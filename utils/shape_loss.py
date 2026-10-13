"""ADAM CHANGE: shape losses that keep the Gaussians near a known object on its surface.

The prior directory (completion_bake.py / the old shape_prior_maps.py) holds, per object, an unsigned distance
to the mesh surface on a regular grid: distance/<label>.npz with `origin` [3], `spacing` (m) and `values`
[X, Y, Z] clamped at `dilate` (objects.json). Trilinear interpolation of that grid gives d(x), differentiable
in x. The normal n(x) (detached) comes from central differences of the SIGNED field d * (1 - 2 * inside), with
`inside` the grid's solid mask (winding number): the unsigned field has a kink on the surface, where its central
difference cancels along the normal, and that is exactly where the Gaussians sit. Without an inside mask the
unsigned field is used and normals on the surface are unreliable.

Ownership: a Gaussian belongs to the object whose surface is nearest if that distance is below `radius`,
recomputed every `refresh` iterations and whenever the Gaussian count changes. Every loss is a mean over the
owned Gaussians (of the chosen target set) of a per-Gaussian penalty in squared metres:

    centre    d^2                                     the centre on the surface
    surface   d^2 + spread * (n^T Sigma n - tol^2)_+  ... and thin across it: n^T Sigma n is the variance along
                                                      the normal, so d^2 + n^T Sigma n = E[dist^2 to the local plane]
    band      E_X[(d(X) - delta)_+^2]                 X ~ N(mu, Sigma), `samples` reparameterised draws: the whole
                                                      Gaussian within `delta` of the surface, curved or not

A loss takes (prior, xyz, scaling, rotation, owner, **params) and returns (value, number of Gaussians it
acted on). xyz / scaling / rotation are the model's activated tensors; owner is -1 where a Gaussian is not
constrained. Device-agnostic (CPU tests).
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def quaternion_to_matrix(q):
    """[N, 4] (w, x, y, z) -> [N, 3, 3], as the fork's build_rotation but on any device."""
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


class ShapePrior:
    def __init__(self, path, radius=0.15, refresh=100, device="cuda"):
        meta = json.load(open(os.path.join(path, "objects.json")))
        self.labels = [o["label"] for o in meta["objects"]]
        self.radius, self.refresh, self.device = radius, refresh, device
        clamp = float(meta.get("dilate", 0.3))
        self.grids, self.inside = [], []
        for o in meta["objects"]:
            g = np.load(os.path.join(path, o["distance"]))
            self.grids.append((torch.tensor(g["origin"], device=device), float(g["spacing"]), clamp,
                               torch.tensor(g["values"].astype(np.float32), device=device)[None, None]))  # [1, 1, X, Y, Z]
            self.inside.append(torch.tensor(g["inside"].astype(np.float32), device=device)[None, None] if "inside" in g else None)
        self.owner = None  # [N] long, -1 = unconstrained
        self.last_refresh = -1

    def sample_grid(self, k, xyz, signed=False):
        """Trilinear sample of object k's distance at xyz [N, 3], differentiable in xyz; clamp outside the grid.
        signed: negative inside the solid (needs the inside mask; falls back to unsigned)."""
        origin, spacing, clamp, values = self.grids[k]
        shape = torch.tensor(values.shape[2:], device=xyz.device, dtype=xyz.dtype)
        u = (xyz - origin) / spacing / (shape - 1) * 2 - 1  # [-1, 1] across the grid, index order (x, y, z)
        # grid_sample indexes its last dimension with the first coordinate, so feed (z, y, x)
        grid = u.flip(-1)[None, None, None]
        d = F.grid_sample(values, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0, 0]
        if signed and self.inside[k] is not None:
            d = d * (1 - 2 * F.grid_sample(self.inside[k], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0, 0])
        return torch.where((u.abs() > 1).any(-1), torch.full_like(d, clamp), d)

    @torch.no_grad()
    def distances(self, xyz):
        """[N, K] distance to every object's surface."""
        return torch.stack([self.sample_grid(k, xyz) for k in range(len(self.grids))], 1)

    @torch.no_grad()
    def assign(self, xyz):
        d, k = self.distances(xyz).min(1)
        self.owner = torch.where(d < self.radius, k, torch.full_like(k, -1))
        return self.owner

    def maybe_refresh(self, xyz, iteration):
        if self.owner is None or self.owner.shape[0] != xyz.shape[0] or iteration - self.last_refresh >= self.refresh:
            self.assign(xyz.detach())
            self.last_refresh = iteration
        return self.owner

    def distance(self, xyz, owner):
        """d(x) on each Gaussian's own object, differentiable in xyz; zero where owner < 0."""
        d = torch.zeros(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)
        for k in range(len(self.grids)):
            m = owner == k
            if m.any():
                d = d.index_put((m.nonzero()[:, 0],), self.sample_grid(k, xyz[m]))
        return d

    @torch.no_grad()
    def normal(self, xyz, owner):
        """Unit normal of each Gaussian's object surface at the Gaussian, by central differences of one grid spacing."""
        n = torch.zeros_like(xyz)
        for k in range(len(self.grids)):
            m = owner == k
            if m.any():
                h = self.grids[k][1]
                x = xyz[m]
                g = torch.stack([self.sample_grid(k, x + h * e, signed=True) - self.sample_grid(k, x - h * e, signed=True)
                                 for e in torch.eye(3, device=xyz.device, dtype=xyz.dtype)], 1)
                n[m] = g / g.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return n

    def stats(self):
        """Owned Gaussians per object."""
        counts = torch.bincount(self.owner[self.owner >= 0], minlength=len(self.labels)) if self.owner is not None else torch.zeros(len(self.labels))
        return dict(zip(self.labels, counts.tolist()))


# --- losses ------------------------------------------------------------------------

def _zero(xyz):
    return xyz.sum() * 0.0, 0


def centre(prior, xyz, scaling, rotation, owner, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    d = prior.distance(xyz[m], owner[m])
    return (d ** 2).mean(), int(m.sum())


def surface(prior, xyz, scaling, rotation, owner, spread=1.0, tolerance=0.005, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    d = prior.distance(xyz[m], owner[m])
    n = prior.normal(xyz[m].detach(), owner[m])
    v = (quaternion_to_matrix(rotation[m]).transpose(1, 2) @ n[..., None])[..., 0]  # normal in the Gaussian's frame
    variance = (scaling[m] ** 2 * v ** 2).sum(1)  # n^T R S^2 R^T n
    return (d ** 2 + spread * torch.relu(variance - tolerance ** 2)).mean(), int(m.sum())


def band(prior, xyz, scaling, rotation, owner, delta=0.02, samples=16, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    R = quaternion_to_matrix(rotation[m])
    eps = torch.randn(samples, int(m.sum()), 3, device=xyz.device, dtype=xyz.dtype)
    X = xyz[m] + (R @ (scaling[m] * eps)[..., None])[..., 0]  # [S, M, 3]
    d = prior.distance(X.reshape(-1, 3), owner[m].repeat(samples))
    return (torch.relu(d - delta) ** 2).mean(), int(m.sum())


LOSSES = {"centre": centre, "surface": surface, "band": band}


def target_mask(gaussians, targets):
    """Which Gaussians a shape loss may act on: ordinary | completion | all."""
    if targets == "all":
        return torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
    return gaussians.completed if targets == "completion" else ~gaussians.completed
