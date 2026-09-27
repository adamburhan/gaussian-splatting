"""Ray-interval shape prior on the Gaussians' positions and extents (--shape_prior <dir>).

The directory is baked offline (tools/synthetic_capture/shape_prior_maps.py) from fixed world-frame
object meshes: objects.json, one unsigned distance grid per object, and per training frame the
sparse ray intervals (pixel, object, a, b): first and last hit of the pixel's ray on the object, in
metres along the unit ray.

Ownership. Every `refresh` iterations, and whenever the number of Gaussians changed (densification,
pruning), each Gaussian centre is assigned to the closest object whose surface is within `radius`,
else left unassigned; distances come from the grids by trilinear interpolation. Proximity only
chooses ownership; there is no attraction to the surface.

Loss. For the current training camera, each assigned Gaussian is projected to its pixel; if that
pixel carries an interval [a, b] for the Gaussian's own object (other objects on the ray are ignored,
occlusion is not), the Gaussian's spread along its centre ray d = (mu - c) / |mu - c|,
T ~ N(m, s^2) with m = |mu - c| and s^2 = d^T Sigma d, is penalised outside [a - delta, b + delta]:
L = E[(A - T)_+^2 + (T - B)_+^2] = H(A - m, s) + H(m - B, s), H(z, s) = (z^2 + s^2) Phi(z/s) + z s phi(z/s).
Mode "centre" drops the spread: (A - m)_+^2 + (m - B)_+^2. Penalties are averaged per object, then
over the objects with a constrained Gaussian in this camera. Units: squared metres, no normalisation
by interval width, no opacity weighting. Gradients reach positions, scales and rotations (through the
covariance); ownership, pixels, rays and endpoints are detached.

Accepted limitations: proximity can capture neighbouring floor Gaussians; assignments change as
centres move; rays that miss the mesh, or hit it once, carry no constraint; the outer hit envelope
permits gaps between the parts of an object.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def _one_sided(t, var):
    """E[ReLU(T)^2] for T ~ N(t, var) = (t^2 + v) Phi(z) + t s phi(z), z = t / s, in double with the
    complementary error function so the deep negative tail stays accurate and non-negative."""
    dtype = t.dtype
    t = t.double()
    v = var.double().clamp_min(1e-12)
    s = v.sqrt()
    z = t / s
    cdf = 0.5 * torch.erfc(-z / math.sqrt(2.0))
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    return ((t.square() + v) * cdf + t * s * pdf).clamp_min(0.0).to(dtype)


def interval_penalty(m, var, a, b, mode="full"):
    """Per-Gaussian penalty for T ~ N(m, var) against [a, b]; mode "centre" ignores var."""
    if mode == "centre":
        return torch.relu(a - m).square() + torch.relu(m - b).square()
    return _one_sided(a - m, var) + _one_sided(m - b, var)


def directional_variance(covariance, d):
    """d^T Sigma d per Gaussian from the packed (xx, xy, xz, yy, yz, zz) rows of get_covariance; d [N, 3]."""
    c = covariance
    return (c[:, 0] * d[:, 0] * d[:, 0] + c[:, 3] * d[:, 1] * d[:, 1] + c[:, 5] * d[:, 2] * d[:, 2]
            + 2 * (c[:, 1] * d[:, 0] * d[:, 1] + c[:, 2] * d[:, 0] * d[:, 2] + c[:, 4] * d[:, 1] * d[:, 2]))


class ShapePrior:
    def __init__(self, path, radius=0.15, delta=0.05, refresh=100, mode="full", device="cuda"):
        meta = json.load(open(os.path.join(path, "objects.json")))
        self.labels = [o["label"] for o in meta["objects"]]
        self.width, self.height = meta["width"], meta["height"]
        self.radius, self.delta, self.refresh, self.mode, self.device = radius, delta, refresh, mode, device
        self.grids = []
        for o in meta["objects"]:
            g = np.load(os.path.join(path, o["distance"]))
            self.grids.append((torch.tensor(g["origin"], device=device), float(g["spacing"]), float(meta["dilate"]),
                               torch.tensor(g["values"].astype(np.float32), device=device)[None, None]))  # [1, 1, X, Y, Z]
        # per frame: keys = object * H * W + pixel, sorted, for a binary-search lookup without dense maps
        self.frames = {}
        for name in meta["frames"]:
            iv = np.load(os.path.join(path, "intervals", f"{name}.npz"))
            key = iv["object"].astype(np.int64) * (self.width * self.height) + iv["pixel"].astype(np.int64)
            order = np.argsort(key)
            self.frames[name] = tuple(torch.tensor(x[order], device=device) for x in (key, iv["a"], iv["b"]))
        self.owner = None  # [N] long, -1 = unassigned
        self.last_refresh = -1

    # --- ownership ---------------------------------------------------------------

    @torch.no_grad()
    def distances(self, xyz):
        """[N, K] unsigned distance to each object's surface, clamped at the grid margin outside the grid."""
        out = []
        for origin, spacing, clamp, values in self.grids:
            shape = torch.tensor(values.shape[2:], device=xyz.device, dtype=xyz.dtype)
            u = (xyz - origin) / spacing / (shape - 1) * 2 - 1  # [-1, 1] across the grid, index order (x, y, z)
            # grid_sample indexes its last dimension with the first coordinate, so feed (z, y, x)
            d = F.grid_sample(values, u.flip(-1)[None, None, None], mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0, 0]
            outside = (u.abs() > 1).any(-1)
            out.append(torch.where(outside, torch.full_like(d, clamp), d))
        return torch.stack(out, 1)

    @torch.no_grad()
    def assign(self, xyz):
        d, k = self.distances(xyz).min(1)
        self.owner = torch.where(d < self.radius, k, torch.full_like(k, -1))
        return self.owner

    def maybe_refresh(self, xyz, iteration):
        """Recompute ownership on the refresh cadence and whenever the Gaussian count changed."""
        if self.owner is None or self.owner.shape[0] != xyz.shape[0] or iteration - self.last_refresh >= self.refresh:
            self.assign(xyz.detach())
            self.last_refresh = iteration
        return self.owner

    # --- loss --------------------------------------------------------------------

    def lookup(self, cam, xyz):
        """For every Gaussian: its object's interval at its pixel in this camera, or invalid.
        Returns (valid [N] bool, a [N], b [N]) with the endpoints already widened by delta."""
        n = xyz.shape[0]
        key_tab, a_tab, b_tab = self.frames[os.path.splitext(cam.image_name)[0]]
        if len(key_tab) == 0:  # a frame where no ray hits any constrained object
            zeros = torch.zeros(n, device=xyz.device)
            return torch.zeros(n, dtype=torch.bool, device=xyz.device), zeros, zeros
        with torch.no_grad():
            view = cam.world_view_transform  # stored transposed: camera coordinates = [x, 1] @ view (OpenCV: z forward, y down)
            x_cam = xyz @ view[:3, :3] + view[3, :3]
            z = x_cam[:, 2]
            fx = self.width / (2 * math.tan(cam.FoVx / 2))
            fy = self.height / (2 * math.tan(cam.FoVy / 2))
            zs = z.clamp_min(1e-3)
            u = torch.round(fx * x_cam[:, 0] / zs + self.width / 2 - 0.5).long()
            v = torch.round(fy * x_cam[:, 1] / zs + self.height / 2 - 0.5).long()
            inside = (z > 1e-3) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height) & (self.owner >= 0)
            key = self.owner.clamp_min(0) * (self.width * self.height) + v.clamp(0, self.height - 1) * self.width + u.clamp(0, self.width - 1)
            idx = torch.searchsorted(key_tab, key).clamp_max(len(key_tab) - 1)
            found = inside & (key_tab[idx] == key)
            a = torch.where(found, a_tab[idx] - self.delta, torch.zeros(n, device=xyz.device))
            b = torch.where(found, b_tab[idx] + self.delta, torch.zeros(n, device=xyz.device))
        return found, a, b

    def loss(self, cam, xyz, covariance):
        """Scalar; zero (with a graph) when nothing is constrained. Also returns the number constrained."""
        valid, a, b = self.lookup(cam, xyz)
        if not valid.any():
            return xyz.sum() * 0.0, 0
        mu, a, b, owner = xyz[valid], a[valid], b[valid], self.owner[valid]
        delta = mu - cam.camera_center
        m = delta.norm(dim=1)
        d = (delta / m[:, None]).detach()
        var = directional_variance(covariance[valid], d)
        penalty = interval_penalty(m, var, a, b, self.mode)
        # mean within each object, then mean over the objects present, so big objects do not dominate
        per_object = torch.zeros(len(self.labels), device=xyz.device).index_add(0, owner, penalty)
        count = torch.bincount(owner, minlength=len(self.labels)).to(penalty.dtype)
        present = count > 0
        return (per_object[present] / count[present]).mean(), int(valid.sum())

    @torch.no_grad()
    def stats(self):
        """Assigned Gaussians per object."""
        counts = torch.bincount(self.owner.clamp_min(0)[self.owner >= 0], minlength=len(self.labels))
        return {label: int(c) for label, c in zip(self.labels, counts)}
