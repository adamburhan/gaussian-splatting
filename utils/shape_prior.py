"""Shape prior on the Gaussians' positions and extents (--shape_prior <dir>). Modes: "containment" (expected
containment in the object's solid volume), "containment_centre" (the centre only), "intervals" (ray
intervals in the current camera), "intervals_centre" (ray intervals, centre only).

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
Mode "intervals_centre" drops the spread: (A - m)_+^2 + (m - B)_+^2. Penalties are averaged per object, then
over the objects with a constrained Gaussian in this camera. Units: squared metres, no normalisation
by interval width, no opacity weighting. Gradients reach positions, scales and rotations (through the
covariance); ownership, pixels, rays and endpoints are detached.

Containment (mode "containment"). The bake also stores each object's solid volume V_o (generalized
winding number > 0.5). With d(x, V_o) = 0 inside and the distance to the surface outside, every
owned Gaussian pays, in every iteration regardless of the camera or occlusion,
L_i = E_{X ~ N(mu_i, Sigma_i)}[(d(X, V_o) - delta)_+^2], estimated with `samples` reparameterised draws
X = mu_i + R_i diag(s_i) eps ("containment_centre": the centre only). Same aggregation and units as above;
gradients reach positions, scales and rotations through the samples and the trilinear grid.

Accepted limitations: proximity can capture neighbouring floor Gaussians; assignments change as
centres move; rays that miss the mesh, or hit it once, carry no constraint; the outer hit envelope
permits gaps between the parts of an object; the solid of an open mesh is what its winding number
says (a cavity between closed parts counts as outside).
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


def quaternion_to_matrix(q):
    """[N, 4] (w, x, y, z), normalised inside -> [N, 3, 3], as the fork's build_rotation."""
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


MODES = ("containment", "containment_centre", "intervals", "intervals_centre")


class ShapePrior:
    def __init__(self, path, radius=0.15, delta=0.05, refresh=100, mode="containment", samples=16, device="cuda"):
        if mode not in MODES:
            raise ValueError(f"shape prior mode {mode!r}; choose one of {MODES}")
        if mode == "containment_centre":
            samples = 0
        meta = json.load(open(os.path.join(path, "objects.json")))
        self.labels = [o["label"] for o in meta["objects"]]
        self.width, self.height = meta["width"], meta["height"]
        self.radius, self.delta, self.refresh, self.mode, self.samples, self.device = radius, delta, refresh, mode, samples, device
        self.grids, self.inside = [], []
        for o in meta["objects"]:
            g = np.load(os.path.join(path, o["distance"]))
            self.grids.append((torch.tensor(g["origin"], device=device), float(g["spacing"]), float(meta["dilate"]),
                               torch.tensor(g["values"].astype(np.float32), device=device)[None, None]))  # [1, 1, X, Y, Z]
            self.inside.append(torch.tensor(g["inside"].astype(np.float32), device=device)[None, None] if "inside" in g else None)
        if mode.startswith("containment") and any(m is None for m in self.inside):
            raise ValueError("containment needs the solid volume: re-bake the prior with inside masks")
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

    def sample_grid(self, k, xyz, field):
        """Trilinear sample of object k's `field` grid at xyz [N, 3]; differentiable in xyz."""
        origin, spacing, clamp, values = self.grids[k]
        shape = torch.tensor(values.shape[2:], device=xyz.device, dtype=xyz.dtype)
        u = (xyz - origin) / spacing / (shape - 1) * 2 - 1  # [-1, 1] across the grid, index order (x, y, z)
        # grid_sample indexes its last dimension with the first coordinate, so feed (z, y, x)
        v = F.grid_sample(field, u.flip(-1)[None, None, None], mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0, 0]
        return v, (u.abs() > 1).any(-1)

    @torch.no_grad()
    def distances(self, xyz):
        """[N, K] unsigned distance to each object's surface, clamped at the grid margin outside the grid."""
        out = []
        for k, (origin, spacing, clamp, values) in enumerate(self.grids):
            d, outside = self.sample_grid(k, xyz, values)
            out.append(torch.where(outside, torch.full_like(d, clamp), d))
        return torch.stack(out, 1)

    def containment_distance(self, k, xyz):
        """d(x, V_k): zero inside object k's solid, the distance to its surface outside; differentiable in xyz."""
        d, outside = self.sample_grid(k, xyz, self.grids[k][3])
        inside, _ = self.sample_grid(k, xyz, self.inside[k])
        d = torch.where(outside, torch.full_like(d, self.grids[k][2]), d)
        return torch.where(inside > 0.5, torch.zeros_like(d), d)

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

    def loss(self, cam, xyz, covariance=None, scaling=None, rotation=None):
        """Scalar; zero (with a graph) when nothing is constrained. Also returns the number constrained.
        Ray modes use `covariance` (packed); containment uses `scaling` [N, 3] and `rotation` [N, 4]."""
        if self.mode.startswith("containment"):
            return self.containment_loss(xyz, scaling, rotation)
        valid, a, b = self.lookup(cam, xyz)
        if not valid.any():
            return xyz.sum() * 0.0, 0
        mu, a, b, owner = xyz[valid], a[valid], b[valid], self.owner[valid]
        delta = mu - cam.camera_center
        m = delta.norm(dim=1)
        d = (delta / m[:, None]).detach()
        var = directional_variance(covariance[valid], d)
        penalty = interval_penalty(m, var, a, b, "centre" if self.mode == "intervals_centre" else "full")
        # mean within each object, then mean over the objects present, so big objects do not dominate
        per_object = torch.zeros(len(self.labels), device=xyz.device).index_add(0, owner, penalty)
        count = torch.bincount(owner, minlength=len(self.labels)).to(penalty.dtype)
        present = count > 0
        return (per_object[present] / count[present]).mean(), int(valid.sum())

    def containment_loss(self, xyz, scaling, rotation):
        owned = self.owner >= 0
        if not owned.any():
            return xyz.sum() * 0.0, 0
        n_samples = max(self.samples, 1)
        total, present = xyz.new_zeros(()), 0
        R = quaternion_to_matrix(rotation) if self.samples > 0 else None
        for k in range(len(self.labels)):
            mine = self.owner == k
            if not mine.any():
                continue
            mu = xyz[mine]
            if self.samples > 0:
                eps = torch.randn(mu.shape[0], n_samples, 3, device=xyz.device)
                offsets = torch.einsum("nkj,nij->nki", eps * scaling[mine][:, None, :], R[mine])  # R diag(s) eps
                pts = (mu[:, None, :] + offsets).reshape(-1, 3)
            else:
                pts = mu
            penalty = torch.relu(self.containment_distance(k, pts) - self.delta).square().reshape(mu.shape[0], -1).mean(1)
            total = total + penalty.mean()
            present += 1
        return total / present, int(owned.sum())

    @torch.no_grad()
    def stats(self):
        """Assigned Gaussians per object."""
        counts = torch.bincount(self.owner.clamp_min(0)[self.owner >= 0], minlength=len(self.labels))
        return {label: int(c) for label, c in zip(self.labels, counts)}
