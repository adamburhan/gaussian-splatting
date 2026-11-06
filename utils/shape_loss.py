"""ADAM CHANGE: shape losses that keep the Gaussians near a known object on its surface.

The prior directory (completion_bake.py / the old shape_prior_maps.py) holds, per object, an unsigned distance
to the mesh surface on a regular grid: distance/<label>.npz with `origin` [3], `spacing` (m) and `values`
[X, Y, Z] clamped at `dilate` (objects.json). Trilinear interpolation of that grid gives d(x), differentiable
in x. The normal n(x) (detached) comes from central differences of the SIGNED field d * (1 - 2 * inside), with
`inside` the grid's solid mask (winding number): the unsigned field has a kink on the surface, where its central
difference cancels along the normal, and that is exactly where the Gaussians sit. Without an inside mask the
unsigned field is used and normals on the surface are unreliable.

Ownership: a Gaussian belongs to the object whose surface is nearest if that distance is below the object's
radius, max(`radius`, `radius_scale` x the object's box diagonal), recomputed every `refresh` iterations and
whenever the Gaussian count changes. With `ownership` "unseen" it must
also lie within `radius` of one of that object's completion samples (set_seeds: the unseen surface), so the
seen surface stays unconstrained. Every loss is a mean over the owned Gaussians (of the chosen target set) of a
per-Gaussian penalty in squared metres:

    centre    p(d)                                    the centre on the surface
    surface   p(d) + spread * (n^T Sigma n - tol^2)_+  ... and thin across it: n^T Sigma n is the variance along
                                                      the normal, so d^2 + n^T Sigma n = E[dist^2 to the local plane]
    band      E_X[(d(X) - delta)_+^2]                 X ~ N(mu, Sigma), `samples` reparameterised draws: the whole
                                                      Gaussian within `delta` of the surface, curved or not

p(d) is d^2, or with `robust` "huber" the Huber penalty of scale `robust_scale`: quadratic within the scale,
linear beyond, so a Gaussian far from the surface is pulled with a bounded force.

Each per-Gaussian penalty may be weighted by the Gaussian's opacity (`opacity_weight`): "none" (weight 1),
"detached" (alpha as a constant: transparent Gaussians stop pulling gradient) or "trainable" (gradient flows
into alpha too: a Gaussian far from the surface may fade instead of moving).

A loss takes (prior, xyz, scaling, rotation, owner, opacity=None, opacity_weight="none", **params) and returns
(value, number of Gaussians it acted on). xyz / scaling / rotation / opacity are the model's activated tensors;
owner is -1 where a Gaussian is not constrained. Device-agnostic (CPU tests).

seed_colour pulls each completion Gaussian's DC colour toward the mean DC colour (detached) of the ordinary
Gaussians of the same object within `colour_radius`; the neighbour sets are rebuilt with the ownership.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

CDIST_BLOCK = 2 ** 24  # entries of one pairwise-distance block


def quaternion_to_matrix(q):
    """[N, 4] (w, x, y, z) -> [N, 3, 3], as the fork's build_rotation but on any device."""
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


class ShapePrior:
    def __init__(self, path, radius=0.15, refresh=100, device="cuda", ownership="surface", radius_scale=0.0):
        meta = json.load(open(os.path.join(path, "objects.json")))
        self.labels = [o["label"] for o in meta["objects"]]
        self.refresh, self.device, self.ownership = refresh, device, ownership
        clamp = float(meta.get("dilate", 0.3))
        self.grids, self.inside, diagonals = [], [], []
        for o in meta["objects"]:
            g = np.load(os.path.join(path, o["distance"]))
            self.grids.append((torch.tensor(g["origin"], device=device), float(g["spacing"]), clamp,
                               torch.tensor(g["values"].astype(np.float32), device=device)[None, None]))  # [1, 1, X, Y, Z]
            self.inside.append(torch.tensor(g["inside"].astype(np.float32), device=device)[None, None] if "inside" in g else None)
            diagonals.append(np.linalg.norm((np.array(g["values"].shape) - 1) * float(g["spacing"]) - 2 * clamp))
        self.radius = torch.tensor(np.maximum(radius, radius_scale * np.array(diagonals)), device=device, dtype=torch.float32)  # [K]
        self.owner = None  # [N] long, -1 = unconstrained
        self.last_refresh = -1
        self.seeds = None  # per object, its completion samples [M_k, 3] (ownership "unseen")
        self.pairs, self.pairs_key = None, None  # seed_colour's (seed, neighbour) index pairs and the ownership they match
        self.pose = None  # (R [K, 3, 3], t [K, 3], c [K, 3]) when the objects are rigid bodies (train.py object_pose)

    def canonical(self, k, xyz):
        """World points into object k's grid frame under the current pose: p = R^T (x - c - t) + c."""
        if self.pose is None:
            return xyz
        R, t, c = self.pose
        return (xyz - c[k] - t[k]) @ R[k] + c[k]

    def to_world(self, k, xyz):
        if self.pose is None:
            return xyz
        R, t, c = self.pose
        return (xyz - c[k]) @ R[k].T + c[k] + t[k]

    def set_seeds(self, xyz, owner, names):
        """Completion samples [M, 3] and their object index [M] into `names` (the completion PLY numbers the
        selected objects; objects.json holds the accepted ones), kept per object of this prior."""
        index = torch.tensor([self.labels.index(n) if n in self.labels else -1 for n in names], device=owner.device)
        owner = index[owner]
        self.seeds = [xyz[owner == k].to(self.device) for k in range(len(self.grids))]

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
        return torch.stack([self.sample_grid(k, self.canonical(k, xyz)) for k in range(len(self.grids))], 1)

    @torch.no_grad()
    def seed_distance(self, xyz, owner):
        """Distance from each Gaussian to the nearest completion sample of its object; inf where it has none."""
        d = torch.full((xyz.shape[0],), float("inf"), device=xyz.device, dtype=xyz.dtype)
        for k, seeds in enumerate(self.seeds):
            m = (owner == k).nonzero()[:, 0]
            if len(m) and len(seeds):
                seeds = self.to_world(k, seeds)
                d[m] = torch.cat([torch.cdist(xyz[i], seeds).min(1).values for i in m.split(max(1, CDIST_BLOCK // len(seeds)))])
        return d

    @torch.no_grad()
    def assign(self, xyz):
        d, k = self.distances(xyz).min(1)
        radius = self.radius[k]
        owner = torch.where(d < radius, k, torch.full_like(k, -1))
        if self.ownership == "unseen":
            owner = torch.where(self.seed_distance(xyz, owner) < radius, owner, torch.full_like(k, -1))
        self.owner = owner
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
                d = d.index_put((m.nonzero()[:, 0],), self.sample_grid(k, self.canonical(k, xyz[m])))
        return d

    @torch.no_grad()
    def normal(self, xyz, owner):
        """Unit normal of each Gaussian's object surface at the Gaussian, by central differences of one grid spacing."""
        n = torch.zeros_like(xyz)
        for k in range(len(self.grids)):
            m = owner == k
            if m.any():
                h = self.grids[k][1]
                x = self.canonical(k, xyz[m])
                g = torch.stack([self.sample_grid(k, x + h * e, signed=True) - self.sample_grid(k, x - h * e, signed=True)
                                 for e in torch.eye(3, device=xyz.device, dtype=xyz.dtype)], 1)
                g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-9)
                n[m] = g if self.pose is None else g @ self.pose[0][k].T
        return n

    def stats(self):
        """Owned Gaussians per object."""
        counts = torch.bincount(self.owner[self.owner >= 0], minlength=len(self.labels)) if self.owner is not None else torch.zeros(len(self.labels))
        return dict(zip(self.labels, counts.tolist()))


# --- losses ------------------------------------------------------------------------

def _zero(xyz):
    return xyz.sum() * 0.0, 0


def _weights(opacity, m, mode):
    """Per-Gaussian weight of the penalty: 1, or the opacity (constant or trainable)."""
    if mode == "none":
        return 1.0
    w = opacity[m].reshape(-1)
    return w.detach() if mode == "detached" else w


def _penalty(d, robust, scale):
    """d^2, or Huber: d^2 within `scale`, linear beyond."""
    if robust == "huber":
        return torch.where(d.abs() <= scale, d ** 2, scale * (2 * d.abs() - scale))
    return d ** 2


def centre(prior, xyz, scaling, rotation, owner, opacity=None, opacity_weight="none", robust="none", robust_scale=0.03, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    d = prior.distance(xyz[m], owner[m])
    return (_weights(opacity, m, opacity_weight) * _penalty(d, robust, robust_scale)).mean(), int(m.sum())


def surface(prior, xyz, scaling, rotation, owner, opacity=None, opacity_weight="none", spread=1.0, tolerance=0.005,
            robust="none", robust_scale=0.03, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    d = prior.distance(xyz[m], owner[m])
    n = prior.normal(xyz[m].detach(), owner[m])
    v = (quaternion_to_matrix(rotation[m]).transpose(1, 2) @ n[..., None])[..., 0]  # normal in the Gaussian's frame
    variance = (scaling[m] ** 2 * v ** 2).sum(1)  # n^T R S^2 R^T n
    penalty = _penalty(d, robust, robust_scale) + spread * torch.relu(variance - tolerance ** 2)
    return (_weights(opacity, m, opacity_weight) * penalty).mean(), int(m.sum())


def band(prior, xyz, scaling, rotation, owner, opacity=None, opacity_weight="none", delta=0.02, samples=16, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    R = quaternion_to_matrix(rotation[m])
    eps = torch.randn(samples, int(m.sum()), 3, device=xyz.device, dtype=xyz.dtype)
    X = xyz[m] + (R @ (scaling[m] * eps)[..., None])[..., 0]  # [S, M, 3]
    d = prior.distance(X.reshape(-1, 3), owner[m].repeat(samples)).reshape(samples, -1)
    return (_weights(opacity, m, opacity_weight) * (torch.relu(d - delta) ** 2).mean(0)).mean(), int(m.sum())


LOSSES = {"centre": centre, "surface": surface, "band": band}


def object_fit(prior, observed, R, t, c, samples=2048, scale=0.03):
    """The pose's evidence from the visible side: Huber distance of each object's observed points (the map cloud,
    world frame) to its mesh under the pose (R, t, c), differentiable in the pose. Mean over objects with points."""
    total, count = None, 0
    for k, q in enumerate(observed):
        if q is None or not len(q):
            continue
        if len(q) > samples:
            q = q[torch.randint(len(q), (samples,), device=q.device)]
        d = prior.sample_grid(k, (q - c[k] - t[k]) @ R[k] + c[k])
        term = torch.where(d < scale, 0.5 * d**2 / scale, d - 0.5 * scale).mean()
        total = term if total is None else total + term
        count += 1
    return (total / count if count else R.sum() * 0.0), count


@torch.no_grad()
def seed_neighbours(xyz, owner, completed, radius):
    """Index pairs (seed, ordinary Gaussian) of the same object within radius, as two [P] tensors."""
    seeds, neighbours = [], []
    for k in owner[completed & (owner >= 0)].unique():
        s, o = ((owner == k) & completed).nonzero()[:, 0], ((owner == k) & ~completed).nonzero()[:, 0]
        if not len(o):
            continue
        for i in s.split(max(1, CDIST_BLOCK // len(o))):
            a, b = (torch.cdist(xyz[i], xyz[o]) < radius).nonzero(as_tuple=True)
            seeds.append(i[a])
            neighbours.append(o[b])
    empty = torch.zeros(0, dtype=torch.long, device=xyz.device)
    return (torch.cat(seeds), torch.cat(neighbours)) if seeds else (empty, empty)


def seed_colour(prior, xyz, features_dc, owner, completed, radius=0.10):
    """Mean over the seeds with neighbours of the squared DC difference to their neighbours' mean, and that count."""
    if prior.pairs_key != (prior.last_refresh, xyz.shape[0]):
        prior.pairs = seed_neighbours(xyz.detach(), owner, completed, radius)
        prior.pairs_key = (prior.last_refresh, xyz.shape[0])
    seed, neighbour = prior.pairs
    if not len(seed):
        return _zero(features_dc)
    count = torch.bincount(seed, minlength=xyz.shape[0])
    total = torch.zeros_like(features_dc).index_add_(0, seed, features_dc[neighbour].detach())
    has = count > 0
    return ((features_dc[has] - total[has] / count[has, None, None]) ** 2).sum(-1).mean(), int(has.sum())


def target_mask(gaussians, targets):
    """Which Gaussians a shape loss may act on: ordinary | completion | all."""
    if targets == "all":
        return torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
    return gaussians.completed if targets == "completion" else ~gaussians.completed
