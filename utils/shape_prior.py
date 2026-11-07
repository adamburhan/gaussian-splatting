"""Surface ownership and optional fixed support; independent of Gaussian seeding."""

import json
from pathlib import Path

import numpy as np
import torch

from utils.surfaces import load_surface

CDIST_BLOCK = 2**24


class ShapePrior:
    def __init__(
        self,
        path,
        radius=0.15,
        refresh=100,
        device="cuda",
        ownership="surface",
        radius_scale=0.0,
        backend="grid",
    ):
        self.path = Path(path)
        meta = json.loads((self.path / "objects.json").read_text())
        self.labels = [o["label"] for o in meta["objects"]]
        self.refresh, self.device, self.ownership = refresh, device, ownership
        if ownership not in ("surface", "unseen", "evidence"):
            raise ValueError(f"unknown ownership: {ownership}")
        self.surfaces = [
            load_surface(self.path, o, backend, float(meta.get("dilate", 0.3)), device)
            for o in meta["objects"]
        ]
        self.support = (
            [
                SurfaceSupport(self.path / o["support"], meta["support_radius"])
                for o in meta["objects"]
            ]
            if ownership == "evidence"
            else None
        )
        self.radius = torch.tensor(
            [max(radius, radius_scale * s.diagonal) for s in self.surfaces],
            device=device,
            dtype=torch.float32,
        )
        self.owner, self.last_refresh, self.seeds = None, -1, None
        self.pairs, self.pairs_key, self.pose = None, None, None

    def load_ownership(self, file=""):
        """Ownership samples are an input to the constraint, independent of seeding."""
        from plyfile import PlyData

        meta = json.loads((self.path / "objects.json").read_text())
        file = (
            Path(file)
            if file
            else self.path / meta.get("ownership_samples", "completion_harmonic.ply")
        )
        v = PlyData.read(str(file))["vertex"]
        names = json.loads(file.with_suffix(".json").read_text())["objects"]
        xyz = torch.tensor(
            np.stack([v[k] for k in ("x", "y", "z")], 1), device=self.device
        )
        owners = torch.tensor(
            np.asarray(v["object"]).astype(np.int64), device=self.device
        )
        self.set_seeds(xyz, owners, names)

    def canonical(self, k, xyz):
        """World points into the hypothesis frame: p = R^T (x - c - t) + c."""
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
        selected objects; objects.json holds the accepted ones), kept per object of this prior.
        """
        index = torch.tensor(
            [self.labels.index(n) if n in self.labels else -1 for n in names],
            device=owner.device,
        )
        owner = index[owner]
        self.seeds = [
            xyz[owner == k].to(self.device) for k in range(len(self.surfaces))
        ]

    @torch.no_grad()
    def distances(self, xyz):
        """[N, K] distance to every object's surface."""
        return torch.stack(
            [
                self.surfaces[k].distance(self.canonical(k, xyz))
                for k in range(len(self.surfaces))
            ],
            1,
        )

    @torch.no_grad()
    def seed_distance(self, xyz, owner):
        """Distance from each Gaussian to the nearest completion sample of its object; inf where it has none."""
        d = torch.full(
            (xyz.shape[0],), float("inf"), device=xyz.device, dtype=xyz.dtype
        )
        for k, seeds in enumerate(self.seeds):
            m = (owner == k).nonzero()[:, 0]
            if len(m) and len(seeds):
                seeds = self.to_world(k, seeds)
                d[m] = torch.cat(
                    [
                        torch.cdist(xyz[i], seeds).min(1).values
                        for i in m.split(max(1, CDIST_BLOCK // len(seeds)))
                    ]
                )
        return d

    @torch.no_grad()
    def assign(self, xyz):
        distances = self.distances(xyz)
        if self.support is not None:
            for j, surface in enumerate(self.surfaces):
                x = self.canonical(j, xyz)
                distances[:, j].masked_fill_(
                    self.support[j].weight(surface.closest(x), surface.normal(x)) <= 0,
                    float("inf"),
                )
        d, k = distances.min(1)
        radius = self.radius[k]
        owner = torch.where(d < radius, k, torch.full_like(k, -1))
        if self.ownership == "unseen":
            owner = torch.where(
                self.seed_distance(xyz, owner) < radius, owner, torch.full_like(k, -1)
            )
        self.owner = owner
        return self.owner

    def maybe_refresh(self, xyz, iteration):
        if (
            self.owner is None
            or self.owner.shape[0] != xyz.shape[0]
            or iteration - self.last_refresh >= self.refresh
        ):
            self.assign(xyz.detach())
            self.last_refresh = iteration
        return self.owner

    def distance(self, xyz, owner):
        """d(x) on each Gaussian's own object, differentiable in xyz; zero where owner < 0."""
        d = torch.zeros(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)
        for k in range(len(self.surfaces)):
            m = owner == k
            if m.any():
                d = d.index_put(
                    (m.nonzero()[:, 0],),
                    self.surfaces[k].distance(self.canonical(k, xyz[m])),
                )
        return d

    @torch.no_grad()
    def normal(self, xyz, owner):
        """Detached unit normals from the selected surface backend."""
        n = torch.zeros_like(xyz)
        for k in range(len(self.surfaces)):
            m = owner == k
            if m.any():
                g = self.surfaces[k].normal(self.canonical(k, xyz[m]))
                n[m] = g if self.pose is None else g @ self.pose[0][k].T
        return n

    @torch.no_grad()
    def weight(self, xyz, owner):
        """Fixed evidence weights; never normalize by their sum."""
        if self.support is None:
            return torch.ones(len(xyz), device=xyz.device, dtype=xyz.dtype)
        weights = torch.zeros(len(xyz), device=xyz.device, dtype=xyz.dtype)
        for k, surface in enumerate(self.surfaces):
            m = owner == k
            if m.any():
                x = self.canonical(k, xyz[m])
                weights[m] = self.support[k].weight(
                    surface.closest(x), surface.normal(x)
                )
        return weights

    def stats(self):
        """Owned Gaussians per object."""
        counts = (
            torch.bincount(self.owner[self.owner >= 0], minlength=len(self.labels))
            if self.owner is not None
            else torch.zeros(len(self.labels))
        )
        return dict(zip(self.labels, counts.tolist()))


class SurfaceSupport:
    """Nearest-sample influence on the surface, including zero-weight samples.

    Keeping rejected samples prevents querying only the nearest accepted patch.
    Resolution is set by preparation sampling, not by a distance volume.
    """

    def __init__(self, path, radius):
        from scipy.spatial import cKDTree

        with np.load(path) as data:
            self.tree = cKDTree(data["points"])
            self.weights = data["weight"].astype(np.float32)
            self.normals = data["normals"] if "normals" in data else None
        if (
            not np.isfinite(self.weights).all()
            or ((self.weights < 0) | (self.weights > 1)).any()
        ):
            raise ValueError(f"invalid support weights: {path}")
        self.radius = radius

    def weight(self, projected, normal=None):
        d, index = self.tree.query(projected.detach().cpu().numpy())
        w = np.zeros(len(projected), np.float32)
        valid = d <= self.radius
        if self.normals is not None and normal is not None:
            # Avoid borrowing support from the opposite side of a thin surface.
            ids = np.flatnonzero(valid)
            valid[ids] &= (
                self.normals[index[ids]] * normal.detach().cpu().numpy()[ids]
            ).sum(1) > 0
        w[valid] = self.weights[index[valid]]
        return torch.as_tensor(w, device=projected.device, dtype=projected.dtype)
