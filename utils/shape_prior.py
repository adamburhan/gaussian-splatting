"""Assign Gaussians to distance-grid priors; test with pytest tests/test_shape_loss.py."""

import json
from pathlib import Path

import numpy as np
import torch

from utils.surfaces import GridSurface

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
    ):
        self.path = Path(path)
        meta = json.loads((self.path / "objects.json").read_text())
        self.labels = [o["label"] for o in meta["objects"]]
        self.refresh, self.device, self.ownership = refresh, device, ownership
        if ownership not in ("surface", "unseen"):
            raise ValueError(f"unknown ownership: {ownership}")
        self.surfaces = [
            GridSurface(self.path / o["distance"], float(meta.get("dilate", 0.3)), device)
            for o in meta["objects"]
        ]
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
        """Detached unit normals from the distance grid."""
        n = torch.zeros_like(xyz)
        for k in range(len(self.surfaces)):
            m = owner == k
            if m.any():
                g = self.surfaces[k].normal(self.canonical(k, xyz[m]))
                n[m] = g if self.pose is None else g @ self.pose[0][k].T
        return n

    def stats(self):
        """Owned Gaussians per object."""
        counts = (
            torch.bincount(self.owner[self.owner >= 0], minlength=len(self.labels))
            if self.owner is not None
            else torch.zeros(len(self.labels))
        )
        return dict(zip(self.labels, counts.tolist()))
