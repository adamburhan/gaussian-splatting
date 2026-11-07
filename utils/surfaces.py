"""Fixed surfaces queried in their preparation frame; gradients flow only to x.

Mesh uses Open3D's CPU BVH as a reference backend. Its nearest point is detached:
the envelope theorem gives the distance gradient away from nearest-face ties.
No voxelization or watertightness is required. Profile transfers before scaling up.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class GridSurface:
    def __init__(self, path, clamp, device):
        with np.load(path) as g:
            self.origin = torch.tensor(g["origin"], device=device)
            self.spacing, self.clamp = float(g["spacing"]), clamp
            self.values = torch.tensor(g["values"].astype(np.float32), device=device)[
                None, None
            ]
            self.inside = (
                torch.tensor(g["inside"].astype(np.float32), device=device)[None, None]
                if "inside" in g
                else None
            )
        self.diagonal = float(
            np.linalg.norm(
                (np.array(self.values.shape[2:]) - 1) * self.spacing - 2 * clamp
            )
        )

    def sample(self, x, signed=False):
        shape = x.new_tensor(self.values.shape[2:])
        u = (x - self.origin) / self.spacing / (shape - 1) * 2 - 1
        grid = u.flip(-1)[None, None, None]
        d = F.grid_sample(
            self.values,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0, 0, 0]
        if signed and self.inside is not None:
            d = d * (
                1
                - 2
                * F.grid_sample(
                    self.inside,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )[0, 0, 0, 0]
            )
        return torch.where((u.abs() > 1).any(-1), torch.full_like(d, self.clamp), d)

    def distance(self, x):
        return self.sample(x)

    def normal(self, x):
        h = self.spacing
        g = torch.stack(
            [
                self.sample(x + h * e, True) - self.sample(x - h * e, True)
                for e in torch.eye(3, device=x.device)
            ],
            1,
        )
        return F.normalize(g, dim=1, eps=1e-9)

    def closest(self, x):
        # Use unsigned-distance descent so interior points project outward too.
        h = self.spacing
        g = torch.stack(
            [
                self.sample(x + h * e) - self.sample(x - h * e)
                for e in torch.eye(3, device=x.device)
            ],
            1,
        )
        return x - self.distance(x)[:, None] * F.normalize(g, dim=1, eps=1e-9)


class MeshSurface:
    def __init__(self, path, device):
        import open3d as o3d

        with np.load(path) as mesh:
            vertices, faces = mesh["vertices"].astype(np.float32), mesh["faces"].astype(
                np.uint32
            )
        if not len(faces):
            raise ValueError(f"empty surface: {path}")
        self.diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))

    def query(self, x):
        import open3d as o3d

        if not len(x):
            return x.detach(), x.detach()
        result = self.scene.compute_closest_points(
            o3d.core.Tensor(x.detach().cpu().float().numpy())
        )
        return tuple(
            torch.as_tensor(result[key].numpy(), device=x.device, dtype=x.dtype)
            for key in ("points", "primitive_normals")
        )

    def closest(self, x):
        return self.query(x)[0]

    def distance(self, x):
        return (x - self.closest(x)).norm(dim=1)

    def normal(self, x):
        return self.query(x)[1]


class PlaneSurface:
    """Finite rectangular patch: origin, orthonormal tangent axes, half extents."""

    def __init__(self, spec, device):
        self.origin = torch.tensor(spec["origin"], dtype=torch.float32, device=device)
        self.axes = torch.tensor(spec["axes"], dtype=torch.float32, device=device)
        self.half = torch.tensor(
            spec["half_extent"], dtype=torch.float32, device=device
        )
        if self.axes.shape != (2, 3) or not torch.allclose(
            self.axes @ self.axes.T, torch.eye(2, device=device), atol=1e-5
        ):
            raise ValueError("plane axes must be two orthonormal tangent vectors")
        if self.half.shape != (2,) or (self.half <= 0).any():
            raise ValueError("plane half extents must be positive")
        self.n = torch.linalg.cross(self.axes[0], self.axes[1])
        self.diagonal = float((2 * self.half).norm())

    def closest(self, x):
        uv = ((x - self.origin) @ self.axes.T).clamp(-self.half, self.half)
        return self.origin + uv @ self.axes

    def distance(self, x):
        return (x - self.closest(x)).norm(dim=1)

    def normal(self, x):
        return self.n.expand_as(x)


def load_surface(root, entry, backend, clamp, device):
    spec = entry.get("surface", {})
    if spec.get("type") == "plane":
        return PlaneSurface(spec, device)
    if backend == "grid":
        return GridSurface(Path(root) / entry["distance"], clamp, device)
    if backend == "mesh":
        if spec.get("type") != "mesh":
            raise ValueError(
                "mesh queries need a prepared surface archive; prepare the prior again in a new directory"
            )
        return MeshSurface(Path(root) / spec["file"], device)
    raise ValueError(f"unknown surface backend: {backend}")
