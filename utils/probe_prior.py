"""Fixed probe caches and object membership. No scene-depth prediction or grid."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

from utils.graphics_utils import getProjectionMatrix
from utils.probe_loss import surface_likelihood


def associate(points, clouds, radius, margin):
    """Nearest observed object, rejecting distant or ambiguous associations."""
    if radius <= 0 or margin < 0:
        raise ValueError(
            "association needs a positive radius and nonnegative ambiguity margin"
        )
    best, second = np.full(len(points), np.inf), np.full(len(points), np.inf)
    owner = np.full(len(points), -1, np.int64)
    for k, cloud in enumerate(clouds):
        if not len(cloud):
            continue
        distance = cKDTree(cloud).query(points)[0]
        nearer = distance < best
        second = np.where(nearer, best, np.minimum(second, distance))
        best = np.minimum(best, distance)
        owner[nearer] = k
    finite = np.isfinite(best)
    ambiguous = np.zeros(len(points), dtype=bool)
    ambiguous[finite] = (second[finite] - best[finite]) < margin
    owner[(best > radius) | ambiguous] = -1
    return owner


def read_owned_cloud(path, labels):
    path = Path(path)
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.stack([vertex[k] for k in ("x", "y", "z")], -1)
    names = json.loads(path.with_suffix(".json").read_text())["objects"]
    mapping = np.array(
        [labels.index(n) if n in labels else -1 for n in names], dtype=np.int64
    )
    ids = np.asarray(vertex["object"], dtype=np.int64)
    if len(ids) and (ids.min() < 0 or ids.max() >= len(mapping)):
        raise ValueError(f"invalid object index in {path}")
    return xyz, mapping[ids]


class ProbePrior:
    def __init__(self, path):
        self.path = Path(path)
        manifest = self.path / "probes.json"
        self.meta = json.loads(manifest.read_text())
        if self.meta["version"] != 1 or self.meta["evidence_split"] != "train":
            raise ValueError("unsupported probe cache or non-training evidence")
        self.labels = [obj["label"] for obj in self.meta["objects"]]
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("probe labels must be unique")
        self.fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for file, digest in self.meta["files"].items():
            if hashlib.sha256((self.path / file).read_bytes()).hexdigest() != digest:
                raise ValueError(f"probe cache changed: {file}; prepare a new cache")
        # Keep caches in host memory; one small image is transferred per step.
        self.probes = []
        for obj in self.meta["objects"]:
            views = []
            for file in obj["views"]:
                if file not in self.meta["files"]:
                    raise ValueError(f"probe file is not fingerprinted: {file}")
                with np.load(self.path / file) as data:
                    view = dict(data)
                d, s, c, w = [
                    view[k]
                    for k in (
                        "target_depth",
                        "position_uncertainty",
                        "confidence",
                        "integration_weight",
                    )
                ]
                if (
                    d.ndim != 2
                    or d.shape[0] != d.shape[1]
                    or not all(np.isfinite(v).all() for v in (d, s, c, w))
                    or (s <= 0).any()
                    or ((c < 0) | (c >= 1)).any()
                    or (w < 0).any()
                    or ((w > 0) & (d <= 0)).any()
                    or any(v.shape != d.shape for v in (s, c, w))
                ):
                    raise ValueError(f"invalid probe values: {file}")
                camera = view["world_to_camera"]
                if (
                    camera.shape != (4, 4)
                    or not np.isfinite(camera).all()
                    or not np.allclose(
                        camera[:3, :3] @ camera[:3, :3].T, np.eye(3), atol=1e-5
                    )
                    or not 0 < float(view["fov"]) < np.pi
                ):
                    raise ValueError(f"invalid probe camera: {file}")
                views.append(view)
            total = sum(float(v["integration_weight"].sum()) for v in views)
            if not views or not np.isclose(total, 1, atol=1e-5):
                raise ValueError(
                    "each object's fixed quadrature weights must sum to one"
                )
            self.probes.append(views)

    def initialize_identity(self, gaussians, completion, radius, margin):
        if "probe_fingerprint" in gaussians.training_state:
            if gaussians.training_state["probe_fingerprint"] != self.fingerprint:
                raise ValueError("checkpoint and probe cache differ")
            if len(gaussians.instance_id) != len(gaussians.get_xyz):
                raise ValueError("checkpoint has no valid instance identities")
            return
        if gaussians.training_state:
            raise ValueError(
                "cannot attach new fixed identities to an older training checkpoint"
            )
        xyz, owners = read_owned_cloud(self.path / "observed.ply", self.labels)
        clouds = [xyz[owners == k] for k in range(len(self.labels))]
        points = gaussians.get_xyz.detach().cpu().numpy()
        ids = associate(points, clouds, radius, margin)
        if completion:
            _, seed_ids = read_owned_cloud(completion, self.labels)
            source = gaussians.completion_index.detach().cpu().numpy()
            seeded = source >= 0
            ids[seeded] = seed_ids[source[seeded]]
        gaussians.instance_id = torch.as_tensor(ids, device=gaussians.get_xyz.device)
        print(
            "probe membership:",
            {name: int((ids == k).sum()) for k, name in enumerate(self.labels)},
        )

    def loss(self, gaussians, pipe, iteration):
        from gaussian_renderer import render_compatibility

        if not self.probes:
            return gaussians.get_xyz.sum() * 0
        # A deterministic cycle: equal object weight, then equal view weight.
        k = (iteration - 1) % len(self.probes)
        views = self.probes[k]
        v = views[((iteration - 1) // len(self.probes)) % len(views)]
        device = gaussians.get_xyz.device
        tensor = lambda key: torch.as_tensor(v[key], device=device, dtype=torch.float32)
        target, sigma = (
            tensor("target_depth")[None],
            tensor("position_uncertainty")[None],
        )
        h, w = target.shape[-2:]
        fov = float(v["fov"])
        view = tensor("world_to_camera").T.contiguous()
        proj = getProjectionMatrix(0.01, 1000.0, fov, fov).T.to(device)
        camera = SimpleNamespace(
            image_height=h,
            image_width=w,
            FoVx=fov,
            FoVy=fov,
            world_view_transform=view,
            full_proj_transform=view @ proj,
            camera_center=view.inverse()[3, :3],
        )
        b = render_compatibility(
            camera, gaussians, pipe, gaussians.instance_id == k, target, sigma
        )
        # Weights already sum to one across ALL views. Correct for choosing one view.
        return len(views) * surface_likelihood(
            b, tensor("confidence")[None], tensor("integration_weight")[None]
        )
