"""Match Gaussian density to fixed mesh samples; test with pytest tests/test_density_prior.py."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def rotation(quaternion):
    w, x, y, z = torch.nn.functional.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), -1).reshape(-1, 3, 3)


def block_density(points, means, scales, quaternions, opacity):
    local = torch.einsum("qni,nij->qnj", points[:, None] - means, rotation(quaternions))
    exponent = -0.5 * (local / scales.clamp_min(1e-6)).square().sum(-1)
    return (exponent.exp() * opacity.reshape(-1)).sum(-1)


def gaussian_density(points, means, scales, quaternions, opacity, chunk=32768):
    """Sum all scene Gaussians. Chunking changes memory use, not spatial support."""
    value = points.new_zeros(len(points))
    for start in range(0, len(means), chunk):
        args = [points, *[p[start:start+chunk] for p in (means, scales, quaternions, opacity)]]
        if torch.is_grad_enabled():
            value = value + checkpoint(block_density, *args, use_reentrant=False,
                                       preserve_rng_state=False)
        else:
            value = value + block_density(*args)
    return value


RADII = (0.03, 0.06, 0.125, 0.25, 0.5)  # voxel sizes [m] of the culling tiers; Gaussians larger than the last are summed densely


def pair_density(points, means, rotations, scales, opacity, p, g):
    local = torch.einsum("qi,qij->qj", points[p] - means[g], rotations[g])
    exponent = -0.5 * (local / scales[g].clamp_min(1e-6)).square().sum(-1)
    return torch.zeros(len(points), dtype=points.dtype, device=points.device).index_add(0, p, exponent.exp() * opacity.reshape(-1)[g])


def culled_density(points, means, scales, quaternions, opacity, sigmas=3.5, chunk=2_000_000):
    """gaussian_density restricted to the Gaussians whose centre lies within `sigmas` of their largest scale from the
    point: a Gaussian is binned by its extent into a voxel tier of size >= sigmas * max scale, and a point gathers
    the 27 cells around it in every tier, so every dropped Gaussian contributes less than exp(-sigmas^2 / 2)."""
    extent = sigmas * scales.detach().max(-1).values
    tier = torch.bucketize(extent, torch.tensor(RADII, device=extent.device), right=False)  # RADII[tier] >= extent
    value = points.new_zeros(len(points))
    dense = tier >= len(RADII)
    if dense.any():
        value = value + gaussian_density(points, means[dense], scales[dense], quaternions[dense], opacity[dense])
    rotations = rotation(quaternions)
    origin = torch.minimum(means.detach().min(0).values, points.min(0).values) - 1.0
    for k, radius in enumerate(RADII):
        members = (tier == k).nonzero().squeeze(1)
        if not len(members):
            continue
        cells = torch.floor((means[members].detach() - origin) / radius).long()
        keys = (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
        order = keys.argsort()
        keys = keys[order]
        centre = torch.floor((points - origin) / radius).long()
        shifts = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=points.device)] * 3, indexing="ij"), -1).reshape(-1, 3)
        query = (centre[:, None] + shifts[None]).clamp_min(0)  # [M, 27, 3]
        qkeys = ((query[..., 0] << 42) | (query[..., 1] << 21) | query[..., 2]).reshape(-1)
        start = torch.searchsorted(keys, qkeys, right=False)
        count = torch.searchsorted(keys, qkeys, right=True) - start
        total = int(count.sum())
        if not total:
            continue
        p = torch.arange(len(points), device=points.device).repeat_interleave(27).repeat_interleave(count)
        first = (count.cumsum(0) - count).repeat_interleave(count)
        g = members[order[start.repeat_interleave(count) + torch.arange(total, device=points.device) - first]]
        for a in range(0, total, chunk):
            args = (points, means, rotations, scales, opacity, p[a:a + chunk], g[a:a + chunk])
            value = value + (checkpoint(pair_density, *args, use_reentrant=False, preserve_rng_state=False)
                             if torch.is_grad_enabled() else pair_density(*args))
    return value


class DensityPrior:
    def __init__(self, config_file, device, weight, samples, log):
        self.cfg = json.loads(Path(config_file).read_text())
        cache = Path(config_file).parent / self.cfg["cache"]
        if hashlib.sha256(cache.read_bytes()).hexdigest() != self.cfg["cache_sha256"]:
            raise ValueError("Density cache fingerprint mismatch")
        data = np.load(cache)
        self.points, self.target, self.confidence = [
            torch.as_tensor(data[k], device=device, dtype=torch.float32)
            for k in ("points", "target", "confidence")]
        self.batch = samples
        self.weight = weight
        self.log = Path(log)
        assert self.points.shape[:-1] == self.target.shape == self.confidence.shape
        assert all(torch.isfinite(t).all() for t in (self.points, self.target, self.confidence))
        assert ((self.confidence >= 0) & (self.confidence <= 1)).all()

    def loss(self, gaussians, iteration):
        if not len(self.points):
            return gaussians.get_xyz.sum() * 0
        indices = (torch.arange(self.batch, device=self.points.device)
                   + (iteration - 1) * self.batch) % len(self.points)
        points = self.points[indices].reshape(-1, 3)
        target = self.target[indices].reshape(-1)
        confidence = self.confidence[indices].reshape(-1)
        started = time.perf_counter()
        rho = culled_density(points, gaussians.get_xyz, gaussians.get_scaling,
                             gaussians.get_rotation, gaussians.get_opacity)
        # Fixed sample denominator: low confidence must reduce absolute influence.
        loss = (confidence * (rho - target).square()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite density loss at {iteration}")
        if iteration == 1 or iteration % 100 == 0:
            entry = dict(iteration=iteration, raw=float(loss.detach()), weight=self.weight,
                         mean_confidence=float(confidence.mean()), mean_density=float(rho.detach().mean()),
                         gaussians=len(gaussians.get_xyz), points=len(points), seconds=round(time.perf_counter() - started, 4))
            with self.log.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print("DENSITY", json.dumps(entry), flush=True)
        return self.weight * loss
