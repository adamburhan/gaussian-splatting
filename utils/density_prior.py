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


SIGMAS = 3.5  # a Gaussian is ignored at a point beyond this many of its scales: it adds less than exp(-SIGMAS^2 / 2)


def pair_density(points, means, rotations, scales, opacity, p, g):
    local = torch.einsum("qi,qij->qj", points[p] - means[g], rotations[g])
    exponent = -0.5 * (local / scales[g].clamp_min(1e-6)).square().sum(-1)
    return torch.zeros(len(points), dtype=points.dtype, device=points.device).index_add(0, p, exponent.exp() * opacity.reshape(-1)[g])


def near_pairs(points, means, rotations, scales, sigmas, block):
    """The (point, Gaussian) pairs that can contribute: the point lies within `sigmas` of the Gaussian's largest scale
    from its centre and within `sigmas` of its smallest scale from its plane. Two dense tests without gradients; every
    pair inside the `sigmas` ellipsoid passes both."""
    with torch.no_grad():
        normal = rotations[torch.arange(len(means), device=means.device), :, scales.argmin(-1)]
        reach, thin = sigmas * scales.max(-1).values, sigmas * scales.min(-1).values
        origin = points.mean(0)  # centred coordinates keep the products exact enough in float32
        x, m = points - origin, means - origin
        height = (m * normal).sum(-1)
        p, g = [], []
        step = max(1, block // max(len(m), 1))
        for start in range(0, len(x), step):
            chunk = x[start:start + step]
            near = (torch.cdist(chunk, m) <= reach) & ((chunk @ normal.T - height).abs() <= thin)
            rows, columns = near.nonzero(as_tuple=True)
            p.append(rows + start)
            g.append(columns)
        return torch.cat(p), torch.cat(g)


def local_density(points, means, scales, quaternions, opacity, sigmas=SIGMAS, block=2**24):
    """gaussian_density without the pairs beyond `sigmas`; the Gaussians given are the candidates near the points."""
    rotations = rotation(quaternions)
    p, g = near_pairs(points, means, rotations, scales, sigmas, block)
    return pair_density(points, means, rotations, scales, opacity, p, g)


class DensityPrior:
    REFRESH = 100  # iterations between candidate updates: the densification interval
    MARGIN = 0.1   # [m] how far a candidate's reach may fall short of an object's probes, for motion between updates

    def __init__(self, config_file, device, weight, samples, log):
        self.cfg = json.loads(Path(config_file).read_text())
        cache = Path(config_file).parent / self.cfg["cache"]
        if hashlib.sha256(cache.read_bytes()).hexdigest() != self.cfg["cache_sha256"]:
            raise ValueError("Density cache fingerprint mismatch")
        data = np.load(cache)
        self.points, self.target, self.confidence = [
            torch.as_tensor(data[k], device=device, dtype=torch.float32)
            for k in ("points", "target", "confidence")]
        self.object = torch.as_tensor(data["object"], device=device, dtype=torch.long)  # per surface sample
        self.batch = samples
        self.weight = weight
        self.log = Path(log)
        assert self.points.shape[:-1] == self.target.shape == self.confidence.shape
        assert all(torch.isfinite(t).all() for t in (self.points, self.target, self.confidence))
        assert ((self.confidence >= 0) & (self.confidence <= 1)).all()
        probes = [self.points[self.object == k].reshape(-1, 3) for k in range(int(self.object.max()) + 1 if len(self.object) else 0)]
        self.boxes = [(p.min(0).values, p.max(0).values) for p in probes]
        self.candidates, self.count = None, 0

    def refresh(self, means, scales):
        """Per object, the Gaussians whose reach may touch the box of its probes before the next update."""
        with torch.no_grad():
            reach = (2 * SIGMAS * scales.max(-1).values + self.MARGIN)[:, None]
            self.candidates = [((means + reach >= low) & (means - reach <= high)).all(-1).nonzero().squeeze(1)
                               for low, high in self.boxes]
            self.count = len(means)

    def loss(self, gaussians, iteration):
        if not len(self.points):
            return gaussians.get_xyz.sum() * 0
        means, scales = gaussians.get_xyz, gaussians.get_scaling
        quaternions, opacity = gaussians.get_rotation, gaussians.get_opacity
        if self.candidates is None or len(means) != self.count or iteration % self.REFRESH == 1:
            self.refresh(means, scales)
        indices = (torch.arange(self.batch, device=self.points.device)
                   + (iteration - 1) * self.batch) % len(self.points)
        owner = self.object[indices]
        started = time.perf_counter()
        errors, densities = [], []
        for k in owner.unique().tolist():
            rows, near = indices[owner == k], self.candidates[k]
            rho = local_density(self.points[rows].reshape(-1, 3), means[near], scales[near], quaternions[near], opacity[near])
            # Fixed sample denominator: low confidence must reduce absolute influence.
            errors.append(self.confidence[rows].reshape(-1) * (rho - self.target[rows].reshape(-1)).square())
            densities.append(rho.detach())
        loss = torch.cat(errors).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite density loss at {iteration}")
        if iteration == 1 or iteration % 100 == 0:
            entry = dict(iteration=iteration, raw=float(loss.detach()), weight=self.weight,
                         mean_confidence=float(self.confidence[indices].mean()), mean_density=float(torch.cat(densities).mean()),
                         gaussians=len(means), points=len(indices) * self.points.shape[1], seconds=round(time.perf_counter() - started, 4))
            with self.log.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print("DENSITY", json.dumps(entry), flush=True)
        return self.weight * loss
