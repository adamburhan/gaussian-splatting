"""Density priors from fixed mesh samples: DensityPrior matches the density at points, SummedPrior sums the opacity
along normals."""
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


def line_opacity(origins, normals, tolerance, reach, end, means, scales, quaternions, opacity, fade=True, sigmas=SIGMAS, block=2**24):
    """Per line (origin + t normal), the opacity it meets at the object and in free space. A Gaussian counts with its
    opacity times its peak on the line, over the part of its extent along the line that lies at t in [-tolerance, end];
    its share goes from the object to free space as t goes from tolerance to reach. Without `fade` the opacity met in
    free space gives the opacities no gradient: a Gaussian leaves by moving in or thinning. The Gaussians given are
    the candidates near the lines."""
    with torch.no_grad():  # the (line, Gaussian) pairs that can count: the centre lies within `sigmas` scales of the stretch
        centre = origins.mean(0)  # centred coordinates keep the products exact enough in float32
        o, m, size = origins - centre, means - centre, sigmas * scales.max(-1).values
        base = (o * normals).sum(-1)
        line, g = [], []
        step = max(1, block // max(len(m), 1))
        for first in range(0, len(o), step):
            rows = slice(first, first + step)
            along = normals[rows] @ m.T - base[rows, None]
            near = ((torch.cdist(o[rows], m).square() - along.square() <= size.square())
                    & (along >= -tolerance[rows, None] - size) & (along <= end[rows, None] + size))
            found = near.nonzero(as_tuple=True)
            line.append(found[0] + first)
            g.append(found[1])
        line, g = torch.cat(line), torch.cat(g)
    rotations, inverse = rotation(quaternions[g]), 1 / scales[g].clamp_min(1e-6)
    offset = torch.einsum("pi,pij->pj", origins[line] - means[g], rotations) * inverse
    direction = torch.einsum("pi,pij->pj", normals[line], rotations) * inverse
    a = direction.square().sum(-1)
    t = -(direction * offset).sum(-1) / a  # where on the line the Gaussian peaks [m from the surface]
    peak = torch.exp(-0.5 * (offset.square().sum(-1) - a * t.square()).clamp_min(0))
    # the Gaussian's extent along the line, at three points of its profile there (the centre and +-sqrt(3) widths,
    # weights 2/3 and 1/6): the part of it that sticks out of the tolerance counts as free space
    at = t[:, None] + a.rsqrt()[:, None] * t.new_tensor([-3 ** 0.5, 0.0, 3 ** 0.5])
    counted = t.new_tensor([1 / 6, 2 / 3, 1 / 6]) * ((at >= -tolerance[line, None]) & (at <= end[line, None]))
    outside = ((at - tolerance[line, None]) / (reach[line] - tolerance[line])[:, None]).clamp(0, 1)
    met = opacity.reshape(-1)[g]
    zeros = torch.zeros(len(origins), dtype=means.dtype, device=means.device)
    # detached: a Gaussian leaves free space by fading, moving in or thinning along the line, not by dodging the lines
    leaving = met if fade else met.detach()
    return (zeros.index_add(0, line, met * peak * (counted * (1 - outside)).sum(-1)),
            zeros.index_add(0, line, leaving * peak.detach() * (counted * outside).sum(-1)))


class SummedPrior(DensityPrior):
    """Along the normal through each surface sample of the prior (prior/density.py sample_lines), the opacity met within
    the tolerance of the surface must reach 1 and the opacity met in the free stretch beyond it must be 0: per normal,
    the squared shortfall of the first plus the second. `pull_only`: the second moves and thins Gaussians, never fades them."""

    def __init__(self, config_file, device, weight, samples, log, pull_only=False):
        self.cfg = json.loads(Path(config_file).read_text())
        cache = Path(config_file).parent / self.cfg["cache"]
        if hashlib.sha256(cache.read_bytes()).hexdigest() != self.cfg["cache_sha256"]:
            raise ValueError("Density cache fingerprint mismatch")
        data = np.load(cache)
        self.points, self.normals, self.tolerance, self.reach, self.end, self.confidence = [
            torch.as_tensor(data[k], device=device, dtype=torch.float32)
            for k in ("points", "normals", "tolerance", "reach", "end", "weight")]
        self.object = torch.as_tensor(data["object"], device=device, dtype=torch.long)
        self.batch, self.weight, self.log, self.fade = samples, weight, Path(log), not pull_only
        ends = torch.stack((self.points - self.tolerance[:, None] * self.normals, self.points + self.end[:, None] * self.normals))
        self.boxes = [(ends[:, self.object == k].reshape(-1, 3).min(0).values, ends[:, self.object == k].reshape(-1, 3).max(0).values)
                      for k in range(int(self.object.max()) + 1 if len(self.object) else 0)]
        self.candidates, self.count = None, 0

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
        errors, at_object, in_free = [], [], []
        for k in owner.unique().tolist():
            rows, near = indices[owner == k], self.candidates[k]
            found, free = line_opacity(self.points[rows], self.normals[rows], self.tolerance[rows], self.reach[rows], self.end[rows],
                                       means[near], scales[near], quaternions[near], opacity[near], self.fade)
            errors.append(self.confidence[rows] * (torch.relu(1 - found).square() + free))
            at_object.append(found.detach())
            in_free.append(free.detach())
        loss = torch.cat(errors).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite density loss at {iteration}")
        if iteration == 1 or iteration % 100 == 0:
            entry = dict(iteration=iteration, raw=float(loss.detach()), weight=self.weight,
                         at_object=float(torch.cat(at_object).mean()), in_free=float(torch.cat(in_free).mean()),
                         gaussians=len(means), lines=len(indices), seconds=round(time.perf_counter() - started, 4))
            with self.log.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print("DENSITY", json.dumps(entry), flush=True)
        return self.weight * loss


def density_prior(config_file, *arguments, pull_only=False):
    """The loss a cache was prepared for: SummedPrior for normals, DensityPrior for points with targets."""
    cfg = json.loads(Path(config_file).read_text())
    lines = "normals" in np.load(Path(config_file).parent / cfg["cache"]).files
    return SummedPrior(config_file, *arguments, pull_only=pull_only) if lines else DensityPrior(config_file, *arguments)
