"""Interval shape prior on the Gaussians' spatial distribution (--shape_prior <json>).

A region selects Gaussians by centre (axis-aligned box, re-evaluated every iteration) and constrains
their mass along a unit normal n to an allowed interval [a, b] in the coordinate n^T x. With
H_i ~ N(n^T mu_i, n^T Sigma_i n) the penalty is E[ReLU(a - H_i)^2 + ReLU(H_i - b)^2] in closed form:
E[(a - H)^2 1{H < a}] = (t^2 + v) Phi(t / s) + t s phi(t / s), t = a - m, v = s^2, and symmetrically
above b. A Gaussian centred in the interval but thick across it still pays through v. The Gaussian's
tails mean the interval is not an exact zero-gradient zone; the pressure is negligible several
standard deviations inside.

Loss = sum over regions of mean over selected Gaussians of o_i * penalty_i / scale^2, an
opacity-weighted spatial penalty (the image contribution also depends on footprint and occlusion).
`scale` is a fixed normalisation length so the interval width and the loss strength can be varied
independently. Regions are summed, so each has equal nominal weight and overlapping regions stack.

mode "full" is the expectation above; mode "centre" drops the covariance, ReLU(a - m)^2 + ReLU(m - b)^2,
the ablation isolating surface positioning from control of the spread. detach_opacity=True stops the
loss from lowering opacity, so only geometry can resolve a violation.

JSON: {"regions": [{"name": ..., "select_min": [x, y, z], "select_max": [x, y, z],
                     "normal": [nx, ny, nz], "interval": [a, b], "scale": 0.01}, ...]}
"""

import json
import math

import torch

VAR_FLOOR = 1e-12  # m^2; the same floored variance is used everywhere so the zero-variance limit is exact


def load_shape_prior(path, device="cuda"):
    regions = []
    for r in json.load(open(path))["regions"]:
        lo, hi = torch.tensor(r["select_min"], dtype=torch.float32), torch.tensor(r["select_max"], dtype=torch.float32)
        n = torch.tensor(r["normal"], dtype=torch.float32)
        a, b = (float(v) for v in r["interval"])
        scale = float(r.get("scale", (b - a) / 2))
        if not (torch.isfinite(lo).all() and torch.isfinite(hi).all() and torch.isfinite(n).all() and math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"shape prior region {r['name']}: non-finite values")
        if not (lo < hi).all():
            raise ValueError(f"shape prior region {r['name']}: select_min must be below select_max")
        if abs(n.norm().item() - 1) > 1e-6:
            raise ValueError(f"shape prior region {r['name']}: normal must be a unit vector (interval is in n^T x)")
        if not a < b or scale <= 0:
            raise ValueError(f"shape prior region {r['name']}: need a < b and scale > 0")
        regions.append(dict(name=r["name"], select_min=lo.to(device), select_max=hi.to(device),
                            normal=n.to(device), interval=(a, b), scale=scale))
    return regions


def _one_sided(t, var):
    """E[ReLU(T)^2] for T ~ N(t, var) = (t^2 + v) Phi(z) + t s phi(z), z = t / s, in double with the
    complementary error function so the deep negative tail stays accurate and non-negative."""
    dtype = t.dtype
    t = t.double()
    v = var.double().clamp_min(VAR_FLOOR)
    s = v.sqrt()
    z = t / s
    cdf = 0.5 * torch.erfc(-z / math.sqrt(2.0))
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    return ((t.square() + v) * cdf + t * s * pdf).clamp_min(0.0).to(dtype)


def interval_penalty(m, var, a, b, mode="full"):
    """Per-Gaussian penalty for H ~ N(m, var) against [a, b]; mode "centre" ignores var."""
    if mode == "centre":
        return torch.relu(a - m).square() + torch.relu(m - b).square()
    return _one_sided(a - m, var) + _one_sided(m - b, var)


def normal_variance(covariance, n):
    """n^T Sigma n from the packed (xx, xy, xz, yy, yz, zz) rows of GaussianModel.get_covariance."""
    c = covariance
    return (c[:, 0] * n[0] * n[0] + c[:, 3] * n[1] * n[1] + c[:, 5] * n[2] * n[2]
            + 2 * (c[:, 1] * n[0] * n[1] + c[:, 2] * n[0] * n[2] + c[:, 4] * n[1] * n[2]))


def shape_prior_loss(xyz, covariance, opacity, regions, mode="full", detach_opacity=False):
    """xyz [N, 3]; covariance [N, 6] packed; opacity [N] or [N, 1]. Scalar; 0 when nothing is selected."""
    opacity = opacity.reshape(-1)
    if detach_opacity:
        opacity = opacity.detach()
    total = xyz.new_zeros(())
    for r in regions:
        inside = ((xyz >= r["select_min"]) & (xyz <= r["select_max"])).all(1)
        if not inside.any():
            continue
        n = r["normal"]
        a, b = r["interval"]
        penalty = interval_penalty(xyz[inside] @ n, normal_variance(covariance[inside], n), a, b, mode)
        total = total + (opacity[inside] * penalty).mean() / r["scale"] ** 2
    return total


@torch.no_grad()
def shape_prior_stats(xyz, covariance, opacity, regions):
    """Per region: count, mean centre excursion outside [a, b] [m], mean normal std [m], mean opacity."""
    opacity = opacity.reshape(-1)
    stats = {}
    for r in regions:
        inside = ((xyz >= r["select_min"]) & (xyz <= r["select_max"])).all(1)
        if not inside.any():
            stats[r["name"]] = dict(count=0)
            continue
        n = r["normal"]
        a, b = r["interval"]
        m = xyz[inside] @ n
        excursion = torch.relu(a - m) + torch.relu(m - b)
        std = normal_variance(covariance[inside], n).clamp_min(0).sqrt()
        stats[r["name"]] = dict(count=int(inside.sum()), excursion_m=excursion.mean().item(),
                                normal_std_m=std.mean().item(), opacity=opacity[inside].mean().item())
    return stats
