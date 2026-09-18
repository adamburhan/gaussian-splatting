"""Interval shape prior on the Gaussians' spatial distribution (--shape_prior <json>).

A region selects Gaussians by centre (axis-aligned box) and constrains their mass along a normal n to
an allowed interval [a, b]: with H_i ~ N(n^T mu_i, n^T Sigma_i n) the penalty is
E[ReLU(a - H_i)^2 + ReLU(H_i - b)^2], closed form via the normal CDF and PDF, so a Gaussian that is
centred in the interval but thick across it still pays. The penalty is weighted by opacity (mass that
renders), normalised by the squared half-width (1 = one half-width of violation), and averaged over
the selected Gaussians. It acts on parameters directly, whether or not a training camera sees them.

JSON: {"regions": [{"name": ..., "select_min": [x, y, z], "select_max": [x, y, z],
                     "normal": [nx, ny, nz], "interval": [a, b]}, ...]}
"""

import json
import math

import torch


def load_shape_prior(path, device="cuda"):
    regions = []
    for r in json.load(open(path))["regions"]:
        n = torch.tensor(r["normal"], dtype=torch.float32, device=device)
        regions.append(dict(
            name=r["name"],
            select_min=torch.tensor(r["select_min"], dtype=torch.float32, device=device),
            select_max=torch.tensor(r["select_max"], dtype=torch.float32, device=device),
            normal=n / n.norm(), interval=tuple(float(v) for v in r["interval"])))
    return regions


def _one_sided(t, var):
    """E[ReLU(T)^2] for T ~ N(t, var) written as s^2 [(1 + z^2) Phi(z) + z phi(z)], z = t / s."""
    s = var.clamp_min(1e-12).sqrt()
    z = t / s
    return var * ((1 + z * z) * torch.special.ndtr(z) + z * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))


def interval_penalty(m, var, a, b):
    """Per-Gaussian E[ReLU(a - H)^2 + ReLU(H - b)^2] for H ~ N(m, var)."""
    return _one_sided(a - m, var) + _one_sided(m - b, var)


def shape_prior_loss(xyz, covariance, opacity, regions):
    """xyz [N, 3]; covariance [N, 6] packed (xx, xy, xz, yy, yz, zz) as GaussianModel.get_covariance;
    opacity [N] or [N, 1]. Returns a scalar; 0 when no Gaussian is selected."""
    opacity = opacity.reshape(-1)
    total = xyz.new_zeros(())
    for r in regions:
        inside = ((xyz >= r["select_min"]) & (xyz <= r["select_max"])).all(1)
        if not inside.any():
            continue
        n = r["normal"]
        c = covariance[inside]
        var = (c[:, 0] * n[0] * n[0] + c[:, 3] * n[1] * n[1] + c[:, 5] * n[2] * n[2]
               + 2 * (c[:, 1] * n[0] * n[1] + c[:, 2] * n[0] * n[2] + c[:, 4] * n[1] * n[2]))
        a, b = r["interval"]
        penalty = interval_penalty(xyz[inside] @ n, var, a, b)
        total = total + (opacity[inside] * penalty).mean() / ((b - a) / 2) ** 2
    return total
