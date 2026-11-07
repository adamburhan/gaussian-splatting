"""Opacity-aware surface compatibility; independent of cameras and CUDA."""

import torch


def compatibility(alpha, depth, target, sigma):
    """Reference for front-to-back ray samples [..., N], including escape mass.

    Alpha is the pixel footprint, not the Gaussian's raw opacity. Samples must
    already be sorted. The native renderer applies its usual alpha thresholds.
    """
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(alpha[..., :1]), 1 - alpha), -1), -1
    )[..., :-1]
    kernel = torch.exp(-0.5 * ((depth - target[..., None]) / sigma[..., None]).square())
    return (transmittance * alpha * kernel).sum(-1)


def surface_likelihood(compatibility, confidence, integration_weight):
    """Fixed quadrature weights, NOT renormalized by confidence or opacity.

    Confidence must be fixed in [0,1); cache validation enforces this. A missing
    object has finite cost -log(1-confidence), not zero cost.
    """
    return (-torch.log1p(-confidence * (1 - compatibility)) * integration_weight).sum()
