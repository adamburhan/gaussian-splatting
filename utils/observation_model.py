"""Depth observation models for the inverse-depth L1 term in train.py.

The native loss treats the sensor value at every pixel as one target. Near a depth discontinuity the
sensor (an upsampled 256x192 LiDAR map) blends the two surfaces into a ramp that lies on neither. The
observation models below share the native residual outside the discontinuity band and differ only
inside it:

    unimodal   |D - s|                       native
    masked     0                             no supervision in the band
    bimodal    min(|D - h_near|, |D - h_far|) either local surface, never the ramp

with D the rendered inverse depth, s the sensor inverse depth and h_near >= h_far the inverse depths of
the two surfaces around the pixel. The caller keeps the native reduction (mean over all pixels) and
weight schedule, so the arms differ only in where loss mass exists.

This module depends on numpy and torch only so it can be unit-tested outside the training environment.
"""

import numpy as np
import torch
import torch.nn.functional as F


def extract_hypotheses(depth, threshold, window=13, dilation=1, min_gap_abs=0.1, min_gap_rel=0.05):
    """Two-surface hypotheses from a metric depth map [h, w] (metres, 0 = no reading), at its own
    resolution. Returns (band, hyp_near, hyp_far): band marks pixels next to a discontinuity whose
    local near/far gap is meaningful; the hypotheses are the p90 / p10 of inverse depth in a
    window x window neighbourhood (percentiles rather than min/max so flying pixels do not count).
    Every band pixel's window must reach both surfaces: with a ramp of r pixels across the edge and
    d pixels of dilation that is window >= 2 (r + d) + 3, i.e. 13 for the 2-3 pixel ramps of the
    upsampled iPhone LiDAR. Hypotheses are float32 inverse depth (1/m), 0 where the window has no reading."""
    valid = depth > 0
    inv = np.where(valid, 1.0 / np.maximum(depth, 1e-3), np.nan).astype(np.float32)
    logd = np.log(np.where(valid, depth, np.nan))

    # a jump between neighbouring pixels marks both of them (log ratio, so relative to depth)
    edge = np.zeros(depth.shape, dtype=bool)
    jump_x = np.abs(np.diff(logd, axis=1)) > threshold
    jump_y = np.abs(np.diff(logd, axis=0)) > threshold
    edge[:, :-1] |= jump_x
    edge[:, 1:] |= jump_x
    edge[:-1, :] |= jump_y
    edge[1:, :] |= jump_y
    for _ in range(dilation):
        grown = edge.copy()
        grown[1:, :] |= edge[:-1, :]
        grown[:-1, :] |= edge[1:, :]
        grown[:, 1:] |= edge[:, :-1]
        grown[:, :-1] |= edge[:, 1:]
        edge = grown

    pad = window // 2
    padded = np.pad(inv, pad, constant_values=np.nan)
    patches = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    patches = patches.reshape(*inv.shape, window * window)
    with np.errstate(all="ignore"):
        hyp_far, hyp_near = np.nanpercentile(patches, [10, 90], axis=-1)  # low inverse depth = far
        gap = 1.0 / hyp_far - 1.0 / hyp_near  # metres between the two surfaces
        meaningful = (gap > min_gap_abs) | (gap > min_gap_rel / hyp_near)
    band = edge & np.isfinite(gap) & meaningful
    return band, np.nan_to_num(hyp_near).astype(np.float32), np.nan_to_num(hyp_far).astype(np.float32)


def depth_residual(invdepth, cam, model):
    """Per-pixel residual [1, H, W] of the rendered inverse depth for one observation model, masked by
    the camera's valid-depth mask. The caller reduces it; unimodal reproduces the native expression."""
    device = invdepth.device
    target = cam.invdepthmap.to(device)
    valid = cam.depth_mask.to(device)
    if model == "unimodal":
        return torch.abs((invdepth - target) * valid)

    def at_render_size(t):
        return F.interpolate(t.to(device)[None], size=invdepth.shape[-2:], mode="nearest")[0]

    band = at_render_size(cam.band)
    unimodal = torch.abs(invdepth - target)
    if model == "masked":
        return unimodal * (1 - band) * valid
    if model == "bimodal":
        near, far = at_render_size(cam.hyp_near), at_render_size(cam.hyp_far)
        bimodal = torch.minimum(torch.abs(invdepth - near), torch.abs(invdepth - far))
        return torch.where(band > 0, bimodal, unimodal) * valid
    raise ValueError(f"unknown observation model {model!r}")
