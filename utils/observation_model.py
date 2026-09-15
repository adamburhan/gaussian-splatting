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

D is the mean of the ray's weight distribution w_i = alpha_i T_i over the Gaussians' inverse depths
u_i, so any loss on D is blind to how the mass is arranged: a pixel split between the near and the far
surface renders a "phantom" middle depth without any Gaussian being there. The distributional arms put
the observation model on every Gaussian and take the expectation under the ray distribution,

    unimodal_dist   sqrt( E_w[(u - s)^2] )                        = |D - s| for a single surface
    bimodal_dist    sqrt( E_w[(u - h_near)^2 (u - h_far)^2] ) / |h_near - h_far|   in the band

which needs only the moments M_k = sum_i w_i u_i^k. Those come out of the unmodified rasterizer by
rendering u^k as colours (render_moments), so gradients reach positions through u and opacities
through w. The missing mass 1 - sum_i w_i is the background, an atom at u = 0 (infinitely far): that is
what the native D = sum_i w_i u_i already assumes, and it is what makes an under-covered pixel pay for
its transparency instead of escaping supervision. Units are inverse depth, like the native residual,
so the native weight schedule applies.

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
    Both pixels across every jump are marked, so the band spans the ramp plus one pixel on each side
    before the dilation. Every band pixel's window must reach both surfaces: with a ramp of r pixels
    and d pixels of dilation that is window >= 2 (r + d) + 3; 13 covers the 2-3 pixel ramps of the
    upsampled iPhone LiDAR. Hypotheses are float32 inverse depth (1/m), 0 outside the band."""
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

    # windows only at edge pixels (a few percent of the image), otherwise the percentiles dominate loading
    pad = window // 2
    padded = np.pad(inv, pad, constant_values=np.nan)
    patches = np.lib.stride_tricks.sliding_window_view(padded, (window, window))[edge]
    hyp_near = np.zeros(depth.shape, dtype=np.float32)
    hyp_far = np.zeros(depth.shape, dtype=np.float32)
    band = np.zeros(depth.shape, dtype=bool)
    if edge.any():
        # nearest-rank percentiles over the valid samples of each window (NaN sorts last)
        ordered = np.sort(patches.reshape(len(patches), -1), axis=-1)
        count = np.isfinite(ordered).sum(-1)
        rows = np.arange(len(ordered))
        near = ordered[rows, np.round(0.9 * (count - 1)).clip(0).astype(int)]  # high inverse depth = near
        far = ordered[rows, np.round(0.1 * (count - 1)).clip(0).astype(int)]
        with np.errstate(all="ignore"):
            gap = 1.0 / far - 1.0 / near  # metres between the two surfaces
            meaningful = np.isfinite(gap) & ((gap > min_gap_abs) | (gap > min_gap_rel / near))
        rows, cols = np.nonzero(edge)
        band[rows[meaningful], cols[meaningful]] = True
        hyp_near[rows[meaningful], cols[meaningful]] = near[meaningful]
        hyp_far[rows[meaningful], cols[meaningful]] = far[meaningful]
    return band, hyp_near, hyp_far


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


DISTRIBUTIONAL = ("unimodal_dist", "bimodal_dist")
NEEDS_HYPOTHESES = ("masked", "bimodal", "bimodal_dist")


def inverse_depths(cam, xyz):
    """View-space inverse depth of every Gaussian centre. The rasterizer culls z < 0.2, so clamping
    there keeps u^4 bounded for the moment renders."""
    view = cam.world_view_transform  # stored transposed: camera coordinates = [x, 1] @ view
    z = xyz @ view[:3, 2] + view[3, 2]
    return 1.0 / z.clamp(min=0.2)


def render_moments(cam, gaussians, pipe, separate_sh, model):
    """Moments M_k = sum_i w_i u_i^k of the ray distribution over inverse depth, rendered as colours over
    a black background: M1, M2 for unimodal_dist; M1..M4 for bimodal_dist. Values are [H, W]. The
    background completes the distribution at u = 0, so no opacity channel is needed."""
    from gaussian_renderer import render  # training-environment import, kept out of the module scope

    u = inverse_depths(cam, gaussians.get_xyz)
    black = torch.zeros(3, dtype=torch.float32, device=u.device)

    def pass_(channels):
        return render(cam, gaussians, pipe, black, override_color=torch.stack(channels, 1), separate_sh=separate_sh)["render"]

    if model == "unimodal_dist":
        m = pass_([u, u * u, u * u])
        return dict(M1=m[0], M2=m[1])
    m = pass_([u, u * u, u * u * u])
    n = pass_([u * u * u * u, u, u])
    return dict(M1=m[0], M2=m[1], M3=m[2], M4=n[0])


def distributional_residual(moments, cam, model):
    """Per-pixel residual [1, H, W] of the distributional observation models from rendered moments.
    The expectation runs over the completed ray distribution (background at u = 0), so the zeroth
    moment is 1 and a pixel pays for the mass it lacks."""
    M1, M2 = moments["M1"], moments["M2"]
    device = M1.device
    target = cam.invdepthmap.to(device)[0]
    valid = cam.depth_mask.to(device)[0]

    def at_render_size(t):
        return F.interpolate(t.to(device)[None], size=M1.shape, mode="nearest")[0, 0]

    spread = M2 - 2 * target * M1 + target * target  # E[(u - s)^2]
    unimodal = torch.sqrt(spread.clamp(min=0) + 1e-8)
    if model == "unimodal_dist":
        return (unimodal * valid)[None]
    if model != "bimodal_dist":
        raise ValueError(f"unknown distributional observation model {model!r}")
    band = at_render_size(cam.band)
    a, b = at_render_size(cam.hyp_near), at_render_size(cam.hyp_far)
    M3, M4 = moments["M3"], moments["M4"]
    # E[(u - a)^2 (u - b)^2] expanded in the moments
    quartic = M4 - 2 * (a + b) * M3 + (a * a + 4 * a * b + b * b) * M2 - 2 * a * b * (a + b) * M1 + a * a * b * b
    bimodal = torch.sqrt(quartic.clamp(min=0) + 1e-8) / (b - a).abs().clamp(min=1e-3)
    return (torch.where(band > 0, bimodal, unimodal) * valid)[None]
