"""Gaussian-to-surface penalties in squared metres.

Ownership and evidence live in shape_prior; geometry queries live in surfaces.
The surface loss is a local tangent approximation, not rendered-surface agreement.
"""

import torch

from utils.shape_prior import ShapePrior, CDIST_BLOCK


def quaternion_to_matrix(q):
    """[N, 4] (w, x, y, z) -> [N, 3, 3], as the fork's build_rotation but on any device."""
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


# --- losses ------------------------------------------------------------------------

def _zero(xyz):
    return xyz.sum() * 0.0, 0


def _weights(opacity, m, mode):
    """Per-Gaussian weight of the penalty: 1, or the opacity (constant or trainable)."""
    if mode == "none":
        return 1.0
    w = opacity[m].reshape(-1)
    return w.detach() if mode == "detached" else w


def _penalty(d, robust, scale):
    """d^2, or Huber: d^2 within `scale`, linear beyond."""
    if robust == "huber":
        return torch.where(d.abs() <= scale, d ** 2, scale * (2 * d.abs() - scale))
    return d ** 2


def centre(prior, xyz, scaling, rotation, owner, opacity=None, opacity_weight="none", robust="none", robust_scale=0.03, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    d = prior.distance(xyz[m], owner[m])
    return (prior.weight(xyz[m], owner[m]) * _weights(opacity, m, opacity_weight) * _penalty(d, robust, robust_scale)).mean(), int(m.sum())


def surface(prior, xyz, scaling, rotation, owner, opacity=None, opacity_weight="none", spread=1.0, tolerance=0.005,
            robust="none", robust_scale=0.03, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    d = prior.distance(xyz[m], owner[m])
    n = prior.normal(xyz[m].detach(), owner[m])
    v = (quaternion_to_matrix(rotation[m]).transpose(1, 2) @ n[..., None])[..., 0]  # normal in the Gaussian's frame
    variance = (scaling[m] ** 2 * v ** 2).sum(1)  # n^T R S^2 R^T n
    penalty = _penalty(d, robust, robust_scale) + spread * torch.relu(variance - tolerance ** 2)
    return (prior.weight(xyz[m], owner[m]) * _weights(opacity, m, opacity_weight) * penalty).mean(), int(m.sum())


def band(prior, xyz, scaling, rotation, owner, opacity=None, opacity_weight="none", delta=0.02, samples=16, **_):
    m = owner >= 0
    if not m.any():
        return _zero(xyz)
    R = quaternion_to_matrix(rotation[m])
    eps = torch.randn(samples, int(m.sum()), 3, device=xyz.device, dtype=xyz.dtype)
    X = xyz[m] + (R @ (scaling[m] * eps)[..., None])[..., 0]  # [S, M, 3]
    d = prior.distance(X.reshape(-1, 3), owner[m].repeat(samples)).reshape(samples, -1)
    return (prior.weight(xyz[m], owner[m]) * _weights(opacity, m, opacity_weight) * (torch.relu(d - delta) ** 2).mean(0)).mean(), int(m.sum())


LOSSES = {"centre": centre, "surface": surface, "band": band}


def object_fit(prior, observed, R, t, c, samples=2048, scale=0.03):
    """The pose's evidence from the visible side: Huber distance of each object's observed points (the map cloud,
    world frame) to its mesh under the pose (R, t, c), differentiable in the pose. Mean over objects with points."""
    total, count, per_object = None, 0, {}
    for k, q in enumerate(observed):
        if q is None or not len(q):
            continue
        if len(q) > samples:
            q = q[torch.randint(len(q), (samples,), device=q.device)]
        d = prior.surfaces[k].distance((q - c[k] - t[k]) @ R[k] + c[k])
        term = torch.where(d < scale, 0.5 * d**2 / scale, d - 0.5 * scale).mean()
        per_object[k] = term
        total = term if total is None else total + term
        count += 1
    return (total / count if count else R.sum() * 0.0), count, per_object


@torch.no_grad()
def seed_neighbours(xyz, owner, completed, radius):
    """Index pairs (seed, ordinary Gaussian) of the same object within radius, as two [P] tensors."""
    seeds, neighbours = [], []
    for k in owner[completed & (owner >= 0)].unique():
        s, o = ((owner == k) & completed).nonzero()[:, 0], ((owner == k) & ~completed).nonzero()[:, 0]
        if not len(o):
            continue
        for i in s.split(max(1, CDIST_BLOCK // len(o))):
            a, b = (torch.cdist(xyz[i], xyz[o]) < radius).nonzero(as_tuple=True)
            seeds.append(i[a])
            neighbours.append(o[b])
    empty = torch.zeros(0, dtype=torch.long, device=xyz.device)
    return (torch.cat(seeds), torch.cat(neighbours)) if seeds else (empty, empty)


def seed_colour(prior, xyz, features_dc, owner, completed, radius=0.10):
    """Mean over the seeds with neighbours of the squared DC difference to their neighbours' mean, and that count."""
    if prior.pairs_key != (prior.last_refresh, xyz.shape[0]):
        prior.pairs = seed_neighbours(xyz.detach(), owner, completed, radius)
        prior.pairs_key = (prior.last_refresh, xyz.shape[0])
    seed, neighbour = prior.pairs
    if not len(seed):
        return _zero(features_dc)
    count = torch.bincount(seed, minlength=xyz.shape[0])
    total = torch.zeros_like(features_dc).index_add_(0, seed, features_dc[neighbour].detach())
    has = count > 0
    return ((features_dc[has] - total[has] / count[has, None, None]) ** 2).sum(-1).mean(), int(has.sum())


def target_mask(gaussians, targets):
    """Which Gaussians a shape loss may act on: ordinary | completion | all."""
    if targets == "all":
        return torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
    return gaussians.completed if targets == "completion" else ~gaussians.completed
