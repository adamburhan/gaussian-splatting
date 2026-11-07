"""Optional fixed influence carried by prepared completion samples."""

import numpy as np
from plyfile import PlyData


def seed_weights(path):
    vertex = PlyData.read(str(path))["vertex"]
    if "prior_weight" not in vertex.data.dtype.names:
        return np.ones(len(vertex), np.float32)
    weights = np.asarray(vertex["prior_weight"], np.float32)
    if not np.isfinite(weights).all() or ((weights <= 0) | (weights > 1)).any():
        raise ValueError(f"invalid seed weights: {path}")
    return weights
