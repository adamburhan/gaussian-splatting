"""Interpolate the current method’s distance grids; test with pytest tests/test_shape_loss.py."""

import numpy as np
import torch
import torch.nn.functional as F


class GridSurface:
    def __init__(self, path, clamp, device):
        with np.load(path) as g:
            self.origin = torch.tensor(g["origin"], device=device)
            self.spacing, self.clamp = float(g["spacing"]), clamp
            self.values = torch.tensor(g["values"].astype(np.float32), device=device)[
                None, None
            ]
            self.inside = (
                torch.tensor(g["inside"].astype(np.float32), device=device)[None, None]
                if "inside" in g
                else None
            )
        self.diagonal = float(
            np.linalg.norm(
                (np.array(self.values.shape[2:]) - 1) * self.spacing - 2 * clamp
            )
        )

    def sample(self, x, signed=False):
        shape = x.new_tensor(self.values.shape[2:])
        u = (x - self.origin) / self.spacing / (shape - 1) * 2 - 1
        grid = u.flip(-1)[None, None, None]
        d = F.grid_sample(
            self.values,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0, 0, 0]
        if signed and self.inside is not None:
            d = d * (
                1
                - 2
                * F.grid_sample(
                    self.inside,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )[0, 0, 0, 0]
            )
        return torch.where((u.abs() > 1).any(-1), torch.full_like(d, self.clamp), d)

    def distance(self, x):
        return self.sample(x)

    def normal(self, x):
        h = self.spacing
        g = torch.stack(
            [
                self.sample(x + h * e, True) - self.sample(x - h * e, True)
                for e in torch.eye(3, device=x.device)
            ],
            1,
        )
        return F.normalize(g, dim=1, eps=1e-9)
