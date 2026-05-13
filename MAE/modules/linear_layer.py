from __future__ import annotations
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int] | None = None,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        dims = [in_dim] + (hidden_dims or []) + [out_dim]

        layers = []
        if norm is not None:
            layers.append(norm)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())


        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, in_dim) -> (B, N, out_dim)
        return self.net(x)
