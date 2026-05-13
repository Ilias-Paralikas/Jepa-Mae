from __future__ import annotations
import torch
import torch.nn as nn
from .modules.linear_layer import LinearLayer


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        embed_dim: int,
        hidden_dims: list[int] | None = None,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.projection = LinearLayer(patch_dim, embed_dim, hidden_dims, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C,patch_dim) — patchified image
        Returns:
            (B, N, embed_dim)
        """
        return self.projection(x)
