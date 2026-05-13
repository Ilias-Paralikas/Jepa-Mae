from __future__ import annotations
import torch
import torch.nn as nn


class Masking(nn.Module):
    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, N, D) — patch embeddings
            mask: (N_keep,) indices of patches to keep, or None to generate randomly
        Returns:
            visible: (B, N_keep, D) — kept patches
            indices: (N_keep,)      — indices of the kept patches
        """
        B, N, D = x.shape

        if mask is None:
            n_keep = int(N * (1 - self.mask_ratio))
            indices = torch.randperm(N, device=x.device)[:n_keep]
            indices = indices.sort().values
        else:
            indices = mask

        visible = x[:, indices, :]
        return visible, indices
