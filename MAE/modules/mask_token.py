import torch
import torch.nn as nn


class MaskToken(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.token, std=0.02)

    def forward(
        self,
        visible: torch.Tensor,
        indices: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        """
        Reconstruct the full token sequence by inserting mask tokens at
        all positions not present in indices.

        Args:
            visible:     (B, N_keep, embed_dim) — encoder output
            indices:     (N_keep,)              — positions of visible patches
            num_patches: N                      — total number of patches
        Returns:
            (B, N, embed_dim)
        """
        B, _, D = visible.shape
        out = self.token.expand(B, num_patches, D).clone()
        out[:, indices] = visible
        return out
