import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, N, embed_dim) — visible patch embeddings
            indices: (N,)              — positions of the provided patches
        Returns:
            (B, N, embed_dim)
        """
        return x + self.pos_embed[indices]
