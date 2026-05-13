import torch
import torch.nn as nn
from .modules.positional_embedding import PositionalEmbedding
from .modules.transformer_block import TransformerBlock


class Encoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, N_keep, embed_dim) — visible patch embeddings
            indices: (N_keep,)              — positions of the visible patches
        Returns:
            (B, N_keep, embed_dim)
        """
        x = self.pos_embed(x, indices)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
