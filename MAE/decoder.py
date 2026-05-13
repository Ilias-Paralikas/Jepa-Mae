import torch
import torch.nn as nn
from .modules.mask_token import MaskToken
from .modules.positional_embedding import PositionalEmbedding
from .modules.transformer_block import TransformerBlock


class Decoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = MaskToken(decoder_embed_dim)
        self.pos_embed = PositionalEmbedding(num_patches, decoder_embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, N_keep, encoder_embed_dim) — encoder output
            indices: (N_keep,)                      — positions of visible patches
        Returns:
            (B, N, decoder_embed_dim)
        """
        x = self.projection(x)
        x = self.mask_token(x, indices, self.num_patches)
        all_indices = torch.arange(self.num_patches, device=x.device)
        x = self.pos_embed(x, all_indices)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
