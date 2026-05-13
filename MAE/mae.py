from __future__ import annotations
import torch
import torch.nn as nn
from .modules.masking import Masking
from .modules.linear_layer import LinearLayer
from .encoder import Encoder
from .decoder import Decoder


class MAE(nn.Module):
    def __init__(
        self,
        num_patches: int,
        patch_dim: int,
        encoder_embed_dim: int,
        encoder_depth: int,
        encoder_num_heads: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mask_ratio: float = 0.75,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.masking = Masking(mask_ratio)
        self.encoder = Encoder(num_patches, encoder_embed_dim, encoder_depth, encoder_num_heads, mlp_ratio, dropout)
        self.decoder = Decoder(num_patches, encoder_embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, dropout)
        self.to_rgb = LinearLayer(decoder_embed_dim, patch_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, N, encoder_embed_dim) — patch embeddings
            mask: (N_keep,) indices to keep, or None to generate randomly
        Returns:
            rgb:     (B, N, patch_dim) — reconstructed pixel values for all patches
            indices: (N_keep,)         — indices of the visible (unmasked) patches
        """
        visible, indices = self.masking(x, mask)
        encoded = self.encoder(visible, indices)
        decoded = self.decoder(encoded, indices)
        rgb = self.to_rgb(decoded)
        return rgb, indices
