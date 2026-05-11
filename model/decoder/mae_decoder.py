import torch
import torch.nn as nn

from model.components.positional_embed import get_2d_sincos_pos_embed
from model.blocks.transformer_block import TransformerBlock


class MAEDecoder(nn.Module):
    """
    MAE decoder: project encoder tokens -> insert mask tokens -> restore order
    -> transformer -> pixel prediction head.

    Input:
        latent     : (B, 1 + len_keep, encoder_embed_dim)
        ids_restore: (B, N)

    Output:
        pred: (B, N, patch_size^2 * in_chans)  — predicted pixels for every patch
    """

    def __init__(self, num_patches: int, encoder_embed_dim: int,
                 decoder_embed_dim: int = 512, depth: int = 8, num_heads: int = 16,
                 mlp_ratio: float = 4.0, patch_size: int = 16, in_chans: int = 3,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = norm_layer(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

        self._init_weights(num_patches)

    def _init_weights(self, num_patches):
        grid_size = int(num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, latent, ids_restore):
        x = self.decoder_embed(latent)      # (B, 1+len_keep, decoder_dim)

        B = x.shape[0]
        num_kept = x.shape[1] - 1          # subtract cls token
        num_masked = ids_restore.shape[1] - num_kept

        # append mask tokens and restore original patch order
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # drop cls, concat mask tokens
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1]))
        x = torch.cat([x[:, :1, :], x_], dim=1)            # re-attach cls token

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        pred = self.pred(x[:, 1:, :])   # remove cls token -> (B, N, p^2*C)
        return pred
