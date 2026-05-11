import torch
import torch.nn as nn

from model.components.patch_embed import PatchEmbed
from model.components.positional_embed import get_2d_sincos_pos_embed
from model.blocks.transformer_block import TransformerBlock


class MAEEncoder(nn.Module):
    """
    MAE encoder: patchify -> randomly mask -> transformer on visible tokens only.

    Returns:
        latent     : (B, 1 + len_keep, encoder_embed_dim)  cls token prepended
        mask       : (B, N)  — 1 = masked, 0 = visible
        ids_restore: (B, N)  — inverse permutation to restore original patch order
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, mask_ratio: float = 0.75,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # sin-cos pos embed is fixed (not learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        grid_size = int(self.patch_embed.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)          # ascending noise -> kept patches first
        ids_restore = ids_shuffle.argsort(dim=1)    # inverse permutation

        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # mask: 1 = masked (removed), 0 = visible (kept)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_kept, mask, ids_restore

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]   # pos embed without cls token

        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore
