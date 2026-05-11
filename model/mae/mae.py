import torch
import torch.nn as nn

from model.encoder import MAEEncoder
from model.decoder import MAEDecoder


class MAE(nn.Module):
    """
    Masked Autoencoder (He et al., 2022).

    forward() returns (loss, pred, mask).
    reconstruct() returns (original, masked_img, recon_img) tensors for visualisation.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 encoder_embed_dim: int = 768, encoder_depth: int = 12, encoder_num_heads: int = 12,
                 decoder_embed_dim: int = 512, decoder_depth: int = 8, decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.0, mask_ratio: float = 0.75, norm_pix_loss: bool = True):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.encoder = MAEEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, mask_ratio=mask_ratio,
        )
        num_patches = self.encoder.patch_embed.num_patches

        self.decoder = MAEDecoder(
            num_patches=num_patches, encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            patch_size=patch_size, in_chans=in_chans,
        )

    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------

    def patchify(self, imgs):
        """(B, C, H, W) -> (B, N, patch_size^2 * C)"""
        p, C = self.patch_size, self.in_chans
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(imgs.shape[0], h * w, p * p * C)
        return x

    def unpatchify(self, patches, h, w):
        """(B, N, patch_size^2 * C) -> (B, C, H, W)"""
        p, C, B = self.patch_size, self.in_chans, patches.shape[0]
        x = patches.reshape(B, h, w, p, p, C)
        x = torch.einsum("bhwpqc->bchpwq", x)
        x = x.reshape(B, C, h * p, w * p)
        return x

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(self, imgs):
        latent, mask, ids_restore = self.encoder(imgs)
        pred = self.decoder(latent, ids_restore)    # (B, N, p^2*C)
        loss = self._compute_loss(imgs, pred, mask)
        return loss, pred, mask

    def _compute_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)    # (B, N, p^2*C)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        loss = (pred - target) ** 2     # (B, N, p^2*C)
        loss = loss.mean(dim=-1)        # (B, N)  — per-patch MSE
        loss = (loss * mask).sum() / mask.sum()   # mean over masked patches only
        return loss

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(self, imgs):
        """
        Run a forward pass and return image tensors suitable for plotting.

        Returns:
            original   : (B, C, H, W)
            masked_img : (B, C, H, W)  — visible patches only, masked patches are zero
            recon_img  : (B, C, H, W)  — predicted patches blended with visible patches
        """
        latent, mask, ids_restore = self.encoder(imgs)
        pred = self.decoder(latent, ids_restore)    # (B, N, p^2*C) in (possibly norm) space

        target = self.patchify(imgs)    # original patches (B, N, p^2*C)

        if self.norm_pix_loss:
            # denormalise predictions back to pixel space
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred = pred * (var + 1e-6).sqrt() + mean

        h = w = imgs.shape[2] // self.patch_size
        m = mask.unsqueeze(-1)                          # (B, N, 1)

        recon_patches = pred * m + target * (1 - m)    # pred where masked, orig elsewhere
        masked_patches = target * (1 - m)              # zero where masked

        recon_img = self.unpatchify(recon_patches, h, w)
        masked_img = self.unpatchify(masked_patches, h, w)

        return imgs, masked_img, recon_img
