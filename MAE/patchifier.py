import torch
import torch.nn as nn


class Patchifier(nn.Module):
    def __init__(self, patch_size: int, img_size: int, in_channels: int = 3):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split image into non-overlapping patches.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, patch_size^2 * C) where N = (H/patch_size) * (W/patch_size)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p

        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from patches.

        Args:
            x: (B, N, patch_size^2 * C)
        Returns:
            (B, C, H, W)
        """
        B, N, _ = x.shape
        p = self.patch_size
        C = self.in_channels
        h = w = int(N ** 0.5)

        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(B, C, h * p, w * p)
        return x
