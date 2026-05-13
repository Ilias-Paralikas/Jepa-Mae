import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        visible_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss computed only on masked (non-visible) patches.

        Args:
            pred:            (B, N, patch_dim) — MAE reconstructed patches
            target:          (B, N, patch_dim) — original image patches
            visible_indices: (N_keep,)         — indices of the visible patches
        Returns:
            scalar loss
        """
        N = pred.shape[1]
        masked = torch.ones(N, dtype=torch.bool, device=pred.device)
        masked[visible_indices] = False

        return F.mse_loss(pred[:, masked], target[:, masked])
