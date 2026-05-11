import numpy as np


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """2D sin-cos positional embedding used in MAE / ViT."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)          # each (H, W)
    grid = np.stack([grid_w, grid_h], axis=0)              # (2, H, W)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    pos_embed = _embed_from_grid(embed_dim, grid)          # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


def _embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _embed_1d(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = _embed_1d(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def _embed_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)      # (D/2,)
    pos = pos.reshape(-1)               # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D/2)
