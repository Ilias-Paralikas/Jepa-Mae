import torch
from torch.utils.data import Dataset


class PlaceholderDataset(Dataset):
    """
    Returns random images in [-1, 1] (tanh space) with a dummy label.
    Replace this class with your own dataset — keep __getitem__ returning
    (image, target) where image is (C, H, W).
    """

    def __init__(self, num_samples: int = 10000, img_size: int = 256, in_chans: int = 3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.in_chans = in_chans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.rand(self.in_chans, self.img_size, self.img_size).mul(2).sub(1)
        return img, torch.empty(0)
