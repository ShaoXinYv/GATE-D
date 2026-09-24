from __future__ import annotations
import math
import torch
from torch.utils.data import DataLoader
from data.dataset import CompactBlurGyroDataset

def psnr_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = 10.0 / math.log(10.0)
    return scale * torch.log((prediction - target).square().mean((1, 2, 3)) + 1e-08).mean()

def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.clamp(0.0, 1.0) - target).square().mean((1, 2, 3)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)

def epoch_loader(dataset: CompactBlurGyroDataset, *, epoch: int, seed: int, batch_size: int, workers: int) -> DataLoader:
    """Make epoch sampling identical for uninterrupted and resumed training."""
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=False, generator=generator)
