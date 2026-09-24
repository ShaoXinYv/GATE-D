from __future__ import annotations
import torch
import torch.nn.functional as F

def restoration_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    charbonnier = ((prediction - target).square() + 1e-06).sqrt().mean()
    mu_x = F.avg_pool2d(prediction, 11, 1, 5)
    mu_y = F.avg_pool2d(target, 11, 1, 5)
    sigma_x = F.avg_pool2d(prediction.square(), 11, 1, 5) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), 11, 1, 5) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, 11, 1, 5) - mu_x * mu_y
    c1, c2 = (0.01 ** 2, 0.03 ** 2)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)).clamp_min(1e-08)).mean()
    return charbonnier + 0.1 * (1.0 - ssim)
