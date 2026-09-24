from __future__ import annotations
import torch
import torch.nn.functional as F

def gaussian_ssim(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float=1.0, window_size: int=11, sigma: float=1.5) -> torch.Tensor:
    """Per-image RGB SSIM using the standard valid Gaussian window."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('prediction and target must have the same NCHW shape')
    if window_size % 2 != 1 or min(prediction.shape[-2:]) < window_size:
        raise ValueError('window_size must be odd and fit inside the image')
    coordinates = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel_1d = torch.exp(-coordinates.square() / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = prediction.shape[1]
    kernel = kernel_2d[None, None].expand(channels, 1, -1, -1)
    mean_x = F.conv2d(prediction, kernel, groups=channels)
    mean_y = F.conv2d(target, kernel, groups=channels)
    mean_x2 = mean_x.square()
    mean_y2 = mean_y.square()
    mean_xy = mean_x * mean_y
    variance_x = F.conv2d(prediction.square(), kernel, groups=channels) - mean_x2
    variance_y = F.conv2d(target.square(), kernel, groups=channels) - mean_y2
    covariance = F.conv2d(prediction * target, kernel, groups=channels) - mean_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mean_xy + c1) * (2.0 * covariance + c2)
    denominator = (mean_x2 + mean_y2 + c1) * (variance_x + variance_y + c2)
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).mean((1, 2, 3))
