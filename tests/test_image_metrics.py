from __future__ import annotations
import numpy as np
import torch
from skimage.metrics import structural_similarity
from models.image_metrics import gaussian_ssim

def test_gaussian_ssim_identity_and_symmetry() -> None:
    generator = torch.Generator().manual_seed(17)
    left = torch.rand(2, 3, 32, 40, generator=generator)
    right = torch.rand(2, 3, 32, 40, generator=generator)
    assert torch.allclose(gaussian_ssim(left, left), torch.ones(2), atol=1e-06)
    assert torch.allclose(gaussian_ssim(left, right), gaussian_ssim(right, left), atol=1e-06)

def test_gaussian_ssim_matches_skimage_reference() -> None:
    generator = torch.Generator().manual_seed(23)
    left = torch.rand(1, 3, 32, 40, generator=generator)
    right = torch.rand(1, 3, 32, 40, generator=generator)
    expected = structural_similarity(left[0].permute(1, 2, 0).numpy(), right[0].permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    actual = float(gaussian_ssim(left, right)[0])
    assert np.isclose(actual, expected, atol=2e-06)
