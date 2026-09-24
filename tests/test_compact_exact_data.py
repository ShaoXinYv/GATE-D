import torch
from data.dataset import CompactExactDataset

def test_exact_dataset_crop_supports_geometry_only_samples() -> None:
    dataset = object.__new__(CompactExactDataset)
    dataset.crop_size = 4
    dataset.random_crop = False
    blur = torch.zeros(3, 8, 10)
    depth = torch.ones(1, 8, 10)
    intrinsics = torch.tensor([100.0, 100.0, 5.0, 4.0])
    cropped_blur, cropped_sharp, cropped_depth, cropped_intrinsics = dataset._crop(blur, None, depth, intrinsics)
    assert cropped_blur.shape == (3, 4, 4)
    assert cropped_sharp is None
    assert cropped_depth.shape == (1, 4, 4)
    torch.testing.assert_close(cropped_intrinsics, torch.tensor([100.0, 100.0, 2.0, 2.0]))
