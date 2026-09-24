from __future__ import annotations
import torch
from models.geometry import pixel_grid

def renderer_rotation_increment(angle: torch.Tensor) -> torch.Tensor:
    """Match the renderer's Rx(-x) Ry(-y) Rz(-z) convention exactly."""
    x, y, z = angle.unbind(-1)
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    cx, cy, cz = (torch.cos(x), torch.cos(y), torch.cos(z))
    sx, sy, sz = (torch.sin(x), torch.sin(y), torch.sin(z))
    rx = torch.stack((one, zero, zero, zero, cx, sx, zero, -sx, cx), dim=-1).reshape(*angle.shape[:-1], 3, 3)
    ry = torch.stack((cy, zero, -sy, zero, one, zero, sy, zero, cy), dim=-1).reshape(*angle.shape[:-1], 3, 3)
    rz = torch.stack((cz, sz, zero, -sz, cz, zero, zero, zero, one), dim=-1).reshape(*angle.shape[:-1], 3, 3)
    return rx @ ry @ rz

def rotations_to_flow(rotations: torch.Tensor, intrinsics: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Project center-relative rotations to ordered dense flow."""
    batch, samples = rotations.shape[:2]
    height, width = image_size
    center = samples // 2
    center_inverse = rotations[:, center].transpose(-1, -2)
    relative = rotations @ center_inverse[:, None]
    xx, yy = pixel_grid(height, width, device=rotations.device, dtype=rotations.dtype)
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=0).reshape(1, 3, -1)
    k = torch.zeros(batch, 3, 3, device=rotations.device, dtype=rotations.dtype)
    k[:, 0, 0], k[:, 1, 1] = (intrinsics[:, 0], intrinsics[:, 1])
    k[:, 0, 2], k[:, 1, 2] = (intrinsics[:, 2], intrinsics[:, 3])
    k[:, 2, 2] = 1.0
    homography = k[:, None] @ relative @ torch.linalg.inv(k)[:, None]
    warped = homography @ pixels[:, None]
    warped_xy = warped[:, :, :2] / warped[:, :, 2:3].clamp_min(1e-08)
    reference = pixels[:, :2].reshape(1, 1, 2, height, width)
    flow = warped_xy.reshape(batch, samples, 2, height, width) - reference
    keep = [index for index in range(samples) if index != center]
    return flow[:, keep].reshape(batch, 2 * (samples - 1), height, width)

def integrate_dense_gyro_segment(segment: torch.Tensor, *, interpolation_factor: int=8, timestamp_scale: float=1e-09, gyro_scale: float=1.0) -> torch.Tensor:
    """Reproduce the renderer's linearly interpolated dense gyro integration."""
    if segment.ndim != 2 or segment.shape[1] != 4:
        raise ValueError('segment must have shape [N, 4] as timestamp, wx, wy, wz')
    timestamps = segment[:, 0]
    angular_velocity = segment[:, 1:4]
    rotations = [torch.eye(3, device=segment.device, dtype=segment.dtype)]
    current = rotations[0]
    for index in range(segment.shape[0] - 1):
        dt = (timestamps[index + 1] - timestamps[index]) * timestamp_scale / interpolation_factor
        for substep in range(interpolation_factor):
            alpha = substep / interpolation_factor
            omega = (1.0 - alpha) * angular_velocity[index] + alpha * angular_velocity[index + 1]
            current = renderer_rotation_increment(omega * dt * gyro_scale) @ current
            rotations.append(current)
    rotations = torch.stack(rotations)
    center_inverse = rotations[len(rotations) // 2].transpose(-1, -2)
    return rotations @ center_inverse

def eight_exposure_indices(sample_count: int) -> torch.Tensor:
    """Four pre-center and four post-center indices, matching legacy labels."""
    if sample_count < 9 or sample_count % 2 == 0:
        raise ValueError('Expected an odd dense exposure trajectory with at least 9 samples')
    center = sample_count // 2
    before = torch.linspace(0, center, 5).round().long()[:-1]
    after = torch.linspace(center, sample_count - 1, 5).round().long()[1:]
    return torch.cat((before, after))
