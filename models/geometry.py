from __future__ import annotations
import torch
import torch.nn.functional as F

def pixel_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing='ij')
    return (xx, yy)

def normalize_inverse_depth(inv_depth: torch.Tensor, valid: torch.Tensor | None=None) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(inv_depth)
    valid = valid.to(inv_depth.dtype)
    mean = (inv_depth * valid).sum((2, 3), keepdim=True)
    mean = mean / valid.sum((2, 3), keepdim=True).clamp_min(1.0)
    return inv_depth / mean.clamp_min(1e-06) * valid

def project_pose_trajectory(depth: torch.Tensor, poses: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project center-to-sample SE(3) poses using reference-camera depth."""
    batch, _, height, width = depth.shape
    xx, yy = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    xx = xx.reshape(1, -1).expand(batch, -1)
    yy = yy.reshape(1, -1).expand(batch, -1)
    z_ref = depth[:, 0].reshape(batch, -1)
    fx, fy, cx, cy = (intrinsics[:, index:index + 1] for index in range(4))
    reference = torch.stack(((xx - cx) / fx * z_ref, (yy - cy) / fy * z_ref, z_ref, torch.ones_like(z_ref)), dim=1)
    target = poses @ reference[:, None]
    z = target[:, :, 2]
    u = fx[:, None] * target[:, :, 0] / z.clamp_min(1e-08) + cx[:, None]
    v = fy[:, None] * target[:, :, 1] / z.clamp_min(1e-08) + cy[:, None]
    flow = torch.stack((u - xx[:, None], v - yy[:, None]), dim=2)
    valid = torch.isfinite(flow).all(2) & torch.isfinite(z_ref[:, None]) & (z_ref[:, None] > 0) & (z > 1e-08)
    flow = torch.where(valid[:, :, None], flow, torch.zeros_like(flow))
    return (flow.reshape(batch, 2 * poses.shape[1], height, width), valid.reshape(batch, poses.shape[1], height, width))

def resize_flow_trajectory(trajectory: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    source_h, source_w = trajectory.shape[-2:]
    target_h, target_w = size
    resized = F.interpolate(trajectory, size=size, mode='bilinear', align_corners=False)
    resized = resized.clone()
    resized[:, 0::2] *= target_w / max(source_w, 1)
    resized[:, 1::2] *= target_h / max(source_h, 1)
    return resized

def geometry_tensor(trajectory: torch.Tensor, inverse_depth: torch.Tensor, confidence: torch.Tensor, flow_scale: float) -> torch.Tensor:
    """Receiver geometry with an unmodified known gyro anchor.

    `confidence` is an auxiliary reliability field. It must not attenuate the
    trajectory here: residual attenuation is applied before composition, while
    the gyro-derived rotation flow remains a known physical measurement.
    """
    reliability = confidence.clamp(0.0, 1.0)
    return torch.cat((trajectory / flow_scale, inverse_depth, reliability), dim=1)
