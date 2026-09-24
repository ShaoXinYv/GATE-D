from __future__ import annotations
from dataclasses import dataclass
import torch
from models.geometry import normalize_inverse_depth, project_pose_trajectory

@dataclass(frozen=True)
class GaugeFixedGeometry:
    inverse_depth: torch.Tensor
    translations: torch.Tensor
    gauge: torch.Tensor

def gauge_fix_depth_translation(depth: torch.Tensor, translations: torch.Tensor, valid: torch.Tensor) -> GaugeFixedGeometry:
    """Fix the monocular depth/translation gauge to mean inverse depth one.

    Scaling depth and translation by the same positive value leaves perspective
    projection unchanged.  This deterministic convention removes metric scene
    scale before comparing relative-depth predictors across scenes.
    """
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError('depth must have shape [B,1,H,W]')
    if translations.ndim != 3 or translations.shape[-1] != 3:
        raise ValueError('translations must have shape [B,T,3]')
    if valid.shape != depth.shape:
        raise ValueError('valid must match depth')
    finite = torch.isfinite(depth) & (depth > 0)
    mask = valid.to(dtype=depth.dtype) * finite.to(dtype=depth.dtype)
    inverse = torch.where(finite, depth.reciprocal(), torch.zeros_like(depth))
    gauge = (inverse * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp_min(1.0)
    gauge = gauge.clamp_min(1e-08)
    normalized_inverse = normalize_inverse_depth(inverse, mask)
    return GaugeFixedGeometry(inverse_depth=normalized_inverse, translations=translations * gauge[:, None, None], gauge=gauge)

def poses_from_rotation_translation(rotations: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError('rotations must have shape [B,T,3,3]')
    if translations.shape != (*rotations.shape[:2], 3):
        raise ValueError('translations must have shape [B,T,3]')
    batch, steps = translations.shape[:2]
    poses = torch.eye(4, device=translations.device, dtype=translations.dtype)
    poses = poses.reshape(1, 1, 4, 4).repeat(batch, steps, 1, 1)
    poses[:, :, :3, :3] = rotations
    poses[:, :, :3, 3] = translations
    return poses

def project_gyro_anchored_translation(inverse_depth: torch.Tensor, rotations: torch.Tensor, translations: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Project rotation and translation jointly under a relative-depth gauge."""
    depth = inverse_depth.clamp_min(1e-06).reciprocal()
    poses = poses_from_rotation_translation(rotations, translations)
    flow, _valid = project_pose_trajectory(depth, poses, intrinsics)
    return flow
