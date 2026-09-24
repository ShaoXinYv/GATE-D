from __future__ import annotations
import torch

def gyro_phase_energy(gyro_flow: torch.Tensor) -> torch.Tensor:
    """RMS change between adjacent coarse-phase rotation-flow means."""
    if gyro_flow.ndim != 4 or gyro_flow.shape[1] != 16:
        raise ValueError('gyro_flow must have shape [B,16,H,W]')
    batch, _channels, height, width = gyro_flow.shape
    phase = gyro_flow.reshape(batch, 4, 2, 2, height, width).mean(2)
    differences = phase[:, 1:] - phase[:, :-1]
    return differences.square().sum(2).mean((1, 2, 3)).sqrt()

def gyro_phase_observability(gyro_flow: torch.Tensor, *, tau_px: float=0.5) -> torch.Tensor:
    """Continuous evidence that coarse phase labels alter the rotation operator."""
    if tau_px <= 0:
        raise ValueError('tau_px must be positive')
    energy = gyro_phase_energy(gyro_flow)
    return 1.0 - torch.exp(-energy / tau_px)

def fit_translations_from_gyro_residual_flow(residual_flow: torch.Tensor, rotations: torch.Tensor, inverse_depth: torch.Tensor, intrinsics: torch.Tensor, valid: torch.Tensor | None=None, *, ridge: float=0.0001) -> torch.Tensor:
    """Fit translations from additive residual flow under known gyro rotation.

    ``residual_flow`` is the exact additive field ``full_flow - gyro_flow``.
    Unlike :func:`fit_translations_from_dense_flow`, this solver does not treat
    that residual as a standalone identity-rotation projection. It reconstructs
    the total pixel correspondence using the known rotation, then solves the
    perspective equations for translation. This preserves the rotation and
    translation composition used by the renderer.
    """
    if residual_flow.ndim != 4 or residual_flow.shape[1] % 2:
        raise ValueError('residual_flow must have shape [B,2T,H,W]')
    batch, channels, height, width = residual_flow.shape
    steps = channels // 2
    if rotations.shape != (batch, steps, 3, 3):
        raise ValueError('rotations must have shape [B,T,3,3]')
    if inverse_depth.shape != (batch, 1, height, width):
        raise ValueError('inverse_depth must have shape [B,1,H,W]')
    if intrinsics.shape != (batch, 4):
        raise ValueError('intrinsics must have shape [B,4]')
    if ridge < 0:
        raise ValueError('ridge must be non-negative')
    if valid is None:
        valid = torch.ones_like(inverse_depth)
    if valid.shape != inverse_depth.shape:
        raise ValueError('valid must match inverse_depth')
    yy, xx = torch.meshgrid(torch.arange(height, device=residual_flow.device, dtype=residual_flow.dtype), torch.arange(width, device=residual_flow.device, dtype=residual_flow.dtype), indexing='ij')
    xx = xx.reshape(1, -1).expand(batch, -1)
    yy = yy.reshape(1, -1).expand(batch, -1)
    depth = inverse_depth[:, 0].reshape(batch, -1).clamp_min(1e-06).reciprocal()
    fx, fy, cx, cy = (intrinsics[:, index:index + 1] for index in range(4))
    reference = torch.stack(((xx - cx) / fx * depth, (yy - cy) / fy * depth, depth), dim=1)
    rotated = torch.einsum('btij,bjn->btin', rotations, reference)
    rotation_u = fx[:, None] * rotated[:, :, 0] / rotated[:, :, 2].clamp_min(1e-06)
    rotation_v = fy[:, None] * rotated[:, :, 1] / rotated[:, :, 2].clamp_min(1e-06)
    rotation_u = rotation_u + cx[:, None]
    rotation_v = rotation_v + cy[:, None]
    residual = residual_flow.reshape(batch, steps, 2, -1)
    observed_u = rotation_u + residual[:, :, 0]
    observed_v = rotation_v + residual[:, :, 1]
    normalized_u = (observed_u - cx[:, None]) / fx[:, None]
    normalized_v = (observed_v - cy[:, None]) / fy[:, None]
    target_x = normalized_u * rotated[:, :, 2] - rotated[:, :, 0]
    target_y = normalized_v * rotated[:, :, 2] - rotated[:, :, 1]
    weight = valid[:, 0].reshape(batch, 1, -1).to(residual_flow)
    count = weight.sum(-1).expand(-1, steps).clamp_min(1.0)
    sx = (weight * normalized_u).sum(-1)
    sy = (weight * normalized_v).sum(-1)
    sxx_yy = (weight * (normalized_u.square() + normalized_v.square())).sum(-1)
    ata = residual_flow.new_zeros(batch, steps, 3, 3)
    ata[:, :, 0, 0] = count
    ata[:, :, 1, 1] = count
    ata[:, :, 0, 2] = ata[:, :, 2, 0] = -sx
    ata[:, :, 1, 2] = ata[:, :, 2, 1] = -sy
    ata[:, :, 2, 2] = sxx_yy
    atb = torch.stack(((weight * target_x).sum(-1), (weight * target_y).sum(-1), -(weight * (normalized_u * target_x + normalized_v * target_y)).sum(-1)), dim=-1)
    identity = torch.eye(3, device=residual_flow.device, dtype=residual_flow.dtype)
    scale = ata.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1.0)
    regularized = ata + float(ridge) * scale[..., None] * identity
    return torch.linalg.solve(regularized, atb[..., None]).squeeze(-1)
