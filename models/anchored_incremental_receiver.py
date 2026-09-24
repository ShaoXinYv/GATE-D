from __future__ import annotations
import torch
from models.geometry import geometry_tensor
from models.receiver import MultiScalePhysicalReceiver

def gyro_foundation_restore(receiver: MultiScalePhysicalReceiver, blur: torch.Tensor, gyro: torch.Tensor, clamp_output: bool=True) -> torch.Tensor:
    valid = torch.ones(gyro.shape[0], 1, *gyro.shape[-2:], device=gyro.device, dtype=gyro.dtype)
    geometry = geometry_tensor(gyro, valid, torch.zeros_like(valid), receiver.phys_inverse_bottleneck.flow_scale)
    restored = receiver(blur, gyro, geometry)
    return restored.clamp(0.0, 1.0) if clamp_output else restored
