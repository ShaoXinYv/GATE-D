from __future__ import annotations
import argparse
from pathlib import Path
import torch
from train_foundation import build_receiver

def translation_reliability(total: torch.Tensor, gyro: torch.Tensor, tau: float) -> torch.Tensor:
    """Continuous deterministic confidence derived from residual flow energy."""
    if tau <= 0:
        raise ValueError('reliability tau must be positive')
    residual = total - gyro
    batch, channels, height, width = residual.shape
    if channels % 2:
        raise ValueError('trajectory channels must contain x/y pairs')
    magnitude = residual.reshape(batch, channels // 2, 2, height, width)
    squared = magnitude.square().sum(2)
    magnitude = ((squared + 1e-12).sqrt() - 1e-06).clamp_min(0.0)
    magnitude = magnitude.mean(1, keepdim=True)
    return 1.0 - torch.exp(-magnitude / float(tau))

def set_zero_safe_trainable_scope(receiver) -> list[torch.nn.Parameter]:
    """Freeze the foundation and train only differential physical adapters."""
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    receiver.enable_residual_physical_adapters()
    trainable = []
    for block in receiver.residual_physical_blocks():
        for parameter in block.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable

def load_foundation(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    receiver = build_receiver(argparse.Namespace(**state['architecture']), device)
    receiver.load_state_dict(state['receiver_state_dict'], strict=True)
    return receiver
