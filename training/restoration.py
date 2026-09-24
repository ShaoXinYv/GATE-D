from __future__ import annotations
import torch
from models.geometry import geometry_tensor
from models.operator_repr import phase_product_flow_measure
from training.receiver_setup import translation_reliability

def phase_restore_trainable(receiver, blur: torch.Tensor, gyro: torch.Tensor, total: torch.Tensor, active_probability: torch.Tensor, reliability_tau: float, *, return_receiver_features: bool=False, clamp_output: bool=True) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    if getattr(receiver, 'force_zero_residual', False):
        total = gyro
        active_probability = torch.zeros_like(active_probability)
    batch, channels, height, width = total.shape
    gyro_steps = gyro.reshape(batch, channels // 2, 2, height, width)
    residual_steps = (total - gyro).reshape_as(gyro_steps)
    groups = getattr(receiver, 'exposure_phase_groups', 4)
    active_measure = phase_product_flow_measure(gyro_steps, residual_steps, groups)
    anchor_measure = phase_product_flow_measure(gyro_steps, torch.zeros_like(residual_steps), groups)
    inverse_depth = total.new_ones(batch, 1, height, width)
    zero = torch.zeros_like(inverse_depth)
    anchor_geometry = geometry_tensor(gyro, inverse_depth, zero, receiver.phys_inverse_bottleneck.flow_scale)
    posterior_reliability = translation_reliability(total, gyro, reliability_tau) * active_probability[:, None, None, None]
    total_geometry = geometry_tensor(total, inverse_depth, posterior_reliability, receiver.phys_inverse_bottleneck.flow_scale)
    restored = receiver(blur, gyro, anchor_geometry, return_aux=return_receiver_features, residual_geometry=total_geometry, residual_flow_atoms=active_measure.atoms, residual_atom_weights=active_measure.weights, anchor_flow_atoms=anchor_measure.atoms, anchor_atom_weights=anchor_measure.weights)
    if not return_receiver_features:
        return restored.clamp(0.0, 1.0) if clamp_output else restored
    features = tuple((entry['residual_feature_delta'] for entry in restored.auxiliary['phys_inverse'] if 'residual_feature_delta' in entry))
    if not features:
        raise RuntimeError('receiver did not expose residual physical features')
    return (restored.image.clamp(0.0, 1.0), features)
