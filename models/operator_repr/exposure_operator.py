from __future__ import annotations
from dataclasses import dataclass
import torch
from models.receiver import warp_feature
from models.operator_repr.phase_partition import contiguous_phase_ids

@dataclass(frozen=True)
class PhaseFlowMeasure:
    atoms: torch.Tensor
    weights: torch.Tensor

    def validate(self, *, atol: float=1e-06) -> None:
        if self.atoms.ndim != 5 or self.atoms.shape[2] != 2:
            raise ValueError('flow atoms must have shape [B,S,2,H,W]')
        if self.weights.shape != self.atoms.shape[:2]:
            raise ValueError('atom weights must have shape [B,S]')
        if not bool(torch.isfinite(self.atoms).all()):
            raise ValueError('flow atoms must be finite')
        if not bool(torch.isfinite(self.weights).all()):
            raise ValueError('atom weights must be finite')
        if bool((self.weights < 0).any()):
            raise ValueError('atom weights must be non-negative')
        expected = torch.ones(self.weights.shape[0], device=self.weights.device, dtype=self.weights.dtype)
        if not torch.allclose(self.weights.sum(1), expected, atol=atol, rtol=0.0):
            raise ValueError('atom weights must sum to one')

def phase_product_flow_measure(gyro_flow: torch.Tensor, residual_flow: torch.Tensor, phases: int, *, include_center: bool=True, phase_ids: torch.Tensor | None=None, paired_total_flow: torch.Tensor | None=None) -> PhaseFlowMeasure:
    """Build a non-negative phase-conditioned exposure-flow measure.

    Within each phase, gyro and residual samples form a product measure. This
    retains both marginals while deliberately discarding their fine pairing.
    """
    if gyro_flow.ndim != 5 or gyro_flow.shape[2] != 2:
        raise ValueError('gyro_flow must have shape [B,T,2,H,W]')
    if residual_flow.shape != gyro_flow.shape:
        raise ValueError('residual_flow must match gyro_flow')
    if paired_total_flow is not None and paired_total_flow.shape != gyro_flow.shape:
        raise ValueError('paired_total_flow must match gyro_flow')
    batch, steps, _, height, width = gyro_flow.shape
    if phase_ids is None:
        phase_ids = contiguous_phase_ids(steps, phases, device=gyro_flow.device)
    if phase_ids.ndim == 1:
        phase_ids = phase_ids[None].expand(batch, -1)
    if phase_ids.shape != (batch, steps):
        raise ValueError('phase_ids must have shape [T] or [B,T]')
    phase_ids = phase_ids.to(device=gyro_flow.device, dtype=torch.long)
    expected_ids = torch.arange(phases, device=gyro_flow.device)
    for row in phase_ids:
        if not torch.equal(torch.unique(row, sorted=True), expected_ids):
            raise ValueError('every phase id must occur at least once')
        if bool((row[1:] < row[:-1]).any()):
            raise ValueError('phase ids must be contiguous and nondecreasing')
    if bool((phase_ids == phase_ids[:1]).all()):
        shared_ids = phase_ids[0]
        atom_groups: list[torch.Tensor] = []
        weight_groups: list[torch.Tensor] = []
        denominator = steps + int(include_center)
        for phase in range(phases):
            selected = torch.nonzero(shared_ids == phase, as_tuple=False).flatten()
            count = int(selected.numel())
            gyro = gyro_flow.index_select(1, selected)
            residual = residual_flow.index_select(1, selected)
            atoms = gyro[:, :, None] + residual[:, None, :]
            if paired_total_flow is not None:
                diagonal = torch.arange(count, device=gyro_flow.device)
                atoms[:, diagonal, diagonal] = paired_total_flow.index_select(1, selected)
            atoms = atoms.reshape(batch, count * count, 2, height, width)
            atom_groups.append(atoms)
            weight_groups.append(gyro_flow.new_full((batch, count * count), 1.0 / (denominator * count)))
        if include_center:
            center_group = next((phase for phase in range(phases) if int(torch.nonzero(shared_ids == phase, as_tuple=False).min()) >= steps // 2), len(atom_groups))
            atom_groups.insert(center_group, gyro_flow.new_zeros((batch, 1, 2, height, width)))
            weight_groups.insert(center_group, gyro_flow.new_full((batch, 1), 1.0 / denominator))
        measure = PhaseFlowMeasure(atoms=torch.cat(atom_groups, dim=1), weights=torch.cat(weight_groups, dim=1))
        measure.validate()
        return measure
    sample_atoms: list[torch.Tensor] = []
    sample_weights: list[torch.Tensor] = []
    atom_groups: list[torch.Tensor] = []
    weight_groups: list[torch.Tensor] = []
    denominator = steps + int(include_center)
    for batch_index in range(batch):
        atom_groups.clear()
        weight_groups.clear()
        for phase in range(phases):
            selected = torch.nonzero(phase_ids[batch_index] == phase, as_tuple=False).flatten()
            count = int(selected.numel())
            gyro = gyro_flow[batch_index].index_select(0, selected)
            residual = residual_flow[batch_index].index_select(0, selected)
            atoms = gyro[:, None] + residual[None, :]
            if paired_total_flow is not None:
                diagonal = torch.arange(count, device=gyro_flow.device)
                atoms[diagonal, diagonal] = paired_total_flow[batch_index].index_select(0, selected)
            atom_groups.append(atoms.reshape(count * count, 2, height, width))
            weight_groups.append(gyro_flow.new_full((count * count,), 1.0 / (denominator * count)))
        if include_center:
            row_ids = phase_ids[batch_index]
            center_group = next((phase for phase in range(phases) if int(torch.nonzero(row_ids == phase, as_tuple=False).min()) >= steps // 2), len(atom_groups))
            atom_groups.insert(center_group, gyro_flow.new_zeros((1, 2, height, width)))
            weight_groups.insert(center_group, gyro_flow.new_full((1,), 1.0 / denominator))
        sample_atoms.append(torch.cat(atom_groups, dim=0))
        sample_weights.append(torch.cat(weight_groups, dim=0))
    maximum_atoms = max((value.shape[0] for value in sample_atoms))
    padded_atoms = gyro_flow.new_zeros((batch, maximum_atoms, 2, height, width))
    padded_weights = gyro_flow.new_zeros((batch, maximum_atoms))
    for batch_index, (atoms, weights) in enumerate(zip(sample_atoms, sample_weights)):
        count = atoms.shape[0]
        padded_atoms[batch_index, :count] = atoms
        padded_weights[batch_index, :count] = weights
    measure = PhaseFlowMeasure(atoms=padded_atoms, weights=padded_weights)
    measure.validate()
    return measure

def apply_weighted_flow_measure(feature: torch.Tensor, measure: PhaseFlowMeasure, *, sign: float) -> torch.Tensor:
    """Apply an empirical flow measure without averaging flow coordinates."""
    measure.validate()
    if feature.ndim != 4 or feature.shape[0] != measure.atoms.shape[0]:
        raise ValueError('feature must have shape [B,C,H,W] with matching batch')
    if feature.shape[-2:] != measure.atoms.shape[-2:]:
        raise ValueError('feature and flow atoms must have matching spatial size')
    warped = [warp_feature(feature, measure.atoms[:, index], sign) for index in range(measure.atoms.shape[1])]
    equal_weight = measure.weights.new_full(measure.weights.shape, 1.0 / measure.weights.shape[1])
    if torch.equal(measure.weights, equal_weight):
        return torch.stack(warped).mean(0)
    values = torch.stack(warped, dim=1)
    return (values * measure.weights[:, :, None, None, None]).sum(1)
