import pytest
import torch
from models.operator_repr import PhaseFlowMeasure, apply_weighted_flow_measure, phase_product_flow_measure
from models.receiver import TrajectoryPhysicalInverseBlock

def test_q8_matches_existing_paired_forward_operator() -> None:
    torch.manual_seed(7)
    feature = torch.randn(2, 5, 11, 13)
    gyro = torch.randn(2, 8, 2, 11, 13) * 0.15
    residual = torch.randn_like(gyro) * 0.1
    total = gyro + residual
    measure = phase_product_flow_measure(gyro, residual, phases=8)
    block = TrajectoryPhysicalInverseBlock(5)
    expected = block.forward_operator(feature, list(total.unbind(1)))
    actual = apply_weighted_flow_measure(feature, measure, sign=-1.0)
    assert torch.equal(actual, expected)

def test_physical_block_q8_measure_matches_existing_path() -> None:
    torch.manual_seed(11)
    block = TrajectoryPhysicalInverseBlock(8)
    with torch.no_grad():
        block.delta.weight.normal_(0.0, 0.01)
    blur = torch.randn(1, 8, 10, 12)
    estimate = torch.randn_like(blur)
    gyro = torch.randn(1, 8, 2, 10, 12) * 0.1
    residual = torch.randn_like(gyro) * 0.05
    total = gyro + residual
    geometry = torch.cat((total.reshape(1, 16, 10, 12) / block.flow_scale, torch.ones(1, 2, 10, 12)), dim=1)
    expected, _ = block(blur, estimate, geometry)
    measure = phase_product_flow_measure(gyro, residual, phases=8)
    actual, _ = block(blur, estimate, geometry, flow_atoms=measure.atoms, atom_weights=measure.weights)
    assert torch.equal(actual, expected)

@pytest.mark.parametrize('phases', [1, 2, 4, 8])
def test_zero_residual_total_and_anchor_are_exactly_equal(phases: int) -> None:
    gyro = torch.randn(1, 8, 2, 8, 10)
    zero = torch.zeros_like(gyro)
    total = phase_product_flow_measure(gyro, zero, phases)
    anchor = phase_product_flow_measure(gyro, zero, phases)
    assert torch.equal(total.atoms, anchor.atoms)
    assert torch.equal(total.weights, anchor.weights)

def test_pure_translation_operator_is_phase_invariant() -> None:
    torch.manual_seed(13)
    feature = torch.randn(1, 4, 9, 12)
    gyro = torch.zeros(1, 8, 2, 9, 12)
    residual = torch.randn_like(gyro) * 0.2
    outputs = [apply_weighted_flow_measure(feature, phase_product_flow_measure(gyro, residual, phases), sign=-1.0) for phases in (1, 2, 4, 8)]
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], atol=1e-06, rtol=1e-06)

def test_invalid_measure_is_rejected() -> None:
    measure = PhaseFlowMeasure(atoms=torch.zeros(1, 2, 2, 4, 4), weights=torch.tensor([[1.1, -0.1]]))
    with pytest.raises(ValueError, match='non-negative'):
        measure.validate()
