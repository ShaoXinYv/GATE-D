import importlib

import pytest
import torch

from models.configuration import configure_variant
from models.receiver import MultiScalePhysicalReceiver
from train_restoration import distillation_weight


@pytest.mark.parametrize('name', ['train_geometry', 'train_foundation', 'train_restoration', 'evaluate'])
def test_entrypoint_imports(name):
    assert callable(importlib.import_module(name).main)


def test_teacher_and_student_configuration():
    base = torch.nn.Identity()
    receiver = torch.nn.Identity()
    configure_variant(base, receiver, 'ordered_single')
    assert receiver.exposure_phase_groups == 8
    assert receiver.residual_reference_mode == 'identity'
    configure_variant(base, receiver, 'full')
    assert receiver.exposure_phase_groups == 4
    assert receiver.residual_reference_mode == 'rotation'
    with pytest.raises(ValueError):
        configure_variant(base, receiver, 'ordered_single_no_solver')


def test_retired_receiver_modes_are_rejected():
    with pytest.raises(ValueError):
        MultiScalePhysicalReceiver(motion_block_mode='depth_aware')


def test_distillation_schedule():
    assert distillation_weight(0.4, 'repair', 15) == 0.4
    assert distillation_weight(0.4, 'joint', 1) == 0.4
    assert distillation_weight(0.4, 'joint', 20) == 0
    assert distillation_weight(0.4, 'joint', 60) == 0
