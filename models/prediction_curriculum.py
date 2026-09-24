from __future__ import annotations
import torch

def enable_joint_receiver(receiver: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Open the active image/gyro backbone and differential physical blocks."""
    if receiver.physics_mode != 'none' or receiver.motion_block_mode != 'clean_gyro':
        raise ValueError('joint curriculum expects the audited clean_gyro foundation')
    if receiver.output_gate_mode != 'fixed_one':
        raise ValueError('joint curriculum requires fixed_one image output')
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    receiver.enable_residual_physical_adapters()
    modules = [receiver.intro, receiver.intro_gyro, receiver.ending, receiver.encoders, receiver.downs, receiver.gyro_refine_blks, receiver.middle_blks, receiver.gyro_deblurring_blks, receiver.ups, receiver.decoders, *receiver.residual_physical_blocks()]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in receiver.parameters() if parameter.requires_grad]
