def configure_variant(base, receiver, variant):
    if variant not in ('ordered_single', 'full'):
        raise ValueError(variant)
    base.disable_translation_solver = False
    receiver.exposure_phase_groups = 8 if variant == 'ordered_single' else 4
    receiver.residual_reference_mode = 'identity' if variant == 'ordered_single' else 'rotation'
    receiver.residual_operator_override = None
    receiver.force_zero_residual = False
