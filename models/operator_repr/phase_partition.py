from __future__ import annotations
import torch

def contiguous_phase_ids(steps: int, phases: int, *, device: torch.device | None=None) -> torch.Tensor:
    """Assign ordered exposure slots to balanced contiguous phase bins."""
    if steps <= 0:
        raise ValueError('steps must be positive')
    if phases <= 0 or phases > steps:
        raise ValueError('phases must satisfy 1 <= phases <= steps')
    indices = torch.arange(steps, device=device)
    return torch.div(indices * phases, steps, rounding_mode='floor')
