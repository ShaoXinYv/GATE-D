from __future__ import annotations
import argparse
import torch
from training.receiver_setup import load_foundation

def load_receiver(args: argparse.Namespace, device: torch.device):
    if args.foundation_checkpoint is None or args.receiver_checkpoint is None:
        return (None, None, None)
    foundation = load_foundation(args.foundation_checkpoint, device).eval()
    receiver = load_foundation(args.foundation_checkpoint, device).eval()
    state = torch.load(args.receiver_checkpoint, map_location=device, weights_only=False)
    receiver.enable_residual_physical_adapters()
    receiver.load_state_dict(state['receiver_state_dict'], strict=True)
    reliability_tau = float(state.get('reliability_tau', 1.0))
    for module in (foundation, receiver):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return (foundation, receiver, reliability_tau)
