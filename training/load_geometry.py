from __future__ import annotations
import argparse
from collections import defaultdict
import torch
from models.checkpoint import load_bundle

def load_geometry(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    bundle = load_bundle(args.initial_checkpoint, device)
    base = bundle.motion.base
    del bundle
    payload = torch.load(args.geometry_checkpoint, map_location='cpu', weights_only=False)
    base.load_state_dict(payload['base_state_dict'], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base

def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    groups['overall'] = rows
    for row in rows:
        groups[str(row['motion_mode'])].append(row)
    output: dict[str, object] = {}
    for group, items in groups.items():
        numeric = [key for key, value in items[0].items() if isinstance(value, (int, float)) and key != 'index']
        summary = {key: sum((float(item[key]) for item in items)) / len(items) for key in numeric}
        summary['count'] = len(items)
        foundation = float(summary['foundation_psnr'])
        for key in tuple(summary):
            if key.endswith('_psnr') and key != 'foundation_psnr':
                stem = key[:-5] if key.endswith('_psnr') else key
                summary[stem + '_gain'] = float(summary[key]) - foundation
        output[group] = summary
    return output
