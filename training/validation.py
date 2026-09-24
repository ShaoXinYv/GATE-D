from __future__ import annotations
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from models.anchored_incremental_receiver import gyro_foundation_restore
from data.dataset import CompactExactDataset
from models.operator_repr import project_gyro_anchored_translation
from models.phase_set_motion import fit_translations_from_gyro_residual_flow
from training.load_geometry import summarize
from training.sampling import scene_balanced_order
from training.image_loss import psnr
from train_geometry import prediction, set_trainable
from training.restoration import phase_restore_trainable

def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_geometry_mode(base: torch.nn.Module, joint: bool, stage: str='last_encoder') -> list[dict[str, object]]:
    if not joint:
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        base.eval()
        return []
    groups = set_trainable(base, stage)
    base.eval()
    for module in (base.res_traj, base.motion_decoder, base.motion_fuse, base.gyro_embed):
        if any((parameter.requires_grad for parameter in module.parameters())):
            module.train()
    return groups

def solve_total(base: torch.nn.Module, batch: dict[str, torch.Tensor], inference_only: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if inference_only:
        batch = {**batch, 'valid': torch.ones_like(batch['blur'][:, :1])}
    residual, inverse_depth = prediction(base, batch)
    if getattr(base, 'disable_translation_solver', False):
        return (batch['gyro_flow'] + residual, residual, inverse_depth)
    with torch.autocast(device_type=residual.device.type, enabled=False):
        residual_fp32 = residual.float()
        depth_fp32 = inverse_depth.float()
        rotations = batch['gyro_rotations'].float()
        intrinsics = batch['intrinsics'].float()
        valid = batch['valid'].float()
        translations = fit_translations_from_gyro_residual_flow(residual_fp32, rotations, depth_fp32, intrinsics, valid)
        total = project_gyro_anchored_translation(depth_fp32, rotations, translations, intrinsics)
    return (total, residual, inverse_depth)

def cross_scene_reverse(records: list[dict[str, object]]) -> torch.Tensor:
    count = len(records)
    if count <= 1:
        return torch.arange(count)
    scenes = [str(record.get('scene', '')) for record in records]
    for shift in range(1, count):
        order = [(index + shift) % count for index in range(count)]
        if all((scenes[index] != scenes[order[index]] for index in range(count))):
            return torch.tensor(order)
    return torch.arange(count - 1, -1, -1)

@torch.inference_mode()
def evaluate(base: torch.nn.Module, receiver: torch.nn.Module, foundation: torch.nn.Module, dataset: CompactExactDataset, device: torch.device, workers: int, batch_size: int, reliability_tau: float, selection: str='legacy_safe') -> dict[str, object]:
    base.eval()
    receiver.eval()
    foundation.eval()
    if selection == 'prediction_psnr':
        seed_all(99173)
    order = scene_balanced_order(dataset.records, 99173)
    loader = DataLoader(Subset(dataset, order), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    rows: list[dict[str, object]] = []
    offset = 0
    for raw in tqdm(loader, desc='receiver-validation', leave=False):
        batch = {key: value.to(device, non_blocking=True).float() for key, value in raw.items() if torch.is_tensor(value)}
        blur, sharp, gyro = (batch['blur'], batch['sharp'], batch['gyro_flow'])
        predicted_total, residual, inverse_depth = solve_total(base, batch, selection == 'prediction_psnr')
        active = torch.ones(len(blur), device=device)
        foundation_image = gyro_foundation_restore(foundation, blur, gyro, True)
        predicted_image = phase_restore_trainable(receiver, blur, gyro, predicted_total, active, reliability_tau)
        target_image = phase_restore_trainable(receiver, blur, gyro, batch['full_trajectory'], active, reliability_tau)
        student_gyro = None
        if selection == 'prediction_psnr':
            student_gyro = phase_restore_trainable(receiver, blur, gyro, gyro, torch.zeros_like(active), reliability_tau)
        batch_records = [dataset.records[index] for index in order[offset:offset + len(blur)]]
        permutation = cross_scene_reverse(batch_records).to(device)
        shuffled_translations = fit_translations_from_gyro_residual_flow(residual[permutation], batch['gyro_rotations'], inverse_depth, batch['intrinsics'], torch.ones_like(batch['valid']) if selection == 'prediction_psnr' else batch['valid'])
        shuffled_total = project_gyro_anchored_translation(inverse_depth, batch['gyro_rotations'], shuffled_translations, batch['intrinsics'])
        if getattr(base, 'disable_translation_solver', False):
            shuffled_total = gyro + residual[permutation]
        shuffled_image = phase_restore_trainable(receiver, blur, gyro, shuffled_total, active, reliability_tau)
        values = {'foundation_psnr': psnr(foundation_image, sharp), 'predicted_psnr': psnr(predicted_image, sharp), 'shuffled_psnr': psnr(shuffled_image, sharp), 'target_psnr': psnr(target_image, sharp)}
        if student_gyro is not None:
            values['student_gyro_psnr'] = psnr(student_gyro, sharp)
        for local, name in enumerate(raw['name']):
            record = batch_records[local]
            rows.append({'index': offset + local, 'name': str(name), 'scene': str(record.get('scene', '')), 'motion_mode': str(record.get('motion_mode', '')), **{key: float(value[local]) for key, value in values.items()}})
        offset += len(blur)
    summary = summarize(rows)
    modes = summary
    overall = modes['overall']
    metrics = {'overall': float(overall['predicted_gain']), 'translation': float(modes['translation']['predicted_gain']), 'mixed': float(modes['mixed']['predicted_gain']), 'rotation': float(modes['rotation']['predicted_gain']), 'correct_minus_shuffled': float(overall['predicted_gain'] - overall['shuffled_gain']), 'target': float(overall['target_gain'])}
    gate = {'overall_positive': metrics['overall'] > 0.0, 'translation_nonnegative': metrics['translation'] >= 0.0, 'mixed_positive': metrics['mixed'] > 0.0, 'rotation_above_floor': metrics['rotation'] >= -0.05, 'correct_beats_shuffled_by_0p05': metrics['correct_minus_shuffled'] >= 0.05, 'target_positive': metrics['target'] > 0.0}
    gate['safe'] = all(gate.values())
    penalties = 8.0 * max(0.0, -metrics['translation']) + 6.0 * max(0.0, -metrics['mixed']) + 5.0 * max(0.0, -0.05 - metrics['rotation']) + 4.0 * max(0.0, 0.05 - metrics['correct_minus_shuffled'])
    score = metrics['overall'] + 0.25 * metrics['translation'] + 0.25 * metrics['mixed'] - penalties
    if selection == 'prediction_psnr':
        score = sum((row['predicted_psnr'] for row in rows)) / len(rows)
        student = sum((row['student_gyro_psnr'] for row in rows)) / len(rows)
        metrics['prediction_psnr'] = score
        metrics['student_gyro_psnr'] = student
        metrics['prediction_minus_student_gyro'] = score - student
    return {'summary': summary, 'metrics': metrics, 'gate': gate, 'score': score}
