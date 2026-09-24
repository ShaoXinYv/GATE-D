from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from models.checkpoint import load_bundle
from data.dataset import CompactExactDataset
from models.geometry import normalize_inverse_depth
from models.operator_repr import apply_weighted_flow_measure, gauge_fix_depth_translation, phase_product_flow_measure, project_gyro_anchored_translation
from models.phase_set_motion import fit_translations_from_gyro_residual_flow, gyro_phase_observability

def atomic_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, destination)

def fold_legacy_endpoint_into_phase_head(base: torch.nn.Module) -> None:
    """Make res_traj exactly reproduce the old endpoint-plus-delta output."""
    with torch.no_grad():
        trajectory_weight = base.res_traj.weight.clone()
        trajectory_bias = base.res_traj.bias.clone()
        endpoint_weight = base.res_endpoint.weight
        endpoint_bias = base.res_endpoint.bias
        for step in range(8):
            if step < 4:
                endpoint_pair = 0
                coefficient = (4 - step) / 4.0
            else:
                endpoint_pair = 1
                coefficient = (step - 3) / 4.0
            for axis in range(2):
                output = 2 * step + axis
                endpoint = 2 * endpoint_pair + axis
                trajectory_weight[output] += coefficient * endpoint_weight[endpoint]
                trajectory_bias[output] += coefficient * endpoint_bias[endpoint]
        base.res_traj.weight.copy_(trajectory_weight)
        base.res_traj.bias.copy_(trajectory_bias)

def load_geometry(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    bundle = load_bundle(args.initial_checkpoint, device)
    base = bundle.motion.base
    del bundle
    if args.geometry_checkpoint is not None:
        payload = torch.load(args.geometry_checkpoint, map_location='cpu', weights_only=False)
        base.load_state_dict(payload['base_state_dict'], strict=True)
    else:
        fold_legacy_endpoint_into_phase_head(base)
    return base.to(device)

def set_trainable(base: torch.nn.Module, stage: str) -> list[dict[str, object]]:
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    groups: list[dict[str, object]] = []
    for parameter in base.res_traj.parameters():
        parameter.requires_grad_(True)
    groups.append({'params': list(base.res_traj.parameters()), 'lr_scale': 1.0})
    if stage in {'decoder', 'last_encoder', 'last_two_encoders'}:
        decoder_parameters = []
        for module in (base.motion_decoder, base.motion_fuse, base.gyro_embed):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                decoder_parameters.append(parameter)
        groups.append({'params': decoder_parameters, 'lr_scale': 0.2})
    if stage in {'last_encoder', 'last_two_encoders'}:
        encoder_parameters = []
        stages = (3, 4) if stage == 'last_two_encoders' else (4,)
        tokens = tuple((f'{prefix}{index}' for index in stages for prefix in ('patch_embed', 'block', 'norm')))
        for name, parameter in base.backbone.named_parameters():
            if any((token in name for token in tokens)):
                parameter.requires_grad_(True)
                encoder_parameters.append(parameter)
        if not encoder_parameters:
            raise RuntimeError('could not identify the last MSCAN encoder stage')
        groups.append({'params': encoder_parameters, 'lr_scale': 0.1 if stage == 'last_two_encoders' else 0.03})
    return groups

def prediction(base: torch.nn.Module, batch: dict[str, torch.Tensor], inference_only: bool=False):
    _endpoint, dense, inverse_depth, _confidence, _features = base(batch['blur'], batch['gyro_flow'])
    return (dense * float(base.flow_scale), normalize_inverse_depth(inverse_depth, torch.ones_like(batch['blur'][:, :1]) if inference_only else batch['valid']))

def reduce_resolution(flow: torch.Tensor, valid: torch.Tensor, divisor: int) -> tuple[torch.Tensor, torch.Tensor]:
    if divisor <= 1:
        return (flow, valid)
    size = (max(1, flow.shape[-2] // divisor), max(1, flow.shape[-1] // divisor))
    return (F.interpolate(flow, size=size, mode='bilinear', align_corners=False), F.interpolate(valid, size=size, mode='nearest'))

def flow_scale(flow: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = flow.shape
    steps = channels // 2
    values = flow.reshape(batch, steps, 2, height, width)
    weight = valid[:, None]
    numerator = (values.square() * weight).sum((1, 2, 3, 4))
    denominator = weight.sum((1, 2, 3, 4)).clamp_min(1.0) * steps * 2
    return (numerator / denominator).clamp_min(1e-12).sqrt()

def robust_distance(left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    difference = left - right
    error = torch.sqrt(difference.square() + 1e-06)
    weight = valid[:, None, None]
    return (error * weight).sum((-1, -2, -3)) / (weight.sum((-1, -2, -3)).clamp_min(1.0) * 2)

def balanced_active_mean(values: torch.Tensor, active: torch.Tensor, observability: torch.Tensor, *, enabled: bool, threshold: float) -> torch.Tensor:
    """Balance low/high gyro evidence without using a motion-class label."""
    active = active.to(values)
    if not enabled:
        return (values * active).sum() / active.sum().clamp_min(1.0)
    low = active * (observability < threshold).to(values)
    high = active * (observability >= threshold).to(values)
    terms = []
    if bool(low.sum() > 0):
        terms.append((values * low).sum() / low.sum())
    if bool(high.sum() > 0):
        terms.append((values * high).sum() / high.sum())
    return torch.stack(terms).mean() if terms else values.new_zeros(())

def exposure_covariance(flow: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=flow.device.type, enabled=False):
        flow = flow.float()
        valid = valid.float()
        batch, channels, height, width = flow.shape
        points = flow.reshape(batch, channels // 2, 2, height, width)
        weight = valid[:, None].to(flow)
        denominator = (weight.sum((1, 3, 4)) * points.shape[1]).clamp_min(1.0)
        mean = (points * weight).sum((1, 3, 4)) / denominator
        centered = points - mean[:, None, :, None, None]
        return torch.einsum('btihw,btjhw->bij', centered * weight, centered) / denominator[:, None]

def covariance_moment_error(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_covariance = exposure_covariance(predicted, valid)
    target_covariance = exposure_covariance(target, valid)
    predicted_trace = predicted_covariance.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-08)
    target_trace = target_covariance.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-08)
    trace_error = (predicted_trace.log() - target_trace.log()).abs()
    predicted_shape = predicted_covariance / predicted_trace[:, None, None]
    target_shape = target_covariance / target_trace[:, None, None]
    anisotropy_error = (predicted_shape - target_shape).square().sum((1, 2)).sqrt()
    return (trace_error + anisotropy_error, trace_error, anisotropy_error)

def operator_error(predicted_residual: torch.Tensor, target_residual: torch.Tensor, gyro: torch.Tensor, sharp: torch.Tensor, *, divisor: int) -> torch.Tensor:
    size = (max(1, sharp.shape[-2] // divisor), max(1, sharp.shape[-1] // divisor))
    sharp_small = F.interpolate(sharp, size=size, mode='area')
    scale_x = size[1] / sharp.shape[-1]
    scale_y = size[0] / sharp.shape[-2]
    if abs(scale_x - scale_y) > 1e-08:
        raise ValueError('operator loss requires isotropic resizing')

    def resize_flow(flow: torch.Tensor) -> torch.Tensor:
        return F.interpolate(flow, size=size, mode='bilinear', align_corners=False) * scale_x
    batch = len(sharp)
    gyro_steps = resize_flow(gyro).reshape(batch, 8, 2, *size)
    predicted_steps = resize_flow(predicted_residual).reshape(batch, 8, 2, *size)
    target_steps = resize_flow(target_residual).reshape_as(predicted_steps)
    predicted_measure = phase_product_flow_measure(gyro_steps, predicted_steps, 4)
    target_measure = phase_product_flow_measure(gyro_steps, target_steps, 4)
    predicted_action = apply_weighted_flow_measure(sharp_small, predicted_measure, sign=-1.0)
    target_action = apply_weighted_flow_measure(sharp_small, target_measure, sign=-1.0)
    return F.smooth_l1_loss(predicted_action, target_action, beta=0.01, reduction='none').mean((1, 2, 3))

def geometry_losses(predicted: torch.Tensor, target: torch.Tensor, predicted_depth: torch.Tensor, target_depth: torch.Tensor, gyro: torch.Tensor, sharp: torch.Tensor, valid: torch.Tensor, *, loss_divisor: int, minimum_motion: float, phase_weight: float=0.25, scale_weight: float=0.25, moment_weight: float=0.0, operator_weight: float=0.0, operator_divisor: int=8, zero_weight: float=1.0, depth_weight: float=0.05, phase_observability: bool=False, observability_tau: float=0.5, balance_gyro: bool=False, balance_threshold: float=0.05, representation: str='phase4') -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if representation != 'phase4':
        raise ValueError(f'unknown representation: {representation}')
    predicted_small, valid_small = reduce_resolution(predicted, valid, loss_divisor)
    target_small, _ = reduce_resolution(target, valid, loss_divisor)
    batch, channels, height, width = predicted_small.shape
    predicted_steps = predicted_small.reshape(batch, 8, 2, height, width)
    target_steps = target_small.reshape_as(predicted_steps)
    predicted_scale = flow_scale(predicted_small, valid_small)
    target_scale = flow_scale(target_small, valid_small)
    active = target_scale > minimum_motion
    normalized_predicted = predicted_steps / predicted_scale[:, None, None, None, None]
    normalized_target = target_steps / target_scale[:, None, None, None, None]
    pairwise = robust_distance(normalized_predicted[:, :, None], normalized_target[:, None, :], valid_small)
    global_shape = 0.5 * (pairwise.min(2).values.mean(1) + pairwise.min(1).values.mean(1))
    phase_predicted = normalized_predicted.reshape(batch, 4, 2, 2, height, width)
    phase_target = normalized_target.reshape_as(phase_predicted)
    direct = robust_distance(phase_predicted, phase_target, valid_small).sum(2)
    swapped = robust_distance(phase_predicted, phase_target.flip(2), valid_small).sum(2)
    phase_shape = torch.minimum(direct, swapped).mean(1)
    active_weight = active.float()
    observability = gyro_phase_observability(gyro, tau_px=observability_tau)
    shape_loss = balanced_active_mean(global_shape, active_weight, observability, enabled=balance_gyro, threshold=balance_threshold)
    phase_observability_weight = observability if phase_observability else torch.ones_like(observability)
    effective_phase_weight = active_weight * phase_observability_weight
    phase_loss = (phase_shape * effective_phase_weight).sum() / effective_phase_weight.sum().clamp_min(1.0)
    scale_error = (predicted_scale.add(0.0001).log() - target_scale.add(0.0001).log()).abs()
    scale_loss = balanced_active_mean(scale_error, active_weight, observability, enabled=balance_gyro, threshold=balance_threshold)
    moment_error, trace_error, anisotropy_error = covariance_moment_error(predicted_small, target_small, valid_small)
    moment_loss = balanced_active_mean(moment_error, active_weight, observability, enabled=balance_gyro, threshold=balance_threshold)
    if operator_weight > 0:
        forward_operator_error = operator_error(predicted, target, gyro, sharp, divisor=operator_divisor)
        forward_operator_loss = balanced_active_mean(forward_operator_error, active_weight, observability, enabled=balance_gyro, threshold=balance_threshold)
    else:
        forward_operator_loss = predicted.new_zeros(())
    low = (~active).float()
    zero_loss = (predicted_scale * low).sum() / low.sum().clamp_min(1.0)
    target_inverse = normalize_inverse_depth(target_depth.clamp_min(1e-06).reciprocal(), valid)
    depth_difference = (predicted_depth.clamp_min(1e-06).log() - target_inverse.clamp_min(1e-06).log()).abs()
    depth_loss = (depth_difference * valid).sum() / valid.sum().clamp_min(1.0)
    total = shape_loss + phase_weight * phase_loss + scale_weight * scale_loss + moment_weight * moment_loss + operator_weight * forward_operator_loss + zero_weight * zero_loss + depth_weight * depth_loss
    return (total, {'loss': total.detach(), 'global_shape': shape_loss.detach(), 'phase_shape': phase_loss.detach(), 'log_scale': scale_loss.detach(), 'moment': moment_loss.detach(), 'moment_trace': balanced_active_mean(trace_error, active_weight, observability, enabled=balance_gyro, threshold=balance_threshold).detach(), 'moment_anisotropy': balanced_active_mean(anisotropy_error, active_weight, observability, enabled=balance_gyro, threshold=balance_threshold).detach(), 'operator': forward_operator_loss.detach(), 'zero': zero_loss.detach(), 'depth': depth_loss.detach(), 'gyro_observability': observability.mean().detach(), 'low_gyro_fraction': (observability < balance_threshold).float().mean().detach(), 'predicted_scale': predicted_scale.mean().detach(), 'target_scale': target_scale.mean().detach()})

def tensor_batch(raw: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True).float() for key, value in raw.items() if torch.is_tensor(value)}

def make_loader(manifest: Path, max_samples: int, batch_size: int, workers: int, crop_size: int, random_crop: bool, shuffle: bool) -> tuple[CompactExactDataset, DataLoader]:
    complete = CompactExactDataset(manifest, crop_size=crop_size, random_crop=random_crop, return_pose_components=True)
    dataset = complete
    if max_samples > 0:
        dataset = Subset(complete, range(min(max_samples, len(complete))))
    return (complete, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, drop_last=shuffle and len(dataset) >= batch_size))

def train_epoch(base: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    base.eval()
    for module in (base.res_traj, base.motion_decoder, base.motion_fuse, base.gyro_embed):
        if any((parameter.requires_grad for parameter in module.parameters())):
            module.train()
    if getattr(args, 'freeze_normalization', False):
        for module in base.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for raw in tqdm(loader, desc='train', leave=False):
        batch = tensor_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=args.amp):
            predicted, predicted_depth = prediction(base, batch, args.stage == 'last_two_encoders')
            loss, metrics = geometry_losses(predicted, batch['residual_trajectory'], predicted_depth, batch['depth'], batch['gyro_flow'], batch['sharp'], batch['valid'], loss_divisor=args.loss_divisor, minimum_motion=args.minimum_motion, phase_weight=args.phase_weight, scale_weight=args.scale_weight, moment_weight=args.moment_weight, operator_weight=args.operator_weight, operator_divisor=args.operator_divisor, zero_weight=args.zero_weight, depth_weight=args.depth_weight, phase_observability=args.phase_observability, observability_tau=args.observability_tau, balance_gyro=args.balance_gyro, balance_threshold=args.balance_threshold, representation=getattr(args, 'representation', 'phase4'))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([parameter for parameter in base.parameters() if parameter.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        size = len(batch['blur'])
        for key, value in metrics.items():
            totals[key] += float(value) * size
        count += size
    return {key: value / max(count, 1) for key, value in totals.items()}

@torch.inference_mode()
def validate(base: torch.nn.Module, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    base.eval()
    if args.stage == 'last_two_encoders':
        torch.manual_seed(99173)
        torch.cuda.manual_seed_all(99173)
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for raw in tqdm(loader, desc='validation', leave=False):
        batch = tensor_batch(raw, device)
        predicted, predicted_depth = prediction(base, batch, args.stage == 'last_two_encoders')
        _loss, metrics = geometry_losses(predicted, batch['residual_trajectory'], predicted_depth, batch['depth'], batch['gyro_flow'], batch['sharp'], batch['valid'], loss_divisor=args.loss_divisor, minimum_motion=args.minimum_motion, phase_weight=args.phase_weight, scale_weight=args.scale_weight, moment_weight=args.moment_weight, operator_weight=args.operator_weight, operator_divisor=args.operator_divisor, zero_weight=args.zero_weight, depth_weight=args.depth_weight, phase_observability=args.phase_observability, observability_tau=args.observability_tau, balance_gyro=args.balance_gyro, balance_threshold=args.balance_threshold, representation=getattr(args, 'representation', 'phase4'))
        translations = batch['full_poses'][:, :, :3, 3]
        fixed = gauge_fix_depth_translation(batch['depth'], translations, batch['valid'])
        solve_depth = predicted_depth if args.stage == 'last_two_encoders' else fixed.inverse_depth
        solve_valid = torch.ones_like(batch['valid']) if args.stage == 'last_two_encoders' else batch['valid']
        fitted = fit_translations_from_gyro_residual_flow(predicted, batch['gyro_rotations'], solve_depth, batch['intrinsics'], solve_valid)
        total = project_gyro_anchored_translation(solve_depth, batch['gyro_rotations'], fitted, batch['intrinsics'])
        total_error = total - batch['full_trajectory']
        steps = total_error.shape[1] // 2
        epe = total_error.reshape(len(total_error), steps, 2, *total_error.shape[-2:])
        epe = epe.square().sum(2).sqrt().mean(1, keepdim=True)
        epe = (epe * batch['valid']).sum((1, 2, 3)) / batch['valid'].sum((1, 2, 3)).clamp_min(1.0)
        size = len(batch['blur'])
        for key, value in metrics.items():
            totals[key] += float(value) * size
        totals['solver_total_epe'] += float(epe.sum())
        count += size
    return {key: value / max(count, 1) for key, value in totals.items()}

def save_checkpoint(path: Path, base: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, history: list[dict[str, object]], args: argparse.Namespace, scheduler=None, scaler=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'format': 'iaai_residual_multiphase_geometry_v1', 'base_state_dict': base.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'history': history, 'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None, 'scaler_state_dict': scaler.state_dict() if scaler is not None else None, 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}, temporary)
    os.replace(temporary, path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-manifest', required=True, type=Path)
    parser.add_argument('--validation-manifest', required=True, type=Path)
    parser.add_argument('--initial-checkpoint', required=True, type=Path)
    parser.add_argument('--geometry-checkpoint', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stage', choices=('head', 'decoder', 'last_encoder', 'last_two_encoders'), default='head')
    parser.add_argument('--representation', choices=('phase4',), default='phase4')
    parser.add_argument('--train-depth', action='store_true')
    parser.add_argument('--freeze-normalization', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=5e-05)
    parser.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    parser.add_argument('--minimum-lr-ratio', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--loss-divisor', type=int, default=4)
    parser.add_argument('--minimum-motion', type=float, default=0.05)
    parser.add_argument('--phase-weight', type=float, default=0.25)
    parser.add_argument('--scale-weight', type=float, default=0.25)
    parser.add_argument('--moment-weight', type=float, default=0.0)
    parser.add_argument('--operator-weight', type=float, default=0.0)
    parser.add_argument('--operator-divisor', type=int, default=8)
    parser.add_argument('--zero-weight', type=float, default=1.0)
    parser.add_argument('--depth-weight', type=float, default=0.05)
    parser.add_argument('--phase-observability', action='store_true')
    parser.add_argument('--observability-tau', type=float, default=0.5)
    parser.add_argument('--balance-gyro', action='store_true')
    parser.add_argument('--balance-threshold', type=float, default=0.05)
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--max-train-samples', type=int, default=0)
    parser.add_argument('--max-validation-samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20260901)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--minimum-free-gb', type=float, default=10.0)
    args = parser.parse_args()
    if min(args.phase_weight, args.scale_weight, args.moment_weight, args.operator_weight, args.zero_weight, args.depth_weight) < 0:
        raise ValueError('loss weights must be non-negative')
    if args.observability_tau <= 0 or not 0 <= args.balance_threshold <= 1:
        raise ValueError('invalid observability configuration')
    if args.operator_divisor <= 0:
        raise ValueError('operator divisor must be positive')
    if not 0 <= args.minimum_lr_ratio <= 1:
        raise ValueError('minimum LR ratio must lie in [0, 1]')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_geometry(args, device)
    parameter_groups = set_trainable(base, args.stage)
    if args.train_depth:
        for module, scale in ((base.inv_depth, 1.0), (base.depth_decoder, 0.2)):
            params = list(module.parameters())
            for parameter in params:
                parameter.requires_grad_(True)
            parameter_groups.append({'params': params, 'lr_scale': scale})
    optimizer = torch.optim.AdamW([{'params': group['params'], 'lr': args.learning_rate * float(group['lr_scale'])} for group in parameter_groups], weight_decay=0.0001)
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda completed: args.minimum_lr_ratio + (1.0 - args.minimum_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * completed / max(args.epochs, 1))))
    else:
        scheduler = None
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    _train_dataset, train_loader = make_loader(args.train_manifest, args.max_train_samples, args.batch_size, args.workers, args.crop_size, True, True)
    _validation_dataset, validation_loader = make_loader(args.validation_manifest, args.max_validation_samples, args.batch_size, args.workers, args.crop_size, False, False)
    history: list[dict[str, object]] = []
    best_score = math.inf
    start_epoch = 1
    if args.resume and (args.output_dir / 'last.pt').is_file():
        state = torch.load(args.output_dir / 'last.pt', map_location=device, weights_only=False)
        for key in ('epochs', 'stage', 'lr_schedule', 'learning_rate'):
            if state['args'][key] != getattr(args, key):
                raise ValueError(f'resume protocol changed: {key}')
        for key, default in (('representation', 'phase4'), ('train_depth', False), ('freeze_normalization', False)):
            if state['args'].get(key, default) != getattr(args, key):
                raise ValueError(f'resume protocol changed: {key}')
        base.load_state_dict(state['base_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler is not None:
            if state.get('scheduler_state_dict') is None:
                raise ValueError('checkpoint predates resumable scheduler support')
            scheduler.load_state_dict(state['scheduler_state_dict'])
        if state.get('scaler_state_dict') is not None:
            scaler.load_state_dict(state['scaler_state_dict'])
        history = state['history']
        best_score = min((row['selection_score'] for row in history))
        start_epoch = int(state['epoch']) + 1
    for epoch in range(start_epoch, args.epochs + 1):
        if shutil.disk_usage(args.output_dir).free < args.minimum_free_gb * 1024 ** 3:
            raise RuntimeError('free space below configured floor; latest checkpoint retained')
        if args.stage == 'last_two_encoders':
            random.seed(args.seed + epoch)
            np.random.seed(args.seed + epoch)
            torch.manual_seed(args.seed + epoch)
            torch.cuda.manual_seed_all(args.seed + epoch)
        training = train_epoch(base, train_loader, optimizer, scaler, args, device)
        validation = validate(base, validation_loader, args, device)
        score = validation['global_shape'] + args.scale_weight * validation['log_scale'] + args.moment_weight * validation['moment'] + args.operator_weight * validation['operator']
        row = {'epoch': epoch, 'train': training, 'validation': validation, 'selection_score': score, 'learning_rates': [group['lr'] for group in optimizer.param_groups]}
        history.append(row)
        atomic_json({'status': 'running', 'history': history}, args.output_dir / 'status.json')
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(args.output_dir / 'last.pt', base, optimizer, epoch, history, args, scheduler, scaler)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.output_dir / 'candidates' / f'epoch_{epoch:03d}.pt', base, optimizer, epoch, history, args, scheduler, scaler)
        if score < best_score:
            best_score = score
            save_checkpoint(args.output_dir / 'best.pt', base, optimizer, epoch, history, args, scheduler, scaler)
        print(json.dumps(row), flush=True)
    atomic_json({'status': 'completed', 'best_score': best_score, 'history': history}, args.output_dir / 'status.json')
if __name__ == '__main__':
    main()
