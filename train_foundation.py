from __future__ import annotations
import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from data.dataset import CompactBlurGyroDataset
from models.anchored_incremental_receiver import gyro_foundation_restore
from models.receiver import MultiScalePhysicalReceiver
from training.sampling import scene_balanced_order
from training.image_loss import epoch_loader, psnr, psnr_loss

def atomic_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, destination)

def atomic_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, destination)

def parse_blocks(value: str) -> tuple[int, ...]:
    result = tuple((int(item) for item in value.split(',')))
    if len(result) != 3 or min(result) < 1:
        raise argparse.ArgumentTypeError('blocks must contain three positive integers')
    return result

def build_receiver(args: argparse.Namespace, device: torch.device) -> MultiScalePhysicalReceiver:
    receiver = MultiScalePhysicalReceiver(width=args.width, encoder_blocks=args.encoder_blocks, middle_blocks=args.middle_blocks, decoder_blocks=args.decoder_blocks, physics_mode=getattr(args, 'physics_mode', 'all_adjoint'), motion_block_mode=getattr(args, 'motion_block_mode', 'depth_aware'), output_gate_mode=getattr(args, 'output_gate_mode', 'learned')).to(device)
    for parameter in receiver.geometry.parameters():
        parameter.requires_grad_(False)
    if receiver.physics_mode == 'none':
        for block in receiver.physical_blocks():
            for parameter in block.parameters():
                parameter.requires_grad_(False)
    return receiver

def gyro_foundation(receiver: MultiScalePhysicalReceiver, blur: torch.Tensor, gyro: torch.Tensor, clamp_output: bool=True) -> torch.Tensor:
    return gyro_foundation_restore(receiver, blur, gyro, clamp_output=clamp_output)

def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    scenes: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['motion_mode'])].append(row)
        scenes[str(row['scene'])].append(row)

    def one(items: list[dict[str, object]]) -> dict[str, float]:
        return {'psnr': sum((float(item['psnr']) for item in items)) / len(items), 'count': len(items)}
    return {'overall': one(rows), 'by_motion_mode': {mode: one(items) for mode, items in sorted(grouped.items())}, 'by_scene': {scene: one(items) for scene, items in sorted(scenes.items())}}

@torch.inference_mode()
def evaluate(receiver: MultiScalePhysicalReceiver, manifest: Path, device: torch.device, crop_size: int, workers: int, max_samples: int) -> dict[str, object]:
    dataset = CompactBlurGyroDataset(manifest, crop_size=crop_size)
    records = dataset.records[:min(max_samples, len(dataset))] if max_samples else dataset.records
    subset = Subset(dataset, range(len(records)))
    loader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    receiver.eval()
    rows = []
    for raw in tqdm(loader, desc='anchored-foundation-eval', leave=False):
        blur = raw['blur'].to(device, non_blocking=True).float()
        sharp = raw['sharp'].to(device, non_blocking=True).float()
        gyro = raw['gyro_flow'].to(device, non_blocking=True).float()
        values = psnr(gyro_foundation(receiver, blur, gyro), sharp)
        for index, name in enumerate(raw['name']):
            record = records[len(rows)]
            rows.append({'name': str(name), 'scene': str(record.get('scene', '')), 'motion_mode': str(record.get('motion_mode', 'unknown')), 'psnr': float(values[index])})
    return {**summarize(rows), 'records': rows}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-manifest', required=True, type=Path)
    parser.add_argument('--val-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--max-train-samples', type=int, default=0)
    parser.add_argument('--max-val-samples', type=int, default=200)
    parser.add_argument('--validation-every', type=int, default=5)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--encoder-blocks', type=parse_blocks, default=(2, 2, 2))
    parser.add_argument('--middle-blocks', type=int, default=16)
    parser.add_argument('--decoder-blocks', type=parse_blocks, default=(1, 1, 1))
    parser.add_argument('--motion-block-mode', choices=('clean_gyro',), default='clean_gyro')
    parser.add_argument('--physics-mode', choices=('none',), default='none')
    parser.add_argument('--output-gate-mode', choices=('fixed_one',), default='fixed_one')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=29000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest = args.output_dir / 'latest.pt'
    best = args.output_dir / 'best.pt'
    receiver = build_receiver(args, device)
    parameter_count = sum((parameter.numel() for parameter in receiver.parameters()))
    trainable_parameters = [parameter for parameter in receiver.parameters() if parameter.requires_grad]
    trainable_parameter_count = sum((parameter.numel() for parameter in trainable_parameters))
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-07)
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_psnr = -math.inf
    if args.resume and latest.is_file():
        state = torch.load(latest, map_location=device, weights_only=False)
        receiver.load_state_dict(state['receiver_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        history = list(state['history'])
        start_epoch = int(state['epoch']) + 1
        best_psnr = float(state['best_psnr'])
    full = CompactBlurGyroDataset(args.train_manifest, crop_size=args.crop_size, random_crop=True)
    order = scene_balanced_order(full.records, args.seed)
    limit = len(full) if args.max_train_samples <= 0 else min(args.max_train_samples, len(full))
    dataset = Subset(full, order[:limit])
    for epoch in range(start_epoch, args.epochs + 1):
        loader = epoch_loader(dataset, epoch=epoch, seed=args.seed, batch_size=args.batch_size, workers=args.workers)
        receiver.train()
        losses = []
        for raw in tqdm(loader, desc=f'anchored-foundation-{epoch}'):
            blur = raw['blur'].to(device, non_blocking=True).float()
            sharp = raw['sharp'].to(device, non_blocking=True).float()
            gyro = raw['gyro_flow'].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            output = gyro_foundation(receiver, blur, gyro, clamp_output=False)
            loss = psnr_loss(output, sharp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row: dict[str, object] = {'epoch': epoch, 'train_loss': sum(losses) / len(losses), 'lr': optimizer.param_groups[0]['lr']}
        validation = None
        if epoch % args.validation_every == 0 or epoch == args.epochs:
            validation = evaluate(receiver, args.val_manifest, device, args.crop_size, args.workers, args.max_val_samples)
            value = float(validation['overall']['psnr'])
            row['validation_psnr'] = value
            if value > best_psnr:
                best_psnr = value
                atomic_save({'format': 'anchored_receiver_foundation_v1', 'epoch': epoch, 'best_psnr': best_psnr, 'receiver_state_dict': receiver.state_dict(), 'architecture': {'width': args.width, 'encoder_blocks': args.encoder_blocks, 'middle_blocks': args.middle_blocks, 'decoder_blocks': args.decoder_blocks, 'motion_block_mode': args.motion_block_mode, 'physics_mode': args.physics_mode, 'output_gate_mode': args.output_gate_mode}, 'protocol': {'inference_inputs': ['blur', 'gyro', 'intrinsics'], 'trajectory': 'gyro_only', 'loss': 'PSNRLoss', 'training_output_clamped': False, 'evaluation_output_clamped': True, 'motion_mode_used': False, 'depth_or_gt_trajectory_used': False, 'parameter_count': parameter_count, 'trainable_parameter_count': trainable_parameter_count, 'precision': 'fp32', 'optimizer': 'Adam', 'scheduler': 'CosineAnnealingLR', 'scene_balanced_order': True, 'foundation_physics_mode': args.physics_mode, 'output_gate_mode': args.output_gate_mode}, 'validation': validation}, best)
        history.append(row)
        atomic_save({'format': 'anchored_receiver_foundation_resume_v1', 'epoch': epoch, 'best_psnr': best_psnr, 'receiver_state_dict': receiver.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'history': history, 'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}, latest)
        (args.output_dir / 'history.jsonl').write_text('\n'.join((json.dumps(item) for item in history)) + '\n', encoding='utf-8')
        atomic_json({'status': 'running', 'epoch': epoch, 'epochs': args.epochs, 'best_psnr': best_psnr, 'parameter_count': parameter_count}, args.output_dir / 'progress.json')
        print(json.dumps(row), flush=True)
    atomic_json({'status': 'completed', 'epoch': args.epochs, 'epochs': args.epochs, 'best_psnr': best_psnr, 'parameter_count': parameter_count, 'best_checkpoint': str(best)}, args.output_dir / 'completed.json')
if __name__ == '__main__':
    main()
