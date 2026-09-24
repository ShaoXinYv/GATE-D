"""Evaluate the final model on a compact-data manifest."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data.dataset import CompactExactDataset
from models.image_metrics import gaussian_ssim
from training.validation import seed_all
from train_restoration import load_pair, restore


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'initial-checkpoint', 'foundation-checkpoint', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--crop-size', type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for restoration evaluation')
    if args.batch_size < 1 or args.crop_size < 32:
        raise ValueError('Invalid batch size or crop size')
    device = torch.device('cuda')
    base, receiver, foundation, tau = load_pair(args, args.checkpoint, device, 'ordered_single')
    del foundation
    base.eval()
    receiver.eval()
    dataset = CompactExactDataset(args.manifest, crop_size=args.crop_size,
                                  random_crop=False, return_pose_components=True)
    if not len(dataset):
        raise ValueError('Empty evaluation manifest')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    if args.image_dir:
        args.image_dir.mkdir(parents=True, exist_ok=True)
    seed_all(99173)
    rows = []
    for raw in tqdm(loader, desc='evaluation'):
        # Supervision fields never enter the prediction path.
        inputs = {k: raw[k].to(device).float()
                  for k in ('blur', 'gyro_flow', 'gyro_rotations', 'intrinsics')}
        sharp = raw['sharp'].to(device).float()
        output, _, _ = restore(base, receiver, inputs, tau)
        output = output.clamp(0, 1)
        psnr = -10 * (output - sharp).square().flatten(1).mean(1).clamp_min(1e-12).log10()
        ssim = gaussian_ssim(output, sharp)
        for i in range(len(output)):
            record = dataset.records[len(rows)]
            rows.append({'name': record['name'], 'motion_mode': record.get('motion_mode', 'unknown'),
                         'psnr': float(psnr[i]), 'ssim': float(ssim[i])})
            if args.image_dir:
                save_image(output[i], args.image_dir / f'{len(rows) - 1:06d}.png')
    summary = {}
    for group in ('overall', 'rotation', 'translation', 'mixed'):
        selected = rows if group == 'overall' else [r for r in rows if r['motion_mode'] == group]
        if selected:
            summary[group] = {'count': len(selected), **{
                metric: sum(r[metric] for r in selected) / len(selected)
                for metric in ('psnr', 'ssim')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'records': rows,
        'protocol': {'crop': args.crop_size, 'batch_size': args.batch_size,
                     'seed': 99173, 'precision': 'float32', 'quantize': False}}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
