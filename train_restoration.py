import argparse
import math
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from data.dataset import CompactExactDataset
from models.configuration import configure_variant
from models.prediction_curriculum import enable_joint_receiver
from training.load_receiver import load_receiver
from training.load_geometry import load_geometry
from training.receiver_setup import set_zero_safe_trainable_scope
from training.sampling import scene_balanced_order
from train_geometry import geometry_losses
from training.losses import restoration_loss
from training.validation import atomic_json, atomic_save, evaluate, seed_all, set_geometry_mode, solve_total
from training.restoration import phase_restore_trainable

def load_pair(args, checkpoint, device, variant):
    config = argparse.Namespace(initial_checkpoint=args.initial_checkpoint, geometry_checkpoint=checkpoint, foundation_checkpoint=args.foundation_checkpoint, receiver_checkpoint=checkpoint)
    base = load_geometry(config, device)
    foundation, receiver, tau = load_receiver(config, device)
    configure_variant(base, receiver, variant)
    return (base, receiver, foundation, tau)

def parameter_groups(base, receiver, stage):
    parameters = set_zero_safe_trainable_scope(receiver) if stage == 'repair' else enable_joint_receiver(receiver)
    groups = [{'params': parameters, 'lr': 1e-05}]
    geometry = set_geometry_mode(base, stage == 'joint', 'decoder')
    for index, group in enumerate(geometry):
        groups.append({'params': group['params'], 'lr': 2e-06 if index == 0 else 5e-07})
    return groups

def model_mode(base, receiver, stage):
    set_geometry_mode(base, stage == 'joint', 'decoder')
    receiver.train(stage == 'joint')
    for block in receiver.residual_physical_blocks():
        block.train()
    for module in list(base.modules()) + list(receiver.modules()):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

def restore(base, receiver, batch, tau):
    inputs = {k: batch[k] for k in ('blur', 'gyro_flow', 'gyro_rotations', 'intrinsics')}
    total, residual, depth = solve_total(base, inputs, inference_only=True)
    output = phase_restore_trainable(receiver, batch['blur'], batch['gyro_flow'], total, torch.ones(len(total), device=total.device), tau, clamp_output=False)
    return (output, residual, depth)

def geometry_loss(residual, depth, batch):
    loss, _ = geometry_losses(residual, batch['residual_trajectory'], depth, batch['depth'], batch['gyro_flow'], batch['sharp'], batch['valid'], loss_divisor=4, minimum_motion=0.05, phase_weight=0.25, scale_weight=0.25, moment_weight=0.0, operator_weight=0.0, operator_divisor=8, zero_weight=1.0, depth_weight=0.05, phase_observability=True, observability_tau=0.5, balance_gyro=True, balance_threshold=0.05, representation='phase4')
    return loss

def distillation_weight(initial, stage, epoch):
    return initial if stage == 'repair' else initial * max(0.0, 1.0 - (epoch - 1) / 19.0)

def guarded_save(state, path, minimum_free_gb):

    def tensor_bytes(value):
        if torch.is_tensor(value):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum((tensor_bytes(v) for v in value.values()))
        if isinstance(value, (list, tuple)):
            return sum((tensor_bytes(v) for v in value))
        return 0
    required = minimum_free_gb * 1024 ** 3 + tensor_bytes(state) + 64 * 1024 ** 2
    if shutil.disk_usage(path.parent).free < required:
        raise RuntimeError('Insufficient atomic-checkpoint reserve above configured floor')
    atomic_save(state, path)

def calibrate_weight(output, sharp, teacher):
    probe = output.detach().requires_grad_(True)
    a = torch.autograd.grad(restoration_loss(probe, sharp), probe)[0].norm()
    b = torch.autograd.grad((probe - teacher).abs().mean(), probe)[0].norm()
    return float((0.25 * a / b.clamp_min(1e-12)).clamp(0.01, 1.0))

def main():
    p = argparse.ArgumentParser()
    for name in ['initial-checkpoint', 'foundation-checkpoint', 'student-checkpoint', 'teacher-checkpoint', 'train-manifest', 'validation-manifest', 'output-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stage', choices=['repair', 'joint'], required=True)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--max-train-samples', type=int, default=0)
    p.add_argument('--max-validation-samples', type=int, default=0)
    p.add_argument('--validation-every', type=int, default=5)
    p.add_argument('--paper-variant', choices=['ordered_single'], default='ordered_single')
    p.add_argument('--minimum-free-gb', type=float, default=2.0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Host CUDA required; no CPU fallback')
    if args.epochs < 1 or args.validation_every < 1:
        raise ValueError('Invalid epoch schedule')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.resolve() in [args.teacher_checkpoint.resolve().parent, args.student_checkpoint.resolve().parent]:
        raise ValueError('A separate output directory is required')
    device = torch.device('cuda')
    seed_all(args.seed)
    base, receiver, foundation, tau = load_pair(args, args.student_checkpoint, device, args.paper_variant)
    teacher_base, teacher, unused, teacher_tau = load_pair(args, args.teacher_checkpoint, device, 'full')
    del unused
    teacher_base.eval()
    teacher.eval()
    for module in (teacher_base, teacher):
        module.requires_grad_(False)
    optimizer = torch.optim.AdamW(parameter_groups(base, receiver, args.stage), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-07)
    train = CompactExactDataset(args.train_manifest, crop_size=256, random_crop=True, return_pose_components=True)
    order = scene_balanced_order(train.records, args.seed)
    if args.max_train_samples:
        order = order[:args.max_train_samples]
    train = Subset(train, order)
    if len(train) < args.batch_size:
        raise ValueError('Training set smaller than one full batch')
    val = CompactExactDataset(args.validation_manifest, crop_size=256, random_crop=False, return_pose_components=True)
    if args.max_validation_samples:
        val.records = val.records[:args.max_validation_samples]
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata['paper_variant'] = args.paper_variant
    latest = args.output_dir / 'latest.pt'
    history, start, best, weight = ([], 1, -math.inf, None)
    if args.stage == 'joint':
        source = torch.load(args.student_checkpoint, map_location='cpu', weights_only=False)
        weight = source['distillation_initial_weight']
        del source
    if latest.exists():
        if not args.resume:
            raise RuntimeError('Existing run requires --resume')
        state = torch.load(latest, map_location=device, weights_only=False)
        if state['args'] != metadata:
            raise ValueError('Resume configuration mismatch')
        base.load_state_dict(state['base_state_dict'], strict=True)
        receiver.load_state_dict(state['receiver_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start, history, best, weight = (state['epoch'] + 1, state['history'], state['best_score'], state['distillation_initial_weight'])
        del state

    def validate():
        return evaluate(base, receiver, foundation, val, device, args.workers, 2, tau, 'prediction_psnr')

    def payload(epoch):
        return dict(format='matched_final_ablation_v1', epoch=epoch, base_state_dict=base.state_dict(), receiver_state_dict=receiver.state_dict(), reliability_tau=tau, args=metadata, history=history, best_score=best, distillation_initial_weight=weight, protocol={'inputs': ['blur', 'gyro', 'intrinsics'], 'teacher_at_inference': False, 'geometry_supervision': 'unchanged phase4; ordered exposure only', 'paper_variant': args.paper_variant, 'teacher': str(args.teacher_checkpoint), 'batchnorm_statistics': 'frozen'})

    def save_best(epoch):
        guarded_save(payload(epoch), args.output_dir / 'best_diagnostic.pt', args.minimum_free_gb)
    if not history:
        result = validate()
        best = float(result['score'])
        history.append({'epoch': 0, 'validation': result})
        seed_all(args.seed)
        model_mode(base, receiver, args.stage)
        if weight is None:
            calibration = []
            for index, raw in enumerate(DataLoader(train, batch_size=args.batch_size, num_workers=0)):
                batch = {k: v.to(device).float() for k, v in raw.items() if torch.is_tensor(v)}
                with torch.no_grad():
                    output, _, _ = restore(base, receiver, batch, tau)
                    target, _, _ = restore(teacher_base, teacher, batch, teacher_tau)
                calibration.append(calibrate_weight(output, batch['sharp'], target))
                if index == 3:
                    break
            weight = sum(calibration) / len(calibration)
            atomic_json({'output_gradient_target_ratio': 0.25, 'weights': calibration, 'selected': weight, 'data': 'training crops only'}, args.output_dir / 'calibration.json')
        save_best(0)
    for epoch in range(start, args.epochs + 1):
        if shutil.disk_usage(args.output_dir).free < args.minimum_free_gb * 1024 ** 3:
            raise RuntimeError('Space below configured floor; preserve latest checkpoint')
        seed_all(args.seed + epoch)
        model_mode(base, receiver, args.stage)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True, generator=torch.Generator().manual_seed(args.seed + epoch))
        wd = distillation_weight(weight, args.stage, epoch)
        totals = dict(loss=0.0, restore=0.0, distill=0.0, geometry=0.0)
        for raw in tqdm(loader, desc=f'{args.stage}-{epoch}'):
            batch = {k: v.to(device).float() for k, v in raw.items() if torch.is_tensor(v)}
            image, residual, depth = restore(base, receiver, batch, tau)
            primary = restoration_loss(image, batch['sharp'])
            distill = primary.new_zeros(())
            if wd > 0:
                with torch.no_grad():
                    target, _, _ = restore(teacher_base, teacher, batch, teacher_tau)
                distill = (image - target).abs().mean()
            geometry = geometry_loss(residual, depth, batch) if args.stage == 'joint' else primary.new_zeros(())
            loss = primary + wd * distill + 0.01 * geometry
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for g in optimizer.param_groups for q in g['params']], 1.0, error_if_nonfinite=True)
            optimizer.step()
            for key, value in zip(totals, [loss, primary, distill, geometry]):
                totals[key] += float(value.detach())
        scheduler.step()
        result = validate() if epoch % args.validation_every == 0 or epoch == args.epochs else None
        history.append({'epoch': epoch, 'train': {k: v / len(loader) for k, v in totals.items()}, 'distillation_weight': wd, 'updates': len(loader), 'validation': result})
        improved = result is not None and float(result['score']) > best
        if improved:
            best = float(result['score'])
        state = payload(epoch)
        state.update(optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict())
        guarded_save(state, latest, args.minimum_free_gb)
        if improved:
            save_best(epoch)
        atomic_json({'epoch': epoch, 'best_score': best, 'last': history[-1]}, args.output_dir / 'progress.json')
    atomic_json({'status': 'completed', 'epochs': args.epochs, 'best_score': best, 'checkpoint': str(args.output_dir / 'best_diagnostic.pt')}, args.output_dir / 'completed.json')
if __name__ == '__main__':
    main()
