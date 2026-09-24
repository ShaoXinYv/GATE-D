from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from models.geometry import project_pose_trajectory
from models.gyro import eight_exposure_indices, integrate_dense_gyro_segment, rotations_to_flow

def _image_tensor(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()

def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

def gyro_flow_from_record(record: dict[str, object], root: Path, intrinsics: torch.Tensor, image_size: tuple[int, int], gyro_scale_multiplier: float=1.0, gyro_sources: dict[Path, torch.Tensor] | None=None) -> torch.Tensor:
    """Build gyro flow without reading depth or exact exposure poses."""
    gyro_source = record.get('gyro_source')
    if gyro_source:
        source_path = _resolve(root, str(gyro_source)).resolve()
        if gyro_sources is not None and source_path in gyro_sources:
            source = gyro_sources[source_path]
        else:
            source = torch.from_numpy(np.loadtxt(source_path, dtype=np.float64))
        start = int(record['gyro_start'])
        intervals = int(record.get('gyro_num', 11))
        segment = source[start:start + intervals + 1]
        if len(segment) != intervals + 1:
            raise ValueError(f'gyro slice [{start}:{start + intervals + 1}] exceeds {source_path}')
    else:
        pose_pack = torch.load(_resolve(root, str(record['pose'])), map_location='cpu')
        segment = pose_pack['gyro_segment'].double()
    dense_rotation = integrate_dense_gyro_segment(segment.double(), interpolation_factor=int(record.get('gyro_interpolation_factor', 8)), gyro_scale=float(record.get('gyro_scale', 1.0)) * gyro_scale_multiplier).float()
    indices = eight_exposure_indices(len(dense_rotation))
    selected_rotation = dense_rotation[indices]
    rotation_with_center = torch.cat((selected_rotation[:4], torch.eye(3)[None], selected_rotation[4:]), dim=0)[None]
    return rotations_to_flow(rotation_with_center, intrinsics[None], image_size)[0]

class CompactExactDataset(Dataset):
    """JSONL compact dataset.

    Required per record:
      name, blur, sharp, depth, pose, intrinsics=[fx,fy,cx,cy]
    The pose pack contains center_to_all_pose and the raw gyro_segment.
    """

    def __init__(self, manifest: str | Path, *, crop_size: int | None=None, random_crop: bool=False, return_pose_components: bool=False, include_sharp: bool=True) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.resolve().parent
        self.crop_size = crop_size
        self.random_crop = bool(random_crop)
        self.return_pose_components = bool(return_pose_components)
        self.include_sharp = bool(include_sharp)
        if crop_size is not None and crop_size <= 0:
            raise ValueError('crop_size must be positive')
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def _crop(self, blur: torch.Tensor, sharp: torch.Tensor | None, depth: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if self.crop_size is None:
            return (blur, sharp, depth, intrinsics)
        height, width = blur.shape[-2:]
        size = self.crop_size
        if size > height or size > width:
            raise ValueError(f'crop_size={size} exceeds sample size {(height, width)}')
        if self.random_crop:
            top = int(torch.randint(height - size + 1, ()).item())
            left = int(torch.randint(width - size + 1, ()).item())
        else:
            top = (height - size) // 2
            left = (width - size) // 2
        region = (slice(top, top + size), slice(left, left + size))
        cropped_intrinsics = intrinsics.clone()
        cropped_intrinsics[2] -= left
        cropped_intrinsics[3] -= top
        return (blur[:, region[0], region[1]], None if sharp is None else sharp[:, region[0], region[1]], depth[:, region[0], region[1]], cropped_intrinsics)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        blur = _image_tensor(_resolve(self.root, record['blur']))
        sharp = _image_tensor(_resolve(self.root, record['sharp'])) if self.include_sharp else None
        depth_array = np.load(_resolve(self.root, record['depth'])).astype(np.float32)
        depth = torch.from_numpy(depth_array).unsqueeze(0)
        pose_pack = torch.load(_resolve(self.root, record['pose']), map_location='cpu')
        intrinsics = torch.tensor(record['intrinsics'], dtype=torch.float32)
        blur, sharp, depth, intrinsics = self._crop(blur, sharp, depth, intrinsics)
        all_poses = pose_pack['center_to_all_pose'].float()
        pose_indices = eight_exposure_indices(len(all_poses))
        selected_poses = all_poses[pose_indices]
        full_flow, full_valid = project_pose_trajectory(depth[None], selected_poses[None], intrinsics[None])
        dense_rotation = integrate_dense_gyro_segment(pose_pack['gyro_segment'].double(), interpolation_factor=int(record.get('gyro_interpolation_factor', 8)), gyro_scale=float(record.get('gyro_scale', 1.0))).float()
        gyro_indices = eight_exposure_indices(len(dense_rotation))
        selected_rotation = dense_rotation[gyro_indices]
        rotation_with_center = torch.cat((selected_rotation[:4], torch.eye(3)[None], selected_rotation[4:]), dim=0)[None]
        gyro_flow = rotations_to_flow(rotation_with_center, intrinsics[None], depth.shape[-2:])[0]
        valid = full_valid.all(1, keepdim=True).float()[0]
        sample: dict[str, object] = {'name': str(record['name']), 'blur': blur, 'depth': depth, 'intrinsics': intrinsics, 'gyro_flow': gyro_flow, 'full_trajectory': full_flow[0], 'residual_trajectory': full_flow[0] - gyro_flow, 'valid': valid}
        if sharp is not None:
            sample['sharp'] = sharp
        if self.return_pose_components:
            sample['full_poses'] = selected_poses
            sample['gyro_rotations'] = selected_rotation
        return sample

class CompactBlurGyroDataset(Dataset):
    """Deployment-input dataset with an optional sharp supervision target.

    This path never reads depth or exact exposure poses. The pose pack is used
    only as the compact container for the measured ``gyro_segment``.
    """

    def __init__(self, manifest: str | Path, *, crop_size: int | None=None, random_crop: bool=False) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.resolve().parent
        self.crop_size = crop_size
        self.random_crop = bool(random_crop)
        if crop_size is not None and crop_size <= 0:
            raise ValueError('crop_size must be positive')
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        source_paths = {_resolve(self.root, str(record['gyro_source'])).resolve() for record in self.records if record.get('gyro_source')}
        self.gyro_sources = {path: torch.from_numpy(np.loadtxt(path, dtype=np.float64)) for path in source_paths}

    def __len__(self) -> int:
        return len(self.records)

    def _crop(self, blur: torch.Tensor, sharp: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.crop_size is None:
            return (blur, sharp, intrinsics)
        height, width = blur.shape[-2:]
        size = self.crop_size
        if size > height or size > width:
            raise ValueError(f'crop_size={size} exceeds sample size {(height, width)}')
        if self.random_crop:
            top = int(torch.randint(height - size + 1, ()).item())
            left = int(torch.randint(width - size + 1, ()).item())
        else:
            top = (height - size) // 2
            left = (width - size) // 2
        cropped_intrinsics = intrinsics.clone()
        cropped_intrinsics[2] -= left
        cropped_intrinsics[3] -= top
        return (blur[:, top:top + size, left:left + size], sharp[:, top:top + size, left:left + size], cropped_intrinsics)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        blur = _image_tensor(_resolve(self.root, record['blur']))
        sharp = _image_tensor(_resolve(self.root, record['sharp']))
        intrinsics = torch.tensor(record['intrinsics'], dtype=torch.float32)
        blur, sharp, intrinsics = self._crop(blur, sharp, intrinsics)
        gyro_flow = gyro_flow_from_record(record, self.root, intrinsics, blur.shape[-2:], gyro_sources=self.gyro_sources)
        return {'name': str(record['name']), 'blur': blur, 'sharp': sharp, 'intrinsics': intrinsics, 'gyro_flow': gyro_flow}
