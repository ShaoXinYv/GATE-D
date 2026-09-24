from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.third_party.iaai.modules import ConvModule, LightHamHead, MSCAN
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    return (image - mean) / std

class LowLevelEncoder(nn.Module):
    """The low-level branch used by the official Image-as-an-IMU model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvModule(3, 64, kernel_size=3, padding=1)
        self.conv2 = ConvModule(64, 64, kernel_size=3, padding=1)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature = self.conv2(self.conv1(data['image']))
        return {'features': feature}

class IAAIResidualBackbone(nn.Module):
    """Stage-G-compatible IAAI visual geometry and gyro residual backbone."""

    def __init__(self, flow_scale: float=20.0, imagenet_norm: bool=True) -> None:
        super().__init__()
        self.flow_scale = float(flow_scale)
        self.imagenet_norm = bool(imagenet_norm)
        self.backbone = MSCAN()
        self.low_level = LowLevelEncoder()
        self.motion_decoder = LightHamHead()
        self.depth_decoder = LightHamHead()
        self.gyro_embed = nn.Sequential(nn.Conv2d(16, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 64, 3, 1, 1), nn.GELU())
        self.motion_fuse = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 64, 3, 1, 1), nn.GELU())
        self.res_endpoint = nn.Conv2d(64, 4, 1)
        self.res_traj = nn.Conv2d(64, 16, 1)
        self.inv_depth = nn.Conv2d(64, 1, 1)
        self.confidence = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.res_endpoint.weight)
        nn.init.zeros_(self.res_endpoint.bias)
        nn.init.zeros_(self.res_traj.weight)
        nn.init.zeros_(self.res_traj.bias)

    def forward(self, blur: torch.Tensor, gyro_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        height, width = blur.shape[-2:]
        pad_h = (32 - height % 32) % 32
        pad_w = (32 - width % 32) % 32
        blur_input = F.pad(blur, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else blur
        gyro_input = F.pad(gyro_flow, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else gyro_flow
        image = imagenet_normalize(blur_input) if self.imagenet_norm else blur_input
        data = {'image': image}
        high_level = self.backbone(data)['features']
        features = {'hl': high_level, 'll': self.low_level(data)['features']}
        image_motion = self.motion_decoder(features)
        depth_feature = self.depth_decoder(features)
        gyro_feature = self.gyro_embed(gyro_input / self.flow_scale)
        if gyro_feature.shape[-2:] != image_motion.shape[-2:]:
            gyro_feature = F.interpolate(gyro_feature, size=image_motion.shape[-2:], mode='bilinear', align_corners=False)
        motion_feature = self.motion_fuse(torch.cat((image_motion, gyro_feature), dim=1))
        endpoint = self.res_endpoint(motion_feature)[..., :height, :width]
        dense_delta = self.res_traj(motion_feature)[..., :height, :width]
        inverse_depth = F.softplus(self.inv_depth(depth_feature)[..., :height, :width]) + 1e-06
        confidence = torch.sigmoid(self.confidence(motion_feature)[..., :height, :width])
        return (endpoint, dense_delta, inverse_depth, confidence, {'image_motion_feat': image_motion[..., :height, :width], 'gyro_feat': gyro_feature[..., :height, :width], 'motion_feat': motion_feature[..., :height, :width], 'depth_feat': depth_feature[..., :height, :width], 'backbone_features': high_level})
