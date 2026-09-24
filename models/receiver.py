"""NAFNet-style restoration with GyroDeblurNet-inspired gyro conditioning."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from models.geometry import resize_flow_trajectory

class LayerNorm2d(nn.Module):

    def __init__(self, channels: int, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).square().mean(1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[:, None, None] + self.bias[:, None, None]

class SimpleGate(nn.Module):

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second

class NAFBlock(nn.Module):

    def __init__(self, channels: int, expand: int=2) -> None:
        super().__init__()
        hidden = channels * expand
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv3 = nn.Conv2d(hidden // 2, channels, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden // 2, hidden // 2, 1))
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(channels, hidden, 1)
        self.conv5 = nn.Conv2d(hidden // 2, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature = self.sg(self.conv2(self.conv1(self.norm1(value))))
        feature = self.conv3(feature * self.sca(feature))
        first = value + self.dropout1(feature) * self.beta
        feature = self.conv5(self.sg(self.conv4(self.norm2(first))))
        return first + self.dropout2(feature) * self.gamma

class SpatialAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 1, 7, padding='same', bias=False), nn.BatchNorm2d(1, eps=1e-05, momentum=0.01))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((value.max(1, keepdim=True).values, value.mean(1, keepdim=True)), dim=1)
        return value * torch.sigmoid(self.conv(pooled))

class IdentityGyro(nn.Module):

    def forward(self, _blur: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        return gyro

class IdentityTwo(nn.Module):

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (first, second)

class GyroRefinementBlock(nn.Module):

    def __init__(self, c_gyro: int, c_blur: int) -> None:
        super().__init__()
        self.conv_ca_weight = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_gyro + c_blur, c_gyro, 3, padding=1))
        self.conv_gyro = nn.Conv2d(c_gyro, c_gyro, 3, padding=1)
        self.conv_down = nn.Conv2d(c_gyro, 2 * c_gyro, 3, stride=2, padding=1)

    def forward(self, blur: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        weight = self.conv_ca_weight(torch.cat((blur, gyro), dim=1))
        return self.conv_down(F.relu(self.conv_gyro(gyro * weight)))

class OutputLocalGate(nn.Module):

    def __init__(self, channels: int, image_channels: int=3) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.body = nn.Sequential(nn.Conv2d(channels + image_channels, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, 2.0)

    def forward(self, feature: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.body(torch.cat((feature, image), dim=1)))

class GyroConditionedDeformableConv2d(nn.Module):
    """Gyro-conditioned alignment without a dedicated translation predictor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.offset = nn.Conv2d(2 * channels, 18, 3, padding=1)
        self.modulator = nn.Conv2d(2 * channels, 9, 3, padding=1)
        self.regular = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)

    def forward(self, feature: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        condition = torch.cat((feature, gyro), dim=1)
        return deform_conv2d(feature, self.offset(condition), self.regular.weight, self.regular.bias, padding=1, mask=2.0 * torch.sigmoid(self.modulator(condition)))

class CleanGyroDeblurringBlock(nn.Module):
    """Gyro-conditioned block without dedicated depth/translation heads."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alignment = GyroConditionedDeformableConv2d(channels)
        self.appearance = NAFBlock(channels)
        self.spatial_attention = SpatialAttention()
        self.fusion = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, feature: torch.Tensor, gyro: torch.Tensor, return_aux: bool=False):
        aligned = self.alignment(feature, gyro)
        appearance = self.appearance(self.spatial_attention(aligned))
        output = self.fusion(torch.cat((aligned, appearance), dim=1))
        auxiliary = {'dedicated_translation_head_disabled': torch.ones((), device=feature.device, dtype=feature.dtype)}
        return (output, gyro, auxiliary) if return_aux else (output, gyro)

class ImageGeometryHead(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels, 32)
        self.shared = nn.Sequential(LayerNorm2d(channels), nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(), nn.Conv2d(hidden, hidden, 1), nn.GELU())
        self.depth = nn.Conv2d(hidden, 1, 1)
        self.flow = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.depth.weight)
        nn.init.zeros_(self.depth.bias)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(feature)
        return (F.softplus(self.depth(shared)) + 1e-06, self.flow(shared))

def warp_feature(feature: torch.Tensor, flow: torch.Tensor, sign: float) -> torch.Tensor:
    batch, _, height, width = feature.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=feature.device, dtype=feature.dtype), torch.arange(width, device=feature.device, dtype=feature.dtype), indexing='ij')
    grid_x = xx[None] + sign * flow[:, 0]
    grid_y = yy[None] + sign * flow[:, 1]
    grid = torch.stack((2.0 * grid_x / max(width - 1, 1) - 1.0, 2.0 * grid_y / max(height - 1, 1) - 1.0), dim=-1)
    return F.grid_sample(feature, grid, mode='bilinear', padding_mode='border', align_corners=True)

class TrajectoryPhysicalInverseBlock(nn.Module):
    """One identity-safe forward/approximate-adjoint feature correction."""

    def __init__(self, channels: int, flow_scale: float=20.0, gate_bias: float=4.0) -> None:
        super().__init__()
        self.flow_scale = float(flow_scale)
        self.sign = -1.0
        hidden = max(channels // 2, 64)
        self.pre = nn.Sequential(LayerNorm2d(3 * channels + 2), nn.Conv2d(3 * channels + 2, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU())
        self.delta = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    @staticmethod
    def _with_center(flows: list[torch.Tensor]) -> list[torch.Tensor]:
        return flows[:4] + [torch.zeros_like(flows[0])] + flows[4:]

    def _flows(self, geometry: torch.Tensor, size: tuple[int, int]) -> list[torch.Tensor]:
        trajectory = resize_flow_trajectory(geometry[:, :16] * self.flow_scale, size)
        return [trajectory[:, 2 * index:2 * index + 2] for index in range(8)]

    def forward_operator(self, feature: torch.Tensor, flows: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([warp_feature(feature, flow, self.sign) for flow in self._with_center(flows)]).mean(0)

    def adjoint_operator(self, residual: torch.Tensor, flows: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([warp_feature(residual, flow, -self.sign) for flow in self._with_center(flows)]).mean(0)

    @staticmethod
    def normalized_atom_weights(flow_atoms: torch.Tensor, atom_weights: torch.Tensor) -> torch.Tensor:
        if flow_atoms.ndim != 5 or flow_atoms.shape[2] != 2:
            raise ValueError('flow atoms must have shape [B,S,2,H,W]')
        if atom_weights.shape != flow_atoms.shape[:2]:
            raise ValueError('atom weights must have shape [B,S]')
        if not bool(torch.isfinite(flow_atoms).all()) or not bool(torch.isfinite(atom_weights).all()):
            raise ValueError('flow atoms and weights must be finite')
        if bool((atom_weights < 0).any()):
            raise ValueError('atom weights must be non-negative')
        mass = atom_weights.sum(1, keepdim=True)
        if bool((mass <= 0).any()):
            raise ValueError('atom weights must have positive mass')
        return atom_weights / mass

    def resize_flow_atoms(self, flow_atoms: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        batch, atoms, channels, height, width = flow_atoms.shape
        flow_atoms = flow_atoms / self.flow_scale * self.flow_scale
        return resize_flow_trajectory(flow_atoms.reshape(batch * atoms, channels, height, width), size).reshape(batch, atoms, channels, *size)

    def weighted_forward_operator(self, feature: torch.Tensor, flow_atoms: torch.Tensor, atom_weights: torch.Tensor) -> torch.Tensor:
        warped = [warp_feature(feature, flow_atoms[:, index], self.sign) for index in range(flow_atoms.shape[1])]
        equal_weight = atom_weights.new_full(atom_weights.shape, 1.0 / atom_weights.shape[1])
        if torch.equal(atom_weights, equal_weight):
            return torch.stack(warped).mean(0)
        values = torch.stack(warped, dim=1)
        return (values * atom_weights[:, :, None, None, None]).sum(1)

    def weighted_adjoint_operator(self, residual: torch.Tensor, flow_atoms: torch.Tensor, atom_weights: torch.Tensor) -> torch.Tensor:
        warped = [warp_feature(residual, flow_atoms[:, index], -self.sign) for index in range(flow_atoms.shape[1])]
        equal_weight = atom_weights.new_full(atom_weights.shape, 1.0 / atom_weights.shape[1])
        if torch.equal(atom_weights, equal_weight):
            return torch.stack(warped).mean(0)
        values = torch.stack(warped, dim=1)
        return (values * atom_weights[:, :, None, None, None]).sum(1)

    def forward(self, blur_feature, estimate, geometry, *, operator='all_adjoint', flow_atoms=None, atom_weights=None):
        if operator != 'all_adjoint':
            raise ValueError('Only the final exposure/backprojection operator is supported')
        if (flow_atoms is None) != (atom_weights is None):
            raise ValueError('Flow atoms and weights must be provided together')
        flows = self._flows(geometry, estimate.shape[-2:])
        if flow_atoms is None:
            forward = self.forward_operator(estimate, flows)
            residual = blur_feature - forward
            backprojection = self.adjoint_operator(residual, flows)
        else:
            weights = self.normalized_atom_weights(flow_atoms, atom_weights)
            atoms = self.resize_flow_atoms(flow_atoms, estimate.shape[-2:])
            forward = self.weighted_forward_operator(estimate, atoms, weights)
            residual = blur_feature - forward
            backprojection = self.weighted_adjoint_operator(residual, atoms, weights)
        auxiliary = F.interpolate(geometry[:, 16:18], size=estimate.shape[-2:], mode='bilinear', align_corners=False)
        context = self.pre(torch.cat((estimate, residual, backprojection, auxiliary), dim=1))
        gate = torch.sigmoid(self.gate(context))
        output = estimate + gate * self.delta(context)
        return (output, {'phys_residual': residual, 'phys_backproj': backprojection, 'phys_gate': gate, 'uncertainty_scale': None, 'physical_hypothesis_count': flow_atoms.shape[1] if flow_atoms is not None else 1})

@dataclass
class ReceiverOutput:
    image: torch.Tensor
    auxiliary: dict[str, object]

class MultiScalePhysicalReceiver(nn.Module):
    """Checkpoint-compatible all-scale, one-step physical receiver."""

    def __init__(self, width: int=32, encoder_blocks: tuple[int, ...]=(2, 2, 2), middle_blocks: int=16, decoder_blocks: tuple[int, ...]=(1, 1, 1), flow_scale: float=20.0, gate_bias: float=4.0, physics_mode: str='none', motion_block_mode: str='clean_gyro', output_gate_mode: str='fixed_one') -> None:
        super().__init__()
        if physics_mode not in {'none'}:
            raise ValueError(f'unknown physics_mode={physics_mode}')
        self.physics_mode = physics_mode
        if output_gate_mode not in {'fixed_one'}:
            raise ValueError(f'unknown output_gate_mode={output_gate_mode}')
        self.output_gate_mode = output_gate_mode
        if motion_block_mode not in {'clean_gyro'}:
            raise ValueError(f'unknown motion_block_mode={motion_block_mode}')
        self.motion_block_mode = motion_block_mode
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        gyro_channels = width * 2
        self.intro_gyro = nn.Conv2d(16, gyro_channels, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        # These registered heads retain the trained checkpoint's parameter keys.
        self.output_gate = OutputLocalGate(width, 3)
        if output_gate_mode == 'fixed_one':
            for parameter in self.output_gate.parameters():
                parameter.requires_grad_(False)
        self.geometry = ImageGeometryHead(width)
        self.encoders = nn.ModuleList()
        self.gyro_refine_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.gyro_deblurring_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for block_count in encoder_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)]))
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, 2))
            channels *= 2
        self.phys_inverse_bottleneck = TrajectoryPhysicalInverseBlock(channels, flow_scale, gate_bias)
        self.gyro_refine_blks.extend((IdentityGyro(), GyroRefinementBlock(gyro_channels, width * 2), GyroRefinementBlock(gyro_channels * 2, width * 4)))
        for _ in range(4):
            self.middle_blks.append(nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blocks // 4)]))
        motion_blocks = {'clean_gyro': CleanGyroDeblurringBlock}
        motion_block = motion_blocks[motion_block_mode]
        for _ in range(3):
            self.gyro_deblurring_blks.append(motion_block(channels))
        self.gyro_deblurring_blks.append(IdentityTwo())
        decoder_channels = []
        for block_count in decoder_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, 2 * channels, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            decoder_channels.append(channels)
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)]))
        self.phys_inverse_decoders = nn.ModuleList([TrajectoryPhysicalInverseBlock(value, flow_scale, gate_bias) if index < 2 else nn.Identity() for index, value in enumerate(decoder_channels)])
        self.residual_phys_inverse_bottleneck: nn.Module = nn.Identity()
        self.residual_phys_inverse_decoders = nn.ModuleList([nn.Identity() for _ in decoder_channels])
        self.padder_size = 2 ** len(self.encoders)

    def physical_blocks(self) -> list[TrajectoryPhysicalInverseBlock]:
        return [self.phys_inverse_bottleneck, *[block for block in self.phys_inverse_decoders if isinstance(block, TrajectoryPhysicalInverseBlock)]]

    def enable_residual_physical_adapters(self, gate_bias: float=4.0) -> None:
        """Attach multiscale adapters whose response is zero at zero translation."""
        if isinstance(self.residual_phys_inverse_bottleneck, TrajectoryPhysicalInverseBlock):
            return
        device = self.intro.weight.device
        dtype = self.intro.weight.dtype
        bottleneck_channels = self.phys_inverse_bottleneck.delta.out_channels
        self.residual_phys_inverse_bottleneck = TrajectoryPhysicalInverseBlock(bottleneck_channels, self.phys_inverse_bottleneck.flow_scale, gate_bias).to(device=device, dtype=dtype)
        adapters: list[nn.Module] = []
        for physical in self.phys_inverse_decoders:
            if isinstance(physical, TrajectoryPhysicalInverseBlock):
                adapters.append(TrajectoryPhysicalInverseBlock(physical.delta.out_channels, physical.flow_scale, gate_bias).to(device=device, dtype=dtype))
            else:
                adapters.append(nn.Identity())
        self.residual_phys_inverse_decoders = nn.ModuleList(adapters)

    def residual_physical_blocks(self) -> list[TrajectoryPhysicalInverseBlock]:
        return [block for block in (self.residual_phys_inverse_bottleneck, *self.residual_phys_inverse_decoders) if isinstance(block, TrajectoryPhysicalInverseBlock)]

    def _residual_physical_update(self, adapter: nn.Module, blur_feature: torch.Tensor, estimate: torch.Tensor, anchor_geometry: torch.Tensor, total_geometry: torch.Tensor | None, total_flow_atoms: torch.Tensor | None=None, total_atom_weights: torch.Tensor | None=None, anchor_flow_atoms: torch.Tensor | None=None, anchor_atom_weights: torch.Tensor | None=None) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if total_geometry is None or not isinstance(adapter, TrajectoryPhysicalInverseBlock):
            return (estimate, None)
        residual_operator = 'all_adjoint' if self.physics_mode == 'none' else self.physics_mode
        residual_operator = getattr(self, 'residual_operator_override', None) or residual_operator
        total_response, total_aux = adapter(blur_feature, estimate, total_geometry, operator=residual_operator, flow_atoms=total_flow_atoms, atom_weights=total_atom_weights)
        if getattr(self, 'residual_reference_mode', 'rotation') == 'identity':
            anchor_response = estimate
        else:
            anchor_response, _ = adapter(blur_feature, estimate, anchor_geometry, operator=residual_operator, flow_atoms=anchor_flow_atoms, atom_weights=anchor_atom_weights)
        reliability = F.interpolate(total_geometry[:, 17:18], size=estimate.shape[-2:], mode='bilinear', align_corners=False)
        delta = reliability * (total_response - anchor_response)
        total_aux = {**total_aux, 'residual_feature_delta': delta}
        return (estimate + delta, total_aux)

    def _pad(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(value, (0, pad_w, 0, pad_h))

    def forward(self, blur, gyro_flow, geometry, return_aux=False, *, residual_geometry=None, residual_flow_atoms=None, residual_atom_weights=None, anchor_flow_atoms=None, anchor_atom_weights=None):
        height, width = blur.shape[-2:]
        image = self._pad(blur)
        gyro_flow = resize_flow_trajectory(gyro_flow, (image.shape[-2] // 2, image.shape[-1] // 2))
        geometry = self._pad(geometry)
        if residual_geometry is not None:
            if residual_geometry.shape != (blur.shape[0], 18, height, width):
                raise ValueError('Residual geometry must match the image')
            if not self.residual_physical_blocks():
                raise ValueError('Enable residual adapters before use')
            residual_geometry = self._pad(residual_geometry)
        measures = (residual_flow_atoms, residual_atom_weights, anchor_flow_atoms, anchor_atom_weights)
        if any((value is not None for value in measures)):
            if not all((value is not None for value in measures)) or residual_geometry is None:
                raise ValueError('Complete residual and anchor measures are required')
            for atoms, weights in ((residual_flow_atoms, residual_atom_weights), (anchor_flow_atoms, anchor_atom_weights)):
                if atoms.ndim != 5 or atoms.shape[0] != blur.shape[0] or atoms.shape[2:] != (2, height, width):
                    raise ValueError('Flow atoms must have shape [B,S,2,H,W]')
                if weights.shape != atoms.shape[:2]:
                    raise ValueError('Weights must match flow atoms')

            def pad_atoms(atoms):
                batch, count, channels, h, w = atoms.shape
                return self._pad(atoms.reshape(batch * count, channels, h, w)).reshape(batch, count, channels, *image.shape[-2:])
            residual_flow_atoms = pad_atoms(residual_flow_atoms)
            anchor_flow_atoms = pad_atoms(anchor_flow_atoms)
        feature = self.intro(image)
        gyro = self.intro_gyro(gyro_flow)
        skips = []
        auxiliary = {'motion': [], 'phys_inverse': []}
        for index, (encoder, down, refine) in enumerate(zip(self.encoders, self.downs, self.gyro_refine_blks)):
            feature = encoder(feature)
            skips.append(feature)
            if index:
                gyro = refine(feature, gyro)
            feature = down(feature)
        feature, physics = self._residual_physical_update(self.residual_phys_inverse_bottleneck, feature, feature, geometry, residual_geometry, residual_flow_atoms, residual_atom_weights, anchor_flow_atoms, anchor_atom_weights)
        if physics is not None:
            auxiliary['phys_inverse'].append(physics)
        for middle, gyro_block in zip(self.middle_blks, self.gyro_deblurring_blks):
            feature = middle(feature)
            if return_aux and isinstance(gyro_block, CleanGyroDeblurringBlock):
                feature, gyro, motion = gyro_block(feature, gyro, True)
                auxiliary['motion'].append(motion)
            else:
                feature, gyro = gyro_block(feature, gyro)
        for decoder, up, skip, adapter in zip(self.decoders, self.ups, reversed(skips), self.residual_phys_inverse_decoders):
            feature = up(feature) + skip
            feature, physics = self._residual_physical_update(adapter, skip, feature, geometry, residual_geometry, residual_flow_atoms, residual_atom_weights, anchor_flow_atoms, anchor_atom_weights)
            if physics is not None:
                auxiliary['phys_inverse'].append(physics)
            feature = decoder(feature)
        strong = self.ending(feature) + image
        output_gate = torch.ones_like(image[:, :1])
        output = (strong * output_gate + image * (1.0 - output_gate))[..., :height, :width]
        if not return_aux:
            return output
        depth, flow = self.geometry(feature)
        auxiliary.update(depth=depth[..., :height, :width], flow=flow[..., :height, :width], output_gate=output_gate[..., :height, :width])
        return ReceiverOutput(output, auxiliary)
