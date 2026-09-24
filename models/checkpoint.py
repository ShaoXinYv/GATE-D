from types import SimpleNamespace
import torch
from models.motion import IAAIResidualBackbone

def load_bundle(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'gyro_factorized_deblur_v2':
        raise ValueError('Expected gyro_factorized_deblur_v2 initialization')
    config = payload['config']['motion']
    base = IAAIResidualBackbone(flow_scale=config['flow_scale'], imagenet_norm=config['imagenet_normalize'])
    prefix = 'motion.base.'
    state = {k[len(prefix):]: v for k, v in payload['model_state_dict'].items() if k.startswith(prefix)}
    base.load_state_dict(state, strict=True)
    return SimpleNamespace(motion=SimpleNamespace(base=base.to(device)))
