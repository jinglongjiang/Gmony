import torch
import numpy as np

def convert(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert(v) for v in obj)
    return obj

ckpt = torch.load('/workspace/nav_data/mamba/camrl/4090/ppo_91.8/rl_model_ep10000.pth', map_location='cpu', weights_only=False)
ckpt_clean = convert(ckpt)
torch.save(ckpt_clean, '/workspace/nav_data/mamba/camrl/4090/ppo_91.8/rl_model_ep10000_compat.pth')
print('Done!')
