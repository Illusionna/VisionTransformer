import torch
import platform
from utils.interface import TestActivate


if __name__ == '__main__':
    params = {
        'batch_size': 128,
        'train_test_valid_set_map': './cache/Fruits-map.json',
        'weight': './cache/log/Fruits-train_loss(0.12103)valid_loss(0.04023).pt',
        'device': torch.device(
            ('mps' if platform.system() == 'Darwin' else 'cuda')
            if torch.cuda.is_available() else 'cpu'
        )
    }
    TestActivate(params)