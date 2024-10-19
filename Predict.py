import torch
import platform
from utils.interface import PredictActivate


if __name__ == '__main__':
    params = {
        'batch_size': 1536,
        'info_path': './cache/Fruits-info.json',
        'predict_images_dir': './datasets/Unknown-Fruits',
        'weight': './cache/log/Fruits-train_loss(0.12103)valid_loss(0.04023).pt',
        'device': torch.device(
            ('mps' if platform.system() == 'Darwin' else 'cuda')
            if torch.cuda.is_available() else 'cpu'
        )
    }
    PredictActivate(params)