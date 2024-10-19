import torch
import platform
from utils.net import Transformer
from utils.interface import TrainActivate


if __name__ == '__main__':
    DATASET_NAME = 'Fruits'
    device = torch.device(
        ('mps' if platform.system() == 'Darwin' else 'cuda')
        if torch.cuda.is_available() else 'cpu'
    )
    net = Transformer(info_path = f'./cache/{DATASET_NAME}-info.json').to(device)
    optimizer = torch.optim.Adam(params = net.parameters(), lr = 3e-5)
    parameters: dict = {
        'train_test_valid_set_map': f'./cache/{DATASET_NAME}-map.json',
        'epoch': 200,
        'batch_size': 256,
        'model': net,
        'device': device,
        'optimizer': optimizer,
        'scheduler': torch.optim.lr_scheduler.StepLR(
            optimizer = optimizer,
            step_size = 1,
            gamma = 0.7
        ),
        'criterion': torch.nn.CrossEntropyLoss()
    }
    TrainActivate(parameters)