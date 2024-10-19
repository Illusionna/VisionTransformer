import os
from utils.efficient import ViT
from utils.tool import ReadJSON
from utils.linformer import Linformer


class Transformer:
    def __new__(cls, info_path: str, channel_nums: int = 3) -> ViT:
        efficientTransformer = Linformer(
            dim = 128,
            seq_len = 7 * 7 + 1,  # 7x7 patches + 1 cls-token.
            depth = 12,
            k = 64,
            heads = 8
        )
        model = ViT(
            dim = 128,
            image_size = 224,
            patch_size = 32,
            num_classes = len(ReadJSON(info_path)['map']),
            transformer = efficientTransformer,
            channels = channel_nums,
        )
        path = os.path.normpath(os.path.join(os.path.dirname(info_path), f"{os.path.basename(info_path)[:os.path.basename(info_path).rfind('-')]}-net.txt"))
        with open(path, mode = 'w', encoding = 'utf-8') as f:
            f.write(str(model))
        print(''.join(f'\033[38;5;{idx}m#' for idx in range(120, 180, 1)), end = '\033[0m\n')
        print(f'ViT structure is saved to:\n\t"{path}"')
        print(''.join(f'\033[38;5;{idx}m#' for idx in range(0, 60, 1)), end = '\033[0m\n')
        return model