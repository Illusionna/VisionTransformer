import os
import json
import numpy
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from collections.abc import Generator
from torchvision.transforms import (
    Compose, Resize, ToTensor,
    RandomResizedCrop, RandomHorizontalFlip, CenterCrop
)


def WriteJSON(path: str, data: dict, indent: int | None) -> None:
    """>>> WriteJSON('./config.json', data, indent = 4)

    Args:
        path (str): _description_
        data (dict): _description_
        indent (int | None): _description_
    """
    with open(path, mode = 'w', encoding = 'utf-8') as f:
        f.write(
            json.dumps(
                obj = data,
                ensure_ascii = False, indent = indent
            )
        )


def ReadJSON(path: str) -> dict:
    """>>> print(ReadJSON('./config.json'))

    Args:
        path (str): _description_

    Returns:
        dict: _description_
    """
    with open(path, mode = 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    return data


def RandomSeed(seed: int) -> None:
    """>>> RandomSeed(42)

    Args:
        seed (int): _description_
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ProgressBar(iterable: object, size: int, prefix: str, suffix: str) -> Generator[int]:
    r""">>> L = ['1.png', '2.png', ..., 'n.png']
    >>> for i in ProgressBar(L, size = 25, prefix = 'await', suffix = '\t'):
    >>>     pass

    Args:
        iterable (object): _description_
        size (int): _description_
        prefix (str): _description_
        suffix (str): _description_

    Yields:
        Generator[int]: _description_
    """
    total = len(iterable)
    for idx, item in enumerate(iterable):
        idx = -~idx
        percent = f'{100 * (idx / total):.1f}'
        bar = '\033[32m█\033[0m' * int(size * idx // total) + '-' * (size - int(size * idx // total))
        print(f'\r{prefix}: |{bar}| {percent}%{suffix}', end = '')
        yield item
    print()


def PrintPreprocessing(show: bool, path: str) -> None:
    """>>> PrintPreprocessing(True, './cache/Bloodcells-info.json')

    Args:
        show (bool): _description_
        path (str): _description_
    """
    if show == True:
        data = ReadJSON(path)
        print(''.join(f'\033[38;5;{idx}m#' for idx in range(151, 211, 1)), end = '\033[0m\n')
        print('Detail Information:')
        print('\t1. Dataset map:')
        for key, value in data['map'].items():
            print(f'\t\t{key} -- {value}')
        print('\t2. Distribution:')
        for key, value in data['distribution'].items():
            print(f'\t\t{key} -- {value} image(s)')
        print(f"\t3. Total number of images: {data['total']}")
        print(f"\t4. Number of train_set: {data['train_size']} image(s)")
        print(f"\t5. Number of test_set: {data['test_size']} image(s)")
        print(f"\t6. Number of valid_set: {data['valid_size']} image(s)")
        print(f'''\t7. Information location: "{data['info_path']}"''')
        print(f'''\t8. Dataset list location: "{data['map_path']}"''')
        if (data['width'] != -1) and (data['height'] != -1):
            print(f"\t9. Average width of all images: {data['width']} px")
            print(f"\tX. Average height of all images: {data['height']} px")
        print(''.join(f'\033[38;5;{idx}m#' for idx in range(91, 151, 1)), end = '\033[0m\n')


def train_test_split(arrays: list, test_size: float) -> tuple[list, list]:
    """>>> train_test_split([1, 2, 3, 4, 5], 0.2)

    Args:
        arrays (list): _description_
        test_size (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        tuple[list, list]: _description_
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError('At least one array required as input')
    test = random.sample(arrays, int(n_arrays * test_size))
    return list(item for item in arrays if item not in test), test


def Process(dir: str, ratio: list, show: bool, resolution: bool = False) -> None:
    """>>> Process('./datasets/Fruits', 42, [70, 20, 10], True, True)

    Args:
        dir (str): _description_
        ratio (list): _description_
        show (bool): _description_
        resolution (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    def __Statistic__() -> tuple[dict, dict, dict, int]:
        """>>> __Statistic__()

        Returns:
            tuple[dict, dict, dict, int]: _description_
        """
        category_map: dict = {
            key: value
            for value, key in enumerate(
                path for path in iter(os.listdir(dir))
                if os.path.isdir(os.path.join(dir, path))
            )
        }
        distribution: dict = {
            key: len(os.listdir(os.path.join(dir, key)))
            for key in category_map.keys()
        }
        total = sum(i for i in distribution.values())
        return category_map, dict(map(reversed, category_map.items())), distribution, total

    def __DatasetMap__() -> tuple[int, int, int]:
        """>>> __DatasetMap__()

        Returns:
            tuple[int, int, int]: _description_
        """
        __Func__ = lambda train_test_valid_set: list(
            map(
                lambda x: [x, category_map[os.path.basename(os.path.dirname(x))]],
                iter(train_test_valid_set)
            )
        )
        train_set, test_set = train_test_split(
            dataset_path_list,
            test_size = ratio[1] / sum(ratio)
        )
        test_set, valid_set = train_test_split(
            test_set,
            test_size = ratio[-1] / (ratio[1] + ratio[-1])
        )
        WriteJSON(
            path = os.path.join('cache', f'{os.path.basename(dir)}-map.json'),
            data = {
                'train_set': __Func__(train_set),
                'test_set': __Func__(test_set),
                'valid_set': __Func__(valid_set)
            },
            indent = None
        )
        return len(train_set), len(test_set), len(valid_set)

    def __Resolution__() -> tuple[int, int]:
        """>>> __Resolution__()

        Returns:
            tuple[int, int]: _description_
        """
        if resolution == False:
            return -1, -1
        accumulate_width = accumulate_height = 0
        count = 0
        for i in ProgressBar(dataset_path_list, size = 25, prefix = 'Processing', suffix = '\t'):
            try:
                with Image.open(i) as image:
                    accumulate_width = accumulate_width + image.width
                    accumulate_height = accumulate_height + image.height
                    count = -~count
            except:
                print(f'\033[31m* Failed to open image: "{i}"\033[0m')
        if count == 0:
            return -1, -1
        return int(accumulate_width / count), int(accumulate_height / count)

    os.makedirs('cache', exist_ok = True)
    params = dict()
    category_map, reversed_category_map, distribution, total = __Statistic__()
    params.update({'map': category_map, 'unmap': reversed_category_map, 'distribution': distribution, 'total': total})
    dataset_path_list = list(
        os.path.normpath(os.path.join(dir, category, path))
        for category in category_map.keys()
        for path in iter(os.listdir(os.path.join(dir, category)))
    )
    train_size, test_size, valid_test = __DatasetMap__()
    width, height = __Resolution__()
    params.update({'train_size': train_size, 'test_size': test_size, 'valid_size': valid_test, 'width': width, 'height': height})
    path = os.path.join('cache', f'{os.path.basename(dir)}-info.json')
    WriteJSON(
        path = path,
        data = {
            **params,
            'info_path': path,
            'map_path': os.path.join('cache', f'{os.path.basename(dir)}-map.json')
        },
        indent = 4
    )
    PrintPreprocessing(show, path)


def MultistepCompose() -> tuple[Compose, Compose, Compose]:
    """>>> MultistepCompose()

    Returns:
        tuple[Compose, Compose, Compose]: _description_
    """
    TrainCompose = Compose(
        [
            Resize((224, 224)),
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor()
        ]
    )
    TestCompose = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor()
        ]
    )
    ValidCompose = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor()
        ]
    )
    return TrainCompose, TestCompose, ValidCompose


class CustomLoader(Dataset):
    def __init__(self, *args, set_list: list, label_list: list, __Func__: Compose, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_list = set_list
        self.label_list = label_list
        self.__Func__ = __Func__

    def __getitem__(self, idx: int) -> tuple[list[list], int]:
        image = Image.open(self.set_list[idx])
        return self.__Func__(image), self.label_list[idx]

    def __len__(self) -> int:
        return len(self.set_list)