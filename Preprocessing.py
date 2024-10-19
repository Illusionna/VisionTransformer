import os
from utils.tool import RandomSeed, Process


if __name__ == '__main__':
    os.system('')
    print('\033[H\033[J', end = '')
    RandomSeed(42)
    Process(
        dir = './datasets/Fruits',
        ratio = [70, 20, 10],
        show = True,
        resolution = True
    )