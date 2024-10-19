import os
import time
import torch
import collections
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.tool import CustomLoader, MultistepCompose, ReadJSON, WriteJSON, ProgressBar


def TrainActivate(params: dict) -> None:
    """>>> TrainActivate(parameters)

    Args:
        params (dict): _description_
    """
    def __Loader__() -> tuple[DataLoader, DataLoader]:
        data = ReadJSON(path = params['train_test_valid_set_map'])
        image_paths, labels = zip(*data['train_set'])      
        TrainLoader = DataLoader(
            dataset = CustomLoader(
                set_list = image_paths,
                label_list = labels,
                __Func__ = MultistepCompose()[0]
            ),
            batch_size = params['batch_size'],
            shuffle = True
        )
        image_paths, labels = zip(*data['valid_set'])
        ValidLoader = DataLoader(
            dataset = CustomLoader(
                set_list = image_paths,
                label_list = labels,
                __Func__ = MultistepCompose()[-1]
            ),
            batch_size = params['batch_size'],
            shuffle = True
        )
        return TrainLoader, ValidLoader

    def __Record__() -> None:
        record['epoch'].append(iteration + 1)
        record['train_accuracy'].append(float(train_accuracy))
        record['train_loss'].append(float(train_loss))
        record['valid_accuracy'].append(float(valid_accuracy))
        record['valid_loss'].append(float(valid_loss))

    start = time.time()
    dir = os.path.join(os.path.dirname(params['train_test_valid_set_map']), 'log')
    os.makedirs(dir, exist_ok = True)
    TrainLoader, ValidLoader = __Loader__()
    record = collections.defaultdict(list)
    for iteration in range(0, params['epoch'], 1):
        stamp = time.time()
        train_loss = train_accuracy = 0
        bar = ProgressBar(
            iterable = TrainLoader, size = 25,
            prefix = 'Training', suffix = f"  Epoch: {iteration + 1} / {params['epoch']}  "
        )
        for X, Y in bar:
            X = X.to(params['device'])
            Y = Y.to(params['device'])
            output = params['model'](X)
            Loss = params['criterion'](output, Y)
            params['optimizer'].zero_grad()
            Loss.backward()
            params['optimizer'].step()
            N = len(TrainLoader)
            train_accuracy += ((output.argmax(dim = 1)) == Y).float().mean() / N
            train_loss = train_loss + Loss / N
        with torch.no_grad():
            valid_accuracy = valid_loss = 0
            for X, Y in ValidLoader:
                X = X.to(params['device'])
                Y = Y.to(params['device'])
                output = params['model'](X)
                Loss = params['criterion'](output, Y)
                N = len(ValidLoader)
                valid_accuracy += ((output.argmax(dim = 1)) == Y).float().mean() / N
                valid_loss = valid_loss + Loss / N
        save_path = os.path.normpath(
            os.path.join(
                dir,
                f"{os.path.basename(params['train_test_valid_set_map'])[:os.path.basename(params['train_test_valid_set_map']).rfind('-')]}-epoch({iteration+1})train_loss({train_loss:.5f})valid_loss({valid_loss:.5f}).pt"
            )
        )
        torch.save(
            obj = params['model'],
            f = save_path
        )
        print('Loss:')
        print(f'\ttrain: {train_loss:.7f}'); print(f'\tvalid: {valid_loss:.7f}')
        print('Accuracy:')
        print(f'\ttrain: {100 * train_accuracy:.3f}%')
        print(f'\tvalid: {100 * valid_accuracy:.3f}%')
        print(f'Weight is saved to: "{save_path}"')
        __Record__()
        now = time.strftime('%H:%M:%S', time.gmtime(time.time() - stamp))
        print(f'Cost: {now}\n')
    train_result_path = os.path.normpath(os.path.join(os.path.dirname(params['train_test_valid_set_map']), f"{os.path.basename(params['train_test_valid_set_map'])[:os.path.basename(params['train_test_valid_set_map']).rfind('-')]}-train.json"))
    WriteJSON(train_result_path, record, indent = 4)
    end = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print(f'<<---------------- Total Cost: {end} ---------------->>')
    print(''.join(f'\033[38;5;{idx}m#' for idx in range(60, 120, 1)), end = '\033[0m\n')


def TestActivate(params: dict) -> None:
    """>>> TestActivate(parameters)

    Args:
        params (dict): _description_
    """
    def __Loader__() -> DataLoader:
        data = ReadJSON(path = params['train_test_valid_set_map'])
        image_paths, labels = zip(*data['test_set'])
        return DataLoader(
            dataset = CustomLoader(
                set_list = image_paths,
                label_list = labels,
                __Func__ = MultistepCompose()[1]
            ),
            batch_size = params['batch_size'],
            shuffle = True
        )

    start = time.time()
    TestLoader = __Loader__()
    model = torch.load(f = params['weight'], weights_only = False).to(params['device'])
    model.eval()
    test_accuracy = 0
    for X, Y in ProgressBar(TestLoader, size = 25, prefix = 'Testing', suffix = '\t'):
        X = X.to(params['device'])
        Y = Y.to(params['device'])
        output = model(X)
        test_accuracy += ((output.argmax(dim=1)) == Y).float().mean() / len(TestLoader)
    print(''.join(f'\033[38;5;{idx}m#' for idx in range(196, 256, 1)), end = '\033[0m\n')
    print(f'Test Accuracy: {100 * test_accuracy:.3f}%')
    path = os.path.join(
        os.path.dirname(params['train_test_valid_set_map']),
        f"{os.path.basename(params['train_test_valid_set_map'])[:os.path.basename(params['train_test_valid_set_map']).rfind('-')]}-test.json"
    )
    WriteJSON(path = path, data = {'test_accuracy': float(test_accuracy)}, indent = None)
    print(f'Test accuracy is saved to: "{path}"')
    end = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print(f'<<---------------- Total Cost: {end} ---------------->>')
    print(''.join(f'\033[38;5;{idx}m#' for idx in range(136, 196, 1)), end = '\033[0m\n')


def PredictActivate(params: dict) -> None:
    """>>> PredictActivate(parameters)

    Args:
        params (dict): _description_
    """
    def __PredictCompose__() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]
        )

    def __Func__() -> dict:
        result.update(
            dict(
                zip(
                    stack,
                    iter(category_unmap[str(i)] for i in predict_tag.tolist())
                )
            )
        )

    def __PrintResult__() -> None:
        print('Result view:')
        if len(result) <= 4:
            for key, value in result.items():
                print(f'  "{os.path.normpath(key)}" ---- {value}')
        else:
            ABBREVIATION = 0
            for key, value in result.items():
                if ABBREVIATION < 3:
                    print(f'  "{os.path.normpath(key)}" ---- {value}')
                else:
                    print(f'  ............ ---- ............')
                    break
                ABBREVIATION = -~ABBREVIATION

    start = time.time()
    category_unmap = ReadJSON(path = params['info_path'])['unmap']
    n = len(os.listdir(params['predict_images_dir']))
    predict_loader = list(
        list(
            os.path.join(params['predict_images_dir'], filename)
            for filename in os.listdir(params['predict_images_dir'])
        )[idx:(idx + params['batch_size'])]
        for idx in range(0, n, params['batch_size'])
    )
    model = torch.load(f = params['weight'], weights_only = False).to(params['device'])
    model.eval()
    result = dict()
    for stack in ProgressBar(predict_loader, size = 25, prefix = 'Predicting', suffix = '\t'):
        with torch.no_grad():
            output = model(
                torch.stack(
                    list(
                        __PredictCompose__()(Image.open(image)) for image in stack
                    )
                ).to(params['device'])
            )
            predict_tag = output.argmax(dim=1)
            __Func__()
    print(''.join(f'\033[38;5;{idx}m#' for idx in range(140, 200, 1)), end = '\033[0m\n')
    __PrintResult__()
    path = os.path.join(
        os.path.dirname(params['info_path']),
        f"{os.path.basename(params['info_path'])[:os.path.basename(params['info_path']).rfind('-')]}-predict.json"
    )
    WriteJSON(path = path, data = result, indent = 4)
    print(''.join(f'\033[38;5;{idx}m#' for idx in range(80, 140, 1)), end = '\033[0m\n')
    print(f'Prediction result is saved to:\n\t"{os.path.normpath(path)}"')
    end = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print(f'<<---------------- Total Cost: {end} ---------------->>')
    print(''.join(f'\033[38;5;{idx}m#' for idx in range(20, 80, 1)), end = '\033[0m\n')