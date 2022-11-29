import json
import os
import random
from collections import deque

import numpy as np
import torch
from torch import nn

def fix_seed(seed):
    """
    固定random np torch的随机数种子。

    :param seed:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def shingle(sequence, size):
    """
    生成时间窗口。

    Parameters
    ----------
    sequence : iterable
               时间序列
    size : int
           窗口长度
    """
    iterator = iter(sequence)
    init = (next(iterator) for _ in range(size))
    window = deque(init, maxlen=size)
    if len(window) < size:
        raise IndexError('Sequence smaller than window size')
    yield np.asarray(window)
    for elem in iterator:
        window.append(elem)
        yield np.asarray(window)


def save_model_to_json(model: nn.Module, path: str):
    """
    把模型保存成json
    :param model:
    :param path:
    :return:
    """
    model_dict = model.state_dict()
    for key in model_dict:
        model_dict[key] = model_dict[key].tolist()
    with open(path, 'w') as f:
        json.dump(model_dict, f)


def load_model_from_json(path: str):
    """
    从json中读模型字典
    TODO: 直接返回模型
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        model_dict = json.load(f)
    for key in model_dict:
        model_dict[key] = torch.Tensor(model_dict[key])
    return model_dict
