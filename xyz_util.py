from collections import deque
import numpy as np
import torch
import random
import os


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





