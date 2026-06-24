import numpy as np


def ensure_array(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        raise TypeError(f'invalid type `{type(x)}`')


def ensure_1d_array(x, name):
    """将一维、行向量或列向量规范为一维数组。"""
    array = ensure_array(x)
    if array.ndim == 1:
        result = array
    elif array.ndim == 2 and 1 in array.shape:
        result = array.reshape(-1)
    else:
        raise ValueError(f"{name} 必须是一维数组、行向量或列向量。")

    if result.size == 0:
        raise ValueError(f"{name} 不能为空。")
    return result


def ensure_matching_1d_arrays(y_true, y_pred):
    """规范成对目标向量，并在计算前阻止长度广播。"""
    y_true = ensure_1d_array(y_true, "y_true")
    y_pred = ensure_1d_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的长度必须相同。")
    return y_true, y_pred
