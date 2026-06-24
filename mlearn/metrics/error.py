import numpy as np
from .utils import ensure_matching_1d_arrays


def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error.

    计算均方误差（MSE）。

    Parameters:
    - y_true: numpy.ndarray, shape为 (n_samples,) 的真实目标值。
    - y_pred: numpy.ndarray, shape为 (n_samples,) 的预测目标值。

    Returns:
    - mse: float, mean squared error.
    - mse: float，均方误差。
    """
    y_true, y_pred = ensure_matching_1d_arrays(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """
    Compute the root mean squared error.

    计算均方根误差（RMSE）。

    Parameters:
    - y_true: numpy.ndarray, shape为 (n_samples,) 的真实目标值。
    - y_pred: numpy.ndarray, shape为 (n_samples,) 的预测目标值。

    Returns:
    - rmse: float，均方根误差。
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def log_loss(y_true, y_pred):
    y_true, y_pred = ensure_matching_1d_arrays(y_true, y_pred)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
