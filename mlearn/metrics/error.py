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


def log_loss(y_true, y_pred, pos_label=1):
    y_true, y_pred = ensure_matching_1d_arrays(y_true, y_pred)
    classes = np.unique(y_true)
    if len(classes) > 2:
        raise ValueError("log_loss 仅支持二分类目标。")
    if pos_label not in classes:
        raise ValueError(f"pos_label={pos_label!r} 不存在于 y_true。")
    if not np.issubdtype(y_pred.dtype, np.number):
        raise TypeError("y_pred 必须包含数值概率。")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred 必须只包含有限概率。")
    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("y_pred 概率必须位于 [0, 1]。")

    encoded_y = (y_true == pos_label).astype(float)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        encoded_y * np.log(y_pred)
        + (1 - encoded_y) * np.log(1 - y_pred)
    )
