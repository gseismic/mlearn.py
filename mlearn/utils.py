import numbers

import numpy as np


def validate_X(X, n_features=None, allow_1d=False):
    """校验并规范机器学习模型使用的有限实数特征矩阵。"""
    X = np.asarray(X)
    if allow_1d and X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("X 必须是二维特征矩阵。")
    if X.shape[0] == 0:
        raise ValueError("X 必须至少包含一个样本。")
    if X.shape[1] == 0:
        raise ValueError("X 必须至少包含一个特征。")
    if not np.issubdtype(X.dtype, np.number) or np.issubdtype(
        X.dtype,
        np.complexfloating
    ):
        raise TypeError("X 必须包含实数特征。")
    if not np.all(np.isfinite(X)):
        raise ValueError("X 必须只包含有限值。")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"X 应包含 {n_features} 个特征，实际为 {X.shape[1]} 个。"
        )
    return X


def validate_y(y, numeric=False):
    """将一维或单列目标规范为一维，并校验缺失值。"""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    elif y.ndim != 1:
        raise ValueError("y 必须是一维数组或单列二维数组。")
    if y.size == 0:
        raise ValueError("y 必须至少包含一个样本。")

    if numeric:
        if not np.issubdtype(y.dtype, np.number) or np.issubdtype(
            y.dtype,
            np.complexfloating
        ):
            raise TypeError("回归目标 y 必须包含实数。")
        if not np.all(np.isfinite(y)):
            raise ValueError("回归目标 y 必须只包含有限值。")
        return y.astype(float, copy=False)

    if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
        raise ValueError("分类目标 y 必须只包含有限标签。")
    if y.dtype == object and any(value is None for value in y):
        raise ValueError("分类目标 y 不能包含 None。")
    return y


def validate_X_y(X, y, numeric_y=False):
    """统一校验监督学习特征和目标，并确保样本数一致。"""
    X = validate_X(X)
    y = validate_y(y, numeric=numeric_y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X 和 y 的样本数量必须相同。")
    return X, y


def validate_positive_integer(value, name, minimum=1, allow_none=False):
    """校验正整数类超参数，并拒绝布尔值。"""
    if allow_none and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} 必须是整数。")
    if value < minimum:
        raise ValueError(f"{name} 必须不小于 {minimum}。")


def validate_positive_real(value, name):
    """校验有限正实数超参数。"""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} 必须是实数。")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限正数。")


def validate_nonnegative_real(value, name):
    """校验有限非负实数超参数。"""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} 必须是实数。")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} 必须是有限非负数。")
