import numbers

import numpy as np


def resolve_max_features(max_features, n_features):
    """将特征子采样配置解析为当前训练数据上的正整数。"""
    if n_features < 1:
        raise ValueError("X 必须至少包含一个特征。")

    if max_features is None:
        return n_features

    if isinstance(max_features, bool):
        raise TypeError("max_features 不能是布尔值。")

    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError(f"不支持的 max_features 字符串：{max_features!r}。")

    if isinstance(max_features, numbers.Integral):
        if not 1 <= max_features <= n_features:
            raise ValueError(
                f"整数 max_features 必须位于 [1, {n_features}]。"
            )
        return int(max_features)

    if isinstance(max_features, numbers.Real):
        if not 0 < max_features <= 1:
            raise ValueError("浮点 max_features 必须位于 (0, 1]。")
        return max(1, int(max_features * n_features))

    raise TypeError("max_features 必须是 None、整数、浮点比例、'sqrt' 或 'log2'。")


def split_threshold(left, right):
    """计算不会因端点直接相加而溢出的分裂阈值。"""
    left_float = float(left)
    right_float = float(right)
    midpoint = left_float / 2 + right_float / 2
    if (
        not np.isfinite(midpoint)
        or midpoint <= left_float
        or midpoint >= right_float
    ):
        return right
    return midpoint
