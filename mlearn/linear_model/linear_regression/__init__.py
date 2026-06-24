from importlib import import_module

from .numpy import LinearRegression
from . import numpy


def __getattr__(name):
    """按需加载可选后端，避免 NumPy 用户被迫安装 PyTorch。"""
    if name == "torch":
        module = import_module(f"{__name__}.torch")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
