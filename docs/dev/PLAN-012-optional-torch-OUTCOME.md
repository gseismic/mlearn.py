# PLAN-012 实施结果：隔离 PyTorch 可选依赖

## 实施内容

1. 删除线性回归包初始化时的立即 PyTorch 导入。
2. 使用模块级 `__getattr__` 首次访问时加载 `torch` 子模块。
3. 保持 NumPy `LinearRegression` 为默认公开实现。
4. PyTorch 测试在依赖缺失时使用 `pytest.importorskip`。
5. `setup.py` 增加 `torch` 可选 extra。

## Review 结果

1. 延迟加载只处理属性名 `torch`，其他缺失属性正常抛出 `AttributeError`。
2. 加载后模块缓存到全局命名空间，不会重复导入或递归。
3. NumPy 后端默认导入不触发 PyTorch。
4. 缺少 PyTorch 时仅显式访问该后端会抛 `ModuleNotFoundError`。
5. 现有 `linear_model.linear_regression.torch.LinearRegression` 路径保持有效。
6. 未发现需要追加修复的问题。

## 验证结果

- 有 PyTorch 的 Conda Python：
  - NumPy 和 PyTorch 后端均可导入
  - `python -m pytest -q`：`39 passed`
- 无 PyTorch 的系统 Python：
  - NumPy 后端可正常导入
  - 显式访问 PyTorch 后端时才提示缺少依赖
  - `pytest -q`：`38 passed, 1 skipped`

## 与计划差异

无。
