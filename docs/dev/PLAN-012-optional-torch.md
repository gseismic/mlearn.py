# PLAN-012：隔离 PyTorch 可选依赖

## 问题

`mlearn.linear_model.linear_regression.__init__` 无条件导入 PyTorch 后端。即使用户只使用 NumPy
线性回归，环境未安装 PyTorch 时也无法导入 `mlearn.linear_model`。基础安装依赖中又没有
声明 PyTorch，导致包行为与安装元数据不一致。

## 目标

1. 默认导入 NumPy 线性回归不需要 PyTorch。
2. 保持 `linear_model.linear_regression.torch` 的访问方式。
3. 首次访问 PyTorch 后端时再加载依赖。
4. 测试环境没有 PyTorch 时只跳过后端测试，不阻断其他测试。
5. 安装元数据声明可选 PyTorch extra。

## 实施步骤

1. 删除包初始化中的立即 PyTorch 导入。
2. 使用模块级 `__getattr__` 延迟加载 `torch` 子模块。
3. PyTorch 测试使用 `pytest.importorskip`。
4. 在 `setup.py` 增加 `extras_require={"torch": ["torch"]}`。
5. 分别在有、无 PyTorch 的解释器中验证导入和测试。

## 验证

1. 系统 Python 未安装 PyTorch时可以导入 `mlearn.linear_model`。
2. 系统 `pytest -q` 除 PyTorch 测试跳过外全部通过。
3. 安装 PyTorch 的解释器运行全部测试。
4. 现有 PyTorch 后端访问路径保持有效。

## Review 检查点

1. 延迟加载是否会递归调用自身。
2. NumPy 后端是否仍作为默认 `LinearRegression`。
3. 缺少 PyTorch 时是否只在实际访问后端时失败。
