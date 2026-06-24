# PLAN-011 实施结果：补齐测试断言与稳定执行方式

## 实施内容

1. 为原有 11 个仅打印测试增加明确的数值、形状或精度断言。
2. 删除所有测试文件中的 `import config`。
3. 删除 6 个通过相对路径修改 `sys.path` 的 `config.py`。
4. 新增 `pytest.ini`，固定测试目录和项目根 Python 路径。
5. 清理测试中的无用指标导入。

## Review 结果

1. 使用 AST 检查全部 39 个测试函数。
2. 每个测试均包含 `assert`、`np.testing.assert_*` 或 `pytest.raises`。
3. 测试目录不存在 `import config`、`sys.path.insert` 或剩余 `config.py`。
4. 新断言使用稳定的精度或行为边界，能够捕获算法退化。
5. 系统 pytest 已能找到本地 `mlearn`，证明路径配置问题已修复。

## 验证结果

- AST 断言审计：
  - 缺少断言测试数：`0`
- `python -m pytest -q`
  - 结果：`39 passed`
- `pytest -q`
  - 项目路径导入成功
  - 剩余失败：系统 Python 未安装可选 PyTorch，但 NumPy 线性回归导入时强制加载 PyTorch
  - 该问题由 `PLAN-012` 继续修复

## 与计划差异

普通 `pytest` 的项目路径问题已经消除，但暴露出独立的可选依赖强制导入问题。该问题按既定
拆分在下一计划处理，避免测试质量变更与包依赖行为混合提交。
