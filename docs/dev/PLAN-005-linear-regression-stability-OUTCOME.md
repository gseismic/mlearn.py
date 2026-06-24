# PLAN-005 实施结果：修复线性回归数值稳定性

## 实施内容

1. 使用 `np.asarray` 规范输入并校验特征、目标维度和样本数。
2. 单列二维目标转换为一维目标，多输出目标明确拒绝。
3. 使用 `np.linalg.lstsq(..., rcond=None)` 替代正规方程显式求逆。
4. 保持 `coef_` 一维和 `intercept_` 标量的现有 API。
5. 新增秩亏、欠定和多输出目标测试。

## Review 结果

1. 仅在 `fit_intercept=True` 时添加常数列。
2. 共线和欠定设计矩阵返回最小范数最小二乘解。
3. `fit_intercept=False` 的参数和预测行为保持正确。
4. 不再构造 `XᵀX`，避免条件数平方和奇异矩阵求逆。
5. 未发现需要追加修复的问题。

## 验证结果

- `python -m pytest tests/linear_model/test_linear_regression.py -q`
  - 结果：`5 passed`
- 完全共线特征预测：`[1.0, 2.0, 3.0]`
- 欠定参数与 `np.linalg.lstsq` 结果一致。
- `python -m pytest -q`
  - 结果：`23 passed`

## 与计划差异

无。
