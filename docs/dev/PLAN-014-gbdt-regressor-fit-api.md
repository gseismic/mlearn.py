# PLAN-014：修复 GBDT 回归训练 API

## 问题

`GBDTRegressor.fit` 是库内唯一训练完成后不返回 `self` 的估计器，导致
`GBDTRegressor(...).fit(X, y).predict(X)` 链式调用失败，也偏离项目声明的 sklearn-like
API。

## 目标

1. `GBDTRegressor.fit` 返回当前模型实例。
2. 不改变训练状态和预测算法。
3. 支持构造、训练、预测链式调用。

## 实施步骤

1. 在训练循环完成后返回 `self`。
2. 增加返回对象身份和链式预测测试。
3. 运行 GBDT 回归测试和全量测试。

## 验证

1. `fit` 返回值与模型对象使用 `is` 比较为真。
2. 链式调用返回正确形状预测。
3. 现有 GBDT 回归预测测试保持通过。

## Review 检查点

1. `return self` 是否位于完整训练循环之后。
2. 是否意外改变循环缩进或状态更新。
