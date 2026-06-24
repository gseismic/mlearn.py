# PLAN-008 实施结果：修复回归树 `min_impurity_decrease` 语义

## 实施内容

1. 保留 `_best_split` 返回节点原始 SSE 减少量。
2. 在停止判断处将 SSE 减少量除以根训练样本数。
3. 使用全局样本加权下降量与 `min_impurity_decrease` 比较。
4. 增加 1、2、10 倍数据复制下的分裂一致性测试。

## Review 结果

1. 实现等价于标准 CART 的全局样本加权不纯度下降公式。
2. 分母始终为根训练样本数，不随递归节点变化。
3. `_best_split` 的候选排序仍使用原始 SSE，行为未改变。
4. 特征重要性继续使用节点 SSE，未发生重复归一化。
5. 与 scikit-learn 在所有测试复制倍数和阈值下节点数一致。
6. 未发现需要追加修复的问题。

## 验证结果

- `python -m pytest tests/tree/test_tree_regressor.py -q`
  - 结果：`5 passed`
- 复制倍数 `1/2/10`：
  - 阈值 `0.2`：本库和参考实现均为 3 个节点
  - 阈值 `0.3`：本库和参考实现均为 1 个节点
- `python -m pytest -q`
  - 结果：`29 passed`

## 与计划差异

无。
