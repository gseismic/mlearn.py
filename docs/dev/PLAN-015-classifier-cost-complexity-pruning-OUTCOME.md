# PLAN-015 实施结果：修复分类树代价复杂度剪枝

## 实施内容

1. 分类树保存根训练样本数用于风险归一化。
2. 子树风险改为所有叶节点样本加权 Gini 风险之和。
3. 复杂度项改为 `ccp_alpha * 叶节点数量`。
4. 单叶风险与子树风险使用同一根样本数尺度。
5. 删除只处理直接双叶子的限制。
6. 剪枝节点通过统一 `_make_leaf` 构造。
7. 增加明确 alpha 边界测试。

## Review 结果

1. 简单三节点树在 `alpha=0.5` 时与参考实现同时剪为单叶。
2. 13 节点深树在多组 alpha 下与 scikit-learn 节点数完全一致。
3. 任意深度子树均可通过自底向上代价比较整体替换。
4. 剪枝叶节点保留样本数和类别计数。
5. 剪枝后特征重要性为有限值。
6. 未发现需要追加修复的问题。

## 验证结果

- `python -m pytest tests/tree/test_tree_classifier.py -q`
  - 结果：`9 passed`
- 简单树和深树多组 alpha 节点数与 scikit-learn 一致。
- `python -m pytest -q`
  - 结果：`44 passed`
- 系统 `pytest -q`
  - 结果：`43 passed, 1 skipped`

## 与计划差异

无。
