# PLAN-006 实施结果：修复集成模型重复训练状态

## 实施内容

1. 随机森林分类器和回归器在每次 `fit` 开始时重置 `trees`。
2. GBDT 分类器在每次 `fit` 开始时重置 `trees`。
3. GBDT 回归器在每次 `fit` 开始时重置 `trees_`。
4. 四个模型分别增加重复训练回归测试。

## Review 结果

1. 所有清空动作都位于训练循环之前。
2. 首次和第二次训练后，基学习器数量均严格等于 `n_estimators`。
3. 保留旧树对象引用后复核，新一轮树与旧树没有对象复用。
4. 单次训练算法和预测聚合方式未改变。
5. 未发现需要追加修复的问题。

## 验证结果

- `python -m pytest tests/ensemble/random_forest tests/ensemble/gbdt -q`
  - 结果：`9 passed`
- 四个模型手工重复训练：
  - 树数量：均为配置值
  - 旧树复用：均为 `False`
- `python -m pytest -q`
  - 结果：`27 passed`

## 与计划差异

无。
