# PLAN-013 实施结果：隔离模型随机状态

## 实施内容

1. 分类树和回归树增加 `random_state` 与私有 `Generator`。
2. 每次 `fit` 从构造参数重新创建私有随机生成器。
3. 特征子集抽样改用树私有生成器。
4. 两种随机森林使用本地生成器执行 bootstrap。
5. 森林为每棵树派生独立整数子种子。
6. 回归森林增加 `random_state` 参数。
7. 新增全局状态隔离、子种子唯一和预测可复现测试。

## Review 结果

1. 生产代码不存在 `np.random.seed`、全局 `choice`、`rand` 或 `randn` 调用。
2. bootstrap 与特征抽样全部使用模型私有生成器。
3. 相同构造种子的模型产生相同预测。
4. 每次 `fit` 重新初始化生成器，重复训练保持可复现。
5. 森林内各树子种子唯一，避免相同特征抽样序列。
6. 训练前后调用方全局随机序列完全一致。
7. 未发现需要追加修复的问题。

## 验证结果

- `python -m pytest tests/tree tests/ensemble/random_forest tests/ensemble/gbdt -q`
  - 结果：`33 passed`
- 手工全局随机状态检查：`True`
- 分类和回归森林子种子唯一性：`True`
- `python -m pytest -q`
  - 结果：`42 passed`
- 系统 `pytest -q`
  - 结果：`41 passed, 1 skipped`

## 与计划差异

无。
