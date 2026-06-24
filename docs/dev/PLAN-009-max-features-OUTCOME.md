# PLAN-009 实施结果：修复 `max_features` 解析与重训行为

## 实施内容

1. 新增共享 `resolve_max_features` 解析器。
2. 支持 `None`、正整数、`(0, 1]` 浮点比例、`sqrt` 和 `log2`。
3. 解析结果限制在 `[1, n_features]`，非法配置明确失败。
4. 分类树、回归树和随机森林增加训练态 `max_features_`。
5. 用户配置 `max_features` 不再在训练时被改写。
6. `_best_split` 根据实际特征子矩阵列数遍历。
7. 增加单特征、小比例、非法值和换维度重训测试。

## Review 结果

1. 仓库中 `self.max_features` 只在构造函数赋值。
2. 树特征抽样统一使用 `self.max_features_`。
3. `1.0` 浮点比例解析为全部特征，整数 `1` 解析为一个特征。
4. 单特征 `sqrt/log2` 和任意正比例至少解析为一个特征。
5. 首轮 Review 发现布尔值会落入实数分支，已显式拒绝并增加测试。
6. 二次 Review 未发现剩余问题。

## 验证结果

- `python -m pytest tests/tree tests/ensemble/random_forest -q`
  - 结果：`21 passed`
- 参数矩阵覆盖 `n_features=1/2/5` 与全部支持类型。
- `True/False` 均明确抛出 `TypeError`。
- `python -m pytest -q`
  - 结果：`35 passed`

## 与计划差异

Review 阶段新增布尔值输入校验，防止 Python 的 `bool` 数值继承关系绕过类型约束。
