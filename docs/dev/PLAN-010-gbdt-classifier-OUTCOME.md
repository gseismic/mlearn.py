# PLAN-010 实施结果：修复 GBDT 分类算法与标签处理

## 实施内容

1. 使用 `classes_` 和连续二元编码支持任意两种标签。
2. 单类别和多类别目标明确抛出 `ValueError`。
3. 使用正类比例 log-odds 初始化 raw score。
4. 每轮回归树拟合逻辑损失负梯度 `y-p`。
5. 每个叶节点按 `sum(y-p) / sum(p(1-p))` 执行 Newton 更新。
6. 使用裁剪 raw score 的稳定 sigmoid 计算概率。
7. `predict` 将二元结果映射回原始标签。
8. 增加字符串标签、概率、损失下降和非法类别数测试。

## Review 结果

1. 每轮叶值更新使用该轮更新前概率计算的梯度和 Hessian。
2. 递归分裂掩码与回归树预测路径一致，每个样本进入唯一叶节点。
3. `predict_proba` 第一列和第二列分别对应 `classes_[0]`、`classes_[1]`。
4. 极端 raw score 下概率保持有限且每行和为 1。
5. 固定数据和参数下，本库训练 log loss 与 scikit-learn 结果逐值一致。
6. 未发现需要追加修复的问题。

## 验证结果

- `python -m pytest tests/ensemble/gbdt/test_gbdt_classifier.py -q`
  - 结果：`7 passed`
- 与 scikit-learn 比较：
  - 10 棵树：两者均为 `0.44946013492845`
  - 100 棵树：两者均为 `0.23576053647051043`
- 极端输入下概率有限，字符串标签预测正确。
- `python -m pytest -q`
  - 结果：`39 passed`

## 与计划差异

无。
