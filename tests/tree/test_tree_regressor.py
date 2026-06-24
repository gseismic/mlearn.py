import config
import numpy as np
from mlearn import tree
from mlearn.metrics import r2_score


def get_dataset():
    np.random.seed(0)
    X = np.random.rand(100, 2)  # shape: (100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)
    return X, y

def test_tree_regressor_hello():
    # 示例数据
    X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y_train = np.array([0.1, 1.1, 1.0, 0.0])

    # 创建决策树分类器实例
    clf = tree.DecisionTreeRegressor(max_depth=3, ccp_alpha=0.05)

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测
    X_test = np.array([[0, 0], [1, 1]])
    predictions = clf.predict(X_test)
    print("Predictions:", predictions)

    y_true = np.array([0.1, 1.1])  # 真实目标值

    score = r2_score(y_true, predictions)
    print("r2_score:", score)
    
    importances = clf.feature_importance()
    print("feature_importance:", importances)


def test_tree_regressor_feature_importance_uses_training_statistics():
    """验证特征重要性使用真实节点 SSE，而不是把叶均值当作样本。"""
    rng = np.random.default_rng(86)
    X = rng.normal(size=(40, 3))
    y = 5 * X[:, 0] + 2 * X[:, 1] + rng.normal(scale=2, size=40)

    np.random.seed(86)
    model = tree.DecisionTreeRegressor(max_depth=3).fit(X, y)
    importances = model.feature_importance()

    assert np.argmax(importances) == 0
    np.testing.assert_allclose(np.sum(importances), 1.0)
    assert np.all(np.isfinite(importances))


def test_tree_regressor_leaf_feature_importance_is_zero():
    """验证没有分裂的回归树返回有限的零重要性。"""
    X = np.ones((4, 1))
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = tree.DecisionTreeRegressor().fit(X, y)

    np.testing.assert_array_equal(model.feature_importance(), np.zeros(1))


def test_tree_regressor_cost_complexity_pruning_uses_leaf_count():
    """验证剪枝比较使用叶节点 SSE 和叶节点数，而不是全部节点数。"""
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0])

    kept = tree.DecisionTreeRegressor(
        max_depth=3,
        ccp_alpha=10.0
    ).fit(X, y)
    pruned = tree.DecisionTreeRegressor(
        max_depth=3,
        ccp_alpha=30.0
    ).fit(X, y)

    assert kept._count_nodes(kept.tree) == 3
    np.testing.assert_array_equal(kept.predict(X), y)
    assert pruned._count_nodes(pruned.tree) == 1
    np.testing.assert_array_equal(pruned.predict(X), np.full(len(y), 5.0))
    np.testing.assert_array_equal(pruned.feature_importance(), np.zeros(1))


def test_tree_regressor_min_impurity_decrease_is_replication_invariant():
    """验证完全复制数据集不会改变最小不纯度下降的分裂判断。"""
    base_X = np.array([[0.0], [1.0]])
    base_y = np.array([0.0, 1.0])

    for repeat in (1, 2, 10):
        X = np.repeat(base_X, repeat, axis=0)
        y = np.repeat(base_y, repeat)

        split_model = tree.DecisionTreeRegressor(
            min_impurity_decrease=0.2
        ).fit(X, y)
        leaf_model = tree.DecisionTreeRegressor(
            min_impurity_decrease=0.3
        ).fit(X, y)

        assert split_model._count_nodes(split_model.tree) == 3
        assert leaf_model._count_nodes(leaf_model.tree) == 1


if __name__ == '__main__':
    if 1:
        test_tree_regressor_hello()
