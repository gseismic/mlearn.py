import numpy as np
from mlearn import ensemble


def get_dataset():
    np.random.seed(0)
    X = np.random.rand(100, 2)  # shape: (100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)
    return X, y


def test_randomtree_classifier_hello():
    # 创建一些示例数据 / Create some sample data
    rng = np.random.default_rng(0)
    X = rng.random((100, 5))  # shape: (100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)

    # 创建并训练模型 / Create and train the model
    rf = ensemble.RandomForestClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=0
    )
    rf.fit(X, y)  # X shape: (100, 5), y shape: (100,)

    # 进行预测 / Make predictions
    X_test = rng.random((10, 5))  # shape: (10, 5)
    predictions = rf.predict(X_test)  # shape: (10,)
    assert predictions.shape == (10,)

    # 计算准确率 / Calculate accuracy
    y_pred = rf.predict(X)  # shape: (100,)
    accuracy = np.mean(y_pred == y)
    assert accuracy >= 0.9


def test_random_forest_classifier_refit_replaces_trees():
    """验证重复训练不会把新树追加到旧森林。"""
    X, y = get_dataset()
    model = ensemble.RandomForestClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=0
    )

    model.fit(X, y)
    model.fit(X, y)

    assert len(model.trees) == 3


def test_random_forest_classifier_log2_single_feature_uses_one():
    """验证单特征数据的 log2 配置不会解析为零。"""
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    model = ensemble.RandomForestClassifier(
        n_estimators=3,
        max_depth=2,
        max_features="log2",
        random_state=0
    ).fit(X, y)

    assert model.max_features == "log2"
    assert model.max_features_ == 1
    np.testing.assert_array_equal(model.predict(X), y)


def test_random_forest_classifier_does_not_change_global_random_state():
    """验证森林训练不会重置或消耗调用方的全局随机序列。"""
    X, y = get_dataset()
    np.random.seed(123)
    expected = np.random.random(3)

    np.random.seed(123)
    ensemble.RandomForestClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=7
    ).fit(X, y)
    actual = np.random.random(3)

    np.testing.assert_array_equal(actual, expected)


def test_random_forest_classifier_random_state_is_reproducible():
    """验证相同随机种子生成相同分类森林预测。"""
    X, y = get_dataset()
    first = ensemble.RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=11
    ).fit(X, y)
    second = ensemble.RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=11
    ).fit(X, y)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert len({tree.random_state for tree in first.trees}) == 5


if __name__ == '__main__':
    if 1:
        test_randomtree_classifier_hello()
