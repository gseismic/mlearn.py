import numpy as np
from mlearn import ensemble


def get_dataset():
    np.random.seed(0)
    X = np.random.rand(100, 2)  # shape: (100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)
    return X, y


def test_randomtree_classifier_hello():
    # 创建一些示例数据 / Create some sample data
    # np.random.seed(0)
    X = np.random.rand(100, 5)  # shape: (100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)

    # 创建并训练模型 / Create and train the model
    rf = ensemble.RandomForestClassifier(n_estimators=10, max_depth=3)
    rf.fit(X, y)  # X shape: (100, 5), y shape: (100,)

    # 进行预测 / Make predictions
    X_test = np.random.rand(10, 5)  # shape: (10, 5)
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


if __name__ == '__main__':
    if 1:
        test_randomtree_classifier_hello()
