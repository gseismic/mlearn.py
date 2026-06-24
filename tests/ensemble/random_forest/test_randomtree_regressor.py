import numpy as np
from mlearn import ensemble
from mlearn.metrics import r2_score


def get_dataset():
    np.random.seed(0)
    X = np.random.rand(100, 2)  # shape: (100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)
    return X, y


def test_randomtree_classifier_hello():
    # 创建一些示例数据 / Create some sample data
    rng = np.random.default_rng(0)
    X = rng.random((100, 5))  # shape: (100, 5)
    y = 3*X[:, 0] + 2*X[:, 1] + rng.normal(size=100) * 0.0001  # shape: (100,)

    # 创建并训练模型 / Create and train the model
    rf = ensemble.RandomForestRegressor(
        n_estimators=20,
        max_depth=4,
        random_state=0
    )
    rf.fit(X, y)  # X shape: (100, 5), y shape: (100,)

    # 进行预测 / Make predictions
    X_test = rng.random((10, 5))  # shape: (10, 5)
    predictions = rf.predict(X_test)  # shape: (10,)
    assert predictions.shape == (10,)

    # 计算均方误差 / Calculate mean squared error
    y_pred = rf.predict(X)  # shape: (100,)
    r2 = r2_score(y, y_pred)
    assert r2 >= 0.85


def test_random_forest_regressor_refit_replaces_trees():
    """验证重复训练后的回归森林只包含本轮生成的树。"""
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = ensemble.RandomForestRegressor(n_estimators=3, max_depth=2)

    model.fit(X, y)
    model.fit(X, y)

    assert len(model.trees) == 3


def test_random_forest_regressor_recomputes_max_features_on_refit():
    """验证字符串配置根据每次训练的特征维度重新解析。"""
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = ensemble.RandomForestRegressor(
        n_estimators=2,
        max_depth=1,
        max_features="sqrt"
    )

    model.fit(np.arange(16, dtype=float).reshape(4, 4), y)
    assert model.max_features_ == 2

    model.fit(np.arange(4, dtype=float).reshape(4, 1), y)
    assert model.max_features == "sqrt"
    assert model.max_features_ == 1


def test_random_forest_regressor_random_state_is_isolated_and_reproducible():
    """验证回归森林的本地随机源既隔离全局状态又可复现。"""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 3))
    y = 2 * X[:, 0] - X[:, 1]

    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    first = ensemble.RandomForestRegressor(
        n_estimators=5,
        max_depth=3,
        random_state=11
    ).fit(X, y)
    actual = np.random.random(3)
    second = ensemble.RandomForestRegressor(
        n_estimators=5,
        max_depth=3,
        random_state=11
    ).fit(X, y)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(first.predict(X), second.predict(X))
    assert len({tree.random_state for tree in first.trees}) == 5


if __name__ == '__main__':
    if 1:
        test_randomtree_classifier_hello()
