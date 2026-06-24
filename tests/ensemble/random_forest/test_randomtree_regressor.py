import config
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
    # np.random.seed(0)
    X = np.random.rand(100, 5)  # shape: (100, 5)
    y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100) * 0.0001  # shape: (100,)

    # 创建并训练模型 / Create and train the model
    rf = ensemble.RandomForestRegressor(n_estimators=20, max_depth=3)
    rf.fit(X, y)  # X shape: (100, 5), y shape: (100,)

    # 进行预测 / Make predictions
    X_test = np.random.rand(10, 5)  # shape: (10, 5)
    predictions = rf.predict(X_test)  # shape: (10,)
    print("Predictions:", predictions)

    # 计算均方误差 / Calculate mean squared error
    y_pred = rf.predict(X)  # shape: (100,)
    r2 = r2_score(y, y_pred)
    print("R2 Score:", r2)


def test_random_forest_regressor_refit_replaces_trees():
    """验证重复训练后的回归森林只包含本轮生成的树。"""
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = ensemble.RandomForestRegressor(n_estimators=3, max_depth=2)

    model.fit(X, y)
    model.fit(X, y)

    assert len(model.trees) == 3


if __name__ == '__main__':
    if 1:
        test_randomtree_classifier_hello()
