import config
import numpy as np
from mlearn import ensemble
from mlearn.metrics import r2_score, accuracy_score, log_loss


def test_gbdt_regressor_basic():
    X = np.array([[1], [2], [3], [4], [5]])  # 特征数据 / Feature data (5 samples, 1 feature)
    y = np.array([300, 450, 550, 600, 700])  # 目标数据 / Target data (5 samples)

    model = ensemble.GBDTRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    X_test = np.array([[1.0], [3.0], [4.0]])
    predictions = model.predict(X_test)
    print("Predictions:", predictions)

    # 计算准确率 / Calculate accuracy
    # accuracy = np.mean(y_pred == y)
    # print("Accuracy:", accuracy)


def test_gbdt_regressor_refit_replaces_trees():
    """验证重复训练后的回归 GBDT 不保留旧基学习器。"""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = ensemble.GBDTRegressor(n_estimators=3, max_depth=1)

    model.fit(X, y)
    model.fit(X, y)

    assert len(model.trees_) == 3


if __name__ == '__main__':
    if 1:
        test_gbdt_regressor_basic()
