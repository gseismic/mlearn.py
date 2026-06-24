import config
import numpy as np
import pytest
from mlearn import ensemble
from mlearn.metrics import accuracy_score, log_loss

def test_gbdt_classifier_basic():
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])

    model = ensemble.gbdt.GBDTClassifier(n_estimators=3, learning_rate=0.01, max_depth=3)
    model.fit(X, y)

    X_test = X
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    
    score = accuracy_score(y, predictions)
    print("Accuracy:", score)


def test_gbdt_classifier_basic2():
    np.random.seed(42)
    X = np.random.rand(1000, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    gbdt = ensemble.gbdt.GBDTClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
    gbdt.fit(X_train, y_train)

    y_pred_train = gbdt.predict(X_train)
    y_pred_test = gbdt.predict(X_test)
    y_prob_train = gbdt.predict_proba(X_train)[:, 1]
    y_prob_test = gbdt.predict_proba(X_test)[:, 1]

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    loss_train = log_loss(y_train, y_prob_train)
    loss_test = log_loss(y_test, y_prob_test)

    print(f"训练集准确率 | Train Accuracy: {acc_train:.4f}")
    print(f"测试集准确率 | Test Accuracy: {acc_test:.4f}")
    print(f"训练集对数损失 | Train Log Loss: {loss_train:.4f}")
    print(f"测试集对数损失 | Test Log Loss: {loss_test:.4f}")


def test_gbdt_classifier_refit_replaces_trees():
    """验证重复训练后的分类 GBDT 不保留旧基学习器。"""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = ensemble.gbdt.GBDTClassifier(n_estimators=3, max_depth=1)

    model.fit(X, y)
    model.fit(X, y)

    assert len(model.trees) == 3


def test_gbdt_classifier_supports_binary_string_labels():
    """验证二分类标签编码和概率列顺序与原始类别一致。"""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["down", "down", "up", "up"])
    model = ensemble.gbdt.GBDTClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=1
    ).fit(X, y)

    probabilities = model.predict_proba(X)
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.classes_, np.array(["down", "up"]))
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
    assert np.all(np.isfinite(probabilities))


def test_gbdt_classifier_newton_updates_reduce_log_loss():
    """验证基学习器更新确实降低二分类逻辑损失。"""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 3))
    y = (2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
    model = ensemble.gbdt.GBDTClassifier(
        n_estimators=20,
        learning_rate=0.1,
        max_depth=2
    ).fit(X, y)

    initial_probability = np.full(len(y), np.mean(y))
    assert log_loss(y, model.predict_proba(X)[:, 1]) < log_loss(
        y,
        initial_probability
    )


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 0, 0]),
        np.array([0, 1, 2]),
    ]
)
def test_gbdt_classifier_rejects_non_binary_targets(labels):
    """验证模型不会为单类别或多类别目标生成无效 log-odds。"""
    with pytest.raises(ValueError, match="恰好两个类别"):
        ensemble.gbdt.GBDTClassifier().fit(
            np.arange(len(labels), dtype=float).reshape(-1, 1),
            labels
        )


if __name__ == '__main__':
    if 1:
        test_gbdt_classifier_basic()
    if 1:
        test_gbdt_classifier_basic2()
