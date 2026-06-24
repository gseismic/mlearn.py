import numpy as np
import pytest

from mlearn import ensemble, tree


@pytest.mark.parametrize(
    ("model", "y"),
    [
        (tree.DecisionTreeClassifier(), np.array([0, 1, 0])),
        (tree.DecisionTreeRegressor(), np.array([0.0, 1.0, 2.0])),
        (ensemble.RandomForestClassifier(n_estimators=2), np.array([0, 1, 0])),
        (ensemble.RandomForestRegressor(n_estimators=2), np.array([0.0, 1.0, 2.0])),
        (ensemble.GBDTClassifier(n_estimators=2), np.array([0, 1, 0])),
        (ensemble.GBDTRegressor(n_estimators=2), np.array([0.0, 1.0, 2.0])),
    ]
)
def test_estimators_reject_mismatched_sample_counts(model, y):
    """验证所有树和集成模型在训练入口检查 X/y 样本数。"""
    with pytest.raises(ValueError, match="样本数量必须相同"):
        model.fit(np.ones((3, 1)), y[:2])


@pytest.mark.parametrize(
    ("model", "y"),
    [
        (tree.DecisionTreeClassifier(), np.array([], dtype=int)),
        (tree.DecisionTreeRegressor(), np.array([], dtype=float)),
        (ensemble.RandomForestClassifier(n_estimators=2), np.array([], dtype=int)),
        (ensemble.RandomForestRegressor(n_estimators=2), np.array([], dtype=float)),
        (ensemble.GBDTClassifier(n_estimators=2), np.array([], dtype=int)),
        (ensemble.GBDTRegressor(n_estimators=2), np.array([], dtype=float)),
    ]
)
def test_estimators_reject_empty_training_data(model, y):
    """验证空训练数据不会进入分裂或 bootstrap 逻辑。"""
    with pytest.raises(ValueError, match="至少包含一个样本"):
        model.fit(np.empty((0, 1)), y)


def test_regressors_accept_column_targets_and_single_sample_prediction():
    """验证回归目标统一为一维，预测支持一维单样本。"""
    X = np.arange(5, dtype=float).reshape(-1, 1)
    y = np.arange(5, dtype=float).reshape(-1, 1)

    tree_model = tree.DecisionTreeRegressor().fit(X, y)
    gbdt_model = ensemble.GBDTRegressor(n_estimators=3).fit(X, y)

    assert tree_model.predict(np.array([2.0])).shape == (1,)
    assert gbdt_model.predict(np.array([2.0])).shape == (1,)
    assert gbdt_model.residuals_.shape == (5,)


def test_predict_rejects_wrong_feature_count_and_unfitted_models():
    """验证预测在树遍历前检查拟合状态和特征数量。"""
    with pytest.raises(ValueError, match="尚未训练"):
        tree.DecisionTreeRegressor().predict([[1.0]])

    model = ensemble.RandomForestClassifier(
        n_estimators=2,
        random_state=0
    ).fit(np.ones((4, 2)), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="应包含 2 个特征"):
        model.predict([[1.0]])


def test_estimators_reject_non_finite_features():
    """验证 NaN 不会进入阈值排序并生成无效分裂。"""
    with pytest.raises(ValueError, match="只包含有限值"):
        tree.DecisionTreeClassifier().fit(
            np.array([[0.0], [np.nan]]),
            np.array([0, 1])
        )
