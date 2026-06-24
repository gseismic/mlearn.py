import numpy as np
import pytest

from mlearn import ensemble, tree


X = np.arange(4, dtype=float).reshape(-1, 1)
Y_CLASS = np.array([0, 0, 1, 1])
Y_REG = np.array([0.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("model", "target"),
    [
        (ensemble.RandomForestClassifier(n_estimators=0), Y_CLASS),
        (ensemble.RandomForestRegressor(n_estimators=0), Y_REG),
        (ensemble.GBDTClassifier(n_estimators=0), Y_CLASS),
        (ensemble.GBDTRegressor(n_estimators=0), Y_REG),
    ]
)
def test_ensemble_models_require_positive_n_estimators(model, target):
    """验证零基学习器不会生成空预测或 NaN 聚合。"""
    with pytest.raises(ValueError, match="n_estimators"):
        model.fit(X, target)


@pytest.mark.parametrize(
    ("model", "target", "message"),
    [
        (tree.DecisionTreeClassifier(max_depth=0), Y_CLASS, "max_depth"),
        (tree.DecisionTreeRegressor(min_samples_split=1), Y_REG, "min_samples_split"),
        (tree.DecisionTreeClassifier(ccp_alpha=-0.1), Y_CLASS, "ccp_alpha"),
        (
            tree.DecisionTreeRegressor(min_impurity_decrease=-0.1),
            Y_REG,
            "min_impurity_decrease"
        ),
    ]
)
def test_trees_reject_invalid_growth_parameters(model, target, message):
    """验证非法树生长和剪枝参数在训练前明确失败。"""
    with pytest.raises(ValueError, match=message):
        model.fit(X, target)


@pytest.mark.parametrize(
    ("model", "target"),
    [
        (ensemble.GBDTClassifier(learning_rate=0), Y_CLASS),
        (ensemble.GBDTRegressor(learning_rate=0), Y_REG),
    ]
)
def test_gbdt_requires_positive_learning_rate(model, target):
    """验证非正学习率不会静默生成常数模型。"""
    with pytest.raises(ValueError, match="learning_rate"):
        model.fit(X, target)


def test_boolean_hyperparameters_are_not_accepted_as_numbers():
    """验证 Python 布尔值不会绕过整数或实数参数校验。"""
    with pytest.raises(TypeError, match="n_estimators"):
        ensemble.RandomForestClassifier(n_estimators=True).fit(X, Y_CLASS)

    with pytest.raises(TypeError, match="learning_rate"):
        ensemble.GBDTRegressor(learning_rate=True).fit(X, Y_REG)
