import numpy as np
import pytest
from mlearn import linear_model

def get_dataset():
    # 创建一些示例数据
    np.random.seed(0)
    X = np.random.rand(100, 1)  # shape: (100, 1)
    y = 2 + 3 * X + np.random.randn(100, 1) * 0.1  # shape: (100, 1)
    return X, y


def test_linear_numpy_basic():
    X, y = get_dataset()

    # 创建并训练模型
    model = linear_model.LinearRegression()
    model.fit(X, y)  # X shape: (100, 1), y shape: (100, 1)

    np.testing.assert_allclose(model.coef_, np.array([3.0]), atol=0.1)
    np.testing.assert_allclose(model.intercept_, 2.0, atol=0.1)
    assert model.score(X, y) > 0.98

    # 进行预测
    X_test = np.array([[0.5]])  # shape: (1, 1)
    prediction = model.predict(X_test)
    assert prediction.shape == (1,)
    np.testing.assert_allclose(prediction, np.array([3.5]), atol=0.1)


def test_linear_torch_basic():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    X, y = get_dataset()

    # 创建并训练模型
    model = linear_model.linear_regression.torch.LinearRegression()
    optim = 'sgd'
    optim_kwargs = {'lr': 0.01}
    model.fit(X, y, epochs=2000, optim=optim, **optim_kwargs)

    np.testing.assert_allclose(model.coef_, np.array([3.0]), atol=0.15)
    np.testing.assert_allclose(model.intercept_, 2.0, atol=0.15)
    assert model.score(X, y) > 0.98

    # 进行预测
    X_test = np.array([[0.5]])  # shape: (1, 1)
    prediction = model.predict(X_test)
    assert prediction.shape == (1,)
    np.testing.assert_allclose(prediction, np.array([3.5]), atol=0.15)


def test_linear_numpy_supports_rank_deficient_features():
    """验证完全共线特征通过最小二乘求解，不再要求正规方程可逆。"""
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])

    model = linear_model.LinearRegression().fit(X, y)

    np.testing.assert_allclose(model.predict(X), y)
    assert model.coef_.shape == (2,)
    assert np.ndim(model.intercept_) == 0


def test_linear_numpy_matches_lstsq_for_underdetermined_data():
    """验证特征多于样本时返回 NumPy 最小范数最小二乘解。"""
    X = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    y = np.array([2.0, 3.0])
    design = np.column_stack((np.ones(len(X)), X))
    expected, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    model = linear_model.LinearRegression().fit(X, y)

    np.testing.assert_allclose(
        np.concatenate(([model.intercept_], model.coef_)),
        expected
    )


def test_linear_numpy_rejects_multioutput_target():
    """验证当前单输出模型不会静默丢弃多输出目标列。"""
    with pytest.raises(ValueError, match="单列二维数组"):
        linear_model.LinearRegression().fit(
            np.ones((3, 1)),
            np.ones((3, 2))
        )


def test_linear_numpy_predict_validates_and_normalizes_features():
    """验证列表和一维单样本预测使用训练特征契约。"""
    model = linear_model.LinearRegression().fit(
        [[0.0], [1.0], [2.0]],
        [1.0, 3.0, 5.0]
    )

    np.testing.assert_allclose(model.predict([[3.0]]), np.array([7.0]))
    np.testing.assert_allclose(model.predict(np.array([3.0])), np.array([7.0]))

    with pytest.raises(ValueError, match="应包含 1 个特征"):
        model.predict([[1.0, 2.0]])

    with pytest.raises(ValueError, match="尚未训练"):
        linear_model.LinearRegression().predict([[1.0]])


def test_linear_numpy_constant_target_score_is_finite():
    """验证常量目标评分复用公共 R² 语义，不再除零。"""
    X = np.arange(5, dtype=float).reshape(-1, 1)
    y = np.ones(5)
    model = linear_model.LinearRegression().fit(X, y)

    assert model.score(X, y) == 1.0


def test_linear_numpy_no_intercept_resets_intercept_attribute():
    """验证无截距训练后的公开截距属性为 None。"""
    model = linear_model.LinearRegression(fit_intercept=False).fit(
        [[1.0], [2.0]],
        [2.0, 4.0]
    )

    assert model.intercept_ is None


def test_linear_torch_constant_target_score_is_finite():
    """验证 PyTorch 后端与 NumPy 后端使用相同 R² 语义。"""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    X = np.arange(5, dtype=float).reshape(-1, 1)
    y = np.ones(5)
    model = linear_model.linear_regression.torch.LinearRegression()
    model.fit(X, y, epochs=20, optim="lbfgs", lr=0.5)

    assert model.score(X, y) == 1.0


if __name__ == '__main__':
    if 1:
        test_linear_numpy_basic()
    if 1:
        test_linear_torch_basic()
