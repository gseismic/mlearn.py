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
    pytest.importorskip("torch")
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


if __name__ == '__main__':
    if 1:
        test_linear_numpy_basic()
    if 1:
        test_linear_torch_basic()
