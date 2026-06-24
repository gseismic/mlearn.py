import numpy as np
import pytest

from mlearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def test_metrics_basic():
    # 示例数据
    y_true_class = [0, 1, 1, 0]
    y_pred_class = [0, 1, 0, 1]

    y_true_reg = [3, -0.5, 2, 7]
    y_pred_reg = [2.5, 0.0, 2, 8]

    # 分类指标
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class)
    recall = recall_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class)

    assert accuracy == 0.5
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5

    # 回归指标
    mse = mean_squared_error(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_reg, y_pred_reg)

    assert mse == 0.375
    np.testing.assert_allclose(rmse, np.sqrt(0.375))
    np.testing.assert_allclose(r2, 0.9486081370449679)


def test_regression_metrics_prevent_column_vector_broadcasting():
    """验证列向量和一维向量会先规范形状，不会广播成矩阵。"""
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([1.0, 2.0])

    assert mean_squared_error(y_true, y_pred) == 0.0
    assert r2_score(y_true, y_pred) == 1.0


def test_log_loss_accepts_lists():
    """验证对数损失与其他指标一样支持列表输入。"""
    loss = log_loss([0, 1], [0.1, 0.9])

    np.testing.assert_allclose(loss, -np.log(0.9))


def test_metrics_reject_mismatched_or_matrix_targets():
    """验证非法形状在算术运算前明确失败。"""
    with pytest.raises(ValueError, match="长度必须相同"):
        mean_squared_error([1, 2], [1])

    with pytest.raises(ValueError, match="必须是一维数组"):
        accuracy_score(np.ones((2, 2)), np.ones((2, 2)))


def test_r2_constant_target_handles_floating_point_residuals():
    """验证常量目标的数值舍入误差不会被误判为失败预测。"""
    y_true = np.ones(5)

    assert r2_score(y_true, y_true + 1e-12) == 1.0
    assert r2_score(y_true, np.zeros(5)) == 0.0


def test_binary_scores_support_arbitrary_positive_labels():
    """验证二分类指标按 pos_label 执行 one-vs-rest 统计。"""
    y_true = np.array(["no", "yes", "yes"])
    y_pred = np.array(["no", "yes", "no"])

    assert precision_score(y_true, y_pred, pos_label="yes") == 1.0
    assert recall_score(y_true, y_pred, pos_label="yes") == 0.5
    np.testing.assert_allclose(
        f1_score(y_true, y_pred, pos_label="yes"),
        2 / 3
    )

    assert precision_score([1, 2, 2], [1, 2, 1]) == 0.5
    assert recall_score([1, 2, 2], [1, 2, 1]) == 1.0


def test_log_loss_supports_arbitrary_binary_labels():
    """验证对数损失先按正类标签编码，再应用二元交叉熵。"""
    loss = log_loss([1, 2], [0.8, 0.2], pos_label=1)

    np.testing.assert_allclose(loss, -np.log(0.8))


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([0, 1, 2], [0.1, 0.5, 0.9], "仅支持二分类"),
        ([0, 1], [np.nan, 0.8], "有限概率"),
        ([0, 1], [-0.1, 1.1], r"\[0, 1\]"),
    ]
)
def test_log_loss_rejects_invalid_targets_and_probabilities(
    y_true,
    y_pred,
    message
):
    """验证无效二分类概率不会被静默裁剪或传播为 NaN。"""
    with pytest.raises(ValueError, match=message):
        log_loss(y_true, y_pred)


if __name__ == '__main__':
    if 1:
        test_metrics_basic()
