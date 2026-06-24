import config
import numpy as np
from mlearn import tree
from mlearn.metrics import accuracy_score


def get_dataset():
    np.random.seed(0)
    X = np.random.rand(100, 2)  # shape: (100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # shape: (100,)
    return X, y  # X shape: (100, 2), y shape: (100,)

def test_tree_classifier_hello():
    # 示例数据 | Example data
    X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y_train = np.array([0, 1, 1, 0])

    # 创建决策树分类器实例
    clf = tree.DecisionTreeClassifier(max_depth=3, ccp_alpha=0.1)

    # 训练模型 | Train the model
    clf.fit(X_train, y_train)

    # 预测 | Predict    
    X_test = np.array([[0, 0], [1, 1]])
    predictions = clf.predict(X_test)
    print("Predictions:", predictions)

    # 计算准确率 | Calculate accuracy
    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print("Accuracy:", accuracy)

def test_tree_classifier():
    X, y = get_dataset()
    # print("X:", X)
    # print("y:", y)

    # 创建并训练模型 | Create and train the model
    clf = tree.DecisionTreeClassifier(max_depth=3, ccp_alpha=0.0)
    clf.fit(X, y)  # X shape: (100, 2), y shape: (100,)

    # 进行预测 | Predict
    X_test = np.array([[0.5, 0.5], [0.8, 0.8]])  # shape: (2, 2)
    predictions = clf.predict(X_test)  # shape: (2,)
    print("Predictions:", predictions)

    # 计算准确率 | Calculate accuracy   
    y_pred = clf.predict(X)  # shape: (100,)
    accuracy = np.mean(y_pred == y)
    print("Accuracy:", accuracy)


def test_tree_classifier_supports_non_consecutive_integer_labels():
    """验证分类树不会把非连续整数标签直接作为数组下标。"""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([10, 10, 20, 20])

    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), y)
    np.testing.assert_array_equal(clf.classes_, np.array([10, 20]))


def test_tree_classifier_supports_string_labels():
    """验证公开预测会从内部类别编码恢复为字符串标签。"""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["down", "down", "up", "up"])

    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), y)


def test_tree_classifier_constant_features_create_leaf():
    """验证没有合法阈值时生成叶节点，而不是使用空阈值继续分裂。"""
    X = np.ones((4, 1))
    y = np.array([0, 1, 0, 1])

    clf = tree.DecisionTreeClassifier().fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), np.zeros(4, dtype=int))
    np.testing.assert_array_equal(clf.feature_importance(), np.zeros(1))


def test_tree_classifier_allows_zero_gain_split_for_xor():
    """验证根节点零收益时仍可继续生长并在下一层分开 XOR 标签。"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    clf = tree.DecisionTreeClassifier(max_depth=2).fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), y)


if __name__ == '__main__':
    if 1:
        test_tree_classifier_hello()
    if 1:
        test_tree_classifier()
