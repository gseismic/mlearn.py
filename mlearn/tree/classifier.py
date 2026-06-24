import numpy as np
from collections import Counter
from .utils import resolve_max_features
from ..utils import validate_X, validate_X_y

"""
算法思想:
    1. 从根节点开始，递归地对每个节点进行分裂，直到满足停止条件。
    2. 在每个节点，随机选择特征子集，并寻找最佳分裂特征和阈值。
    3. 根据最佳分裂特征和阈值，将数据分裂为左右两个子节点。
    4. 递归地对左右两个子节点进行分裂，直到满足停止条件。
    5. 返回训练好的决策树。

 Algorithm idea:
    1. Start from the root node and recursively split each node until the stopping condition is met.
    2. At each node, randomly select a subset of features and find the best feature and threshold for splitting.
    3. Split the data into two child nodes based on the best feature and threshold.
    4. Recursively split the left and right child nodes until the stopping condition is met.
    5. Return the trained decision tree.
"""

class DecisionTreeClassifier:
    """决策树分类器 / Decision Tree Classifier
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None,
                 ccp_alpha=0.0, random_state=None):
        """初始化决策树 / Initialize the decision tree
        
        Args:
            - max_depth: int, 决策树的最大深度。None 表示无限制。| int, the maximum depth of the decision tree. None means no limit.
            - min_samples_split: int, 分裂节点所需的最小样本数。| int, the minimum number of samples required to split a node.
            - max_features: int, 每次分裂时考虑的最大特征数。None 表示使用所有特征。| int, the maximum number of features to consider for splitting. None means using all features.   
        Notes:
            - 如果设置了max_features，则每次分裂时随机只考虑max_features个特征，选择特征具有`随机性` | If max_features is set, only max_features features are randomly selected for each split, with a random selection of features.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_features_ = None  # 当前训练数据解析后的特征子采样数量
        self.tree = None
        self.n_classes = None # 类别数
        self.n_features = None # 特征数 
        self.n_samples = None  # 根训练样本数，用于剪枝风险归一化
        self.classes_ = None  # 原始类别标签，树内部使用连续整数编码
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self._rng = None
    
    def fit(self, X, y):
        """训练决策树模型 / Train the decision tree model"""
        X, y = validate_X_y(X, y)
        self._rng = np.random.default_rng(self.random_state)
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        self.max_features_ = resolve_max_features(
            self.max_features,
            self.n_features
        )
        self.tree = self._grow_tree(X, encoded_y)
        if self.ccp_alpha > 0:
            self.tree = self._prune_tree(self.tree, X, encoded_y)
        return self

    def predict(self, X):
        """使用训练好的模型进行预测 / Make predictions using the trained model"""
        if self.tree is None:
            raise ValueError("模型尚未训练，请先调用 fit。")
        X = validate_X(X, n_features=self.n_features, allow_1d=True)
        encoded_predictions = np.array(
            [self._predict_tree(x, self.tree) for x in X],
            dtype=int
        )
        return self.classes_[encoded_predictions]

    def feature_importance(self):
        """计算特征重要性/ Calculate feature importance
        
        Returns:
            - importance: numpy.ndarray, shape (n_features,), 特征重要性。| numpy.ndarray, shape (n_features,), feature importance.
        """
        importance = np.zeros(self.n_features)
        self._feature_importance(self.tree, importance)
        total_importance = np.sum(importance)
        if total_importance == 0:
            return importance
        return importance / total_importance

    def _feature_importance(self, node, importance):
        if 'value' in node:
            return
        
        feature_idx = node['feature_idx']
        importance[feature_idx] += self._node_impurity_decrease(node)
        
        self._feature_importance(node['left'], importance)
        self._feature_importance(node['right'], importance)

    def _node_impurity_decrease(self, node):
        """计算节点分裂导致的不纯度减少量 / Calculate the impurity reduction caused by node splitting
        
        Args:
            - node: dict, 当前节点。| dict, the current node.
        Returns:
            - impurity_decrease: float, 不纯度减少量。| float, the impurity reduction.
        """
        # 如果是叶子节点，不纯度减少量为0
        if 'value' in node:
            return 0

        # 获取父节点、左子节点和右子节点的样本数和不纯度
        n_parent = node['n_samples']
        n_left = node['left']['n_samples']
        n_right = node['right']['n_samples']

        impurity_parent = self._calculate_gini(node['class_counts'])
        impurity_left = self._calculate_gini(node['left']['class_counts'])
        impurity_right = self._calculate_gini(node['right']['class_counts'])

        # 计算不纯度减少量
        impurity_decrease = (
            impurity_parent 
            - (n_left / n_parent) * impurity_left 
            - (n_right / n_parent) * impurity_right
        )

        return n_parent * impurity_decrease

    def _prune_tree(self, node, X, y):
        """按全局加权 Gini 风险和叶节点数量执行代价复杂度剪枝。"""
        if 'value' in node:
            return node

        # 递归剪枝左右子树
        node['left'] = self._prune_tree(node['left'], X[X[:, node['feature_idx']] < node['threshold']], 
                                        y[X[:, node['feature_idx']] < node['threshold']])
        node['right'] = self._prune_tree(node['right'], X[X[:, node['feature_idx']] >= node['threshold']], 
                                         y[X[:, node['feature_idx']] >= node['threshold']])

        subtree_cost = (
            self._leaf_impurity(node) / self.n_samples
            + self.ccp_alpha * self._count_leaves(node)
        )
        leaf_cost = (
            node['n_samples'] * self._calculate_gini(node['class_counts'])
            / self.n_samples
            + self.ccp_alpha
        )

        if leaf_cost <= subtree_cost:
            return self._make_leaf(y)

        return node

    def _leaf_impurity(self, node):
        """汇总子树叶节点的样本加权 Gini 风险。"""
        if 'value' in node:
            return (
                node['n_samples']
                * self._calculate_gini(node['class_counts'])
            )
        return (
            self._leaf_impurity(node['left'])
            + self._leaf_impurity(node['right'])
        )

    def _count_leaves(self, node):
        """计算子树叶节点数量。"""
        if 'value' in node:
            return 1
        return self._count_leaves(node['left']) + self._count_leaves(node['right'])
    
    def _calculate_gini(self, class_counts):
        # Gini = 1 - Σ(pi^2)
        # 基尼不纯度的值范围是[0, 1-1/k] / The value range of Gini impurity is [0, 1-1/k]   
        n_samples = sum(class_counts)
        if n_samples == 0:
            return 0
        gini = 1.0 - sum((count / n_samples) ** 2 for count in class_counts)
        return gini

    def _grow_tree(self, X, y, depth=0):
        """递归生成决策树 / Recursively grow the decision tree"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 检查停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            return self._make_leaf(y)

        # 随机选择特征子集
        feature_idxs = self._rng.choice(
            n_features,
            self.max_features_,
            replace=False
        )

        # 寻找最佳分裂
        best_feature, best_threshold = self._best_split(X[:, feature_idxs], y)

        # 常量特征等场景不存在合法阈值，当前节点应直接成为叶节点
        if best_feature is None or best_threshold is None:
            return self._make_leaf(y)

        # 分裂数据
        left_idxs = X[:, feature_idxs[best_feature]] < best_threshold
        right_idxs = ~left_idxs

        # 递归生成左右子树
        left_tree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_tree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        # class_counts 是每个类别的样本数 / class_counts is the number of samples for each class
        return {
            'feature_idx': feature_idxs[best_feature],
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'n_samples': n_samples,
            'class_counts': [np.sum(y == c) for c in range(self.n_classes)],
            'impurity': self._calculate_gini([np.sum(y == c) for c in range(self.n_classes)])
        }

    def _make_leaf(self, y):
        """构造多数类叶节点，并保留剪枝和特征重要性所需统计量。"""
        return {
            'value': Counter(y).most_common(1)[0][0],
            'n_samples': len(y),
            'class_counts': [np.sum(y == c) for c in range(self.n_classes)]
        }
        
    def _best_split(self, X, y):
        """寻找最佳分裂特征和阈值 / Find the best feature and threshold for splitting"""
        m = X.shape[0]
        if m <= 1:
            return None, None

        # 计算父节点的基尼不纯度 / Calculate the Gini impurity of the parent node
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds, classes = zip(*sorted(zip(X[:, feature], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                # 如果基尼不纯度小于最佳基尼不纯度，则更新最佳基尼不纯度和最佳特征、阈值 / If the Gini impurity is less than the best Gini impurity, update the best Gini impurity, best feature, and threshold 
                if best_feature is None or gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return best_feature, best_threshold

    def _predict_tree(self, x, tree):
        """使用决策树进行单个样本的预测 / Make a prediction for a single sample using the decision tree"""
        if 'value' in tree:
            return tree['value']

        if x[tree['feature_idx']] < tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])
