import numpy as np
from .utils import resolve_max_features

class DecisionTreeRegressor:
    """决策树回归器 / Decision Tree Regressor"""
    
    def __init__(self, 
                 max_depth=None, 
                 min_samples_split=2, 
                 max_features=None, 
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0):
        """初始化决策树 / Initialize the decision tree  
        
        Args:
            - max_depth: int, 决策树的最大深度。None 表示无限制。| int, the maximum depth of the decision tree. None means no limit.
            - min_samples_split: int, 分裂节点所需的最小样本数。| int, the minimum number of samples required to split a node.
            - max_features: int, 每次分裂时考虑的最大特征数。None 表示使用所有特征。| int, the maximum number of features to consider for splitting. None means using all features. 
            - ccp_alpha: float, 复杂度参数。| float, the complexity parameter.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_features_ = None  # 当前训练数据解析后的特征子采样数量
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None
        self.n_features = None
        self.n_samples = None
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y):
        """训练决策树模型 / Train the decision tree model"""
        # X shape: (n_samples, n_features), y shape: (n_samples,)
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        self.max_features_ = resolve_max_features(
            self.max_features,
            self.n_features
        )
        self.tree = self._grow_tree(X, y)
        if self.ccp_alpha > 0:
            self.tree = self._prune_tree(self.tree, X, y)
        return self

    def _prune_tree(self, tree, X, y):
        """按全局平均平方误差和叶节点数量执行代价复杂度剪枝。"""
        if 'value' in tree:
            return tree

        # 递归剪枝子树
        left_mask = X[:, tree['feature_idx']] < tree['threshold']
        tree['left'] = self._prune_tree(tree['left'], X[left_mask], y[left_mask])
        tree['right'] = self._prune_tree(tree['right'], X[~left_mask], y[~left_mask])

        subtree_cost = (
            self._leaf_squared_error(tree) / self.n_samples
            + self.ccp_alpha * self._count_leaves(tree)
        )
        leaf_cost = (
            tree['squared_error'] / self.n_samples
            + self.ccp_alpha
        )

        if leaf_cost <= subtree_cost:
            return self._make_leaf(y)
        
        return tree

    def _leaf_squared_error(self, tree):
        """汇总子树叶节点的残差平方和。"""
        if 'value' in tree:
            return tree['squared_error']
        return (
            self._leaf_squared_error(tree['left'])
            + self._leaf_squared_error(tree['right'])
        )

    def _count_leaves(self, tree):
        """计算子树叶节点数量。"""
        if 'value' in tree:
            return 1
        return self._count_leaves(tree['left']) + self._count_leaves(tree['right'])

    def _count_nodes(self, tree):
        """计算树中的节点数"""
        if 'value' in tree:
            return 1
        return 1 + self._count_nodes(tree['left']) + self._count_nodes(tree['right'])

    def predict(self, X):
        """使用训练好的模型进行预测 / Make predictions using the trained model"""
        # X shape: (n_samples, n_features), return shape: (n_samples,)
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """递归生成决策树 / Recursively grow the decision tree"""
        # X shape: (n_samples, n_features), y shape: (n_samples,)
        n_samples, n_features = X.shape

        # 检查停止条件 / Check stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            return self._make_leaf(y)

        # 随机选择特征子集 / Randomly select a subset of features
        feature_idxs = np.random.choice(n_features, self.max_features_, replace=False)

        # 寻找最佳分裂 / Find the best split
        best_feature, best_threshold, impurity_decrease = self._best_split(X[:, feature_idxs], y)
        weighted_impurity_decrease = impurity_decrease / self.n_samples

        # 如果无法找到有效的分裂，返回叶节点
        if best_feature is None or best_threshold is None or \
           weighted_impurity_decrease < self.min_impurity_decrease:
            return self._make_leaf(y)

        # 分裂数据 / Split the data
        left_idxs = X[:, feature_idxs[best_feature]] < best_threshold
        right_idxs = ~left_idxs

        # 递归生成左右子树 / Recursively grow left and right subtrees
        left_tree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_tree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {
            'feature_idx': feature_idxs[best_feature],
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'n_samples': n_samples,
            'squared_error': self._squared_error(y)
        }

    def _make_leaf(self, y):
        """构造叶节点并保存特征重要性和剪枝所需的真实训练统计量。"""
        return {
            'value': np.mean(y),
            'n_samples': len(y),
            'squared_error': self._squared_error(y)
        }

    def _squared_error(self, y):
        """计算目标值相对节点均值的残差平方和。"""
        if len(y) == 0:
            return 0.0
        return np.sum((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        """寻找最佳分裂特征和阈值 / Find the best feature and threshold for splitting"""
        # X shape: (n_samples, max_features), y shape: (n_samples,)
        m = X.shape[0]
        if m <= 1:
            return None, None, 0.0

        # 计算父节点的方差 / Calculate the variance of the parent node
        parent_var = np.var(y) * m
        best_var = parent_var
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds, targets = zip(*sorted(zip(X[:, feature], y)))
            left_sum = 0
            right_sum = sum(targets)
            left_count = 0
            right_count = m

            for i in range(1, m):
                left_sum += targets[i - 1]
                right_sum -= targets[i - 1]
                left_count += 1
                right_count -= 1

                if thresholds[i] == thresholds[i - 1]:
                    continue

                # left_mse = np.mean((left_y - np.mean(left_y)) ** 2) if len(left_y) > 0 else 0
                left_var = np.var(targets[:i]) * left_count
                right_var = np.var(targets[i:]) * right_count
                total_var = left_var + right_var

                if total_var < best_var:
                    best_var = total_var
                    best_feature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2


        impurity_decrease = parent_var - best_var
        return best_feature, best_threshold, impurity_decrease

    def _predict_tree(self, x, tree):
        """使用决策树进行单个样本的预测 / Make a prediction for a single sample using the decision tree"""
        # x shape: (n_features,)
        if 'value' in tree:
            return tree['value']

        if x[tree['feature_idx']] < tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])

    def feature_importance(self, weighted=True):
        """计算特征重要性 | Calculate feature importance"""
        # 方差减少 | Variance reduction
        if self.tree is None:
            raise ValueError("The model is not trained, please call the fit method first.")
        
        importance = np.zeros(self.n_features)
        self._feature_importance(self.tree, importance, weighted)
        
        # 归一化特征重要性
        total_importance = np.sum(importance)
        if total_importance == 0:
            return importance
        return importance / total_importance

    def _feature_importance(self, node, importance, weighted=False):
        """递归计算特征重要性 | Recursively calculate feature importance"""
        if 'value' in node:
            return
        
        feature = node['feature_idx']
        left = node['left']
        right = node['right']
        
        # 计算该节点的样本数和方差减少
        n_samples = self._count_samples(node)
        variance_reduction = self._calculate_variance_reduction(node)
        
        # 更新特征重要性
        if weighted:
            importance[feature] += variance_reduction * n_samples
        else:
            importance[feature] += variance_reduction
        
        # 递归处理子节点
        self._feature_importance(left, importance, weighted)
        self._feature_importance(right, importance, weighted)

    def _count_samples(self, node):
        """计算节点中的样本数 | Calculate the number of samples in the node"""
        return node['n_samples']

    def _calculate_variance_reduction(self, node):
        """计算节点的方差减少 | Calculate the variance reduction of the node"""
        if 'value' in node:
            return 0
        
        n_total = node['n_samples']
        children_error = (
            node['left']['squared_error'] + node['right']['squared_error']
        )
        return (node['squared_error'] - children_error) / n_total
