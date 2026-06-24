import numpy as np
from mlearn.tree.regressor import DecisionTreeRegressor


class GBDTClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.F0 = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y 必须是一维类别数组。")

        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("GBDTClassifier 仅支持恰好两个类别。")

        self.trees = []
        positive_rate = np.mean(encoded_y)
        self.F0 = np.log(positive_rate / (1 - positive_rate))
        F = np.full(len(encoded_y), self.F0, dtype=float)
        
        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            residual = encoded_y - p
            hessian = p * (1 - p)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                         min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            self._update_leaf_values(
                tree.tree,
                X,
                residual,
                hessian
            )
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees.append(tree)
        
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        F = np.full(len(X), self.F0, dtype=float)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        probas = self._sigmoid(F)
        return np.vstack((1-probas, probas)).T

    def predict(self, X):
        encoded_predictions = (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(int)
        return self.classes_[encoded_predictions]

    def _sigmoid(self, raw_prediction):
        """稳定计算二分类正类概率。"""
        clipped = np.clip(raw_prediction, -709, 709)
        return 1 / (1 + np.exp(-clipped))

    def _update_leaf_values(self, node, X, residual, hessian):
        """按逻辑损失的 Newton 步更新每个回归树叶节点输出。"""
        if 'value' in node:
            denominator = np.sum(hessian)
            node['value'] = (
                np.sum(residual) / denominator
                if denominator > np.finfo(float).eps
                else 0.0
            )
            return

        left_mask = X[:, node['feature_idx']] < node['threshold']
        self._update_leaf_values(
            node['left'],
            X[left_mask],
            residual[left_mask],
            hessian[left_mask]
        )
        self._update_leaf_values(
            node['right'],
            X[~left_mask],
            residual[~left_mask],
            hessian[~left_mask]
        )
