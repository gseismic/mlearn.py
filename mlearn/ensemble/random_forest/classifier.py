import numpy as np
from collections import Counter
from ...tree.classifier import DecisionTreeClassifier
from ...tree.utils import resolve_max_features


class RandomForestClassifier:
    """随机森林分类器 / Random Forest Classifier
    
    选择票数最多的类别作为最终类别 / Choose the class with the most votes as the final class
    """
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, max_features='sqrt',
                 random_state=None):
        """初始化随机森林 / Initialize the random forest"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """训练随机森林模型 / Train the random forest model"""
        self.trees = []
        rng = np.random.default_rng(self.random_state)
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.max_features_ = resolve_max_features(
            self.max_features,
            self.n_features
        )
        
        # 训练每棵决策树 / Train each decision tree
        for _ in range(self.n_estimators):
            # 使用自助采样选择训练数据 / Use bootstrap sampling to select training data
            idxs = rng.choice(len(X), len(X), replace=True)
            tree_seed = rng.integers(0, np.iinfo(np.int32).max)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features_,
                random_state=tree_seed
            )
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        """使用随机森林进行预测 / Make predictions using the random forest"""
        # 收集每棵树的预测结果 / Collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 使用多数投票确定最终预测 / Use majority voting to determine the final prediction
        # return np.array([np.bincount(p).argmax() for p in tree_preds.T])
        return np.array([Counter(pred).most_common(1)[0][0] for pred in tree_preds.T])
