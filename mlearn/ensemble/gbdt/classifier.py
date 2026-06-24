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

    def fit(self, X, y):
        self.trees = []
        self.F0 = np.log(np.mean(y) / (1 - np.mean(y)))
        F = np.full(len(y), self.F0)
        
        for _ in range(self.n_estimators):
            p = 1 / (1 + np.exp(-F))
            residual = y - p
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                         min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees.append(tree)
        
        return self

    def predict_proba(self, X):
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        probas = 1 / (1 + np.exp(-F))
        return np.vstack((1-probas, probas)).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
