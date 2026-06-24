import numpy as np
from ...tree.regressor import DecisionTreeRegressor
from ...utils import validate_X, validate_X_y


class GBDTRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators  # 基学习器的数量
        self.learning_rate = learning_rate  # 学习率
        self.max_depth = max_depth  # 回归树的最大深度
        self.trees_ = []
        self.initial_prediction_ = None  # 存储初始均值预测 / To store the initial mean prediction
    
    def fit(self, X, y):
        # 训练GBDT模型
        X, y = validate_X_y(X, y, numeric_y=True)
        self.trees_ = []
        self.n_features_ = X.shape[1]
        self.initial_prediction_ = np.mean(y)  # 初始预测为目标均值 / Initial prediction is the mean of the target
        self.y_pred_ = np.full(len(y), self.initial_prediction_, dtype=float)
        self.residuals_ = y - self.y_pred_
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, self.residuals_)
            self.trees_.append(tree)

            # 每棵树都是固定的学习权重
            # XXX learning_rate是不是应该和n_estimators相关
            self.y_pred_ += self.learning_rate * tree.predict(X)
            self.residuals_ = y - self.y_pred_

        return self
    
    def predict(self, X):
        """
        预测新数据 / Predict new data
        参数:
        X: numpy.ndarray, shape (n_samples, n_features) - 输入特征数据 / Input feature data
        返回:
        numpy.ndarray, shape (n_samples,) - 预测的目标值 / Predicted target values
        """
        if self.initial_prediction_ is None:
            raise ValueError("模型尚未训练，请先调用 fit。")
        X = validate_X(X, n_features=self.n_features_, allow_1d=True)
        predictions = np.full(X.shape[0], self.initial_prediction_, dtype=float)
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
