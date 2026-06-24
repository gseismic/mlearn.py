import numpy as np

from ...metrics import r2_score
from ...utils import validate_X, validate_X_y


class LinearRegression:

    def __init__(self, fit_intercept=True):
        """
        初始化线性回归模型
        
        参数:
        fit_intercept (bool): 是否拟合截距,默认为True
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None  # 将存储系数,shape: (n_features,)
        self.intercept_ = None  # 将存储截距,shape: ()
        self.n_features_in_ = None

    def fit(self, X, y):
        """
        训练线性回归模型
        
        参数:
        X (numpy.ndarray): 输入特征矩阵,shape: (n_samples, n_features)
        y (numpy.ndarray): 目标变量,shape: (n_samples,) 或 (n_samples, 1)
        
        返回:
        self: 训练后的模型实例
        """
        if not isinstance(self.fit_intercept, bool):
            raise TypeError("fit_intercept 必须是布尔值。")
        X, y = validate_X_y(X, y, numeric_y=True)
        self.n_features_in_ = X.shape[1]
        
        if self.fit_intercept:
            # 添加一列1用于拟合截距
            X = np.column_stack((np.ones(X.shape[0]), X))  # shape: (n_samples, n_features + 1)
        
        # 使用 SVD 驱动的最小二乘求解，兼容秩亏和欠定设计矩阵
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        if self.fit_intercept:
            self.intercept_ = theta[0]  # 截距,shape: ()
            self.coef_ = theta[1:]  # 系数,shape: (n_features,)
        else:
            self.intercept_ = None
            self.coef_ = theta  # 系数,shape: (n_features,)

        return self

    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数:
        X (numpy.ndarray): 输入特征矩阵,shape: (n_samples, n_features)
        
        返回:
        numpy.ndarray: 预测结果,shape: (n_samples,)
        """
        if self.coef_ is None:
            raise ValueError("模型尚未训练，请先调用 fit。")
        X = validate_X(
            X,
            n_features=self.n_features_in_,
            allow_1d=True
        )
        if self.fit_intercept:
            # 添加一列1用于计算截距
            X = np.column_stack((np.ones(X.shape[0]), X))  # shape: (n_samples, n_features + 1)
            return X.dot(np.concatenate(([self.intercept_], self.coef_)))  # shape: (n_samples,)
        else:
            return X.dot(self.coef_)  # shape: (n_samples,)

    def score(self, X, y):
        """
        计算模型的R²分数
        
        参数:
        X (numpy.ndarray): 输入特征矩阵,shape: (n_samples, n_features)
        y (numpy.ndarray): 真实标签,shape: (n_samples,) 或 (n_samples, 1)
        
        返回:
        float: R²分数
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
