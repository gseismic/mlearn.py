import numpy as np


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

    def fit(self, X, y):
        """
        训练线性回归模型
        
        参数:
        X (numpy.ndarray): 输入特征矩阵,shape: (n_samples, n_features)
        y (numpy.ndarray): 目标变量,shape: (n_samples,) 或 (n_samples, 1)
        
        返回:
        self: 训练后的模型实例
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X 必须是二维特征矩阵。")
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        elif y.ndim != 1:
            raise ValueError("y 必须是一维数组或单列二维数组。")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 和 y 的样本数量必须相同。")
        
        if self.fit_intercept:
            # 添加一列1用于拟合截距
            X = np.column_stack((np.ones(X.shape[0]), X))  # shape: (n_samples, n_features + 1)
        
        # 使用 SVD 驱动的最小二乘求解，兼容秩亏和欠定设计矩阵
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        if self.fit_intercept:
            self.intercept_ = theta[0]  # 截距,shape: ()
            self.coef_ = theta[1:]  # 系数,shape: (n_features,)
        else:
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
        y_pred = self.predict(X)  # shape: (n_samples,)
        
        # 确保y是一维数组
        if y.ndim == 2:
            y = y.ravel()  # shape: (n_samples,)
        
        u = ((y - y_pred) ** 2).sum()  # 残差平方和,shape: ()
        v = ((y - y.mean()) ** 2).sum()  # 总离差平方和,shape: ()
        return 1 - u/v  # R²分数,shape: ()
