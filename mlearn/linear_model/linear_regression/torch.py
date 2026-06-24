import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ...metrics import r2_score
from ...utils import validate_X, validate_X_y

class LinearRegression(nn.Module):

    def __init__(self, fit_intercept=True, verbose=0):
        """
        初始化线性回归模型
        
        参数:
        fit_intercept (bool): 是否拟合截距项,默认为True
        """
        super(LinearRegression, self).__init__()
        self.fit_intercept = fit_intercept
        self.linear = None
        self.verbose = verbose
        self.n_features_in_ = None

    def forward(self, x):
        """
        定义模型的前向传播
        
        参数:
        x (torch.Tensor): 输入数据,shape: (batch_size, input_dim)

        返回:
        torch.Tensor: 模型的输出,shape: (batch_size, 1)
        """
        return self.linear(x)

    def fit(self, X, y, optim='sgd', epochs=1000, **optim_kwargs):
        """
        训练模型
        
        参数:
        X (numpy.ndarray): 输入特征,shape: (n_samples, input_dim)
        y (numpy.ndarray): 目标变量,shape: (n_samples,) 或 (n_samples, 1)
        epochs (int): 训练轮数,默认为1000
        lr (float): 学习率,默认为0.01

        返回:
        self: 训练后的模型实例
        """
        if not isinstance(self.fit_intercept, bool):
            raise TypeError("fit_intercept 必须是布尔值。")
        X, y = validate_X_y(X, y, numeric_y=True)
        self.n_features_in_ = X.shape[1]

        # 将numpy数组转换为PyTorch张量
        X = torch.FloatTensor(X)  # shape: (n_samples, input_dim)
        y = torch.FloatTensor(y).view(-1, 1)  # shape: (n_samples, 1)

        # self.linear.weight shape: (1, input_dim)
        # self.linear.bias shape: (1,) (如果fit_intercept为True)
        input_dim = X.shape[1]
        if self.fit_intercept:
            self.linear = nn.Linear(input_dim, 1)  # 包含截距的线性层
        else:
            self.linear = nn.Linear(input_dim, 1, bias=False)  # 不包含截距的线性层

        # 定义损失函数和优化器
        criterion = nn.MSELoss()  # 均方误差损失
        # TODO: 使用自适应优化器
        if optim == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), **optim_kwargs)  # 随机梯度下降优化器
        elif optim == 'lbfgs':
            optimizer = torch.optim.LBFGS(self.parameters(), **optim_kwargs)
        else:
            raise ValueError(f'not supported optimizer `{optim}`')

        for epoch in range(epochs):
            # 前向传播
            outputs = self(X)  # shape: (n_samples, 1)

            if optim == 'lbfgs':
                def closure():
                    optimizer.zero_grad()   # 清空梯度
                    output = self(X)       # 计算预测值
                    loss = criterion(output, y)  # 计算损失
                    loss.backward()         # 反向传播
                    return loss

                optimizer.step(closure)
                loss = closure()
            else:
                # 反向传播和优化
                loss = criterion(outputs, y)  # loss is a scalar
                optimizer.zero_grad()  # 清零梯度
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数

            # 打印损失值
            if self.verbose >= 1:
                if (epoch + 1) % 1 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


        return self

    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数:
        X (numpy.ndarray): 输入特征,shape: (n_samples, input_dim)

        返回:
        numpy.ndarray: 预测结果,shape: (n_samples,)
        """
        if self.linear is None:
            raise ValueError("模型尚未训练，请先调用 fit。")
        X = validate_X(
            X,
            n_features=self.n_features_in_,
            allow_1d=True
        )
        X = torch.FloatTensor(X)  # shape: (n_samples, input_dim)
        with torch.no_grad():  # 不计算梯度
            return self(X).numpy().flatten()  # 输出 shape: (n_samples,)

    def score(self, X, y):
        """
        计算模型的R²分数
        
        参数:
        X (numpy.ndarray): 输入特征,shape: (n_samples, input_dim)
        y (numpy.ndarray): 真实标签,shape: (n_samples,) 或 (n_samples, 1)

        返回:
        float: R²分数
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    @property
    def coef_(self):
        """
        获取模型的系数
        
        返回:
        numpy.ndarray: 模型系数,shape: (input_dim,)
        """
        return self.linear.weight.detach().numpy().flatten()

    @property
    def intercept_(self):
        """
        获取模型的截距
        
        返回:
        float: 模型截距,如果没有拟合截距则返回None
        """
        return self.linear.bias.item() if self.fit_intercept else None
