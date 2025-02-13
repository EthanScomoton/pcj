import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

# 生成虚拟时间序列数据（使用正弦波+随机噪声作为例子）
np.random.seed(42)
n = 3000
t = np.linspace(0, 50, n)
y = 10 * np.sin(0.2 * t) + np.random.normal(scale=2, size=n)

# 拆分训练和测试集
train_size = int(len(y) * 0.8)
train, test = y[:train_size], y[train_size:]

# 拟合自回归模型
model = AutoReg(train, lags=5)
model_fit = model.fit()

# 预测
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# 创建绘图
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# 图1：原始时间序列数据
ax[0, 0].plot(t, y, label='Time Series Data', color='red')
ax[0, 0].set_title('Original Time Series Data', fontsize=12)
ax[0, 0].set_xlabel('Time')
ax[0, 0].set_ylabel('Value')
ax[0, 0].legend()

# 图2：自相关函数 (ACF)
plot_acf(y, lags=30, ax=ax[0, 1], color='blue')
ax[0, 1].set_title('Autocorrelation Function (ACF)', fontsize=12)

# 图3：偏自相关函数 (PACF)
plot_pacf(y, lags=30, ax=ax[1, 0], color='green')
ax[1, 0].set_title('Partial Autocorrelation Function (PACF)', fontsize=12)

# 图4：模型残差及其自相关性
residuals = test - predictions
ax[1, 1].plot(residuals, label='Residuals', color='purple')
ax[1, 1].set_title('Residuals Time Series', fontsize=12)
ax[1, 1].set_xlabel('Time')
ax[1, 1].legend()

# 调整布局
plt.tight_layout()
plt.show()
