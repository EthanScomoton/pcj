import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 生成虚拟时间序列数据（带噪声的）
np.random.seed(42)
n = 3000
time = np.arange(n)
noise = np.random.normal(0, 1, n)
data = np.sin(0.2 * time) + noise  # 原始信号 + 噪声

# 将时间序列数据保存到DataFrame中
df = pd.DataFrame({'time': time, 'value': data})

# 构建并拟合 MA(2) 模型
model = ARIMA(df['value'], order=(0, 0, 2))
model_fit = model.fit()

# 预测值
df['predicted'] = model_fit.fittedvalues

# 计算残差
df['residuals'] = df['value'] - df['predicted']

# 设置图形颜色和样式
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 原始数据与预测数据对比
axes[0, 0].plot(df['time'], df['value'], color='blue', label='Original Data', linewidth=2)
axes[0, 0].plot(df['time'], df['predicted'], color='red', linestyle='--', label='MA(2) Predicted', linewidth=2)
axes[0, 0].set_title('Original vs Predicted')
axes[0, 0].legend(loc='upper right')

# 图2: 残差图
axes[0, 1].plot(df['time'], df['residuals'], color='green', label='Residuals', linewidth=2)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_title('Residuals Analysis')
axes[0, 1].legend(loc='upper right')

# 图3: ACF (自相关函数) 图
plot_acf(df['residuals'], lags=20, ax=axes[1, 0], color='magenta')
axes[1, 0].set_title('ACF of Residuals')

# 图4: PACF (偏自相关函数) 图
plot_pacf(df['residuals'], lags=20, ax=axes[1, 1], color='orange')
axes[1, 1].set_title('PACF of Residuals')

# 调整布局
plt.tight_layout()
plt.show()
