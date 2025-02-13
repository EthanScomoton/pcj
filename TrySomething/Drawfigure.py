import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据，假设为相关系数矩阵
data = {
    'Cooling': [1, 0.8, 0.6, 0.7, -0.4],
    'Heating': [0.8, 1, 0.7, 0.6, -0.3],
    'Power': [0.6, 0.7, 1, 0.9, -0.2],
    'DBT': [0.7, 0.6, 0.9, 1, -0.1],
    'RH': [-0.4, -0.3, -0.2, -0.1, 1]
}

df = pd.DataFrame(data, index=['Cooling', 'Heating', 'Power', 'DBT', 'RH'])

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'}, linewidths=0.5)

plt.title("Heat map of different types of loads and external factors")
plt.show()
