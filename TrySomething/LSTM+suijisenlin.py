import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import seaborn as sns

# 生成虚拟气象数据
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=500, freq="D")
temperature = 10 + 10 * np.sin(2 * np.pi * dates.dayofyear / 500) + np.random.normal(0, 2, 500)

data = pd.DataFrame({"Date": dates, "Temperature": temperature})

# 数据可视化（图1：原始时间序列）
plt.figure(figsize=(10,6))
sns.lineplot(data=data,x="Date",y="Temperature",color="blue")
plt.title("Synthetic TemperatureTime Series",fontsize=16)
plt.xlabel("Date",fontsize=12)
plt.ylabel("Temperature(°C)",fontsize=12)
plt.grid()
plt.show()

# 数据集准备
class WeatherDataset(Dataset):
    def __init__(self, data, sequence_length=30):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

sequence_length = 30
data_values = data["Temperature"].values
dataset = WeatherDataset(data_values, sequence_length=sequence_length)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

lstm_model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# 训练LSTM并记录损失值（图2：训练损失曲线）
epochs = 20
losses = []
for epoch in range(epochs):
    epoch_loss = 0
    lstm_model.train()
    for x, y in dataloader:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        optimizer.zero_grad()
        outputs = lstm_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(dataloader))

# 可视化LSTM训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1),losses,marker="o",color="green")
plt.title("LSTM Training Loss Curve",fontsize=16)
plt.xlabel("Epoch",fontsize=12)
plt.ylabel("MSE Loss",fontsize=12)
plt.grid()
plt.show()

# LSTM特征提取
lstm_model.eval()
lstm_features = []
for i in range(len(data_values) - sequence_length):
    x = torch.tensor(data_values[i:i + sequence_length], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        lstm_features.append(lstm_model(x).item())

# 随机森林训练
lstm_features = np.array(lstm_features).reshape(-1, 1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
train_size = int(len(lstm_features) * 0.8)
rf_model.fit(lstm_features[:train_size], data_values[sequence_length:sequence_length + train_size])

# 随机森林单独预测效果（图3）
rf_predictions_train = rf_model.predict(lstm_features[:train_size])
actual_train = data_values[sequence_length:sequence_length + train_size]

plt.figure(figsize=(12, 8))
plt.plot(range(len(actual_train)), actual_train, label="Actual (Train)", color="blue")
plt.plot(range(len(rf_predictions_train)), rf_predictions_train, label="RF Predicted (Train)", color="orange")
plt.title("Random Forest Training Predictions", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend()
plt.grid()
plt.show()

# 混合模型预测效果（图4）
rf_predictions = rf_model.predict(lstm_features[train_size:])
actual = data_values[sequence_length + train_size:]

plt.figure(figsize=(12, 8))
plt.plot(range(len(actual)), actual, label="Actual (Test)", color="blue")
plt.plot(range(len(rf_predictions)), rf_predictions, label="Predicted (LSTM + RF)", color="red")
plt.title("LSTM + Random Forest Predictions", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend()
plt.grid()
plt.show()