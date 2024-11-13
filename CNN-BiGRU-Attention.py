import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import weibull_min

# 设置参数
time_steps = 1000  # 时间步数
num_features = 5   # 特征数 (光伏、风能、储能、电网供电、港口能源需求)
num_classes = 10   # 类别数量

# 超参数
learning_rate = 0.001  # 学习率
num_epochs = 20        # 训练轮数
batch_size = 32        # 批次大小
weight_decay = 1e-4    # L2正则化防止过拟合

# 自定义注意力机制模块
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        attn_weights = self.attention(x)  # (batch_size, seq_length, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # 对时间步进行softmax
        weighted = x * attn_weights  # 加权输入
        output = torch.sum(weighted, dim=1)  # 对序列维度求和，得到整体表示
        return output

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Conv1d卷积层 + 池化层
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # BiGRU层
        self.bigru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        # 注意力机制
        self.attention = Attention(input_dim=256)

        # 全连接层
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x形状: (batch_size, time_steps, features)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, features, time_steps)

        # 卷积 + 池化层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))

        # 转换为 (batch_size, seq_length, features)
        x = x.permute(0, 2, 1)

        # BiGRU层
        gru_out, _ = self.bigru(x)

        # 注意力机制
        attn_out = self.attention(gru_out)

        # 全连接层
        x = F.relu(self.fc1(attn_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 输入特征：光伏发电、风能发电、储能电量、电网供电、港口能源需求
np.random.seed(42)  # 固定随机数种子，确保结果可复现
num_samples = 2000  # 样本数量


def generate_solar_power(time_steps, num_samples, latitude=30):

    dt = 1  # 时间步长（小时）
    t = np.arange(0, time_steps) * dt  # 每个时间步对应的时间（小时）
    
    P_solar_one = 0.4  # 单块最大功率（kW）
    P_solar_Panel = 200  # 光伏面板数量
    P_solar_max = P_solar_one * P_solar_Panel  # 光伏系统的最大功率输出

    solar_power = np.zeros((num_samples, time_steps))
    
    for sample in range(num_samples):
        day_of_year = np.random.randint(1, 366) # 随机日期
        declination = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
        hour_angle = np.degrees(np.arccos(-np.tan(np.deg2rad(latitude)) * np.tan(np.deg2rad(declination))))
        sunrise = 12 - hour_angle / 15  # 日出时间（小时）
        sunset = 12 + hour_angle / 15   # 日落时间（小时）
        sunrise = max(0, sunrise)
        sunset = min(24, sunset)
        
        P_solar = np.zeros_like(t)
        for i in range(len(t)):
            current_time = t[i] % 24  # 以24小时为周期
            if sunrise <= current_time <= sunset:
                P_solar[i] = P_solar_max * np.sin(np.pi * (current_time - sunrise) / (sunset - sunrise))
        P_solar[P_solar < 0] = 0
        E_solar = P_solar * dt
        solar_power[sample, :] = E_solar
    return solar_power

def generate_wind_power(time_steps, num_samples, k_weibull=2, c_weibull=8, v_in=5, v_rated=8, v_out=12, P_wind_rated=1000, N_wind_turbine=3):

    # 时间参数
    dt = 1  # 时间步长 (小时)
    t = np.arange(0, time_steps) * dt  # 每个时间步对应的时间（小时）
    
    # 初始化风能发电功率矩阵
    wind_power = np.zeros((num_samples, time_steps))
    
    # 生成每个样本的风速和风能发电功率
    for sample in range(num_samples):
        # 生成符合Weibull分布的风速
        np.random.seed(sample)  # 使用样本编号作为随机数种子，确保每个样本的风速不同
        v_wind = weibull_min.rvs(k_weibull, scale=c_weibull, size=len(t))
        
        # 初始化当前样本的风能发电功率
        P_wind = np.zeros_like(t)
        
        for i in range(len(v_wind)):
            v = v_wind[i]
            if v < v_in or v >= v_out:
                P_wind[i] = 0
            elif v_in <= v < v_rated:
                # 计算风速在启动风速和额定风速之间时的功率
                P_wind[i] = P_wind_rated * ((v - v_in) / (v_rated - v_in)) ** 3
            else:
                # 风速在额定风速到停机风速之间时，输出额定功率
                P_wind[i] = P_wind_rated
        
        # 考虑风电机组数量
        P_wind *= N_wind_turbine
        
        # 计算发电量 (kWh)，假设每个时间步长为dt小时
        E_wind = P_wind * dt
        
        # 将该样本的风能发电数据存储到wind_power矩阵中
        wind_power[sample, :] = E_wind
    
    return wind_power




# 生成模拟数据：随机生成数值，模拟不同能源的发电量与需求
solar_power = generate_solar_power(time_steps, num_samples, latitude=30)    # 光伏发电
wind_power = generate_wind_power(time_steps, num_samples, k_weibull=2, c_weibull=8, v_in=5, v_rated=8, v_out=12, P_wind_rated=1000, N_wind_turbine=3)      # 风能发电
energy_demand = generate_energy_demand(time_steps, num_samples)  # 港口能源需求
storage_power = generate_storage_power(time_steps, num_samples)  # 储能电量
grid_power = generate_grid_power(time_steps, num_samples, energy_demand)  # 电网供电

# 将所有特征组合成输入数据
inputs = np.stack([solar_power, wind_power, storage_power, grid_power, energy_demand], axis=2)
inputs = torch.tensor(inputs, dtype=torch.float32)

# 生成随机标签（假设有10个不同的能源调度策略）
labels = torch.randint(0, num_classes, (num_samples,))

# 创建数据集和数据加载器
dataset = TensorDataset(inputs, labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型实例并移动到GPU（如果可用）
model = MyModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for batch_inputs, batch_labels in data_loader:
        # 将数据移动到设备
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        # 前向传播
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item() * batch_inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == batch_labels.data)

    # 计算平均损失和准确率
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects.double() / num_samples

    print(f'第 {epoch + 1} 个周期，Loss: {epoch_loss:.4f}，Accuracy: {epoch_acc:.4f}')

print('训练完成。')