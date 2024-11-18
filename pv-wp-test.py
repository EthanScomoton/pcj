import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

# 定义输入数据维度参数
num_samples = 365         # 样本数量 (天数)
time_steps = 1000         # 每个样本的时间步长
num_features = 6          # 输入特征的数量
num_classes = 2           # 类别数量：二分类问题

# 超参数
learning_rate = 1e-4
num_epochs = 50
batch_size = 64
weight_decay = 1e-4

# 自定义注意力机制模块
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        attn_weights = self.attention(x)  # (batch_size, seq_length, 1)
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=1)  # 对时间步进行softmax
        weighted = x * attn_weights  # 加权输入
        output = torch.sum(weighted, dim=1)  # 对序列维度求和，得到整体表示
        return output

# 定义模型
class MyModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MyModel, self).__init__()

        # 卷积层用于处理储能设备和用能设备的特征
        self.conv1 = nn.Conv1d(in_channels=num_features-2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # BiGRU层用于储能设备和用能设备的协同运行
        self.bigru = nn.GRU(input_size=256, hidden_size=256, batch_first=True, bidirectional=True)

        # 注意力机制用于风能和光伏的预测
        self.attention = Attention(input_dim=2)  # 风能和光伏数据有2个维度

        # 全连接层和归一化层
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 + 256, 128)  # Attention输出(512) + BiGRU输出(256)
        self.bn_fc1 = nn.BatchNorm1d(num_features=128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, time_steps, num_features)
        wind_solar_input = x[:, :, :2]  # 前两个特征是风能和光伏
        storage_demand_input = x[:, :, 2:]  # 剩下的特征是储能设备和用能设备

        # 处理储能和用能设备数据
        storage_demand_input = storage_demand_input.permute(0, 2, 1)  # 调整为 (batch_size, num_features-2, time_steps)
        x = self.conv1(storage_demand_input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool3(x)

        # 调整形状以适应GRU输入
        x = x.permute(0, 2, 1)  # 调整为 (batch_size, seq_length/8, 256)
        r_out, _ = self.bigru(x)  # r_out 形状: (batch_size, seq_length/8, 512)

        # 注意力机制处理风能和光伏数据
        attn_output = self.attention(wind_solar_input)  # attn_output 形状: (batch_size, 512)

        # 拼接Attention输出和BiGRU输出
        combined_output = torch.cat((attn_output, r_out[:, -1, :]), dim=1)  # (batch_size, 512 + 256)

        # 全连接层
        x = self.dropout(combined_output)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 生成数据的函数
def generate_solar_power(time_steps, latitude=30):
    dt = 0.1  # 时间步长
    t_per_day = np.arange(0, time_steps) * dt
    P_solar_one = 0.4  # 单块最大功率
    P_solar_Panel = 200
    P_solar_max = P_solar_one * P_solar_Panel
    days_in_year = 365
    solar_power = np.zeros((days_in_year, time_steps))
    
    for day in range(1, days_in_year + 1):
        day_of_year = day
        declination = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
        hour_angle = np.degrees(np.arccos(-np.tan(np.deg2rad(latitude)) * np.tan(np.deg2rad(declination))))
        sunrise = max(0, 12 - hour_angle / 15)
        sunset = min(24, 12 + hour_angle / 15)
        
        P_solar = np.zeros_like(t_per_day)
        for i in range(len(t_per_day)):
            current_time = t_per_day[i] % 24
            if sunrise <= current_time <= sunset:
                P_solar[i] = P_solar_max * np.sin(np.pi * (current_time - sunrise) / (sunset - sunrise))
        P_solar[P_solar < 0] = 0
        E_solar = P_solar * dt
        solar_power[day - 1, :] = E_solar

    return solar_power


def generate_wind_power(time_steps, num_samples, k_weibull=2, c_weibull=8, v_in=5, v_rated=8, v_out=12, P_wind_rated=1000, N_wind_turbine=3):
    dt = 1
    t = np.arange(0, time_steps) * dt
    wind_power = np.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        np.random.seed(sample)
        v_wind = weibull_min.rvs(k_weibull, scale=c_weibull, size=len(t))
        P_wind = np.zeros_like(t)
        
        for i in range(len(v_wind)):
            v = v_wind[i]
            if v < v_in or v >= v_out:
                P_wind[i] = 0
            elif v_in <= v < v_rated:
                P_wind[i] = P_wind_rated * ((v - v_in) / (v_rated - v_in)) ** 3
            else:
                P_wind[i] = P_wind_rated
        P_wind *= N_wind_turbine
        E_wind = P_wind * dt
        wind_power[sample, :] = E_wind    

    return wind_power


def generate_energy_demand(time_steps, num_samples):
    t = np.linspace(0, 24, time_steps)
    base_demand = 500
    demand_variation = 200
    energy_demand = np.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        daily_demand = base_demand + demand_variation * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 50, size=time_steps)
        daily_demand = np.clip(daily_demand, 0, None)
        energy_demand[sample, :] = daily_demand
    
    return energy_demand


def generate_storage_power(time_steps, num_samples, E_max=50000, P_charge_max=1000, P_discharge_max=1000, target_soc=0.8, soc_tolerance=0.05):
    storage_power = np.zeros((num_samples, time_steps))
    E_storage = np.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        E_storage[sample, 0] = E_max * target_soc

        for t in range(1, time_steps):
            current_soc = E_storage[sample, t-1] / E_max

            if current_soc < (target_soc - soc_tolerance):
                charge_power = min(P_charge_max, E_max - E_storage[sample, t-1])
                E_storage[sample, t] = E_storage[sample, t-1] + charge_power
                storage_power[sample, t] = charge_power
            elif current_soc > (target_soc + soc_tolerance):
                discharge_power = min(P_discharge_max, E_storage[sample, t-1])
                E_storage[sample, t] = E_storage[sample, t-1] - discharge_power
                storage_power[sample, t] = -discharge_power
            else:
                E_storage[sample, t] = E_storage[sample, t-1]
                storage_power[sample, t] = 0

    return storage_power, E_storage

def generate_grid_power(time_steps, num_samples, energy_demand, renewable_power, storage_power, E_storage, E_max, target_soc=0.8, soc_tolerance=0.05, P_charge_max=1000):
    grid_power = np.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        for t in range(time_steps):
            remaining_demand = energy_demand[sample, t] - renewable_power[sample, t] - storage_power[sample, t]
            
            if remaining_demand > 0:
                grid_power[sample, t] = remaining_demand
            else:
                grid_power[sample, t] = 0

            current_soc = E_storage[sample, t] / E_max
            if current_soc < (target_soc - soc_tolerance) and remaining_demand <= 0:
                charge_power = min(P_charge_max, E_max - E_storage[sample, t])
                grid_power[sample, t] += charge_power
                E_storage[sample, t] += charge_power

    return grid_power


# 生成模拟数据
solar_power = generate_solar_power(time_steps, latitude=30)
wind_power = generate_wind_power(time_steps, num_samples)
renewable_power = solar_power + wind_power
energy_demand = generate_energy_demand(time_steps, num_samples)
storage_power, E_storage = generate_storage_power(time_steps, num_samples)
grid_power = generate_grid_power(time_steps, num_samples, energy_demand, renewable_power, storage_power, E_storage, E_max=50000)

# 将所有特征组合成输入数据
inputs = np.stack([solar_power, wind_power, storage_power, grid_power, energy_demand, renewable_power], axis=2)
inputs = torch.tensor(inputs, dtype=torch.float32)

# 生成随机标签
labels = torch.randint(0, num_classes, (num_samples,))

# 创建数据集和数据加载器
dataset = TensorDataset(inputs, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel(num_features=num_features, num_classes=num_classes)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 学习率调度器
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return max(0.0, 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps))))

scheduler = LambdaLR(optimizer, lr_lambda)

# 训练和验证模型
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_inputs, batch_labels in progress_bar:
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_size = batch_inputs.size(0)
        num_samples += batch_size

        optimizer.zero_grad()

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        scheduler.step()

        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == batch_labels)

        progress_bar.set_postfix({'Loss': running_loss / num_samples, 'Acc': (running_corrects.double() / num_samples).item()})

    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects.double() / num_samples
    train_acc_history.append(epoch_acc.cpu().item())

    # 验证
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_samples = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs = val_inputs.to(device, non_blocking=True)
            val_labels = val_labels.to(device, non_blocking=True)
            val_size = val_inputs.size(0)
            val_samples += val_size

            outputs = model(val_inputs)
            loss = criterion(outputs, val_labels)

            val_loss += loss.item() * val_size
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == val_labels)

    val_loss /= val_samples
    val_acc = val_corrects.double() / val_samples
    val_acc_history.append(val_acc.cpu().item())

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# 训练完成后绘制准确率曲线
def plot_accuracy(train_acc_history, val_acc_history):
    epochs = range(1, len(train_acc_history) + 1)
    plt.plot(epochs, train_acc_history, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc_history, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_accuracy(train_acc_history, val_acc_history)