import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm  # 进度条库，用于显示进度条
from scipy.stats import weibull_min

# 设置参数
time_steps = 1000  # 时间步数
num_features = 6
num_classes = 2   # 类别数量

# 超参数
learning_rate = 1e-3  # 学习率
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
        # 卷积层和归一化层
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(num_features=256)

        # BiGRU层
        self.bigru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        # 注意力机制层
        self.attention = Attention(input_dim=256)

        # 全连接层和批归一化层
        self.fc1 = nn.Linear(256, 64)
        self.bn_fc1 = nn.BatchNorm1d(num_features=64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x形状: (batch_size, time_steps, features)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, features, time_steps)

        # 卷积层1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 卷积层2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)

        # 卷积层3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 卷积层4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)

        # 卷积层5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        # 转换为 (batch_size, seq_length, features)
        x = x.permute(0, 2, 1)

        # 双向GRU层
        gru_out, _ = self.bigru(x)

        # 注意力机制
        attn_out = self.attention(gru_out)

        # 全连接层
        x = self.fc1(attn_out)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

np.random.seed(42)  # 固定随机数种子，确保结果可复现
num_samples = 2000  # 样本数量


def generate_solar_power(time_steps, num_samples, latitude=30):
    dt = 0.1  # 时间步长（小时）
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
    dt = 1  # 时间步长 (小时)
    t = np.arange(0, time_steps) * dt  # 每个时间步对应的时间（小时）
    wind_power = np.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        # 生成符合Weibull分布的风速
        np.random.seed(sample)  # 随机数种子，确保每个样本的风速不同
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
    t = np.linspace(0, 24, time_steps)  # 一天中的时间点
    base_demand = 500  # 平均能源需求基线 (kW)
    demand_variation = 200  # 能源需求波动 (kW)
    energy_demand = np.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        # 基于正弦波模拟一天的需求变化，并添加随机扰动
        daily_demand = base_demand + demand_variation * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 50, size=time_steps)
        daily_demand = np.clip(daily_demand, 0, None)  # 需求不能为负
        energy_demand[sample, :] = daily_demand
    
    return energy_demand


def generate_storage_power(time_steps, num_samples, E_max=50000, P_charge_max=1000, P_discharge_max=1000):
    storage_power = np.zeros((num_samples, time_steps))
    E_storage = np.zeros((num_samples, time_steps))  # 储能状态 (kWh)

    for sample in range(num_samples):
        E_storage[sample, 0] = E_max * 0.5  # 初始储能水平为 50%
        for t in range(1, time_steps):
            # 随机决定充电或放电
            if np.random.rand() > 0.5:
                # 充电
                charge_power = min(P_charge_max, E_max - E_storage[sample, t-1])
                E_storage[sample, t] = E_storage[sample, t-1] + charge_power
                storage_power[sample, t] = charge_power
            else:
                # 放电
                discharge_power = min(P_discharge_max, E_storage[sample, t-1])
                E_storage[sample, t] = E_storage[sample, t-1] - discharge_power
                storage_power[sample, t] = -discharge_power

    return storage_power


def generate_grid_power(time_steps, num_samples, energy_demand, renewable_power, storage_power):
    grid_power = np.zeros((num_samples, time_steps))
    
    for sample in range(num_samples):
        for t in range(time_steps):
            remaining_demand = energy_demand[sample, t] - renewable_power[sample, t] - storage_power[sample, t]
            grid_power[sample, t] = max(0, remaining_demand)  # 电网只提供正向功率
    
    return grid_power


# 生成模数据
solar_power = generate_solar_power(time_steps, num_samples, latitude=30)
wind_power = generate_wind_power(time_steps, num_samples, k_weibull=2, c_weibull=8, v_in=5, v_rated=8, v_out=12, P_wind_rated=1000, N_wind_turbine=3)
renewable_power = solar_power + wind_power
energy_demand = generate_energy_demand(time_steps, num_samples)
storage_power = generate_storage_power(time_steps, num_samples)
grid_power = generate_grid_power(time_steps, num_samples, energy_demand, renewable_power, storage_power)

# 将所有特征组合成输入数据
inputs = np.stack([solar_power, wind_power, storage_power, grid_power, energy_demand, renewable_power], axis=2)
inputs = torch.tensor(inputs, dtype=torch.float32)

# 生成随机标签2
labels = torch.randint(0, num_classes, (num_samples,))

# 创建数据集和数据加载器
dataset = TensorDataset(inputs, labels)

# 创建数据加载器（优化 num_workers 和 pin_memory）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 首选GPU，也可CPU
model = MyModel(num_features=num_features, num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 定义学习率调度器（线性预热 + 余弦退火）
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)  # 前10%步骤用于热身

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return max(0.0, 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps))))

scheduler = LambdaLR(optimizer, lr_lambda)

# 使用自动混合精度
scaler = torch.cuda.amp.GradScaler()

# 早停机制参数
patience = 5  # 在验证集上若干个周期无提升则停止
best_val_loss = float('inf')
counter = 0

# TensorBoard
writer = SummaryWriter(log_dir='runs/experiment1')

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            num_samples += inputs.size(0)

    val_loss = running_loss / num_samples
    val_acc = running_corrects.double() / num_samples
    return val_loss, val_acc

scheduler = LambdaLR(optimizer, lr_lambda)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0

    # 使用 tqdm 包装训练集数据加载器
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_inputs, batch_labels in progress_bar:
        # 将数据移动到设备，并使用 non_blocking=True
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_size = batch_inputs.size(0)
        num_samples += batch_size

        # 提前零梯度
        optimizer.zero_grad()

        # 前向传播（使用自动混合精度）
        with torch.cuda.amp.autocast():
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

        # 反向传播和优化
        scaler.scale(loss).backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数和学习率
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # 计算损失和准确率
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == batch_labels)

        # 更新进度条描述
        progress_bar.set_postfix({'Loss': running_loss / num_samples, 'Acc': (running_corrects.double() / num_samples).item()})

    # 计算整个 epoch 的平均损失和准确率
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects.double() / num_samples

    # 在 TensorBoard 中记录训练指标
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

    # 在验证集上评估
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # 在 TensorBoard 中记录验证指标
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/Accuracy', val_acc, epoch)

    # 打印结果
    print(f'\nEpoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 早停机制和保存最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # 保存最优模型
        torch.save(model.state_dict(), 'best_model.pth')
        print("模型已保存！")
    else:
        counter += 1
        if counter >= patience:
            print("验证集损失未降低，提前停止训练")
            break

# 关闭 TensorBoard
writer.close()
print('训练完成。')