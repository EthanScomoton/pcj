import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as numpy  
from tqdm import tqdm  # 进度条库，用于显示进度条
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

# 设置参数
num_samples = 365  # 样本数量 (天数)
time_steps = 1000  # 时间步数
num_features = 6
num_classes = 2   # 类别数量

# 超参数
learning_rate = 1e-4  # 学习率
num_epochs = 50        # 训练轮数
batch_size = 64        # 批次大小
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

        # 第一组卷积层
        self.conv1 = nn.Conv1d(in_channels=num_features - 2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 第二组卷积层
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 第三组卷积层
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # BiGRU层
        self.bigru = nn.GRU(input_size=256, hidden_size=256, batch_first=True, bidirectional=True)

        # 注意力机制层 (用于风能和光伏的预测)
        self.attention = Attention(input_dim=2)  # Wind and solar data have 2 features
        self.fc_attn = nn.Linear(2, 512)         # Transform attention output to match BiGRU output

        # 全连接层和归一化层
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 + 512, 128)  # 调整为 512 + 512
        self.bn_fc1 = nn.BatchNorm1d(num_features=128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, seq_length, num_features)
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_length)

        # 分离输入中的风能和光伏数据以及储能和用能设备的数据
        wind_solar_input = x[:, :2, :]    # (batch_size, 2, seq_length)
        storage_demand_input = x[:, 2:, :]  # (batch_size, num_features - 2, seq_length)

        # 调整 wind_solar_input 的形状供注意力机制使用
        wind_solar_input = wind_solar_input.permute(0, 2, 1)  # (batch_size, seq_length, 2)

        # 第一组卷积层
        x = self.conv1(storage_demand_input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)  # (batch_size, 64, seq_length/2)

        # 第二组卷积层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)  # (batch_size, 128, seq_length/4)

        # 第三组卷积层
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool3(x)  # (batch_size, 256, seq_length/8)

        # BiGRU 层
        x = x.permute(0, 2, 1)  # (batch_size, seq_length/8, 256)
        r_out, _ = self.bigru(x)  # (batch_size, seq_length/8, 512)

        # 注意力机制
        attn_output = self.attention(wind_solar_input)  # (batch_size, 2)
        attn_output = self.fc_attn(attn_output)         # (batch_size, 512)

        # 将注意力机制和 BiGRU 的输出拼接在一起
        combined_output = torch.cat((attn_output, r_out[:, -1, :]), dim=1)  # (batch_size, 512 + 512)

        # 全连接层
        x = self.dropout(combined_output)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # 输出 x 的形状: (batch_size, num_classes)
    

def generate_solar_power(time_steps, latitude=30):
    dt = 0.1  # 时间步长（小时）
    t_per_day = numpy.arange(0, time_steps) * dt  # 每天的时间步（小时）
    P_solar_one = 0.4  # 单块最大功率（kW）
    P_solar_Panel = 200  # 光伏面板数量
    P_solar_max = P_solar_one * P_solar_Panel  # 光伏系统的最大功率输出
    days_in_year = 365  # 一年中的天数
    solar_power = numpy.zeros((days_in_year, time_steps))
    
    for day in range(1, days_in_year + 1):
        day_of_year = day  # 当前是第几天
        # 计算太阳赤纬角
        declination = 23.45 * numpy.sin(numpy.deg2rad(360 * (284 + day_of_year) / 365))
        # 计算日角
        hour_angle = numpy.degrees(numpy.arccos(-numpy.tan(numpy.deg2rad(latitude)) * numpy.tan(numpy.deg2rad(declination))))
        # 计算日出和日落时间
        sunrise = 12 - hour_angle / 15  # 日出时间（小时）
        sunset = 12 + hour_angle / 15   # 日落时间（小时）
        sunrise = max(0, sunrise)
        sunset = min(24, sunset)
        
        P_solar = numpy.zeros_like(t_per_day)
        for i in range(len(t_per_day)):
            current_time = t_per_day[i] % 24  # 当前时间（小时）
            if sunrise <= current_time <= sunset:
                P_solar[i] = P_solar_max * numpy.sin(numpy.pi * (current_time - sunrise) / (sunset - sunrise))
        P_solar[P_solar < 0] = 0
        E_solar = P_solar * dt
        solar_power[day - 1, :] = E_solar

    return solar_power


def generate_wind_power(time_steps, num_samples, k_weibull=2, c_weibull=8, v_in=5, v_rated=8, v_out=12, P_wind_rated=1000, N_wind_turbine=3):
    dt = 1  # 时间步长 (小时)
    t = numpy.arange(0, time_steps) * dt  # 每个时间步对应的时间（小时）
    wind_power = numpy.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        # 生成符合Weibull分布的风速
        numpy.random.seed(sample)  # 随机数种子，确保每个样本的风速不同
        v_wind = weibull_min.rvs(k_weibull, scale=c_weibull, size=len(t))
        P_wind = numpy.zeros_like(t)
        
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
    t = numpy.linspace(0, 24, time_steps)  # 一天中的时间点
    base_demand = 500  # 平均能源需求基线 (kW)
    demand_variation = 200  # 能源需求波动 (kW)
    energy_demand = numpy.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        # 基于正弦波模拟一天的需求变化，并添加随机扰动
        daily_demand = base_demand + demand_variation * numpy.sin(2 * numpy.pi * t / 24) + numpy.random.normal(0, 50, size=time_steps)
        daily_demand = numpy.clip(daily_demand, 0, None)  # 需求不能为负
        energy_demand[sample, :] = daily_demand
    
    return energy_demand


def generate_storage_power(time_steps, num_samples, E_max=50000, P_charge_max=1000, P_discharge_max=1000, target_soc=0.8, soc_tolerance=0.05):
    storage_power = numpy.zeros((num_samples, time_steps))
    E_storage = numpy.zeros((num_samples, time_steps))  # 储能状态 (kWh)

    for sample in range(num_samples):
        E_storage[sample, 0] = E_max * target_soc  # 初始储能水平为80%

        for t in range(1, time_steps):
            current_soc = E_storage[sample, t-1] / E_max  # 当前的SOC

            # 如果当前SOC低于目标SOC且电网有多余电力，充电
            if current_soc < (target_soc - soc_tolerance):
                charge_power = min(P_charge_max, E_max - E_storage[sample, t-1])
                E_storage[sample, t] = E_storage[sample, t-1] + charge_power
                storage_power[sample, t] = charge_power
            # 如果当前SOC高于目标SOC且有多余的可再生能源，放电
            elif current_soc > (target_soc + soc_tolerance):
                discharge_power = min(P_discharge_max, E_storage[sample, t-1])
                E_storage[sample, t] = E_storage[sample, t-1] - discharge_power
                storage_power[sample, t] = -discharge_power
            # 否则，保持当前状态
            else:
                E_storage[sample, t] = E_storage[sample, t-1]
                storage_power[sample, t] = 0

    return storage_power, E_storage


def generate_grid_power(time_steps, num_samples, energy_demand, renewable_power, storage_power, E_storage, E_max, target_soc=0.8, soc_tolerance=0.05, P_charge_max=1000):
    grid_power = numpy.zeros((num_samples, time_steps))

    for sample in range(num_samples):
        for t in range(time_steps):
            remaining_demand = energy_demand[sample, t] - renewable_power[sample, t] - storage_power[sample, t]
            
            # 如果有剩余需求，电网补给供电
            if remaining_demand > 0:
                grid_power[sample, t] = remaining_demand  # 电网补充不足的部分
            else:
                grid_power[sample, t] = 0

            # 当电力需求较低且储能设备SOC低于目标SOC时，电网为储能设备充电
            current_soc = E_storage[sample, t] / E_max
            if current_soc < (target_soc - soc_tolerance) and remaining_demand <= 0:
                charge_power = min(P_charge_max, E_max - E_storage[sample, t])
                grid_power[sample, t] += charge_power  # 电网为储能设备充电
                E_storage[sample, t] += charge_power  # 更新储能设备SOC

    return grid_power


# 生成模数据
solar_power = generate_solar_power(time_steps, latitude=30)
wind_power = generate_wind_power(time_steps, num_samples)
renewable_power = solar_power + wind_power
energy_demand = generate_energy_demand(time_steps, num_samples)
storage_power, E_storage = generate_storage_power(time_steps, num_samples)
grid_power = generate_grid_power(time_steps, num_samples, energy_demand, renewable_power, storage_power, E_storage, E_max=50000)

# 将所有特征组合成输入数据
inputs = numpy.stack([solar_power, wind_power, storage_power, grid_power, energy_demand, renewable_power], axis=2)
inputs = torch.tensor(inputs, dtype=torch.float32)

# 生成随机标签
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
if device.type == 'cuda':
    scaler = torch.cuda.amp.GradScaler()  # 仅在 CUDA 可用时启用 GradScaler
else:
    scaler = None  # 如果没有 CUDA，不需要使用 GradScaler

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

train_acc_history = []
val_acc_history = []

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

        # 前向传播和反向传播（根据是否使用 CUDA 选择性使用自动混合精度）
        if scaler:
            # 如果使用 GPU 和 GradScaler
            with torch.cuda.amp.autocast():
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 如果是 CPU
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()

        # 更新学习率
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

    # 记录训练和验证准确率
    train_acc_history.append(epoch_acc.cpu().item())
    val_acc_history.append(val_acc.cpu().item())

    # 打印结果
    print(f'\nEpoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 早停机制和保存最优模型
    #if val_loss < best_val_loss:
        #best_val_loss = val_loss
        #counter = 0
        # 保存最优模型
        #torch.save(model.state_dict(), 'best_model.pth')
        #print("模型已保存！")
    #else:
        #counter += 1
        #if counter >= patience:
            #print("验证集损失未降低，提前停止训练")
            #break

# 绘制训练和验证集准确率的折线图
def plot_accuracy(train_acc_history, val_acc_history):
    epochs = range(1, len(train_acc_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_history, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_acc_history, label='Validation Accuracy', marker='o')

    plt.title('Train and Validation Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# 在训练完成后调用此函数
plot_accuracy(train_acc_history, val_acc_history)

# 关闭 TensorBoard
writer.close()
print('训练完成。')