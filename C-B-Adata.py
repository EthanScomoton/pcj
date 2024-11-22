import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm  # 进度条库，用于显示进度条
import matplotlib.pyplot as plt
import os

# 设置参数
num_classes = 2   # 类别数量

# 超参数
learning_rate = 1e-4   # 学习率
num_epochs = 50        # 训练轮数
batch_size = 64        # 批次大小
weight_decay = 1e-4    # L2正则化防止过拟合

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # 检查CUDA是否可用


# 数据读取与预处理,数据保存在 data/ 目录下的 CSV 文件中
def load_and_preprocess_data():
    # 读取可再生能源数据，包括影响因素
    renewable_df = pd.read_csv('data/renewable_data.csv')   
    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])

    # 读取负荷数据，包括影响因素
    load_df = pd.read_csv('data/load_data.csv')   
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')  # 根据时间戳合并数据，inner内连接表示时间戳都存在的列才会被保留

    # 对分类特征进行独热编码，将分类特征转换为数值特征
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['season', 'holiday', 'weather', 'temperature', 'hour', 'ship_grade', 'work_time', 'dock_position']   #Total features
    encoder = OneHotEncoder(sparse=False) #结果将以密集矩阵的形式返回
    encoded_features = encoder.fit_transform(data_df[categorical_features])   #fit：分析 categorical_features 的所有可能值（即分类的类别）。transform：将这些分类特征转换为独热编码格式。encoded_features：返回一个 NumPy 数组，表示独热编码后的特征矩阵。

    encoded_feature_names = encoder.get_feature_names_out(categorical_features)  # 获取新特征的列名 

    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)  # 创建新的DataFrame，pd.DataFrame将独热编码结果（encoded_features）转换为一个新的 DataFrame，列名为 encoded_feature_names

    data_df = pd.concat([data_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)  # pd.concat将编码后的特征与原数据合并，将原始数据框 data_df 和新的编码结果 encoded_df 按列（axis=1）合并。使用 reset_index(drop=True) 确保合并时索引对齐。

    data_df.drop(columns=categorical_features, inplace=True)  # 删除原始的分类特征列

    # 处理时间特征，将 time_of_day 转换为正弦和余弦形式，这些转换可以将时间特征映射到一个单位圆上，捕捉时间的周期性
    data_df['time_of_day_sin'] = np.sin(2 * np.pi * data_df['time_of_day'] / 24)
    data_df['time_of_day_cos'] = np.cos(2 * np.pi * data_df['time_of_day'] / 24)
    data_df.drop(columns=['time_of_day'], inplace=True)  #删除原始的 time_of_day 列，因为它已经被 time_of_day_sin 和 time_of_day_cos 替代


    # 数值型特征进行标准化，变成 N（0，1）分布
    from sklearn.preprocessing import StandardScaler

    # 数值型特征，不包括时间戳和目标变量
    numeric_features = ['solar_power', 'wind_power', 'unload_time', 'energy_demand']

    scaler = StandardScaler()
    data_df[numeric_features] = scaler.fit_transform(data_df[numeric_features]) #fit: 学习数据的均值和标准差。transform: 使用这些统计量将数据归一化。

    #feature_columns：定义模型的输入特征列表，标准化后的数值特征：solar_power, wind_power, unload_time, energy_demand。时间特征的周期性表示：time_of_day_sin, time_of_day_cos。独热编码后的分类特征：encoded_feature_names。
    feature_columns = ['solar_power', 'wind_power', 'unload_time', 'energy_demand', 'time_of_day_sin', 'time_of_day_cos'] + list(encoded_feature_names)  

    inputs = data_df[feature_columns].values  # 转换为 NumPy 数组。inputs 是一个二维数组，形状为 (num_rows, num_features)

    # 假设标签为 'target' 列，表示分类目标。labels 是一个一维数组，长度为 num_rows
    labels = data_df['target'].values

    # 定义序列长度，例如以一天的数据为一个序列（24小时）
    seq_length = 24

    # 确保数据长度是序列长度的整数倍。如果数据的总行数 len(data_df) 不是 seq_length 的整数倍，最后的部分数据无法构成完整的序列，因此需要截取到最接近的整数倍
    num_samples = len(data_df) // seq_length
    inputs = inputs[:num_samples * seq_length]
    labels = labels[:num_samples * seq_length]

    # 重塑输入数据和标签
    inputs = inputs.reshape(num_samples, seq_length, -1)
    labels = labels.reshape(num_samples, seq_length)

    # 为简单起见，我们采用每个序列最后一个时间步的标签作为整体标签
    labels = labels[:, -1]

    # 获取特征数量
    num_features = inputs.shape[2]

    return inputs, labels, num_features

# 调用数据加载函数
inputs, labels, num_features = load_and_preprocess_data()

# 将 NumPy 数组转换为 Torch 张量
inputs_tensor = torch.tensor(inputs, dtype=torch.float32) #适合神经网络中的浮点运算
labels_tensor = torch.tensor(labels, dtype=torch.long)  # 分类任务中的目标变量

# 创建数据集和数据加载器
dataset = TensorDataset(inputs_tensor, labels_tensor)  #将inputs_tensor 和 labels_tensor 打包成一个数据集对象，方便后续按批次加载

# 划分训练集和验证集，将数据集按比例划分为训练集（80%）和验证集（20%）。使用 torch.utils.data.random_split，随机划分数据
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

#创建训练集和验证集的加载器，支持按批次加载数据。batch_size: 每批次加载的样本数量。shuffle: 是否随机打乱数据（训练集通常需要打乱，验证集不需要）。num_workers: 数据加载的工作线程数量，0 表示在主线程中加载。pin_memory: 如果使用 GPU，可以启用以提高数据传输效率。train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 定义位置编码（Positional Encoding）因为transformer本身不具备内置的顺序感知能力
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        #初始化
        pe = torch.zeros(max_len, d_model) #d_model每个时间步的特征数 max_len最多5000个时间步 pe初始化一个大小为 (max_len, d_model) 的零矩阵，用于存储每个时间步的编码值
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)     # 偶数位置，正弦和余弦提供了周期性，使得模型能够捕捉序列中时间步之间的相对关系。不同频率（由 div_term 控制）使得模型能够感知短期和长期依赖
        pe[:, 1::2] = torch.cos(position * div_term)     # 奇数位置

        #增加 Batch 维度并注册为缓冲区
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_length, batch_size, d_model)。seq_length: 当前序列的长度（注意可能比 max_len 小）。batch_size: 序列的批次大小。d_model: 每个时间步的特征维度。
        x = x + self.pe[:x.size(0)]
        return x

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

        return output  # (batch_size, input_dim)

# 定义模型
class MyModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MyModel, self).__init__()

        # 假设风能和光伏的影响因素有 n_wind_solar_features 个
        # 根据数据中风能和光伏相关的特征数量来确定
        n_wind_solar_features = 2  # 例如，这里将前两个特征视为风能和光伏发电量

        # 计算其余特征数量
        n_other_features = num_features - n_wind_solar_features

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=n_other_features, out_channels=64, kernel_size=3, padding=1)
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

        # Transformer编码器层
        self.pos_encoder = PositionalEncoding(d_model=256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 注意力机制层
        self.attention = Attention(input_dim=n_wind_solar_features)
        self.fc_attn = nn.Linear(n_wind_solar_features, 512)

        # 全连接层
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 + 256, 128)
        self.bn_fc1 = nn.BatchNorm1d(num_features=128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, seq_length, num_features)
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_length)

        # 根据特征的排列方式，分割输入
        n_wind_solar_features = 2  # 风能和光伏特征数量
        wind_solar_input = x[:, :n_wind_solar_features, :]    # (batch_size, n_wind_solar_features, seq_length)
        other_input = x[:, n_wind_solar_features:, :]         # (batch_size, n_other_features, seq_length)

        # 调整 wind_solar_input 的形状供注意力机制使用
        wind_solar_input = wind_solar_input.permute(0, 2, 1)  # (batch_size, seq_length, n_wind_solar_features)

        # 第一组卷积层
        x = self.conv1(other_input)  # 输入维度 (batch_size, n_other_features, seq_length)
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

        # Transformer编码器
        x = x.permute(2, 0, 1)  # (seq_length/8, batch_size, 256)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # 取最后一个时间步的输出 (batch_size, 256)

        # 注意力机制
        attn_output = self.attention(wind_solar_input)  # (batch_size, n_wind_solar_features)
        attn_output = self.fc_attn(attn_output)         # (batch_size, 512)

        # 将注意力机制和 Transformer 的输出拼接在一起
        combined_output = torch.cat((attn_output, x), dim=1)  # (batch_size, 512 + 256)

        # 全连接层
        x = self.dropout(combined_output)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # (batch_size, num_classes)

# 实例化模型
model = MyModel(num_features=num_features, num_classes=num_classes)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 定义学习率调度器
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
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# 早停机制参数
patience = 5  # 在验证集上若干个周期无提升则停止
best_val_loss = float('inf')
counter = 0

# TensorBoard
writer = SummaryWriter(log_dir='runs/experiment1')

# 训练和验证函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
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

# 训练模型
train_acc_history = []
val_acc_history = []
global_step = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0

    # 使用 tqdm 包装训练集数据加载器
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_inputs, batch_labels in progress_bar:
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_size = batch_inputs.size(0)
        num_samples += batch_size

        optimizer.zero_grad()

        if scaler:
            # 使用自动混合精度
            with torch.cuda.amp.autocast():
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
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

        global_step += 1

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