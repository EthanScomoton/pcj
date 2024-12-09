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
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

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

    # 分离特征组
    renewable_features = ['season', 'holiday', 'weather', 'temperature', 'hour']
    load_features = ['ship_grade', 'work_time', 'dock_position']
    
    # 分别进行独热编码
    encoder_renewable = OneHotEncoder(sparse=False)
    encoder_load = OneHotEncoder(sparse=False)
    
    encoded_renewable = encoder_renewable.fit_transform(data_df[renewable_features])
    encoded_load = encoder_load.fit_transform(data_df[load_features])
    
    renewable_feature_names = encoder_renewable.get_feature_names_out(renewable_features)
    load_feature_names = encoder_load.get_feature_names_out(load_features)
    
    # 创建对应的DataFrame
    renewable_df = pd.DataFrame(encoded_renewable, columns=renewable_feature_names)
    load_df = pd.DataFrame(encoded_load, columns=load_feature_names)
    
    # 合并数据
    data_df = pd.concat([
        data_df.reset_index(drop=True),
        renewable_df.reset_index(drop=True),
        load_df.reset_index(drop=True)
    ], axis=1)
    
    # 删除原始分类列
    data_df.drop(columns=renewable_features + load_features, inplace=True)
   
    # 提取特征和目标
    feature_columns = list(renewable_feature_names) + list(load_feature_names)
    inputs = data_df[feature_columns].values
    labels = data_df['target'].values  # 假设目标列为 'target'

    return renewable_feature_names, load_feature_names, data_df, inputs, labels

# 调用数据加载函数
inputs, labels, renewable_feature_names, load_feature_names = load_and_preprocess_data()

# 定义特征维度
renewable_dim = len(renewable_feature_names)
load_dim = len(load_feature_names)
num_features = inputs.shape[1]

# 将 NumPy 数组转换为 Torch 张量
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# 计算类别权重
def calculate_class_weights(labels):
    label_counts = Counter(labels.numpy())
    total_samples = len(labels)
    num_classes = len(label_counts)

    weights = torch.zeros(num_classes)
    for label, count in label_counts.items():
        weights[label] = (total_samples / (num_classes * count)) ** 0.5  # 平滑处理，避免权重过大

    print("类别分布:", dict(label_counts))
    print("类别权重:", weights)

    return weights

class_weights = calculate_class_weights(labels_tensor).to(device)

# 创建数据集和数据加载器
dataset = TensorDataset(inputs_tensor, labels_tensor)  #将inputs_tensor 和 labels_tensor 打包成一个数据集对象，方便后续按批次加载

# 划分训练集和验证集，将数据集按比例划分为训练集（80%）和验证集（20%）。使用 torch.utils.data.random_split，随机划分数据
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

#创建训练集和验证集的加载器，支持按批次加载数据。batch_size: 每批次加载的样本数量。shuffle: 是否随机打乱数据（训练集通常需要打乱，验证集不需要）。num_workers: 数据加载的工作线程数量，0 表示在主线程中加载。pin_memory: 如果使用 GPU，可以启用以提高数据传输效率。
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
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
    def __init__(self, num_features, num_classes, renewable_dim, load_dim):
        super(MyModel, self).__init__()

        self.renewable_dim = renewable_dim
        self.load_dim = load_dim

        # 可再生能源特征处理
        self.renewable_encoder = nn.Sequential(
            nn.Linear(self.renewable_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # 负荷特征处理
        self.load_encoder = nn.Sequential(
            nn.Linear(self.load_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # 交互关系的BiGRU
        self.interaction_bigru = nn.GRU(
            input_size=128,  # 64(renewable) + 64(load)
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 时序特征处理的BiGRU（如果有时序特征）
        temporal_input_dim = num_features - self.renewable_dim - self.load_dim
        if temporal_input_dim > 0:
            self.temporal_bigru = nn.GRU(
                input_size=temporal_input_dim,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
        else:
            self.temporal_bigru = None

        # Transformer编码器
        self.pos_encoder = PositionalEncoding(d_model=256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 注意力层
        self.attention = Attention(input_dim=128)  # 64*2 due to bidirectional

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(128 + (256 if self.temporal_bigru else 0) + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 提取特征
        renewable_features = x[:, :self.renewable_dim]
        load_features = x[:, self.renewable_dim:self.renewable_dim + self.load_dim]
        temporal_features = x[:, self.renewable_dim + self.load_dim:]

        # 编码可再生能源和负荷特征
        renewable_encoded = self.renewable_encoder(renewable_features)  # (batch_size, 64)
        load_encoded = self.load_encoder(load_features)  # (batch_size, 64)

        # 合并编码后的特征
        combined_features = torch.cat([renewable_encoded, load_encoded], dim=-1)  # (batch_size, 128)
        combined_features = combined_features.unsqueeze(1)  # (batch_size, 1, 128)

        # 使用BiGRU建模交互关系
        interaction_out, _ = self.interaction_bigru(combined_features)  # (batch_size, 1, 128)

        # 注意力机制
        attention_out = self.attention(interaction_out)  # (batch_size, 128)

        # 处理时序特征（如果有）
        if self.temporal_bigru:
            temporal_features = temporal_features.unsqueeze(1)  # (batch_size, 1, feature_dim)
            temporal_out, _ = self.temporal_bigru(temporal_features)  # (batch_size, 1, 256)

            # Transformer处理
            transformer_input = temporal_out.permute(1, 0, 2)  # (seq_len=1, batch_size, 256)
            transformer_input = self.pos_encoder(transformer_input)
            transformer_out = self.transformer_encoder(transformer_input)  # (seq_len=1, batch_size, 256)
            transformer_out = transformer_out.permute(1, 0, 2)  # (batch_size, seq_len=1, 256)

            temporal_out = temporal_out[:, -1, :]  # (batch_size, 256)
            transformer_out = transformer_out[:, -1, :]  # (batch_size, 256)
        else:
            # 如果没有时序特征，用零填充
            temporal_out = torch.zeros(x.size(0), 256).to(x.device)
            transformer_out = torch.zeros(x.size(0), 256).to(x.device)

        # 特征融合
        combined = torch.cat([
            attention_out,        # (batch_size, 128)
            temporal_out,         # (batch_size, 256)
            transformer_out       # (batch_size, 256)
        ], dim=1)  # (batch_size, 128 + 256 + 256)

        # 输出预测
        output = self.fc(combined)  # (batch_size, num_classes)

        return output

# 初始化模型
model = MyModel(num_features=num_features, num_classes=num_classes, renewable_dim=renewable_dim, load_dim=load_dim)
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 定义学习率调度器
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)  # 前10%步骤用于热身

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return max(0.0, 1 - (current_step - warmup_steps) / (total_steps - warmup_steps))

scheduler = LambdaLR(optimizer, lr_lambda)

# 自动混合精度
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

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
    all_preds = []
    all_labels = []

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

            # 记录所有预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / num_samples
    val_acc = running_corrects.double() / num_samples

    # 生成分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))

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

    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

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

        # 计算损失和准确率
        running_loss += loss.item() * batch_inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == batch_labels).item()

        # 更新进度条描述
        progress_bar.set_postfix({'Loss': running_loss / num_samples, 'Acc': (running_corrects.double() / num_samples).item()})

        # 更新学习率
        scheduler.step()
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
def plot_metrics(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
    epochs = range(1, len(train_acc_history) + 1)

    plt.figure(figsize=(12, 6))

    # 准确率
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_history, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_acc_history, label='Validation Accuracy', marker='o')
    plt.title('Train and Validation Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 损失
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_history, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss_history, label='Validation Loss', marker='o')
    plt.title('Train and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 在训练完成后调用此函数
plot_metrics(train_acc_history, val_acc_history)

# 关闭 TensorBoard
writer.close()
print('训练完成。')