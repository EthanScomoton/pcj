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
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# 定义超参数
learning_rate = 1e-5   # 学习率
num_epochs = 100        # 训练轮数
batch_size = 512       # 批次大小
weight_decay = 5e-3    # L2正则化防止过拟合
patience = 5           # 早停轮数
num_workers = 0        # 数据加载器的工作进程数

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

# ---------------------------
# 1. 加载和预处理数据
# ---------------------------
def load_and_preprocess_data():
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data.csv')
    load_df = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])

    data_df = pd.merge(renewable_df, load_df, 
                       on='timestamp',  
                       how='inner')     

    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour'] = data_df['timestamp'].dt.hour
    data_df['month'] = data_df['timestamp'].dt.month

    # 时间相关特征的 sin/cos 编码
    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month']-1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month']-1) / 12)

    renewable_features = ['season', 'holiday', 'weather', 'temperature', 'working_hours',
                          'E_PV', 'E_storage_discharge', 'E_grid', 'ESCFR', 'ESCFG']
    load_features = ['ship_grade', 'dock_position', 'destination']

    # 目标值（能耗）
    y_raw = data_df['energyconsumption'].values.astype(float)
    y_log = np.log1p(y_raw)  # 对数变换

    # 使用 LabelEncoder 进行编码
    label_encoders = {}
    for feature in renewable_features + load_features:
        le = LabelEncoder()
        data_df[feature] = le.fit_transform(data_df[feature].fillna("Unknown"))
        label_encoders[feature] = le

    time_feature_cols = [
        'dayofweek_sin', 'dayofweek_cos',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos'
    ]

    feature_columns = renewable_features + load_features + time_feature_cols

    # 从 data_df 中取出特征
    X_raw = data_df[feature_columns].values

    # 数值标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    return X_scaled, y_log, renewable_features, load_features, scaler_X

# 调用数据加载函数
inputs, labels_log, renewable_features, load_features, scaler_X = load_and_preprocess_data()

# 定义特征维度
renewable_dim = len(renewable_features)
load_dim = len(load_features)
num_features = inputs.shape[1]

# 将 NumPy 数组转换为 Torch 张量
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
labels_tensor = torch.tensor(labels_log, dtype=torch.float32)

# 创建数据集和数据加载器
dataset = TensorDataset(inputs_tensor, labels_tensor)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=num_workers, pin_memory=True)

# ---------------------------
# 2. 定义辅助模块
# ---------------------------
# 位置编码（Positional Encoding）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

# 简单注意力模块
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
        attn_weights = F.softmax(attn_weights, dim=1)  
        weighted = x * attn_weights
        output = torch.sum(weighted, dim=1)  # (batch_size, input_dim)
        return output

# ---------------------------
# 3. 定义主模型 EModel
# ---------------------------
class EModel(nn.Module):
    def __init__(self, num_features, renewable_dim, load_dim):
        super(EModel, self).__init__()
        self.renewable_dim = renewable_dim
        self.load_dim = load_dim

        # 初始化时全部特征的权重为1，可学习参数
        self.feature_importance = nn.Parameter(
            torch.ones(num_features), requires_grad=True
        )

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
            input_size = 128,  
            hidden_size = 64,
            num_layers = 1,
            batch_first = True,
            bidirectional = True,
            dropout = 0.3
        )

        # 时序特征处理的BiGRU
        temporal_input_dim = num_features - self.renewable_dim - self.load_dim
        if temporal_input_dim > 0:
            self.temporal_bigru = nn.GRU(
                input_size = temporal_input_dim,
                hidden_size = 128,
                num_layers = 2,
                batch_first = True,
                bidirectional = True,
                dropout = 0.2
            )
        else:
            self.temporal_bigru = None

        # Transformer
        self.pos_encoder = PositionalEncoding(d_model = 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model = 256, nhead = 8, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 2)

        # Attention
        self.attention = Attention(input_dim=128) 

        # 输出层：回归 -> 输出维度设为1
        self.fc = nn.Sequential(
            nn.Linear(128 + 256 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # 最终输出一个值用于回归
        )

    def forward(self, x):
        # -------- 对输入特征做权重加成 --------
        # shape: (batch_size, num_features)
        x = x * self.feature_importance  # 广播机制，逐元素乘

        # 分离特征
        renewable_features = x[:, :self.renewable_dim]
        load_features = x[:, self.renewable_dim:self.renewable_dim + self.load_dim]
        temporal_features = x[:, self.renewable_dim + self.load_dim:]

        # 编码可再生能源和负荷特征
        renewable_encoded = self.renewable_encoder(renewable_features)  
        load_encoded = self.load_encoder(load_features)  

        # 合并编码后的特征
        combined_features = torch.cat([renewable_encoded, load_encoded], dim=-1)  # (batch_size, 128)
        combined_features = combined_features.unsqueeze(1)  # (batch_size, 1, 128)

        # 使用BiGRU建模交互关系
        interaction_out, _ = self.interaction_bigru(combined_features)  # (batch_size, 1, 128 * 2 = 128)
        # 取双向拼接后的 128 （64 * 2）
        attention_out = self.attention(interaction_out)  # (batch_size, 128)

        # 动态检查时序特征是否为空
        if self.temporal_bigru is not None and temporal_features.size(1) > 0:
            # temporal_features: (batch_size, temporal_input_dim)
            # 在此示例里，如果只是一条序列信息，实际上 seq_len=1；如需处理更长序列，可自行修改
            temporal_features = temporal_features.unsqueeze(1)  # (batch_size, 1, feature_dim)
            temporal_out, _ = self.temporal_bigru(temporal_features)  # (batch_size, 1, 2*128=256)

            # Transformer处理
            # 先把 (batch_size, seq_len, feature_dim) -> (seq_len, batch_size, feature_dim)
            transformer_input = temporal_out.permute(1, 0, 2)  
            transformer_input = self.pos_encoder(transformer_input)
            transformer_out = self.transformer_encoder(transformer_input)  
            # 变回来 (batch_size, seq_len, feature_dim)
            transformer_out = transformer_out.permute(1, 0, 2)  

            # 取最后时刻
            temporal_out = temporal_out[:, -1, :]  
            transformer_out = transformer_out[:, -1, :]  
        else:
            # 如果没有时序特征，用零填充
            temporal_out = torch.zeros(x.size(0), 256).to(x.device)
            transformer_out = torch.zeros(x.size(0), 256).to(x.device)

        # 特征融合
        merged = torch.cat([attention_out, temporal_out, transformer_out], dim=1)
        output = self.fc(merged)  # (batch, 1)
        return output

# ---------------------------
# 4. 训练准备
# ---------------------------
model = EModel(num_features=num_features, 
               renewable_dim=renewable_dim, 
               load_dim=load_dim).to(device)

criterion = nn.MSELoss()  # 对数域下的 MSE
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 学习率调度
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return max(0.0, 1 - (current_step - warmup_steps) / (total_steps - warmup_steps))

scheduler = LambdaLR(optimizer, lr_lambda)

# 自动混合精度
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

# 早停
best_val_loss = float('inf')
counter = 0

writer = SummaryWriter(log_dir='runs/experiment_log_transform')

# 回归评估函数：返回 MSE, RMSE
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_samples = 0

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_inputs)
                    loss = criterion(outputs.squeeze(-1), batch_labels)
            else:
                outputs = model(batch_inputs)
                loss = criterion(outputs.squeeze(-1), batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples += batch_inputs.size(0)

            preds_list.append(outputs.squeeze(-1).cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())

    val_loss = running_loss / num_samples  # MSE 在对数域下
    
    # 计算 RMSE 在对数域下
    preds_arr = np.concatenate(preds_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    rmse_log = np.sqrt(mean_squared_error(labels_arr, preds_arr))

    return val_loss, rmse_log

# ---------------------------
# 5. 训练循环
# ---------------------------
train_mse_history = []
val_mse_history = []
train_rmse_history = []
val_rmse_history = []

# 用于记录每个 epoch 的特征权重变化
feature_importance_history = []

global_step = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_inputs, batch_labels in progress_bar:
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(batch_inputs)
                loss = criterion(outputs.squeeze(-1), batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_inputs)
            loss = criterion(outputs.squeeze(-1), batch_labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_inputs.size(0)
        num_samples += batch_inputs.size(0)

        progress_bar.set_postfix({
            'Train_MSE_log': running_loss / num_samples
        })

        scheduler.step()
        global_step += 1

    epoch_train_mse = running_loss / num_samples
    epoch_train_rmse = np.sqrt(epoch_train_mse)

    # 验证集
    val_mse, val_rmse = evaluate(model, val_loader, criterion, device)

    train_mse_history.append(epoch_train_mse)
    val_mse_history.append(val_mse)
    train_rmse_history.append(epoch_train_rmse)
    val_rmse_history.append(val_rmse)

    writer.add_scalar('Train/MSE_log', epoch_train_mse, epoch)
    writer.add_scalar('Train/RMSE_log', epoch_train_rmse, epoch)
    writer.add_scalar('Val/MSE_log', val_mse, epoch)
    writer.add_scalar('Val/RMSE_log', val_rmse, epoch)

    print(f"\nEpoch {epoch+1}/{num_epochs}, "
          f"Train MSE (log): {epoch_train_mse:.4f}, Train RMSE (log): {epoch_train_rmse:.4f}, "
          f"Val MSE (log): {val_mse:.4f}, Val RMSE (log): {val_rmse:.4f}")

    # 记录此时的特征权重
    current_weights = model.feature_importance.detach().cpu().numpy().copy()
    feature_importance_history.append(current_weights)

    # 早停逻辑
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        counter = 0
        torch.save(model.state_dict(), 'best_model_log_transform.pth')
        print("模型已保存！")
    else:
        counter += 1
        if counter >= patience:
            print("验证集没有更好的 MSE (log)，提前停止训练")
            break

writer.close()
print("训练完成。")

# ---------------------------
# 6. 画图与可视化
# ---------------------------
def plot_metrics(train_mse_history, val_mse_history, train_rmse_history, val_rmse_history):
    epochs = range(1, len(train_mse_history) + 1)

    plt.figure(figsize=(12,5))

    # MSE(log)
    plt.subplot(1,2,1)
    plt.plot(epochs, train_mse_history, 'o-', label = 'Train MSE (log)')
    plt.plot(epochs, val_mse_history, 'o-', label = 'Val MSE (log)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (log domain)')
    plt.title('Train/Val MSE (log)')
    plt.legend()
    plt.grid(True)

    # RMSE(log)
    plt.subplot(1,2,2)
    plt.plot(epochs, train_rmse_history, 'o-', label = 'Train RMSE (log)')
    plt.plot(epochs, val_rmse_history, 'o-', label = 'Val RMSE (log)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (log domain)')
    plt.title('Train/Val RMSE (log)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_feature_importance_history(feature_importance_history, feature_names=None):
    """
    可视化每个 Epoch 的特征权重。
    feature_importance_history: list[np.ndarray], 每个元素是某个 epoch 的特征权重
    """
    feature_importance_history = np.array(feature_importance_history)  # shape: (num_epochs, num_features)
    num_epochs, num_features = feature_importance_history.shape
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(num_features)]

    plt.figure(figsize=(10,6))
    for i in range(num_features):
        plt.plot(range(1, num_epochs+1), feature_importance_history[:, i], 
                 label=f"{feature_names[i]}")

    plt.xlabel("Epoch")
    plt.ylabel("Feature Weight")
    plt.title("Feature Importance over Epochs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 调用画图函数
plot_metrics(train_mse_history, val_mse_history, train_rmse_history, val_rmse_history)

# 如果想要基于具体特征列名，可以这样传入：
all_feature_names = renewable_features + load_features + [
    'dayofweek_sin','dayofweek_cos','hour_sin','hour_cos',
    'month_sin','month_cos'
]
plot_feature_importance_history(feature_importance_history, feature_names=all_feature_names)