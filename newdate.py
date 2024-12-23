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
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 定义超参数
learning_rate = 1e-5   # 学习率
num_epochs = 200       # 训练轮数
batch_size = 256       # 批次大小
weight_decay = 1e-4    # L2正则化防止过拟合
patience = 5           # 早停轮数

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

# 加载和预处理数据
def load_and_preprocess_data():
    renewable_df = pd.read_csv('/Users/ethan/Desktop/renewable_data.csv')
    load_df = pd.read_csv('/Users/ethan/Desktop/load_data.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])

    data_df = pd.concat([renewable_df, load_df], axis=1)

    renewable_features = ['season','holiday','weather','temperature','working_hours','E_PV','E_storage_discharge','E_grid','ESCFR','ESCFG'
    ]
    load_features = ['ship_grade','dock_position','destination']

    # 提取目标(能耗)

    y_raw = data_df['energyconsumption'].values.astype(float)
    # 对数变换: log( y + 1 )
    y_log = np.log1p(y_raw)

    encoder_renewable = OneHotEncoder(sparse_output=False)
    encoder_load = OneHotEncoder(sparse_output=False)

    encoded_renewable = encoder_renewable.fit_transform(data_df[renewable_features])
    encoded_load = encoder_load.fit_transform(data_df[load_features])

    renewable_feature_names = encoder_renewable.get_feature_names_out(renewable_features)
    load_feature_names = encoder_load.get_feature_names_out(load_features)

    renewable_df_encoded = pd.DataFrame(encoded_renewable, columns=renewable_feature_names)
    load_df_encoded = pd.DataFrame(encoded_load, columns=load_feature_names)

    data_df = pd.concat([data_df, renewable_df_encoded, load_df_encoded], axis=1)

    data_df.drop(columns=renewable_features + load_features, inplace=True)

    feature_columns = list(renewable_feature_names) + list(load_feature_names)
    X_raw = data_df[feature_columns].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    return X_scaled, y_log, renewable_feature_names, load_feature_names, scaler_X

# 调用数据加载函数
inputs, labels_log, renewable_feature_names, load_feature_names, scaler_X = load_and_preprocess_data()

# 定义特征维度
renewable_dim = len(renewable_feature_names)
load_dim = len(load_feature_names)
num_features = inputs.shape[1]

# 将 NumPy 数组转换为 Torch 张量
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
labels_tensor = torch.tensor(labels_log, dtype=torch.float32)

# 创建数据集和数据加载器
dataset = TensorDataset(inputs_tensor, labels_tensor)

# 划分训练集和验证集
dataset = TensorDataset(inputs_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 定义位置编码（Positional Encoding）
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
        x = x + self.pe[:x.size(0), :]
        return x

# 自定义注意力模块
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

# 定义回归模型
class EModel(nn.Module):
    def __init__(self, num_features, renewable_dim, load_dim):
        super(EModel, self).__init__()
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 6)

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
        interaction_out, _ = self.interaction_bigru(combined_features)  # (batch_size, 1, 128)

        # 注意力机制
        attention_out = self.attention(interaction_out)  # (batch_size, 128)

        # 动态检查时序特征是否为空
        if temporal_features.size(1) > 0:
            temporal_features = temporal_features.unsqueeze(1)  # (batch_size, 1, feature_dim)
            temporal_out, _ = self.temporal_bigru(temporal_features)  # (batch_size, 1, 256)

            # Transformer处理
            transformer_input = temporal_out.permute(1, 0, 2)  
            transformer_input = self.pos_encoder(transformer_input)
            transformer_out = self.transformer_encoder(transformer_input)  
            transformer_out = transformer_out.permute(1, 0, 2)  

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

# 初始化模型
model = EModel(num_features=num_features, renewable_dim=renewable_dim, load_dim=load_dim).to(device)

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
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# 早停
best_val_loss = float('inf')
counter = 0

# TensorBoard
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
                with torch.cuda.amp.autocast():
                    outputs = model(batch_inputs)
                    loss = criterion(outputs.squeeze(-1), batch_labels) 
            else:
                outputs = model(batch_inputs)
                loss = criterion(outputs.squeeze(-1), batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples += batch_inputs.size(0)

            preds_list.append(outputs.squeeze(-1).cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())

    val_loss = running_loss / num_samples  # MSE in log domain
    
    # 计算 RMSE in log domain
    preds_arr = np.concatenate(preds_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    rmse_log = np.sqrt(mean_squared_error(labels_arr, preds_arr))

    return val_loss, rmse_log

# 训练循环
train_mse_history = []
val_mse_history = []
train_rmse_history = []
val_rmse_history = []

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

        if scaler:
            with torch.cuda.amp.autocast():
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

# 画图：训练和验证集 MSE、RMSE
def plot_metrics(train_mse_history, val_mse_history, train_rmse_history, val_rmse_history):
    epochs = range(1, len(train_mse_history) + 1)

    plt.figure(figsize=(12,5))

    # MSE(log)
    plt.subplot(1,2,1)
    plt.plot(epochs, train_mse_history, 'o-', label='Train MSE (log)')
    plt.plot(epochs, val_mse_history, 'o-', label='Val MSE (log)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (log domain)')
    plt.title('Train/Val MSE (log)')
    plt.legend()
    plt.grid(True)

    # RMSE(log)
    plt.subplot(1,2,2)
    plt.plot(epochs, train_rmse_history, 'o-', label='Train RMSE (log)')
    plt.plot(epochs, val_rmse_history, 'o-', label='Val RMSE (log)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (log domain)')
    plt.title('Train/Val RMSE (log)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_metrics(train_mse_history, val_mse_history, train_rmse_history, val_rmse_history)