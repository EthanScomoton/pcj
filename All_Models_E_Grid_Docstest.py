import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib as mpl

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from torch.nn.utils import clip_grad_norm_
from x_transformers import RMSNorm, Lion, TransformerWrapper, Decoder, RotaryEmbedding

# ---------------------------
# 全局字体及样式设置
# ---------------------------
mpl.rcParams.update({
    'font.size': 18,          # 全局默认字体大小
    'axes.labelsize': 18,     # 坐标轴标签（xlabel, ylabel）的字体大小
    'axes.titlesize': 20,     # 图表标题（title）的字体大小
    'xtick.labelsize': 16,    # x轴刻度标签的字体大小
    'ytick.labelsize': 16     # y轴刻度标签的字体大小
})

# ---------------------------
# 0. 全局超参数
# ---------------------------
learning_rate = 1e-3
num_epochs    = 150
batch_size    = 128
weight_decay  = 1e-6
patience      = 8
num_workers   = 0
window_size   = 20

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
#   1. 数据加载
# =====================================================================
def load_data():
    """
    加载原始的可再生能源与负荷数据，并按时间戳做 merge。
    """
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data10.csv')
    load_df      = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data10.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp']      = pd.to_datetime(load_df['timestamp'])

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')
    data_df.sort_values('timestamp', inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    return data_df


# =====================================================================
#   2. 特征工程
# =====================================================================
def feature_engineering(data_df):
    """
    - E_grid 的 EWMA 平滑
    - 时间特征构造
    - 分类特征 LabelEncoder
    - 不做数值标准化（避免数据泄露）
    """
    span = 10
    data_df['E_grid'] = data_df['E_grid'].ewm(span=span, adjust=False).mean()

    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour']      = data_df['timestamp'].dt.hour
    data_df['month']     = data_df['timestamp'].dt.month

    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month'] - 1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month'] - 1) / 12)

    # 对分类特征进行 LabelEncoder
    renewable_features = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_PV', 'E_wind', 'E_storage_discharge',
        'ESCFR', 'ESCFG'
    ]
    load_features = [
        'ship_grade', 'dock_position', 'destination', 'energyconsumption'
    ]
    for col in renewable_features + load_features:
        if col in data_df.columns:
            le = LabelEncoder()
            data_df[col] = le.fit_transform(data_df[col].astype(str))

    time_feature_cols = [
        'dayofweek_sin', 'dayofweek_cos',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos'
    ]
    feature_columns = renewable_features + load_features + time_feature_cols
    target_column   = 'E_grid'
    
    return data_df, feature_columns, target_column


# =====================================================================
#   3. 序列构造
# =====================================================================
def create_sequences(X_data, y_data, window_size):
    X_list, y_list = [], []
    num_samples = X_data.shape[0]
    for i in range(num_samples - window_size):
        seq_x = X_data[i : i + window_size, :]
        seq_y = y_data[i + window_size]
        X_list.append(seq_x)
        y_list.append(seq_y)
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    return X_arr, y_arr


# =====================================================================
#   4. 模型定义
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(-(torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x, step_offset=0):
        seq_len = x.size(1)
        pos_enc = self.pe[step_offset: step_offset + seq_len, 0, :]
        return x + pos_enc.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, d_model, nhead=8, num_encoder_layers=2, num_decoder_layers=2):
        super(Transformer, self).__init__()
        # 使用Rotary Position Embedding替代传统位置编码
        self.rotary_emb = RotaryEmbedding(dim=d_model//2)
        
        # 优化后的Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=0.1,
            batch_first=True,
            activation='gelu'  # 使用GELU激活函数
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers,
            norm=RMSNorm(d_model)  # 使用RMSNorm
        )

        # 优化后的Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=RMSNorm(d_model)
        )

    def forward(self, src, tgt):
        # 应用旋转位置编码
        src = self.rotary_emb(src)
        tgt = self.rotary_emb(tgt)
        
        # Encoder处理
        memory = self.transformer_encoder(src)
        
        # Decoder处理
        out = self.transformer_decoder(tgt, memory)
        return out


class CNNBlock(nn.Module):
    def __init__(self, feature_dim, hidden_size, dropout=0):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size=7, padding=3)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(2 * hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  # => [batch_size, feature_dim, seq_len]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = x.transpose(1, 2)  # => [batch_size, seq_len, 2*hidden_size]
        return x


class Attention(nn.Module):
    def __init__(self, input_dim, dropout=0.1):  # 添加dropout参数
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)  # 使用传入的dropout

    def forward(self, x):
        attn_weights = self.attention(x)       # [batch_size, seq_len, 1]
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = x * attn_weights
        return torch.sum(weighted, dim=1)      # => [batch_size, input_dim]


class EModel_FeatureWeight(nn.Module):
    def __init__(self, feature_dim, lstm_hidden_size=128, lstm_num_layers=2, lstm_dropout=0.2):
        super(EModel_FeatureWeight, self).__init__()
        self.feature_dim = feature_dim
        
        # 特征重要性权重
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)
        
        # 改进的LSTM模块
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        # LSTM权重初始化
        self._init_lstm_weights()
        
        # Transformer模块
        self.transformer_block = Transformer(
            d_model=2*lstm_hidden_size, 
            nhead=4, 
            num_encoder_layers=2, 
            num_decoder_layers=2
        )
        
        # Attention模块
        self.attention = Attention(input_dim=2*lstm_hidden_size)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(2*lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def _init_lstm_weights(self):
        """按照min-LSTM论文的方法初始化LSTM权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1.0)

    def forward(self, x):
        # 特征加权
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # Transformer处理
        transformer_out = self.transformer_block(lstm_out, lstm_out)
        
        # Attention处理
        attn_out = self.attention(transformer_out)
        
        # 最终预测
        out = self.fc(attn_out)
        return out.squeeze(-1)


class EModel_CNN_Transformer(nn.Module):
    def __init__(self, feature_dim, hidden_size=128, num_layers=2, dropout=0.1):
        super(EModel_CNN_Transformer, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # 特征重要性权重
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)
        
        # CNN特征提取
        self.cnn_block = CNNBlock(feature_dim, hidden_size, dropout)

        # 优化后的Transformer模块
        self.transformer_block = Transformer(
            d_model=2*hidden_size,
            nhead=8,
            num_encoder_layers=num_layers,  # 使用传入的num_layers
            num_decoder_layers=num_layers,  # 使用传入的num_layers
            dropout=dropout  # 添加dropout参数
        )
        # 注意力机制
        self.attention = Attention(input_dim=2*hidden_size)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # 残差连接
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, 2 * hidden_size),
            nn.GELU()
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 特征加权
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)
        
        # 残差连接
        residual = self.residual(x[:, -1, :])
        
        # CNN特征提取
        cnn_out = self.cnn_block(x)
        
        # Transformer处理
        transformer_out = self.transformer_block(cnn_out, cnn_out)
        
        # 添加残差连接
        transformer_out = transformer_out + residual.unsqueeze(1)
        
        # 注意力机制
        attn_out = self.attention(transformer_out)
        
        # 最终预测
        out = self.fc(attn_out)
        return out.squeeze(-1)


# =====================================================================
#   5. 评估工具: 计算 MSELoss / RMSE / MAPE / R^2 / SMAPE / MAE
# =====================================================================
def evaluate(model, dataloader, criterion, device='cuda'):
    model.eval()
    running_loss, num_samples = 0.0, 0
    preds_list, labels_list = [], []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples  += batch_inputs.size(0)

            preds_list.append(outputs.cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())

    val_loss   = running_loss / num_samples
    preds_arr  = np.concatenate(preds_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)

    # RMSE
    rmse_std = np.sqrt(mean_squared_error(labels_arr, preds_arr))

    # MAPE
    nonzero_mask_mape = (labels_arr != 0)
    if np.sum(nonzero_mask_mape) > 0:
        mape_std = np.mean(np.abs((labels_arr[nonzero_mask_mape] - preds_arr[nonzero_mask_mape]) 
                                  / labels_arr[nonzero_mask_mape])) * 100.0
    else:
        mape_std = 0.0

    # R^2
    ss_res = np.sum((labels_arr - preds_arr)**2)
    ss_tot = np.sum((labels_arr - np.mean(labels_arr))**2)
    r2_std = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # SMAPE
    numerator = np.abs(labels_arr - preds_arr)
    denominator = np.abs(labels_arr) + np.abs(preds_arr)
    nonzero_mask_smape = (denominator != 0)
    if np.sum(nonzero_mask_smape) > 0:
        smape_val = 100.0 * 2.0 * np.mean(numerator[nonzero_mask_smape] / denominator[nonzero_mask_smape])
    else:
        smape_val = 0.0

    # MAE
    mae_val = np.mean(np.abs(labels_arr - preds_arr))

    return val_loss, rmse_std, mape_std, r2_std, smape_val, mae_val, preds_arr, labels_arr


# =====================================================================
#   6. 训练工具: 同时记录训练集、验证集指标
# =====================================================================
def train_model(model, train_loader, val_loader, model_name='Model', learning_rate=1e-4, weight_decay=1e-2):
    """
    每个 epoch:
      1) evaluate(model, train_loader, ...)
      2) evaluate(model, val_loader, ...)
      3) 记录训练与验证各项指标
      4) early stopping
    """
    criterion = nn.MSELoss()
    optimizer = Lion(
        model.parameters(),
        lr = learning_rate,  # 保持原有学习率
        weight_decay = weight_decay  # 保持原有权重衰减
    )

    total_steps  = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(0.0, 1 - (current_step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss = float('inf')
    counter = 0
    global_step = 0

    # 历史记录
    train_loss_history = []
    train_rmse_history = []
    train_mape_history = []
    train_r2_history   = []
    train_smape_history= []
    train_mae_history  = []

    val_loss_history   = []
    val_rmse_history   = []
    val_mape_history   = []
    val_r2_history     = []
    val_smape_history  = []
    val_mae_history    = []

    for epoch in range(num_epochs):
        # === 训练阶段 ===
        model.train()
        running_loss, num_samples = 0.0, 0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss  = criterion(preds, batch_labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples  += batch_inputs.size(0)
            global_step  += 1

        train_loss_epoch = running_loss / num_samples

        # === 计算 train set 上的各指标 ===
        train_loss_eval, train_rmse_eval, train_mape_eval, train_r2_eval, train_smape_eval, train_mae_eval, _, _ = evaluate(model, train_loader, criterion)

        # === 计算 val set 上的各指标 ===
        val_loss_eval, val_rmse_eval, val_mape_eval, val_r2_eval, val_smape_eval, val_mae_eval, _, _ = evaluate(model, val_loader, criterion)

        # === 保存各类指标到历史数组 ===
        train_loss_history.append(train_loss_eval)
        train_rmse_history.append(train_rmse_eval)
        train_mape_history.append(train_mape_eval)
        train_r2_history.append(train_r2_eval)
        train_smape_history.append(train_smape_eval)
        train_mae_history.append(train_mae_eval)

        val_loss_history.append(val_loss_eval)
        val_rmse_history.append(val_rmse_eval)
        val_mape_history.append(val_mape_eval)
        val_r2_history.append(val_r2_eval)
        val_smape_history.append(val_smape_eval)
        val_mae_history.append(val_mae_eval)

        # === 打印日志 ===
        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"TrainLoss: {train_loss_epoch:.4f}, "
              f"ValLoss: {val_loss_eval:.4f}, "
              f"ValRMSE(std): {val_rmse_eval:.4f}, "
              f"ValMAPE(%): {val_mape_eval:.2f}, "
              f"ValR^2: {val_r2_eval:.4f}, "
              f"ValSMAPE(%): {val_smape_eval:.2f}, "
              f"ValMAE(std): {val_mae_eval:.4f}")

        # === Early Stopping ===
        if val_loss_eval < best_val_loss:
            best_val_loss = val_loss_eval
            counter       = 0
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            print(f"[{model_name}] 模型已保存 (best val_loss={best_val_loss:.4f}).")
        else:
            counter += 1
            if counter >= patience:
                print(f"[{model_name}] 验证集无改善，提前停止。")
                break

    # 整合返回
    return {
        "train_loss":  train_loss_history,
        "train_rmse":  train_rmse_history,
        "train_mape":  train_mape_history,
        "train_r2":    train_r2_history,
        "train_smape": train_smape_history,
        "train_mae":   train_mae_history,

        "val_loss":    val_loss_history,
        "val_rmse":    val_rmse_history,
        "val_mape":    val_mape_history,
        "val_r2":      val_r2_history,
        "val_smape":   val_smape_history,
        "val_mae":     val_mae_history
    }


# =====================================================================
#   7. 可视化与辅助函数
# =====================================================================
def plot_correlation_heatmap(df, feature_cols, title="Heat map"):
    df_encoded = df.copy()
    for col in feature_cols:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    corr_matrix = df_encoded[feature_cols].corr()

    colors_list = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("red_yellow_green", colors_list, N=256)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        cmap=cmap_custom,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=1,
        linecolor='white',
        cbar=True,
        vmin=-1,
        vmax=1
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def analyze_target_distribution(data_df, target_col):
    if target_col not in data_df.columns:
        print(f"[Warning] '{target_col}' not in data_df. Skip analysis.")
        return

    print(f"\n[Target Analysis] Basic stats of '{target_col}':")
    print(data_df[target_col].describe())

    plt.figure(figsize=(6,4))
    plt.hist(data_df[target_col], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of '{target_col}' in original domain")
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_Egrid_over_time(data_df):
    plt.figure(figsize=(10, 5))
    plt.plot(data_df['timestamp'], data_df['E_grid'], color='blue', marker='o', markersize=3, linewidth=1)
    plt.xlabel('Timestamp')
    plt.ylabel('E_grid')
    plt.title('E_grid over Time (entire dataset)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions_comparison(y_actual_real, y_pred_model1_real, y_pred_model2_real,
                               model1_name='Model1', model2_name='Model2'):
    plt.figure(figsize=(10,5))
    x_axis = np.arange(len(y_actual_real))

    plt.plot(x_axis, y_actual_real,        'red',        label='Actual', linewidth=1)
    plt.plot(x_axis, y_pred_model1_real,   'lightgreen', label=model1_name, linewidth=1)
    plt.plot(x_axis, y_pred_model2_real,   'skyblue',    label=model2_name, linewidth=1)
    plt.xlabel('Index')
    plt.ylabel('Value (real domain)')
    plt.title(f'Comparison: Actual vs {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_test_predictions_over_time(test_timestamps, y_actual_real, y_pred_real):
    plt.figure(figsize=(10,5))
    plt.plot(test_timestamps, y_actual_real, color='red',  label='Actual E_grid', linewidth=1)
    plt.plot(test_timestamps, y_pred_real,   color='blue', label='Predicted E_grid', linewidth=1, linestyle='--')
    plt.xlabel('Timestamp (Test Data)')
    plt.ylabel('E_grid (real domain)')
    plt.title('Comparison of Actual vs Predicted E_grid over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_curves_allmetrics(hist_dict, model_name='Model'):
    """
    统一绘制 Loss / RMSE / MAPE / R^2 / SMAPE / MAE 共6种指标的曲线
    同时绘制 train 和 val
    """
    epochs = range(1, len(hist_dict["train_loss"]) + 1)
    plt.figure(figsize=(15, 12))

    # (1) Loss
    plt.subplot(3, 2, 1)
    plt.plot(epochs, hist_dict["train_loss"], 'r-o', label='Train Loss', markersize=4)
    plt.plot(epochs, hist_dict["val_loss"],   'b-o', label='Val Loss',   markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (std)')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    # (2) RMSE
    plt.subplot(3, 2, 2)
    plt.plot(epochs, hist_dict["train_rmse"], 'r-o', label='Train RMSE', markersize=4)
    plt.plot(epochs, hist_dict["val_rmse"],   'b-o', label='Val RMSE',   markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (std)')
    plt.title('RMSE')
    plt.legend()
    plt.grid(True)

    # (3) MAPE
    plt.subplot(3, 2, 3)
    plt.plot(epochs, hist_dict["train_mape"], 'r-o', label='Train MAPE', markersize=4)
    plt.plot(epochs, hist_dict["val_mape"],   'b-o', label='Val MAPE',   markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE')
    plt.legend()
    plt.grid(True)

    # (4) R^2
    plt.subplot(3, 2, 4)
    plt.plot(epochs, hist_dict["train_r2"], 'r-o', label='Train R^2', markersize=4)
    plt.plot(epochs, hist_dict["val_r2"],   'b-o', label='Val R^2',   markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('R^2')
    plt.title('R^2')
    plt.legend()
    plt.grid(True)

    # (5) SMAPE
    plt.subplot(3, 2, 5)
    plt.plot(epochs, hist_dict["train_smape"], 'r-o', label='Train SMAPE', markersize=4)
    plt.plot(epochs, hist_dict["val_smape"],   'b-o', label='Val SMAPE',   markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('SMAPE (%)')
    plt.title('SMAPE')
    plt.legend()
    plt.grid(True)

    # (6) MAE
    plt.subplot(3, 2, 6)
    plt.plot(epochs, hist_dict["train_mae"], 'r-o', label='Train MAE', markersize=4)
    plt.plot(epochs, hist_dict["val_mae"],   'b-o', label='Val MAE',   markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('MAE (std)')
    plt.title('MAE')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"Training Curves for {model_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


# =====================================================================
#   8. 主函数
# =====================================================================
def main(use_log_transform=True, min_egrid_threshold=1.0):
    print("[Info] 1) Loading raw data...")
    data_df = load_data()

    # -- 画热力图前，不先过滤 E_grid=0
    feature_cols_to_plot = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_grid'
    ]
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns]
    plot_correlation_heatmap(data_df, feature_cols_to_plot, title="Heat map (including E_grid=0)")

    # -- 过滤掉 E_grid=0 的行
    data_df = data_df[data_df['E_grid'] != 0].copy()
    data_df.reset_index(drop=True, inplace=True)

    # -- 特征工程（不做标准化）
    data_df, feature_cols, target_col = feature_engineering(data_df)

    # -- 再过滤过小值
    data_df = data_df[data_df[target_col] > min_egrid_threshold].copy()
    data_df.reset_index(drop=True, inplace=True)

    analyze_target_distribution(data_df, target_col)
    plot_Egrid_over_time(data_df)

    # -- 根据时间序列切分 train / val / test
    X_all_raw = data_df[feature_cols].values
    y_all_raw = data_df[target_col].values
    timestamps_all = data_df['timestamp'].values

    total_samples = len(data_df)
    train_size = int(0.8 * total_samples)
    val_size   = int(0.1 * total_samples)
    test_size  = total_samples - train_size - val_size

    X_train_raw = X_all_raw[:train_size]
    y_train_raw = y_all_raw[:train_size]
    X_val_raw   = X_all_raw[train_size : train_size + val_size]
    y_val_raw   = y_all_raw[train_size : train_size + val_size]
    X_test_raw  = X_all_raw[train_size + val_size:]
    y_test_raw  = y_all_raw[train_size + val_size:]

    train_timestamps = timestamps_all[:train_size]
    val_timestamps   = timestamps_all[train_size : train_size + val_size]
    test_timestamps  = timestamps_all[train_size + val_size:]

    # -- 可选：对目标做对数变换
    if use_log_transform:
        y_train_raw = np.log1p(y_train_raw)
        y_val_raw   = np.log1p(y_val_raw)
        y_test_raw  = np.log1p(y_test_raw)

    # -- 分别在 train 上 fit 标准化器，val/test 只 transform
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train = scaler_X.transform(X_train_raw)
    X_val   = scaler_X.transform(X_val_raw)
    X_test  = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler().fit(y_train_raw.reshape(-1,1))
    y_train  = scaler_y.transform(y_train_raw.reshape(-1,1)).ravel()
    y_val    = scaler_y.transform(y_val_raw.reshape(-1,1)).ravel()
    y_test   = scaler_y.transform(y_test_raw.reshape(-1,1)).ravel()

    # -- 序列数据构造
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   window_size)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  window_size)

    print(f"[Info] Train/Val/Test sizes: {X_train_seq.shape[0]}, {X_val_seq.shape[0]}, {X_test_seq.shape[0]}")
    print(f"[Info] Feature dim: {X_train_seq.shape[-1]}, window_size: {window_size}")

    train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
    val_dataset   = TensorDataset(torch.from_numpy(X_val_seq),   torch.from_numpy(y_val_seq))
    test_dataset  = TensorDataset(torch.from_numpy(X_test_seq),  torch.from_numpy(y_test_seq))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -- 构建模型
    feature_dim = X_train_seq.shape[-1]
    modelA = EModel_FeatureWeight(feature_dim).to(device)
    modelB = EModel_CNN_Transformer(feature_dim).to(device)

    # -- 训练 EModel_FeatureWeight
    print("\n========== Train EModel_FeatureWeight ==========")
    histA = train_model(modelA, train_loader, val_loader, model_name='EModel_FeatureWeight')

    # -- 训练 EModel_CNN_Transformer
    print("\n========== Train EModel_CNN_Transformer ==========")
    histB = train_model(modelB, train_loader, val_loader, model_name='EModel_CNN_Transformer')

    # -- 加载最优权重
    best_modelA = EModel_FeatureWeight(feature_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth'))

    best_modelB = EModel_CNN_Transformer(feature_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_CNN_Transformer.pth'))

    # -- 测试集评估
    criterion_test = nn.SmoothL1Loss(beta=1.0)
    (_, test_rmseA_std, test_mapeA_std, test_r2A_std, test_smapeA_std, test_maeA_std, predsA_std, labelsA_std) = evaluate(best_modelA, test_loader, criterion_test)
    (_, test_rmseB_std, test_mapeB_std, test_r2B_std, test_smapeB_std, test_maeB_std, predsB_std, labelsB_std) = evaluate(best_modelB, test_loader, criterion_test)

    print("\n========== [Test in Standardized Domain] ==========")
    print(f"[EModel_FeatureWeight]  RMSE: {test_rmseA_std:.4f}, MAPE: {test_mapeA_std:.2f}, R^2: {test_r2A_std:.4f}, "
          f"SMAPE: {test_smapeA_std:.2f}, MAE: {test_maeA_std:.4f}")
    print(f"[EModel_CNN_Transformer] RMSE: {test_rmseB_std:.4f}, MAPE: {test_mapeB_std:.2f}, R^2: {test_r2B_std:.4f}, "
          f"SMAPE: {test_smapeB_std:.2f}, MAE: {test_maeB_std:.4f}")

    # -- 反标准化 + (可选)反log
    predsA_real_std = scaler_y.inverse_transform(predsA_std.reshape(-1,1)).ravel()
    predsB_real_std = scaler_y.inverse_transform(predsB_std.reshape(-1,1)).ravel()
    labelsA_real_std = scaler_y.inverse_transform(labelsA_std.reshape(-1,1)).ravel()
    labelsB_real_std = scaler_y.inverse_transform(labelsB_std.reshape(-1,1)).ravel()

    if use_log_transform:
        predsA_real = np.expm1(predsA_real_std)
        predsB_real = np.expm1(predsB_real_std)
        labelsA_real = np.expm1(labelsA_real_std)
        labelsB_real = np.expm1(labelsB_real_std)
    else:
        predsA_real = predsA_real_std
        predsB_real = predsB_real_std
        labelsA_real = labelsA_real_std
        labelsB_real = labelsB_real_std

    # -- 在原域上计算RMSE
    from sklearn.metrics import mean_squared_error
    test_rmseA_real = np.sqrt(mean_squared_error(labelsA_real, predsA_real))
    test_rmseB_real = np.sqrt(mean_squared_error(labelsB_real, predsB_real))

    print("\n========== [Test in Real Domain] ==========")
    print(f"[EModel_FeatureWeight] => RMSE(real): {test_rmseA_real:.2f}")
    print(f"[EModel_CNN_Transformer] => RMSE(real): {test_rmseB_real:.2f}")

    # -- 可视化(以modelA为例)
    plot_test_predictions_over_time(test_timestamps[window_size:], labelsA_real, predsA_real)
    plot_predictions_comparison(
        y_actual_real      = labelsA_real,
        y_pred_model1_real = predsA_real,
        y_pred_model2_real = predsB_real,
        model1_name='EModel_FeatureWeight',
        model2_name='EModel_CNN_Transformer'
    )

    # -- 绘制训练曲线(包括 Loss / RMSE / MAPE / R^2 / SMAPE / MAE)
    plot_training_curves_allmetrics(histA, model_name='EModel_FeatureWeight')
    plot_training_curves_allmetrics(histB, model_name='EModel_CNN_Transformer')

    print("[Info] Done!")


if __name__ == "__main__":
    main(use_log_transform=True, min_egrid_threshold=1.0)