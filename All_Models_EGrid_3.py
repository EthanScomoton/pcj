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

# ---------------------------
# 全局字体及样式设置
# ---------------------------
mpl.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# ---------------------------
# 0. 全局超参数 (做了若干改动)
# ---------------------------
learning_rate = 1e-4    # 调低学习率
num_epochs    = 300     # 增加最大训练轮数
batch_size    = 128
weight_decay  = 1e-6
patience      = 20      # 加大 Early Stopping 的耐心
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
    加载原始数据，并按时间戳merge。
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
    - E_grid 做 EWMA 平滑
    - 时间特征
    - 分类特征 LabelEncoder
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
#   4. 模型定义 (加大Dropout)
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
    def __init__(self, d_model, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dropout=0.2):
        super(Transformer, self).__init__()
        self.encoder_pe = PositionalEncoding(d_model)
        self.decoder_pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, tgt):
        src_enc = self.encoder_pe(src)
        memory  = self.transformer_encoder(src_enc)
        tgt_enc = self.decoder_pe(tgt)
        out     = self.transformer_decoder(tgt_enc, memory)
        out     = self.norm(out)
        return out


class CNNBlock(nn.Module):
    def __init__(self, feature_dim, hidden_size, dropout=0.2):
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
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        attn_weights = self.attention(x)       # [batch_size, seq_len, 1]
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = x * attn_weights
        return torch.sum(weighted, dim=1)


class EModel_FeatureWeight(nn.Module):
    def __init__(self, feature_dim, dropout=0.2):
        super(EModel_FeatureWeight, self).__init__()
        self.feature_dim = feature_dim
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.transformer_block = Transformer(
            d_model=2*128, 
            nhead=4, 
            num_encoder_layers=2, 
            num_decoder_layers=2,
            dropout=dropout
        )
        self.attention = Attention(input_dim=2*128)

        self.fc = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer_block(lstm_out, lstm_out)
        attn_out = self.attention(transformer_out)
        out = self.fc(attn_out)
        return out.squeeze(-1)


class EModel_CNN_Transformer(nn.Module):
    def __init__(self, feature_dim, hidden_size=128, num_layers=2, dropout=0.2):
        super(EModel_CNN_Transformer, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)
        self.cnn_block = CNNBlock(feature_dim, hidden_size, dropout)

        self.transformer_block = Transformer(
            d_model=2*hidden_size,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=dropout
        )
        self.attention = Attention(input_dim=2*hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, 2 * hidden_size),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)
        residual = self.residual(x[:, -1, :])
        cnn_out = self.cnn_block(x)
        transformer_out = self.transformer_block(cnn_out, cnn_out)
        transformer_out = transformer_out + residual.unsqueeze(1)
        attn_out = self.attention(transformer_out)
        out = self.fc(attn_out)
        return out.squeeze(-1)


# =====================================================================
#   5. 评估工具
# =====================================================================
def evaluate(model, dataloader, criterion):
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
#   6. 训练工具
# =====================================================================
def train_model(model, train_loader, val_loader, model_name='Model'):
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 重新计算 total_steps
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
    val_loss_history   = []

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
        train_loss_history.append(train_loss_epoch)

        # === 验证 ===
        val_loss_eval, _, _, _, _, _, _, _ = evaluate(model, val_loader, criterion)
        val_loss_history.append(val_loss_eval)

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"TrainLoss: {train_loss_epoch:.4f}, "
              f"ValLoss: {val_loss_eval:.4f}")

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

    return train_loss_history, val_loss_history


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


def plot_train_val_loss(train_loss_history, val_loss_history, model_name="Model"):
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss_history, 'r-o', label='Train Loss', markersize=3)
    plt.plot(epochs, val_loss_history,   'b-o', label='Val Loss',   markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (std)')
    plt.title(f'Train/Val Loss for {model_name}')
    plt.legend()
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


# =====================================================================
#   8. 主函数 (加入更激进的过滤策略)
# =====================================================================
def main(use_log_transform=True, min_egrid_threshold=20000, max_egrid_threshold=200000):
    print("[Info] 1) Loading raw data...")
    data_df = load_data()

    # -- (可选) 先查看在未过滤 E_grid=0 的情况下特征热力图
    feature_cols_to_plot = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_grid'
    ]
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns]
    plot_correlation_heatmap(data_df, feature_cols_to_plot, title="Heat map (including E_grid=0)")

    # -- 过滤掉 E_grid <= 0 的行
    data_df = data_df[data_df['E_grid'] > 0].copy()
    data_df.reset_index(drop=True, inplace=True)

    # -- 基础特征工程
    data_df, feature_cols, target_col = feature_engineering(data_df)

    # -- 更激进过滤 (demo: 限定 E_grid 在[2e4, 2e5]之间)
    data_df = data_df[(data_df[target_col] >= min_egrid_threshold) & (data_df[target_col] <= max_egrid_threshold)]
    data_df.reset_index(drop=True, inplace=True)

    # -- 分析过滤后的目标列
    analyze_target_distribution(data_df, target_col)
    plot_Egrid_over_time(data_df)

    # -- 时间序列拆分
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

    # -- 对数变换
    if use_log_transform:
        y_train_raw = np.log1p(y_train_raw)
        y_val_raw   = np.log1p(y_val_raw)
        y_test_raw  = np.log1p(y_test_raw)

    # -- 标准化
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train = scaler_X.transform(X_train_raw)
    X_val   = scaler_X.transform(X_val_raw)
    X_test  = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler().fit(y_train_raw.reshape(-1,1))
    y_train  = scaler_y.transform(y_train_raw.reshape(-1,1)).ravel()
    y_val    = scaler_y.transform(y_val_raw.reshape(-1,1)).ravel()
    y_test   = scaler_y.transform(y_test_raw.reshape(-1,1)).ravel()

    # -- 序列构造
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

    # -- 构建模型 (加大dropout，减少过拟合)
    feature_dim = X_train_seq.shape[-1]
    modelA = EModel_FeatureWeight(feature_dim, dropout=0.2).to(device)
    modelB = EModel_CNN_Transformer(feature_dim, dropout=0.2).to(device)

    # -- 训练 ModelA
    print("\n========== Train EModel_FeatureWeight ==========")
    train_lossA, val_lossA = train_model(modelA, train_loader, val_loader, model_name='EModel_FeatureWeight')

    # -- 训练 ModelB
    print("\n========== Train EModel_CNN_Transformer ==========")
    train_lossB, val_lossB = train_model(modelB, train_loader, val_loader, model_name='EModel_CNN_Transformer')

    # -- 加载最优权重
    best_modelA = EModel_FeatureWeight(feature_dim, dropout=0.2).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth'))

    best_modelB = EModel_CNN_Transformer(feature_dim, dropout=0.2).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_CNN_Transformer.pth'))

    # -- 测试集评估
    criterion_test = nn.SmoothL1Loss(beta=1.0)
    (_, test_rmseA_std, test_mapeA_std, test_r2A_std, test_smapeA_std, test_maeA_std, predsA_std, labelsA_std) = evaluate(best_modelA, test_loader, criterion_test)
    (_, test_rmseB_std, test_mapeB_std, test_r2B_std, test_smapeB_std, test_maeB_std, predsB_std, labelsB_std) = evaluate(best_modelB, test_loader, criterion_test)

    print("\n========== [Test in Standardized Domain] ==========")
    print(f"[EModel_FeatureWeight]  RMSE: {test_rmseA_std:.4f}, MAPE: {test_mapeA_std:.2f}%, R^2: {test_r2A_std:.4f}, "
          f"SMAPE: {test_smapeA_std:.2f}%, MAE: {test_maeA_std:.4f}")
    print(f"[EModel_CNN_Transformer] RMSE: {test_rmseB_std:.4f}, MAPE: {test_mapeB_std:.2f}%, R^2: {test_r2B_std:.4f}, "
          f"SMAPE: {test_smapeB_std:.2f}%, MAE: {test_maeB_std:.4f}")

    # -- 反标准化 + (可选) 反 log
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

    # -- 在原域上计算 RMSE
    test_rmseA_real = np.sqrt(mean_squared_error(labelsA_real, predsA_real))
    test_rmseB_real = np.sqrt(mean_squared_error(labelsB_real, predsB_real))

    print("\n========== [Test in Real Domain] ==========")
    print(f"[EModel_FeatureWeight] => RMSE(real): {test_rmseA_real:.2f}")
    print(f"[EModel_CNN_Transformer] => RMSE(real): {test_rmseB_real:.2f}")

    # -- 可视化 (modelA)
    plot_test_predictions_over_time(test_timestamps[window_size:], labelsA_real, predsA_real)
    plot_predictions_comparison(
        y_actual_real      = labelsA_real,
        y_pred_model1_real = predsA_real,
        y_pred_model2_real = predsB_real,
        model1_name='EModel_FeatureWeight',
        model2_name='EModel_CNN_Transformer'
    )

    # -- 训练曲线
    plot_train_val_loss(train_lossA, val_lossA, model_name='EModel_FeatureWeight')
    plot_train_val_loss(train_lossB, val_lossB, model_name='EModel_CNN_Transformer')

    print("[Info] Done!")


if __name__ == "__main__":
    # 使用更激进过滤区间 [2e4, 2e5]，并对数变换
    main(use_log_transform=True, min_egrid_threshold=20000, max_egrid_threshold=200000)