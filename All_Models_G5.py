#更新了BiGRU模块，将第一段代码（分类任务的 BiGRU）的“显式零初始化 hidden state”以及“双向 GRU 的核心 forward 流程”融入到了第二段回归任务的 EModel_BiGRU 中。
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors  # 自定义颜色

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from torch.nn.utils import clip_grad_norm_

# ---------------------------
# 0. 全局超参数
# ---------------------------
learning_rate = 1e-5
num_epochs    = 100
batch_size    = 128
weight_decay  = 1e-4
patience      = 10
num_workers   = 0
window_size   = 3

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_data():
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data10.csv')
    load_df      = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data10.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp']      = pd.to_datetime(load_df['timestamp'])

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')
    data_df.sort_values('timestamp', inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    return data_df

def feature_engineering(data_df):
    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour']      = data_df['timestamp'].dt.hour
    data_df['month']     = data_df['timestamp'].dt.month

    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month'] - 1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month'] - 1) / 12)

    renewable_features = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_PV', 'E_storage_discharge',
        'E_grid', 'ESCFR', 'ESCFG'
    ]
    load_features = ['ship_grade', 'dock_position', 'destination']

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
    target_column   = 'energyconsumption'

    data_selected = data_df[feature_columns + [target_column]].copy()

    scaler_X = StandardScaler()
    data_selected[feature_columns] = scaler_X.fit_transform(data_selected[feature_columns].values)

    scaler_y = StandardScaler()
    data_selected[[target_column]] = scaler_y.fit_transform(data_selected[[target_column]].values)

    data_all = data_selected.values
    return data_all, feature_columns, target_column, scaler_y

def create_sequences(data_all, window_size, feature_dim):
    X_list, y_list = [], []
    num_samples = data_all.shape[0]

    for i in range(num_samples - window_size):
        seq_x = data_all[i : i + window_size, :feature_dim]
        seq_y = data_all[i + window_size, feature_dim]
        X_list.append(seq_x)
        y_list.append(seq_y)

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    return X_arr, y_arr

# ---------------------------
# 2. 模型结构
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(-(torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, 0, :]

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, d_model, nhead=8, num_encoder_layers=2, num_decoder_layers=2):
        super(EncoderDecoderTransformer, self).__init__()
        self.encoder_pe = PositionalEncoding(d_model)
        self.decoder_pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src):
        # Encoder
        src_enc = self.encoder_pe(src)
        memory  = self.transformer_encoder(src_enc)

        # Decoder (此处仅做示意，如果是多步预测需修改 tgt)
        tgt = self.decoder_pe(src)
        out = self.transformer_decoder(tgt, memory)
        return out

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
        attn_weights = self.attention(x)
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = x * attn_weights
        return torch.sum(weighted, dim=1)

class EModel_FeatureWeight(nn.Module):
    def __init__(self, feature_dim):
        super(EModel_FeatureWeight, self).__init__()
        self.feature_dim = feature_dim
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.transformer_block = EncoderDecoderTransformer(
            d_model=2*128, nhead=8, num_encoder_layers=2, num_decoder_layers=2
        )
        self.attention = Attention(input_dim=2*128)
        self.fc = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 利用可学习的特征权重
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer_block(lstm_out)
        attn_out = self.attention(transformer_out)
        out = self.fc(attn_out)
        return out.squeeze(-1)

class EModel_BiGRU(nn.Module):
    def __init__(self, feature_dim, hidden_size=128, num_layers=2, dropout=0.3):
        super(EModel_BiGRU, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 可学习的特征权重
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)

        self.bigru = nn.GRU(
            input_size=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.transformer_block = EncoderDecoderTransformer(
            d_model=2 * self.hidden_size,  # 双向，因此是 2 * hidden_size
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        self.attention = Attention(input_dim=2 * self.hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 利用可学习的特征权重
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)

        # 显式初始化 hidden state，形状: [num_layers * 2, batch_size, hidden_size]
        # 注意：需要将 hidden state 放到与 x 相同的 device 上
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # 与第一段 BiGRU 类似的调用方式
        gru_out, _ = self.bigru(x, h0)

        transformer_out = self.transformer_block(gru_out)
        attn_out = self.attention(transformer_out)
        out = self.fc(attn_out)    # 维度: [batch_size, 1]

        # 用 squeeze(-1) 将 [batch_size, 1] 变成 [batch_size]
        return out.squeeze(-1)

# ---------------------------
# 3. 训练与评价工具
# ---------------------------
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
    preds_arr  = np.concatenate(preds_list, axis=0)  # shape: [num_samples,]
    labels_arr = np.concatenate(labels_list, axis=0) # shape: [num_samples,]

    # ---- 1. 计算 RMSE（标准化域下）----
    rmse_std = np.sqrt(mean_squared_error(labels_arr, preds_arr))

    # ---- 2. 计算 MAPE（标准化域下）----
    # 注意：如果某些 y_i = 0，会导致除零。通常可加个极小值或过滤，但以下示例不额外处理。
    mape_std = np.mean(np.abs((labels_arr - preds_arr) / labels_arr)) * 100.0

    # ---- 3. 计算 R^2（标准化域下）----
    ss_res = np.sum((labels_arr - preds_arr)**2)
    ss_tot = np.sum((labels_arr - np.mean(labels_arr))**2)
    r2_std = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return val_loss, rmse_std, mape_std, r2_std, preds_arr, labels_arr


def train_model(model, train_loader, val_loader, model_name='Model', feature_names=None):
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    train_loss_history = []
    val_loss_history   = []
    val_rmse_history   = []
    val_mape_history   = []  # ← 新增
    val_r2_history     = []  # ← 新增

    feature_importance_history = []

    for epoch in range(num_epochs):
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

        # === 训练集平均损失 ===
        train_loss = running_loss / num_samples
        # === 验证集指标 ===
        val_loss, val_rmse_std, val_mape_std, val_r2_std, _, _ = evaluate(model, val_loader, criterion)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_rmse_history.append(val_rmse_std)
        val_mape_history.append(val_mape_std)
        val_r2_history.append(val_r2_std)

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss(std): {train_loss:.4f}, "
              f"Val Loss(std): {val_loss:.4f}, "
              f"RMSE(std): {val_rmse_std:.4f}, "
              f"MAPE(%): {val_mape_std:.2f}, "
              f"R^2: {val_r2_std:.4f}")

        if hasattr(model, 'feature_importance'):
            current_fi = model.feature_importance.detach().cpu().numpy().copy()
            feature_importance_history.append(current_fi)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter       = 0
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            print(f"[{model_name}] 模型已保存。")
        else:
            counter += 1
            if counter >= patience:
                print(f"[{model_name}] 验证集无改善，提前停止。")
                break

    return (
        train_loss_history, 
        val_loss_history, 
        val_rmse_history,
        val_mape_history,   # ← 返回 MAPE
        val_r2_history,     # ← 返回 R^2
        feature_importance_history
    )

# ---------------------------
# 4. 可视化
# ---------------------------
def plot_predictions_comparison(y_actual_real, y_pred_model1_real, y_pred_model2_real, model1_name='Model1', model2_name='Model2'):
    plt.figure(figsize=(10,5))
    x_axis = np.arange(len(y_actual_real))

    plt.plot(x_axis, y_actual_real,        'r-o',  label='Actual', linewidth=1)
    plt.plot(x_axis, y_pred_model1_real,   'g--*', label=model1_name, linewidth=1)
    plt.plot(x_axis, y_pred_model2_real,   'b-.*', label=model2_name, linewidth=1)
    plt.xlabel('Index')
    plt.ylabel('Value (real domain)')
    plt.title(f'Comparison: Actual vs {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_loss_history, val_loss_history, val_rmse_history, model_name='Model'):
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(10,5))

    plt.plot(epochs, train_loss_history, 'r-o',  label='Train Loss(std)')
    plt.plot(epochs, val_loss_history,   'b-o',  label='Val Loss(std)')
    plt.plot(epochs, val_rmse_history,   'g--*', label='Val RMSE(std)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss / RMSE (standardized domain)')
    plt.title(f'Training Curves for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_two_model_val_rmse(val_rmseA, val_rmseB, labelA='ModelA', labelB='ModelB'):
    epochsA = range(1, len(val_rmseA) + 1)
    epochsB = range(1, len(val_rmseB) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochsA, val_rmseA, 'r-o', label=f'{labelA} Val RMSE (std)')
    plt.plot(epochsB, val_rmseB, 'b-o', label=f'{labelB} Val RMSE (std)')

    plt.xlabel('Epoch')
    plt.ylabel('RMSE (standardized domain)')
    plt.title('Validation RMSE Comparison (standardized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, feature_cols, title = "Heat map of different types of loads and external factors"):
    df_encoded = df.copy()
    for col in feature_cols:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    corr_matrix = df_encoded[feature_cols].corr()

    colors_list = ["red", "magenta", "purple"]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("my_rdpu", colors_list, N=256)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr_matrix,
        cmap=cmap_custom,
        annot=False,
        square=True,
        linewidths=1,
        linecolor='white',
        cbar=True
    )
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ===== 新增函数 & 逻辑：使用 target_col 做可视化或统计分析 =====
def analyze_target_distribution(data_df, target_col):
    """
    在原始 DataFrame（未标准化的 domain）中对目标列进行可视化或统计分析。
    这里演示绘制一个简单的直方图、查看描述统计。
    """
    if target_col not in data_df.columns:
        print(f"[Warning] '{target_col}' is not in data_df columns. Skip analysis.")
        return

    # 打印基本统计信息
    print(f"\n[Target Analysis] Basic stats of '{target_col}':")
    print(data_df[target_col].describe())

    # 绘制目标列的直方图
    plt.figure(figsize=(6,4))
    plt.hist(data_df[target_col], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of '{target_col}' in original domain")
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_curves_extended(
    train_loss_history, 
    val_loss_history, 
    val_rmse_history,
    val_mape_history,
    val_r2_history,
    model_name='Model'
):
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(12, 8))

    # (1) Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss_history, 'r-o', label='Train Loss(std)')
    plt.plot(epochs, val_loss_history,   'b-o', label='Val Loss(std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (std)')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    # (2) RMSE
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_rmse_history, 'g--*', label='Val RMSE(std)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (std)')
    plt.title('RMSE')
    plt.legend()
    plt.grid(True)

    # (3) MAPE
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_mape_history, 'm-*', label='Val MAPE(%)')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE')
    plt.legend()
    plt.grid(True)

    # (4) R^2
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_r2_history, 'c-o', label='Val R^2')
    plt.xlabel('Epoch')
    plt.ylabel('R^2')
    plt.title('R^2')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Training Metrics for {model_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    print("[Info] Loading and preprocessing data...")
    data_df = load_data()

    # ========== 基于原始 df 画 NxN 热力图 ==========
    feature_cols_to_plot = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'energyconsumption'
    ]
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns]
    plot_correlation_heatmap(data_df, feature_cols_to_plot)

    # ===== 在进入标准化流水线前，先对目标列做一个分布分析（使用 target_col）=====
    analyze_target_distribution(data_df, "energyconsumption")

    # ========== 原有特征工程 + 训练 + 测试逻辑 ==========
    data_all, feature_cols, target_col, scaler_y = feature_engineering(data_df)
    feature_dim = len(feature_cols)

    # 这里多了一次对 target_col 的使用，可以显示输出看一下
    print(f"[Info] Model will predict target column: {target_col}")

    X_all, y_all = create_sequences(data_all, window_size=window_size, feature_dim=feature_dim)
    print(f"[Info] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")

    total_samples = X_all.shape[0]
    train_size    = int(0.8 * total_samples)
    val_size      = int(0.1 * total_samples)
    test_size     = total_samples - train_size - val_size

    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val,   y_val   = X_all[train_size : train_size+val_size], y_all[train_size : train_size+val_size]
    X_test,  y_test  = X_all[train_size+val_size:], y_all[train_size+val_size:]
    print(f"[Info] Split data => Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("[Info] Building models...")
    modelA = EModel_FeatureWeight(feature_dim).to(device)
    modelB = EModel_BiGRU(feature_dim).to(device)

    print("\n========== Train EModel_FeatureWeight ==========")
    (train_lossA, val_lossA, val_rmseA, _) = train_model(
        modelA, train_loader, val_loader, 
        model_name='EModel_FeatureWeight', 
        feature_names=feature_cols  # 虽然目前内部未用，但示例保留
    )

    print("\n========== Train EModel_BiGRU ==========")
    (train_lossB, val_lossB, val_rmseB, _) = train_model(
        modelB, train_loader, val_loader, 
        model_name='EModel_BiGRU', 
        feature_names=feature_cols
    )

    # 加载最优权重 & 测试
    best_modelA = EModel_FeatureWeight(feature_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth', weights_only=True))
    best_modelB = EModel_BiGRU(feature_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_BiGRU.pth', weights_only=True))

    print("\n[Info] Testing...")
    criterion = nn.SmoothL1Loss(beta=1.0)
    test_lossA, test_rmseA_std, predsA_std, labelsA_std = evaluate(best_modelA, test_loader, criterion)
    test_lossB, test_rmseB_std, predsB_std, _           = evaluate(best_modelB, test_loader, criterion)

    print("\n========== Test Results (standardized domain) ==========")
    print(f"[EModel_FeatureWeight] => Test Loss(std): {test_lossA:.4f}, RMSE(std): {test_rmseA_std:.4f}")
    print(f"[EModel_BiGRU]         => Test Loss(std): {test_lossB:.4f}, RMSE(std): {test_rmseB_std:.4f}")

    # 反标准化
    predsA_real  = scaler_y.inverse_transform(predsA_std.reshape(-1,1)).ravel()
    predsB_real  = scaler_y.inverse_transform(predsB_std.reshape(-1,1)).ravel()
    labelsA_real = scaler_y.inverse_transform(labelsA_std.reshape(-1,1)).ravel()

    # 在原空间重新计算RMSE
    test_rmseA_real = np.sqrt(mean_squared_error(labelsA_real, predsA_real))
    test_rmseB_real = np.sqrt(mean_squared_error(labelsA_real, predsB_real))

    print("\n========== Test Results (real domain) ==========")
    print(f"[EModel_FeatureWeight] => RMSE(real): {test_rmseA_real:.4f}")
    print(f"[EModel_BiGRU]         => RMSE(real): {test_rmseB_real:.4f}")

    print("\n========== Train EModel_FeatureWeight ==========")
    (train_lossA, val_lossA, val_rmseA, val_mapeA, val_r2A, _) = train_model(
        modelA, train_loader, val_loader, 
        model_name='EModel_FeatureWeight', 
        feature_names=feature_cols  
    )

    print("\n========== Train EModel_BiGRU ==========")
    (train_lossB, val_lossB, val_rmseB, val_mapeB, val_r2B, _) = train_model(
        modelB, train_loader, val_loader, 
        model_name='EModel_BiGRU', 
        feature_names=feature_cols
    )

    # 可视化对比
    plot_predictions_comparison(
        y_actual_real      = labelsA_real,
        y_pred_model1_real = predsA_real,
        y_pred_model2_real = predsB_real,
        model1_name        = 'EModel_FeatureWeight',
        model2_name        = 'EModel_BiGRU'
    )
    plot_training_curves(train_lossA, val_lossA, val_rmseA, model_name='EModel_FeatureWeight')
    plot_training_curves(train_lossB, val_lossB, val_rmseB, model_name='EModel_BiGRU')
    plot_two_model_val_rmse(val_rmseA, val_rmseB, labelA='EModel_FeatureWeight', labelB='EModel_BiGRU')

    plot_training_curves_extended(
        train_lossA, 
        val_lossA, 
        val_rmseA, 
        val_mapeA, 
        val_r2A,
        model_name='EModel_FeatureWeight'
    )

    # 同理 B 模型
    plot_training_curves_extended(
        train_lossB, 
        val_lossB, 
        val_rmseB, 
        val_mapeB, 
        val_r2B,
        model_name='EModel_BiGRU'
    )

    print("[Info] Done!")

if __name__ == "__main__":
    main()
