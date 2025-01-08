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

# -------------------------------------------------
# 0. 全局超参数
# -------------------------------------------------
learning_rate = 3e-4
num_epochs    = 200
batch_size    = 64
weight_decay  = 1e-4
patience      = 25
num_workers   = 0
window_size   = 5

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------
# 1. 数据加载与预处理
# -------------------------------------------------
def load_data():
    """
    读取并合并可再生能源数据和负载数据，根据自己的数据路径进行修改
    """
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data10.csv')
    load_df      = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data10.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp']      = pd.to_datetime(load_df['timestamp'])

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')
    data_df.sort_values('timestamp', inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    return data_df

def feature_engineering(data_df):
    """
    对原始 DataFrame 进行特征工程（时间特征展开、标签编码、标准化等），
    返回: data_all, feature_columns, target_column, scaler_y
    """
    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour']      = data_df['timestamp'].dt.hour
    data_df['month']     = data_df['timestamp'].dt.month

    # 周期性特征
    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month'] - 1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month'] - 1) / 12)

    # 假设下面这些是数据中的离散型特征，需要做 LabelEncoder
    renewable_features = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_PV', 'E_storage_discharge',
        'ESCFR', 'ESCFG'
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

    # ------ 将目标列改为 E_grid ------
    target_column = 'E_grid'

    # 仅选择所需的 feature_columns 和 target_column
    data_selected = data_df[feature_columns + [target_column]].copy()

    # 标准化
    scaler_X = StandardScaler()
    data_selected[feature_columns] = scaler_X.fit_transform(data_selected[feature_columns].values)

    scaler_y = StandardScaler()
    data_selected[[target_column]] = scaler_y.fit_transform(data_selected[[target_column]].values)

    data_all = data_selected.values  # [N, feature_dim + 1]
    return data_all, feature_columns, target_column, scaler_y

def create_sequences(data_all, window_size, feature_dim):
    """
    构造单步预测的序列数据:
      - 输入 X: 连续 window_size 个时间步的 feature_dim 个特征
      - 输出 y: 第 window_size+1 个时间步的目标值
    """
    X_list, y_list = [], []
    num_samples = data_all.shape[0]

    for i in range(num_samples - window_size):
        seq_x = data_all[i : i + window_size, :feature_dim]     # 连续 window_size 行的前 feature_dim 列
        seq_y = data_all[i + window_size, feature_dim]          # 第 window_size 行的第 feature_dim 列 (目标)
        X_list.append(seq_x)
        y_list.append(seq_y)

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    return X_arr, y_arr

# -------------------------------------------------
# 2. 模型结构
# -------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 注意: 与标准 Transformer 类似，但这里用了 -log(10000.0)
        div_term = torch.exp(-(torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x, step_offset=0):
        """
        x: [batch_size, seq_len, d_model]
        step_offset: 在多步预测中，如果要对新的时间步加不同位置编码，可加上 offset
        """
        seq_len = x.size(1)
        pos_enc = self.pe[step_offset : step_offset + seq_len, 0, :]
        return x + pos_enc.unsqueeze(0)  # 广播加到 x 上


class Transformer(nn.Module):
    """
    演示用的 Transformer，Encoder + Decoder 结构。
    这里仅做单步预测时，把 Decoder 输入直接设为 src；如要多步，可自行扩展。
    """
    def __init__(self, d_model, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super(Transformer, self).__init__()
        self.encoder_pe = PositionalEncoding(d_model)
        self.decoder_pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src):
        """
        src: [batch_size, seq_len, d_model]
        此处仅示意，把 src 当作 tgt (不做多步预测场景)
        """
        # Encoder
        src_enc = self.encoder_pe(src)
        memory  = self.transformer_encoder(src_enc)

        # Decoder
        tgt = self.decoder_pe(src)
        out = self.transformer_decoder(tgt, memory)
        return out

class CNNBlock(nn.Module):
    """
    三阶 CNN，用于提取时序局部信息
    """
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
        """
        x: [batch_size, seq_len, feature_dim]
        输出: [batch_size, seq_len, 2*hidden_size]
        """
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
    """
    简单可学习注意力: 先计算权重 w, 再加权求和
    """
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        先算注意力分布 => 再与 x 相乘 => 最后在 seq_len 上聚合
        """
        attn_weights = self.attention(x)        # [batch_size, seq_len, 1]
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = x * attn_weights             # 广播 => [batch_size, seq_len, input_dim]
        return torch.sum(weighted, dim=1)       # => [batch_size, input_dim]

class EModel_FeatureWeight(nn.Module):
    """
    LSTM + Transformer + Attention 的组合示例。
    1) 用可学习 feature_importance 做特征加权
    2) 用双层双向 LSTM 提取时序
    3) 用 Transformer 进一步处理
    4) 用 Attention 聚合
    5) 全连接层输出 => 预测1个值
    """
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
        # LSTM 输出是 2*128 => 作为 Transformer 的 d_model
        self.transformer_block = Transformer(
            d_model=2*128,
            nhead=4, 
            num_encoder_layers=2, 
            num_decoder_layers=2
        )
        self.attention = Attention(input_dim=2*128)
        self.fc = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, feature_dim]
        """
        # 可学习特征权重
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)

        lstm_out, _ = self.lstm(x)  # => [batch_size, seq_len, 2*128]

        transformer_out = self.transformer_block(lstm_out)
        attn_out        = self.attention(transformer_out)
        out             = self.fc(attn_out)
        return out.squeeze(-1)

class EModel_CNN_Transformer(nn.Module):
    """
    CNN + Transformer + Attention 的组合示例
    1) 可学习特征权重
    2) 三阶 CNN 提取时序
    3) Transformer 处理 (假设 src=tgt)
    4) Attention
    5) 全连接层输出
    """
    def __init__(self, feature_dim, hidden_size=128, num_layers=2, dropout=0.2):
        super(EModel_CNN_Transformer, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)
        self.cnn_block = CNNBlock(feature_dim, hidden_size, dropout)

        # Transformer: d_model=2*hidden_size
        self.transformer_block = Transformer(
            d_model=2*hidden_size,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        self.attention = Attention(input_dim=2*hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, feature_dim]
        """
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)

        # 三阶 CNN
        cnn_out = self.cnn_block(x)  # => [batch_size, seq_len, 2*hidden_size]

        # 在单步预测场景下，把 cnn_out 当作 src/tgt 都给 Transformer
        src = cnn_out
        tgt = cnn_out
        transformer_out = self.transformer_block(src)  # 这里为了示例仅传 src
        # 若严格要 src、tgt 分离，可在 Transformer里改签名 forward(src,tgt)

        # Attention
        attn_out = self.attention(transformer_out)  # => [batch_size, 2*hidden_size]

        # 全连接
        out = self.fc(attn_out)  # => [batch_size, 1]
        return out.squeeze(-1)

# -------------------------------------------------
# 3. 训练与评价工具
# -------------------------------------------------
def evaluate(model, dataloader, criterion):
    """
    在验证集 / 测试集上进行评估
    返回: val_loss, rmse_std, mape_std, r2_std, preds_arr, labels_arr
    """
    model.eval()
    running_loss, num_samples = 0.0, 0
    preds_list, labels_list   = [], []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_inputs)  # => [batch_size]
            loss = criterion(outputs, batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples  += batch_inputs.size(0)

            preds_list.append(outputs.cpu().numpy()) 
            labels_list.append(batch_labels.cpu().numpy())

    val_loss   = running_loss / num_samples
    preds_arr  = np.concatenate(preds_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)

    # ---- 1. 计算 RMSE（标准化域下）----
    rmse_std = np.sqrt(mean_squared_error(labels_arr, preds_arr))

    # ---- 2. 计算 MAPE（标准化域下）----
    #  如果标签可能有 0，需要小心，这里演示未做过滤
    mape_std = np.mean(np.abs((labels_arr - preds_arr) / labels_arr)) * 100.0

    # ---- 3. 计算 R^2（标准化域下）----
    ss_res = np.sum((labels_arr - preds_arr)**2)
    ss_tot = np.sum((labels_arr - np.mean(labels_arr))**2)
    r2_std = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return val_loss, rmse_std, mape_std, r2_std, preds_arr, labels_arr

def train_model(model, train_loader, val_loader, model_name='Model', feature_names=None):
    """
    单步预测的训练流程 (可扩展)
    """
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps  = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    
    def lr_lambda(current_step):
        # 先 warmup 再线性衰减
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
    val_mape_history   = []
    val_r2_history     = []
    feature_importance_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss, num_samples = 0.0, 0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            preds = model(batch_inputs)  # => [batch_size]
            loss  = criterion(preds, batch_labels)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples  += batch_inputs.size(0)
            global_step  += 1

        train_loss = running_loss / num_samples
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

        # 若模型中有 feature_importance 参数，可以在此记录
        if hasattr(model, 'feature_importance'):
            current_fi = model.feature_importance.detach().cpu().numpy().copy()
            feature_importance_history.append(current_fi)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
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
        val_mape_history,
        val_r2_history,
        feature_importance_history
    )

# -------------------------------------------------
# 4. 可视化
# -------------------------------------------------
def plot_predictions_comparison(y_actual_real, y_pred_model1_real, y_pred_model2_real, model1_name='Model1', model2_name='Model2'):
    """
    比较两个模型的预测值与真实值，可视化折线图
    """
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
    """
    简单版本：在同一张图上绘制 训练/验证Loss、验证RMSE
    """
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
    """
    对比两个模型在每个 Epoch 验证集 RMSE 的曲线
    """
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

def plot_correlation_heatmap(df, feature_cols, title="Heat map of different loads and external factors"):
    """
    在原始 DataFrame 上对指定的列做相关性可视化
    """
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

def analyze_target_distribution(data_df, target_col):
    """
    在原始 DataFrame 中对目标列进行直方图可视化、描述性统计
    """
    if target_col not in data_df.columns:
        print(f"[Warning] '{target_col}' is not in data_df columns. Skip analysis.")
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

def plot_training_curves_extended(train_loss_history, 
                                  val_loss_history, 
                                  val_rmse_history,
                                  val_mape_history,
                                  val_r2_history,
                                  model_name='Model'):
    """
    扩展版本：分别对 Loss、RMSE、MAPE、R^2 四个指标单独作图
    """
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

# -------------------------------------------------
# 5. 主函数
# -------------------------------------------------
def main():
    print("[Info] Loading and preprocessing data...")
    data_df = load_data()

    # (1) 可选：查看原始 DataFrame 的部分信息
    # print(data_df.head())

    # (2) 画热力图，查看相关性
    feature_cols_to_plot = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_grid'
    ]
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns]
    plot_correlation_heatmap(data_df, feature_cols_to_plot)

    # (3) 分析目标列在原始域下的分布
    analyze_target_distribution(data_df, "E_grid")

    # (4) 特征工程 + 序列构建
    data_all, feature_cols, target_col, scaler_y = feature_engineering(data_df)
    feature_dim = len(feature_cols)
    print(f"[Info] Model will predict target column: {target_col}")

    X_all, y_all = create_sequences(data_all, window_size=window_size, feature_dim=feature_dim)
    print(f"[Info] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")

    total_samples = X_all.shape[0]
    train_size    = int(0.8 * total_samples)
    val_size      = int(0.1 * total_samples)
    test_size     = total_samples - train_size - val_size
    print(f"[Info] Train: {train_size}, Val: {val_size}, Test: {test_size}")

    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val,   y_val   = X_all[train_size : train_size+val_size], y_all[train_size : train_size+val_size]
    X_test,  y_test  = X_all[train_size+val_size:], y_all[train_size+val_size:]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # (5) 构建两种模型
    print("[Info] Building models...")
    modelA = EModel_FeatureWeight(feature_dim).to(device)
    modelB = EModel_CNN_Transformer(feature_dim).to(device)

    # (6) 训练
    print("\n========== Train EModel_FeatureWeight ==========")
    train_lossA, val_lossA, val_rmseA, _, _, _ = train_model(
        modelA, train_loader, val_loader, 
        model_name='EModel_FeatureWeight', 
        feature_names=feature_cols  
    )

    print("\n========== Train EModel_CNN_Transformer ==========")
    train_lossB, val_lossB, val_rmseB, _, _, _ = train_model(
        modelB, train_loader, val_loader, 
        model_name='EModel_CNN_Transformer', 
        feature_names=feature_cols
    )

    # (7) 加载最优权重并测试
    best_modelA = EModel_FeatureWeight(feature_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth', map_location=device))

    best_modelB = EModel_CNN_Transformer(feature_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_CNN_Transformer.pth', map_location=device))

    print("\n[Info] Testing...")
    criterion = nn.SmoothL1Loss(beta=1.0)

    # Evaluate A
    test_lossA, test_rmseA_std, predsA_std, labelsA_std, _, _ = evaluate(best_modelA, test_loader, criterion)
    # Evaluate B
    test_lossB, test_rmseB_std, predsB_std, labelsB_std, _, _ = evaluate(best_modelB, test_loader, criterion)

    print("\n========== Test Results (standardized domain) ==========")
    print(f"[EModel_FeatureWeight] => Test Loss(std): {test_lossA:.4f}, RMSE(std): {test_rmseA_std:.4f}")
    print(f"[EModel_CNN_Transformer]         => Test Loss(std): {test_lossB:.4f}, RMSE(std): {test_rmseB_std:.4f}")

    # (8) 反标准化 => 计算原域下 RMSE
    predsA_real  = scaler_y.inverse_transform(predsA_std.reshape(-1,1)).ravel()
    predsB_real  = scaler_y.inverse_transform(predsB_std.reshape(-1,1)).ravel()
    labelsA_real = scaler_y.inverse_transform(labelsA_std.reshape(-1,1)).ravel()
    labelsB_real = scaler_y.inverse_transform(labelsB_std.reshape(-1,1)).ravel()

    test_rmseA_real = np.sqrt(mean_squared_error(labelsA_real, predsA_real))
    test_rmseB_real = np.sqrt(mean_squared_error(labelsB_real, predsB_real))

    print("\n========== Test Results (real domain) ==========")
    print(f"[EModel_FeatureWeight] => RMSE(real): {test_rmseA_real:.4f}")
    print(f"[EModel_CNN_Transformer]         => RMSE(real): {test_rmseB_real:.4f}")

    # (9) 再训练一次，收集扩展指标 (MAPE, R^2)，并可做可视化
    print("\n========== Train EModel_FeatureWeight (with extended metrics) ==========")
    train_lossA, val_lossA, val_rmseA, val_mapeA, val_r2A, _ = train_model(
        modelA, train_loader, val_loader, 
        model_name='EModel_FeatureWeight', 
        feature_names=feature_cols  
    )

    print("\n========== Train EModel_CNN_Transformer (with extended metrics) ==========")
    train_lossB, val_lossB, val_rmseB, val_mapeB, val_r2B, _ = train_model(
        modelB, train_loader, val_loader, 
        model_name='EModel_CNN_Transformer', 
        feature_names=feature_cols
    )

    # (10) 可视化比较预测结果
    plot_predictions_comparison(
        y_actual_real      = labelsA_real,
        y_pred_model1_real = predsA_real,
        y_pred_model2_real = predsB_real,
        model1_name        = 'EModel_FeatureWeight',
        model2_name        = 'EModel_CNN_Transformer'
    )

    # (11) 训练曲线可视化 (简易版本)
    plot_training_curves(train_lossA, val_lossA, val_rmseA, model_name='EModel_FeatureWeight')
    plot_training_curves(train_lossB, val_lossB, val_rmseB, model_name='EModel_CNN_Transformer')

    # (12) 两个模型在验证集 RMSE 对比
    plot_two_model_val_rmse(val_rmseA, val_rmseB, labelA='EModel_FeatureWeight', labelB='EModel_CNN_Transformer')

    # (13) 扩展版本训练曲线 (含 MAPE, R^2)
    plot_training_curves_extended(
        train_lossA, 
        val_lossA, 
        val_rmseA, 
        val_mapeA, 
        val_r2A,
        model_name='EModel_FeatureWeight'
    )
    plot_training_curves_extended(
        train_lossB, 
        val_lossB, 
        val_rmseB, 
        val_mapeB, 
        val_r2B,
        model_name='EModel_CNN_Transformer'
    )

    print("[Info] Done!")

if __name__ == "__main__":
    main()