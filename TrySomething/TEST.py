import math
import numpy as np
import pandas as pd
import pywt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors  # 颜色
import matplotlib as mpl  # 用于统一修改全局字体大小

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from torch.nn.utils import clip_grad_norm_
from sklearn.decomposition import PCA  # 用于示例性的降维
from scipy import stats  # 用于示例性的异常值处理

# ---------------------------
# 全局字体及样式设置
# ---------------------------
mpl.rcParams.update({
    'font.size': 16,        # 整体文字大小
    'axes.labelsize': 16,   # 坐标轴标签文字大小
    'axes.titlesize': 18,   # 图表标题文字大小
    'xtick.labelsize': 14,  # x 轴刻度文字大小
    'ytick.labelsize': 14   # y 轴刻度文字大小
})

# ---------------------------
# 0. 全局超参数
# ---------------------------
learning_rate = 1e-4
num_epochs    = 150
batch_size    = 128
weight_decay  = 1e-6
patience      = 5
num_workers   = 0
window_size   = 20
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ 数据预处理辅助函数 ============

def fill_missing_values(df, method='ffill'):
    if method in ['ffill', 'bfill']:
        df.fillna(method=method, inplace=True)
    elif method == 'mean':
        for col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
    # 其他填充策略可自行扩展
    return df

def remove_outliers_by_zscore(df, cols, threshold=3.0):
    for col in cols:
        if col not in df.columns:
            continue
        # 去除缺失值，否则 stats.zscore 会报错
        df_clean = df[col].dropna()
        z_scores = np.abs(stats.zscore(df_clean))
        # 标记需要过滤的行
        filter_idx = z_scores > threshold
        # 将超过阈值的行置为 np.nan 或直接删除
        outlier_indices = df_clean[filter_idx].index
        df.loc[outlier_indices, col] = np.nan
    # 对异常值处理后再进行一次缺失值处理
    df = fill_missing_values(df, method='ffill')
    return df

def wavelet_denoising(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)
    # 将最高频率分量置零(示例性)，以实现简单降噪
    coeffs[-1] = np.zeros_like(coeffs[-1])
    reconstructed = pywt.waverec(coeffs, wavelet, mode='periodization')
    
    # 如果重构后长度不一致，就截断或填充
    if len(reconstructed) > len(signal):
        reconstructed = reconstructed[:len(signal)]
    elif len(reconstructed) < len(signal):
        # 这里简单示例用 np.pad 让它和原 signal 一样长，也可以只截断 DataFrame
        reconstructed = np.pad(
            reconstructed,
            (0, len(signal) - len(reconstructed)),
            mode='edge'
        )
    
    return reconstructed

def feature_selection_by_correlation(df, target, threshold=0.1):
    corr_matrix = df.corr()[target].abs()
    selected_features = corr_matrix[corr_matrix >= threshold].index.tolist()
    return selected_features

# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_data():
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data10.csv')
    load_df      = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data10.csv')

    # 1) 转换时间格式
    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp']      = pd.to_datetime(load_df['timestamp'])

    # 2) 合并
    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')
    data_df.sort_values('timestamp', inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    # 3) 数据清洗：缺失值、异常值处理
    data_df = fill_missing_values(data_df, method='ffill')

    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
    data_df = remove_outliers_by_zscore(data_df, cols=numeric_cols, threshold=3.0)
    
    # 这里还可以选择删除全部为 0 或全部为同一值的列，以保证数据质量

    return data_df

def feature_engineering(data_df):
    # 1) 特征提取：日期、时间、历史负荷等
    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour']      = data_df['timestamp'].dt.hour
    data_df['month']     = data_df['timestamp'].dt.month

    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month'] - 1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month'] - 1) / 12)

    # 2) 其他特征
    renewable_features = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_PV', 'E_wind', 'E_storage_discharge',
        'ESCFR', 'ESCFG'
    ]
    load_features = [
        'ship_grade', 'dock_position', 'destination', 'energyconsumption'
    ]

    # 对分类特征进行编码
    for col in renewable_features + load_features:
        if col in data_df.columns:
            le = LabelEncoder()
            data_df[col] = le.fit_transform(data_df[col].astype(str))

    # 3) 数据分解与去噪（对 'E_grid' 做小波去噪）
    if 'E_grid' in data_df.columns:
        e_grid_denoised = wavelet_denoising(data_df['E_grid'].values, wavelet='db4', level=1)
        
        # 确保去噪结果与 data_df 长度匹配
        if len(e_grid_denoised) != len(data_df):
            print(f"[Warning] wavelet_denoising produced length {len(e_grid_denoised)} != {len(data_df)}.")
            min_len = min(len(e_grid_denoised), len(data_df))
            e_grid_denoised = e_grid_denoised[:min_len]
            data_df = data_df.iloc[:min_len].copy()
        
        data_df['E_grid'] = e_grid_denoised

    # 4) 特征选择：根据与目标列 E_grid 的相关性进行筛选
    target_column = 'E_grid'
    base_feature_cols = renewable_features + load_features + [
        'dayofweek_sin', 'dayofweek_cos',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos'
    ]
    exist_features = [col for col in base_feature_cols if col in data_df.columns]
    
    # 先暂时将所有存在的特征与目标列合并做相关性分析
    correlation_df = data_df[exist_features + [target_column]].copy()
    selected_cols  = feature_selection_by_correlation(correlation_df, target_column, threshold=0.05)
    # 这里也可以做更严格的筛选，比如 threshold=0.1，根据实际需要调整

    # 最终用于建模的特征
    # 如果 selected_cols 中已包含 target 列，需要去除
    feature_columns = [col for col in selected_cols if col != target_column]
    
    # 也可以在这里做 PCA 或其他降维操作
    # 先做标准化
    scaler_X = StandardScaler()
    # 注意：要只对 feature_columns 做标准化
    data_df[feature_columns] = scaler_X.fit_transform(data_df[feature_columns].values.reshape(-1, len(feature_columns)))

    # 对目标列进行标准化
    scaler_y = StandardScaler()
    data_df[[target_column]] = scaler_y.fit_transform(data_df[[target_column]].values)

    # 如果想做 PCA，请在标准化之后进行
    # 例如只做一个演示：将特征降至前 8 个主成分
    # 若特征非常多才适合 PCA，这里仅作示例
    if len(feature_columns) > 8:
        pca = PCA(n_components=8)
        pca_features = pca.fit_transform(data_df[feature_columns].values)
        # 用新的特征替换原有列
        pca_cols = [f'pca_{i}' for i in range(pca_features.shape[1])]
        pca_df = pd.DataFrame(pca_features, columns=pca_cols)
        for col in feature_columns:
            data_df.drop(col, axis=1, inplace=True)
        data_df = pd.concat([data_df, pca_df], axis=1)
        # 更新新的特征列表
        feature_columns = pca_cols

    # 最终保留必要列
    data_selected = data_df[feature_columns + [target_column]].copy()
    data_all = data_selected.values

    return data_all, feature_columns, target_column, scaler_y, data_df

def create_sequences(data_all, window_size, feature_dim):
    X_list, y_list = [], []
    num_samples = data_all.shape[0]

    for i in range(num_samples - window_size):
        seq_x = data_all[i : i + window_size, :feature_dim]
        seq_y = data_all[i + window_size, feature_dim]  # 第 feature_dim 列是 target
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
        pe = pe.unsqueeze(1)  # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x, step_offset=0):
        seq_len = x.size(1)
        pos_enc = self.pe[step_offset: step_offset + seq_len, 0, :]
        return x + pos_enc.unsqueeze(0)  # [batch_size, seq_len, d_model]

class Transformer(nn.Module):
    def __init__(self, d_model, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super(Transformer, self).__init__()
        self.encoder_pe = PositionalEncoding(d_model)
        self.decoder_pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src, tgt):
        # === Encoder ===
        src_enc = self.encoder_pe(src)
        memory  = self.transformer_encoder(src_enc)

        # === Decoder ===
        tgt_enc = self.decoder_pe(tgt)
        out     = self.transformer_decoder(tgt_enc, memory)
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
        # 输入 x: [batch_size, seq_len, feature_dim]
        x = x.transpose(1, 2)  # 转换为 [batch_size, feature_dim, seq_len]
        
        x = self.conv1(x)  # [batch_size, hidden_size, seq_len]
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)  # [batch_size, hidden_size, seq_len]
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)  # [batch_size, 2 * hidden_size, seq_len]
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = x.transpose(1, 2)  # 转换回 [batch_size, seq_len, 2 * hidden_size]
        return x

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        # x 形状: [batch_size, seq_len, input_dim]
        attn_weights = self.attention(x)       # [batch_size, seq_len, 1]
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = x * attn_weights            # 广播乘法
        return torch.sum(weighted, dim=1)      # [batch_size, input_dim]

class EModel_FeatureWeight(nn.Module):
    """
    LSTM + Transformer + Attention 的组合，额外引入 feature_importance 作为可学习权重。
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
            dropout=0
        )
        self.transformer_block = Transformer(
            d_model=2*128,  # 与 LSTM 双向 => 输出 2*hidden_size
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        self.attention = Attention(input_dim=2*128)
        self.fc = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 利用可学习的特征权重
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)

        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, 2*128]

        # 传入 src 和 tgt，这里简单做法是让 tgt = lstm_out
        transformer_out = self.transformer_block(lstm_out, lstm_out)

        # Attention
        attn_out = self.attention(transformer_out)

        # 全连接
        out = self.fc(attn_out)
        return out.squeeze(-1)

class EModel_CNN_Transformer(nn.Module):
    """
    三阶 CNN + Transformer + Attention 的组合，额外引入 feature_importance 作为可学习权重。
    """
    def __init__(self, feature_dim, hidden_size=128, num_layers=2, dropout=0):
        super(EModel_CNN_Transformer, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 可学习的特征权重
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)

        # 三阶 CNN 模块
        self.cnn_block = CNNBlock(feature_dim, hidden_size, dropout)

        # Transformer模块 (Encoder + Decoder)
        self.transformer_block = Transformer(
            d_model=2 * hidden_size,  # CNN 输出的维度
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # Attention模块
        self.attention = Attention(input_dim=2 * hidden_size)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 利用可学习的特征权重
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)

        # 三阶CNN模块
        cnn_out = self.cnn_block(x)  # [batch_size, seq_len, 2 * hidden_size]

        # Transformer
        src = cnn_out  
        tgt = cnn_out  
        transformer_out = self.transformer_block(src, tgt)  # [batch_size, seq_len, 2 * hidden_size]

        # Attention
        attn_out = self.attention(transformer_out)  # [batch_size, 2 * hidden_size]

        # 全连接层
        out = self.fc(attn_out)  # [batch_size, 1]
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
    preds_arr  = np.concatenate(preds_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)

    # ---- 1. 计算 RMSE（标准化域下）----
    rmse_std = np.sqrt(mean_squared_error(labels_arr, preds_arr))

    # ---- 2. 计算 MAPE（标准化域下）----
    nonzero_mask = (labels_arr != 0)
    mape_std = (
        np.mean(np.abs((labels_arr[nonzero_mask] - preds_arr[nonzero_mask]) / labels_arr[nonzero_mask])) * 100.0 
        if np.sum(nonzero_mask) > 0 else 0.0
    )

    # ---- 3. 计算 R^2（标准化域下）----
    ss_res = np.sum((labels_arr - preds_arr) ** 2)
    ss_tot = np.sum((labels_arr - np.mean(labels_arr)) ** 2)
    r2_std = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return val_loss, rmse_std, mape_std, r2_std, preds_arr, labels_arr

def train_model(model, train_loader, val_loader, model_name='Model', feature_names=None):
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
            preds = model(batch_inputs)
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

        if hasattr(model, 'feature_importance'):
            current_fi = model.feature_importance.detach().cpu().numpy().copy()
            feature_importance_history.append(current_fi)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter       = 0
            # 注意：如果本地 PyTorch 版本低，可直接使用 torch.save(model.state_dict(), "xxx.pth")
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

# ---------------------------
# 4. 可视化
# ---------------------------
def plot_predictions_comparison(y_actual_real, y_pred_model1_real, y_pred_model2_real, model1_name='Model1', model2_name='Model2'):
    plt.figure(figsize=(10,5))
    x_axis = np.arange(len(y_actual_real))

    plt.plot(x_axis, y_actual_real,        'r-o',  label='Actual', linewidth=1, markersize=4)
    plt.plot(x_axis, y_pred_model1_real,   'g--*', label=model1_name, linewidth=1, markersize=4)
    plt.plot(x_axis, y_pred_model2_real,   'b-.*', label=model2_name, linewidth=1, markersize=4)
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

    plt.plot(epochs, train_loss_history, 'r-o',  label='Train Loss(std)', markersize=4)
    plt.plot(epochs, val_loss_history,   'b-o',  label='Val Loss(std)',   markersize=4)
    plt.plot(epochs, val_rmse_history,   'g--*', label='Val RMSE(std)',   markersize=4)

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
    plt.plot(epochsA, val_rmseA, 'r-o', label=f'{labelA} Val RMSE (std)', markersize=4)
    plt.plot(epochsB, val_rmseB, 'b-o', label=f'{labelB} Val RMSE (std)', markersize=4)

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

    # 自定义从红 -> 黄 -> 绿的平滑渐变色表
    colors_list = [
        (1.0, 0.0, 0.0),  # 红
        (1.0, 1.0, 0.0),  # 黄
        (0.0, 1.0, 0.0)   # 绿
    ]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("red_yellow_green", colors_list, N=256)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        cmap=cmap_custom,
        annot=True,     # 显示数值
        fmt=".2f",      # 保留两位小数
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
    """
    在原始 DataFrame（未标准化的 domain）中对目标列进行可视化或统计分析。
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
    plt.plot(epochs, train_loss_history, 'r-o', label='Train Loss(std)', markersize=4)
    plt.plot(epochs, val_loss_history,   'b-o', label='Val Loss(std)',   markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (std)')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    # (2) RMSE
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_rmse_history, 'g--*', label='Val RMSE(std)', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (std)')
    plt.title('RMSE')
    plt.legend()
    plt.grid(True)

    # (3) MAPE
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_mape_history, 'm-*', label='Val MAPE(%)', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE')
    plt.legend()
    plt.grid(True)

    # (4) R^2
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_r2_history, 'c-o', label='Val R^2', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('R^2')
    plt.title('R^2')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Training Metrics for {model_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

# =========== 可视化函数 ===========
def plot_Egrid_over_time(data_df):
    """
    绘制整段时间序列上 E_grid 值的变化趋势（原始域数据）。
    data_df 需要包含 'timestamp' 和 'E_grid' 两列，并保证已按时间排序。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data_df['timestamp'], data_df['E_grid'], color='blue', marker='o', markersize=3, linewidth=1)
    plt.xlabel('Timestamp')
    plt.ylabel('E_grid')
    plt.title('E_grid over Time (entire dataset)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_test_predictions_over_time(test_timestamps, y_actual_real, y_pred_real):
    """
    在时间轴上同时画出 E_grid 的实际值和预测值随时间变化曲线（针对测试集）。
    test_timestamps: 测试集对应的 timestamp 列（长度与 y_actual_real 一致）。
    """
    plt.figure(figsize=(10,5))
    plt.plot(test_timestamps, y_actual_real, 'r-o',  label='Actual E_grid',   markersize=3, linewidth=1)
    plt.plot(test_timestamps, y_pred_real,   'b--*', label='Predicted E_grid', markersize=3, linewidth=1)
    plt.xlabel('Timestamp (Test Data)')
    plt.ylabel('E_grid (real domain)')
    plt.title('Comparison of Actual vs Predicted E_grid over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    print("[Info] Loading and preprocessing data...")
    data_df = load_data()

    # ========== 基于原始 df 画 NxN 热力图（此处保留 E_grid=0 的数据） ==========
    # 将 E_grid 纳入以便观察其与其它字段的相关性
    feature_cols_to_plot = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_grid'
    ]
    # 根据实际存在的列再做筛选
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns]
    if len(feature_cols_to_plot) > 1:
        plot_correlation_heatmap(data_df, feature_cols_to_plot)

    # ========= 在绘制完热力图后，可选删除 E_grid == 0 的行，并重置索引 =========
    if 'E_grid' in data_df.columns:
        data_df = data_df[data_df['E_grid'] != 0].copy()
        data_df.reset_index(drop=True, inplace=True)

        # 分析目标列 E_grid 的分布（此时已经不含 E_grid=0）
        analyze_target_distribution(data_df, "E_grid")

        # ========== 在整段时间序列上，画出 E_grid 的变化趋势（此时已无 0 值）==========
        plot_Egrid_over_time(data_df)

    # ========== 特征工程 + 序列构建 + 训练测试拆分 ==========
    data_all, feature_cols, target_col, scaler_y, data_df_afterFE = feature_engineering(data_df)
    feature_dim = len(feature_cols)

    print(f"[Info] Model will predict target column: {target_col}")
    print(f"[Info] Selected feature columns: {feature_cols}")

    X_all, y_all = create_sequences(data_all, window_size=window_size, feature_dim=feature_dim)
    print(f"[Info] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")

    total_samples = X_all.shape[0]
    train_size    = int(0.8 * total_samples)
    val_size      = int(0.1 * total_samples)
    test_size     = total_samples - train_size - val_size

    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val,   y_val   = X_all[train_size : train_size + val_size], y_all[train_size : train_size + val_size]
    X_test,  y_test  = X_all[train_size + val_size:], y_all[train_size + val_size:]
    print(f"[Info] Split data => Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ========== 生成与序列数据对应的时间戳 (已去除 E_grid=0 的行后) ==========
    timestamps_all = data_df_afterFE['timestamp'].values[window_size:]
    train_timestamps = timestamps_all[:train_size]
    val_timestamps   = timestamps_all[train_size : train_size + val_size]
    test_timestamps  = timestamps_all[train_size + val_size :]

    print("[Info] Building models...")
    modelA = EModel_FeatureWeight(feature_dim).to(device)
    modelB = EModel_CNN_Transformer(feature_dim).to(device)

    print("\n========== Train EModel_FeatureWeight ==========")
    (train_lossA, val_lossA, val_rmseA, _, _, _) = train_model(
        modelA, train_loader, val_loader, 
        model_name='EModel_FeatureWeight', 
        feature_names=feature_cols  
    )

    print("\n========== Train EModel_CNN_Transformer ==========")
    (train_lossB, val_lossB, val_rmseB, _, _, _) = train_model(
        modelB, train_loader, val_loader, 
        model_name='EModel_CNN_Transformer', 
        feature_names=feature_cols
    )

    # 加载最优权重 & 测试
    best_modelA = EModel_FeatureWeight(feature_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth', weights_only=True))

    best_modelB = EModel_CNN_Transformer(feature_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_CNN_Transformer.pth', weights_only=True))

    print("\n[Info] Testing...")
    criterion = nn.SmoothL1Loss(beta=1.0)

    # Evaluate A
    test_lossA, test_rmseA_std, test_mapeA_std, test_r2A_std, predsA_std, labelsA_std = evaluate(best_modelA, test_loader, criterion)
    # Evaluate B
    test_lossB, test_rmseB_std, test_mapeB_std, test_r2B_std, predsB_std, labelsB_std = evaluate(best_modelB, test_loader, criterion)

    print("\n========== Test Results (standardized domain) ==========")
    print(f"[EModel_FeatureWeight] => Test Loss(std): {test_lossA:.4f}, RMSE(std): {test_rmseA_std:.4f}")
    print(f"[EModel_CNN_Transformer] => Test Loss(std): {test_lossB:.4f}, RMSE(std): {test_rmseB_std:.4f}")
    print(f"[EModel_FeatureWeight] => Test MAPE(std): {test_mapeA_std:.4f}, R^2(std): {test_r2A_std:.4f}")
    print(f"[EModel_CNN_Transformer] => Test MAPE(std): {test_mapeB_std:.4f}, R^2(std): {test_r2B_std:.4f}")

    # 反标准化
    predsA_real  = scaler_y.inverse_transform(predsA_std.reshape(-1,1)).ravel()
    predsB_real  = scaler_y.inverse_transform(predsB_std.reshape(-1,1)).ravel()
    labelsA_real = scaler_y.inverse_transform(labelsA_std.reshape(-1,1)).ravel()
    labelsB_real = scaler_y.inverse_transform(labelsB_std.reshape(-1,1)).ravel()

    # 在原空间重新计算RMSE
    test_rmseA_real = np.sqrt(mean_squared_error(labelsA_real, predsA_real))
    test_rmseB_real = np.sqrt(mean_squared_error(labelsB_real, predsB_real))

    print("\n========== Test Results (real domain) ==========")
    print(f"[EModel_FeatureWeight] => RMSE(real): {test_rmseA_real:.4f}")
    print(f"[EModel_CNN_Transformer] => RMSE(real): {test_rmseB_real:.4f}")

    print("\n========== Train EModel_FeatureWeight (with extended metrics) ==========")
    (train_lossA, val_lossA, val_rmseA, val_mapeA, val_r2A, _) = train_model(
        modelA, train_loader, val_loader, 
        model_name='EModel_FeatureWeight', 
        feature_names=feature_cols  
    )

    print("\n========== Train EModel_CNN_Transformer (with extended metrics) ==========")
    (train_lossB, val_lossB, val_rmseB, val_mapeB, val_r2B, _) = train_model(
        modelB, train_loader, val_loader, 
        model_name='EModel_CNN_Transformer', 
        feature_names=feature_cols
    )

    # ========== 在时间轴上同时画出测试集的实际值和预测值 (示例：modelA) ==========
    plot_test_predictions_over_time(test_timestamps, labelsA_real, predsA_real)
    # ========== 额外示例：可视化训练集和验证集的时序预测结果 ==========

    # 1) 计算训练集上的预测结果
    train_lossA, train_rmseA_std, train_mapeA_std, train_r2A_std, predsA_train_std, labelsA_train_std = evaluate(best_modelA, train_loader, criterion)
    predsA_train_real  = scaler_y.inverse_transform(predsA_train_std.reshape(-1, 1)).ravel()
    labelsA_train_real = scaler_y.inverse_transform(labelsA_train_std.reshape(-1, 1)).ravel()

    # 2) 计算验证集上的预测结果
    val_lossA, val_rmseA_std, val_mapeA_std, val_r2A_std, predsA_val_std, labelsA_val_std = evaluate(best_modelA, val_loader, criterion)
    predsA_val_real  = scaler_y.inverse_transform(predsA_val_std.reshape(-1, 1)).ravel()
    labelsA_val_real = scaler_y.inverse_transform(labelsA_val_std.reshape(-1, 1)).ravel()

    # 3) 分别绘制训练集、验证集的预测曲线
    print("\n[Info] Plot predictions for Training Set:")
    plot_test_predictions_over_time(train_timestamps, labelsA_train_real, predsA_train_real)

    print("\n[Info] Plot predictions for Validation Set:")
    plot_test_predictions_over_time(val_timestamps, labelsA_val_real, predsA_val_real)

    # 其他可视化对比
    plot_predictions_comparison(
        y_actual_real      = labelsA_real,
        y_pred_model1_real = predsA_real,
        y_pred_model2_real = predsB_real,
        model1_name        = 'EModel_FeatureWeight',
        model2_name        = 'EModel_CNN_Transformer'
    )
    plot_training_curves(train_lossA, val_lossA, val_rmseA, model_name='EModel_FeatureWeight')
    plot_training_curves(train_lossB, val_lossB, val_rmseB, model_name='EModel_CNN_Transformer')
    plot_two_model_val_rmse(val_rmseA, val_rmseB, labelA='EModel_FeatureWeight', labelB='EModel_CNN_Transformer')

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