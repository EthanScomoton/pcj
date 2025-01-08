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

    renewable_features = ['season', 'holiday', 'weather', 'temperature','working_hours', 'E_PV', 'E_storage_discharge','E_grid', 'ESCFR', 'ESCFG']
    load_features = ['ship_grade', 'dock_position', 'destination']

    for col in renewable_features + load_features:
        if col in data_df.columns:
            le = LabelEncoder()
            data_df[col] = le.fit_transform(data_df[col].astype(str))

    time_feature_cols = ['dayofweek_sin', 'dayofweek_cos','hour_sin', 'hour_cos','month_sin', 'month_cos']

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
    def __init__(self, d_model, nhead = 8, num_encoder_layers = 2, num_decoder_layers = 2):
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
    def __init__(self, feature_dim):
        super(EModel_BiGRU, self).__init__()
        self.feature_dim = feature_dim
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)

        self.bigru = nn.GRU(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
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
        gru_out, _ = self.bigru(x)
        transformer_out = self.transformer_block(gru_out)
        attn_out = self.attention(transformer_out)
        out = self.fc(attn_out)
        return out.squeeze(-1)

# ---------------------------
# 3. 训练与评价工具
# ---------------------------
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, num_samples = 0.2, 0
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
    rmse_std   = np.sqrt(mean_squared_error(labels_arr, preds_arr))
    return val_loss, rmse_std, preds_arr, labels_arr

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

    # 虽然保留了记录 feature_importance 的逻辑，但本示例不再绘制特征权重热力图
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
        val_loss, val_rmse_std, _, _ = evaluate(model, val_loader, criterion)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_rmse_history.append(val_rmse_std)

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss(std): {train_loss:.4f}, "
              f"Val Loss(std): {val_loss:.4f}, "
              f"RMSE(std): {val_rmse_std:.4f}")

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

    return (train_loss_history, 
            val_loss_history, 
            val_rmse_history, 
            feature_importance_history)

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

# =============== 新增的热力图函数（替换原先的特征权重可视化） ===============
def plot_correlation_heatmap(df, feature_cols, title = "Heat map of different types of loads and external factors"):
    """
    绘制 NxN 热力图，横纵坐标为同一批特征(如 [season, holiday, weather, temperature, ...])。
    - 不显示数字 (annot=False)
    - 使用从红到紫的色系，增强对比度
    - 若列中含有字符串（如 'winter'），则先用 LabelEncoder 转为数值
    """
    df_encoded = df.copy()
    for col in feature_cols:
        # 如果这一列是字符串/对象类型，先做一个简单的编码
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # 1) 计算 NxN 矩阵 (以相关系数为例)
    corr_matrix = df_encoded[feature_cols].corr()

    # 2) 自定义一个从红到粉到紫的颜色映射
    colors_list = ["red", "magenta", "purple"]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("my_rdpu", colors_list, N=256)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr_matrix,
        cmap=cmap_custom,   # 红 -> 紫 渐变
        annot=False,        # 不显示数字
        square=True,
        linewidths=1,       # 格子分割线宽度
        linecolor='white',  # 分割线颜色
        cbar=True           # 右侧显示色条
    )
    plt.title(title)
    plt.xticks(rotation = 45, ha='right')
    plt.tight_layout()
    plt.show()

# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    print("[Info] Loading and preprocessing data...")
    data_df = load_data()

    # ========== 在做特征工程前，或之后，也可基于原始 df 画 NxN 热力图 ==========
    # 比如先简单地在原始 DataFrame 上做一下选择
    feature_cols_to_plot = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'energyconsumption'
    ]
    # 先判断列是否在 data_df 中
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns]

    # 绘制热力图 (红->紫, 不显示数字)
    plot_correlation_heatmap(data_df, feature_cols_to_plot)

    # ========== 后续原有流程(特征工程 + 训练 + 测试) ==========
    data_all, feature_cols, target_col, scaler_y = feature_engineering(data_df)
    feature_dim = len(feature_cols)

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
        feature_names=feature_cols
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

    print("[Info] Done!")

if __name__ == "__main__":
    main()
