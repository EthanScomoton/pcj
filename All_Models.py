import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from torch.nn.utils import clip_grad_norm_

# ---------------------------
# 0. 全局超参数
# ---------------------------
learning_rate = 1e-4       # 学习率调低
num_epochs = 300           # 训练轮数适当调高
batch_size = 128           # 批次大小
weight_decay = 1e-4        # L2正则化
patience = 10              # 早停轮数增大
num_workers = 0            # DataLoader 进程数
window_size = 5           # 多步时序窗口

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_data():
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data1.csv')
    load_df = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data1.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])

    # 按 inner merge，得到完整记录
    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')

    # 时间排序，保证后面切分序列时是按时间顺序排列
    data_df.sort_values('timestamp', inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    
    return data_df

def feature_engineering(data_df):

    # 提取时间特征
    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour']      = data_df['timestamp'].dt.hour
    data_df['month']     = data_df['timestamp'].dt.month

    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month']-1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month']-1) / 12)

    # 需要手动指定哪些是可再生能源特征，哪些是负荷特征
    renewable_features = ['season','holiday','weather','temperature','working_hours',
                          'E_PV','E_storage_discharge','E_grid','ESCFR','ESCFG']
    load_features = ['ship_grade','dock_position','destination']
    
    # LabelEncoder
    for col in renewable_features + load_features:
        if col in data_df.columns:
            le = LabelEncoder()
            data_df[col] = le.fit_transform(data_df[col].astype(str))
    
    # 组合特征列
    time_feature_cols = [
        'dayofweek_sin','dayofweek_cos',
        'hour_sin','hour_cos',
        'month_sin','month_cos'
    ]
    feature_columns = renewable_features + load_features + time_feature_cols
    
    # 目标列
    target_column = 'energyconsumption'
    
    # 对目标做对数变换
    data_df['target_log'] = np.log1p(data_df[target_column].values)

    selected_cols = feature_columns + ['target_log']
    data_selected = data_df[selected_cols].copy()

    # 数值标准化 (只对特征做标准化, 不对 'target_log' 做)
    scaler_X = StandardScaler()

    data_selected[feature_columns] = scaler_X.fit_transform(
        data_selected[feature_columns].values
    )

    # 转成 numpy 数组，用于后面多步时序的 create_sequences
    data_all = data_selected.values  # shape: (num_samples, num_features + 1)
    return data_all, feature_columns, 'target_log'

def create_sequences(data_all, window_size, feature_dim):
    """
    data_all 的列顺序: [feature_1, feature_2, ..., feature_n, target_log]
    window_size: 使用过去多少步
    feature_dim: 特征数 (不含 target_log)
    
    返回:
    X: (samples, window_size, feature_dim)
    y: (samples,)
    """
    X_list, y_list = [], []
    num_samples = data_all.shape[0]
    
    for i in range(num_samples - window_size):
        # 取过去 window_size 行作为一个序列
        seq_x = data_all[i : i + window_size, :feature_dim]    # shape: (window_size, feature_dim)
        seq_y = data_all[i + window_size, feature_dim]         # 第 feature_dim 列是 target_log
        X_list.append(seq_x)
        y_list.append(seq_y)
    
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    return X_arr, y_arr

# ---------------------------
# 2. 模型结构
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, 0, :]
        return x

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
        attn_weights = F.softmax(attn_weights, dim = 1)
        weighted = x * attn_weights  # (batch_size, seq_length, input_dim)
        output = torch.sum(weighted, dim = 1)  # (batch_size, input_dim)
        return output

# 1: EModel_FeatureWeight (LSTM + Transformer + Attention)
class EModel_FeatureWeight(nn.Module):
    def __init__(self, feature_dim):
        super(EModel_FeatureWeight, self).__init__()
        self.feature_dim = feature_dim

        # 可学习特征权重, shape = (feature_dim,)
        self.feature_importance = nn.Parameter(
            torch.ones(feature_dim), requires_grad = True
        )

        self.lstm = nn.LSTM(
            input_size = feature_dim,
            hidden_size = 128,
            num_layers = 2,
            batch_first = True,
            bidirectional = True,
            dropout = 0.2
        )

        # Transformer
        self.pos_encoder = PositionalEncoding(d_model = 2 * 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model = 2 * 128, nhead = 8, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 2)

        # Attention
        self.attention = Attention(input_dim = 2 * 128)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x shape: (batch, seq_len, feature_dim)
        """
        # 对特征做通道级别的权重加成
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)

        # LSTM
        lstm_out, _ = self.lstm(x)  # shape: (batch, seq_len, 2*hidden_size=256)

        # Transformer
        t_out = self.pos_encoder(lstm_out)
        t_out = self.transformer_encoder(t_out)  # shape: (batch, seq_len, 256)

        # Attention
        attn_out = self.attention(t_out)  # (batch, 256)

        # 输出
        out = self.fc(attn_out)  # (batch, 1)
        return out.squeeze(-1)

# 2: EModel_BiGRU (双向GRU + Transformer + Attention)，含特征权重
class EModel_BiGRU(nn.Module):
    def __init__(self, feature_dim):
        super(EModel_BiGRU, self).__init__()
        self.feature_dim = feature_dim
        
        # 学习特征权重
        self.feature_importance = nn.Parameter(
            torch.ones(feature_dim), requires_grad=True
        )

        self.bigru = nn.GRU(
            input_size = feature_dim,
            hidden_size = 128,
            num_layers = 2,
            batch_first = True,
            bidirectional = True,
            dropout = 0.3
        )

        # Transformer
        self.pos_encoder = PositionalEncoding(d_model=2*128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = 2 * 128,
            nhead = 8,
            batch_first = True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = 2
        )

        # Attention
        self.attention = Attention(input_dim = 2 * 128)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x shape: (batch, seq_len, feature_dim)
        """
        # 对特征维度进行可学习权重乘法 (通道级别)
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)
        
        # BiGRU
        gru_out, _ = self.bigru(x)  # shape: (batch, seq_len, 2 * 128)

        # Transformer
        t_out = self.pos_encoder(gru_out)
        t_out = self.transformer_encoder(t_out)

        # Attention
        attn_out = self.attention(t_out)

        # 输出
        out = self.fc(attn_out)
        return out.squeeze(-1)

# ---------------------------
# 3. 训练与评估工具
# ---------------------------
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    num_samples = 0

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples += batch_inputs.size(0)

            preds_list.append(outputs.cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())

    val_loss = running_loss / num_samples   # MSE or SmoothL1 in log domain
    preds_arr = np.concatenate(preds_list, axis = 0)
    labels_arr = np.concatenate(labels_list, axis = 0)
    rmse_log = np.sqrt(mean_squared_error(labels_arr, preds_arr))
    return val_loss, rmse_log, preds_arr, labels_arr

def train_model(model, train_loader, val_loader, model_name = 'Model'):
    # 使用 SmoothL1Loss (Huber Loss)
    criterion = nn.SmoothL1Loss(beta=1.0)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 一些学习率调度策略 (可选)
    total_steps = num_epochs * len(train_loader)
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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = criterion(preds, batch_labels)
            loss.backward()

            # --------- 梯度裁剪以稳定训练 ---------
            clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples += batch_inputs.size(0)
            global_step += 1

        train_loss = running_loss / num_samples
        val_loss, val_rmse, _, _ = evaluate(model, val_loader, criterion)

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss(log): {train_loss:.4f}, "
              f"Val Loss(log): {val_loss:.4f}, RMSE(log): {val_rmse:.4f}")

        # Early Stopping监控
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

# ---------------------------
# 4. 可视化对比曲线
# ---------------------------
def plot_predictions_comparison(y_actual_log, y_pred_model1, y_pred_model2, model1_name='Model1', model2_name='Model2'):
    """
    在同一张图上画 Actual(log), Model1, Model2 三条曲线
    y_actual_log: 对数域下的真实值
    """
    plt.figure(figsize=(10,5))
    x_axis = np.arange(len(y_actual_log))

    plt.plot(x_axis, y_actual_log, 'r-o', label='Actual (log)', linewidth=1)
    plt.plot(x_axis, y_pred_model1, 'g--*', label=model1_name, linewidth=1)
    plt.plot(x_axis, y_pred_model2, 'b-.*', label=model2_name, linewidth=1)

    plt.xlabel('Index')
    plt.ylabel('Value (log domain)')
    plt.title(f'Comparison: Actual (log) vs {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    # 1) 加载 & 特征工程
    print("[Info] Loading and preprocessing data...")
    data_df = load_data()
    data_all, feature_cols, target_col = feature_engineering(data_df)

    # 2) 构建多步时序数据
    feature_dim = len(feature_cols)
    X_all, y_all = create_sequences(
        data_all, 
        window_size=window_size, 
        feature_dim=feature_dim
    )
    print(f"[Info] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")

    # 3) 划分数据集: 8:1:1
    total_samples = X_all.shape[0]
    train_size = int(0.8 * total_samples)
    val_size   = int(0.1 * total_samples)
    test_size  = total_samples - train_size - val_size

    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val,   y_val   = X_all[train_size : train_size+val_size], y_all[train_size : train_size+val_size]
    X_test,  y_test  = X_all[train_size+val_size : ], y_all[train_size+val_size : ]
    print(f"[Info] Split data => Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # 4) 构建 DataLoader
    train_dataset = TensorDataset(
        torch.from_numpy(X_train), 
        torch.from_numpy(y_train)
    )
    val_dataset   = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    test_dataset  = TensorDataset(
        torch.from_numpy(X_test), 
        torch.from_numpy(y_test)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_loader   = DataLoader(
        val_dataset,   
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    test_loader  = DataLoader(
        test_dataset,  
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )

    # 5) 实例化模型
    print("[Info] Building models...")
    modelA = EModel_FeatureWeight(feature_dim).to(device)
    modelB = EModel_BiGRU(feature_dim).to(device)

    # 6) 训练模型
    print("\n========== Train EModel_FeatureWeight ==========")
    train_model(modelA, train_loader, val_loader, model_name='EModel_FeatureWeight')

    print("\n========== Train EModel_BiGRU ==========")
    train_model(modelB, train_loader, val_loader, model_name='EModel_BiGRU')

    # 7) 加载最优权重
    print("\n[Info] Loading best model weights...")
    best_modelA = EModel_FeatureWeight(feature_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth'))

    best_modelB = EModel_BiGRU(feature_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_BiGRU.pth'))

    # 8) 测试集评估
    print("\n[Info] Testing...")
    criterion = nn.SmoothL1Loss(beta=1.0)
    test_lossA, test_rmseA, predsA, labelsA = evaluate(best_modelA, test_loader, criterion)
    test_lossB, test_rmseB, predsB, _       = evaluate(best_modelB, test_loader, criterion)

    print("\n========== Test Results (Original) ==========")
    print(f"[EModel_FeatureWeight] => Test Loss(log): {test_lossA:.4f}, RMSE(log): {test_rmseA:.4f}")
    print(f"[EModel_BiGRU]         => Test Loss(log): {test_lossB:.4f}, RMSE(log): {test_rmseB:.4f}")

    # ---------------------------
    # 这里插入后处理平滑的过程
        # ---------------------------
    def smooth_predictions(preds, window=5):
        """
        简单的滑动平均平滑, window 可以根据需求调整
        """
        smoothed = np.copy(preds)
        for i in range(window, len(preds)):
            smoothed[i] = np.mean(preds[i-window:i])
        return smoothed

    # 对两个模型的预测分别做平滑
    predsA_smooth = smooth_predictions(predsA, window=5)
    predsB_smooth = smooth_predictions(predsB, window=5)

    # 如果你需要查看平滑后的 RMSE，可在这里单独计算
    test_rmseA_smooth = np.sqrt(mean_squared_error(labelsA, predsA_smooth))
    test_rmseB_smooth = np.sqrt(mean_squared_error(labelsA, predsB_smooth))

    print("\n========== Test Results (Smoothed) ==========")
    print(f"[EModel_FeatureWeight] => Smoothed RMSE(log): {test_rmseA_smooth:.4f}")
    print(f"[EModel_BiGRU]         => Smoothed RMSE(log): {test_rmseB_smooth:.4f}")

    # 1) 将平滑后的预测值 predsA_smooth / predsB_smooth 从对数域转回原空间
    predsA_smooth_real = np.expm1(predsA_smooth)
    predsB_smooth_real = np.expm1(predsB_smooth)
    labels_real        = np.expm1(labelsA)  # labelsA 也在 log(1 + y) 域，需一起转回

    # 2) 如果需要查看平滑后在原空间的 RMSE
    test_rmseA_smooth_real = np.sqrt(mean_squared_error(labels_real, predsA_smooth_real))
    test_rmseB_smooth_real = np.sqrt(mean_squared_error(labels_real, predsB_smooth_real))

    print("\n========== Test Results (Smoothed, Real Domain) ==========")
    print(f"[EModel_FeatureWeight] => Smoothed RMSE(real): {test_rmseA_smooth_real:.4f}")
    print(f"[EModel_BiGRU]         => Smoothed RMSE(real): {test_rmseB_smooth_real:.4f}")

    # 3) 可视化：在原空间
    def plot_predictions_comparison_real(y_actual, y_pred_model1, y_pred_model2,
                                        model1_name='Model1', model2_name='Model2'):
        plt.figure(figsize=(10,5))
        x_axis = np.arange(len(y_actual))

        plt.plot(x_axis, y_actual, 'r-o', label='Actual (real)', linewidth=1)
        plt.plot(x_axis, y_pred_model1, 'g--*', label=model1_name, linewidth=1)
        plt.plot(x_axis, y_pred_model2, 'b-.*', label=model2_name, linewidth=1)

        plt.xlabel('Index')
        plt.ylabel('Value (real domain)')
        plt.title(f'Comparison: Actual (real) vs {model1_name} vs {model2_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plot_predictions_comparison_real(
        y_actual        = labels_real,
        y_pred_model1   = predsA_smooth_real,
        y_pred_model2   = predsB_smooth_real,
        model1_name     = 'EModel_FeatureWeight (smooth, real)',
        model2_name     = 'EModel_BiGRU (smooth, real)'
    )

    # 最后可视化时，使用平滑后的预测结果
    plot_predictions_comparison(
        y_actual_log    = labelsA,
        y_pred_model1   = predsA_smooth,  # 使用平滑后预测
        y_pred_model2   = predsB_smooth,  # 使用平滑后预测
        model1_name     = 'EModel_FeatureWeight (smooth)',
        model2_name     = 'EModel_BiGRU (smooth)'
    )
    print("[Info] Done!")

if __name__ == "__main__":
    main()