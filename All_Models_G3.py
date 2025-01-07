import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    """
    读取并合并可再生能源与负载数据，返回时间排序后的完整DataFrame
    """
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data1.csv')
    load_df      = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data1.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp']      = pd.to_datetime(load_df['timestamp'])

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')
    data_df.sort_values('timestamp', inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    return data_df

def feature_engineering(data_df):
    """
    对合并后的数据进行特征工程和标准化，返回处理后的所有数据矩阵、特征列名、目标列名以及目标的scaler
    """
    # 提取时间特征
    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour']      = data_df['timestamp'].dt.hour
    data_df['month']     = data_df['timestamp'].dt.month

    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month'] - 1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month'] - 1) / 12)

    # 待LabelEncoder的列
    renewable_features = ['season', 'holiday', 'weather', 'temperature','working_hours', 'E_PV', 'E_storage_discharge','E_grid', 'ESCFR', 'ESCFG']
    load_features      = ['ship_grade', 'dock_position', 'destination']
    
    for col in renewable_features + load_features:  # 遍历 renewable_features 和 load_features 列表中的每一个列名
        if col in data_df.columns:                  # 如果列名在数据框 data_df 的列中
            le = LabelEncoder()                     # 创建一个 LabelEncoder 实例
            data_df[col] = le.fit_transform(data_df[col].astype(str))  # 对该列进行标签编码，并将结果赋值回该列

    time_feature_cols = [
        'dayofweek_sin', 'dayofweek_cos',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos'
    ]
    feature_columns = renewable_features + load_features + time_feature_cols
    target_column   = 'energyconsumption'

    data_selected = data_df[feature_columns + [target_column]].copy()

    # 对特征做标准化
    scaler_X = StandardScaler()
    data_selected[feature_columns] = scaler_X.fit_transform(data_selected[feature_columns].values)

    # 对目标做标准化
    scaler_y = StandardScaler()
    data_selected[[target_column]] = scaler_y.fit_transform(data_selected[[target_column]].values)

    data_all = data_selected.values  # shape: (num_samples, feature_dim + 1)
    return data_all, feature_columns, target_column, scaler_y

def create_sequences(data_all, window_size, feature_dim):
    """
    将多步时序映射到 (X, y) 形式:
    X: (samples, window_size, feature_dim)
    y: (samples,)
    """
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
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 将前 seq_len 的位置编码加到 x 上
        return x + self.pe[:seq_len, 0, :]

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, d_model, nhead = 8, num_encoder_layers = 2, num_decoder_layers = 2):
        super(EncoderDecoderTransformer, self).__init__()

        self.encoder_pe = PositionalEncoding(d_model)
        self.decoder_pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model = d_model, nhead = nhead, batch_first = True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_decoder_layers)

    def forward(self, src):
        """
        src: [batch_size, seq_len, d_model]
        """
        # --- Encoder ---
        # 对 encoder 端输入进行位置编码
        src_enc = self.encoder_pe(src)
        memory  = self.transformer_encoder(src_enc)  # [batch_size, seq_len, d_model]

        # --- Decoder ---
        # 此处示例：简单将 src_enc 的某种形式当做 decoder 的输入
        #   1）若为多步预测，可使用“shift”后的序列、或零向量、或历史真实值等做 decoder 端输入（tgt）。
        #   2）也可视任务需求替换成更合适的 tgt。
        tgt = self.decoder_pe(src) 
        out = self.transformer_decoder(tgt, memory)  # [batch_size, seq_len, d_model]

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
        attn_weights = self.attention(x)    # [batch, seq, 1]
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim = 1)
        weighted = x * attn_weights         # [batch, seq, feature]
        return torch.sum(weighted, dim = 1)   # [batch, feature]

class EModel_FeatureWeight(nn.Module):
    def __init__(self, feature_dim):
        super(EModel_FeatureWeight, self).__init__()
        self.feature_dim = feature_dim
        # 特征权重可学习
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad = True)

        # LSTM
        self.lstm = nn.LSTM(
            input_size = feature_dim,
            hidden_size = 128,
            num_layers = 2,
            batch_first = True,
            bidirectional = True,
            dropout = 0.2
        )

        # 因为lstm是双向128维输出 => 每个时刻输出的维度为 256
        self.transformer_block = EncoderDecoderTransformer(d_model = 2 * 128, nhead = 8, num_encoder_layers = 2, num_decoder_layers = 2)

        # Attention + 输出层
        self.attention = Attention(input_dim = 2 * 128)
        self.fc = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, feature_dim]
        """
        # 1) 特征加权
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)

        # 2) LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, 2 * hidden_size (=256)]

        # 3) 送入包含 encoder+decoder 的 Transformer
        #    src形状 => [batch, seq_len, d_model (=256)]
        transformer_out = self.transformer_block(lstm_out)

        # 4) Attention
        attn_out = self.attention(transformer_out)

        # 5) 输出
        out = self.fc(attn_out)  # [batch, 1]
        return out.squeeze(-1)

class EModel_BiGRU(nn.Module):
    def __init__(self, feature_dim):
        super(EModel_BiGRU, self).__init__()
        self.feature_dim = feature_dim
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad = True)

        self.bigru = nn.GRU(
            input_size = feature_dim,
            hidden_size = 128,
            num_layers = 2,
            batch_first = True,
            bidirectional = True,
            dropout = 0.3
        )

        # 整合 encoder+decoder 的 Transformer 
        self.transformer_block = EncoderDecoderTransformer(d_model = 2 * 128, nhead = 8, num_encoder_layers = 2, num_decoder_layers = 2)

        self.attention = Attention(input_dim = 2 * 128)
        self.fc = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, feature_dim]
        """
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)
        gru_out, _ = self.bigru(x)  # [batch, seq_len, 2 * hidden_size (=256)]

        transformer_out = self.transformer_block(gru_out)
        attn_out        = self.attention(transformer_out)
        out             = self.fc(attn_out)
        return out.squeeze(-1)

# ---------------------------
# 3. 训练与评价工具
# ---------------------------
def evaluate(model, dataloader, criterion):
    """
    在验证或测试集上评估模型，返回平均loss、RMSE以及预测值和标签(标准化空间下)
    """
    model.eval()
    running_loss, num_samples = 0.0, 0
    preds_list, labels_list   = [], []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_inputs)
            loss    = criterion(outputs, batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples  += batch_inputs.size(0)

            preds_list.append(outputs.cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())

    val_loss   = running_loss / num_samples
    preds_arr  = np.concatenate(preds_list, axis = 0)
    labels_arr = np.concatenate(labels_list, axis = 0)
    rmse_std   = np.sqrt(mean_squared_error(labels_arr, preds_arr))
    return val_loss, rmse_std, preds_arr, labels_arr

def train_model(model, train_loader, val_loader, model_name='Model'):
    """
    训练并验证单个模型，返回训练过程中的损失和RMSE曲线，以便后续绘图或比较
    """
    criterion = nn.SmoothL1Loss(beta = 1.0)  # 可改成MSELoss
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度器(含warmup)
    total_steps  = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(0.0, 1 - (current_step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    counter       = 0
    global_step   = 0

    train_loss_history = []
    val_loss_history   = []
    val_rmse_history   = []

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

            clip_grad_norm_(model.parameters(), max_norm = 5.0)
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

        # Early Stopping 及最优模型保存
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

    return train_loss_history, val_loss_history, val_rmse_history

# ---------------------------
# 4. 可视化
# ---------------------------
def plot_predictions_comparison(
    y_actual_real, y_pred_model1_real, y_pred_model2_real,
    model1_name='Model1', model2_name='Model2'
):
    """
    对比两个模型在真实值下的预测情况
    """
    plt.figure(figsize=(10,5))
    x_axis = np.arange(len(y_actual_real))

    plt.plot(x_axis, y_actual_real,        'r-o',  label = 'Actual', linewidth = 1)
    plt.plot(x_axis, y_pred_model1_real,   'g--*', label=model1_name, linewidth = 1)
    plt.plot(x_axis, y_pred_model2_real,   'b-.*', label=model2_name, linewidth = 1)
    plt.xlabel('Index')
    plt.ylabel('Value (real domain)')
    plt.title(f'Comparison: Actual vs {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_loss_history, val_loss_history, val_rmse_history, model_name = 'Model'):
    """
    绘制单个模型的训练和验证损失曲线，以及验证RMSE曲线
    """
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(10,5))

    plt.plot(epochs, train_loss_history, 'r-o',  label = 'Train Loss(std)')
    plt.plot(epochs, val_loss_history,   'b-o',  label = 'Val Loss(std)')
    plt.plot(epochs, val_rmse_history,   'g--*', label = 'Val RMSE(std)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss / RMSE (standardized domain)')
    plt.title(f'Training Curves for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_two_model_val_rmse(val_rmseA, val_rmseB, labelA='ModelA', labelB = 'ModelB'):
    """
    将两个模型在验证集上的RMSE曲线放在同一张图对比
    """
    epochsA = range(1, len(val_rmseA) + 1)
    epochsB = range(1, len(val_rmseB) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochsA, val_rmseA, 'r-o', label = f'{labelA} Val RMSE (std)')
    plt.plot(epochsB, val_rmseB, 'b-o', label = f'{labelB} Val RMSE (std)')

    plt.xlabel('Epoch')
    plt.ylabel('RMSE (standardized domain)')
    plt.title('Validation RMSE Comparison (standardized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    print("[Info] Loading and preprocessing data...")
    data_df   = load_data()
    data_all, feature_cols, target_col, scaler_y = feature_engineering(data_df)

    feature_dim = len(feature_cols)
    X_all, y_all = create_sequences(data_all, window_size = window_size, feature_dim = feature_dim)
    print(f"[Info] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")

    # 数据集切分
    total_samples = X_all.shape[0]
    train_size    = int(0.8 * total_samples)
    val_size      = int(0.1 * total_samples)
    test_size     = total_samples - train_size - val_size

    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val,   y_val   = X_all[train_size : train_size+val_size], y_all[train_size : train_size+val_size]
    X_test,  y_test  = X_all[train_size+val_size:], y_all[train_size+val_size:]
    print(f"[Info] Split data => Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # 构建DataLoader
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,  num_workers = num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size = batch_size, shuffle = False, num_workers = num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size = batch_size, shuffle = False, num_workers = num_workers)

    print("[Info] Building models...")
    modelA = EModel_FeatureWeight(feature_dim).to(device)
    modelB = EModel_BiGRU(feature_dim).to(device)

    print("\n========== Train EModel_FeatureWeight ==========")
    train_lossA, val_lossA, val_rmseA = train_model(modelA, train_loader, val_loader, model_name = 'EModel_FeatureWeight')

    print("\n========== Train EModel_BiGRU ==========")
    train_lossB, val_lossB, val_rmseB = train_model(modelB, train_loader, val_loader, model_name = 'EModel_BiGRU')

    # 加载最优权重
    best_modelA = EModel_FeatureWeight(feature_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth'))

    best_modelB = EModel_BiGRU(feature_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_BiGRU.pth'))

    print("\n[Info] Testing...")
    criterion = nn.SmoothL1Loss(beta = 1.0)
    test_lossA, test_rmseA_std, predsA_std, labelsA_std = evaluate(best_modelA, test_loader, criterion)
    test_lossB, test_rmseB_std, predsB_std, _           = evaluate(best_modelB, test_loader, criterion)

    print("\n========== Test Results (standardized domain) ==========")
    print(f"[EModel_FeatureWeight] => Test Loss(std): {test_lossA:.4f}, RMSE(std): {test_rmseA_std:.4f}")
    print(f"[EModel_BiGRU]         => Test Loss(std): {test_lossB:.4f}, RMSE(std): {test_rmseB_std:.4f}")

    # 反标准化
    predsA_real   = scaler_y.inverse_transform(predsA_std.reshape(-1,1)).ravel()
    predsB_real   = scaler_y.inverse_transform(predsB_std.reshape(-1,1)).ravel()
    labelsA_real  = scaler_y.inverse_transform(labelsA_std.reshape(-1,1)).ravel()

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

    plot_training_curves(train_lossA, val_lossA, val_rmseA, model_name = 'EModel_FeatureWeight')
    plot_training_curves(train_lossB, val_lossB, val_rmseB, model_name = 'EModel_BiGRU')

    plot_two_model_val_rmse(val_rmseA, val_rmseB, labelA='EModel_FeatureWeight', labelB = 'EModel_BiGRU')

    print("[Info] Done!")

if __name__ == "__main__":
    main()
