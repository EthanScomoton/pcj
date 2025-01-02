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

# ---------------------------
# 0. 全局超参数
# ---------------------------
learning_rate = 1e-5   # 学习率
num_epochs = 100       # 训练轮数
batch_size = 512       # 批次大小
weight_decay = 5e-3    # L2正则化
patience = 5           # 早停轮数
num_workers = 0        # DataLoader 进程数

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  


# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_and_preprocess_data():
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data.csv')
    load_df = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')

    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour'] = data_df['timestamp'].dt.hour
    data_df['month'] = data_df['timestamp'].dt.month

    # 时间特征: sin/cos
    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month']-1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month']-1) / 12)

    renewable_features = ['season','holiday','weather','temperature','working_hours',
                          'E_PV','E_storage_discharge','E_grid','ESCFR','ESCFG']
    load_features = ['ship_grade','dock_position','destination']

    # 目标值
    y_raw = data_df['energyconsumption'].values.astype(float)
    y_log = np.log1p(y_raw)  # 对数变换

    # LabelEncoder
    label_encoders = {}
    for feature in renewable_features + load_features:
        le = LabelEncoder()
        data_df[feature] = le.fit_transform(data_df[feature].fillna("Unknown"))
        label_encoders[feature] = le

    time_feature_cols = [
        'dayofweek_sin','dayofweek_cos',
        'hour_sin','hour_cos',
        'month_sin','month_cos'
    ]

    feature_columns = renewable_features + load_features + time_feature_cols

    X_raw = data_df[feature_columns].values

    # 数值标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    return X_scaled, y_log, renewable_features, load_features, scaler_X


# ---------------------------
# 2. 辅助模块
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() 
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(0), :]
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
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = x * attn_weights
        output = torch.sum(weighted, dim=1)  # (batch_size, input_dim)
        return output

# ---------------------------
# 3A. 之前的 EModel (带 feature_importance + LSTM)
# ---------------------------
class EModel_FeatureWeight(nn.Module):
    def __init__(self, num_features, renewable_dim, load_dim):
        super(EModel_FeatureWeight, self).__init__()
        self.renewable_dim = renewable_dim
        self.load_dim = load_dim

        # 可学习特征权重
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

        # LSTM 处理时序特征
        self.temporal_input_dim = num_features - self.renewable_dim - self.load_dim
        if self.temporal_input_dim > 0:
            self.temporal_lstm = nn.LSTM(
                input_size = self.temporal_input_dim,
                hidden_size = 128,
                num_layers = 2,
                batch_first = True,
                bidirectional = True,
                dropout = 0.2
            )
        else:
            self.temporal_lstm = None

        # Transformer
        self.pos_encoder = PositionalEncoding(d_model = 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Attention
        self.attention = Attention(input_dim=128)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(128 + 256 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # 对输入特征做权重加成
        x = x * self.feature_importance

        # 分割特征
        renewable_features = x[:, :self.renewable_dim]
        load_features = x[:, self.renewable_dim:self.renewable_dim + self.load_dim]
        temporal_features = x[:, self.renewable_dim + self.load_dim:]

        # 编码
        renewable_encoded = self.renewable_encoder(renewable_features)
        load_encoded = self.load_encoder(load_features)

        # 合并后 -> (batch, 128) -> (batch, 1, 128)
        combined_features = torch.cat([renewable_encoded, load_encoded], dim=-1)
        combined_features = combined_features.unsqueeze(1)

        # BiGRU
        interaction_out, _ = self.interaction_bigru(combined_features)
        attention_out = self.attention(interaction_out)

        # LSTM + Transformer
        if self.temporal_lstm is not None and temporal_features.size(1) > 0:
            temporal_features = temporal_features.unsqueeze(1)
            temporal_out, _ = self.temporal_lstm(temporal_features)

            transformer_input = temporal_out.permute(1, 0, 2)
            transformer_input = self.pos_encoder(transformer_input)
            transformer_out = self.transformer_encoder(transformer_input)
            transformer_out = transformer_out.permute(1, 0, 2)

            temporal_out = temporal_out[:, -1, :]
            transformer_out = transformer_out[:, -1, :]
        else:
            temporal_out = torch.zeros(x.size(0), 256).to(x.device)
            transformer_out = torch.zeros(x.size(0), 256).to(x.device)

        merged = torch.cat([attention_out, temporal_out, transformer_out], dim=1)
        output = self.fc(merged)
        return output.squeeze(-1)


# ---------------------------
# 3B. 新的 EModel_BiGRU (双BiGRU + Transformer) 
# ---------------------------
class EModel_BiGRU(nn.Module):
    def __init__(self, num_features, renewable_dim, load_dim):
        super(EModel_BiGRU, self).__init__()
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
                dropout = 0.3
            )
        else:
            self.temporal_bigru = None

        # Transformer
        self.pos_encoder = PositionalEncoding(d_model = 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model = 256, nhead = 8, batch_first = True)
        # 这里你写了 num_layers=13，这里可以改成 13 或者其它；为了演示，改小一点也行
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 2)

        # Attention
        self.attention = Attention(input_dim=128)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(128 + 256 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # 分离特征
        renewable_features = x[:, :self.renewable_dim]
        load_features = x[:, self.renewable_dim:self.renewable_dim + self.load_dim]
        temporal_features = x[:, self.renewable_dim + self.load_dim:]

        # 编码
        renewable_encoded = self.renewable_encoder(renewable_features)  
        load_encoded = self.load_encoder(load_features)  

        # 合并编码
        combined_features = torch.cat([renewable_encoded, load_encoded], dim=-1)
        combined_features = combined_features.unsqueeze(1)

        # BiGRU for interaction
        interaction_out, _ = self.interaction_bigru(combined_features)
        attention_out = self.attention(interaction_out)

        # 时序 BiGRU + Transformer
        if self.temporal_bigru is not None and temporal_features.size(1) > 0:
            temporal_features = temporal_features.unsqueeze(1)
            temporal_out, _ = self.temporal_bigru(temporal_features)

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

        merged = torch.cat([attention_out, temporal_out, transformer_out], dim=1)
        output = self.fc(merged)
        return output.squeeze(-1)


# ---------------------------
# 4. 训练与评估工具
# ---------------------------
def evaluate(model, dataloader, criterion, device):
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

    val_loss = running_loss / num_samples  # MSE (log domain)
    preds_arr = np.concatenate(preds_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    rmse_log = np.sqrt(mean_squared_error(labels_arr, preds_arr))
    return val_loss, rmse_log, preds_arr, labels_arr

def train_model(model, train_loader, val_loader, model_name='Model'):
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
            scheduler.step()

            preds = model(batch_inputs)
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples += batch_inputs.size(0)
            global_step += 1

        train_loss = running_loss / num_samples

        # 在验证集上评估
        val_loss, val_rmse, _, _ = evaluate(model, val_loader)

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"Train MSE(log): {train_loss:.4f}, "
              f"Val MSE(log): {val_loss:.4f}, RMSE(log): {val_rmse:.4f}")

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
# 5. 画出多模型与实际值的对比曲线
# ---------------------------
def plot_predictions_comparison(y_actual, y_pred_model1, y_pred_model2, model1_name='Model1', model2_name='Model2'):
    """
    在同一张图上画 Actual, Model1, Model2 三条曲线
    你可以扩展到更多模型，比如 Model3, Model4 等。
    """
    plt.figure(figsize=(10,5))
    x_axis = np.arange(len(y_actual))

    plt.plot(x_axis, y_actual, 'r-o', label='Actual', linewidth=1)
    plt.plot(x_axis, y_pred_model1, 'b--*', label=model1_name, linewidth=1)
    plt.plot(x_axis, y_pred_model2, 'g-.*', label=model2_name, linewidth=1)

    plt.xlabel('Index')
    plt.ylabel('Value (log domain)')  # 因为我们做了对数变换
    plt.title(f'Comparison: Actual vs {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------
# 6. 主函数
# ---------------------------
def main():
    # 1) 加载数据
    X_scaled, y_log, renewable_features, load_features, scaler_X, data_df = load_and_preprocess_data()
    inputs_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    labels_tensor = torch.tensor(y_log, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_features = X_scaled.shape[1]
    renewable_dim = len(renewable_features)
    load_dim = len(load_features)

    # 2) 实例化两种模型
    modelA = EModel_FeatureWeight(num_features, renewable_dim, load_dim).to(device)
    modelB = EModel_BiGRU(num_features, renewable_dim, load_dim).to(device)

    # 3) 训练并保存
    print("\n========== Train EModel_FeatureWeight ==========")
    train_model(modelA, train_loader, val_loader, model_name='EModel_FeatureWeight')

    print("\n========== Train EModel_BiGRU ==========")
    train_model(modelB, train_loader, val_loader, model_name='EModel_BiGRU')

    # 4) 加载最优权重（可选，如果刚训练完也可直接用 modelA, modelB）
    best_modelA = EModel_FeatureWeight(num_features, renewable_dim, load_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth'))

    best_modelB = EModel_BiGRU(num_features, renewable_dim, load_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_BiGRU.pth'))

    # 5) 在验证集上推理（或换成测试集）
    val_loader_for_eval = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    # 只取一个 batch，包含全部验证集
    with torch.no_grad():
        for X_val, Y_val in val_loader_for_eval:
            X_val = X_val.to(device)
            Y_val = Y_val.cpu().numpy()   # shape: (val_size, )

            predsA = best_modelA(X_val).cpu().numpy()  # shape: (val_size,)
            predsB = best_modelB(X_val).cpu().numpy()

            # 画图: Y_val, predsA, predsB
            plot_predictions_comparison(
                y_actual=Y_val,
                y_pred_model1=predsA,
                y_pred_model2=predsB,
                model1_name='EModel_FeatureWeight',
                model2_name='EModel_BiGRU'
            )
            break  # 我们只需要一次

if __name__ == "__main__":
    main()