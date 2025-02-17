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

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from torch.nn.utils import clip_grad_norm_
from lion_pytorch import Lion

# Hugging Face Trainer and related imports
from transformers import BertLayer, BertConfig, Trainer, TrainingArguments
import logging
from dataclasses import dataclass

# 全局图形设置
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 22,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

# 设置随机种子和设备
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################
# 配置管理：定义配置类
#########################
@dataclass
class ModelConfig:
    feature_dim: int = 0  # 数据集确定后赋值
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    attention_heads: int = 8
    transformer_layers: int = 2

#########################
# 数据加载模块（添加异常捕获）
#########################
def load_data():
    """
    [数据加载模块]
    - 从 CSV 文件中加载可再生能源和负载数据，
    - 按时间戳合并、排序、重置索引。
    """
    try:
        renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data10.csv')
        load_df      = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data10.csv')
    except Exception as e:
        logging.error("加载 CSV 文件失败: %s", e)
        raise e

    try:
        renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])
        load_df['timestamp']      = pd.to_datetime(load_df['timestamp'])
    except Exception as e:
        logging.error("时间戳转换失败: %s", e)
        raise e

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')
    data_df.sort_values('timestamp', inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    return data_df

#########################
# 特征工程模块（保持原有逻辑）
#########################
def feature_engineering(data_df):
    """
    [特征工程模块]
    - 使用 EWMA 平滑 'E_grid'
    - 构建时间特征（weekday, hour, month 及其 sin/cos 变换）
    - 对部分类别特征应用 LabelEncoder
    """
    span = 8
    data_df['E_grid'] = data_df['E_grid'].ewm(span=span, adjust=False).mean()
    
    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek
    data_df['hour'] = data_df['timestamp'].dt.hour
    data_df['month'] = data_df['timestamp'].dt.month

    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin'] = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos'] = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin'] = np.sin(2 * np.pi * (data_df['month'] - 1) / 12)
    data_df['month_cos'] = np.cos(2 * np.pi * (data_df['month'] - 1) / 12)
    
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
    target_column = 'E_grid'
    return data_df, feature_columns, target_column

#########################
# 序列构造模块（保持原有逻辑）
#########################
def create_sequences(X_data, y_data, window_size):
    X_list, y_list = [], []
    num_samples = X_data.shape[0]
    for i in range(num_samples - window_size):
        seq_x = X_data[i: i + window_size, :]
        seq_y = y_data[i + window_size]
        X_list.append(seq_x)
        y_list.append(seq_y)
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    return X_arr, y_arr

#########################
# 模型定义模块（添加 Transformer 编码器，整合 LSTM 和注意力）
#########################
class EModel_FeatureWeight(nn.Module):
    """
    [整合 LSTM、注意力与 Transformer Encoder 的模型]
    """
    def __init__(self, config: ModelConfig):
        super(EModel_FeatureWeight, self).__init__()
        self.feature_dim = config.feature_dim
        
        # 特征门控机制
        self.feature_gate = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Sigmoid()
        )
        
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0
        )
        self._init_lstm_weights()

        # Transformer Encoder 层（使用 BertLayer）
        bert_config = BertConfig(
            hidden_size=2 * config.lstm_hidden_size,
            num_hidden_layers=config.transformer_layers,
            num_attention_heads=config.attention_heads,
            intermediate_size=4 * config.lstm_hidden_size,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            output_hidden_states=False,
            is_decoder=False
        )
        self.transformer_encoder = nn.ModuleList([
            BertLayer(bert_config) for _ in range(config.transformer_layers)
        ])
        
        # 全局注意力聚合（简单实现）
        self.attention = nn.Sequential(
            nn.Linear(2 * config.lstm_hidden_size, 2 * config.lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(2 * config.lstm_hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 最终全连接层
        self.fc = nn.Sequential(
            nn.Linear(4 * config.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 输出 mu 和 logvar
        )

    def _init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)

    def forward(self, x, labels=None):
        # 动态特征加权
        gate = self.feature_gate(torch.mean(x, dim=1))
        x = x * gate.unsqueeze(1)
        
        # LSTM 处理
        lstm_out, _ = self.lstm(x)   # [batch, seq_len, 2 * hidden_size]
        
        # Transformer Encoder 处理
        for layer in self.transformer_encoder:
            lstm_out = layer(lstm_out)[0]
        
        # 全局注意力聚合
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_applied = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, 2 * hidden_size]
        
        # 拼接最后时刻输出和注意力聚合后的特征，得到最终预测
        output = self.fc(torch.cat([lstm_out[:, -1, :], attn_applied], dim=1))
        mu, logvar = torch.chunk(output, 2, dim=1)
        noise = 0.1 * torch.randn_like(mu, device=x.device) * torch.exp(0.5 * logvar)
        pred = mu + noise
        pred = pred.squeeze(-1)  # 形状：[batch]
        
        # 若传入标签则计算损失，返回符合 Trainer 要求的字典格式
        if labels is not None:
            loss = F.mse_loss(pred, labels)
            return {"loss": loss, "logits": pred}
        return {"logits": pred}

#########################
# 自定义数据集：适配 Hugging Face Trainer
#########################
class CustomTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        # 返回的键与模型 forward 参数对应：'x' 和 'labels'
        return {
            "x": torch.tensor(self.X[index], dtype=torch.float),
            "labels": torch.tensor(self.y[index], dtype=torch.float)
        }

#########################
# 实时推理流处理器：用于在线或流式预测
#########################
class StreamProcessor:
    def __init__(self, model, window_size):
        self.model = model
        self.window_size = window_size
        self.buffer = []
    def process_stream(self, new_data):
        """
        参数：
          new_data: 新的观测数据（列表形式）
        返回：
          当缓冲数据达到窗口长度时，返回预测值，否则返回 None
        """
        self.buffer.extend(new_data)
        if len(self.buffer) >= self.window_size:
            seq = np.array(self.buffer[-self.window_size:], dtype=np.float32)
            seq = torch.tensor(seq).unsqueeze(0)  # [1, window_size, feature_dim]
            with torch.no_grad():
                outputs = self.model(seq.to(device))
                prediction = outputs["logits"]
            return prediction.item()
        return None

#########################
# 计算指标函数（用于 Trainer 评估）
#########################
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return {"rmse": rmse}

#########################
# 主函数：数据加载、预处理、训练以及实时推理验证
#########################
def main(use_log_transform=True, min_egrid_threshold=1.0, window_size=20):
    # 数据加载（包含异常捕获）
    data_df = load_data()
    
    # 特征工程
    data_df, feature_cols, target_col = feature_engineering(data_df)
    
    # 过滤掉 E_grid 为0及小于阈值的样本
    data_df = data_df[data_df['E_grid'] != 0]
    data_df = data_df[data_df[target_col] > min_egrid_threshold].copy()
    data_df.reset_index(drop=True, inplace=True)

    # 数据分割
    X_all_raw = data_df[feature_cols].values
    y_all_raw = data_df[target_col].values
    total_samples = len(data_df)
    train_size = int(0.8 * total_samples)
    val_size   = int(0.1 * total_samples)
    test_size  = total_samples - train_size - val_size

    X_train_raw = X_all_raw[:train_size]
    y_train_raw = y_all_raw[:train_size]
    X_val_raw = X_all_raw[train_size:train_size+val_size]
    y_val_raw = y_all_raw[train_size:train_size+val_size]
    X_test_raw = X_all_raw[train_size+val_size:]
    y_test_raw = y_all_raw[train_size+val_size:]

    # 可选：对目标值进行对数转换
    if use_log_transform:
        y_train_raw = np.log1p(y_train_raw)
        y_val_raw = np.log1p(y_val_raw)
        y_test_raw = np.log1p(y_test_raw)

    # 标准化特征和目标
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train = scaler_X.transform(X_train_raw)
    X_val = scaler_X.transform(X_val_raw)
    X_test = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler().fit(y_train_raw.reshape(-1, 1))
    y_train = scaler_y.transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

    # 构造时序序列
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, window_size)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)

    print(f"[Info] 训练/验证/测试样本数: {X_train_seq.shape[0]}, {X_val_seq.shape[0]}, {X_test_seq.shape[0]}")
    print(f"[Info] 特征维度: {X_train_seq.shape[-1]}, 窗口大小: {window_size}")

    # 配置参数，设置模型输入维度
    config = ModelConfig(
        feature_dim=X_train_seq.shape[-1],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        lstm_dropout=0.2,
        attention_heads=8,
        transformer_layers=2
    )
    model = EModel_FeatureWeight(config).to(device)

    # 构建自定义数据集，用于 Hugging Face Trainer
    train_dataset = CustomTensorDataset(X_train_seq, y_train_seq)
    val_dataset = CustomTensorDataset(X_val_seq, y_val_seq)
    test_dataset = CustomTensorDataset(X_test_seq, y_test_seq)

    # 定义训练参数（使用混合精度 fp16 并记录日志）
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=150,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=1e-4,
        weight_decay=3e-4,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        fp16=True
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 开始训练（Trainer 内部会进行评估与保存最佳模型）
    trainer.train()

    # 在测试集上评估
    test_results = trainer.predict(test_dataset)
    print("\n测试集评估结果:")
    print(test_results.metrics)

    # 实时推理示例：使用流处理器进行在线预测
    stream_processor = StreamProcessor(model, window_size)
    # 模拟新数据：取测试集序列中最后一条的部分数据，追加新的观测值
    new_data = list(X_test_seq[-1])
    new_prediction = stream_processor.process_stream(new_data)
    print("流式预测结果:", new_prediction)

if __name__ == "__main__":
    main(use_log_transform=True, min_egrid_threshold=1.0, window_size=20)