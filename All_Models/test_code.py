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

from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lion_pytorch import Lion

# Hugging Face Trainer and related imports
from transformers import BertLayer, BertConfig, Trainer, TrainingArguments
import logging
from dataclasses import dataclass
from tqdm.auto import tqdm  # 用于自定义进度条（如果需要手动控制）

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
# 绘图模块（整合 All_Models_EGrid_Paper 中的绘图比较方法，采用非阻塞更新）
#########################
def plot_predictions_comparison(y_actual, y_pred1, y_pred2,
                                model1_name="Model1",
                                model2_name="Model2"):
    """
    [Visualization Module - Prediction Comparison]
    - Compare and plot the actual values with the predictions from two models.
    - 使用 plt.draw() 与 plt.pause() 避免每次调用时重复清空进度条显示
    """
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(len(y_actual))
    plt.plot(x_axis, y_actual, 'red', label='Actual', linewidth=1)
    plt.plot(x_axis, y_pred1, 'lightgreen', label=model1_name, linewidth=1)
    plt.plot(x_axis, y_pred2, 'skyblue', label=model2_name, linewidth=1)
    plt.xlabel('Index')
    plt.ylabel('Value (Real Domain)')
    plt.title(f'Actual vs {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 非阻塞方式更新图像，不会关闭进度条区域
    plt.draw()
    plt.pause(0.001)

def plot_test_predictions_over_time(timestamps, y_actual, y_pred):
    """
    [Visualization Module - Test Set Predictions]
    - Plot actual and predicted values over time for the test set.
    - 使用非阻塞式绘图，保证进度条不被重新绘制
    """
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, y_actual, color='red', label='Actual', linewidth=1)
    plt.plot(timestamps, y_pred, color='blue', label='Predicted', 
             linewidth=1, linestyle='--')
    plt.xlabel('Timestamp')
    plt.ylabel('Value (Real Domain)')
    plt.title('Test Set: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

#########################
# 数据加载、特征工程、序列构造等函数……
#########################
def load_data():
    # 示例代码：加载 CSV 数据
    try:
        df = pd.read_csv('data.csv')
    except Exception as e:
        logging.error("加载 CSV 文件失败: %s", e)
        raise e
    return df

def feature_engineering(df):
    # 示例代码：返回处理后的数据、特征列和目标列
    feature_cols = [c for c in df.columns if c != 'target']
    target_col = 'target'
    return df, feature_cols, target_col

def create_sequences(X, y, window_size):
    # 构造用于时序模型的滑动窗口数据
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

#########################
# 自定义数据集（返回键统一为 x 与 labels）
#########################
class CustomTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __getitem__(self, index):
        return {"x": self.X[index], "labels": self.y[index]}
    def __len__(self):
        return len(self.X)

#########################
# 模型定义（示例，需保证 forward 接受 x 和 labels）
#########################
class EModel_FeatureWeight(nn.Module):
    def __init__(self, config: ModelConfig):
        super(EModel_FeatureWeight, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size = config.feature_dim,
            hidden_size = config.lstm_hidden_size,
            num_layers = config.lstm_num_layers,
            dropout = config.lstm_dropout,
            batch_first=True
        )
        # 添加 Transformer Encoder 层（示例，简单封装 BertLayer）
        bert_config = BertConfig(
            hidden_size = config.lstm_hidden_size,
            num_attention_heads = config.attention_heads,
            num_hidden_layers = config.transformer_layers,
            intermediate_size = config.lstm_hidden_size * 4,
            hidden_dropout_prob = config.lstm_dropout,
            attention_probs_dropout_prob = config.lstm_dropout
        )
        self.transformer_layers = nn.ModuleList([BertLayer(bert_config) for _ in range(config.transformer_layers)])
        self.fc = nn.Linear(config.lstm_hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, labels=None):
        # x: [batch_size, window_size, feature_dim]
        lstm_out, _ = self.lstm(x)  # 输出 shape: [batch_size, window_size, lstm_hidden_size]
        # 取序列的最后一时刻输出
        out = lstm_out[:, -1, :]  # shape: [batch_size, lstm_hidden_size]
        # 经过 Transformer Encoder 层（示例：简单堆叠，不考虑 mask 等细节）
        for layer in self.transformer_layers:
            out = layer(out.unsqueeze(1))[0].squeeze(1)
        pred = self.fc(out).squeeze(-1)  # shape: [batch_size]
        loss = None
        if labels is not None:
            loss = self.loss_fn(pred, labels)
        # 返回 Trainer 所需的字典，需要包含 loss 与 logits
        if loss is not None:
            return {"loss": loss, "logits": pred}
        else:
            return {"logits": pred}

#########################
# 计算指标函数（用于 Trainer 评估）
#########################
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return {"rmse": rmse}

#########################
# 主函数：数据加载、预处理、训练、评估以及实时推理示例
#########################
def main(use_log_transform=True, min_egrid_threshold=1.0, window_size=20):
    # 数据加载与预处理
    data_df = load_data()
    data_df, feature_cols, target_col = feature_engineering(data_df)
    data_df = data_df[data_df[target_col] > min_egrid_threshold].copy()
    data_df.reset_index(drop=True, inplace=True)

    X_all_raw = data_df[feature_cols].values
    y_all_raw = data_df[target_col].values

    total_samples = len(data_df)
    train_size = int(0.8 * total_samples)
    val_size   = int(0.1 * total_samples)
    test_size  = total_samples - train_size - val_size

    X_train_raw = X_all_raw[:train_size]
    y_train_raw = y_all_raw[:train_size]
    X_val_raw   = X_all_raw[train_size:train_size+val_size]
    y_val_raw   = y_all_raw[train_size:train_size+val_size]
    X_test_raw  = X_all_raw[train_size+val_size:]
    y_test_raw  = y_all_raw[train_size+val_size:]

    if use_log_transform:
        y_train_raw = np.log1p(y_train_raw)
        y_val_raw   = np.log1p(y_val_raw)
        y_test_raw  = np.log1p(y_test_raw)

    scaler_X = StandardScaler().fit(X_train_raw)
    X_train = scaler_X.transform(X_train_raw)
    X_val   = scaler_X.transform(X_val_raw)
    X_test  = scaler_X.transform(X_test_raw)
    scaler_y = StandardScaler().fit(y_train_raw.reshape(-1,1))
    y_train = scaler_y.transform(y_train_raw.reshape(-1,1)).ravel()
    y_val   = scaler_y.transform(y_val_raw.reshape(-1,1)).ravel()
    y_test  = scaler_y.transform(y_test_raw.reshape(-1,1)).ravel()

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   window_size)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  window_size)

    print(f"[Info] 训练/验证/测试样本数: {X_train_seq.shape[0]}, {X_val_seq.shape[0]}, {X_test_seq.shape[0]}")
    print(f"[Info] 特征维度: {X_train_seq.shape[-1]}, 窗口大小: {window_size}")

    # 配置模型超参数，并构建模型
    config = ModelConfig(feature_dim=X_train_seq.shape[-1],
                         lstm_hidden_size=128,
                         lstm_num_layers=2,
                         lstm_dropout=0.2,
                         attention_heads=8,
                         transformer_layers=2)
    model = EModel_FeatureWeight(config).to(device)

    # 使用自定义数据集，返回键为 "x" 与 "labels"
    train_dataset = CustomTensorDataset(X_train_seq, y_train_seq)
    val_dataset   = CustomTensorDataset(X_val_seq,   y_val_seq)
    test_dataset  = CustomTensorDataset(X_test_seq,  y_test_seq)

    # 定义 Trainer 训练参数，并通过 tqdm_kwargs 固定进度条位置
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
        fp16=True,
        disable_tqdm=False,
        tqdm_kwargs={"position": 0, "leave": True}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 开始训练与评估（Trainer 内部自动调用进度条，不会反复重绘）
    trainer.train()
    test_results = trainer.predict(test_dataset)
    print("\n测试集评估结果:")
    print(test_results.metrics)

    # 示例：对部分测试序列进行实时流式推理（此处仅为示例）
    # 假设 stream_processor 已经按照需求定义完成
    stream_processor = StreamProcessor(model, window_size)
    new_data = list(X_test_seq[-1])
    new_prediction = stream_processor.process_stream(new_data)
    print("流式预测结果:", new_prediction)

    # 评估结束后调用绘图比较模块（对预测结果进行可视化）
    # 假设经过逆标准化与逆对数变换处理后得到原始域数据：
    # 此处示例中 y_test_seq 仅供参考，实际需根据你的数据处理流程调整
    preds_real = scaler_y.inverse_transform(test_results.predictions.reshape(-1, 1)).ravel()
    labels_real = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
    # 使用绘图函数展示测试集中实际值与预测值对比（示例）
    test_timestamps = data_df['timestamp'].values[train_size+val_size+window_size:]
    plot_test_predictions_over_time(test_timestamps, labels_real, preds_real)
    # 如果有两个模型对比，可调用：
    plot_predictions_comparison(labels_real, preds_real, preds_real, 
                                model1_name='EModel_FeatureWeight', model2_name='EModel_Another')
    
    # 绘图后保持窗口开启
    plt.show()

# 示例流处理器（实时推理，不影响进度条绘制）
class StreamProcessor:
    def __init__(self, model, window_size):
        self.model = model
        self.window_size = window_size
        self.buffer = []

    def process_stream(self, new_data):
        self.buffer.extend(new_data)
        if len(self.buffer) >= self.window_size:
            seq = torch.tensor(self.buffer[-self.window_size:], dtype=torch.float32)
            seq = seq.unsqueeze(0).to(device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(seq)
            return outputs["logits"].item()
        return None

if __name__ == "__main__":
    main(use_log_transform=True, min_egrid_threshold=1.0, window_size=20)