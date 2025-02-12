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

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from torch.nn.utils import clip_grad_norm_
from x_transformers import RMSNorm
from lion_pytorch import Lion
from rotary_embedding_torch import RotaryEmbedding


# 【图形全局样式设置】
mpl.rcParams.update({
    'font.size': 18,          # 全局默认字体大小
    'axes.labelsize': 18,     # 坐标轴标签字体大小
    'axes.titlesize': 20,     # 图表标题字体大小
    'xtick.labelsize': 16,    # x轴刻度字体大小
    'ytick.labelsize': 16     # y轴刻度字体大小
})

# 【全局超参数】
learning_rate     = 1e-4   # 学习率
num_epochs        = 150    # 训练轮数
batch_size        = 128    # 批处理大小
weight_decay      = 1e-5   # 权重衰减
patience          = 12     # 耐心值
num_workers       = 0      # 工作线程数
window_size       = 20     # 序列窗口大小
lstm_hidden_size  = 128    # LSTM隐藏层大小
lstm_num_layers   = 2      # LSTM层数

# 【随机种子与设备设置】
torch.manual_seed(42)       # 随机种子 = 42
np.random.seed(42)          # 随机种子 = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  


# 1. 数据加载模块
def load_data():
    """
    [数据加载模块]
    - 从 CSV 文件加载可再生能源数据和负荷数据
    - 根据时间戳进行合并、排序，并重置索引
    """
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data10.csv')
    load_df      = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data10.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])  # timestamp转换为datetime
    load_df['timestamp']      = pd.to_datetime(load_df['timestamp'])       # timestamp转换为datetime

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')  # 内连接合并数据
    data_df.sort_values('timestamp', inplace=True)                        # 按timestamp排序
    data_df.reset_index(drop=True, inplace=True)                           # 重置索引
    return data_df

# 2. 特征工程模块
def feature_engineering(data_df):
    """
    [特征工程模块]
    - 对 'E_grid' 进行 EWMA 平滑 (span = 10)
    - 构造时间特征：dayofweek, hour, month 及其 sin/cos变换
    - 对分类特征进行 LabelEncoder 编码
    - 返回：处理后数据、特征列列表、目标列名
    """
    span = 10  # EWMA平滑参数
    data_df['E_grid'] = data_df['E_grid'].ewm(span = span, adjust = False).mean()  # E_grid平滑

    # 构造时间特征
    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek  # dayofweek = 星期几
    data_df['hour']      = data_df['timestamp'].dt.hour       # hour = 小时
    data_df['month']     = data_df['timestamp'].dt.month      # month = 月份

    # sin/cos变换
    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month'] - 1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month'] - 1) / 12)

    # 分类特征编码
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
            data_df[col] = le.fit_transform(data_df[col].astype(str))  # 编码：col = 转换后的数值

    time_feature_cols = [
        'dayofweek_sin', 'dayofweek_cos',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos'
    ]
    feature_columns = renewable_features + load_features + time_feature_cols  # feature_columns = 所有特征列
    target_column   = 'E_grid'
    
    return data_df, feature_columns, target_column


# 3. 序列构造模块
def create_sequences(X_data, y_data, window_size):
    """
    [序列构造模块]
    - 根据给定窗口大小构造时序数据
    参数:
      X_data: 特征数据 (numpy数组)
      y_data: 目标数据 (numpy数组)
      window_size: 序列窗口大小 = window_size
    返回:
      X_arr: 序列化后的特征数据
      y_arr: 对应目标数据
    """
    X_list, y_list = [], []
    num_samples = X_data.shape[0]  # 样本总数 = num_samples
    for i in range(num_samples - window_size):  # 遍历构造序列
        seq_x = X_data[i : i + window_size, :]  # 当前窗口特征序列
        seq_y = y_data[i + window_size]           # 当前窗口对应目标值
        X_list.append(seq_x)
        y_list.append(seq_y)
    X_arr = np.array(X_list, dtype = np.float32)
    y_arr = np.array(y_list, dtype = np.float32)
    return X_arr, y_arr

# 4. 模型定义模块
class PositionalEncoding(nn.Module):
    """
    [位置编码模块]
    - 为输入序列添加位置编码
    参数:
      d_model: 模型维度 = d_model
      max_len: 最大序列长度 = 5000（默认）
    """
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # pe.shape = [max_len, d_model]
        position = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)  # position.shape = [max_len, 1]
        div_term = torch.exp(-(torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model))  # div_term.shape = [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位使用cos
        pe = pe.unsqueeze(1)  # pe.shape = [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x, step_offset = 0):
        """
        参数:
          x: 输入张量, shape = [batch_size, seq_len, d_model]
          step_offset: 序列偏移量 (默认 0)
        返回:
          添加位置编码后的 x
        """
        seq_len = x.size(1)  # 序列长度 = seq_len
        pos_enc = self.pe[step_offset : step_offset + seq_len, 0, :]  # 取对应位置编码
        return x + pos_enc.unsqueeze(0)

class Transformer(nn.Module):
    """
    [Transformer模块]
    - 包含编码器和解码器结构
    参数:
      d_model: 模型维度 = d_model
      nhead: 注意力头数 = nhead
      num_encoder_layers: 编码器层数 = num_encoder_layers
      num_decoder_layers: 解码器层数 = num_decoder_layers
      dropout: dropout概率 = dropout
    """
    def __init__(self, d_model, nhead = 12, num_encoder_layers = 3, num_decoder_layers = 3, dropout = 0.1):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim = d_model // 2)  # rotary_emb：旋转位置编码（未使用于forward中）
        
        # 编码器层定义
        encoder_layer = nn.TransformerEncoderLayer(
            d_model          = d_model,
            nhead            = nhead,
            dim_feedforward  = 4 * d_model,
            dropout          = dropout,
            batch_first      = True,
            activation       = 'gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers = num_encoder_layers,
            norm       = RMSNorm(d_model)         # 使用 RMSNorm 归一化
        )

        # 解码器层定义
        decoder_layer = nn.TransformerDecoderLayer(
            d_model          = d_model,
            nhead            = nhead,
            dim_feedforward  = 4 * d_model,
            dropout          = dropout,
            batch_first      = True,
            activation       = 'gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers = num_decoder_layers,
            norm       = RMSNorm(d_model)
        )

    def forward(self, src, tgt):
        """
        参数:
          src: 源输入张量
          tgt: 目标输入张量
        返回:
          Transformer解码器输出
        """
        memory = self.transformer_encoder(src)  # memory = 编码器输出
        return self.transformer_decoder(tgt, memory)

class CNNBlock(nn.Module):
    """
    [CNN模块]
    - 通过多层卷积进行特征提取
    参数:
      feature_dim: 输入特征维度 = feature_dim
      hidden_size: 隐藏层大小 = hidden_size
      dropout: dropout概率 = dropout
    """
    def __init__(self, feature_dim, hidden_size, dropout = 0.1):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, hidden_size, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size = 7, padding = 3)

        self.bn1 = nn.BatchNorm1d(hidden_size)        # bn1: 对应conv1输出
        self.bn2 = nn.BatchNorm1d(hidden_size)        # bn2: 对应conv2输出
        self.bn3 = nn.BatchNorm1d(2 * hidden_size)      # bn3: 对应conv3输出

        self.dropout = nn.Dropout(dropout)            # dropout = dropout
        self.relu    = nn.ReLU()                      # 激活函数：ReLU

    def forward(self, x):
        """
        参数:
          x: 输入张量, shape = [batch_size, seq_len, feature_dim]
        返回:
          CNN特征输出, shape = [batch_size, seq_len, 2*hidden_size]
        """
        x = x.transpose(1, 2)  # 转换为 [batch_size, feature_dim, seq_len]
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
        x = x.transpose(1, 2)  # 恢复为 [batch_size, seq_len, 2*hidden_size]
        return x

class Attention(nn.Module):
    """
    [注意力模块]
    - 对输入进行注意力加权汇聚
    参数:
      input_dim: 输入维度 = input_dim
      dropout: dropout概率 = dropout
    """
    def __init__(self, input_dim, dropout = 0.1):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # 映射维度：input_dim -> input_dim
            nn.Tanh(),
            nn.Linear(input_dim, 1)             # 映射为单值
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        参数:
          x: 输入张量, shape = [batch_size, seq_len, input_dim]
        返回:
          汇聚后的特征, shape = [batch_size, input_dim]
        """
        attn_weights = self.attention(x)       # attn_weights.shape = [batch_size, seq_len, 1]
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim = 1)  # 对时间步维度做softmax
        weighted = x * attn_weights            # 元素级加权
        return torch.sum(weighted, dim = 1)    # 汇聚为 [batch_size, input_dim]

class EModel_FeatureWeight(nn.Module):
    """
    [模型一：特征加权基于LSTM的模型]
    参数:
      - feature_dim: 输入特征维度 = feature_dim
      - lstm_hidden_size: LSTM隐藏层大小 = lstm_hidden_size
      - lstm_num_layers: LSTM层数 = lstm_num_layers
      - lstm_dropout: LSTM dropout概率 = lstm_dropout
      - use_local_attn: 是否使用局部注意力，默认为 False
      - local_attn_window_size: 局部注意力窗口大小，默认为 5
    """
    def __init__(self, 
                 feature_dim, 
                 lstm_hidden_size = 128, 
                 lstm_num_layers = 2, 
                 lstm_dropout = 0.2,
                 use_local_attn = False,
                 local_attn_window_size = 5
                ):
        super(EModel_FeatureWeight, self).__init__()
        self.feature_dim = feature_dim
        
        # 特征门控机制：全连接层 + Sigmoid，计算各特征权重
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),  # 映射：feature_dim -> feature_dim
            nn.Sigmoid()
        )
        
        # Temporal attention：选择局部或全局（MLP）注意力
        if use_local_attn:
            from local_attention.local_attention import LocalAttention
            self.temporal_attn = LocalAttention(
                dim = 2 * lstm_hidden_size,
                local_window_size = local_attn_window_size,
                causal = False                      # causal = False（非自回归模式）
            )
        else:
            self.temporal_attn = Attention(input_dim = 2 * lstm_hidden_size)
        
        # 特征注意力层：对 LSTM 输出在特征维度上加权汇聚
        self.feature_attn = nn.Sequential(
            nn.Linear(window_size, 1),  # 映射：窗口大小 = window_size -> 1
            nn.Sigmoid()
        )
        # 特征投影层，将注意力后的结果映射到 2*lstm_hidden_size
        self.feature_proj = nn.Linear(2 * lstm_hidden_size, 2 * lstm_hidden_size)
        
        # 可学习的特征重要性权重，初始值全1
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad = True)
        
        # 双向 LSTM 模块，用于捕捉时序信息
        self.lstm = nn.LSTM(
            input_size    = feature_dim,
            hidden_size   = lstm_hidden_size,
            num_layers    = lstm_num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        self._init_lstm_weights()  # 按照 min-LSTM 论文初始化LSTM权重
        
        # 全局注意力模块，对LSTM输出进行汇聚
        self.attention = Attention(input_dim = 2 * lstm_hidden_size)
        
        # 全连接层：最终预测
        self.fc = nn.Sequential(
            nn.Linear(4 * lstm_hidden_size, 128),  # 输入维度 = 4*lstm_hidden_size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)                       # 输出维度 = 2，分别对应 mu 和 logvar
        )

    def _init_lstm_weights(self):
        """
        [LSTM权重初始化]
        - 按照 min-LSTM 论文方法初始化权重
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)  # 设置遗忘门偏置为1

    def forward(self, x):
        """
        参数:
          x: 输入张量, shape = [batch_size, seq_len, feature_dim]
        返回:
          输出预测值, shape = [batch_size]
        """
        # 动态特征加权：计算特征门并加权
        gate = self.feature_gate(x.mean(dim = 1))  # gate.shape = [batch_size, feature_dim]
        x = x * gate.unsqueeze(1)                   # 加权后的 x, shape = [batch_size, seq_len, feature_dim]

        # LSTM处理，获得双向LSTM输出
        lstm_out, _ = self.lstm(x)  # lstm_out.shape = [batch_size, seq_len, 2*lstm_hidden_size]

        # 时间注意力：对LSTM输出进行加权汇聚
        temporal = self.temporal_attn(lstm_out)  # temporal.shape = [batch_size, 2*lstm_hidden_size]
        
        # 特征注意力：对特征维度进行加权汇聚
        feature_raw = self.feature_attn(lstm_out.transpose(1, 2))  # 先转置，再映射, shape = [batch_size, 2*lstm_hidden_size, 1]
        feature_raw = feature_raw.squeeze(-1)                      # squeeze后 shape = [batch_size, 2*lstm_hidden_size]
        feature = self.feature_proj(feature_raw)                   # 投影后 feature.shape = [batch_size, 2*lstm_hidden_size]

        # 拼接两个分支：temporal 与 feature，合并后 shape = [batch_size, 4*lstm_hidden_size]
        combined = torch.cat([temporal, feature], dim = 1)
        output = self.fc(combined)  # 全连接层映射到2维
        mu, logvar = torch.chunk(output, 2, dim = 1)  # 拆分为 mu 和 logvar
        
        # reparameterization trick: 生成噪声并与 mu 相加
        noise = 0.1 * torch.randn_like(mu, device = x.device) * torch.exp(0.5 * logvar)
        output = mu + noise
        
        return output.squeeze(-1)

class EModel_CNN_Transformer(nn.Module):
    """
    [模型二：基于CNN和Transformer的模型]
    参数:
      - feature_dim: 输入特征维度 = feature_dim
      - hidden_size: CNN隐藏层大小 = hidden_size
      - num_layers: Transformer层数 = num_layers
      - dropout: dropout概率 = dropout
    """
    def __init__(self, feature_dim, hidden_size = 128, num_layers = 2, dropout = 0.1):
        super().__init__()
        self.feature_importance = nn.Parameter(torch.ones(feature_dim))  # 可学习特征权重, shape = [feature_dim]
        
        self.cnn_block = CNNBlock(feature_dim, hidden_size, dropout)       # CNN模块
        
        self.transformer_block = Transformer(
            d_model            = 2 * hidden_size,  # d_model = 2*hidden_size
            nhead              = 8,
            num_encoder_layers = num_layers,
            num_decoder_layers = num_layers,
            dropout            = dropout
        )
        
        self.attention = Attention(input_dim = 2 * hidden_size, dropout = dropout)  # 注意力模块
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),  # 映射: 2*hidden_size -> 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)                 # 映射: 128 -> 1（最终预测）
        )
        
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, 2 * hidden_size),  # 残差映射: feature_dim -> 2*hidden_size
            nn.GELU()
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        参数:
          x: 输入张量, shape = [batch_size, seq_len, feature_dim]
        返回:
          预测值, shape = [batch_size]
        """
        # 应用特征重要性权重
        x = x * self.feature_importance.unsqueeze(0).unsqueeze(0)
        residual = self.residual(x[:, -1, :])  # 取序列最后一步, shape = [batch_size, 2*hidden_size]
        
        cnn_out = self.cnn_block(x)  # CNN模块输出, shape = [batch_size, seq_len, 2*hidden_size]
        transformer_out = self.transformer_block(cnn_out, cnn_out)  # Transformer模块输出, shape 同上
        transformer_out += residual.unsqueeze(1)  # 加入残差
        
        attn_out = self.attention(transformer_out)  # 注意力汇聚, shape = [batch_size, 2*hidden_size]
        return self.fc(attn_out).squeeze(-1)


# 5. 评估模块
def evaluate(model, dataloader, criterion, device = 'cuda'):
    """
    [评估模块]
    - 在给定数据集上计算损失以及多种指标：
      RMSE, MAPE, R², SMAPE, MAE
    参数:
      model: 待评估模型
      dataloader: 数据加载器
      criterion: 损失函数
      device: 计算设备 = device
    返回:
      val_loss, rmse, mape, r2, smape, mae, preds, labels
    """
    model.eval()
    running_loss, num_samples = 0.0, 0
    preds_list, labels_list = [], []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs  = batch_inputs.to(device)
            batch_labels  = batch_labels.to(device)

            outputs = model(batch_inputs)
            loss    = criterion(outputs, batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            num_samples  += batch_inputs.size(0)

            preds_list.append(outputs.cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())

    val_loss   = running_loss / num_samples
    preds_arr  = np.concatenate(preds_list, axis = 0)
    labels_arr = np.concatenate(labels_list, axis = 0)

    # RMSE计算
    rmse_std = np.sqrt(mean_squared_error(labels_arr, preds_arr))

    # MAPE计算（过滤标签为0的情况）
    nonzero_mask_mape = (labels_arr != 0)
    if np.sum(nonzero_mask_mape) > 0:
        mape_std = np.mean(np.abs((labels_arr[nonzero_mask_mape] - preds_arr[nonzero_mask_mape]) / labels_arr[nonzero_mask_mape])) * 100.0
    else:
        mape_std = 0.0

    # R²计算
    ss_res = np.sum((labels_arr - preds_arr) ** 2)
    ss_tot = np.sum((labels_arr - np.mean(labels_arr)) ** 2)
    r2_std = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # SMAPE计算
    numerator = np.abs(labels_arr - preds_arr)
    denominator = np.abs(labels_arr) + np.abs(preds_arr)
    nonzero_mask_smape = (denominator != 0)
    if np.sum(nonzero_mask_smape) > 0:
        smape_val = 100.0 * 2.0 * np.mean(numerator[nonzero_mask_smape] / denominator[nonzero_mask_smape])
    else:
        smape_val = 0.0

    # MAE计算
    mae_val = np.mean(np.abs(labels_arr - preds_arr))

    return val_loss, rmse_std, mape_std, r2_std, smape_val, mae_val, preds_arr, labels_arr


# 6. 训练模块
def train_model(model, train_loader, val_loader, model_name = 'Model', learning_rate = 1e-4, weight_decay = 1e-2, num_epochs = num_epochs):
    """
    [训练模块]
    - 在训练集和验证集上训练模型，同时记录各项指标
    - 实现Early Stopping策略
    参数:
      model: 待训练模型
      train_loader: 训练集数据加载器
      val_loader: 验证集数据加载器
      model_name: 模型名称 = model_name
      learning_rate: 学习率 = learning_rate
      weight_decay: 权重衰减 = weight_decay
      num_epochs: 训练轮数 = num_epochs
    返回:
      各项指标历史记录字典
    """
    criterion = nn.MSELoss()
    optimizer = Lion(
        model.parameters(),
        lr           = learning_rate,  # learning_rate = learning_rate
        weight_decay = weight_decay    # weight_decay = weight_decay
    )

    total_steps  = num_epochs * len(train_loader)  # 总步数 = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)           # warmup_steps = 10% 总步数

    scheduler = LambdaLR(optimizer, lambda step: 
        min(step / warmup_steps, 1.0) if step < warmup_steps else 
        max(0.0, 1 - (step - warmup_steps) / (total_steps - warmup_steps))
    )
    best_val_loss = float('inf')
    counter       = 0
    global_step   = 0

    # 历史记录初始化
    train_loss_history   = []
    train_rmse_history   = []
    train_mape_history   = []
    train_r2_history     = []
    train_smape_history  = []
    train_mae_history    = []

    val_loss_history     = []
    val_rmse_history     = []
    val_mape_history     = []
    val_r2_history       = []
    val_smape_history    = []
    val_mae_history      = []

    for epoch in range(num_epochs):
        # 【训练阶段】
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

        train_loss_epoch = running_loss / num_samples

        # 在训练集上计算指标
        train_loss_eval, train_rmse_eval, train_mape_eval, train_r2_eval, train_smape_eval, train_mae_eval, _, _ = evaluate(model, train_loader, criterion)
        # 在验证集上计算指标
        val_loss_eval, val_rmse_eval, val_mape_eval, val_r2_eval, val_smape_eval, val_mae_eval, _, _ = evaluate(model, val_loader, criterion)

        # 保存各指标历史记录
        train_loss_history.append(train_loss_eval)
        train_rmse_history.append(train_rmse_eval)
        train_mape_history.append(train_mape_eval)
        train_r2_history.append(train_r2_eval)
        train_smape_history.append(train_smape_eval)
        train_mae_history.append(train_mae_eval)

        val_loss_history.append(val_loss_eval)
        val_rmse_history.append(val_rmse_eval)
        val_mape_history.append(val_mape_eval)
        val_r2_history.append(val_r2_eval)
        val_smape_history.append(val_smape_eval)
        val_mae_history.append(val_mae_eval)

        # 打印日志信息
        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"TrainLoss: {train_loss_epoch:.4f}, "
              f"ValLoss: {val_loss_eval:.4f}, "
              f"ValRMSE(std): {val_rmse_eval:.4f}, "
              f"ValMAPE(%): {val_mape_eval:.2f}, "
              f"ValR^2: {val_r2_eval:.4f}, "
              f"ValSMAPE(%): {val_smape_eval:.2f}, "
              f"ValMAE(std): {val_mae_eval:.4f}")

        # Early Stopping判断
        if val_loss_eval < best_val_loss:
            best_val_loss = val_loss_eval
            counter       = 0
            torch.save(model.state_dict(), f"best_{model_name}.pth")  # 保存最佳模型权重
            print(f"[{model_name}] 模型已保存 (best val_loss = {best_val_loss:.4f}).")
        else:
            counter += 1
            if counter >= patience:
                print(f"[{model_name}] 验证集无改善，提前停止。")
                break

    return {
        "train_loss":   train_loss_history,
        "train_rmse":   train_rmse_history,
        "train_mape":   train_mape_history,
        "train_r2":     train_r2_history,
        "train_smape":  train_smape_history,
        "train_mae":    train_mae_history,
        "val_loss":     val_loss_history,
        "val_rmse":     val_rmse_history,
        "val_mape":     val_mape_history,
        "val_r2":       val_r2_history,
        "val_smape":    val_smape_history,
        "val_mae":      val_mae_history
    }


# 7. 可视化与辅助函数模块
def plot_correlation_heatmap(df, feature_cols, title = "Heat map"):
    """
    [可视化模块 - 相关性热图]
    - 绘制给定特征列的相关性热图
    参数:
      df: 数据集
      feature_cols: 特征列列表
      title: 图表标题 = title
    """
    df_encoded = df.copy()
    for col in feature_cols:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    corr_matrix = df_encoded[feature_cols].corr()

    colors_list = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("red_yellow_green", colors_list, N = 256)

    plt.figure(figsize = (8, 6))
    sns.heatmap(
        corr_matrix,
        cmap      = cmap_custom,
        annot     = True,
        fmt       = ".2f",
        square    = True,
        linewidths= 1,
        linecolor = 'white',
        cbar      = True,
        vmin      = -1,
        vmax      = 1
    )
    plt.title(title, fontsize = 16)
    plt.xticks(rotation = 45, ha = 'right')
    plt.tight_layout()
    plt.show()

def analyze_target_distribution(data_df, target_col):
    """
    [可视化模块 - 目标变量分布]
    - 输出目标变量基本统计信息并绘制直方图
    参数:
      data_df: 数据集
      target_col: 目标列名称 = target_col
    """
    if target_col not in data_df.columns:
        print(f"[Warning] '{target_col}' 不在数据集中，跳过分析。")
        return

    print(f"\n[Target Analysis] '{target_col}' 的基本统计信息：")
    print(data_df[target_col].describe())

    plt.figure(figsize = (6, 4))
    plt.hist(data_df[target_col], bins = 30, color = 'skyblue', edgecolor = 'black')
    plt.title(f"'{target_col}'原始分布")
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_Egrid_over_time(data_df):
    """
    [可视化模块 - 时间序列图]
    - 绘制数据集中 'E_grid' 随时间变化的折线图
    参数:
      data_df: 数据集
    """
    plt.figure(figsize = (10, 5))
    plt.plot(data_df['timestamp'], data_df['E_grid'], color = 'blue', marker = 'o', markersize = 3, linewidth = 1)
    plt.xlabel('Timestamp')
    plt.ylabel('E_grid')
    plt.title('全数据集：E_grid随时间变化')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_comparison(y_actual_real, y_pred_model1_real, y_pred_model2_real, model1_name = 'Model1', model2_name = 'Model2'):
    """
    [可视化模块 - 预测对比图]
    - 对比绘制实际值与两个模型的预测值
    参数:
      y_actual_real: 实际值
      y_pred_model1_real: 模型1预测值
      y_pred_model2_real: 模型2预测值
      model1_name: 模型1名称 = model1_name
      model2_name: 模型2名称 = model2_name
    """
    plt.figure(figsize = (10, 5))
    x_axis = np.arange(len(y_actual_real))
    plt.plot(x_axis, y_actual_real,      'red',      label = 'Actual', linewidth = 1)
    plt.plot(x_axis, y_pred_model1_real, 'lightgreen', label = model1_name, linewidth = 1)
    plt.plot(x_axis, y_pred_model2_real, 'skyblue',    label = model2_name, linewidth = 1)
    plt.xlabel('Index')
    plt.ylabel('Value (real domain)')
    plt.title(f'实际值 vs {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_test_predictions_over_time(test_timestamps, y_actual_real, y_pred_real):
    """
    [可视化模块 - 测试集预测图]
    - 绘制测试集上实际值与预测值随时间变化的对比图
    参数:
      test_timestamps: 测试集时间戳
      y_actual_real: 实际值
      y_pred_real: 预测值
    """
    plt.figure(figsize = (10, 5))
    plt.plot(test_timestamps, y_actual_real, color = 'red',  label = 'Actual E_grid', linewidth = 1)
    plt.plot(test_timestamps, y_pred_real,   color = 'blue', label = 'Predicted E_grid', linewidth = 1, linestyle = '--')
    plt.xlabel('Timestamp (Test Data)')
    plt.ylabel('E_grid (real domain)')
    plt.title('测试集：实际 vs 预测 E_grid')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_curves_allmetrics(hist_dict, model_name = 'Model'):
    """
    [可视化模块 - 训练曲线]
    - 同时绘制 Loss、RMSE、MAPE、R²、SMAPE、MAE 曲线（训练集和验证集）
    参数:
      hist_dict: 包含各指标历史记录的字典
      model_name: 模型名称 = model_name
    """
    epochs = range(1, len(hist_dict["train_loss"]) + 1)
    plt.figure(figsize = (15, 12))

    # Loss曲线
    plt.subplot(3, 2, 1)
    plt.plot(epochs, hist_dict["train_loss"], 'r-o', label = 'Train Loss', markersize = 4)
    plt.plot(epochs, hist_dict["val_loss"],   'b-o', label = 'Val Loss', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (std)')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    # RMSE曲线
    plt.subplot(3, 2, 2)
    plt.plot(epochs, hist_dict["train_rmse"], 'r-o', label = 'Train RMSE', markersize = 4)
    plt.plot(epochs, hist_dict["val_rmse"],   'b-o', label = 'Val RMSE', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (std)')
    plt.title('RMSE')
    plt.legend()
    plt.grid(True)

    # MAPE曲线
    plt.subplot(3, 2, 3)
    plt.plot(epochs, hist_dict["train_mape"], 'r-o', label = 'Train MAPE', markersize = 4)
    plt.plot(epochs, hist_dict["val_mape"],   'b-o', label = 'Val MAPE', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE')
    plt.legend()
    plt.grid(True)

    # R²曲线
    plt.subplot(3, 2, 4)
    plt.plot(epochs, hist_dict["train_r2"], 'r-o', label = 'Train R^2', markersize = 4)
    plt.plot(epochs, hist_dict["val_r2"],   'b-o', label = 'Val R^2', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('R^2')
    plt.title('R^2')
    plt.legend()
    plt.grid(True)

    # SMAPE曲线
    plt.subplot(3, 2, 5)
    plt.plot(epochs, hist_dict["train_smape"], 'r-o', label = 'Train SMAPE', markersize = 4)
    plt.plot(epochs, hist_dict["val_smape"],   'b-o', label = 'Val SMAPE', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('SMAPE (%)')
    plt.title('SMAPE')
    plt.legend()
    plt.grid(True)

    # MAE曲线
    plt.subplot(3, 2, 6)
    plt.plot(epochs, hist_dict["train_mae"], 'r-o', label = 'Train MAE', markersize = 4)
    plt.plot(epochs, hist_dict["val_mae"],   'b-o', label = 'Val MAE', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('MAE (std)')
    plt.title('MAE')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"训练曲线 - {model_name}", fontsize = 16)
    plt.tight_layout()
    plt.show()

# 8. 主函数
def main(use_log_transform = True, min_egrid_threshold = 1.0):
    """
    [主函数]
    - 流程：数据加载 -> 特征工程 -> 数据分割 -> 序列构造 -> 模型构建 -> 模型训练 -> 评估与可视化
    参数:
      use_log_transform: 是否对目标值进行对数变换 = True/False
      min_egrid_threshold: E_grid最小阈值，过滤过小值 = 1.0
    """
    print("[Info] 1) 正在加载原始数据...")
    data_df = load_data()

    # 绘制热力图（未过滤E_grid=0前）
    feature_cols_to_plot = ['season', 'holiday', 'weather', 'temperature', 'working_hours', 'E_grid']
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns]
    plot_correlation_heatmap(data_df, feature_cols_to_plot, title = "Heat map")

    # 过滤掉 E_grid = 0 的行
    data_df = data_df[data_df['E_grid'] != 0].copy()
    data_df.reset_index(drop = True, inplace = True)

    # 特征工程（不进行标准化，避免数据泄露）
    data_df, feature_cols, target_col = feature_engineering(data_df)

    # 过滤掉过小的 E_grid 值
    data_df = data_df[data_df[target_col] > min_egrid_threshold].copy()
    data_df.reset_index(drop = True, inplace = True)

    analyze_target_distribution(data_df, target_col)
    plot_Egrid_over_time(data_df)

    # 根据时间序列划分训练、验证、测试集
    X_all_raw = data_df[feature_cols].values
    y_all_raw = data_df[target_col].values
    timestamps_all = data_df['timestamp'].values

    total_samples = len(data_df)
    train_size = int(0.8 * total_samples)      # 训练集比例 = 80%
    val_size   = int(0.1 * total_samples)        # 验证集比例 = 10%
    test_size  = total_samples - train_size - val_size  # 测试集比例 = 剩余

    X_train_raw = X_all_raw[:train_size]
    y_train_raw = y_all_raw[:train_size]
    X_val_raw   = X_all_raw[train_size : train_size + val_size]
    y_val_raw   = y_all_raw[train_size : train_size + val_size]
    X_test_raw  = X_all_raw[train_size + val_size:]
    y_test_raw  = y_all_raw[train_size + val_size:]

    train_timestamps = timestamps_all[:train_size]
    val_timestamps   = timestamps_all[train_size : train_size + val_size]
    test_timestamps  = timestamps_all[train_size + val_size:]

    # 对目标值进行对数变换（如设定）
    if use_log_transform:
        y_train_raw = np.log1p(y_train_raw)
        y_val_raw   = np.log1p(y_val_raw)
        y_test_raw  = np.log1p(y_test_raw)

    # 对特征分别进行标准化：训练集 fit, 验证/测试集 transform
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train = scaler_X.transform(X_train_raw)
    X_val   = scaler_X.transform(X_val_raw)
    X_test  = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler().fit(y_train_raw.reshape(-1, 1))
    y_train  = scaler_y.transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val    = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test   = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

    # 构造时序数据序列
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   window_size)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  window_size)

    print(f"[Info] Train/Val/Test样本数: {X_train_seq.shape[0]}, {X_val_seq.shape[0]}, {X_test_seq.shape[0]}")
    print(f"[Info] 特征维度: {X_train_seq.shape[-1]}, 窗口大小: {window_size}")

    train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
    val_dataset   = TensorDataset(torch.from_numpy(X_val_seq),   torch.from_numpy(y_val_seq))
    test_dataset  = TensorDataset(torch.from_numpy(X_test_seq),  torch.from_numpy(y_test_seq))

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,  num_workers = num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size = batch_size, shuffle = False, num_workers = num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size = batch_size, shuffle = False, num_workers = num_workers)

    # 模型构建
    feature_dim = X_train_seq.shape[-1]
    modelA = EModel_FeatureWeight(
        feature_dim       = feature_dim,
        lstm_hidden_size  = lstm_hidden_size, 
        lstm_num_layers   = lstm_num_layers,
        lstm_dropout      = 0.2
    ).to(device)
    
    modelB = EModel_CNN_Transformer(
        feature_dim = feature_dim,
        hidden_size = 128,
        num_layers  = 2,
        dropout     = 0.1
    ).to(device)

    # 训练模型：EModel_FeatureWeight
    print("\n========== 训练模型：EModel_FeatureWeight ==========")
    histA = train_model(
        model         = modelA,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )

    # 训练模型：EModel_CNN_Transformer
    print("\n========== 训练模型：EModel_CNN_Transformer ==========")
    histB = train_model(
        model         = modelB,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_CNN_Transformer',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )

    # 加载最佳权重
    best_modelA = EModel_FeatureWeight(feature_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth'))

    best_modelB = EModel_CNN_Transformer(feature_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_CNN_Transformer.pth'))

    # 在测试集上评估（标准化域）
    criterion_test = nn.SmoothL1Loss(beta = 1.0)
    (_, test_rmseA_std, test_mapeA_std, test_r2A_std, test_smapeA_std, test_maeA_std, predsA_std, labelsA_std) = evaluate(best_modelA, test_loader, criterion_test)
    (_, test_rmseB_std, test_mapeB_std, test_r2B_std, test_smapeB_std, test_maeB_std, predsB_std, labelsB_std) = evaluate(best_modelB, test_loader, criterion_test)

    print("\n========== [测试集（标准化域）评估] ==========")
    print(f"[EModel_FeatureWeight]  RMSE: {test_rmseA_std:.4f}, MAPE: {test_mapeA_std:.2f}, R²: {test_r2A_std:.4f}, SMAPE: {test_smapeA_std:.2f}, MAE: {test_maeA_std:.4f}")
    print(f"[EModel_CNN_Transformer] RMSE: {test_rmseB_std:.4f}, MAPE: {test_mapeB_std:.2f}, R²: {test_r2B_std:.4f}, SMAPE: {test_smapeB_std:.2f}, MAE: {test_maeB_std:.4f}")

    # 反标准化与（可选）反对数变换
    predsA_real_std = scaler_y.inverse_transform(predsA_std.reshape(-1, 1)).ravel()
    predsB_real_std = scaler_y.inverse_transform(predsB_std.reshape(-1, 1)).ravel()
    labelsA_real_std = scaler_y.inverse_transform(labelsA_std.reshape(-1, 1)).ravel()
    labelsB_real_std = scaler_y.inverse_transform(labelsB_std.reshape(-1, 1)).ravel()

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

    # 计算原域下RMSE
    test_rmseA_real = np.sqrt(mean_squared_error(labelsA_real, predsA_real))
    test_rmseB_real = np.sqrt(mean_squared_error(labelsB_real, predsB_real))

    print("\n========== [测试集（原始域）评估] ==========")
    print(f"[EModel_FeatureWeight] => RMSE(real): {test_rmseA_real:.2f}")
    print(f"[EModel_CNN_Transformer] => RMSE(real): {test_rmseB_real:.2f}")

    # 数据集统计信息
    print(f"\n[数据统计] 总样本数: {total_samples}")
    print(f"训练集: {train_size} ({train_size / total_samples:.1%})")
    print(f"验证集: {val_size} ({val_size / total_samples:.1%})")
    print(f"测试集: {test_size} ({test_size / total_samples:.1%})")

    # 时间分布直方图绘制函数
    def plot_dataset_distribution(timestamps, title):
        plt.figure(figsize = (10, 4))
        plt.hist(pd.to_datetime(timestamps), bins = 50, color = 'skyblue', edgecolor = 'black')
        plt.title(f'{title} - 时间分布')
        plt.xlabel('Timestamp')
        plt.ylabel('Count')
        plt.grid(axis = 'y')
        plt.tight_layout()
        plt.show()

    plot_dataset_distribution(train_timestamps, '训练集')
    plot_dataset_distribution(val_timestamps, '验证集')
    plot_dataset_distribution(test_timestamps, '测试集')

    # 可视化预测结果（以modelA为例）
    plot_test_predictions_over_time(test_timestamps[window_size:], labelsA_real, predsA_real)
    plot_predictions_comparison(
        y_actual_real      = labelsA_real,
        y_pred_model1_real = predsA_real,
        y_pred_model2_real = predsB_real,
        model1_name        = 'EModel_FeatureWeight',
        model2_name        = 'EModel_CNN_Transformer'
    )

    # 绘制训练曲线（各项指标）
    plot_training_curves_allmetrics(histA, model_name = 'EModel_FeatureWeight')
    plot_training_curves_allmetrics(histB, model_name = 'EModel_CNN_Transformer')

    print("[Info] 处理完毕！")

if __name__ == "__main__":
    main(use_log_transform = True, min_egrid_threshold = 1.0)
