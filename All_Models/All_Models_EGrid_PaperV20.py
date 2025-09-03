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
from lion_pytorch import Lion
import matplotlib.ticker as ticker 

try:
    import lightgbm as lgb
except Exception as e:
    lgb = None
    print("[Warning] lightgbm not found, LightGBM model will be skipped. pip install lightgbm")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception as e:
    SARIMAX = None
    print("[Warning] statsmodels not found, SARIMA model will be skipped. pip install statsmodels")

# Global style settings for plots
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 26,          # Global default font size
    'axes.labelsize': 26,     # Axis label font size
    'axes.titlesize': 28,     # Chart title font size
    'xtick.labelsize': 24,    # x-axis tick label size
    'ytick.labelsize': 24     # y-axis tick label size
})

# Fixed colors for all models (consistent across plots)
MODEL_COLORS = {
    'Model1':   '#1f77b4',  # 蓝色
    'Model2':   '#ff7f0e',  # 橙色
    'Model3':   '#2ca02c',  # 绿色
    'Model4':   '#d62728',  # 红色
    'Model5':   '#9467bd',  # 紫色
    'PatchTST': '#8c564b',  # 棕色
    'LightGBM': '#e377c2',  # 粉色
    'SARIMA':   '#7f7f7f',  # 灰色
    's-naive':  '#bcbd22'   # 黄绿色
}

# Global hyperparameters
learning_rate     = 1e-4   # Learning rate
num_epochs        = 150    # Number of training epochs
batch_size        = 128    # Batch size
weight_decay      = 1e-4   # Weight decay
patience          = 12     # Patience for early stopping
num_workers       = 0      # Number of worker threads
window_size       = 20     # Sequence window size

# Set random seed and device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# 统一季节周期（小时数据通常用24；如你的数据是日级或其他周期，请改这里）
seasonal_period = 24

# 1. Data Loading Module
def load_data():
    """
    [Data Loading Module]
    - Load renewable energy and load data from CSV files.
    - Merge and sort by timestamp, then reset the index.
    """
    renewable_df = pd.read_csv(r'C:\Users\Administrator\Desktop\renewable_data10.csv')
    load_df      = pd.read_csv(r'C:\Users\Administrator\Desktop\load_data10.csv')

    renewable_df['timestamp'] = pd.to_datetime(renewable_df['timestamp'])  # Convert timestamp to datetime
    load_df['timestamp']      = pd.to_datetime(load_df['timestamp'])       # Convert timestamp to datetime

    data_df = pd.merge(renewable_df, load_df, on='timestamp', how='inner')  # Merge data using inner join
    data_df.sort_values('timestamp', inplace=True)                         # Sort by timestamp
    data_df.reset_index(drop=True, inplace=True)                           # Reset index
    return data_df

# 2. Feature Engineering Module
def feature_engineering(data_df):
    """
    [Feature Engineering Module]
    - Apply EWMA smoothing to 'E_grid' (span = 10)
    - Construct time features: dayofweek, hour, month and their sin/cos transformations
    - Encode categorical features using LabelEncoder
    - Returns: processed data, list of feature columns, target column name
    """
    span = 8  # EWMA smoothing parameter
    data_df['E_grid'] = data_df['E_grid'].ewm(span = span, adjust = False).mean()  # Smooth E_grid

    # Construct time features
    data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek  # dayofweek: weekday index
    data_df['hour']      = data_df['timestamp'].dt.hour       # hour: hour of the day
    data_df['month']     = data_df['timestamp'].dt.month      # month: month of the year

    # Sin/Cos transformations
    data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7)
    data_df['hour_sin']      = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos']      = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['month_sin']     = np.sin(2 * np.pi * (data_df['month'] - 1) / 12)
    data_df['month_cos']     = np.cos(2 * np.pi * (data_df['month'] - 1) / 12)

    # Categorical feature encoding
    renewable_features = [
        'season', 'holiday', 'weather', 'temperature',
        'working_hours', 'E_wind', 'E_storage_discharge',
        'ESCFR', 'ESCFG','v_wind', 'wind_direction','E_PV'
    ]
    load_features = [
        'ship_grade', 'dock_position', 'destination', 'energyconsumption'
    ]
    for col in renewable_features + load_features:
        if col in data_df.columns:
            le = LabelEncoder()
            data_df[col] = le.fit_transform(data_df[col].astype(str))  # Encode feature

    time_feature_cols = [
        'dayofweek_sin', 'dayofweek_cos',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos'
    ]
    feature_columns = renewable_features + load_features + time_feature_cols  # All feature columns
    target_column   = 'E_grid'
    
    return data_df, feature_columns, target_column


# 3. Sequence Construction Module
def create_sequences(X_data, y_data, window_size):
    """
    [Sequence Construction Module]
    - Construct time series data based on the specified window size.
    Parameters:
      X_data: Feature data (numpy array)
      y_data: Target data (numpy array)
      window_size: Sequence window size
    Returns:
      X_arr: Sequenced feature data
      y_arr: Corresponding target data
    """
    X_list, y_list = [], []
    num_samples = X_data.shape[0]  # Total number of samples
    for i in range(num_samples - window_size):  # Construct sequences using sliding window
        seq_x = X_data[i : i + window_size, :]  # Feature sequence for current window
        seq_y = y_data[i + window_size]           # Target value corresponding to the current window
        X_list.append(seq_x)
        y_list.append(seq_y)
    X_arr = np.array(X_list, dtype = np.float32)
    y_arr = np.array(y_list, dtype = np.float32)
    return X_arr, y_arr

# 4. Model Definition Module
class PositionalEncoding(nn.Module):
    """
    [Positional Encoding Module]
    - Add positional encoding to the input sequence.
    Parameters:
      d_model: Model dimension
      max_len: Maximum sequence length (default: 5000)
    """
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]
        position = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(-(torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model))  # Shape: [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x, step_offset = 0):
        """
        Parameters:
          x: Input tensor with shape [batch_size, seq_len, d_model]
          step_offset: Sequence offset (default 0)
        Returns:
          x with added positional encoding
        """
        seq_len = x.size(1)  # Sequence length
        pos_enc = self.pe[step_offset : step_offset + seq_len, 0, :]  # Get corresponding positional encoding
        return x + pos_enc.unsqueeze(0)

class Attention(nn.Module):
    """
    [Attention Module]
    - Apply attention-based weighted aggregation to the input.
    Parameters:
      input_dim: Input dimension
      dropout: Dropout probability
    """
    def __init__(self, input_dim, dropout = 0.1):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Parameters:
          x: Input tensor with shape [batch_size, seq_len, input_dim]
        Returns:
          Aggregated feature with shape [batch_size, input_dim]
        """
        attn_weights = self.attention(x)       # Shape: [batch_size, seq_len, 1]
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim = 1)  # Softmax over time steps
        weighted = x * attn_weights            # Element-wise weighting
        return torch.sum(weighted, dim = 1)    # Aggregate to shape [batch_size, input_dim]

class CNN_FeatureGate(nn.Module):
    """
    CNN-based feature gating mechanism to replace the simple feature gate in EModel_FeatureWeight2
    """
    def __init__(self, feature_dim, seq_len):
        super(CNN_FeatureGate, self).__init__()
        # First convolutional layer: kernel_size=3, filters=4
        self.conv1 = nn.Conv1d(
            in_channels=feature_dim, 
            out_channels=4, 
            kernel_size=3, 
            padding=1  # Zero padding
        )
        
        # First max pooling layer
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional layer: kernel_size=3, filters=8
        self.conv2 = nn.Conv1d(
            in_channels=4, 
            out_channels=8, 
            kernel_size=3, 
            padding=1  # Zero padding
        )
        
        # Second max pooling layer
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions and pooling
        self.flattened_size = 8 * (seq_len // 4)
        
        # Fully connected hidden layer with 10 units
        self.fc1 = nn.Linear(self.flattened_size, 10)
        
        # Output layer to generate feature weights
        self.fc2 = nn.Linear(10, feature_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Parameters:
          x: Input tensor with shape [batch_size, seq_len, feature_dim]
        Returns:
          Feature weights with shape [batch_size, feature_dim]
        """
        # Transpose for 1D convolution: [batch_size, feature_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        # First conv + activation + pool
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Second conv + activation + pool
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        
        return x

class EModel_FeatureWeight1(nn.Module):
    """
    [Model 1: LSTM-based Model with Feature Weighting]
    Parameters:
      - feature_dim: Input feature dimension
      - lstm_hidden_size: LSTM hidden size
      - lstm_num_layers: Number of LSTM layers
      - lstm_dropout: LSTM dropout probability
      - use_local_attn: Whether to use local attention 
      - local_attn_window_size: Window size for local attention
    """
    def __init__(self, 
                 feature_dim, 
                 lstm_hidden_size = 256, 
                 lstm_num_layers = 2, 
                 lstm_dropout = 0.2,
                 use_local_attn = False,
                 local_attn_window_size = 5
                ):
        super(EModel_FeatureWeight1, self).__init__()
        self.feature_dim = feature_dim
        self.use_local_attn = use_local_attn  # 保存是否使用局部注意力的标识
        
        # Feature gating mechanism: fully connected layer + Sigmoid to compute feature weights
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Temporal attention: choose between local or global (MLP) attention
        if use_local_attn:
            from local_attention.local_attention import LocalAttention
            self.temporal_attn = LocalAttention(
                dim = 2 * lstm_hidden_size,
                window_size = local_attn_window_size,   # 使用正确的参数名
                causal = False
            )
        else:
            self.temporal_attn = Attention(input_dim = 2 * lstm_hidden_size)
        
        # Feature attention layer: aggregate LSTM output over feature dimensions
        self.feature_attn = nn.Sequential(
            nn.Linear(window_size, 1),
            nn.Sigmoid()
        )
        # Feature projection layer
        self.feature_proj = nn.Linear(2 * lstm_hidden_size, 2 * lstm_hidden_size)
        
        # Learnable feature importance weights, initialized to 1
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad = True)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size    = feature_dim,
            hidden_size   = lstm_hidden_size,
            num_layers    = lstm_num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = lstm_dropout if lstm_num_layers > 1 else 0
        )
        self._init_lstm_weights()
        
        # Global attention module
        self.attention = Attention(input_dim = 2 * lstm_hidden_size)
        
        # Fully connected layer for final prediction
        self.fc = nn.Sequential(
            nn.Linear(4 * lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 输出两个值: mu 和 logvar
        )

    def _init_lstm_weights(self):
        """
        [LSTM Weight Initialization]
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)  # Set forget gate bias to 1

    def forward(self, x):
        """
        Parameters:
          x: Input tensor with shape [batch_size, seq_len, feature_dim]
        Returns:
          Predicted output with shape [batch_size]
        """
        # Apply dynamic feature weighting
        gate = self.feature_gate(x.mean(dim = 1))
        x = x * gate.unsqueeze(1)

        # Process with LSTM to obtain bidirectional output
        lstm_out, _ = self.lstm(x)
        
        # Temporal attention:
        if self.use_local_attn:
            # 调用LocalAttention，将query, key, value均设置为lstm_out
            temporal = self.temporal_attn(lstm_out, lstm_out, lstm_out)
            # 聚合沿时间步（例如使用求和或者平均），使得维度变为二维
            temporal = temporal.sum(dim=1)
            # 如果希望归一化，可以改为：temporal = temporal.mean(dim=1)
        else:
            temporal = self.temporal_attn(lstm_out)
        
        # Feature attention over the feature dimensions
        feature_raw = self.feature_attn(lstm_out.transpose(1, 2))
        feature_raw = feature_raw.squeeze(-1)
        feature = self.feature_proj(feature_raw)

        # Concatenate the two branches
        combined = torch.cat([temporal, feature], dim = 1)
        output = self.fc(combined)
        mu, logvar = torch.chunk(output, 2, dim = 1)
        
        # Reparameterization trick with noise
        noise = 0.1 * torch.randn_like(mu, device = x.device) * torch.exp(0.5 * logvar)
        output = mu + noise
        
        return output.squeeze(-1)
    
class EModel_FeatureWeight2(nn.Module):
    """
    Modified version of EModel_FeatureWeight2 with CNN-based feature gating
    
    Parameters:
      - feature_dim: Input feature dimension
      - window_size: Sequence window size (needed for CNN)
      - lstm_hidden_size: LSTM hidden size
      - lstm_num_layers: Number of LSTM layers
      - lstm_dropout: LSTM dropout probability
      - use_local_attn: Whether to use local attention
      - local_attn_window_size: Window size for local attention
    """
    def __init__(self,
                 feature_dim,
                 window_size=window_size,  # Using global window_size
                 lstm_hidden_size=256,
                 lstm_num_layers=2,
                 lstm_dropout=0.2,
                 use_local_attn=True,
                 local_attn_window_size=5
                ):
        super(EModel_FeatureWeight2, self).__init__()
        self.feature_dim = feature_dim
        self.use_local_attn = use_local_attn
        
        # Replace feature gating with CNN-based mechanism
        self.feature_gate = CNN_FeatureGate(feature_dim, window_size)
        
        # Temporal attention: choose between local or global attention
        if use_local_attn:
            from local_attention.local_attention import LocalAttention
            self.temporal_attn = LocalAttention(
                dim=2 * lstm_hidden_size,
                window_size=local_attn_window_size,
                causal=False
            )
        else:
            self.temporal_attn = Attention(input_dim=2 * lstm_hidden_size)
        
        # Feature attention layer
        self.feature_attn = nn.Sequential(
            nn.Linear(window_size, 1),
            nn.Sigmoid()
        )
        
        # Feature projection layer
        self.feature_proj = nn.Linear(2 * lstm_hidden_size, 2 * lstm_hidden_size)
        
        # Learnable feature importance weights
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        self._init_lstm_weights()
        
        # Global attention module
        self.attention = Attention(input_dim=2 * lstm_hidden_size)
        
        # Fully connected layer for final prediction
        self.fc = nn.Sequential(
            nn.Linear(4 * lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # Output mu and logvar
        )
    
    def _init_lstm_weights(self):
        """LSTM Weight Initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)  # Set forget gate bias to 1
    
    def forward(self, x):
        """
        Parameters:
          x: Input tensor with shape [batch_size, seq_len, feature_dim]
        Returns:
          Predicted output with shape [batch_size]
        """
        # Apply CNN-based feature gating
        gate = self.feature_gate(x)
        x = x * gate.unsqueeze(1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x)
        
        # Temporal attention
        if self.use_local_attn:
            temporal = self.temporal_attn(lstm_out, lstm_out, lstm_out)
            temporal = temporal.sum(dim=1)  # Aggregate along time steps
        else:
            temporal = self.temporal_attn(lstm_out)
        
        # Feature attention
        feature_raw = self.feature_attn(lstm_out.transpose(1, 2))
        feature_raw = feature_raw.squeeze(-1)
        feature = self.feature_proj(feature_raw)
        
        # Concatenate branches
        combined = torch.cat([temporal, feature], dim=1)
        output = self.fc(combined)
        mu, logvar = torch.chunk(output, 2, dim=1)
        
        # Reparameterization trick with noise
        noise = 0.1 * torch.randn_like(mu, device=x.device) * torch.exp(0.5 * logvar)
        output = mu + noise
        return output.squeeze(-1)
    
class EModel_FeatureWeight3(nn.Module):
    """
    Parameters:
      - feature_dim: Input feature dimension
      - gru_hidden_size: GRU hidden size (固定为10)
      - gru_num_layers: Number of GRU layers
      - gru_dropout: GRU dropout probability
      - use_local_attn: Whether to use local attention 
      - local_attn_window_size: Window size for local attention
    """
    def __init__(self, 
                 feature_dim, 
                 gru_hidden_size = 128, 
                 gru_num_layers = 2, 
                 gru_dropout = 0.2,
                 use_local_attn = True,
                 local_attn_window_size = 5
                ):
        super(EModel_FeatureWeight3, self).__init__()
        self.feature_dim = feature_dim
        self.use_local_attn = use_local_attn  # 保存是否使用局部注意力的标识
        
        # Feature gating mechanism: fully connected layer + Sigmoid to compute feature weights
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Temporal attention: choose between local or global (MLP) attention
        if use_local_attn:
            from local_attention.local_attention import LocalAttention
            self.temporal_attn = LocalAttention(
                dim = 2 * gru_hidden_size,  # 双向GRU，隐藏层大小*2
                window_size = local_attn_window_size,   # 使用正确的参数名
                causal = False
            )
        else:
            self.temporal_attn = Attention(input_dim = 2 * gru_hidden_size)  # 双向GRU，隐藏层大小*2
        
        # Feature attention layer: aggregate GRU output over feature dimensions
        self.feature_attn = nn.Sequential(
            nn.Linear(window_size, 1),
            nn.Sigmoid()
        )
        # Feature projection layer
        self.feature_proj = nn.Linear(2 * gru_hidden_size, 2 * gru_hidden_size)  # 双向GRU，隐藏层大小*2
        
        # Learnable feature importance weights, initialized to 1
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad = True)
        
        # Bidirectional GRU 
        self.gru = nn.GRU(
            input_size    = feature_dim,
            hidden_size   = gru_hidden_size,  # 隐藏层大小为10
            num_layers    = gru_num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = gru_dropout if gru_num_layers > 1 else 0
        )
        self._init_gru_weights()  # 初始化GRU权重
        
        # Global attention module
        self.attention = Attention(input_dim = 2 * gru_hidden_size)  # 双向GRU，隐藏层大小*2
        
        # Fully connected layer for final prediction
        self.fc = nn.Sequential(
            nn.Linear(4 * gru_hidden_size, 128),  # 因为将两个大小为2*gru_hidden_size的特征连接在一起
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 输出两个值: mu 和 logvar
        )

    def _init_gru_weights(self):
        """
        [GRU Weight Initialization]
        """
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        """
        Parameters:
          x: Input tensor with shape [batch_size, seq_len, feature_dim]
        Returns:
          Predicted output with shape [batch_size]
        """
        # Apply dynamic feature weighting
        gate = self.feature_gate(x.mean(dim = 1))
        x = x * gate.unsqueeze(1)

        # Process with GRU to obtain bidirectional output (替换LSTM)
        gru_out, _ = self.gru(x)
        
        # Temporal attention:
        if self.use_local_attn:
            # 调用LocalAttention，将query, key, value均设置为gru_out
            temporal = self.temporal_attn(gru_out, gru_out, gru_out)
            # 聚合沿时间步（例如使用求和或者平均），使得维度变为二维
            temporal = temporal.sum(dim=1)
            # 如果希望归一化，可以改为：temporal = temporal.mean(dim=1)
        else:
            temporal = self.temporal_attn(gru_out)
        
        # Feature attention over the feature dimensions
        feature_raw = self.feature_attn(gru_out.transpose(1, 2))
        feature_raw = feature_raw.squeeze(-1)
        feature = self.feature_proj(feature_raw)

        # Concatenate the two branches
        combined = torch.cat([temporal, feature], dim = 1)
        output = self.fc(combined)
        mu, logvar = torch.chunk(output, 2, dim = 1)
        
        # Reparameterization trick with noise
        noise = 0.1 * torch.randn_like(mu, device = x.device) * torch.exp(0.5 * logvar)
        output = mu + noise
        
        return output.squeeze(-1)
    
class EModel_FeatureWeight4(nn.Module):
    """
    [Model 4: LSTM-based Model with Feature Weighting]
    Parameters:
      - feature_dim: Input feature dimension
      - lstm_hidden_size: LSTM hidden size
      - lstm_num_layers: Number of LSTM layers
      - lstm_dropout: LSTM dropout probability
      - use_local_attn: Whether to use local attention 
      - local_attn_window_size: Window size for local attention
      - window_size: Window size for feature attention
    """
    def __init__(self, 
                 feature_dim, 
                 lstm_hidden_size = 256, 
                 lstm_num_layers = 2, 
                 lstm_dropout = 0.1,
                 use_local_attn = True,
                 local_attn_window_size = 10,
                 window_size = 20,
                 feature_importance = None
                ):
        super(EModel_FeatureWeight4, self).__init__()
        self.feature_dim = feature_dim
        self.use_local_attn = use_local_attn  # 保存是否使用局部注意力的标识
        self.window_size = window_size  # 保存窗口大小
        
        # Feature gating mechanism: fully connected layer + Sigmoid to compute feature weights
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Temporal attention: choose between local or global (MLP) attention
        if use_local_attn:
            from local_attention.local_attention import LocalAttention
            self.temporal_attn = LocalAttention(
                dim = 2 * lstm_hidden_size,
                window_size = local_attn_window_size,   # 使用正确的参数名
                causal = False
            )
        else:
            self.temporal_attn = Attention(input_dim = 2 * lstm_hidden_size)
        
        # Feature attention layer: aggregate LSTM output over feature dimensions
        self.feature_attn = nn.Sequential(
            nn.Linear(self.window_size, self.window_size * 2),
            nn.GELU(),
            nn.LayerNorm(self.window_size * 2),
            nn.Linear(self.window_size * 2, 1),
            nn.Sigmoid()
        )
        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, 2 * lstm_hidden_size),
            nn.LayerNorm(2 * lstm_hidden_size),
            nn.GELU()
        )
        
        # 初始化特征重要性权重，如果提供了特征重要性，则使用它
        if feature_importance is not None and len(feature_importance) == feature_dim:
            self.feature_importance = nn.Parameter(torch.tensor(feature_importance, dtype=torch.float32), requires_grad=True)
        else:
            self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)
        
        # LSTM 输出层归一化与 Dropout
        self.lstm_norm = nn.LayerNorm(2 * lstm_hidden_size)
        self.lstm_dropout = nn.Dropout(0.1)
      
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size    = feature_dim,
            hidden_size   = lstm_hidden_size,
            num_layers    = lstm_num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = lstm_dropout if lstm_num_layers > 1 else 0
        )
        self._init_lstm_weights()
        
        # Global attention module
        self.attention = Attention(input_dim = 2 * lstm_hidden_size)
        
        # Fully connected layer for final prediction
        self.fc = nn.Sequential(
            nn.Linear(4 * lstm_hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def _init_lstm_weights(self):
        """
        [LSTM Weight Initialization]
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)  # Set forget gate bias to 1

    def forward(self, x):
        """
        Parameters:
          x: Input tensor with shape [batch_size, seq_len, feature_dim]
        Returns:
          Predicted output with shape [batch_size]
        """
        # Apply dynamic feature weighting
        gate = self.feature_gate(x.mean(dim = 1))
        x = x * gate.unsqueeze(1)

        # Process with LSTM to obtain bidirectional output
        lstm_out, _ = self.lstm(x)
        
        # Temporal attention:
        if self.use_local_attn:
            # 调用LocalAttention，将query, key, value均设置为lstm_out
            temporal = self.temporal_attn(lstm_out, lstm_out, lstm_out)
            # 聚合沿时间步（例如使用求和或者平均），使得维度变为二维
            temporal = temporal.sum(dim=1)
            # 如果希望归一化，可以改为：temporal = temporal.mean(dim=1)
        else:
            temporal = self.temporal_attn(lstm_out)
        
        # Feature attention over the feature dimensions
        feature_raw = self.feature_attn(lstm_out.transpose(1, 2))
        feature_raw = feature_raw.squeeze(-1)
        feature = self.feature_proj(feature_raw)

        # Concatenate the two branches
        combined = torch.cat([temporal, feature], dim = 1)
        output = self.fc(combined)
        mu, logvar = torch.chunk(output, 2, dim = 1)
        
        # Reparameterization trick with noise
        noise = 0.1 * torch.randn_like(mu, device = x.device) * torch.exp(0.5 * logvar)
        output = mu + noise
        
        return output.squeeze(-1)

class EModel_FeatureWeight5(nn.Module):

    def __init__(self,
                 feature_dim,
                 window_size=window_size,          # 序列窗口长度
                 lstm_hidden_size=256,
                 lstm_num_layers=2,
                 lstm_dropout=0.2, 
                ):
        # ------- 修正父类调用名称 ------- #
        super(EModel_FeatureWeight5, self).__init__()

        self.feature_dim = feature_dim

        # 1) CNN-FeatureGate
        self.feature_gate = CNN_FeatureGate(feature_dim, window_size)

        # 2) MLP-Attention
        self.temporal_attn = Attention(input_dim=2 * lstm_hidden_size)

        # 3) 双向 LSTM 及其初始化（
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )

        self._init_lstm_weights()

        # 5) 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, 128),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)   # 输出 μ 与 log σ²
        )
    
    def _init_lstm_weights(self):
        """LSTM权重初始化"""
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
        x: [batch, seq_len, feature_dim]
        """
        # ---------- 动态特征加权 ---------- #
        gate = self.feature_gate(x)                # [batch, feature_dim]
        x = x * gate.unsqueeze(1)                  # [batch, seq_len, feature_dim]

        # ---------- LSTM ---------- #
        lstm_out, _ = self.lstm(x)                 # [batch, seq_len, 2*hidden]

        # ---------- Temporal Attention ---------- #
        temporal = self.temporal_attn(lstm_out)    # [batch, 2*hidden]

        mu, logvar = torch.chunk(self.fc(temporal), 2, dim=1)

        # Re-parameterization trick
        noise   = 0.1 * torch.randn_like(mu) * torch.exp(0.5 * logvar)
        output  = mu + noise
        return output.squeeze(-1)

class PatchTST(nn.Module):
    """
    简化版 PatchTST:
    - 将时间维度切成 patch（长度 patch_len，步长 patch_stride）
    - 每个 patch 展开后线性投影到 d_model
    - 加位置编码，过 TransformerEncoder
    - 池化得到序列表征，回归到 1 步预测
    说明：我们沿用统一的滑窗输入 [B, window, feature_dim]。
    """
    def __init__(self, feature_dim, window_size, patch_len=5, patch_stride=5,
                 d_model=256, n_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_dim  = feature_dim
        self.window_size  = window_size
        self.patch_len    = patch_len
        self.patch_stride = patch_stride

        assert patch_len <= window_size
        self.num_patches = 1 + (window_size - patch_len) // patch_stride
        self.in_dim = patch_len * feature_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=4*d_model,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(self.in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=5000)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, x):
        # x: [B, window, feature_dim]
        B, W, F = x.shape
        patches = []
        for s in range(0, W - self.patch_len + 1, self.patch_stride):
            px = x[:, s:s+self.patch_len, :]                     # [B, patch_len, F]
            px = px.reshape(B, -1)                               # [B, patch_len*F]
            patches.append(px)
        patches = torch.stack(patches, dim=1)                    # [B, num_patches, patch_len*F]
        z = self.proj(patches)                                   # [B, num_patches, d_model]
        z = self.pos_enc(z)                                      # 位置编码（沿 patch 维）
        z = self.encoder(z)                                      # [B, num_patches, d_model]
        z = z.mean(dim=1)                                        # 池化
        y = self.head(z).squeeze(-1)                             # [B]
        return y

def apply_feature_gating_to_sequences(X_seq, feature_importance):
    """
    X_seq: [n_samples, window_size, feature_dim]
    feature_importance: [feature_dim], 来自 Pearson+MIC 的综合权重
    """
    if feature_importance is None:
        return X_seq
    w = feature_importance.astype(np.float32).reshape(1, 1, -1)
    return X_seq * w

def flatten_sequences_for_tabular(X_seq):
    """
    将 [n, window, feat] 展成 [n, window*feat]，作为 LightGBM 的滞后特征输入。
    历法特征已在特征工程阶段加入并被统一标准化，这里统一展开即可。
    """
    n, w, f = X_seq.shape
    return X_seq.reshape(n, w * f)

def train_lightgbm(X_train_tab, y_train_seq, X_val_tab, y_val_seq, patience_rounds=50, seed=42):
    if lgb is None:
        return None
    train_set = lgb.Dataset(X_train_tab, label=y_train_seq)
    val_set   = lgb.Dataset(X_val_tab,   label=y_val_seq, reference=train_set)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,      # 训练协议统一：使用早停与固定随机种子；学习率为树模型合理默认
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "verbosity": -1
    }
    # LightGBM 4.x: use callbacks for early stopping and logging
    callbacks = []
    try:
        callbacks.append(lgb.early_stopping(stopping_rounds=patience_rounds))
        callbacks.append(lgb.log_evaluation(period=100))
    except Exception:
        # Fallback for older versions; if callbacks are unavailable, train without them
        pass
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=20000,
        valid_sets=[train_set, val_set],
        valid_names=["train","valid"],
        callbacks=callbacks
    )
    return booster

def fit_predict_sarima(y_train_val_std, n_test, m=24, search_small=True, seed=42):
    """
    在标准化后的 y 序列上拟合 SARIMA（与深度/树模型的评估域一致），预测后再按主流程反标准化。
    y_train_val_std: 训练+验证集（标准化域）
    n_test: 需要预测的步数（与 y_test_seq 对齐）
    m: 季节周期
    """
    if SARIMAX is None:
        return None
    np.random.seed(seed)
    # 极小网格，避免耗时
    pdq = [(0,1,1), (1,1,0), (1,1,1)]
    PDQ = [(0,1,1), (1,1,0)]
    best_aic, best_cfg, best_res = 1e18, None, None
    for (p,d,q) in pdq if search_small else [(1,1,1)]:
        for (P,D,Q) in PDQ if search_small else [(0,1,1)]:
            try:
                model = SARIMAX(y_train_val_std, order=(p,d,q), seasonal_order=(P,D,Q,m), enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False, maxiter=200)
                if res.aic < best_aic:
                    best_aic, best_cfg, best_res = res.aic, ((p,d,q),(P,D,Q,m)), res
            except Exception:
                continue
    if best_res is None:
        return None
    forecast = best_res.forecast(steps=n_test)  # 标准化域
    return np.asarray(forecast, dtype=np.float32)

def predict_snaive_std(y_all_std, start_index, n_test, m=24):
    """
    s-naive: y_hat[t] = y[t-m]，在标准化域直接做。
    y_all_std: 整体标准化后的 y（train+val+test 拼接）
    start_index: 测试集第一条标签对应的全局索引（注意滑窗偏移）
    n_test: 需要预测的样本数（与 y_test_seq 对齐）
    m: 季节周期
    """
    preds = []
    for i in range(n_test):
        t = start_index + i
        ref = t - m
        if ref < 0 or ref >= len(y_all_std):
            preds.append(y_all_std[t-1] if t-1 >= 0 else y_all_std[0])
        else:
            preds.append(y_all_std[ref])
    return np.asarray(preds, dtype=np.float32)

# 5. Evaluation Module
def evaluate(model, dataloader, criterion, device = device):
    """
    [Evaluation Module]
    - Compute loss and multiple metrics (RMSE, MAPE, R², mse, MAE) on the given dataset.
    Parameters:
      model: Model to evaluate
      dataloader: DataLoader for the dataset
      criterion: Loss function
    Returns:
      val_loss, rmse, mape, r2, mse, mae, preds, labels
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

    # Safety: guard against NaN/Inf in predictions or labels
    if np.isnan(preds_arr).any() or np.isinf(preds_arr).any():
        print("[Warning] NaN/Inf detected in predictions. They will be replaced by 0 for metrics computation.")
    if np.isnan(labels_arr).any() or np.isinf(labels_arr).any():
        print("[Warning] NaN/Inf detected in labels. They will be replaced by 0 for metrics computation.")
    preds_arr  = np.nan_to_num(preds_arr, nan = 0.0, posinf = 0.0, neginf = 0.0)
    labels_arr = np.nan_to_num(labels_arr, nan = 0.0, posinf = 0.0, neginf = 0.0)

    # Compute RMSE
    rmse_std = np.sqrt(mean_squared_error(labels_arr, preds_arr))

    # Compute MAPE (excluding labels equal to 0)
    nonzero_mask_mape = (labels_arr != 0)
    if np.sum(nonzero_mask_mape) > 0:
        mape_std = np.mean(np.abs((labels_arr[nonzero_mask_mape] - preds_arr[nonzero_mask_mape]) / labels_arr[nonzero_mask_mape])) * 100.0
    else:
        mape_std = 0.0

    # Compute R²
    ss_res = np.sum((labels_arr - preds_arr) ** 2)
    ss_tot = np.sum((labels_arr - np.mean(labels_arr)) ** 2)
    r2_std = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Compute mse
    mse_val = np.mean((labels_arr - preds_arr) ** 2)

    # Compute MAE
    mae_val = np.mean(np.abs(labels_arr - preds_arr))

    return val_loss, rmse_std, mape_std, r2_std, mse_val, mae_val, preds_arr, labels_arr


# 6. Training Module
def train_model(model, train_loader, val_loader, model_name = 'Model', learning_rate = learning_rate, weight_decay = weight_decay, num_epochs = num_epochs):
    """
    [Training Module]
    - Train the model on the training and validation sets while recording various metrics.
    - Implements Early Stopping.
    Parameters:
      model: Model to train
      train_loader: DataLoader for the training set
      val_loader: DataLoader for the validation set
      model_name: Model name
      learning_rate: Learning rate
      weight_decay: Weight decay
      num_epochs: Number of training epochs
    Returns:
      A dictionary of metric histories.
    """
    criterion = nn.MSELoss()
    optimizer = Lion(
        model.parameters(),
        lr           = learning_rate,
        weight_decay = weight_decay
    )

    total_steps  = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = LambdaLR(optimizer, lambda step: 
        min(step / warmup_steps, 1.0) if step < warmup_steps else 
        max(0.0, 1 - (step - warmup_steps) / (total_steps - warmup_steps))
    )
    best_val_loss = float('inf')
    counter       = 0
    global_step   = 0

    # Initialize history dictionaries
    train_loss_history   = []
    train_rmse_history   = []
    train_mape_history   = []
    train_r2_history     = []
    train_mse_history    = []
    train_mae_history    = []

    val_loss_history     = []
    val_rmse_history     = []
    val_mape_history     = []
    val_r2_history       = []
    val_mse_history      = []
    val_mae_history      = []

    for epoch in range(num_epochs):
        # Training phase
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

        # Evaluate on training and validation sets
        train_loss_eval, train_rmse_eval, train_mape_eval, train_r2_eval, train_mse_eval, train_mae_eval, _, _ = evaluate(model, train_loader, criterion)
        val_loss_eval, val_rmse_eval, val_mape_eval, val_r2_eval, val_mse_eval, val_mae_eval, _, _ = evaluate(model, val_loader, criterion)

        # Save metric histories
        train_loss_history.append(train_loss_eval)
        train_rmse_history.append(train_rmse_eval)
        train_mape_history.append(train_mape_eval)
        train_r2_history.append(train_r2_eval)
        train_mse_history.append(train_mse_eval)
        train_mae_history.append(train_mae_eval)

        val_loss_history.append(val_loss_eval)
        val_rmse_history.append(val_rmse_eval)
        val_mape_history.append(val_mape_eval)
        val_r2_history.append(val_r2_eval)
        val_mse_history.append(val_mse_eval)
        val_mae_history.append(val_mae_eval)

        # Print log information
        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"TrainLoss: {train_loss_epoch:.4f}, "
              f"ValLoss: {val_loss_eval:.4f}, "
              f"ValRMSE: {val_rmse_eval:.4f}, "
              f"ValMAPE: {val_mape_eval:.2f}%, "
              f"ValR^2: {val_r2_eval:.4f}, "
              f"Valmse: {val_mse_eval:.2f}%, "
              f"ValMAE: {val_mae_eval:.4f}")

        # Early Stopping check
        if val_loss_eval < best_val_loss:
            best_val_loss = val_loss_eval
            counter       = 0
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            print(f"[{model_name}] Model saved (best val_loss = {best_val_loss:.4f}).")
        else:
            counter += 1
            if counter >= patience:
                print(f"[{model_name}] No improvement on validation set. Early stopping.")
                break

    return {
        "train_loss":   train_loss_history,
        "train_rmse":   train_rmse_history,
        "train_mape":   train_mape_history,
        "train_r2":     train_r2_history,
        "train_mse":  train_mse_history,
        "train_mae":    train_mae_history,
        "val_loss":     val_loss_history,
        "val_rmse":     val_rmse_history,
        "val_mape":     val_mape_history,
        "val_r2":       val_r2_history,
        "val_mse":    val_mse_history,
        "val_mae":      val_mae_history
    }


# 7. Visualization and Helper Functions
def plot_correlation_heatmap(df, feature_cols):
    """
    [Visualization Module - Correlation Heatmap]
    - Plot the correlation heatmap for the specified feature columns.
    Parameters:
      df: Dataset
      feature_cols: List of feature columns
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
        cmap       = cmap_custom,
        annot      = True,
        fmt        = ".2f",
        square     = True,
        linewidths = 1,
        linecolor  = 'white',
        cbar       = True,
        vmin       = -1,
        vmax       = 1
    )
    plt.xticks(rotation = 45, ha = 'right')
    plt.tight_layout()
    plt.show()

def analyze_target_distribution(data_df, target_col):
    """
    [Visualization Module - Target Distribution]
    - Print basic statistics and plot a histogram for the target variable.
    Parameters:
      data_df: Dataset
      target_col: Target column name
    """
    if target_col not in data_df.columns:
        print(f"[Warning] '{target_col}' is not in the dataset, skipping analysis.")
        return

    print(f"\n[Target Analysis] Basic statistics for '{target_col}':")
    print(data_df[target_col].describe())

    plt.figure(figsize = (6, 4)) 
    plt.hist(data_df[target_col], bins = 30, color = 'skyblue', edgecolor = 'black')
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_Egrid_over_time(data_df):
    """
    [Visualization Module - Time Series Plot]
    - Plot the time series of 'E_grid' from the dataset.
    Parameters:
      data_df: Dataset
    """
    plt.figure(figsize = (10, 5))
    plt.plot(data_df['timestamp'], data_df['E_grid'], color = 'blue', marker = 'o', markersize = 3, linewidth = 1)
    plt.xlabel('Timestamp')
    plt.ylabel('E_grid (kW·h)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_comparison(y_actual_real, predictions_dict, timestamps):
    plt.figure(figsize=(14, 6))
    if timestamps is not None:
        x_axis = pd.to_datetime(timestamps)  
    else:
        x_axis = np.arange(len(y_actual_real))
    plt.plot(x_axis, y_actual_real, 'black', label='Actual', linewidth=1.2, alpha=1)
    
    colors = ['green', 'blue']
    
    # 为每个模型的预测绘制曲线
    for i, (model_name, pred_values) in enumerate(predictions_dict.items()):
        color = MODEL_COLORS.get(model_name, colors[i % len(colors)])
        plt.plot(x_axis, pred_values, color=color, label=model_name, linewidth=0.9, linestyle='-', alpha=0.9)
    
    plt.xlabel('Timestamp of TrainValue')
    plt.ylabel('Grid Energy Compensation Value (kW·h)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)   # 让时间轴标签斜着显示
    plt.show()


def plot_training_curves_allmetrics(hist_dict, model_name = 'Model'):
    """
    [Visualization Module - Training Curves]
    - Plot training and validation curves for Loss, RMSE, MAPE, R², mse, and MAE.
    Parameters:
      hist_dict: Dictionary containing metric histories
      model_name: Model name (default: 'Model')
    """
    epochs = range(1, len(hist_dict["train_loss"]) + 1)
    plt.figure(figsize = (15, 12))

    # Loss curve
    plt.subplot(3, 2, 1)
    plt.plot(epochs, hist_dict["train_loss"], 'r-o', label = 'Train Loss', markersize = 4)
    plt.plot(epochs, hist_dict["val_loss"], 'b-o', label = 'Val Loss', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    # RMSE curve
    plt.subplot(3, 2, 2)
    plt.plot(epochs, hist_dict["train_rmse"], 'r-o', label = 'Train RMSE', markersize = 4)
    plt.plot(epochs, hist_dict["val_rmse"], 'b-o', label = 'Val RMSE', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE')
    plt.legend()
    plt.grid(True)

    # MAPE curve
    plt.subplot(3, 2, 3)
    plt.plot(epochs, hist_dict["train_mape"], 'r-o', label = 'Train MAPE', markersize = 4)
    plt.plot(epochs, hist_dict["val_mape"], 'b-o', label = 'Val MAPE', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE')
    plt.legend()
    plt.grid(True)

    # R² curve
    plt.subplot(3, 2, 4)
    plt.plot(epochs, hist_dict["train_r2"], 'r-o', label = 'Train R^2', markersize = 4)
    plt.plot(epochs, hist_dict["val_r2"], 'b-o', label = 'Val R^2', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R²')
    plt.legend()
    plt.grid(True)

    # mse curve
    plt.subplot(3, 2, 5)
    plt.plot(epochs, hist_dict["train_mse"], 'r-o', label = 'Train mse', markersize = 4)
    plt.plot(epochs, hist_dict["val_mse"], 'b-o', label = 'Val mse', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('mse (%)')
    plt.title('mse')
    plt.legend()
    plt.grid(True)

    # MAE curve
    plt.subplot(3, 2, 6)
    plt.plot(epochs, hist_dict["train_mae"], 'r-o', label = 'Train MAE', markersize = 4)
    plt.plot(epochs, hist_dict["val_mae"], 'b-o', label = 'Val MAE', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"Training Curves - {model_name}", fontsize = 16)
    plt.tight_layout()
    plt.show()

def plot_dataset_distribution(timestamps, title):
    """
    [Visualization Module - Dataset Time Distribution]
    - Plot the time distribution of the dataset based on timestamps.
    Parameters:
      timestamps: Timestamps of the dataset
    """
    plt.figure(figsize = (10, 4))
    plt.hist(pd.to_datetime(timestamps), bins = 50, color = 'skyblue', edgecolor = 'black')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.grid(axis = 'y')
    plt.tight_layout()
    plt.show()

# 添加新函数计算特征重要性
def calculate_feature_importance(data_df, feature_cols, target_col):
    """
    计算特征与目标变量间的相关性，获取特征重要性权重
    
    参数:
        data_df: 数据DataFrame
        feature_cols: 特征列名称列表
        target_col: 目标变量列名称
        
    返回:
        特征重要性权重（绝对值相关系数）numpy数组
    """
    # 复制数据以避免修改原始数据
    df_encoded = data_df.copy()
    
    # 初始化特征重要性数组
    feature_importance = np.ones(len(feature_cols))
    
    # 计算每个特征与目标变量的Pearson相关系数
    for i, col in enumerate(feature_cols):
        # 确保特征列和目标列是数值型
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        # 计算Pearson相关系数
        corr = df_encoded[col].corr(df_encoded[target_col])
        
        # 使用相关系数的绝对值作为重要性
        feature_importance[i] = abs(corr)
    
    # 确保没有零值（设置最小值为0.1）
    feature_importance = np.maximum(feature_importance, 0.1)
    
    # 归一化，使最大值为1
    if np.max(feature_importance) > 0:
        feature_importance = feature_importance / np.max(feature_importance)
    
    # 打印特征重要性信息
    importance_info = [(feature_cols[i], importance) for i, importance in enumerate(feature_importance)]
    importance_info.sort(key=lambda x: x[1], reverse=True)
    
    print("\n基于Pearson相关系数的特征对E_grid影响度（线性相关性）：")
    for feature, importance in importance_info:
        print(f"{feature}: {importance:.4f}")
    
    return feature_importance

# 基于 Maximum Information Coefficient (MIC) 的特征重要性计算
# ------------------------------------------------------------------
def calculate_feature_importance_mic(data_df, feature_cols, target_col):
    """
    使用 MIC 计算特征重要性，返回 0~1 归一化后的权重数组。
    需要安装 minepy:  pip install minepy
    """
    from minepy import MINE
    df_encoded = data_df.copy()
    mic_importance = np.ones(len(feature_cols), dtype=np.float32)

    mine = MINE(alpha=0.6, c=15)                 # 官方推荐参数
    for i, col in enumerate(feature_cols):
        # 类别型特征先标签编码
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        x = df_encoded[col].values
        y = df_encoded[target_col].values

        mine.compute_score(x, y)
        mic_val = mine.mic()
        mic_importance[i] = max(mic_val, 0.1)     # 最小 0.1，避免 0

    # 归一化到 0~1
    mic_importance /= mic_importance.max()

    # 输出排序结果
    ranking = sorted(zip(feature_cols, mic_importance),
                     key=lambda z: z[1], reverse=True)
    print("\n基于MIC的特征对E_grid影响度（线性+非线性相关性）：")
    for feat, score in ranking:
        print(f"{feat}: {score:.4f}")

    return mic_importance


def plot_predictions_date_range(y_actual_real, predictions_dict, timestamps, start_date, end_date):
    """
    Plot predictions for a specific date range.
    
    Parameters
    ----------
    y_actual_real : np.ndarray
        Ground-truth values in original scale.
    predictions_dict : Dict[str, np.ndarray]
        Mapping of model name to its prediction array.
    timestamps : array-like
        Corresponding timestamps.
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    """
    # Convert timestamps to pandas DatetimeIndex
    time_index = pd.to_datetime(timestamps)
    
    # 自定义模型颜色映射和透明度
    model_settings = {
        name: {'color': MODEL_COLORS.get(name, '#808080'), 'alpha': 0.9, 'linewidth': 1.5, 'linestyle': '-'}
        for name in predictions_dict.keys()
    }
    # Emphasize Model4 and Model5 as before, but keep fixed colors
    if 'Model4' in model_settings:
        model_settings['Model4'].update({'alpha': 1.0, 'linewidth': 1.9, 'linestyle': ':'})
    if 'Model5' in model_settings:
        model_settings['Model5'].update({'alpha': 1.0, 'linewidth': 2.0, 'linestyle': '-'})
    
    # Create mask for date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (time_index >= start) & (time_index <= end)
    
    if mask.sum() < 2:
        print(f"[Warning] Not enough samples in date range {start_date} to {end_date}")
        return
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    # 绘制实际值
    plt.plot(time_index[mask], y_actual_real[mask], 
             label='Actual', color='black', linewidth=1.5, zorder=10)
    
    # 先绘制非重点模型（使其在底层）
    for model_name, preds in predictions_dict.items():
        if model_name not in ['Model4', 'Model5']:
            settings = model_settings.get(model_name, {'color': '#808080', 'alpha': 0.4, 'linewidth': 1.0, 'linestyle': '-'})
            plt.plot(time_index[mask], preds[mask], 
                     label=model_name, 
                     color=settings['color'], 
                     linewidth=settings['linewidth'], 
                     alpha=settings['alpha'], 
                     linestyle=settings['linestyle'],
                     zorder=5)
    
    # 再绘制重点模型（使其在顶层）
    for model_name, preds in predictions_dict.items():
        if model_name in ['Model4', 'Model5']:
            settings = model_settings.get(model_name)
            plt.plot(time_index[mask], preds[mask], 
                     label=model_name, 
                     color=settings['color'], 
                     linewidth=settings['linewidth'], 
                     alpha=settings['alpha'], 
                     linestyle=settings['linestyle'],
                     zorder=15)
    
    plt.xlabel('Timestamp')
    plt.ylabel('Grid Energy Compensation Value (kW·h)')
    
    # 调整图例顺序
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0]  # Actual first
    # Model4 and Model5 next
    for i, label in enumerate(labels):
        if label in ['Model4', 'Model5'] and i != 0:
            order.append(i)
    # Other models last
    for i, label in enumerate(labels):
        if label not in ['Model4', 'Model5', 'Actual'] and i != 0:
            order.append(i)
    
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
               loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Add y-axis major locator for better readability
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20000))
    
    plt.show()

def plot_value_and_error_histograms(y_actual_real, predictions_dict, bins=30):
    """
    Draw histograms of (1) actual value distribution and (2) prediction error distribution.

    Prediction error is defined as ``prediction – actual``.
    One histogram per model is overlayed for error distribution.
    """
    plt.figure(figsize=(14, 5))

    # -------- Histogram of actual values -------- #
    plt.subplot(1, 2, 1)
    plt.hist(y_actual_real, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('Grid Energy Compensation Value (kW·h)')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y')

    # -------- Histogram of prediction errors -------- #
    plt.subplot(1, 2, 2)
    
    # 固定的模型颜色映射
    model_colors = dict(MODEL_COLORS)
    
    # 使用固定的bin边界范围，确保所有图表一致
    # 基于xlim(-20000, 20000)设置固定的bin边界
    bin_edges = np.linspace(-20000, 20000, bins + 1)
    
    for model_name, preds in predictions_dict.items():
        errors = preds - y_actual_real
        # 使用固定的颜色映射，如果模型名不在映射中，使用默认颜色
        color = model_colors.get(model_name, '#808080')  # 灰色作为默认
        
        plt.hist(errors,
                 bins=bin_edges,        # 使用固定的bin边界
                 alpha=0.5,
                 label=model_name,
                 color=color,
                 edgecolor='black')

    plt.xlim(-20000, 20000)
    plt.xlabel('Prediction Error of Model and Actual Value(kW·h)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

def plot_error_max_curve(y_actual_real,
                         predictions_dict,
                         bins: int = 30,
                         smooth_sigma: float = 1.0,
                         # 新增参数
                         alpha: float = 0.9,                # 默认透明度
                         model5_alpha: float = 1.0,         # Model5 的透明度  
                         model5_linewidth: float = 2.3,     # Model5 的线条粗细
                         model5_color: str = '#DC143C',     # Model5 的颜色（深红色）
                         default_linewidth: float = 1.7):   # 其他模型的默认线条粗细
    """
    为 predictions_dict 中的每个模型绘制一条曲线：
    曲线上的点来自该模型误差直方图每个 bin 的最高计数，
    再经高斯平滑后连接而成。

    参数
    ----
    y_actual_real : ndarray
        真实值（原始尺度）。
    predictions_dict : Dict[str, ndarray]
        {'模型名': 预测值}。
    bins : int
        直方图分箱数量（应与直方图保持一致）。
    smooth_sigma : float
        高斯平滑 σ；设为 0 可关闭平滑。
    alpha : float
        所有曲线的默认透明度（0-1），默认0.9。
    model5_alpha : float
        Model5 的特殊透明度（0-1），默认1.0（不透明）。
    model5_linewidth : float
        Model5 的线条粗细，默认2.0。
    model5_color : str
        Model5 的颜色，默认深红色。
    default_linewidth : float
        其他模型的默认线条粗细，默认1.7。
    """
    from scipy.ndimage import gaussian_filter1d

    # -------- 1. 计算各模型直方图计数 -------- #
    bin_edges = None
    model_curves = {}
    for model_name, preds in predictions_dict.items():
        errors = preds - y_actual_real
        counts, edges = np.histogram(errors, bins=bins)
        model_curves[model_name] = counts          # 每个 bin 的最高点即计数
        if bin_edges is None:
            bin_edges = edges                      # 记录一次 bin 边界

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # -------- 2. 绘制顺滑曲线 -------- #
    plt.figure(figsize=(10, 5))
    # Use fixed colors for consistency
    colors = mpl.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':',(0, (3, 1, 1, 1))]
    marker_styles = ['^', 's', 'o', 'h', 'p']
    
    for i, (model_name, counts) in enumerate(model_curves.items()):
        curve = (gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)
                 if smooth_sigma > 0 else counts)
        
        # 检查是否为 Model5，应用特殊设置
        if model_name == 'Model5':
            plt.plot(bin_centers,
                     curve,
                     label=model_name,
                     color=MODEL_COLORS.get('Model5', model5_color),              # 使用 Model5 的特殊颜色（优先固定颜色表）
                     linewidth=model5_linewidth,      # 使用 Model5 的特殊线条粗细
                     marker=marker_styles[i % len(marker_styles)],
                     linestyle='-',                   # 固定为实线
                     markersize=10,
                     alpha=model5_alpha,              # 使用 Model5 的特殊透明度
                     zorder=10)                       # 确保 Model5 在最上层
        else:
            plt.plot(bin_centers,
                     curve,
                     label=model_name,
                     color=MODEL_COLORS.get(model_name, colors[i % len(colors)]),
                     linewidth=default_linewidth,     # 使用默认线条粗细
                     marker=marker_styles[i % len(marker_styles)],
                     linestyle=line_styles[i % len(line_styles)],
                     markersize=10,
                     alpha=alpha)                     # 使用默认透明度

    # -------- 3. 图形美化 -------- #
    #plt.title('Smoothed Histogram Curves of Prediction Errors')
    plt.xlim(-20000, 20000)
    plt.xlabel('Prediction Error of Model and Actual Value(kW·h)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 8. Main Function
def main(use_log_transform = True, min_egrid_threshold = 1.0):
    """
    [Main Function]
    - Workflow: Data Loading -> Feature Engineering -> Data Splitting -> Sequence Construction -> Model Building -> Training -> Evaluation and Visualization.
    Parameters:
      use_log_transform: Whether to apply logarithmic transformation to target values (True/False)
      min_egrid_threshold: Minimum threshold for E_grid, filter out values below this (default: 1.0)
    """
    print("Loading raw data...")
    data_df = load_data()

    # Plot correlation heatmap (before filtering E_grid = 0)
    feature_cols_to_plot = ['season', 'holiday', 'weather', 'temperature', 'working_hours', 'E_wind','E_PV','v_wind', 'wind_direction', 'E_grid']
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns] 
    plot_correlation_heatmap(data_df, feature_cols_to_plot)

    # Filter out rows with E_grid = 0
    data_df = data_df[data_df['E_grid'] != 0].copy()
    data_df.reset_index(drop = True, inplace = True)

    # Feature engineering (without standardization to avoid data leakage)
    data_df, feature_cols, target_col = feature_engineering(data_df)
    
    # ------------- 计算特征重要性 -------------
    pearson_importance = calculate_feature_importance(
        data_df, feature_cols, target_col)

    mic_importance = calculate_feature_importance_mic(
        data_df, feature_cols, target_col)

    # 0.5*Pearson + 0.5*MIC 加权平均
    combined_importance = 0.5 * pearson_importance + 0.5 * mic_importance

    # 打印三列对比
    print("\n========== 特征对 E_grid 的影响因子分析 ==========")
    print("注：Pearson衡量线性相关性，MIC衡量线性+非线性相关性")
    print("影响因子范围: 0~1 (越接近1表示该特征对E_grid影响越大)")
    print("-" * 85)
    print(f"{'特征名称':>25} | {'Pearson系数':>12} | {'MIC系数':>12} | {'综合影响因子':>12}")
    print("-" * 85)
    for feat, p_val, m_val, c_val in zip(feature_cols, pearson_importance, mic_importance, combined_importance):
        # 处理NaN值的显示
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        m_str = f"{m_val:.4f}" if not np.isnan(m_val) else "N/A"
        c_str = f"{c_val:.4f}" if not np.isnan(c_val) else "N/A"
        print(f"{feat:>25} | {p_str:>12} | {m_str:>12} | {c_str:>12}")
    print("-" * 85) 

    # 后续模型使用 combined_importance
    feature_importance = combined_importance

    # Filter out small E_grid values
    data_df = data_df[data_df[target_col] > min_egrid_threshold].copy()
    data_df.reset_index(drop = True, inplace = True)

    analyze_target_distribution(data_df, target_col)
    plot_Egrid_over_time(data_df)

    # Split data into training, validation, and test sets by time series
    X_all_raw = data_df[feature_cols].values
    y_all_raw = data_df[target_col].values
    timestamps_all = data_df['timestamp'].values

    total_samples = len(data_df)
    train_size = int(0.8 * total_samples)      # 80% for training
    val_size   = int(0.1 * total_samples)        # 10% for validation
    test_size  = total_samples - train_size - val_size  # Remaining for testing

    X_train_raw = X_all_raw[:train_size]
    y_train_raw = y_all_raw[:train_size]
    X_val_raw   = X_all_raw[train_size : train_size + val_size]
    y_val_raw   = y_all_raw[train_size : train_size + val_size]
    X_test_raw  = X_all_raw[train_size + val_size:]
    y_test_raw  = y_all_raw[train_size + val_size:]

    train_timestamps = timestamps_all[:train_size]
    val_timestamps   = timestamps_all[train_size : train_size + val_size]
    test_timestamps = timestamps_all[train_size + val_size + window_size:]

    # Apply logarithmic transformation to target values if set
    if use_log_transform:
        y_train_raw = np.log1p(y_train_raw)
        y_val_raw   = np.log1p(y_val_raw)
        y_test_raw  = np.log1p(y_test_raw)

    # Standardize features: fit on training set, transform on validation/test sets
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train = scaler_X.transform(X_train_raw)
    X_val   = scaler_X.transform(X_val_raw)
    X_test  = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler().fit(y_train_raw.reshape(-1, 1))
    y_train  = scaler_y.transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val    = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test   = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

    # Construct time series sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   window_size)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  window_size)

    print(f"[Info] Train/Val/Test samples: {X_train_seq.shape[0]}, {X_val_seq.shape[0]}, {X_test_seq.shape[0]}")
    print(f"[Info] Feature dimension: {X_train_seq.shape[-1]}, Window size: {window_size}")

    train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
    val_dataset   = TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq))
    test_dataset  = TensorDataset(torch.from_numpy(X_test_seq), torch.from_numpy(y_test_seq))

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,  num_workers = num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size = batch_size, shuffle = False, num_workers = num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size = batch_size, shuffle = False, num_workers = num_workers)

    # 为新增三种模型（PatchTST / LightGBM / SARIMA & s-naive）准备：FeatureGating + 序列展开
    X_train_seq_g = apply_feature_gating_to_sequences(X_train_seq, feature_importance)
    X_val_seq_g   = apply_feature_gating_to_sequences(X_val_seq,   feature_importance)
    X_test_seq_g  = apply_feature_gating_to_sequences(X_test_seq,  feature_importance)
    # Safety: replace any NaN/Inf introduced by preprocessing
    X_train_seq_g = np.nan_to_num(X_train_seq_g, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_seq_g   = np.nan_to_num(X_val_seq_g,   nan=0.0, posinf=0.0, neginf=0.0)
    X_test_seq_g  = np.nan_to_num(X_test_seq_g,  nan=0.0, posinf=0.0, neginf=0.0)

    # Safety: labels for PatchTST loaders
    y_train_seq_safe = np.nan_to_num(y_train_seq, nan=0.0, posinf=0.0, neginf=0.0)
    y_val_seq_safe   = np.nan_to_num(y_val_seq,   nan=0.0, posinf=0.0, neginf=0.0)

    # LightGBM: 将序列展开为滞后表（含历法特征已在特征工程中加入）
    X_train_tab = flatten_sequences_for_tabular(X_train_seq_g)
    X_val_tab   = flatten_sequences_for_tabular(X_val_seq_g)
    X_test_tab  = flatten_sequences_for_tabular(X_test_seq_g)
    # Safety: LightGBM 不接受 NaN/Inf
    X_train_tab = np.nan_to_num(X_train_tab, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_tab   = np.nan_to_num(X_val_tab,   nan=0.0, posinf=0.0, neginf=0.0)
    X_test_tab  = np.nan_to_num(X_test_tab,  nan=0.0, posinf=0.0, neginf=0.0)

    # Build models
    feature_dim = X_train_seq.shape[-1]
    model1 = EModel_FeatureWeight1(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2,
        lstm_dropout      = 0.2
    ).to(device)
    
    model2 = EModel_FeatureWeight2(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2,
        lstm_dropout      = 0.2
    ).to(device)

    model3 = EModel_FeatureWeight3(
        feature_dim       = feature_dim,
        gru_hidden_size   = 128,  # 固定为10个隐藏层节点
        gru_num_layers    = 2, 
        gru_dropout       = 0.2,
        use_local_attn    = True,
        local_attn_window_size = 5
    ).to(device)

    model4 = EModel_FeatureWeight4(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2,
        lstm_dropout      = 0.1,
        feature_importance = feature_importance  # 使用加权后的特征重要性
    ).to(device)

    model5 = EModel_FeatureWeight5(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2,
        lstm_dropout      = 0.2
    ).to(device)

    # Transformer 代表：PatchTST
    patch_model = PatchTST(
        feature_dim=feature_dim,
        window_size=window_size,
        patch_len=5,
        patch_stride=5,
        d_model=256, n_heads=8, num_layers=2, dropout=0.1
    ).to(device)

    print("\n========== Training Model: PatchTST ==========")
    hist_patch = train_model(
        model         = patch_model,
        train_loader  = DataLoader(TensorDataset(torch.from_numpy(X_train_seq_g), torch.from_numpy(y_train_seq_safe)), batch_size = batch_size, shuffle = True,  num_workers = num_workers),
        val_loader    = DataLoader(TensorDataset(torch.from_numpy(X_val_seq_g),   torch.from_numpy(y_val_seq_safe)),   batch_size = batch_size, shuffle = False, num_workers = num_workers),
        model_name    = 'PatchTST',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )

    # Train Model: EModel_FeatureWeight1
    print("\n========== Training Model: 1 ==========")
    hist1 = train_model(
        model         = model1,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight1',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )
    # Train Model: EModel_FeatureWeight2
    print("\n========== Training Model: 2 ==========")
    hist2 = train_model(
        model         = model2,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight2',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )

    # Train Model: EModel_FeatureWeight3
    print("\n========== Training Model: 3 ==========")
    hist3 = train_model(
        model         = model3,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight3',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )
    # Train Model: EModel_FeatureWeight4
    print("\n========== Training Model: 4 ==========")
    hist4 = train_model(
        model         = model4,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight4',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )
    # Train Model: EModel_FeatureWeight5
    print("\n========== Training Model: 5 ==========")
    hist5 = train_model(
        model         = model5,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight5',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )

    # 修改加载模型部分的代码
    best_model1 = EModel_FeatureWeight1(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2
    ).to(device)
    best_model1.load_state_dict(torch.load('best_EModel_FeatureWeight1.pth', map_location=device, weights_only=True), strict=False)

    best_model2 = EModel_FeatureWeight2(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2
    ).to(device)
    best_model2.load_state_dict(torch.load('best_EModel_FeatureWeight2.pth', map_location=device, weights_only=True), strict=False)

    best_model3 = EModel_FeatureWeight3(
        feature_dim       = feature_dim,
        gru_hidden_size   = 128,  # 固定为10个隐藏层节点
        gru_num_layers    = 2
    ).to(device)
    best_model3.load_state_dict(torch.load('best_EModel_FeatureWeight3.pth', map_location=device, weights_only=True), strict=False)
    
    best_model4 = EModel_FeatureWeight4(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2
    ).to(device)
    best_model4.load_state_dict(torch.load('best_EModel_FeatureWeight4.pth', map_location=device, weights_only=True), strict=False)

    best_model5 = EModel_FeatureWeight5(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2
    ).to(device)
    best_model5.load_state_dict(torch.load('best_EModel_FeatureWeight5.pth', map_location=device, weights_only=True), strict=False)

    # Evaluate on test set (standardized domain)
    criterion_test = nn.SmoothL1Loss(beta = 1.0)
    (_, test_rmse1_std, test_mape1_std, test_r21_std, test_mse1_std, test_mae1_std, preds1_std, labels1_std) = evaluate(best_model1, test_loader, criterion_test)
    (_, test_rmse2_std, test_mape2_std, test_r22_std, test_mse2_std, test_mae2_std, preds2_std, labels2_std) = evaluate(best_model2, test_loader, criterion_test)
    (_, test_rmse3_std, test_mape3_std, test_r23_std, test_mse3_std, test_mae3_std, preds3_std, labels3_std) = evaluate(best_model3, test_loader, criterion_test)
    (_, test_rmse4_std, test_mape4_std, test_r24_std, test_mse4_std, test_mae4_std, preds4_std, labels4_std) = evaluate(best_model4, test_loader, criterion_test)
    (_, test_rmse5_std, test_mape5_std, test_r25_std, test_mse5_std, test_mae5_std, preds5_std, labels5_std) = evaluate(best_model5, test_loader, criterion_test)

    print("\n========== [Test Set Evaluation (Standardized Domain)] ==========")
    print(f"[EModel_FeatureWeight1]  RMSE: {test_rmse1_std:.4f}, MAPE: {test_mape1_std:.2f}%, R²: {test_r21_std:.4f}, mse: {test_mse1_std:.2f}%, MAE: {test_mae1_std:.4f}")
    print(f"[EModel_FeatureWeight2]  RMSE: {test_rmse2_std:.4f}, MAPE: {test_mape2_std:.2f}%, R²: {test_r22_std:.4f}, mse: {test_mse2_std:.2f}%, MAE: {test_mae2_std:.4f}")
    print(f"[EModel_FeatureWeight3]  RMSE: {test_rmse3_std:.4f}, MAPE: {test_mape3_std:.2f}%, R²: {test_r23_std:.4f}, mse: {test_mse3_std:.2f}%, MAE: {test_mae3_std:.4f}")
    print(f"[EModel_FeatureWeight4]  RMSE: {test_rmse4_std:.4f}, MAPE: {test_mape4_std:.2f}%, R²: {test_r24_std:.4f}, mse: {test_mse4_std:.2f}%, MAE: {test_mae4_std:.4f}")
    print(f"[EModel_FeatureWeight5]  RMSE: {test_rmse5_std:.4f}, MAPE: {test_mape5_std:.2f}%, R²: {test_r25_std:.4f}, mse: {test_mse5_std:.2f}%, MAE: {test_mae5_std:.4f}")

    # ----标准化域上的预测 ----
    # 1) PatchTST
    best_patch = PatchTST(feature_dim=feature_dim, window_size=window_size, patch_len=5, patch_stride=5,
                          d_model=256, n_heads=8, num_layers=2, dropout=0.1).to(device)
    best_patch.load_state_dict(torch.load('best_PatchTST.pth', map_location=device, weights_only=True), strict=False)
    (_, rmse_p_std, mape_p_std, r2_p_std, mse_p_std, mae_p_std, preds_patch_std, labels_patch_std) = evaluate(best_patch, test_loader, nn.SmoothL1Loss(beta=1.0))

    # 2) LightGBM（与训练/验证域一致：标准化后的 y_seq）
    # Safety: y 序列也需要无 NaN/Inf
    y_train_seq_safe = np.nan_to_num(y_train_seq, nan=0.0, posinf=0.0, neginf=0.0)
    y_val_seq_safe   = np.nan_to_num(y_val_seq,   nan=0.0, posinf=0.0, neginf=0.0)
    booster = train_lightgbm(X_train_tab, y_train_seq_safe, X_val_tab, y_val_seq_safe, patience_rounds=50, seed=42)
    if booster is not None:
        best_iter = getattr(booster, 'best_iteration', None)
        if best_iter is None or best_iter <= 0:
            preds_lgb_std = booster.predict(X_test_tab)
        else:
            preds_lgb_std = booster.predict(X_test_tab, num_iteration=best_iter)
    else:
        preds_lgb_std = np.zeros_like(y_test_seq)

    # 3) SARIMA（用训练+验证的标准化 y，预测 test_seq 步）
    y_train_val_std = np.concatenate([y_train, y_val])
    y_train_val_std = np.nan_to_num(y_train_val_std, nan=0.0, posinf=0.0, neginf=0.0)
    n_test_seq = len(y_test_seq)
    preds_sarima_std = fit_predict_sarima(y_train_val_std, n_test=n_test_seq, m=seasonal_period, search_small=True, seed=42)
    if preds_sarima_std is None:
        preds_sarima_std = np.zeros_like(y_test_seq)

    # 4) s-naive：y[t] = y[t-m]（标准化域）
    y_all_std = np.concatenate([y_train, y_val, y_test])
    y_all_std = np.nan_to_num(y_all_std, nan=0.0, posinf=0.0, neginf=0.0)
    start_idx = train_size + val_size + window_size
    preds_snaive_std = predict_snaive_std(y_all_std, start_index=start_idx, n_test=n_test_seq, m=seasonal_period)

    # ---- 反标准化并（可选）反log，与现有五模保持一致 ----
    preds_patch_real_std   = scaler_y.inverse_transform(preds_patch_std.reshape(-1, 1)).ravel()
    preds_lgb_real_std     = scaler_y.inverse_transform(preds_lgb_std.reshape(-1, 1)).ravel()
    preds_sarima_real_std  = scaler_y.inverse_transform(preds_sarima_std.reshape(-1, 1)).ravel()
    preds_snaive_real_std  = scaler_y.inverse_transform(preds_snaive_std.reshape(-1, 1)).ravel()
    labels_new_real_std    = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()

    if use_log_transform:
        preds_patch_real  = np.expm1(preds_patch_real_std)
        preds_lgb_real    = np.expm1(preds_lgb_real_std)
        preds_sarima_real = np.expm1(preds_sarima_real_std)
        preds_snaive_real = np.expm1(preds_snaive_real_std)
        labels_new_real   = np.expm1(labels_new_real_std)
    else:
        preds_patch_real  = preds_patch_real_std
        preds_lgb_real    = preds_lgb_real_std
        preds_sarima_real = preds_sarima_real_std
        preds_snaive_real = preds_snaive_real_std
        labels_new_real   = labels_new_real_std
    
    # Inverse standardization and (optionally) inverse logarithmic transformation
    preds1_real_std = scaler_y.inverse_transform(preds1_std.reshape(-1, 1)).ravel()
    preds2_real_std = scaler_y.inverse_transform(preds2_std.reshape(-1, 1)).ravel()
    preds3_real_std = scaler_y.inverse_transform(preds3_std.reshape(-1, 1)).ravel()
    preds4_real_std = scaler_y.inverse_transform(preds4_std.reshape(-1, 1)).ravel()
    preds5_real_std = scaler_y.inverse_transform(preds5_std.reshape(-1, 1)).ravel()
    labels1_real_std = scaler_y.inverse_transform(labels1_std.reshape(-1, 1)).ravel()
    labels2_real_std = scaler_y.inverse_transform(labels2_std.reshape(-1, 1)).ravel()
    labels3_real_std = scaler_y.inverse_transform(labels3_std.reshape(-1, 1)).ravel()
    labels4_real_std = scaler_y.inverse_transform(labels4_std.reshape(-1, 1)).ravel()
    labels5_real_std = scaler_y.inverse_transform(labels5_std.reshape(-1, 1)).ravel()

    if use_log_transform:
        preds1_real = np.expm1(preds1_real_std)
        preds2_real = np.expm1(preds2_real_std)
        preds3_real = np.expm1(preds3_real_std)
        preds4_real = np.expm1(preds4_real_std)
        preds5_real = np.expm1(preds5_real_std)
        labels1_real = np.expm1(labels1_real_std)
        labels2_real = np.expm1(labels2_real_std)
        labels3_real = np.expm1(labels3_real_std)
        labels4_real = np.expm1(labels4_real_std)
        labels5_real = np.expm1(labels5_real_std)
    else:
        preds1_real = preds1_real_std
        preds2_real = preds2_real_std
        preds3_real = preds3_real_std
        preds4_real = preds4_real_std
        preds5_real = preds5_real_std
        labels1_real = labels1_real_std
        labels2_real = labels2_real_std
        labels3_real = labels3_real_std
        labels4_real = labels4_real_std
        labels5_real = labels5_real_std

    # Compute RMSE in original domain
    test_rmse1_real = np.sqrt(mean_squared_error(labels1_real, preds1_real))
    test_rmse2_real = np.sqrt(mean_squared_error(labels2_real, preds2_real))
    test_rmse3_real = np.sqrt(mean_squared_error(labels3_real, preds3_real))
    test_rmse4_real = np.sqrt(mean_squared_error(labels4_real, preds4_real))
    test_rmse5_real = np.sqrt(mean_squared_error(labels5_real, preds5_real))

    # 使用 Model4 与其他模型对比，生成全长与缩放窗口图，以及分布直方图
    for m_name, m_preds in [('Model1', preds1_real),
                           ('Model2', preds2_real),
                           ('Model3', preds3_real),
                           ('Model4', preds4_real)]:
        pair_preds = {'Model5': preds5_real, m_name: m_preds}
        
        plot_value_and_error_histograms(
            y_actual_real = labels5_real,
            predictions_dict = pair_preds,
            bins = 30
        )

    all_model_preds = {
        'Model1': preds1_real,
        'Model2': preds2_real,
        'Model3': preds3_real,
        'Model4': preds4_real,
        'Model5': preds5_real
    }

    # 使用 Model4 与其他模型对比，生成全长与缩放窗口图，以及分布直方图
    primary_preds = {'Model4': preds4_real, 'Model5': preds5_real}

    # 合并原5模 + 新3模
    all_model_preds_ext = dict(all_model_preds)
    all_model_preds_ext.update({
        'PatchTST': preds_patch_real,
        'LightGBM': preds_lgb_real,
        'SARIMA':   preds_sarima_real,
        's-naive':  preds_snaive_real
    })

    # 可视化（全模型）
    plot_predictions_date_range(
        y_actual_real = labels_new_real,
        predictions_dict = all_model_preds_ext,
        timestamps = test_timestamps,
        start_date = '2024-11-25',
        end_date   = '2024-12-13'
    )
    plot_value_and_error_histograms(
        y_actual_real = labels_new_real,
        predictions_dict = all_model_preds_ext,
        bins = 30
    )
    
    # 绘制两个时间段的图表
    plot_predictions_date_range(
        y_actual_real = labels5_real,
        predictions_dict = primary_preds,
        timestamps = test_timestamps,
        start_date = '2024-11-25',
        end_date = '2024-12-13'
    )

    plot_predictions_date_range(
        y_actual_real = labels5_real,
        predictions_dict = primary_preds,
        timestamps = test_timestamps,
        start_date = '2024-12-13',
        end_date = '2025-01-01'
    )

    # 绘制两个时间段的图表（所有模型）
    plot_predictions_date_range(
        y_actual_real = labels5_real,
        predictions_dict = all_model_preds,
        timestamps = test_timestamps,
        start_date = '2024-11-25',
        end_date = '2024-12-13'
    )

    plot_predictions_date_range(
        y_actual_real = labels5_real,
        predictions_dict = all_model_preds,
        timestamps = test_timestamps,
        start_date = '2024-12-13',
        end_date = '2025-01-01'
    )

    plot_error_max_curve(
        y_actual_real = labels5_real,
        predictions_dict = primary_preds,
        bins = 30,
        smooth_sigma = 1.0
    )
    print("[Info] Processing complete!")

    print("\n========== [Test Set Evaluation (Original Domain)] ==========")
    print(f"[EModel_FeatureWeight1] => RMSE (original): {test_rmse1_real:.2f}")
    print(f"[EModel_FeatureWeight2] => RMSE (original): {test_rmse2_real:.2f}")
    print(f"[EModel_FeatureWeight3] => RMSE (original): {test_rmse3_real:.2f}")
    print(f"[EModel_FeatureWeight4] => RMSE (original): {test_rmse4_real:.2f}")
    print(f"[EModel_FeatureWeight5] => RMSE (original): {test_rmse5_real:.2f}")

    # Dataset statistics
    print(f"\n[Dataset Statistics] Total samples: {total_samples}")
    print(f"Training set: {train_size} ({train_size / total_samples:.1%})")
    print(f"Validation set: {val_size} ({val_size / total_samples:.1%})")
    print(f"Test set: {test_size} ({test_size / total_samples:.1%})")

    # Plot dataset time distribution
    plot_dataset_distribution(train_timestamps, 'Training Set')
    plot_dataset_distribution(val_timestamps, 'Validation Set')
    plot_dataset_distribution(test_timestamps, 'Test Set')

    # ----------------- 1) 五个模型整体对比 -----------------

    plot_value_and_error_histograms(
        y_actual_real = labels5_real,
        predictions_dict = all_model_preds,
        bins = 30
    )

    plot_error_max_curve(
        y_actual_real = labels5_real,
        predictions_dict = all_model_preds,  # 或 primary_preds
        bins = 30,
        smooth_sigma = 1.0
    )

    # ----------------- 2) 与 Model4 的两两对比 -----------------
    # 可读性更高，用循环依次绘制
    for m_name, m_preds in [('Model1', preds1_real),
                            ('Model2', preds2_real),
                            ('Model3', preds3_real),
                            ('Model5', preds5_real)]:   # 追加 Model5
        plot_predictions_comparison(
            y_actual_real = labels5_real,
            predictions_dict = {'Model5': preds5_real, m_name: m_preds},
            timestamps = test_timestamps
        )

    # 标准化域（与已有5模一致口径）
    print(f"[PatchTST]   RMSE: {rmse_p_std:.4f}, MAPE: {mape_p_std:.2f}%, R²: {r2_p_std:.4f}, mse: {mse_p_std:.2f}%, MAE: {mae_p_std:.4f}")
    # LightGBM / SARIMA / s-naive 在标准化域的指标（与 evaluate 口径一致）：
    def _calc_std_metrics(y_true_std, y_pred_std):
        rmse = np.sqrt(mean_squared_error(y_true_std, y_pred_std))
        nonzero = (y_true_std != 0)
        mape = (np.mean(np.abs((y_true_std[nonzero] - y_pred_std[nonzero]) / y_true_std[nonzero])) * 100.0) if np.sum(nonzero) else 0.0
        ss_res = np.sum((y_true_std - y_pred_std) ** 2)
        ss_tot = np.sum((y_true_std - np.mean(y_true_std)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        mse = np.mean((y_true_std - y_pred_std) ** 2)
        mae = np.mean(np.abs(y_true_std - y_pred_std))
        return rmse, mape, r2, mse, mae

    rmse_lgb, mape_lgb, r2_lgb, mse_lgb, mae_lgb = _calc_std_metrics(y_test_seq, preds_lgb_std)
    rmse_sar, mape_sar, r2_sar, mse_sar, mae_sar = _calc_std_metrics(y_test_seq, preds_sarima_std)
    rmse_snv, mape_snv, r2_snv, mse_snv, mae_snv = _calc_std_metrics(y_test_seq, preds_snaive_std)

    print(f"[LightGBM]   RMSE: {rmse_lgb:.4f}, MAPE: {mape_lgb:.2f}%, R²: {r2_lgb:.4f}, mse: {mse_lgb:.2f}%, MAE: {mae_lgb:.4f}")
    print(f"[SARIMA]     RMSE: {rmse_sar:.4f}, MAPE: {mape_sar:.2f}%, R²: {r2_sar:.4f}, mse: {mse_sar:.2f}%, MAE: {mae_sar:.4f}")
    print(f"[s-naive]    RMSE: {rmse_snv:.4f}, MAPE: {mape_snv:.2f}%, R²: {r2_snv:.4f}, mse: {mse_snv:.2f}%, MAE: {mae_snv:.4f}")

    # 原始域 RMSE（与已有5模一致口径）
    rmse_patch_real  = np.sqrt(mean_squared_error(labels_new_real, preds_patch_real))
    rmse_lgb_real    = np.sqrt(mean_squared_error(labels_new_real, preds_lgb_real))
    rmse_sarima_real = np.sqrt(mean_squared_error(labels_new_real, preds_sarima_real))
    rmse_snaive_real = np.sqrt(mean_squared_error(labels_new_real, preds_snaive_real))
    print(f"[PatchTST] => RMSE (original): {rmse_patch_real:.2f}")
    print(f"[LightGBM] => RMSE (original): {rmse_lgb_real:.2f}")
    print(f"[SARIMA]   => RMSE (original): {rmse_sarima_real:.2f}")
    print(f"[s-naive]  => RMSE (original): {rmse_snaive_real:.2f}")
    
    # ----------------- 3) 训练曲线 -----------------
    plot_training_curves_allmetrics(hist1, model_name = 'EModel_FeatureWeight1')
    plot_training_curves_allmetrics(hist2, model_name = 'EModel_FeatureWeight2')
    plot_training_curves_allmetrics(hist3, model_name = 'EModel_FeatureWeight3')
    plot_training_curves_allmetrics(hist4, model_name = 'EModel_FeatureWeight4')
    plot_training_curves_allmetrics(hist5, model_name = 'EModel_FeatureWeight5')

    print("[Info] Processing complete!")

if __name__ == "__main__":
    main(use_log_transform = True, min_egrid_threshold = 1.0)