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
 
# Global style settings for plots
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 22,          # Global default font size
    'axes.labelsize': 22,     # Axis label font size
    'axes.titlesize': 24,     # Chart title font size
    'xtick.labelsize': 20,    # x-axis tick label size
    'ytick.labelsize': 20     # y-axis tick label size
})

# Global hyperparameters
learning_rate     = 1e-3   # Learning rate
num_epochs        = 150    # Number of training epochs
batch_size        = 128    # Batch size
weight_decay      = 1e-4   # Weight decay
patience          = 12     # Patience for early stopping
num_workers       = 0      # Number of worker threads
window_size       = 20     # Sequence window size

# Set random seed and device
torch.manual_seed(42)       # Random seed = 42
np.random.seed(42)          # Random seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  


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
    
class EModel_FeatureWeight21(nn.Module):
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
                 use_local_attn = True,
                 local_attn_window_size = 5
                ):
        super(EModel_FeatureWeight21, self).__init__()
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
        
        # Bidirectional GRU (替换LSTM)
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
    """
    def __init__(self, 
                 feature_dim, 
                 lstm_hidden_size = 256, 
                 lstm_num_layers = 2, 
                 lstm_dropout = 0.2,
                 use_local_attn = True,
                 local_attn_window_size = 5
                ):
        super(EModel_FeatureWeight4, self).__init__()
        self.feature_dim = feature_dim
        self.use_local_attn = use_local_attn  # 保存是否使用局部注意力的标识
        
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
            nn.Linear(window_size, window_size * 2),
            nn.GELU(),
            nn.LayerNorm(window_size * 2),
            nn.Linear(window_size * 2, 1),
            nn.Sigmoid()
        )
        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, 2 * lstm_hidden_size),
            nn.LayerNorm(2 * lstm_hidden_size),
            nn.GELU()
        )
        
        self.feature_importance = nn.Parameter(torch.ones(feature_dim), requires_grad=True)
        
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
    """
    [Model 5: LSTM-based Model with Feature Weighting]
    Parameters:
      - feature_dim: Input feature dimension
      - lstm_hidden_size: LSTM hidden size
      - lstm_num_layers: Number of LSTM layers
      - lstm_dropout: LSTM dropout probability
      - use_local_attn: Whether to use local attention (default: False)
      - local_attn_window_size: Window size for local attention (default: 5)
    """
    def __init__(self, 
                 feature_dim, 
                 lstm_hidden_size = 256, 
                 lstm_num_layers = 3, 
                 lstm_dropout = 0.1,
                 use_local_attn = False,
                 local_attn_window_size = 5,
                 proj_dim = 512           # 新增投影维度参数
                ):
        
        super(EModel_FeatureWeight5, self).__init__()
        self.feature_dim = feature_dim
        self.use_local_attn = use_local_attn  # 保存该标识

        # 增强特征门控机制
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.GELU(),
            nn.Linear(feature_dim*2, feature_dim),
            nn.Sigmoid()
        )
        
        # 改进的局部注意力机制
        if use_local_attn:
            from local_attention.local_attention import LocalAttention
            self.temporal_attn = LocalAttention(
                dim = 2 * lstm_hidden_size,
                window_size = local_attn_window_size,
                causal = False,
                dropout = 0.1,
                prenorm = True
            )
        else:
            self.temporal_attn = Attention(input_dim = 2 * lstm_hidden_size)
        
        # 增强特征注意力层
        self.feature_attn = nn.Sequential(
            nn.Linear(window_size, window_size*2),
            nn.GELU(),
            nn.Linear(window_size*2, 1),
            nn.Sigmoid()
        )
        
        # 增强特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 可学习的特征重要性权重（增加初始化范围）
        self.feature_importance = nn.Parameter(torch.rand(feature_dim)*2-1, requires_grad=True)
        
        # 增强LSTM配置
        self.lstm = nn.LSTM(
            input_size    = feature_dim,
            hidden_size   = lstm_hidden_size,
            num_layers    = lstm_num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = lstm_dropout if lstm_num_layers > 1 else 0
        )
        self._init_lstm_weights()
        
        # 增强全连接层
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size + proj_dim, 256),  # 正确拼接后的维度
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
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
                param.data[(n // 4):(n // 2)].fill_(1.0)

    def forward(self, x):
        """
        Parameters:
          x: Input tensor with shape [batch_size, seq_len, feature_dim]
        Returns:
          Predicted output with shape [batch_size]
        """
        # Dynamic feature weighting
        gate = self.feature_gate(x.mean(dim = 1))
        x = x * gate.unsqueeze(1)

        # Process with LSTM
        lstm_out, _ = self.lstm(x)
        
        # Temporal attention
        if self.use_local_attn:
            temporal = self.temporal_attn(lstm_out, lstm_out, lstm_out)
            temporal = temporal.sum(dim=1)  # 聚合到2D
        else:
            temporal = self.temporal_attn(lstm_out)
        
        # Feature attention over feature dimensions
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
    train_mse_history  = []
    train_mae_history    = []

    val_loss_history     = []
    val_rmse_history     = []
    val_mape_history     = []
    val_r2_history       = []
    val_mse_history    = []
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
    plt.title(f"'{target_col}' Original Distribution")
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
    plt.ylabel('E_grid')
    plt.title('E_grid Over Time (Full Dataset)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_comparison(y_actual_real, predictions_dict, colors=None, timestamps=None):

    plt.figure(figsize = (14, 6))
    x_axis = np.arange(len(y_actual_real))


    plt.plot(x_axis, y_actual_real, '#3A3B98', label='Actual', linewidth=2, alpha=0.8)

    colors = ['#E6B422', '#4CAF50', '#E85D75', '#17A2B8', '#5D8AA8']
    for (model_name, pred_values), color in zip(predictions_dict.items(), colors):
        plt.plot(x_axis, pred_values, color=color, label=model_name, linewidth=1.5, linestyle='--', alpha=0.9)

    plt.xlabel('Timestamp')
    plt.ylabel('E_grid Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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
      title: Title for the plot
    """
    plt.figure(figsize = (10, 4))
    plt.hist(pd.to_datetime(timestamps), bins = 50, color = 'skyblue', edgecolor = 'black')
    plt.title(f'{title} - Time Distribution')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.grid(axis = 'y')
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
    print("[Info] 1) Loading raw data...")
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

    model21 = EModel_FeatureWeight21(
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
        lstm_dropout      = 0.1
    ).to(device)

    model5 = EModel_FeatureWeight5(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 3,
        lstm_dropout      = 0.1
    ).to(device)

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

    print("\n========== Training Model: 21 ==========")
    hist21 = train_model(
        model         = model21,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight21',
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

    best_model21 = EModel_FeatureWeight21(
        feature_dim       = feature_dim,
        lstm_hidden_size  = 256, 
        lstm_num_layers   = 2
    ).to(device)
    best_model21.load_state_dict(torch.load('best_EModel_FeatureWeight21.pth', map_location=device, weights_only=True), strict=False)

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
        lstm_num_layers   = 3
    ).to(device)
    best_model5.load_state_dict(torch.load('best_EModel_FeatureWeight5.pth', map_location=device, weights_only=True), strict=False)

    # Evaluate on test set (standardized domain)
    criterion_test = nn.SmoothL1Loss(beta = 1.0)
    (_, test_rmse1_std, test_mape1_std, test_r21_std, test_mse1_std, test_mae1_std, preds1_std, labels1_std) = evaluate(best_model1, test_loader, criterion_test)
    (_, test_rmse2_std, test_mape2_std, test_r22_std, test_mse2_std, test_mae2_std, preds2_std, labels2_std) = evaluate(best_model2, test_loader, criterion_test)
    (_, test_rmse21_std, test_mape21_std, test_r221_std, test_mse21_std, test_mae21_std, preds21_std, labels21_std) = evaluate(best_model21, test_loader, criterion_test)
    (_, test_rmse3_std, test_mape3_std, test_r23_std, test_mse3_std, test_mae3_std, preds3_std, labels3_std) = evaluate(best_model3, test_loader, criterion_test)
    (_, test_rmse4_std, test_mape4_std, test_r24_std, test_mse4_std, test_mae4_std, preds4_std, labels4_std) = evaluate(best_model4, test_loader, criterion_test)
    (_, test_rmse5_std, test_mape5_std, test_r25_std, test_mse5_std, test_mae5_std, preds5_std, labels5_std) = evaluate(best_model5, test_loader, criterion_test)

    print("\n========== [Test Set Evaluation (Standardized Domain)] ==========")
    print(f"[EModel_FeatureWeight1]  RMSE: {test_rmse1_std:.4f}, MAPE: {test_mape1_std:.2f}%, R²: {test_r21_std:.4f}, mse: {test_mse1_std:.2f}%, MAE: {test_mae1_std:.4f}")
    print(f"[EModel_FeatureWeight2]  RMSE: {test_rmse2_std:.4f}, MAPE: {test_mape2_std:.2f}%, R²: {test_r22_std:.4f}, mse: {test_mse2_std:.2f}%, MAE: {test_mae2_std:.4f}")
    print(f"[EModel_FeatureWeight21]  RMSE: {test_rmse21_std:.4f}, MAPE: {test_mape21_std:.2f}%, R²: {test_r221_std:.4f}, mse: {test_mse21_std:.2f}%, MAE: {test_mae21_std:.4f}")
    print(f"[EModel_FeatureWeight3]  RMSE: {test_rmse3_std:.4f}, MAPE: {test_mape3_std:.2f}%, R²: {test_r23_std:.4f}, mse: {test_mse3_std:.2f}%, MAE: {test_mae3_std:.4f}")
    print(f"[EModel_FeatureWeight4]  RMSE: {test_rmse4_std:.4f}, MAPE: {test_mape4_std:.2f}%, R²: {test_r24_std:.4f}, mse: {test_mse4_std:.2f}%, MAE: {test_mae4_std:.4f}")
    print(f"[EModel_FeatureWeight5]  RMSE: {test_rmse5_std:.4f}, MAPE: {test_mape5_std:.2f}%, R²: {test_r25_std:.4f}, mse: {test_mse5_std:.2f}%, MAE: {test_mae5_std:.4f}")

    # Inverse standardization and (optionally) inverse logarithmic transformation
    preds1_real_std = scaler_y.inverse_transform(preds1_std.reshape(-1, 1)).ravel()
    preds2_real_std = scaler_y.inverse_transform(preds2_std.reshape(-1, 1)).ravel()
    preds21_real_std = scaler_y.inverse_transform(preds21_std.reshape(-1, 1)).ravel()
    preds3_real_std = scaler_y.inverse_transform(preds3_std.reshape(-1, 1)).ravel()
    preds4_real_std = scaler_y.inverse_transform(preds4_std.reshape(-1, 1)).ravel()
    preds5_real_std = scaler_y.inverse_transform(preds5_std.reshape(-1, 1)).ravel()
    labels1_real_std = scaler_y.inverse_transform(labels1_std.reshape(-1, 1)).ravel()
    labels2_real_std = scaler_y.inverse_transform(labels2_std.reshape(-1, 1)).ravel()
    labels21_real_std = scaler_y.inverse_transform(labels21_std.reshape(-1, 1)).ravel()
    labels3_real_std = scaler_y.inverse_transform(labels3_std.reshape(-1, 1)).ravel()
    labels4_real_std = scaler_y.inverse_transform(labels4_std.reshape(-1, 1)).ravel()
    labels5_real_std = scaler_y.inverse_transform(labels5_std.reshape(-1, 1)).ravel()

    if use_log_transform:
        preds1_real = np.expm1(preds1_real_std)
        preds2_real = np.expm1(preds2_real_std)
        preds21_real = np.expm1(preds21_real_std)
        preds3_real = np.expm1(preds3_real_std)
        preds4_real = np.expm1(preds4_real_std)
        preds5_real = np.expm1(preds5_real_std)
        labels1_real = np.expm1(labels1_real_std)
        labels2_real = np.expm1(labels2_real_std)
        labels21_real = np.expm1(labels21_real_std)
        labels3_real = np.expm1(labels3_real_std)
        labels4_real = np.expm1(labels4_real_std)
        labels5_real = np.expm1(labels5_real_std)
    else:
        preds1_real = preds1_real_std
        preds2_real = preds2_real_std
        preds21_real = preds21_real_std
        preds3_real = preds3_real_std
        preds4_real = preds4_real_std
        preds5_real = preds5_real_std
        labels1_real = labels1_real_std
        labels2_real = labels2_real_std
        labels21_real = labels21_real_std
        labels3_real = labels3_real_std
        labels4_real = labels4_real_std
        labels5_real = labels5_real_std

    # Compute RMSE in original domain
    test_rmse1_real = np.sqrt(mean_squared_error(labels1_real, preds1_real))
    test_rmse2_real = np.sqrt(mean_squared_error(labels2_real, preds2_real))
    test_rmse21_real = np.sqrt(mean_squared_error(labels21_real, preds21_real))
    test_rmse3_real = np.sqrt(mean_squared_error(labels3_real, preds3_real))
    test_rmse4_real = np.sqrt(mean_squared_error(labels4_real, preds4_real))
    test_rmse5_real = np.sqrt(mean_squared_error(labels5_real, preds5_real))

    print("\n========== [Test Set Evaluation (Original Domain)] ==========")
    print(f"[EModel_FeatureWeight1] => RMSE (original): {test_rmse1_real:.2f}")
    print(f"[EModel_FeatureWeight2] => RMSE (original): {test_rmse2_real:.2f}")
    print(f"[EModel_FeatureWeight21] => RMSE (original): {test_rmse21_real:.2f}")
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

    predictions_dict = {
    'Model1': preds1_real,
    'Model2': preds2_real,
    'Model21': preds21_real,
    'Model3': preds3_real,
    'Model4': preds4_real,
    'Model5': preds5_real
    }

    plot_predictions_comparison(
        y_actual_real=labels4_real,
        predictions_dict={'Model4': preds1_real, 'Model5': preds3_real},
        timestamps=train_timestamps  # 训练集对应的时间戳
    )

    plot_predictions_comparison(
        y_actual_real=labels1_real,
        predictions_dict={'Model1': preds1_real, 'Model4': preds4_real},
        timestamps=train_timestamps  # 训练集对应的时间戳
    )
    plot_predictions_comparison(
        y_actual_real=labels2_real,
        predictions_dict={'Model2': preds1_real, 'Model4': preds4_real},
        timestamps=train_timestamps  # 训练集对应的时间戳
    )

    plot_predictions_comparison(
        y_actual_real=labels3_real,
        predictions_dict={'Model3': preds1_real, 'Model4': preds4_real},
        timestamps=train_timestamps  # 训练集对应的时间戳
    )

    # Plot training curves for various metrics
    plot_training_curves_allmetrics(hist1, model_name = 'EModel_FeatureWeight1')
    plot_training_curves_allmetrics(hist2, model_name = 'EModel_FeatureWeight2')
    plot_training_curves_allmetrics(hist21, model_name = 'EModel_FeatureWeight21')
    plot_training_curves_allmetrics(hist3, model_name = 'EModel_FeatureWeight3')
    plot_training_curves_allmetrics(hist4, model_name = 'EModel_FeatureWeight4')
    plot_training_curves_allmetrics(hist5, model_name = 'EModel_FeatureWeight5')

    print("[Info] Processing complete!")

if __name__ == "__main__":
    main(use_log_transform = True, min_egrid_threshold = 1.0)