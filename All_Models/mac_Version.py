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
learning_rate     = 1e-4   # Learning rate
num_epochs        = 150    # Number of training epochs
batch_size        = 128    # Batch size
weight_decay      = 3e-4   # Weight decay
patience          = 10     # Patience for early stopping
num_workers       = 0      # Number of worker threads
window_size       = 20     # Sequence window size
lstm_hidden_size  = 128    # LSTM hidden size
lstm_num_layers   = 2      # Number of LSTM layers

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
    renewable_df = pd.read_csv('/Users/ethan/Desktop/renewable_data10.csv')
    load_df      = pd.read_csv('/Users/ethan/Desktop/load_data10.csv')

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
        'working_hours', 'E_PV', 'E_wind', 'E_storage_discharge',
        'ESCFR', 'ESCFG'
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

class EModel_FeatureWeight(nn.Module):
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
                 lstm_hidden_size = 128, 
                 lstm_num_layers = 2, 
                 lstm_dropout = 0.2,
                 use_local_attn = True,
                 local_attn_window_size = 5
                ):
        super(EModel_FeatureWeight, self).__init__()
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
    [Model 1: LSTM-based Model with Feature Weighting]
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
                 lstm_hidden_size = 128, 
                 lstm_num_layers = 2, 
                 lstm_dropout = 0.1,
                 use_local_attn = False,
                 local_attn_window_size = 5
                ):
        super(EModel_FeatureWeight2, self).__init__()
        self.feature_dim = feature_dim
        self.use_local_attn = use_local_attn  # 保存该标识

        # Feature gating mechanism
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Temporal attention
        if use_local_attn:
            from local_attention.local_attention import LocalAttention
            self.temporal_attn = LocalAttention(
                dim = 2 * lstm_hidden_size,
                window_size = local_attn_window_size,  # 使用正确的参数名
                causal = False
            )
        else:
            self.temporal_attn = Attention(input_dim = 2 * lstm_hidden_size)
        
        # Feature attention layer
        self.feature_attn = nn.Sequential(
            nn.Linear(window_size, 1),
            nn.Sigmoid()
        )
        # Feature projection layer
        self.feature_proj = nn.Linear(2 * lstm_hidden_size, 2 * lstm_hidden_size)
        
        # Learnable feature importance weights
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
        
        self.attention = Attention(input_dim = 2 * lstm_hidden_size)
        
        # Fully connected layer for final prediction
        self.fc = nn.Sequential(
            nn.Linear(4 * lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 输出两个值
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
    - Compute loss and multiple metrics (RMSE, MAPE, R², SMAPE, MAE) on the given dataset.
    Parameters:
      model: Model to evaluate
      dataloader: DataLoader for the dataset
      criterion: Loss function
    Returns:
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

    # Compute SMAPE
    numerator = np.abs(labels_arr - preds_arr)
    denominator = np.abs(labels_arr) + np.abs(preds_arr)
    nonzero_mask_smape = (denominator != 0)
    if np.sum(nonzero_mask_smape) > 0:
        smape_val = 100.0 * 2.0 * np.mean(numerator[nonzero_mask_smape] / denominator[nonzero_mask_smape])
    else:
        smape_val = 0.0

    # Compute MAE
    mae_val = np.mean(np.abs(labels_arr - preds_arr))

    return val_loss, rmse_std, mape_std, r2_std, smape_val, mae_val, preds_arr, labels_arr


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
    train_smape_history  = []
    train_mae_history    = []

    val_loss_history     = []
    val_rmse_history     = []
    val_mape_history     = []
    val_r2_history       = []
    val_smape_history    = []
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
        train_loss_eval, train_rmse_eval, train_mape_eval, train_r2_eval, train_smape_eval, train_mae_eval, _, _ = evaluate(model, train_loader, criterion)
        val_loss_eval, val_rmse_eval, val_mape_eval, val_r2_eval, val_smape_eval, val_mae_eval, _, _ = evaluate(model, val_loader, criterion)

        # Save metric histories
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

        # Print log information
        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"TrainLoss: {train_loss_epoch:.4f}, "
              f"ValLoss: {val_loss_eval:.4f}, "
              f"ValRMSE: {val_rmse_eval:.4f}, "
              f"ValMAPE: {val_mape_eval:.2f}%, "
              f"ValR^2: {val_r2_eval:.4f}, "
              f"ValSMAPE: {val_smape_eval:.2f}%, "
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
        "train_smape":  train_smape_history,
        "train_mae":    train_mae_history,
        "val_loss":     val_loss_history,
        "val_rmse":     val_rmse_history,
        "val_mape":     val_mape_history,
        "val_r2":       val_r2_history,
        "val_smape":    val_smape_history,
        "val_mae":      val_mae_history
    }


# 7. Visualization and Helper Functions
def plot_correlation_heatmap(df, feature_cols, title = "Heat map"):
    """
    [Visualization Module - Correlation Heatmap]
    - Plot the correlation heatmap for the specified feature columns.
    Parameters:
      df: Dataset
      feature_cols: List of feature columns
      title: Plot title (default: "Heat map")
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
    plt.title(title, fontsize = 16)
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

def plot_predictions_comparison(y_actual_real, y_pred_model1_real, y_pred_model2_real, model1_name = 'Model1', model2_name = 'Model2'):
    """
    [Visualization Module - Prediction Comparison]
    - Compare and plot the actual values and the predictions from two models.
    Parameters:
      y_actual_real: Actual values
      y_pred_model1_real: Predictions from model 1
      y_pred_model2_real: Predictions from model 2
      model1_name: Name of model 1 (default: 'Model1')
      model2_name: Name of model 2 (default: 'Model2')
    """
    plt.figure(figsize = (10, 5))
    x_axis = np.arange(len(y_actual_real))
    plt.plot(x_axis, y_actual_real, 'red', label = 'Actual', linewidth = 1)
    plt.plot(x_axis, y_pred_model1_real, 'lightgreen', label = model1_name, linewidth = 1)
    plt.plot(x_axis, y_pred_model2_real, 'skyblue', label = model2_name, linewidth = 1)
    plt.xlabel('Index')
    plt.ylabel('Value (Real Domain)')
    plt.title(f'Actual vs {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_test_predictions_over_time(test_timestamps, y_actual_real, y_pred_real):
    """
    [Visualization Module - Test Set Predictions]
    - Plot actual and predicted values over time for the test set.
    Parameters:
      test_timestamps: Timestamps for the test set
      y_actual_real: Actual values
      y_pred_real: Predicted values
    """
    plt.figure(figsize = (10, 5))
    plt.plot(test_timestamps, y_actual_real, color = 'red', label = 'Actual E_grid', linewidth = 1)
    plt.plot(test_timestamps, y_pred_real, color = 'blue', label = 'Predicted E_grid', linewidth = 1, linestyle = '--')
    plt.xlabel('Timestamp (Test Data)')
    plt.ylabel('E_grid (Real Domain)')
    plt.title('Test Set: Actual vs Predicted E_grid')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_curves_allmetrics(hist_dict, model_name = 'Model'):
    """
    [Visualization Module - Training Curves]
    - Plot training and validation curves for Loss, RMSE, MAPE, R², SMAPE, and MAE.
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

    # SMAPE curve
    plt.subplot(3, 2, 5)
    plt.plot(epochs, hist_dict["train_smape"], 'r-o', label = 'Train SMAPE', markersize = 4)
    plt.plot(epochs, hist_dict["val_smape"], 'b-o', label = 'Val SMAPE', markersize = 4)
    plt.xlabel('Epoch')
    plt.ylabel('SMAPE (%)')
    plt.title('SMAPE')
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
    feature_cols_to_plot = ['season', 'holiday', 'weather', 'temperature', 'working_hours', 'E_grid']
    feature_cols_to_plot = [c for c in feature_cols_to_plot if c in data_df.columns]
    plot_correlation_heatmap(data_df, feature_cols_to_plot, title = "Heat map")

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
    test_timestamps  = timestamps_all[train_size + val_size:]

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
    modelA = EModel_FeatureWeight(
        feature_dim       = feature_dim,
        lstm_hidden_size  = lstm_hidden_size, 
        lstm_num_layers   = lstm_num_layers,
        lstm_dropout      = 0.2
    ).to(device)
    
    modelB = EModel_FeatureWeight2(
        feature_dim       = feature_dim,
        lstm_hidden_size  = lstm_hidden_size, 
        lstm_num_layers   = lstm_num_layers,
        lstm_dropout      = 0.2
    ).to(device)

    # Train Model: EModel_FeatureWeight
    print("\n========== Training Model: A ==========")
    histA = train_model(
        model         = modelA,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )

    # Train Model: EModel_FeatureWeight2
    print("\n========== Training Model: B ==========")
    histB = train_model(
        model         = modelB,
        train_loader  = train_loader,
        val_loader    = val_loader,
        model_name    = 'EModel_FeatureWeight2',
        learning_rate = learning_rate,
        weight_decay  = weight_decay,
        num_epochs    = num_epochs
    )

    # Load best weights
    best_modelA = EModel_FeatureWeight(feature_dim).to(device)
    best_modelA.load_state_dict(torch.load('best_EModel_FeatureWeight.pth'))

    best_modelB = EModel_FeatureWeight2(feature_dim).to(device)
    best_modelB.load_state_dict(torch.load('best_EModel_FeatureWeight2.pth'))

    # Evaluate on test set (standardized domain)
    criterion_test = nn.SmoothL1Loss(beta = 1.0)
    (_, test_rmseA_std, test_mapeA_std, test_r2A_std, test_smapeA_std, test_maeA_std, predsA_std, labelsA_std) = evaluate(best_modelA, test_loader, criterion_test)
    (_, test_rmseB_std, test_mapeB_std, test_r2B_std, test_smapeB_std, test_maeB_std, predsB_std, labelsB_std) = evaluate(best_modelB, test_loader, criterion_test)

    print("\n========== [Test Set Evaluation (Standardized Domain)] ==========")
    print(f"[EModel_FeatureWeight]  RMSE: {test_rmseA_std:.4f}, MAPE: {test_mapeA_std:.2f}%, R²: {test_r2A_std:.4f}, SMAPE: {test_smapeA_std:.2f}%, MAE: {test_maeA_std:.4f}")
    print(f"[EModel_FeatureWeight2] RMSE: {test_rmseB_std:.4f}, MAPE: {test_mapeB_std:.2f}%, R²: {test_r2B_std:.4f}, SMAPE: {test_smapeB_std:.2f}%, MAE: {test_maeB_std:.4f}")

    # Inverse standardization and (optionally) inverse logarithmic transformation
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

    # Compute RMSE in original domain
    test_rmseA_real = np.sqrt(mean_squared_error(labelsA_real, predsA_real))
    test_rmseB_real = np.sqrt(mean_squared_error(labelsB_real, predsB_real))

    print("\n========== [Test Set Evaluation (Original Domain)] ==========")
    print(f"[EModel_FeatureWeight] => RMSE (original): {test_rmseA_real:.2f}")
    print(f"[EModel_CNN] => RMSE (original): {test_rmseB_real:.2f}")

    # Dataset statistics
    print(f"\n[Dataset Statistics] Total samples: {total_samples}")
    print(f"Training set: {train_size} ({train_size / total_samples:.1%})")
    print(f"Validation set: {val_size} ({val_size / total_samples:.1%})")
    print(f"Test set: {test_size} ({test_size / total_samples:.1%})")

    # Plot dataset time distribution
    plot_dataset_distribution(train_timestamps, 'Training Set')
    plot_dataset_distribution(val_timestamps, 'Validation Set')
    plot_dataset_distribution(test_timestamps, 'Test Set')

    # Visualize test predictions (using modelA as an example)
    plot_test_predictions_over_time(test_timestamps[window_size:], labelsA_real, predsA_real)
    plot_predictions_comparison(
        y_actual_real      = labelsA_real,
        y_pred_model1_real = predsA_real,
        y_pred_model2_real = predsB_real,
        model1_name        = 'EModel_FeatureWeight',
        model2_name        = 'EModel_CNN'
    )

    # Plot training curves for various metrics
    plot_training_curves_allmetrics(histA, model_name = 'EModel_FeatureWeight')
    plot_training_curves_allmetrics(histB, model_name = 'EModel_FeatureWeight2')

    print("[Info] Processing complete!")

if __name__ == "__main__":
    main(use_log_transform = True, min_egrid_threshold = 1.0)