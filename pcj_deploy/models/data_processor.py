import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    """
    数据处理类 - 用于特征工程和数据预处理
    """
    def __init__(self, window_size=20):
        """
        初始化数据处理器
        
        参数:
            window_size: 序列窗口大小
        """
        self.window_size = window_size
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.label_encoders = {}
        
    def feature_engineering(self, data_df):
        """
        特征工程 - 应用EWMA平滑和构建时间特征
        
        参数:
            data_df: 包含原始数据的DataFrame
            
        返回:
            处理后的DataFrame，特征列列表，目标列名
        """
        df = data_df.copy()
        
        # 应用EWMA平滑到'E_grid'
        span = 8  # EWMA平滑参数
        if 'E_grid' in df.columns:
            df['E_grid'] = df['E_grid'].ewm(span=span, adjust=False).mean()
        
        # 构建时间特征
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['dayofweek'] = df['timestamp'].dt.dayofweek
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            
            # Sin/Cos变换
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
            df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        # 类别特征编码
        categorical_features = [
            'season', 'holiday', 'weather', 'temperature',
            'working_hours', 'E_wind', 'E_storage_discharge',
            'ESCFR', 'ESCFG', 'v_wind', 'wind_direction', 'E_PV',
            'ship_grade', 'dock_position', 'destination'
        ]
        
        for col in categorical_features:
            if col in df.columns and df[col].dtype == 'object':
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # 定义特征列和目标列
        time_feature_cols = [
            'dayofweek_sin', 'dayofweek_cos',
            'hour_sin', 'hour_cos',
            'month_sin', 'month_cos'
        ]
        
        # 过滤掉不存在的列
        available_categorical = [col for col in categorical_features if col in df.columns]
        available_time_features = [col for col in time_feature_cols if col in df.columns]
        
        feature_columns = available_categorical + available_time_features
        target_column = 'E_grid'
        
        return df, feature_columns, target_column
    
    def create_sequences(self, X_data, y_data=None):
        """
        创建时间序列数据
        
        参数:
            X_data: 特征数据 (numpy数组)
            y_data: 目标数据 (numpy数组)，如果为None则只返回X序列
            
        返回:
            X_arr: 序列化的特征数据
            y_arr: 对应的目标数据 (如果y_data不为None)
        """
        X_list = []
        num_samples = X_data.shape[0]
        
        for i in range(num_samples - self.window_size):
            seq_x = X_data[i:i + self.window_size, :]
            X_list.append(seq_x)
        
        X_arr = np.array(X_list, dtype=np.float32)
        
        if y_data is not None:
            y_list = []
            for i in range(num_samples - self.window_size):
                seq_y = y_data[i + self.window_size]
                y_list.append(seq_y)
            y_arr = np.array(y_list, dtype=np.float32)
            return X_arr, y_arr
        
        return X_arr
    
    def fit_transform(self, X_data, y_data=None, log_transform=True):
        """
        拟合并转换数据
        
        参数:
            X_data: 特征数据
            y_data: 目标数据 (可选)
            log_transform: 是否应用对数变换到目标值
            
        返回:
            转换后的X数据，以及转换后的y数据(如果提供)
        """
        # 拟合并转换特征
        X_scaled = self.feature_scaler.fit_transform(X_data)
        
        if y_data is not None:
            # 应用对数变换(如果需要)
            if log_transform:
                y_data = np.log1p(y_data)
            
            # 拟合并转换目标
            y_scaled = self.target_scaler.fit_transform(y_data.reshape(-1, 1)).ravel()
            return X_scaled, y_scaled
        
        return X_scaled
    
    def transform(self, X_data, y_data=None, log_transform=True):
        """
        转换数据(使用已拟合的缩放器)
        
        参数:
            X_data: 特征数据
            y_data: 目标数据 (可选)
            log_transform: 是否应用对数变换到目标值
            
        返回:
            转换后的X数据，以及转换后的y数据(如果提供)
        """
        # 转换特征
        X_scaled = self.feature_scaler.transform(X_data)
        
        if y_data is not None:
            # 应用对数变换(如果需要)
            if log_transform:
                y_data = np.log1p(y_data)
            
            # 转换目标
            y_scaled = self.target_scaler.transform(y_data.reshape(-1, 1)).ravel()
            return X_scaled, y_scaled
        
        return X_scaled
    
    def inverse_transform_target(self, y_scaled, log_transform=True):
        """
        反向转换目标值到原始尺度
        
        参数:
            y_scaled: 缩放后的目标值
            log_transform: 是否应用了对数变换
            
        返回:
            原始尺度的目标值
        """
        # 反向缩放
        y_unscaled = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        
        # 反向对数变换(如果应用了)
        if log_transform:
            y_unscaled = np.expm1(y_unscaled)
        
        return y_unscaled