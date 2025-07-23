import torch
import numpy as np
from .energy_models import EModel_FeatureWeight5
from .data_processor import DataProcessor

class EnergyPredictor:
    """
    能源预测器 - 用于加载模型和进行预测
    """
    def __init__(self, model_path, feature_dim=18, window_size=20, device=None):
        """
        初始化预测器
        
        参数:
            model_path: 模型权重文件路径
            feature_dim: 特征维度
            window_size: 序列窗口大小
            device: 运行设备 (None表示自动选择)
        """
        # 设置设备
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        
        # 初始化数据处理器
        self.processor = DataProcessor(window_size=window_size)
        
        # 初始化模型
        self.model = EModel_FeatureWeight5(
            feature_dim=feature_dim,
            window_size=window_size,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            lstm_dropout=0.2
        ).to(self.device)
        
        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型成功加载自: {model_path}")
        except Exception as e:
            print(f"加载模型时出错: {e}")
        
        # 设置为评估模式
        self.model.eval()
    
    def predict(self, data_df, log_transform=True):
        """
        使用加载的模型进行预测
        
        参数:
            data_df: 包含特征的DataFrame
            log_transform: 是否应用对数变换
            
        返回:
            预测结果
        """
        # 特征工程
        processed_df, feature_cols, target_col = self.processor.feature_engineering(data_df)
        
        # 提取特征
        X_data = processed_df[feature_cols].values
        
        # 标准化特征
        X_scaled = self.processor.transform(X_data)
        
        # 创建序列
        X_seq = self.processor.create_sequences(X_scaled)
        
        # 转换为张量
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # 反向转换预测结果
        predictions = self.processor.inverse_transform_target(predictions_scaled, log_transform=log_transform)
        
        return predictions
    
    def fit_processor(self, train_data_df, log_transform=True):
        """
        使用训练数据拟合数据处理器
        
        参数:
            train_data_df: 训练数据DataFrame
            log_transform: 是否应用对数变换
        """
        # 特征工程
        processed_df, feature_cols, target_col = self.processor.feature_engineering(train_data_df)
        
        # 提取特征和目标
        X_data = processed_df[feature_cols].values
        y_data = processed_df[target_col].values
        
        # 拟合处理器
        self.processor.fit_transform(X_data, y_data, log_transform=log_transform)
        
        print("数据处理器已成功拟合")