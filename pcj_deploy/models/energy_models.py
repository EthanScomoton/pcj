import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义注意力模块
class Attention(nn.Module):
    """
    注意力模块 - 对输入应用基于注意力的加权聚合
    参数:
      input_dim: 输入维度
      dropout: Dropout概率
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
        参数:
          x: 输入张量，形状为 [batch_size, seq_len, input_dim]
        返回:
          聚合特征，形状为 [batch_size, input_dim]
        """
        attn_weights = self.attention(x)       # 形状: [batch_size, seq_len, 1]
        attn_weights = self.dropout(attn_weights)
        attn_weights = F.softmax(attn_weights, dim = 1)  # 在时间步上进行softmax
        weighted = x * attn_weights            # 元素级加权
        return torch.sum(weighted, dim = 1)    # 聚合为形状 [batch_size, input_dim]

# CNN特征门控机制
class CNN_FeatureGate(nn.Module):
    """
    基于CNN的特征门控机制，用于替代EModel_FeatureWeight2中的简单特征门控
    """
    def __init__(self, feature_dim, seq_len):
        super(CNN_FeatureGate, self).__init__()
        # 第一个卷积层: kernel_size=3, filters=4
        self.conv1 = nn.Conv1d(
            in_channels=feature_dim, 
            out_channels=4, 
            kernel_size=3, 
            padding=1  # 零填充
        )
        
        # 第一个最大池化层
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第二个卷积层: kernel_size=3, filters=8
        self.conv2 = nn.Conv1d(
            in_channels=4, 
            out_channels=8, 
            kernel_size=3, 
            padding=1  # 零填充
        )
        
        # 第二个最大池化层
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积和池化后的展平大小
        self.flattened_size = 8 * (seq_len // 4)
        
        # 具有10个单元的全连接隐藏层
        self.fc1 = nn.Linear(self.flattened_size, 10)
        
        # 输出层，生成特征权重
        self.fc2 = nn.Linear(10, feature_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        参数:
          x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        返回:
          特征权重，形状为 [batch_size, feature_dim]
        """
        # 转置为1D卷积: [batch_size, feature_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        # 第一个卷积 + 激活 + 池化
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        # 第二个卷积 + 激活 + 池化
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 展平输出
        x = x.view(x.size(0), -1)
        
        # 应用全连接层
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        
        return x

# 模型5: 最佳性能模型
class EModel_FeatureWeight5(nn.Module):
    def __init__(self,
                 feature_dim,
                 window_size=20,          # 序列窗口长度
                 lstm_hidden_size=256,
                 lstm_num_layers=2,
                 lstm_dropout=0.2, 
                ):
        super(EModel_FeatureWeight5, self).__init__()

        self.feature_dim = feature_dim

        # 1) CNN-FeatureGate
        self.feature_gate = CNN_FeatureGate(feature_dim, window_size)

        # 2) MLP-Attention
        self.temporal_attn = Attention(input_dim=2 * lstm_hidden_size)

        # 3) 双向 LSTM 及其初始化
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
        noise = 0.1 * torch.randn_like(mu) * torch.exp(0.5 * logvar)
        output = mu + noise
        return output.squeeze(-1)