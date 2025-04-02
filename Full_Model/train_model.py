"""
模型训练脚本 - 使用当前数据集特征维度训练新模型
"""

from All_Models_EGrid_Paper import (
    load_data, feature_engineering, 
    EModel_FeatureWeight4,
    create_sequences,
    train_model
)
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    # 设置设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数配置
    window_size = 20       # 序列长度
    batch_size = 32        # 批次大小
    learning_rate = 1e-3   # 学习率
    weight_decay = 1e-5    # 权重衰减
    num_epochs = 30        # 迭代次数
    num_workers = 0        # 数据加载器的工作进程数
    
    # 加载和处理数据
    print("加载数据...")
    data_df = load_data()
    data_df, feature_cols, target_col = feature_engineering(data_df)
    print(f"数据处理完成，特征列数: {len(feature_cols)}")
    
    # 准备特征和目标数据
    X_data = data_df[feature_cols].values
    y_data = data_df[target_col].values
    
    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_data, y_data, test_size=0.2, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, shuffle=False
    )
    
    # 创建序列数据
    print("创建序列数据...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, window_size)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)
    
    # 转换为张量数据集
    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_seq, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(y_test_seq, dtype=torch.float32)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 构建模型
    feature_dim = X_train_seq.shape[-1]
    model = EModel_FeatureWeight4(
        feature_dim=feature_dim,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.2
    ).to(device)
    
    # 训练模型
    print(f"开始训练模型，特征维度: {feature_dim}...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name='EModel_FeatureWeight4',
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs
    )
    
    # 保存模型
    model_save_path = 'best_EModel_FeatureWeight4.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到 {model_save_path}")
    
    # 显示模型信息
    print("\n模型训练完成!")
    print(f"模型特征维度: {feature_dim}")
    print(f"最佳验证损失: {min(history['val_loss_history']):.4f}")
    print(f"最佳验证RMSE: {min(history['val_rmse_history']):.4f}") 