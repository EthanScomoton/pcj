"""
模型转换脚本 - 将现有预训练模型转换为适应当前数据集特征维度的新模型
"""

from All_Models_EGrid_Paper import (
    load_data, feature_engineering, 
    EModel_FeatureWeight4
)
import torch
import numpy as np
import os

def convert_model_weights(pretrained_path, new_feature_dim, output_path=None):
    """
    将预训练模型权重转换为适应新特征维度的权重
    
    参数:
        pretrained_path: 预训练模型路径
        new_feature_dim: 新的特征维度
        output_path: 输出路径，默认为None（使用默认文件名）
        
    返回:
        转换后的模型
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练模型文件不存在: {pretrained_path}")
        
    # 加载预训练模型状态字典
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    
    # 获取原始特征维度
    orig_feature_dim = pretrained_dict['feature_importance'].size(0)
    print(f"原始特征维度: {orig_feature_dim}, 新特征维度: {new_feature_dim}")
    
    # 创建新模型（具有新的特征维度）
    new_model = EModel_FeatureWeight4(
        feature_dim=new_feature_dim,
        lstm_hidden_size=256,
        lstm_num_layers=2
    )
    
    # 获取新模型的状态字典
    new_model_dict = new_model.state_dict()
    
    # 转换权重
    converted_dict = {}
    for name, param in pretrained_dict.items():
        if 'feature_importance' in name:
            # 调整特征重要性向量
            if new_feature_dim > orig_feature_dim:
                # 扩展特征维度：复制后补充1
                new_param = torch.ones(new_feature_dim, dtype=param.dtype)
                new_param[:orig_feature_dim] = param
                converted_dict[name] = new_param
            else:
                # 缩减特征维度：截取前面的
                converted_dict[name] = param[:new_feature_dim]
        
        elif 'feature_gate.0.weight' in name:
            # 第一个全连接层的权重
            if new_feature_dim > orig_feature_dim:
                # 扩展输入维度
                orig_out_dim = param.size(0)
                new_param = torch.zeros(orig_out_dim, new_feature_dim, dtype=param.dtype)
                new_param[:, :orig_feature_dim] = param
                # 初始化新增加的权重
                if new_feature_dim > orig_feature_dim:
                    nn_init = torch.nn.init.xavier_uniform_
                    nn_init(new_param[:, orig_feature_dim:])
                converted_dict[name] = new_param
            else:
                # 缩减输入维度
                converted_dict[name] = param[:, :new_feature_dim]
        
        elif 'feature_gate.2.weight' in name:
            # 第二个全连接层的权重
            if new_feature_dim > orig_feature_dim:
                # 扩展输出维度
                orig_in_dim = param.size(1)
                new_param = torch.zeros(new_feature_dim, orig_in_dim, dtype=param.dtype)
                new_param[:orig_feature_dim, :] = param
                # 初始化新增加的权重
                if new_feature_dim > orig_feature_dim:
                    nn_init = torch.nn.init.xavier_uniform_
                    nn_init(new_param[orig_feature_dim:, :])
                converted_dict[name] = new_param
            else:
                # 缩减输出维度
                converted_dict[name] = param[:new_feature_dim, :]
        
        elif 'feature_gate.0.bias' in name:
            # 第一层全连接层的偏置
            if new_feature_dim > orig_feature_dim:
                # 保持原有尺寸不变（因为这是输出维度）
                converted_dict[name] = param
            else:
                converted_dict[name] = param
                
        elif 'feature_gate.2.bias' in name:
            # 第二层全连接层的偏置
            if new_feature_dim > orig_feature_dim:
                # 扩展输出维度的偏置
                new_param = torch.zeros(new_feature_dim, dtype=param.dtype)
                new_param[:orig_feature_dim] = param
                converted_dict[name] = new_param
            else:
                # 缩减输出维度的偏置
                converted_dict[name] = param[:new_feature_dim]
                
        elif 'lstm.weight_ih_l0' in name or 'lstm.weight_ih_l0_reverse' in name:
            # LSTM输入权重
            if new_feature_dim > orig_feature_dim:
                # 扩展输入维度
                lstm_out_dim = param.size(0)
                new_param = torch.zeros(lstm_out_dim, new_feature_dim, dtype=param.dtype)
                new_param[:, :orig_feature_dim] = param
                # 初始化新增加的权重
                if new_feature_dim > orig_feature_dim:
                    nn_init = torch.nn.init.xavier_uniform_
                    nn_init(new_param[:, orig_feature_dim:])
                converted_dict[name] = new_param
            else:
                # 缩减输入维度
                converted_dict[name] = param[:, :new_feature_dim]
                
        else:
            # 其他层保持不变
            converted_dict[name] = param
    
    # 加载转换后的权重到新模型
    new_model.load_state_dict(converted_dict)
    
    # 保存转换后的模型
    if output_path is None:
        output_path = f"converted_EModel_FeatureWeight4_{new_feature_dim}.pth"
    
    torch.save(new_model.state_dict(), output_path)
    print(f"转换后的模型已保存到: {output_path}")
    
    return new_model

if __name__ == "__main__":
    # 设置路径
    pretrained_model_path = "best_EModel_FeatureWeight4.pth"
    converted_model_path = "current_EModel_FeatureWeight4.pth"
    
    # 加载当前数据以获取特征维度
    print("加载数据以获取当前特征维度...")
    data_df = load_data()
    data_df, feature_cols, target_col = feature_engineering(data_df)
    current_feature_dim = len(feature_cols)
    print(f"当前数据集特征维度: {current_feature_dim}")
    
    # 尝试转换模型
    try:
        converted_model = convert_model_weights(
            pretrained_path=pretrained_model_path,
            new_feature_dim=current_feature_dim,
            output_path=converted_model_path
        )
        print("模型转换成功！")
    except Exception as e:
        print(f"模型转换失败: {e}")
        
    # 测试加载转换后的模型
    if os.path.exists(converted_model_path):
        try:
            test_model = EModel_FeatureWeight4(
                feature_dim=current_feature_dim,
                lstm_hidden_size=256,
                lstm_num_layers=2
            )
            test_model.load_state_dict(torch.load(converted_model_path))
            print("转换后的模型加载测试成功！")
        except Exception as e:
            print(f"转换后的模型加载测试失败: {e}") 