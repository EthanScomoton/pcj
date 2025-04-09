"""
模型转换脚本 - 将现有预训练模型转换为适应当前数据集特征维度的新模型
"""

from All_Models_EGrid_Paper import (
    load_data, feature_engineering, 
    EModel_FeatureWeight4,
    calculate_feature_importance
)
import torch
import numpy as np
import os

def convert_model_weights(pretrained_path, new_feature_dim, output_path=None, feature_cols=None, data_df=None, target_col=None):
    """
    将预训练模型权重转换为适应新特征维度的权重
    
    参数:
        pretrained_path: 预训练模型路径
        new_feature_dim: 新的特征维度
        output_path: 输出路径，默认为None（使用默认文件名）
        feature_cols: 特征列名称列表
        data_df: 数据DataFrame
        target_col: 目标变量列名称
        
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
    
    # 检查特征重要性是否全都相同
    feature_importance = pretrained_dict['feature_importance'].cpu().numpy()
    is_all_same = np.all(feature_importance == feature_importance[0])
    
    if is_all_same and data_df is not None and target_col is not None and feature_cols is not None:
        print(f"\n警告: 所有特征重要性值都相同 ({feature_importance[0]}). 尝试使用Pearson相关系数计算特征重要性...")
        # 计算特征重要性
        new_feature_importance = calculate_feature_importance(data_df, feature_cols, target_col)
        
        # 更新原始特征重要性（用于显示）
        if len(new_feature_importance) == orig_feature_dim:
            feature_importance = new_feature_importance
        elif len(new_feature_importance) > orig_feature_dim:
            feature_importance = new_feature_importance[:orig_feature_dim]
        else:
            # 如果新特征重要性数组长度小于原始维度，需要进行填充
            temp = np.ones(orig_feature_dim)
            temp[:len(new_feature_importance)] = new_feature_importance
            feature_importance = temp
    
    # 显示特征重要性
    if feature_cols is not None:
        print("\n预训练模型的特征重要性 (前10个):")
        sorted_indices = np.argsort(-feature_importance)
        for i, idx in enumerate(sorted_indices[:10]):
            if idx < len(feature_cols):
                feature_name = feature_cols[idx]
            else:
                feature_name = f"未知特征_{idx}"
            print(f"{i+1:2d}. {feature_name}: {feature_importance[idx]:.4f}")
        
        print("\n当前数据集的特征列表:")
        for i, col in enumerate(feature_cols):
            mark = ""
            if i >= orig_feature_dim:
                mark = " (新增)"
            print(f"{i+1:2d}. {col}{mark}")
        
        if new_feature_dim > orig_feature_dim:
            print(f"\n需要添加的特征数量: {new_feature_dim - orig_feature_dim}")
            for i in range(orig_feature_dim, new_feature_dim):
                if i < len(feature_cols):
                    print(f"  - 添加: {feature_cols[i]}")
        elif new_feature_dim < orig_feature_dim:
            print(f"\n需要删除的特征数量: {orig_feature_dim - new_feature_dim}")
            for i in range(new_feature_dim, orig_feature_dim):
                if i < len(feature_cols):
                    print(f"  - 删除: {feature_cols[i]}")
    
    # 创建新模型（具有新的特征维度）
    new_model = EModel_FeatureWeight4(
        feature_dim=new_feature_dim,
        lstm_hidden_size=256,
        lstm_num_layers=2
    )
    
    # 获取新模型的状态字典
    new_model_dict = new_model.state_dict()
    
    # 计算当前数据集的特征重要性（如果需要）
    if new_feature_dim != orig_feature_dim and data_df is not None and target_col is not None and feature_cols is not None:
        print("\n计算新数据集的特征重要性...")
        new_feature_importance = calculate_feature_importance(data_df, feature_cols, target_col)
    
    # 转换权重
    converted_dict = {}
    for name, param in pretrained_dict.items():
        if 'feature_importance' in name:
            # 调整特征重要性向量
            if new_feature_dim > orig_feature_dim:
                # 如果有新计算的特征重要性则使用它
                if 'new_feature_importance' in locals() and len(new_feature_importance) == new_feature_dim:
                    converted_dict[name] = torch.tensor(new_feature_importance, dtype=param.dtype)
                else:
                    # 扩展特征维度：复制后补充1
                    new_param = torch.ones(new_feature_dim, dtype=param.dtype)
                    new_param[:orig_feature_dim] = param
                    converted_dict[name] = new_param
            else:
                # 缩减特征维度：截取前面的
                converted_dict[name] = param[:new_feature_dim]
        
        elif 'feature_gate.0.weight' in name:
            # 第一个全连接层的权重 - 注意中间层维度可能也不同
            orig_out_dim = param.size(0)  # 中间层维度
            orig_in_dim = param.size(1)   # 输入特征维度
            
            # 新模型中间层维度计算（保持与原始模型的中间层倍数关系）
            new_mid_dim = (orig_out_dim * new_feature_dim) // orig_in_dim
            
            # 创建新参数矩阵
            new_param = torch.zeros(new_mid_dim, new_feature_dim, dtype=param.dtype)
            
            # 复制共同部分并缩放
            if orig_in_dim <= new_feature_dim and orig_out_dim <= new_mid_dim:
                # 扩展情况
                new_param[:orig_out_dim, :orig_in_dim] = param
                
                # 初始化新增部分
                nn_init = torch.nn.init.xavier_uniform_
                if orig_out_dim < new_mid_dim:
                    nn_init(new_param[orig_out_dim:, :])
                if orig_in_dim < new_feature_dim:
                    nn_init(new_param[:, orig_in_dim:])
            else:
                # 缩减情况
                new_param = param[:new_mid_dim, :new_feature_dim]
                
            converted_dict[name] = new_param
            
            # 记录中间层维度供后续使用
            new_middle_dim = new_mid_dim
            
        elif 'feature_gate.0.bias' in name:
            # 第一层全连接层的偏置
            orig_dim = param.size(0)
            
            # 计算新的中间层维度（如果未定义，使用与原模型同样的增长率）
            if 'new_middle_dim' not in locals():
                new_middle_dim = (orig_dim * new_feature_dim) // orig_feature_dim
                
            # 创建新偏置
            new_param = torch.zeros(new_middle_dim, dtype=param.dtype)
            
            # 复制共同部分
            min_dim = min(orig_dim, new_middle_dim)
            new_param[:min_dim] = param[:min_dim]
            
            converted_dict[name] = new_param
            
        elif 'feature_gate.2.weight' in name:
            # 第二个全连接层的权重
            orig_out_dim = param.size(0)  # 输出特征维度
            orig_in_dim = param.size(1)   # 中间层维度
            
            # 使用前面计算的中间层维度
            if 'new_middle_dim' not in locals():
                new_middle_dim = (orig_in_dim * new_feature_dim) // orig_feature_dim
                
            # 创建新参数矩阵
            new_param = torch.zeros(new_feature_dim, new_middle_dim, dtype=param.dtype)
            
            # 复制共同部分
            min_out_dim = min(orig_out_dim, new_feature_dim)
            min_in_dim = min(orig_in_dim, new_middle_dim)
            new_param[:min_out_dim, :min_in_dim] = param[:min_out_dim, :min_in_dim]
            
            # 初始化新增部分
            nn_init = torch.nn.init.xavier_uniform_
            if min_out_dim < new_feature_dim:
                nn_init(new_param[min_out_dim:, :])
            if min_in_dim < new_middle_dim:
                nn_init(new_param[:, min_in_dim:])
                
            converted_dict[name] = new_param
            
        elif 'feature_gate.2.bias' in name:
            # 第二层全连接层的偏置
            new_param = torch.zeros(new_feature_dim, dtype=param.dtype)
            min_dim = min(param.size(0), new_feature_dim)
            new_param[:min_dim] = param[:min_dim]
            converted_dict[name] = new_param
                
        elif 'lstm.weight_ih_l0' in name or 'lstm.weight_ih_l0_reverse' in name:
            # LSTM输入权重
            orig_out_dim = param.size(0)  # LSTM隐藏维度*4
            orig_in_dim = param.size(1)   # 输入特征维度
            
            # 创建新参数矩阵
            new_param = torch.zeros(orig_out_dim, new_feature_dim, dtype=param.dtype)
            
            # 复制共同部分
            min_in_dim = min(orig_in_dim, new_feature_dim)
            new_param[:, :min_in_dim] = param[:, :min_in_dim]
            
            # 初始化新增部分
            if min_in_dim < new_feature_dim:
                nn_init = torch.nn.init.xavier_uniform_
                nn_init(new_param[:, min_in_dim:])
                
            converted_dict[name] = new_param
                
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
    
    # 打印当前数据集所有特征
    print("\n当前数据集的所有特征:")
    for i, col in enumerate(feature_cols):
        print(f"{i+1:2d}. {col}")
    
    # 尝试转换模型
    try:
        converted_model = convert_model_weights(
            pretrained_path=pretrained_model_path,
            new_feature_dim=current_feature_dim,
            output_path=converted_model_path,
            feature_cols=feature_cols,
            data_df=data_df,
            target_col=target_col
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
            
            # 输出转换后模型的特征重要性
            print("\n转换后模型的特征重要性:")
            feature_importance = test_model.feature_importance.detach().cpu().numpy()
            
            # 按照重要性降序排序进行显示
            importance_info = [(i, importance) for i, importance in enumerate(feature_importance)]
            importance_info.sort(key=lambda x: x[1], reverse=True)
            
            for i, (idx, importance) in enumerate(importance_info):
                if idx < len(feature_cols):
                    print(f"{i+1:2d}. {feature_cols[idx]}: {importance:.4f}")
                else:
                    print(f"{i+1:2d}. 未知特征_{idx}: {importance:.4f}")
                    
        except Exception as e:
            print(f"转换后的模型加载测试失败: {e}") 