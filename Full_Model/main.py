from OSS import optimize_storage_size, visualize_optimization_results
from IES import IntegratedEnergySystem
from All_Models_EGrid_Paper import (load_data, feature_engineering, EModel_FeatureWeight4)
from EF import print_feature_info, get_feature_names

# 主函数 
if __name__ == "__main__":
    # 导入必要的库
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    print("开始执行港口综合能源系统优化...")
    
    # 加载您的数据
    print("正在加载数据...")
    data_df = load_data()
    data_df, feature_cols, target_col = feature_engineering(data_df)
    print(f"数据加载完成，特征列数: {len(feature_cols)}")
    
    # 输出特征详细信息
    print("\n=== 数据集特征详细信息 ===")
    print_feature_info(data_df)
    
    print("\n当前数据集的26个特征名称:")
    for i, col in enumerate(feature_cols):
        print(f"{i+1:2d}. {col}")

    # 设置设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 检查预训练模型可用性
    model_path = 'best_EModel_FeatureWeight4.pth'
    
    # 创建预测模型
    print("创建预测模型...")
    if os.path.exists(model_path):
        # 加载模型来获取正确的feature_dim
        pretrained_model_dict = torch.load(model_path, map_location=device)
        # 从预训练模型参数中获取特征维度
        feature_dim_from_model = pretrained_model_dict['feature_importance'].size(0)
        print(f"预训练模型特征维度: {feature_dim_from_model}")
        
        # 保存模型期望的特征权重信息
        feature_importance = pretrained_model_dict['feature_importance'].cpu().numpy()
        model_weights_info = [(i, w) for i, w in enumerate(feature_importance)]
        model_weights_info.sort(key=lambda x: x[1], reverse=True)
        
        print("\n模型期望的22个特征重要性排序:")
        for i, (idx, weight) in enumerate(model_weights_info):
            if idx < len(feature_cols):
                feature_name = feature_cols[idx]
            else:
                feature_name = f"未知特征_{idx}"
            print(f"{i+1:2d}. 特征 {idx:2d} ({feature_name}): {weight:.4f}")
            
        # 对数据集中超出模型期望的特征进行分析
        if len(feature_cols) > feature_dim_from_model:
            print("\n数据集中超出模型期望的特征:")
            for i in range(feature_dim_from_model, len(feature_cols)):
                print(f"额外特征 {i+1}: {feature_cols[i]}")
                
        print("\n=== 特征对比分析 ===")
        print("预训练模型特征数量: {}, 当前数据集特征数量: {}".format(
            feature_dim_from_model, len(feature_cols)
        ))
        
        if feature_dim_from_model == len(feature_cols):
            print("特征数量匹配")
        else:
            print("特征数量不匹配，需要进行特征适配")
            
            # 计算特征重要性的平均值，用于评估特征的重要程度
            avg_importance = np.mean(feature_importance)
            print(f"模型特征重要性平均值: {avg_importance:.4f}")
            
            # 找出重要性高于平均值的特征
            important_features = [(i, w) for i, w in enumerate(feature_importance) if w > avg_importance]
            important_features.sort(key=lambda x: x[1], reverse=True)
            
            print("\n高重要性特征:")
            for i, (idx, weight) in enumerate(important_features):
                if idx < len(feature_cols):
                    feature_name = feature_cols[idx]
                else:
                    feature_name = f"未知特征_{idx}"
                print(f"{i+1:2d}. 特征 {idx:2d} ({feature_name}): {weight:.4f}")
        
        best_model = EModel_FeatureWeight4(
            feature_dim=feature_dim_from_model,  # 使用预训练模型的特征维度
            lstm_hidden_size=256,
            lstm_num_layers=2
        ).to(device)
    else:
        # 如果没有找到预训练模型，使用当前数据的特征维度
        best_model = EModel_FeatureWeight4(
            feature_dim=len(feature_cols),
            lstm_hidden_size=256,
            lstm_num_layers=2
        ).to(device)
        print(f"警告：未找到预训练模型，使用当前数据的特征维度: {len(feature_cols)}")

    # 加载训练好的模型权重
    print("\n加载模型权重...")
    try:
        best_model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型权重加载成功")
    except Exception as e:
        print(f"警告: 无法加载模型权重: {e}")
        print("将使用未训练的模型继续运行，但结果可能不准确")
    
    # 生成示例电价数据
    print("生成电价数据...")
    # 在实际应用中，加载真实的分时电价数据
    timestamps = data_df['timestamp']
    prices = []
    
    for ts in timestamps:
        hour = ts.hour
        weekday = ts.weekday()
        
        # 基础电价 - 分峰平谷时段
        if 8 <= hour < 12 or 14 <= hour < 18:  # 峰时段
            price = 1.2  # 1.2元/kWh
        elif 12 <= hour < 14 or 18 <= hour < 21:  # 平时段
            price = 0.8  # 0.8元/kWh
        else:  # 谷时段
            price = 0.4  # 0.4元/kWh
            
        # 周末折扣
        if weekday >= 5:  # 周六或周日
            price *= 0.9
            
        prices.append(price)
    
    price_df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices
    })
    print(f"生成了 {len(prices)} 条电价数据")
    
    # 运行储能规模优化
    print("正在优化储能系统规模...")
    optimization_results = optimize_storage_size(
        demand_data=data_df,
        renewable_data=data_df,  # 使用同一数据框，因为其中包含可再生能源数据
        price_data=price_df,
        min_capacity=1000,
        max_capacity=5000,
        step=500,
        min_power=200,
        max_power=1000,
        power_step=100
    )
    
    # 可视化优化结果
    print("可视化优化结果...")
    visualize_optimization_results(optimization_results)
    
    # 部署最优系统并模拟运行
    optimal_capacity = optimization_results['best_config']['capacity']
    optimal_power = optimization_results['best_config']['power']
    
    print(f"\n使用最优配置 ({optimal_capacity} kWh, {optimal_power} kW) 进行系统模拟...")
    optimal_system = IntegratedEnergySystem(
        bess_capacity_kwh=optimal_capacity,
        bess_power_kw=optimal_power,
        prediction_model=best_model
    )
    
    # 运行模拟
    print("模拟最优系统运行...")
    simulation_results = optimal_system.simulate_operation(
        historic_data=data_df,
        time_steps=min(24*30, len(data_df)),  # 1个月或全部数据
        price_data=price_df
    )
    
    # 与基准系统比较
    print("模拟基准系统运行...")
    baseline_system = IntegratedEnergySystem(
        bess_capacity_kwh=0,  # 无储能
        bess_power_kw=0,      # 无储能
        prediction_model=best_model
    )
    
    baseline_results = baseline_system.simulate_operation(
        historic_data=data_df,
        time_steps=min(24*30, len(data_df)),
        price_data=price_df
    )
    
    # 计算KPI
    print("计算关键绩效指标...")
    kpis = optimal_system.calculate_kpis(simulation_results, baseline_results)
    
    # 打印KPI
    print("\n系统性能关键指标:")
    print(f"峰值削减: {kpis['peak_reduction']:.2f}%")
    print(f"电网购电减少: {kpis['grid_energy_reduction']:.2f}%")
    print(f"成本节省: {kpis['cost_savings']:.2f}%")
    print(f"自消费率: {kpis['self_consumption_rate']:.2f}%")
    
    # 可视化模拟结果
    print("\n正在生成结果可视化...")
    optimal_system.visualize_results(simulation_results)
    
    print("港口综合能源系统优化分析完成!")