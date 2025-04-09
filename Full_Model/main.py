from OSS import optimize_storage_size, visualize_optimization_results
from IES import IntegratedEnergySystem
from All_Models_EGrid_Paper import (load_data, feature_engineering, EModel_FeatureWeight4, calculate_feature_importance)
from EF import print_feature_info, get_feature_names
from convert_model import convert_model_weights
# 主函数 
if __name__ == "__main__":

    import torch
    import pandas as pd
    import os
    
    print("开始执行港口综合能源系统优化...")
    
    # 加载您的数据
    print("正在加载数据...")
    data_df = load_data()
    data_df, feature_cols, target_col = feature_engineering(data_df)
    
    # 修正可能的特征不一致问题
    actual_feature_names = get_feature_names(data_df)
    if len(feature_cols) != len(actual_feature_names):
        print(f"警告: feature_cols长度({len(feature_cols)})与实际特征数量({len(actual_feature_names)})不一致")
        print("使用实际特征名称进行后续处理")
        feature_cols = actual_feature_names
    
    print(f"数据加载完成，特征列数: {len(feature_cols)}")
    
    # 计算特征重要性
    print("\n计算特征重要性...")
    feature_importance = calculate_feature_importance(data_df, feature_cols, target_col)
    
    # 输出特征详细信息
    print("\n=== 数据集特征详细信息 ===")
    print_feature_info(data_df)
    
    print("\n当前数据集的特征名称:")
    for i, col in enumerate(feature_cols):
        print(f"{i+1:2d}. {col}")

    # 设置设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 检查预训练模型可用性
    model_path = 'best_EModel_FeatureWeight4.pth'
    
    # 创建预测模型
    print("创建预测模型...")
    
    # 使用当前特征维度和计算的特征重要性创建模型
    feature_dim_to_use = len(feature_cols)
    best_model = EModel_FeatureWeight4(
        feature_dim=feature_dim_to_use,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        feature_importance=feature_importance  # 使用计算的特征重要性
    ).to(device)
    
    print(f"\n创建模型使用特征维度: {feature_dim_to_use}")
    
    # 如果预训练模型存在，尝试加载权重
    if os.path.exists(model_path):
        try:
            pretrained_model_dict = torch.load(model_path, map_location=device)
            # 从预训练模型参数中获取特征维度
            feature_dim_from_model = pretrained_model_dict['feature_importance'].size(0)
            print(f"预训练模型特征维度: {feature_dim_from_model}")
            
            # 检查特征维度是否匹配
            if feature_dim_from_model == feature_dim_to_use:
                print("特征维度匹配，直接加载预训练模型权重")
                best_model.load_state_dict(torch.load(model_path, map_location=device))
                print("模型权重加载成功")
            else:
                print(f"特征维度不匹配 (预训练: {feature_dim_from_model}, 当前: {feature_dim_to_use}), 尝试转换模型...")
                from convert_model import convert_model_weights
                
                # 转换模型权重
                converted_model = convert_model_weights(
                    pretrained_path=model_path,
                    new_feature_dim=feature_dim_to_use,
                    output_path="current_EModel_FeatureWeight4.pth",
                    feature_cols=feature_cols,
                    data_df=data_df,
                    target_col=target_col
                )
                
                # 加载转换后的模型
                best_model.load_state_dict(torch.load("current_EModel_FeatureWeight4.pth", map_location=device))
                print("转换后的模型加载成功")
        except Exception as e:
            print(f"警告: 无法加载模型权重: {e}")
            print("将使用未训练的模型继续运行，但结果可能不准确")
    else:
        print(f"警告：未找到预训练模型 {model_path}")
    
    # 显示模型的特征重要性
    print("\n模型的特征重要性:")
    model_importance = best_model.feature_importance.detach().cpu().numpy()
    importance_info = [(i, w) for i, w in enumerate(model_importance)]
    importance_info.sort(key=lambda x: x[1], reverse=True)
    
    for i, (idx, weight) in enumerate(importance_info):
        if idx < len(feature_cols):
            feature_name = feature_cols[idx]
        else:
            feature_name = f"未知特征_{idx}"
        print(f"{i+1:2d}. 特征 {idx:2d} ({feature_name}): {weight:.4f}")
    
    # 生成示例电价数据
    print("\n生成电价数据...")
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