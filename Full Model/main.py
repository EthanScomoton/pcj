# 主函数 - 完整示例
if __name__ == "__main__":
    # 导入必要的库
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 加载您的数据
    data_df = load_data()
    data_df, feature_cols, target_col = feature_engineering(data_df)
    
    # 加载您的最佳预测模型(假设model4是您的最佳模型)
    best_model = EModel_FeatureWeight4(
        feature_dim=len(feature_cols),
        lstm_hidden_size=256,
        lstm_num_layers=2
    ).to(device)
    best_model.load_state_dict(torch.load('best_EModel_FeatureWeight4.pth', 
                                           map_location=device))
    
    # 生成示例电价数据
    # 在实际应用中，您应加载真实的分时电价数据
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
    simulation_results = optimal_system.simulate_operation(
        historic_data=data_df,
        time_steps=min(24*30, len(data_df)),  # 1个月或全部数据
        price_data=price_df
    )
    
    # 与基准系统比较
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