from OSS import optimize_storage_size, visualize_optimization_results
from IES import IntegratedEnergySystem
from All_Models_EGrid_Paper import (load_data, feature_engineering, EModel_FeatureWeight4, calculate_feature_importance)
from EF import print_feature_info
from convert_model import convert_model_weights
# 主函数 
if __name__ == "__main__":
    
    import warnings
    # 忽略 cvxpy 的精度警告，这是正常的数值计算波动
    warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')
    
    import torch
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import platform
    from sklearn.preprocessing import StandardScaler

    # Set English font globally to avoid display issues
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # Solve minus sign display issue
    
    print("开始执行港口综合能源系统优化...")
    
    # 加载您的数据
    print("正在加载数据...")
    data_df = load_data()

    # --- 数据预处理：重采样为1小时固定间隔 ---
    print("正在对数据进行重采样和规整化（1小时间隔）以匹配物理模拟...")
    # 1. 确保时间戳格式正确
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    
    # 2. 去除重复时间戳
    data_df = data_df.drop_duplicates(subset=['timestamp'])
    
    # 3. 设置索引并重采样
    data_df = data_df.set_index('timestamp')
    
    # 分离数值列和非数值列分别处理
    numeric_cols = data_df.select_dtypes(include=['number']).columns
    
    # 数值列：线性插值
    df_numeric = data_df[numeric_cols].resample('1H').mean().interpolate(method='linear')
    
    # 非数值列（如天气类别）：前向填充
    other_cols = [c for c in data_df.columns if c not in numeric_cols]
    if other_cols:
        df_other = data_df[other_cols].resample('1H').ffill()
        data_df = pd.concat([df_numeric, df_other], axis=1)
    else:
        data_df = df_numeric
        
    # 重置索引
    data_df = data_df.reset_index()
    
    # 4. 处理可能产生的 NaN (尤其是 E_grid)
    if 'E_grid' in data_df.columns:
        data_df['E_grid'] = data_df['E_grid'].fillna(method='ffill').fillna(0)
    
    print(f"数据重采样完成: {len(data_df)}行 (1小时间隔)")
    # ---------------------------------------

    data_df, feature_cols, target_col = feature_engineering(data_df)
    
    actual_feature_names = feature_cols
    print(f"数据加载完成: {len(data_df)}行, {len(actual_feature_names)}个特征列")
    print(f"平均负荷: {data_df['E_grid'].mean():.2f} kW")
    print(f"最大负荷: {data_df['E_grid'].max():.2f} kW")

    # 确保feature_cols与实际特征名称一致
    if len(feature_cols) != len(actual_feature_names) or set(feature_cols) != set(actual_feature_names):
        print(f"警告: feature_cols({len(feature_cols)})与实际特征列({len(actual_feature_names)})不一致，使用实际特征列")
        feature_cols = actual_feature_names
    
    print(f"数据加载完成，特征列数: {len(feature_cols)}")
    
    # --- 准备归一化器 (Scalers) ---
    print("\n准备特征和目标变量归一化器...")
    # 按照训练时的逻辑准备 Scalers
    # 1. 提取特征矩阵和目标向量
    # 注意：这里需要确保使用与训练时相同的逻辑。如果训练时过滤了数据，这里也应该尽量匹配。
    # 假设 main.py 中的 data_df 是全量数据
    X_all = data_df[feature_cols].values
    y_all = data_df[target_col].values
    
    # 2. 划分训练集 (前80%) 用于拟合 Scaler
    train_size = int(0.8 * len(data_df))
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    
    # 3. 对目标变量进行 log1p 变换 (与训练代码一致)
    y_train_log = np.log1p(y_train)
    
    # 4. 拟合 Scalers
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train_log.reshape(-1, 1))
    print("归一化器准备完成")
    # ---------------------------
    
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
    model_path = os.path.join(os.path.dirname(__file__), 'best_EModel_FeatureWeight4.pth')
    
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
    if os.path.exists(model_path)==True:
        try:
            pretrained_model_dict = torch.load(model_path, map_location=device, weights_only=True)
            # 从预训练模型参数中获取特征维度
            feature_dim_from_model = pretrained_model_dict['feature_importance'].size(0)
            print(f"预训练模型特征维度: {feature_dim_from_model}")
            
            # 检查特征维度是否匹配
            if feature_dim_from_model == feature_dim_to_use:
                print("特征维度匹配，直接加载预训练模型权重")
                best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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
                best_model.load_state_dict(torch.load("current_EModel_FeatureWeight4.pth", map_location=device, weights_only=True))
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
        price_data=price_df,
        min_capacity=10000,  # 允许从0开始搜索
        max_capacity=40000,
        step=10000,
        min_power=8000,     # 允许从0开始搜索
        max_power=38000,
        power_step=6000,
        feature_cols=feature_cols,
        scaler_X=scaler_X,
        scaler_y=scaler_y
    )
    
    # 可视化优化结果
    print("可视化优化结果...")
    visualize_optimization_results(optimization_results)
    
    # 部署最优系统并模拟运行
    optimal_capacity = optimization_results['best_config']['capacity']
    optimal_power = optimization_results['best_config']['power']
    
    print(f"\n使用最优配置 ({optimal_capacity} kWh, {optimal_power} kW) 进行系统模拟...")
    optimal_system = IntegratedEnergySystem(
        capacity_kwh=optimal_capacity,
        bess_power_kw=optimal_power,
        prediction_model=best_model,
        feature_cols=feature_cols,
        scaler_X=scaler_X,
        scaler_y=scaler_y
    )
    
    # 运行模拟
    print("模拟最优系统运行...")
    # 使用所有可用数据进行模拟，或者至少3个月的数据
    sim_horizon = min(24*90, len(data_df))
    
    simulation_results = optimal_system.simulate_operation(
        historic_data=data_df,
        time_steps=sim_horizon,
        price_data=price_df
    )
    
    # 与基准系统比较
    print("模拟基准系统运行...")
    baseline_system = IntegratedEnergySystem(
        capacity_kwh=0,  # 无储能
        bess_power_kw=0,      # 无储能
        prediction_model=best_model,
        feature_cols=feature_cols,
        scaler_X=scaler_X,
        scaler_y=scaler_y
    )
    
    baseline_results = baseline_system.simulate_operation(
        historic_data=data_df,
        time_steps=sim_horizon,
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

    # 打印前24小时的详细运行数据进行调试
    print("\n=== 前24小时运行详情调试 ===")
    debug_df = simulation_results.head(24)[['timestamp', 'actual_demand', 'bess_power', 'bess_soc', 'cost']]
    # 添加电价列以便对照
    debug_df['price'] = price_df.head(24)['price'].values
    print(debug_df.to_string())
    
    print("港口综合能源系统优化分析完成!")