from IES import IntegratedEnergySystem
from EF  import calculate_economic_metrics
from All_Models_EGrid_Paper import (EModel_FeatureWeight4)
import torch
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def optimize_storage_size(demand_data, 
                        price_data = None, 
                        min_capacity = 500, 
                        max_capacity = 5000, 
                        step = 100,
                        min_power = 200, 
                        max_power = 1000, 
                        power_step = 100):
    """
    基于经济性指标寻找最优储能规模
    
    参数:
        demand_data: 包含需求数据的DataFrame
        price_data: 包含电价数据的DataFrame(可选)
        min_capacity: 考虑的最小储能容量(kWh)
        max_capacity: 考虑的最大储能容量(kWh)
        step: 容量步长(kWh)
        min_power: 考虑的最小功率(kW)
        max_power: 考虑的最大功率(kW)
        power_step: 功率步长(kW)
        
    返回:
        包含优化结果的字典
    """
    results = []
    
    # 创建一个共享的预测模型实例
    feature_cols = [c for c in demand_data.columns if c not in ['timestamp', 'E_grid', 'dayofweek', 'hour', 'month']]
    feature_dim = len(feature_cols)
    print(f"使用特征维度: {feature_dim}")
    
    prediction_model = EModel_FeatureWeight4(
        feature_dim=feature_dim,
        lstm_hidden_size=256,
        lstm_num_layers=2
    ).to(device)
    
    # 加载训练好的模型权重
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'best_EModel_FeatureWeight4.pth')
        if os.path.exists(model_path):
            # 检查模型特征维度是否匹配
            pretrained_dict = torch.load(model_path, map_location=device, weights_only=True)
            model_feature_dim = pretrained_dict['feature_importance'].size(0)
             
            if model_feature_dim == feature_dim:
                # 特征维度匹配，直接加载
                prediction_model.load_state_dict(pretrained_dict)
                print(f"成功加载预训练模型，特征维度: {model_feature_dim}")
            else:
                # 特征维度不匹配，尝试转换模型
                print(f"特征维度不匹配 (模型: {model_feature_dim}, 当前: {feature_dim})，尝试转换模型...")
                from convert_model import convert_model_weights
                 
                # 转换模型权重
                try:
                    converted_model = convert_model_weights(
                        pretrained_path=model_path,
                        new_feature_dim=feature_dim,
                        output_path="current_EModel_FeatureWeight4.pth",
                        feature_cols=feature_cols,
                        data_df=demand_data,
                        target_col='E_grid'
                    )
                     
                    # 使用转换后的模型，确保模型在同一设备上
                    prediction_model = converted_model
                    prediction_model = converted_model.to(device)
                    print("成功加载转换后的模型")
                except Exception as e:
                    print(f"模型转换失败: {e}")
                    print("将使用未训练的模型继续运行")
        else:
             print(f"未找到预训练模型: {model_path}")
             print("将使用未训练的模型继续运行")
    except Exception as e:
        print(f"警告：无法加载预训练模型，使用未训练的模型: {e}")
    
    # 配置模型为评估模式
    prediction_model.eval()
    
    for capacity in range(min_capacity, max_capacity + step, step):
        for power in range(min_power, max_power + power_step, power_step):
            # 创建使用当前容量和功率的综合系统
            system = IntegratedEnergySystem(
                capacity_kwh=capacity,
                bess_power_kw=power,
                prediction_model=prediction_model  # 使用模型实例
            )
            
            # 使用储能系统的基准场景 (无储能)
            # 修改点：将基准系统的容量和功率设为0，代表没有电池
            baseline_system = IntegratedEnergySystem(
                capacity_kwh=0,       # 修改为0，确保基准是没有电池的
                bess_power_kw=0,      # 修改为0
                prediction_model=prediction_model
            )
            
            # 运行模拟
            try:        
                sim_time_steps = min(72, len(demand_data))  # 使用较短的时间段进行优化
                
                baseline_results = baseline_system.simulate_operation(
                    historic_data = demand_data,
                    time_steps = sim_time_steps,
                    price_data = price_data
                )
                
                system_results = system.simulate_operation(
                    historic_data=demand_data, 
                    time_steps=sim_time_steps,
                    price_data=price_data
                )
                
                # 计算关键绩效指标
                kpis = system.calculate_kpis(system_results, baseline_results)
                
                # 计算经济指标
                # 容量 1000元/kWh, 功率 500元/kW
                investment_cost = capacity * 1000 + power * 500
                
                economic_metrics = calculate_economic_metrics(
                    costs=[baseline_results['cost'].sum(), system_results['cost'].sum()],
                    investment_cost=investment_cost
                )
                
                # 存储结果
                results.append({
                    'capacity': capacity,
                    'power': power,
                    'npv': economic_metrics['NPV'],
                    'payback_period': economic_metrics['payback_period'],
                    'irr': economic_metrics['IRR'],
                    'annual_savings': economic_metrics['annual_savings'],
                    'peak_reduction': kpis.get('peak_reduction', 0),
                    'grid_energy_reduction': kpis.get('grid_energy_reduction', 0),
                    'self_consumption_rate': kpis.get('self_consumption_rate', 0)
                })
                
                print(f"完成配置评估: 容量={capacity}kWh, 功率={power}kW, NPV={economic_metrics['NPV']:.2f}")
                
            except Exception as e:
                print(f"配置评估失败: 容量={capacity}kWh, 功率={power}kW, 错误: {e}")
                import traceback
                traceback.print_exc()
                # 添加默认的失败结果，以便优化可以继续
                results.append({
                    'capacity': capacity,
                    'power': power,
                    'npv': float('-inf'),  # 使用负无穷表示失败的配置
                    'payback_period': float('inf'),
                    'irr': None,
                    'annual_savings': 0,
                    'peak_reduction': 0,
                    'grid_energy_reduction': 0,
                    'self_consumption_rate': 0
                })
    
    # 筛选有效结果并寻找净现值最高的配置
    valid_results = [r for r in results if r['npv'] != float('-inf')]
    if valid_results:
        best_config = max(valid_results, key=lambda x: x['npv'])
    else:
        # 如果没有有效结果，使用第一个结果作为最佳配置
        best_config = results[0] if results else {
            'capacity': min_capacity,
            'power': min_power,
            'npv': 0,
            'payback_period': float('inf'),
            'irr': None,
            'annual_savings': 0,
            'peak_reduction': 0,
            'grid_energy_reduction': 0,
            'self_consumption_rate': 0
        }
    
    return {
        'all_results': results,
        'best_config': best_config
    }

def visualize_optimization_results(results):
    """
    可视化优化结果
    
    参数:
        results: 包含优化结果的字典
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import platform
    import numpy as np
    
    # --- 字体设置修正 ---
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system_name == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    # --------------------

    # 转换为DataFrame
    df = pd.DataFrame(results['all_results'])
    
    # 创建数据透视表用于热图
    npv_pivot = df.pivot(index = 'capacity', columns = 'power', values = 'npv')
    payback_pivot = df.pivot(index = 'capacity', columns = 'power', values = 'payback_period')
    
    # --- 处理 inf 值以便绘图 ---
    # 将 inf 替换为 NaN，这样 matplotlib 会将其留白或显示特定颜色，而不是报错或显示空白
    payback_pivot = payback_pivot.replace([np.inf, -np.inf], np.nan)
    # -------------------------

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize = (18, 8))
    
    # NPV热图
    im1 = axes[0].imshow(npv_pivot, cmap='viridis', aspect='auto', origin='lower')
    axes[0].set_title('净现值(NPV)')
    axes[0].set_xlabel('功率(kW)')
    axes[0].set_ylabel('容量(kWh)')
    axes[0].set_xticks(range(len(npv_pivot.columns)))
    axes[0].set_yticks(range(len(npv_pivot.index)))
    axes[0].set_xticklabels(npv_pivot.columns)
    axes[0].set_yticklabels(npv_pivot.index)
    plt.colorbar(im1, ax=axes[0], label='NPV (元)')
    
    # 标记最佳NPV
    best_capacity = results['best_config']['capacity']
    best_power = results['best_config']['power']
    
    # 只有当最佳配置存在于 pivot 表中时才标记 (防止索引错误)
    if best_capacity in npv_pivot.index and best_power in npv_pivot.columns:
        best_idx = (list(npv_pivot.index).index(best_capacity), list(npv_pivot.columns).index(best_power))
        axes[0].plot(best_idx[1], best_idx[0], 'r*', markersize=15)
    
    # 回收期热图
    # 使用 'cool' 颜色映射，并将 NaN 设为灰色
    current_cmap = plt.cm.cool
    current_cmap.set_bad(color='lightgray')
    
    im2 = axes[1].imshow(payback_pivot, cmap=current_cmap, aspect='auto', origin='lower')
    axes[1].set_title('回收期 (灰色表示无法回收)')
    axes[1].set_xlabel('功率(kW)')
    axes[1].set_ylabel('容量(kWh)')
    axes[1].set_xticks(range(len(payback_pivot.columns)))
    axes[1].set_yticks(range(len(payback_pivot.index)))
    axes[1].set_xticklabels(payback_pivot.columns)
    axes[1].set_yticklabels(payback_pivot.index)
    plt.colorbar(im2, ax=axes[1], label='年')
    
    plt.tight_layout()
    plt.show()
    
    # 打印最佳配置的详细信息
    print(f"最佳配置:")
    print(f"容量: {best_capacity} kWh")
    print(f"功率: {best_power} kW")
    print(f"净现值(NPV): {results['best_config']['npv']:.2f} 元")
    
    irr_val = results['best_config']['irr']
    if irr_val is not None:
        print(f"内部收益率(IRR): {irr_val * 100:.2f}%")
    else:
        print(f"内部收益率(IRR): 无解")
        
    print(f"回收期: {results['best_config']['payback_period']:.2f} 年")
    print(f"年节省: {results['best_config']['annual_savings']:.2f} 元")
    print(f"峰值削减: {results['best_config']['peak_reduction']:.2f}%")
    print(f"电网用电减少: {results['best_config']['grid_energy_reduction']:.2f}%")
    print(f"自消费率: {results['best_config']['self_consumption_rate']:.2f}%")