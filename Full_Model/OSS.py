from IES import IntegratedEnergySystem
from EF  import calculate_economic_metrics
from All_Models_EGrid_Paper import (EModel_FeatureWeight4)

def optimize_storage_size(demand_data, renewable_data, price_data = None, min_capacity = 100, max_capacity = 2000, step = 100,min_power = 50, max_power = 500, power_step = 50):
    """
    基于经济性指标寻找最优储能规模
    
    参数:
        demand_data: 包含需求数据的DataFrame
        renewable_data: 包含可再生发电数据的DataFrame
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
    
    for capacity in range(min_capacity, max_capacity + step, step):
        for power in range(min_power, max_power + power_step, power_step):
            # 创建使用当前容量和功率的综合系统
            system = IntegratedEnergySystem(
                bess_capacity_kwh=capacity,
                bess_power_kw=power,
                prediction_model = EModel_FeatureWeight4  # 使用您的最佳模型
            )
            
            # 模拟不使用储能系统的基准场景
            baseline_system = IntegratedEnergySystem(
                bess_capacity_kwh = 0,  # 无储能
                bess_power_kw = 0,      # 无储能
                prediction_model = EModel_FeatureWeight4
            )
            
            # 运行模拟
            baseline_results = baseline_system.simulate_operation(
                historic_data = demand_data,
                time_steps = min(24*30, len(demand_data)),  # 1个月或全部数据
                price_data = price_data
            )
            
            system_results = system.simulate_operation(
                historic_data = demand_data,
                time_steps = min(24*30, len(demand_data)),  # 1个月或全部数据
                price_data = price_data
            )
            
            # 计算关键绩效指标
            kpis = system.calculate_kpis(system_results, baseline_results)
            
            # 计算经济指标
            # 假设储能成本为2000元/kWh和800元/kW
            investment_cost = capacity * 2000 + power * 800
            
            economic_metrics = calculate_economic_metrics(
                costs=[baseline_results['cost'].sum(), system_results['cost'].sum()],
                investment_cost = investment_cost
            )
            
            # 存储结果
            results.append({
                'capacity': capacity,
                'power': power,
                'npv': economic_metrics['NPV'],
                'payback_period': economic_metrics['payback_period'],
                'irr': economic_metrics['IRR'],
                'annual_savings': economic_metrics['annual_savings'],
                'peak_reduction': kpis['peak_reduction'],
                'grid_energy_reduction': kpis['grid_energy_reduction'],
                'self_consumption_rate': kpis['self_consumption_rate']
            })
    
    # 寻找净现值最高的配置
    best_config = max(results, key = lambda x: x['npv'])
    
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
    
    # 转换为DataFrame
    df = pd.DataFrame(results['all_results'])
    
    # 创建数据透视表用于热图
    npv_pivot = df.pivot(index = 'capacity', column = 'power', values = 'npv')
    payback_pivot = df.pivot(index = 'capacity', columns = 'power', values = 'payback_period')
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize = (18, 8))
    
    # NPV热图
    im1 = axes[0].imshow(npv_pivot, cmap='viridis')
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
    best_idx = (list(npv_pivot.index).index(best_capacity), list(npv_pivot.columns).index(best_power))
    axes[0].plot(best_idx[1], best_idx[0], 'r*', markersize=15)
    
    # 回收期热图
    im2 = axes[1].imshow(payback_pivot, cmap='cool')
    axes[1].set_title('回收期')
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
    print(f"回收期: {results['best_config']['payback_period']:.2f} 年")
    print(f"内部收益率(IRR): {results['best_config']['irr']*100:.2f}%")
    print(f"年节省: {results['best_config']['annual_savings']:.2f} 元")
    print(f"峰值削减: {results['best_config']['peak_reduction']:.2f}%")
    print(f"电网用电减少: {results['best_config']['grid_energy_reduction']:.2f}%")
    print(f"自消费率: {results['best_config']['self_consumption_rate']:.2f}%")