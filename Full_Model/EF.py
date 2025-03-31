import numpy as np

def extract_features(df, index):
    """从数据框中提取特定索引处的特征"""
    # 根据具体数据结构实现
    # 这里假设我们提取当前行的所有特征列
    if index >= len(df):
        return None
    
    # 假设我们的特征是除了timestamp和E_grid之外的所有列
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'E_grid']]
    return df.iloc[index][feature_cols].values

def get_renewable_forecast(df, start_index, horizon):
    """从数据框中获取可再生能源预测"""
    # 根据具体数据结构实现
    pv_forecast = []
    wind_forecast = []
    
    for i in range(horizon):
        if start_index + i < len(df):
            pv_forecast.append(df.iloc[start_index + i].get('E_PV', 0))
            wind_forecast.append(df.iloc[start_index + i].get('E_wind', 0))
        else:
            # 如果超出数据范围，使用最后已知值
            pv_forecast.append(pv_forecast[-1] if pv_forecast else 0)
            wind_forecast.append(wind_forecast[-1] if wind_forecast else 0)
    
    return {'pv': np.array(pv_forecast), 'wind': np.array(wind_forecast)}

def calculate_economic_metrics(costs, investment_cost, discount_rate=0.05, lifetime=10):
    """
    计算储能系统的经济指标
    
    参数:
        costs: 时间序列成本数组
        investment_cost: 初始投资成本
        discount_rate: 年折现率
        lifetime: 系统寿命(年)
        
    返回:
        包含经济指标的字典
    """
    import numpy as np
    
    # 计算年节省
    baseline_cost = costs[0]  # 假设第一个值为基准
    annual_savings = baseline_cost - np.mean(costs[1:])
    
    # 计算净现值(NPV)
    cash_flows = [-investment_cost]  # 初始投资
    for year in range(1, lifetime + 1):
        cash_flows.append(annual_savings)
    
    npv = 0
    for t, cf in enumerate(cash_flows):
        npv += cf / (1 + discount_rate) ** t
    
    # 计算回收期(简单)
    payback_period = investment_cost / annual_savings if annual_savings > 0 else float('inf')
    
    # 计算内部收益率(IRR)
    def irr_function(r, cash_flows):
        return sum([cf / (1 + r) ** t for t, cf in enumerate(cash_flows)])
    
    # 二分查找IRR
    irr = None
    if cash_flows[0] < 0 and sum(cash_flows) > 0:  # 只有在可能有正IRR时
        r_low, r_high = 0.0, 1.0
        while r_high - r_low > 0.0001:
            r_mid = (r_low + r_high) / 2
            npv_mid = irr_function(r_mid, cash_flows)
            if npv_mid > 0:
                r_low = r_mid
            else:
                r_high = r_mid
        irr = r_low
    
    return {
        'NPV': npv,
        'payback_period': payback_period,
        'IRR': irr,
        'annual_savings': annual_savings
    }