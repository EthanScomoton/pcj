import numpy as np

def extract_features(df, index):
    """
    从DataFrame中提取特征
    
    参数:
        df: 包含数据的DataFrame
        index: 要提取特征的行索引
    
    返回:
        特征向量
    """
    # 如果索引超出范围，返回最后一行的特征
    if index >= len(df):
        index = len(df) - 1
        
    # 提取所有特征列(除了timestamp和目标变量)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'E_grid']]
    
    # 提取该行的特征
    features = df.iloc[index][feature_cols].values
    
    return features

def get_feature_names(df):
    """
    获取数据集中的特征名称
    
    参数:
        df: 包含数据的DataFrame
    
    返回:
        特征列名称列表
    """
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'E_grid']]
    return feature_cols

def get_feature_info(df):
    """
    获取数据集特征的详细信息
    
    参数:
        df: 包含数据的DataFrame
    
    返回:
        特征信息字典
    """
    feature_cols = get_feature_names(df)
    
    feature_info = {
        'count': len(feature_cols),
        'names': feature_cols,
        'types': [str(df[col].dtype) for col in feature_cols],
        'samples': [df[col].iloc[0] for col in feature_cols if len(df) > 0],
        'non_null_counts': [df[col].count() for col in feature_cols],
        'unique_values': [df[col].nunique() for col in feature_cols]
    }
    
    return feature_info

def print_feature_info(df):
    """
    打印数据集特征的详细信息
    
    参数:
        df: 包含数据的DataFrame
    """
    info = get_feature_info(df)
    
    print(f"特征总数: {info['count']}")
    print("\n特征详细信息:")
    print("=" * 80)
    print(f"{'序号':4s} {'特征名称':25s} {'数据类型':12s} {'非空值数量':12s} {'唯一值数量':12s}")
    print("-" * 80)
    
    for i, (name, dtype, non_null, unique) in enumerate(zip(
        info['names'], info['types'], info['non_null_counts'], info['unique_values']
    )):
        print(f"{i+1:4d} {name:25s} {dtype:12s} {non_null:12d} {unique:12d}")

def get_renewable_forecast(df, start_index, n_steps):
    """
    获取可再生能源预测
    
    参数:
        df: 包含数据的DataFrame
        start_index: 开始索引
        n_steps: 预测步数
    
    返回:
        包含PV和风能预测的字典
    """
    # 处理索引超出范围的情况
    max_index = min(start_index + n_steps, len(df))
    
    # 获取可再生能源数据
    if 'E_PV' in df.columns:
        pv_data = df['E_PV'].iloc[start_index:max_index].values
    else:
        pv_data = np.zeros(n_steps)
        
    if 'E_wind' in df.columns:
        wind_data = df['E_wind'].iloc[start_index:max_index].values
    else:
        wind_data = np.zeros(n_steps)
    
    # 如果预测步数超出数据范围，用最后一个值填充
    if len(pv_data) < n_steps:
        pv_data = np.pad(pv_data, (0, n_steps - len(pv_data)), 'edge')
        
    if len(wind_data) < n_steps:
        wind_data = np.pad(wind_data, (0, n_steps - len(wind_data)), 'edge')
    
    return {
        'pv': pv_data,
        'wind': wind_data
    }

def calculate_economic_metrics(costs, investment_cost, discount_rate=0.05, lifetime=10):
    """
    计算储能系统的经济指标，包括净现值(NPV)、回收期、内部收益率(IRR)等。
    """
    import numpy as np
    # 计算年度成本节省
    baseline_cost = costs[0]          # 基准方案总成本
    system_cost = np.mean(costs[1:])  # 储能方案总成本（取平均值代表典型年）
    annual_savings = baseline_cost - system_cost
    # 构建现金流列表：第0年为-投资，其后每年为annual_savings
    cash_flows = [-investment_cost] + [annual_savings] * lifetime
    # 计算净现值 NPV
    npv = sum(cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cash_flows))
    # 计算简单回收期
    payback_period = investment_cost / annual_savings if annual_savings > 0 else float('inf')
    # 定义内部收益率计算函数
    def irr_function(r, flows):
        return sum(cf / ((1 + r) ** t) for t, cf in enumerate(flows))
    # 计算内部收益率 IRR
    irr = None
    if cash_flows[0] < 0:  # 存在初始投资
        total = sum(cash_flows[1:])  # 总收益（不含初始投资）
        if total > -cash_flows[0]:
            # 项目净收益为正，存在正IRR，用二分查找0~1区间
            r_low, r_high = 0.0, 1.0
            while r_high - r_low > 1e-4:
                r_mid = (r_low + r_high) / 2
                npv_mid = irr_function(r_mid, cash_flows)
                if npv_mid > 0:
                    r_low = r_mid
                else:
                    r_high = r_mid
            irr = r_low
        elif abs(total + cash_flows[0]) < 1e-6:
            # 项目刚好盈亏平衡，IRR = 0
            irr = 0.0
        elif annual_savings > 0:
            # 项目净收益为负但有正向现金流，计算负IRR（在-0.99～0区间查找）
            r_low, r_high = -0.99, 0.0
            while r_high - r_low > 1e-4:
                r_mid = (r_low + r_high) / 2
                npv_mid = irr_function(r_mid, cash_flows)
                if npv_mid > 0:
                    # 折现率偏低（过于负），提高折现率
                    r_low = r_mid
                else:
                    # 折现率偏高，降低折现率
                    r_high = r_mid
            irr = r_high
        else:
            # 无任何正收益，设定IRR为-100%
            irr = -1.0
    # 返回经济指标结果
    return {
        'NPV': npv,
        'payback_period': payback_period,
        'IRR': irr,
        'annual_savings': annual_savings
    }