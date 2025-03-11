class RenewableEnergyOptimizer:
    def __init__(self, max_pv_curtailment=0.2, max_wind_curtailment=0.3):
        """
        可再生能源优化器 - 用于调整光伏和风电输出
        
        参数:
            max_pv_curtailment: 允许的最大光伏削减比例
            max_wind_curtailment: 允许的最大风电削减比例
        """
        self.max_pv_curtailment = max_pv_curtailment
        self.max_wind_curtailment = max_wind_curtailment
    
    def optimize_renewables(self, pv_forecast, wind_forecast, demand_forecast, 
                           bess_capacity, grid_prices=None):
        """
        优化可再生能源发电和储能利用
        
        参数:
            pv_forecast: 预测的光伏发电量
            wind_forecast: 预测的风电发电量
            demand_forecast: 预测的需求量
            bess_capacity: 储能系统容量
            grid_prices: 电网电价(可选)
            
        返回:
            包含优化结果的字典
        """
        import cvxpy as cp
        import numpy as np
        
        horizon = len(pv_forecast)
        
        # 定义变量
        pv_used = cp.Variable(horizon, nonneg=True)
        wind_used = cp.Variable(horizon, nonneg=True)
        
        # 光伏和风电削减约束
        constraints = [
            pv_used <= pv_forecast,  # 不能超过可用发电量
            pv_used >= pv_forecast * (1 - self.max_pv_curtailment),  # 限制削减比例
            wind_used <= wind_forecast,  # 不能超过可用发电量
            wind_used >= wind_forecast * (1 - self.max_wind_curtailment)  # 限制削减比例
        ]
        
        # 如果有电价数据，则设置目标为最大化经济价值
        if grid_prices is not None:
            objective = cp.Maximize(cp.sum(cp.multiply(pv_used + wind_used, grid_prices)))
        else:
            # 否则最大化可再生能源利用率
            objective = cp.Maximize(cp.sum(pv_used + wind_used))
        
        # 求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return {
            'pv_utilization': pv_used.value,
            'wind_utilization': wind_used.value,
            'total_renewable_energy': np.sum(pv_used.value + wind_used.value),
            'status': problem.status
        }