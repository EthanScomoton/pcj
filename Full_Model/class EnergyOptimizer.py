class EnergyOptimizer:
    def __init__(self, bess, time_step_hours=1.0, planning_horizon=24):
        """
        能源优化控制器
        
        参数:
            bess: 储能系统实例
            time_step_hours: 优化时间步长(小时)
            planning_horizon: 提前规划的时间步数
        """
        self.bess = bess
        self.time_step = time_step_hours
        self.horizon = planning_horizon
        
    def optimize_schedule(self, predicted_demand, renewable_gen=None, grid_prices=None):
        """
        优化充放电计划
        
        参数:
            predicted_demand: 未来周期的预测能源需求数组
            renewable_gen: 预测的可再生能源发电量数组(可选)
            grid_prices: 预测的电网电价数组(可选)
            
        返回:
            包含优化结果的字典
        """
        import cvxpy as cp
        import numpy as np
        
        # 如果未提供可再生能源数据，假设为零
        if renewable_gen is None:
            renewable_gen = np.zeros_like(predicted_demand)
            
        # 如果未提供电价数据，假设为平均值
        if grid_prices is None:
            grid_prices = np.ones_like(predicted_demand)
            
        # 定义优化变量
        bess_charge = cp.Variable(self.horizon, nonneg=True)  # 储能充电功率
        bess_discharge = cp.Variable(self.horizon, nonneg=True)  # 储能放电功率
        grid_import = cp.Variable(self.horizon, nonneg=True)  # 从电网购电功率
        
        # 荷电状态演变
        soc = cp.Variable(self.horizon + 1)  # 包含初始和最终SOC
        
        # 初始SOC约束
        constraints = [soc[0] == self.bess.get_soc()]
        
        # SOC演变约束
        for t in range(self.horizon):
            # 下一时刻SOC = 当前SOC + (充电效率*充电量 - 放电量/放电效率)/容量
            constraints.append(
                soc[t+1] == soc[t] + 
                (self.bess.charging_efficiency * bess_charge[t] * self.time_step - 
                 bess_discharge[t] * self.time_step / self.bess.discharging_efficiency) / 
                self.bess.capacity
            )
            
        # SOC边界约束
        constraints.extend([
            soc >= self.bess.min_soc,
            soc <= self.bess.max_soc
        ])
        
        # 功率约束
        constraints.extend([
            bess_charge <= self.bess.max_power,
            bess_discharge <= self.bess.max_power
        ])
        
        # 功率平衡: 电网输入 + 可再生能源 + 储能放电 - 储能充电 = 需求
        for t in range(self.horizon):
            constraints.append(
                grid_import[t] + renewable_gen[t] + bess_discharge[t] - bess_charge[t] == predicted_demand[t]
            )
            
        # 目标函数: 最小化电网购电成本
        objective = cp.Minimize(cp.sum(cp.multiply(grid_import, grid_prices)))
        
        # 求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            print(f"警告: 问题状态为 {problem.status}")
        
        # 返回优化结果
        return {
            'bess_charge': bess_charge.value,
            'bess_discharge': bess_discharge.value,
            'grid_import': grid_import.value,
            'soc_profile': soc.value,
            'total_cost': problem.value,
            'status': problem.status
        }