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
        
        # 1. 确保输入数据格式正确
        predicted_demand = np.array(predicted_demand).flatten()
        
        if renewable_gen is None:
            renewable_gen = np.zeros_like(predicted_demand)
        else:
            renewable_gen = np.array(renewable_gen).flatten()
            
        if grid_prices is None:
            grid_prices = np.ones_like(predicted_demand)
        else:
            grid_prices = np.array(grid_prices).flatten()
            
        # 2. 特殊情况处理：如果没有储能容量（基准情况），直接计算结果，无需调用优化器
        if self.bess.capacity <= 1e-3:
            zeros = np.zeros(self.horizon)
            # 电网输入 = 需求 - 可再生能源（因为没有电池帮忙）
            grid_import = predicted_demand - renewable_gen
            
            # 计算总成本
            total_cost = np.sum(grid_import * grid_prices)
            
            return {
                'bess_charge': zeros,
                'bess_discharge': zeros,
                'grid_import': grid_import,
                'soc_profile': np.zeros(self.horizon + 1),
                'total_cost': total_cost,
                'status': 'optimal'  # 伪装成最优状态
            }

        # 3. 正常优化流程
        # 定义优化变量
        bess_charge = cp.Variable(self.horizon, nonneg=True)  # 储能充电功率
        bess_discharge = cp.Variable(self.horizon, nonneg=True)  # 储能放电功率
        grid_import = cp.Variable(self.horizon)               # 既可正也可负
        
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
        
        # NOTE: OPTIMAL_INACCURATE 也算可行
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # print(f"警告: 问题状态为 {problem.status}") # 暂时屏蔽警告，避免刷屏
            zeros = np.zeros(self.horizon)
            return {
                'bess_charge': zeros,
                'bess_discharge': zeros,
                'grid_import': predicted_demand - renewable_gen,
                'soc_profile': np.repeat(self.bess.get_soc(), self.horizon + 1),
                'total_cost': float('inf'),
                'status': problem.status
            }
        
        # 返回优化结果
        return {
            'bess_charge': bess_charge.value,
            'bess_discharge': bess_discharge.value,
            'grid_import': grid_import.value,
            'soc_profile': soc.value,
            'total_cost': problem.value,
            'status': problem.status
        }