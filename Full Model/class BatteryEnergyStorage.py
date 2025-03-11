class BatteryEnergyStorage:
    def __init__(self, capacity_kwh, max_power_kw, initial_soc=0.5, charging_efficiency=0.95, discharging_efficiency=0.95, min_soc=0.1, max_soc=0.9):
        """
        初始化储能系统
        
        参数:
            capacity_kwh: 储能总容量(kWh)
            max_power_kw: 最大充放电功率(kW)
            initial_soc: 初始荷电状态(0-1)
            charging_efficiency: 充电效率(0-1)
            discharging_efficiency: 放电效率(0-1)
            min_soc: 最小允许荷电状态(0-1)
            max_soc: 最大允许荷电状态(0-1)
        """
        self.capacity = capacity_kwh
        self.max_power = max_power_kw
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        
        # 初始状态
        self.soc = initial_soc
        self.energy_stored = initial_soc * capacity_kwh
        
        # 记录历史数据
        self.soc_history = []
        self.power_history = []
        
    def charge(self, power_kw, time_hours=1.0):
        """以给定功率充电指定时间"""
        # 限制功率不超过最大值
        power_kw = min(power_kw, self.max_power)
        
        # 计算要添加的能量(考虑效率)
        energy_to_add = power_kw * time_hours * self.charging_efficiency
        
        # 检查是否超过容量上限
        new_energy = min(self.energy_stored + energy_to_add, self.capacity * self.max_soc)
        actual_energy_added = new_energy - self.energy_stored
        
        # 计算实际使用的功率
        actual_power = actual_energy_added / (time_hours * self.charging_efficiency)
        
        # 更新状态
        self.energy_stored = new_energy
        self.soc = self.energy_stored / self.capacity
        
        # 记录历史
        self.soc_history.append(self.soc)
        self.power_history.append(actual_power)
        
        return actual_power
    
    def discharge(self, power_kw, time_hours=1.0):
        """以给定功率放电指定时间"""
        # 限制功率不超过最大值
        power_kw = min(power_kw, self.max_power)
        
        # 计算要移除的能量(考虑效率)
        energy_to_remove = power_kw * time_hours / self.discharging_efficiency
        
        # 检查是否低于容量下限
        energy_available = self.energy_stored - (self.capacity * self.min_soc)
        actual_energy_removed = min(energy_to_remove, energy_available)
        
        # 计算实际提供的功率
        actual_power = actual_energy_removed * self.discharging_efficiency / time_hours
        
        # 更新状态
        self.energy_stored -= actual_energy_removed
        self.soc = self.energy_stored / self.capacity
        
        # 记录历史
        self.soc_history.append(self.soc)
        self.power_history.append(-actual_power)  # 放电为负值
        
        return actual_power
        
    def get_soc(self):
        """获取当前荷电状态"""
        return self.soc