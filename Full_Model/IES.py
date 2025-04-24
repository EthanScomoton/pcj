from BES import BatteryEnergyStorage
from EO import EnergyOptimizer
from REO import RenewableEnergyOptimizer
from EF import extract_features, get_renewable_forecast
import torch
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
class IntegratedEnergySystem:
    def __init__(self, capacity_kwh, bess_power_kw, prediction_model=None):
        """
        港口综合能源系统
        
        参数:
            bess_capacity_kwh: 储能系统容量(kWh)
            bess_power_kw: 储能系统功率(kW)
            prediction_model: 预训练的需求预测模型(可选)
        """
        # 初始化组件
        self.bess = BatteryEnergyStorage(
            capacity_kwh=capacity_kwh,
            max_power_kw=bess_power_kw
        )
        self.energy_optimizer = EnergyOptimizer(self.bess)
        self.renewable_optimizer = RenewableEnergyOptimizer()
        self.prediction_model = prediction_model
        
        # 检查模型的特征维度
        if prediction_model is not None:
            self.expected_feature_dim = prediction_model.feature_dim
            print(f"模型期望的特征维度: {self.expected_feature_dim}")
        else:
            self.expected_feature_dim = None
        
        # 结果存储
        self.results = {
            'timestamps': [],
            'actual_demand': [],
            'predicted_demand': [],
            'renewable_generation': [], 
            'grid_import': [],
            'bess_power': [],
            'bess_soc': [],
            'costs': []
        }
    
    def adapt_features(self, features):
        """调整特征维度以匹配模型要求"""
        if self.expected_feature_dim is None:
            return features
            
        # 确保features是numpy数组
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
            
        # 获取当前特征维度
        current_dim = features.shape[-1]
        
        # 如果维度匹配，直接返回
        if current_dim == self.expected_feature_dim:
            return features
            
        # 如果当前特征维度大于模型期望维度，截取前expected_feature_dim个特征
        if current_dim > self.expected_feature_dim:
            print(f"特征维度调整: 从{current_dim}截取到{self.expected_feature_dim}")
            if len(features.shape) == 1:
                return features[:self.expected_feature_dim]
            elif len(features.shape) == 2:
                return features[:, :self.expected_feature_dim]
            else:  # 3D tensor
                return features[:, :, :self.expected_feature_dim]
                
        # 如果当前特征维度小于模型期望维度，用零填充
        if current_dim < self.expected_feature_dim:
            print(f"特征维度调整: 从{current_dim}填充到{self.expected_feature_dim}")
            
            if len(features.shape) == 1:
                padded = np.zeros(self.expected_feature_dim, dtype=np.float32)
                padded[:current_dim] = features
            elif len(features.shape) == 2:
                padded = np.zeros((features.shape[0], self.expected_feature_dim), dtype=np.float32)
                padded[:, :current_dim] = features
            else:  # 3D tensor
                padded = np.zeros((features.shape[0], features.shape[1], self.expected_feature_dim), dtype=np.float32)
                padded[:, :, :current_dim] = features
                
            return padded
            
        # 不应该到达这里
        return features

    def predict_demand(self, features):
        """使用预测模型预测能源需求"""
        if self.prediction_model is None:
            raise ValueError("未提供预测模型")
    
        # 将 features 转换为 PyTorch 支持的浮点类型
        if isinstance(features, np.ndarray) and features.dtype == np.dtype('O'):
            try:
                # 方法1: 尝试直接转换
                features = features.astype(np.float32)
            except:
                try:
                    # 方法2: 如果是混合数组，尝试堆叠并转换
                    features = np.vstack(features).astype(np.float32)
                except:
                    # 方法3: 通过列表中转转换
                    features = np.array(features.tolist(), dtype=np.float32)
    
        # 调整特征维度
        features = self.adapt_features(features)
    
        with torch.no_grad():
            # 确保输入是三维张量: [batch_size, seq_len, feature_dim]
            # 如果是一维特征向量，添加batch和seq维度
            if len(features.shape) == 1:
                # 将一维特征转换为 [1, 1, feature_dim]
                features = features.reshape(1, 1, -1)
            # 如果是二维特征矩阵但缺少seq维度
            elif len(features.shape) == 2:
                # 将二维特征转换为 [batch_size, 1, feature_dim]
                features = features.reshape(features.shape[0], 1, -1)
            
            # 获取序列长度
            seq_len = features.shape[1]
        
            # 重要修复: 首先确保序列长度匹配window_size (20)
            window_size = 20  # 模型期望的窗口大小
            if seq_len < window_size:
                # 扩展序列到window_size
                if seq_len == 1:
                    # 如果只有一个时间步，简单复制
                    features = np.repeat(features, window_size, axis=1)
                else:
                    # 否则，循环填充到window_size
                    repeats_needed = int(np.ceil(window_size / seq_len))
                    repeated = np.repeat(features, repeats_needed, axis=1)
                    features = repeated[:, :window_size, :]
                print(f"序列长度调整: 从{seq_len}扩展到{window_size}")
                seq_len = window_size  # 更新序列长度
        
            # 然后检查local attention是否需要额外调整
            if hasattr(self.prediction_model, 'use_local_attn') and self.prediction_model.use_local_attn:
                # 尝试从模型中获取window_size
                local_attn_window_size = 5  # 默认值
                if hasattr(self.prediction_model.temporal_attn, 'window_size'):
                    local_attn_window_size = self.prediction_model.temporal_attn.window_size
            
                # 如果序列长度不能被local_attn_window_size整除，添加填充
                if seq_len % local_attn_window_size != 0:
                    # 填充到窗口大小的整数倍
                    padding_len = local_attn_window_size - (seq_len % local_attn_window_size)
                    padding = np.repeat(features[:, -1:, :], padding_len, axis=1)
                    features = np.concatenate([features, padding], axis=1)
                    print(f"Local attention序列长度调整: 从{seq_len}填充到{features.shape[1]}")
                
            inputs = torch.tensor(features, dtype=torch.float32).to(device)
            outputs = self.prediction_model(inputs)
    
        return outputs.cpu().numpy()
    
    def simulate_operation(self, historic_data, time_steps, price_data=None):
        """
        模拟系统运行一段时间
        
        参数:
            historic_data: 包含历史数据的DataFrame
            time_steps: 要模拟的时间步数
            price_data: 包含电价数据的DataFrame(可选)
            
        返回:
            包含模拟结果的DataFrame
        """
        # 初始化结果存储
        results = []
        
        # 时间步长(小时)
        time_step_hours = 1.0  # 假设每小时数据
        
        for t in range(time_steps):
            # 提取当前时间步的特征
            current_features = extract_features(historic_data, t)
            
            # 预测未来24小时需求
            predicted_demand = []
            for h in range(24):
                future_features = extract_features(historic_data, t + h)
                pred = self.predict_demand(future_features)
                predicted_demand.append(pred)
                
            predicted_demand = np.array(predicted_demand)
            
            # 获取可再生能源预测
            renewable_forecast = get_renewable_forecast(historic_data, t, 24)
            
            # 获取电价(如果有)
            if price_data is not None:
                grid_prices = price_data.iloc[t:t+24]['price'].values
            else:
                grid_prices = None
            
            # 优化可再生能源
            renewable_opt = self.renewable_optimizer.optimize_renewables(
                pv_forecast=renewable_forecast['pv'],
                wind_forecast=renewable_forecast['wind'],
                demand_forecast=predicted_demand,
                bess_capacity=self.bess.capacity,
                grid_prices=grid_prices
            )
            
            # 优化储能系统
            energy_opt = self.energy_optimizer.optimize_schedule(
                predicted_demand=predicted_demand,
                renewable_gen=renewable_opt['pv_utilization'] + renewable_opt['wind_utilization'],
                grid_prices=grid_prices
            )
            
            # 应用第一个时间步的决策
            if energy_opt['bess_charge'][0] > 0:
                # 充电
                actual_power = self.bess.charge(energy_opt['bess_charge'][0], time_step_hours)
            else:
                # 放电
                actual_power = self.bess.discharge(energy_opt['bess_discharge'][0], time_step_hours)
            
            # 记录实际结果
            actual_demand = historic_data.iloc[t]['E_grid']
            actual_renewable = renewable_forecast['pv'][0] + renewable_forecast['wind'][0]
            actual_grid_import = actual_demand - actual_renewable - (
                actual_power if energy_opt['bess_discharge'][0] > 0 else -actual_power
            )
            
            # 计算成本
            if price_data is not None:
                cost = actual_grid_import * price_data.iloc[t]['price']
            else:
                cost = actual_grid_import  # 如果没有价格数据，以能源为代理
                
            # 存储结果
            results.append({
                'timestamp': historic_data.iloc[t]['timestamp'],
                'actual_demand': actual_demand,
                'predicted_demand': predicted_demand[0],
                'renewable_generation': actual_renewable,
                'grid_import': actual_grid_import,
                'bess_power': actual_power if energy_opt['bess_discharge'][0] > 0 else -actual_power,
                'bess_soc': self.bess.get_soc(),
                'cost': cost
            })
        
        return pd.DataFrame(results)

    def visualize_results(self, results_df):
        """
        可视化模拟结果
        
        参数:
            results_df: 包含模拟结果的DataFrame
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # 创建包含子图的图表
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        
        # 将时间戳转换为datetime格式(如果需要)
        if not pd.api.types.is_datetime64_any_dtype(results_df['timestamp']):
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        
        # 图1: 需求、可再生能源和电网输入
        axes[0].plot(results_df['timestamp'], results_df['actual_demand'], 'b-', label='实际需求')
        axes[0].plot(results_df['timestamp'], results_df['predicted_demand'], 'b--', label='预测需求')
        axes[0].plot(results_df['timestamp'], results_df['renewable_generation'], 'g-', label='可再生能源发电')
        axes[0].plot(results_df['timestamp'], results_df['grid_import'], 'r-', label='电网输入')
        axes[0].set_ylabel('功率 (kW)')
        axes[0].set_title('能源平衡')
        axes[0].legend()
        axes[0].grid(True)
        
        # 图2: 储能充放电功率
        axes[1].plot(results_df['timestamp'], results_df['bess_power'], 'k-')
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('功率 (kW)')
        axes[1].set_title('储能功率 (+ 充电, - 放电)')
        axes[1].grid(True)
        
        # 图3: 储能荷电状态
        axes[2].plot(results_df['timestamp'], results_df['bess_soc'] * 100, 'b-')
        axes[2].set_ylabel('荷电状态 (%)')
        axes[2].set_title('储能荷电状态')
        axes[2].set_ylim(0, 100)
        axes[2].grid(True)
        
        # 图4: 成本
        axes[3].plot(results_df['timestamp'], results_df['cost'], 'r-')
        axes[3].set_ylabel('成本')
        axes[3].set_title('能源成本')
        axes[3].grid(True)
        
        # 格式化x轴
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        plt.xlabel('时间')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def calculate_kpis(self, results_df, baseline_df=None):
        """
        计算关键绩效指标
        
        参数:
            results_df: 包含模拟结果的DataFrame
            baseline_df: 包含基准结果的DataFrame(可选)
            
        返回:
            包含关键绩效指标的字典
        """
        # 初始化KPI
        kpis = {}
        
        # 电网总用电量
        kpis['total_grid_energy'] = results_df['grid_import'].sum()
        
        # 可再生能源总发电量
        kpis['total_renewable_energy'] = results_df['renewable_generation'].sum()
        
        # 电网峰值需求
        kpis['peak_grid_demand'] = results_df['grid_import'].max()
        
        # 总成本
        kpis['total_cost'] = results_df['cost'].sum()
        
        # 自消费率
        total_demand = results_df['actual_demand'].sum()
        kpis['self_consumption_rate'] = (kpis['total_renewable_energy'] / total_demand) * 100
        
        # 电网依赖度
        kpis['grid_dependency_ratio'] = (kpis['total_grid_energy'] / total_demand) * 100
        
        # 如果提供了基准数据，计算改进效果
        if baseline_df is not None:
            baseline_grid_energy = baseline_df['grid_import'].sum()
            baseline_peak = baseline_df['grid_import'].max()
            baseline_cost = baseline_df['cost'].sum()
            
            kpis['grid_energy_reduction'] = ((baseline_grid_energy - kpis['total_grid_energy']) / 
                                            baseline_grid_energy) * 100
            kpis['peak_reduction'] = ((baseline_peak - kpis['peak_grid_demand']) / 
                                    baseline_peak) * 100
            kpis['cost_savings'] = ((baseline_cost - kpis['total_cost']) / 
                                   baseline_cost) * 100
            
        return kpis