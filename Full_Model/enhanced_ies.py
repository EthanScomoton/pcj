"""
策略感知综合能源系统
=====================
对应调研报告 §四.4.2 滚动时域 MPC —— 将 .pth 预测模型连接到运行调度
- 接受任意 BaseStrategy 子类
- 预测按索引缓存（同一窗口不重复前向）
- 记录经济性 & 环保性所需的逐时指标
"""
import numpy as np
import pandas as pd

from IES import IntegratedEnergySystem
from EF  import get_renewable_forecast


class StrategyAwareIES(IntegratedEnergySystem):

    def __init__(self, *args,
                 strategy=None,
                 conformal_predictor=None,
                 carbon_tracker=None,
                 carbon_intensity_hourly=None,
                 allow_grid_export: bool = False,
                 export_price_ratio: float = 0.4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy        = strategy
        self.conformal       = conformal_predictor
        self.carbon_tracker  = carbon_tracker
        self.carbon_intensity_hourly = (None if carbon_intensity_hourly is None
                                        else np.asarray(carbon_intensity_hourly, dtype=float))
        self.allow_grid_export = allow_grid_export
        self.export_price_ratio = export_price_ratio

    # ------------------------------------------------------------------
    # 预先缓存 time_steps + horizon 个窗口的点预测，避免重复前向传播
    # ------------------------------------------------------------------
    def precompute_predictions(self, historic_data, time_steps, horizon: int = 24):
        n_needed = time_steps + horizon
        preds = np.zeros(n_needed, dtype=float)
        print(f"[StrategyAwareIES] 预计算 {n_needed} 步负荷预测 ...")
        for idx in range(n_needed):
            seq = self._build_window_sequence(historic_data, idx)
            preds[idx] = float(self.predict_demand(seq))
        print("[StrategyAwareIES] 预测缓存完成。")
        return preds

    # ------------------------------------------------------------------
    def simulate_with_strategy(self, historic_data, time_steps,
                               price_data=None,
                               predictions_by_index=None,
                               horizon: int = 24):
        """
        使用 self.strategy 模拟 time_steps 小时的运行。

        Parameters
        ----------
        historic_data : pd.DataFrame   (含 timestamp, E_grid, E_PV, E_wind, 特征列)
        time_steps    : int           仿真小时数
        price_data    : pd.DataFrame  (含 timestamp, price)，可选
        predictions_by_index : ndarray 预先缓存的点预测(可跨策略共享)
        horizon       : int           MPC 滚动预测窗口长度
        """
        assert self.strategy is not None, "必须提供 strategy"

        if predictions_by_index is None:
            predictions_by_index = self.precompute_predictions(
                historic_data, time_steps, horizon)

        dt = 1.0
        records = []

        for t in range(time_steps):
            # ---- 24h 点预测 & 保形上界 ----
            pred_demand = predictions_by_index[t:t + horizon]
            if len(pred_demand) < horizon:
                pred_demand = np.pad(pred_demand, (0, horizon - len(pred_demand)), 'edge')

            pred_upper = (self.conformal.predict_upper(pred_demand)
                          if self.conformal is not None else None)

            # ---- 可再生 24h 预报 ----
            re_fc = get_renewable_forecast(historic_data, t, horizon)
            renewable_gen = re_fc['pv'] + re_fc['wind']

            # ---- 电价 ----
            if price_data is not None:
                prices = price_data.iloc[t:t + horizon]['price'].values
                if len(prices) < horizon:
                    prices = np.pad(prices, (0, horizon - len(prices)), 'edge')
            else:
                prices = np.ones(horizon)

            # ---- 动态碳强度 ----
            ci = None
            if self.carbon_intensity_hourly is not None:
                ci = self.carbon_intensity_hourly[t:t + horizon]
                if len(ci) < horizon:
                    ci = np.pad(ci, (0, horizon - len(ci)), 'edge')

            # ---- 调度决策 ----
            sched = self.strategy.optimize(
                bess=self.bess,
                pred_demand=pred_demand,
                renewable_gen=renewable_gen,
                grid_prices=prices,
                carbon_intensity=ci,
                pred_upper=pred_upper,
            )

            charge0    = float(np.asarray(sched['bess_charge']).flatten()[0])
            discharge0 = float(np.asarray(sched['bess_discharge']).flatten()[0])

            if charge0 >= discharge0:
                actual_power = self.bess.charge(charge0, dt)
                bess_signed  = -actual_power   # 充电记为负
            else:
                actual_power = self.bess.discharge(discharge0, dt)
                bess_signed  =  actual_power   # 放电记为正

            # ---- 实际潮流 ----
            actual_demand    = float(historic_data.iloc[t]['E_grid'])
            actual_renewable = float(renewable_gen[0])
            actual_grid = actual_demand - actual_renewable - bess_signed

            if not self.allow_grid_export:
                actual_grid = max(0.0, actual_grid)

            # ---- 成本 ----
            p0 = float(prices[0])
            if actual_grid > 0:
                cost = actual_grid * p0
            else:
                cost = actual_grid * p0 * self.export_price_ratio

            # ---- CO2 ----
            if self.carbon_intensity_hourly is not None:
                ef_t = float(self.carbon_intensity_hourly[t])
            elif self.carbon_tracker is not None:
                ef_t = self.carbon_tracker.ef
            else:
                ef_t = 0.0
            # grid(kWh) × ef(tCO2/MWh) = g CO2 ; 换算为 kg:  grid*ef
            co2_kg = max(0.0, actual_grid) * ef_t

            records.append({
                'timestamp':            historic_data.iloc[t]['timestamp'],
                'actual_demand':        actual_demand,
                'predicted_demand':     float(pred_demand[0]),
                'predicted_upper':      (float(pred_upper[0]) if pred_upper is not None
                                         else float(pred_demand[0])),
                'renewable_generation': actual_renewable,
                'grid_import':          actual_grid,
                'bess_power':           bess_signed,
                'bess_soc':             self.bess.get_soc(),
                'cost':                 cost,
                'co2_kg':               co2_kg,
                'price':                p0,
                'carbon_intensity':     ef_t,
            })

        return pd.DataFrame(records)
