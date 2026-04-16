"""
碳排放追踪模块
================
对应调研报告 §四、4.5 碳交易模块；§六、组合2 多目标优化
- 中国区域电网 CO2 排放因子 (tCO2/MWh)
- 动态边际排放强度（峰/谷时段差异化）
- 碳成本 & 环保 KPI 计算
- 支持在目标函数中作为 CVXPY Parameter 嵌入
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# 中国区域电网 CO2 排放因子 (参考 生态环境部 2023 更新数据, 单位: tCO2/MWh)
# ---------------------------------------------------------------
GRID_EMISSION_FACTORS = {
    'north':        0.8843,  # 华北
    'east':         0.7035,  # 华东 (港口多分布在此)
    'south':        0.3869,  # 南方
    'central':      0.5655,  # 华中
    'northeast':    0.6673,  # 东北
    'northwest':    0.6448,  # 西北
    'national_avg': 0.5810,  # 全国平均
}

# 参考：报告 §四.4.5  CEA 2024 已突破 100 元/tCO2, 2025 约 100-120 元
DEFAULT_CARBON_PRICE_CNY_PER_TON = 120.0


class CarbonTracker:
    """
    碳排放追踪器
        grid_emission_factor:  tCO2 / MWh
        carbon_price:          元 / tCO2
    """

    def __init__(self,
                 grid_emission_factor: float = GRID_EMISSION_FACTORS['east'],
                 carbon_price: float = DEFAULT_CARBON_PRICE_CNY_PER_TON,
                 renewable_emission_factor: float = 0.0):
        self.ef = float(grid_emission_factor)
        self.price = float(carbon_price)
        self.re_ef = float(renewable_emission_factor)

    # ----- 单点计算 -----
    def emissions_kg(self, grid_import_kwh: float) -> float:
        """电网购电引发的 CO2 排放 (kg)"""
        return max(0.0, grid_import_kwh) / 1000.0 * self.ef * 1000.0  # kWh→MWh→tCO2→kg

    def carbon_cost_cny(self, grid_import_kwh: float) -> float:
        """碳成本 (元)"""
        return max(0.0, grid_import_kwh) / 1000.0 * self.ef * self.price

    # ----- 序列汇总 -----
    def summary(self, grid_import_kwh_series) -> dict:
        arr = np.asarray(grid_import_kwh_series, dtype=float)
        pos = np.maximum(arr, 0.0)
        total_mwh = pos.sum() / 1000.0
        total_co2_t = total_mwh * self.ef
        return {
            'total_grid_kwh':                   float(arr.sum()),
            'total_grid_import_kwh':            float(pos.sum()),
            'total_emissions_tonCO2':           float(total_co2_t),
            'total_carbon_cost_CNY':            float(total_co2_t * self.price),
            'avg_emission_intensity_tCO2_MWh':  self.ef,
            'equivalent_trees_year':            float(total_co2_t * 45),  # ~每棵树年吸 22 kg CO2
            'equivalent_cars_year':             float(total_co2_t / 4.6), # 平均乘用车 4.6 tCO2/yr
        }


def make_dynamic_carbon_intensity(timestamps,
                                  base_ef: float = GRID_EMISSION_FACTORS['east'],
                                  peak_multiplier: float = 1.25,
                                  valley_multiplier: float = 0.75) -> np.ndarray:
    """
    构造按小时变化的动态边际排放因子 (tCO2/MWh):
    - 峰时段火电占比更高 → 放大
    - 谷时段水电/核电/风光占比更高 → 缩小
    """
    ts = pd.to_datetime(timestamps)
    out = np.empty(len(ts), dtype=float)
    for i, t in enumerate(ts):
        h = t.hour
        if 8 <= h < 12 or 14 <= h < 18:       # 峰
            out[i] = base_ef * peak_multiplier
        elif 0 <= h < 6:                       # 谷
            out[i] = base_ef * valley_multiplier
        else:                                  # 平
            out[i] = base_ef
    return out
