"""
能源调度策略库
===============
对应调研报告：
    §三、3.3 不确定性量化        →  RobustMPCStrategy
    §四、4.2 滚动时域 MPC        →  EconomicMPCStrategy / CarbonAwareMPCStrategy
    §四、4.5 碳交易模块           →  CarbonAwareMPCStrategy
    §三、3.4 强化学习替代         →  (预留 RLStrategy 接口)
    §五、5.2 规则型削峰填谷      →  PeakShavingRuleStrategy (对照组)

统一接口：
    strategy.optimize(bess, pred_demand, renewable_gen, grid_prices,
                      carbon_intensity=None, pred_upper=None)
返回：
    {'bess_charge': np.ndarray[H],
     'bess_discharge': np.ndarray[H],
     'grid_import':   np.ndarray[H],
     'status': str}
"""
from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp


# ========================================================================
# 基类
# ========================================================================
class BaseStrategy(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def optimize(self, bess, pred_demand, renewable_gen, grid_prices,
                 carbon_intensity=None, pred_upper=None):
        pass


# ========================================================================
# 0) 无储能基准
# ========================================================================
class BaselineStrategy(BaseStrategy):
    name = "Baseline (No Storage)"
    description = "无储能基准：仅可再生 + 电网"

    def optimize(self, bess, pred_demand, renewable_gen, grid_prices,
                 carbon_intensity=None, pred_upper=None):
        horizon = len(pred_demand)
        return {
            'bess_charge':    np.zeros(horizon),
            'bess_discharge': np.zeros(horizon),
            'grid_import':    np.asarray(pred_demand) - np.asarray(renewable_gen),
            'status':         'optimal',
        }


# ========================================================================
# 参数化 MPC 核心 —— 构建一次 CVXPY 问题，每步仅更新 Parameter 并热启动
#   （调研报告 §四、4.2 关键实现技巧）
# ========================================================================
class _ParametrizedMPC:
    """
    参数化 MPC 核心。相比初版的改进：
      · 加入需量电费项 (peak_charge_weight * max_grid_import) → 激励削峰
      · 提高循环正则权重 5e-3 以抑制无意义微循环
      · 降低终端 SOC 惩罚至 500，让电池更灵活地参与套利
    """

    def __init__(self, bess, horizon: int = 24, time_step: float = 1.0,
                 include_carbon: bool = False,
                 include_peak_charge: bool = True,
                 peak_charge_weight: float = 1.5):
        self.bess = bess
        self.horizon = horizon
        self.dt = time_step
        self.include_carbon = include_carbon
        self.include_peak_charge = include_peak_charge

        # ---- 可更新的 CVXPY Parameters ----
        self.load_p  = cp.Parameter(horizon, value=np.zeros(horizon))
        self.pv_p    = cp.Parameter(horizon, value=np.zeros(horizon))
        self.price_p = cp.Parameter(horizon, value=np.ones(horizon))
        self.soc_init = cp.Parameter(value=bess.get_soc() if bess.capacity > 0 else 0.5)

        self.carbon_p      = cp.Parameter(horizon, value=np.zeros(horizon), nonneg=True) \
                             if include_carbon else None
        self.carbon_weight = cp.Parameter(nonneg=True, value=0.0) \
                             if include_carbon else None

        # ---- 决策变量 ----
        self.Pc = cp.Variable(horizon, nonneg=True)
        self.Pd = cp.Variable(horizon, nonneg=True)
        self.Pg = cp.Variable(horizon, nonneg=True)      # 仅购电，禁止上网
        self.Pre_use = cp.Variable(horizon, nonneg=True) # 可再生使用量（允许少量弃电以避免不可行）
        self.soc = cp.Variable(horizon + 1)
        self.peak = cp.Variable(nonneg=True) if include_peak_charge else None

        cap = max(float(bess.capacity), 1e-6)
        cons = [
            self.soc[0] == self.soc_init,
            self.soc >= bess.min_soc,
            self.soc <= bess.max_soc,
            self.Pc <= bess.max_power,
            self.Pd <= bess.max_power,
            self.Pre_use <= self.pv_p,      # 不能超过预测出力
        ]
        for t in range(horizon):
            cons.append(
                self.soc[t + 1] == self.soc[t] +
                (bess.charging_efficiency * self.Pc[t] * self.dt -
                 self.Pd[t] * self.dt / bess.discharging_efficiency) / cap
            )
            cons.append(
                self.Pg[t] + self.Pre_use[t] + self.Pd[t] - self.Pc[t] == self.load_p[t]
            )

        # ---- 目标函数 ----
        obj_cost = cp.sum(cp.multiply(self.Pg, self.price_p))     # 电量电费
        obj_reg  = 5e-3 * cp.sum(self.Pc + self.Pd)              # 抑制微循环（↑ 从 1e-4）
        obj_soc  = 500 * cp.abs(self.soc[horizon] - self.soc_init)  # 终端 SOC（↓ 从 1e3）
        obj_curt = 1e-3 * cp.sum(self.pv_p - self.Pre_use)       # 轻度鼓励消纳可再生

        objective = obj_cost + obj_reg + obj_soc + obj_curt
        if include_carbon:
            objective += self.carbon_weight * cp.sum(cp.multiply(self.Pg, self.carbon_p))
        if include_peak_charge:
            cons.append(self.Pg <= self.peak)
            # peak_charge_weight ≈ 需量电价/30 (日化)，典型 1.0~2.0 元/kW
            objective += peak_charge_weight * self.peak

        self.problem = cp.Problem(cp.Minimize(objective), cons)

    # --------------------------------------------------------------------
    def solve(self, soc_init, load, pv, price,
              carbon=None, carbon_weight=0.0):
        self.soc_init.value = float(max(self.bess.min_soc, min(self.bess.max_soc, soc_init)))
        self.load_p.value   = np.asarray(load,  dtype=float).flatten()[:self.horizon]
        self.pv_p.value     = np.asarray(pv,    dtype=float).flatten()[:self.horizon]
        self.price_p.value  = np.asarray(price, dtype=float).flatten()[:self.horizon]
        if self.include_carbon:
            self.carbon_p.value      = np.maximum(0.0, np.asarray(carbon, dtype=float)
                                                  .flatten()[:self.horizon])
            self.carbon_weight.value = max(0.0, float(carbon_weight))

        status = None
        for solver in (cp.OSQP, cp.CLARABEL, None):
            try:
                if solver is None:
                    self.problem.solve(warm_start=True, verbose=False)
                else:
                    self.problem.solve(solver=solver, warm_start=True, verbose=False)
                status = self.problem.status
                if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    break
            except Exception:
                continue

        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.Pc.value is None:
            z = np.zeros(self.horizon)
            return {
                'bess_charge':    z,
                'bess_discharge': z,
                'grid_import':    self.load_p.value - self.pv_p.value,
                'soc_profile':    np.repeat(self.soc_init.value, self.horizon + 1),
                'status':         str(status),
            }

        return {
            'bess_charge':    self.Pc.value,
            'bess_discharge': self.Pd.value,
            'grid_import':    self.Pg.value,
            'soc_profile':    self.soc.value,
            'status':         status,
        }


# ========================================================================
# 1) 经济最优 MPC
# ========================================================================
class EconomicMPCStrategy(BaseStrategy):
    name = "Economic MPC"
    description = "参数化 MPC：电费 + 需量电费 联合最小化，含削峰"

    def __init__(self, horizon: int = 24, peak_charge_weight: float = 1.5):
        self.horizon = horizon
        self.peak_charge_weight = peak_charge_weight
        self._mpc = None

    def optimize(self, bess, pred_demand, renewable_gen, grid_prices,
                 carbon_intensity=None, pred_upper=None):
        if bess.capacity <= 1e-3:
            return BaselineStrategy().optimize(bess, pred_demand, renewable_gen, grid_prices)
        if self._mpc is None or self._mpc.bess is not bess:
            self._mpc = _ParametrizedMPC(bess, self.horizon,
                                          include_carbon=False,
                                          include_peak_charge=True,
                                          peak_charge_weight=self.peak_charge_weight)
        return self._mpc.solve(bess.get_soc(), pred_demand, renewable_gen, grid_prices)


# ========================================================================
# 2) 碳感知 MPC —— 电费 + 碳排双目标
# ========================================================================
class CarbonAwareMPCStrategy(BaseStrategy):
    name = "Carbon-Aware MPC"
    description = "成本 + 动态碳排 多目标 MPC + 削峰 (报告 §四.4.5)"

    def __init__(self, horizon: int = 24, carbon_price_cny_per_ton: float = 100.0,
                 peak_charge_weight: float = 1.5, carbon_sensitivity: float = 3.0):
        self.horizon = horizon
        self.carbon_price = float(carbon_price_cny_per_ton)
        self.peak_charge_weight = peak_charge_weight
        self.carbon_sensitivity = float(carbon_sensitivity)
        self._mpc = None

    def optimize(self, bess, pred_demand, renewable_gen, grid_prices,
                 carbon_intensity=None, pred_upper=None):
        if bess.capacity <= 1e-3:
            return BaselineStrategy().optimize(bess, pred_demand, renewable_gen, grid_prices)
        if carbon_intensity is None:
            return EconomicMPCStrategy(self.horizon,
                                       peak_charge_weight=self.peak_charge_weight).optimize(
                bess, pred_demand, renewable_gen, grid_prices)

        if self._mpc is None or self._mpc.bess is not bess:
            self._mpc = _ParametrizedMPC(bess, self.horizon,
                                          include_carbon=True,
                                          include_peak_charge=True,
                                          peak_charge_weight=self.peak_charge_weight)

        # carbon_sensitivity 放大碳成本在目标函数中的权重，提升减排力度
        cw = self.carbon_price / 1000.0 * self.carbon_sensitivity
        return self._mpc.solve(bess.get_soc(), pred_demand, renewable_gen, grid_prices,
                               carbon=carbon_intensity, carbon_weight=cw)


# ========================================================================
# 3) 鲁棒 MPC —— 使用保形预测上界
# ========================================================================
class RobustMPCStrategy(BaseStrategy):
    name = "Robust MPC (Conformal)"
    description = "保形预测上界 + 削峰，应对极端负荷不确定性"

    def __init__(self, horizon: int = 24, conformal_predictor=None,
                 safety_factor: float = 1.0, peak_charge_weight: float = 1.5):
        self.horizon = horizon
        self.conformal = conformal_predictor
        self.safety_factor = float(safety_factor)
        self.peak_charge_weight = peak_charge_weight
        self._mpc = None

    def optimize(self, bess, pred_demand, renewable_gen, grid_prices,
                 carbon_intensity=None, pred_upper=None):
        if bess.capacity <= 1e-3:
            return BaselineStrategy().optimize(bess, pred_demand, renewable_gen, grid_prices)

        pd_arr = np.asarray(pred_demand, dtype=float)
        if pred_upper is None and self.conformal is not None:
            pred_upper = self.conformal.predict_upper(pd_arr)
        if pred_upper is None:
            pred_upper = pd_arr * 1.10

        # 在点预测与上界间线性插值
        robust_demand = pd_arr + self.safety_factor * (np.asarray(pred_upper) - pd_arr)

        if self._mpc is None or self._mpc.bess is not bess:
            self._mpc = _ParametrizedMPC(bess, self.horizon,
                                          include_carbon=False,
                                          include_peak_charge=True,
                                          peak_charge_weight=self.peak_charge_weight)
        return self._mpc.solve(bess.get_soc(), robust_demand, renewable_gen, grid_prices)


# ========================================================================
# 4) 规则型削峰填谷 （对照组）
# ========================================================================
class PeakShavingRuleStrategy(BaseStrategy):
    name = "Rule-Based Peak Shaving"
    description = "规则型：谷电充、峰电放；用于对照 MPC 类策略的提升"

    def __init__(self, horizon: int = 24,
                 peak_ratio: float = 1.10, valley_ratio: float = 0.90):
        self.horizon = horizon
        self.peak_ratio = peak_ratio
        self.valley_ratio = valley_ratio

    def optimize(self, bess, pred_demand, renewable_gen, grid_prices,
                 carbon_intensity=None, pred_upper=None):
        H = min(len(pred_demand), self.horizon)
        pred_demand   = np.asarray(pred_demand, dtype=float)[:H]
        renewable_gen = np.asarray(renewable_gen, dtype=float)[:H]
        grid_prices   = np.asarray(grid_prices,   dtype=float)[:H]

        charge, discharge = np.zeros(H), np.zeros(H)
        if bess.capacity <= 1e-3:
            return {
                'bess_charge':    charge,
                'bess_discharge': discharge,
                'grid_import':    pred_demand - renewable_gen,
                'status':         'optimal',
            }

        soc = bess.get_soc()
        cap = bess.capacity
        avg_price = float(np.mean(grid_prices)) if len(grid_prices) else 1.0

        # 计算净需求 & 峰值目标——充电时不得推高购电峰值
        net = np.maximum(0.0, pred_demand - renewable_gen)
        peak_target = float(np.percentile(net, 75))

        for t in range(H):
            net_t = pred_demand[t] - renewable_gen[t]
            if grid_prices[t] < avg_price * self.valley_ratio and soc < bess.max_soc:
                room = (bess.max_soc - soc) * cap
                c = min(bess.max_power, room * 0.5)
                # 关键修复：限制充电量，使 grid_import = net_t + c 不超过 peak_target
                headroom = max(0.0, peak_target - max(0.0, net_t))
                c = min(c, headroom)
                charge[t] = max(0.0, c)
                soc += charge[t] * bess.charging_efficiency / cap
            elif (grid_prices[t] > avg_price * self.peak_ratio
                  and soc > bess.min_soc and net_t > peak_target):
                # 仅在净需求超过 peak_target 时放电，削峰至目标线
                excess = net_t - peak_target
                avail = (soc - bess.min_soc) * cap
                d = min(bess.max_power, excess, avail * 0.5)
                discharge[t] = max(0.0, d)
                soc -= discharge[t] / bess.discharging_efficiency / cap

        grid = np.maximum(0.0, pred_demand - renewable_gen - discharge + charge)
        return {
            'bess_charge':    charge,
            'bess_discharge': discharge,
            'grid_import':    grid,
            'status':         'optimal',
        }


# ========================================================================
# 策略注册表（方便 main.py 批量运行）
# ========================================================================
def build_default_strategy_suite(conformal_predictor=None,
                                 horizon: int = 24,
                                 carbon_price_cny_per_ton: float = 100.0,
                                 peak_charge_weight: float = 1.5,
                                 carbon_sensitivity: float = 3.0):
    """
    返回可直接使用的默认策略集合 —— 对应报告 §六 提炼的创新组合
    """
    return [
        BaselineStrategy(),
        PeakShavingRuleStrategy(horizon=horizon),
        EconomicMPCStrategy(horizon=horizon,
                            peak_charge_weight=peak_charge_weight),
        CarbonAwareMPCStrategy(horizon=horizon,
                               carbon_price_cny_per_ton=carbon_price_cny_per_ton,
                               peak_charge_weight=peak_charge_weight,
                               carbon_sensitivity=carbon_sensitivity),
        RobustMPCStrategy(horizon=horizon,
                          conformal_predictor=conformal_predictor,
                          safety_factor=1.0,
                          peak_charge_weight=peak_charge_weight),
    ]
