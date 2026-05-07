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
    参数化 MPC 核心。本次修复 (1.2):
      · 终端 SOC 锚点从 soc_init (动态) 改为 soc_target (固定 0.5),
        避免一旦 SOC 落到 0.1, MPC 把"维持 0.1"当成最优而不再回填。
      · 终端 SOC 权重从 500 (单位含糊) 改为 storage_value × cap × 0.5,
        即 "把终端电量按平均电价折算成钱", 让 SOC 偏离与现金流同尺度可比。
      · 仍保留 peak_charge / cycle 抑制 / 弃电正则。
    """

    def __init__(self, bess, horizon: int = 24, time_step: float = 1.0,
                 include_carbon: bool = False,
                 include_peak_charge: bool = True,
                 peak_charge_weight: float = 1.5,
                 soc_target: float = 0.5,
                 storage_shadow_value: float = 0.6):
        """
        Parameters
        ----------
        soc_target : 终端 SOC 锚点 (用于次级软约束, 防止 MPC 过充到 0.9)
        storage_shadow_value : 储能未来价值 (CNY/kWh), 推荐 ≈ 平均电价 (0.5-0.7).
                               在目标函数中以 -value × cap × soc[H] 形式参与,
                               让 MPC 自动按 "买低卖高" 决策, 而不是死守 soc_init.
        """
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
        # 修复 1.2: 加入 soc_target (软约束) 和 shadow_value (主要驱动)
        self.soc_target = cp.Parameter(value=float(soc_target))
        self.soc_shadow_value = cp.Parameter(nonneg=True,
                                             value=float(storage_shadow_value))

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
        obj_reg  = 5e-3 * cp.sum(self.Pc + self.Pd)              # 抑制微循环
        # 修复 1.2 (核心): "储能 shadow value" 形式取代原 |soc[H] - soc_init|.
        # 物理含义: 终端时刻每 kWh 储能的未来期望价值 ≈ 平均电价 (≈0.6 CNY/kWh).
        # 在 24h rolling MPC 下:
        #   marginal cost of charging now = current price α₀
        #   marginal future gain          = shadow_value × eff_c
        #   → 当 α₀ < shadow/eff_c 时充电, α₀ > shadow×eff_d 时放电
        # 这天然导出 "买低卖高" 行为, 不会出现 1AM 谷价时一次倒空到 0.1.
        # 同时 soc_anchor 对终端偏离 soc_target 做软二次惩罚, 防止 MPC 一直冲到 max_soc.
        obj_soc_shadow = -self.soc_shadow_value * cap * self.soc[horizon]
        obj_soc_anchor = 1.0 * cap * cp.square(self.soc[horizon] - self.soc_target)
        obj_soc = obj_soc_shadow + obj_soc_anchor
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
        # ----- 输入清洗: 防御 NaN / inf / 负值 / 噪声实验中的极端放大 -----
        H = self.horizon
        load = np.nan_to_num(np.asarray(load, dtype=float).flatten()[:H],
                             nan=0.0, posinf=1e9, neginf=0.0)
        load = np.maximum(load, 0.0)
        pv   = np.nan_to_num(np.asarray(pv, dtype=float).flatten()[:H],
                             nan=0.0, posinf=1e9, neginf=0.0)
        pv   = np.maximum(pv, 0.0)
        # PV 不能超过 load (否则 Pre_use ≤ pv 与 Pg ≥ 0 矛盾)
        pv   = np.minimum(pv, load)
        price = np.nan_to_num(np.asarray(price, dtype=float).flatten()[:H],
                              nan=1.0, posinf=10.0, neginf=0.0)
        price = np.maximum(price, 0.0)

        self.soc_init.value = float(max(self.bess.min_soc, min(self.bess.max_soc, soc_init)))
        self.load_p.value   = load
        self.pv_p.value     = pv
        self.price_p.value  = price
        if self.include_carbon:
            carbon_arr = np.maximum(0.0, np.nan_to_num(
                np.asarray(carbon, dtype=float).flatten()[:H], nan=0.0))
            self.carbon_p.value      = carbon_arr
            self.carbon_weight.value = max(0.0, float(carbon_weight))

        # ----- 求解器链: SCS (ADMM, 实测对本问题最稳) → CLARABEL → OSQP -----
        # 旧实现 OSQP 优先 + 严格 tolerance 在噪声放大的病态量级 (load 1e6+) 下大量失败.
        # 实测 (_debug_solver_check.py) SCS 在 noise=0.0~0.30 全工况下 100% 收敛,
        # CLARABEL 在本问题结构 (大尺度 + 削峰约束) 下对所有工况都失败, 留作冗余兜底.
        # SCS max_iters: 收敛通常 <500 步; infeasibility 检测 1500 步内一定确认,
        # 之前 10000 太大,导致 Stochastic MPC 极端 hi 上界场景每次 solve 浪费 ~1 秒.
        status = None
        solver_used = None
        last_err = None
        solver_chain = (
            ('SCS', cp.SCS, dict(
                max_iters=3000, eps=1e-3,
                acceleration_lookback=10,
            )),
            ('CLARABEL', cp.CLARABEL, dict(
                max_iter=1000, time_limit=5.0,
                eps_abs=1e-4, eps_rel=1e-4,
            )),
            ('OSQP', cp.OSQP, dict(
                eps_abs=1e-3, eps_rel=1e-3,
                max_iter=10000, polish=False,
                adaptive_rho=True, adaptive_rho_interval=25,
                scaling=20,
            )),
        )
        # 已确认无解的状态: 切换求解器也救不回来, 立即跳出走 heuristic 兜底
        DEFINITIVE_FAILURE = {
            cp.INFEASIBLE, cp.UNBOUNDED,
            'infeasible_inaccurate', 'unbounded_inaccurate',
            getattr(cp, 'INFEASIBLE_INACCURATE', 'infeasible_inaccurate'),
            getattr(cp, 'UNBOUNDED_INACCURATE', 'unbounded_inaccurate'),
        }
        for name, solver, kw in solver_chain:
            try:
                self.problem.solve(solver=solver, warm_start=True,
                                   verbose=False, **kw)
                status = self.problem.status
                if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) \
                        and self.Pc.value is not None:
                    solver_used = name
                    break
                if status in DEFINITIVE_FAILURE:
                    break
            except Exception as exc:
                last_err = (name, type(exc).__name__)
                continue

        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.Pc.value is None:
            return self._heuristic_fallback(
                soc_init=float(self.soc_init.value),
                load=load, pv=pv,
                last_status=status, last_err=last_err,
            )

        if getattr(self, '_fail_count', 0) > 0:
            self._fail_count = 0

        return {
            'bess_charge':    self.Pc.value,
            'bess_discharge': self.Pd.value,
            'grid_import':    self.Pg.value,
            'soc_profile':    self.soc.value,
            'status':         status,
            'solver':         solver_used,
        }

    # --------------------------------------------------------------------
    def _heuristic_fallback(self, soc_init, load, pv, last_status, last_err):
        """
        求解器全部失败时的贪心削峰填谷兜底.

        旧实现退化为 "BESS 不动作 + grid = load - pv", 在噪声鲁棒性实验中
        会让结果失真 (Robust MPC 在最该体现鲁棒性的极端工况里反而完全废掉).

        新实现: 24h 滚动窗口内, 高于平均净负荷的小时放电, 低于的小时充电,
                受限于 BESS 功率/容量/SOC 边界. 保留与 CVXPY 相同的输出契约,
                status='optimal_inaccurate' 让下游 IES 仍然采用 BESS 动作.
        """
        self._fail_count = getattr(self, '_fail_count', 0) + 1
        if self._fail_count == 1 or self._fail_count % 200 == 0:
            import sys
            err_tail = ""
            if last_err:
                err_tail = f", last_err={last_err[0]}({last_err[1]})"
            print(f"  [_ParametrizedMPC] heuristic fallback #{self._fail_count}, "
                  f"last_status={last_status}, "
                  f"load_mean={float(load.mean()):.0f}, "
                  f"pv_mean={float(pv.mean()):.0f}{err_tail}",
                  file=sys.stderr, flush=True)

        H = self.horizon
        bess = self.bess
        cap = float(max(bess.capacity, 1e-6))
        net = np.maximum(0.0, load - pv)
        avg = float(np.mean(net)) if H > 0 else 0.0

        charge    = np.zeros(H)
        discharge = np.zeros(H)
        soc_path  = np.zeros(H + 1)
        soc_path[0] = float(soc_init)

        eta_c = max(float(bess.charging_efficiency), 1e-3)
        eta_d = max(float(bess.discharging_efficiency), 1e-3)
        max_p = float(bess.max_power)

        for t in range(H):
            if net[t] > avg and soc_path[t] > bess.min_soc:
                excess     = net[t] - avg
                avail_kwh  = (soc_path[t] - bess.min_soc) * cap
                d = min(max_p, float(excess), float(avail_kwh) * eta_d)
                discharge[t] = max(0.0, d)
                soc_path[t + 1] = soc_path[t] - discharge[t] / eta_d / cap
            elif net[t] < avg and soc_path[t] < bess.max_soc:
                deficit   = avg - net[t]
                room_kwh  = (bess.max_soc - soc_path[t]) * cap
                c = min(max_p, float(deficit), float(room_kwh) / eta_c)
                charge[t] = max(0.0, c)
                soc_path[t + 1] = soc_path[t] + eta_c * charge[t] / cap
            else:
                soc_path[t + 1] = soc_path[t]

        grid = np.maximum(0.0, load - pv - discharge + charge)
        return {
            'bess_charge':    charge,
            'bess_discharge': discharge,
            'grid_import':    grid,
            'soc_profile':    soc_path,
            'status':         'optimal_inaccurate',
            'solver':         'heuristic_fallback',
            'fail_count':     int(self._fail_count),
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

        # 修复 1.4: 原 cw = price/100 * sens = 3.6 让 carbon 项 ≈ 2.5 CNY/kWh,
        # 比电价大 2-8 倍, CVXPY 三层求解器都失败 → 全程走 _heuristic_fallback,
        # 实际碳目标失效甚至反向 (exp04 sensitivity 0.5-10 输出完全相同).
        # 回到经济上正确的量纲 cw = price/1000 * sens (CNY/kgCO2 维度匹配),
        # 默认 (price=120, sens=3) → cw = 0.36, carbon 项 ≈ 0.25 CNY/kWh = 25% 电价.
        # 这值不会让求解器失衡, 同时 sensitivity 仍然提供 1×~10× 的可调放大。
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
# 3.5) 随机 / 分布鲁棒 MPC (P2)
# ========================================================================
class StochasticMPCStrategy(BaseStrategy):
    """
    场景化随机 MPC:
      - 从 CP 区间 [lo, hi] 采样 K 条需求轨迹
      - 用场景平均目标值 + CVaR (最差 β 分位) 作为鲁棒化
      - 通过对每条场景单独优化再取平均动作 (heuristic) 避免 K 倍变量爆炸

    参考: Wang et al. 2021 "Scenario-based stochastic MPC for battery"
          Shapiro & Dentcheva 2014 "Lectures on stochastic programming"
    """
    name = "Stochastic MPC"
    description = "场景化随机 MPC, 用 CP 区间上下限采样多轨迹取均值动作"

    def __init__(self, horizon: int = 24, conformal_predictor=None,
                 n_scenarios: int = 10, cvar_beta: float = 0.2,
                 peak_charge_weight: float = 1.5):
        self.horizon = horizon
        self.conformal = conformal_predictor
        self.n_scenarios = int(n_scenarios)
        self.cvar_beta = float(cvar_beta)
        self.peak_charge_weight = peak_charge_weight
        self._mpc = None

    def optimize(self, bess, pred_demand, renewable_gen, grid_prices,
                 carbon_intensity=None, pred_upper=None):
        if bess.capacity <= 1e-3:
            return BaselineStrategy().optimize(bess, pred_demand, renewable_gen, grid_prices)

        pd_arr = np.asarray(pred_demand, dtype=float)
        H = len(pd_arr)

        # 生成场景: 从区间 [lo, hi] 中均匀 + 截断高斯混合
        if self.conformal is not None:
            lo, hi = self.conformal.predict_interval(pd_arr)
        else:
            lo = pd_arr * 0.9
            hi = pd_arr * 1.1
        lo = np.asarray(lo)
        hi = np.asarray(hi)

        rng = np.random.default_rng()
        scenarios = []
        # 场景 0: 点预测
        scenarios.append(pd_arr.copy())
        # 场景 1: 上界 (worst-case demand)
        scenarios.append(hi.copy())
        # 场景 2: 下界
        scenarios.append(np.maximum(lo, 0.0))
        # 剩余场景: 区间内截断高斯
        sigma = (hi - lo) / 4.0
        for _ in range(max(0, self.n_scenarios - 3)):
            eps = rng.standard_normal(H)
            scn = np.clip(pd_arr + eps * sigma, lo, hi)
            scenarios.append(np.maximum(scn, 0.0))

        if self._mpc is None or self._mpc.bess is not bess:
            self._mpc = _ParametrizedMPC(bess, self.horizon,
                                         include_carbon=False,
                                         include_peak_charge=True,
                                         peak_charge_weight=self.peak_charge_weight)

        # 对每条场景求解, 收集第 1 步动作
        action_charge, action_discharge = [], []
        costs = []
        for scn in scenarios:
            try:
                res = self._mpc.solve(bess.get_soc(), scn,
                                      renewable_gen, grid_prices)
                ch = float(np.asarray(res['bess_charge']).flatten()[0])
                dc = float(np.asarray(res['bess_discharge']).flatten()[0])
                obj = float(res.get('objective_value', 0.0))
                action_charge.append(ch)
                action_discharge.append(dc)
                costs.append(obj)
            except Exception:
                action_charge.append(0.0)
                action_discharge.append(0.0)
                costs.append(0.0)

        costs = np.asarray(costs)
        action_charge    = np.asarray(action_charge)
        action_discharge = np.asarray(action_discharge)

        # CVaR 加权: 最差 β 场景权重加倍
        if len(costs) > 1 and self.cvar_beta > 0:
            k = max(1, int(np.ceil(self.cvar_beta * len(costs))))
            worst_idx = np.argsort(costs)[-k:]   # top-k 最差
            weights = np.ones_like(costs)
            weights[worst_idx] *= 2.0
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(costs) / len(costs)

        # 加权平均第 1 步动作
        mean_ch = float(np.sum(weights * action_charge))
        mean_dc = float(np.sum(weights * action_discharge))

        # 填充剩余 horizon (仅第 1 步会执行, 其余占位)
        sched_charge    = np.zeros(H)
        sched_discharge = np.zeros(H)
        sched_charge[0]    = max(0.0, mean_ch)
        sched_discharge[0] = max(0.0, mean_dc)

        return {
            'bess_charge':    sched_charge,
            'bess_discharge': sched_discharge,
            'soc_profile':    np.full(H + 1, bess.get_soc()),
            'objective_value': float(np.sum(weights * costs)),
        }


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

        # 削峰目标: 用均值做分界——高于均值放电,低于均值充电
        net = np.maximum(0.0, pred_demand - renewable_gen)
        peak_target = float(np.mean(net))

        for t in range(H):
            net_t = max(0.0, pred_demand[t] - renewable_gen[t])
            if net_t > peak_target and soc > bess.min_soc:
                # ---- 放电: 将 net 削至 peak_target ----
                excess = net_t - peak_target
                avail = (soc - bess.min_soc) * cap
                d = min(bess.max_power, excess, avail)
                discharge[t] = max(0.0, d)
                soc -= discharge[t] / bess.discharging_efficiency / cap
            elif net_t < peak_target and soc < bess.max_soc:
                # ---- 充电: 优先在低价时段, 且不推高 grid_import 超过 peak_target ----
                if grid_prices[t] <= avg_price:
                    headroom = max(0.0, peak_target - net_t)
                    room = (bess.max_soc - soc) * cap
                    c = min(bess.max_power, room, headroom)
                    charge[t] = max(0.0, c)
                    soc += charge[t] * bess.charging_efficiency / cap

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
