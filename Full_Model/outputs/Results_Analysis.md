# Full_Model/outputs 结果分析

仿真口径：8733 小时 (≈363.9 天，年化系数 ≈ 1.00)；储能 162 MWh / 129.6 MW；电价含需量电费 38 元/kW·月；碳价 120 元/tCO₂。共 7 个调度策略 × 11 个汇总图 × 6 组进阶实验 (exp03–08)。

---

## 1. 主表 (strategy_comparison / strategy_ranked)

按综合评分排序（评分 = 节费 + 减碳 + 削峰 + 本地消纳的归一化加权）：

| 排名 | 策略 | 节费% | 减碳% | 削峰% | 本地消纳% | 循环次数 | NPV (CNY) | 回收期 (yr) | score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **DRL-SAC** | +1.60 | +0.61 | −0.66 | 23.38 | 917.5 | −4.46e7 | 10.97 | **0.966** |
| 2 | Rule-Based Peak Shaving | +0.84 | +0.29 |   0.00 | 19.61 | 671.0 | −8.77e7 | 23.93 | 0.724 |
| 3 | Baseline (No Storage) |  0.00 |  0.00 |  0.00 | 16.77 |   0   |     —   |  inf  | 0.448 |
| 4 | Carbon-Aware MPC | +0.04 | −0.17 |  0.00 | 20.24 | 415.5 | −1.33e8 |  inf  | 0.379 |
| 5 | Stochastic MPC | −0.17 | −0.06 | −2.04 | 21.18 | 575.0 | −1.45e8 |  inf  | 0.287 |
| 6 | Economic MPC | −0.70 | −0.05 | −4.88 | 21.15 | 604.5 | −1.76e8 |  inf  | 0.054 |
| 7 | Robust MPC (Conformal) | −0.70 | −0.05 | −4.88 | 21.15 | 604.5 | −1.76e8 |  inf  | 0.054 |

**要点：**
- 只有 DRL-SAC 与 Rule-Based 在所有维度上同时优于 Baseline；其余 4 个 MPC 变体在经济性上**反而比无储能更差**。
- DRL-SAC 与 Rule-Based 构成唯一的 Pareto 前沿（`pareto_front.csv` 一致），`decision_recommender.json` 在五种偏好（economic / green / peak / balanced / grid_indep）下全部推荐 DRL-SAC。
- 所有策略 NPV 均为负值；DRL-SAC 回收期 10.97 年是七者中唯一有限值。

---

## 2. 时序文件 (ts_*.csv) 直接核对

每个 CSV 8734 行，含 actual/predicted/predicted_upper/renewable/grid/bess/soc/cost/co2/price/carbon_intensity 共 12 列。

| 策略 | 电网取电 (GWh) | 充/放电 (MWh) | 充电小时占比 | 放电小时占比 | 空闲占比 | SOC σ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 746.3 | 0/0 | — | — | 100% | — |
| Rule | 749.0 | 28157/25469 | 26.7% | 21.1% | 52.3% | 0.27 |
| Economic MPC | 750.6 | 43628/39313 | 14.3% | 12.4% | 73.3% | 0.37 |
| Carbon-Aware MPC | 749.7 | 34513/31086 |  9.8% |  7.9% | 82.4% | 0.35 |
| Robust MPC | 750.6 | 43628/39313 | 14.3% | 12.4% | 73.3% | 0.37 |
| Stochastic MPC | 750.5 | 43719/39518 | 15.1% | 12.5% | 72.4% | 0.37 |
| DRL-SAC | 752.8 | 65714/59245 | 20.9% | 26.3% | 52.7% | 0.35 |

**要点：**
- 三类 MPC（Economic / Carbon / Stochastic / Robust）**70–82% 的时间储能完全闲置**，仅在 24h 窗口里偶发地套利，这与它们没能跑赢 Rule-Based 直接对应。
- DRL-SAC 是最激进的：循环次数最高 (917)、放电小时占比 26.3%，因此它能拿到最高的本地消纳率（23.38%）。

---

## 3. 关键异常 / 数据健康问题（必须解决）

### 3.1 预测时序为"完美预测"（**关键问题**）

`ts_Baseline_No_Storage.csv` 等所有时序文件里：
```
predicted_demand ≡ actual_demand    （8733 行全部相等）
predicted_upper  ≡ predicted_demand （CP 区间宽度恒为 0）
```
这意味着本批结果是在 **oracle / perfect-foresight** 模式下跑出来的；既没有调用 LSTM 也没有调用 conformal predictor。结论：
- §2.2 LSTM + §2.3 conformal 在 paper 的描述是正确的，但**当前的结果文件无法用来验证那两节的有效性**。
- Robust MPC 与 Economic MPC 各项指标（cost/peak/CO₂/cycles）**逐位完全相同**，正是因为 CP 上界等于点预测，鲁棒裕度恒为 0，控制器退化为 Economic MPC。
- exp08 报告 "daily_error_sum = 0, CP coverage = 100%" 也是同一原因。

**建议**：重新跑一次将 `predict_demand` 改回真实 LSTM 输出，并让 `conformal_predictor` 在标准模式下生成非零 `q̂`。

### 3.2 exp05 求解器全程退化为启发式

`experiment_summary.json` 中 exp05：
```
Econ-MPC   solver_counts = {'FALLBACK': 200}  fail_count = 200 / 200 steps
Carbon-MPC solver_counts = {'FALLBACK': 200}  fail_count = 200 / 200 steps
```
即三档求解器链 SCS → CLARABEL → OSQP **在计时实验里 100% 失败**。所以 exp05_mpc_timing_*.png 反映的其实是 `_heuristic_fallback` 的耗时（Econ ≈ 36 ms / 步，Carbon ≈ 0.3 ms / 步，差异极大，提示两者走的并不是同一兜底路径）。论文里"per-step solve < 1 s"的论断在这个数据集上无法被直接支撑。

### 3.3 MPC 反而**抬高**了月度峰值

Economic / Robust / Stochastic 的 Peak Reduction 分别是 −4.88% / −4.88% / −2.04%。结合时序数据：BESS 在低负荷时段充电时 `grid_import = load + Pc` 创造出新的峰值。`strategies._ParametrizedMPC` 已经引入了 `running_peak_p` 显式控制，但当前结果显示该机制在长周期下还是被低估了。建议把 `peak_charge_weight` 从 1.5 调大，或在 main.py 里把 `running_peak` 从外部按月置零。

### 3.4 BESS 严重过配

`exp06` 扫了 0–300 MWh 的容量，全部容量 NPV 都为负，最大也只是最小测试点 5 MWh：
```
optimal_capacity_kwh = 5000   optimal_NPV = -6.09e+06
```
即在 600 CNY/kWh 投资 + 现行电价/碳价/TOU 结构下，**本港口 BESS 本身就没有正收益**。论文中沿用的 162 MWh 是当年 NPV 最大化得出的（main.py 注释中提到 exp06 的"NPV 最大化"），但当前 exp06 数据并不支持这个结论 — 需要核对 exp06 是否换了价格输入或运行了更长寿期。

### 3.5 exp04 碳灵敏度扫描"不动"

```
sens ∈ {0.5, 1, 2, 3, 5, 7, 10} → 全部 energy_cost=8.04e8, co2=544720, total=8.69e8
```
说明 Carbon-MPC 的 carbon_weight 调参对最终 KPI 没有任何影响 — 与 3.1 同源（CP 上界为 0、Carbon-MPC 在大多数小时退化到边界解），或与 3.2 同源（求解器全程兜底，忽略 `carbon_weight`）。

---

## 4. 进阶实验摘要

### exp03  噪声鲁棒性（5 trials × 4 noise levels）
| σ | Econ peak (kW) | Robust peak (kW) | Robust 相对优势 |
|---:|---:|---:|---:|
| 0.0 | 435 634 | 435 634 | 0 |
| 0.1 | 442 177 | **446 064** | −0.88%（Robust **更差**） |
| 0.2 | 457 194 | 457 194 | 0 |
| 0.3 | 479 649 | 479 649 | 0 |

在当前的 CP 区间宽度 = 0 的条件下，Robust MPC 不可能比 Economic MPC 更好；σ=0.1 出现的 −0.88% 反向劣势来自单次实验偶然性（σ_cost 也增加 ~43%）。

### exp04  碳价 / 灵敏度扫描
- 碳价 0–300 元/tCO₂，Carbon-aware 与 Economic 在 `crossover_price = 83.31 元/tCO₂` 处相交：
  - 碳价 < 83 元/tCO₂：Economic MPC 总成本更低。
  - 碳价 > 83 元/tCO₂：Carbon-aware 略胜（差距 ≤ 2 M CNY，占比 < 0.3%）。
- 灵敏度 0.5–10 无差异（见 §3.5）。

### exp05  滚动 MPC 计时（详见 §3.2）

### exp06  BESS 容量扫描（详见 §3.4）

### exp07  TOU 峰谷比扫描
峰谷比 2→5，节省百分比 −0.43% → −0.47%（越拉大越亏）；Economic 与 Robust 数值逐位相同。说明在当前 CP 区间为 0、求解器表现欠佳的组合下，提高 TOU 价差**不会**自动改善套利收益。

### exp08  极端日
报告的 worst_day = 2024-01-02，daily_error_sum = 0、max_hourly_error = 0、CP 覆盖率 100%；这同样是 §3.1 oracle 预测的副产物，没有真正考察极端日的预测/控制响应。

---

## 5. 主要图表对应的物理含义

| 图 | 内容 |
|---|---|
| 01_kpi_bars.png | 7 策略 KPI 柱状对比 |
| 02_cost_breakdown.png | 电量 vs 需量 vs 碳费 三段堆叠 |
| 03_pareto.png / 10_pareto_front.png | 成本-碳排-峰值 Pareto 投影 |
| 04_improvement_heatmap.png | 三维改进矩阵 |
| 05_radar.png | 五维 KPI 雷达图 |
| 06_load_duration.png | 持续负荷曲线（CDF） |
| 07_daily_profile.png | 典型日 24h 平均 |
| 08_soc_distribution.png | 各策略 SOC 直方 |
| 09_timeseries.png | 多策略时序对比片段 |
| experiments/exp03_* | 噪声 5 试次 trial / 总成本 / 峰值散点 |
| experiments/exp04_* | 碳价扫描 / 灵敏度 / 边际减排 |
| experiments/exp05_* | MPC 单步时延 violin / CDF / trace |
| experiments/exp06_* | BESS NPV / IRR / payback / 灵敏度 |
| experiments/exp07_* | TOU 峰谷比扫描下的节费曲线 |
| experiments/exp08_* | 极端日预测误差 / 能流 / 累积 |

---

## 6. 结论与下一步

1. **DRL-SAC 是当前结果集中唯一稳定优于 Baseline 的策略**，在四类 MPC 全部退化（70–80% 闲置 + 月度峰值反弹）的对比下尤为突出。论文中如果要保留 §2.5 五策略对比，必须先解决 §3.1–3.3 的复现问题。
2. **必须重跑一次完整年度仿真**，开启真实 LSTM 预测 + 真实 conformal 校准（`q̂ > 0`），否则 Robust / Stochastic / 论文中"conformal-robust"的核心论点缺乏证据。
3. **求解器链需要诊断**：exp05 显示 SCS/CLARABEL/OSQP 全部失败，可能是 load/price 量纲未做缩放或 warm start 状态被重置。
4. **BESS 容量与价签**：基于 exp06，162 MWh 在当前价签下无法回本。论文 §2.4 写 "exp06 NPV-optimal" 与当前 outputs 不一致，需要重新对齐。
5. **峰值管理参数**：把 `peak_charge_weight` 从 1.5 提到 ≥ 4 或在 Economic MPC 中将 `running_peak_p` 强制重置为月初零，再观察 Peak Reduction 是否能由 −4.88% 转正。
6. **建议保留的图**：01/03/05/10 与 exp06 全部、exp04_cost_decomposition、exp07_tou_savings_curve 可直接用于论文 §3 Results；exp03 / exp05 / exp08 需要在数据修复后重画。
