# 输出文件结果检查报告

**仿真期**：8733 小时（约 363.9 天）；**储能**：162 MWh / 129.6 MW；**碳价**：120 元/tCO₂；**需量电费**：38 元/kW/月。

总体结论：图表与 CSV 数据本身在大多数主图（01–10）上是 **彼此一致** 的；但同时暴露出若干 **实质性的模型问题** 与 **若干图/表标注缺陷**。其中 4 处属于严重模型缺陷（影响结论可信度），5 处属于明显的图表/口径问题。下面按文件逐一列出并附根因定位。

---

## 一、主输出 outputs/01–10

### 01_kpi_bars.png — ✅ 与 CSV 一致
- 4 个面板（Total Cost / CO₂ / Peak / Grid Indep）数值与 `strategy_comparison.csv` 完全吻合：例如 Baseline 总成本 618,399,823 CNY 即 Energy + Demand 之和。
- 注意：此处 "Total Operating Cost" **不含碳成本**。

### 02_cost_breakdown.png — ⚠️ 标题/口径不一致
- 标题写 "Total Cost Breakdown (Energy + Demand + CO₂)" 并显示 Baseline 683.4M。
- 但 01 图与 CSV 的 "Total Cost" 仅 = Energy + Demand = 618.4M，**不含 CO₂**。
- 同一报告中 "Total Cost" 出现两个不同口径会引起读者误读。**建议**：把 02 图改名为 "Cost + Carbon Cost Stack" 或在 01 图脚注里说明 "Total Cost 不含碳费"。

### 03_pareto.png — ⚠️ 视觉缺陷
- 图例上有 "Pareto frontier" 红虚线条目，但实际图上**没有线**（在 (Cost, CO₂) 二维投影下唯一非劣点是 Stochastic MPC，单点画不出折线）。
- 此外 Baseline 与 DRL-SAC 几乎完全重合，导致标签互相遮挡。
- **建议**：要么把 frontier 线项目删掉，要么改为高亮散点（star marker）；并对 DRL-SAC 加偏移避免遮挡。

### 04_improvement_heatmap.png — 🚩 严重口径错误
- 标题写 "Improvement vs Baseline (%)"，色阶以 0 居中。
- 前 3 列 (Cost Savings, CO₂ Reduction, Peak Reduction) **是**相对 baseline 的改进百分比（正确）。
- 后 3 列 (Grid Indep., Load Factor, RE Penetration) **却是绝对值**（如 Baseline 显示 16.77、0.20、16.77，而非 0），含义被混进同一个色阶里，会被误读为 "baseline 也有 16.77% 改进"。
- 根因：`analysis.py:plot_improvement_heatmap` 直接从 comparison_df 取这 6 列原值，未对绝对指标做 Δ 处理。
- **修复**：① 把后 3 列改为 (strategy − baseline) 的差值；或 ② 单独画一张 "Absolute Metrics" 子图，避免混入相对色阶。

### 05_radar.png — ⚠️ 字号过小
- 五维雷达内容正确（Cost Save / CO₂ / Peak / Grid Indep / Load Factor），但分辨率低、图例与坐标重叠，难辨识。**建议**：增大画布并把图例放外侧。

### 06_load_duration.png — ✅ 数据合理
- 各策略的 LDC 几乎重合，因储能容量相对峰值 (129.6 / 435.6 ≈ 30%) 看似很大，但能量量级（162 MWh ≪ 总用电 750 GWh） 仅 0.02%。曲线相近符合物理预期。

### 07_daily_profile.png — 🚩 严重发现
- 子图 2（Avg SOC）：**DRL-SAC 的 SOC 横线钉死在 90%**，与 BESS Power ≈ 0 一致 → DRL 智能体几乎不动。
- 子图 3（Avg BESS Power）：MPC 系列在 11–16 时呈大幅充电峰（−20 MW 以上），随后放电。这种 "白天太阳出力时充电" 在港口负荷曲线下是合理的（白天谷价 + 光伏富余）。**但充电幅度大于光伏富余时反推出净电网导入增加**，正是 02 图与 KPI 中 MPC 峰值 +25.76% 的来源。

### 08_soc_distribution.png — ✅ 合理
- Rule-PS 主体集中在 30%（基本不深放）；MPC 三档分布广（10%–90%），符合 MPC 在每个 24h 滚动时域结尾几乎用尽 BESS 容量的特征；Stochastic MPC 偏均匀。
- 注：DRL-SAC、Baseline 因 SOC 单点退化未参与 violin（合理过滤）。

### 09_timeseries.png — ✅ 合理
- 168 h 切片显示 SOC 边界打到 [10%, 90%]、累计 CO₂ / Cost 单调上升，斜率分层清晰。

### 10_pareto_front.png — ✅ 与 P2 推荐器一致
- 在 (Cost, CO₂)、(Cost, Peak)、(CO₂, Peak) 三个 2D 投影下，红星标出 Pareto 集合 = {Rule-PS, Stochastic MPC}，与 `decision_recommender.json` 完全一致。

---

## 二、experiments/exp01–exp05

### 🚩 跨 exp01/exp02 共有的根本问题：底层预测器严重偏差

直接从 `outputs/ts_Baseline_No_Storage.csv` 抽样：

| 指标 | actual_demand | predicted_demand |
|---|---|---|
| min | 0 | **−50,847**（不可能为负） |
| max | 502,677 | **1,353,131**（真值的 2.7×） |
| mean | 102,674 | 129,264（**+25.9% 系统性偏高**） |

预测残差 std=131 GW（几乎等于真值的均值水平），且 `predicted_upper` 列与 `predicted_demand` **完全相同**（说明上界没有被写入 CSV，或者保形带被覆盖）。这是后续 CP / UQ 实验全部失真的源头。建议优先排查：
1. `main.py:537` 的 `weights_only=True` 是否静默回退到随机权重；
2. `EModel_FeatureWeight4.pth` 是否真的加载成功；
3. 训练/验证/部署阶段的特征 scaler 是否一致（labelEncode + 数值列混用极易错位）。

### exp01_cp_coverage_by_condition.png — 🚩 严重 + 视觉缺陷
- 三幅条形图全部 "看不见柱子"：因为 90% 名义覆盖目标的实际经验覆盖只有 1.1%–43.1%，远低于 y 轴下限 70%。
- 数值标签被甩在子图框外（成漂浮的浮动文字），属于绘图 bug。
- 实际覆盖远低于 90% 是模型问题（见上方根因）。

### exp01_cp_residual_dist.png — 🟡 校准内一致但分布右偏
- q(90%) = 0.2225, q(95%) = 0.2771（归一化分数 |r|/max(|ŷ|,ε)）。校准集自身的分位数 OK。
- 但测试集相对校准集存在显著漂移，叠加预测器偏置 → 在线覆盖崩塌。

### exp01_cp_rolling_coverage.png — 🚩 严重
- 0–6800 h 的滚动覆盖几乎为 0；到第 6800 h 后才跳到 ~95%。这意味着前 78% 的仿真期里 CP 区间几乎完全错过实际值；只有最后 1900 h 接近名义。
- 与上面的预测偏置一致（前段误差极大）。

### exp01_cp_interval_timeseries.png — 🚩 严重
- 灰色 actual 曲线在 20–80k kW，蓝色 predicted 曲线却贴在 0 附近，CP 区间紧紧裹住 ≈0 的预测。这说明在该窗口 LSTM 输出几乎为零，误差几乎与真值同量级，CP 无能为力。

### exp01_cp_scatter_coverage.png — 🚩 严重
- 散点呈 actual ≪ predicted 的尾部分布（红色 miss 70.4%）。预测器在高负荷段被严重高估，CP 覆盖只有 29.6%。

### exp02_uq_coverage_width.png — 🚩
- MC Dropout 覆盖 5%, CP 27.5%（均远低于 90% 名义）。证实 §一 中的预测器问题不可被任何 UQ 方法弥补。

### exp02_uq_intervals_timeseries.png — 🟡
- CP 上界出现尖峰高至 1e6+（与 predicted_demand 的离群相符），实际曲线低得多。

### exp02_uq_width_violin.png — ⚠️ 视觉 bug
- CP 小提琴下端延伸到 −50,000 kW（区间宽度物理上不应为负）。这是 KDE 过度平滑伪影，但容易被读者解读为 "区间反转"。**建议**绘图时手动 clip x ≥ 0。

### exp02_uq_width_vs_demand.png — ⚠️ 同上
- 也有少量散点 y < 0（同 KDE 平滑或 CQR 上下界倒序，conformal_predictor.py 中 lower 已 clip 至 0，可能是 plotted 列对应的是预归一化分数）。

### exp02_uq_winkler_score.png — ✅
- CP 中位 693k 优于 MC Dropout 1.0M。相对比较成立。

### exp03_noise_*.png — 🚩 严重 + 误导
- `exp03_noise_raw_trials.csv` 显示：**noise=0.2 与 0.3 时, Economic 和 Robust 五次试验都得到完全相同的成本 618,399,823.117，恰等于 Baseline**。Robust MPC 在 noise=0.1 五次试验也完全等于 Baseline。
- 根因：`strategies.py:_ParametrizedMPC.solve` 的 fallback 分支
  ```python
  if status not in (OPTIMAL, OPTIMAL_INACCURATE) or self.Pc.value is None:
      return {'bess_charge': zeros, 'bess_discharge': zeros,
              'grid_import': load - pv, ...}
  ```
  当噪声把 `noised_pred` 推到极端值 → 24h 时域内 MPC 解不可行/求解器超时 → 静默退化为 baseline，且 5 次试验都退化到同一确定性轨迹，**完全抹掉了 noise 的随机性**。这导致：
  - Robust 在所有噪声水平的成本都是常量（图 03_noise_absolute_cost.png 中 Robust 列恒为 61840.0×10k），
  - "Robust vs Economic 优势" 图变成 0% / 0% 的哑铃图（exp03_noise_robust_advantage.png），
  - Cost Degradation 跌入 ~0 浮点噪声水平。
- **结论**：当前 exp03 不能作为论据；需要在 fallback 时显式打 warning，并把失败 trial 标记为 NaN 而非用 baseline 替换。

### exp04_carbon_*.png — 🚩 严重
- `exp04_carbon_cost_emission.png`：Carbon-Aware MPC 与 Economic MPC 的两条线 **完全重合**，CO₂ 排放在 50–300 元/tCO₂ 区间几乎是平的（≈ 540,000 t）。
- `exp04_marginal_abatement.png`：碳感知 MPC 相对经济 MPC 的 "额外减排" 全部为 0；右侧 MAC 子图为空。
- `exp04_carbon_sensitivity_scan.png`：右侧标题中的中文出现 "□□□" 字符方块（matplotlib 中文字体未注册）。
- 根因：`strategies.py:CarbonAwareMPCStrategy` 中 `cw = carbon_price/1000.0 * carbon_sensitivity`。在 carbon_price=120, carbon_sensitivity=3.0 时 cw=0.36；目标函数中 `carbon_weight × Σ(Pg × carbon_p)` 的量级远小于 `Σ(Pg × price_p)`（电价 ≈ 0.6–1.5 元/kWh，碳成本 ≈ 0.6×0.36 ≈ 0.22 元/kWh），权重相差悬殊，碳项基本被电价主导，Carbon-Aware 收敛到与 Economic 几乎一致的解。
- **修复**：① 直接把 cw 取为 carbon_price（不要除以 1000，因为 carbon_p 单位是 kg/kWh，1000 已经把它从 t/MWh 转成 kg/kWh），或者 ② 把 carbon_sensitivity 调到 50–100。

### exp05_mpc_timing_*.png — 🚩 严重
- `exp05_solver_summary.png` 标注 **OSQP (0%), Steps: 0**，但 violin/CDF/trace 又给出几十毫秒的 Econ-MPC、亚毫秒的 Carbon-MPC。
- Solver 计数为 0 + Carbon-MPC 中位 0.3 ms，强烈暗示 Carbon-MPC 没有真正进入 OSQP 求解，可能直接走了缓存或 fallback 路径。
- 根因：`experiments.py:experiment_mpc_timing` 中 `solver_used` 只在内层 try 里 `sname` 赋值，并依赖名称识别失败，对 `solver=None`（CVXPY 默认派发，可能是 SCS/CLARABEL）时没有分类计入。最终输出 "OSQP (0%)" 是 **指标 bug**，并不代表求解器没工作。
- **修复**：直接读 `problem.solver_stats.solver_name`（CVXPY 自动填充），并把 0.3 ms 的 Carbon-MPC 单独排查（看是否 horizon=24 的同一参数化问题被复用，根本没重设参数）。

---

## 三、综合排序的核心问题

排序按对论文/报告结论的影响程度：

1. **预测器系统性高估、产生负值与 2.7× 离群**（出现在 02_cost_breakdown 之外的所有不确定性图）。这是基础模型层 bug，必须先修。
2. **MPC fallback 静默回退到 baseline**（exp03 噪声实验完全失效；同样会污染主结果中所有解失败的小时）。
3. **Carbon-Aware MPC 的碳权重数量级错误** → exp04 不能体现碳价敏感性。
4. **DRL-SAC 智能体未真正训练**：8732/8733 时刻 BESS 动作为 0，SOC 死锁在 90%。当前结果不应作为强化学习对比的依据。
5. **04_improvement_heatmap 图把绝对值与相对值混入同一色阶**。
6. **02_cost_breakdown 的 "Total Cost" 与 01/CSV 口径不同**（前者含 CO₂，后者不含）。
7. **exp05 求解器统计字段错误**（OSQP 0% 是统计 bug，不是求解器没运行）。
8. **exp02 violin/scatter 出现负宽度**（KDE 平滑伪影或 CQR 上下界倒序，需要 clip 或断言）。
9. **exp04 carbon_sensitivity_scan 中文字体显示为方块**（matplotlib font 配置缺失）。
10. **MPC 反而把峰值抬高 22–26%**（peak_charge_weight ≈ 1.27 元/kW/日，相对每日电费量级仍偏小，权重整体太弱）。

---

## 四、建议的最小改动顺序

1. 在 `_ParametrizedMPC.solve` 的 fallback 路径加 `warnings.warn` 并把状态写到 ts CSV（exp03 立即可见真实情况）。
2. 把 `CarbonAwareMPCStrategy` 中 `cw = carbon_price/1000.0 * carbon_sensitivity` 改为 `cw = carbon_price * carbon_sensitivity`（或在 main.py 把 carbon_sensitivity 调到 50+）后重跑 exp04。
3. 修 `analysis.py:plot_improvement_heatmap`，对后 3 列改用 `value − baseline_value`。
4. 修 `experiments.py:experiment_mpc_timing`，统计来源改为 `problem.solver_stats.solver_name`。
5. 在 `outputs/ts_*.csv` 落盘前 `np.clip(predicted_demand, 0, None)` 并补一列实际 lower/upper（眼下 `predicted_upper` 与 `predicted_demand` 完全相同，明显是写入 bug）。
6. 给 exp01_cp_coverage_by_condition 调整 y 轴 `[0, 100]`，并把 `ax.bar()` 的 height 参数改为实际 series（看起来 bar 调用接收的高度可能是 0 或 NaN，导致只剩下文本标签）。
7. DRL-SAC 在当前训练步数下不可用，建议先在主对比表里做 "实验性" 标注，或等 SAC 训练步数 ≥ 20k 且加入 reward shaping 后再纳入。

---

报告生成时间：2026-04-27
