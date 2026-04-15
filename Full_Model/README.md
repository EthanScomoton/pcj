# 港口综合能源系统 —— 策略选择平台

> 基于《深度学习驱动的干散货港口综合能源系统：技术蓝图与代码升级方案》调研报告重构。
> 使用预训练 LSTM 模型 (`best_EModel_FeatureWeight4.pth`) 进行负荷预测，
> 通过多种调度策略对港口储能系统进行滚动优化，
> 最终输出 **经济性 + 环保性 + 技术性** 三维综合分析。

---

## 一、整体运行思路

本平台是围绕 **"预测 → 多策略优化 → 对比评估"** 三段式架构搭建的：

```
          ┌──────────────────────────────────────┐
          │  数据层: renewable_data + load_data   │
          │    ↓  重采样 1h → 特征工程 → 归一化     │
          └──────────────────────────────────────┘
                           ↓
          ┌──────────────────────────────────────┐
          │  预测层: .pth 模型 + 保形预测校准      │
          │   点预测 μ  +  预测区间 [μ-q̂, μ+q̂]    │
          └──────────────────────────────────────┘
                           ↓  (预测共享)
   ┌────────┬────────┬─────────┬──────────┬─────────────┐
   │Baseline│ 规则型 │Econ MPC │Carbon MPC│Robust MPC   │
   │无储能  │削峰填谷│电费最小 │成本+碳排 │保形鲁棒      │
   └────┬───┴────┬───┴────┬────┴─────┬────┴─────┬───────┘
        └────────┴────────┴──────────┴──────────┘
                           ↓
          ┌──────────────────────────────────────┐
          │  仿真层: StrategyAwareIES (逐小时)    │
          │   电池充放电 → SOC 更新 → 潮流核算    │
          └──────────────────────────────────────┘
                           ↓
          ┌──────────────────────────────────────┐
          │  分析层: 经济 / 环保 / 技术 KPI        │
          │   加权评分 → Pareto 前沿 → 最终报告    │
          └──────────────────────────────────────┘
```

**核心设计要点**

1. **预测与决策解耦**：预训练 `.pth` 模型只负责预测，调度决策由可替换的 **策略类** 完成
2. **预测一次，多策略共享**：对 N 小时仿真只计算 `N + 24` 次模型前向，5 种策略共用，降低 24 倍计算量
3. **参数化 CVXPY**：MPC 问题只构建一次，每步热启动求解，比原代码快 5–10 倍
4. **不确定性驱动的鲁棒调度**：保形预测给出 (1−α) 覆盖率的预测区间，鲁棒策略以上界进行调度
5. **双维度目标**：同时将电费、分时电价、动态边际碳强度、电网排放因子纳入目标函数

---

## 二、文件结构与职责

共 15 个文件。按模块分类：

### 1. 平台入口
| 文件 | 说明 |
|------|------|
| **`main.py`** | 平台主流程。**8 个阶段**依次执行：数据加载 → Scaler 拟合 → 模型加载 → 保形校准 → 电价与碳强度构造 → 预测缓存 → 多策略仿真 → 综合分析与可视化 |
| **`README.md`** | 本文档 |

### 2. 新增核心模块（5 个，来自报告 P0 优先级）
| 文件 | 对应报告章节 | 关键类 / 函数 |
|------|------------|------------|
| **`strategies.py`** | §四.4.2 / §三.3.3 / §四.4.5 | `BaseStrategy`、`_ParametrizedMPC`、`BaselineStrategy`、`PeakShavingRuleStrategy`、`EconomicMPCStrategy`、`CarbonAwareMPCStrategy`、`RobustMPCStrategy`、`build_default_strategy_suite` |
| **`enhanced_ies.py`** | §四.4.2 | `StrategyAwareIES`（继承 `IntegratedEnergySystem`），提供 `precompute_predictions` 和 `simulate_with_strategy` |
| **`conformal_predictor.py`** | §三.3.3 | `ConformalPredictor`：无分布假设区间估计 |
| **`carbon_module.py`** | §四.4.5 | `CarbonTracker`、`make_dynamic_carbon_intensity`、区域电网排放因子常量 |
| **`analysis.py`** | §六 | `compute_economic_kpis`、`compute_environmental_kpis`、`compute_technical_kpis`、`build_comparison_table`、`score_strategies`、`plot_strategy_kpis`、`plot_time_series`、`plot_pareto_cost_vs_co2`、`format_final_report` |

### 3. 原有支撑模块（7 个，继承使用）
| 文件 | 说明 |
|------|------|
| **`All_Models_EGrid_Paper.py`** | 所有预测模型类（含 `EModel_FeatureWeight4`）+ `load_data` + `feature_engineering` + `calculate_feature_importance` |
| **`IES.py`** | `IntegratedEnergySystem` 父类，负责 `predict_demand`、滑窗特征构造、维度对齐 |
| **`BES.py`** | `BatteryEnergyStorage` —— 电池物理模型 (SOC / 充放电效率 / 功率限制) |
| **`EO.py`** | `EnergyOptimizer` —— 原版 CVXPY 调度器（被 `IES` 在 `__init__` 中实例化） |
| **`REO.py`** | `RenewableEnergyOptimizer` —— 可再生能源优化器（同上被 IES 依赖） |
| **`EF.py`** | 特征提取、可再生预报、经济指标（`calculate_economic_metrics`） |
| **`convert_model.py`** | 当训练时特征维度与运行时不一致时，自动对 `.pth` 权重做维度填充/截断 |

### 4. 资产文件
| 文件 | 说明 |
|------|------|
| **`best_EModel_FeatureWeight4.pth`** | 预训练的 EModel_FeatureWeight4 权重 (~11.9 MB) |

---

## 三、每个文件的内部逻辑详解

### 3.1 `main.py` —— 平台编排器

**职责**：把 8 个阶段串联起来，执行全流程。

| 阶段 | 做什么 | 调用谁 |
|------|--------|--------|
| 1 | 读取 CSV、去重、1h 重采样、插值、特征工程 | `load_data`、`feature_engineering` |
| 2 | 按前 80% 训练集拟合 `scaler_X / scaler_y`（`log1p` + `StandardScaler`） | `sklearn.StandardScaler` |
| 3 | 创建 `EModel_FeatureWeight4`，加载 `.pth`；维度不匹配则调 `convert_model_weights` | `All_Models_EGrid_Paper`、`convert_model` |
| 4 | 在 80%–90% 数据段构造校准窗口，算残差 → 保形 q̂ | `ConformalPredictor.calibrate_from_sequences` |
| 5 | 生成分时电价（峰/平/谷/周末折扣）+ 动态碳强度 | `make_dynamic_carbon_intensity` |
| 6 | 一次性计算 `SIM_HOURS + 24` 个窗口的负荷预测，5 策略共享 | `StrategyAwareIES.precompute_predictions` |
| 7 | 依次对 5 种策略进行仿真，收集时序 + KPI | `StrategyAwareIES.simulate_with_strategy` |
| 8 | 对比表、加权打分、柱状/时序/Pareto 可视化，导出 CSV + PNG + TXT | `analysis.*` |

**可配置项（`PlatformConfig` 类）**：仿真时长、MPC 窗口、储能容量功率、电网区域、碳价、保形 α、评分权重、输出目录。

---

### 3.2 `strategies.py` —— 调度策略库

**设计模式**：抽象基类 `BaseStrategy` 统一接口，具体策略通过 `optimize(...)` 返回 24h 调度计划字典 `{bess_charge, bess_discharge, grid_import, status}`。

**关键类**

#### `_ParametrizedMPC` —— 参数化 CVXPY 问题（性能核心）
- 在 `__init__` 中 **一次性构建** CVXPY 问题结构
- 使用 `cp.Parameter` 表示负荷、PV、电价、初始 SOC、碳强度、碳权重等输入
- 决策变量：`Pc`（充电功率，非负）、`Pd`（放电功率，非负）、`Pg`（购电，非负）、`Pre_use`（可再生利用，允许弃电）、`soc`
- 约束：SOC 动力学、SOC 边界 `[min_soc, max_soc]`、功率限制、功率平衡、可再生不超过预报
- 目标：购电成本 + 轻正则 + 终端 SOC 回补 + 轻度鼓励消纳 (+ 可选碳排项)
- `solve()` 只更新 `Parameter.value`，并通过 `warm_start=True` 热启动求解（OSQP → CLARABEL → default 三级回退）

#### 5 个具体策略

| 策略 | 场景 | 目标 |
|------|------|------|
| **`BaselineStrategy`** | 对照基准：无储能 | grid = load − renewable |
| **`PeakShavingRuleStrategy`** | 朴素对照组 | 谷电充 / 峰电放，基于均价阈值 |
| **`EconomicMPCStrategy`** | 经济最优 | min Σ(Pg·price) |
| **`CarbonAwareMPCStrategy`** | 成本+碳排双目标 | min Σ(Pg·price) + (carbon_price/1000)·Σ(Pg·intensity) |
| **`RobustMPCStrategy`** | 鲁棒对抗不确定性 | 用保形上界替代点预测作 demand 输入 |

`build_default_strategy_suite()` 一键返回默认策略集合。

---

### 3.3 `enhanced_ies.py` —— 策略感知仿真引擎

继承自 `IntegratedEnergySystem`，新增：

- **`precompute_predictions(historic_data, time_steps, horizon)`**：为所有需要的索引（`range(time_steps + horizon)`）提前计算负荷点预测，存成 ndarray。5 种策略重复使用同一份预测结果，避免重复前向。

- **`simulate_with_strategy(historic_data, time_steps, price_data, predictions_by_index, horizon)`**：逐小时循环：
  1. 从缓存中取 24h 预测
  2. 若提供保形预测器则计算上界
  3. 取 PV/风电预报（`get_renewable_forecast`）
  4. 取电价 + 动态碳强度切片
  5. 调 `strategy.optimize(...)` 得到 24h 计划
  6. **只执行第一步**的充/放电（MPC 核心思想）
  7. 记录潮流、成本、CO₂、SOC 等 12 个字段

支持参数：`allow_grid_export`（是否允许向电网反送电）、`export_price_ratio`（反送电价折扣）。

---

### 3.4 `conformal_predictor.py` —— 保形预测

**原理**：在校准集上计算 |点预测 − 真值| 的分布 → 取 ⌈(m+1)(1−α)⌉/m 分位数作为 q̂ → 预测区间 = [μ − q̂, μ + q̂]。

**API**
- `calibrate_from_sequences(model, X_cal, y_cal, device, scaler_y, use_log_y)`：在已归一化的窗口上前向、反归一化、计算残差、拟合 q̂
- `calibrate_from_residuals(residuals)`：如果已有残差数组可直接校准
- `predict_upper(pred)` / `predict_lower(pred)` / `predict_interval(pred)`
- `summary()`：返回 α、目标覆盖率、q̂、校准样本数

**数学保证**：1−α 覆盖率有理论保证（交换性假设下）。

---

### 3.5 `carbon_module.py` —— 碳排放追踪

**常量**
- `GRID_EMISSION_FACTORS`：中国六大区域电网 CO₂ 排放因子（tCO₂/MWh）
  - 华北 0.8843 / 华东 0.7035 / 南方 0.3869 / 华中 0.5655 / 东北 0.6673 / 西北 0.6448 / 全国 0.5810
- `DEFAULT_CARBON_PRICE_CNY_PER_TON = 100`（2024 年 CEA 价格水平）

**核心类 `CarbonTracker`**
- `emissions_kg(grid_kwh)` → CO₂ 排放 (kg)
- `carbon_cost_cny(grid_kwh)` → 碳成本 (元)
- `summary(series)` → 总量汇总 + 等效植树 / 减车数

**动态边际强度 `make_dynamic_carbon_intensity`**
- 峰时段 (08-12, 14-18)：×1.25 （火电高占比）
- 谷时段 (00-06)：×0.75 （水电/核电/风光高占比）
- 其它：×1.0

→ 为 `CarbonAwareMPCStrategy` 提供 24h 碳强度预测输入。

---

### 3.6 `analysis.py` —— 三维 KPI 与可视化

| 函数 | 输出 |
|------|------|
| `compute_economic_kpis` | 总成本、平均小时成本、总购电量、NPV、IRR、回收期、年节省 |
| `compute_environmental_kpis` | 总 CO₂（kg/吨）、可再生自消费率、碳成本、等效植树、等效减车 |
| `compute_technical_kpis` | 峰值购电、平均/最小/最大 SOC、循环次数、总充/总放电量 |
| `build_comparison_table` | 返回 DataFrame：每策略一行，含成本/碳排/峰值/SCR/循环次数/NPV/回收期 + 相对基准的节省率/减排率/削峰率 |
| `score_strategies` | 按 cost / co2 / peak 权重归一化加权得分并排序 |
| `plot_strategy_kpis` | 柱状图：4 个维度对比 |
| `plot_time_series` | 时序对比图：grid_import / SOC / 累计 CO₂ / 累计成本 |
| `plot_pareto_cost_vs_co2` | Pareto 散点图：成本 vs 碳排 |
| `format_final_report` | 生成纯文本综合分析报告 |

---

### 3.7 原有支撑模块（继承使用）

#### `All_Models_EGrid_Paper.py`
- `load_data()`：读取 `renewable_data10.csv` + `load_data10.csv` 并合并
- `feature_engineering(df)`：EWMA 平滑、时间特征 sin/cos 编码、天气/季节等类别编码
- `EModel_FeatureWeight4`：双向 LSTM + 特征门控 + 双注意力（时间 + 特征）+ 2 头输出 (μ, logvar) + 重参数化
- `calculate_feature_importance`：基于 Pearson 相关系数估算特征重要性

#### `IES.py`
- `IntegratedEnergySystem`：父类
  - 构造时实例化 `BatteryEnergyStorage`、`EnergyOptimizer`、`RenewableEnergyOptimizer`
  - `predict_demand(features)`：特征归一化 → 模型前向 → 反归一化（`inverse_transform` + `expm1`）
  - `_build_window_sequence`：构造 `window_size=20` 的滑窗特征序列，缺口用首行填充
  - `adapt_features`：特征维度自动对齐（截断或零填充）

#### `BES.py`
- `BatteryEnergyStorage`：SOC 范围 [0.1, 0.9]，充放电效率 95%，容量=0 时直接退化为空电池分支

#### `EO.py`
- 原 `EnergyOptimizer`（非参数化版本）—— 被 IES 在构造时实例化但新流程中不再直接调度使用

#### `REO.py`
- `RenewableEnergyOptimizer`：PV 弃电上限 20%，风电上限 30%

#### `EF.py`
- `extract_features(df, idx, feature_cols)`：按指定列顺序提取特征行
- `get_renewable_forecast(df, start, n)`：PV/风电切片
- `calculate_economic_metrics(costs, investment, ...)`：NPV / 回收期 / IRR（二分法求解）

#### `convert_model.py`
- `convert_model_weights(pretrained_path, new_feature_dim, ...)`：对 `feature_importance`、LSTM 输入权重、FC 中间层做维度填充/截断，保存新的 `.pth`

---

## 四、端到端运行步骤

### 4.1 环境依赖

```bash
pip install torch pandas numpy matplotlib scikit-learn cvxpy osqp
```

### 4.2 启动

```bash
cd Full_Model
python main.py
```

### 4.3 输出

所有结果保存至 `Full_Model/outputs/`：

| 文件 | 内容 |
|------|------|
| `strategy_comparison.csv` | 原始 KPI 对比表 |
| `strategy_ranked.csv` | 加权评分排名 |
| `ts_<策略名>.csv` | 每个策略的逐小时时序（12 列） |
| `kpi_bars.png` | 4 维度 KPI 柱状对比图 |
| `timeseries.png` | 前 168h 的 grid / SOC / CO₂ / 成本时序 |
| `pareto.png` | 成本 vs 碳排 Pareto 散点图 |
| `final_report.txt` | 文本综合分析报告（含推荐策略详情） |

### 4.4 关键配置项（修改 `main.py` 中的 `PlatformConfig` 类）

```python
SIM_HOURS          = 24 * 30       # 仿真时长（小时）
MPC_HORIZON        = 24            # MPC 滚动窗口
BESS_CAPACITY_KWH  = 20000         # 储能容量
BESS_POWER_KW      = 16000         # 储能功率
GRID_REGION        = 'east'        # 电网排放因子区域
CARBON_PRICE       = 100.0         # 碳价 元/tCO₂
CONFORMAL_ALPHA    = 0.10          # 保形预测 α（90% 覆盖率）
SCORE_WEIGHTS      = {'cost': 0.40, 'co2': 0.35, 'peak': 0.25}
```

---

## 五、调用依赖关系图

```
                  main.py
                     │
      ┌──────────────┼─────────────────────────┐
      │              │                          │
   All_Models    enhanced_ies              analysis / carbon_module /
   _EGrid_Paper      │                     conformal_predictor
                     │                            │
                  IES.py                      strategies.py ←── enhanced_ies 调用
                     │                            │
      ┌──────┬──────┴──────┬──────┐               │
    BES.py  EO.py      REO.py   EF.py          (自包含 + cvxpy)
      │
  convert_model.py ←── main.py 用于权重转换
```

---

## 六、平台扩展建议（对应报告 P1-P3）

如果后续想进一步扩展，可以参考：

| 报告章节 | 扩展方向 | 新增模块建议 |
|---------|---------|------------|
| §二.2.2 场景2 | 船舶到港 + 岸电联合调度 | `shore_power_scheduler.py` |
| §三.3.1 | 多任务联合预测负荷+PV+风电 | 修改 `EModel_FeatureWeight4` 增加多头 |
| §三.3.4 | RL 替代 CVXPY 处理非线性退化 | `rl_dispatcher.py`（SB3 + Gym） |
| §四.4.1 | 数字孪生 | `digital_twin/` 目录 |
| §四.4.3 | 多智能体 ADMM 去中心化调度 | `multi_agent/` |
| §四.4.4 | 氢能耦合 | `hydrogen_module.py` |

---

## 七、与调研报告的对应关系一览

| 报告要求 | 是否实现 | 实现位置 |
|---------|--------|---------|
| P0: MPC 滚动时域 + 参数化 CVXPY | ✅ | `strategies.py::_ParametrizedMPC` |
| P0: 保形预测不确定性量化 | ✅ | `conformal_predictor.py` + `RobustMPCStrategy` |
| P1: 多目标成本 + 碳排 | ✅ | `CarbonAwareMPCStrategy` + `carbon_module.py` |
| P1: 电网排放因子 + 动态碳强度 | ✅ | `make_dynamic_carbon_intensity` |
| §六 经济性 + 环保性综合分析 | ✅ | `analysis.py` 三函数 + Pareto 图 + 文本报告 |
| §五.5.1 架构升级 | 部分 | 已模块化，尚未做 FastAPI 微服务 |
| §三.3.4 RL 调度 | 未做 | 预留接口 |
| §三.3.1 多任务学习 | 未做 | 架构支持，后续可改 |
