"""
经济性 + 环保性 + 技术性 综合分析模块
=====================================
提供 9 种维度的可视化：
    1. plot_strategy_kpis        四象限柱状 (总成本 / CO2 / 峰值 / SCR)  带数值标注
    2. plot_time_series          时序曲线 (grid / SOC / 累计CO2 / 累计成本)
    3. plot_pareto_cost_vs_co2   Pareto 前沿  带智能避让标签
    4. plot_radar_chart          雷达图 —— 5 维归一化综合能力
    5. plot_improvement_heatmap  相对基准改进率热力图
    6. plot_cost_breakdown       成本结构堆叠条 (电费 + 碳成本)
    7. plot_load_duration_curves 负荷持续曲线 (按幅值降序)
    8. plot_daily_profile        日典型曲线 (24h 平均)
    9. plot_soc_distribution     SOC 分布小提琴图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =======================================================================
# 全局样式 —— 修复 CO₂ 显示 / 统一字号
# =======================================================================
CO2_LABEL = r'CO$_2$'

STRATEGY_ALIAS = {
    "Baseline (No Storage)":   "Baseline",
    "Rule-Based Peak Shaving": "Rule-PS",
    "Economic MPC":            "Econ-MPC",
    "Carbon-Aware MPC":        "Carbon-MPC",
    "Robust MPC (Conformal)":  "Robust-MPC",
}

COLOR_PALETTE = {
    "Baseline (No Storage)":   "#7f7f7f",
    "Rule-Based Peak Shaving": "#ff9f1c",
    "Economic MPC":            "#2e86ab",
    "Carbon-Aware MPC":        "#2ca02c",
    "Robust MPC (Conformal)":  "#a23b72",
}


def _apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.unicode_minus': False,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 13,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
    })


def _short(name):
    return STRATEGY_ALIAS.get(name, name)


def _color(name):
    return COLOR_PALETTE.get(name, '#4C78A8')


def _annotate_bars(ax, bars, values, fmt='{:.0f}', offset_frac=0.01):
    ymax = ax.get_ylim()[1]
    off = ymax * offset_frac
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + off,
                fmt.format(v),
                ha='center', va='bottom', fontsize=8.5, fontweight='medium')


# =======================================================================
# 单策略 KPI 计算
# =======================================================================
def compute_economic_kpis(results_df, investment_cost=0.0,
                          lifetime=10, discount_rate=0.05,
                          baseline_total_cost=None,
                          demand_charge_rate=0.0,
                          bess_capacity_kwh=0.0,
                          om_cost_per_kwh_year=10.0,
                          annual_degradation_pct=2.0):
    """
    经济性 KPI。改进点（对应问题 ① ③）：
      · 加入 **需量电费** (demand_charge_rate * peak_grid_kW)
      · NPV/IRR 考虑 **O&M 成本** 和 **容量衰减**
    """
    total_energy_cost = float(results_df['cost'].sum())
    peak_grid_kw = float(results_df['grid_import'].max())
    demand_charge = peak_grid_kw * demand_charge_rate      # 需量电费
    total_cost = total_energy_cost + demand_charge

    sim_hours = int(len(results_df))
    kpi = {
        'total_cost_CNY':       total_cost,
        'energy_cost_CNY':      total_energy_cost,
        'demand_charge_CNY':    demand_charge,
        'peak_grid_kW':         peak_grid_kw,
        'simulation_hours':     sim_hours,
        'avg_hourly_cost_CNY':  total_cost / max(1, sim_hours),
        'total_grid_kwh':       float(results_df['grid_import'].sum()),
    }
    if baseline_total_cost is not None and investment_cost > 0:
        # ---- 年化节省（含需量电费差） ----
        raw_annual_savings = ((baseline_total_cost - total_cost)
                              * (8760.0 / max(1, sim_hours)))
        annual_om = bess_capacity_kwh * om_cost_per_kwh_year

        # ---- 带衰减 + O&M 的现金流 ----
        cash_flows = [-investment_cost]
        for yr in range(1, lifetime + 1):
            deg = max(0.0, 1.0 - annual_degradation_pct / 100.0 * (yr - 1))
            cash_flows.append(raw_annual_savings * deg - annual_om)

        # NPV
        npv = sum(cf / (1 + discount_rate) ** t
                  for t, cf in enumerate(cash_flows))
        # 简单回收期
        first_year_net = raw_annual_savings - annual_om
        payback = (investment_cost / first_year_net
                   if first_year_net > 0 else float('inf'))

        # IRR（二分查找）
        def _npv_at(r, flows):
            return sum(cf / (1 + r) ** t for t, cf in enumerate(flows))

        irr = None
        total_cf = sum(cash_flows[1:])
        if total_cf > investment_cost:
            lo, hi = 0.0, 1.0
            for _ in range(120):
                mid = (lo + hi) / 2
                if _npv_at(mid, cash_flows) > 0:
                    lo = mid
                else:
                    hi = mid
            irr = lo
        elif abs(total_cf - investment_cost) < 1e-6:
            irr = 0.0
        elif first_year_net > 0:
            lo, hi = -0.99, 0.0
            for _ in range(120):
                mid = (lo + hi) / 2
                if _npv_at(mid, cash_flows) > 0:
                    lo = mid
                else:
                    hi = mid
            irr = hi
        else:
            irr = -1.0

        kpi.update({
            'NPV_CNY':             npv,
            'payback_period':      payback,
            'IRR':                 irr,
            'annual_savings_CNY':  first_year_net,
            'annual_om_CNY':       annual_om,
            'investment_cost_CNY': investment_cost,
        })
    return kpi


def compute_environmental_kpis(results_df, carbon_price_cny_per_ton=100.0):
    total_co2_kg = float(results_df['co2_kg'].sum())
    total_re_kwh = float(results_df['renewable_generation'].sum())
    total_demand = float(results_df['actual_demand'].sum())
    total_grid   = float(results_df['grid_import'].sum())

    # ---- 修正 grid_independence_rate（问题 ④）----
    # 旧公式: (demand - grid) / demand → 因往返损耗导致储能反而降低独立率
    # 新公式: 每小时 min(demand, renewable + discharge) / demand → 统计本地资源实际覆盖率
    re_arr = results_df['renewable_generation'].values
    bp_arr = results_df['bess_power'].values
    dm_arr = results_df['actual_demand'].values
    local_supply = re_arr + np.maximum(bp_arr, 0.0)      # 可再生 + 储能放电
    local_served = np.minimum(dm_arr, local_supply)       # 实际本地满足量
    grid_indep = (float(local_served.sum()) / total_demand * 100.0
                  if total_demand > 0 else 0.0)

    return {
        'total_CO2_kg':                    total_co2_kg,
        'total_CO2_tons':                  total_co2_kg / 1000.0,
        'total_renewable_kwh':             total_re_kwh,
        'renewable_self_consumption_rate': (total_re_kwh / total_demand * 100.0)
                                            if total_demand > 0 else 0.0,
        'grid_independence_rate':          grid_indep,
        'carbon_cost_CNY':                 (total_co2_kg / 1000.0) * carbon_price_cny_per_ton,
        'equivalent_trees_year':           (total_co2_kg / 1000.0) * 45.0,
        'equivalent_cars_year':            (total_co2_kg / 1000.0) / 4.6,
    }


def compute_technical_kpis(results_df):
    bp = results_df['bess_power'].values
    soc = results_df['bess_soc'].values
    sign = np.sign(bp)
    nz = sign[sign != 0]
    transitions = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    cycles = transitions / 2.0
    grid = results_df['grid_import'].values
    return {
        'peak_demand_kW':       float(np.max(grid)),
        'avg_demand_kW':        float(np.mean(grid)),
        'load_factor':          float(np.mean(grid) / np.max(grid)) if np.max(grid) > 0 else 0,
        'avg_soc':              float(np.mean(soc)),
        'min_soc':              float(np.min(soc)),
        'max_soc':              float(np.max(soc)),
        'soc_range':            float(np.max(soc) - np.min(soc)),
        'estimated_cycles':     float(cycles),
        'total_charge_kwh':     float(np.sum(np.maximum(-bp, 0.0))),
        'total_discharge_kwh':  float(np.sum(np.maximum(bp, 0.0))),
    }


# =======================================================================
# 多策略对比表
# =======================================================================
def build_comparison_table(strategy_results, baseline_name="Baseline (No Storage)"):
    baseline = strategy_results.get(baseline_name)
    rows = []
    for name, data in strategy_results.items():
        eco, env, tech = data['economic'], data['environmental'], data['technical']
        row = {
            'Strategy':              name,
            'Total Cost (CNY)':      round(eco['total_cost_CNY'], 2),
            'Energy Cost (CNY)':     round(eco.get('energy_cost_CNY', eco['total_cost_CNY']), 2),
            'Demand Charge (CNY)':   round(eco.get('demand_charge_CNY', 0), 2),
            'CO2 (tons)':            round(env['total_CO2_tons'], 3),
            'Carbon Cost (CNY)':     round(env['carbon_cost_CNY'], 2),
            'Peak Grid (kW)':        round(tech['peak_demand_kW'], 2),
            'Load Factor':           round(tech['load_factor'], 3),
            'Renewable SCR (%)':     round(env['renewable_self_consumption_rate'], 2),
            'Grid Indep. (%)':       round(env['grid_independence_rate'], 2),
            'BESS Cycles':           round(tech['estimated_cycles'], 1),
            'Avg SOC':               round(tech['avg_soc'], 3),
            'NPV (CNY)':             round(eco.get('NPV_CNY', np.nan), 2),
            'Payback (yr)':          (round(eco.get('payback_period', np.nan), 2)
                                      if eco.get('payback_period') not in (None, np.inf)
                                      else np.inf),
        }
        if baseline is not None and name != baseline_name:
            bc   = baseline['economic']['total_cost_CNY']
            bco2 = baseline['environmental']['total_CO2_tons']
            bpk  = baseline['technical']['peak_demand_kW']
            row['Cost Savings (%)']   = round(((bc - eco['total_cost_CNY']) / bc * 100)
                                              if bc > 0 else 0, 2)
            row['CO2 Reduction (%)']  = round(((bco2 - env['total_CO2_tons']) / bco2 * 100)
                                              if bco2 > 0 else 0, 2)
            row['Peak Reduction (%)'] = round(((bpk - tech['peak_demand_kW']) / bpk * 100)
                                              if bpk > 0 else 0, 2)
        else:
            row['Cost Savings (%)']   = 0.0
            row['CO2 Reduction (%)']  = 0.0
            row['Peak Reduction (%)'] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def score_strategies(comparison_df, weights=None):
    if weights is None:
        weights = {'cost': 0.4, 'co2': 0.35, 'peak': 0.25}
    df = comparison_df.copy()

    def _norm(s):
        lo, hi = float(s.min()), float(s.max())
        return np.zeros(len(s)) if (hi - lo) < 1e-9 else (s - lo) / (hi - lo)

    df['score'] = (weights['cost'] * _norm(df['Cost Savings (%)']) +
                   weights['co2']  * _norm(df['CO2 Reduction (%)']) +
                   weights['peak'] * _norm(df['Peak Reduction (%)']))
    return df.sort_values('score', ascending=False).reset_index(drop=True)


# =======================================================================
# 可视化 1: 4 象限 KPI 柱状图 (带数值标注)
# =======================================================================
def plot_strategy_kpis(strategy_results, save_path=None):
    _apply_style()
    names = list(strategy_results.keys())
    short = [_short(n) for n in names]
    colors = [_color(n) for n in names]

    cost = [d['economic']['total_cost_CNY']                   for d in strategy_results.values()]
    co2  = [d['environmental']['total_CO2_tons']              for d in strategy_results.values()]
    peak = [d['technical']['peak_demand_kW']                  for d in strategy_results.values()]
    gi   = [d['environmental']['grid_independence_rate']      for d in strategy_results.values()]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    bars = axes[0, 0].bar(short, cost, color=colors, edgecolor='black', linewidth=0.6)
    axes[0, 0].set_title('Total Operating Cost', fontweight='bold')
    axes[0, 0].set_ylabel('CNY')
    axes[0, 0].set_ylim(0, max(cost) * 1.15)
    _annotate_bars(axes[0, 0], bars, cost, fmt='{:,.0f}')

    bars = axes[0, 1].bar(short, co2, color=colors, edgecolor='black', linewidth=0.6)
    axes[0, 1].set_title(f'Total {CO2_LABEL} Emissions', fontweight='bold')
    axes[0, 1].set_ylabel('tons')
    axes[0, 1].set_ylim(0, max(co2) * 1.15)
    _annotate_bars(axes[0, 1], bars, co2, fmt='{:.1f}')

    bars = axes[1, 0].bar(short, peak, color=colors, edgecolor='black', linewidth=0.6)
    axes[1, 0].set_title('Peak Grid Demand', fontweight='bold')
    axes[1, 0].set_ylabel('kW')
    axes[1, 0].set_ylim(0, max(peak) * 1.15)
    _annotate_bars(axes[1, 0], bars, peak, fmt='{:,.0f}')

    bars = axes[1, 1].bar(short, gi, color=colors, edgecolor='black', linewidth=0.6)
    axes[1, 1].set_title('Grid Independence Rate', fontweight='bold')
    axes[1, 1].set_ylabel('%')
    axes[1, 1].set_ylim(0, max(gi) * 1.2 + 1)
    _annotate_bars(axes[1, 1], bars, gi, fmt='{:.2f}%')

    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=15)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment('right')

    fig.suptitle('Strategy KPI Comparison', fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] KPI 对比图: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 可视化 2: 时序对比
# =======================================================================
def plot_time_series(strategy_results, save_path=None, max_hours=168):
    _apply_style()
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                             constrained_layout=True)

    for name, d in strategy_results.items():
        ts = d['timeseries'].head(max_hours)
        c = _color(name)
        axes[0].plot(ts['timestamp'], ts['grid_import'],            label=_short(name), alpha=0.85, color=c, linewidth=1.2)
        axes[1].plot(ts['timestamp'], ts['bess_soc'] * 100,         label=_short(name), alpha=0.85, color=c, linewidth=1.2)
        axes[2].plot(ts['timestamp'], ts['co2_kg'].cumsum() / 1000, label=_short(name), alpha=0.85, color=c, linewidth=1.2)
        axes[3].plot(ts['timestamp'], ts['cost'].cumsum(),          label=_short(name), alpha=0.85, color=c, linewidth=1.2)

    axes[0].set_ylabel('Grid Import (kW)')
    axes[1].set_ylabel('SOC (%)'); axes[1].set_ylim(0, 100)
    axes[2].set_ylabel(f'Cumulative {CO2_LABEL} (tons)')
    axes[3].set_ylabel('Cumulative Cost (CNY)')
    axes[3].set_xlabel('Time')

    axes[0].set_title(f'Strategy Comparison — First {max_hours}h', fontweight='bold')

    for ax in axes:
        ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1.0),
                  frameon=True, framealpha=0.9)

    fig.autofmt_xdate(rotation=25)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 时序对比图: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 可视化 3: Pareto 前沿 (智能避让标签)
# =======================================================================
def plot_pareto_cost_vs_co2(strategy_results, save_path=None):
    _apply_style()
    names = list(strategy_results.keys())
    cost = np.array([d['economic']['total_cost_CNY']       for d in strategy_results.values()])
    co2  = np.array([d['environmental']['total_CO2_tons']  for d in strategy_results.values()])
    colors = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    # 绘点
    for i, n in enumerate(names):
        ax.scatter(cost[i], co2[i], s=220, c=colors[i], zorder=4,
                   edgecolors='black', linewidth=1.2, label=_short(n))

    # Pareto 前沿（最小化 cost 和 co2）
    order = np.argsort(cost)
    front_x, front_y = [], []
    best_y = np.inf
    for idx in order:
        if co2[idx] < best_y:
            best_y = co2[idx]
            front_x.append(cost[idx])
            front_y.append(co2[idx])
    ax.plot(front_x, front_y, '--', color='#c44569', alpha=0.7,
            linewidth=1.5, zorder=3, label='Pareto frontier')

    # 智能标签偏移（避免重叠）
    x_range = cost.max() - cost.min()
    y_range = co2.max() - co2.min()
    placed = []
    for i, n in enumerate(names):
        xo, yo = 0.015 * x_range, 0.02 * y_range
        # 检测近邻冲突
        for (px, py) in placed:
            if abs(cost[i] - px) < 0.05 * x_range and abs(co2[i] - py) < 0.05 * y_range:
                yo = -0.04 * y_range
                xo = 0.02 * x_range
                break
        placed.append((cost[i], co2[i]))
        ax.annotate(_short(n), (cost[i], co2[i]),
                    xytext=(cost[i] + xo, co2[i] + yo),
                    fontsize=10, fontweight='medium',
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', ec=colors[i], alpha=0.85, lw=1))

    ax.set_xlabel('Total Operating Cost (CNY)')
    ax.set_ylabel(f'Total {CO2_LABEL} Emissions (tons)')
    ax.set_title('Pareto Frontier: Economic vs Environmental', fontweight='bold')
    ax.legend(loc='best', frameon=True)

    # 扩展显示范围，避免标签贴边
    ax.set_xlim(cost.min() - 0.03 * x_range, cost.max() + 0.08 * x_range)
    ax.set_ylim(co2.min() - 0.05 * y_range, co2.max() + 0.08 * y_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] Pareto 图: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 可视化 4: 雷达图 —— 5 维综合能力
# =======================================================================
def plot_radar_chart(strategy_results, save_path=None, baseline_name="Baseline (No Storage)"):
    _apply_style()
    names = list(strategy_results.keys())

    # 5 维指标 （值越高越好）
    metrics = ['Cost Save', f'{CO2_LABEL} Reduce', 'Peak Shave',
               'Grid Indep.', 'Load Factor']

    def _build(d, baseline):
        bc  = baseline['economic']['total_cost_CNY']
        bco = baseline['environmental']['total_CO2_tons']
        bpk = baseline['technical']['peak_demand_kW']
        return [
            max(0, (bc - d['economic']['total_cost_CNY']) / bc * 100) if bc > 0 else 0,
            max(0, (bco - d['environmental']['total_CO2_tons']) / bco * 100) if bco > 0 else 0,
            max(0, (bpk - d['technical']['peak_demand_kW']) / bpk * 100) if bpk > 0 else 0,
            d['environmental']['grid_independence_rate'],
            d['technical']['load_factor'] * 100,
        ]

    baseline = strategy_results[baseline_name]
    values_by_strategy = {n: _build(d, baseline) for n, d in strategy_results.items()}

    # 归一化到 [0, 1]
    all_vals = np.array(list(values_by_strategy.values()))
    maxs = np.where(all_vals.max(axis=0) < 1e-9, 1.0, all_vals.max(axis=0))
    normed = {n: np.asarray(v) / maxs for n, v in values_by_strategy.items()}

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True),
                           constrained_layout=True)

    for name in names:
        if name == baseline_name:
            continue  # baseline 都是 0,  跳过
        vals = normed[name].tolist() + [normed[name][0]]
        c = _color(name)
        ax.plot(angles, vals, 'o-', linewidth=2, color=c, label=_short(name))
        ax.fill(angles, vals, color=c, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_title('Multi-Dimensional Strategy Capability (Normalized)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 雷达图: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 可视化 5: 改进率热力图
# =======================================================================
def plot_improvement_heatmap(comparison_df, save_path=None):
    _apply_style()
    cols = ['Cost Savings (%)', 'CO2 Reduction (%)', 'Peak Reduction (%)',
            'Grid Indep. (%)', 'Load Factor', 'Renewable SCR (%)']
    df = comparison_df.set_index('Strategy')[cols].copy()
    df.index = [_short(x) for x in df.index]

    fig, ax = plt.subplots(figsize=(11, max(3, 0.6 * len(df) + 1.5)),
                           constrained_layout=True)

    data = df.values.astype(float)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto',
                   vmin=-max(1, abs(data).max()), vmax=max(1, abs(data).max()))

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=22, ha='right')
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f'{v:.2f}',
                    ha='center', va='center',
                    color='black' if abs(v) < 0.6 * abs(data).max() else 'white',
                    fontsize=9, fontweight='medium')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Improvement vs Baseline (%)', fontsize=9)
    ax.set_title('Strategy Improvement Heatmap', fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 改进率热力图: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 可视化 6: 成本结构堆叠
# =======================================================================
def plot_cost_breakdown(strategy_results, save_path=None):
    _apply_style()
    names = list(strategy_results.keys())
    short = [_short(n) for n in names]
    energy   = [d['economic'].get('energy_cost_CNY', d['economic']['total_cost_CNY'])
                for d in strategy_results.values()]
    demand   = [d['economic'].get('demand_charge_CNY', 0)
                for d in strategy_results.values()]
    carbon   = [d['environmental']['carbon_cost_CNY']
                for d in strategy_results.values()]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    b1 = ax.bar(short, energy, color='#4C78A8', label='Energy Cost',
                edgecolor='black', linewidth=0.6)
    b2 = ax.bar(short, demand, bottom=energy, color='#F58518',
                label='Demand Charge', edgecolor='black', linewidth=0.6)
    bot3 = [e + d for e, d in zip(energy, demand)]
    b3 = ax.bar(short, carbon, bottom=bot3, color='#E45756',
                label=f'{CO2_LABEL} Cost', edgecolor='black', linewidth=0.6)

    total = [e + d + c for e, d, c in zip(energy, demand, carbon)]
    ymax = max(total) * 1.15
    ax.set_ylim(0, ymax)
    for i, t in enumerate(total):
        ax.text(i, t + ymax * 0.01, f'{t:,.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='medium')

    ax.set_ylabel('Cost (CNY)')
    ax.set_title(f'Total Cost Breakdown (Energy + Demand Charge + {CO2_LABEL})',
                 fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment('right')
    ax.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 成本结构图: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 可视化 7: 负荷持续曲线
# =======================================================================
def plot_load_duration_curves(strategy_results, save_path=None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    for name, d in strategy_results.items():
        sorted_grid = np.sort(d['timeseries']['grid_import'].values)[::-1]
        hours = np.arange(len(sorted_grid))
        ax.plot(hours, sorted_grid, label=_short(name),
                color=_color(name), linewidth=1.8, alpha=0.85)

    ax.set_xlabel('Hours (sorted by magnitude, descending)')
    ax.set_ylabel('Grid Import (kW)')
    ax.set_title('Load Duration Curves — Grid Import',
                 fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.axhline(0, color='k', lw=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 负荷持续曲线: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 可视化 8: 24h 日典型曲线
# =======================================================================
def plot_daily_profile(strategy_results, save_path=None):
    _apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             constrained_layout=True)

    for name, d in strategy_results.items():
        ts = d['timeseries'].copy()
        ts['hour'] = pd.to_datetime(ts['timestamp']).dt.hour
        grouped = ts.groupby('hour').agg({
            'grid_import': 'mean',
            'bess_soc':    'mean',
            'bess_power':  'mean',
        })
        c = _color(name)
        axes[0].plot(grouped.index, grouped['grid_import'],
                     marker='o', markersize=4, linewidth=1.8,
                     label=_short(name), color=c, alpha=0.85)
        axes[1].plot(grouped.index, grouped['bess_soc'] * 100,
                     marker='o', markersize=4, linewidth=1.8,
                     label=_short(name), color=c, alpha=0.85)
        axes[2].plot(grouped.index, grouped['bess_power'],
                     marker='o', markersize=4, linewidth=1.8,
                     label=_short(name), color=c, alpha=0.85)

    axes[0].set_ylabel('Avg Grid Import (kW)')
    axes[0].set_title('24-Hour Average Daily Profile', fontweight='bold')
    axes[1].set_ylabel('Avg SOC (%)'); axes[1].set_ylim(0, 100)
    axes[2].set_ylabel('Avg BESS Power (kW)\n(+disch / −charge)')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_xlabel('Hour of Day')
    axes[2].set_xticks(range(0, 24, 2))

    for ax in axes:
        ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1.0), frameon=True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 日典型曲线: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 可视化 9: SOC 分布小提琴图
# =======================================================================
def plot_soc_distribution(strategy_results, save_path=None):
    _apply_style()
    names, soc_list, colors = [], [], []
    for n, d in strategy_results.items():
        if d['technical']['soc_range'] < 1e-6:
            continue  # 无储能策略跳过
        names.append(_short(n))
        soc_list.append(d['timeseries']['bess_soc'].values * 100)
        colors.append(_color(n))

    if not soc_list:
        print("  [plot] SOC 分布：无储能策略有意义数据，跳过")
        return

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    parts = ax.violinplot(soc_list, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
    for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if key in parts:
            parts[key].set_edgecolor('black')
            parts[key].set_linewidth(0.9)

    ax.axhline(10, color='red', ls='--', alpha=0.5, label='min SOC 10%')
    ax.axhline(90, color='red', ls='--', alpha=0.5, label='max SOC 90%')
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('SOC (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Battery SOC Distribution', fontweight='bold')
    ax.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] SOC 分布图: {save_path}")
    plt.show()
    plt.close(fig)


# =======================================================================
# 一键生成全部图
# =======================================================================
def generate_all_plots(strategy_results, comparison_df, output_dir,
                       max_ts_hours=168):
    os.makedirs(output_dir, exist_ok=True)
    print("\n  >> 生成可视化 ...")
    plot_strategy_kpis       (strategy_results,  os.path.join(output_dir, '01_kpi_bars.png'))
    plot_cost_breakdown      (strategy_results,  os.path.join(output_dir, '02_cost_breakdown.png'))
    plot_pareto_cost_vs_co2  (strategy_results,  os.path.join(output_dir, '03_pareto.png'))
    plot_improvement_heatmap (comparison_df,     os.path.join(output_dir, '04_improvement_heatmap.png'))
    plot_radar_chart         (strategy_results,  os.path.join(output_dir, '05_radar.png'))
    plot_load_duration_curves(strategy_results,  os.path.join(output_dir, '06_load_duration.png'))
    plot_daily_profile       (strategy_results,  os.path.join(output_dir, '07_daily_profile.png'))
    plot_soc_distribution    (strategy_results,  os.path.join(output_dir, '08_soc_distribution.png'))
    plot_time_series         (strategy_results,  os.path.join(output_dir, '09_timeseries.png'),
                              max_hours=max_ts_hours)


# =======================================================================
# 文本报告
# =======================================================================
def format_final_report(strategy_results, scored_df, bess_config,
                        sim_hours, carbon_price,
                        demand_charge_rate=0.0):
    lines = []
    lines.append("=" * 80)
    lines.append("     港口综合能源系统 — 策略选择平台  最终分析报告")
    lines.append("=" * 80)
    lines.append(f"  仿真时长     : {sim_hours} 小时 ({sim_hours/24:.1f} 天)")
    lines.append(f"  储能配置     : {bess_config['capacity_kwh']} kWh / "
                 f"{bess_config['power_kw']} kW")
    lines.append(f"  碳价         : {carbon_price:.1f} 元/tCO2")
    lines.append(f"  需量电费     : {demand_charge_rate:.1f} 元/kW/月")
    lines.append("-" * 80)

    lines.append("\n【经济+环保+技术 综合排名】")
    cols_show = ['Strategy', 'Cost Savings (%)', 'CO2 Reduction (%)',
                 'Peak Reduction (%)', 'Grid Indep. (%)', 'score']
    lines.append(scored_df[cols_show].to_string(index=False))

    best_name = scored_df.iloc[0]['Strategy']
    best = strategy_results[best_name]
    lines.append("\n" + "=" * 80)
    lines.append(f"  推荐策略: {best_name}")
    lines.append("=" * 80)
    lines.append(" · 经济性指标")
    for k, v in best['economic'].items():
        lines.append(f"     {k:30s}: {v}")
    lines.append(" · 环保性指标")
    for k, v in best['environmental'].items():
        lines.append(f"     {k:30s}: {v}")
    lines.append(" · 技术性指标")
    for k, v in best['technical'].items():
        lines.append(f"     {k:30s}: {v}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
