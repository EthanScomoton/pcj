"""
经济性 + 环保性 + 技术性 综合分析模块
=====================================
对应调研报告 §五、§六：
- 经济性: 总成本、节省率、NPV、IRR、回收期
- 环保性: CO2 总量、减排率、可再生自消费率、等效植树
- 技术性: 峰值削减、SOC 利用、循环次数

提供 build_comparison_table / score_strategies / 可视化接口。
"""
import numpy as np
import pandas as pd

from EF import calculate_economic_metrics


# =======================================================================
# 单策略 KPI 计算
# =======================================================================
def compute_economic_kpis(results_df: pd.DataFrame,
                          investment_cost: float = 0.0,
                          lifetime: int = 10,
                          discount_rate: float = 0.05,
                          baseline_total_cost: float = None):
    total_cost = float(results_df['cost'].sum())
    sim_hours  = int(len(results_df))
    kpi = {
        'total_cost_CNY':       total_cost,
        'simulation_hours':     sim_hours,
        'avg_hourly_cost_CNY':  total_cost / max(1, sim_hours),
        'total_grid_kwh':       float(results_df['grid_import'].sum()),
    }
    # NPV/IRR 仅在提供基准与投资成本时计算
    if baseline_total_cost is not None and investment_cost > 0:
        econ = calculate_economic_metrics(
            costs=[baseline_total_cost, total_cost],
            investment_cost=investment_cost,
            discount_rate=discount_rate,
            lifetime=lifetime,
            simulation_hours=sim_hours,
        )
        kpi.update({
            'NPV_CNY':         econ['NPV'],
            'payback_period':  econ['payback_period'],
            'IRR':             econ['IRR'],
            'annual_savings_CNY': econ['annual_savings'],
            'investment_cost_CNY': investment_cost,
        })
    return kpi


def compute_environmental_kpis(results_df: pd.DataFrame,
                               carbon_price_cny_per_ton: float = 100.0):
    total_co2_kg = float(results_df['co2_kg'].sum())
    total_re_kwh = float(results_df['renewable_generation'].sum())
    total_demand = float(results_df['actual_demand'].sum())
    return {
        'total_CO2_kg':                  total_co2_kg,
        'total_CO2_tons':                total_co2_kg / 1000.0,
        'total_renewable_kwh':           total_re_kwh,
        'renewable_self_consumption_rate': (total_re_kwh / total_demand * 100.0)
                                           if total_demand > 0 else 0.0,
        'carbon_cost_CNY':               (total_co2_kg / 1000.0) * carbon_price_cny_per_ton,
        'equivalent_trees_year':         (total_co2_kg / 1000.0) * 45.0,
        'equivalent_cars_year':          (total_co2_kg / 1000.0) / 4.6,
    }


def compute_technical_kpis(results_df: pd.DataFrame):
    bp = results_df['bess_power'].values
    soc = results_df['bess_soc'].values
    sign = np.sign(bp)
    # 去除 0 位置不计入切换
    nonzero = sign[sign != 0]
    transitions = int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
    cycles = transitions / 2.0
    return {
        'peak_demand_kW':        float(results_df['grid_import'].max()),
        'avg_soc':               float(np.mean(soc)),
        'min_soc':               float(np.min(soc)),
        'max_soc':               float(np.max(soc)),
        'soc_range':             float(np.max(soc) - np.min(soc)),
        'estimated_cycles':      float(cycles),
        'total_charge_kwh':      float(np.sum(np.maximum(-bp, 0.0))),
        'total_discharge_kwh':   float(np.sum(np.maximum( bp, 0.0))),
    }


# =======================================================================
# 多策略对比
# =======================================================================
def build_comparison_table(strategy_results: dict,
                           baseline_name: str = "Baseline (No Storage)") -> pd.DataFrame:
    baseline = strategy_results.get(baseline_name)
    rows = []
    for name, data in strategy_results.items():
        eco, env, tech = data['economic'], data['environmental'], data['technical']
        row = {
            'Strategy':              name,
            'Total Cost (CNY)':      round(eco['total_cost_CNY'], 2),
            'CO2 (tons)':            round(env['total_CO2_tons'], 3),
            'Carbon Cost (CNY)':     round(env['carbon_cost_CNY'], 2),
            'Peak Grid (kW)':        round(tech['peak_demand_kW'], 2),
            'Renewable SCR (%)':     round(env['renewable_self_consumption_rate'], 2),
            'BESS Cycles':           round(tech['estimated_cycles'], 1),
            'Avg SOC':               round(tech['avg_soc'], 3),
            'NPV (CNY)':             round(eco.get('NPV_CNY', np.nan), 2),
            'Payback (yr)':          round(eco.get('payback_period', np.nan), 2) \
                                     if eco.get('payback_period') not in (None, np.inf) else np.inf,
        }

        if baseline is not None and name != baseline_name:
            bc  = baseline['economic']['total_cost_CNY']
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


def score_strategies(comparison_df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    if weights is None:
        weights = {'cost': 0.4, 'co2': 0.35, 'peak': 0.25}
    df = comparison_df.copy()

    def _norm(series):
        lo, hi = float(series.min()), float(series.max())
        rng = hi - lo
        if rng < 1e-9:
            return np.zeros(len(series))
        return (series - lo) / rng

    df['score'] = (weights['cost'] * _norm(df['Cost Savings (%)']) +
                   weights['co2']  * _norm(df['CO2 Reduction (%)']) +
                   weights['peak'] * _norm(df['Peak Reduction (%)']))
    return df.sort_values('score', ascending=False).reset_index(drop=True)


# =======================================================================
# 可视化
# =======================================================================
def plot_strategy_kpis(strategy_results: dict, save_path: str = None):
    import matplotlib.pyplot as plt

    names = list(strategy_results.keys())
    cost  = [d['economic']['total_cost_CNY']                 for d in strategy_results.values()]
    co2   = [d['environmental']['total_CO2_tons']            for d in strategy_results.values()]
    peak  = [d['technical']['peak_demand_kW']                for d in strategy_results.values()]
    scr   = [d['environmental']['renewable_self_consumption_rate']
             for d in strategy_results.values()]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2', '#72B7B2']
    colors = [palette[i % len(palette)] for i in range(len(names))]

    axes[0, 0].bar(names, cost, color=colors)
    axes[0, 0].set_title('Total Operating Cost (CNY)')
    axes[0, 1].bar(names, co2,  color='#2ca02c')
    axes[0, 1].set_title('Total CO₂ Emissions (tons)')
    axes[1, 0].bar(names, peak, color='#d62728')
    axes[1, 0].set_title('Peak Grid Demand (kW)')
    axes[1, 1].bar(names, scr,  color='#ff7f0e')
    axes[1, 1].set_title('Renewable Self-Consumption Rate (%)')

    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Strategy Comparison: Economic vs Environmental vs Technical',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] KPI 对比图已保存: {save_path}")
    plt.show()


def plot_time_series(strategy_results: dict, save_path: str = None, max_hours: int = 168):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    for name, d in strategy_results.items():
        ts = d['timeseries'].head(max_hours)
        axes[0].plot(ts['timestamp'], ts['grid_import'],     label=name, alpha=0.8)
        axes[1].plot(ts['timestamp'], ts['bess_soc'] * 100,  label=name, alpha=0.8)
        axes[2].plot(ts['timestamp'], ts['co2_kg'].cumsum() / 1000.0, label=name, alpha=0.8)
        axes[3].plot(ts['timestamp'], ts['cost'].cumsum(),   label=name, alpha=0.8)

    axes[0].set_ylabel('Grid Import (kW)')
    axes[0].set_title(f'Strategy Comparison — First {max_hours}h')
    axes[1].set_ylabel('SOC (%)'); axes[1].set_ylim(0, 100)
    axes[2].set_ylabel('Cumulative CO₂ (tons)')
    axes[3].set_ylabel('Cumulative Cost (CNY)')
    axes[3].set_xlabel('Time')

    for ax in axes:
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 时序对比图已保存: {save_path}")
    plt.show()


def plot_pareto_cost_vs_co2(strategy_results: dict, save_path: str = None):
    """Pareto 前沿：成本 vs CO2"""
    import matplotlib.pyplot as plt

    names = list(strategy_results.keys())
    cost  = [d['economic']['total_cost_CNY']      for d in strategy_results.values()]
    co2   = [d['environmental']['total_CO2_tons'] for d in strategy_results.values()]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(cost, co2, s=140, c='#d62728', zorder=3)
    for n, c, e in zip(names, cost, co2):
        ax.annotate(n, (c, e), textcoords='offset points', xytext=(8, 6), fontsize=9)

    ax.set_xlabel('Total Operating Cost (CNY)')
    ax.set_ylabel('Total CO₂ Emissions (tons)')
    ax.set_title('Pareto Frontier: Economic vs Environmental')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] Pareto 图已保存: {save_path}")
    plt.show()


# =======================================================================
# 文本报告
# =======================================================================
def format_final_report(strategy_results: dict,
                        scored_df: pd.DataFrame,
                        bess_config: dict,
                        sim_hours: int,
                        carbon_price: float) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("     港口综合能源系统 — 策略选择平台  最终分析报告")
    lines.append("=" * 80)
    lines.append(f"  仿真时长          : {sim_hours} 小时 ({sim_hours/24:.1f} 天)")
    lines.append(f"  储能配置          : {bess_config['capacity_kwh']} kWh / "
                 f"{bess_config['power_kw']} kW")
    lines.append(f"  碳价              : {carbon_price:.1f} 元/tCO₂")
    lines.append("-" * 80)

    lines.append("\n【经济性 & 环保性 综合排名（加权得分）】")
    lines.append(scored_df.to_string(index=False))

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
