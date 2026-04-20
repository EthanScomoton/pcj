"""
高级分析实验模块
=================
8 个实验，对应论文扩展分析与敏感性研究：
  1. CP 覆盖率实证验证
  2. CP vs 重参数化 UQ 对比
  3. 预测误差注入鲁棒性实验
  4. 碳价敏感性分析
  5. MPC 计算时间分析
  6. BESS 容量优化 — NPV/IRR 曲线
  7. TOU 峰谷比敏感性
  8. 极端事件案例分析
"""
import os, time, json, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from enhanced_ies import StrategyAwareIES
from strategies import (
    EconomicMPCStrategy, CarbonAwareMPCStrategy,
    RobustMPCStrategy, BaselineStrategy, _ParametrizedMPC,
)
from conformal_predictor import ConformalPredictor
from carbon_module import CarbonTracker
from analysis import (
    compute_economic_kpis, compute_environmental_kpis,
    compute_technical_kpis, _apply_style, _short, _color,
    CO2_LABEL, COLOR_PALETTE,
)


# =====================================================================
# 公共工具
# =====================================================================
def _season(month):
    if month in (3, 4, 5):   return 'Spring'
    if month in (6, 7, 8):   return 'Summer'
    if month in (9, 10, 11): return 'Autumn'
    return 'Winter'


def _hour_bin(h):
    if h < 6:   return 'Night 0-6'
    if h < 12:  return 'Morning 6-12'
    if h < 18:  return 'Afternoon 12-18'
    return 'Evening 18-24'


def _run_sim(strategy, data_df, predictions_by_index, model, cfg,
             price_df, carbon_intensity_hourly, carbon_tracker,
             conformal, scaler_X, scaler_y, feature_cols, sim_hours,
             capacity_kwh=None, bess_power_kw=None):
    """快速仿真辅助——复用预计算缓存。"""
    cap = capacity_kwh if capacity_kwh is not None else cfg.BESS_CAPACITY_KWH
    pwr = bess_power_kw if bess_power_kw is not None else cfg.BESS_POWER_KW
    ies = StrategyAwareIES(
        capacity_kwh=cap, bess_power_kw=pwr,
        prediction_model=model,
        feature_cols=feature_cols,
        scaler_X=scaler_X, scaler_y=scaler_y,
        strategy=strategy,
        conformal_predictor=(conformal if isinstance(strategy, RobustMPCStrategy)
                             else None),
        carbon_tracker=carbon_tracker,
        carbon_intensity_hourly=carbon_intensity_hourly,
        allow_grid_export=False, verbose=False,
    )
    ts = ies.simulate_with_strategy(
        historic_data=data_df, time_steps=sim_hours,
        price_data=price_df,
        predictions_by_index=predictions_by_index,
        horizon=cfg.MPC_HORIZON,
    )
    return ts


def _make_price_series(timestamps, peak, flat, valley, super_peak=None):
    """按 main.py 时段结构生成电价序列。"""
    sp = super_peak if super_peak is not None else peak
    prices = []
    for ts in timestamps:
        h, wd = ts.hour, ts.weekday()
        if 10 <= h < 12 or 14 <= h < 17:
            p = sp
        elif 8 <= h < 10 or 12 <= h < 14 or 17 <= h < 19:
            p = peak
        elif 19 <= h < 22 or 6 <= h < 8:
            p = flat
        else:
            p = valley
        if wd >= 5:
            p *= 0.9
        prices.append(p)
    return pd.DataFrame({'timestamp': timestamps, 'price': prices})


# =====================================================================
# 实验 1 — CP 覆盖率实证验证
# =====================================================================
def experiment_cp_coverage(conformal, data_df, predictions_by_index,
                           sim_hours, output_dir):
    print("  [Exp 1] CP 覆盖率实证验证 ...")
    _apply_style()

    actuals = data_df['E_total'].values[:sim_hours]
    preds   = predictions_by_index[:sim_hours].copy()
    q90 = conformal.q_hat
    # 计算 95% q_hat
    res = conformal.calibration_residuals
    m = len(res)
    q95_level = min(np.ceil((m + 1) * 0.95) / m, 1.0)
    q95 = float(np.quantile(res, q95_level))

    covered_90 = ((actuals >= preds - q90) & (actuals <= preds + q90))
    covered_95 = ((actuals >= preds - q95) & (actuals <= preds + q95))
    emp_90 = covered_90.mean() * 100
    emp_95 = covered_95.mean() * 100

    # 条件覆盖率
    ts = data_df['timestamp'].values[:sim_hours]
    months = pd.to_datetime(ts).month
    hours  = pd.to_datetime(ts).hour
    seasons = np.array([_season(m) for m in months])
    hour_bins = np.array([_hour_bin(h) for h in hours])
    load_terc = pd.qcut(actuals, 3, labels=['Low', 'Mid', 'High'])

    cond = {}
    for label, arr in [('Season', seasons), ('Hour', hour_bins),
                       ('Load', np.asarray(load_terc))]:
        cond[label] = {}
        for g in np.unique(arr):
            mask = arr == g
            if mask.sum() > 0:
                cond[label][g] = covered_90[mask].mean() * 100

    # 区间宽度分布
    widths = 2 * q90 * np.ones(sim_hours)  # 固定宽度

    # ---- 图 1: 条件覆盖率 (按季节/时段/负荷水平) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle('CP Empirical Coverage (90% Nominal)', fontweight='bold')
    for ax, (label, cdict) in zip(axes, cond.items()):
        names = list(cdict.keys())
        vals  = [cdict[n] for n in names]
        bars = ax.bar(names, vals, color='#2e86ab', edgecolor='black', linewidth=0.5)
        ax.axhline(90, color='red', ls='--', lw=1.2, label='Nominal 90%')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'By {label}')
        ax.set_ylim(70, 105)
        ax.legend(fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.5,
                    f'{v:.1f}%', ha='center', fontsize=8)
    plt.savefig(os.path.join(output_dir, 'exp01_cp_coverage_by_condition.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 2: 校准残差分布 + 分位阈值 ----
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(conformal.calibration_residuals, bins=50, color='#4C78A8',
            edgecolor='black', alpha=0.8)
    ax.axvline(q90, color='red', ls='--', lw=1.5, label=f'q(90%)={q90:.0f}')
    ax.axvline(q95, color='orange', ls='--', lw=1.5, label=f'q(95%)={q95:.0f}')
    ax.set_xlabel('Absolute Residual (kW)')
    ax.set_ylabel('Count')
    ax.set_title('Calibration Residual Distribution & Quantile Thresholds',
                 fontweight='bold')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'exp01_cp_residual_dist.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 3: 滑动窗口覆盖率时序 (窗口=24h) ----
    win = 24
    rolling_cov = np.convolve(covered_90.astype(float),
                              np.ones(win) / win, mode='valid') * 100
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.plot(rolling_cov, color='#2e86ab', lw=1.2, alpha=0.85)
    ax.axhline(90, color='red', ls='--', lw=1.2, label='Nominal 90%')
    ax.fill_between(range(len(rolling_cov)), rolling_cov, 90,
                    where=rolling_cov < 90, alpha=0.25, color='red',
                    label='Under-coverage')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Rolling 24h Coverage (%)')
    ax.set_title('Sliding-Window Coverage Rate over Simulation',
                 fontweight='bold')
    ax.set_ylim(50, 105)
    ax.legend(fontsize=9)
    plt.savefig(os.path.join(output_dir, 'exp01_cp_rolling_coverage.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 4: 预测区间时序可视化 (展示 168h / 一周) ----
    vis_len = min(168, sim_hours)
    x_ax = np.arange(vis_len)
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.plot(x_ax, actuals[:vis_len], 'k-', lw=1.3, label='Actual', zorder=3)
    ax.plot(x_ax, preds[:vis_len], '--', color='#2e86ab', lw=1.0, label='Predicted')
    ax.fill_between(x_ax, preds[:vis_len] - q90, preds[:vis_len] + q90,
                    alpha=0.22, color='#2e86ab', label='CP 90% Band')
    ax.fill_between(x_ax, preds[:vis_len] - q95, preds[:vis_len] + q95,
                    alpha=0.10, color='#F58518', label='CP 95% Band')
    # 标记未覆盖点
    miss_mask = ~covered_90[:vis_len]
    if miss_mask.any():
        ax.scatter(x_ax[miss_mask], actuals[:vis_len][miss_mask],
                   color='red', s=18, zorder=4, label='Miss (90%)')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Demand (kW)')
    ax.set_title('Prediction Intervals with Coverage Misses (First Week)',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=3)
    plt.savefig(os.path.join(output_dir, 'exp01_cp_interval_timeseries.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 5: 实际 vs 预测散点 + 覆盖着色 ----
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(preds[covered_90], actuals[covered_90],
               s=6, alpha=0.3, color='#2e86ab', label=f'Covered ({emp_90:.1f}%)')
    ax.scatter(preds[~covered_90], actuals[~covered_90],
               s=12, alpha=0.7, color='red', label=f'Miss ({100-emp_90:.1f}%)')
    lims = [min(actuals.min(), preds.min()) * 0.9,
            max(actuals.max(), preds.max()) * 1.1]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Predicted Demand (kW)')
    ax.set_ylabel('Actual Demand (kW)')
    ax.set_title('Actual vs Predicted — Coverage Coloring', fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    plt.savefig(os.path.join(output_dir, 'exp01_cp_scatter_coverage.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    result = {
        'empirical_coverage_90': emp_90,
        'empirical_coverage_95': emp_95,
        'q_hat_90': q90, 'q_hat_95': q95,
        'conditional_coverage': cond,
    }
    print(f"    覆盖率: 90% nominal -> {emp_90:.1f}%, 95% nominal -> {emp_95:.1f}%")
    print(f"    -> 5 张图已保存")
    return result


# =====================================================================
# 实验 2 — CP vs 重参数化 UQ 对比
# =====================================================================
def experiment_cp_vs_reparam(model, conformal, data_df,
                             predictions_by_index,
                             scaler_X, scaler_y, feature_cols,
                             sim_hours, output_dir, device=None,
                             n_sample=200):
    print("  [Exp 2] CP vs 重参数化 UQ 对比 ...")
    _apply_style()
    if device is None:
        device = next(model.parameters()).device

    actuals = data_df['E_total'].values[:sim_hours]
    preds   = predictions_by_index[:sim_hours].copy()
    q90 = conformal.q_hat

    # 提取 mu 和 logvar
    sample_idx = np.linspace(0, sim_hours - 1, min(n_sample, sim_hours), dtype=int)
    window_size = getattr(model, 'window_size', 20)

    mu_arr, sigma_arr = [], []
    model.eval()
    for idx in sample_idx:
        start = max(0, idx - window_size + 1)
        seq = data_df[feature_cols].iloc[start:idx + 1].values.astype(np.float32)
        if len(seq) < window_size:
            seq = np.pad(seq, ((window_size - len(seq), 0), (0, 0)), mode='edge')
        seq_scaled = scaler_X.transform(seq).astype(np.float32)
        x = torch.tensor(seq_scaled[np.newaxis], dtype=torch.float32).to(device)

        # hook 拦截 fc 层输出 (shape [1, 2])
        fc_output = {}
        def _hook(mod, inp, out):
            fc_output['val'] = out.detach().cpu()
        handle = model.fc.register_forward_hook(_hook)
        with torch.no_grad():
            _ = model(x)
        handle.remove()

        raw = fc_output['val'].numpy().flatten()
        mu_raw, logvar_raw = raw[0], raw[1]
        # 反归一化 mu
        if scaler_y is not None:
            mu_real = scaler_y.inverse_transform([[mu_raw]])[0, 0]
            mu_real = np.expm1(mu_real)
        else:
            mu_real = mu_raw
        sigma_real = 0.1 * np.exp(0.5 * logvar_raw) * mu_real * 0.5  # 近似缩放
        mu_arr.append(mu_real)
        sigma_arr.append(max(sigma_real, 1.0))

    mu_arr = np.array(mu_arr)
    sigma_arr = np.array(sigma_arr)
    actuals_s = actuals[sample_idx]
    preds_s   = preds[sample_idx]

    # 重参数化 90% 区间 (Gaussian z=1.645)
    reparam_lower = mu_arr - 1.645 * sigma_arr
    reparam_upper = mu_arr + 1.645 * sigma_arr
    cov_reparam = ((actuals_s >= reparam_lower) & (actuals_s <= reparam_upper)).mean() * 100

    # CP 90% 区间
    cp_lower = preds_s - q90
    cp_upper = preds_s + q90
    cov_cp = ((actuals_s >= cp_lower) & (actuals_s <= cp_upper)).mean() * 100

    width_reparam = (reparam_upper - reparam_lower).mean()
    width_cp = 2 * q90

    # ---- 图 1: 覆盖率 + 区间宽度联合对比 (双柱状图) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle('CP vs Reparameterization UQ Comparison', fontweight='bold', fontsize=13)
    methods = ['Reparam\n(uncalibrated)', 'Conformal\nPrediction']
    colors_m = ['#E45756', '#2e86ab']
    # 左: 覆盖率
    bars = axes[0].bar(methods, [cov_reparam, cov_cp],
                       color=colors_m, edgecolor='black', linewidth=0.6, width=0.5)
    axes[0].axhline(90, color='gray', ls='--', lw=1.2, label='Nominal 90%')
    for b, v in zip(bars, [cov_reparam, cov_cp]):
        axes[0].text(b.get_x() + b.get_width()/2, v + 1,
                     f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Empirical Coverage (%)')
    axes[0].set_title('Coverage Rate')
    axes[0].set_ylim(0, 110)
    axes[0].legend(fontsize=9)
    # 右: 平均区间宽度
    bars2 = axes[1].bar(methods, [width_reparam, width_cp],
                        color=colors_m, edgecolor='black', linewidth=0.6, width=0.5)
    for b, v in zip(bars2, [width_reparam, width_cp]):
        axes[1].text(b.get_x() + b.get_width()/2, v + v*0.02,
                     f'{v:.0f}', ha='center', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Interval Width (kW)')
    axes[1].set_title('Interval Sharpness')
    plt.savefig(os.path.join(output_dir, 'exp02_uq_coverage_width.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 2: 区间宽度分布箱线图 + 小提琴 ----
    reparam_widths = reparam_upper - reparam_lower
    cp_widths_arr = np.full(len(sample_idx), 2 * q90)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    vp = ax.violinplot([reparam_widths, cp_widths_arr], positions=[1, 2],
                       showmedians=True, showextrema=True)
    vp['bodies'][0].set_facecolor('#E45756')
    vp['bodies'][0].set_alpha(0.6)
    vp['bodies'][1].set_facecolor('#2e86ab')
    vp['bodies'][1].set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Reparam', 'CP'])
    ax.set_ylabel('Interval Width (kW)')
    ax.set_title('Interval Width Distribution (Violin)', fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'exp02_uq_width_violin.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 3: 168h 时序区间可视化 ----
    vis_len = min(168, len(sample_idx))
    vis_idx = sample_idx[:vis_len]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True, sharex=True)
    fig.suptitle('Prediction Intervals: CP vs Reparameterization', fontweight='bold')
    x_axis = np.arange(vis_len)
    # 上图: CP
    axes[0].plot(x_axis, actuals[vis_idx], 'k-', lw=1.3, label='Actual', zorder=3)
    axes[0].plot(x_axis, preds[vis_idx], '--', color='#2e86ab', lw=0.9, alpha=0.7)
    axes[0].fill_between(x_axis, preds[vis_idx] - q90, preds[vis_idx] + q90,
                         alpha=0.3, color='#2e86ab', label=f'CP 90% (w={2*q90:.0f} kW)')
    miss_cp = ~((actuals[vis_idx] >= preds[vis_idx] - q90) &
                (actuals[vis_idx] <= preds[vis_idx] + q90))
    if miss_cp.any():
        axes[0].scatter(x_axis[miss_cp], actuals[vis_idx][miss_cp],
                        color='red', s=14, zorder=4, label='Miss')
    axes[0].set_ylabel('Demand (kW)')
    axes[0].set_title('Conformal Prediction')
    axes[0].legend(fontsize=8, ncol=3)
    # 下图: Reparam
    axes[1].plot(x_axis, actuals[vis_idx], 'k-', lw=1.3, label='Actual', zorder=3)
    axes[1].plot(x_axis, mu_arr[:vis_len], '--', color='#E45756', lw=0.9, alpha=0.7)
    axes[1].fill_between(x_axis, reparam_lower[:vis_len], reparam_upper[:vis_len],
                         alpha=0.3, color='#E45756',
                         label=f'Reparam 90% (w_avg={width_reparam:.0f} kW)')
    miss_rp = ~((actuals[vis_idx] >= reparam_lower[:vis_len]) &
                (actuals[vis_idx] <= reparam_upper[:vis_len]))
    if miss_rp.any():
        axes[1].scatter(x_axis[miss_rp], actuals[vis_idx][miss_rp],
                        color='red', s=14, zorder=4, label='Miss')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Demand (kW)')
    axes[1].set_title('Reparameterization (Gaussian)')
    axes[1].legend(fontsize=8, ncol=3)
    plt.savefig(os.path.join(output_dir, 'exp02_uq_intervals_timeseries.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 4: 区间宽度 vs 实际负荷 (自适应性对比) ----
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(actuals_s, reparam_widths, s=15, alpha=0.5,
               color='#E45756', label='Reparam (adaptive)')
    ax.axhline(2 * q90, color='#2e86ab', ls='-', lw=2,
               label=f'CP (fixed={2*q90:.0f} kW)')
    ax.set_xlabel('Actual Demand (kW)')
    ax.set_ylabel('Interval Width (kW)')
    ax.set_title('Interval Width vs Demand Level — Adaptiveness',
                 fontweight='bold')
    ax.legend(fontsize=9)
    plt.savefig(os.path.join(output_dir, 'exp02_uq_width_vs_demand.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 5: Winkler Score 对比 (区间质量综合指标) ----
    alpha_val = 0.10
    def winkler_score(lower, upper, actual, alpha):
        w = upper - lower
        pen = np.where(actual < lower, 2 / alpha * (lower - actual),
                np.where(actual > upper, 2 / alpha * (actual - upper), 0.0))
        return w + pen
    ws_cp = winkler_score(preds_s - q90, preds_s + q90, actuals_s, alpha_val)
    ws_rp = winkler_score(reparam_lower, reparam_upper, actuals_s, alpha_val)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    bp = ax.boxplot([ws_rp, ws_cp], labels=['Reparam', 'CP'], patch_artist=True,
                    widths=0.4, showfliers=False)
    bp['boxes'][0].set_facecolor('#E45756')
    bp['boxes'][1].set_facecolor('#2e86ab')
    for b in bp['boxes']:
        b.set_alpha(0.7)
    ax.set_ylabel('Winkler Score (lower is better)')
    ax.set_title('Interval Quality: Winkler Score Comparison', fontweight='bold')
    med_rp, med_cp = np.median(ws_rp), np.median(ws_cp)
    ax.text(1, med_rp, f' med={med_rp:.0f}', va='center', fontsize=9, color='#E45756')
    ax.text(2, med_cp, f' med={med_cp:.0f}', va='center', fontsize=9, color='#2e86ab')
    plt.savefig(os.path.join(output_dir, 'exp02_uq_winkler_score.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    result = {
        'cp_coverage_90': cov_cp, 'reparam_coverage_90': cov_reparam,
        'cp_mean_width': width_cp, 'reparam_mean_width': width_reparam,
        'winkler_cp_median': float(med_cp), 'winkler_reparam_median': float(med_rp),
    }
    print(f"    CP: cov={cov_cp:.1f}%, width={width_cp:.0f} | "
          f"Reparam: cov={cov_reparam:.1f}%, width={width_reparam:.0f}")
    print(f"    Winkler (median): CP={med_cp:.0f}, Reparam={med_rp:.0f}")
    print(f"    -> 5 张图已保存")
    return result


# =====================================================================
# 实验 3 — 预测误差注入鲁棒性实验
# =====================================================================
def experiment_noise_robustness(
    data_df, predictions_by_index, model, conformal, cfg,
    price_df, carbon_intensity_hourly, carbon_tracker,
    scaler_X, scaler_y, feature_cols, sim_hours,
    baseline_total_cost, output_dir,
    noise_levels=None,
):
    print("  [Exp 3] 噪声注入鲁棒性实验 ...")
    _apply_style()
    if noise_levels is None:
        noise_levels = [0.0, 0.10, 0.20, 0.30]

    pcw = getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 38.0) / 30.0
    strategies_cfg = {
        'Economic MPC':  lambda: EconomicMPCStrategy(
            horizon=cfg.MPC_HORIZON, peak_charge_weight=pcw),
        'Robust MPC':    lambda: RobustMPCStrategy(
            horizon=cfg.MPC_HORIZON, conformal_predictor=conformal,
            safety_factor=1.0, peak_charge_weight=pcw),
    }

    records = []
    np.random.seed(42)
    for nl in noise_levels:
        noised = predictions_by_index.copy()
        if nl > 0:
            noised *= (1.0 + nl * np.random.randn(len(noised)))
            noised = np.maximum(noised, 0.0)

        for sname, sfactory in strategies_cfg.items():
            strat = sfactory()
            ts = _run_sim(strat, data_df, noised, model, cfg,
                          price_df, carbon_intensity_hourly, carbon_tracker,
                          conformal, scaler_X, scaler_y, feature_cols, sim_hours)
            eco = compute_economic_kpis(
                ts, demand_charge_rate=getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 0))
            tech = compute_technical_kpis(ts)
            records.append({
                'noise': nl, 'strategy': sname,
                'total_cost': eco['total_cost_CNY'],
                'peak_kW': tech['peak_demand_kW'],
            })
        print(f"    noise={nl:.0%} done")

    df = pd.DataFrame(records)

    # 退化率
    for sname in strategies_cfg:
        base_cost = df[(df['strategy'] == sname) & (df['noise'] == 0)]['total_cost'].values[0]
        base_peak = df[(df['strategy'] == sname) & (df['noise'] == 0)]['peak_kW'].values[0]
        mask = df['strategy'] == sname
        df.loc[mask, 'cost_degrad_%'] = (df.loc[mask, 'total_cost'] - base_cost) / base_cost * 100
        df.loc[mask, 'peak_degrad_%'] = (df.loc[mask, 'peak_kW'] - base_peak) / max(base_peak, 1) * 100

    # ---- 图 1: 成本 & 峰值退化曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle('Prediction Noise Robustness', fontweight='bold')
    for sname, color, mk in [('Economic MPC', '#2e86ab', 'o'),
                              ('Robust MPC', '#a23b72', 's')]:
        sub = df[df['strategy'] == sname]
        axes[0].plot(sub['noise'] * 100, sub['cost_degrad_%'], f'{mk}-',
                     color=color, label=sname, lw=2, markersize=7)
        axes[1].plot(sub['noise'] * 100, sub['peak_degrad_%'], f'{mk}-',
                     color=color, label=sname, lw=2, markersize=7)
    axes[0].set_xlabel('Noise Level (%)')
    axes[0].set_ylabel('Cost Degradation (%)')
    axes[0].set_title('Total Cost Degradation')
    axes[0].axhline(0, color='gray', ls='--', lw=0.7)
    axes[0].legend()
    axes[1].set_xlabel('Noise Level (%)')
    axes[1].set_ylabel('Peak Degradation (%)')
    axes[1].set_title('Peak Demand Degradation')
    axes[1].axhline(0, color='gray', ls='--', lw=0.7)
    axes[1].legend()
    plt.savefig(os.path.join(output_dir, 'exp03_noise_degradation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 2: 绝对成本柱状图 (分噪声水平) ----
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    nls = sorted(df['noise'].unique())
    x_pos = np.arange(len(nls))
    w = 0.35
    for i, (sname, color) in enumerate([('Economic MPC', '#2e86ab'),
                                         ('Robust MPC', '#a23b72')]):
        vals = [df[(df['strategy'] == sname) & (df['noise'] == nl)]['total_cost'].values[0]
                for nl in nls]
        bars = ax.bar(x_pos + i * w, np.array(vals) / 1e4, w,
                      label=sname, color=color, edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v/1e4:.1f}', ha='center', fontsize=7.5)
    ax.set_xticks(x_pos + w / 2)
    ax.set_xticklabels([f'{nl:.0%}' for nl in nls])
    ax.set_xlabel('Prediction Noise Level')
    ax.set_ylabel('Total Cost (x10k CNY)')
    ax.set_title('Absolute Cost at Different Noise Levels', fontweight='bold')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'exp03_noise_absolute_cost.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 3: 鲁棒增益 (Robust vs Economic 差值) ----
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for nl in nls:
        ec = df[(df['strategy'] == 'Economic MPC') & (df['noise'] == nl)]['total_cost'].values[0]
        rb = df[(df['strategy'] == 'Robust MPC') & (df['noise'] == nl)]['total_cost'].values[0]
        gain = (ec - rb) / ec * 100
        color = '#2ca02c' if gain > 0 else '#E45756'
        bar = ax.bar(f'{nl:.0%}', gain, color=color, edgecolor='black', linewidth=0.5)
        ax.text(bar[0].get_x() + bar[0].get_width()/2,
                gain + (0.1 if gain > 0 else -0.3),
                f'{gain:.2f}%', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Prediction Noise Level')
    ax.set_ylabel('Robust MPC Cost Advantage (%)')
    ax.set_title('Robust MPC Advantage over Economic MPC at Each Noise Level',
                 fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'exp03_noise_robust_advantage.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    完成, {len(records)} 条记录 -> 3 张图已保存")
    return df.to_dict('records')


# =====================================================================
# 实验 4 — 碳价敏感性分析
# =====================================================================
def experiment_carbon_sensitivity(
    data_df, predictions_by_index, model, conformal, cfg,
    price_df, carbon_intensity_hourly,
    scaler_X, scaler_y, feature_cols, sim_hours,
    baseline_total_cost, output_dir,
    carbon_prices=None,
):
    print("  [Exp 4] 碳价敏感性分析 ...")
    _apply_style()
    if carbon_prices is None:
        carbon_prices = list(range(50, 301, 50))

    pcw = getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 38.0) / 30.0
    dc_rate = getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 0)

    # Economic MPC 只跑一次（不受碳价影响）
    econ_strat = EconomicMPCStrategy(horizon=cfg.MPC_HORIZON, peak_charge_weight=pcw)
    ct_dummy = CarbonTracker(carbon_price=100)
    econ_ts = _run_sim(econ_strat, data_df, predictions_by_index, model, cfg,
                       price_df, carbon_intensity_hourly, ct_dummy,
                       None, scaler_X, scaler_y, feature_cols, sim_hours)
    econ_eco = compute_economic_kpis(econ_ts, demand_charge_rate=dc_rate)

    records = []
    for cp_val in carbon_prices:
        # Carbon-Aware MPC (目标函数含碳价)
        ca_strat = CarbonAwareMPCStrategy(
            horizon=cfg.MPC_HORIZON, carbon_price_cny_per_ton=cp_val,
            peak_charge_weight=pcw,
            carbon_sensitivity=getattr(cfg, 'CARBON_SENSITIVITY', 3.0))
        ct = CarbonTracker(carbon_price=cp_val)
        ca_ts = _run_sim(ca_strat, data_df, predictions_by_index, model, cfg,
                         price_df, carbon_intensity_hourly, ct,
                         None, scaler_X, scaler_y, feature_cols, sim_hours)
        ca_eco = compute_economic_kpis(ca_ts, demand_charge_rate=dc_rate)
        ca_env = compute_environmental_kpis(ca_ts, carbon_price_cny_per_ton=cp_val)

        econ_env = compute_environmental_kpis(econ_ts, carbon_price_cny_per_ton=cp_val)

        records.append({
            'carbon_price': cp_val,
            'econ_total': econ_eco['total_cost_CNY'] + econ_env['carbon_cost_CNY'],
            'econ_co2':   econ_env['total_CO2_tons'],
            'carbon_total': ca_eco['total_cost_CNY'] + ca_env['carbon_cost_CNY'],
            'carbon_co2':   ca_env['total_CO2_tons'],
        })
        print(f"    碳价={cp_val} 元/t done")

    df = pd.DataFrame(records)

    # 交叉点
    diff = df['econ_total'].values - df['carbon_total'].values
    crossover = None
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            x1, x2 = df['carbon_price'].iloc[i], df['carbon_price'].iloc[i+1]
            crossover = x1 + (x2 - x1) * abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
            break

    # ---- 图 1: 总成本 + CO2 排放 双轴 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle('Carbon Price Sensitivity Analysis', fontweight='bold')
    axes[0].plot(df['carbon_price'], df['econ_total'] / 1e4, 'o-',
                 color='#2e86ab', label='Economic MPC', lw=2, markersize=6)
    axes[0].plot(df['carbon_price'], df['carbon_total'] / 1e4, 's-',
                 color='#2ca02c', label='Carbon-Aware MPC', lw=2, markersize=6)
    if crossover:
        axes[0].axvline(crossover, color='gray', ls=':', lw=1.5)
        axes[0].annotate(f'Crossover ~ {crossover:.0f} CNY/t',
                         xy=(crossover, axes[0].get_ylim()[1] * 0.9),
                         fontsize=9, ha='center', color='gray')
    axes[0].set_xlabel('Carbon Price (CNY/tCO2)')
    axes[0].set_ylabel('Total Cost (x10k CNY)')
    axes[0].set_title('Energy + Carbon Cost')
    axes[0].legend()

    axes[1].plot(df['carbon_price'], df['econ_co2'], 'o-',
                 color='#2e86ab', label='Economic MPC', lw=2, markersize=6)
    axes[1].plot(df['carbon_price'], df['carbon_co2'], 's-',
                 color='#2ca02c', label='Carbon-Aware MPC', lw=2, markersize=6)
    axes[1].set_xlabel('Carbon Price (CNY/tCO2)')
    axes[1].set_ylabel(f'{CO2_LABEL} Emissions (tons)')
    axes[1].set_title(f'{CO2_LABEL} Emissions by Carbon Price')
    axes[1].legend()
    plt.savefig(os.path.join(output_dir, 'exp04_carbon_cost_emission.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 2: 碳感知策略的 CO2 减排量 & 边际减排成本 ----
    co2_saved = df['econ_co2'] - df['carbon_co2']
    cost_extra = df['carbon_total'] - df['econ_total']
    marginal_cost = np.where(co2_saved > 0.001,
                             cost_extra / co2_saved, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle('Marginal Abatement Analysis', fontweight='bold')
    # 左: CO2 减排量
    axes[0].bar(df['carbon_price'], co2_saved, color='#2ca02c',
                edgecolor='black', linewidth=0.5, width=30)
    axes[0].set_xlabel('Carbon Price (CNY/tCO2)')
    axes[0].set_ylabel(f'{CO2_LABEL} Reduction (tons)')
    axes[0].set_title(f'Additional {CO2_LABEL} Reduction by Carbon-Aware MPC')
    for i, (x, y) in enumerate(zip(df['carbon_price'], co2_saved)):
        if not np.isnan(y):
            axes[0].text(x, y + y*0.02, f'{y:.2f}', ha='center', fontsize=8)
    # 右: 边际减排成本
    valid_mac = ~np.isnan(marginal_cost)
    axes[1].plot(df['carbon_price'][valid_mac], marginal_cost[valid_mac],
                 'D-', color='#d62728', lw=2, markersize=7)
    axes[1].axhline(0, color='gray', ls='--', lw=0.8)
    axes[1].set_xlabel('Carbon Price (CNY/tCO2)')
    axes[1].set_ylabel('Marginal Abatement Cost (CNY/tCO2)')
    axes[1].set_title('Marginal Cost of Carbon-Aware Strategy')
    plt.savefig(os.path.join(output_dir, 'exp04_marginal_abatement.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 3: 成本结构堆叠 (能量费 + 碳费) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle('Cost Decomposition at Different Carbon Prices', fontweight='bold')
    x_idx = np.arange(len(df))
    bar_w = 0.4
    for ax_i, (label, cost_key, co2_key) in enumerate([
        ('Economic MPC', 'econ_total', 'econ_co2'),
        ('Carbon-Aware MPC', 'carbon_total', 'carbon_co2'),
    ]):
        energy_part = df[cost_key] - df[co2_key] * df['carbon_price']
        carbon_part = df[co2_key] * df['carbon_price']
        axes[ax_i].bar(x_idx, energy_part / 1e4, color='#4C78A8',
                       label='Energy Cost', edgecolor='black', linewidth=0.3)
        axes[ax_i].bar(x_idx, carbon_part / 1e4, bottom=energy_part / 1e4,
                       color='#72B7B2', label='Carbon Cost', edgecolor='black', linewidth=0.3)
        axes[ax_i].set_xticks(x_idx)
        axes[ax_i].set_xticklabels([f'{int(p)}' for p in df['carbon_price']])
        axes[ax_i].set_xlabel('Carbon Price (CNY/t)')
        axes[ax_i].set_ylabel('Cost (x10k CNY)')
        axes[ax_i].set_title(label)
        axes[ax_i].legend(fontsize=8)
    plt.savefig(os.path.join(output_dir, 'exp04_cost_decomposition.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    result = {'crossover_price': crossover, 'data': df.to_dict('records')}
    print(f"    交叉碳价: {crossover:.0f} CNY/t" if crossover else "    未找到交叉点")
    print(f"    -> 3 张图已保存")
    return result


# =====================================================================
# 实验 5 — MPC 计算时间分析
# =====================================================================
def experiment_mpc_timing(
    data_df, predictions_by_index, cfg,
    price_df, carbon_intensity_hourly,
    sim_hours, output_dir,
    n_steps=200,
):
    print("  [Exp 5] MPC 计算时间分析 ...")
    _apply_style()
    import cvxpy as cp_mod
    from BES import BatteryEnergyStorage

    bess = BatteryEnergyStorage(cfg.BESS_CAPACITY_KWH, cfg.BESS_POWER_KW)
    pcw = getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 38.0) / 30.0
    horizon = cfg.MPC_HORIZON

    configs = {
        'Econ-MPC':   dict(include_carbon=False, include_peak_charge=True, peak_charge_weight=pcw),
        'Carbon-MPC': dict(include_carbon=True,  include_peak_charge=True, peak_charge_weight=pcw),
    }

    all_records = {}
    for cname, ckwargs in configs.items():
        mpc = _ParametrizedMPC(bess, horizon, **ckwargs)
        times_warm, times_cold = [], []
        solver_used = {'OSQP': 0, 'CLARABEL': 0, 'default': 0}

        indices = np.linspace(0, min(sim_hours - 1, len(predictions_by_index) - horizon),
                              n_steps, dtype=int)
        for idx in indices:
            load = predictions_by_index[idx:idx + horizon]
            if len(load) < horizon:
                load = np.pad(load, (0, horizon - len(load)), 'edge')
            pv = data_df['E_PV'].iloc[idx:idx + horizon].values.astype(float)
            if len(pv) < horizon:
                pv = np.pad(pv, (0, horizon - len(pv)), 'edge')
            pr = price_df.iloc[idx:idx + horizon]['price'].values.astype(float)
            if len(pr) < horizon:
                pr = np.pad(pr, (0, horizon - len(pr)), 'edge')
            ci = carbon_intensity_hourly[idx:idx + horizon] if ckwargs['include_carbon'] else None
            if ci is not None and len(ci) < horizon:
                ci = np.pad(ci, (0, horizon - len(ci)), 'edge')
            cw = 0.36 if ckwargs['include_carbon'] else 0.0

            # warm start 计时
            mpc.soc_init.value = float(bess.get_soc())
            mpc.load_p.value = load
            mpc.pv_p.value = pv
            mpc.price_p.value = pr
            if ckwargs['include_carbon'] and ci is not None:
                mpc.carbon_p.value = np.maximum(0, ci)
                mpc.carbon_weight.value = cw

            t0 = time.perf_counter()
            for solver in (cp_mod.OSQP, cp_mod.CLARABEL, None):
                try:
                    if solver is None:
                        mpc.problem.solve(warm_start=True, verbose=False)
                    else:
                        mpc.problem.solve(solver=solver, warm_start=True, verbose=False)
                    status = mpc.problem.status
                    if status in (cp_mod.OPTIMAL, cp_mod.OPTIMAL_INACCURATE):
                        sname = 'OSQP' if solver == cp_mod.OSQP else (
                            'CLARABEL' if solver == cp_mod.CLARABEL else 'default')
                        solver_used[sname] += 1
                        break
                except Exception:
                    continue
            t1 = time.perf_counter()
            times_warm.append(t1 - t0)

        times_warm = np.array(times_warm) * 1000  # ms
        all_records[cname] = {
            'p50_ms': float(np.percentile(times_warm, 50)),
            'p95_ms': float(np.percentile(times_warm, 95)),
            'mean_ms': float(times_warm.mean()),
            'solver_counts': dict(solver_used),
            'times_ms': times_warm.tolist(),
        }

    names = list(all_records.keys())

    # ---- 图 1: 求解时间分布 (小提琴 + 箱线) ----
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    time_data = [all_records[n]['times_ms'] for n in names]
    colors_v = ['#4C78A8', '#2ca02c']
    vp = ax.violinplot(time_data, showmedians=False, showextrema=False)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors_v[i % len(colors_v)])
        body.set_alpha(0.5)
    bp = ax.boxplot(time_data, labels=names, patch_artist=True,
                    widths=0.15, showfliers=False)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors_v[i % len(colors_v)])
        box.set_alpha(0.8)
    for i, n in enumerate(names):
        r = all_records[n]
        ax.text(i + 1.2, r['p50_ms'],
                f'p50={r["p50_ms"]:.1f}ms', fontsize=8, va='center')
        ax.text(i + 1.2, r['p95_ms'],
                f'p95={r["p95_ms"]:.1f}ms', fontsize=8, va='center', color='red')
    ax.set_ylabel('Solve Time (ms)')
    ax.set_title('MPC Solve Time Distribution (Warm Start)', fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'exp05_mpc_timing_violin.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 2: Solver fallback 堆叠柱 ----
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    solver_names = ['OSQP', 'CLARABEL', 'default']
    bottom = np.zeros(len(names))
    colors_s = ['#4C78A8', '#F58518', '#E45756']
    for si, sn in enumerate(solver_names):
        vals = [all_records[n]['solver_counts'].get(sn, 0)
                / max(1, sum(all_records[n]['solver_counts'].values())) * 100
                for n in names]
        bars = ax.bar(names, vals, bottom=bottom, color=colors_s[si], label=sn,
                      edgecolor='black', linewidth=0.5)
        for j, (b, v) in enumerate(zip(bars, vals)):
            if v > 3:
                ax.text(b.get_x() + b.get_width()/2, bottom[j] + v/2,
                        f'{v:.0f}%', ha='center', fontsize=8, color='white',
                        fontweight='bold')
        bottom += np.array(vals)
    ax.set_ylabel('Fraction (%)')
    ax.set_title('Solver Usage Breakdown', fontweight='bold')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'exp05_solver_fallback.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 3: CDF 累积分布函数 ----
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for i, n in enumerate(names):
        sorted_t = np.sort(all_records[n]['times_ms'])
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t) * 100
        ax.plot(sorted_t, cdf, '-', color=colors_v[i % len(colors_v)],
                lw=2, label=n)
    ax.axhline(50, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax.axhline(95, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.8, 52, 'p50', fontsize=8, color='gray')
    ax.text(ax.get_xlim()[1] * 0.8, 97, 'p95', fontsize=8, color='gray')
    ax.set_xlabel('Solve Time (ms)')
    ax.set_ylabel('Cumulative Probability (%)')
    ax.set_title('CDF of MPC Solve Times', fontweight='bold')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'exp05_mpc_timing_cdf.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 4: 求解时间随步数变化 (热启动效果) ----
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for i, n in enumerate(names):
        ax.plot(all_records[n]['times_ms'], '-', alpha=0.7,
                color=colors_v[i % len(colors_v)], lw=0.9, label=n)
    ax.set_xlabel('MPC Step Index')
    ax.set_ylabel('Solve Time (ms)')
    ax.set_title('Solve Time Trace — Warm Start Convergence', fontweight='bold')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'exp05_mpc_timing_trace.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    for n in names:
        r = all_records[n]
        print(f"    {n}: p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms, "
              f"solver={r['solver_counts']}")
    print(f"    -> 4 张图已保存")
    return all_records


# =====================================================================
# 实验 6 — BESS 容量优化
# =====================================================================
def experiment_bess_capacity(
    data_df, predictions_by_index, model, conformal, cfg,
    price_df, carbon_intensity_hourly, carbon_tracker,
    scaler_X, scaler_y, feature_cols, sim_hours,
    baseline_total_cost, invest_cost, output_dir,
    capacities=None,
):
    print("  [Exp 6] BESS 容量优化 ...")
    _apply_style()
    if capacities is None:
        capacities = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000]

    c_rate = cfg.BESS_POWER_KW / max(cfg.BESS_CAPACITY_KWH, 1)
    pcw = getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 38.0) / 30.0
    dc_rate = getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 0)

    records = []
    for cap in capacities:
        pwr = cap * c_rate
        inv = cap * cfg.BESS_INVEST_CNY_PER_KWH + pwr * cfg.BESS_INVEST_CNY_PER_KW
        strat = EconomicMPCStrategy(horizon=cfg.MPC_HORIZON, peak_charge_weight=pcw)
        if cap == 0:
            strat = BaselineStrategy()

        ts = _run_sim(strat, data_df, predictions_by_index, model, cfg,
                      price_df, carbon_intensity_hourly, carbon_tracker,
                      conformal, scaler_X, scaler_y, feature_cols, sim_hours,
                      capacity_kwh=cap, bess_power_kw=pwr)
        eco = compute_economic_kpis(
            ts, investment_cost=inv if cap > 0 else 0,
            baseline_total_cost=baseline_total_cost,
            demand_charge_rate=dc_rate,
            bess_capacity_kwh=cap,
        )
        records.append({
            'capacity_kwh': cap, 'investment': inv,
            'total_cost': eco['total_cost_CNY'],
            'NPV': eco.get('NPV_CNY', np.nan),
            'IRR': eco.get('IRR', np.nan),
            'payback': eco.get('payback_period', np.nan),
            'annual_savings': eco.get('annual_savings_CNY', np.nan),
        })
        print(f"    {cap:6d} kWh → cost={eco['total_cost_CNY']:.0f}, "
              f"NPV={eco.get('NPV_CNY', 0):.0f}")

    df = pd.DataFrame(records)

    # 最优容量
    valid = df.dropna(subset=['NPV'])
    best_idx = valid['NPV'].idxmax() if len(valid) > 0 else 0
    best_cap = df.loc[best_idx, 'capacity_kwh'] if best_idx in df.index else 0
    x = df['capacity_kwh'] / 1000

    # ---- 图 1: NPV / IRR / 年化净收益 三合一 ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.suptitle('BESS Capacity Optimization', fontweight='bold')

    axes[0].plot(x, df['NPV'] / 1e4, 'o-', color='#2e86ab', lw=2, markersize=6)
    axes[0].fill_between(x, 0, df['NPV'] / 1e4,
                         where=df['NPV'] > 0, alpha=0.15, color='#2ca02c')
    axes[0].fill_between(x, 0, df['NPV'] / 1e4,
                         where=df['NPV'] <= 0, alpha=0.15, color='#E45756')
    axes[0].axhline(0, color='gray', ls='--', lw=0.8)
    if best_cap > 0:
        axes[0].axvline(best_cap / 1000, color='red', ls=':', lw=1.5,
                        label=f'Optimal {best_cap/1000:.0f} MWh')
        axes[0].legend(fontsize=9)
    axes[0].set_xlabel('Capacity (MWh)')
    axes[0].set_ylabel('NPV (x10k CNY)')
    axes[0].set_title('Net Present Value')

    irr_pct = df['IRR'].apply(lambda v: v * 100 if not np.isnan(v) else np.nan)
    axes[1].plot(x, irr_pct, 's-', color='#2ca02c', lw=2, markersize=6)
    axes[1].axhline(5, color='red', ls='--', lw=1, label='Discount Rate 5%')
    axes[1].set_xlabel('Capacity (MWh)')
    axes[1].set_ylabel('IRR (%)')
    axes[1].set_title('Internal Rate of Return')
    axes[1].legend(fontsize=9)

    axes[2].plot(x, df['annual_savings'] / 1e4, '^-', color='#F58518', lw=2, markersize=6)
    axes[2].axhline(0, color='gray', ls='--', lw=0.8)
    axes[2].set_xlabel('Capacity (MWh)')
    axes[2].set_ylabel('Annual Net Savings (x10k CNY)')
    axes[2].set_title('Annual Net Savings')
    plt.savefig(os.path.join(output_dir, 'exp06_bess_npv_irr.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 2: 回收期 vs 容量 ----
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    payback = df['payback'].values.copy()
    finite_pb = payback[~np.isnan(payback) & (payback < 100)]
    ax.plot(x, np.where(np.isnan(payback), np.nan, payback),
            'D-', color='#d62728', lw=2, markersize=7)
    ax.axhline(10, color='gray', ls='--', lw=1, label='10-year threshold')
    ax.set_xlabel('Capacity (MWh)')
    ax.set_ylabel('Payback Period (years)')
    ax.set_title('Simple Payback Period vs BESS Capacity', fontweight='bold')
    ax.legend(fontsize=9)
    if len(finite_pb) > 0:
        ax.set_ylim(0, max(15, finite_pb.max() * 1.2))
    plt.savefig(os.path.join(output_dir, 'exp06_bess_payback.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 3: 投资额 vs 年化收益 (气泡图: 大小=NPV) ----
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    inv_arr = df['investment'].values / 1e4
    sav_arr = df['annual_savings'].values / 1e4
    npv_arr = df['NPV'].values
    sizes = np.clip(np.abs(npv_arr) / max(1, np.nanmax(np.abs(npv_arr))) * 300, 30, 500)
    colors_b = ['#2ca02c' if v > 0 else '#E45756' for v in npv_arr]
    for i in range(len(df)):
        if not np.isnan(npv_arr[i]):
            ax.scatter(inv_arr[i], sav_arr[i], s=sizes[i], c=colors_b[i],
                       edgecolor='black', linewidth=0.5, alpha=0.7, zorder=3)
            ax.annotate(f'{x.iloc[i]:.0f}MWh', (inv_arr[i], sav_arr[i]),
                        fontsize=7.5, ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Investment (x10k CNY)')
    ax.set_ylabel('Annual Net Savings (x10k CNY)')
    ax.set_title('Investment vs Return (bubble size = |NPV|)', fontweight='bold')
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=10, label='NPV > 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E45756',
               markersize=10, label='NPV < 0'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.savefig(os.path.join(output_dir, 'exp06_bess_investment_return.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 4: 总成本 vs 容量 (含无储能基线) ----
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(x, df['total_cost'] / 1e4, 'o-', color='#2e86ab', lw=2,
            markersize=6, label='Total Operating Cost')
    if baseline_total_cost is not None:
        ax.axhline(baseline_total_cost / 1e4, color='#7f7f7f', ls='--', lw=1.5,
                   label=f'Baseline (no BESS) = {baseline_total_cost/1e4:.1f}')
    ax.set_xlabel('Capacity (MWh)')
    ax.set_ylabel('Total Cost (x10k CNY)')
    ax.set_title('Operating Cost vs BESS Capacity', fontweight='bold')
    ax.legend(fontsize=9)
    plt.savefig(os.path.join(output_dir, 'exp06_bess_total_cost.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    最优容量: {best_cap/1000:.0f} MWh -> 4 张图已保存")
    return {'optimal_capacity_kwh': best_cap, 'data': df.to_dict('records')}


# =====================================================================
# 实验 7 — TOU 峰谷比敏感性
# =====================================================================
def experiment_tou_sensitivity(
    data_df, predictions_by_index, model, conformal, cfg,
    carbon_intensity_hourly, carbon_tracker,
    scaler_X, scaler_y, feature_cols, sim_hours,
    output_dir,
    ratios=None,
):
    print("  [Exp 7] TOU 峰谷比敏感性 ...")
    _apply_style()
    if ratios is None:
        ratios = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    pcw = getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 38.0) / 30.0
    dc_rate = getattr(cfg, 'DEMAND_CHARGE_CNY_PER_KW_MONTH', 0)
    avg_price = 0.80  # 保持加权均值近似不变

    strategies_cfg = {
        'Economic MPC': lambda: EconomicMPCStrategy(
            horizon=cfg.MPC_HORIZON, peak_charge_weight=pcw),
        'Robust MPC': lambda: RobustMPCStrategy(
            horizon=cfg.MPC_HORIZON, conformal_predictor=conformal,
            safety_factor=1.0, peak_charge_weight=pcw),
    }

    records = []
    for ratio in ratios:
        valley = 2 * avg_price / (ratio + 1)
        peak = valley * ratio
        sp = peak * 1.15  # 尖峰比峰高 15%
        flat = (peak + valley) / 2

        pdf = _make_price_series(data_df['timestamp'], peak, flat, valley, sp)

        # Baseline (无储能)
        bl_strat = BaselineStrategy()
        bl_ts = _run_sim(bl_strat, data_df, predictions_by_index, model, cfg,
                         pdf, carbon_intensity_hourly, carbon_tracker,
                         None, scaler_X, scaler_y, feature_cols, sim_hours,
                         capacity_kwh=0, bess_power_kw=0)
        bl_cost = compute_economic_kpis(bl_ts, demand_charge_rate=dc_rate)['total_cost_CNY']

        for sname, sfactory in strategies_cfg.items():
            strat = sfactory()
            ts = _run_sim(strat, data_df, predictions_by_index, model, cfg,
                          pdf, carbon_intensity_hourly, carbon_tracker,
                          conformal, scaler_X, scaler_y, feature_cols, sim_hours)
            eco = compute_economic_kpis(ts, demand_charge_rate=dc_rate)
            savings_pct = (bl_cost - eco['total_cost_CNY']) / bl_cost * 100 if bl_cost > 0 else 0
            records.append({
                'ratio': ratio, 'strategy': sname,
                'savings_pct': savings_pct,
                'total_cost': eco['total_cost_CNY'],
                'peak_price': peak, 'valley_price': valley,
            })
        print(f"    ratio={ratio:.1f} done (peak={peak:.2f}, valley={valley:.2f})")

    df = pd.DataFrame(records)

    # ---- 图 1: 节省率 vs 峰谷比 ----
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for sname, color, mk in [('Economic MPC', '#2e86ab', 'o'),
                              ('Robust MPC', '#a23b72', 's')]:
        sub = df[df['strategy'] == sname]
        ax.plot(sub['ratio'], sub['savings_pct'], f'{mk}-', color=color,
                label=sname, lw=2, markersize=7)
        for _, row in sub.iterrows():
            ax.annotate(f'{row["savings_pct"]:.1f}%',
                        (row['ratio'], row['savings_pct']),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=7.5, ha='center')
    ax.set_xlabel('Peak / Valley Price Ratio')
    ax.set_ylabel('Cost Savings vs Baseline (%)')
    ax.set_title('Storage Value vs TOU Peak-Valley Ratio', fontweight='bold')
    ax.legend()
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    plt.savefig(os.path.join(output_dir, 'exp07_tou_savings_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 2: 绝对成本对比 (分组柱状图) ----
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ratios_u = sorted(df['ratio'].unique())
    x_pos = np.arange(len(ratios_u))
    bar_w = 0.35
    strat_colors = [('Economic MPC', '#2e86ab'), ('Robust MPC', '#a23b72')]
    for i, (sname, color) in enumerate(strat_colors):
        vals = [df[(df['strategy'] == sname) & (df['ratio'] == r)]['total_cost'].values[0]
                for r in ratios_u]
        ax.bar(x_pos + i * bar_w, np.array(vals) / 1e4, bar_w,
               label=sname, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos + bar_w / 2)
    ax.set_xticklabels([f'{r:.1f}:1' for r in ratios_u])
    ax.set_xlabel('Peak-Valley Ratio')
    ax.set_ylabel('Total Cost (x10k CNY)')
    ax.set_title('Absolute Cost at Different TOU Ratios', fontweight='bold')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'exp07_tou_absolute_cost.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 3: 电价结构可视化 (各方案的峰/平/谷定价) ----
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sub_econ = df[df['strategy'] == 'Economic MPC'].drop_duplicates('ratio')
    valley_vals = [2 * avg_price / (r + 1) for r in ratios_u]
    peak_vals = [v * r for v, r in zip(valley_vals, ratios_u)]
    flat_vals = [(p + v) / 2 for p, v in zip(peak_vals, valley_vals)]
    sp_vals = [p * 1.15 for p in peak_vals]
    ax.plot(ratios_u, sp_vals, 'v-', color='#d62728', lw=2, label='Super-Peak', markersize=6)
    ax.plot(ratios_u, peak_vals, '^-', color='#F58518', lw=2, label='Peak', markersize=6)
    ax.plot(ratios_u, flat_vals, 'D-', color='#4C78A8', lw=2, label='Flat', markersize=6)
    ax.plot(ratios_u, valley_vals, 'o-', color='#2ca02c', lw=2, label='Valley', markersize=6)
    ax.fill_between(ratios_u, valley_vals, sp_vals, alpha=0.08, color='gray')
    ax.set_xlabel('Peak-Valley Ratio')
    ax.set_ylabel('Price (CNY/kWh)')
    ax.set_title('TOU Price Structure at Each Ratio', fontweight='bold')
    ax.legend(fontsize=9)
    plt.savefig(os.path.join(output_dir, 'exp07_tou_price_structure.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    -> 3 张图已保存")
    return df.to_dict('records')


# =====================================================================
# 实验 8 — 极端事件案例分析
# =====================================================================
def experiment_extreme_event(strategy_results, data_df,
                             predictions_by_index, conformal,
                             sim_hours, output_dir):
    print("  [Exp 8] 极端事件案例分析 ...")
    _apply_style()

    # 取任一策略的时序数据
    first_key = list(strategy_results.keys())[0]
    ts0 = strategy_results[first_key]['timeseries']

    errors = np.abs(ts0['actual_demand'].values - ts0['predicted_demand'].values)
    timestamps = pd.to_datetime(ts0['timestamp'].values)
    dates = timestamps.date

    # 按天汇总误差
    daily_err = {}
    for i, d in enumerate(dates):
        daily_err.setdefault(d, 0)
        daily_err[d] += errors[i]
    worst_day = max(daily_err, key=daily_err.get)
    day_mask = np.array([d == worst_day for d in dates])
    hours_in_day = np.arange(day_mask.sum())

    q90 = conformal.q_hat if conformal.q_hat is not None else 0

    actual_day = ts0['actual_demand'].values[day_mask]
    pred_day   = ts0['predicted_demand'].values[day_mask]

    # ---- 图 1: 需求 vs 预测 + CP 带 + 误差阴影 ----
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True,
                             sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Extreme Event Day: {worst_day} (Max Daily Prediction Error)',
                 fontweight='bold')
    # 上: 需求曲线
    axes[0].plot(hours_in_day, actual_day, 'k-', lw=2, label='Actual Demand', zorder=3)
    axes[0].plot(hours_in_day, pred_day, '--', color='#2e86ab', lw=1.8, label='Predicted')
    axes[0].fill_between(hours_in_day, pred_day - q90, pred_day + q90,
                         alpha=0.25, color='#2e86ab', label='CP 90% Band')
    miss = ~((actual_day >= pred_day - q90) & (actual_day <= pred_day + q90))
    if miss.any():
        axes[0].scatter(hours_in_day[miss], actual_day[miss],
                        color='red', s=40, zorder=4, label='CP Miss', marker='x')
    axes[0].set_ylabel('Demand (kW)')
    axes[0].legend(fontsize=9)
    # 下: 逐时误差
    err_day = actual_day - pred_day
    colors_err = ['#2ca02c' if e >= 0 else '#E45756' for e in err_day]
    axes[1].bar(hours_in_day, err_day, color=colors_err, edgecolor='black', linewidth=0.3)
    axes[1].axhline(0, color='gray', lw=0.8)
    axes[1].axhline(q90, color='red', ls=':', lw=1, label=f'+q90={q90:.0f}')
    axes[1].axhline(-q90, color='red', ls=':', lw=1, label=f'-q90')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Error (kW)')
    axes[1].legend(fontsize=8, ncol=2)
    plt.savefig(os.path.join(output_dir, 'exp08_extreme_demand_error.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 2: 各策略响应 (grid / BESS / SOC) ----
    strat_names = [n for n in strategy_results if n != 'Baseline (No Storage)']
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True, sharex=True)
    fig.suptitle(f'Strategy Responses on {worst_day}', fontweight='bold')
    labels_y = ['Grid Import (kW)', 'BESS Power (kW)', 'SOC']
    cols   = ['grid_import', 'bess_power', 'bess_soc']
    for ai, (col, ylabel) in enumerate(zip(cols, labels_y)):
        for sn in strat_names:
            ts_s = strategy_results[sn]['timeseries']
            vals = ts_s[col].values[day_mask]
            axes[ai].plot(hours_in_day, vals, label=_short(sn),
                          color=_color(sn), lw=1.6)
        axes[ai].set_ylabel(ylabel)
        axes[ai].legend(fontsize=8, ncol=2)
    axes[2].set_xlabel('Hour of Day')
    plt.savefig(os.path.join(output_dir, 'exp08_extreme_strategy_response.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 3: 累计成本 & CO2 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f'Cumulative Metrics on {worst_day}', fontweight='bold')
    for sn in strategy_results:
        ts_s = strategy_results[sn]['timeseries']
        cost_day = ts_s['cost'].values[day_mask]
        co2_day  = ts_s['co2_kg'].values[day_mask]
        axes[0].plot(hours_in_day, np.cumsum(cost_day), label=_short(sn),
                     color=_color(sn), lw=1.6)
        axes[1].plot(hours_in_day, np.cumsum(co2_day) / 1000, label=_short(sn),
                     color=_color(sn), lw=1.6)
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('Cumulative Cost (CNY)')
    axes[0].set_title('Cost')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel(f'Cumulative {CO2_LABEL} (tons)')
    axes[1].set_title('Emissions')
    axes[1].legend(fontsize=8)
    plt.savefig(os.path.join(output_dir, 'exp08_extreme_cumulative.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 4: 能量流堆叠面积图 (最优策略) ----
    # 选 Economic MPC 作代表
    repr_name = 'Economic MPC' if 'Economic MPC' in strategy_results else strat_names[0]
    ts_r = strategy_results[repr_name]['timeseries']
    re_day = ts_r['renewable_generation'].values[day_mask]
    gr_day = ts_r['grid_import'].values[day_mask]
    bess_d = np.maximum(ts_r['bess_power'].values[day_mask], 0)   # 放电
    bess_c = np.maximum(-ts_r['bess_power'].values[day_mask], 0)  # 充电

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.fill_between(hours_in_day, 0, re_day,
                    alpha=0.6, color='#2ca02c', label='Renewable')
    ax.fill_between(hours_in_day, re_day, re_day + bess_d,
                    alpha=0.6, color='#F58518', label='BESS Discharge')
    ax.fill_between(hours_in_day, re_day + bess_d, re_day + bess_d + gr_day,
                    alpha=0.6, color='#4C78A8', label='Grid Import')
    ax.plot(hours_in_day, actual_day, 'k-', lw=2, label='Actual Demand', zorder=3)
    ax.fill_between(hours_in_day, -bess_c, 0, alpha=0.4, color='#a23b72',
                    label='BESS Charge')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'Energy Flow Breakdown on {worst_day} ({_short(repr_name)})',
                 fontweight='bold')
    ax.legend(fontsize=9, ncol=3)
    plt.savefig(os.path.join(output_dir, 'exp08_extreme_energy_flow.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图 5: 极端日 vs 典型日 预测误差对比 ----
    daily_err_vals = np.array(list(daily_err.values()))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(daily_err_vals, bins=min(30, len(daily_err_vals)),
            color='#4C78A8', edgecolor='black', alpha=0.8, label='All Days')
    ax.axvline(daily_err[worst_day], color='red', ls='--', lw=2,
               label=f'Worst Day: {worst_day}\n(err={daily_err[worst_day]:.0f})')
    ax.set_xlabel('Daily Total Absolute Error (kW)')
    ax.set_ylabel('Count')
    ax.set_title('Daily Prediction Error Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    plt.savefig(os.path.join(output_dir, 'exp08_extreme_error_hist.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    cp_cov_day = float(
        ((actual_day >= pred_day - q90) & (actual_day <= pred_day + q90)).mean() * 100
    )
    result = {
        'worst_day': str(worst_day),
        'daily_error_sum': daily_err[worst_day],
        'max_hourly_error': float(errors[day_mask].max()),
        'cp_coverage_on_day': cp_cov_day,
    }
    print(f"    极端日: {worst_day}, 日误差总和={daily_err[worst_day]:.0f}, "
          f"CP 当日覆盖={cp_cov_day:.1f}%")
    print(f"    -> 5 张图已保存")
    return result


# =====================================================================
# 入口函数 — 一键运行全部 8 个实验
# =====================================================================
def run_all_experiments(
    strategy_results, data_df, predictions_by_index, conformal, model,
    cfg, price_df, carbon_intensity_hourly, carbon_tracker,
    scaler_X, scaler_y, feature_cols, sim_hours,
    baseline_total_cost, invest_cost, output_dir,
):
    """Stage [9/9]: 8 个高级分析实验，输出到 output_dir/experiments/。"""
    exp_dir = os.path.join(output_dir, 'experiments')
    os.makedirs(exp_dir, exist_ok=True)

    results = {}

    # --- 1. CP 覆盖率 ---
    try:
        results['exp01'] = experiment_cp_coverage(
            conformal, data_df, predictions_by_index, sim_hours, exp_dir)
    except Exception as e:
        print(f"  [Exp 1] 失败: {e}")

    # --- 2. CP vs Reparam ---
    try:
        results['exp02'] = experiment_cp_vs_reparam(
            model, conformal, data_df, predictions_by_index,
            scaler_X, scaler_y, feature_cols, sim_hours, exp_dir)
    except Exception as e:
        print(f"  [Exp 2] 失败: {e}")

    # --- 3. 噪声鲁棒性 ---
    try:
        results['exp03'] = experiment_noise_robustness(
            data_df, predictions_by_index, model, conformal, cfg,
            price_df, carbon_intensity_hourly, carbon_tracker,
            scaler_X, scaler_y, feature_cols, sim_hours,
            baseline_total_cost, exp_dir)
    except Exception as e:
        print(f"  [Exp 3] 失败: {e}")

    # --- 4. 碳价敏感性 ---
    try:
        results['exp04'] = experiment_carbon_sensitivity(
            data_df, predictions_by_index, model, conformal, cfg,
            price_df, carbon_intensity_hourly,
            scaler_X, scaler_y, feature_cols, sim_hours,
            baseline_total_cost, exp_dir)
    except Exception as e:
        print(f"  [Exp 4] 失败: {e}")

    # --- 5. MPC 计算时间 ---
    try:
        results['exp05'] = experiment_mpc_timing(
            data_df, predictions_by_index, cfg,
            price_df, carbon_intensity_hourly, sim_hours, exp_dir)
    except Exception as e:
        print(f"  [Exp 5] 失败: {e}")

    # --- 6. BESS 容量优化 ---
    try:
        results['exp06'] = experiment_bess_capacity(
            data_df, predictions_by_index, model, conformal, cfg,
            price_df, carbon_intensity_hourly, carbon_tracker,
            scaler_X, scaler_y, feature_cols, sim_hours,
            baseline_total_cost, invest_cost, exp_dir)
    except Exception as e:
        print(f"  [Exp 6] 失败: {e}")

    # --- 7. TOU 敏感性 ---
    try:
        results['exp07'] = experiment_tou_sensitivity(
            data_df, predictions_by_index, model, conformal, cfg,
            carbon_intensity_hourly, carbon_tracker,
            scaler_X, scaler_y, feature_cols, sim_hours, exp_dir)
    except Exception as e:
        print(f"  [Exp 7] 失败: {e}")

    # --- 8. 极端事件 ---
    try:
        results['exp08'] = experiment_extreme_event(
            strategy_results, data_df, predictions_by_index,
            conformal, sim_hours, exp_dir)
    except Exception as e:
        print(f"  [Exp 8] 失败: {e}")

    # 保存摘要 JSON
    def _safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj)
        if isinstance(obj, dict): return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_safe(v) for v in obj]
        return obj

    summary_path = os.path.join(exp_dir, 'experiment_summary.json')
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(_safe(results), f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  ✅ 实验结果已保存至 {exp_dir}")
    except Exception:
        print(f"\n  ⚠ JSON 保存失败，图片已保存至 {exp_dir}")

    return results
