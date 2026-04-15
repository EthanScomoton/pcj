"""
港口综合能源系统 —— 策略选择平台 (main)
=========================================
基于 《深度学习驱动的干散货港口综合能源系统：技术蓝图与代码升级方案》
实施 P0 优先级改动：
  · MPC 滚动时域优化（参数化 CVXPY, 热启动）
  · 保形预测 + 鲁棒调度
  · 多目标（成本 + 碳排）优化
  · 经济性 & 环保性 综合分析

流程:
  1. 数据加载 / 重采样 / 特征工程
  2. Scaler 拟合
  3. 创建 EModel_FeatureWeight4 并加载 best_EModel_FeatureWeight4.pth
  4. 用验证集校准保形预测器
  5. 生成电价 & 动态碳强度时序
  6. 预计算负荷预测（跨策略共享）
  7. 逐一评估若干策略 → 经济 / 环保 / 技术 KPI
  8. 加权打分，可视化，生成最终报告
"""
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

# -- 本项目模块 --
from All_Models_EGrid_Paper import (
    load_data, feature_engineering,
    EModel_FeatureWeight4, calculate_feature_importance,
)
from EF                 import print_feature_info
from convert_model      import convert_model_weights
from enhanced_ies       import StrategyAwareIES
from strategies         import (
    BaselineStrategy, PeakShavingRuleStrategy,
    EconomicMPCStrategy, CarbonAwareMPCStrategy, RobustMPCStrategy,
)
from conformal_predictor import ConformalPredictor
from carbon_module       import (
    CarbonTracker, make_dynamic_carbon_intensity,
    GRID_EMISSION_FACTORS, DEFAULT_CARBON_PRICE_CNY_PER_TON,
)
from analysis           import (
    compute_economic_kpis, compute_environmental_kpis, compute_technical_kpis,
    build_comparison_table, score_strategies,
    generate_all_plots,
    format_final_report,
)


# =======================================================================
# 平台配置
# =======================================================================
class PlatformConfig:
    # 仿真
    SIM_HOURS       = 24 * 30    # 30 天  (加大可至 24*90)
    MPC_HORIZON     = 24

    # 储能配置 (可由 OSS 优化，或由用户直接设定)
    BESS_CAPACITY_KWH = 20000
    BESS_POWER_KW     = 16000
    BESS_INVEST_CNY_PER_KWH = 600
    BESS_INVEST_CNY_PER_KW  = 200

    # 碳
    GRID_REGION    = 'east'       # 华东
    CARBON_PRICE   = DEFAULT_CARBON_PRICE_CNY_PER_TON  # 100 元/tCO2

    # 保形预测
    CONFORMAL_ALPHA = 0.10         # 90% 覆盖率
    CAL_FRACTION    = 0.10         # 训练后 10% 作为校准集

    # KPI 打分权重
    SCORE_WEIGHTS = {'cost': 0.40, 'co2': 0.35, 'peak': 0.25}

    # 输出目录
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


# =======================================================================
# 主流程
# =======================================================================
def main():
    cfg = PlatformConfig
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # ==================================================================
    # 1) 数据加载 & 预处理
    # ==================================================================
    print("\n[1/8] 数据加载 & 特征工程 ...")
    data_df = load_data()
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df = data_df.drop_duplicates(subset=['timestamp']).set_index('timestamp')

    num_cols = data_df.select_dtypes(include=['number']).columns
    df_num = data_df[num_cols].resample('1h').mean().interpolate(method='linear')
    other_cols = [c for c in data_df.columns if c not in num_cols]
    if other_cols:
        df_oth = data_df[other_cols].resample('1h').ffill()
        data_df = pd.concat([df_num, df_oth], axis=1)
    else:
        data_df = df_num
    data_df = data_df.reset_index()
    if 'E_grid' in data_df.columns:
        data_df['E_grid'] = data_df['E_grid'].ffill().fillna(0)

    data_df, feature_cols, target_col = feature_engineering(data_df)
    print(f"   行数={len(data_df)}, 特征维度={len(feature_cols)}, "
          f"平均负荷={data_df['E_grid'].mean():.1f} kW, "
          f"峰值={data_df['E_grid'].max():.1f} kW")

    # ==================================================================
    # 2) Scaler 拟合
    # ==================================================================
    print("\n[2/8] 归一化器拟合 (前 80% 作训练集) ...")
    X_all = data_df[feature_cols].values.astype(float)
    y_all = data_df[target_col].values.astype(float)
    train_size = int(0.80 * len(data_df))
    scaler_X = StandardScaler().fit(X_all[:train_size])
    scaler_y = StandardScaler().fit(np.log1p(y_all[:train_size]).reshape(-1, 1))

    # ==================================================================
    # 3) 模型加载
    # ==================================================================
    print("\n[3/8] 创建并加载预测模型 (best_EModel_FeatureWeight4.pth) ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")

    feature_importance = calculate_feature_importance(data_df, feature_cols, target_col)
    feat_dim = len(feature_cols)

    model = EModel_FeatureWeight4(
        feature_dim=feat_dim,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        feature_importance=feature_importance,
    ).to(device)

    model_path = os.path.join(os.path.dirname(__file__), 'best_EModel_FeatureWeight4.pth')
    if os.path.exists(model_path):
        try:
            sd = torch.load(model_path, map_location=device, weights_only=True)
            if sd['feature_importance'].size(0) == feat_dim:
                model.load_state_dict(sd)
                print("   ✅ 权重加载成功")
            else:
                print(f"   特征维度不匹配 ({sd['feature_importance'].size(0)}→{feat_dim})，转换 ...")
                converted = convert_model_weights(
                    pretrained_path=model_path,
                    new_feature_dim=feat_dim,
                    output_path=os.path.join(cfg.OUTPUT_DIR, 'converted.pth'),
                    feature_cols=feature_cols,
                    data_df=data_df, target_col=target_col,
                )
                model.load_state_dict(converted.state_dict()
                                      if hasattr(converted, 'state_dict')
                                      else torch.load(os.path.join(cfg.OUTPUT_DIR, 'converted.pth'),
                                                      map_location=device, weights_only=True))
                print("   ✅ 转换权重加载成功")
        except Exception as e:
            print(f"   ⚠ 权重加载失败: {e}; 将使用未训练模型")
    else:
        print(f"   ⚠ 未找到 {model_path}, 使用未训练模型")
    model.eval()

    # ==================================================================
    # 4) 保形预测校准 (§三.3.3)
    # ==================================================================
    print("\n[4/8] 保形预测校准 ...")
    conformal = ConformalPredictor(alpha=cfg.CONFORMAL_ALPHA)
    try:
        win = getattr(model, 'window_size', 20)
        cal_start = int(0.80 * len(data_df))
        cal_end   = min(len(data_df), cal_start + int(cfg.CAL_FRACTION * len(data_df)))
        X_cal, y_cal = [], []
        for idx in range(cal_start + win, cal_end):
            seq = X_all[idx - win: idx]
            seq_scaled = scaler_X.transform(seq)
            X_cal.append(seq_scaled)
            y_cal.append(y_all[idx])
        if len(X_cal) >= 30:
            X_cal_arr = np.stack(X_cal).astype(np.float32)
            y_cal_arr = np.asarray(y_cal, dtype=float)
            qhat = conformal.calibrate_from_sequences(
                model, X_cal_arr, y_cal_arr, device,
                scaler_y=scaler_y, use_log_y=True,
            )
            print(f"   ✅ 保形校准完成: q̂={qhat:.2f} kW, 样本数={len(X_cal)}")
        else:
            print("   ⚠ 校准样本不足，鲁棒 MPC 将使用默认 10% margin")
    except Exception as e:
        print(f"   ⚠ 保形校准失败: {e}")

    # ==================================================================
    # 5) 电价 + 动态碳强度
    # ==================================================================
    print("\n[5/8] 电价与动态碳强度构造 ...")
    prices = []
    for ts in data_df['timestamp']:
        h, wd = ts.hour, ts.weekday()
        if 8 <= h < 12 or 14 <= h < 18:
            p = 1.2
        elif 12 <= h < 14 or 18 <= h < 21:
            p = 0.8
        else:
            p = 0.4
        if wd >= 5:
            p *= 0.9
        prices.append(p)
    price_df = pd.DataFrame({'timestamp': data_df['timestamp'], 'price': prices})

    base_ef = GRID_EMISSION_FACTORS[cfg.GRID_REGION]
    carbon_intensity_hourly = make_dynamic_carbon_intensity(
        data_df['timestamp'], base_ef=base_ef,
        peak_multiplier=1.25, valley_multiplier=0.75,
    )
    carbon_tracker = CarbonTracker(
        grid_emission_factor=base_ef,
        carbon_price=cfg.CARBON_PRICE,
    )
    print(f"   电价: 峰 1.2 / 平 0.8 / 谷 0.4 元, 周末 0.9 折")
    print(f"   基准排放因子: {base_ef:.4f} tCO₂/MWh ({cfg.GRID_REGION}电网)")
    print(f"   碳价: {cfg.CARBON_PRICE:.1f} 元/tCO₂")

    # ==================================================================
    # 6) 预计算负荷预测 (跨策略共享)
    # ==================================================================
    print(f"\n[6/8] 预计算 {cfg.SIM_HOURS} + {cfg.MPC_HORIZON} 小时的负荷预测 ...")
    sim_hours = min(cfg.SIM_HOURS, len(data_df) - cfg.MPC_HORIZON - 1)
    cache_ies = StrategyAwareIES(
        capacity_kwh=cfg.BESS_CAPACITY_KWH,
        bess_power_kw=cfg.BESS_POWER_KW,
        prediction_model=model,
        feature_cols=feature_cols,
        scaler_X=scaler_X, scaler_y=scaler_y,
        strategy=BaselineStrategy(),
        verbose=False,
    )
    predictions_by_index = cache_ies.precompute_predictions(
        data_df, sim_hours, horizon=cfg.MPC_HORIZON
    )

    # ==================================================================
    # 7) 策略评估
    # ==================================================================
    print("\n[7/8] 多策略对比评估 ...")
    strategies_to_run = [
        BaselineStrategy(),                                                     # 无储能
        PeakShavingRuleStrategy(horizon=cfg.MPC_HORIZON),                       # 规则型
        EconomicMPCStrategy(horizon=cfg.MPC_HORIZON),                           # 经济 MPC
        CarbonAwareMPCStrategy(horizon=cfg.MPC_HORIZON,
                               carbon_price_cny_per_ton=cfg.CARBON_PRICE),      # 碳感知 MPC
        RobustMPCStrategy(horizon=cfg.MPC_HORIZON,
                          conformal_predictor=conformal,
                          safety_factor=1.0),                                   # 鲁棒 MPC
    ]

    strategy_results = {}
    baseline_total_cost = None
    invest_cost = (cfg.BESS_CAPACITY_KWH * cfg.BESS_INVEST_CNY_PER_KWH
                   + cfg.BESS_POWER_KW   * cfg.BESS_INVEST_CNY_PER_KW)

    for strat in strategies_to_run:
        print(f"\n  ▶ 策略: {strat.name} — {strat.description}")

        # Baseline 使用 0 容量；其余使用配置容量
        if isinstance(strat, BaselineStrategy):
            cap, pwr = 0, 0
        else:
            cap, pwr = cfg.BESS_CAPACITY_KWH, cfg.BESS_POWER_KW

        ies = StrategyAwareIES(
            capacity_kwh=cap, bess_power_kw=pwr,
            prediction_model=model,
            feature_cols=feature_cols,
            scaler_X=scaler_X, scaler_y=scaler_y,
            strategy=strat,
            conformal_predictor=conformal if isinstance(strat, RobustMPCStrategy) else None,
            carbon_tracker=carbon_tracker,
            carbon_intensity_hourly=carbon_intensity_hourly,
            allow_grid_export=False,
            verbose=False,
        )

        ts = ies.simulate_with_strategy(
            historic_data=data_df,
            time_steps=sim_hours,
            price_data=price_df,
            predictions_by_index=predictions_by_index,
            horizon=cfg.MPC_HORIZON,
        )

        eco = compute_economic_kpis(
            ts,
            investment_cost=0 if cap == 0 else invest_cost,
            baseline_total_cost=baseline_total_cost,
        )
        env = compute_environmental_kpis(ts, carbon_price_cny_per_ton=cfg.CARBON_PRICE)
        tech = compute_technical_kpis(ts)

        strategy_results[strat.name] = {
            'timeseries':    ts,
            'economic':      eco,
            'environmental': env,
            'technical':     tech,
        }

        if isinstance(strat, BaselineStrategy):
            baseline_total_cost = eco['total_cost_CNY']

        print(f"     总成本={eco['total_cost_CNY']:.1f} 元 | "
              f"CO₂={env['total_CO2_tons']:.2f} t | "
              f"峰值={tech['peak_demand_kW']:.1f} kW | "
              f"循环≈{tech['estimated_cycles']:.1f}")

    # 补充计算非 baseline 的 NPV/IRR
    for name, d in strategy_results.items():
        if name == "Baseline (No Storage)":
            continue
        d['economic'] = compute_economic_kpis(
            d['timeseries'],
            investment_cost=invest_cost,
            baseline_total_cost=baseline_total_cost,
        )

    # ==================================================================
    # 8) 综合分析 + 可视化 + 报告
    # ==================================================================
    print("\n[8/8] 综合分析 / 可视化 / 生成报告 ...")
    comp_df   = build_comparison_table(strategy_results, baseline_name="Baseline (No Storage)")
    scored_df = score_strategies(comp_df, weights=cfg.SCORE_WEIGHTS)

    print("\n【多策略对比表】")
    print(comp_df.to_string(index=False))

    print("\n【加权综合排名】")
    print(scored_df[['Strategy', 'Cost Savings (%)', 'CO2 Reduction (%)',
                     'Peak Reduction (%)', 'score']].to_string(index=False))

    # 保存 CSV
    comp_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'strategy_comparison.csv'), index=False)
    scored_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'strategy_ranked.csv'), index=False)

    # 每个策略的时序数据
    for name, d in strategy_results.items():
        safe = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        d['timeseries'].to_csv(os.path.join(cfg.OUTPUT_DIR, f'ts_{safe}.csv'), index=False)

    # 可视化 —— 9 张图，涵盖成本/碳排/峰值/雷达/热力/负荷历时/日周期/SOC 分布/时序
    generate_all_plots(
        strategy_results=strategy_results,
        comparison_df=comp_df,
        output_dir=cfg.OUTPUT_DIR,
        max_ts_hours=min(24*7, sim_hours),
    )

    # 最终文本报告
    report = format_final_report(
        strategy_results, scored_df,
        bess_config={'capacity_kwh': cfg.BESS_CAPACITY_KWH, 'power_kw': cfg.BESS_POWER_KW},
        sim_hours=sim_hours, carbon_price=cfg.CARBON_PRICE,
    )
    print("\n" + report)
    report_path = os.path.join(cfg.OUTPUT_DIR, 'final_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  ✅ 全部输出已保存至: {cfg.OUTPUT_DIR}")


# =======================================================================
if __name__ == "__main__":
    main()
