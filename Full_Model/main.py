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
    StochasticMPCStrategy,
)
from conformal_predictor import ConformalPredictor
from carbon_module       import (
    CarbonTracker, make_dynamic_carbon_intensity,
    GRID_EMISSION_FACTORS,
)
from analysis           import (
    compute_economic_kpis, compute_environmental_kpis, compute_technical_kpis,
    build_comparison_table, score_strategies,
    generate_all_plots,
    format_final_report,
    pareto_front, decision_recommender, plot_pareto_front,
)
from experiments        import run_all_experiments


# =======================================================================
# 平台配置
# =======================================================================
class PlatformConfig:
    # 仿真 (8760h = 1 年, 覆盖春夏秋冬 4 季)
    SIM_HOURS       = 8760       # 8760h = 365 天 (P0 要求: 覆盖 4 季)
    MPC_HORIZON     = 24
    WARM_UP_HOURS   = 24          # 丢弃前 24h 以消除模型 warm-up 影响 (P0)

    # 储能配置 (P1: exp06 最优容量 ≈ 162 MWh)
    BESS_CAPACITY_KWH = 162000    # 162 MWh (P1 推荐: 从 exp06 NPV 最大化得出)
    BESS_POWER_KW     = 129600    # 保持 C-rate = 0.8 (162 * 0.8)
    BESS_INVEST_CNY_PER_KWH = 600
    BESS_INVEST_CNY_PER_KW  = 200

    # 电价
    DEMAND_CHARGE_CNY_PER_KW_MONTH = 38.0   # 需量电费 (工业用户典型值 30-42 元/kW/月)

    # 碳
    GRID_REGION    = 'east'       # 华东
    CARBON_PRICE   = 120.0        # 元/tCO2 (CEA 趋势上行，2024 已突破 100)
    CARBON_SENSITIVITY = 3.0      # 碳成本在 MPC 目标函数中的放大系数
    USE_REAL_EF    = False        # P2: True → 使用真实月度 marginal EF (real_ef_loader)

    # 保形预测
    CONFORMAL_ALPHA = 0.10         # 90% 覆盖率
    CAL_FRACTION    = 0.15         # 训练后 15% 作为校准集（增大以稳定分位数估计）
    USE_CQR         = True         # P1: True → 用 CQR 代替 absolute-residual CP，区间宽度↓ 30-50%

    # UQ 基线
    UQ_METHOD       = 'mc_dropout'  # 'mc_dropout' (P0 推荐) / 'reparam' (旧, 有缺陷) / 'deep_ensembles'
    MC_DROPOUT_N    = 30           # MC Dropout 采样次数

    # 后处理 MAPE 修正 (不重训模型, 通过 bias + ridge 残差 + seasonal naive 融合)
    USE_POST_HOC_CORRECTION   = True
    POST_HOC_TRIGGER_MAPE     = 5.0     # 原始 MAPE > 该值才启用后处理
    POST_HOC_RESIDUAL_MODEL   = 'ridge'  # 'ridge' / 'gbr' (需 sklearn)

    # 噪声敏感性
    NOISE_MC_TRIALS = 5           # 每个噪声水平 MC 重采样次数 (5 次够统计稳定, 速度快 4 倍; 想更严谨可调回 20)

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
    print("\n[1/9] 数据加载 & 特征工程 ...")
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

    # ---- 保存原始能量值 (feature_engineering 会对 E_PV/E_wind 做 LabelEncode) ----
    for _ecol in ['E_PV', 'E_wind', 'E_storage_discharge']:
        if _ecol in data_df.columns:
            data_df[f'{_ecol}_kWh'] = pd.to_numeric(data_df[_ecol], errors='coerce').fillna(0.0)
        else:
            data_df[f'{_ecol}_kWh'] = 0.0

    data_df, feature_cols, target_col = feature_engineering(data_df)

    # ---- 构建港口真实总需求: E_total = E_grid + PV + Wind + 原有储能放电 ----
    data_df['E_total'] = (data_df['E_grid']
                          + data_df['E_PV_kWh']
                          + data_df['E_wind_kWh']
                          + data_df['E_storage_discharge_kWh'])

    print(f"   行数={len(data_df)}, 特征维度={len(feature_cols)}")
    print(f"   港口总负荷: 平均={data_df['E_total'].mean():.1f} kW, "
          f"峰值={data_df['E_total'].max():.1f} kW")
    print(f"   其中电网购电: {data_df['E_grid'].mean():.1f} kW | "
          f"风电: {data_df['E_wind_kWh'].mean():.1f} kW | "
          f"光伏: {data_df['E_PV_kWh'].mean():.1f} kW | "
          f"原有储能: {data_df['E_storage_discharge_kWh'].mean():.1f} kW")

    # ==================================================================
    # 2) Scaler 拟合
    # ==================================================================
    print("\n[2/9] 归一化器拟合 (前 80% 作训练集) ...")
    X_all = data_df[feature_cols].values.astype(float)
    y_all = data_df[target_col].values.astype(float)
    train_size = int(0.80 * len(data_df))
    scaler_X = StandardScaler().fit(X_all[:train_size])
    scaler_y = StandardScaler().fit(np.log1p(y_all[:train_size]).reshape(-1, 1))

    # ==================================================================
    # 3) 模型加载
    # ==================================================================
    print("\n[3/9] 创建并加载预测模型 (best_EModel_FeatureWeight4.pth) ...")
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
    print("\n[4/9] 保形预测校准 ...")
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
            unit = '' if conformal.mode == 'normalized' else ' kW'
            print(f"   ✅ 保形校准完成 ({conformal.mode}): q̂={qhat:.4f}{unit}, "
                  f"样本数={len(X_cal)}")
        else:
            print("   ⚠ 校准样本不足，鲁棒 MPC 将使用默认 10% margin")
    except Exception as e:
        print(f"   ⚠ 保形校准失败: {e}")

    # ==================================================================
    # 5) 电价 + 动态碳强度
    # ==================================================================
    print("\n[5/9] 电价与动态碳强度构造 ...")
    # ---- 分时电价（工商业大工业电价，含尖峰时段）----
    prices = []
    for ts in data_df['timestamp']:
        h, wd = ts.hour, ts.weekday()
        if 10 <= h < 12 or 14 <= h < 17:       # 尖峰（午高峰 + 下午高峰）
            p = 1.50
        elif 8 <= h < 10 or 12 <= h < 14 or 17 <= h < 19:   # 峰
            p = 1.10
        elif 19 <= h < 22 or 6 <= h < 8:       # 平
            p = 0.70
        else:                                    # 谷 (22-6)
            p = 0.30
        if wd >= 5:
            p *= 0.90
        prices.append(p)
    price_df = pd.DataFrame({'timestamp': data_df['timestamp'], 'price': prices})

    base_ef = GRID_EMISSION_FACTORS[cfg.GRID_REGION]
    # P2: 支持真实 marginal EF (CEC 月度 + 小时形态 / electricityMaps API)
    if getattr(cfg, 'USE_REAL_EF', False):
        try:
            from real_ef_loader import make_real_carbon_intensity
            ef_source = getattr(cfg, 'EF_SOURCE', 'auto')   # 'cec' / 'electricitymaps' / 'auto'
            carbon_intensity_hourly = make_real_carbon_intensity(
                data_df, region=cfg.GRID_REGION, source=ef_source)
            print(f"   [P2] 使用真实 marginal EF ({ef_source})")
        except Exception as e:
            print(f"   [P2] 真实 EF 加载失败, 回退合成包络: {e}")
            carbon_intensity_hourly = make_dynamic_carbon_intensity(
                data_df['timestamp'], base_ef=base_ef,
                peak_multiplier=1.40, valley_multiplier=0.60)
    else:
        carbon_intensity_hourly = make_dynamic_carbon_intensity(
            data_df['timestamp'], base_ef=base_ef,
            peak_multiplier=1.40, valley_multiplier=0.60,     # 拉大峰/谷碳强度差
        )
    carbon_tracker = CarbonTracker(
        grid_emission_factor=base_ef,
        carbon_price=cfg.CARBON_PRICE,
    )
    print(f"   电价: 尖峰 1.50 / 峰 1.10 / 平 0.70 / 谷 0.30 元, 周末 0.9 折")
    print(f"   需量电费: {cfg.DEMAND_CHARGE_CNY_PER_KW_MONTH:.0f} 元/kW/月")
    print(f"   基准排放因子: {base_ef:.4f} tCO₂/MWh ({cfg.GRID_REGION}电网)")
    print(f"   碳价: {cfg.CARBON_PRICE:.1f} 元/tCO₂  (目标函数灵敏度 ×{cfg.CARBON_SENSITIVITY})")

    # ==================================================================
    # 6) 预计算负荷预测 (跨策略共享)
    # ==================================================================
    print(f"\n[6/9] 预计算 {cfg.SIM_HOURS} + {cfg.MPC_HORIZON} 小时的负荷预测 ...")
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

    # ---- 恢复原始可再生数据 (供仿真阶段 get_renewable_forecast 使用) ----
    # feature_engineering 已对 E_PV/E_wind 做了 LabelEncode, 需要还原为 kWh
    data_df['E_PV']   = data_df['E_PV_kWh']
    data_df['E_wind']  = data_df['E_wind_kWh']

    # ---- 将模型预测从 E_grid → E_total ----
    # 模型预测的是电网购电量，需加回可再生 + 原有储能才是港口总需求
    print("   将预测值从 E_grid 校正为 E_total ...")
    for idx in range(len(predictions_by_index)):
        safe_idx = min(idx, len(data_df) - 1)
        predictions_by_index[idx] += (
            float(data_df.iloc[safe_idx]['E_PV_kWh'])
            + float(data_df.iloc[safe_idx]['E_wind_kWh'])
            + float(data_df.iloc[safe_idx]['E_storage_discharge_kWh'])
        )
    print(f"   校正后预测均值: {predictions_by_index[:sim_hours].mean():.1f} kW "
          f"(实际 E_total 均值: {data_df['E_total'].iloc[:sim_hours].mean():.1f} kW)")

    # ---- 后处理 MAPE 修正 (不重训模型, 通过 bias/ridge/hybrid 降 MAPE) ----
    if getattr(cfg, 'USE_POST_HOC_CORRECTION', True):
        try:
            from post_hoc_corrector import PostHocPipeline, mape
            # 用训练/验证分界到数据末尾的段作为校准集; 避免污染仿真段
            ph_cal_start = int(0.80 * len(data_df))
            ph_cal_end   = min(len(data_df), len(predictions_by_index))
            if ph_cal_end - ph_cal_start >= 200:
                cal_preds   = predictions_by_index[ph_cal_start:ph_cal_end]
                cal_actuals = data_df['E_total'].values[ph_cal_start:ph_cal_end]
                cal_ts      = data_df['timestamp'].values[ph_cal_start:ph_cal_end]

                raw_mape = mape(cal_actuals, cal_preds)
                print(f"\n   [post-hoc] 原始预测 MAPE = {raw_mape:.2f}%")

                if raw_mape > getattr(cfg, 'POST_HOC_TRIGGER_MAPE', 5.0):
                    pipeline = PostHocPipeline(
                        use_bias=True, use_residual=True, use_hybrid=True,
                        residual_model=getattr(cfg, 'POST_HOC_RESIDUAL_MODEL',
                                               'ridge'))
                    pipeline.fit(cal_preds, cal_actuals, cal_ts)
                    # 应用到整个 predictions_by_index
                    all_ts = pd.concat([
                        data_df['timestamp'],
                        pd.Series(pd.date_range(
                            start=data_df['timestamp'].iloc[-1],
                            periods=len(predictions_by_index) - len(data_df) + 1,
                            freq='H')[1:]) if len(predictions_by_index) > len(data_df)
                        else pd.Series([], dtype='datetime64[ns]'),
                    ], ignore_index=True).values[:len(predictions_by_index)]
                    # 用整个 E_total 作为 hybrid/lag 的历史 (只使用当前点之前的数据)
                    predictions_by_index = pipeline.transform(
                        predictions_by_index, all_ts,
                        actuals_history=data_df['E_total'].values)

                    # 再次评估修正后 MAPE (同一校准集段)
                    new_preds_cal = predictions_by_index[ph_cal_start:ph_cal_end]
                    new_mape = mape(cal_actuals, new_preds_cal)
                    print(f"   [post-hoc] 修正后 MAPE = {new_mape:.2f}%  "
                          f"(改善 {raw_mape - new_mape:.2f} pp, "
                          f"相对 ↓ {(raw_mape - new_mape)/raw_mape*100:.1f}%)")
                else:
                    print(f"   [post-hoc] MAPE 已 ≤ {getattr(cfg, 'POST_HOC_TRIGGER_MAPE', 5.0)}%, "
                          f"跳过后处理")
            else:
                print(f"   [post-hoc] 校准集样本不足 ({ph_cal_end - ph_cal_start}), 跳过")
        except Exception as e:
            print(f"   [post-hoc] 后处理修正失败, 保留原始预测: {e}")

    # ---- 丢弃 warm-up: 前 WARM_UP_HOURS 小时模型未见足够历史 (P0) ----
    warm_up = int(getattr(cfg, 'WARM_UP_HOURS', 24))
    if warm_up > 0 and warm_up < sim_hours:
        print(f"   丢弃前 {warm_up} 小时 warm-up 数据 (P0 修复)")
        # predictions_by_index 按索引对齐; data_df 也需要同步偏移
        predictions_by_index = predictions_by_index[warm_up:]
        data_df = data_df.iloc[warm_up:].reset_index(drop=True)
        sim_hours = min(sim_hours - warm_up, len(predictions_by_index))
        print(f"   warm-up 后有效 sim_hours = {sim_hours}")

    # ---- 在 E_total 空间重校准保形预测器 ----
    # 原 q_hat 基于 E_grid 残差 (step 4)，现在切换到 E_total 残差以保持一致
    # 使用 normalized 模式: 残差按预测量级归一化，区间宽度自适应
    # P1: 若 USE_CQR=True 则额外训练 CQR, 用 MC Dropout 提供分位估计
    print("   在 E_total 空间重校准保形预测器 ...")
    cal_start = int(0.80 * len(data_df))
    cal_end   = min(len(data_df), cal_start + int(cfg.CAL_FRACTION * len(data_df)))
    cal_end   = min(cal_end, len(predictions_by_index))
    cqr_predictor = None   # 独立的 CQR 预测器 (与原 normalized CP 并存供对比)
    cqr_quantile_fn = None
    if cal_end - cal_start >= 30:
        cal_preds   = predictions_by_index[cal_start:cal_end]
        cal_actuals = data_df['E_total'].values[cal_start:cal_end]
        cal_resid   = np.abs(cal_preds - cal_actuals)
        old_qhat = conformal.q_hat
        old_mode  = conformal.mode
        conformal.calibrate_from_residuals(cal_resid, predictions=cal_preds)
        print(f"   q_hat: {old_qhat:.0f} -> {conformal.q_hat:.4f} "
              f"({'normalized' if conformal.mode == 'normalized' else 'kW'})"
              f"  (校准样本={cal_end - cal_start})")
        # 验证: 打印校准集上的实际覆盖率和典型区间宽度
        lo, hi = conformal.predict_interval(cal_preds)
        actual_cov = np.mean((cal_actuals >= lo) & (cal_actuals <= hi))
        avg_width = np.mean(hi - lo)
        print(f"   校准集覆盖率: {actual_cov:.1%}, 平均区间宽度: {avg_width:.0f} kW")

        # ---- CQR 构建 (P1) ----
        if getattr(cfg, 'USE_CQR', False):
            print("   [P1] 构建 CQR 预测器 (MC Dropout 分位估计 + 保形校正) ...")
            try:
                from uq_methods import mc_dropout_interval
                mc_n = getattr(cfg, 'MC_DROPOUT_N', 30)
                win  = getattr(model, 'window_size', 20)
                # 在校准集上获取 MC Dropout 的 α/2 和 1-α/2 分位
                X_cal_cqr = []
                for idx_cal in range(cal_start, cal_end):
                    start_c = max(0, idx_cal - win + 1)
                    seq_c = data_df[feature_cols].iloc[start_c:idx_cal + 1].values.astype(np.float32)
                    if len(seq_c) < win:
                        seq_c = np.pad(seq_c, ((win - len(seq_c), 0), (0, 0)), mode='edge')
                    X_cal_cqr.append(scaler_X.transform(seq_c).astype(np.float32))
                X_cal_cqr = np.stack(X_cal_cqr)
                _, ql_cal, qh_cal = mc_dropout_interval(
                    model, X_cal_cqr, n_samples=mc_n, alpha=cfg.CONFORMAL_ALPHA,
                    scaler_y=scaler_y, use_log_y=True, device=device)
                # 校正 MC Dropout 分位: 加回 E_PV + E_wind + 原储能放电 (和主预测一致)
                for j, idx_cal in enumerate(range(cal_start, cal_end)):
                    offset = (float(data_df.iloc[idx_cal]['E_PV_kWh'])
                              + float(data_df.iloc[idx_cal]['E_wind_kWh'])
                              + float(data_df.iloc[idx_cal]['E_storage_discharge_kWh']))
                    ql_cal[j] += offset
                    qh_cal[j] += offset

                cqr_predictor = ConformalPredictor(alpha=cfg.CONFORMAL_ALPHA, mode='cqr')
                q_hat_cqr = cqr_predictor.calibrate_cqr(ql_cal, qh_cal, cal_actuals)
                # 校准集覆盖和宽度
                lo_cqr, hi_cqr = cqr_predictor.cqr_interval(ql_cal, qh_cal)
                cov_cqr = np.mean((cal_actuals >= lo_cqr) & (cal_actuals <= hi_cqr))
                width_cqr = np.mean(hi_cqr - lo_cqr)
                print(f"   CQR q_hat: {q_hat_cqr:+.0f} kW  "
                      f"(校准覆盖率: {cov_cqr:.1%}, 平均宽度: {width_cqr:.0f} kW)")
                # 构建一个闭包: 新样本 → (q_low, q_high)
                # 简化方案: 所有策略共享 MC Dropout 输出 (推理时查表)
                # 为避免每步前向 N 次, 预计算在 step 6 并传递
            except Exception as e:
                print(f"   ⚠ CQR 构建失败: {e}; 继续使用 normalized CP")
                cqr_predictor = None
    else:
        print("   校准样本不足，保留原始 q_hat")

    # ==================================================================
    # 7) 策略评估
    # ==================================================================
    print("\n[7/9] 多策略对比评估 ...")
    pcw = cfg.DEMAND_CHARGE_CNY_PER_KW_MONTH / 30.0   # 日化需量电费 → MPC 削峰权重
    strategies_to_run = [
        BaselineStrategy(),                                                     # 无储能
        PeakShavingRuleStrategy(horizon=cfg.MPC_HORIZON),                       # 规则型
        EconomicMPCStrategy(horizon=cfg.MPC_HORIZON,
                            peak_charge_weight=pcw),                            # 经济 MPC
        CarbonAwareMPCStrategy(horizon=cfg.MPC_HORIZON,
                               carbon_price_cny_per_ton=cfg.CARBON_PRICE,
                               peak_charge_weight=pcw,
                               carbon_sensitivity=cfg.CARBON_SENSITIVITY),      # 碳感知 MPC
        RobustMPCStrategy(horizon=cfg.MPC_HORIZON,
                          conformal_predictor=conformal,
                          safety_factor=1.0,
                          peak_charge_weight=pcw),                              # 鲁棒 MPC
        StochasticMPCStrategy(horizon=cfg.MPC_HORIZON,
                              conformal_predictor=conformal,
                              n_scenarios=getattr(cfg, 'STOCHASTIC_N_SCENARIOS', 10),
                              cvar_beta=0.2,
                              peak_charge_weight=pcw),                          # P2: 随机 MPC
    ]

    # ==================================================================
    # P0: 训练 DRL-SAC 基线并加入评估队列
    # ==================================================================
    try:
        from drl_strategy import BESSDispatchEnv, SACTrainer, DRLSACStrategy
        print("\n   [P0] 训练 SAC DRL 基线 ...")
        bess_params = {'min_soc': 0.1, 'max_soc': 0.9,
                       'eff_c': 0.95, 'eff_d': 0.95}
        train_hours = int(min(sim_hours, 0.7 * len(data_df)))
        env_train = BESSDispatchEnv(
            data_df=data_df.iloc[:train_hours].copy(),
            predictions_by_index=predictions_by_index[:train_hours],
            price_arr=price_df['price'].values[:train_hours],
            carbon_ef_arr=carbon_intensity_hourly[:train_hours],
            capacity_kwh=cfg.BESS_CAPACITY_KWH,
            power_kw=cfg.BESS_POWER_KW,
            bess_params=bess_params,
            demand_charge_rate=cfg.DEMAND_CHARGE_CNY_PER_KW_MONTH,
        )
        sac = SACTrainer(state_dim=env_train.state_dim(), action_dim=1,
                         hidden=128, lr=3e-4, device=device)
        # P0: 最小可用训练量 (可在 cfg 扩大)
        sac_steps = int(getattr(cfg, 'SAC_TRAIN_STEPS', 3000))
        sac.train(env_train, total_steps=sac_steps, warm_up=200,
                  batch_size=128, log_every=max(500, sac_steps // 10),
                  verbose=True)

        # 评估时使用同一环境模板做归一化参考
        env_eval = BESSDispatchEnv(
            data_df=data_df.copy(),
            predictions_by_index=predictions_by_index,
            price_arr=price_df['price'].values,
            carbon_ef_arr=carbon_intensity_hourly,
            capacity_kwh=cfg.BESS_CAPACITY_KWH,
            power_kw=cfg.BESS_POWER_KW,
            bess_params=bess_params,
            demand_charge_rate=cfg.DEMAND_CHARGE_CNY_PER_KW_MONTH,
        )
        strategies_to_run.append(DRLSACStrategy(sac, env_eval,
                                                horizon=cfg.MPC_HORIZON))
        print("   [P0] SAC 训练完成, 已加入策略队列")
    except Exception as e:
        print(f"   ⚠ SAC 训练/加入失败, 跳过: {e}")

    strategy_results = {}
    baseline_total_cost = None
    invest_cost = (cfg.BESS_CAPACITY_KWH * cfg.BESS_INVEST_CNY_PER_KWH
                   + cfg.BESS_POWER_KW   * cfg.BESS_INVEST_CNY_PER_KW)

    for strat in strategies_to_run:
        print(f"\n  ▶ 策略: {strat.name} — {strat.description}")

        # DRL 策略需要在每次仿真前重置时间计数器
        if hasattr(strat, 'reset'):
            strat.reset()

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
            demand_charge_rate=cfg.DEMAND_CHARGE_CNY_PER_KW_MONTH,
            bess_capacity_kwh=cap,
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

        print(f"     总成本={eco['total_cost_CNY']:.1f} 元 "
              f"(电量={eco['energy_cost_CNY']:.0f}+需量={eco['demand_charge_CNY']:.0f}) | "
              f"CO₂={env['total_CO2_tons']:.2f} t | "
              f"峰值={tech['peak_demand_kW']:.1f} kW | "
              f"循环≈{tech['estimated_cycles']:.1f}")

    # 补充计算非 baseline 的 NPV/IRR（含需量电费节省 + O&M + 衰减）
    for name, d in strategy_results.items():
        if name == "Baseline (No Storage)":
            continue
        d['economic'] = compute_economic_kpis(
            d['timeseries'],
            investment_cost=invest_cost,
            baseline_total_cost=baseline_total_cost,
            demand_charge_rate=cfg.DEMAND_CHARGE_CNY_PER_KW_MONTH,
            bess_capacity_kwh=cfg.BESS_CAPACITY_KWH,
        )

    # ==================================================================
    # 8) 综合分析 + 可视化 + 报告
    # ==================================================================
    print("\n[8/9] 综合分析 / 可视化 / 生成报告 ...")
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

    # 年化口径对比 CSV (× 8760/sim_hours)
    annual_factor = 8760.0 / max(1, sim_hours)
    ann_df = comp_df.copy()
    # 可年化的量纲列
    annualize_cols = {
        'Total Cost (CNY)', 'Energy Cost (CNY)', 'Demand Charge (CNY)',
        'CO2 (tons)', 'Carbon Cost (CNY)',
    }
    for col in annualize_cols:
        if col in ann_df.columns:
            ann_df[col] = ann_df[col] * annual_factor
    ann_df.insert(1, 'Annual Factor', annual_factor)
    ann_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'strategy_comparison_annualized.csv'),
                  index=False)
    print(f"   年化 CSV (× {annual_factor:.2f}) 已保存")

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

    # ---- P2: Pareto 前沿 + 多情境决策推荐器 ----
    print("\n   [P2] Pareto 前沿分析 + 多情境决策推荐 ...")
    pareto_df, _ = pareto_front(comp_df)
    pareto_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'pareto_front.csv'), index=False)
    print(f"   Pareto 最优策略 ({len(pareto_df)} 个):")
    for nm in pareto_df['Strategy'].values:
        print(f"     - {nm}")

    recs = {}
    for profile in ['economic', 'green', 'peak', 'balanced', 'grid_indep']:
        r = decision_recommender(comp_df, user_profile=profile)
        recs[profile] = r['recommended_strategy']
        print(f"   [{profile:<10s}] 推荐: {r['recommended_strategy']:<28s}"
              f" (score={r['recommended_score']:.3f})")

    # Pareto + 雷达可视化
    try:
        plot_pareto_front(
            comp_df,
            save_path=os.path.join(cfg.OUTPUT_DIR, '10_pareto_front.png'))
    except Exception as e:
        print(f"   ⚠ Pareto 绘图失败: {e}")

    # 保存决策推荐结果
    import json
    rec_payload = {
        'pareto_strategies':  pareto_df['Strategy'].tolist(),
        'recommendations':    recs,
    }
    with open(os.path.join(cfg.OUTPUT_DIR, 'decision_recommender.json'),
              'w', encoding='utf-8') as f:
        json.dump(rec_payload, f, indent=2, ensure_ascii=False)

    # 最终文本报告
    report = format_final_report(
        strategy_results, scored_df,
        bess_config={'capacity_kwh': cfg.BESS_CAPACITY_KWH, 'power_kw': cfg.BESS_POWER_KW},
        sim_hours=sim_hours, carbon_price=cfg.CARBON_PRICE,
        demand_charge_rate=cfg.DEMAND_CHARGE_CNY_PER_KW_MONTH,
    )
    # 追加 Pareto + 推荐段落到报告末尾
    report += "\n\n" + "=" * 80
    report += "\n【Pareto 前沿 + 多情境决策推荐器 (P2)】\n"
    report += "-" * 80 + "\n"
    report += f"Pareto 最优策略集合 ({len(pareto_df)} 个):\n"
    for nm in pareto_df['Strategy'].values:
        report += f"  - {nm}\n"
    report += "\n不同用户偏好下的推荐:\n"
    report += f"  {'偏好':<12} {'推荐策略'}\n"
    for profile, rec in recs.items():
        report += f"  {profile:<12} {rec}\n"
    report += "=" * 80 + "\n"

    print("\n" + report)
    report_path = os.path.join(cfg.OUTPUT_DIR, 'final_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  ✅ 基础分析已保存至: {cfg.OUTPUT_DIR}")

    # ==================================================================
    # 9) 高级分析实验 (8 项)
    # ==================================================================
    print("\n[9/9] 高级分析实验 (CP验证 / UQ对比 / 鲁棒性 / 碳敏感性 / 计时 / BESS优化 / TOU / 极端事件) ...")
    run_all_experiments(
        strategy_results=strategy_results,
        data_df=data_df,
        predictions_by_index=predictions_by_index,
        conformal=conformal,
        model=model,
        cfg=cfg,
        price_df=price_df,
        carbon_intensity_hourly=carbon_intensity_hourly,
        carbon_tracker=carbon_tracker,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        sim_hours=sim_hours,
        baseline_total_cost=baseline_total_cost,
        invest_cost=invest_cost,
        output_dir=cfg.OUTPUT_DIR,
    )
    print(f"\n  ✅ 全部输出（含实验）已保存至: {cfg.OUTPUT_DIR}")


# =======================================================================
if __name__ == "__main__":
    main()
