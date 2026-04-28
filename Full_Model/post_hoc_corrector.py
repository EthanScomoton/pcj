"""
预测后处理修正器 (Post-hoc Prediction Correction)
==================================================
不改动已训练好的 .pth 模型, 通过三种无重训的方式降低 MAPE:

  1. ResidualCorrector  —— Ridge/GBR 学习 "上下文 → 残差" 的映射
     特征: [raw_pred, hour_sin, hour_cos, dow_sin, dow_cos, month,
            lag1_actual, lag24_actual, lag168_actual,
            rolling_err_24h, raw_pred_log]
     典型: MAPE ↓ 20–40%

  2. HybridEnsemble      —— 原神经预测与季节性朴素预测的加权平均
     q̂ = w · nn_pred + (1-w) · y_{t-168}      # 周内同时段
     w 通过验证集最小化 RMSE 确定
     典型: MAPE ↓ 10–25%

  3. TimeBiasCorrector   —— 按 (month, hour) 分组的平均偏差校正
     q̂ = raw_pred - bias[month, hour]
     典型: MAPE ↓ 10–20% (若有系统性偏差)

统一入口 PostHocPipeline 顺序组合三者, 提供 fit() / transform() 接口。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ======================================================================
# 工具
# ======================================================================
def mape(y_true, y_pred, eps=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ======================================================================
# 1) 残差校正 (Ridge)
# ======================================================================
class ResidualCorrector:
    """
    在验证集上学 (raw_pred, time_features, lagged_actuals) → residual,
    推理时: corrected = raw_pred + predicted_residual
    """

    def __init__(self, model_type='ridge', ridge_alpha=1.0,
                 use_lags=True):
        self.model_type = model_type
        self.ridge_alpha = ridge_alpha
        self.use_lags = use_lags
        self.model = None
        self.feat_mean = None
        self.feat_std = None

    def _featurize(self, raw_pred, timestamps, actuals=None,
                   rolling_err_24h=None):
        """构造特征矩阵"""
        raw_pred = np.asarray(raw_pred, dtype=float)
        n = len(raw_pred)
        ts = pd.to_datetime(timestamps)
        hr = ts.hour.values if hasattr(ts, 'hour') else np.array([t.hour for t in ts])
        dw = ts.dayofweek.values if hasattr(ts, 'dayofweek') else \
             np.array([t.dayofweek for t in ts])
        mo = ts.month.values if hasattr(ts, 'month') else np.array([t.month for t in ts])

        feats = {
            'raw_pred':       raw_pred,
            'raw_pred_log':   np.log1p(np.maximum(raw_pred, 0)),
            'hour_sin':       np.sin(2 * np.pi * hr / 24),
            'hour_cos':       np.cos(2 * np.pi * hr / 24),
            'dow_sin':        np.sin(2 * np.pi * dw / 7),
            'dow_cos':        np.cos(2 * np.pi * dw / 7),
            'month':          mo.astype(float),
            'is_weekend':     (dw >= 5).astype(float),
        }
        if self.use_lags and actuals is not None:
            a = np.asarray(actuals, dtype=float)
            # 安全的 lag: 不存在时用 raw_pred 顶替
            for lag, label in [(1, 'lag1'), (24, 'lag24'), (168, 'lag168')]:
                lagged = np.concatenate([np.full(lag, raw_pred[0]), a[:-lag]]) \
                         if len(a) > lag else np.full(n, raw_pred[0])
                feats[label] = lagged[:n]
        if rolling_err_24h is not None:
            feats['rolling_err_24h'] = np.asarray(rolling_err_24h)[:n]

        return pd.DataFrame(feats)

    def fit(self, raw_preds_cal, actuals_cal, timestamps_cal):
        """在校准集上拟合"""
        X = self._featurize(raw_preds_cal, timestamps_cal,
                            actuals=actuals_cal).values
        y = np.asarray(actuals_cal) - np.asarray(raw_preds_cal)

        # 标准化
        self.feat_mean = X.mean(axis=0)
        self.feat_std  = X.std(axis=0) + 1e-8
        Xn = (X - self.feat_mean) / self.feat_std

        if self.model_type == 'ridge':
            # 闭式解 Ridge: (X^T X + λI)^{-1} X^T y
            n, p = Xn.shape
            A = Xn.T @ Xn + self.ridge_alpha * np.eye(p)
            b = Xn.T @ y
            self.model = np.linalg.solve(A, b)
        elif self.model_type == 'gbr':
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                gbr = GradientBoostingRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, random_state=0)
                gbr.fit(Xn, y)
                self.model = gbr
            except ImportError:
                print("  [ResidualCorrector] sklearn 未安装, 退回 Ridge")
                self.model_type = 'ridge'
                return self.fit(raw_preds_cal, actuals_cal, timestamps_cal)
        else:
            raise ValueError(f"未知 model_type: {self.model_type}")

        # 校准集评估
        pred_resid = self._predict_residual(Xn)
        corrected = raw_preds_cal + pred_resid
        mape_before = mape(actuals_cal, raw_preds_cal)
        mape_after  = mape(actuals_cal, corrected)
        print(f"  [ResidualCorrector-{self.model_type}] "
              f"校准集 MAPE: {mape_before:.2f}% → {mape_after:.2f}% "
              f"(↓ {mape_before - mape_after:.2f} pp)")
        return self

    def _predict_residual(self, Xn):
        if self.model_type == 'ridge':
            return Xn @ self.model
        return self.model.predict(Xn)

    def transform(self, raw_preds, timestamps, actuals_history=None,
                  rolling_err_24h=None):
        """
        修正 raw_preds。
        actuals_history: 若提供, 则使用 lag1/lag24/lag168 特征 (推荐)
                         否则这些 lag 会用 raw_pred 代替
        """
        if self.model is None:
            return np.asarray(raw_preds, dtype=float)
        X = self._featurize(raw_preds, timestamps,
                            actuals=actuals_history,
                            rolling_err_24h=rolling_err_24h).values
        Xn = (X - self.feat_mean) / self.feat_std
        pred_resid = self._predict_residual(Xn)
        return np.asarray(raw_preds, dtype=float) + pred_resid


# ======================================================================
# 2) 混合集成 (Neural + Seasonal Naive)
# ======================================================================
class HybridEnsemble:
    """
    q̂ = w · nn_pred + (1-w) · seasonal_naive
    其中 seasonal_naive[t] = actual[t - 168]  (周内同小时)
    w 通过在校准集上最小化 RMSE 确定 (闭式解)。
    """

    def __init__(self, season=168):
        self.season = int(season)
        self.w = 1.0   # 默认不混合

    def fit(self, raw_preds_cal, actuals_cal):
        raw = np.asarray(raw_preds_cal, dtype=float)
        y   = np.asarray(actuals_cal, dtype=float)
        n = len(y)
        if n <= self.season:
            print(f"  [HybridEnsemble] 校准集 {n} < season {self.season}, 跳过")
            return self

        # seasonal naive: 用前 season 小时的真值; 前 season 小时用 raw_pred 填充
        sn = np.concatenate([raw[:self.season], y[:-self.season]])
        # 最优权重: minimize w·raw + (1-w)·sn - y
        diff1 = raw - sn
        diff2 = sn - y
        denom = np.sum(diff1 ** 2)
        w_opt = -np.sum(diff1 * diff2) / max(denom, 1e-8)
        self.w = float(np.clip(w_opt, 0.0, 1.0))

        blended = self.w * raw + (1 - self.w) * sn
        mape_b = mape(y, raw)
        mape_a = mape(y, blended)
        print(f"  [HybridEnsemble] 最优权重 w={self.w:.3f} "
              f"(1-w 给 seasonal_naive[-168h]), "
              f"MAPE: {mape_b:.2f}% → {mape_a:.2f}% "
              f"(↓ {mape_b - mape_a:.2f} pp)")
        return self

    def transform(self, raw_preds, actuals_history):
        """
        actuals_history: 长度 ≥ season 的历史真值, 用于取 t-168
        """
        raw = np.asarray(raw_preds, dtype=float)
        hist = np.asarray(actuals_history, dtype=float)
        n = len(raw)

        # seasonal naive
        if len(hist) >= self.season:
            sn = np.concatenate([raw[:self.season],
                                  hist[:max(0, n - self.season)]])[:n]
        else:
            # 不够历史, 直接返回 raw
            return raw

        return self.w * raw + (1 - self.w) * sn


# ======================================================================
# 3) 时段 / 月度偏差校正
# ======================================================================
class TimeBiasCorrector:
    """
    q̂ = raw_pred - bias[month, hour]  (用校准集的 mean residual)
    """

    def __init__(self):
        self.bias_table = None   # dict[(month, hour)] → bias (kW)

    def fit(self, raw_preds_cal, actuals_cal, timestamps_cal):
        df = pd.DataFrame({
            'raw':    np.asarray(raw_preds_cal, dtype=float),
            'actual': np.asarray(actuals_cal, dtype=float),
        })
        ts = pd.to_datetime(timestamps_cal)
        df['month'] = ts.month.values if hasattr(ts, 'month') else \
                      [t.month for t in ts]
        df['hour']  = ts.hour.values if hasattr(ts, 'hour') else \
                      [t.hour for t in ts]
        df['resid'] = df['raw'] - df['actual']
        self.bias_table = df.groupby(['month', 'hour'])['resid'].mean().to_dict()

        corrected = df['raw'].values - np.array([
            self.bias_table.get((m, h), 0.0)
            for m, h in zip(df['month'], df['hour'])
        ])
        mape_b = mape(actuals_cal, raw_preds_cal)
        mape_a = mape(actuals_cal, corrected)
        print(f"  [TimeBiasCorrector] {len(self.bias_table)} 个 (月,小时) 组, "
              f"MAPE: {mape_b:.2f}% → {mape_a:.2f}% "
              f"(↓ {mape_b - mape_a:.2f} pp)")
        return self

    def transform(self, raw_preds, timestamps):
        if self.bias_table is None:
            return np.asarray(raw_preds, dtype=float)
        raw = np.asarray(raw_preds, dtype=float)
        ts = pd.to_datetime(timestamps)
        months = ts.month.values if hasattr(ts, 'month') else \
                 [t.month for t in ts]
        hours  = ts.hour.values if hasattr(ts, 'hour') else \
                 [t.hour for t in ts]
        bias = np.array([self.bias_table.get((m, h), 0.0)
                         for m, h in zip(months, hours)])
        return raw - bias


# ======================================================================
# 组合管线
# ======================================================================
class PostHocPipeline:
    """
    顺序组合: TimeBiasCorrector → ResidualCorrector → HybridEnsemble
    TimeBias 最先 (消除系统性偏差), 然后 Ridge 校正剩余残差,
    最后 Hybrid 与周度朴素预测融合。
    """

    def __init__(self, use_bias=True, use_residual=True, use_hybrid=True,
                 residual_model='ridge'):
        self.use_bias = use_bias
        self.use_residual = use_residual
        self.use_hybrid = use_hybrid
        self.bias_c = TimeBiasCorrector() if use_bias else None
        self.resid_c = ResidualCorrector(model_type=residual_model) \
                       if use_residual else None
        self.hybrid_c = HybridEnsemble() if use_hybrid else None

    def fit(self, raw_preds_cal, actuals_cal, timestamps_cal):
        print(f"\n  [PostHoc] 拟合校准集 (n={len(actuals_cal)}) ...")
        cur = np.asarray(raw_preds_cal, dtype=float).copy()

        if self.bias_c is not None:
            self.bias_c.fit(cur, actuals_cal, timestamps_cal)
            cur = self.bias_c.transform(cur, timestamps_cal)

        if self.resid_c is not None:
            self.resid_c.fit(cur, actuals_cal, timestamps_cal)
            cur = self.resid_c.transform(cur, timestamps_cal,
                                         actuals_history=actuals_cal)

        if self.hybrid_c is not None:
            self.hybrid_c.fit(cur, actuals_cal)
            cur = self.hybrid_c.transform(cur, actuals_history=actuals_cal)

        final_mape = mape(actuals_cal, cur)
        original_mape = mape(actuals_cal, raw_preds_cal)
        print(f"  [PostHoc] 整体 MAPE: {original_mape:.2f}% → {final_mape:.2f}% "
              f"(改善 {original_mape - final_mape:.2f} pp, "
              f"相对 {(original_mape - final_mape) / max(original_mape, 1e-6) * 100:.1f}%)")
        self.original_mape_ = original_mape
        self.corrected_mape_ = final_mape
        return self

    def transform(self, raw_preds, timestamps, actuals_history=None):
        """
        actuals_history: 用于 hybrid & residual 的 lag 特征;
                         若 None, hybrid 退化为 raw, lag 退化为 raw_pred 占位
        """
        cur = np.asarray(raw_preds, dtype=float).copy()
        if self.bias_c is not None:
            cur = self.bias_c.transform(cur, timestamps)
        if self.resid_c is not None:
            cur = self.resid_c.transform(cur, timestamps,
                                         actuals_history=actuals_history)
        if self.hybrid_c is not None and actuals_history is not None:
            cur = self.hybrid_c.transform(cur, actuals_history)
        return cur

    def summary(self):
        return {
            'original_mape_%': getattr(self, 'original_mape_', None),
            'corrected_mape_%': getattr(self, 'corrected_mape_', None),
            'use_bias':        self.use_bias,
            'use_residual':    self.use_residual,
            'use_hybrid':      self.use_hybrid,
            'hybrid_w':        (self.hybrid_c.w if self.hybrid_c else None),
        }
