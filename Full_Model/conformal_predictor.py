"""
保形预测包装器 (Conformal Prediction)
=====================================
对应调研报告 §三、3.3 不确定性量化——推荐方案
- 无分布假设，覆盖率有理论保证 ( 1 - α )
- 后处理方法，不改变原有 .pth 模型
- 输出点预测 + 预测区间，供鲁棒 MPC 使用

支持三种模式:
  - 'absolute'  : 固定宽度区间 ŷ ± q̂
  - 'normalized': 区间宽度随预测量级自适应缩放
                  s_i = |y_i - ŷ_i| / max(|ŷ_i|, ε)
                  区间 = ŷ ± q̂_norm * max(|ŷ|, ε)
  - 'cqr'       : 条件保形分位数回归 (Romano et al. 2019, NeurIPS)
                  需外部提供 q̂_low(x), q̂_high(x) 分位预测
                  s_i = max(q̂_low - y_i, y_i - q̂_high)
                  区间 = [q̂_low - Q̂, q̂_high + Q̂]
                  比 absolute 宽度通常窄 30-50%
"""
import numpy as np
import torch


class ConformalPredictor:
    """
    使用校准集的残差分位数作为非一致性分数 q̂
    predict_upper / predict_interval 返回覆盖率约 (1-α) 的区间

    Parameters
    ----------
    alpha : float
        显著性水平, 目标覆盖率 = 1 - alpha
    mode : str
        'absolute' — 固定宽度 ŷ ± q̂
        'normalized' — 归一化宽度 ŷ ± q̂_norm * max(|ŷ|, ε)
    norm_eps : float
        归一化模式下防止除零的下限 (kW)
    """

    def __init__(self, alpha: float = 0.1, mode: str = 'normalized',
                 norm_eps: float = 1000.0):
        self.alpha = float(alpha)
        self.mode = mode          # 'absolute' / 'normalized' / 'cqr'
        self.norm_eps = float(norm_eps)
        self.q_hat = None
        self.calibration_residuals = None

        # CQR 模式: 存校准集上的分位函数输出, 以便在新样本上查询
        # 实际上 CQR 使用外部分位估计器, 这里只缓存 Q̂
        self._cqr_calibrated = False

    # --------------------------------------------------------------
    def calibrate_from_sequences(self,
                                 model,
                                 X_cal,   # ndarray [N, seq_len, feat_dim]  已归一化
                                 y_cal,   # ndarray [N]  原尺度真实值
                                 device,
                                 scaler_y=None,
                                 use_log_y: bool = True,
                                 batch_size: int = 128):
        """用验证集计算 q̂；X_cal 期望已使用 scaler_X 归一化"""
        model.eval()
        abs_residuals = []
        predictions = []
        n = len(X_cal)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = X_cal[i:i + batch_size]
                x = torch.tensor(batch, dtype=torch.float32).to(device)
                pred = model(x).cpu().numpy().flatten()
                # 反归一化
                if scaler_y is not None:
                    pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
                    if use_log_y:
                        pred = np.expm1(pred)
                y_true = np.asarray(y_cal[i:i + batch_size]).flatten()
                abs_residuals.extend(np.abs(pred - y_true))
                predictions.extend(pred)

        abs_residuals = np.asarray(abs_residuals, dtype=float)
        predictions = np.asarray(predictions, dtype=float)

        if self.mode == 'normalized':
            # 归一化非一致性分数: s = |r| / max(|ŷ|, ε)
            scales = np.maximum(np.abs(predictions), self.norm_eps)
            scores = abs_residuals / scales
            self.calibration_residuals = scores
        else:
            self.calibration_residuals = abs_residuals

        m = len(self.calibration_residuals)
        q_level = min(np.ceil((m + 1) * (1 - self.alpha)) / m, 1.0)
        self.q_hat = float(np.quantile(self.calibration_residuals, q_level))
        return self.q_hat

    def calibrate_from_residuals(self, residuals, predictions=None):
        """
        若已有残差数组可直接校准。

        Parameters
        ----------
        residuals : array-like
            绝对残差 |y - ŷ|
        predictions : array-like, optional
            对应预测值 ŷ，normalized 模式下必须提供
        """
        abs_residuals = np.asarray(residuals, dtype=float)
        if self.mode == 'normalized':
            if predictions is None:
                # 退回 absolute 模式
                print("  [ConformalPredictor] 警告: normalized 模式需要 predictions 参数, "
                      "退回 absolute 模式")
                self.calibration_residuals = abs_residuals
            else:
                predictions = np.asarray(predictions, dtype=float)
                scales = np.maximum(np.abs(predictions), self.norm_eps)
                self.calibration_residuals = abs_residuals / scales
        else:
            self.calibration_residuals = abs_residuals

        m = len(self.calibration_residuals)
        q_level = min(np.ceil((m + 1) * (1 - self.alpha)) / m, 1.0)
        self.q_hat = float(np.quantile(self.calibration_residuals, q_level))
        return self.q_hat

    # --------------------------------------------------------------
    def predict_upper(self, point_pred):
        point_pred = np.asarray(point_pred, dtype=float)
        if self.q_hat is None:
            return point_pred * 1.10
        if self.mode == 'normalized':
            scales = np.maximum(np.abs(point_pred), self.norm_eps)
            return point_pred + self.q_hat * scales
        return point_pred + self.q_hat

    def predict_lower(self, point_pred):
        point_pred = np.asarray(point_pred, dtype=float)
        if self.q_hat is None:
            return point_pred * 0.90
        if self.mode == 'normalized':
            scales = np.maximum(np.abs(point_pred), self.norm_eps)
            return np.maximum(0.0, point_pred - self.q_hat * scales)
        return np.maximum(0.0, point_pred - self.q_hat)

    def predict_interval(self, point_pred):
        return self.predict_lower(point_pred), self.predict_upper(point_pred)

    # --------------------------------------------------------------
    # CQR (Conditional Quantile Regression) —— Romano et al. 2019
    # --------------------------------------------------------------
    def calibrate_cqr(self, q_low_cal, q_high_cal, y_cal):
        """
        CQR 校准: 用分位预测器在校准集上的输出 (q̂_low, q̂_high) 与真值 y 计算
                   非一致性分数 s = max(q̂_low - y, y - q̂_high), 取 (1-α) 分位。

        Parameters
        ----------
        q_low_cal  : array [m]   校准集上的下分位预测 (α/2)
        q_high_cal : array [m]   校准集上的上分位预测 (1-α/2)
        y_cal      : array [m]   校准集真值

        Returns
        -------
        q_hat : float   CQR 校正量 (原尺度, 可加减直接修正区间)
        """
        q_low_cal  = np.asarray(q_low_cal,  dtype=float)
        q_high_cal = np.asarray(q_high_cal, dtype=float)
        y_cal      = np.asarray(y_cal,      dtype=float)

        # 非一致性分数: 真值落在区间外的超出量, 落在区间内则 ≤ 0
        scores = np.maximum(q_low_cal - y_cal, y_cal - q_high_cal)

        self.mode = 'cqr'
        self.calibration_residuals = scores
        m = len(scores)
        q_level = min(np.ceil((m + 1) * (1 - self.alpha)) / m, 1.0)
        self.q_hat = float(np.quantile(scores, q_level))
        self._cqr_calibrated = True
        return self.q_hat

    def cqr_interval(self, q_low_new, q_high_new):
        """
        对新样本的分位预测 (q̂_low, q̂_high) 应用 CQR 校正:
          lower = q̂_low  - Q̂
          upper = q̂_high + Q̂

        若 Q̂ > 0 → 扩宽区间 (分位预测欠覆盖)
        若 Q̂ < 0 → 收窄区间 (分位预测过覆盖, CQR 能自动变窄!)
        """
        if not self._cqr_calibrated or self.q_hat is None:
            # 未校准 → 直接返回原始分位
            return np.asarray(q_low_new), np.asarray(q_high_new)

        q_low_new  = np.asarray(q_low_new,  dtype=float)
        q_high_new = np.asarray(q_high_new, dtype=float)
        lower = np.maximum(0.0, q_low_new  - self.q_hat)
        upper = q_high_new + self.q_hat

        # 修复 Plot 8: 当 Q̂ < 0 (CQR 收窄) 且原始分位预测倒序 / 接近时,
        # 可能产生 upper < lower (负宽度). 强制保证 upper ≥ lower.
        # 倒序时取中点 ± 1e-3 % 的微宽区间, 让 violin/scatter 不出负宽度
        bad = upper < lower
        if np.any(bad):
            mid = (lower[bad] + upper[bad]) / 2
            lower[bad] = mid - 1e-6 * np.abs(mid).clip(min=1.0)
            upper[bad] = mid + 1e-6 * np.abs(mid).clip(min=1.0)
        return lower, upper

    def summary(self) -> dict:
        return {
            'alpha': self.alpha,
            'coverage_target': 1 - self.alpha,
            'q_hat': self.q_hat,
            'mode': self.mode,
            'calibration_size': 0 if self.calibration_residuals is None
                                    else len(self.calibration_residuals),
            'cqr_calibrated': self._cqr_calibrated,
        }
