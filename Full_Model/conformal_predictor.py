"""
保形预测包装器 (Conformal Prediction)
=====================================
对应调研报告 §三、3.3 不确定性量化——推荐方案
- 无分布假设，覆盖率有理论保证 ( 1 - α )
- 后处理方法，不改变原有 .pth 模型
- 输出点预测 + 预测区间，供鲁棒 MPC 使用
"""
import numpy as np
import torch


class ConformalPredictor:
    """
    使用校准集的绝对残差分位数作为非一致性分数 q̂
    predict_upper / predict_interval 返回覆盖率约 (1-α) 的区间
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.q_hat = None
        self.calibration_residuals = None

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
        residuals = []
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
                residuals.extend(np.abs(pred - y_true))

        self.calibration_residuals = np.asarray(residuals, dtype=float)
        m = len(residuals)
        q_level = min(np.ceil((m + 1) * (1 - self.alpha)) / m, 1.0)
        self.q_hat = float(np.quantile(self.calibration_residuals, q_level))
        return self.q_hat

    def calibrate_from_residuals(self, residuals):
        """若已有残差数组可直接校准"""
        self.calibration_residuals = np.asarray(residuals, dtype=float)
        m = len(residuals)
        q_level = min(np.ceil((m + 1) * (1 - self.alpha)) / m, 1.0)
        self.q_hat = float(np.quantile(self.calibration_residuals, q_level))
        return self.q_hat

    # --------------------------------------------------------------
    def predict_upper(self, point_pred):
        point_pred = np.asarray(point_pred, dtype=float)
        if self.q_hat is None:
            return point_pred * 1.10
        return point_pred + self.q_hat

    def predict_lower(self, point_pred):
        point_pred = np.asarray(point_pred, dtype=float)
        if self.q_hat is None:
            return point_pred * 0.90
        return np.maximum(0.0, point_pred - self.q_hat)

    def predict_interval(self, point_pred):
        return self.predict_lower(point_pred), self.predict_upper(point_pred)

    def summary(self) -> dict:
        return {
            'alpha': self.alpha,
            'coverage_target': 1 - self.alpha,
            'q_hat': self.q_hat,
            'calibration_size': 0 if self.calibration_residuals is None
                                    else len(self.calibration_residuals),
        }
