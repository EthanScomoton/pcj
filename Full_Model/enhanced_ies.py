"""
策略感知综合能源系统
=====================
对应调研报告 §四.4.2 滚动时域 MPC —— 将 .pth 预测模型连接到运行调度
- 接受任意 BaseStrategy 子类
- 预测按索引缓存（同一窗口不重复前向）
- 记录经济性 & 环保性所需的逐时指标
"""
import numpy as np
import pandas as pd

from IES import IntegratedEnergySystem
from EF  import get_renewable_forecast


class StrategyAwareIES(IntegratedEnergySystem):

    def __init__(self, *args,
                 strategy=None,
                 conformal_predictor=None,
                 carbon_tracker=None,
                 carbon_intensity_hourly=None,
                 allow_grid_export: bool = False,
                 export_price_ratio: float = 0.4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy        = strategy
        self.conformal       = conformal_predictor
        self.carbon_tracker  = carbon_tracker
        self.carbon_intensity_hourly = (None if carbon_intensity_hourly is None
                                        else np.asarray(carbon_intensity_hourly, dtype=float))
        self.allow_grid_export = allow_grid_export
        self.export_price_ratio = export_price_ratio

    # ------------------------------------------------------------------
    # 预先缓存 time_steps + horizon 个窗口的点预测，避免重复前向传播
    # Apple Silicon 优化:
    #   - MPS / CUDA: 批量推理 (省 kernel launch), 典型 10-30×
    #   - CPU:        LSTM 时序依赖 + batch 开销, 批量通常更慢 → 退回逐样本
    # ------------------------------------------------------------------
    def precompute_predictions(self, historic_data, time_steps, horizon: int = 24,
                               batch_size: int = 256):
        n_needed = time_steps + horizon
        preds = np.zeros(n_needed, dtype=float)

        # 探测设备, 决定走批量 vs 逐样本
        use_batch = False
        try:
            import torch
            if self.prediction_model is not None:
                dev = next(self.prediction_model.parameters()).device
                # 仅在加速卡上启用批量 (CPU 上 LSTM 时序批处理反而更慢)
                use_batch = dev.type in ('mps', 'cuda')
        except Exception:
            pass

        if not use_batch:
            print(f"[StrategyAwareIES] 预计算 {n_needed} 步负荷预测 (逐样本, CPU 更快) ...")
            for idx in range(n_needed):
                seq = self._build_window_sequence(historic_data, idx)
                preds[idx] = float(self.predict_demand(seq))
            print("[StrategyAwareIES] 预测缓存完成。")
            return preds

        print(f"[StrategyAwareIES] 预计算 {n_needed} 步负荷预测 (batched on {dev}) ...")

        # 尝试走批量通路 (MPS/CUDA 上显著更快)
        try:
            device = dev

            # 1) 构造所有窗口序列
            seqs = []
            for idx in range(n_needed):
                s = self._build_window_sequence(historic_data, idx)
                seqs.append(s)
            seqs = np.stack(seqs).astype(np.float32)   # [N, win, feat_dim_raw]

            # 2) 同 predict_demand 的特征维度适配 + 归一化
            #    一次性 batch 处理, 避免逐样本 scaler.transform / adapt_features
            N, W, F = seqs.shape
            flat = seqs.reshape(N * W, F)
            if hasattr(self, 'expected_feature_dim') and \
               self.expected_feature_dim is not None and \
               F != self.expected_feature_dim:
                # 裁剪或零填充
                if F > self.expected_feature_dim:
                    flat = flat[:, :self.expected_feature_dim]
                else:
                    pad = np.zeros((flat.shape[0],
                                    self.expected_feature_dim - F),
                                   dtype=np.float32)
                    flat = np.concatenate([flat, pad], axis=1)

            if self.scaler_X is not None:
                flat = self.scaler_X.transform(flat).astype(np.float32)
            seqs_scaled = flat.reshape(N, W, -1)

            # 3) 分 batch 前向
            self.prediction_model.eval()
            out_list = []
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    x = torch.from_numpy(seqs_scaled[i:i + batch_size]).to(device)
                    y = self.prediction_model(x)
                    if y.ndim > 1 and y.shape[-1] >= 2:
                        y = y[..., 0]
                    out_list.append(y.detach().cpu().numpy().flatten())
            preds_norm = np.concatenate(out_list)

            # 4) 反归一化 (log1p → StandardScaler 链)
            if self.scaler_y is not None:
                preds_inv = self.scaler_y.inverse_transform(
                    preds_norm.reshape(-1, 1)).flatten()
            else:
                preds_inv = preds_norm
            # 训练时用 log1p(y), 推理后需要 expm1
            preds = np.expm1(preds_inv)
            preds = np.maximum(preds, 0.0)
            print(f"[StrategyAwareIES] 批量推理完成 "
                  f"(batch_size={batch_size}, device={device}).")
            return preds

        except Exception as e:
            # 回退: 逐样本 (原实现)
            print(f"[StrategyAwareIES] 批量通路失败 ({e}), 回退逐样本")
            for idx in range(n_needed):
                seq = self._build_window_sequence(historic_data, idx)
                preds[idx] = float(self.predict_demand(seq))
            print("[StrategyAwareIES] 预测缓存完成 (fallback)。")
            return preds

    # ------------------------------------------------------------------
    def simulate_with_strategy(self, historic_data, time_steps,
                               price_data=None,
                               predictions_by_index=None,
                               horizon: int = 24):
        """
        使用 self.strategy 模拟 time_steps 小时的运行。

        Apple Silicon / 性能优化:
          - 把 pandas iloc 改成 numpy 数组访问 (~10× 降 overhead)
          - 可再生预报一次性向量化, 避免每步 pandas iloc+pad
          - 电价 / 碳强度 / 需求也全部预缓存
          - 结果列表分层 append, 最后一次 DataFrame 构造
        """
        assert self.strategy is not None, "必须提供 strategy"

        if predictions_by_index is None:
            predictions_by_index = self.precompute_predictions(
                historic_data, time_steps, horizon)

        dt = 1.0

        # ---- 一次性构造 numpy 缓存 ----
        # actual demand (E_total 优先, 回退 E_grid)
        if 'E_total' in historic_data.columns:
            actual_demand_arr = historic_data['E_total'].values.astype(float)
        else:
            actual_demand_arr = historic_data['E_grid'].values.astype(float)

        # 可再生 (PV + Wind) 作为长时序向量; 后面切片即可
        if 'E_PV' in historic_data.columns:
            pv_arr = historic_data['E_PV'].values.astype(float)
        else:
            pv_arr = np.zeros(len(historic_data), dtype=float)
        if 'E_wind' in historic_data.columns:
            wind_arr = historic_data['E_wind'].values.astype(float)
        else:
            wind_arr = np.zeros(len(historic_data), dtype=float)
        renewable_arr = pv_arr + wind_arr
        # 末端 pad 以便切片到 t+horizon 时不溢出
        if len(renewable_arr) < time_steps + horizon:
            renewable_arr = np.concatenate([
                renewable_arr,
                np.full(time_steps + horizon - len(renewable_arr),
                        renewable_arr[-1] if len(renewable_arr) > 0 else 0.0)
            ])

        # 电价数组
        if price_data is not None:
            price_arr = price_data['price'].values.astype(float)
            if len(price_arr) < time_steps + horizon:
                price_arr = np.concatenate([
                    price_arr,
                    np.full(time_steps + horizon - len(price_arr),
                            price_arr[-1] if len(price_arr) > 0 else 1.0)
                ])
        else:
            price_arr = np.ones(time_steps + horizon, dtype=float)

        # 碳强度数组
        if self.carbon_intensity_hourly is not None:
            ci_arr = np.asarray(self.carbon_intensity_hourly, dtype=float)
            if len(ci_arr) < time_steps + horizon:
                ci_arr = np.concatenate([
                    ci_arr,
                    np.full(time_steps + horizon - len(ci_arr),
                            ci_arr[-1] if len(ci_arr) > 0 else 0.0)
                ])
        else:
            ci_arr = None

        # timestamp 只在结果 DataFrame 需要
        ts_arr = historic_data['timestamp'].values if 'timestamp' in historic_data.columns \
                 else np.arange(time_steps)

        # predictions_by_index pad (防切片不足)
        preds_arr = np.asarray(predictions_by_index, dtype=float)
        if len(preds_arr) < time_steps + horizon:
            preds_arr = np.concatenate([
                preds_arr,
                np.full(time_steps + horizon - len(preds_arr),
                        preds_arr[-1] if len(preds_arr) > 0 else 0.0)
            ])

        # 预分配结果数组
        rec_actual_demand     = np.empty(time_steps)
        rec_predicted_demand  = np.empty(time_steps)
        rec_predicted_upper   = np.empty(time_steps)
        rec_renewable         = np.empty(time_steps)
        rec_grid              = np.empty(time_steps)
        rec_bess_power        = np.empty(time_steps)
        rec_bess_soc          = np.empty(time_steps)
        rec_cost              = np.empty(time_steps)
        rec_co2_kg            = np.empty(time_steps)
        rec_price             = np.empty(time_steps)
        rec_ef                = np.empty(time_steps)

        has_conformal = self.conformal is not None
        export_ratio = self.export_price_ratio
        allow_export = self.allow_grid_export
        ef_fallback = (self.carbon_tracker.ef if self.carbon_tracker is not None
                       else 0.0)

        for t in range(time_steps):
            # ---- 24h 切片 (numpy view, 零拷贝) ----
            pred_demand   = preds_arr[t:t + horizon]
            renewable_gen = renewable_arr[t:t + horizon]
            prices        = price_arr[t:t + horizon]
            ci_slice      = ci_arr[t:t + horizon] if ci_arr is not None else None

            pred_upper = (self.conformal.predict_upper(pred_demand)
                          if has_conformal else None)

            # ---- 调度决策 ----
            sched = self.strategy.optimize(
                bess=self.bess,
                pred_demand=pred_demand,
                renewable_gen=renewable_gen,
                grid_prices=prices,
                carbon_intensity=ci_slice,
                pred_upper=pred_upper,
            )

            # sched 里的 bess_charge/bess_discharge 可能是 list/ndarray, 仅取第 1 步
            bc = sched['bess_charge']
            bd = sched['bess_discharge']
            charge0    = float(bc[0]) if hasattr(bc, '__len__') else float(bc)
            discharge0 = float(bd[0]) if hasattr(bd, '__len__') else float(bd)

            if charge0 >= discharge0:
                actual_power = self.bess.charge(charge0, dt)
                bess_signed  = -actual_power
            else:
                actual_power = self.bess.discharge(discharge0, dt)
                bess_signed  =  actual_power

            # ---- 实际潮流 ----
            actual_demand    = actual_demand_arr[t]
            actual_renewable = renewable_arr[t]
            actual_grid = actual_demand - actual_renewable - bess_signed
            if not allow_export:
                actual_grid = max(0.0, actual_grid)

            # ---- 成本 ----
            p0 = prices[0]
            if actual_grid > 0:
                cost = actual_grid * p0
            else:
                cost = actual_grid * p0 * export_ratio

            # ---- CO2 ----
            ef_t = ci_arr[t] if ci_arr is not None else ef_fallback
            co2_kg = max(0.0, actual_grid) * ef_t

            # ---- 写入预分配数组 ----
            rec_actual_demand[t]    = actual_demand
            rec_predicted_demand[t] = pred_demand[0]
            rec_predicted_upper[t]  = (pred_upper[0] if pred_upper is not None
                                        else pred_demand[0])
            rec_renewable[t]        = actual_renewable
            rec_grid[t]             = actual_grid
            rec_bess_power[t]       = bess_signed
            rec_bess_soc[t]         = self.bess.get_soc()
            rec_cost[t]             = cost
            rec_co2_kg[t]           = co2_kg
            rec_price[t]            = p0
            rec_ef[t]               = ef_t

        # 一次性构造 DataFrame (比 8760 次 dict append 快 5-10×)
        return pd.DataFrame({
            'timestamp':            ts_arr[:time_steps],
            'actual_demand':        rec_actual_demand,
            'predicted_demand':     rec_predicted_demand,
            'predicted_upper':      rec_predicted_upper,
            'renewable_generation': rec_renewable,
            'grid_import':          rec_grid,
            'bess_power':           rec_bess_power,
            'bess_soc':             rec_bess_soc,
            'cost':                 rec_cost,
            'co2_kg':               rec_co2_kg,
            'price':                rec_price,
            'carbon_intensity':     rec_ef,
        })
