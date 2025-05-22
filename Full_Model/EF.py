import numpy as np
import pandas as pd

# === 1. 特征提取模块 =====================================================
def extract_features(df, index, window_size: int = 20):
    """
    返回索引 index 处 *向前* window_size 个时间步的特征窗口，
    形状固定为 (1, window_size, feature_dim)，供 LSTM-Attention 网络直接输入。

    参数
    ----
    df : pandas.DataFrame
        已完成特征工程、按时间升序排序的数据集。
    index : int
        当前预测时刻对应的行索引 (0-based)。
    window_size : int, default 20
        模型期望的时间窗口长度。

    返回
    ----
    np.ndarray
        (1, window_size, feature_dim) 的 float32 数组。
    """
    # ---- 1. 边界处理 ----------------------------------------------------
    if index >= len(df):
        index = len(df) - 1
    if index < 0:
        raise ValueError("index 必须大于等于 0")

    # ---- 2. 选取特征列 --------------------------------------------------
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'E_grid']]

    # ---- 3. 构造滑动窗口 ------------------------------------------------
    end_idx   = index
    start_idx = max(0, end_idx - window_size + 1)
    slice_df  = df.iloc[start_idx:end_idx + 1][feature_cols]

    # 若窗口不足 window_size 行，用首行做前向填充
    if len(slice_df) < window_size:
        pad_len  = window_size - len(slice_df)
        pad_part = pd.DataFrame(
            np.repeat(slice_df.iloc[[0]].values, pad_len, axis=0),
            columns=slice_df.columns
        )
        slice_df = pd.concat([pad_part, slice_df], ignore_index=True)

    # ---- 4. 返回三维张量 -------------------------------------------------
    features = slice_df.values.astype(np.float32)          # (window_size, feature_dim)
    return np.expand_dims(features, axis=0)                # (1, window_size, feature_dim)

# === 2. 其它工具函数（保持原样，无改动） =================================
def get_feature_names(df):
    """获取特征列名称列表（排除 timestamp 与目标变量）。"""
    return [c for c in df.columns if c not in ['timestamp', 'E_grid']]

def get_feature_info(df):
    """打印数据集各特征的统计信息，用于快速浏览数据质量。"""
    print(df.describe(include='all').T)

# === 3. 经济评价模块（原文件其余内容保持不变） ===========================
def calculate_economic_metrics(costs, investment_cost,
                               discount_rate: float = 0.05,
                               lifetime: int = 10):
    """
    计算 NPV、简单回收期、内部收益率 (IRR)。
    其实现与旧版一致，此处略。
    """
    # —— 1. 现金流 --------------------------------------------------------
    baseline_cost  = costs[0]          # 基准方案
    system_cost    = np.mean(costs[1:])  # 储能方案（取平均值）
    annual_savings = baseline_cost - system_cost
    cash_flows     = [-investment_cost] + [annual_savings] * lifetime

    # —— 2. NPV ----------------------------------------------------------
    npv = sum(cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cash_flows))

    # —— 3. Payback ------------------------------------------------------
    payback_period = investment_cost / annual_savings if annual_savings > 0 else float('inf')

    # —— 4. IRR (二分法) -------------------------------------------------
    def irr_func(r):  # 内部辅助
        return sum(cf / ((1 + r) ** t) for t, cf in enumerate(cash_flows))

    # 判断净收益符号，决定搜索区间
    if irr_func(0) * irr_func(1) < 0:            # 有正实根
        low, high = 0.0, 1.0
    else:                                        # 现金流同号，IRR 在 (-0.99, 0)
        low, high = -0.99, 0.0

    while high - low > 1e-4:
        mid = (low + high) / 2
        if irr_func(mid) > 0:
            low = mid
        else:
            high = mid
    irr = low

    return {"NPV": npv, "Payback": payback_period, "IRR": irr}
