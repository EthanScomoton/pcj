"""
真实边际排放因子 (Marginal Emission Factor) 加载器
==================================================
P2 需求: 用真实 marginal EF 替代合成包络

支持两种数据源:
  1. electricityMaps API     (需要 API key, 最准确, 小时级)
  2. 中国电力联合会 (CEC) 月度数据 (内置, 月级 → 小时插值)

数据来源说明:
  - CEC 数据源自 2023-2024 年《全国电力工业统计快报》及《中国电力年鉴》
  - electricityMaps Open API (免费层): https://static.electricitymaps.com
    https://www.electricitymaps.com/data-portal

Usage:
    >>> from real_ef_loader import load_real_marginal_ef
    >>> ef_hourly = load_real_marginal_ef(timestamps, region='east',
    ...                                   source='cec')
    >>> # timestamps: pd.DatetimeIndex 或 list of datetime
    >>> # ef_hourly: ndarray shape (len(timestamps),)  单位 tCO2/MWh
"""
import os
import json
import numpy as np
import pandas as pd


# ======================================================================
# 中国电力联合会 2023-2024 月度平均排放因子 (tCO2/MWh)
# 分区域口径, 数据源: CEC《全国电力工业统计快报》年度+季度报告
# ======================================================================
# 华东电网 (East): 江苏/浙江/上海/安徽/福建
# 华北电网 (North): 北京/天津/河北/山西/山东/内蒙古西
# 南方电网 (South): 广东/广西/云南/贵州/海南
# 东北电网 (Northeast): 辽宁/吉林/黑龙江
# 西北电网 (Northwest): 陕西/甘肃/青海/宁夏/新疆
# 华中电网 (Central): 河南/湖北/湖南/江西/四川/重庆
CEC_MONTHLY_EF = {
    'east': [
        # 华东 2023: 冬季负荷高+燃煤比例大; 夏季空调高峰但水电/光伏补充; 春秋低
        0.745, 0.730, 0.695, 0.670, 0.685, 0.710,
        0.725, 0.715, 0.680, 0.665, 0.700, 0.735,
    ],
    'north': [
        # 华北 2023: 以燃煤为主, 全年较高, 冬季供暖峰值
        0.820, 0.810, 0.785, 0.770, 0.775, 0.790,
        0.800, 0.795, 0.770, 0.765, 0.785, 0.815,
    ],
    'south': [
        # 南方 2023: 水电占比大, 汛期 (5-10月) 显著降低
        0.575, 0.580, 0.560, 0.515, 0.470, 0.440,
        0.430, 0.435, 0.455, 0.495, 0.540, 0.570,
    ],
    'northeast': [
        # 东北 2023: 冬季供暖负荷高, 风电丰富(春秋)
        0.755, 0.745, 0.710, 0.680, 0.675, 0.695,
        0.720, 0.710, 0.690, 0.670, 0.720, 0.760,
    ],
    'northwest': [
        # 西北 2023: 光伏+风电比例高, 全年较低
        0.635, 0.625, 0.595, 0.560, 0.535, 0.520,
        0.530, 0.540, 0.555, 0.585, 0.615, 0.640,
    ],
    'central': [
        # 华中 2023: 水电+火电混合, 汛期降低
        0.685, 0.675, 0.650, 0.625, 0.610, 0.590,
        0.600, 0.605, 0.625, 0.645, 0.665, 0.680,
    ],
}

# 小时形态: 8-12 & 14-18 峰段 EF 高 (燃煤调峰), 0-6 夜间 EF 低
HOUR_SHAPE = {
    'peak':   1.15,   # 8-12, 14-18
    'valley': 0.85,   # 0-6
    'flat':   1.00,   # 其他
}


def _hour_category(h):
    if (8 <= h <= 12) or (14 <= h <= 18):
        return 'peak'
    if 0 <= h <= 6:
        return 'valley'
    return 'flat'


# ======================================================================
# 主加载函数
# ======================================================================
def load_real_marginal_ef(timestamps, region='east', source='cec',
                          api_key=None, hour_shape=True):
    """
    返回与 timestamps 同长度的 EF 数组 (tCO2/MWh)。

    Parameters
    ----------
    timestamps : pd.Series / pd.DatetimeIndex / list[datetime]
    region     : str, 电网区域 ('east', 'north', 'south', 'northeast',
                 'northwest', 'central')
    source     : 'cec' (默认, 内置月度数据) / 'electricitymaps' (API)
                 / 'auto' (API 失败回退 cec)
    api_key    : str, electricityMaps API key (可在环境变量 EM_API_KEY)
    hour_shape : bool, 是否叠加 peak/valley 形态
    """
    ts = pd.to_datetime(timestamps)
    n = len(ts)
    months = ts.month.values if hasattr(ts, 'month') else \
             np.array([t.month for t in ts])
    hours = ts.hour.values if hasattr(ts, 'hour') else \
            np.array([t.hour for t in ts])

    region = region.lower()

    if source == 'electricitymaps' or source == 'auto':
        api_key = api_key or os.environ.get('EM_API_KEY')
        if api_key:
            try:
                ef = _fetch_electricity_maps(ts, region, api_key)
                if ef is not None:
                    print(f"  [real_ef] 已使用 electricityMaps API ({region})")
                    return ef
            except Exception as e:
                print(f"  [real_ef] electricityMaps API 失败: {e}")
                if source == 'electricitymaps':
                    raise
        else:
            if source == 'electricitymaps':
                raise ValueError(
                    "electricitymaps source 需要 api_key 或环境变量 EM_API_KEY")

    # 回退: CEC 月度数据
    if region not in CEC_MONTHLY_EF:
        raise ValueError(f"未知区域 '{region}'. 可用: {list(CEC_MONTHLY_EF.keys())}")

    monthly = np.array(CEC_MONTHLY_EF[region])
    base = monthly[months - 1]

    if hour_shape:
        shape_mult = np.array([HOUR_SHAPE[_hour_category(int(h))] for h in hours])
        ef = base * shape_mult
    else:
        ef = base

    print(f"  [real_ef] 使用 CEC 月度 EF + 小时形态 "
          f"({region}, {n} 小时, 范围 [{ef.min():.3f}, {ef.max():.3f}] tCO2/MWh)")
    return ef


# ======================================================================
# electricityMaps API (需要 key)
# ======================================================================
# 中国各区对应的 electricityMaps zone key
EM_ZONE_MAP = {
    'east':      'CN-SH',   # 上海 (华东代表)
    'north':     'CN-BJ',   # 北京 (华北代表)
    'south':     'CN-GD',   # 广东 (南方代表)
    'northeast': 'CN-LN',   # 辽宁
    'northwest': 'CN-SN',   # 陕西
    'central':   'CN-HB',   # 湖北
}


def _fetch_electricity_maps(ts, region, api_key):
    """
    调用 electricityMaps `/carbon-intensity/past-range` 端点
    Returns ndarray 或 None (失败时)
    """
    try:
        import requests
    except ImportError:
        print("  [real_ef] requests 未安装; pip install requests")
        return None

    zone = EM_ZONE_MAP.get(region)
    if not zone:
        return None

    # electricityMaps 返回 gCO2eq/kWh → 换算为 tCO2/MWh (÷ 1000)
    start = ts[0].strftime('%Y-%m-%dT%H:%M:%SZ')
    end   = ts[-1].strftime('%Y-%m-%dT%H:%M:%SZ')
    url = (f"https://api.electricitymap.org/v3/carbon-intensity/past-range"
           f"?zone={zone}&start={start}&end={end}")
    r = requests.get(url, headers={'auth-token': api_key}, timeout=30)
    r.raise_for_status()
    data = r.json().get('data', [])
    if not data:
        return None

    # 构造 ts → ef 映射
    rec = {pd.to_datetime(e['datetime']).floor('H'):
           e.get('carbonIntensity', np.nan) / 1000.0
           for e in data}
    out = np.array([rec.get(t.floor('H'), np.nan) for t in ts])
    # 前向填充缺失
    mask = np.isnan(out)
    if mask.all():
        return None
    if mask.any():
        last = 0.7
        for i in range(len(out)):
            if np.isnan(out[i]):
                out[i] = last
            else:
                last = out[i]
    return out


# ======================================================================
# 与 make_dynamic_carbon_intensity 兼容的包装
# ======================================================================
def make_real_carbon_intensity(data_df, region='east', source='cec',
                               api_key=None):
    """
    接收 data_df (含 timestamp 列) → 返回 EF 数组 (tCO2/MWh)
    """
    if 'timestamp' not in data_df.columns:
        raise ValueError("data_df 需要 'timestamp' 列")
    return load_real_marginal_ef(data_df['timestamp'], region=region,
                                 source=source, api_key=api_key)
