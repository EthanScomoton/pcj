from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 直接在此处填写三份 CSV 的“绝对路径”
CSV_FILES = [
    Path('/Users/ethanzhu/Desktop/part_top3_per_day.csv'),
    Path('/Users/ethanzhu/Desktop/part_4th_per_day.csv'),
    Path('/Users/ethanzhu/Desktop/part_5th_per_day.csv'),
]

# 按文件名指定颜色
COLOR_MAP = {
    'part_top3_per_day.csv': '#1f77b4',
    'part_4th_per_day.csv': '#ff7f0e',
    'part_5th_per_day.csv': '#2ca02c',
}

# 统一固定的 y 轴范围
FIXED_Y_LIM = (0, 400000)

# ------------------------ 全局绘图风格 ------------------------ #
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 26,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24
})

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """读取 CSV，解析时间戳并按时间排序，返回包含 timestamp 与 E_grid 的 DataFrame。"""
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns or 'E_grid' not in df.columns:
        raise ValueError(f"文件 {csv_path} 缺少必需列：'timestamp' 或 'E_grid'。")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').dropna(subset=['E_grid']).reset_index(drop=True)
    return df

def remove_outliers_and_interpolate(df: pd.DataFrame, cap_value: float = 400000.0) -> pd.DataFrame:
    """将 E_grid 中大于 cap_value 的点置为 NaN，并使用按时间插值实现平滑过渡。"""
    cleaned = df.copy()
    # 标记并移除超过阈值的点
    cleaned.loc[cleaned['E_grid'] > cap_value, 'E_grid'] = np.nan
    # 按时间索引做插值，平滑过渡
    cleaned = cleaned.set_index('timestamp')
    cleaned['E_grid'] = cleaned['E_grid'].interpolate(method='time', limit_direction='both')
    cleaned = cleaned.reset_index()
    return cleaned

def plot_egrid_timeseries(df: pd.DataFrame, color: str, ylim=FIXED_Y_LIM):
    fig = plt.figure(figsize=(14, 3))
    plt.plot(df['timestamp'], df['E_grid'], marker='.', linewidth=1, color=color)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Timestamp')
    plt.ylabel('E_grid (kWh)')
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=0)

    plt.tight_layout()
    return fig

def main():
    # 校验路径数量与存在性
    if len(CSV_FILES) != 3:
        raise ValueError("请在 CSV_FILES 中填写三个 CSV 文件的绝对路径。")
    for p in CSV_FILES:
        if not p.exists():
            raise FileNotFoundError(f"找不到文件：{p}")

    # 读取并预处理数据（删除 >400000 的点并平滑插值过渡）
    dfs = [load_and_prepare(p) for p in CSV_FILES]
    dfs = [remove_outliers_and_interpolate(df, cap_value=FIXED_Y_LIM[1]) for df in dfs]

    # 逐个绘制（不保存，只显示），无标题，统一 y 轴为 (0, 400000)，颜色按文件名指定
    for csv_path, df in zip(CSV_FILES, dfs):
        color = COLOR_MAP.get(csv_path.name, '#1f77b4')
        plot_egrid_timeseries(df, color=color, ylim=FIXED_Y_LIM)

    # 一次性显示所有图窗口
    plt.show()

if __name__ == '__main__':
    main()