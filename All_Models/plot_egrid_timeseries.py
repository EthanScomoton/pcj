import argparse
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------ 全局绘图风格（与原代码一致） ------------------------ #
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 26,          # 全局默认字体大小
    'axes.labelsize': 26,     # 坐标轴标签大小
    'axes.titlesize': 28,     # 标题字号
    'xtick.labelsize': 24,    # x 轴刻度字号
    'ytick.labelsize': 24     # y 轴刻度字号
})

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """读取 CSV，解析时间戳并按时间排序，返回包含 timestamp 与 E_grid 的 DataFrame。"""
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns or 'E_grid' not in df.columns:
        raise ValueError(f"文件 {csv_path} 缺少必需列：'timestamp' 或 'E_grid'。")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').dropna(subset=['E_grid']).reset_index(drop=True)
    return df

def compute_global_ylim(dfs):
    """计算多个数据集的统一 y 轴范围（上下各留 5% 边距）。"""
    y_min = min(df['E_grid'].min() for df in dfs)
    y_max = max(df['E_grid'].max() for df in dfs)
    if y_max <= y_min:
        return (y_min - 1.0, y_max + 1.0)
    pad = 0.05 * (y_max - y_min)
    return (y_min - pad, y_max + pad)

def plot_egrid_timeseries(df: pd.DataFrame, title: str, outfile: Path, ylim=None):
    """按时间绘制 E_grid 折线图并保存。"""
    fig = plt.figure(figsize=(14, 3))
    # 不显式指定颜色，以遵循通用绘图规范；保留标记与线宽便于还原“点+线”风格
    plt.plot(df['timestamp'], df['E_grid'], marker='.', linewidth=1)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Timestamp')
    plt.ylabel('E_grid')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=0)

    plt.tight_layout()
    fig.savefig(outfile, dpi=240)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Plot E_grid over time from CSV files.')
    parser.add_argument('--inputs', nargs='+', required=True, help='输入 CSV 文件路径（一个或多个）。')
    parser.add_argument('--output-dir', default='.', help='输出图片目录，默认当前目录。')
    parser.add_argument('--unify-ylims', action='store_true', help='对所有图统一 y 轴范围以便比较。')
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取所有数据
    dfs = [load_and_prepare(p) for p in input_paths]

    # 统一 y 轴范围（可选）
    ylim = compute_global_ylim(dfs) if args.unify_ylims and len(dfs) > 1 else None

    # 逐个绘制
    for csv_path, df in zip(input_paths, dfs):
        title = f"E_grid over Time — {csv_path.stem}"
        outfile = outdir / f"{csv_path.stem}_egrid_timeseries.png"
        plot_egrid_timeseries(df, title=title, outfile=outfile, ylim=ylim)
        print(f"[Saved] {outfile}")

if __name__ == '__main__':
    main()
    
