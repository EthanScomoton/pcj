import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

# Input files (created earlier)
f_top3 = Path("/Users/ethanzhu/Desktop/part_top3_per_day.csv")
f_4th  = Path("/Users/ethanzhu/Desktop/part_4th_per_day.csv")
f_5th  = Path("/Users/ethanzhu/Desktop/part_5th_per_day.csv")

# Read & parse
def load_and_prepare(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").dropna(subset=["E_grid"]).reset_index(drop=True)
    return df

df_top3 = load_and_prepare(f_top3)
df_4th  = load_and_prepare(f_4th)
df_5th  = load_and_prepare(f_5th)

# Determine a consistent y-limit across all three charts for fair comparison
y_min = min(df_top3["E_grid"].min(), df_4th["E_grid"].min(), df_5th["E_grid"].min())
y_max = max(df_top3["E_grid"].max(), df_4th["E_grid"].max(), df_5th["E_grid"].max())
pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
y_lower = y_min - pad
y_upper = y_max + pad

# Helper to plot and save one figure
def plot_series(df, title):
    fig = plt.figure(figsize=(14, 3))
    plt.plot(df["timestamp"], df["E_grid"], marker=".", linewidth=1)
    plt.ylim(y_lower, y_upper)
    plt.xlabel("Timestamp")
    plt.ylabel("E_grid")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # Improve datetime ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_series(df_top3, "E_grid over Time — First Three Records per Day")
plot_series(df_4th,  "E_grid over Time — Fourth Record per Day")
plot_series(df_5th,  "E_grid over Time — Fifth Record per Day")
