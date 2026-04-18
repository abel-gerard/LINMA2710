#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys

PERF_DIR = "performance"
FILES = {
    "matrix":             os.path.join(PERF_DIR, "matrix.csv"),
    "distributed_matrix": os.path.join(PERF_DIR, "distributed_matrix.csv"),
    "opencl_naive":       os.path.join(PERF_DIR, "opencl_naive.csv"),
    "opencl_tiled":       os.path.join(PERF_DIR, "opencl_tiled_16.csv"),
    "opencl_tiled_4":     os.path.join(PERF_DIR, "opencl_tiled_4.csv"),
    "opencl_tiled_8":     os.path.join(PERF_DIR, "opencl_tiled_8.csv"),
    "opencl_tiled_16":    os.path.join(PERF_DIR, "opencl_tiled_16.csv"),
}

def load(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.", file=sys.stderr)
        return None
    return pd.read_csv(path)

def plot_with_band(ax, df, label, color):
    ax.plot(df["dim"], df["avg_us"], label=label, color=color, marker="o", markersize=4)
    ax.fill_between(
        df["dim"],
        df["avg_us"] - df["std_us"],
        df["avg_us"] + df["std_us"],
        alpha=0.2,
        color=color,
    )

def style_loglog(ax, title):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Matrix dimension (N×N)")
    ax.set_ylabel("Time (µs)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

COLORS = {
    "matrix":             "#2196F3",  # blue         - CPU
    "distributed_matrix": "#FF9800",  # orange       - distributed
    "opencl_naive":       "#F44336",  # red          - naive GPU
    "opencl_tiled":       "#1B5E20",  # dark green   - tiled GPU (16)
    "opencl_tiled_4":     "#81C784",  # light green  - tiled TILE=4
    "opencl_tiled_8":     "#388E3C",  # mid green    - tiled TILE=8
    "opencl_tiled_16":    "#1B5E20",  # dark green   - tiled TILE=16
}

dfs = {k: load(v) for k, v in FILES.items()}

os.makedirs(PERF_DIR, exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(8, 5))
LABELS = {
    "matrix":             "CPU",
    "distributed_matrix": "Distributed (MPI)",
    "opencl_naive":       "OpenCL naive",
}
for key, label in LABELS.items():
    if dfs[key] is not None:
        plot_with_band(ax1, dfs[key], label, COLORS[key])

style_loglog(ax1, "Matrix multiplication — all methods")
fig1.tight_layout()
fig1.savefig(os.path.join(PERF_DIR, "plot_all_methods.pdf"))

fig2, ax2 = plt.subplots(figsize=(8, 5))
for key in ("opencl_naive", "opencl_tiled"):
    if dfs[key] is not None:
        plot_with_band(ax2, dfs[key], {"opencl_naive": "OpenCL naive", "opencl_tiled": "OpenCL tiled (TILE=16)"}[key], COLORS[key])

style_loglog(ax2, "OpenCL — naive vs tiled")
fig2.tight_layout()
fig2.savefig(os.path.join(PERF_DIR, "plot_opencl_comparison.pdf"))

fig3, ax3 = plt.subplots(figsize=(8, 5))
TILE_LABELS = {
    "opencl_tiled_4":  "Tiled TILE=4",
    "opencl_tiled_8":  "Tiled TILE=8",
    "opencl_tiled_16": "Tiled TILE=16",
}
for key, label in TILE_LABELS.items():
    if dfs[key] is not None:
        plot_with_band(ax3, dfs[key], label, COLORS[key])

style_loglog(ax3, "OpenCL tiled — tile size comparison")
fig3.tight_layout()
fig3.savefig(os.path.join(PERF_DIR, "plot_tile_comparison.pdf"))
print("Saved performance/plot_tile_comparison.pdf")

plt.show()