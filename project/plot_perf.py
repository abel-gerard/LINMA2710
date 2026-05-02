import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys


PERF_DIR = "performance"
FILES = {
    "matrix":             os.path.join(PERF_DIR, "matrix.csv"),
    "omp_t1":             os.path.join(PERF_DIR, "omp_t1.csv"),
    "omp_t2":             os.path.join(PERF_DIR, "omp_t2.csv"),
    "omp_t4":             os.path.join(PERF_DIR, "omp_t4.csv"),
    "omp_t8":             os.path.join(PERF_DIR, "omp_t8.csv"),
    "omp_t16":            os.path.join(PERF_DIR, "omp_t16.csv"),
    "distributed_matrix": os.path.join(PERF_DIR, "distributed_matrix.csv"),
    "opencl_naive":       os.path.join(PERF_DIR, "opencl_naive.csv"),
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
    ax.set_xlabel(fr"Matrix dimension $(N\times N)$")
    ax.set_ylabel(fr"Time ($µs$)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

COLORS = {
    "matrix":             "#ff00dd",
    "omp_t1":             "#8400ff",
    "omp_t2":             "#001aff",
    "omp_t4":             "#21ebf2",
    "omp_t8":             "#21f2ac",
    "omp_t16":            "#90f221",
    "distributed_matrix": "#FF9800",
    "opencl_naive":       "#F44336",
    "opencl_tiled_4":     "#81C784",
    "opencl_tiled_8":     "#388E3C",
    "opencl_tiled_16":    "#1B5E20",
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

style_loglog(ax1, "OpenCL (all methods)")
fig1.tight_layout()
fig1.savefig(os.path.join(PERF_DIR, "plot_all_methods.svg"))

fig2, ax2 = plt.subplots(figsize=(8, 5))
for key in ("opencl_naive", "opencl_tiled_16"):
    if dfs[key] is not None:
        plot_with_band(ax2, dfs[key], {"opencl_naive": "OpenCL naive", "opencl_tiled_16": "OpenCL tiled (TILE=16)"}[key], COLORS[key])

style_loglog(ax2, "OpenCL (naive vs tiled)")
fig2.tight_layout()
fig2.savefig(os.path.join(PERF_DIR, "plot_opencl_comparison.svg"))

fig3, ax3 = plt.subplots(figsize=(8, 5))
TILE_LABELS = {
    "opencl_tiled_4":  "Tiled TILE=4",
    "opencl_tiled_8":  "Tiled TILE=8",
    "opencl_tiled_16": "Tiled TILE=16",
}
for key, label in TILE_LABELS.items():
    if dfs[key] is not None:
        plot_with_band(ax3, dfs[key], label, COLORS[key])

style_loglog(ax3, "OpenCL tiled (tile size comparison)")
fig3.tight_layout()
fig3.savefig(os.path.join(PERF_DIR, "plot_tile_comparison.svg"))

fig4, ax4 = plt.subplots(figsize=(8, 5))
OMP_LABELS = {
    "matrix":      "matrix (no OMP)",
    "omp_t1":      "matrix (with OMP, T=1)",
    "omp_t2":      "matrix (with OMP, T=2)",
    "omp_t4":      "matrix (with OMP, T=4)",
    "omp_t8":      "matrix (with OMP, T=8)",
    "omp_t16":     "matrix (with OMP, T=16)",
}
for key in ("matrix", "omp_t1", "omp_t2", "omp_t4", "omp_t8", "omp_t16"):
    if dfs[key] is not None:
        plot_with_band(ax4, dfs[key], OMP_LABELS[key], COLORS[key])

style_loglog(ax4, "Matrix (w/ vs w/o OMP)")
fig4.tight_layout()
fig4.savefig(os.path.join(PERF_DIR, "plot_matrix_comparison.svg"))

plt.show()
