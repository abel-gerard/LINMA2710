import matplotlib.pyplot as plt
import numpy as np


def plot_perf(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = {}
    for line in lines:
        name, dim, time = line.strip().split(',')
        data.setdefault(name, []).append((int(dim), np.float64(time)))

    for name, values in data.items():
        values.sort(key=lambda x: x[0])
        dims, times = zip(*values)
        plt.loglog(dims, times, marker='o', label=name)

    plt.xlabel('Dimension')
    plt.ylabel('Time')
    plt.title('Performance Comparison')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_perf("perf.csv")