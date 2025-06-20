import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
from scipy.sparse import diags

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.linear_algebra.mc_solvers import MonteCarloLinearSolver


def create_sparse_diagonally_dominant(size):
    """Membuat matriks sparse yang diagonally dominant."""
    diagonals = [np.ones(size - 1) * -1, np.ones(size) * 4, np.ones(size - 1) * -1]
    return diags(diagonals, [-1, 0, 1], shape=(size, size)).toarray()


def main():
    # --- Konfigurasi Eksperimen untuk Core i3 ---
    scenarios = [
        (200, 8000, 0.5),
        (200, 8000, 0.9),
        (400, 10000, 0.5),
        (400, 10000, 0.9),
    ]

    # Rentang prosesor yang relevan untuk 2 Cores / 4 Threads
    processor_counts = [1, 2, 3, 4]
    epsilon = 1e-4

    results = {}

    # --- Jalankan Eksperimen untuk Setiap Skenario ---
    for size, N, gamma in scenarios:
        key = f"size_{size}_N_{N}_gamma_{gamma}"
        print(f"\n{'=' * 20}\n--- Skenario: {key} ---\n{'=' * 20}")
        results[key] = {'processors': [], 'times': []}
        A = create_sparse_diagonally_dominant(size)
        x_true = np.ones(size)
        b = A @ x_true
        solver = MonteCarloLinearSolver(A, b)

        for p_count in processor_counts:
            is_parallel = p_count > 1
            print(f"  Menjalankan dengan {p_count} prosesor...")
            start_time = time.time()
            solver.solve_slae(gamma, epsilon, N, parallel=is_parallel, num_processes=p_count)
            exec_time = time.time() - start_time
            results[key]['processors'].append(p_count)
            results[key]['times'].append(exec_time)
            print(f"    Waktu: {exec_time:.4f} detik")

    # --- Membuat Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    styles = [{'marker': 'o', 'linestyle': '-', 'color': 'C0'}, {'marker': 'o', 'linestyle': '--', 'color': 'C0'}, {'marker': 's', 'linestyle': '-', 'color': 'C1'}]
    for i, (key, data) in enumerate(results.items()):
        baseline_time = data['times'][0]
        if baseline_time > 0:
            speedup = [baseline_time / t for t in data['times']]
            size, _, gamma = key.split('_')[1], key.split('_')[3], key.split('_')[5]
            label = f"Size={size}, γ={gamma}"
            style = styles[i % len(styles)]
            ax.plot(data['processors'], speedup, marker=style['marker'], linestyle=style['linestyle'], color=style['color'], label=label)
    ax.plot(processor_counts, processor_counts, linestyle=':', color='black', alpha=0.7, label='Ideal Speedup')
    ax.set_title("Speedup of Sparse SLAE Solver", fontsize=16)
    ax.set_xlabel("Number of Processors", fontsize=12)
    ax.set_ylabel("Speedup (Waktu(1 Core) / Waktu(N Cores))", fontsize=12)
    ax.set_xticks(processor_counts)
    ax.set_xlim(left=0.8, right=max(processor_counts) * 1.05)
    ax.set_ylim(bottom=0)
    legend = ax.legend(title="Konfigurasi (Ukuran Matriks, Gamma)", fontsize=10)
    plt.setp(legend.get_title(), fontsize='11')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'repro_fig2_multi_scenario.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot disimpan di: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()