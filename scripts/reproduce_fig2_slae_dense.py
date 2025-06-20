import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.linear_algebra.mc_solvers import MonteCarloLinearSolver


def create_dense_diagonally_dominant(size):
    """Membuat matriks dense yang dijamin diagonally dominant."""
    A = np.random.rand(size, size)
    diag_values = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A)) + 1
    np.fill_diagonal(A, diag_values)
    return A



def main():
    # --- Konfigurasi Eksperimen untuk Core i3 ---
    # Ukuran yang lebih kecil agar tidak terlalu lama di Core i3
    matrix_sizes_to_test = [100, 200]

    # Rentang prosesor yang relevan untuk 2 Cores / 4 Threads
    processor_counts = [1, 2, 3, 4]

    # Parameter MC dijaga konstan
    gamma = 0.5
    N = 4000  # Beban kerja yang wajar untuk dense matrix
    epsilon = 1e-4

    results = {}

    # --- Jalankan Eksperimen ---
    for size in matrix_sizes_to_test:
        key = f"size_{size}"
        print(f"\n{'=' * 20}\n--- Skenario DENSE: {key} ---\n{'=' * 20}")
        results[key] = {'processors': [], 'times': []}
        A = create_dense_diagonally_dominant(size)
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

    # --- Membuat Plot Profesional ---
    # ... (Bagian plotting Anda sudah benar, tidak perlu diubah) ...
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    markers = ['o', 's', 'D', '^', 'v']
    for i, (key, data) in enumerate(results.items()):
        baseline_time = data['times'][0]
        if baseline_time > 0:
            speedup = [baseline_time / t for t in data['times']]
            label = f"Size = {key.split('_')[1]}"
            ax.plot(data['processors'], speedup, marker=markers[i % len(markers)], linestyle='-', color=colors[i], label=label)
    ax.plot(processor_counts, processor_counts, linestyle='--', color='red', alpha=0.8, label='Ideal Speedup')
    ax.set_title("Speedup of Dense SLAE Solver for Various Matrix Sizes", fontsize=16)
    ax.set_xlabel("Number of Processors", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_xticks(processor_counts)
    ax.legend(title="Matrix Size", fontsize=10)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'repro_fig2_dense_multi_size.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot profesional disimpan di: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()