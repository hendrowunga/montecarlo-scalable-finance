import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
from scipy.sparse import diags

# Tambahkan path src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Impor kelasnya
from src.linear_algebra.mc_solvers import MonteCarloLinearSolver


# ... (Fungsi create_sparse_diagonally_dominant tetap sama) ...
def create_sparse_diagonally_dominant(size):
    diagonals = [np.ones(size - 1) * -1, np.ones(size) * 4, np.ones(size - 1) * -1]
    return diags(diagonals, [-1, 0, 1], shape=(size, size)).toarray()


def main():
    matrix_sizes = [500]  # Naikkan dari 200 ke 1000

    # Gunakan lebih banyak prosesor jika Anda punya
    processor_counts = [1, 2, 4, 6, 8]  # Sesuaikan dengan jumlah inti CPU Anda

    gamma = 0.5

    # Ini yang paling penting: tingkatkan beban kerja secara signifikan
    N = 10000  # Naikkan dari 1000 ke 50,000
    epsilon = 1e-4

    results = {}

    # --- Jalankan Eksperimen ---
    for size in matrix_sizes:
        print(f"\n--- Menguji ukuran matriks: {size}x{size} ---")
        A = create_sparse_diagonally_dominant(size)
        # x_true dan b tidak perlu diubah, hanya untuk setup
        x_true = np.ones(size)
        b = A @ x_true

        # Buat instance solver
        solver = MonteCarloLinearSolver(A, b)

        key = f"size_{size}_N_{N}"  # Beri nama yang lebih deskriptif
        results[key] = {'processors': [], 'times': []}

        # Baseline sekuensial (1 prosesor)
        print(f"  Menjalankan sekuensial (1 prosesor)...")
        start_time_seq = time.time()
        # Untuk 1 prosesor, kita bisa panggil versi non-paralel agar tidak ada overhead pool
        solver.solve_slae(gamma, epsilon, N, parallel=False)
        time_seq = time.time() - start_time_seq

        results[key]['processors'].append(1)
        results[key]['times'].append(time_seq)
        print(f"    Waktu sekuensial: {time_seq:.4f} detik")

        # Jalankan untuk jumlah prosesor lainnya
        for p_count in processor_counts:
            if p_count == 1:
                continue  # Sudah kita jalankan

            print(f"  Menjalankan dengan {p_count} prosesor...")
            start_time = time.time()
            solver.solve_slae(gamma, epsilon, N, parallel=True, num_processes=p_count)
            exec_time = time.time() - start_time

            results[key]['processors'].append(p_count)
            results[key]['times'].append(exec_time)
            print(f"    Waktu: {exec_time:.4f} detik")

    # --- Buat Plot ---
    plt.figure(figsize=(10, 8))
    for key, data in results.items():
        baseline_time = data['times'][0]
        speedup = [baseline_time / t for t in data['times']]
        plt.plot(data['processors'], speedup, marker='o', linestyle='-', label=key)

    plt.plot(processor_counts, processor_counts, linestyle='--', color='gray', label='Ideal Speedup')

    plt.title("Speedup of Sparse SLAE on Local Machine")
    plt.xlabel("Number of Processors")
    plt.ylabel("Speedup")
    plt.xticks(processor_counts)
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'repro_fig2_left_oop.png')
    plt.savefig(output_path)
    print(f"\nPlot disimpan di: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()