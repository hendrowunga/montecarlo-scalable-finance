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
    # --- Parameter Eksperimen ---
    size = 50  # Inversi matriks sangat berat, mulai dari yang sangat kecil
    processor_counts = [1, 2, 3, 4]

    # Parameter untuk MI
    N = 2000  # Jumlah walk per baris
    m = 20  # Panjang setiap walk

    results = {}

    print(f"\n--- Menguji Inversi Matriks DENSE: {size}x{size} (Eksekusi Penuh) ---")
    # Kita butuh matriks A di mana I-A konvergen, A = I - B dengan ||B|| < 1
    # Kita bisa gunakan matriks diagonally dominant dan skalakan
    A_temp = create_dense_diagonally_dominant(size)
    # Pastikan norma matriks L = I - A kurang dari 1
    L_temp = np.identity(size) - A_temp
    A = A_temp / (np.max(np.sum(np.abs(L_temp), axis=1)) + 1)

    solver = MonteCarloLinearSolver(A)

    key = f"mi_dense_size_{size}"
    results[key] = {'processors': [], 'times': []}

    for p_count in processor_counts:
        is_parallel = p_count > 1
        print(f"  Menjalankan dengan {p_count} prosesor...")

        start_time = time.time()
        # Panggil metode invert_matrix TANPA benchmark_rows
        solver.invert_matrix(N, m, parallel=is_parallel, num_processes=p_count)
        exec_time = time.time() - start_time

        results[key]['processors'].append(p_count)
        results[key]['times'].append(exec_time)
        print(f"    Waktu: {exec_time:.4f} detik")

    # --- Buat Plot ---
    plt.figure(figsize=(10, 8))
    for key, data in results.items():
        baseline_time = data['times'][0]
        if baseline_time > 0:
            speedup = [baseline_time / t for t in data['times']]
            plt.plot(data['processors'], speedup, marker='o', linestyle='-', label=key)

    plt.plot(processor_counts, processor_counts, linestyle='--', color='gray', label='Ideal Speedup')
    plt.title("Speedup of Dense Matrix Inversion")
    plt.xlabel("Number of Processors")
    plt.ylabel("Speedup")
    plt.xticks(processor_counts)
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'repro_mi_dense.png')
    plt.savefig(output_path)
    print(f"\nPlot disimpan di: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()