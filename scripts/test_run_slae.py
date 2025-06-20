import numpy as np
import sys
import os

# Menambahkan direktori src ke path agar kita bisa mengimpor modul kita
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 1. Impor KELASNYA, bukan fungsinya
from src.linear_algebra.mc_solvers import MonteCarloLinearSolver

def main():
    print("Menjalankan tes sederhana untuk kelas MonteCarloLinearSolver...")

    # Buat masalah yang bisa diselesaikan
    n = 5
    A = np.random.rand(n, n) + np.diag([n] * n)  # Diagonally dominant
    x_true = np.ones(n)
    b = A @ x_true

    # Parameter untuk solver MC
    gamma = 0.5
    epsilon = 1e-4
    N = 10000

    print("Matriks A:\n", A)
    print("Vektor b:\n", b)
    print("Solusi sebenarnya x_true:\n", x_true)

    # 2. Buat sebuah INSTANCE dari kelas solver
    # Masukkan matriks A dan vektor b saat inisialisasi
    solver = MonteCarloLinearSolver(A, b)

    # 3. Panggil METODE solve_slae dari objek solver
    # Kita set parallel=False untuk pengujian sekuensial sederhana ini
    x_mc = solver.solve_slae(gamma=gamma, epsilon=epsilon, N=N, parallel=False)

    print("\nSolusi dari Monte Carlo x_mc:\n", x_mc)

    # Hitung error
    error = np.linalg.norm(x_true - x_mc) / np.linalg.norm(x_true)
    print(f"\nRelative error: {error:.4f}")


if __name__ == "__main__":
    main()