import numpy as np

# Import parallel_runner di level atas. Kita akan atasi potensi circular dependency
# dengan cara memanggil dari skrip, bukan antar modul src.
from ..utils.parallel_runner import run_parallel


# ==============================================================================
# FUNGSI PEMBANTU (WORKER FUNCTIONS) UNTUK PARALELISASI
# Fungsi-fungsi ini harus berada di level atas modul agar bisa di-"pickle" oleh multiprocessing.
# ==============================================================================

def _slae_walk_worker(args):
    """
    Fungsi pembantu untuk menjalankan satu random walk untuk SLAE.
    Dirancang untuk digunakan dengan multiprocessing.pool.
    """
    i_start, H, g, P, epsilon, max_len = args
    n = H.shape[0]

    i_current = i_start
    W = 1.0
    theta = g[i_current]  # Estimator

    for _ in range(max_len):
        # Pilih state selanjutnya
        if np.all(P[i_current, :] == 0):
            break  # Terjebak jika baris probabilitas nol

        i_next = np.random.choice(n, p=P[i_current, :])

        # Hindari pembagian dengan nol jika probabilitas sangat kecil
        if P[i_current, i_next] == 0:
            break

        # Update bobot (weight) dan estimator
        W = W * H[i_current, i_next] / P[i_current, i_next]
        theta += W * g[i_next]

        # Cek kondisi berhenti
        if abs(W) < epsilon:
            break

        i_current = i_next

    return theta


def _mi_walk_worker(args):
    """
    Fungsi pembantu untuk menjalankan satu random walk untuk Inversi Matriks (MI).
    """
    i_start, L, P, m = args
    n = L.shape[0]

    W = 1.0
    current_point = i_start

    # xi adalah vektor yang mengestimasi baris ke-i dari matriks invers
    xi = np.zeros(n)
    xi[current_point] += W

    for _ in range(m):
        if np.all(P[current_point, :] == 0):
            break

        next_point = np.random.choice(n, p=P[current_point, :])

        if P[current_point, next_point] == 0:
            break

        W = W * L[current_point, next_point] / P[current_point, next_point]
        xi[next_point] += W
        current_point = next_point

    return xi


# ==============================================================================
# KELAS UTAMA UNTUK SOLVER ALJABAR LINEAR
# ==============================================================================

class MonteCarloLinearSolver:
    """
    Sebuah kelas untuk menyelesaikan masalah aljabar linear menggunakan metode Monte Carlo.
    Mendukung penyelesaian Sistem Persamaan Linear (SLAE) dan Inversi Matriks (MI).
    """

    def __init__(self, A, b=None):
        """
        Inisialisasi solver dengan matriks A dan (opsional) vektor b.

        Args:
            A (np.ndarray): Matriks n x n.
            b (np.ndarray, optional): Vektor n x 1 untuk masalah Ax = b.
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matriks A harus persegi.")

        self.A = A
        self.b = b
        self.n = A.shape[0]

        # Atribut untuk menyimpan hasil pra-pemrosesan
        self.H = None
        self.g = None
        self.P_slae = None
        self.L = None
        self.P_mi = None

    def _preprocess_slae(self, gamma):
        """Melakukan pra-pemrosesan untuk masalah SLAE (Ax=b)."""
        if self.b is None:
            raise ValueError("Vektor b diperlukan untuk menyelesaikan SLAE.")

        diag_A = np.diag(self.A)
        if np.any(diag_A == 0):
            raise ValueError("Matriks A memiliki nol di diagonal, tidak dapat melanjutkan.")

        D_inv = np.diag(1.0 / diag_A)
        self.H = np.identity(self.n) - gamma * (D_inv @ self.A)
        self.g = gamma * (D_inv @ self.b)

        H_abs = np.abs(self.H)
        row_sums = np.sum(H_abs, axis=1)
        row_sums[row_sums == 0] = 1  # Hindari pembagian dengan nol
        self.P_slae = H_abs / row_sums[:, np.newaxis]
        print("Pra-pemrosesan untuk SLAE selesai.")

    def _preprocess_mi(self):
        """Melakukan pra-pemrosesan untuk masalah Inversi Matriks."""
        self.L = np.identity(self.n) - self.A

        L_abs = np.abs(self.L)
        row_sums = np.sum(L_abs, axis=1)
        row_sums[row_sums == 0] = 1
        self.P_mi = L_abs / row_sums[:, np.newaxis]
        print("Pra-pemrosesan untuk Inversi Matriks selesai.")

    def solve_slae(self, gamma, epsilon, N, max_len=100, parallel=False, num_processes=4):
        """
        Menyelesaikan sistem Ax = b.

        Args:
            gamma (float): Parameter relaksasi.
            epsilon (float): Ambang batas untuk menghentikan walk.
            N (int): Jumlah random walks per komponen.
            max_len (int): Panjang maksimum walk.
            parallel (bool): Jika True, gunakan pemrosesan paralel.
            num_processes (int): Jumlah proses worker untuk paralelisme.

        Returns:
            np.ndarray: Vektor solusi x.
        """
        self._preprocess_slae(gamma)

        x = np.zeros(self.n)

        for i in range(self.n):
            # Buat daftar tugas untuk komponen x_i
            tasks = [(i, self.H, self.g, self.P_slae, epsilon, max_len)] * N

            if parallel:
                # Jalankan secara paralel
                results = run_parallel(_slae_walk_worker, tasks, num_processes)
            else:
                # Jalankan secara sekuensial
                results = [_slae_walk_worker(task) for task in tasks]

            x[i] = np.mean(results)
            print(f"Komponen x[{i}] selesai dihitung.")

        return x

    def invert_matrix(self, N, m, parallel=False, num_processes=4):
        """
        Menghitung invers dari matriks A.

        Args:
            N (int): Jumlah random walks per baris matriks invers.
            m (int): Panjang setiap random walk.
            parallel (bool): Jika True, gunakan pemrosesan paralel.
            num_processes (int): Jumlah proses worker.

        Returns:
            np.ndarray: Matriks invers C = A^-1.
        """
        self._preprocess_mi()

        C = np.zeros((self.n, self.n))

        for i in range(self.n):
            # Buat daftar tugas untuk baris C[i, :]
            tasks = [(i, self.L, self.P_mi, m)] * N

            if parallel:
                results = run_parallel(_mi_walk_worker, tasks, num_processes)
            else:
                results = [_mi_walk_worker(task) for task in tasks]

            # Hasilnya adalah daftar vektor xi. Kita perlu merata-ratakannya.
            C[i, :] = np.mean(np.array(results), axis=0)
            print(f"Baris C[{i}] selesai dihitung.")

        return C