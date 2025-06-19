import numpy as np
from ..utils.parallel_runner import run_parallel


# ==============================================================================
# FUNGSI INITIALIZER DAN WORKER UNTUK PARALELISASI
# ==============================================================================

def _init_worker_globals(H_global, g_global, P_slae_global, L_global, P_mi_global):
    global H, g, P_slae, L, P_mi
    H, g, P_slae, L, P_mi = H_global, g_global, P_slae_global, L_global, P_mi_global

def _slae_walk_worker(args):
    i_start, epsilon, max_len = args
    n = H.shape[0]
    i_current = i_start
    W = 1.0
    theta = g[i_current]
    for _ in range(max_len):
        if np.all(P_slae[i_current, :] == 0): break
        i_next = np.random.choice(n, p=P_slae[i_current, :])
        if P_slae[i_current, i_next] == 0: break
        W *= H[i_current, i_next] / P_slae[i_current, i_next]
        theta += W * g[i_next]
        if abs(W) < epsilon: break
        i_current = i_next
    return theta

def _mi_walk_worker(args):
    i_start, m = args
    n = L.shape[0]
    current_point = i_start
    W = 1.0
    xi = np.zeros(n)
    xi[current_point] += W
    for _ in range(m):
        if np.all(P_mi[current_point, :] == 0): break
        next_point = np.random.choice(n, p=P_mi[current_point, :])
        if P_mi[current_point, next_point] == 0: break
        W *= L[current_point, next_point] / P_mi[current_point, next_point]
        xi[next_point] += W
        current_point = next_point
    return xi

# ==============================================================================
# KELAS UTAMA UNTUK SOLVER ALJABAR LINEAR
# ==============================================================================

class MonteCarloLinearSolver:
    def __init__(self, A, b=None):
        if A.shape[0] != A.shape[1]: raise ValueError("Matriks A harus persegi.")
        self.A, self.b, self.n = A, b, A.shape[0]
        self.H, self.g, self.P_slae, self.L, self.P_mi = [None] * 5

    def _preprocess_slae(self, gamma):
        if self.b is None: raise ValueError("Vektor b diperlukan.")
        diag_A = np.diag(self.A)
        if np.any(diag_A == 0): raise ValueError("Nol di diagonal.")
        D_inv = np.diag(1.0 / diag_A)
        self.H = np.identity(self.n) - gamma * (D_inv @ self.A)
        self.g = gamma * (D_inv @ self.b)
        H_abs = np.abs(self.H)
        row_sums = np.sum(H_abs, axis=1)
        row_sums[row_sums == 0] = 1
        self.P_slae = H_abs / row_sums[:, np.newaxis]

    def _preprocess_mi(self):
        self.L = np.identity(self.n) - self.A
        L_abs = np.abs(self.L)
        row_sums = np.sum(L_abs, axis=1)
        row_sums[row_sums == 0] = 1
        self.P_mi = L_abs / row_sums[:, np.newaxis]

    def solve_slae(self, gamma, epsilon, N, max_len=100, parallel=False, num_processes=4):
        # Panggil pra-pemrosesan dengan argumen yang benar
        self._preprocess_slae(gamma)

        x = np.zeros(self.n)
        for i in range(self.n):
            tasks = [(i, epsilon, max_len)] * N
            if parallel:
                init_args = (self.H, self.g, self.P_slae, None, None)
                results = run_parallel(_slae_walk_worker, tasks, num_processes, initializer=_init_worker_globals, initargs=init_args)
            else:
                _init_worker_globals(self.H, self.g, self.P_slae, None, None)
                results = [_slae_walk_worker(task) for task in tasks]
            x[i] = np.mean(results)
            # Menggunakan \r untuk menimpa baris yang sama, membuat output lebih rapi
            print(f"  Menghitung... Komponen x[{i+1}/{self.n}] selesai.", end='\r')
        print(f"\n  Selesai untuk {num_processes if parallel else 1} prosesor.")
        return x

    def invert_matrix(self, N, m, parallel=False, num_processes=4):
        self._preprocess_mi()
        C = np.zeros((self.n, self.n))
        for i in range(self.n):
            tasks = [(i, m)] * N
            if parallel:
                init_args = (None, None, None, self.L, self.P_mi)
                results = run_parallel(_mi_walk_worker, tasks, num_processes, initializer=_init_worker_globals, initargs=init_args)
            else:
                _init_worker_globals(None, None, None, self.L, self.P_mi)
                results = [_mi_walk_worker(task) for task in tasks]
            C[i, :] = np.mean(np.array(results), axis=0)
            print(f"  Menghitung... Baris C[{i+1}/{self.n}] selesai.", end='\r')
        print(f"\n  Selesai untuk {num_processes if parallel else 1} prosesor.")
        return C