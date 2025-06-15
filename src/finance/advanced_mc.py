import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from ..utils.parallel_runner import run_parallel

# --- KELAS UNTUK OPSI ASIA (PATH-DEPENDENT) ---
def _asian_option_walk_worker(args):
    """Worker untuk satu jalur simulasi Opsi Asia."""
    S, E, r, sigma, T, num_steps = args
    dt = T / num_steps
    path = np.zeros(num_steps + 1)
    path[0] = S
    for i in range(1, num_steps + 1):
        Z = np.random.normal()
        path[i] = path[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    average_price = np.mean(path[1:])
    payoff = np.maximum(average_price - E, 0)
    return payoff

class AsianOptionPricer:
    """Menangani masalah keuangan yang lebih kompleks (path-dependent)."""
    def __init__(self, S, E, r, sigma, T, num_steps):
        self.S, self.E, self.r, self.sigma, self.T = S, E, r, sigma, T
        self.num_steps = num_steps

    def price(self, M, parallel=True, num_processes=4):
        tasks = [(self.S, self.E, self.r, self.sigma, self.T, self.num_steps)] * M
        if parallel:
            payoffs = run_parallel(_asian_option_walk_worker, tasks, num_processes)
        else:
            payoffs = [_asian_option_walk_worker(task) for task in tasks]
        return np.exp(-self.r * self.T) * np.mean(payoffs)

# --- KELAS UNTUK QUASI-MONTE CARLO (QMC) ---
class QMCEuropeanPricer:
    """Membandingkan dengan metode yang lebih canggih (QMC)."""
    def __init__(self, S, E, r, sigma, T):
        self.S, self.E, self.r, self.sigma, self.T = S, E, r, sigma, T

    def price(self, M):
        """Harga menggunakan sekuens Sobol. Tidak perlu paralelisasi karena sudah vektorial."""
        M_power_of_2 = int(2**np.ceil(np.log2(M)))
        sobol_engine = Sobol(d=1, scramble=True)
        uniform_samples = sobol_engine.random(n=M_power_of_2).flatten()
        normal_samples = norm.ppf(uniform_samples)
        ST = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * normal_samples)
        payoffs = np.maximum(ST - self.E, 0)
        return np.exp(-self.r * self.T) * np.mean(payoffs)