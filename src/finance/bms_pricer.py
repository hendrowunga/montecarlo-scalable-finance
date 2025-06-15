import numpy as np
from ..utils.parallel_runner import run_parallel


def _bms_walk_worker_extended(args):
    """
    Fungsi worker yang diperluas untuk mendukung variabel antitetik.
    """
    S, E, r, sigma, T, use_antithetic = args

    if use_antithetic:
        # Menghasilkan satu bilangan acak dan pasangannya (-Z)
        Z = np.random.normal()
        ST1 = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        ST2 = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * (-Z))
        payoff1 = np.maximum(ST1 - E, 0)
        payoff2 = np.maximum(ST2 - E, 0)
        # Mengembalikan rata-rata dari dua payoff sebagai satu sampel tunggal
        return (payoff1 + payoff2) / 2.0
    else:
        # Perilaku standar seperti sebelumnya
        Z = np.random.normal()
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        payoff = np.maximum(ST - E, 0)
        return payoff


class MonteCarloBSMPricer:
    """
    Kelas ini sekarang mendukung baik simulasi standar maupun dengan variabel antitetik.
    """

    def __init__(self, S, E, r, sigma, T):
        self.S, self.E, self.r, self.sigma, self.T = S, E, r, sigma, T

    def price_option(self, M, parallel=False, num_processes=4, use_antithetic=False, fault_compensation_factor=0.0):
        """
        Menghitung harga opsi.
        Args:
            use_antithetic (bool): Aktifkan untuk menggunakan pengurangan variansi.
        """
        num_simulations = int(M * (1 + fault_compensation_factor))

        # Jika antitetik, setiap walk menghasilkan 2 jalur, jadi kita hanya butuh M/2 walk.
        if use_antithetic:
            # Pastikan jumlah simulasi genap jika dibagi 2
            num_simulations = (num_simulations // 2) * 2
            tasks = [(self.S, self.E, self.r, self.sigma, self.T, True)] * (num_simulations // 2)
        else:
            tasks = [(self.S, self.E, self.r, self.sigma, self.T, False)] * num_simulations

        if parallel and len(tasks) > 1000:
            payoffs = run_parallel(_bms_walk_worker_extended, tasks, num_processes)
        else:
            payoffs = [_bms_walk_worker_extended(task) for task in tasks]

        return np.exp(-self.r * self.T) * np.mean(payoffs)