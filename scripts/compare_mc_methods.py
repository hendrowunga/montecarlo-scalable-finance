import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.finance.bms_pricer import MonteCarloBSMPricer
from src.finance.advanced_mc import AsianOptionPricer, QMCEuropeanPricer


def black_scholes_analytic(S, E, r, sigma, T):
    d1 = (np.log(S / E) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - E * np.exp(-r * T) * norm.cdf(d2)


def main():
    S, E, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    num_steps_asian = 50
    sample_sizes = np.logspace(8, 18, num=11, base=2).astype(int)
    true_price = black_scholes_analytic(S, E, r, sigma, T)

    bms_pricer = MonteCarloBSMPricer(S, E, r, sigma, T)
    qmc_pricer = QMCEuropeanPricer(S, E, r, sigma, T)
    asian_pricer = AsianOptionPricer(S, E, r, sigma, T, num_steps_asian)

    errors_mc, errors_qmc = [], []

    print("Menghitung error konvergensi untuk MC vs QMC...")
    for M in sample_sizes:
        print(f"  Jumlah Sampel: {M}")
        price_mc = bms_pricer.price_option(M, parallel=True)
        price_qmc = qmc_pricer.price(M)
        errors_mc.append(abs(price_mc - true_price))
        errors_qmc.append(abs(price_qmc - true_price))

    print(f"\nEstimasi Harga Opsi Asia (M={sample_sizes[-1]}):")
    price_asian = asian_pricer.price(M=sample_sizes[-1], parallel=True)
    print(f"  Harga = {price_asian:.5f}")

    plt.figure(figsize=(12, 7))
    plt.loglog(sample_sizes, errors_mc, 'o-', label='Standard Monte Carlo Error')
    plt.loglog(sample_sizes, errors_qmc, 's-', label='Quasi-Monte Carlo (Sobol) Error')
    plt.loglog(sample_sizes, 1 / np.sqrt(sample_sizes), 'k--', alpha=0.5, label='Referensi O(N⁻⁰·⁵)')
    plt.loglog(sample_sizes, 1 / sample_sizes, 'k:', alpha=0.5, label='Referensi O(N⁻¹)')

    plt.title('Laju Konvergensi: Standard MC vs. Quasi-MC')
    plt.xlabel('Jumlah Sampel (N)')
    plt.ylabel('Error Absolut (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'convergence_mc_vs_qmc.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot konvergensi disimpan di: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()