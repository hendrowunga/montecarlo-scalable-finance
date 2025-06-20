import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.finance.bms_pricer import MonteCarloBSMPricer


def main():
    S, E, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    num_simulations = 100000
    num_runs = 50

    print("Membandingkan Standard MC vs. Antithetic Variates...")
    pricer = MonteCarloBSMPricer(S, E, r, sigma, T)

    standard_mc_results = []
    antithetic_mc_results = []

    for i in range(num_runs):
        print(f"  Menjalankan eksperimen ke-{i + 1}/{num_runs}...")
        price_std = pricer.price_option(M=num_simulations, parallel=True, use_antithetic=False)
        standard_mc_results.append(price_std)

        price_anti = pricer.price_option(M=num_simulations, parallel=True, use_antithetic=True)
        antithetic_mc_results.append(price_anti)

    std_dev_standard = np.std(standard_mc_results)
    std_dev_antithetic = np.std(antithetic_mc_results)
    variance_reduction = (std_dev_standard ** 2 - std_dev_antithetic ** 2) / std_dev_standard ** 2

    print("\n" + "=" * 50)
    print("Hasil Akhir:")
    print(f"Std Dev (Standard MC):   {std_dev_standard:.5f}")
    print(f"Std Dev (Antithetic MC): {std_dev_antithetic:.5f}")
    print(f"Pengurangan Variansi:    {variance_reduction:.2%}")
    print("=" * 50)

    plt.figure(figsize=(12, 6))
    plt.hist(standard_mc_results, bins=15, alpha=0.7, label=f'Standard MC (Std Dev: {std_dev_standard:.4f})')
    plt.hist(antithetic_mc_results, bins=15, alpha=0.7, label=f'Antithetic MC (Std Dev: {std_dev_antithetic:.4f})')
    plt.title("Distribusi Harga Opsi dari Simulasi Berulang")
    plt.xlabel("Harga Opsi Terhitung")
    plt.ylabel("Frekuensi")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures',
                               'variance_reduction_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot perbandingan disimpan di: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()