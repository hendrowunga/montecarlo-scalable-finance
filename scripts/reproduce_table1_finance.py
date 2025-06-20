import numpy as np
import time
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.finance.bms_pricer import MonteCarloBSMPricer


def main():
    # --- Parameter dari Paper ---
    S, r, sigma, T = 100, 0.05, 0.2, 1.0  # Asumsi parameter pasar umum

    exercise_prices = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    # iteration_steps asli dari paper
    # iteration_steps = [10000, 100000, 250000, 500000, 1000000, 10000000, 100000000]

    # Kita tidak akan menjalankan 100 juta di mesin lokal, jadi kita kurangi
    iteration_steps_local = [10000, 100000, 500000, 1000000, 5000000]

    results_data = []
    num_procs = 4  # Gunakan 4 prosesor untuk benchmark ini

    for E in exercise_prices:
        row_E = {'Experiment': f'E{E}'}
        row_EF = {'Experiment': f'EF{E}'}

        # Buat pricer sekali per harga exercise
        pricer = MonteCarloBSMPricer(S, E, r, sigma, T)

        print(f"\n--- Menjalankan untuk Harga Exercise E = {E} ---")
        for M in iteration_steps_local:
            print(f"  Trials M = {M}")

            # Jalankan tanpa kompensasi kesalahan
            start_time = time.time()
            pricer.price_option(M, parallel=True, num_processes=num_procs)
            exec_time = time.time() - start_time
            row_E[str(M)] = exec_time

            # Jalankan DENGAN kompensasi kesalahan (misal, 5% ekstra seperti di paper)
            start_time_ef = time.time()
            pricer.price_option(M, parallel=True, num_processes=num_procs, fault_compensation_factor=0.05)
            exec_time_ef = time.time() - start_time_ef
            row_EF[str(M)] = exec_time_ef

        results_data.append(row_E)
        results_data.append(row_EF)

    # --- Buat dan Simpan Tabel ---
    df = pd.DataFrame(results_data)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables', 'repro_table1.csv')
    df.to_csv(output_path, index=False, float_format='%.5f')

    print("\n" + "=" * 50)
    print("Hasil Tabel 1 (Versi Lokal):")
    print(df)
    print(f"\nTabel disimpan di: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()