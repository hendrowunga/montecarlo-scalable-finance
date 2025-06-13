import multiprocessing
from tqdm import tqdm # Library bagus untuk progress bar

def run_parallel(func, tasks, num_processes):
    """
    Fungsi generik untuk menjalankan tugas secara paralel.

    Args:
        func (function): Fungsi yang akan dijalankan untuk setiap tugas (misal, _walk).
        tasks (list): Daftar argumen untuk setiap pemanggilan fungsi.
        num_processes (int): Jumlah proses worker yang akan dibuat.

    Returns:
        list: Daftar hasil dari setiap tugas.
    """
    # Menggunakan with statement untuk memastikan pool ditutup dengan benar
    with multiprocessing.Pool(processes=num_processes) as pool:
        # tqdm akan memberikan kita progress bar yang bagus
        results = list(tqdm(pool.imap(func, tasks), total=len(tasks), desc=f"Running on {num_processes} cores"))
    return results