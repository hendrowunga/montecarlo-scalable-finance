import multiprocessing
from tqdm import tqdm


def run_parallel(func, tasks, num_processes, initializer=None, initargs=None):
    """
    Fungsi generik untuk menjalankan tugas secara paralel.
    Mendukung initializer dan chunksize untuk optimasi.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    chunksize = max(1, len(tasks) // (num_processes * 4))

    with multiprocessing.Pool(processes=num_processes,
                              initializer=initializer,
                              initargs=initargs) as pool:
        results = list(tqdm(pool.imap(func, tasks, chunksize=chunksize),
                            total=len(tasks),
                            desc=f"Running on {num_processes} cores"))
    return results