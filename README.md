# Efisiensi Monte Carlo dalam Keuangan Komputasional: Sebuah Studi Komparatif

Proyek ini merupakan implementasi, analisis kritis, dan ekstensi dari makalah ilmiah **"Monte Carlo scalable algorithms for Computational Finance"** oleh V. N. Alexandrov, C. González Martel, dan J. Straßburg. Proyek ini bertujuan untuk mereplikasi hasil yang disajikan, mengidentifikasi keterbatasan metodologisnya, dan memperbaikinya dengan mengimplementasikan teknik-teknik standar industri yang lebih canggih.

## Paper Acuan

Proyek ini didasarkan pada metodologi dan hasil yang disajikan dalam paper berikut. Disarankan untuk membaca paper ini untuk memahami konteks awal dari penelitian ini.

|                                                                  Tampilan Paper                                                                  | Informasi |
|:------------------------------------------------------------------------------------------------------------------------------------------------:| :--- |
| [<img src="/results/tables/paper.png" alt="Paper Screenshot" width="250"/>](https://www.sciencedirect.com/science/article/pii/S187705091100346X) | **Judul:** Monte Carlo scalable algorithms for Computational Finance<br>**Penulis:** V. N. Alexandrov, C. González Martel, J. Straßburg<br>**Publikasi:** Procedia Computer Science, Vol. 4, 2011<br>**DOI:** [10.1016/j.procs.2011.04.185](https://doi.org/10.1016/j.procs.2011.04.185)<br>**Link Akses:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S187705091100346X) |

## Struktur Proyek

Repositori ini disusun secara logis untuk memisahkan kode sumber, skrip eksekusi, hasil, dan ekstensi. Struktur ini mencerminkan evolusi proyek dari replikasi hingga inovasi.

```
montecarlo-scalable-finance/
├── results/                    # Output dari skrip (gambar dan tabel)
│   ├── figures/
│   │   ├── convergence_mc_vs_qmc.png
│   │   ├── repro_fig2_dense_multi_size.png
│   │   ├── repro_fig2_multi_scenario.png
│   │   ├── repro_mi_dense.png
│   │   └── variance_reduction_comparison.png
│   └── tables/
│       └── repro_table1.csv
├── scripts/                    # Skrip yang dapat dieksekusi untuk benchmark & analisis
│   ├── compare_mc_methods.py
│   ├── compare_variance_reduction.py
│   ├── reproduce_fig2_mi_dense.py
│   ├── reproduce_fig2_slae_dense.py
│   ├── reproduce_fig2_slae_sparse.py
│   ├── reproduce_table1_finance.py
│   └── test_run_slae.py
├── src/                        
│   ├── finance/
│   │   ├── advanced_mc.py      # Implementasi Opsi Asia & QMC
│   │   └── bms_pricer.py       # Implementasi Opsi Eropa & Var. Antitetik
│   ├── linear_algebra/
│   │   └── mc_solvers.py       # Implementasi solver MC untuk aljabar linear
│   └── utils/
│       └── parallel_runner.py  # Utilitas untuk eksekusi paralel
├── .gitignore
├── README.md                   
└── requirements.txt                    
```

### Penjelasan Struktur

-   **`src/`**: Berisi semua logika inti proyek.
    -   `linear_algebra/`: Modul untuk mereplikasi bagian aljabar linear dari paper.
    -   `finance/`: Modul untuk bagian keuangan, termasuk implementasi dasar (`bms_pricer.py`) dan ekstensi canggih (`advanced_mc.py`).
    -   `utils/`: Utilitas pendukung, terutama untuk menangani komputasi paralel secara efisien.
-   **`scripts/`**: Berisi semua skrip untuk menjalankan eksperimen. Nama file dengan prefix `reproduce_` bertujuan untuk mereplikasi hasil dari paper, sedangkan `compare_` digunakan untuk menganalisis ekstensi dan inovasi yang kami buat.
-   **`results/`**: Direktori ini menyimpan semua artefak yang dihasilkan oleh skrip, seperti gambar plot dan file data CSV.

## Instalasi dan Pengaturan

Untuk menjalankan kode dalam repositori ini, ikuti langkah-langkah berikut.

1.  **Clone Repositori**
    ```bash
    git clone https://github.com/hendrowunga/montecarlo-scalable-finance.git
    cd montecarlo-scalable-finance
    ```

2.  **Buat dan Aktifkan Lingkungan Virtual**
    Sangat disarankan untuk menggunakan lingkungan virtual untuk mengelola dependensi.
    ```bash
    # Membuat lingkungan virtual
    python3 -m venv .venv

    # Mengaktifkan lingkungan (Linux/macOS)
    source .venv/bin/activate

    # Mengaktifkan lingkungan (Windows)
    .\.venv\Scripts\activate
    ```

3.  **Instal Dependensi**
    Instal semua paket yang diperlukan dari file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Cara Menjalankan Eksperimen

Semua eksperimen dapat dijalankan melalui skrip yang ada di dalam direktori `scripts/`.

#### A. Mereplikasi Hasil Paper

Untuk mereplikasi hasil benchmark skalabilitas dan tabel waktu dari paper acuan:
```bash
# Menghasilkan plot speedup untuk matriks sparse
python scripts/reproduce_fig2_slae_sparse.py

# Menghasilkan plot speedup untuk matriks dense
python scripts/reproduce_fig2_slae_dense.py

# Menghasilkan tabel waktu untuk pricer keuangan
python scripts/reproduce_table1_finance.py
```

#### B. Menganalisis Ekstensi dan Inovasi

Untuk menjalankan analisis komparatif dari perbaikan yang telah kami implementasikan:
```bash
# Membandingkan MC standar vs. MC dengan Variabel Antitetik
python scripts/compare_variance_reduction.py

# Membandingkan laju konvergensi MC standar vs. QMC
python scripts/compare_mc_methods.py
```

## Rencana dan Status Proyek

Proyek ini mengikuti alur kerja yang terstruktur dari replikasi hingga inovasi.

1.  [x] **Implementasi Inti & Replikasi:** Mengimplementasikan dan memvalidasi semua algoritma dari paper (Aljabar Linear dan Keuangan). Semua skrip `reproduce_` telah diselesaikan.
2.  [x] **Identifikasi Keterbatasan:** Menganalisis kekurangan paper, terutama pada pengabaian teknik efisiensi statistik.
3.  [x] **Implementasi Ekstensi & Inovasi:**
    -   [x] Menerapkan teknik pengurangan variansi (Variabel Antitetik).
    -   [x] Mengimplementasikan metode Quasi-Monte Carlo (QMC).
    -   [x] Menambahkan dukungan untuk masalah yang lebih kompleks (Opsi Asia).
4.  [x] **Analisis Komparatif:** Melakukan benchmark dan menghasilkan visualisasi untuk membuktikan keunggulan dari metode-metode yang diperluas.

**Status Proyek: Selesai.** Semua tujuan awal telah tercapai.
