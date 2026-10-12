# Deteksi URL Phishing Real-Time Berbasis Client-Side Browser Extension
### Komparasi Kinerja Algoritma Random Forest, XGBoost, dan CatBoost via ONNX Runtime Web

> **Repositori Tugas Akhir**  
> **Program Studi:** Informatika  
> **NIM:** 2211102226  

---

## 📌 Ringkasan 

Pertahanan peramban konvensional (seperti *Google Safe Browsing* dan basis data tanda tangan antivirus) mengandalkan daftar hitam (*blacklist*) terpusat. Pendekatan ini memiliki *detection lag* rata-rata 4,5 hari, padahal situs phishing modern memiliki median masa hidup hanya sekitar 5,46 jam. Pendekatan berbasis *server-side* atau API eksternal juga menimbulkan risiko kebocoran privasi riwayat penjelajahan dan latensi jaringan.

Penelitian ini merancang dan mengimplementasikan **ekstensi Google Chrome mandiri (*standalone*)** yang melakukan inferensi Machine Learning secara **100% di sisi klien (*client-side*)** menggunakan **ONNX Runtime Web (WASM)**. Sistem ini membandingkan kinerja deteksi dan efisiensi komputasi antara tiga algoritma utama:
1. **Random Forest (RF):** Model *bagging* acuan yang dioptimalkan (*bonsai trees*) untuk ukuran berkas ringan dan *False Positive Rate* (FPR) mendekati 0%.
2. **XGBoost (XGB):** Model *gradient boosting* yang dipadukan dengan *pipeline* reduksi dimensi berganda (*StandardScaler* → *SelectKBest* ANOVA F → *RFECV* → *PCA 95%*) terinspirasi dari arsitektur *PhishGuard*.
3. **CatBoost (CB):** Model *gradient boosting* yang memakai fitur mentah yang sama dengan Random Forest (TF-IDF kata + fitur struktural); ditambahkan sebagai model ketiga pada ekstensi versi 3.4.

Ketiga model dikonversi ke format ONNX dan dapat diuji secara langsung serta bergantian melalui antarmuka ekstensi peramban.

---

## Arsitektur Sistem & Alur Deteksi

Deteksi berjalan secara otomatis pada saat pengguna melakukan navigasi URL di peramban Chrome:

```
                      [ Pengguna Membuka URL ]
                                  │
                                  ▼
               [ Event: webNavigation.onBeforeNavigate ]
                                  │
                                  ▼
                     Apakah URL ada di Whitelist?
                     ├─── Ya  ───▶ [ Bypass / Izinkan ]
                     └─── Tidak ──▶ [ Ekstraksi 1.509 Fitur di JS ]
                                           │
                                           ▼
                                 [ ONNX Inference ]
                         (phishing_rf.onnx / phishing_xgb.onnx)
                                           │
                                           ▼
            ┌──────────────────────────────┴──────────────────────────────┐
            ▼                                                             ▼
   Probabilitas Phishing > 80%                               Probabilitas Phishing 60% – 80%
      (Tingkat Bahaya Tinggi)                                       (Mencurigakan)
            │                                                             │
            ▼                                                             ▼
  [ Blokir Pra-Eksekusi ]                                     [ Inject Warning Banner ]
  Redirect ke `blocked.html`                                    via `content.js` pada DOM
  (sebelum script web berjalan)                               (dengan opsi Trust / Dismiss)
```

---

## 📁 Peta Struktur Repositori

Semua skrip dijalankan **dari root repositori** (`python nama_skrip.py`). Tiga modul inti (`features.py`, `sampling.py`, `ml_pipelines.py`) sengaja berada di root: berkas model `.pkl` menyimpan nama modul tersebut, sehingga memindahkannya akan membuat model lama tidak dapat dimuat.

```
Tugas Akhir/
│
├── phishingdetectorExt/           # PRODUK: paket Chrome Extension (Manifest V3)
│   ├── manifest.json              # Konfigurasi ekstensi & deklarasi permissions
│   ├── background.js              # Service worker: pemindaian navigasi, sesi ONNX, message API
│   ├── popup.html / popup.js      # Popup: pemindaian manual, pilihan model, kelola whitelist
│   ├── content.js                 # Banner peringatan pada situs mencurigakan
│   ├── blocked.html / blocked.js  # Halaman blokir (session bypass & whitelist)
│   ├── utils.js                   # Ekstraksi fitur leksikal, harus identik dengan features.py
│   ├── bench.html / bench.js      # Benchmark latensi di dalam browser (bench_urls.json)
│   ├── phishing_rf.onnx           # Random Forest (tidak dilacak git; lihat MODELS.md)
│   ├── phishing_xgb.onnx          # XGBoost
│   ├── phishing_catboost.onnx     # CatBoost
│   ├── tfidf_data.json            # Kosakata TF-IDF untuk RF dan CatBoost
│   ├── tfidf_data_xgb.json        # Kosakata TF-IDF untuk XGBoost
│   ├── xgb_preprocessing.json     # StandardScaler, indeks seleksi fitur, dan matriks PCA XGBoost
│   └── libs/                      # ONNX Runtime Web (WASM & WASM-SIMD)
│
├── ── MODUL INTI (root) ──
├── features.py                    # Satu-satunya sumber kebenaran fitur di sisi Python
├── sampling.py                    # Protokol data: bersihkan → split 80/20 (seed 42) → validasi 10%
├── ml_pipelines.py                # Pipeline RF dan XGBoost dengan hyperparameter tetap
│
├── ── PELATIHAN & EKSPOR ──
├── hybrid/train_hybrid_model_v2.py  # Pelatihan Random Forest (--no-undersample untuk varian full)
├── hybrid/test_hybrid.py          # Uji interaktif sebuah model RF (.pkl)
├── Xgboost/                       # Pelatihan/ekspor XGBoost versi Colab (+ artefak Colab lama)
├── Catboost/                      # Pelatihan CatBoost, perbaikan keluaran ONNX, verifikasi ONNX
├── export_to_web.py               # RF (.pkl) → ONNX + tfidf_data.json
├── export_xgb_onnx.py, xgb_export.py  # XGBoost → ONNX + berkas preprocessing JSON
│
├── ── BENCHMARK ──
├── benchmark_sampling.py          # Undersampling vs full (melatih RF/XGB; ± 8 menit)
├── benchmark_all_models.py        # PKL vs ONNX pada test set bersama
├── benchmark_deployed_latency.py  # Latensi model yang di-deploy (Python)
├── benchmark_memory.py            # Memori dan latensi
├── benchmark_browser.py           # Membandingkan hasil bench.html dengan Python
├── benchmark_ablation.py          # Ablasi fitur: skema, www., kelompok fitur, n-gram karakter
│
├── ── ANALISIS ──
├── analyze_dataset_split.py       # Overlap host latih/uji dan statistik awalan URL
├── analyze_feature_importance.py  # Bobot fitur RF dan XGBoost
├── analyze_thresholds.py          # Sensitivitas ambang keputusan (60% / 80%)
├── analyze_overfitting.py         # Kinerja data latih vs data uji
├── run_realworld_test.py          # Validasi awal 25/25 URL (PhishTank)
├── explore_char_external.py       # Penilaian eksploratif n-gram karakter pada daftar eksternal
├── make_thesis_figures.py         # Grafik untuk naskah (docs/figures/)
│
├── ── HASIL (bukti; angka di naskah diambil dari sini) ──
├── sampling_results/              # Studi sampling, ambang, bobot fitur, split data, overfitting, CatBoost
├── ablation_results/              # Ablasi skema/www./kelompok fitur (10 Okt 2026)
├── ablation_char_results/         # Ablasi n-gram karakter + penilaian eksternal eksploratif
├── external_500/                  # Uji eksternal Tranco + PhishTank: protokol, daftar beku, laporan, audit
├── realworld_test/                # Validasi awal pada 25/25 URL
├── browser_bench/                 # Pengujian di browser: panduan, hasil, tangkapan layar
├── rf_models/                     # Model RF biner (tidak dilacak git)
├── docs/                          # Laporan PKL vs ONNX, catatan keterbatasan ekstensi, figures/
│
├── tests/                         # Uji paritas fitur (pytest) dan navigasi ekstensi (node --test)
├── legacy/                        # Skrip generasi awal, disimpan sebagai arsip (lihat legacy/README.md)
├── MODELS.md                      # Catatan resmi model yang di-deploy: ukuran dan SHA-256
├── URL dataset.csv                # Dataset (Mendeley 2024, 450.176 baris setelah dibersihkan)
└── README.md                      # Dokumen ini
```

Berkas `*.pkl` hasil pelatihan dan `phishing_rf*.onnx` tidak dilacak git (lihat `.gitignore`); klon baru perlu melatih ulang sebelum sebagian besar benchmark dapat dijalankan.

---

## Metodologi Rekayasa Fitur & Model

### 1. Dataset
Menggunakan dataset gabungan dari **Mendeley Phishing URL Dataset (2024)** berjumlah 450.176 baris setelah pembersihan data (penghapusan baris kosong dan duplikasi).
- **Pembagian Data:** *Stratified Hold-out* 80% data latih (360.140 URL) dan 20% data uji bersama (90.036 URL: 69.148 *legitimate*, 20.888 *phishing*).

### 2. Ekstraksi Fitur Leksikal (1.509 Fitur)
Tidak seperti metode berbasis konten yang mengunduh seluruh halaman HTML, sistem ini hanya mengekstraksi representasi dari *string* URL sehingga sangat cepat dan menjaga privasi:
- **1.500 Fitur TF-IDF:** Tokenisasi URL hierarkis berbasis delimiter garis miring (`/`), tanda hubung (`-`), dan titik (`.`). Menghapus kata umum seperti `com` dan `www`.
- **9 Fitur Struktural / Leksikal URL:**
  | No | Nama Fitur | Definisi Teknis | Rasional Keamanan |
  |:--:|:---|:---|:---|
  | 1 | `url_length` | Total panjang karakter URL | URL phishing sering kali panjang untuk menyembunyikan nama domain asli. |
  | 2 | `dot_count` | Jumlah kemunculan titik (`.`) | Menandakan kedalaman subdomain (*subdomain nesting*). |
  | 3 | `slash_count` | Jumlah garis miring (`/`) | Mengukur kedalaman direktori jalur web. |
  | 4 | `dash_count` | Jumlah tanda hubung (`-`) | Umum digunakan pada teknik *typosquatting* (misal: `paypal-secure`). |
  | 5 | `at_count` | Jumlah simbol `@` | Simbol `@` mengabaikan teks sebelumnya dalam otoritas URL. |
  | 6 | `digit_ratio` | Rasio jumlah angka terhadap panjang URL | URL berbahaya otomatis sering mengandung deretan angka acak. |
  | 7 | `entropy` | *Shannon Entropy* dalam satuan bits | Mengukur keacakan karakter (*domain generation algorithms*). |
  | 8 | `is_common_tld` | Variabel biner TLD populer (`.com`, `.org`, `.id`, dll.) | Domain tidak lazim sering digunakan karena murah atau gratis. |
  | 9 | `subdomain_level`| Kedalaman level subdomain dari nama *host* | Indikasi peniruan domain bertingkat. |

*Catatan Paritas:* Logika ekstraksi fitur di `phishingdetectorExt/utils.js` telah diselaraskan 100% dengan `features.py` pada lingkungan Python (termasuk normalisasi URL berakhiran *trailing slash*).

### 3. Spesifikasi Model yang Di-deploy pada Ekstensi

| Parameter / Aspek | Random Forest (Acuan) | XGBoost (Komparasi) |
|---|---|---|
| **Varian Data Latih** | Full Dataset (360.140 URL) | Full Dataset (360.140 URL) |
| **Dimensi Fitur Masukan** | 1.509 fitur langsung | 1.509 → 177 (*importance pruning*) → 149 (*PCA*) |
| **Konfigurasi Utama** | 100 pohon, `max_depth=15`, `min_samples_split=10`, `min_samples_leaf=4` | 700 pohon, `max_depth=5`, `learning_rate=0.1`, `reg_alpha=0.1`, `reg_lambda=2` |
| **Ukuran Berkas Model** | `phishing_rf.onnx` (**730 KB**) | `phishing_xgb.onnx` (**1.291 KB**) + Preprocessing (**655 KB**) |
| **Karakteristik Deteksi** | Sangat konservatif, nyaris tanpa *false positive* (FPR = 0,001%). | Sangat sensitif/agresif, menangkap lebih banyak variasi phishing. |

> Catatan: ekstensi **v3.4** menambahkan model **CatBoost** (fitur mentah yang sama dengan RF); tabel di atas membandingkan pasangan RF–XGBoost. Status artefak terkini ada di `MODELS.md`.

---

## Hasil Pengujian & Benchmark

Seluruh pengujian dievaluasi pada data uji bersama (*unseen test set*) sebesar **90.036 URL**:

### 1. Evaluasi Kinerja Klasifikasi (Studi Sampling)

| Model | Data Latih | Akurasi | Presisi (Phishing) | Recall (Phishing) | F1-Score | False Positive Rate (FPR) | Jumlah False Positive |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RF (Undersampled)** | 167.100 | 99,24% | 99,70% | 97,02% | 98,34% | 0,088% | 61 |
| **RF (Full Dataset) ⭐** | 360.140 | 99,04% | **100,00%** | 95,87% | 97,89% | **0,001%** | **1** |
| **XGBoost (Undersampled) ⭐**| 167.100 | 99,46% | 98,83% | **98,87%** | **98,85%** | 0,354% | 245 |
| **XGBoost (Full Dataset)** | 360.140 | **99,55%** | 99,57% | 98,48% | 99,03% | 0,127% | 88 |

*Keterangan: Tanda ⭐ menandakan model yang diintegrasikan ke dalam paket Chrome Extension*

### 2. Evaluasi Format PKL vs ONNX (Efisiensi Komputasi)

| Model & Format | Akurasi / F1 | Latensi Rata-rata per URL | Penggunaan RAM | Ukuran Berkas |
|---|:---:|:---:|:---:|:---:|
| **Random Forest (.pkl)** | 99,04% / 97,89% | 28,5 ms | ~255,7 MB | 7.529 KB |
| **Random Forest (.onnx)** | 99,00% / 97,80% | **5,9 ms** | **163,1 MB** | **730 KB** |
| **XGBoost (.pkl)** | 99,46% / 98,85% | 5,3 ms | ~213,7 MB | 4.672 KB |
| **XGBoost (.onnx)** | 99,43% / 98,76% | **9,3 ms** | **169,6 MB** | **1.907 KB** |

*Catatan:* Konversi ke ONNX berhasil mempertahankan metrik statistik (>99% identik) sekaligus memangkas ukuran berkas hingga **90%** dan menurunkan latensi inferensi secara drastis.

### 3. Validasi pada Data Nyata (*PhishTank Live Feed*)

Pengujian dilakukan terhadap 25 URL phishing aktif dari feed PhishTank (dikonfirmasi daring) dan 25 URL *legitimate* populer yang rentan *false positive*:
- **Random Forest:** Recall 0/25 (0%), False Positive 0/25 (0%). Bersifat sangat defensif karena dataset 2024 mengasosiasikan HTTPS dengan URL aman, sedangkan phishing modern saat ini hampir seluruhnya telah menggunakan HTTPS.
- **XGBoost:** Recall 7/25 (28%), False Positive 4/25 (16%). Mampu mendeteksi sebagian phishing aktif pada infrastruktur *free-hosting* (*pages.dev*, *web.app*, dsb.), namun menghasilkan false positive pada layanan tertentu (misal: Steam).
- **CatBoost:** Recall 6/25 (24%), False Positive 5/25 (20%). Memakai fitur mentah yang sama dengan Random Forest; pola deteksinya menyerupai XGBoost.

Temuan ini menegaskan keunggulan menyediakan **tiga model yang dapat dipilih pengguna**: Random Forest untuk pengguna yang mengutamakan tanpa gangguan (*zero false alarms*), serta XGBoost dan CatBoost untuk pengguna yang menginginkan proteksi lebih proaktif. Uji eksternal 500/500 pada daftar beku (`external_500/`) menunjukkan pola yang sama pada skala lebih besar: ketiga model masih bergantung pada pintasan dataset (skema dan `www.`) dan belum layak diklaim untuk produksi — rinciannya di `external_500/external_report.md`.

---

## Panduan Menjalankan Eksperimen (Reproducibility)

### 1. Menyiapkan Lingkungan Python
Pastikan Python 3.10 atau Anaconda telah terpasang:
```bash
# Pemasangan dependensi utama
pip install numpy pandas scikit-learn xgboost onnxruntime skl2onnx joblib
```

### 2. Melatih Model
```bash
# Melatih Random Forest (Varian Full Dataset yang digunakan pada ekstensi)
python hybrid/train_hybrid_model_v2.py --no-undersample --out rf_models/rf_full.pkl

# Melatih Random Forest varian Undersampled
python hybrid/train_hybrid_model_v2.py --out rf_models/phishing_optimized.pkl

# Menjalankan pipeline lengkap XGBoost (Scaler -> SelectKBest -> RFECV -> PCA -> Tuning)
python Xgboost/train_xgb.py
```

### 3. Menjalankan Uji Benchmark Komparatif
```bash
# Menjalankan evaluasi sampling (Undersampling vs Non-Undersampling)
python benchmark_sampling.py

# Menjalankan benchmark komparasi format PKL vs ONNX
python benchmark_all_models.py

# Menjalankan pengujian konsumsi memori dan latensi
python benchmark_memory.py

# Menjalankan pengujian validasi URL nyata dari PhishTank
python run_realworld_test.py
```

---

## Panduan Pemasangan Ekstensi Chrome

Untuk menguji langsung ekstensi di peramban Google Chrome:

1. Buka Google Chrome dan navigasikan ke `chrome://extensions/`.
2. Aktifkan sakelar **Developer mode** di pojok kanan atas.
3. Klik tombol **Load unpacked** di pojok kiri atas.
4. Pilih folder `phishingdetectorExt/` dari direktori repositori ini.
5. Ekstensi **"Phishing Detector (RF + XGBoost)"** akan muncul di bilah alat peramban.

### Fitur Antarmuka Ekstensi
- **Model Switcher:** Pengguna dapat berpindah antara model *Random Forest*, *XGBoost*, dan *CatBoost* secara instan melalui *dropdown* di bagian atas jendela popup.
- **Auto-Scan & Pre-Execution Blocking:** Intersepsi navigasi via `webNavigation.onBeforeNavigate` memblokir akses ke situs phishing (>80%) sebelum aset berbahaya dieksekusi.
- **Warning Banner:** Menampilkan spanduk peringatan melayang dengan opsi *Trust Site* dan *Details* untuk URL mencurigakan (probabilitas 60%–80%).
- **Blocked Page (`blocked.html`):** Menyediakan tombol *Go Back*, *Trust Site*, dan *Continue* (dilengkapi mekanisme *Session Bypass* satu kali sehingga tidak memicu pengalihan berulang).
- **Manajemen Whitelist:** Pengguna dapat menambah atau menghapus domain aman yang dikecualikan dari pemindaian.

---

## Kesimpulan & Rencana Pengembangan Lanjutan


### Rekomendasi Sesuai Arahan Pembimbing
1. **Penambahan Fitur Character n-grams:** *Sudah dilaksanakan.* Blok n-gram karakter (3–5 karakter pada host, TF-IDF 1.500 n-gram) diimplementasikan, dilatih, dan dievaluasi pada Bab IV Subbab 4.9 (`ablation_char_results/`). Hasilnya: memperkuat pintasan `www.` dan tidak memenuhi batas FPR pada daftar eksternal, sehingga tidak dipasang di ekstensi.
2. **Evaluasi Model Ketiga:** *Sudah dilaksanakan.* CatBoost dipasang sebagai model ketiga pada ekstensi versi 3.4 (disebut "model dasar" pada draf) dan dibandingkan dengan Random Forest dan XGBoost pada Bab IV.

---

