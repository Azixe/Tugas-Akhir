# Deteksi URL Phishing Real-Time Berbasis Client-Side Browser Extension
### Komparasi Kinerja Algoritma Random Forest dan XGBoost via ONNX Runtime Web

> **Repositori Tugas Akhir**  
> **Program Studi:** Informatika  
> **NIM:** 2211102226  

---

## 📌 Ringkasan 

Pertahanan peramban konvensional (seperti *Google Safe Browsing* dan basis data tanda tangan antivirus) mengandalkan daftar hitam (*blacklist*) terpusat. Pendekatan ini memiliki *detection lag* rata-rata 4,5 hari, padahal situs phishing modern memiliki median masa hidup hanya sekitar 5,46 jam. Pendekatan berbasis *server-side* atau API eksternal juga menimbulkan risiko kebocoran privasi riwayat penjelajahan dan latensi jaringan.

Penelitian ini merancang dan mengimplementasikan **ekstensi Google Chrome mandiri (*standalone*)** yang melakukan inferensi Machine Learning secara **100% di sisi klien (*client-side*)** menggunakan **ONNX Runtime Web (WASM)**. Sistem ini membandingkan kinerja deteksi dan efisiensi komputasi antara dua algoritma utama:
1. **Random Forest (RF):** Model *bagging* acuan yang dioptimalkan (*bonsai trees*) untuk ukuran berkas ringan dan *False Positive Rate* (FPR) mendekati 0%.
2. **XGBoost (XGB):** Model *gradient boosting* yang dipadukan dengan *pipeline* reduksi dimensi berganda (*StandardScaler* → *SelectKBest* ANOVA F → *RFECV* → *PCA 95%*) terinspirasi dari arsitektur *PhishGuard*.

Kedua model dikonversi ke format ONNX dan dapat diuji secara langsung serta bergantian melalui antarmuka ekstensi peramban.

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

Struktur proyek disusun secara modular agar memudahkan penelusuran tahapan penelitian:

```
Tugas Akhir/
│
├── phishingdetectorExt/           # PRODUK: Paket Chrome Extension (Manifest V3)
│   ├── manifest.json              # Konfigurasi ekstensi & deklarasi permissions
│   ├── background.js              # Service Worker (intersepsi onBeforeNavigate, ONNX runner)
│   ├── popup.html / popup.js      # Antarmuka popup (pemindaian manual, toggle RF/XGB, kelola whitelist)
│   ├── content.js                 # Injeksi banner peringatan non-intrusif pada situs mencurigakan
│   ├── blocked.html               # Halaman peringatan blokir (dengan session bypass & whitelist)
│   ├── utils.js                   # Ekstraksi fitur leksikal 100% konsisten dengan Python
│   ├── phishing_rf.onnx           # Model Random Forest terkompresi (~730 KB)
│   ├── tfidf_data.json            # Kosakata TF-IDF (1.500 token) untuk RF
│   ├── phishing_xgb.onnx          # Model XGBoost ONNX (~1,25 MB)
│   ├── tfidf_data_xgb.json        # Kosakata TF-IDF untuk XGBoost
│   ├── xgb_preprocessing.json     # Bobot StandardScaler, indeks seleksi (179), & matriks PCA (151)
│   └── libs/                      # Pustaka ONNX Runtime Web (WASM & WASM-SIMD)
│
├── hybrid/                        # Pelatihan model Random Forest
│   └── train_hybrid_model_v2.py   # Pelatihan RF (opsi --no-undersample atau undersampling)
│
├── Xgboost/                       # Pelatihan model XGBoost
│   ├── train_xgb.py               # Pipeline: Scaler → SelectKBest → RFECV → PCA → XGBoost
│   ├── export_xgb_onnx_colab.py   # Ekspor ONNX di Google Colab
│   └── ...
│
├── sampling_results/              # Artefak dan laporan studi sampling (Undersampling vs Non-Undersampling)
│   ├── sampling_report.md         # Laporan perbandingan metrik evaluasi
│   ├── sampling_metrics.csv       # Metrik terkuantisasi
│   └── *.pkl / *.onnx             # Artefak model hasil studi sampling
│
├── features.py                    # SINGLE SOURCE OF TRUTH fitur (9 struktural + TF-IDF + preprocessing)
├── sampling.py                    # Protokol pemisahan data terkontrol (stratified 80/20, seed 42)
├── export_to_web.py               # Skrip konversi model scikit-learn (.pkl) ke format ONNX
├── export_xgb_onnx.py             # Skrip konversi model XGBoost + preprocessing ke format ONNX
├── benchmark_sampling.py          # Skrip eksperimen perbandingan undersampling vs non-undersampling
├── benchmark_all_models.py        # Pengujian perbandingan format PKL vs ONNX
├── benchmark_memory.py            # Pengukuran latensi dan alokasi memori
├── run_realworld_test.py          # Pengujian validasi URL nyata (PhishTank feed aktif)
├── MODELS.md                      # Dokumentasi rekam jejak, ukuran berkas, & SHA-256 checksum model
├── URL dataset.csv                # Dataset utama (Mendeley Phishing Dataset 2024, ~450k baris)
└── README.md                      # Dokumentasi komprehensif repositori (dokumen ini)
```

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
| **Varian Data Latih** | Full Dataset (360.140 URL) | Undersampled 1:1 (167.100 URL) |
| **Dimensi Fitur Masukan** | 1.509 fitur langsung | 1.509 → 179 (*SelectKBest+RFECV*) → 151 (*PCA*) |
| **Konfigurasi Utama** | 100 pohon, `max_depth=15`, `min_samples_split=10`, `min_samples_leaf=4` | 700 pohon, `max_depth=5`, `learning_rate=0.1`, `reg_alpha=0.1`, `reg_lambda=2` |
| **Ukuran Berkas Model** | `phishing_rf.onnx` (**730 KB**) | `phishing_xgb.onnx` (**1.253 KB**) + Preprocessing (**654 KB**) |
| **Karakteristik Deteksi** | Sangat konservatif, nyaris tanpa *false positive* (FPR = 0,001%). | Sangat sensitif/agresif, menangkap lebih banyak variasi phishing. |

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

Pengujian dilakukan terhadap 25 URL phishing aktif dari feed PhishTank (dikonfirmasi daring) dan 20 URL *legitimate* populer yang rentan *false positive*:
- **Random Forest:** Recall 0/25 (0%), False Positive 0/20 (0%). Bersifat sangat defensif karena dataset 2024 mengasosiasikan HTTPS dengan URL aman, sedangkan phishing modern saat ini hampir seluruhnya telah menggunakan HTTPS.
- **XGBoost:** Recall 7/25 (28%), False Positive 2/20 (10%). Mampu mendeteksi sebagian phishing aktif pada infrastruktur *free-hosting* (*pages.dev*, *web.app*, dsb.), namun menghasilkan false positive pada layanan tertentu (misal: Steam).

Temuan ini menegaskan keunggulan menyediakan **dua model yang dapat dipilih pengguna**: Random Forest untuk pengguna yang mengutamakan tanpa gangguan (*zero false alarms*), dan XGBoost untuk pengguna yang menginginkan proteksi lebih proaktif.

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
- **Model Switcher:** Pengguna dapat berpindah antara model *Random Forest* dan *XGBoost* secara instan melalui *dropdown* di bagian atas jendela popup.
- **Auto-Scan & Pre-Execution Blocking:** Intersepsi navigasi via `webNavigation.onBeforeNavigate` memblokir akses ke situs phishing (>80%) sebelum aset berbahaya dieksekusi.
- **Warning Banner:** Menampilkan spanduk peringatan melayang dengan opsi *Trust Site* dan *Details* untuk URL mencurigakan (probabilitas 60%–80%).
- **Blocked Page (`blocked.html`):** Menyediakan tombol *Go Back*, *Trust Site*, dan *Continue* (dilengkapi mekanisme *Session Bypass* satu kali sehingga tidak memicu pengalihan berulang).
- **Manajemen Whitelist:** Pengguna dapat menambah atau menghapus domain aman yang dikecualikan dari pemindaian.

---

## Kesimpulan & Rencana Pengembangan Lanjutan


### Rekomendasi Sesuai Arahan Pembimbing
1. **Penambahan Fitur Character n-grams:** Mengintegrasikan n-gram karakter (sub-kata 3–5 karakter) pada vektor masukan untuk memperkuat daya deteksi terhadap teknik pemalsuan domain (*typosquatting* / *combosquatting*).
2. **Evaluasi Model Ketiga:** Melakukan eksperimen komparasi tambahan dengan algoritma alternatif (misal: *CatBoost* atau *LightGBM*) untuk menganalisis efisiensi komputasi lebih lanjut di lingkungan web.

---

