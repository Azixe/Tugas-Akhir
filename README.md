# Phishing URL Detection System

Sistem deteksi URL phishing menggunakan Machine Learning (Random Forest) yang berjalan langsung di browser sebagai Chrome Extension.

## 📁 Struktur Project

```
Tugas Akhir/
├── hybrid/
│   └── train_hybrid_model_v2.py   # Training model
├── export_to_web.py               # Export ke ONNX + JSON
├── test_model.py                  # Testing model
├── URL dataset.csv                # Dataset
│
└── phishingdetectorExt/           # Chrome Extension
    ├── manifest.json
    ├── background.js              # Auto-scan service worker
    ├── popup.html/js              # UI manual scan + whitelist
    ├── content.js                 # Warning banner injection
    ├── blocked.html               # Halaman blokir
    ├── utils.js                   # Shared functions
    ├── phishing_rf.onnx           # Model ONNX (~900KB)
    ├── tfidf_data.json            # Vocabulary + IDF weights
    └── libs/
        ├── ort.min.js             # ONNX Runtime
        └── ort-wasm*.wasm         # WASM backend
```

---

## 🔄 Alur Kerja

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Training   │───▶│   Export    │───▶│  Extension  │───▶│  Detection  │
│  (Python)   │    │  (ONNX)     │    │  (Browser)  │    │  (Realtime) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 1️⃣ Training Model

### File: `hybrid/train_hybrid_model_v2.py`

```bash
python hybrid/train_hybrid_model_v2.py
```

**Proses:**
1. Load dataset URL (legitimate + phishing)
2. Ekstraksi fitur:
   - **TF-IDF** (1500 features) - tokenisasi URL
   - **Structural** (9 features) - length, entropy, subdomain, dll
3. Train Random Forest Classifier
4. Evaluasi: Accuracy, Precision, Recall, F1-Score
5. Simpan model: `hybrid_model.pkl`

**Fitur Struktural:**
| # | Fitur | Deskripsi |
|---|-------|-----------|
| 1 | url_length | Panjang URL |
| 2 | dot_count | Jumlah titik |
| 3 | slash_count | Jumlah slash |
| 4 | dash_count | Jumlah dash |
| 5 | at_count | Jumlah @ |
| 6 | digit_ratio | Rasio angka |
| 7 | entropy | Shannon entropy |
| 8 | is_common_tld | TLD umum (com/org/net) |
| 9 | subdomain_level | Kedalaman subdomain |

---

## 2️⃣ Export ke ONNX

### File: `export_to_web.py`

```bash
python export_to_web.py
```

**Output:**
- `phishing_rf.onnx` - Model dalam format ONNX
- `tfidf_data.json` - Vocabulary + IDF weights

**Kenapa ONNX?**
- Bisa dijalankan di browser via ONNX Runtime Web (WASM)
- Tidak perlu server/API - semua lokal
- Lebih privasi - URL tidak dikirim ke mana-mana

---

## 3️⃣ Chrome Extension

### Instalasi

1. Buka `chrome://extensions/`
2. Aktifkan **Developer mode**
3. Klik **Load unpacked**
4. Pilih folder `phishingdetectorExt/`

### Cara Kerja

```
User buka URL
      │
      ▼
┌─────────────────┐
│ background.js   │──▶ Cek whitelist
│ (Service Worker)│
└────────┬────────┘
         │ tidak di whitelist
         ▼
┌─────────────────┐
│ Feature Extract │──▶ TF-IDF + Structural
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ONNX Inference  │──▶ phishing_rf.onnx
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│              KEPUTUSAN                  │
├─────────────────────────────────────────┤
│ Confidence > 80%  → BLOCK (redirect)    │
│ Confidence 60-80% → WARNING (banner)    │
│ Confidence < 60%  → SAFE (no action)    │
└─────────────────────────────────────────┘
```

### Fitur Extension

- ✅ **Auto-scan** - Otomatis scan setiap navigasi
- ✅ **Manual scan** - Scan via popup
- ✅ **Warning banner** - Non-intrusive alert
- ✅ **Block page** - Halaman peringatan
- ✅ **Whitelist** - Kelola exception
- ✅ **Benchmark** - Log inference time

---

## 4️⃣ Benchmark

Lihat console Service Worker (`chrome://extensions/` → service worker):

```
[BG] ⏱️ Inference: 12.34ms
[BG] ✅ https://google.com 8.2%
[BG] ⏱️ Inference: 15.21ms
[BG] ⚠️ https://suspicious.tk 78.5%
```

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|-------|-----------|
| Training | Python, scikit-learn |
| Model | Random Forest |
| Export | skl2onnx, ONNX |
| Runtime | ONNX Runtime Web (WASM) |
| Extension | Chrome Manifest V3 |
| Storage | chrome.storage.local |

---

## 📝 TODO

- [ ] Implementasi model XGBoost untuk perbandingan
- [ ] A/B testing RF vs XGBoost
- [ ] UI untuk pilih model

---

## 👤 Author

Tugas Akhir - 2025
