# Panduan Pengujian Ekstensi — P-01 / P-02 / P-03 / Uji Fungsional

> Sesi: 2026-09-21 · Ekstensi: **v3.2** (model tetap v3.1: RF `625c9333…`, XGB `daa8c8d4…`)
> Prasyarat: ekstensi sudah di-*reload* di `chrome://extensions` (tombol ⟳) setelah commit terakhir.
> ID ekstensi: `jabeplfgcjflfbdbiflpcclngefpkbhm` — cek di `chrome://extensions` kalau berbeda.

---

## P-01 — Latensi scan di browser (otomatis)

Halaman bench memanggil jalur `scan` yang sama dengan popup, untuk 1.000 URL
(`bench_urls.json`, identik dengan benchmark Python: common split, seed 42).

1. **Reload ekstensi** di `chrome://extensions`. Jangan buka popup dulu.
2. Buka tab baru: `chrome-extension://<ID>/bench.html`
3. Pastikan "URLs per model" = `1000`.
4. Klik **Run both** (RF lalu XGB). Progress dan ringkasan tampil di halaman.
   Perkiraan: beberapa menit untuk 2 × 1.000 scan.
5. Setelah status **Finished**, klik **Download JSON**.
   File `bench_browser_v3.2_<timestamp>.json` tersimpan di folder **Downloads**.
6. Beri tahu asisten — file akan diambil langsung dari `/mnt/c/Users/Enzo/Downloads`.

Catatan penting:
- **Jangan buka DevTools service worker sebelum run** (mempertahankan worker tetap hidup → angka init model tidak cold).
- Angka `Init (ms)` per model = muat ONNX pertama kali (cold per model). Model pertama juga termasuk startup WASM.
- Kalau ada error beruntun, run berhenti otomatis dan error tercatat di JSON.

---

## P-02 — Memori ekstensi (Chrome Task Manager)

Persiapan: tutup tab yang tidak perlu, satu jendela Chrome saja.

1. Tekan `Shift+Esc` (Task Manager).
2. Cari baris **"Extension: Phishing Detector"**, lihat kolom *Memory footprint*.
3. Catat kondisi berikut (screenshot tiap kondisi):

| Kondisi | Memori (MB) | Screenshot |
|---|---|---|
| Idle setelah load (popup ditutup, belum scan) | | |
| Setelah scan halaman legitimate (mis. `https://www.youtube.com/`) | | |
| Setelah scan halaman phishing (dari hasil P-01, URL terpilih) | | |
| Setelah ganti model RF ⇄ XGB di popup | | |

Target dari Bab 3: idle **< 50 MB**, peak **< 150 MB**.
Cara scan: buka popup → **Scan Current Tab**.

---

## P-03 — CPU saat scan

1. Masih di Task Manager, kolom *CPU*.
2. Lakukan scan (popup atau buka halaman), amati spike CPU baris ekstensi.
3. Screenshot saat spike tertinggi + catat angkanya.

| Kondisi | CPU (%) | Screenshot |
|---|---|---|
| Idle | | |
| Saat scan halaman | | |

Target: spike **< 30%**.

---

## Uji fungsional halaman blokir (klik)

Model **XGBoost** (pilih di popup). Dua URL legitimate yang memang diprediksi
tinggi oleh XGB (false positive dari validasi sebelumnya, aman dikunjungi):

- `https://steamcommunity.com/id/WhyIzMyLifeLikeDiz/` (≈89,9%)
- `https://steamcommunity.com/market/listings/730/AK-47%20%7C%20Redline%20%28Field-Tested%29` (≈94,6%)

Langkah:
1. Buka URL pertama → `blocked.html` harus muncul (persentase tampil).
2. Uji tombol:
   - **Go Back** → kembali.
   - **Continue** → konfirmasi → halaman terbuka sekali. Buka URL yang sama lagi → diblokir lagi (bypass sekali pakai).
   - **Trust Site** → domain masuk whitelist; uji dengan URL kedua, lalu **hapus `steamcommunity.com` dari whitelist** lewat popup setelah selesai.
3. Alternatif uji tampilan tanpa model:
   `chrome-extension://<ID>/blocked.html?url=https%3A%2F%2Fexample.com&conf=95`

Catatan: kalau situs melakukan redirect, bypass hanya berlaku untuk URL persis yang diblokir
(lihat `docs/extension-known-issues.md`).

---

## Uji banner peringatan (skor 60–80%)

URL uji diisi setelah P-01 selesai (dipilih dari hasil bench supaya pasti masuk
band 60–80% pada model yang dipakai):

| Model | URL | Banner muncul? | Trust/Dismiss/Details berfungsi? |
|---|---|---|---|
| | | | |

Langkah: pilih model di popup → kunjungi URL → banner oranye harus muncul di atas
halaman dengan teks "…% phishing probability".

---

## (Opsional) Latensi navigasi nyata — log service worker

Dilakukan **setelah P-01** (kalau DevTools dibuka lebih dulu, worker tetap hidup dan angka init P-01 tidak cold lagi).

1. `chrome://extensions` → kartu ekstensi → klik **"service worker"** (DevTools terbuka).
2. Kunjungi 5–10 situs (campur legitimate + phishing dari daftar uji).
3. Screenshot baris log `[BG] [RF/XGB] … (X.XXms)`.

---

## Checklist artefak

- [ ] JSON P-01 dari folder Downloads
- [ ] Tabel P-02 terisi + screenshot
- [ ] Tabel P-03 terisi + screenshot
- [ ] Screenshot halaman blokir (klik tombol)
- [ ] Screenshot banner peringatan
- [ ] Screenshot popup SAFE (`Low Risk (x%)`) dan PHISHING (`High Risk (x%)`)
