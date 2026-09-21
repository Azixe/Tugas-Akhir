# Manual checks — P-02 / P-03 / functional (2026-09-21)

> Same machine and session as the P-01 run: extension v3.2 on Chrome 153 (Windows, 12 logical cores, 32 GB).
> Screenshots live in this folder.

## P-02 — Extension memory (Chrome Task Manager)

| Condition | Task Manager | ≈ MiB | Screenshot |
|---|---:|---:|---|
| Idle after load (popup closed) | 52,952K | 51.7 | `image.png` |
| After a legitimate scan (YouTube) | 121,728K | 118.9 | `image-1.png` |
| After switching models (RF ⇄ XGB) | 127,296K | 124.3 | `image-5.png` |
| Blocked page open ("⚠ Blocked" entry) | 138,276K | 135.0 | `image-4.png` |

- The "⚠ Blocked" row is a **separate extension process** for `blocked.html`, not the service worker.
- The process entry only appears once the MV3 service worker has started; opening the popup briefly showed ~110 MB before settling at 52.9 MB.
- Against the Bab 3 targets (idle < 50 MB, peak < 150 MB): idle is marginally above (51.7 MiB), peak is below (135.0 MiB).

## P-03 — CPU (Chrome Task Manager)

| Condition | CPU | Screenshot |
|---|---:|---|
| Idle | 0.0% | `image-2.png` |
| Scan with the model loaded | 4.5% | `image-3.png` |
| Model switch / session init transient | 49.3% | `image-5.png` |

- Steady-state scans stay at ~4.5% (target < 30%). The 49.3% reading is the one-time
  ONNX/WASM session load (~240 ms); because MV3 evicts the idle service worker, this
  transient can recur on the first scan after an eviction.

## Functional tests

| Check | Result | Screenshot |
|---|---|---|
| Warning banner (RF, 70.0%, crafted safe URL) | Banner displayed; Trust / Details / Dismiss work | `image-6.png` |
| Blocked page (XGB, live PhishTank URL, 81.4%) | URL + confidence render; Go Back / Continue (one-time bypass) / Trust Site all work | `Halaman Blokir.png` |
| Popup PHISHING | "High Risk (81.4%)" · XGBoost · 2.6 ms | `High risk pop-up.png` |
| Popup SAFE | "Low Risk (7.8%)" · Random Forest · 1.0 ms | `Safe pop-up.png` |

- The blocked URL (`zyncrogeldplatform.de`) was a fresh PhishTank `online-valid` sample
  collected on 2026-09-21.
- The blocked-page screenshot also proves the MV3 CSP fix: previously the inline script
  was blocked, so the page showed "Loading…" and the buttons were dead.
- The SAFE popup used `https://www.youtube.com/`; the popup scan path scores the active
  tab directly and does not consult the static whitelist (by design).
