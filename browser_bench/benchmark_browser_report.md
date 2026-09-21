# Browser Benchmark — P-01 Latency vs Python (deployed models)

- Date: 2026-09-21T10:36:20.247Z · Extension v3.2 (deployed models, hashes below)
- Browser: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 · 12 logical cores · 32 GB RAM
- URL set: `bench_urls.json`, 1000 URLs (common test split, seed 42) — identical to the Python benchmark
- Method: 5 warm-up scans per model, then per-URL timing through the extension's scan path (popup-like message path); Python side re-measured here with `features.py` + the deployed ONNX files on the same URLs

## File integrity

| File | SHA-256 | Checked against bench output |
|---|---|---|
| `phishing_rf.onnx` | `625c9333151667c4…` | MATCH |
| `phishing_xgb.onnx` | `daa8c8d49a200ac3…` | MATCH |
| `tfidf_data.json` | `594309360bc559b6…` | MATCH |
| `tfidf_data_xgb.json` | `57f715b79a084bd3…` | MATCH |
| `xgb_preprocessing.json` | `9a87f5a92e9d31fd…` | MATCH |

## Latency (ms, warm; min / median / avg / p95 / max)

| Model | Init (cold) | Browser round-trip | Browser pipeline | Python pipeline (same files, same URLs) |
|---|---|---|---|---|
| RF | 410.0 | 0.6 / 0.8 / 0.912 / 1.5 / 2.9 | 0.1 / 0.2 / 0.263 / 0.5 / 1.4 | 4.295 / 5.372 / 5.294 / 5.905 / 15.889 |
| XGB | 239.9 | 0.6 / 0.9 / 1.063 / 1.8 / 3.6 | 0.1 / 0.3 / 0.349 / 0.6 / 0.8 | 5.768 / 7.105 / 7.009 / 7.719 / 10.501 |

## Verdict distribution

| Model | Browser | Python |
|---|---|---|
| RF | PHISHING 74 · SUSPICIOUS 146 · SAFE 780 | PHISHING 74 · SUSPICIOUS 146 · SAFE 780 |
| XGB | PHISHING 242 · SUSPICIOUS 0 · SAFE 758 | PHISHING 242 · SUSPICIOUS 0 · SAFE 758 |

## Browser vs Python parity (identical URLs)

| Model | Label agreement | Verdict agreement | Mean \|Δp\| | Max \|Δp\| | Diffs > 1pp | Errors |
|---|---|---|---|---|---|---|
| RF | 100.0% | 100.0% | 0.0pp | 0.0001pp | 0 | 0 |
| XGB | 100.0% | 100.0% | 0.0pp | 0.0pp | 0 | 0 |

## Scope notes

- Init times: each model pays its own first load (ONNX fetch + session creation); the first model initialized
  also includes ONNX Runtime Web/WASM startup (RF 410 ms first vs XGB 240 ms second in this run).
- Browser numbers are **warm** and cover the extension scan path (message → feature extraction → ONNX → response).
  A real navigation adds the cold model init (first column) and page-load effects; model init is measured separately per model.
- Python numbers are the same computation on `onnxruntime` CPU, not a runtime-equivalent comparison (WASM vs native);
  they are reported as a sanity baseline, not as a browser expectation.
- Parity was 100% (labels and verdicts) for both models on all 1000 URLs, max |Δp| 0.0001pp, which validates the
  JS feature pipeline end-to-end against `features.py` (including the `is_common_tld` trailing-slash fix).
- P-02/P-03 (Task Manager memory/CPU) and the functional click tests are in the "Manual checks" section below.


---

# Manual checks — P-02 / P-03 / functional (2026-09-21)

> Same machine and session as the P-01 run: extension v3.2 on Chrome 153 (Windows, 12 logical cores, 32 GB).
> Screenshots live in this folder.

## P-02 — Extension memory (Chrome Task Manager)

| Condition | Task Manager | ≈ MiB | Screenshot |
|---|---:|---:|---|
| Idle after load (popup closed) | 52,952K | 51.7 | `screenshots/p02_memory_idle.png` |
| After a legitimate scan (YouTube) | 121,728K | 118.9 | `screenshots/p02_memory_scan_legit.png` |
| After switching models (RF ⇄ XGB) | 127,296K | 124.3 | `screenshots/p02_p03_model_switch.png` |
| Blocked page open ("⚠ Blocked" entry) | 138,276K | 135.0 | `screenshots/p02_memory_blocked_page.png` |

- The "⚠ Blocked" row is a **separate extension process** for `blocked.html`, not the service worker.
- The process entry only appears once the MV3 service worker has started; opening the popup briefly showed ~110 MB before settling at 52.9 MB.
- Against the Bab 3 targets (idle < 50 MB, peak < 150 MB): idle is marginally above (51.7 MiB), peak is below (135.0 MiB).

## P-03 — CPU (Chrome Task Manager)

| Condition | CPU | Screenshot |
|---|---:|---|
| Idle | 0.0% | `screenshots/p03_cpu_idle.png` |
| Scan with the model loaded | 4.5% | `screenshots/p03_cpu_scan.png` |
| Model switch / session init transient | 49.3% | `screenshots/p02_p03_model_switch.png` |

- Steady-state scans stay at ~4.5% (target < 30%). The 49.3% reading is the one-time
  ONNX/WASM session load (~240 ms); because MV3 evicts the idle service worker, this
  transient can recur on the first scan after an eviction.

## Functional tests

| Check | Result | Screenshot |
|---|---|---|
| Warning banner (RF, 70.0%, crafted safe URL) | Banner displayed; Trust / Details / Dismiss work | `screenshots/functional_banner_70pct.png` |
| Blocked page (XGB, live PhishTank URL, 81.4%) | URL + confidence render; Go Back / Continue (one-time bypass) / Trust Site all work | `screenshots/functional_blocked_page_81pct.png` |
| Popup PHISHING | "High Risk (81.4%)" · XGBoost · 2.6 ms | `screenshots/popup_phishing_xgb.png` |
| Popup SAFE | "Low Risk (7.8%)" · Random Forest · 1.0 ms | `screenshots/popup_safe_rf.png` |

- The blocked URL (`zyncrogeldplatform.de`) was a fresh PhishTank `online-valid` sample
  collected on 2026-09-21.
- The blocked-page screenshot also proves the MV3 CSP fix: previously the inline script
  was blocked, so the page showed "Loading…" and the buttons were dead.
- The SAFE popup used `https://www.youtube.com/`; the popup scan path scores the active
  tab directly and does not consult the static whitelist (by design).
