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
- P-02/P-03 (Task Manager memory/CPU) and the blocked/banner click tests are recorded separately in the guide and added here once measured.

