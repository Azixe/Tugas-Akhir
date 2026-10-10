# CatBoost ONNX deployment verification (persistent evidence)

Deployed file: `phishingdetectorExt\phishing_catboost.onnx`; native artifact: `sampling_results/cb_full.pkl`.
Protocol: common split, seed 42 (test set = 90,036 URLs).

## 1. Native ↔ ONNX parity (first 300 test URLs)

| Metric | Value |
|---|---:|
| Label agreement | 100.0% |
| Max abs prob. diff | 8.63e-08 |

## 2. ONNX metrics on the full test set vs native

| Metric | Native (pkl) | ONNX (deployed) |
|---|---:|---:|
| Accuracy (%) | 99.59 | 99.59 |
| Precision (%) | 99.90 | 99.90 |
| Recall (%) | 98.35 | 98.35 |
| F1 (%) | 99.12 | 99.12 |
| FPR (%) | 0.0304 | 0.0304 |
| FP / FN | 21 / 345 | 21 / 345 |

## 3. Extension package size (P-04, criterion < 30 MB)

| Measure | Size |
|---|---:|
| Folder on disk | 24.3 MB |
| Deflate zip (≈ .crx) | 6.17 MB |

> n=300 for parity (sampled), full 90,036 for metrics.
