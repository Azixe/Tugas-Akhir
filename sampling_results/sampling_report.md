# Undersampling vs Non-Undersampling — Benchmark Report

Generated: 2026-09-29T08:31:32+00:00
Dataset: `URL dataset.csv` — 450176 rows after cleaning
Protocol: `sampling.py` (stratified 80/20, seed 42); all models evaluated on the same 90036-URL test set

- Train pool: 360140 (276590 legit / 83550 phishing)
- Common test: 90036 (69148 legit / 20888 phishing)

| Model | Train size | Accuracy | Precision (phish) | Recall (phish) | F1 (phish) | FPR | FP | FN | Avg latency (ms) | Size (KB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF (undersampled) | 167100 | 0.9924 | 0.9970 | 0.9702 | 0.9834 | 0.088% | 61 | 623 | 25.2896 | 4320 |
| RF (full dataset) | 360140 | 0.9904 | 1.0000 | 0.9587 | 0.9789 | 0.001% | 1 | 863 | 27.7285 | 7529 |
| XGBoost (undersampled) | 167100 | 0.9946 | 0.9883 | 0.9887 | 0.9885 | 0.354% | 245 | 237 | 4.1354 | 4672 |
| XGBoost (full dataset) | 360140 | 0.9955 | 0.9957 | 0.9848 | 0.9903 | 0.127% | 88 | 317 | 4.1839 | 7830 |
| CatBoost (undersampled) | 167100 | 0.9957 | 0.9955 | 0.9860 | 0.9907 | 0.136% | 94 | 293 | 7.7031 | 4030 |
| CatBoost (full dataset) | 360140 | 0.9959 | 0.9990 | 0.9835 | 0.9912 | 0.030% | 21 | 345 | 7.5249 | 7191 |

## Notes for Bab 4

- Same architecture and hyperparameters per model family; only the training data differs.
- FPR is the key metric: compare FP counts at the same recall level.
- Latency measured locally (Python), not in the browser — see the ONNX benchmark for extension numbers.
