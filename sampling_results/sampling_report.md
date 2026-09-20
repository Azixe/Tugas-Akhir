# Undersampling vs Non-Undersampling — Benchmark Report

Generated: 2026-09-20T09:16:41+00:00
Dataset: `URL dataset.csv` — 450176 rows after cleaning
Protocol: `sampling.py` (stratified 80/20, seed 42); all models evaluated on the same 90036-URL test set

- Train pool: 360140 (276590 legit / 83550 phishing)
- Common test: 90036 (69148 legit / 20888 phishing)

| Model | Train size | Accuracy | Precision (phish) | Recall (phish) | F1 (phish) | FPR | FP | FN | Avg latency (ms) | Size (KB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF (undersampled) | 167100 | 0.9924 | 0.9970 | 0.9702 | 0.9834 | 0.088% | 61 | 623 | 26.4855 | 4320 |
| RF (full dataset) | 360140 | 0.9904 | 1.0000 | 0.9587 | 0.9789 | 0.001% | 1 | 863 | 23.9623 | 7529 |
| XGBoost (undersampled) | 167100 | 0.9946 | 0.9883 | 0.9887 | 0.9885 | 0.354% | 245 | 237 | 3.8863 | 4672 |
| XGBoost (full dataset) | 360140 | 0.9955 | 0.9957 | 0.9848 | 0.9903 | 0.127% | 88 | 317 | 3.3934 | 7830 |

## Notes for Bab 4

- Same architecture and hyperparameters per model family; only the training data differs.
- FPR is the key metric: compare FP counts at the same recall level.
- Latency measured locally (Python), not in the browser — see the ONNX benchmark for extension numbers.
