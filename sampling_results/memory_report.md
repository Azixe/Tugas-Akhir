# Memory Benchmark (Python environment)

Fresh process per model; RSS sampled every 5.0 ms during 1000 single-URL scans.

| Model | Baseline (MB) | Setelah load (MB) | Load delta (MB) | Min (MB) | Max (MB) | Avg (MB) | Peak delta (MB) | Scan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF (.pkl) | 154.1 | 254.7 | 100.6 | 256.0 | 256.7 | 256.4 | 102.5 | 38.1 |
| RF (.onnx) | 154.0 | 164.6 | 10.6 | 164.9 | 165.0 | 165.0 | 11.0 | 5.5 |
| XGBoost (.pkl) | 154.0 | 261.0 | 107.1 | 261.7 | 261.8 | 261.8 | 107.8 | 4.4 |
| XGBoost (.onnx) | 154.4 | 170.2 | 15.9 | 170.8 | 170.9 | 170.9 | 16.5 | 7.3 |
| CatBoost (.pkl) | 154.3 | 282.2 | 127.9 | 284.8 | 284.8 | 284.8 | 130.5 | 7.5 |
| CatBoost (.onnx) | 154.2 | 167.9 | 13.7 | 168.2 | 168.3 | 168.2 | 14.1 | 5.2 |

> Baseline ≈ 154.1 MB: Python interpreter + shared libraries (sklearn / xgboost / onnxruntime) imported before the model is loaded, identical across entries.
> Python-process RSS. Browser (extension) memory is measured separately via Chrome Task Manager (P-02) and is not the same quantity.
