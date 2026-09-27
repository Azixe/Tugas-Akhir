# Memory Benchmark (Python environment)

Fresh process per model; RSS sampled every 5.0 ms during 1000 single-URL scans.

| Model | Baseline (MB) | Setelah load (MB) | Load delta (MB) | Min (MB) | Max (MB) | Avg (MB) | Peak delta (MB) | Scan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF (.pkl) | 153.6 | 254.3 | 100.7 | 255.3 | 255.9 | 255.7 | 102.3 | 27.5 |
| RF (.onnx) | 153.4 | 162.8 | 9.3 | 163.1 | 163.1 | 163.1 | 9.7 | 5.4 |
| XGBoost (.pkl) | 153.5 | 259.2 | 105.7 | 259.9 | 259.9 | 259.9 | 106.4 | 4.2 |
| XGBoost (.onnx) | 153.6 | 169.1 | 15.5 | 169.6 | 169.7 | 169.7 | 16.1 | 6.8 |

> Baseline ≈ 153.6 MB: Python interpreter + shared libraries (sklearn / xgboost / onnxruntime) imported before the model is loaded, identical across entries.
> Python-process RSS. Browser (extension) memory is measured separately via Chrome Task Manager (P-02) and is not the same quantity.
