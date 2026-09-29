# Latency Benchmark (Python environment, deployed artifacts)

Full pipeline per URL (feature extraction + preprocessing + model) on the shared 1000-URL sample (`memory_test_urls.json`), 20 warm-up scans each.

| Model | Min (ms) | Max (ms) | Avg (ms) | P95 (ms) | Model-only avg (ms) |
|---|---:|---:|---:|---:|---:|
| RF (.pkl) | 22.491 | 136.792 | 25.799 | 29.634 | - |
| RF (.onnx) | 4.365 | 9.206 | 5.24 | 5.816 | 0.0378 |
| XGBoost (.pkl) | 3.136 | 85.018 | 3.877 | 4.568 | - |
| XGBoost (.onnx) | 5.797 | 28.558 | 7.198 | 8.436 | 0.051 |
| CatBoost (.pkl) | 6.576 | 58.354 | 7.208 | 7.663 | - |
| CatBoost (.onnx) | 4.37 | 8.993 | 5.192 | 5.765 | 0.0459 |

> System-scheduling outliers show up in Max; Avg/P95 are the stable measures.
> Model-only = ONNX session time on pre-computed features (feature extraction excluded); measured for the .onnx entries.
