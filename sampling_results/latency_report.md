# Latency Benchmark (Python environment, deployed artifacts)

Full pipeline per URL (feature extraction + preprocessing + model) on the shared 1000-URL sample (`memory_test_urls.json`), 20 warm-up scans each.

| Model | Min (ms) | Max (ms) | Avg (ms) | P95 (ms) |
|---|---:|---:|---:|---:|
| RF (.pkl) | 21.959 | 123.608 | 24.308 | 25.288 |
| RF (.onnx) | 4.434 | 8.501 | 5.242 | 5.77 |
| XGBoost (.pkl) | 3.223 | 83.249 | 3.973 | 4.526 |
| XGBoost (.onnx) | 5.902 | 21.967 | 7.118 | 7.767 |

> System-scheduling outliers show up in Max; Avg/P95 are the stable measures.
