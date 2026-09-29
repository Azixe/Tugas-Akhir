# CatBoost — size / accuracy trade-off (2026-09-29)

Full-dataset variant on the shared protocol (`sampling.py`, common split, seed 42, 90,036-URL test set), depth 6, lr 0.1. Only `iterations` varies:

| Iterations | ONNX size | Accuracy | Precision (phish) | Recall (phish) | F1 | FPR | FP | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 1,618 KB | 99.59% | 99.90% | 98.35% | 99.12% | 0.0304% | 21 | 73 s |
| 500 | 2,716 KB | 99.62% | 99.89% | 98.46% | 99.17% | 0.0333% | 23 | 94 s |
| 700 | 3,815 KB | 99.63% | 99.88% | 98.50% | 99.19% | 0.0347% | 24 | 136 s |

Reading: metrics plateau after ~300 iterations (Δ ≤ 0.15 pp) while the ONNX export grows 2.4× (1.6 → 3.8 MB). The 300-iteration export meets the thesis's < 2 MB model-size criterion; 700 does not. The deployed CatBoost should use 300 iterations.

Reproduce (from the repo root):
```
python Catboost/train_catboost.py --variant full --iterations 300 --outdir <dir>
python Catboost/train_catboost.py --variant full --iterations 500 --outdir <dir>
python Catboost/train_catboost.py --variant full                     # 700 (study default)
```
