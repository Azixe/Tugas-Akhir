# legacy/

Earlier-generation scripts, kept for history. Nothing in the current pipeline imports them and
no report in the repo was produced by them. Prefer the modules in the repo root
(`features.py`, `sampling.py`, `ml_pipelines.py`) and the scripts listed in `README.md`.

Moved here on 2026-10-12 (they used to sit in the repo root and in `hybrid/`). Like every other
script in this repo they expect to be run **from the repo root**, e.g. `python legacy/ML_model.py`.

| File | What it was | State |
|---|---|---|
| `ML_model.py` | First Random Forest trainer (TF-IDF only), writes `phishing_detector_model.pkl` | Runs; superseded by `hybrid/train_hybrid_model_v2.py` |
| `test_model.py` | Interactive test of `phishing_detector_model.pkl` | Needs that pickle in the working directory |
| `train_hybrid_model.py` | First TF-IDF + structural trainer (own copy of the feature code) | Reads `hybrid/URL dataset.csv`, which is not in the repo |
| `benchmark_models.py` | First RF vs XGBoost ONNX benchmark | Runs; superseded by `benchmark_all_models.py` |
| `verify_onnx.py` | Prints the inputs/outputs of `phishing_rf.onnx` in the working directory | Run it from `phishingdetectorExt/` |
| `tfidf_data.json` | TF-IDF vocabulary exported in April 2026 | Not the deployed one; the extension uses `phishingdetectorExt/tfidf_data.json` |
| `train_xgb.py` | Local XGBoost trainer with SelectKBest grid search → RFECV → PCA (moved from `Xgboost/` on 2026-10-20) | Not used for any reported result; the deployed XGBoost comes from `ml_pipelines.train_xgb_pipeline` |
| `benchmark_results_legacy_protocol.md` | PKL vs ONNX benchmark on the old two-model protocol (moved from `docs/` on 2026-10-20) | Superseded by `docs/benchmark_pkl_vs_onnx_full.md` |
