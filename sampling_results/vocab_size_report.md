# TF-IDF vocabulary size — Random Forest (in-distribution)

Generated: 2026-10-15T12:03:23+00:00
Protocol: `sampling.py` (stratified 80/20, seed 42); train pool 360140 URLs, common test set 90036 URLs, threshold 0.5.
Random Forest: `ml_pipelines.RF_HPARAMS` (n_estimators=100, n_jobs=-1, random_state=42, max_depth=15, min_samples_split=10, min_samples_leaf=4).
Sizes are the two files the extension ships for this model, exported as `export_to_web.py` does.

| max_features | Vocabulary | Accuracy | Precision | Recall | F1 | FPR | FP | FN | ONNX (KB) | tfidf JSON (KB) | Tree nodes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 500 | 0.9926 | 0.9990 | 0.9689 | 0.9837 | 0.029% | 20 | 650 | 915 | 17 | 24522 |
| 1000 | 1000 | 0.9911 | 0.9996 | 0.9621 | 0.9805 | 0.012% | 8 | 791 | 824 | 34 | 22152 |
| 1500 | 1500 | 0.9909 | 0.9994 | 0.9614 | 0.9800 | 0.019% | 13 | 807 | 746 | 52 | 20114 |
| 3000 | 3000 | 0.9681 | 1.0000 | 0.8627 | 0.9263 | 0.000% | 0 | 2868 | 607 | 105 | 16492 |
| 5000 | 5000 | 0.9689 | 1.0000 | 0.8660 | 0.9282 | 0.000% | 0 | 2798 | 528 | 178 | 14436 |

- The deployed RF (`phishingdetectorExt/phishing_rf.onnx`) was fitted before the seed was added, so the 1,500 row is a seeded retrain of the same configuration, not the deployed file.
- The test set comes from the same random split as the training data; these numbers say nothing about URLs from other sources (see `external_500/` and `external_posthoc/`).
