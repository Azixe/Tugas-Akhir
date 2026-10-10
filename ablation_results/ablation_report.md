# Feature Ablation — scheme, `www.` and feature groups

Generated: 2026-10-10T12:42:19+00:00
Dataset: `URL dataset.csv` — 450176 rows after cleaning
Protocol: `sampling.py` common split (stratified 80/20, seed 42), then `validation_split` (stratified 10% of the train pool).

- Fit set: 324126 (248931 legit / 75195 phishing)
- Validation set: 36014 (27659 legit / 8355 phishing)
- Common test set: 90036 (NOT scored in this run)
- Empty-string token at training time: dropped
- Training data: full fit set (no undersampling); CatBoost iterations: 300

## Validation set — use this table to choose an arm

| Arm | Model | Inputs | Accuracy | Precision (phish) | Recall (phish) | F1 (phish) | FPR | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | Random Forest | 1509 | 0.9916 | 0.9996 | 0.9643 | 0.9817 | 0.011% | 3 | 298 |
| `all` | XGBoost | 1509 | 0.9956 | 0.9941 | 0.9868 | 0.9904 | 0.177% | 49 | 110 |
| `all` | CatBoost | 1509 | 0.9961 | 0.9985 | 0.9847 | 0.9916 | 0.043% | 12 | 128 |
| `no_scheme` | Random Forest | 1509 | 0.9016 | 0.9961 | 0.5783 | 0.7318 | 0.069% | 19 | 3523 |
| `no_scheme` | XGBoost | 1509 | 0.9744 | 0.9710 | 0.9172 | 0.9433 | 0.828% | 229 | 692 |
| `no_scheme` | CatBoost | 1509 | 0.9653 | 0.9825 | 0.8661 | 0.9206 | 0.466% | 129 | 1119 |
| `no_scheme_www` | Random Forest | 1509 | 0.8418 | 0.9822 | 0.3238 | 0.4870 | 0.177% | 49 | 5650 |
| `no_scheme_www` | XGBoost | 1509 | 0.9455 | 0.9179 | 0.8401 | 0.8773 | 2.271% | 628 | 1336 |
| `no_scheme_www` | CatBoost | 1509 | 0.9306 | 0.9336 | 0.7543 | 0.8344 | 1.620% | 448 | 2053 |
| `tfidf_only` | Random Forest | 1500 | 0.9880 | 0.9999 | 0.9485 | 0.9735 | 0.004% | 1 | 430 |
| `tfidf_only` | XGBoost | 1500 | 0.9930 | 0.9933 | 0.9764 | 0.9848 | 0.199% | 55 | 197 |
| `tfidf_only` | CatBoost | 1500 | 0.9935 | 0.9991 | 0.9728 | 0.9858 | 0.025% | 7 | 227 |
| `struct_only` | Random Forest | 9 | 0.9287 | 0.9743 | 0.7114 | 0.8224 | 0.568% | 157 | 2411 |
| `struct_only` | XGBoost | 9 | 0.9307 | 0.9427 | 0.7469 | 0.8334 | 1.370% | 379 | 2115 |
| `struct_only` | CatBoost | 9 | 0.9293 | 0.9501 | 0.7337 | 0.8280 | 1.164% | 322 | 2225 |
| `tfidf_only_no_scheme_www` | Random Forest | 1500 | 0.8273 | 0.9850 | 0.2594 | 0.4106 | 0.119% | 33 | 6188 |
| `tfidf_only_no_scheme_www` | XGBoost | 1500 | 0.9186 | 0.9085 | 0.7217 | 0.8044 | 2.195% | 607 | 2325 |
| `tfidf_only_no_scheme_www` | CatBoost | 1500 | 0.9049 | 0.9247 | 0.6423 | 0.7580 | 1.580% | 437 | 2989 |
| `struct_only_no_scheme_www` | Random Forest | 9 | 0.8610 | 0.8275 | 0.5063 | 0.6282 | 3.189% | 882 | 4125 |
| `struct_only_no_scheme_www` | XGBoost | 9 | 0.8652 | 0.7869 | 0.5745 | 0.6641 | 4.700% | 1300 | 3555 |
| `struct_only_no_scheme_www` | CatBoost | 9 | 0.8601 | 0.7906 | 0.5397 | 0.6415 | 4.317% | 1194 | 3846 |

## Notes

- Hyperparameters are fixed per model family; only the feature construction differs between arms.
- Models are fitted on the fit set (90% of the train pool), so these numbers are not directly comparable with `sampling_results/sampling_report.md`, where models see the whole train pool.
- Validation and test rows come from the same random split as the training data (in-distribution; see `sampling_results/dataset_split_report.md` for host overlap). They do not measure generalisation to live URLs.
