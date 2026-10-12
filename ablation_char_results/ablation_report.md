# Feature Ablation — scheme, `www.` and feature groups

Generated: 2026-10-12T14:00:51+00:00
Dataset: `URL dataset.csv` — 450176 rows after cleaning
Protocol: `sampling.py` common split (stratified 80/20, seed 42), then `validation_split` (stratified 10% of the train pool).

- Fit set: 324126 (248931 legit / 75195 phishing)
- Validation set: 36014 (27659 legit / 8355 phishing)
- Common test set: 90036 (scored below)
- Empty-string token at training time: dropped
- Training data: full fit set (no undersampling); CatBoost iterations: 300
- Character block (`char*` arms only): TF-IDF over 3-5 character n-grams of the host, 1500 terms

## Validation set — use this table to choose an arm

| Arm | Model | Inputs | Accuracy | Precision (phish) | Recall (phish) | F1 (phish) | FPR | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | Random Forest | 1509 | 0.9916 | 0.9996 | 0.9643 | 0.9817 | 0.011% | 3 | 298 |
| `all` | XGBoost | 1509 | 0.9956 | 0.9941 | 0.9868 | 0.9904 | 0.177% | 49 | 110 |
| `all` | CatBoost | 1509 | 0.9961 | 0.9985 | 0.9847 | 0.9916 | 0.043% | 12 | 128 |
| `no_scheme_www` | Random Forest | 1509 | 0.8418 | 0.9822 | 0.3238 | 0.4870 | 0.177% | 49 | 5650 |
| `no_scheme_www` | XGBoost | 1509 | 0.9455 | 0.9179 | 0.8401 | 0.8773 | 2.271% | 628 | 1336 |
| `no_scheme_www` | CatBoost | 1509 | 0.9306 | 0.9336 | 0.7543 | 0.8344 | 1.620% | 448 | 2053 |
| `char` | Random Forest | 3009 | 0.9921 | 0.9944 | 0.9713 | 0.9827 | 0.166% | 46 | 240 |
| `char` | XGBoost | 3009 | 0.9929 | 0.9940 | 0.9755 | 0.9847 | 0.177% | 49 | 205 |
| `char` | CatBoost | 3009 | 0.9973 | 0.9959 | 0.9926 | 0.9942 | 0.123% | 34 | 62 |
| `char_no_scheme_www` | Random Forest | 3009 | 0.8504 | 0.9711 | 0.3659 | 0.5315 | 0.329% | 91 | 5298 |
| `char_no_scheme_www` | XGBoost | 3009 | 0.9473 | 0.9271 | 0.8389 | 0.8808 | 1.992% | 551 | 1346 |
| `char_no_scheme_www` | CatBoost | 3009 | 0.9369 | 0.9285 | 0.7890 | 0.8531 | 1.837% | 508 | 1763 |
| `char_only` | Random Forest | 1509 | 0.9685 | 0.9925 | 0.8710 | 0.9278 | 0.199% | 55 | 1078 |
| `char_only` | XGBoost | 1509 | 0.9791 | 0.9845 | 0.9247 | 0.9537 | 0.441% | 122 | 629 |
| `char_only` | CatBoost | 1509 | 0.9774 | 0.9881 | 0.9136 | 0.9494 | 0.333% | 92 | 722 |
| `char_only_no_scheme_www` | Random Forest | 1509 | 0.8390 | 0.9309 | 0.3307 | 0.4880 | 0.741% | 205 | 5592 |
| `char_only_no_scheme_www` | XGBoost | 1509 | 0.9227 | 0.8826 | 0.7692 | 0.8220 | 3.091% | 855 | 1928 |
| `char_only_no_scheme_www` | CatBoost | 1509 | 0.9082 | 0.8802 | 0.6997 | 0.7796 | 2.878% | 796 | 2509 |

## Common test set — final pass (not for selection)

| Arm | Model | Inputs | Accuracy | Precision (phish) | Recall (phish) | F1 (phish) | FPR | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | Random Forest | 1509 | 0.9915 | 0.9993 | 0.9639 | 0.9813 | 0.022% | 15 | 754 |
| `all` | XGBoost | 1509 | 0.9951 | 0.9947 | 0.9841 | 0.9894 | 0.158% | 109 | 333 |
| `all` | CatBoost | 1509 | 0.9959 | 0.9991 | 0.9831 | 0.9911 | 0.026% | 18 | 352 |
| `no_scheme_www` | Random Forest | 1509 | 0.8406 | 0.9873 | 0.3172 | 0.4801 | 0.123% | 85 | 14263 |
| `no_scheme_www` | XGBoost | 1509 | 0.9474 | 0.9259 | 0.8406 | 0.8812 | 2.033% | 1406 | 3330 |
| `no_scheme_www` | CatBoost | 1509 | 0.9315 | 0.9378 | 0.7547 | 0.8363 | 1.513% | 1046 | 5124 |
| `char` | Random Forest | 3009 | 0.9925 | 0.9938 | 0.9736 | 0.9836 | 0.182% | 126 | 552 |
| `char` | XGBoost | 3009 | 0.9939 | 0.9936 | 0.9800 | 0.9868 | 0.189% | 131 | 417 |
| `char` | CatBoost | 3009 | 0.9977 | 0.9970 | 0.9931 | 0.9950 | 0.090% | 62 | 145 |
| `char_no_scheme_www` | Random Forest | 3009 | 0.8512 | 0.9755 | 0.3677 | 0.5341 | 0.279% | 193 | 13208 |
| `char_no_scheme_www` | XGBoost | 3009 | 0.9468 | 0.9305 | 0.8328 | 0.8790 | 1.879% | 1299 | 3492 |
| `char_no_scheme_www` | CatBoost | 3009 | 0.9385 | 0.9378 | 0.7873 | 0.8560 | 1.578% | 1091 | 4442 |
| `char_only` | Random Forest | 1509 | 0.9708 | 0.9920 | 0.8813 | 0.9334 | 0.214% | 148 | 2480 |
| `char_only` | XGBoost | 1509 | 0.9810 | 0.9849 | 0.9324 | 0.9580 | 0.431% | 298 | 1411 |
| `char_only` | CatBoost | 1509 | 0.9792 | 0.9892 | 0.9202 | 0.9535 | 0.302% | 209 | 1667 |
| `char_only_no_scheme_www` | Random Forest | 1509 | 0.8400 | 0.9381 | 0.3322 | 0.4906 | 0.662% | 458 | 13949 |
| `char_only_no_scheme_www` | XGBoost | 1509 | 0.9229 | 0.8865 | 0.7658 | 0.8218 | 2.962% | 2048 | 4891 |
| `char_only_no_scheme_www` | CatBoost | 1509 | 0.9100 | 0.8875 | 0.7011 | 0.7834 | 2.686% | 1857 | 6243 |

## Notes

- Hyperparameters are fixed per model family; only the feature construction differs between arms.
- Models are fitted on the fit set (90% of the train pool), so these numbers are not directly comparable with `sampling_results/sampling_report.md`, where models see the whole train pool.
- Validation and test rows come from the same random split as the training data (in-distribution; see `sampling_results/dataset_split_report.md` for host overlap). They do not measure generalisation to live URLs.
