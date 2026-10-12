# Overfitting Check — train pool vs common test set (deployed variants)

Generated: 2026-10-12T10:58:28+00:00
Protocol: `sampling.py` (stratified 80/20, seed 42). Train pool: 360140 URLs; common test set: 90036 URLs.
Models: `sampling_results/{rf,xgb,cb}_full.pkl` (full-dataset variants, the ones deployed in v3.4).

## Train vs test at the full ensemble size (threshold 0.5)

| Model | Split | Accuracy | Precision | Recall | F1 | FPR | Log loss | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | train | 0.9903 | 0.9998 | 0.9584 | 0.9787 | 0.005% | 0.1465 | 15 | 3475 |
| Random Forest | test | 0.9904 | 1.0000 | 0.9587 | 0.9789 | 0.001% | 0.1464 | 1 | 863 |
| XGBoost | train | 0.9986 | 0.9998 | 0.9941 | 0.9969 | 0.007% | 0.0065 | 18 | 492 |
| XGBoost | test | 0.9955 | 0.9957 | 0.9848 | 0.9903 | 0.127% | 0.0180 | 88 | 317 |
| CatBoost | train | 0.9963 | 0.9988 | 0.9854 | 0.9920 | 0.036% | 0.0165 | 100 | 1221 |
| CatBoost | test | 0.9959 | 0.9990 | 0.9835 | 0.9912 | 0.030% | 0.0185 | 21 | 345 |

## Train-test gap (train minus test, percentage points; log loss as test minus train)

| Model | Accuracy gap | Recall gap | F1 gap | FPR gap | Log loss gap |
|---|---:|---:|---:|---:|---:|
| Random Forest | -0.009 | -0.028 | -0.021 | +0.0040 | -0.0001 |
| XGBoost | +0.308 | +0.929 | +0.669 | -0.1208 | +0.0115 |
| CatBoost | +0.040 | +0.190 | +0.087 | +0.0058 | +0.0020 |

## Random Forest — metrics as the ensemble grows

| Trees / iterations | Train F1 | Test F1 | Train log loss | Test log loss |
|---:|---:|---:|---:|---:|
| 1 | 0.9534 | 0.9540 | 0.0603 | 0.0617 |
| 5 | 0.9805 | 0.9805 | 0.0700 | 0.0702 |
| 10 | 0.9805 | 0.9802 | 0.1147 | 0.1147 |
| 20 | 0.9769 | 0.9770 | 0.1310 | 0.1309 |
| 40 | 0.9773 | 0.9773 | 0.1285 | 0.1283 |
| 60 | 0.9774 | 0.9773 | 0.1413 | 0.1412 |
| 80 | 0.9770 | 0.9770 | 0.1470 | 0.1469 |
| 100 | 0.9787 | 0.9789 | 0.1465 | 0.1464 |

Lowest test log loss at 1 (0.0617); at the full size it is 0.1464.

## XGBoost — metrics as the ensemble grows

| Trees / iterations | Train F1 | Test F1 | Train log loss | Test log loss |
|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.4647 | 0.4647 |
| 10 | 0.9255 | 0.9254 | 0.2036 | 0.2037 |
| 25 | 0.9526 | 0.9531 | 0.0889 | 0.0901 |
| 50 | 0.9716 | 0.9700 | 0.0466 | 0.0489 |
| 100 | 0.9845 | 0.9811 | 0.0271 | 0.0312 |
| 200 | 0.9915 | 0.9871 | 0.0169 | 0.0227 |
| 300 | 0.9940 | 0.9889 | 0.0129 | 0.0200 |
| 400 | 0.9954 | 0.9893 | 0.0104 | 0.0188 |
| 500 | 0.9961 | 0.9899 | 0.0087 | 0.0183 |
| 600 | 0.9966 | 0.9903 | 0.0075 | 0.0181 |
| 700 | 0.9969 | 0.9903 | 0.0065 | 0.0180 |

Lowest test log loss at 700 (0.0180); at the full size it is 0.0180.

## CatBoost — metrics as the ensemble grows

| Trees / iterations | Train F1 | Test F1 | Train log loss | Test log loss |
|---:|---:|---:|---:|---:|
| 1 | 0.9816 | 0.9824 | 0.4648 | 0.4646 |
| 10 | 0.9851 | 0.9855 | 0.0556 | 0.0552 |
| 25 | 0.9867 | 0.9870 | 0.0311 | 0.0311 |
| 50 | 0.9879 | 0.9880 | 0.0259 | 0.0264 |
| 100 | 0.9895 | 0.9892 | 0.0219 | 0.0230 |
| 150 | 0.9907 | 0.9898 | 0.0198 | 0.0212 |
| 200 | 0.9916 | 0.9907 | 0.0181 | 0.0198 |
| 250 | 0.9919 | 0.9910 | 0.0171 | 0.0189 |
| 300 | 0.9920 | 0.9912 | 0.0165 | 0.0185 |

Lowest test log loss at 300 (0.0185); at the full size it is 0.0185.

## Reading

- A model overfits when it keeps improving on the data it was fitted on while getting worse on held-out data.
- For Random Forest, log loss rises as trees are added, by the same amount on train and test: averaging depth-limited trees pulls the probabilities toward the middle. It is the same conservatism seen in the threshold analysis, not a train-test divergence.
- Train and test here come from the same random split of one dataset, so this check covers in-distribution overfitting only. It does not measure behaviour on URLs from other sources.
