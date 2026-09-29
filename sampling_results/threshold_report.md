# Decision-Threshold Sensitivity — deployed models (shared test set)

90,036 URLs from the common test split (seed 42); probabilities from the deployed full-dataset artifacts.

## Random Forest

| Threshold | Precision | Recall | F1 | FPR | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| ≥ 0.5 | 1.0000 | 0.9587 | 0.9789 | 0.0014% | 1 | 863 |
| ≥ 0.6 | 1.0000 | 0.8817 | 0.9371 | 0.0000% | 0 | 2472 |
| ≥ 0.7 | 1.0000 | 0.6693 | 0.8019 | 0.0000% | 0 | 6908 |
| ≥ 0.8 | 1.0000 | 0.2845 | 0.4430 | 0.0000% | 0 | 14945 |
| ≥ 0.9 | 1.0000 | 0.0475 | 0.0907 | 0.0000% | 0 | 19896 |

Extension rule simulation (block >0.8, warn 0.6–0.8): blocked 5943 TP / 0 FP · warned 12473 TP / 0 FP · detected (≥0.6) 18416 TP / 0 FP

## XGBoost

| Threshold | Precision | Recall | F1 | FPR | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| ≥ 0.5 | 0.9957 | 0.9848 | 0.9903 | 0.1273% | 88 | 317 |
| ≥ 0.6 | 0.9968 | 0.9831 | 0.9899 | 0.0954% | 66 | 354 |
| ≥ 0.7 | 0.9974 | 0.9809 | 0.9891 | 0.0766% | 53 | 398 |
| ≥ 0.8 | 0.9980 | 0.9781 | 0.9880 | 0.0578% | 40 | 458 |
| ≥ 0.9 | 0.9987 | 0.9725 | 0.9854 | 0.0376% | 26 | 574 |

Extension rule simulation (block >0.8, warn 0.6–0.8): blocked 20430 TP / 40 FP · warned 104 TP / 26 FP · detected (≥0.6) 20534 TP / 66 FP

## CatBoost

| Threshold | Precision | Recall | F1 | FPR | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| ≥ 0.5 | 0.9990 | 0.9835 | 0.9912 | 0.0304% | 21 | 345 |
| ≥ 0.6 | 0.9992 | 0.9823 | 0.9907 | 0.0231% | 16 | 370 |
| ≥ 0.7 | 0.9994 | 0.9809 | 0.9901 | 0.0174% | 12 | 399 |
| ≥ 0.8 | 0.9996 | 0.9790 | 0.9892 | 0.0116% | 8 | 439 |
| ≥ 0.9 | 0.9999 | 0.9759 | 0.9877 | 0.0043% | 3 | 504 |

Extension rule simulation (block >0.8, warn 0.6–0.8): blocked 20449 TP / 8 FP · warned 69 TP / 8 FP · detected (≥0.6) 20518 TP / 16 FP
