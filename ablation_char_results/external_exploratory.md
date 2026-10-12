# Character n-gram arms on the frozen external lists — EXPLORATORY

Generated: 2026-10-12T15:08:06+00:00

> Not part of the pre-registered external test. The frozen lists were collected and scored before these
> models existed and their results were known when the character block was designed. Nothing here can
> qualify a model for deployment; a candidate that looks good needs a fresh sample.

Lists: `external_500/frozen/` — 1000 legit (469 www / 531 non-www host), 500 phishing, 500 prefix-sensitivity domains. Thresholds: detected > 60%, blocked > 80%.
The `all` and `no_scheme_www` rows are the 10 Oct ablation models, re-scored here as a check.

## Legit false positives and phishing recall

| Arm | Model | Detected FPR | Blocked FPR | FP on www hosts | FP on non-www hosts | Recall, browser | Recall, as listed |
|---|---|---:|---:|---:|---:|---:|---:|
| `all` | Random Forest | 2.0% (20/1000) | 0.0% (0/1000) | 0.0% (0/469) | 3.8% (20/531) | 3.4% (17/500) | 22.6% (113/500) |
| `all` | XGBoost | 30.4% (304/1000) | 27.7% (277/1000) | 2.1% (10/469) | 55.4% (294/531) | 42.4% (212/500) | 58.4% (292/500) |
| `all` | CatBoost | 34.2% (342/1000) | 33.5% (335/1000) | 0.0% (0/469) | 64.4% (342/531) | 29.4% (147/500) | 49.2% (246/500) |
| `no_scheme_www` | Random Forest | 0.5% (5/1000) | 0.0% (0/1000) | 0.0% (0/469) | 0.9% (5/531) | 1.4% (7/500) | 1.4% (7/500) |
| `no_scheme_www` | XGBoost | 18.0% (180/1000) | 11.1% (111/1000) | 14.1% (66/469) | 21.5% (114/531) | 56.8% (284/500) | 58.6% (293/500) |
| `no_scheme_www` | CatBoost | 11.5% (115/1000) | 5.5% (55/1000) | 7.9% (37/469) | 14.7% (78/531) | 32.2% (161/500) | 45.4% (227/500) |
| `char` | Random Forest | 51.1% (511/1000) | 0.8% (8/1000) | 0.0% (0/469) | 96.2% (511/531) | 86.8% (434/500) | 88.4% (442/500) |
| `char` | XGBoost | 40.5% (405/1000) | 36.8% (368/1000) | 0.9% (4/469) | 75.5% (401/531) | 81.2% (406/500) | 85.4% (427/500) |
| `char` | CatBoost | 39.5% (395/1000) | 26.1% (261/1000) | 0.0% (0/469) | 74.4% (395/531) | 86.2% (431/500) | 90.4% (452/500) |
| `char_no_scheme_www` | Random Forest | 0.6% (6/1000) | 0.0% (0/1000) | 0.2% (1/469) | 0.9% (5/531) | 2.2% (11/500) | 2.2% (11/500) |
| `char_no_scheme_www` | XGBoost | 21.3% (213/1000) | 13.3% (133/1000) | 15.1% (71/469) | 26.7% (142/531) | 56.0% (280/500) | 54.6% (273/500) |
| `char_no_scheme_www` | CatBoost | 16.2% (162/1000) | 7.6% (76/1000) | 10.2% (48/469) | 21.5% (114/531) | 54.8% (274/500) | 60.8% (304/500) |
| `char_only` | Random Forest | 53.0% (530/1000) | 50.7% (507/1000) | 0.0% (0/469) | 99.8% (530/531) | 94.0% (470/500) | 94.0% (470/500) |
| `char_only` | XGBoost | 51.5% (515/1000) | 48.4% (484/1000) | 2.6% (12/469) | 94.7% (503/531) | 92.2% (461/500) | 91.8% (459/500) |
| `char_only` | CatBoost | 54.0% (540/1000) | 52.2% (522/1000) | 2.1% (10/469) | 99.8% (530/531) | 94.4% (472/500) | 94.4% (472/500) |
| `char_only_no_scheme_www` | Random Forest | 0.4% (4/1000) | 0.1% (1/1000) | 0.0% (0/469) | 0.8% (4/531) | 2.2% (11/500) | 2.6% (13/500) |
| `char_only_no_scheme_www` | XGBoost | 21.3% (213/1000) | 11.8% (118/1000) | 16.6% (78/469) | 25.4% (135/531) | 58.8% (294/500) | 58.4% (292/500) |
| `char_only_no_scheme_www` | CatBoost | 19.0% (190/1000) | 8.7% (87/1000) | 14.7% (69/469) | 22.8% (121/531) | 58.0% (290/500) | 60.0% (300/500) |

## Prefix sensitivity (domains detected out of 500)

| Arm | Model | https://domain/ | https://www.domain/ | http://domain/ |
|---|---|---:|---:|---:|
| `all` | Random Forest | 4 | 0 | 500 |
| `all` | XGBoost | 181 | 7 | 500 |
| `all` | CatBoost | 178 | 1 | 500 |
| `no_scheme_www` | Random Forest | 0 | 0 | 0 |
| `no_scheme_www` | XGBoost | 61 | 61 | 61 |
| `no_scheme_www` | CatBoost | 40 | 40 | 40 |
| `char` | Random Forest | 491 | 0 | 499 |
| `char` | XGBoost | 278 | 3 | 494 |
| `char` | CatBoost | 198 | 0 | 500 |
| `char_no_scheme_www` | Random Forest | 0 | 0 | 0 |
| `char_no_scheme_www` | XGBoost | 80 | 80 | 80 |
| `char_no_scheme_www` | CatBoost | 62 | 62 | 62 |
| `char_only` | Random Forest | 499 | 0 | 499 |
| `char_only` | XGBoost | 457 | 10 | 458 |
| `char_only` | CatBoost | 491 | 2 | 491 |
| `char_only_no_scheme_www` | Random Forest | 0 | 0 | 0 |
| `char_only_no_scheme_www` | XGBoost | 83 | 83 | 83 |
| `char_only_no_scheme_www` | CatBoost | 82 | 82 | 82 |
