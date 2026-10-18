# External test — post-hoc analyses

Generated: 2026-10-12T17:37:16+00:00

> **Post hoc.** None of this was in `external_500/PROTOCOL.md`; it was computed after the external
> results were known and cannot change the deployment decision. Lists: `external_500/frozen/`
> (1000 legit URLs from 500 domains, 500 phishing URLs, browser variant).
> "model dasar" = the three deployed ONNX files scored through the extension's feature path;
> back-ticked rows are ablation arms (models fitted on the 324,126-URL fit set).

## 1. Threshold-free metrics on the external lists

| Model | Family | ROC-AUC | PR-AUC | Best recall at FPR ≤ 2% | Best recall at FPR ≤ 1% |
|---|---|---:|---:|---:|---:|
| model dasar | Random Forest | 0.495 | 0.345 | 2.4% | 1.6% |
| model dasar | XGBoost | 0.536 | 0.375 | 3.0% | 2.0% |
| model dasar | CatBoost | 0.553 | 0.362 | 2.4% | 1.2% |
| `all` | Random Forest | 0.526 | 0.353 | 3.4% | 3.0% |
| `all` | XGBoost | 0.588 | 0.408 | 6.0% | 3.2% |
| `all` | CatBoost | 0.555 | 0.365 | 3.2% | 1.4% |
| `no_scheme` | Random Forest | 0.496 | 0.341 | 2.0% | 1.8% |
| `no_scheme` | XGBoost | 0.611 | 0.411 | 3.6% | 1.8% |
| `no_scheme` | CatBoost | 0.535 | 0.365 | 4.4% | 3.0% |
| `no_scheme_www` | Random Forest | 0.654 | 0.480 | 7.8% | 4.6% |
| `no_scheme_www` | XGBoost | 0.759 | 0.580 | 8.0% | 4.0% |
| `no_scheme_www` | CatBoost | 0.706 | 0.516 | 6.6% | 1.8% |
| `char` | Random Forest | 0.612 | 0.388 | 4.0% | 3.2% |
| `char` | XGBoost | 0.766 | 0.554 | 4.8% | 2.8% |
| `char` | CatBoost | 0.792 | 0.586 | 4.6% | 1.8% |
| `char_no_scheme_www` | Random Forest | 0.713 | 0.515 | 2.6% | 2.4% |
| `char_no_scheme_www` | XGBoost | 0.759 | 0.573 | 9.6% | 2.6% |
| `char_no_scheme_www` | CatBoost | 0.776 | 0.610 | 8.0% | 4.4% |

"Best recall" uses the threshold that an oracle would pick on this very set, so it is an upper bound.
A classifier that ranks at random has ROC-AUC 0.5 and PR-AUC 0.333 (the phishing share of the lists).

## 2. Uncertainty of the headline rates (model dasar, detected = score > 60%)

| Family | Detected FPR | Wilson 95% | Domain-cluster bootstrap 95% | Detected recall | Wilson 95% |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.5% (5/1000) | 0.2% – 1.2% | 0.1% – 1.0% | 1.0% (5/500) | 0.4% – 2.3% |
| XGBoost | 25.1% (251/1000) | 22.5% – 27.9% | 21.8% – 28.2% | 30.6% (153/500) | 26.7% – 34.8% |
| CatBoost | 34.3% (343/1000) | 31.4% – 37.3% | 30.8% – 37.6% | 30.2% (151/500) | 26.3% – 34.4% |

The bootstrap resamples whole domains (2000 replicates, seed 42) because each domain contributes two URLs.

## 3. Residual shortcuts: TLD and URL shape

Scheme and a leading `www.` are stripped before measuring, so this is what remains after the two
known shortcuts are removed.

| URL set | n | Host ends in `.com` | Median URL length | Homepage only (no path) |
|---|---:|---:|---:|---:|
| Training, legitimate | 276590 | 76.8% | 41 | 13.8% |
| Training, phishing | 83550 | 48.8% | 42 | 7.6% |
| External, legitimate | 1000 | 52.8% | 19.5 | 43.7% |
| External, phishing | 500 | 29.0% | 30 | 58.6% |

External legitimate URLs flagged (detected), by TLD and by whether the host starts with `www.`:

| Model | Family | `.com` | other TLD | `.com`, www | `.com`, no www | other, www | other, no www |
|---|---|---:|---:|---:|---:|---:|---:|
| model dasar | Random Forest | 0.0% (0/528) | 1.1% (5/472) | 0.0% (0/262) | 0.0% (0/266) | 0.0% (0/207) | 1.9% (5/265) |
| model dasar | XGBoost | 15.0% (79/528) | 36.4% (172/472) | 2.3% (6/262) | 27.4% (73/266) | 4.8% (10/207) | 61.1% (162/265) |
| model dasar | CatBoost | 23.7% (125/528) | 46.2% (218/472) | 0.0% (0/262) | 47.0% (125/266) | 0.0% (0/207) | 82.3% (218/265) |
| `no_scheme_www` | Random Forest | 0.4% (2/528) | 0.6% (3/472) | 0.0% (0/262) | 0.8% (2/266) | 0.0% (0/207) | 1.1% (3/265) |
| `no_scheme_www` | XGBoost | 9.7% (51/528) | 27.3% (129/472) | 9.2% (24/262) | 10.2% (27/266) | 20.3% (42/207) | 32.8% (87/265) |
| `no_scheme_www` | CatBoost | 5.1% (27/528) | 18.6% (88/472) | 4.2% (11/262) | 6.0% (16/266) | 12.6% (26/207) | 23.4% (62/265) |

## 4. Deployed models on the common test set, extension feature path

ONNX files and exported vocabularies exactly as shipped (empty-string token filtered, as in
`utils.js`). "At 0.5" is the model's own label; "> 60%" is the extension's detection threshold.

| Family | Operating point | Accuracy | Precision | Recall | FPR | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | at 0.5 | 0.9900 | 0.9999 | 0.9570 | 0.0029% | 2 | 899 |
| Random Forest | > 60% | 0.9708 | 1.0000 | 0.8741 | 0.0000% | 0 | 2629 |
| XGBoost | at 0.5 | 0.9947 | 0.9953 | 0.9820 | 0.1388% | 96 | 377 |
| XGBoost | > 60% | 0.9945 | 0.9962 | 0.9802 | 0.1128% | 78 | 414 |
| CatBoost | at 0.5 | 0.9959 | 0.9990 | 0.9833 | 0.0304% | 21 | 348 |
| CatBoost | > 60% | 0.9957 | 0.9993 | 0.9822 | 0.0202% | 14 | 371 |
