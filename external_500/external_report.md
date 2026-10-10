# External 500/500 evaluation — live Tranco legit + PhishTank phishing

Generated: 2026-10-10T14:36:01+00:00 — harness commit `8a63af0`, extension v3.4
Protocol: `external_500/PROTOCOL.md` (frozen before collection; amendment log inside).
Lists (frozen, hashed): `legit_urls.csv` 1000 rows sha256 `8a0829d0b08c8f29…`; `phishing_urls.csv` 500 rows sha256 `af5f9f8b3721c9b3…`; `prefix_sensitivity.csv` 500 rows sha256 `ea0c5636a1584ef0…`
ONNX models: RF `625c9333151667c4…`, XGB `4efd7ead699770a0…`, CatBoost `791a503e72426426…`
Thresholds (extension, strict): detected >60, blocked >80 (percent).

- Legit: 1000 URLs (500 headline), phishing: 500 URLs (15 live at resolution, 2 hosts seen in training, 64 free-hosting).

## Legit — headline 500 (browser URLs; detected >60, blocked >80)

| Model | detected FP | detected FPR | blocked FP | blocked FPR |
|---|---:|---:|---:|---:|
| v3.4 Random Forest | 2/500 | 0.4% | 0/500 | 0.0% |
| v3.4 XGBoost | 129/500 | 25.8% | 100/500 | 20.0% |
| v3.4 CatBoost | 172/500 | 34.4% | 169/500 | 33.8% |
| `all` Random Forest | 9/500 | 1.8% | 0/500 | 0.0% |
| `all` XGBoost | 158/500 | 31.6% | 142/500 | 28.4% |
| `all` CatBoost | 172/500 | 34.4% | 169/500 | 33.8% |
| `no_scheme` Random Forest | 126/500 | 25.2% | 0/500 | 0.0% |
| `no_scheme` XGBoost | 248/500 | 49.6% | 209/500 | 41.8% |
| `no_scheme` CatBoost | 247/500 | 49.4% | 244/500 | 48.8% |
| `no_scheme_www` Random Forest | 3/500 | 0.6% | 0/500 | 0.0% |
| `no_scheme_www` XGBoost | 97/500 | 19.4% | 54/500 | 10.8% |
| `no_scheme_www` CatBoost | 60/500 | 12.0% | 28/500 | 5.6% |
| `tfidf_only` Random Forest | 0/500 | 0.0% | 0/500 | 0.0% |
| `tfidf_only` XGBoost | 17/500 | 3.4% | 15/500 | 3.0% |
| `tfidf_only` CatBoost | 7/500 | 1.4% | 3/500 | 0.6% |
| `struct_only` Random Forest | 236/500 | 47.2% | 206/500 | 41.2% |
| `struct_only` XGBoost | 241/500 | 48.2% | 201/500 | 40.2% |
| `struct_only` CatBoost | 239/500 | 47.8% | 217/500 | 43.4% |
| `tfidf_only_no_scheme_www` Random Forest | 3/500 | 0.6% | 0/500 | 0.0% |
| `tfidf_only_no_scheme_www` XGBoost | 113/500 | 22.6% | 64/500 | 12.8% |
| `tfidf_only_no_scheme_www` CatBoost | 57/500 | 11.4% | 29/500 | 5.8% |
| `struct_only_no_scheme_www` Random Forest | 20/500 | 4.0% | 3/500 | 0.6% |
| `struct_only_no_scheme_www` XGBoost | 25/500 | 5.0% | 8/500 | 1.6% |
| `struct_only_no_scheme_www` CatBoost | 28/500 | 5.6% | 8/500 | 1.6% |

## Legit — all 1.000 rows (protocol rule inputs; baseline + candidates)

| Model | detected FPR (all) | blocked FPR (all) | detected FPR landing | detected FPR deep | worst band (detected FPR) |
|---|---:|---:|---:|---:|---:|
| v3.4 Random Forest | 0.5% (5/1000) | 0.0% | 0.2% | 0.8% | 500k-1M 2.0% |
| v3.4 XGBoost | 25.1% (251/1000) | 19.9% | 15.4% | 34.8% | 500k-1M 34.5% |
| v3.4 CatBoost | 34.3% (343/1000) | 33.5% | 22.0% | 46.6% | 500k-1M 50.5% |
| `no_scheme` Random Forest | 24.6% (246/1000) | 0.0% | 21.8% | 27.4% | 500k-1M 40.5% |
| `no_scheme` XGBoost | 48.9% (489/1000) | 40.6% | 50.4% | 47.4% | 500k-1M 67.5% |
| `no_scheme` CatBoost | 49.5% (495/1000) | 48.3% | 49.8% | 49.2% | 500k-1M 69.5% |
| `no_scheme_www` Random Forest | 0.5% (5/1000) | 0.0% | 0.2% | 0.8% | 500k-1M 1.0% |
| `no_scheme_www` XGBoost | 18.0% (180/1000) | 11.1% | 13.0% | 23.0% | 500k-1M 27.0% |
| `no_scheme_www` CatBoost | 11.5% (115/1000) | 5.5% | 9.0% | 14.0% | 500k-1M 16.0% |

## Phishing — browser variant, 500 (detected recall >60, blocked >80)

| Model | detected TP | detected recall | blocked TP | blocked recall |
|---|---:|---:|---:|---:|
| v3.4 Random Forest | 5/500 | 1.0% | 0/500 | 0.0% |
| v3.4 XGBoost | 153/500 | 30.6% | 124/500 | 24.8% |
| v3.4 CatBoost | 151/500 | 30.2% | 139/500 | 27.8% |
| `all` Random Forest | 17/500 | 3.4% | 1/500 | 0.2% |
| `all` XGBoost | 212/500 | 42.4% | 174/500 | 34.8% |
| `all` CatBoost | 147/500 | 29.4% | 142/500 | 28.4% |
| `no_scheme` Random Forest | 133/500 | 26.6% | 1/500 | 0.2% |
| `no_scheme` XGBoost | 312/500 | 62.4% | 282/500 | 56.4% |
| `no_scheme` CatBoost | 233/500 | 46.6% | 200/500 | 40.0% |
| `no_scheme_www` Random Forest | 7/500 | 1.4% | 1/500 | 0.2% |
| `no_scheme_www` XGBoost | 284/500 | 56.8% | 211/500 | 42.2% |
| `no_scheme_www` CatBoost | 161/500 | 32.2% | 84/500 | 16.8% |
| `tfidf_only` Random Forest | 0/500 | 0.0% | 0/500 | 0.0% |
| `tfidf_only` XGBoost | 91/500 | 18.2% | 55/500 | 11.0% |
| `tfidf_only` CatBoost | 22/500 | 4.4% | 12/500 | 2.4% |
| `struct_only` Random Forest | 184/500 | 36.8% | 169/500 | 33.8% |
| `struct_only` XGBoost | 195/500 | 39.0% | 179/500 | 35.8% |
| `struct_only` CatBoost | 186/500 | 37.2% | 167/500 | 33.4% |
| `tfidf_only_no_scheme_www` Random Forest | 4/500 | 0.8% | 0/500 | 0.0% |
| `tfidf_only_no_scheme_www` XGBoost | 226/500 | 45.2% | 182/500 | 36.4% |
| `tfidf_only_no_scheme_www` CatBoost | 156/500 | 31.2% | 45/500 | 9.0% |
| `struct_only_no_scheme_www` Random Forest | 111/500 | 22.2% | 29/500 | 5.8% |
| `struct_only_no_scheme_www` XGBoost | 139/500 | 27.8% | 68/500 | 13.6% |
| `struct_only_no_scheme_www` CatBoost | 121/500 | 24.2% | 42/500 | 8.4% |

## Phishing — variants and subsets (baseline + candidates)

| Model | as-listed recall | live-only recall | seen-host recall | unseen-host recall | free-hosting recall | recall gain vs v3.4 |
|---|---:|---:|---:|---:|---:|---:|
| v3.4 Random Forest | 14.0% | 0.0% (n=15) | 0.0% (n=2) | 1.004% (n=498) | 0.0% (n=64) | — |
| v3.4 XGBoost | 46.0% | 46.667% (n=15) | 50.0% (n=2) | 30.522% (n=498) | 50.0% (n=64) | — |
| v3.4 CatBoost | 49.2% | 40.0% (n=15) | 50.0% (n=2) | 30.12% (n=498) | 14.062% (n=64) | — |
| `no_scheme` Random Forest | 27.4% | 20.0% (n=15) | 0.0% (n=2) | 26.707% (n=498) | 0.0% (n=64) | +25.60 pp |
| `no_scheme` XGBoost | 65.2% | 60.0% (n=15) | 50.0% (n=2) | 62.45% (n=498) | 65.625% (n=64) | +31.80 pp |
| `no_scheme` CatBoost | 59.0% | 40.0% (n=15) | 50.0% (n=2) | 46.586% (n=498) | 10.938% (n=64) | +16.40 pp |
| `no_scheme_www` Random Forest | 1.4% | 0.0% (n=15) | 0.0% (n=2) | 1.406% (n=498) | 0.0% (n=64) | +0.40 pp |
| `no_scheme_www` XGBoost | 58.6% | 60.0% (n=15) | 50.0% (n=2) | 56.827% (n=498) | 60.938% (n=64) | +26.20 pp |
| `no_scheme_www` CatBoost | 45.4% | 33.333% (n=15) | 50.0% (n=2) | 32.129% (n=498) | 25.0% (n=64) | +2.00 pp |

## McNemar — candidates vs v3.4 of the same family (browser variant, detected >60)

| Family | Candidate | v3.4-only correct (c) | candidate-only correct (b) | p (detected) | p (blocked) | recall gain |
|---|---|---:|---:|---:|---:|---:|
| Random Forest | `no_scheme` | 0 | 128 | 0.0 | 1.0 | +25.60 pp |
| Random Forest | `no_scheme_www` | 2 | 4 | 0.6875 | 1.0 | +0.40 pp |
| XGBoost | `no_scheme` | 2 | 161 | 0.0 | 0.0 | +31.80 pp |
| XGBoost | `no_scheme_www` | 32 | 163 | 0.0 | 0.0 | +26.20 pp |
| CatBoost | `no_scheme` | 6 | 88 | 0.0 | 0.0 | +16.40 pp |
| CatBoost | `no_scheme_www` | 89 | 99 | 0.51168 | 1.4e-05 | +2.00 pp |

## Prefix sensitivity — synthetic `https://d/` vs `https://www.d/` vs `http://d/` (headline domains)

| Model | detected https | detected www | detected http | flips (any of 3) | flips https vs http |
|---|---:|---:|---:|---:|---:|
| v3.4 Random Forest | 1/500 | 0/500 | 499/500 | 499 | 498 |
| v3.4 XGBoost | 134/500 | 6/500 | 500/500 | 494 | 366 |
| v3.4 CatBoost | 179/500 | 2/500 | 500/500 | 498 | 321 |
| `no_scheme` Random Forest | 190/500 | 0/500 | 190/500 | 190 | 0 |
| `no_scheme` XGBoost | 456/500 | 31/500 | 456/500 | 425 | 0 |
| `no_scheme` CatBoost | 491/500 | 9/500 | 491/500 | 482 | 0 |
| `no_scheme_www` Random Forest | 0/500 | 0/500 | 0/500 | 0 | 0 |
| `no_scheme_www` XGBoost | 61/500 | 61/500 | 61/500 | 0 | 0 |
| `no_scheme_www` CatBoost | 40/500 | 40/500 | 40/500 | 0 | 0 |

## Deployment rule (protocol §c) — evaluation

### Random Forest

Baseline v3.4: detected FPR 0.5% | blocked FPR 0.0% | detected recall 1.0%.

| Candidate | c1 FPR ≤2% & ≤v3.4 | c2 blocked ≤1% | c3 +≥10pp & p<0.05 | pass |
|---|---|---|---|---|
| `no_scheme` | no (24.6%) | yes (0.0%) | yes (+25.60 pp, p=0.0) | fail |
| `no_scheme_www` | yes (0.5%) | yes (0.0%) | no (+0.40 pp, p=0.6875) | fail |

**Decision: keep v3.4**

### XGBoost

Baseline v3.4: detected FPR 25.1% | blocked FPR 19.9% | detected recall 30.6%.

| Candidate | c1 FPR ≤2% & ≤v3.4 | c2 blocked ≤1% | c3 +≥10pp & p<0.05 | pass |
|---|---|---|---|---|
| `no_scheme` | no (48.9%) | no (40.6%) | yes (+31.80 pp, p=0.0) | fail |
| `no_scheme_www` | no (18.0%) | no (11.1%) | yes (+26.20 pp, p=0.0) | fail |

**Decision: keep v3.4**

### CatBoost

Baseline v3.4: detected FPR 34.3% | blocked FPR 33.5% | detected recall 30.2%.

| Candidate | c1 FPR ≤2% & ≤v3.4 | c2 blocked ≤1% | c3 +≥10pp & p<0.05 | pass |
|---|---|---|---|---|
| `no_scheme` | no (49.5%) | no (48.3%) | yes (+16.40 pp, p=0.0) | fail |
| `no_scheme_www` | no (11.5%) | no (5.5%) | no (+2.00 pp, p=0.51168) | fail |

**Decision: keep v3.4**


## Context — in-distribution vs external (all arms)

| Arm | Family | validation recall | test recall | external recall | validation FPR | test FPR | external FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| `all` | Random Forest | 96.43% | 96.39% | 3.40% | 0.011% | 0.022% | 2.000% |
| `all` | XGBoost | 98.68% | 98.41% | 42.40% | 0.177% | 0.158% | 30.400% |
| `all` | CatBoost | 98.47% | 98.31% | 29.40% | 0.043% | 0.026% | 34.200% |
| `no_scheme` | Random Forest | 57.83% | 57.74% | 26.60% | 0.069% | 0.082% | 24.600% |
| `no_scheme` | XGBoost | 91.72% | 92.09% | 62.40% | 0.828% | 0.771% | 48.900% |
| `no_scheme` | CatBoost | 86.61% | 87.36% | 46.60% | 0.466% | 0.418% | 49.500% |
| `no_scheme_www` | Random Forest | 32.38% | 31.72% | 1.40% | 0.177% | 0.123% | 0.500% |
| `no_scheme_www` | XGBoost | 84.01% | 84.06% | 56.80% | 2.271% | 2.033% | 18.000% |
| `no_scheme_www` | CatBoost | 75.43% | 75.47% | 32.20% | 1.620% | 1.513% | 11.500% |
| `tfidf_only` | Random Forest | 94.85% | 95.00% | 0.00% | 0.004% | 0.000% | 0.000% |
| `tfidf_only` | XGBoost | 97.64% | 97.55% | 18.20% | 0.199% | 0.153% | 3.500% |
| `tfidf_only` | CatBoost | 97.28% | 97.01% | 4.40% | 0.025% | 0.036% | 1.300% |
| `struct_only` | Random Forest | 71.14% | 72.19% | 36.80% | 0.568% | 0.607% | 47.500% |
| `struct_only` | XGBoost | 74.69% | 75.54% | 39.00% | 1.370% | 1.390% | 48.100% |
| `struct_only` | CatBoost | 73.37% | 74.55% | 37.20% | 1.164% | 1.053% | 48.000% |
| `tfidf_only_no_scheme_www` | Random Forest | 25.94% | 25.42% | 0.80% | 0.119% | 0.106% | 0.400% |
| `tfidf_only_no_scheme_www` | XGBoost | 72.17% | 71.95% | 45.20% | 2.195% | 2.103% | 22.300% |
| `tfidf_only_no_scheme_www` | CatBoost | 64.23% | 64.32% | 31.20% | 1.580% | 1.466% | 11.100% |
| `struct_only_no_scheme_www` | Random Forest | 50.63% | 50.79% | 22.20% | 3.189% | 2.884% | 3.100% |
| `struct_only_no_scheme_www` | XGBoost | 57.45% | 56.76% | 27.80% | 4.700% | 4.428% | 5.000% |
| `struct_only_no_scheme_www` | CatBoost | 53.97% | 53.55% | 24.20% | 4.317% | 3.947% | 4.800% |

## Notes

- v3.4 rows score the deployed ONNX models through the extension feature path (`features.extract_features_onnx` / `preprocess_xgb`); arm rows score each cached `.pkl` through `benchmark_sampling.make_predictors`. Baseline and candidates are therefore both scored the way they would run in production/analysis.
- Cross-check: the v3.4 browser numbers reproduce the 25/25 real-world test pattern (there: RF 0/25 recall, 0 FP; XGB 28.0% recall / 16.0% FPR; CatBoost 24.0% / 20.0%) at 20x the sample size, and the batched scorer was re-verified against per-URL `run_realworld_test.evaluate` scoring with an exact count match on the same 1.000 URL variants.
- Collection environment: Bitdefender real-time protection was active during collection; individual fetches may have been blocked by its web filtering (recorded as drops or non-answering entries, never silently dropped). Phishing liveness (33/500 answered, 15 live) is therefore a lower bound; scoring is URL-only, so headline metrics do not depend on liveness.
- Phishing entries that no longer resolve are still scored (URL-only analysis); the live-only row reports the subset that answered when resolved.
- The `all` arm is trained on the fit set (90% of the train pool), so it is close to but not identical with v3.4 (full train pool).
- Training host overlap: 2 of 500 phishing hosts also appear in the training dataset.

