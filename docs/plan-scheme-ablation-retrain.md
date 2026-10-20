# Plan — scheme / `www.` ablation, retrain, and larger external validation

> Source: **independent review, 2026-10-10**. Branch: `feat/scheme-ablation-retrain`.
> Baseline to preserve: **extension v3.4** (RF full / XGB full 177→149 / CatBoost full @ 300 iters).
> Never overwrite v3.4 artifacts — always pass explicit `--out` / `--outdir` paths.

## Why

Bab 4 shows the deployed models exploit dataset shortcuts (`http:`/`https:` tokens dominate
feature importance) and collapse on live data (25/25: RF 0%, XGBoost 28%, CatBoost 24% recall).
This work tests the shortcut hypothesis directly and produces a fairer feature comparison.

## Before the retrain — decisions that change the training run

1. **Measure the `www.` shortcut first** — add it to `analyze_dataset_split.py`; no training
   needed. Review figure to verify: 99.79% of legitimate URLs start with scheme + `www.`
   vs 13.04% of phishing — which would make `www.` as strong a shortcut as the scheme.
2. **Strip the scheme from the string, not just the tokens** — URL length, slash count and
   entropy also see `https://`; dropping only the `http:`/`https:` tokens leaves a small leak.
   The same change must go into `phishingdetectorExt/utils.js`.
3. **Run as an ablation, not a single retrain** — arms:
   (a) all features, (b) no scheme, (c) no scheme + no `www.`, (d) TF-IDF only,
   (e) structural only. This answers Rumusan Masalah 2 directly (no reword needed).
4. **Hold out a validation split from the training pool** — choose the variant and CatBoost
   iterations on it; touch the shared 90,036-URL test set once, at the end.
5. **Decide on the empty-token fix now** — filtering empty tokens in the training tokenizer
   removes the PKL-vs-ONNX gap at the source, but it changes the vocabulary;
   `tests/test_features_parity.py` and `utils.js` must change with it.
6. Keep **v3.4 as the baseline**; explicit `--out` paths so nothing gets overwritten.

## After the retrain

- Re-export & deploy: ONNX + JSON files, `manifest.json` version, `MODELS.md` hashes,
  `bench.js` hash list, then both test suites (`node --test`, `tests/test_features_parity.py`).
- Re-run every report: sampling, thresholds, feature importance, latency, memory, and
  CatBoost through the extension's feature path (**never measured so far** — the Python
  memory/latency benchmarks used the raw pkl path, not the JS-equivalent pipeline).
  The in-browser P-01 benchmark is manual and needs the user.
- **500/500 external test**, scored on v3.4 and the new models with the same frozen list:
  - sample Tranco across rank bands (not just the top 500);
  - include non-`www.` hosts and deep paths, not only homepages;
  - verified online PhishTank entries, deduplicated by host, fetch date recorded;
  - report host overlap with the training data;
  - commit the frozen list to the repo.

## Execution order

1. Measure `www.` first (no training).
2. Implement items 2–5 as flags in `features.py`, `sampling.py`, `benchmark_sampling.py`;
   smoke-test with `--sample-rows`.
3. The full run takes ~8+ minutes and writes artifacts — **ask before launching**.

## Coordination

- The thesis docx is out of scope here (its own tracked-changes workflow).
- **One writer per repo file at a time** — record who owns which file in the
  project hand-off notes when switching.
