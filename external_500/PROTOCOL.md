# External 500/500 evaluation — protocol (frozen before any list is fetched)

> Agreed on 2026-10-10. **This file is committed
> before any list is downloaded.** The lists are frozen and committed before scoring;
> there is no re-sampling and no retraining after scores are seen (rule 6).

## Goal

Measure how the deployed v3.4 models and the scheme/`www.` ablation candidates behave on
*external* data — contemporary phishing URLs and Tranco-ranked legitimate URLs — to answer:
**do the de-shortcut models generalize better on live data than the shortcut-driven v3.4?**

## Ownership

The harness and its verification are kept apart: the scripts that build and score the lists
are not the scripts that re-check them.

| Area | Role |
|---|---|
| `external_500/` harness: list builders, frozen lists, scoring script, report | harness |
| Re-score of the frozen lists and count check (`external_500/audit/`) | audit scripts, written separately from the harness |
| `realworld_test/` + `run_realworld_test.py` (v3.4 25/25 evidence) | untouched |
| `features.py`, `ml_pipelines.py`, `sampling.py`, `benchmark_ablation.py`, `tests/` | model code, frozen for this test |
| `utils.js` + export scripts — only if a candidate passes the deployment rule | model code |

## (a) Legitimate set — Tranco

- **Snapshot:** the default Tranco list as downloaded on the build day. Record its list ID,
  permanent URL, download date and the SHA-256 of the CSV in `external_500/frozen/PROVENANCE.md`.
- **Bands:** five rank bands of 100 domains each, seed 42:
  1–1k · 1k–10k · 10k–100k · 100k–500k · 500k–1M.
- **Eligibility** (draw ≈300 per band in seeded order, keep the first 100 that pass,
  recording how many were dropped per rule):
  1. answers over `https` with an HTML page;
  2. not in the extension's static whitelist (`WHITELIST` in `phishingdetectorExt/utils.js`,
     12 domains: google.com, youtube.com, facebook.com, twitter.com, github.com,
     stackoverflow.com, microsoft.com, apple.com, amazon.com, wikipedia.org, linkedin.com,
     instagram.com);
  3. registrable domain not in the PhishTank feed.
- **Two real URLs per domain, no synthetic ones:**
  - *Landing*: the final URL after redirects from `https://<domain>/` (the site decides `www.` or not);
  - *Deep*: one seeded same-site link from the landing page with a non-empty path and **not a
    static asset** (links ending in css/js/images/fonts/media/archives are skipped — the intent
    is a navigable page).
- **Headline legit set:** 500 URLs, one per domain, seeded 50/50 landing/deep within each band.
- **FPR** is also reported on **all ~1.000** legit URLs, split by landing/deep and by band.
- **Prefix sensitivity** (separate table, not part of FPR): for every domain score
  `https://<d>/`, `https://www.<d>/` and `http://<d>/`; report how often the verdict flips.
  This is the direct test of the shortcut (mirrors the scheme-sensitivity table in the 25/25 report).

## (b) Phishing set — PhishTank

Feed: `online-valid`; record fetch timestamp, SHA-256 and row count. Method identical to the
25/25 collection (https upgrade + curl-resolved redirect = *browser* variant, primary;
*as-listed* = reference; HTTP code recorded), with four tightenings:

1. **Recency:** only entries submitted in the last 60 days; widen only if fewer than ≈1.500
   unique hosts remain (the 25/25 sample contained entries from 2024, which can predate the
   training dataset).
2. **Training overlap:** drop exact URL matches with `URL dataset.csv`. Keep host matches,
   but report recall separately for hosts **seen** and **not seen** in training.
3. **Dedup and cap:** one URL per hostname, and at most 50 of the 500 from any single hosting
   suffix (e.g. `pages.dev`) so one platform cannot dominate.
4. **Variants:** browser variant is primary; as-listed is the reference. Add a **live-only
   recall** row (entries that answered when resolved).

Headline phishing set: 500 URLs, seed 42, sampled across the qualifying hosts.

## Scoring

- **Operating points:** the extension's own thresholds, unchanged — strict `>`:
  **detected** = score > 60; **blocked** = score > 80. **No threshold tuning on this set.**
- **Models:** v3.4 (deployed RF/XGB/CatBoost ONNX) is the baseline; all seven ablation arms ×
  three families are scored and reported, but only `no_scheme` and `no_scheme_www` are
  **candidates**. TF-IDF-only and structural-only arms are analysis.
- Legit URLs: report detected-FPR and blocked-FPR (overall and by landing/deep, by band).
- Phishing URLs: report detected recall, blocked recall (browser variant; as-listed reference;
  live-only subset; training-seen vs unseen hosts).
- Report host overlap of the phishing set with the training data.

## (c) Deployment rule — fixed before any list is fetched

1. Candidates: `no_scheme` and `no_scheme_www` arms for each model family. v3.4 = baseline.
2. Operating points: as above (detected >60, blocked >80); no tuning.
3. A candidate replaces the v3.4 model **of its family** only if **all three** hold:
   - detected FPR on the ~1.000 legit URLs is **≤ 2,0%** and **no higher than the v3.4 model's**;
   - blocked FPR on the same URLs is **≤ 1,0%**;
   - detected recall on the 500 phishing URLs (browser variant) beats the v3.4 model by
     **≥ 10 percentage points**, with an **exact McNemar test at p < 0,05**.
4. Several pass in one family: highest recall wins, then lower FPR.
5. None pass: v3.4 stays deployed and the thesis reports that removing the shortcuts did not
   produce a deployable lexical model. That is a legitimate result.
6. **Single use:** the lists are frozen and committed before scoring; no re-sampling and no
   retraining after scores are seen.
7. A winner is retrained on the **full train pool** with the same switches, exported, and
   re-scored on the same frozen list. Both sets of numbers are reported.

## Reproducibility

- Seeds: Tranco draw + 50/50 landing/deep + deep-link pick = 42; PhishTank sample = 42.
- Fetching is done with `curl` (consistent with the 25/25 method), polite timeouts, and
  records every HTTP code; failures are recorded, never silently dropped.
- Frozen artifacts committed: `external_500/frozen/legit_urls.csv`,
  `external_500/frozen/phishing_urls.csv`, `external_500/frozen/PROVENANCE.md`
  (raw-input hashes), plus the scoring report.
- Raw downloads (full Tranco CSV, full PhishTank feed) are **not** committed — only their
  provenance hashes.

## Amendment log

- **2026-10-10 — deep-link definition:** added the static-asset exclusion to §(a) after smoke
  tests showed `…/assets/index-*.css` being picked as a deep link. Made **before any real list
  was fetched** (only smoke-test fetches of a handful of candidates); no scores were seen.
- **2026-10-10 — scorer hotfix after the freeze (logged 2026-10-13):** the commit that added the
  results also changed `score_external.py` in the code that evaluates the deployment rule: a
  key-access fix (`mc['p']` → `mc['detected']['p']`) for a `KeyError` on the first full run,
  plus report notes. Thresholds, metrics and the rule's logic are unchanged, and the scoring
  pass reproduced identical counts (see the note in `external_report.md`). It is recorded here
  because it happened after the lists were frozen.
- **Timestamps:** the order above (protocol, amendment, frozen lists, results) is the order of
  the local commits. The branch was first pushed to the remote on 2026-10-12, so there is no
  third-party timestamp for the individual steps.
