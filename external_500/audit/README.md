# Independent audit — external 500/500

Re-verification of the external evaluation, done on 2026-10-10 under the role split in
`external_500/PROTOCOL.md` (the harness builds and scores; these audit scripts re-score).
The scripts were written separately from the harness, do not import it, and are committed
unchanged from the audit run so the check is kept as repo evidence.

## Files

- `audit_lists.py <repo_root>` — read-only compliance check of the frozen lists against
  `PROTOCOL.md`: band sizes, seeded 50/50 landing/deep, one headline URL per domain,
  uniqueness, 60-day recency, hosting-suffix caps, whitelist / PhishTank / suffix
  exclusions, scheme and liveness counts, training overlap.
- `audit_rescore.py <repo_root> <scratch_dir> [js]` — independent re-score of every frozen
  URL: per-URL ONNX calls for v3.4 and direct pipeline calls for the 21 ablation models;
  compares every count with `external_metrics.json`. With `js` it also checks the Python
  feature emulation against the real `utils.js` (needs node; uses `js_features.mjs`).
- `js_features.mjs` — exports features from the extension's `utils.js` for the `js` mode.

## Audit result

- All **303 compared counts match** `external_500/external_metrics.json`; the deployment
  outcome (keep v3.4 in all three families) is confirmed.
- List compliance: 5 bands × 100 domains, 50/50 landing/deep, one headline URL per domain;
  500 unique phishing hosts, hosting cap respected, all submitted within 60 days, zero
  exact training-URL matches, 2 hosts seen in training.
- The `utils.js` feature emulation matches the Python path to float precision.
- Findings that changed the report wording (all folded into `external_report.md`):
  recall range for the https-upgrade assumption (98/500 listed http, 1 answered);
  the legit FPR is driven by the missing `www.` (XGB 3.4% vs 44.3%; CB 0.0% vs 64.6%);
  the as-listed slash caveat (117 bare-host listings; CatBoost `no_scheme_www` 41→14 on
  the 49 slash-only rows); plus construction details (co.uk / com.au / co.id removed by
  last-two-label matching; 2 whitelisted-host deep rows; 24 deep non-200s; 6 deep
  redirects off the landing site; 69 legit landing hosts also present in training).

## Reproduce

```
python external_500/audit/audit_lists.py <repo_root>
python external_500/audit/audit_rescore.py <repo_root> <scratch_dir> [js]
```
