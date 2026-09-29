"""Threshold sensitivity of the deployed models on the common test split.

Motivation: the extension uses a block/warn rule (>80% block, 60-80% warn) that
was never justified against data. This script measures, per deployed model
(RF / XGBoost / CatBoost), the classification metrics at several decision
thresholds on the shared 90,036-URL test set, and simulates the extension rule
(block >0.8, warn 0.6-0.8) so its trade-off is documented.

Usage (repo root):
    python analyze_thresholds.py            # full test set (default)
    python analyze_thresholds.py --samples 20000

Outputs:
    sampling_results/threshold_report.md
    sampling_results/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_pipelines  # noqa: E402
import sampling  # noqa: E402

OUTDIR = 'sampling_results'
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
CHUNK = 20000


def probs_rf(model, urls):
    return model.predict_proba(urls)[:, 1]


def probs_xgb(art, urls):
    out = []
    for i in range(0, len(urls), CHUNK):
        out.append(ml_pipelines.predict_proba_with_xgb(art, list(urls[i:i + CHUNK]))[:, 1])
    return np.concatenate(out)


def probs_cb(art, urls):
    fx, model = art['feature_extractor'], art['model']
    out = []
    for i in range(0, len(urls), CHUNK):
        X = ml_pipelines._to_dense(fx.transform(urls[i:i + CHUNK]))
        out.append(model.predict_proba(X)[:, 1])
    return np.concatenate(out)


def metrics_at(y, probs, t):
    pred = probs >= t
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {'threshold': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'fpr': round(fpr, 6), 'f1': round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0}


def main():
    ap = argparse.ArgumentParser(description="Decision-threshold sensitivity on the shared test set.")
    ap.add_argument('--samples', type=int, default=0, help='cap test rows (0 = full test set)')
    args = ap.parse_args()

    df = sampling.load_clean_dataset()
    _, test = sampling.common_split(df)
    if args.samples:
        test = test.sample(n=min(args.samples, len(test)), random_state=sampling.SEED).reset_index(drop=True)
    y = test['label'].values
    urls = test['url'].values
    print(f"Test URLs: {len(urls)} ({int((y==1).sum())} phishing / {int((y==0).sum())} legitimate)")

    models = {}
    print("RF probabilities ...", flush=True)
    models['rf'] = probs_rf(joblib.load(os.path.join(OUTDIR, 'rf_full.pkl')), urls)
    print("XGB probabilities ...", flush=True)
    models['xgb'] = probs_xgb(joblib.load(os.path.join(OUTDIR, 'xgb_full.pkl')), urls)
    print("CatBoost probabilities ...", flush=True)
    models['cb'] = probs_cb(joblib.load(os.path.join(OUTDIR, 'cb_full.pkl')), urls)

    labels = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'cb': 'CatBoost'}
    result = {'n_test': int(len(urls)), 'block_threshold': 0.8, 'warn_threshold': 0.6, 'models': {}}

    lines = [
        "# Decision-Threshold Sensitivity — deployed models (shared test set)",
        "",
        f"{len(urls):,} URLs from the common test split (seed {sampling.SEED}); probabilities from the "
        "deployed full-dataset artifacts.",
        "",
    ]
    for key, probs in models.items():
        rows = [metrics_at(y, probs, t) for t in THRESHOLDS]
        blocked = probs >= 0.8
        warned = (probs >= 0.6) & (probs < 0.8)
        rule = {
            'blocked_tp': int(((y == 1) & blocked).sum()), 'blocked_fp': int(((y == 0) & blocked).sum()),
            'warned_tp': int(((y == 1) & warned).sum()), 'warned_fp': int(((y == 0) & warned).sum()),
            'detected_tp': int(((y == 1) & (probs >= 0.6)).sum()),
            'flagged_fp': int(((y == 0) & (probs >= 0.6)).sum()),
        }
        result['models'][key] = {'label': labels[key], 'by_threshold': rows, 'extension_rule': rule}

        lines += [f"## {labels[key]}", "",
                  "| Threshold | Precision | Recall | F1 | FPR | FP | FN |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| ≥ {r['threshold']:.1f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                         f"{r['f1']:.4f} | {r['fpr']*100:.4f}% | {r['fp']} | {r['fn']} |")
        lines += ["",
                  f"Extension rule simulation (block >0.8, warn 0.6–0.8): blocked {rule['blocked_tp']} TP / "
                  f"{rule['blocked_fp']} FP · warned {rule['warned_tp']} TP / {rule['warned_fp']} FP · "
                  f"detected (≥0.6) {rule['detected_tp']} TP / {rule['flagged_fp']} FP", ""]

    with open(os.path.join(OUTDIR, 'thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(OUTDIR, 'threshold_report.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("Saved: threshold_report.md / thresholds.json")
    for key, m in result['models'].items():
        r = m['by_threshold'][3]  # t=0.8
        print(f"  {m['label']:14s} @0.8: precision {r['precision']:.4f} recall {r['recall']:.4f} "
              f"FPR {r['fpr']*100:.4f}% FP {r['fp']}")


if __name__ == '__main__':
    main()
