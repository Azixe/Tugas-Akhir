"""Check the deployed-variant models for overfitting (train vs test).

Two views, both on the shared protocol in ``sampling.py``:

1. Train pool (360,140 URLs the models were fitted on) vs common test set
   (90,036 URLs never seen in training): accuracy, precision, recall, F1, FPR
   and log loss at the default 0.5 threshold. A large train-test gap would
   indicate overfitting.
2. Ensemble-size curves: the same metrics as trees / boosting iterations are
   added. An overfitted ensemble keeps improving on train while the test
   curve turns back up.

This measures in-distribution overfitting only. Train and test come from the
same random split, so a small gap here says nothing about external data (see
``external_500/external_report.md``).

Usage (repo root):

    python analyze_overfitting.py
    python analyze_overfitting.py --sample-rows 20000    # quick smoke test
    python analyze_overfitting.py --report-only          # rewrite the report from the saved JSON

Reads : sampling_results/{rf,xgb,cb}_full.pkl
Writes: sampling_results/overfitting_report.md, sampling_results/overfitting.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score,
)

import ml_pipelines
import sampling
from benchmark_sampling import subsample_stratified

OUTDIR = 'sampling_results'
CHUNK = 40000
MODELS = {
    'rf': ('Random Forest', [1, 5, 10, 20, 40, 60, 80, 100]),
    'xgb': ('XGBoost', [1, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700]),
    'cb': ('CatBoost', [1, 10, 25, 50, 100, 150, 200, 250, 300]),
}


def r4(value):
    """Four decimals, rounding half up (stored values carry seven decimals)."""
    return str(Decimal(str(value)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))


def pct3(rate):
    """A rate as a percentage with three decimals, rounding half up."""
    return str((Decimal(str(rate)) * 100).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))


def metrics(y_true, prob):
    pred = (prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'accuracy': round(float(accuracy_score(y_true, pred)), 7),
        'precision': round(float(precision_score(y_true, pred, zero_division=0)), 7),
        'recall': round(float(recall_score(y_true, pred, zero_division=0)), 7),
        'f1': round(float(f1_score(y_true, pred, zero_division=0)), 7),
        'fpr': round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 8),
        'log_loss': round(float(log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1])), 7),
        'fp': int(fp), 'fn': int(fn), 'n': int(len(y_true)),
    }


def staged_probabilities(key, model, urls, checkpoints):
    """Phishing probability per URL for each checkpoint -> array [len(checkpoints), n]."""
    out = np.zeros((len(checkpoints), len(urls)), dtype=np.float32)
    for start in range(0, len(urls), CHUNK):
        batch = urls[start:start + CHUNK]
        sl = slice(start, start + len(batch))
        if key == 'rf':
            X = model.named_steps['features'].transform(batch).astype(np.float32).tocsr()
            trees = model.named_steps['clf'].estimators_
            running = np.zeros(len(batch), dtype=np.float64)
            done = 0
            for i, k in enumerate(checkpoints):
                for tree in trees[done:k]:
                    running += tree.predict_proba(X)[:, 1]
                done = k
                out[i, sl] = running / k
        elif key == 'xgb':
            X = ml_pipelines._xgb_matrix(model, batch)
            for i, k in enumerate(checkpoints):
                out[i, sl] = model['model'].predict_proba(X, iteration_range=(0, k))[:, 1]
        else:
            X = ml_pipelines._to_dense(model['feature_extractor'].transform(batch))
            for i, k in enumerate(checkpoints):
                out[i, sl] = model['model'].predict_proba(X, ntree_end=k)[:, 1]
        del X
        gc.collect()
    return out


def write_report(meta, results):
    lines = ["# Overfitting Check — train pool vs common test set (deployed variants)", ""]
    if meta['smoke_test']:
        lines += ["> **SMOKE TEST (subsampled data) — do not use for the thesis.**", ""]
    lines += [
        f"Generated: {meta['timestamp']}",
        f"Protocol: `sampling.py` (stratified 80/20, seed {sampling.SEED}). Train pool: "
        f"{meta['train']['total']} URLs; common test set: {meta['test']['total']} URLs.",
        "Models: `sampling_results/{rf,xgb,cb}_full.pkl` (full-dataset variants, the ones deployed in v3.4).",
        "",
        "## Train vs test at the full ensemble size (threshold 0.5)",
        "",
        "| Model | Split | Accuracy | Precision | Recall | F1 | FPR | Log loss | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, (name, _) in MODELS.items():
        r = results[key]
        for split in ('train', 'test'):
            m = r[split]
            lines.append(f"| {name} | {split} | {r4(m['accuracy'])} | {r4(m['precision'])} | {r4(m['recall'])} | "
                         f"{r4(m['f1'])} | {pct3(m['fpr'])}% | {r4(m['log_loss'])} | {m['fp']} | {m['fn']} |")
    lines += ["", "## Train-test gap (train minus test, percentage points; log loss as test minus train)", "",
              "| Model | Accuracy gap | Recall gap | F1 gap | FPR gap | Log loss gap |", "|---|---:|---:|---:|---:|---:|"]
    for key, (name, _) in MODELS.items():
        g = results[key]['gap']
        lines.append(f"| {name} | {g['accuracy_pp']:+.3f} | {g['recall_pp']:+.3f} | {g['f1_pp']:+.3f} | "
                     f"{g['fpr_pp']:+.4f} | {g['log_loss']:+.4f} |")
    for key, (name, _) in MODELS.items():
        lines += ["", f"## {name} — metrics as the ensemble grows", "",
                  "| Trees / iterations | Train F1 | Test F1 | Train log loss | Test log loss |",
                  "|---:|---:|---:|---:|---:|"]
        for c in results[key]['curve']:
            lines.append(f"| {c['size']} | {r4(c['train']['f1'])} | {r4(c['test']['f1'])} | "
                         f"{r4(c['train']['log_loss'])} | {r4(c['test']['log_loss'])} |")
        best = min(results[key]['curve'], key=lambda c: c['test']['log_loss'])
        lines += ["", f"Lowest test log loss at {best['size']} "
                      f"({r4(best['test']['log_loss'])}); at the full size it is "
                      f"{r4(results[key]['curve'][-1]['test']['log_loss'])}."]
    lines += ["", "## Reading", "",
              "- A model overfits when it keeps improving on the data it was fitted on while getting worse on held-out data.",
              "- For Random Forest, log loss rises as trees are added, by the same amount on train and test: "
              "averaging depth-limited trees pulls the probabilities toward the middle. It is the same "
              "conservatism seen in the threshold analysis, not a train-test divergence.",
              "- Train and test here come from the same random split of one dataset, so this check covers "
              "in-distribution overfitting only. It does not measure behaviour on URLs from other sources.",
              ""]
    path = os.path.join(OUTDIR, 'overfitting_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return path


def print_summary(results):
    """Console copy of the train/test table in the report (same rounding)."""
    print()
    print(f"{'Model':<15}{'Split':<7}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'F1':>8}{'FPR':>9}{'Log loss':>10}")
    for key, (name, _) in MODELS.items():
        for split in ('train', 'test'):
            m = results[key][split]
            print(f"{name:<15}{split:<7}{r4(m['accuracy']):>9}{r4(m['precision']):>10}{r4(m['recall']):>8}"
                  f"{r4(m['f1']):>8}{pct3(m['fpr']) + '%':>9}{r4(m['log_loss']):>10}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Train vs test overfitting check for the deployed variants.")
    ap.add_argument('--data', default=sampling.DEFAULT_DATASET)
    ap.add_argument('--sample-rows', type=int, default=0, help="Debug: cap the train pool / test set (0 = full)")
    ap.add_argument('--report-only', action='store_true',
                    help="Rewrite the Markdown report from the existing overfitting.json without re-scoring")
    args = ap.parse_args()

    if args.report_only:
        with open(os.path.join(OUTDIR, 'overfitting.json'), encoding='utf-8') as f:
            saved = json.load(f)
        print_summary(saved['models'])
        print("Saved:", write_report(saved['meta'], saved['models']))
        return

    df = sampling.load_clean_dataset(args.data)
    train, test = sampling.common_split(df)
    if args.sample_rows:
        train = subsample_stratified(train, args.sample_rows)
        test = subsample_stratified(test, max(1000, args.sample_rows // 4))
        print("!! SMOKE TEST MODE (subsampled data) — do not use for the thesis")
    print(f"Train pool: {sampling.describe(train)}")
    print(f"Common test: {sampling.describe(test)}")

    results = {}
    for key, (name, checkpoints) in MODELS.items():
        model = joblib.load(os.path.join(OUTDIR, f'{key}_full.pkl'))
        entry = {'curve': []}
        staged = {}
        for split, frame in (('train', train), ('test', test)):
            print(f"  [{name}] scoring {split} ({len(frame)} URLs)...", flush=True)
            staged[split] = staged_probabilities(key, model, frame['url'].values, checkpoints)
            entry[split] = metrics(frame['label'].values, staged[split][-1])
        for i, k in enumerate(checkpoints):
            entry['curve'].append({'size': k,
                                   'train': metrics(train['label'].values, staged['train'][i]),
                                   'test': metrics(test['label'].values, staged['test'][i])})
        tr, te = entry['train'], entry['test']
        entry['gap'] = {'accuracy_pp': round((tr['accuracy'] - te['accuracy']) * 100, 3),
                        'recall_pp': round((tr['recall'] - te['recall']) * 100, 3),
                        'f1_pp': round((tr['f1'] - te['f1']) * 100, 3),
                        'fpr_pp': round((tr['fpr'] - te['fpr']) * 100, 4),
                        'log_loss': round(te['log_loss'] - tr['log_loss'], 7)}
        results[key] = entry
        print(f"  [{name}] train F1 {r4(tr['f1'])} / test F1 {r4(te['f1'])} | "
              f"train log loss {r4(tr['log_loss'])} / test {r4(te['log_loss'])}", flush=True)
        del model, staged
        gc.collect()

    meta = {'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'dataset': args.data, 'seed': sampling.SEED,
            'train': sampling.describe(train), 'test': sampling.describe(test),
            'smoke_test': bool(args.sample_rows)}
    suffix = '_smoke' if args.sample_rows else ''
    with open(os.path.join(OUTDIR, f'overfitting{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'models': results}, f, indent=2)
    print_summary(results)
    if not args.sample_rows:
        print("Saved:", write_report(meta, results))
    print(f"Saved: {os.path.join(OUTDIR, f'overfitting{suffix}.json')}")


if __name__ == '__main__':
    main()
