"""Benchmark: undersampling vs non-undersampling for RF and XGBoost.

Trains (or reuses) four models under the shared protocol in ``sampling.py``
and evaluates all of them on the *same* common test set:

    RF  undersampled   |  RF  full dataset
    XGB undersampled   |  XGB full dataset

Writes machine-readable results and a Markdown report for the thesis
(Bab 4 evidence — log every process, Pak Rahmad no. 1).

Usage (run from the repo root):

    python benchmark_sampling.py                       # train if needed, evaluate
    python benchmark_sampling.py --force-retrain       # retrain everything
    python benchmark_sampling.py --sample-rows 20000   # quick smoke test

Outputs (default ``--outdir sampling_results``):
    rf_under.pkl / rf_full.pkl / xgb_under.pkl / xgb_full.pkl
    sampling_metrics.json / sampling_metrics.csv / sampling_report.md
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

import ml_pipelines
import sampling

MODEL_KEYS = ['rf_under', 'rf_full', 'xgb_under', 'xgb_full']
DISPLAY = {
    'rf_under': 'RF (undersampled)',
    'rf_full': 'RF (full dataset)',
    'xgb_under': 'XGBoost (undersampled)',
    'xgb_full': 'XGBoost (full dataset)',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Undersampling vs non-undersampling benchmark.")
    parser.add_argument('--data', default=sampling.DEFAULT_DATASET,
                        help="Dataset CSV (default: %(default)s)")
    parser.add_argument('--outdir', default='sampling_results',
                        help="Where to store models and results (default: %(default)s)")
    parser.add_argument('--force-retrain', action='store_true',
                        help="Ignore cached .pkl models and retrain")
    parser.add_argument('--sample-rows', type=int, default=0,
                        help="Debug: cap the train pool / test set size (0 = full)")
    parser.add_argument('--latency-samples', type=int, default=500,
                        help="URLs sampled for per-URL latency (default: %(default)s)")
    return parser.parse_args()


def subsample_stratified(df, n, seed=sampling.SEED):
    if n <= 0 or n >= len(df):
        return df
    per_class = max(1, n // 2)
    parts = [g.head(per_class) for _, g in df.groupby('label', sort=True)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_or_load(key, train_df, outdir, force_retrain):
    """Returns (model_object, meta). RF -> sklearn Pipeline; XGB -> artifacts dict."""
    path = os.path.join(outdir, f"{key}.pkl")
    if os.path.exists(path) and not force_retrain:
        print(f"  [{DISPLAY[key]}] loading cached {path}")
        return joblib.load(path), {'reused': True, 'path': path}

    t0 = time.time()
    if key.startswith('rf'):
        print(f"  [{DISPLAY[key]}] training on {len(train_df)} URLs...")
        model = ml_pipelines.build_rf_pipeline()
        model.fit(train_df['url'].values, train_df['label'].values)
    else:
        print(f"  [{DISPLAY[key]}] training on {len(train_df)} URLs...")
        model = ml_pipelines.train_xgb_pipeline(
            train_df['url'].values, train_df['label'].values)

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(model, path, compress=3)
    seconds = time.time() - t0
    size_kb = os.path.getsize(path) / 1024
    print(f"  [{DISPLAY[key]}] trained in {seconds:.1f}s, saved {path} ({size_kb:.0f} KB)")
    return model, {'reused': False, 'path': path, 'train_seconds': round(seconds, 1),
                   'n_train': int(len(train_df))}


def make_predictors(model, key):
    if key.startswith('rf'):
        return (lambda urls: model.predict(urls),
                lambda urls: model.predict_proba(urls))
    return (lambda urls: ml_pipelines.predict_with_xgb(model, urls),
            lambda urls: ml_pipelines.predict_proba_with_xgb(model, urls))


def evaluate(model, key, test_df, latency_samples):
    urls = test_df['url'].values
    y_true = test_df['label'].values
    predict, predict_proba = make_predictors(model, key)

    t0 = time.time()
    y_pred = predict(urls)
    probs = predict_proba(urls)[:, 1]
    predict_seconds = time.time() - t0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    latencies = []
    if latency_samples > 0:
        n = min(latency_samples, len(urls))
        idx = np.random.RandomState(sampling.SEED).choice(len(urls), n, replace=False)
        for i in idx:
            t = time.perf_counter()
            predict([urls[i]])
            latencies.append((time.perf_counter() - t) * 1000)

    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision_phishing': round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        'recall_phishing': round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        'f1_phishing': round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        'fpr': round(float(fpr), 5),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'predict_seconds': round(predict_seconds, 1),
        'latency_avg_ms': round(float(np.mean(latencies)), 4) if latencies else None,
        'latency_p95_ms': round(float(np.percentile(latencies, 95)), 4) if latencies else None,
        'prob_phishing_avg': round(float(np.mean(probs)), 4),
    }


def write_report(outdir, meta, results):
    lines = []
    lines.append("# Undersampling vs Non-Undersampling — Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {meta['timestamp']}")
    lines.append(f"Dataset: `{meta['dataset']}` — {meta['dataset_rows']} rows after cleaning")
    lines.append(f"Protocol: `sampling.py` (stratified 80/20, seed {sampling.SEED}); "
                 f"all models evaluated on the same {meta['test']['total']}-URL test set")
    lines.append("")
    lines.append(f"- Train pool: {meta['train_pool']['total']} "
                 f"({meta['train_pool']['legit']} legit / {meta['train_pool']['phishing']} phishing)")
    lines.append(f"- Common test: {meta['test']['total']} "
                 f"({meta['test']['legit']} legit / {meta['test']['phishing']} phishing)")
    lines.append("")
    lines.append("| Model | Train size | Accuracy | Precision (phish) | Recall (phish) | F1 (phish) | FPR | FP | FN | Avg latency (ms) | Size (KB) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in MODEL_KEYS:
        if key not in results:
            continue
        r, m = results[key]['metrics'], results[key]['meta']
        size = os.path.getsize(m['path']) / 1024 if 'path' in m else 0
        lines.append(
            f"| {DISPLAY[key]} | {m.get('n_train', '-')} | {r['accuracy']:.4f} | "
            f"{r['precision_phishing']:.4f} | {r['recall_phishing']:.4f} | {r['f1_phishing']:.4f} | "
            f"{r['fpr']*100:.3f}% | {r['fp']} | {r['fn']} | "
            f"{r['latency_avg_ms']} | {size:.0f} |")
    lines.append("")
    lines.append("## Notes for Bab 4")
    lines.append("")
    lines.append("- Same architecture and hyperparameters per model family; only the training data differs.")
    lines.append("- FPR is the key metric: compare FP counts at the same recall level.")
    lines.append("- Latency measured locally (Python), not in the browser — see the ONNX benchmark for extension numbers.")
    path = os.path.join(outdir, 'sampling_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("SAMPLING BENCHMARK — undersampled vs full dataset (RF & XGBoost)")
    print("=" * 70)

    df = sampling.load_clean_dataset(args.data)
    train_pool, test = sampling.common_split(df)
    if args.sample_rows:
        train_pool = subsample_stratified(train_pool, args.sample_rows)
        test = subsample_stratified(test, max(1000, args.sample_rows // 4))
        print("!! SMOKE TEST MODE (subsampled data) — do not use for the thesis")

    train_pool_info = sampling.describe(train_pool)
    test_info = sampling.describe(test)
    print(f"Train pool: {train_pool_info}")
    print(f"Common test: {test_info}")

    train_under = sampling.undersample(train_pool)
    print(f"Undersampled train set: {sampling.describe(train_under)}")
    print()

    model_meta, results = {}, {}
    for key in MODEL_KEYS:
        data = train_under if key.endswith('under') else train_pool
        model, meta = train_or_load(key, data, args.outdir, args.force_retrain)
        model_meta[key] = meta
        results[key] = {'meta': meta}
        print(f"  [{DISPLAY[key]}] evaluating on {len(test)} URLs...")
        results[key]['metrics'] = evaluate(model, key, test, args.latency_samples)
        # free memory between variants
        del model
        gc.collect()

    meta = {
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'dataset': args.data,
        'dataset_rows': int(len(df)),
        'seed': sampling.SEED,
        'train_pool': train_pool_info,
        'under_train': sampling.describe(train_under),
        'test': test_info,
        'smoke_test': bool(args.sample_rows),
    }
    payload = {'meta': meta, 'models': results}

    json_path = os.path.join(args.outdir, 'sampling_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(args.outdir, 'sampling_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'train_size', 'accuracy', 'precision_phishing', 'recall_phishing',
                         'f1_phishing', 'fpr', 'fp', 'fn', 'latency_avg_ms', 'latency_p95_ms'])
        for key in MODEL_KEYS:
            r, m = results[key]['metrics'], results[key]['meta']
            writer.writerow([DISPLAY[key], m.get('n_train', ''), r['accuracy'], r['precision_phishing'],
                             r['recall_phishing'], r['f1_phishing'], r['fpr'], r['fp'], r['fn'],
                             r['latency_avg_ms'], r['latency_p95_ms']])

    report_path = write_report(args.outdir, meta, results)

    print("\n" + "=" * 70)
    print("RESULTS — common test set")
    print("=" * 70)
    header = f"{'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>8} {'FP':>6} {'FN':>6}"
    print(header)
    for key in MODEL_KEYS:
        r = results[key]['metrics']
        print(f"{DISPLAY[key]:<24} {r['accuracy']:>7.4f} {r['precision_phishing']:>7.4f} "
              f"{r['recall_phishing']:>7.4f} {r['f1_phishing']:>7.4f} "
              f"{r['fpr']*100:>7.3f}% {r['fp']:>6} {r['fn']:>6}")
    print()
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()