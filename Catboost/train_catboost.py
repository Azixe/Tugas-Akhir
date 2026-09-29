"""Train CatBoost (third comparison model) under the shared sampling protocol.

Same raw features as the deployed Random Forest: 1500 word-level TF-IDF tokens
+ 9 structural features = 1509 inputs. Protocol follows ``sampling.py``:
common stratified 80/20 split (seed 42), so CatBoost is trained and evaluated
under the exact conditions of the RF/XGB sampling study.

Usage (run from the repo root):

    python Catboost/train_catboost.py                    # full + undersampled
    python Catboost/train_catboost.py --variant full
    python Catboost/train_catboost.py --sample-rows 20000 --iterations 100   # smoke test

Outputs (default ``--outdir sampling_results``):
    cb_full.pkl / cb_under.pkl           (joblib: model + feature extractor + params)
    cb_full.onnx / cb_under.onnx         (CatBoost native ONNX export)
    catboost_metrics.json                (test-set metrics per variant)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_pipelines  # noqa: E402
import sampling  # noqa: E402

from catboost import CatBoostClassifier  # noqa: E402

DEFAULT_PARAMS = dict(
    iterations=700,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    random_seed=sampling.SEED,
    thread_count=-1,
    verbose=100,
    allow_writing_files=False,  # keep the repo clean (no catboost_info/)
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def subsample_stratified(df, n, seed=sampling.SEED):
    if n <= 0 or n >= len(df):
        return df
    per_class = max(1, n // 2)
    parts = [g.head(per_class) for _, g in df.groupby('label', sort=True)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def evaluate_variant(model, feature_extractor, test_df):
    X = ml_pipelines._to_dense(feature_extractor.transform(test_df['url'].values))
    y_true = test_df['label'].values
    y_pred = model.predict(X).astype(int)
    probs = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'n_test': int(len(y_true)),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision_phish': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall_phish': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1_phish': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'fpr': round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 6),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'prob_mean_phish': round(float(probs[y_true == 1].mean()), 4),
    }


def train_variant(variant, train_df, test_df, outdir, params, export_onnx):
    key = f'cb_{variant}'
    t0 = time.time()
    print(f"  [{key}] training on {len(train_df)} URLs "
          f"(iterations={params['iterations']}, depth={params['depth']}) ...", flush=True)

    feature_extractor = ml_pipelines.build_feature_union()
    X_train = ml_pipelines._to_dense(
        feature_extractor.fit_transform(train_df['url'].values, train_df['label'].values))
    print(f"  [{key}] feature matrix: {X_train.shape} ({time.time()-t0:.0f}s)", flush=True)

    model = CatBoostClassifier(**params)
    model.fit(X_train, train_df['label'].values)
    train_seconds = round(time.time() - t0, 1)

    os.makedirs(outdir, exist_ok=True)
    pkl_path = os.path.join(outdir, f'{key}.pkl')
    joblib.dump({'model': model, 'feature_extractor': feature_extractor,
                 'params': params, 'variant': variant,
                 'n_train': int(len(train_df))}, pkl_path, compress=3)

    onnx_path = os.path.join(outdir, f'{key}.onnx')
    if export_onnx:
        model.save_model(onnx_path, format='onnx', export_parameters={
            'onnx_domain': 'ai.catboost', 'onnx_model_version': 1,
            'onnx_doc_string': f'CatBoost phishing detector ({variant})',
            'onnx_graph_name': f'CatBoostPhishing_{variant}',
        })
        # CatBoost exports probabilities as seq(map); expose the flat [N,2]
        # tensor instead so the browser/plain-tensor consumers work unchanged.
        from fix_onnx_output import strip_zipmap
        print("  " + strip_zipmap(onnx_path))

    metrics = evaluate_variant(model, feature_extractor, test_df)
    result = {
        'variant': variant,
        'n_train': int(len(train_df)),
        'train_seconds': train_seconds,
        'pkl_kb': round(os.path.getsize(pkl_path) / 1024),
        'onnx_kb': round(os.path.getsize(onnx_path) / 1024) if export_onnx else None,
        'onnx_sha256': sha256(onnx_path) if export_onnx else None,
        'params': params,
        **metrics,
    }
    print(f"  [{key}] trained in {train_seconds}s | acc {metrics['accuracy']:.4f} "
          f"recall {metrics['recall_phish']:.4f} FPR {metrics['fpr']:.6f} "
          f"FP {metrics['fp']} | onnx {result['onnx_kb']} KB", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="Train CatBoost (third model) on the shared protocol.")
    ap.add_argument('--data', default=sampling.DEFAULT_DATASET)
    ap.add_argument('--outdir', default='sampling_results')
    ap.add_argument('--variant', choices=['full', 'under', 'both'], default='both')
    ap.add_argument('--iterations', type=int, default=DEFAULT_PARAMS['iterations'])
    ap.add_argument('--depth', type=int, default=DEFAULT_PARAMS['depth'])
    ap.add_argument('--learning-rate', type=float, default=DEFAULT_PARAMS['learning_rate'])
    ap.add_argument('--sample-rows', type=int, default=0,
                    help="Debug: cap train pool / test set (0 = full)")
    ap.add_argument('--no-onnx', action='store_true', help="skip ONNX export (smoke tests)")
    args = ap.parse_args()

    params = dict(DEFAULT_PARAMS)
    params.update(iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate)

    df = sampling.load_clean_dataset(args.data)
    train_pool, test = sampling.common_split(df)
    print("Protocol: common split (sampling.py, seed 42)")
    print(f"  train pool: {len(train_pool)} | test: {len(test)}")

    if args.sample_rows:
        train_pool = subsample_stratified(train_pool, args.sample_rows)
        test = subsample_stratified(test, max(2000, args.sample_rows // 4))
        print(f"  [debug] capped to train {len(train_pool)} / test {len(test)}")

    results = []
    if args.variant in ('full', 'both'):
        results.append(train_variant('full', train_pool, test, args.outdir, params,
                                     export_onnx=not args.no_onnx))
    if args.variant in ('under', 'both'):
        results.append(train_variant('under', sampling.undersample(train_pool), test,
                                     args.outdir, params, export_onnx=not args.no_onnx))

    metrics_path = os.path.join(args.outdir, 'catboost_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {metrics_path}")
    for r in results:
        print(f"  {r['variant']}: acc {r['accuracy']} recall {r['recall_phish']} "
              f"FPR {r['fpr']} FP {r['fp']} ONNX {r['onnx_kb']} KB")


if __name__ == '__main__':
    main()
