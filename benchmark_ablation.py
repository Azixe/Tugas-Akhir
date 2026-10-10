"""Feature ablation: scheme / ``www.`` shortcuts and feature groups.

Trains RF, XGBoost and CatBoost (full-dataset variant, fixed hyperparameters)
once per ablation arm and compares the arms under one protocol:

    common split (sampling.py)  ->  train pool | common test set
    validation_split(train pool) ->  fit set    | validation set

Every model is fitted on the fit set and compared on the **validation** set.
The common test set is only scored when ``--final-test`` is passed, so arm
selection never looks at it.

Arms (``--arms``, default: the first five):

    all                        historical 1,509-dim features (v3.4 feature path)
    no_scheme                  scheme removed from the URL string
    no_scheme_www              scheme and a leading ``www.`` removed
    tfidf_only                 TF-IDF block only
    struct_only                9 structural features only
    tfidf_only_no_scheme_www   TF-IDF block only, scheme and ``www.`` removed
    struct_only_no_scheme_www  structural only, scheme and ``www.`` removed

Usage (run from the repo root; ``--outdir`` is required and must not be
``sampling_results``, which holds the v3.4 evidence):

    python benchmark_ablation.py --outdir ablation_results
    python benchmark_ablation.py --outdir ablation_results --drop-empty-tokens
    python benchmark_ablation.py --outdir ablation_results --final-test
    python benchmark_ablation.py --outdir ablation_smoke --sample-rows 20000 --cb-iterations 50

Outputs:
    <arm>__<model>.pkl          one artifact per arm and model
    ablation_metrics.json / ablation_report.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib

import ml_pipelines
import sampling
from benchmark_sampling import evaluate, subsample_stratified

ROOT = os.path.dirname(os.path.abspath(__file__))

ARMS = {
    'all': dict(),
    'no_scheme': dict(strip_scheme=True),
    'no_scheme_www': dict(strip_scheme=True, strip_www=True),
    'tfidf_only': dict(groups=('text',)),
    'struct_only': dict(groups=('structural',)),
    'tfidf_only_no_scheme_www': dict(strip_scheme=True, strip_www=True, groups=('text',)),
    'struct_only_no_scheme_www': dict(strip_scheme=True, strip_www=True, groups=('structural',)),
}
DEFAULT_ARMS = ['all', 'no_scheme', 'no_scheme_www', 'tfidf_only', 'struct_only']
MODELS = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'cb': 'CatBoost'}
PROTECTED_OUTDIRS = {'sampling_results', 'phishingdetectorext', 'rf_models'}


def csv_choice(valid):
    def parse(value):
        items = [v.strip() for v in value.split(',') if v.strip()]
        unknown = [v for v in items if v not in valid]
        if unknown or not items:
            raise argparse.ArgumentTypeError(f"choose from {', '.join(valid)} (got {value!r})")
        return items
    return parse


def parse_args():
    parser = argparse.ArgumentParser(description="Scheme / www. / feature-group ablation.")
    parser.add_argument('--data', default=sampling.DEFAULT_DATASET,
                        help="Dataset CSV (default: %(default)s)")
    parser.add_argument('--outdir', required=True,
                        help="Where to store models and results (never sampling_results)")
    parser.add_argument('--arms', type=csv_choice(list(ARMS)), default=DEFAULT_ARMS,
                        help="Comma-separated arms (default: %s)" % ','.join(DEFAULT_ARMS))
    parser.add_argument('--models', type=csv_choice(list(MODELS)), default=list(MODELS),
                        help="Comma-separated model families (default: rf,xgb,cb)")
    parser.add_argument('--drop-empty-tokens', action='store_true',
                        help="Filter the empty-string token at training time, in every arm")
    parser.add_argument('--cb-iterations', type=int, default=300,
                        help="CatBoost iterations (default: %(default)s, the v3.4 setting)")
    parser.add_argument('--final-test', action='store_true',
                        help="Also score the common test set (do this once, after choosing an arm)")
    parser.add_argument('--force-retrain', action='store_true',
                        help="Ignore cached .pkl models and retrain")
    parser.add_argument('--sample-rows', type=int, default=0,
                        help="Debug: cap the train pool / test set size (0 = full)")
    parser.add_argument('--latency-samples', type=int, default=0,
                        help="URLs sampled for per-URL latency (default: %(default)s = skip)")
    args = parser.parse_args()
    if os.path.basename(os.path.normpath(args.outdir)).lower() in PROTECTED_OUTDIRS:
        parser.error(f"--outdir {args.outdir} holds deployed/v3.4 artifacts; pick another directory")
    return args


def check_cached_config(args):
    """Cached models are only valid for the configuration that trained them."""
    path = os.path.join(args.outdir, 'ablation_metrics.json')
    if args.force_retrain or not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        old = json.load(f)['meta']
    current = {'drop_empty_tokens': bool(args.drop_empty_tokens),
               'cb_iterations': args.cb_iterations,
               'smoke_test': bool(args.sample_rows), 'dataset': args.data}
    changed = [k for k, v in current.items() if old.get(k) != v]
    if changed:
        raise SystemExit(
            f"{args.outdir} holds models trained with different settings ({', '.join(changed)}). "
            "Use another --outdir or pass --force-retrain.")


def train_catboost(train_df, iterations, feature_kwargs):
    sys.path.insert(0, os.path.join(ROOT, 'Catboost'))
    from catboost import CatBoostClassifier
    from train_catboost import DEFAULT_PARAMS

    params = dict(DEFAULT_PARAMS, iterations=iterations, verbose=0)
    feature_extractor = ml_pipelines.build_feature_union(**feature_kwargs)
    X_train = ml_pipelines._to_dense(
        feature_extractor.fit_transform(train_df['url'].values, train_df['label'].values))
    model = CatBoostClassifier(**params)
    model.fit(X_train, train_df['label'].values)
    return {'model': model, 'feature_extractor': feature_extractor, 'params': params,
            'n_train': int(len(train_df))}


def train_or_load(arm, family, train_df, args, feature_kwargs):
    """Returns (model_object, meta). Same artifact shapes as benchmark_sampling."""
    path = os.path.join(args.outdir, f"{arm}__{family}.pkl")
    label = f"{arm} / {MODELS[family]}"
    if os.path.exists(path) and not args.force_retrain:
        print(f"  [{label}] loading cached {path}")
        return joblib.load(path), {'reused': True, 'path': path}

    print(f"  [{label}] training on {len(train_df)} URLs...", flush=True)
    t0 = time.time()
    if family == 'rf':
        model = ml_pipelines.build_rf_pipeline(**feature_kwargs)
        model.fit(train_df['url'].values, train_df['label'].values)
    elif family == 'xgb':
        model = ml_pipelines.train_xgb_pipeline(
            train_df['url'].values, train_df['label'].values, **feature_kwargs)
    else:
        model = train_catboost(train_df, args.cb_iterations, feature_kwargs)

    joblib.dump(model, path, compress=3)
    seconds = time.time() - t0
    print(f"  [{label}] trained in {seconds:.1f}s, saved {path} "
          f"({os.path.getsize(path) / 1024:.0f} KB)", flush=True)
    return model, {'reused': False, 'path': path, 'train_seconds': round(seconds, 1)}


def n_input_features(model, family):
    extractor = model.named_steps['features'] if family == 'rf' else model['feature_extractor']
    return int(extractor.transform(['https://www.example.com/']).shape[1])


def metrics_table(results, arms, models, split):
    lines = ["| Arm | Model | Inputs | Accuracy | Precision (phish) | Recall (phish) | F1 (phish) | FPR | FP | FN |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        for family in models:
            entry = results[arm][family]
            r = entry[split]
            lines.append(
                f"| `{arm}` | {MODELS[family]} | {entry['n_features']} | {r['accuracy']:.4f} | "
                f"{r['precision_phishing']:.4f} | {r['recall_phishing']:.4f} | {r['f1_phishing']:.4f} | "
                f"{r['fpr'] * 100:.3f}% | {r['fp']} | {r['fn']} |")
    return lines


def write_report(args, meta, results):
    lines = ["# Feature Ablation — scheme, `www.` and feature groups", ""]
    if meta['smoke_test']:
        lines += ["> **SMOKE TEST (subsampled data) — do not use for the thesis.**", ""]
    lines += [
        f"Generated: {meta['timestamp']}",
        f"Dataset: `{meta['dataset']}` — {meta['dataset_rows']} rows after cleaning",
        f"Protocol: `sampling.py` common split (stratified 80/20, seed {sampling.SEED}), then "
        f"`validation_split` (stratified {int(sampling.VAL_SIZE * 100)}% of the train pool).",
        "",
        f"- Fit set: {meta['fit']['total']} ({meta['fit']['legit']} legit / {meta['fit']['phishing']} phishing)",
        f"- Validation set: {meta['validation']['total']} "
        f"({meta['validation']['legit']} legit / {meta['validation']['phishing']} phishing)",
        f"- Common test set: {meta['test']['total']} "
        f"({'scored below' if meta['final_test'] else 'NOT scored in this run'})",
        f"- Empty-string token at training time: {'dropped' if meta['drop_empty_tokens'] else 'kept (historical)'}",
        f"- Training data: full fit set (no undersampling); CatBoost iterations: {meta['cb_iterations']}",
        "",
        "## Validation set — use this table to choose an arm",
        "",
    ]
    lines += metrics_table(results, meta['arms'], meta['models'], 'validation')
    lines += [""]
    if meta['final_test']:
        lines += ["## Common test set — final pass (not for selection)", ""]
        lines += metrics_table(results, meta['arms'], meta['models'], 'test')
        lines += [""]
    lines += [
        "## Notes",
        "",
        "- Hyperparameters are fixed per model family; only the feature construction differs between arms.",
        "- Models are fitted on the fit set (90% of the train pool), so these numbers are not directly "
        "comparable with `sampling_results/sampling_report.md`, where models see the whole train pool.",
        "- Validation and test rows come from the same random split as the training data "
        "(in-distribution; see `sampling_results/dataset_split_report.md` for host overlap). "
        "They do not measure generalisation to live URLs.",
    ]
    path = os.path.join(args.outdir, 'ablation_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    check_cached_config(args)

    print("=" * 70)
    print("FEATURE ABLATION — scheme / www. / feature groups (RF, XGBoost & CatBoost)")
    print("=" * 70)

    df = sampling.load_clean_dataset(args.data)
    train_pool, test = sampling.common_split(df)
    if args.sample_rows:
        train_pool = subsample_stratified(train_pool, args.sample_rows)
        test = subsample_stratified(test, max(1000, args.sample_rows // 4))
        print("!! SMOKE TEST MODE (subsampled data) — do not use for the thesis")
    fit, val = sampling.validation_split(train_pool)
    print(f"Fit set: {sampling.describe(fit)}")
    print(f"Validation set: {sampling.describe(val)}")
    print(f"Common test set: {len(test)} rows ({'final pass' if args.final_test else 'not scored'})")
    print()

    results = {}
    for arm in args.arms:
        feature_kwargs = dict(ARMS[arm], drop_empty=args.drop_empty_tokens)
        results[arm] = {}
        for family in args.models:
            model, meta = train_or_load(arm, family, fit, args, feature_kwargs)
            entry = {'meta': meta, 'n_features': n_input_features(model, family)}
            if family == 'xgb':
                entry['xgb_feature_counts'] = model['feature_counts']
            entry['validation'] = evaluate(model, family, val, args.latency_samples)
            if args.final_test:
                entry['test'] = evaluate(model, family, test, args.latency_samples)
            results[arm][family] = entry
            r = entry['validation']
            print(f"  [{arm} / {MODELS[family]}] validation: acc {r['accuracy']:.4f} "
                  f"recall {r['recall_phishing']:.4f} FPR {r['fpr'] * 100:.3f}% "
                  f"(FP {r['fp']}, FN {r['fn']})", flush=True)
            del model
            gc.collect()

    meta = {
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'dataset': args.data,
        'dataset_rows': int(len(df)),
        'seed': sampling.SEED,
        'val_size': sampling.VAL_SIZE,
        'fit': sampling.describe(fit),
        'validation': sampling.describe(val),
        'test': sampling.describe(test),
        'arms': args.arms,
        'arm_definitions': {a: {k: list(v) if isinstance(v, tuple) else v for k, v in ARMS[a].items()}
                            for a in args.arms},
        'models': args.models,
        'drop_empty_tokens': bool(args.drop_empty_tokens),
        'cb_iterations': args.cb_iterations,
        'final_test': bool(args.final_test),
        'smoke_test': bool(args.sample_rows),
    }
    json_path = os.path.join(args.outdir, 'ablation_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'arms': results}, f, indent=2)
    report_path = write_report(args, meta, results)

    print()
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
