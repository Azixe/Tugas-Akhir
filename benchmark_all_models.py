"""
Automated Performance Benchmark — PKL vs ONNX × RF vs XGBoost
==============================================================
Evaluates up to four entries on the same held-out test set:
  1. RF (pickle / scikit-learn)
  2. RF (ONNX)
  3. XGBoost (pickle / native)
  4. XGBoost (ONNX)

Two protocols (--protocol):
  legacy (default): balanced undersampled split, 41,776 test URLs
  common:           sampling.py common split on the full dataset, 90,036 URLs

All paths are overridable, so any matching PKL/ONNX pair can be benchmarked.
For the sampling-study models:

  python benchmark_all_models.py --protocol common \
      --rf-pkl sampling_results/rf_under.pkl \
      --rf-onnx sampling_results/rf_under.onnx \
      --rf-tfidf sampling_results/rf_under_tfidf.json \
      --xgb-pkl sampling_results/xgb_under.pkl \
      --xgb-onnx sampling_results/xgb_under.onnx \
      --xgb-tfidf sampling_results/xgb_under_tfidf.json \
      --xgb-prep sampling_results/xgb_under_preprocessing.json \
      --out docs/benchmark_pkl_vs_onnx_under.md

Latency is measured per URL as the full pipeline (feature extraction +
preprocessing + model) — the same path the extension executes at scan time.
A model-only latency is reported for the ONNX entries as well.
"""

import argparse
import contextlib
import pandas as pd
import numpy as np
import time
import json
import os
import sys
import psutil
import joblib

import onnxruntime as ort
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# ============================================================
# SHARED HELPERS (single source of truth: features.py)
# ============================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (  # noqa: E402
    alias_legacy_main, build_tfidf_vector, extract_features_onnx,
    preprocess_xgb, structural_features,
)

# Legacy pkls reference __main__.make_tokens / StructuralFeatureExtractor
alias_legacy_main()

# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="PKL vs ONNX benchmark for RF and XGBoost.")
    parser.add_argument('--data', default='URL dataset.csv',
                        help="Dataset CSV (default: %(default)s)")
    parser.add_argument('--protocol', choices=['legacy', 'common'], default='legacy',
                        help="Test-set protocol (default: %(default)s)")
    parser.add_argument('--rf-pkl', default=os.path.join('rf_models', 'phishing_optimized.pkl'))
    parser.add_argument('--xgb-pkl', default=os.path.join('Xgboost', 'phishing_xgb_pipeline.pkl'))
    parser.add_argument('--rf-onnx', default=os.path.join('phishingdetectorExt', 'phishing_rf.onnx'))
    parser.add_argument('--xgb-onnx', default=os.path.join('phishingdetectorExt', 'phishing_xgb.onnx'))
    parser.add_argument('--rf-tfidf', default=os.path.join('phishingdetectorExt', 'tfidf_data.json'))
    parser.add_argument('--xgb-tfidf', default=os.path.join('phishingdetectorExt', 'tfidf_data_xgb.json'))
    parser.add_argument('--xgb-prep', default=os.path.join('phishingdetectorExt', 'xgb_preprocessing.json'))
    parser.add_argument('--latency-samples', type=int, default=1000,
                        help="URLs sampled for per-URL latency (default: %(default)s)")
    parser.add_argument('--max-test-rows', type=int, default=0,
                        help="Debug: cap the test set size (0 = full)")
    parser.add_argument('--out', default=None,
                        help="Write the full report to this Markdown file")
    return parser.parse_args()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    args = parse_args()
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            with contextlib.redirect_stdout(Tee(sys.stdout, f)):
                run(args)
    else:
        run(args)


def run(args):
    print("=" * 70)
    print("BENCHMARK: PKL vs ONNX × RF vs XGBoost (All 4 Models)")
    print(f"Protocol: {args.protocol}")
    print("=" * 70)

    # --- Paths (from CLI) ---
    RF_ONNX, XGB_ONNX = args.rf_onnx, args.xgb_onnx
    RF_TFIDF, XGB_TFIDF, XGB_PREP = args.rf_tfidf, args.xgb_tfidf, args.xgb_prep
    RF_PKL, XGB_PKL = args.rf_pkl, args.xgb_pkl

    missing = [f for f in [RF_ONNX, XGB_ONNX, RF_TFIDF, XGB_TFIDF, XGB_PREP, RF_PKL, args.data]
               if not os.path.exists(f)]
    if missing:
        for f in missing:
            print(f"  ERROR: {f} not found!")
        print("  (XGB_PKL is optional — the benchmark continues without it)")
        exit()

    # --- Load Dataset ---
    print("\n[1] Loading dataset...")
    if args.protocol == 'common':
        import sampling
        df = sampling.load_clean_dataset(args.data)
        _, test_df = sampling.common_split(df)
        print(f"    Protocol: common (sampling.py) — full dataset {len(df)} rows")
    else:
        df = pd.read_csv(args.data)
        df = df.dropna(subset=['url', 'type'])
        df = df.drop_duplicates(subset=['url'])
        df['label'] = df['type'].map({'phishing': 1, 'legitimate': 0})

        # Undersample (same as the legacy training protocol)
        df_phish = df[df['label'] == 1]
        df_legit = df[df['label'] == 0].sample(n=len(df_phish), random_state=42)
        df_balanced = pd.concat([df_legit, df_phish]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Same test split as training (80/20)
        _, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])
        print("    Protocol: legacy (balanced undersampled split)")

    test_urls = test_df['url'].values
    y_true = test_df['label'].values
    if args.max_test_rows and args.max_test_rows < len(test_urls):
        sel = np.random.RandomState(42).choice(len(test_urls), args.max_test_rows, replace=False)
        test_urls = test_urls[sel]
        y_true = y_true[sel]
        print(f"    !! Debug: test set capped to {len(test_urls)} rows")
    print(f"    Test set: {len(test_urls)} URLs ({sum(y_true==0)} legit, {sum(y_true==1)} phishing)")

    # --- Load PKL Models ---
    print("\n[2] Loading PKL models...")
    import joblib
    import warnings
    warnings.filterwarnings('ignore')

    # RF PKL — this is a full Pipeline (features + classifier)
    print("    Loading RF pkl...")
    rf_pkl_pipeline = joblib.load(RF_PKL)
    print(f"    RF pkl type: {type(rf_pkl_pipeline).__name__}")

    # XGBoost PKL — dict with model + preprocessing objects
    xgb_pkl_loaded = False
    print("    Loading XGBoost pkl...")
    try:
        xgb_pkl_artifacts = joblib.load(XGB_PKL)
        print(f"    XGB pkl keys: {list(xgb_pkl_artifacts.keys())}")
        xgb_pkl_model = xgb_pkl_artifacts['model']
        xgb_pkl_scaler = xgb_pkl_artifacts['scaler']
        xgb_pkl_kbest = xgb_pkl_artifacts['selector_kbest']
        xgb_pkl_pca = xgb_pkl_artifacts['pca']
        xgb_pkl_feature_extractor = xgb_pkl_artifacts['feature_extractor']
        xgb_pkl_rfecv = xgb_pkl_artifacts.get('rfecv', None)
        xgb_pkl_loaded = True
    except Exception as e:
        print(f"    ⚠ XGB pkl failed to load ({type(e).__name__}): {e}")
        print("    ⚠ The Colab pkl was saved under numpy 2.x (numpy._core) — it cannot load on numpy 1.x.")
        print("    ⚠ Use a locally trained pkl instead, e.g. sampling_results/xgb_under.pkl")
        print("    → Skipping XGB (PKL), benchmarking remaining 3 models.")

    # --- Load ONNX Models ---
    print("\n[3] Loading ONNX models...")
    rf_onnx_session = ort.InferenceSession(RF_ONNX, providers=['CPUExecutionProvider'])
    xgb_onnx_session = ort.InferenceSession(XGB_ONNX, providers=['CPUExecutionProvider'])

    # --- Load ONNX preprocessing data ---
    with open(RF_TFIDF) as f:
        rf_tfidf_data = json.load(f)
    with open(XGB_TFIDF) as f:
        xgb_tfidf_data = json.load(f)
    with open(XGB_PREP) as f:
        xgb_prep_data = json.load(f)

    # --- Model Sizes ---
    rf_pkl_size  = os.path.getsize(RF_PKL) / 1024
    xgb_pkl_size = os.path.getsize(XGB_PKL) / 1024
    rf_onnx_size = os.path.getsize(RF_ONNX) / 1024
    xgb_onnx_size = os.path.getsize(XGB_ONNX) / 1024
    xgb_prep_size = os.path.getsize(XGB_PREP) / 1024

    # ============================================================
    # PREDICTIONS — ALL 4 MODELS
    # ============================================================

    sample_count = min(args.latency_samples, len(test_urls))
    sample_idx = np.random.RandomState(42).choice(len(test_urls), sample_count, replace=False)
    results = {}

    # ----- MODEL 1: RF PKL -----
    print("\n[4] RF (Pickle) — predicting...")
    t0 = time.time()
    rf_pkl_preds = rf_pkl_pipeline.predict(test_urls)
    rf_pkl_total_time = time.time() - t0

    # Probability for confidence
    rf_pkl_probs = rf_pkl_pipeline.predict_proba(test_urls)

    # Per-sample latency (full pipeline: features + classifier)
    print(f"    Measuring latency ({sample_count} samples, full pipeline)...")
    rf_pkl_latencies = []
    for idx in sample_idx:
        url_arr = [test_urls[idx]]
        t0 = time.perf_counter()
        rf_pkl_pipeline.predict(url_arr)
        rf_pkl_latencies.append((time.perf_counter() - t0) * 1000)
    rf_pkl_latencies = np.array(rf_pkl_latencies)
    print(f"    Done. Total predict time: {rf_pkl_total_time:.1f}s")

    results['RF (PKL)'] = {
        'labels': rf_pkl_preds, 'latencies': rf_pkl_latencies,
        'size_kb': rf_pkl_size
    }

    # ----- MODEL 2: XGBoost PKL -----
    if xgb_pkl_loaded:
        print("\n[5] XGBoost (Pickle) — extracting features...")
        t0 = time.time()

        # Feature extraction using the saved feature_extractor
        X_test_xgb_raw = xgb_pkl_feature_extractor.transform(test_urls)
        if hasattr(X_test_xgb_raw, 'toarray'):
            X_test_xgb_raw = X_test_xgb_raw.toarray()

        # Apply preprocessing pipeline: scaler → kbest → importance mask → pca
        X_test_scaled = xgb_pkl_scaler.transform(X_test_xgb_raw)
        X_test_kbest = xgb_pkl_kbest.transform(X_test_scaled)
        if xgb_pkl_artifacts.get('important_mask') is not None:
            X_test_imp = X_test_kbest[:, xgb_pkl_artifacts['important_mask']]
        else:
            X_test_imp = X_test_kbest
        X_test_pca = xgb_pkl_pca.transform(X_test_imp)

        feat_time = time.time() - t0
        print(f"    Feature extraction: {feat_time:.1f}s")

        print("    Predicting...")
        t0 = time.time()
        xgb_pkl_preds = xgb_pkl_model.predict(X_test_pca)
        xgb_pkl_total_time = time.time() - t0
        xgb_pkl_probs = xgb_pkl_model.predict_proba(X_test_pca)

        # Per-sample latency (includes feature extraction + preprocessing + predict)
        print(f"    Measuring latency ({sample_count} samples, full pipeline)...")
        xgb_pkl_latencies = []
        for idx in sample_idx:
            url_arr = [test_urls[idx]]
            t0 = time.perf_counter()
            # Full pipeline for single URL
            x_raw = xgb_pkl_feature_extractor.transform(url_arr)
            if hasattr(x_raw, 'toarray'):
                x_raw = x_raw.toarray()
            x_s = xgb_pkl_scaler.transform(x_raw)
            x_k = xgb_pkl_kbest.transform(x_s)
            if xgb_pkl_artifacts.get('important_mask') is not None:
                x_i = x_k[:, xgb_pkl_artifacts['important_mask']]
            else:
                x_i = x_k
            x_p = xgb_pkl_pca.transform(x_i)
            xgb_pkl_model.predict(x_p)
            xgb_pkl_latencies.append((time.perf_counter() - t0) * 1000)
        xgb_pkl_latencies = np.array(xgb_pkl_latencies)
        print(f"    Done. Predict time: {xgb_pkl_total_time:.1f}s")

        results['XGB (PKL)'] = {
            'labels': xgb_pkl_preds, 'latencies': xgb_pkl_latencies,
            'size_kb': xgb_pkl_size
        }
    else:
        print("\n[5] XGBoost (Pickle) — SKIPPED (failed to load)")

    # ----- MODEL 3: RF ONNX -----
    print("\n[6] RF (ONNX) — extracting features...")
    CHUNK = 10000
    rf_onnx_features = []
    t0 = time.time()
    for i in range(0, len(test_urls), CHUNK):
        chunk = test_urls[i:i+CHUNK]
        rf_onnx_features.append(extract_features_onnx(chunk, rf_tfidf_data))
        if (i + CHUNK) % 20000 < CHUNK:
            print(f"      {min(i+CHUNK, len(test_urls))}/{len(test_urls)}...")
    X_rf_onnx = np.vstack(rf_onnx_features).astype(np.float32)
    del rf_onnx_features
    feat_time = time.time() - t0
    print(f"    Features: {X_rf_onnx.shape}, extraction: {feat_time:.1f}s")

    print("    Predicting (batch)...")
    rf_onnx_input = rf_onnx_session.get_inputs()[0].name
    rf_onnx_labels = []
    rf_onnx_probs = []
    for i in range(0, len(X_rf_onnx), 1000):
        batch = X_rf_onnx[i:i+1000]
        res = rf_onnx_session.run(None, {rf_onnx_input: batch})
        rf_onnx_labels.extend(res[0].flatten().tolist())
        rf_onnx_probs.extend(res[1].tolist())
    rf_onnx_labels = np.array(rf_onnx_labels, dtype=int)

    # Latency: full pipeline per URL (feature extraction + model) and model-only
    print(f"    Measuring latency ({sample_count} samples, full pipeline + model-only)...")
    rf_onnx_latencies = []
    rf_onnx_model_latencies = []
    for idx in sample_idx:
        t0 = time.perf_counter()
        feats = extract_features_onnx([test_urls[idx]], rf_tfidf_data)
        rf_onnx_session.run(None, {rf_onnx_input: feats})
        rf_onnx_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        rf_onnx_session.run(None, {rf_onnx_input: X_rf_onnx[idx:idx+1]})
        rf_onnx_model_latencies.append((time.perf_counter() - t0) * 1000)
    rf_onnx_latencies = np.array(rf_onnx_latencies)
    rf_onnx_model_latencies = np.array(rf_onnx_model_latencies)

    results['RF (ONNX)'] = {
        'labels': rf_onnx_labels, 'latencies': rf_onnx_latencies,
        'model_latencies': rf_onnx_model_latencies,
        'size_kb': rf_onnx_size
    }

    # ----- MODEL 4: XGBoost ONNX -----
    print("\n[7] XGBoost (ONNX) — extracting features...")
    xgb_onnx_features = []
    t0 = time.time()
    for i in range(0, len(test_urls), CHUNK):
        chunk = test_urls[i:i+CHUNK]
        raw = extract_features_onnx(chunk, xgb_tfidf_data)
        xgb_onnx_features.append(preprocess_xgb(raw, xgb_prep_data))
        if (i + CHUNK) % 20000 < CHUNK:
            print(f"      {min(i+CHUNK, len(test_urls))}/{len(test_urls)}...")
    X_xgb_onnx = np.vstack(xgb_onnx_features).astype(np.float32)
    del xgb_onnx_features
    feat_time = time.time() - t0
    print(f"    Features: {X_xgb_onnx.shape}, extraction: {feat_time:.1f}s")

    print("    Predicting (batch)...")
    xgb_onnx_input = xgb_onnx_session.get_inputs()[0].name
    xgb_onnx_labels = []
    xgb_onnx_probs = []
    for i in range(0, len(X_xgb_onnx), 1000):
        batch = X_xgb_onnx[i:i+1000]
        res = xgb_onnx_session.run(None, {xgb_onnx_input: batch})
        xgb_onnx_labels.extend(res[0].flatten().tolist())
        xgb_onnx_probs.extend(res[1].tolist())
    xgb_onnx_labels = np.array(xgb_onnx_labels, dtype=int)

    # Latency: full pipeline per URL (features + preprocessing + model) and model-only
    print(f"    Measuring latency ({sample_count} samples, full pipeline + model-only)...")
    xgb_onnx_latencies = []
    xgb_onnx_model_latencies = []
    for idx in sample_idx:
        t0 = time.perf_counter()
        raw = extract_features_onnx([test_urls[idx]], xgb_tfidf_data)
        feats = preprocess_xgb(raw, xgb_prep_data)
        xgb_onnx_session.run(None, {xgb_onnx_input: feats})
        xgb_onnx_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        xgb_onnx_session.run(None, {xgb_onnx_input: X_xgb_onnx[idx:idx+1]})
        xgb_onnx_model_latencies.append((time.perf_counter() - t0) * 1000)
    xgb_onnx_latencies = np.array(xgb_onnx_latencies)
    xgb_onnx_model_latencies = np.array(xgb_onnx_model_latencies)

    results['XGB (ONNX)'] = {
        'labels': xgb_onnx_labels, 'latencies': xgb_onnx_latencies,
        'model_latencies': xgb_onnx_model_latencies,
        'size_kb': xgb_onnx_size + xgb_prep_size
    }

    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    model_names = ['RF (PKL)', 'RF (ONNX)', 'XGB (PKL)', 'XGB (ONNX)']
    model_names = [n for n in model_names if n in results]  # only models that ran

    for name in model_names:
        r = results[name]
        print(f"\n--- {name} ---")
        print(classification_report(y_true, r['labels'], target_names=['Legitimate', 'Phishing'], digits=4))

    # Confusion matrices
    for name in model_names:
        cm = confusion_matrix(y_true, results[name]['labels'])
        print(f"Confusion Matrix ({name}):  TN={cm[0][0]:>6} FP={cm[0][1]:>6} | FN={cm[1][0]:>6} TP={cm[1][1]:>6}")

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE (All 4 Models)")
    print(f"{'=' * 80}")
    header = f"{'Metric':<25}"
    for name in model_names:
        header += f" {name:>12}"
    print(header)
    print("-" * 75)

    # Compute metrics for each model
    metrics = {}
    for name in model_names:
        r = results[name]
        cm = confusion_matrix(y_true, r['labels'])
        metrics[name] = {
            'acc': accuracy_score(y_true, r['labels']),
            'prec': precision_score(y_true, r['labels']),
            'rec': recall_score(y_true, r['labels']),
            'f1': f1_score(y_true, r['labels']),
            'fpr': cm[0][1] / (cm[0][0] + cm[0][1]),
            'lat_avg': r['latencies'].mean(),
            'lat_p95': np.percentile(r['latencies'], 95),
            'lat_max': r['latencies'].max(),
            'lat_model_avg': r['model_latencies'].mean() if 'model_latencies' in r else None,
            'size': r['size_kb'],
        }

    rows = [
        ('Accuracy',           'acc',     '.4f'),
        ('Precision (Phish)',  'prec',    '.4f'),
        ('Recall (Phish)',     'rec',     '.4f'),
        ('F1-Score (Phish)',   'f1',      '.4f'),
        ('False Positive Rate','fpr',     '.4f'),
    ]
    for label, key, fmt in rows:
        line = f"{label:<25}"
        for name in model_names:
            line += f" {metrics[name][key]:>12{fmt}}"
        print(line)

    print("-" * 75)
    lat_rows = [
        ('Avg Latency (ms)',  'lat_avg', '.3f'),
        ('P95 Latency (ms)',  'lat_p95', '.3f'),
        ('Max Latency (ms)',  'lat_max', '.3f'),
        ('Avg Model-only (ms)', 'lat_model_avg', '.3f'),
    ]
    for label, key, fmt in lat_rows:
        line = f"{label:<25}"
        for name in model_names:
            val = metrics[name].get(key)
            line += f" {val:>12{fmt}}" if val is not None else f" {'-':>12}"
        print(line)

    print("-" * 75)
    line = f"{'Model Size (KB)':<25}"
    for name in model_names:
        line += f" {metrics[name]['size']:>12.0f}"
    print(line)

    print(f"{'Process Memory (MB)':<25} {mem_mb:>12.1f}")
    print(f"{'Test Samples':<25} {len(y_true):>12}")

    # ============================================================
    # PKL vs ONNX AGREEMENT (same model, two formats)
    # ============================================================
    print(f"\n{'=' * 80}")
    print("PKL vs ONNX AGREEMENT")
    print(f"{'=' * 80}")
    print("  Note: the ONNX feature path mirrors utils.js and filters empty tokens,")
    print("  while sklearn counts them ('' is in the vocabulary), so tiny divergences")
    print("  (~0.2%) are expected for XGBoost; RF is typically identical.")

    if 'RF (PKL)' in results and 'RF (ONNX)' in results:
        rf_match = np.sum(results['RF (PKL)']['labels'] == results['RF (ONNX)']['labels'])
        print(f"  RF:  {rf_match}/{len(y_true)} predictions agree ({rf_match/len(y_true)*100:.4f}%)")
        if rf_match < len(y_true):
            print("       (if the two files are different variants, differences are expected — see MODELS.md)")

    if 'XGB (PKL)' in results and 'XGB (ONNX)' in results:
        xgb_match = np.sum(results['XGB (PKL)']['labels'] == results['XGB (ONNX)']['labels'])
        print(f"  XGB: {xgb_match}/{len(y_true)} predictions agree ({xgb_match/len(y_true)*100:.4f}%)")
        if xgb_match < len(y_true):
            print("       (small divergence expected from the empty-token emulation described above)")
    elif 'XGB (PKL)' not in results:
        print("  XGB: PKL not available — agreement check skipped")

    # ============================================================
    # THESIS SUCCESS CRITERIA
    # ============================================================
    print(f"\n{'=' * 80}")
    print("THESIS SUCCESS CRITERIA CHECK")
    print(f"{'=' * 80}")
    criteria = [
        ("Accuracy > 95%",    lambda m: m['acc'] > 0.95),
        ("Recall > 90%",      lambda m: m['rec'] > 0.90),
        ("F1-Score > 90%",    lambda m: m['f1'] > 0.90),
        ("FPR < 2%",          lambda m: m['fpr'] < 0.02),
        ("Avg Latency < 100ms", lambda m: m['lat_avg'] < 100),
        ("Model < 2MB",       lambda m: m['size'] < 2048),
    ]
    for crit_name, check_fn in criteria:
        line = f"  {crit_name:<25}"
        for name in model_names:
            if crit_name == "Model < 2MB" and "(PKL)" in name:
                line += f" {name}: {'n/a':>4}  "
            else:
                passed = check_fn(metrics[name])
                line += f" {name}: {'PASS' if passed else 'FAIL':>4}  "
        print(line)

    print(f"\nBenchmark complete.")


if __name__ == "__main__":
    main()
