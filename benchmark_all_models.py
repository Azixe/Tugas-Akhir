"""
Automated Performance Benchmark — PKL vs ONNX × RF vs XGBoost
==============================================================
Tests all 4 model variants on the full dataset:
  1. RF (pickle / scikit-learn)
  2. RF (ONNX)
  3. XGBoost (pickle / native)
  4. XGBoost (ONNX)

Uses exported TF-IDF vocabularies for ONNX models, and the
original pipeline objects for PKL models.
"""

import pandas as pd
import numpy as np
import math
import time
import json
import os
import sys
import psutil
from collections import Counter

import onnxruntime as ort
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# SHARED HELPERS (must be defined before loading pkl)
# ============================================================

def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in Counter(data).values():
        p_x = x / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy

class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']
        for url in X:
            s_url = str(url).lower()
            length = len(s_url)
            dot_count = s_url.count('.')
            slash_count = s_url.count('/')
            dash_count = s_url.count('-')
            at_count = s_url.count('@')
            digit_count = sum(c.isdigit() for c in s_url)
            digit_ratio = digit_count / length if length > 0 else 0
            entropy = shannon_entropy(s_url)
            is_common_tld = 0
            for tld in common_tlds:
                if s_url.endswith(tld) or s_url.endswith(tld + '/'):
                    is_common_tld = 1
                    break
            try:
                clean = s_url.replace("https://", "").replace("http://", "").split('/')[0]
                subdomain_level = clean.count('.')
            except:
                subdomain_level = 0
            features.append([length, dot_count, slash_count, dash_count, at_count,
                             digit_ratio, entropy, is_common_tld, subdomain_level])
        return np.array(features)

def make_tokens(f):
    tokens_by_slash = str(f).encode('utf-8').decode('utf-8').split('/')
    total_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')
        tokens_dot = []
        for j in range(0, len(tokens)):
            temp_tokens = str(tokens[j]).split('.')
            tokens_dot = tokens_dot + temp_tokens
        total_tokens = total_tokens + tokens + tokens_dot
    total_tokens = list(set(total_tokens))
    if 'com' in total_tokens: total_tokens.remove('com')
    if 'www' in total_tokens: total_tokens.remove('www')
    return total_tokens

# ============================================================
# ONNX FEATURE EXTRACTION (from exported JSON vocabularies)
# ============================================================

def build_tfidf_vector(url, tfidf_data):
    tokens = make_tokens(url.lower())
    vocab = tfidf_data['vocabulary']
    idf = tfidf_data['idf']
    n_features = len(vocab)
    vec = np.zeros(n_features, dtype=np.float32)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    for i in range(n_features):
        vec[i] *= idf[i]
    norm = np.sqrt(np.sum(vec * vec))
    if norm > 0:
        vec /= norm
    return vec

def structural_features(url):
    s = url.lower()
    length = len(s)
    common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']
    is_common_tld = 0
    for tld in common_tlds:
        if s.endswith(tld) or s.endswith(tld + '/'):
            is_common_tld = 1; break
    clean = s.replace("https://", "").replace("http://", "").split('/')[0]
    return np.array([length, s.count('.'), s.count('/'), s.count('-'), s.count('@'),
                     sum(c.isdigit() for c in s) / length if length > 0 else 0,
                     shannon_entropy(s), is_common_tld, clean.count('.')], dtype=np.float32)

def extract_features_onnx(urls, tfidf_data):
    features = []
    for url in urls:
        tfidf_vec = build_tfidf_vector(url, tfidf_data)
        struct = structural_features(url)
        features.append(np.concatenate([tfidf_vec, struct]))
    return np.array(features, dtype=np.float32)

def preprocess_xgb(features, prep_data):
    mean = np.array(prep_data['scaler_mean'], dtype=np.float32)
    scale = np.array(prep_data['scaler_scale'], dtype=np.float32)
    scaled = (features - mean) / scale
    selected = scaled[:, prep_data['selected_feature_indices']]
    pca_mean = np.array(prep_data['pca_mean'], dtype=np.float32)
    pca_components = np.array(prep_data['pca_components'], dtype=np.float32)
    return ((selected - pca_mean) @ pca_components.T).astype(np.float32)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK: PKL vs ONNX × RF vs XGBoost (All 4 Models)")
    print("=" * 70)

    # --- Paths ---
    EXT_DIR = 'phishingdetectorExt'
    RF_ONNX  = os.path.join(EXT_DIR, 'phishing_rf.onnx')
    XGB_ONNX = os.path.join(EXT_DIR, 'phishing_xgb.onnx')
    RF_TFIDF  = os.path.join(EXT_DIR, 'tfidf_data.json')
    XGB_TFIDF = os.path.join(EXT_DIR, 'tfidf_data_xgb.json')
    XGB_PREP  = os.path.join(EXT_DIR, 'xgb_preprocessing.json')
    RF_PKL   = os.path.join('rf_models', 'phishing_optimized.pkl')
    XGB_PKL  = os.path.join('Xgboost', 'phishing_xgb_pipeline.pkl')
    DATASET  = 'URL dataset.csv'

    missing = [f for f in [RF_ONNX, XGB_ONNX, RF_TFIDF, XGB_TFIDF, XGB_PREP, RF_PKL, XGB_PKL, DATASET]
               if not os.path.exists(f)]
    if missing:
        for f in missing:
            print(f"  ERROR: {f} not found!")
        exit()

    # --- Load Dataset ---
    print("\n[1] Loading dataset...")
    df = pd.read_csv(DATASET)
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    df['label'] = df['type'].map({'phishing': 1, 'legitimate': 0})

    # Undersample (same as training)
    df_phish = df[df['label'] == 1]
    df_legit = df[df['label'] == 0].sample(n=len(df_phish), random_state=42)
    df_balanced = pd.concat([df_legit, df_phish]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Same test split as training (80/20)
    _, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])
    test_urls = test_df['url'].values
    y_true = test_df['label'].values
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
    # May fail if saved with newer numpy (Colab numpy 2.x vs local 1.x)
    xgb_pkl_loaded = False
    print("    Loading XGBoost pkl...")
    try:
        # Try monkey-patching numpy._core for cross-version compatibility
        import numpy.core
        if not hasattr(np, '_core'):
            sys.modules['numpy._core'] = np.core
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
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
        print(f"    ⚠ XGB pkl failed to load: {e}")
        print(f"    ⚠ This pkl was saved with a newer numpy version (Colab).")
        print(f"    ⚠ To fix: re-export the pkl from Colab with matching numpy,")
        print(f"    ⚠ or run the benchmark on Colab instead.")
        print(f"    → Skipping XGB (PKL), benchmarking remaining 3 models.")

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

    LATENCY_SAMPLES = 1000
    results = {}

    # ----- MODEL 1: RF PKL -----
    print("\n[4] RF (Pickle) — predicting...")
    t0 = time.time()
    rf_pkl_preds = rf_pkl_pipeline.predict(test_urls)
    rf_pkl_total_time = time.time() - t0

    # Probability for confidence
    rf_pkl_probs = rf_pkl_pipeline.predict_proba(test_urls)

    # Per-sample latency
    print("    Measuring latency (1000 samples)...")
    rf_pkl_latencies = []
    sample_idx = np.random.choice(len(test_urls), LATENCY_SAMPLES, replace=False)
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

        # Apply preprocessing pipeline: scaler → kbest → (rfecv?) → pca
        X_test_scaled = xgb_pkl_scaler.transform(X_test_xgb_raw)
        X_test_kbest = xgb_pkl_kbest.transform(X_test_scaled)
        if xgb_pkl_rfecv is not None:
            X_test_rfecv = xgb_pkl_rfecv.transform(X_test_kbest)
        else:
            X_test_rfecv = X_test_kbest
        X_test_pca = xgb_pkl_pca.transform(X_test_rfecv)

        feat_time = time.time() - t0
        print(f"    Feature extraction: {feat_time:.1f}s")

        print("    Predicting...")
        t0 = time.time()
        xgb_pkl_preds = xgb_pkl_model.predict(X_test_pca)
        xgb_pkl_total_time = time.time() - t0
        xgb_pkl_probs = xgb_pkl_model.predict_proba(X_test_pca)

        # Per-sample latency (includes feature extraction + preprocessing + predict)
        print("    Measuring latency (1000 samples, full pipeline)...")
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
            if xgb_pkl_rfecv is not None:
                x_r = xgb_pkl_rfecv.transform(x_k)
            else:
                x_r = x_k
            x_p = xgb_pkl_pca.transform(x_r)
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

    # Latency
    print("    Measuring latency (1000 samples)...")
    rf_onnx_latencies = []
    for idx in sample_idx:
        sample = X_rf_onnx[idx:idx+1]
        t0 = time.perf_counter()
        rf_onnx_session.run(None, {rf_onnx_input: sample})
        rf_onnx_latencies.append((time.perf_counter() - t0) * 1000)
    rf_onnx_latencies = np.array(rf_onnx_latencies)

    results['RF (ONNX)'] = {
        'labels': rf_onnx_labels, 'latencies': rf_onnx_latencies,
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

    # Latency
    print("    Measuring latency (1000 samples)...")
    xgb_onnx_latencies = []
    for idx in sample_idx:
        sample = X_xgb_onnx[idx:idx+1]
        t0 = time.perf_counter()
        xgb_onnx_session.run(None, {xgb_onnx_input: sample})
        xgb_onnx_latencies.append((time.perf_counter() - t0) * 1000)
    xgb_onnx_latencies = np.array(xgb_onnx_latencies)

    results['XGB (ONNX)'] = {
        'labels': xgb_onnx_labels, 'latencies': xgb_onnx_latencies,
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
    ]
    for label, key, fmt in lat_rows:
        line = f"{label:<25}"
        for name in model_names:
            line += f" {metrics[name][key]:>12{fmt}}"
        print(line)

    print("-" * 75)
    line = f"{'Model Size (KB)':<25}"
    for name in model_names:
        line += f" {metrics[name]['size']:>12.0f}"
    print(line)

    print(f"{'Process Memory (MB)':<25} {mem_mb:>12.1f}")
    print(f"{'Test Samples':<25} {len(y_true):>12}")

    # ============================================================
    # PKL vs ONNX PARITY CHECK
    # ============================================================
    print(f"\n{'=' * 80}")
    print("PKL vs ONNX PARITY CHECK")
    print(f"{'=' * 80}")

    if 'RF (PKL)' in results and 'RF (ONNX)' in results:
        rf_match = np.sum(results['RF (PKL)']['labels'] == results['RF (ONNX)']['labels'])
        print(f"  RF:  PKL vs ONNX agree on {rf_match}/{len(y_true)} predictions ({rf_match/len(y_true)*100:.2f}%)")
        if rf_match == len(y_true):
            print("  ✓ RF models are perfectly equivalent")
        else:
            print(f"  ⚠ RF models differ on {len(y_true) - rf_match} predictions")

    if 'XGB (PKL)' in results and 'XGB (ONNX)' in results:
        xgb_match = np.sum(results['XGB (PKL)']['labels'] == results['XGB (ONNX)']['labels'])
        print(f"  XGB: PKL vs ONNX agree on {xgb_match}/{len(y_true)} predictions ({xgb_match/len(y_true)*100:.2f}%)")
        if xgb_match == len(y_true):
            print("  ✓ XGB models are perfectly equivalent")
        else:
            print(f"  ⚠ XGB models differ on {len(y_true) - xgb_match} predictions")
    elif 'XGB (PKL)' not in results:
        print("  XGB: PKL not available — parity check skipped")

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
            passed = check_fn(metrics[name])
            line += f" {name}: {'PASS' if passed else 'FAIL':>4}  "
        print(line)

    print(f"\nBenchmark complete.")
