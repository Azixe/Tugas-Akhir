"""
Automated Performance Benchmark - RF vs XGBoost (ONNX)
======================================================
Uses the EXPORTED TF-IDF vocabularies (same as the extension)
to build feature vectors, ensuring exact alignment with models.
"""

import pandas as pd
import numpy as np
import math
import time
import json
import os
import psutil
from collections import Counter

import onnxruntime as ort
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# ============================================================
# FEATURE EXTRACTION (matches JS extension exactly)
# ============================================================

def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in Counter(data).values():
        p_x = x / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy

def make_tokens(url):
    """Tokenizer matching JS and Python training code."""
    tokens_by_slash = str(url).split('/')
    total_tokens = []
    for part in tokens_by_slash:
        by_dash = part.split('-')
        by_dot = []
        for t in by_dash:
            by_dot.extend(t.split('.'))
        total_tokens.extend(by_dash)
        total_tokens.extend(by_dot)
    total_tokens = list(set(total_tokens))
    if 'com' in total_tokens: total_tokens.remove('com')
    if 'www' in total_tokens: total_tokens.remove('www')
    return [t for t in total_tokens if t]

def build_tfidf_vector(url, tfidf_data):
    """Build TF-IDF vector using exported vocabulary (matches JS)."""
    tokens = make_tokens(url.lower())
    vocab = tfidf_data['vocabulary']
    idf = tfidf_data['idf']
    n_features = len(vocab)

    vec = np.zeros(n_features, dtype=np.float32)

    # Term frequency
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1

    # Apply IDF and L2 normalize
    for i in range(n_features):
        vec[i] *= idf[i]

    norm = np.sqrt(np.sum(vec * vec))
    if norm > 0:
        vec /= norm

    return vec

def structural_features(url):
    """9 structural features matching JS and Python training code."""
    s = url.lower()
    length = len(s)
    dot_count = s.count('.')
    slash_count = s.count('/')
    dash_count = s.count('-')
    at_count = s.count('@')
    digit_count = sum(c.isdigit() for c in s)
    digit_ratio = digit_count / length if length > 0 else 0
    entropy = shannon_entropy(s)

    common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']
    is_common_tld = 0
    for tld in common_tlds:
        if s.endswith(tld) or s.endswith(tld + '/'):
            is_common_tld = 1
            break

    clean = s.replace("https://", "").replace("http://", "").split('/')[0]
    subdomain_level = clean.count('.')

    return np.array([length, dot_count, slash_count, dash_count, at_count,
                     digit_ratio, entropy, is_common_tld, subdomain_level], dtype=np.float32)

def extract_features(urls, tfidf_data):
    """Extract full feature vector for a batch of URLs."""
    features = []
    for url in urls:
        tfidf_vec = build_tfidf_vector(url, tfidf_data)
        struct = structural_features(url)
        features.append(np.concatenate([tfidf_vec, struct]))
    return np.array(features, dtype=np.float32)

# ============================================================
# XGBoost PREPROCESSING
# ============================================================

def preprocess_xgb(features, prep_data):
    """Apply StandardScaler -> feature selection -> PCA (matches JS)."""
    mean = np.array(prep_data['scaler_mean'], dtype=np.float32)
    scale = np.array(prep_data['scaler_scale'], dtype=np.float32)
    scaled = (features - mean) / scale

    indices = prep_data['selected_feature_indices']
    selected = scaled[:, indices]

    pca_mean = np.array(prep_data['pca_mean'], dtype=np.float32)
    pca_components = np.array(prep_data['pca_components'], dtype=np.float32)
    pca_result = (selected - pca_mean) @ pca_components.T

    return pca_result.astype(np.float32)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AUTOMATED BENCHMARK: Random Forest vs XGBoost (ONNX)")
    print("=" * 70)

    EXT_DIR = 'phishingdetectorExt'
    RF_ONNX = os.path.join(EXT_DIR, 'phishing_rf.onnx')
    XGB_ONNX = os.path.join(EXT_DIR, 'phishing_xgb.onnx')
    RF_TFIDF = os.path.join(EXT_DIR, 'tfidf_data.json')
    XGB_TFIDF = os.path.join(EXT_DIR, 'tfidf_data_xgb.json')
    XGB_PREP = os.path.join(EXT_DIR, 'xgb_preprocessing.json')
    DATASET = 'URL dataset.csv'

    for f in [RF_ONNX, XGB_ONNX, RF_TFIDF, XGB_TFIDF, XGB_PREP, DATASET]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found!")
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
    print(f"    Balanced: {len(df_balanced)} URLs")

    # Same test split as training
    _, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])
    test_urls = test_df['url'].values
    y_true = test_df['label'].values
    print(f"    Test set: {len(test_urls)} URLs ({sum(y_true==0)} legit, {sum(y_true==1)} phishing)")

    # --- Load TF-IDF Data ---
    print("\n[2] Loading exported TF-IDF vocabularies...")
    with open(RF_TFIDF) as f:
        rf_tfidf_data = json.load(f)
    with open(XGB_TFIDF) as f:
        xgb_tfidf_data = json.load(f)
    with open(XGB_PREP) as f:
        xgb_prep = json.load(f)
    print(f"    RF vocab: {len(rf_tfidf_data['vocabulary'])} terms")
    print(f"    XGB vocab: {len(xgb_tfidf_data['vocabulary'])} terms")

    # --- Feature Extraction ---
    print("\n[3] Extracting features (using exported vocabularies)...")
    t0 = time.time()

    X_test_rf = extract_features(test_urls, rf_tfidf_data)
    print(f"    RF features: {X_test_rf.shape}")

    X_test_xgb_raw = extract_features(test_urls, xgb_tfidf_data)
    X_test_xgb = preprocess_xgb(X_test_xgb_raw, xgb_prep)
    print(f"    XGB features: {X_test_xgb.shape}")
    print(f"    Feature extraction: {time.time()-t0:.1f}s")

    # --- Load ONNX Models ---
    print("\n[4] Loading ONNX models...")
    rf_session = ort.InferenceSession(RF_ONNX, providers=['CPUExecutionProvider'])
    xgb_session = ort.InferenceSession(XGB_ONNX, providers=['CPUExecutionProvider'])

    rf_onnx_size = os.path.getsize(RF_ONNX) / 1024
    xgb_onnx_size = os.path.getsize(XGB_ONNX) / 1024
    xgb_prep_size = os.path.getsize(XGB_PREP) / 1024

    # --- Predictions ---
    print("\n[5] Running predictions (per-sample latency)...")

    def predict_all(session, X):
        """Predict all samples, measure per-sample latency."""
        input_name = session.get_inputs()[0].name
        labels = []
        probs = []
        latencies = []
        for i in range(len(X)):
            sample = X[i:i+1]
            t0 = time.perf_counter()
            results = session.run(None, {input_name: sample})
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            labels.append(int(results[0][0]))
            probs.append(results[1][0])
            if (i + 1) % 10000 == 0:
                print(f"      {i+1}/{len(X)} done...")
        return np.array(labels), np.array(probs), np.array(latencies)

    print("    RF predictions...")
    rf_labels, rf_probs, rf_latencies = predict_all(rf_session, X_test_rf)

    print("    XGBoost predictions...")
    xgb_labels, xgb_probs, xgb_latencies = predict_all(xgb_session, X_test_xgb)

    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_true, rf_labels, target_names=['Legitimate', 'Phishing'], digits=4))

    print("--- XGBoost Classification Report ---")
    print(classification_report(y_true, xgb_labels, target_names=['Legitimate', 'Phishing'], digits=4))

    rf_cm = confusion_matrix(y_true, rf_labels)
    xgb_cm = confusion_matrix(y_true, xgb_labels)

    print("--- Confusion Matrix (RF) ---")
    print(f"    TN={rf_cm[0][0]:>6}  FP={rf_cm[0][1]:>6}")
    print(f"    FN={rf_cm[1][0]:>6}  TP={rf_cm[1][1]:>6}")

    print("\n--- Confusion Matrix (XGBoost) ---")
    print(f"    TN={xgb_cm[0][0]:>6}  FP={xgb_cm[0][1]:>6}")
    print(f"    FN={xgb_cm[1][0]:>6}  TP={xgb_cm[1][1]:>6}")

    # Metrics
    rf_acc = accuracy_score(y_true, rf_labels)
    rf_prec_val = precision_score(y_true, rf_labels)
    rf_rec = recall_score(y_true, rf_labels)
    rf_f1 = f1_score(y_true, rf_labels)
    rf_fpr = rf_cm[0][1] / (rf_cm[0][0] + rf_cm[0][1])

    xgb_acc = accuracy_score(y_true, xgb_labels)
    xgb_prec_val = precision_score(y_true, xgb_labels)
    xgb_rec = recall_score(y_true, xgb_labels)
    xgb_f1 = f1_score(y_true, xgb_labels)
    xgb_fpr = xgb_cm[0][1] / (xgb_cm[0][0] + xgb_cm[0][1])

    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE (Thesis Table 3.3)")
    print(f"{'=' * 70}")
    print(f"{'Metric':<30} {'Random Forest':>15} {'XGBoost':>15}")
    print(f"{'-' * 62}")
    print(f"{'Accuracy':<30} {rf_acc:>14.4f} {xgb_acc:>14.4f}")
    print(f"{'Precision (Phishing)':<30} {rf_prec_val:>14.4f} {xgb_prec_val:>14.4f}")
    print(f"{'Recall (Phishing)':<30} {rf_rec:>14.4f} {xgb_rec:>14.4f}")
    print(f"{'F1-Score (Phishing)':<30} {rf_f1:>14.4f} {xgb_f1:>14.4f}")
    print(f"{'False Positive Rate':<30} {rf_fpr:>14.4f} {xgb_fpr:>14.4f}")
    print(f"{'-' * 62}")
    print(f"{'Avg Latency (ms)':<30} {rf_latencies.mean():>14.3f} {xgb_latencies.mean():>14.3f}")
    print(f"{'P95 Latency (ms)':<30} {np.percentile(rf_latencies, 95):>14.3f} {np.percentile(xgb_latencies, 95):>14.3f}")
    print(f"{'Max Latency (ms)':<30} {rf_latencies.max():>14.3f} {xgb_latencies.max():>14.3f}")
    print(f"{'-' * 62}")
    print(f"{'ONNX Size (KB)':<30} {rf_onnx_size:>14.0f} {xgb_onnx_size:>14.0f}")
    print(f"{'Total Size (KB)':<30} {rf_onnx_size:>14.0f} {xgb_onnx_size + xgb_prep_size:>14.0f}")
    print(f"{'Process Memory (MB)':<30} {mem_mb:>14.1f}")
    print(f"{'Test Samples':<30} {len(y_true):>14}")

    print(f"\n{'=' * 70}")
    print("THESIS SUCCESS CRITERIA CHECK")
    print(f"{'=' * 70}")
    criteria = [
        ("Accuracy > 95%", rf_acc > 0.95, xgb_acc > 0.95),
        ("Recall > 90%", rf_rec > 0.90, xgb_rec > 0.90),
        ("F1-Score > 90%", rf_f1 > 0.90, xgb_f1 > 0.90),
        ("FPR < 2%", rf_fpr < 0.02, xgb_fpr < 0.02),
        ("Avg Latency < 100ms", rf_latencies.mean() < 100, xgb_latencies.mean() < 100),
        ("Model < 2MB", rf_onnx_size < 2048, (xgb_onnx_size + xgb_prep_size) < 2048),
    ]
    for name, rf_pass, xgb_pass in criteria:
        rf_s = "PASS" if rf_pass else "FAIL"
        xgb_s = "PASS" if xgb_pass else "FAIL"
        print(f"  {name:<25} RF: {rf_s:<6} XGB: {xgb_s}")

    print(f"\nBenchmark complete.")
