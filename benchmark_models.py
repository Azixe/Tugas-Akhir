"""
Automated Performance Benchmark - RF vs XGBoost (ONNX)
======================================================
Uses the EXPORTED TF-IDF vocabularies (same as the extension)
to build feature vectors, ensuring exact alignment with models.
"""

import pandas as pd
import numpy as np
import time
import json
import os
import sys
import psutil

import onnxruntime as ort
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

# ============================================================
# FEATURE EXTRACTION (single source of truth: features.py)
# ============================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (  # noqa: E402
    extract_features_onnx as extract_features,
    preprocess_xgb,
)

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

    # --- Load FULL Dataset ---
    print("\n[1] Loading FULL dataset (all rows, imbalanced)...")
    df = pd.read_csv(DATASET)
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    df['label'] = df['type'].map({'phishing': 1, 'legitimate': 0})

    test_urls = df['url'].values
    y_true = df['label'].values
    n_legit = sum(y_true == 0)
    n_phish = sum(y_true == 1)
    print(f"    Total: {len(test_urls)} URLs ({n_legit} legit, {n_phish} phishing, ratio {n_legit/n_phish:.1f}:1)")

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

    # --- Feature Extraction (in chunks to manage memory) ---
    print("\n[3] Extracting features (using exported vocabularies)...")
    print("    This may take several minutes for 450K URLs...")
    t0 = time.time()
    CHUNK = 10000

    rf_chunks = []
    xgb_chunks = []
    for i in range(0, len(test_urls), CHUNK):
        chunk_urls = test_urls[i:i+CHUNK]
        rf_chunks.append(extract_features(chunk_urls, rf_tfidf_data))
        xgb_raw = extract_features(chunk_urls, xgb_tfidf_data)
        xgb_chunks.append(preprocess_xgb(xgb_raw, xgb_prep).astype(np.float32))
        print(f"      {min(i+CHUNK, len(test_urls))}/{len(test_urls)} done...")

    X_test_rf = np.vstack(rf_chunks).astype(np.float32)
    X_test_xgb = np.vstack(xgb_chunks).astype(np.float32)
    del rf_chunks, xgb_chunks

    print(f"    RF features: {X_test_rf.shape}, XGB features: {X_test_xgb.shape}")
    print(f"    Feature extraction: {(time.time()-t0)/60:.1f} min")

    # --- Load ONNX Models ---
    print("\n[4] Loading ONNX models...")
    rf_session = ort.InferenceSession(RF_ONNX, providers=['CPUExecutionProvider'])
    xgb_session = ort.InferenceSession(XGB_ONNX, providers=['CPUExecutionProvider'])

    rf_onnx_size = os.path.getsize(RF_ONNX) / 1024
    xgb_onnx_size = os.path.getsize(XGB_ONNX) / 1024
    xgb_prep_size = os.path.getsize(XGB_PREP) / 1024

    # --- Predictions (batch + latency sample) ---
    print("\n[5] Running predictions...")

    def predict_batch(session, X, batch_size=1000):
        """Predict in batches for speed."""
        input_name = session.get_inputs()[0].name
        all_labels = []
        all_probs = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            results = session.run(None, {input_name: batch})
            all_labels.extend(results[0].flatten().tolist())
            all_probs.extend(results[1].tolist())
            if (i + batch_size) % 50000 < batch_size:
                print(f"      {min(i+batch_size, len(X))}/{len(X)} done...")
        return np.array(all_labels, dtype=int), np.array(all_probs)

    def measure_latency(session, X, n_samples=1000):
        """Measure per-sample latency on a random sample."""
        input_name = session.get_inputs()[0].name
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        latencies = []
        for idx in indices:
            sample = X[idx:idx+1]
            t0 = time.perf_counter()
            session.run(None, {input_name: sample})
            latencies.append((time.perf_counter() - t0) * 1000)
        return np.array(latencies)

    print("    RF predictions (batch)...")
    rf_labels, rf_probs = predict_batch(rf_session, X_test_rf)
    print("    RF latency sample (1000 URLs)...")
    rf_latencies = measure_latency(rf_session, X_test_rf)

    print("    XGBoost predictions (batch)...")
    xgb_labels, xgb_probs = predict_batch(xgb_session, X_test_xgb)
    print("    XGBoost latency sample (1000 URLs)...")
    xgb_latencies = measure_latency(xgb_session, X_test_xgb)

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
