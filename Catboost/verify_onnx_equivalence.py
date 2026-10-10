"""Persistent evidence for the CatBoost ONNX deployment claims.

Produces ``sampling_results/catboost_onnx_verification.md`` with:

1. Native (``cb_full.pkl``) vs deployed ONNX
   (``phishingdetectorExt/phishing_catboost.onnx``) parity on 300 test URLs:
   label mismatches and max |Δp|.
2. ONNX metrics on the full common test set (90,036 URLs) vs the native
   metrics recorded in ``catboost_metrics.json``.
3. Extension package size: folder bytes and deflate-zip bytes (P-04 basis,
   criterion < 30 MB).

Usage (repo root):
    python Catboost/verify_onnx_equivalence.py
"""

from __future__ import annotations

import json
import os
import sys
import zipfile

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_pipelines  # noqa: E402
import sampling  # noqa: E402

OUTDIR = 'sampling_results'
ONNX_PATH = os.path.join('phishingdetectorExt', 'phishing_catboost.onnx')
EXT_DIR = 'phishingdetectorExt'
CHUNK = 8192


def folder_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def zip_bytes(path: str) -> int:
    tmp = os.path.join(os.environ.get('TEMP', '/tmp'), 'ext_size_probe.zip')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, _, files in os.walk(path):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, path))
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size


def main():
    art = joblib.load(os.path.join(OUTDIR, 'cb_full.pkl'))
    fx, native = art['feature_extractor'], art['model']
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name

    df = sampling.load_clean_dataset()
    _, test = sampling.common_split(df)
    urls, y = test['url'].values, test['label'].values

    # 1) parity on 300 URLs
    Xs = ml_pipelines._to_dense(fx.transform(urls[:300])).astype(np.float32)
    p_native = native.predict_proba(Xs)[:, 1]
    out = sess.run(None, {name: Xs})
    labels, probs = np.asarray(out[0]), np.asarray(out[1])[:, 1]
    agree = float((labels == (p_native > 0.5)).mean() * 100)
    maxdiff = float(np.abs(probs - p_native).max())

    # 2) full test set via ONNX
    preds = []
    for i in range(0, len(urls), CHUNK):
        X = ml_pipelines._to_dense(fx.transform(urls[i:i + CHUNK])).astype(np.float32)
        preds.append(np.asarray(sess.run(None, {name: X})[0]).astype(int))
    y_pred = np.concatenate(preds)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    onnx_metrics = {
        'accuracy': round(float(accuracy_score(y, y_pred)) * 100, 2),
        'precision': round(float(precision_score(y, y_pred)) * 100, 2),
        'recall': round(float(recall_score(y, y_pred)) * 100, 2),
        'f1': round(float(f1_score(y, y_pred)) * 100, 2),
        'fpr': round(float(fp / (fp + tn)) * 100, 4),
        'fp': int(fp), 'fn': int(fn),
    }
    native_metrics = json.load(open(os.path.join(OUTDIR, 'catboost_metrics.json')))[0]

    # 3) extension size (P-04)
    fb = folder_bytes(EXT_DIR)
    zb = zip_bytes(EXT_DIR)

    lines = [
        "# CatBoost ONNX deployment verification (persistent evidence)",
        "",
        f"Deployed file: `{ONNX_PATH}`; native artifact: `{OUTDIR}/cb_full.pkl`.",
        "Protocol: common split, seed 42 (test set = 90,036 URLs).",
        "",
        "## 1. Native ↔ ONNX parity (first 300 test URLs)",
        "",
        "| Metric | Value |", "|---|---:|",
        f"| Label agreement | {agree:.1f}% |",
        f"| Max abs prob. diff | {maxdiff:.2e} |",
        "",
        "## 2. ONNX metrics on the full test set vs native",
        "",
        "| Metric | Native (pkl) | ONNX (deployed) |",
        "|---|---:|---:|",
        f"| Accuracy (%) | {native_metrics['accuracy']*100:.2f} | {onnx_metrics['accuracy']:.2f} |",
        f"| Precision (%) | {native_metrics['precision_phish']*100:.2f} | {onnx_metrics['precision']:.2f} |",
        f"| Recall (%) | {native_metrics['recall_phish']*100:.2f} | {onnx_metrics['recall']:.2f} |",
        f"| F1 (%) | {native_metrics['f1_phish']*100:.2f} | {onnx_metrics['f1']:.2f} |",
        f"| FPR (%) | {native_metrics['fpr']*100:.4f} | {onnx_metrics['fpr']:.4f} |",
        f"| FP / FN | {native_metrics['fp']} / {native_metrics['fn']} | {onnx_metrics['fp']} / {onnx_metrics['fn']} |",
        "",
        "## 3. Extension package size (P-04, criterion < 30 MB)",
        "",
        "| Measure | Size |", "|---|---:|",
        f"| Folder on disk | {fb/1024/1024:.1f} MB |",
        f"| Deflate zip (≈ .crx) | {zb/1024/1024:.2f} MB |",
        "",
        "> n=300 for parity (sampled), full 90,036 for metrics.",
        "",
    ]
    path = os.path.join(OUTDIR, 'catboost_onnx_verification.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == '__main__':
    main()
