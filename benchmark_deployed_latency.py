"""Latency of the four deployed-configuration artifacts on the shared 1000-URL sample.

Full pipeline per URL (feature extraction + preprocessing + model), the same
definition used by `benchmark_all_models.py`, measured with `time.perf_counter`.
This gives the Min / Max / Avg numbers used in Tabel 4.3 of the thesis for the
deployed pair (RF full dataset, XGBoost full dataset).

Usage (from the repo root):
    python benchmark_deployed_latency.py [--samples 1000] [--warmup 20]

Outputs:
    sampling_results/latency_report.md
    sampling_results/latency_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

EXT = 'phishingdetectorExt'
OUTDIR = 'sampling_results'
URLS_FILE = os.path.join(OUTDIR, 'memory_test_urls.json')

MODELS = {
    'rf_pkl':   {'label': 'RF (.pkl)',   'pkl': os.path.join(OUTDIR, 'rf_full.pkl')},
    'rf_onnx':  {'label': 'RF (.onnx)',  'onnx': os.path.join(EXT, 'phishing_rf.onnx'),
                 'tfidf': os.path.join(EXT, 'tfidf_data.json')},
    'xgb_pkl':  {'label': 'XGBoost (.pkl)',  'pkl': os.path.join(OUTDIR, 'xgb_full.pkl')},
    'xgb_onnx': {'label': 'XGBoost (.onnx)', 'onnx': os.path.join(EXT, 'phishing_xgb.onnx'),
                 'tfidf': os.path.join(EXT, 'tfidf_data_xgb.json'),
                 'prep': os.path.join(EXT, 'xgb_preprocessing.json')},
}


def p95(values):
    s = sorted(values)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def measure(key, urls, warmup):
    import joblib
    import onnxruntime as ort

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import features
    import ml_pipelines

    spec = MODELS[key]

    if key == 'rf_pkl':
        model = joblib.load(spec['pkl'])

        def predict(u):
            model.predict([u])
    elif key == 'rf_onnx':
        tfidf = json.load(open(spec['tfidf']))
        sess = ort.InferenceSession(spec['onnx'], providers=['CPUExecutionProvider'])
        name = sess.get_inputs()[0].name

        def predict(u):
            sess.run(None, {name: features.extract_features_onnx([u], tfidf)})
    elif key == 'xgb_pkl':
        model = joblib.load(spec['pkl'])

        def predict(u):
            ml_pipelines.predict_with_xgb(model, [u])
    else:  # xgb_onnx
        tfidf = json.load(open(spec['tfidf']))
        prep = json.load(open(spec['prep']))
        sess = ort.InferenceSession(spec['onnx'], providers=['CPUExecutionProvider'])
        name = sess.get_inputs()[0].name

        def predict(u):
            feats = features.extract_features_onnx([u], tfidf)
            sess.run(None, {name: features.preprocess_xgb(feats, prep)})

    for u in urls[:warmup]:
        predict(u)

    lat = []
    for u in urls:
        t0 = time.perf_counter()
        predict(u)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return {
        'label': spec['label'],
        'n': len(lat),
        'min_ms': round(min(lat), 3),
        'max_ms': round(max(lat), 3),
        'avg_ms': round(statistics.mean(lat), 3),
        'p95_ms': round(p95(lat), 3),
    }


def main():
    ap = argparse.ArgumentParser(description="Deployed-artifact latency benchmark (full pipeline).")
    ap.add_argument('--samples', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--urls-file', default=URLS_FILE)
    args = ap.parse_args()

    urls = json.load(open(args.urls_file))[:args.samples]
    print(f"URLs: {len(urls)} from {args.urls_file} (+{args.warmup} warm-up each)")

    results = []
    for key in MODELS:
        files = [v for k, v in MODELS[key].items() if k != 'label']
        missing = [p for p in files if not os.path.exists(p)]
        if missing:
            print(f"SKIP {MODELS[key]['label']}: missing {missing}")
            continue
        print(f"measuring {MODELS[key]['label']} ...", flush=True)
        r = measure(key, urls, args.warmup)
        print(f"  min/max/avg/p95 = {r['min_ms']}/{r['max_ms']}/{r['avg_ms']}/{r['p95_ms']} ms")
        results.append(r)

    os.makedirs(OUTDIR, exist_ok=True)
    json_path = os.path.join(OUTDIR, 'latency_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    lines = ["# Latency Benchmark (Python environment, deployed artifacts)", "",
             f"Full pipeline per URL (feature extraction + preprocessing + model) on the shared "
             f"{len(urls)}-URL sample (`{'memory_test_urls.json'}`), {args.warmup} warm-up scans each.", "",
             "| Model | Min (ms) | Max (ms) | Avg (ms) | P95 (ms) |", "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['label']} | {r['min_ms']} | {r['max_ms']} | {r['avg_ms']} | {r['p95_ms']} |")
    lines += ["", "> System-scheduling outliers show up in Max; Avg/P95 are the stable measures."]
    md_path = os.path.join(OUTDIR, 'latency_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {json_path}\nSaved: {md_path}")


if __name__ == '__main__':
    main()
