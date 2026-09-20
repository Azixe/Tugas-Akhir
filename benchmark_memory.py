"""Memory benchmark for the four Tabel 4.3 entries (Python environment side).

Measures resident memory (RSS) per model in a *fresh process* so the models
cannot pollute each other:

  1. RF  (.pkl)  sampling_results/rf_full.pkl
  2. RF  (.onnx) phishingdetectorExt/phishing_rf.onnx + tfidf_data.json
  3. XGB (.pkl)  sampling_results/xgb_under.pkl
  4. XGB (.onnx) phishingdetectorExt/phishing_xgb.onnx + preprocessing + tfidf

For each entry: baseline RSS, RSS after loading the model, then RSS sampled
every 5 ms while running single-URL scans (1000 random URLs from the shared
test split). Reports Min/Max/Avg during the scan, matching the proposed
Tabel 3.3 "Memory Usage (Min/Max/Avg)" columns.

NOTE: these are Python-process numbers. Browser memory (Chrome Task Manager,
P-02) is measured separately and is not the same quantity.

Usage (run from the repo root):

    python benchmark_memory.py                 # all four, 1000 URLs
    python benchmark_memory.py --samples 200   # quicker
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

EXT = 'phishingdetectorExt'
OUTDIR = 'sampling_results'
URLS_FILE = os.path.join(OUTDIR, 'memory_test_urls.json')

MODELS = {
    'rf_pkl':   {'label': 'RF (.pkl)',   'pkl': os.path.join(OUTDIR, 'rf_full.pkl')},
    'rf_onnx':  {'label': 'RF (.onnx)',  'onnx': os.path.join(EXT, 'phishing_rf.onnx'),
                 'tfidf': os.path.join(EXT, 'tfidf_data.json')},
    'xgb_pkl':  {'label': 'XGBoost (.pkl)',  'pkl': os.path.join(OUTDIR, 'xgb_under.pkl')},
    'xgb_onnx': {'label': 'XGBoost (.onnx)', 'onnx': os.path.join(EXT, 'phishing_xgb.onnx'),
                 'tfidf': os.path.join(EXT, 'tfidf_data_xgb.json'),
                 'prep': os.path.join(EXT, 'xgb_preprocessing.json')},
}


def parse_args():
    p = argparse.ArgumentParser(description="Python-side memory benchmark.")
    p.add_argument('--samples', type=int, default=1000,
                   help="URLs to scan per model (default: %(default)s)")
    p.add_argument('--interval-ms', type=float, default=5.0,
                   help="RSS sampling interval in ms (default: %(default)s)")
    p.add_argument('--outdir', default=OUTDIR)
    # internal worker mode
    p.add_argument('--worker', default=None, choices=list(MODELS))
    p.add_argument('--urls-file', default=None)
    return p.parse_args()


# ============================================================
# WORKER (one per model, fresh process)
# ============================================================

def worker(args):
    import joblib
    import numpy as np
    import onnxruntime as ort
    import psutil

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import features
    import ml_pipelines

    proc = psutil.Process()

    def rss():
        return proc.memory_info().rss

    urls = json.load(open(args.urls_file))
    urls = urls[:args.samples]
    spec = MODELS[args.worker]

    baseline = statistics.median(rss() for _ in range(3))

    # ---- load model ----
    t0 = time.time()
    if args.worker == 'rf_pkl':
        model = joblib.load(spec['pkl'])
        def predict(u):
            model.predict([u])
    elif args.worker == 'rf_onnx':
        tfidf = json.load(open(spec['tfidf']))
        sess = ort.InferenceSession(spec['onnx'], providers=['CPUExecutionProvider'])
        def predict(u):
            sess.run(None, {sess.get_inputs()[0].name:
                            features.extract_features_onnx([u], tfidf)})
    elif args.worker == 'xgb_pkl':
        model = joblib.load(spec['pkl'])
        def predict(u):
            ml_pipelines.predict_with_xgb(model, [u])
    else:  # xgb_onnx
        tfidf = json.load(open(spec['tfidf']))
        prep = json.load(open(spec['prep']))
        sess = ort.InferenceSession(spec['onnx'], providers=['CPUExecutionProvider'])
        def predict(u):
            X = features.preprocess_xgb(features.extract_features_onnx([u], tfidf), prep)
            sess.run(None, {sess.get_inputs()[0].name: X})
    load_seconds = time.time() - t0
    after_load = rss()

    # ---- warm-up (allocates inference arenas) ----
    for u in urls[:10]:
        predict(u)
    after_warmup = rss()

    # ---- sampled scans ----
    class Sampler(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.samples = []
            self._stop_event = threading.Event()

        def run(self):
            while not self._stop_event.is_set():
                self.samples.append(rss())
                self._stop_event.wait(args.interval_ms / 1000.0)

        def stop(self):
            self._stop_event.set()
            self.join()

    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    for u in urls:
        predict(u)
    scan_seconds = time.time() - t0
    sampler.stop()

    samples = sampler.samples or [rss()]
    mb = 1024 * 1024
    result = {
        'model': args.worker,
        'label': spec['label'],
        'urls': len(urls),
        'baseline_mb': round(baseline / mb, 1),
        'after_load_mb': round(after_load / mb, 1),
        'load_delta_mb': round((after_load - baseline) / mb, 1),
        'after_warmup_mb': round(after_warmup / mb, 1),
        'rss_min_mb': round(min(samples) / mb, 1),
        'rss_max_mb': round(max(samples) / mb, 1),
        'rss_avg_mb': round(statistics.mean(samples) / mb, 1),
        'rss_min_delta_mb': round((min(samples) - baseline) / mb, 1),
        'rss_max_delta_mb': round((max(samples) - baseline) / mb, 1),
        'rss_avg_delta_mb': round((statistics.mean(samples) - baseline) / mb, 1),
        'samples_taken': len(samples),
        'load_seconds': round(load_seconds, 1),
        'scan_seconds': round(scan_seconds, 1),
    }
    print("RESULT_JSON:" + json.dumps(result))


# ============================================================
# PARENT
# ============================================================

def main():
    args = parse_args()
    if args.worker:
        return worker(args)

    os.makedirs(args.outdir, exist_ok=True)

    # build the shared URL list once (same split as the other benchmarks)
    need_build = True
    if os.path.exists(URLS_FILE):
        try:
            need_build = len(json.load(open(URLS_FILE))) != args.samples
        except Exception:
            need_build = True
    if need_build:
        import numpy as np
        import sampling
        df = sampling.load_clean_dataset('URL dataset.csv')
        _, test = sampling.common_split(df)
        idx = np.random.RandomState(42).choice(len(test), args.samples, replace=False)
        urls = test['url'].values[idx].tolist()
        with open(URLS_FILE, 'w') as f:
            json.dump(urls, f)
    print(f"URLs file: {URLS_FILE} ({args.samples} URLs)")

    results = []
    for key in MODELS:
        label = MODELS[key]['label']
        files = [v for k, v in MODELS[key].items() if k != 'label']
        missing = [p for p in files if not os.path.exists(p)]
        if missing:
            print(f"SKIP {label}: missing {missing}")
            continue
        print(f"measuring {label} ...", flush=True)
        cp = subprocess.run(
            [sys.executable, os.path.abspath(__file__), '--worker', key,
             '--urls-file', URLS_FILE, '--samples', str(args.samples),
             '--interval-ms', str(args.interval_ms)],
            capture_output=True, text=True)
        line = [l for l in cp.stdout.splitlines() if l.startswith('RESULT_JSON:')]
        if not line:
            print(f"  ERROR for {label}:\n{cp.stdout[-500:]}\n{cp.stderr[-1000:]}")
            continue
        results.append(json.loads(line[0][len('RESULT_JSON:'):]))
        r = results[-1]
        print(f"  baseline {r['baseline_mb']} MB | after load {r['after_load_mb']} MB "
              f"(+{r['load_delta_mb']}) | scan min/max/avg "
              f"{r['rss_min_mb']}/{r['rss_max_mb']}/{r['rss_avg_mb']} MB")

    # ---- outputs ----
    json_path = os.path.join(args.outdir, 'memory_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    lines = ["# Memory Benchmark (Python environment)", "",
             f"Fresh process per model; RSS sampled every {args.interval_ms} ms during "
             f"{args.samples} single-URL scans.", "",
             "| Model | Baseline (MB) | Setelah load (MB) | Load delta (MB) | Min (MB) | Max (MB) | Avg (MB) | Peak delta (MB) | Scan (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['label']} | {r['baseline_mb']} | {r['after_load_mb']} | "
                     f"{r['load_delta_mb']} | {r['rss_min_mb']} | {r['rss_max_mb']} | "
                     f"{r['rss_avg_mb']} | {r['rss_max_delta_mb']} | {r['scan_seconds']} |")
    lines += ["", f"> Baseline ≈ {results[0]['baseline_mb'] if results else 0:.1f} MB: Python "
              "interpreter + shared libraries (sklearn / xgboost / onnxruntime) imported before "
              "the model is loaded, identical across entries.",
              "> Python-process RSS. Browser (extension) memory is measured separately "
              "via Chrome Task Manager (P-02) and is not the same quantity."]
    md_path = os.path.join(args.outdir, 'memory_report.md')
    with open(md_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == '__main__':
    main()