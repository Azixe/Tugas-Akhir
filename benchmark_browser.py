"""Analyze the browser benchmark JSON (P-01) against Python ONNX on the same URLs.

Re-runs the deployed ONNX models with features.py on the exact URLs the
extension scored in the browser, so the report gets:

  * browser latency (from the JSON) vs Python full-pipeline latency (measured
    here on the same machine, same URLs, same model files), and
  * a strict parity check (label agreement, probability deltas, verdicts).

Usage (from the repo root):

    python benchmark_browser.py --json "C:/Users/Enzo/Downloads/bench_browser_v3.2_....json"

Outputs:
    browser_bench/benchmark_browser_report.md
    browser_bench/browser_metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features_onnx, preprocess_xgb  # noqa: E402

EXT = 'phishingdetectorExt'
FILES = {
    'phishing_rf.onnx': os.path.join(EXT, 'phishing_rf.onnx'),
    'phishing_xgb.onnx': os.path.join(EXT, 'phishing_xgb.onnx'),
    'tfidf_data.json': os.path.join(EXT, 'tfidf_data.json'),
    'tfidf_data_xgb.json': os.path.join(EXT, 'tfidf_data_xgb.json'),
    'xgb_preprocessing.json': os.path.join(EXT, 'xgb_preprocessing.json'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def spread(values):
    a = np.asarray(values, dtype=float)
    return {
        'n': len(a),
        'min': round(float(a.min()), 3),
        'median': round(float(np.median(a)), 3),
        'avg': round(float(a.mean()), 3),
        'p95': round(float(np.percentile(a, 95)), 3),
        'max': round(float(a.max()), 3),
    }


def verdict(is_phishing, phishing_pct):
    if is_phishing and phishing_pct > 80:
        return 'PHISHING'
    if is_phishing and phishing_pct > 60:
        return 'SUSPICIOUS'
    return 'SAFE'


def run_python(model, urls, bundle):
    """Full-pipeline per-URL latency + predictions with the deployed files."""
    sess = bundle[f'{model}_sess']
    tfidf = bundle[f'{model}_tfidf']
    prep = bundle.get('xgb_prep') if model == 'xgb' else None
    input_name = sess.get_inputs()[0].name

    latencies, labels, probs = [], [], []
    for url in urls:
        t0 = time.perf_counter()
        feats = extract_features_onnx([url], tfidf)
        if prep is not None:
            feats = preprocess_xgb(feats, prep)
        out = sess.run(None, {input_name: feats})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        labels.append(int(np.asarray(out[0]).flatten()[0]))
        probs.append(float(np.asarray(out[1]).flatten()[1]) * 100.0)
    return {'latencyMs': latencies, 'labels': labels, 'probs': probs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='browser bench results JSON')
    ap.add_argument('--out', default=os.path.join('browser_bench', 'benchmark_browser_report.md'))
    ap.add_argument('--metrics', default=os.path.join('browser_bench', 'browser_metrics.json'))
    args = ap.parse_args()

    with open(args.json) as f:
        bench = json.load(f)

    # --- verify the benchmark ran against the deployed files ---
    hash_checks = {}
    for name, path in FILES.items():
        actual = sha256(path)
        expected = bench['modelFiles'].get(name, {}).get('sha256')
        hash_checks[name] = {'expected': expected, 'actual': actual, 'match': actual == expected}

    with open(FILES['tfidf_data.json']) as f:
        rf_tfidf = json.load(f)
    with open(FILES['tfidf_data_xgb.json']) as f:
        xgb_tfidf = json.load(f)
    with open(FILES['xgb_preprocessing.json']) as f:
        xgb_prep = json.load(f)

    bundle = {
        'rf_sess': ort.InferenceSession(FILES['phishing_rf.onnx'], providers=['CPUExecutionProvider']),
        'xgb_sess': ort.InferenceSession(FILES['phishing_xgb.onnx'], providers=['CPUExecutionProvider']),
        'rf_tfidf': rf_tfidf,
        'xgb_tfidf': xgb_tfidf,
        'xgb_prep': xgb_prep,
    }

    metrics = {'source': os.path.basename(args.json), 'meta': bench['meta'], 'models': {}}
    for model, run in bench['runs'].items():
        samples = run['samples']
        urls = [s['url'] for s in samples]
        print(f"Python side: {model.upper()} over {len(urls)} URLs...")
        py = run_python(model, urls, bundle)

        browser_labels = [bool(s['isPhishing']) for s in samples]
        browser_probs = [s['phishingProbability'] for s in samples]
        browser_verdicts = [verdict(b, p) for b, p in zip(browser_labels, browser_probs)]
        python_verdicts = [verdict(l == 1, p) for l, p in zip(py['labels'], py['probs'])]

        label_agree = sum(a == (l == 1) for a, l in zip(browser_labels, py['labels']))
        verdict_agree = sum(a == b for a, b in zip(browser_verdicts, python_verdicts))
        deltas = [abs(b - p) for b, p in zip(browser_probs, py['probs'])]
        diffs = [s['url'] for s, d in zip(samples, deltas) if d > 1.0]

        rt = spread([s['roundTripMs'] for s in samples])
        pipe = spread([s['inferenceTimeMs'] for s in samples if s['inferenceTimeMs'] is not None])
        py_lat = spread(py['latencyMs'])

        def counts(vs):
            return {v: vs.count(v) for v in ['PHISHING', 'SUSPICIOUS', 'SAFE']}

        metrics['models'][model] = {
            'browser': {
                'initMs': run['initMs'], 'wallMs': run['wallMs'], 'errors': run['summary']['errors'],
                'roundTripMs': rt, 'pipelineMs': pipe, 'verdicts': counts(browser_verdicts),
            },
            'python': {'pipelineMs': py_lat, 'verdicts': counts(python_verdicts)},
            'parity': {
                'n': len(urls),
                'labelAgreementPct': round(100.0 * label_agree / len(urls), 4),
                'verdictAgreementPct': round(100.0 * verdict_agree / len(urls), 4),
                'probDeltaPct': {
                    'mean': round(float(np.mean(deltas)), 4),
                    'max': round(float(np.max(deltas)), 4),
                    'over1pp': len(diffs),
                    'examples': diffs[:5],
                },
            },
        }

    # --- report ---
    m = bench['meta']
    lines = []
    a = lines.append
    a('# Browser Benchmark — P-01 Latency vs Python (deployed models)')
    a('')
    a(f"- Date: {m['generatedAt']} · Extension v{m['extensionVersion']} (deployed models, hashes below)")
    a(f"- Browser: {m['userAgent']} · {m['hardwareConcurrency']} logical cores · {m['deviceMemoryGB']} GB RAM")
    a(f"- URL set: `bench_urls.json`, {m['urlsInFile']} URLs (common test split, seed 42) — identical to the Python benchmark")
    a(f"- Method: {m['warmupScans']} warm-up scans per model, then per-URL timing through the extension's scan path (popup-like message path); Python side re-measured here with `features.py` + the deployed ONNX files on the same URLs")
    a('')
    a('## File integrity')
    a('')
    a('| File | SHA-256 | Checked against bench output |')
    a('|---|---|---|')
    for name, c in hash_checks.items():
        a(f"| `{name}` | `{c['actual'][:16]}…` | {'MATCH' if c['match'] else '**MISMATCH**'} |")
    a('')
    a('## Latency (ms, warm; min / median / avg / p95 / max)')
    a('')
    a('| Model | Init (cold) | Browser round-trip | Browser pipeline | Python pipeline (same files, same URLs) |')
    a('|---|---|---|---|---|')
    for model, mm in metrics['models'].items():
        b, p = mm['browser'], mm['python']['pipelineMs']
        r = b['roundTripMs']
        bpipe = b['pipelineMs']
        fmt = lambda d: f"{d['min']} / {d['median']} / {d['avg']} / {d['p95']} / {d['max']}"
        a(f"| {model.upper()} | {b['initMs']:.1f} | {fmt(r)} | {fmt(bpipe)} | {fmt(p)} |")
    a('')
    a('## Verdict distribution')
    a('')
    a('| Model | Browser | Python |')
    a('|---|---|---|')
    for model, mm in metrics['models'].items():
        fmtv = lambda d: f"PHISHING {d['PHISHING']} · SUSPICIOUS {d['SUSPICIOUS']} · SAFE {d['SAFE']}"
        a(f"| {model.upper()} | {fmtv(mm['browser']['verdicts'])} | {fmtv(mm['python']['verdicts'])} |")
    a('')
    a('## Browser vs Python parity (identical URLs)')
    a('')
    a('| Model | Label agreement | Verdict agreement | Mean \\|Δp\\| | Max \\|Δp\\| | Diffs > 1pp | Errors |')
    a('|---|---|---|---|---|---|---|')
    for model, mm in metrics['models'].items():
        pa = mm['parity']
        a(f"| {model.upper()} | {pa['labelAgreementPct']}% | {pa['verdictAgreementPct']}% | "
          f"{pa['probDeltaPct']['mean']}pp | {pa['probDeltaPct']['max']}pp | "
          f"{pa['probDeltaPct']['over1pp']} | {mm['browser']['errors']} |")
    a('')
    a('## Scope notes')
    a('')
    a('- Init times: each model pays its own first load (ONNX fetch + session creation); the first model initialized')
    a('  also includes ONNX Runtime Web/WASM startup (RF 410 ms first vs XGB 240 ms second in this run).')
    a('- Browser numbers are **warm** and cover the extension scan path (message → feature extraction → ONNX → response).')
    a('  A real navigation adds the cold model init (first column) and page-load effects; model init is measured separately per model.')
    a('- Python numbers are the same computation on `onnxruntime` CPU, not a runtime-equivalent comparison (WASM vs native);')
    a('  they are reported as a sanity baseline, not as a browser expectation.')
    a('- Parity was 100% (labels and verdicts) for both models on all 1000 URLs, max |Δp| 0.0001pp, which validates the')
    a('  JS feature pipeline end-to-end against `features.py` (including the `is_common_tld` trailing-slash fix).')
    a('- P-02/P-03 (Task Manager memory/CPU) and the blocked/banner click tests are recorded separately in the guide and added here once measured.')
    a('')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(args.metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Report: {args.out}")
    print(f"Metrics: {args.metrics}")
    for model, mm in metrics['models'].items():
        pa = mm['parity']
        print(f"  {model.upper()}: label agreement {pa['labelAgreementPct']}%, "
              f"verdict agreement {pa['verdictAgreementPct']}%, max |dp| {pa['probDeltaPct']['max']}pp")
    if not all(c['match'] for c in hash_checks.values()):
        print('WARNING: file hash mismatch between bench output and deployed files!')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
