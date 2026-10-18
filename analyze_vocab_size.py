"""How much does the TF-IDF vocabulary size matter? Random Forest at several sizes.

The deployed models use a 1,500-term word-level TF-IDF vocabulary. The only recorded reason is a
comment in ``hybrid/train_hybrid_model_v2.py`` ("Size optimization: 5000 -> 1500 TF-IDF tokens").
This script measures the trade-off instead of asserting it: for each vocabulary size it trains the
Random Forest with the fixed hyperparameters of ``ml_pipelines`` (seeded since 2026-10-13), scores
the common test set, and exports the model to ONNX exactly as ``export_to_web.py`` does, to record
the size of the two files the extension ships (``phishing_rf.onnx`` and ``tfidf_data.json``).

In-distribution only: the test set comes from the same random split as the training data.

Usage (repo root):

    python analyze_vocab_size.py                          # 500, 1000, 1500, 3000, 5000 (about 10 min)
    python analyze_vocab_size.py --sizes 1000,1500 --sample-rows 20000   # smoke test

Reads : URL dataset.csv (via sampling.py)
Writes: sampling_results/vocab_size_report.md, sampling_results/vocab_size.json
        (models and ONNX files are written to a temporary folder and deleted)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import ml_pipelines
import sampling
from benchmark_sampling import evaluate, subsample_stratified

OUTDIR = 'sampling_results'


def export_sizes(pipeline, workdir, tag):
    """Bytes of the ONNX model and the TF-IDF JSON, written the way export_to_web.py writes them."""
    tfidf = pipeline.named_steps['features'].transformer_list[0][1]
    rf = pipeline.named_steps['clf']
    tfidf_data = {"vocabulary": {k: int(v) for k, v in tfidf.vocabulary_.items()},
                  "idf": tfidf.idf_.tolist(), "norm": tfidf.norm, "use_idf": tfidf.use_idf,
                  "smooth_idf": tfidf.smooth_idf, "sublinear_tf": tfidf.sublinear_tf}
    json_path = os.path.join(workdir, f'tfidf_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(tfidf_data, f)
    n_features = len(tfidf_data['vocabulary']) + 9
    onnx_model = convert_sklearn(rf, initial_types=[('float_input', FloatTensorType([None, n_features]))],
                                 target_opset=12, options={id(rf): {'zipmap': False}})
    onnx_path = os.path.join(workdir, f'rf_{tag}.onnx')
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    return {'onnx_bytes': os.path.getsize(onnx_path), 'tfidf_json_bytes': os.path.getsize(json_path),
            'n_features': n_features, 'tree_nodes': int(sum(t.tree_.node_count for t in rf.estimators_))}


def main():
    ap = argparse.ArgumentParser(description="Random Forest vs TF-IDF vocabulary size (in-distribution).")
    ap.add_argument('--sizes', default='500,1000,1500,3000,5000', help="Comma-separated max_features values")
    ap.add_argument('--sample-rows', type=int, default=0, help="Debug: cap the train pool / test set (0 = full)")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(',')]

    df = sampling.load_clean_dataset(sampling.DEFAULT_DATASET)
    train, test = sampling.common_split(df)
    if args.sample_rows:
        train = subsample_stratified(train, args.sample_rows)
        test = subsample_stratified(test, max(1000, args.sample_rows // 4))
        print("!! SMOKE TEST MODE (subsampled data) — do not use for the thesis")
    print(f"Train pool: {sampling.describe(train)}")
    print(f"Common test: {sampling.describe(test)}")

    rows = []
    with tempfile.TemporaryDirectory() as workdir:
        for n in sizes:
            t0 = time.time()
            print(f"  [max_features={n}] training on {len(train)} URLs...", flush=True)
            model = ml_pipelines.build_rf_pipeline(max_features=n)
            model.fit(train['url'].values, train['label'].values)
            train_seconds = time.time() - t0
            metrics = evaluate(model, 'rf', test, 0)
            sizes_ = export_sizes(model, workdir, n)
            vocab = len(model.named_steps['features'].transformer_list[0][1].vocabulary_)
            rows.append({'max_features': n, 'vocabulary': vocab, 'train_seconds': round(train_seconds, 1),
                         'accuracy': metrics['accuracy'], 'precision': metrics['precision_phishing'],
                         'recall': metrics['recall_phishing'], 'f1': metrics['f1_phishing'],
                         'fpr': metrics['fpr'], 'fp': metrics['fp'], 'fn': metrics['fn'], **sizes_})
            r = rows[-1]
            print(f"  [max_features={n}] recall {r['recall']:.4f} FPR {100 * r['fpr']:.3f}% | ONNX "
                  f"{r['onnx_bytes'] / 1024:.0f} KB, tfidf JSON {r['tfidf_json_bytes'] / 1024:.0f} KB "
                  f"({train_seconds:.0f}s)", flush=True)

    meta = {'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'seed': sampling.SEED,
            'rf_hparams': {k: v for k, v in ml_pipelines.RF_HPARAMS.items()},
            'train': sampling.describe(train), 'test': sampling.describe(test),
            'smoke_test': bool(args.sample_rows)}
    suffix = '_smoke' if args.sample_rows else ''
    with open(os.path.join(OUTDIR, f'vocab_size{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'rows': rows}, f, indent=2)
    if not args.sample_rows:
        print("Saved:", write_report(meta, rows))
    print_summary(rows)


def write_report(meta, rows):
    lines = ["# TF-IDF vocabulary size — Random Forest (in-distribution)", "",
             f"Generated: {meta['timestamp']}",
             f"Protocol: `sampling.py` (stratified 80/20, seed {meta['seed']}); train pool {meta['train']['total']} "
             f"URLs, common test set {meta['test']['total']} URLs, threshold 0.5.",
             f"Random Forest: `ml_pipelines.RF_HPARAMS` ({', '.join(f'{k}={v}' for k, v in meta['rf_hparams'].items())}).",
             "Sizes are the two files the extension ships for this model, exported as `export_to_web.py` does.",
             "",
             "| max_features | Vocabulary | Accuracy | Precision | Recall | F1 | FPR | FP | FN | ONNX (KB) | tfidf JSON (KB) | Tree nodes |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['max_features']} | {r['vocabulary']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['f1']:.4f} | {100 * r['fpr']:.3f}% | {r['fp']} | {r['fn']} | "
                     f"{r['onnx_bytes'] / 1024:.0f} | {r['tfidf_json_bytes'] / 1024:.0f} | {r['tree_nodes']} |")
    lines += ["",
              "- The deployed RF (`phishingdetectorExt/phishing_rf.onnx`) was fitted before the seed was added, so "
              "the 1,500 row is a seeded retrain of the same configuration, not the deployed file.",
              "- The test set comes from the same random split as the training data; these numbers say nothing "
              "about URLs from other sources (see `external_500/` and `external_posthoc/`).",
              ""]
    path = os.path.join(OUTDIR, 'vocab_size_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return path


def print_summary(rows):
    print()
    print(f"{'max_features':>12}{'Recall':>9}{'FPR':>9}{'F1':>8}{'ONNX KB':>9}{'JSON KB':>9}")
    for r in rows:
        print(f"{r['max_features']:>12}{r['recall']:>9.4f}{100 * r['fpr']:>8.3f}%{r['f1']:>8.4f}"
              f"{r['onnx_bytes'] / 1024:>9.0f}{r['tfidf_json_bytes'] / 1024:>9.0f}")
    print()


if __name__ == '__main__':
    main()
