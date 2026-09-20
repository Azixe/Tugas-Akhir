"""
XGBoost ONNX Export Script — Run on Google Colab
=================================================
Run this AFTER train_xgb_colab.py in the same Colab session,
or after uploading phishing_xgb_pipeline.pkl to Colab.

Setup:
1. Upload to /content/: phishing_xgb_pipeline.pkl, this file, features.py
   and xgb_export.py (shared modules — same files as in the repo root)
2. Run: !python export_xgb_onnx_colab.py

Options:
  --pkl PATH        pipeline .pkl (default: auto-detect in /content)
  --outdir DIR      output directory (default: .)
  --onnx-name NAME / --prep-name NAME / --tfidf-name NAME

Outputs:
  - phishing_xgb.onnx         (XGBoost model for ONNX Runtime)
  - xgb_preprocessing.json    (scaler, feature indices, PCA matrix)
  - tfidf_data_xgb.json       (TF-IDF vocabulary + IDF weights)
"""

# === Install dependencies ===
import subprocess
subprocess.run(['pip', 'install', 'onnxmltools', 'onnxconverter-common', 'onnx', 'skl2onnx', 'onnxruntime'],
               capture_output=True)

import argparse
import os
import sys

import joblib

# Shared modules (upload features.py and xgb_export.py next to this script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import alias_legacy_main  # noqa: E402
from xgb_export import export_artifacts  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Export XGBoost artifacts to ONNX + JSONs.")
    parser.add_argument('--pkl', default=None,
                        help="Input pipeline .pkl (default: auto-detect)")
    parser.add_argument('--outdir', default='.',
                        help="Output directory (default: %(default)s)")
    parser.add_argument('--onnx-name', default='phishing_xgb.onnx')
    parser.add_argument('--prep-name', default='xgb_preprocessing.json')
    parser.add_argument('--tfidf-name', default='tfidf_data_xgb.json')
    return parser.parse_args()


def main():
    args = parse_args()

    # Legacy pickles reference __main__.make_tokens etc.; new ones reference features.*
    alias_legacy_main()

    pkl_paths = [p for p in [
        args.pkl,
        'phishing_xgb_pipeline.pkl',
        '/content/phishing_xgb_pipeline.pkl',
    ] if p]

    artifacts = None
    for p in pkl_paths:
        if os.path.exists(p):
            print(f"Loading from: {p}")
            artifacts = joblib.load(p)
            break

    if artifacts is None:
        print("phishing_xgb_pipeline.pkl not found!")
        print("Run train_xgb_colab.py first, or upload the pkl file.")
        exit()

    print(f"Feature counts: {artifacts['feature_counts']}")
    result = export_artifacts(
        artifacts,
        outdir=args.outdir,
        onnx_name=args.onnx_name,
        prep_name=args.prep_name,
        tfidf_name=args.tfidf_name,
    )

    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(result)

    # === Download files ===
    try:
        from google.colab import files
        print("\nDownloading files...")
        files.download(result['onnx'])
        files.download(result['preprocessing'])
        files.download(result['tfidf'])
    except ImportError:
        print(f"\nFiles saved to: {args.outdir}")


if __name__ == "__main__":
    main()