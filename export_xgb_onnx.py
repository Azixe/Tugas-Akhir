"""Export a trained XGBoost artifacts .pkl to ONNX + JSONs (local, no Colab).

Usage (run from the repo root):

    python export_xgb_onnx.py --pkl sampling_results/xgb_under.pkl \
        --outdir sampling_results \
        --onnx-name xgb_under.onnx \
        --prep-name xgb_under_preprocessing.json \
        --tfidf-name xgb_under_tfidf.json

The extension expects `phishing_xgb.onnx`, `xgb_preprocessing.json` and
`tfidf_data_xgb.json` inside `phishingdetectorExt/` — copy them there after
exporting (or pass matching names).
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import alias_legacy_main  # noqa: E402
from xgb_export import export_artifacts  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Export XGBoost artifacts to ONNX + JSONs.")
    parser.add_argument('--pkl', required=True, help="Input artifacts .pkl")
    parser.add_argument('--outdir', default='.', help="Output directory (default: %(default)s)")
    parser.add_argument('--onnx-name', default='phishing_xgb.onnx',
                        help="ONNX filename (default: %(default)s)")
    parser.add_argument('--prep-name', default='xgb_preprocessing.json',
                        help="Preprocessing JSON filename (default: %(default)s)")
    parser.add_argument('--tfidf-name', default='tfidf_data_xgb.json',
                        help="TF-IDF JSON filename (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    alias_legacy_main()

    print(f"Memuat artifacts: {args.pkl}")
    try:
        artifacts = joblib.load(args.pkl)
    except FileNotFoundError:
        print(f"Error: file tidak ditemukan: {args.pkl}")
        sys.exit(1)

    print(f"Feature counts: {artifacts['feature_counts']}")
    result = export_artifacts(
        artifacts,
        outdir=args.outdir,
        onnx_name=args.onnx_name,
        prep_name=args.prep_name,
        tfidf_name=args.tfidf_name,
    )
    print(f"\nSUKSES! Pipeline: TF-IDF+Structural(1509) -> Scale -> "
          f"Select({result['n_selected_features']}) -> PCA({result['n_input_features']}) -> XGBoost")


if __name__ == "__main__":
    main()