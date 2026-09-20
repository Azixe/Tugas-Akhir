"""
XGBoost ONNX Export Script — Run on Google Colab
=================================================
Run this AFTER train_xgb_colab.py in the same Colab session,
or after uploading phishing_xgb_pipeline.pkl to Colab.

Setup:
1. Upload to /content/: phishing_xgb_pipeline.pkl, this file, and features.py
   (shared feature module — required, same file as in the repo root)
2. Run: !python export_xgb_onnx_colab.py

Options:
  --pkl PATH    pipeline .pkl (default: auto-detect in /content)
  --outdir DIR  output directory (default: .)

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
import hashlib
import json
import os
import sys

import joblib
import numpy as np

# Shared feature definitions (upload features.py next to this script on Colab)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import alias_legacy_main  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Export the XGBoost pipeline to ONNX + JSON.")
    parser.add_argument('--pkl', default=None,
                        help="Input pipeline .pkl (default: auto-detect)")
    parser.add_argument('--outdir', default='.',
                        help="Output directory (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Legacy pickles reference __main__.make_tokens etc.; new ones reference features.*
    alias_legacy_main()

    # === Load pipeline ===
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

    model = artifacts['model']
    scaler = artifacts['scaler']
    selector_kbest = artifacts['selector_kbest']
    important_mask = artifacts['important_mask']
    pca = artifacts['pca']
    feature_extractor = artifacts['feature_extractor']
    feature_counts = artifacts['feature_counts']

    print(f"Feature counts: {feature_counts}")
    print(f"Model: {model}")

    # ============================================================
    # 1. Export TF-IDF vocabulary + IDF weights
    # ============================================================
    print("\n--- Exporting TF-IDF data ---")

    tfidf_vectorizer = feature_extractor.transformer_list[0][1]

    vocab = tfidf_vectorizer.vocabulary_
    idf = tfidf_vectorizer.idf_.tolist()

    vocab_str = {str(k): int(v) for k, v in vocab.items()}

    tfidf_data = {
        'vocabulary': vocab_str,
        'idf': idf,
        'max_features': len(vocab_str)
    }

    tfidf_path = os.path.join(args.outdir, 'tfidf_data_xgb.json')
    with open(tfidf_path, 'w') as f:
        json.dump(tfidf_data, f)
    print(f">> {tfidf_path} ({len(vocab_str)} terms)")

    # ============================================================
    # 2. Export preprocessing pipeline (scaler + feature selection + PCA)
    # ============================================================
    print("\n--- Exporting preprocessing pipeline ---")

    # Get the feature indices that survive both SelectKBest and importance pruning
    kbest_indices = selector_kbest.get_support(indices=True)
    importance_indices = np.where(important_mask)[0]

    # Map importance indices back to the original 1509 feature space
    combined_indices = kbest_indices[importance_indices].tolist()

    print(f"  SelectKBest kept: {len(kbest_indices)} features")
    print(f"  Importance kept: {len(importance_indices)} features")
    print(f"  Combined indices: {len(combined_indices)} features from original {feature_counts['original']}")
    print(f"  PCA components: {pca.n_components_}")

    preprocessing = {
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'selected_feature_indices': combined_indices,
        'pca_mean': pca.mean_.tolist(),
        'pca_components': pca.components_.tolist(),
        'n_pca_components': int(pca.n_components_),
        'n_original_features': feature_counts['original'],
        'n_selected_features': len(combined_indices),
        'n_pca_features': int(pca.n_components_),
    }

    prep_path = os.path.join(args.outdir, 'xgb_preprocessing.json')
    with open(prep_path, 'w') as f:
        json.dump(preprocessing, f)
    prep_size = os.path.getsize(prep_path) / 1024
    print(f">> {prep_path} ({prep_size:.0f} KB)")

    # ============================================================
    # 3. Export XGBoost model to ONNX
    # ============================================================
    print("\n--- Exporting XGBoost to ONNX ---")

    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    n_input_features = int(pca.n_components_)
    print(f"ONNX input shape: [batch, {n_input_features}]")

    initial_type = [('float_input', FloatTensorType([None, n_input_features]))]

    onnx_model = convert_xgboost(
        model, 
        initial_types=initial_type,
        target_opset=11
    )

    onnx_path = os.path.join(args.outdir, 'phishing_xgb.onnx')
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    onnx_size = os.path.getsize(onnx_path) / 1024
    print(f">> {onnx_path} ({onnx_size:.0f} KB)")

    # ============================================================
    # 4. Verify ONNX model
    # ============================================================
    print("\n--- Verifying ONNX model ---")

    import onnx
    onnx_loaded = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_loaded)
    print("ONNX model is valid!")

    for inp in onnx_loaded.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name}, shape={shape}")
    for out in onnx_loaded.graph.output:
        print(f"  Output: {out.name}")

    # ============================================================
    # 5. Quick inference test
    # ============================================================
    print("\n--- Quick inference test ---")

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    dummy_input = np.random.randn(1, n_input_features).astype(np.float32)
    result = session.run(None, {'float_input': dummy_input})
    print(f"  Dummy prediction: label={result[0][0]}, probabilities={result[1][0]}")
    print("  Inference OK!")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print("=" * 60)
    print("Files for Chrome extension:")
    print(f"  1. {onnx_path} ({onnx_size:.0f} KB) - XGBoost model")
    print(f"  2. {prep_path} ({prep_size:.0f} KB) - Scaler + feature selection + PCA")
    print(f"  3. {tfidf_path} ({os.path.getsize(tfidf_path)/1024:.0f} KB) - TF-IDF vocabulary")
    print(f"Pipeline: TF-IDF+Structural(1509) -> Scale -> Select({len(combined_indices)}) -> PCA({n_input_features}) -> XGBoost")
    print("\nsha256:")
    for p in (onnx_path, prep_path, tfidf_path):
        print(f"  {sha256(p)}  {os.path.basename(p)}")

    # === Download files ===
    try:
        from google.colab import files
        print("\nDownloading files...")
        files.download(onnx_path)
        files.download(prep_path)
        files.download(tfidf_path)
    except ImportError:
        print(f"\nFiles saved to: {args.outdir}")


if __name__ == "__main__":
    main()
