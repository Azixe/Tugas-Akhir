"""Export XGBoost pipeline artifacts to ONNX + JSONs (shared logic).

Consumes the artifacts dict produced by ``ml_pipelines.train_xgb_pipeline``
or ``Xgboost/train_xgb_colab.py`` and writes the three files the Chrome
extension needs:

  - <onnx_name>   XGBoost model (input: PCA features)
  - <prep_name>   scaler + feature-selection indices + PCA matrix
  - <tfidf_name>  TF-IDF vocabulary + IDF weights

Used by ``export_xgb_onnx.py`` (local) and
``Xgboost/export_xgb_onnx_colab.py`` (Colab wrapper).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def export_artifacts(artifacts: dict, outdir: str = '.',
                     onnx_name: str = 'phishing_xgb.onnx',
                     prep_name: str = 'xgb_preprocessing.json',
                     tfidf_name: str = 'tfidf_data_xgb.json',
                     verbose: bool = True) -> dict:
    """Write ONNX + preprocessing JSON + TF-IDF JSON. Returns the paths dict."""
    os.makedirs(outdir, exist_ok=True)

    model = artifacts['model']
    scaler = artifacts['scaler']
    selector_kbest = artifacts['selector_kbest']
    important_mask = artifacts['important_mask']
    pca = artifacts['pca']
    feature_extractor = artifacts['feature_extractor']
    feature_counts = artifacts['feature_counts']

    # ---- 1. TF-IDF vocabulary + IDF ----
    tfidf_vectorizer = feature_extractor.transformer_list[0][1]
    vocab_str = {str(k): int(v) for k, v in tfidf_vectorizer.vocabulary_.items()}
    tfidf_path = os.path.join(outdir, tfidf_name)
    with open(tfidf_path, 'w') as f:
        json.dump({
            'vocabulary': vocab_str,
            'idf': tfidf_vectorizer.idf_.tolist(),
            'max_features': len(vocab_str),
        }, f)
    if verbose:
        print(f">> {tfidf_path} ({len(vocab_str)} terms)")

    # ---- 2. Preprocessing (scaler + feature selection + PCA) ----
    kbest_indices = selector_kbest.get_support(indices=True)
    importance_indices = np.where(important_mask)[0]
    combined_indices = kbest_indices[importance_indices].tolist()

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

    prep_path = os.path.join(outdir, prep_name)
    with open(prep_path, 'w') as f:
        json.dump(preprocessing, f)
    if verbose:
        print(f">> {prep_path} ({os.path.getsize(prep_path)/1024:.0f} KB, "
              f"{len(combined_indices)} selected features)")

    # ---- 3. ONNX model ----
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    n_input_features = int(pca.n_components_)
    onnx_model = convert_xgboost(
        model,
        initial_types=[('float_input', FloatTensorType([None, n_input_features]))],
        target_opset=11,
    )
    onnx_path = os.path.join(outdir, onnx_name)
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    if verbose:
        print(f">> {onnx_path} ({os.path.getsize(onnx_path)/1024:.0f} KB, "
              f"input [batch, {n_input_features}])")

    # ---- 4. Verify + quick inference ----
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))

    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, n_input_features).astype(np.float32)
    session.run(None, {'float_input': dummy})

    if verbose:
        print("   ONNX valid, inference OK")
        print("   sha256:")
        for p in (onnx_path, prep_path, tfidf_path):
            print(f"     {sha256(p)}  {os.path.basename(p)}")

    return {
        'onnx': onnx_path,
        'preprocessing': prep_path,
        'tfidf': tfidf_path,
        'n_input_features': n_input_features,
        'n_selected_features': len(combined_indices),
        'feature_counts': feature_counts,
    }