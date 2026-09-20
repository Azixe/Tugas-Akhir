"""Export a trained RF pipeline to ONNX + TF-IDF JSON for the extension.

Usage (run from the repo root):

    # Undersampled variant -> rf_models/phishing_rf3.onnx + rf_models/tfidf_data.json
    python export_to_web.py --model rf_models/phishing_optimized.pkl --outdir rf_models \
        --onnx-name phishing_rf3.onnx

    # Non-undersampled variant (the one deployed in the extension)
    python export_to_web.py --model rf_models/rf_full.pkl --outdir rf_models \
        --onnx-name rf_full.onnx --tfidf-name rf_full_tfidf_data.json

The extension expects `phishing_rf.onnx` + `tfidf_data.json` inside
`phishingdetectorExt/` — copy them there after exporting.
"""

import argparse
import hashlib
import json
import os
import sys

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import alias_legacy_main  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Export the RF pipeline to ONNX + JSON.")
    parser.add_argument('--model', default=os.path.join('rf_models', 'phishing_optimized.pkl'),
                        help="Input .pkl pipeline (default: %(default)s)")
    parser.add_argument('--outdir', default='.',
                        help="Output directory (default: %(default)s)")
    parser.add_argument('--onnx-name', default='phishing_rf.onnx',
                        help="ONNX filename (default: %(default)s)")
    parser.add_argument('--tfidf-name', default='tfidf_data.json',
                        help="TF-IDF JSON filename (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Legacy pickles reference __main__.make_tokens etc.; new ones reference features.*
    alias_legacy_main()

    try:
        print(f"Memuat model pipeline: {args.model}")
        pipeline = joblib.load(args.model)
        print("Model berhasil dimuat.")
    except AttributeError as e:
        print(f"CRITICAL ERROR: {e}")
        print("Pastikan nama fungsi SAMA PERSIS dengan file training.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file .pkl tidak ditemukan: {args.model}")
        sys.exit(1)

    # Ambil komponen
    feature_union = pipeline.named_steps['features']
    tfidf_model = feature_union.transformer_list[0][1]
    rf_model = pipeline.named_steps['clf']

    # --- Ekspor JSON (TF-IDF) ---
    print("Mengekspor TF-IDF vocabulary...")
    vocab_raw = tfidf_model.vocabulary_
    vocab_clean = {k: int(v) for k, v in vocab_raw.items()}
    idf_clean = tfidf_model.idf_.tolist()

    tfidf_data = {
        "vocabulary": vocab_clean,
        "idf": idf_clean,
        "norm": tfidf_model.norm,
        "use_idf": tfidf_model.use_idf,
        "smooth_idf": tfidf_model.smooth_idf,
        "sublinear_tf": tfidf_model.sublinear_tf
    }

    tfidf_path = os.path.join(args.outdir, args.tfidf_name)
    with open(tfidf_path, "w") as f:
        json.dump(tfidf_data, f)
    print(f"-> {tfidf_path} OK ({len(vocab_clean)} terms)")

    # --- Ekspor ONNX (Random Forest) ---
    n_features = len(vocab_clean) + 9  # vocab + 9 structural features
    print(f"Jumlah fitur ONNX: {n_features}")

    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    print("Mengonversi ke ONNX...")

    onnx_model = convert_sklearn(
        rf_model,
        initial_types=initial_type,
        target_opset=12,
        options={id(rf_model): {'zipmap': False}}  # ZipMap tidak didukung browser
    )

    onnx_path = os.path.join(args.outdir, args.onnx_name)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"-> {onnx_path} OK ({os.path.getsize(onnx_path)/1024:.0f} KB)")
    print(f"   sha256 {sha256(onnx_path)}")
    print(f"   sha256 {sha256(tfidf_path)}  ({args.tfidf_name})")
    print("\nSUKSES! File siap untuk Web Extension.")


if __name__ == "__main__":
    main()
