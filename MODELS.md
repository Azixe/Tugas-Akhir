# Model Artifacts — Provenance & Checksums

> Single source of truth for *which model file is which*. Updated 2026-09-20.
> Verify any file with: `sha256sum <path>` (WSL) or `Get-FileHash <path>` (PowerShell).

## Deployed in the extension

| Artifact | Variant | Features | Origin | Size | SHA-256 | Paired files |
|---|---|---|---|---|---|---|
| `phishingdetectorExt/phishing_rf.onnx` | RF — **no undersampling** | 1509 (1500 TF-IDF + 9 struct) | Gen 3, Dec 2025. Original `.pkl` was overwritten and lost | 901 KB | `0a8f995b4869749c13dcd561970c2857ff6c21b7bdc2a08ed741652f06a47f5a` | `phishingdetectorExt/tfidf_data.json` (`3933ff90a34285cf40f88644ab06d9face451f44ed4870e96219d39c96b3748c`) |
| `phishingdetectorExt/phishing_xgb.onnx` | XGB — undersampled | 160 (PCA of 189) | Colab, Apr 2026 | 1,277 KB | `1c1965eb9da03ee459d3657b9a349764a7dcda1ac8658d88d9e31d7f8dc863d7` | `tfidf_data_xgb.json` (`3720ae3c6770b1eb4518d81ff255d24368f7c8f27507998bc185c44479f419d9`), `xgb_preprocessing.json` (`ac8848a21b02dfa23d085cf46daf92788a25272077267de33f56014084cf7d79`) |

## RF variants & history (`rf_models/`)

| Artifact | Variant | Notes | Size | SHA-256 |
|---|---|---|---|---|
| `phishing_optimized.pkl` | RF — **undersampled** (Gen 3) | Output of `hybrid/train_hybrid_model_v2.py` (default). Apr 12 2026 | 4.2 MB | `87250b42c266c8c63a3ea0ca9911479df0445cce026189a75a44276fa9da9e11` |
| `phishing_rf3.onnx` | ONNX of the undersampled RF | Exported Apr 12 2026 from `phishing_optimized.pkl`. Pairs with root `tfidf_data.json` | 670 KB | `4313d8bfd93882d2bfb86715948103974b102a829f851fc970a7be6a22f056ed` |
| `phishing_rf1.onnx` | RF — older generation | Pre-undersampling generation (same size class as deployed model) | 901 KB | `53cc8400650da07c2742ff0ce7fb441224daa22d06d7a5f66fbe14cebf24fc18` |
| `phishing_rf2.onnx` | RF — older generation | Same generation as rf1 | 901 KB | `c41951ea7c0849cf74dcf32958752fcfa62182b9c9e4273bf7ffc272560c06b8` |
| `phishing_hybrid_model.pkl` | RF — Gen 2 (legacy) | TF-IDF (5000) + 7 structural features | 61.5 MB | `7419fa3dde6313e14b94bc92b5e1842f2905fb49ea1e4c8a9bd754fb1387c6e9` |
| `phishing_detector_model.pkl` | RF — Gen 1 (legacy) | TF-IDF only, unlimited vocab | 242 MB | `a43368ae867dc020a1945c0c55d2178310d2d02772bdee4cd4e0f3bd8d649d5b` |

Root `tfidf_data.json` (`4f9cf4bd4c48e0d013171bc9959ad46cd3cf7afb8f09c6505138f611ca7213cf`) pairs with the **undersampled** RF export (`phishing_optimized.pkl` → `phishing_rf3.onnx`). Re-exporting the undersampled pkl reproduces this file byte-for-byte.

## XGBoost (`Xgboost/`)

| Artifact | Variant | Notes | Size | SHA-256 |
|---|---|---|---|---|
| `phishing_xgb_pipeline.pkl` | XGB — undersampled | Full artifact dict (model + scaler + selector + PCA + feature extractor). Colab, Apr 18 2026 | 767 KB | `8e751bb79bdbf5b9d50ccb02edceaf5016482ba1b6b45fe3d46f719c071bb601` |
| `phishing_xgb_model.pkl` | XGB — undersampled | Classifier only | 577 KB | `120ee8bfb06080b5d8a8c59017ba1349c7779c4932c13e063c0b1ce8d4bb5523` |
| `phishing_xgb.onnx` | ONNX of the above | Byte-identical to `phishingdetectorExt/phishing_xgb.onnx` | 1,277 KB | `1c1965eb9da03ee459d3657b9a349764a7dcda1ac8658d88d9e31d7f8dc863d7` |

## Regenerating

```bash
# RF undersampled (default) -> rf_models/phishing_optimized.pkl
python hybrid/train_hybrid_model_v2.py
python export_to_web.py --model rf_models/phishing_optimized.pkl --outdir rf_models --onnx-name phishing_rf3.onnx

# RF non-undersampled (deployed variant)
python hybrid/train_hybrid_model_v2.py --no-undersample --out rf_models/rf_full.pkl
python export_to_web.py --model rf_models/rf_full.pkl --outdir rf_models \
    --onnx-name rf_full.onnx --tfidf-name rf_full_tfidf_data.json

# XGBoost (Colab; upload URL dataset.csv, features.py, and the script)
!python train_xgb_colab.py                          # add --no-undersample for the full variant
!python export_xgb_onnx_colab.py                    # add --outdir / --pkl as needed
```

## Notes & policy

- **Lost artifact:** the original non-undersampled RF `.pkl` was overwritten in April 2026; only `phishingdetectorExt/phishing_rf.onnx` survives. Training/export scripts now require explicit `--out` / `--outdir` so variants cannot clobber each other.
- **Empty-token quirk:** exported TF-IDF vocabularies contain `''` as a term (the training tokenizer keeps empty strings); the JS/benchmark emulation filters empty tokens (`features.build_tfidf_vector`). This is intentional — keep it when comparing pkl vs ONNX.
- **Git tracking:** XGBoost binaries are tracked in git; RF binaries are ignored via `.gitignore` (`phishing_rf*.onnx`, `*.pkl`, …). Provenance is tracked here by SHA-256. If the repo grows, consider `git rm --cached` for the XGB binaries and rely on this manifest + local/Drive copies.
- **Python:** local runs use Anaconda Python 3.10 (`numpy 1.23.5`, `sklearn 1.2.1`). The XGB pkls were saved under NumPy 2.x on Colab and will **not** load locally until re-saved with `numpy<2` (see `task.md`).
