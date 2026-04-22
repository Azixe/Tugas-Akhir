"""
XGBoost ONNX Export Script — Run on Google Colab
=================================================
Run this AFTER train_xgb_colab.py in the same Colab session,
or after uploading phishing_xgb_pipeline.pkl to Colab.

Outputs:
  - phishing_xgb.onnx         (XGBoost model for ONNX Runtime)
  - xgb_preprocessing.json    (scaler, feature indices, PCA matrix)
  - tfidf_data_xgb.json       (TF-IDF vocabulary + IDF weights)
"""

# === Install dependencies ===
import subprocess
subprocess.run(['pip', 'install', 'onnxmltools', 'onnxconverter-common', 'onnx', 'skl2onnx', 'onnxruntime'], 
               capture_output=True)

import joblib
import json
import numpy as np
import math
import os
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

# === These must be defined so pickle can deserialize the pipeline ===

def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in Counter(data).values():
        p_x = x / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy

class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']
        for url in X:
            s_url = str(url).lower()
            length = len(s_url)
            dot_count = s_url.count('.')
            slash_count = s_url.count('/')
            dash_count = s_url.count('-')
            at_count = s_url.count('@')
            digit_count = sum(c.isdigit() for c in s_url)
            digit_ratio = digit_count / length if length > 0 else 0
            entropy = shannon_entropy(s_url)
            is_common_tld = 0
            for tld in common_tlds:
                if s_url.endswith(tld) or s_url.endswith(tld + '/'):
                    is_common_tld = 1
                    break
            try:
                clean = s_url.replace("https://", "").replace("http://", "").split('/')[0]
                subdomain_level = clean.count('.')
            except:
                subdomain_level = 0
            features.append([length, dot_count, slash_count, dash_count, at_count,
                             digit_ratio, entropy, is_common_tld, subdomain_level])
        return np.array(features)

def make_tokens(f):
    tokens_by_slash = str(f).encode('utf-8').decode('utf-8').split('/')
    total_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')
        tokens_dot = []
        for j in range(0, len(tokens)):
            temp_tokens = str(tokens[j]).split('.')
            tokens_dot = tokens_dot + temp_tokens
        total_tokens = total_tokens + tokens + tokens_dot
    total_tokens = list(set(total_tokens))
    if 'com' in total_tokens: total_tokens.remove('com')
    if 'www' in total_tokens: total_tokens.remove('www')
    return total_tokens

# === Load pipeline ===
pkl_paths = [
    'phishing_xgb_pipeline.pkl',
    '/content/phishing_xgb_pipeline.pkl',
]

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

# Extract TfidfVectorizer from the FeatureUnion
tfidf_vectorizer = feature_extractor.transformer_list[0][1]

vocab = tfidf_vectorizer.vocabulary_
idf = tfidf_vectorizer.idf_.tolist()

# Convert vocab keys to strings (JSON requires string keys)
vocab_str = {str(k): int(v) for k, v in vocab.items()}

tfidf_data = {
    'vocabulary': vocab_str,
    'idf': idf,
    'max_features': len(vocab_str)
}

with open('tfidf_data_xgb.json', 'w') as f:
    json.dump(tfidf_data, f)
print(f">> tfidf_data_xgb.json ({len(vocab_str)} terms)")

# ============================================================
# 2. Export preprocessing pipeline (scaler + feature selection + PCA)
# ============================================================
print("\n--- Exporting preprocessing pipeline ---")

# Get the feature indices that survive both SelectKBest and importance pruning
# SelectKBest selects from 1509 -> n_after_kbest
kbest_indices = selector_kbest.get_support(indices=True)

# Importance mask selects from n_after_kbest -> n_after_importance  
importance_indices = np.where(important_mask)[0]

# Combined: map importance indices back to original 1509 feature space
combined_indices = kbest_indices[importance_indices].tolist()

print(f"  SelectKBest kept: {len(kbest_indices)} features")
print(f"  Importance kept: {len(importance_indices)} features")
print(f"  Combined indices: {len(combined_indices)} features from original {feature_counts['original']}")
print(f"  PCA components: {pca.n_components_}")

preprocessing = {
    # StandardScaler params (applied to all 1509 features before selection)
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    
    # Combined feature selection (from 1509 -> n_selected)
    # These are indices into the 1509-feature vector AFTER scaling
    'selected_feature_indices': combined_indices,
    
    # PCA params (applied after feature selection)
    'pca_mean': pca.mean_.tolist(),                    # shape: [n_selected]
    'pca_components': pca.components_.tolist(),         # shape: [n_pca, n_selected]
    'n_pca_components': int(pca.n_components_),
    
    # Metadata
    'n_original_features': feature_counts['original'],
    'n_selected_features': len(combined_indices),
    'n_pca_features': int(pca.n_components_),
}

with open('xgb_preprocessing.json', 'w') as f:
    json.dump(preprocessing, f)

# Print file size
prep_size = os.path.getsize('xgb_preprocessing.json') / 1024
print(f">> xgb_preprocessing.json ({prep_size:.0f} KB)")

# ============================================================
# 3. Export XGBoost model to ONNX
# ============================================================
print("\n--- Exporting XGBoost to ONNX ---")

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

n_input_features = int(pca.n_components_)  # PCA output dimension
print(f"ONNX input shape: [batch, {n_input_features}]")

initial_type = [('float_input', FloatTensorType([None, n_input_features]))]

onnx_model = convert_xgboost(
    model, 
    initial_types=initial_type,
    target_opset=11
)

with open('phishing_xgb.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

onnx_size = os.path.getsize('phishing_xgb.onnx') / 1024
print(f">> phishing_xgb.onnx ({onnx_size:.0f} KB)")

# ============================================================
# 4. Verify ONNX model
# ============================================================
print("\n--- Verifying ONNX model ---")

import onnx
onnx_loaded = onnx.load('phishing_xgb.onnx')
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

session = ort.InferenceSession('phishing_xgb.onnx')
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
print(f"Files for Chrome extension:")
print(f"  1. phishing_xgb.onnx           ({onnx_size:.0f} KB) - XGBoost model")
print(f"  2. xgb_preprocessing.json      ({prep_size:.0f} KB) - Scaler + feature selection + PCA")
print(f"  3. tfidf_data_xgb.json         ({os.path.getsize('tfidf_data_xgb.json')/1024:.0f} KB) - TF-IDF vocabulary")
print(f"\nPipeline: TF-IDF+Structural(1509) -> Scale -> Select({len(combined_indices)}) -> PCA({n_input_features}) -> XGBoost")

# === Download files ===
try:
    from google.colab import files
    print("\nDownloading files...")
    files.download('phishing_xgb.onnx')
    files.download('xgb_preprocessing.json')
    files.download('tfidf_data_xgb.json')
except ImportError:
    print("\nFiles saved to current directory.")
