"""Shared feature extraction for the phishing URL detection pipeline.

Single source of truth for the Python side of the feature pipeline:
training (RF / XGBoost), ONNX export, benchmarks and local test utilities.

It mirrors `phishingdetectorExt/utils.js`:
    tokenize()      -> make_tokens() / build_tfidf_vector()
    structural()    -> StructuralFeatureExtractor / structural_features()
    entropy()       -> shannon_entropy()

Pickle compatibility
--------------------
Artifacts trained from a script's ``__main__`` namespace (all models saved
before 2026-09) reference ``__main__.make_tokens`` etc. Call
``alias_legacy_main()`` before ``joblib.load`` to keep those loadable.
New artifacts saved by scripts that import this module reference
``features.*`` and load anywhere this file is importable.

Colab note
----------
When running a training/export script on Google Colab, upload this file
next to the script (e.g. to /content/) so the import resolves.
"""

import math
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Trusted TLDs (must match COMMON_TLDS in utils.js)
COMMON_TLDS = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']


# ============================================================
# TRAINING-SIDE FEATURES (used by sklearn pipelines / pickles)
# ============================================================

def shannon_entropy(data):
    """Shannon entropy in bits. Matches ``entropy()`` in utils.js."""
    if not data:
        return 0
    entropy = 0
    for x in Counter(data).values():
        p_x = x / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """9 hand-crafted structural URL features.

    Feature order (must match utils.js `structural()`):
    [length, dot_count, slash_count, dash_count, at_count,
     digit_ratio, entropy, is_common_tld, subdomain_level]
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
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
            for tld in COMMON_TLDS:
                if s_url.endswith(tld) or s_url.endswith(tld + '/'):
                    is_common_tld = 1
                    break

            clean = s_url.replace("https://", "").replace("http://", "").split('/')[0]
            subdomain_level = clean.count('.')

            features.append([length, dot_count, slash_count, dash_count, at_count,
                             digit_ratio, entropy, is_common_tld, subdomain_level])

        return np.array(features)


def make_tokens(f):
    """URL tokenizer used by TfidfVectorizer during training.

    Split on '/', '-', '.'; deduplicate; drop 'com' and 'www'.
    NOTE: intentionally keeps empty strings (the historical vocabulary
    contains '' as a term). The inference-side emulation in
    ``build_tfidf_vector`` filters them, matching utils.js.
    """
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


def alias_legacy_main():
    """Expose the shared helpers in ``__main__`` for legacy pickles."""
    import __main__
    __main__.shannon_entropy = shannon_entropy
    __main__.StructuralFeatureExtractor = StructuralFeatureExtractor
    __main__.make_tokens = make_tokens
    __main__.COMMON_TLDS = COMMON_TLDS


# ============================================================
# INFERENCE-SIDE FEATURES (ONNX emulation, mirrors utils.js)
# ============================================================

def build_tfidf_vector(url, tfidf_data):
    """Build a TF-IDF vector from an exported vocabulary (matches JS).

    ``tfidf_data`` is the JSON exported by the training/export scripts:
    {'vocabulary': {...}, 'idf': [...], ...}
    """
    tokens = [t for t in make_tokens(url.lower()) if t]
    vocab = tfidf_data['vocabulary']
    idf = tfidf_data['idf']
    n_features = len(vocab)

    vec = np.zeros(n_features, dtype=np.float32)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1

    for i in range(n_features):
        vec[i] *= idf[i]

    norm = np.sqrt(np.sum(vec * vec))
    if norm > 0:
        vec /= norm
    return vec


def structural_features(url):
    """9 structural features as a float32 array (matches JS)."""
    s = url.lower()
    length = len(s)
    clean = s.replace("https://", "").replace("http://", "").split('/')[0]

    is_common_tld = 0
    for tld in COMMON_TLDS:
        if s.endswith(tld) or s.endswith(tld + '/'):
            is_common_tld = 1
            break

    return np.array([
        length,
        s.count('.'),
        s.count('/'),
        s.count('-'),
        s.count('@'),
        sum(c.isdigit() for c in s) / length if length > 0 else 0,
        shannon_entropy(s),
        is_common_tld,
        clean.count('.'),
    ], dtype=np.float32)


def extract_features_onnx(urls, tfidf_data):
    """Full feature matrix for a batch of URLs (TF-IDF + structural)."""
    features = []
    for url in urls:
        tfidf_vec = build_tfidf_vector(url, tfidf_data)
        struct = structural_features(url)
        features.append(np.concatenate([tfidf_vec, struct]))
    return np.array(features, dtype=np.float32)


def preprocess_xgb(features, prep_data):
    """Apply StandardScaler -> feature selection -> PCA (matches JS)."""
    mean = np.array(prep_data['scaler_mean'], dtype=np.float32)
    scale = np.array(prep_data['scaler_scale'], dtype=np.float32)
    scaled = (features - mean) / scale

    selected = scaled[:, prep_data['selected_feature_indices']]

    pca_mean = np.array(prep_data['pca_mean'], dtype=np.float32)
    pca_components = np.array(prep_data['pca_components'], dtype=np.float32)
    return ((selected - pca_mean) @ pca_components.T).astype(np.float32)
