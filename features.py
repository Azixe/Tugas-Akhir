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
import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Trusted TLDs (must match COMMON_TLDS in utils.js)
COMMON_TLDS = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id']

SCHEME_RE = re.compile(r'^[a-z][a-z0-9+.\-]*://')


def normalize_url(url, strip_scheme=False, strip_www=False):
    """Lowercase a URL and optionally drop its scheme and a leading ``www.``.

    Both switches default to off, which is the historical behaviour of every
    deployed model up to v3.4. They exist for the shortcut ablation: in the
    training data the scheme and the ``www.`` label separate the classes
    almost on their own (see ``sampling_results/dataset_split_report.md``).
    The scheme is removed from the string itself, so length, slash count and
    entropy stop seeing it too, not only the ``http:`` / ``https:`` tokens.

    Not mirrored in utils.js yet: a model trained with either switch on must
    not be deployed before the same normalisation exists there.
    """
    s = str(url).lower()
    m = SCHEME_RE.match(s)
    scheme, rest = (m.group(0), s[m.end():]) if m else ('', s)
    if strip_www and rest.startswith('www.'):
        rest = rest[4:]
    return rest if strip_scheme else scheme + rest


def url_host(url, strip_www=False):
    """Lowercased host of a URL: everything between the scheme and the first '/', '?' or '#'.

    A port or ``user@`` prefix stays in the string; the character n-gram block
    only needs the same cut on both sides, not a full URL parse.
    """
    return re.split(r'[/?#]', normalize_url(url, strip_scheme=True, strip_www=strip_www), 1)[0]


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

    ``strip_scheme`` / ``strip_www`` are the ablation switches of
    ``normalize_url``; the defaults keep the historical features.
    """

    def __init__(self, strip_scheme=False, strip_www=False):
        self.strip_scheme = strip_scheme
        self.strip_www = strip_www

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # getattr: extractors pickled before the switches existed lack them.
        strip_scheme = getattr(self, 'strip_scheme', False)
        strip_www = getattr(self, 'strip_www', False)
        features = []
        for url in X:
            s_url = normalize_url(url, strip_scheme, strip_www)

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


class UrlTokenizer:
    """``make_tokens`` with the ablation switches; picklable for TfidfVectorizer.

    With all switches off it returns exactly what ``make_tokens`` returns for
    a lowercased URL. ``drop_empty`` removes the empty-string token at
    training time, which is what the inference side (utils.js and
    ``build_tfidf_vector``) has always done.
    """

    def __init__(self, strip_scheme=False, strip_www=False, drop_empty=False):
        self.strip_scheme = strip_scheme
        self.strip_www = strip_www
        self.drop_empty = drop_empty

    def __call__(self, f):
        tokens = make_tokens(normalize_url(f, self.strip_scheme, self.strip_www))
        return [t for t in tokens if t] if self.drop_empty else tokens


class UrlText:
    """Text handed to the character n-gram vectorizer; picklable.

    ``scope='host'`` (default) returns ``url_host`` so the n-grams describe the
    domain name, which is where typosquatting and look-alike brands live;
    the scheme never reaches this block. ``scope='url'`` returns the whole
    normalised URL and then honours ``strip_scheme`` as well.

    Not mirrored in utils.js yet: a model trained with this block must not be
    deployed before the same n-gram extraction exists there.
    """

    def __init__(self, strip_scheme=False, strip_www=False, scope='host'):
        if scope not in ('host', 'url'):
            raise ValueError(f"scope must be 'host' or 'url', got {scope!r}")
        self.strip_scheme = strip_scheme
        self.strip_www = strip_www
        self.scope = scope

    def __call__(self, url):
        if self.scope == 'host':
            return url_host(url, self.strip_www)
        return normalize_url(url, self.strip_scheme, self.strip_www)


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
    {'vocabulary': {...}, 'idf': [...], ...}. Optional ``strip_scheme`` /
    ``strip_www`` keys select the normalisation the model was trained with.
    """
    s = normalize_url(url, tfidf_data.get('strip_scheme', False),
                      tfidf_data.get('strip_www', False))
    tokens = [t for t in make_tokens(s) if t]
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


def structural_features(url, strip_scheme=False, strip_www=False):
    """9 structural features as a float32 array (matches JS)."""
    s = normalize_url(url, strip_scheme, strip_www)
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
        struct = structural_features(url, tfidf_data.get('strip_scheme', False),
                                     tfidf_data.get('strip_www', False))
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
