"""Parity test: features.py must reproduce the historical implementations.

The pipeline originally duplicated tokenizer / structural-feature code in
every training, export and benchmark script. After the 2026-09 refactor
those definitions live only in ``features.py``; this test pins the module
to the exact behavior of the old copies.

Run from the repo root (uses plain asserts, no test framework required):

    python tests/test_features_parity.py
"""

import math
import os
import sys
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import features  # noqa: E402

URLS = [
    "https://google.com",
    "https://www.google.com/search?q=test",
    "http://paypal.com.secure-login.xyz/verify",
    "https://steamcommunity.com/id/WhyIzMyLifeLikeDiz/",
    "https://github.com/openai/gpt-4",
    "http://192.168.1.1/admin",
    "https://a.b.c.evil.com/login",
    "http://faceb00k.com/login.php",
    "https://www.kompas.com/read/2024/01/01",
    "http://secure-update-paypal.com/account/verify",
    "https://twitter.com/home",
    "https://drive.google.com/file/d/abc123/view",
    "http://update.microsoft.com.security-check.top/",
    "https://en.wikipedia.org/wiki/Phishing",
    "http://bit.ly/3xYzAbc",
    "https://myaccount.google.com/security",
    "http://login-appleid.com-verify.support/",
    "https://reddit.com/r/programming/comments/abc123/title_here/",
    "http://www.example.co.id/path/to/page?query=12345&x=abc",
    "https://web.whatsapp.com/",
    "http://amaz0n-prime.com/checkout",
    "https://mail.google.com/mail/u/0/",
    "http://dropbox.com.verify-login.club/session",
    "https://stackoverflow.com/questions/12345/how-to-do-x",
    "http://steamcommunitty.com/tradeoffer/new",
    "https://id.wikipedia.org/wiki/Situs_web",
    "http://secure.bank-login.co.id.verify.xyz/",
    "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "http://tinyurl.com/2p8x4z3k",
    "http://xn--pypal-4ve.com/",
]


# ============================================================
# Historical implementations (verbatim from the pre-refactor scripts)
# ============================================================

def old_shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in Counter(data).values():
        p_x = x / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy


class OldStructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_list = []
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
            entropy = old_shannon_entropy(s_url)
            is_common_tld = 0
            for tld in common_tlds:
                if s_url.endswith(tld) or s_url.endswith(tld + '/'):
                    is_common_tld = 1
                    break
            try:
                clean = s_url.replace("https://", "").replace("http://", "").split('/')[0]
                subdomain_level = clean.count('.')
            except:  # noqa: E722 - kept verbatim from the old code
                subdomain_level = 0
            features_list.append([length, dot_count, slash_count, dash_count, at_count,
                                  digit_ratio, entropy, is_common_tld, subdomain_level])
        return np.array(features_list)


def old_make_tokens(f):
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


# ============================================================
# Checks
# ============================================================

def test_tokenizer_parity():
    for url in URLS:
        old = sorted(old_make_tokens(url))
        new = sorted(features.make_tokens(url))
        assert old == new, f"tokenizer mismatch for {url!r}:\n old={old}\n new={new}"
    print(f"tokenizer parity: OK ({len(URLS)} URLs)")


def test_structural_parity():
    old = OldStructuralFeatureExtractor().transform(URLS)
    new = features.StructuralFeatureExtractor().transform(URLS)
    assert new.shape == (len(URLS), 9), f"unexpected shape {new.shape}"
    assert np.allclose(old, new), "structural feature mismatch"
    print("structural parity: OK")


def test_entropy_parity():
    for url in URLS:
        assert abs(old_shannon_entropy(url) - features.shannon_entropy(url)) < 1e-12
    print("entropy parity: OK")


def test_onnx_structural_matches_class():
    class_out = features.StructuralFeatureExtractor().transform(URLS)
    for i, url in enumerate(URLS):
        single = features.structural_features(url)
        assert np.allclose(single, class_out[i].astype(np.float32), atol=1e-6), \
            f"onnx structural mismatch for {url!r}"
    print("onnx structural emulation: OK")


def test_legacy_pickle_loads():
    pkl = os.path.join(ROOT, "rf_models", "phishing_optimized.pkl")
    if not os.path.exists(pkl):
        print("legacy pickle load: SKIPPED (file not found)")
        return
    import joblib
    features.alias_legacy_main()
    pipeline = joblib.load(pkl)
    preds = pipeline.predict(URLS[:5]).tolist()
    assert set(preds) <= {0, 1}, f"unexpected labels {preds}"
    print("legacy pickle load: OK")


# ============================================================
# Ablation switches (scheme / www. / empty token) - off by default
# ============================================================

def test_normalize_url():
    n = features.normalize_url
    assert n("HTTPS://WWW.Example.com/A") == "https://www.example.com/a"
    assert n("https://www.example.com/a", strip_scheme=True) == "www.example.com/a"
    assert n("https://www.example.com/a", strip_www=True) == "https://example.com/a"
    assert n("http://www.example.com/", True, True) == "example.com/"
    assert n("ftp://www.example.com", True, True) == "example.com"
    # only a leading host label is removed, and only the exact label "www"
    assert n("https://www2.example.com/www.x", True, True) == "www2.example.com/www.x"
    assert n("https://a.www.example.com", True, True) == "a.www.example.com"
    assert n("example.com/http://x", True, True) == "example.com/http://x"
    print("normalize_url: OK")


def test_url_tokenizer_defaults_match_make_tokens():
    tok = features.UrlTokenizer()
    for url in URLS:
        assert sorted(tok(url)) == sorted(features.make_tokens(url.lower())), url
    print("UrlTokenizer defaults: OK")


def test_url_tokenizer_switches():
    url = "https://www.paypal.com.secure-login.xyz/verify/"
    plain = features.UrlTokenizer()(url)
    assert "https:" in plain and "" in plain and "www.paypal.com.secure" in plain
    stripped = features.UrlTokenizer(strip_scheme=True, strip_www=True, drop_empty=True)(url)
    assert "https:" not in stripped and "" not in stripped
    assert "paypal.com.secure" in stripped and "www.paypal.com.secure" not in stripped
    assert set(stripped) < set(plain) | {"paypal.com.secure"}
    print("UrlTokenizer switches: OK")


def test_structural_switches():
    for flags in ((True, False), (False, True), (True, True)):
        class_out = features.StructuralFeatureExtractor(*flags).transform(URLS)
        for i, url in enumerate(URLS):
            single = features.structural_features(url, *flags)
            assert np.allclose(single, class_out[i].astype(np.float32), atol=1e-6), (flags, url)
    base = features.structural_features("https://www.example.com/a")
    no_scheme = features.structural_features("https://www.example.com/a", strip_scheme=True)
    both = features.structural_features("https://www.example.com/a", True, True)
    assert base[0] - no_scheme[0] == 8 and base[2] - no_scheme[2] == 2   # length, slashes
    assert no_scheme[0] - both[0] == 4 and no_scheme[1] - both[1] == 1   # length, dots
    assert base[8] == no_scheme[8] == both[8] + 1                        # subdomain_level
    print("structural switches: OK")


def test_inference_tfidf_follows_export_flags():
    data = {'vocabulary': {'https:': 0, 'example.com': 1, 'www.example.com': 2, 'a': 3},
            'idf': [1.0, 1.0, 1.0, 1.0]}
    url = "https://www.example.com/a"
    assert features.build_tfidf_vector(url, data).nonzero()[0].tolist() == [0, 2, 3]
    flagged = dict(data, strip_scheme=True, strip_www=True)
    assert features.build_tfidf_vector(url, flagged).nonzero()[0].tolist() == [1, 3]
    assert features.extract_features_onnx([url], flagged).shape == (1, 4 + 9)
    print("inference TF-IDF flags: OK")


# ============================================================
# Character n-gram block - not part of the default feature union
# ============================================================

def test_url_host_and_text():
    h = features.url_host
    assert h("HTTPS://WWW.PayPal.com/Signin?x=1") == "www.paypal.com"
    assert h("https://www.paypal.com/signin", strip_www=True) == "paypal.com"
    assert h("http://paypal.com.secure-login.xyz") == "paypal.com.secure-login.xyz"
    assert h("example.com?next=http://x/y") == "example.com"
    assert h("https://a.b.c:8080/#frag") == "a.b.c:8080"
    url = "http://www.faceb00k.com/login.php"
    assert features.UrlText()(url) == "www.faceb00k.com"
    assert features.UrlText(strip_www=True)(url) == "faceb00k.com"
    assert features.UrlText(scope='url')(url) == url
    assert features.UrlText(True, True, scope='url')(url) == "faceb00k.com/login.php"
    print("url_host / UrlText: OK")


def test_char_block_is_opt_in():
    import pickle

    import ml_pipelines

    default = ml_pipelines.build_feature_union()
    assert [name for name, _ in default.transformer_list] == ['text_features', 'structural_features']

    union = ml_pipelines.build_feature_union(
        strip_scheme=True, strip_www=True, groups=('text', 'char', 'structural'), char_features=50)
    assert [name for name, _ in union.transformer_list] == \
        ['text_features', 'char_features', 'structural_features']
    X = union.fit_transform(URLS)
    char = union.transformer_list[1][1]
    terms = list(char.vocabulary_)
    assert X.shape[1] == len(union.transformer_list[0][1].vocabulary_) + len(terms) + 9
    assert len(terms) == 50 and all(3 <= len(t) <= 5 for t in terms)
    # host only, prefixes stripped: no scheme and no path separator in any term
    assert not any('/' in t or ':' in t for t in terms)
    hosts = [features.url_host(u, strip_www=True) for u in URLS]
    assert all(any(t in host for host in hosts) for t in terms)
    again = pickle.loads(pickle.dumps(union))
    assert (again.transform(URLS) != union.transform(URLS)).nnz == 0
    print("character n-gram block: OK")


if __name__ == "__main__":
    test_tokenizer_parity()
    test_structural_parity()
    test_entropy_parity()
    test_onnx_structural_matches_class()
    test_legacy_pickle_loads()
    test_normalize_url()
    test_url_tokenizer_defaults_match_make_tokens()
    test_url_tokenizer_switches()
    test_structural_switches()
    test_inference_tfidf_follows_export_flags()
    test_url_host_and_text()
    test_char_block_is_opt_in()
    print("\nAll parity checks passed.")
