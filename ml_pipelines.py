"""Model pipelines for the sampling study (fixed hyperparameters).

- RF: Gen-3 architecture (``RF_HPARAMS``), same as
  ``hybrid/train_hybrid_model_v2.py``.
- XGB: the same stage order as the Colab pipeline
  (``Xgboost/train_xgb_colab.py``) but with the hyperparameters found by the
  Colab search (``BEST_XGB_PARAMS`` / ``IMPORTANCE_PARAMS``). Keeping them
  fixed means the *only* variable between sampling variants is the training
  data (undersampled vs full), which is the point of the comparison.
"""

from __future__ import annotations

import gc
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import StructuralFeatureExtractor, UrlText, UrlTokenizer, make_tokens

SEED = 42

# Gen-3 RF hyperparameters (hybrid/train_hybrid_model_v2.py)
RF_HPARAMS = dict(
    n_estimators=100,
    n_jobs=-1,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
)

# Best XGBoost hyperparameters from the Colab RandomizedSearchCV run
BEST_XGB_PARAMS = dict(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=700,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.1,
    reg_lambda=2,
)

# Feature-importance pruning model (same as the Colab pipeline)
IMPORTANCE_PARAMS = dict(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
)

# The Colab tuning run kept all 1509 features after SelectKBest; fix it so
# every variant runs the identical feature-reduction pipeline.
K_BEST = 'all'


FEATURE_GROUPS = ('text', 'structural')      # the historical 1,509-dim extractor
BLOCK_ORDER = ('text', 'char', 'structural')  # column order when the char block is added

# Character n-gram block (ablation only; see features.UrlText)
CHAR_FEATURES = 1500
CHAR_NGRAM = (3, 5)
CHAR_SCOPE = 'host'


def build_feature_union(max_features: int = 1500, strip_scheme: bool = False,
                        strip_www: bool = False, drop_empty: bool = False,
                        groups=FEATURE_GROUPS, char_features: int = CHAR_FEATURES,
                        char_ngram=CHAR_NGRAM, char_scope: str = CHAR_SCOPE) -> FeatureUnion:
    """TF-IDF + structural feature union, optionally with a character n-gram block.

    The defaults build the historical 1,509-dim extractor. The keyword
    switches are the ablation arms (see ``features.normalize_url``);
    ``groups`` selects blocks: ``'text'`` (word-level TF-IDF), ``'char'``
    (character n-gram TF-IDF, off by default) and ``'structural'``.
    """
    if strip_scheme or strip_www or drop_empty:
        tokenizer = UrlTokenizer(strip_scheme, strip_www, drop_empty)
    else:
        tokenizer = make_tokens  # same object the v3.4 pickles reference
    blocks = {
        'text': ('text_features', TfidfVectorizer(
            tokenizer=tokenizer, token_pattern=None, max_features=max_features)),
        'char': ('char_features', TfidfVectorizer(
            analyzer='char', ngram_range=tuple(char_ngram), lowercase=False,
            preprocessor=UrlText(strip_scheme, strip_www, char_scope),
            max_features=char_features)),
        'structural': ('structural_features',
                       StructuralFeatureExtractor(strip_scheme, strip_www)),
    }
    unknown = set(groups) - set(BLOCK_ORDER)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    return FeatureUnion([blocks[g] for g in BLOCK_ORDER if g in groups])


def build_rf_pipeline(**feature_kwargs) -> Pipeline:
    return Pipeline([
        ('features', build_feature_union(**feature_kwargs)),
        ('clf', RandomForestClassifier(**RF_HPARAMS)),
    ])


def _to_dense(X) -> np.ndarray:
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def train_xgb_pipeline(urls, labels, verbose: bool = True, **feature_kwargs) -> dict:
    """Train the staged XGB pipeline and return an artifacts dict.

    Schema matches ``Xgboost/train_xgb_colab.py`` so the existing export
    scripts can consume it. ``feature_kwargs`` go to ``build_feature_union``.
    """
    t0 = time.time()

    feature_extractor = build_feature_union(**feature_kwargs)
    X_raw = _to_dense(feature_extractor.fit_transform(urls, labels))
    if verbose:
        print(f"    features: {X_raw.shape} ({time.time()-t0:.0f}s)", flush=True)

    # Free each dense copy as soon as the next stage has its own: with the char
    # block the matrix is twice as wide, and three live copies do not fit in RAM.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32, copy=False)
    del X_raw
    gc.collect()

    selector = SelectKBest(score_func=f_classif, k=K_BEST)
    X_kbest = selector.fit_transform(X_scaled, labels).astype(np.float32, copy=False)
    del X_scaled
    gc.collect()

    importance_xgb = XGBClassifier(
        tree_method='hist', device='cpu', random_state=SEED, verbosity=0,
        **IMPORTANCE_PARAMS)
    importance_xgb.fit(X_kbest, labels)

    importances = importance_xgb.feature_importances_
    important_mask = importances > np.median(importances)
    if important_mask.sum() < 20:  # same guard as the Colab script
        top = np.argsort(importances)[-max(20, len(importances) // 2):]
        important_mask = np.zeros(len(importances), dtype=bool)
        important_mask[top] = True

    X_imp = X_kbest[:, important_mask]
    pca = PCA(n_components=0.95, random_state=SEED)
    X_pca = pca.fit_transform(X_imp).astype(np.float32)
    del X_kbest, X_imp
    gc.collect()

    model = XGBClassifier(
        tree_method='hist', device='cpu', random_state=SEED, verbosity=0,
        eval_metric='logloss', **BEST_XGB_PARAMS)
    model.fit(X_pca, labels)

    artifacts = {
        'model': model,
        'scaler': scaler,
        'selector_kbest': selector,
        'important_mask': important_mask,
        'pca': pca,
        'feature_extractor': feature_extractor,
        'best_params': dict(BEST_XGB_PARAMS),
        'feature_counts': {
            'original': len(importances),
            'after_kbest': len(importances),
            'after_importance': int(important_mask.sum()),
            'after_pca': int(pca.n_components_),
        },
    }
    if verbose:
        print(f"    XGB feature counts: {artifacts['feature_counts']} "
              f"(total {time.time()-t0:.0f}s)", flush=True)
    return artifacts


def _xgb_matrix(artifacts: dict, urls) -> np.ndarray:
    """Run the fitted preprocessing chain for a batch of URLs."""
    X_raw = _to_dense(artifacts['feature_extractor'].transform(urls))
    X_scaled = artifacts['scaler'].transform(X_raw)
    X_kbest = artifacts['selector_kbest'].transform(X_scaled)
    X_imp = X_kbest[:, artifacts['important_mask']]
    return artifacts['pca'].transform(X_imp).astype(np.float32)


def predict_with_xgb(artifacts: dict, urls) -> np.ndarray:
    return artifacts['model'].predict(_xgb_matrix(artifacts, urls))


def predict_proba_with_xgb(artifacts: dict, urls) -> np.ndarray:
    return artifacts['model'].predict_proba(_xgb_matrix(artifacts, urls))