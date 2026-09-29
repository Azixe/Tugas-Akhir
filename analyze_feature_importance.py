"""Feature-importance analysis for RQ2 (Bab 4) on the shared protocol.

Two importance sources, both trained on the *train split only* (seed 42):

1. Random Forest — ``feature_importances_`` (MDI) of the deployed
   full-dataset pipeline (``sampling_results/rf_full.pkl``) over the raw
   1,509 features (1,500 TF-IDF tokens + 9 structural).
2. XGBoost — replays the importance-pruning model (200 trees, depth 7) exactly
   as ``ml_pipelines.train_xgb_pipeline`` does: saved feature extractor ->
   saved StandardScaler -> SelectKBest(k='all') -> XGB importances. Reports
   the pruning mask ("importance > median") and whether each of the nine
   structural features survives it.

Usage (repo root):
    python analyze_feature_importance.py

Outputs:
    sampling_results/feature_importance_report.md
    sampling_results/feature_importance.json
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_pipelines  # noqa: E402
import sampling  # noqa: E402

from xgboost import XGBClassifier  # noqa: E402

OUTDIR = 'sampling_results'
STRUCT_NAMES = ['url_length', 'dot_count', 'slash_count', 'dash_count', 'at_count',
                'digit_ratio', 'entropy', 'is_common_tld', 'subdomain_level']


def feature_names(feature_extractor):
    """Full 1,509-name list in FeatureUnion order (text first, then structural)."""
    tfidf = feature_extractor.transformer_list[0][1]
    inv = {i: tok for tok, i in tfidf.vocabulary_.items()}
    n_text = len(tfidf.vocabulary_)
    names = [inv.get(i, f'text#{i}') for i in range(n_text)]
    return names + STRUCT_NAMES, n_text


def top_features(importances, names, n=20):
    order = np.argsort(importances)[::-1]
    return [{'rank': r + 1, 'feature': names[i], 'importance': round(float(importances[i]), 6),
             'type': 'structural' if names[i] in STRUCT_NAMES else 'tfidf'}
            for r, i in enumerate(order[:n])]


def main():
    df = sampling.load_clean_dataset()
    train_pool, _ = sampling.common_split(df)
    urls = train_pool['url'].values
    y = train_pool['label'].values
    print(f"Train split: {len(urls)} URLs (seed {sampling.SEED})")

    result = {}

    # ---------- Random Forest ----------
    print("RF importances (deployed rf_full.pkl) ...", flush=True)
    rf = joblib.load(os.path.join(OUTDIR, 'rf_full.pkl'))
    fx_rf = rf.named_steps['features']
    imp_rf = rf.named_steps['clf'].feature_importances_
    names_rf, n_text_rf = feature_names(fx_rf)
    assert len(imp_rf) == len(names_rf), (len(imp_rf), len(names_rf))

    order_rf = np.argsort(imp_rf)[::-1]
    rank_of = {int(i): r + 1 for r, i in enumerate(order_rf)}
    struct_rf = {name: {'importance': round(float(imp_rf[n_text_rf + j]), 6),
                        'rank': rank_of[n_text_rf + j]}
                 for j, name in enumerate(STRUCT_NAMES)}
    result['rf'] = {
        'n_features': len(imp_rf),
        'n_text': n_text_rf,
        'top20': top_features(imp_rf, names_rf),
        'structural': struct_rf,
    }

    # ---------- XGBoost (importance replay) ----------
    print("XGB importances (replay of the pruning model) ...", flush=True)
    art = joblib.load(os.path.join(OUTDIR, 'xgb_full.pkl'))
    fx_x = art['feature_extractor']
    scaler = art['scaler']
    names_x, n_text_x = feature_names(fx_x)

    X_raw = ml_pipelines._to_dense(fx_x.transform(urls))
    X = scaler.transform(X_raw).astype(np.float32)
    del X_raw
    probe = XGBClassifier(tree_method='hist', device='cpu', random_state=sampling.SEED,
                          verbosity=0, **ml_pipelines.IMPORTANCE_PARAMS)
    probe.fit(X, y)
    imp_x = probe.feature_importances_
    del X

    mask = imp_x > np.median(imp_x)
    order_x = np.argsort(imp_x)[::-1]
    rank_x = {int(i): r + 1 for r, i in enumerate(order_x)}
    struct_x = {name: {'importance': round(float(imp_x[n_text_x + j]), 6),
                       'rank': rank_x[n_text_x + j],
                       'survived_mask': bool(mask[n_text_x + j])}
                for j, name in enumerate(STRUCT_NAMES)}
    result['xgb'] = {
        'n_features': len(imp_x),
        'n_text': n_text_x,
        'mask_threshold': round(float(np.median(imp_x)), 8),
        'n_survived': int(mask.sum()),
        'n_structural_survived': int(sum(mask[n_text_x + j] for j in range(9))),
        'top20': top_features(imp_x, names_x),
        'structural': struct_x,
    }

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, 'feature_importance.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    lines = [
        "# Feature Importance (RQ2) — Random Forest & XGBoost",
        "",
        f"Train split only (common protocol, seed {sampling.SEED}); the test set is never used here.",
        "",
        "## Random Forest — top 20 (MDI, raw 1,509 features)",
        "",
        "| Rank | Feature | Type | Importance |",
        "|---:|---|---|---:|",
    ]
    for r in result['rf']['top20']:
        lines.append(f"| {r['rank']} | `{r['feature']}` | {r['type']} | {r['importance']:.6f} |")
    lines += ["", "### Nine structural features (RF)", "",
              "| Feature | Importance | Rank (of 1,509) |", "|---|---:|---:|"]
    for name, s in struct_rf.items():
        lines.append(f"| `{name}` | {s['importance']:.6f} | {s['rank']} |")

    lines += ["", "## XGBoost — top 20 (importance model, 200 trees / depth 7)", "",
              "| Rank | Feature | Type | Importance |", "|---:|---|---|---:|"]
    for r in result['xgb']['top20']:
        lines.append(f"| {r['rank']} | `{r['feature']}` | {r['type']} | {r['importance']:.6f} |")
    lines += ["", f"### Nine structural features (XGB, mask = importance > median = {result['xgb']['mask_threshold']})", "",
              "| Feature | Importance | Rank (of 1,509) | Survived pruning? |", "|---|---:|---:|:--:|"]
    for name, s in struct_x.items():
        lines.append(f"| `{name}` | {s['importance']:.6f} | {s['rank']} | "
                     f"{'yes' if s['survived_mask'] else 'no'} |")
    lines += ["",
              f"Pruning mask: {result['xgb']['n_survived']} of {result['xgb']['n_features']} features survive "
              f"({result['xgb']['n_structural_survived']}/9 structural).",
              "> TF-IDF importances come from the exact pipeline steps used at training time "
              "(saved extractor + saved scaler + SelectKBest k='all').",
              ""]
    with open(os.path.join(OUTDIR, 'feature_importance_report.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("Saved: feature_importance_report.md / .json")
    print(f"  RF structural ranks: " + ", ".join(f"{k}={v['rank']}" for k, v in struct_rf.items()))
    print(f"  XGB survived: {result['xgb']['n_survived']} ({result['xgb']['n_structural_survived']}/9 structural)")


if __name__ == '__main__':
    main()
