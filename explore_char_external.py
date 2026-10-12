"""Exploratory: score the character n-gram arms on the frozen external lists.

NOT part of the pre-registered external test (``external_500/PROTOCOL.md``). The lists in
``external_500/frozen/`` were collected and scored before these models existed, and their
results were already known when the character block was designed, so this run cannot
qualify a model for deployment. It answers one question cheaply: is there any sign that
character n-grams change the external picture? A pass here would still need a fresh sample.

Reads : external_500/frozen/*.csv (read-only), <models dir>/<arm>__<family>.pkl
Writes: <models dir>/external_exploratory.json, <models dir>/external_exploratory.md

    python explore_char_external.py --models-dir ablation_char_results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np

import features
from benchmark_sampling import make_predictors

FROZEN = os.path.join('external_500', 'frozen')
ARMS = ['all', 'no_scheme_www', 'char', 'char_no_scheme_www', 'char_only', 'char_only_no_scheme_www']
FAMILIES = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'cb': 'CatBoost'}
DETECT, BLOCK = 0.60, 0.80      # extension thresholds, strict >


def rows(name):
    with open(os.path.join(FROZEN, name), newline='', encoding='utf-8') as f:
        return [{k: v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def pct(n, d):
    return f"{100 * n / d:.1f}% ({n}/{d})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--models-dir', required=True)
    args = ap.parse_args()

    legit, phish, prefix = rows('legit_urls.csv'), rows('phishing_urls.csv'), rows('prefix_sensitivity.csv')
    sets = {
        'legit': [r['browser_url'] or r['url'] for r in legit],
        'phish_browser': [r['browser_url'] or r['url'] for r in phish],
        'phish_listed': [r['url'] for r in phish],
        'https': [r['url_https'] for r in prefix],
        'www': [r['url_https_www'] for r in prefix],
        'http': [r['url_http'] for r in prefix],
    }
    has_www = np.array([features.url_host(u).startswith('www.') for u in sets['legit']])
    print(f"legit {len(sets['legit'])} ({int(has_www.sum())} www / {int((~has_www).sum())} non-www), "
          f"phishing {len(sets['phish_browser'])}, prefix domains {len(prefix)}")

    results = {}
    for arm in ARMS:
        for fam in FAMILIES:
            path = os.path.join(args.models_dir, f'{arm}__{fam}.pkl')
            if not os.path.exists(path):
                continue
            _, proba = make_predictors(joblib.load(path), fam)
            p = {k: proba(np.array(v, dtype=object))[:, 1] for k, v in sets.items()}
            det = {k: v > DETECT for k, v in p.items()}
            r = {
                'legit_detected': int(det['legit'].sum()),
                'legit_blocked': int((p['legit'] > BLOCK).sum()),
                'legit_detected_www': int(det['legit'][has_www].sum()),
                'legit_detected_non_www': int(det['legit'][~has_www].sum()),
                'phish_detected_browser': int(det['phish_browser'].sum()),
                'phish_detected_listed': int(det['phish_listed'].sum()),
                'prefix_https': int(det['https'].sum()),
                'prefix_www': int(det['www'].sum()),
                'prefix_http': int(det['http'].sum()),
            }
            results[f'{arm}__{fam}'] = r
            print(f"  [{arm} / {FAMILIES[fam]}] FPR {pct(r['legit_detected'], len(legit))} "
                  f"blocked {pct(r['legit_blocked'], len(legit))} | recall browser "
                  f"{pct(r['phish_detected_browser'], len(phish))} listed {pct(r['phish_detected_listed'], len(phish))}",
                  flush=True)

    n_l, n_p, n_w, n_nw = len(legit), len(phish), int(has_www.sum()), int((~has_www).sum())
    meta = {'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'models_dir': args.models_dir,
            'legit': n_l, 'legit_www': n_w, 'legit_non_www': n_nw, 'phishing': n_p, 'prefix_domains': len(prefix),
            'detect_threshold': DETECT, 'block_threshold': BLOCK, 'exploratory': True}
    with open(os.path.join(args.models_dir, 'external_exploratory.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2)

    lines = [
        "# Character n-gram arms on the frozen external lists — EXPLORATORY",
        "",
        f"Generated: {meta['timestamp']}",
        "",
        "> Not part of the pre-registered external test. The frozen lists were collected and scored before these",
        "> models existed and their results were known when the character block was designed. Nothing here can",
        "> qualify a model for deployment; a candidate that looks good needs a fresh sample.",
        "",
        f"Lists: `external_500/frozen/` — {n_l} legit ({n_w} www / {n_nw} non-www host), {n_p} phishing, "
        f"{len(prefix)} prefix-sensitivity domains. Thresholds: detected > {DETECT:.0%}, blocked > {BLOCK:.0%}.",
        "The `all` and `no_scheme_www` rows are the 10 Oct ablation models, re-scored here as a check.",
        "",
        "## Legit false positives and phishing recall",
        "",
        "| Arm | Model | Detected FPR | Blocked FPR | FP on www hosts | FP on non-www hosts | Recall, browser | Recall, as listed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, r in results.items():
        arm, fam = key.split('__')
        lines.append(f"| `{arm}` | {FAMILIES[fam]} | {pct(r['legit_detected'], n_l)} | {pct(r['legit_blocked'], n_l)} | "
                     f"{pct(r['legit_detected_www'], n_w)} | {pct(r['legit_detected_non_www'], n_nw)} | "
                     f"{pct(r['phish_detected_browser'], n_p)} | {pct(r['phish_detected_listed'], n_p)} |")
    lines += ["", "## Prefix sensitivity (domains detected out of %d)" % len(prefix), "",
              "| Arm | Model | https://domain/ | https://www.domain/ | http://domain/ |", "|---|---|---:|---:|---:|"]
    for key, r in results.items():
        arm, fam = key.split('__')
        lines.append(f"| `{arm}` | {FAMILIES[fam]} | {r['prefix_https']} | {r['prefix_www']} | {r['prefix_http']} |")
    path = os.path.join(args.models_dir, 'external_exploratory.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print("Saved:", path)


if __name__ == '__main__':
    main()
