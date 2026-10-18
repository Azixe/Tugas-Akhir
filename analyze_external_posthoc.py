"""Post-hoc analyses of the external test.

NOT part of the pre-registered protocol (``external_500/PROTOCOL.md``): every number here was
computed after the external results were known. Nothing in this file can change the
deployment decision; it answers follow-up questions about that result.

1. Threshold-free metrics on the frozen lists (ROC-AUC, PR-AUC, best recall at a fixed FPR):
   is the fixed 60/80 threshold the problem, or the ranking itself?
2. Uncertainty of the headline external rates (Wilson intervals, domain-cluster bootstrap).
3. Residual shortcuts after scheme and ``www.`` are removed: TLD and URL shape by class, and
   the external false-positive rate by TLD x ``www.`` stratum.
4. The deployed models on the common test set through the extension's feature path
   (ONNX + exported vocabulary), at 0.5 and at the extension's detection threshold.

Usage (repo root):

    python analyze_external_posthoc.py
    python analyze_external_posthoc.py --skip-test-set     # parts 1-3 only, about a minute

Reads : external_500/frozen/*.csv (read-only), phishingdetectorExt/*.onnx + JSON,
        ablation_results/*.pkl, ablation_char_results/*.pkl (arms that are missing are skipped)
Writes: external_posthoc/posthoc_report.md, external_posthoc/posthoc_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import features
import sampling
from benchmark_sampling import make_predictors

EXT = 'phishingdetectorExt'
FROZEN = os.path.join('external_500', 'frozen')
OUTDIR = 'external_posthoc'
FAMILIES = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'cb': 'CatBoost'}
ONNX = {'rf': 'phishing_rf.onnx', 'xgb': 'phishing_xgb.onnx', 'cb': 'phishing_catboost.onnx'}
ARM_DIRS = {'all': 'ablation_results', 'no_scheme': 'ablation_results', 'no_scheme_www': 'ablation_results',
            'char': 'ablation_char_results', 'char_no_scheme_www': 'ablation_char_results'}
DETECT = 0.60          # extension threshold, strict >
CHUNK = 5000
Z95 = 1.959964


def rows(name):
    with open(os.path.join(FROZEN, name), newline='', encoding='utf-8') as f:
        return [{k: v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


# ---------------------------------------------------------------- deployed models (extension path)
class Deployed:
    """The three shipped ONNX models, fed the way the extension feeds them."""

    def __init__(self):
        load = lambda name: json.load(open(os.path.join(EXT, name), encoding='utf-8'))
        self.rf_tfidf, self.xgb_tfidf = load('tfidf_data.json'), load('tfidf_data_xgb.json')
        self.prep = load('xgb_preprocessing.json')
        self.sessions = {fam: ort.InferenceSession(os.path.join(EXT, name), providers=['CPUExecutionProvider'])
                         for fam, name in ONNX.items()}

    @staticmethod
    def _run(session, X):
        labels, probs = session.run(None, {session.get_inputs()[0].name: X.astype(np.float32)})[:2]
        if isinstance(probs, list):                     # ZipMap output: one {label: p} dict per row
            probs = np.array([row[1] for row in probs], dtype=np.float64)
        else:
            probs = np.asarray(probs, dtype=np.float64)[:, 1]
        return probs, np.asarray(labels).ravel().astype(int)

    def score(self, urls, progress=None):
        """{family: (probability of phishing, model label)} for a list of URLs."""
        out = {fam: ([], []) for fam in FAMILIES}
        for start in range(0, len(urls), CHUNK):
            batch = list(urls[start:start + CHUNK])
            f_rf = features.extract_features_onnx(batch, self.rf_tfidf)
            f_xgb = features.preprocess_xgb(features.extract_features_onnx(batch, self.xgb_tfidf), self.prep)
            for fam, X in (('rf', f_rf), ('xgb', f_xgb), ('cb', f_rf)):   # CatBoost shares RF's vocabulary
                p, lab = self._run(self.sessions[fam], X)
                out[fam][0].append(p)
                out[fam][1].append(lab)
            if progress:
                print(f"    {progress}: {min(start + CHUNK, len(urls))}/{len(urls)}", flush=True)
        return {fam: (np.concatenate(p), np.concatenate(lab)) for fam, (p, lab) in out.items()}


def arm_scores(arm, urls):
    """{family: probability} for one ablation arm, or {} when its pickles are not on disk."""
    out = {}
    for fam in FAMILIES:
        path = os.path.join(ARM_DIRS[arm], f'{arm}__{fam}.pkl')
        if os.path.exists(path):
            _, proba = make_predictors(joblib.load(path), fam)
            out[fam] = proba(np.array(urls, dtype=object))[:, 1].astype(np.float64)
    return out


# ---------------------------------------------------------------- statistics
def threshold_free(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return {'roc_auc': float(roc_auc_score(y, p)), 'pr_auc': float(average_precision_score(y, p)),
            'best_recall_at_fpr_2pct': float(tpr[fpr <= 0.02].max()),
            'best_recall_at_fpr_1pct': float(tpr[fpr <= 0.01].max())}


def wilson(k, n):
    if n == 0:
        return [0.0, 0.0]
    phat = k / n
    denom = 1 + Z95 ** 2 / n
    centre = (phat + Z95 ** 2 / (2 * n)) / denom
    half = Z95 * np.sqrt(phat * (1 - phat) / n + Z95 ** 2 / (4 * n ** 2)) / denom
    return [float(centre - half), float(centre + half)]


def cluster_bootstrap(flags, clusters, n_boot, seed=sampling.SEED):
    """Percentile interval of a rate when whole clusters (domains) are resampled."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    sums = np.array([flags[clusters == c].sum() for c in ids], dtype=float)
    sizes = np.array([(clusters == c).sum() for c in ids], dtype=float)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    rates = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]


def host_of(url):
    return features.url_host(url, strip_www=True).split(':')[0]


def shape_stats(urls):
    stripped = [features.normalize_url(u, True, True) for u in urls]
    hosts = [host_of(u) for u in urls]
    homepage = [s.rstrip('/') == s.split('/')[0].split('?')[0].split('#')[0] and '?' not in s and '#' not in s
                for s in stripped]
    return {'n': len(urls),
            'dot_com_share': float(np.mean([h.endswith('.com') for h in hosts])),
            'median_stripped_length': float(np.median([len(s) for s in stripped])),
            'homepage_share': float(np.mean(homepage))}


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def rate(k, n):
    return f"{pct(k / n) if n else '—'} ({k}/{n})"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Post-hoc analyses of the external test (not pre-registered).")
    ap.add_argument('--skip-test-set', action='store_true', help="Skip part 4 (the 90,036-URL pass)")
    ap.add_argument('--bootstrap', type=int, default=2000, help="Cluster-bootstrap replicates (default: %(default)s)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    legit, phish = rows('legit_urls.csv'), rows('phishing_urls.csv')
    L = [r['browser_url'] or r['url'] for r in legit]
    P = [r['browser_url'] or r['url'] for r in phish]
    urls = L + P
    y = np.array([0] * len(L) + [1] * len(P))
    domains = np.array([r['domain'] for r in legit])
    is_com = np.array([host_of(u).endswith('.com') for u in L])
    has_www = np.array([features.url_host(u).startswith('www.') for u in L])
    print(f"External lists: {len(L)} legit ({len(set(domains))} domains), {len(P)} phishing")

    deployed = Deployed()
    scores = {('model dasar', fam): p for fam, (p, _) in deployed.score(urls).items()}
    for arm in ARM_DIRS:
        for fam, p in arm_scores(arm, urls).items():
            scores[(arm, fam)] = p
    print(f"Scored {len(scores)} model/arm combinations")

    out = {'meta': {'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'post_hoc': True,
                    'legit': len(L), 'phishing': len(P), 'legit_domains': int(len(set(domains))),
                    'detect_threshold': DETECT, 'bootstrap': args.bootstrap, 'seed': sampling.SEED},
           'threshold_free': {}, 'uncertainty': {}, 'strata': {}, 'roc_curves': {}}

    # 1. threshold-free
    for (arm, fam), p in scores.items():
        out['threshold_free'][f'{arm}__{fam}'] = threshold_free(y, p)
    for fam in FAMILIES:
        fpr, tpr, _ = roc_curve(y, scores[('model dasar', fam)])
        out['roc_curves'][fam] = {'fpr': [round(float(v), 5) for v in fpr], 'tpr': [round(float(v), 5) for v in tpr]}

    # 2. uncertainty of the headline rates (model dasar)
    for fam in FAMILIES:
        p = scores[('model dasar', fam)]
        fp_flags, tp_flags = p[y == 0] > DETECT, p[y == 1] > DETECT
        out['uncertainty'][fam] = {
            'fp': int(fp_flags.sum()), 'fpr': float(fp_flags.mean()), 'fpr_wilson95': wilson(int(fp_flags.sum()), len(L)),
            'fpr_cluster_bootstrap95': cluster_bootstrap(fp_flags, domains, args.bootstrap),
            'tp': int(tp_flags.sum()), 'recall': float(tp_flags.mean()),
            'recall_wilson95': wilson(int(tp_flags.sum()), len(P))}

    # 3. residual shortcuts
    df = sampling.load_clean_dataset(sampling.DEFAULT_DATASET)
    train, test = sampling.common_split(df)
    out['shape'] = {'train_legit': shape_stats(train.loc[train['label'] == 0, 'url'].tolist()),
                    'train_phishing': shape_stats(train.loc[train['label'] == 1, 'url'].tolist()),
                    'external_legit': shape_stats(L), 'external_phishing': shape_stats(P)}
    strata = {'com_www': is_com & has_www, 'com_no_www': is_com & ~has_www,
              'other_www': ~is_com & has_www, 'other_no_www': ~is_com & ~has_www,
              'com': is_com, 'other': ~is_com}
    for (arm, fam), p in scores.items():
        if arm in ('model dasar', 'no_scheme_www'):
            flags = p[y == 0] > DETECT
            out['strata'][f'{arm}__{fam}'] = {name: {'fp': int(flags[mask].sum()), 'n': int(mask.sum())}
                                              for name, mask in strata.items()}

    # 4. deployed path on the common test set
    if not args.skip_test_set:
        print(f"Common test set through the extension path: {len(test)} URLs")
        y_test = test['label'].values
        out['test_set_deployed_path'] = {}
        for fam, (p, lab) in deployed.score(test['url'].tolist(), progress='test set').items():
            entry = {}
            for name, pred in (('at_0.5', lab == 1), ('above_0.6', p > DETECT)):
                tp, fp = int((pred & (y_test == 1)).sum()), int((pred & (y_test == 0)).sum())
                fn, tn = int((~pred & (y_test == 1)).sum()), int((~pred & (y_test == 0)).sum())
                entry[name] = {'accuracy': (tp + tn) / len(y_test), 'recall': tp / (tp + fn), 'fpr': fp / (fp + tn),
                               'precision': tp / (tp + fp) if tp + fp else 0.0, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
            out['test_set_deployed_path'][fam] = entry

    with open(os.path.join(OUTDIR, 'posthoc_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    path = write_report(out)
    print_summary(out)
    print(f"Saved: {os.path.join(OUTDIR, 'posthoc_metrics.json')}")
    print(f"Saved: {path}")


def label_of(key):
    arm, fam = key.split('__')
    return (arm if arm == 'model dasar' else f'`{arm}`'), FAMILIES[fam]


def write_report(out):
    m = out['meta']
    lines = [
        "# External test — post-hoc analyses",
        "",
        f"Generated: {m['timestamp']}",
        "",
        "> **Post hoc.** None of this was in `external_500/PROTOCOL.md`; it was computed after the external",
        "> results were known and cannot change the deployment decision. Lists: `external_500/frozen/`",
        f"> ({m['legit']} legit URLs from {m['legit_domains']} domains, {m['phishing']} phishing URLs, browser variant).",
        "> \"model dasar\" = the three deployed ONNX files scored through the extension's feature path;",
        "> back-ticked rows are ablation arms (models fitted on the 324,126-URL fit set).",
        "",
        "## 1. Threshold-free metrics on the external lists",
        "",
        "| Model | Family | ROC-AUC | PR-AUC | Best recall at FPR ≤ 2% | Best recall at FPR ≤ 1% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, r in out['threshold_free'].items():
        arm, fam = label_of(key)
        lines.append(f"| {arm} | {fam} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
                     f"{pct(r['best_recall_at_fpr_2pct'])} | {pct(r['best_recall_at_fpr_1pct'])} |")
    lines += ["",
              "\"Best recall\" uses the threshold that an oracle would pick on this very set, so it is an upper bound.",
              f"A classifier that ranks at random has ROC-AUC 0.5 and PR-AUC {m['phishing'] / (m['legit'] + m['phishing']):.3f} "
              "(the phishing share of the lists).",
              "",
              f"## 2. Uncertainty of the headline rates (model dasar, detected = score > {int(DETECT * 100)}%)",
              "",
              "| Family | Detected FPR | Wilson 95% | Domain-cluster bootstrap 95% | Detected recall | Wilson 95% |",
              "|---|---:|---:|---:|---:|---:|"]
    for fam, u in out['uncertainty'].items():
        lines.append(f"| {FAMILIES[fam]} | {rate(u['fp'], m['legit'])} | {pct(u['fpr_wilson95'][0])} – {pct(u['fpr_wilson95'][1])} | "
                     f"{pct(u['fpr_cluster_bootstrap95'][0])} – {pct(u['fpr_cluster_bootstrap95'][1])} | "
                     f"{rate(u['tp'], m['phishing'])} | {pct(u['recall_wilson95'][0])} – {pct(u['recall_wilson95'][1])} |")
    lines += ["",
              f"The bootstrap resamples whole domains ({m['bootstrap']} replicates, seed {m['seed']}) because each "
              "domain contributes two URLs.",
              "",
              "## 3. Residual shortcuts: TLD and URL shape",
              "",
              "Scheme and a leading `www.` are stripped before measuring, so this is what remains after the two",
              "known shortcuts are removed.",
              "",
              "| URL set | n | Host ends in `.com` | Median URL length | Homepage only (no path) |",
              "|---|---:|---:|---:|---:|"]
    names = {'train_legit': 'Training, legitimate', 'train_phishing': 'Training, phishing',
             'external_legit': 'External, legitimate', 'external_phishing': 'External, phishing'}
    for key, s in out['shape'].items():
        lines.append(f"| {names[key]} | {s['n']} | {pct(s['dot_com_share'])} | {s['median_stripped_length']:g} | "
                     f"{pct(s['homepage_share'])} |")
    lines += ["",
              "External legitimate URLs flagged (detected), by TLD and by whether the host starts with `www.`:",
              "",
              "| Model | Family | `.com` | other TLD | `.com`, www | `.com`, no www | other, www | other, no www |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, s in out['strata'].items():
        arm, fam = label_of(key)
        cells = [rate(s[k]['fp'], s[k]['n']) for k in ('com', 'other', 'com_www', 'com_no_www', 'other_www', 'other_no_www')]
        lines.append(f"| {arm} | {fam} | " + " | ".join(cells) + " |")
    lines += [""]
    if 'test_set_deployed_path' in out:
        lines += ["## 4. Deployed models on the common test set, extension feature path",
                  "",
                  "ONNX files and exported vocabularies exactly as shipped (empty-string token filtered, as in",
                  "`utils.js`). \"At 0.5\" is the model's own label; \"> 60%\" is the extension's detection threshold.",
                  "",
                  "| Family | Operating point | Accuracy | Precision | Recall | FPR | FP | FN |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for fam, entry in out['test_set_deployed_path'].items():
            for name, label in (('at_0.5', 'at 0.5'), ('above_0.6', '> 60%')):
                r = entry[name]
                lines.append(f"| {FAMILIES[fam]} | {label} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                             f"{100 * r['fpr']:.4f}% | {r['fp']} | {r['fn']} |")
        lines += [""]
    path = os.path.join(OUTDIR, 'posthoc_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return path


def print_summary(out):
    """Console copy of the headline rows (same rounding as the report)."""
    print()
    print(f"{'Model dasar':<15}{'ROC-AUC':>9}{'PR-AUC':>8}{'Best recall @FPR<=2%':>22}{'Ext FPR':>9}{'Ext recall':>12}")
    for fam in FAMILIES:
        t, u = out['threshold_free'][f'model dasar__{fam}'], out['uncertainty'][fam]
        print(f"{FAMILIES[fam]:<15}{t['roc_auc']:>9.3f}{t['pr_auc']:>8.3f}{pct(t['best_recall_at_fpr_2pct']):>22}"
              f"{pct(u['fpr']):>9}{pct(u['recall']):>12}")
    if 'test_set_deployed_path' in out:
        print()
        print(f"{'Test set, extension path':<26}{'Recall @0.5':>12}{'FPR @0.5':>10}{'Recall >60%':>13}{'FPR >60%':>10}")
        for fam, e in out['test_set_deployed_path'].items():
            print(f"{FAMILIES[fam]:<26}{e['at_0.5']['recall']:>12.4f}{100 * e['at_0.5']['fpr']:>9.3f}%"
                  f"{e['above_0.6']['recall']:>13.4f}{100 * e['above_0.6']['fpr']:>9.3f}%")
    print()


if __name__ == '__main__':
    main()
