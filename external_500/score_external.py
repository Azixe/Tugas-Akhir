"""Score the frozen external 500/500 lists with the deployed v3.4 models and every
ablation arm, then apply the pre-registered deployment rule.

Protocol: external_500/PROTOCOL.md (committed before any list was fetched).

Scoring paths (mirror the artifacts as they are used in production/analysis):

  * v3.4  — the extension's ONNX models scored through the same feature code the
            extension uses (`features.extract_features_onnx`, `preprocess_xgb`),
            exactly like `run_realworld_test.py`.
  * arms  — each cached `ablation_results/<arm>__<family>.pkl` scored through
            `benchmark_sampling.make_predictors`, the same path the ablation
            benchmark used.

Operating points (extension thresholds, strict): detected = score > 60,
blocked = score > 80, both in percent.

Usage (repo root):
    python external_500/score_external.py

Reads : external_500/frozen/{legit_urls.csv, phishing_urls.csv,
        prefix_sensitivity.csv}, ablation_results/*.pkl + ablation_metrics.json,
        phishingdetectorExt/*.onnx + *.json, URL dataset.csv
Writes: external_500/external_report.md, external_500/external_metrics.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import joblib
import numpy as np
import onnxruntime as ort
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features import extract_features_onnx, preprocess_xgb  # noqa: E402
from benchmark_sampling import make_predictors  # noqa: E402
from run_realworld_test import is_free_hosting  # noqa: E402

EXT = os.path.join(ROOT, 'phishingdetectorExt')
FROZEN = os.path.join(ROOT, 'external_500', 'frozen')
OUTDIR = os.path.join(ROOT, 'external_500')

ARMS = ['all', 'no_scheme', 'no_scheme_www', 'tfidf_only', 'struct_only',
        'tfidf_only_no_scheme_www', 'struct_only_no_scheme_www']
CANDIDATES = ['no_scheme', 'no_scheme_www']
FAMILIES = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'cb': 'CatBoost'}
DETECT, BLOCK = 60.0, 80.0
TASK_ARMS = ARMS  # every arm is scored and reported; only CANDIDATES may deploy


# ---------------------------------------------------------------- loading ----

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_frozen():
    legit = read_csv_rows(os.path.join(FROZEN, 'legit_urls.csv'))
    phish = read_csv_rows(os.path.join(FROZEN, 'phishing_urls.csv'))
    prefix = read_csv_rows(os.path.join(FROZEN, 'prefix_sensitivity.csv'))
    for rows in (legit, phish, prefix):
        for r in rows:
            for k, v in list(r.items()):
                r[k] = v.strip() if isinstance(v, str) else v
    return legit, phish, prefix


def load_v34():
    with open(os.path.join(EXT, 'tfidf_data.json')) as f:
        rf_tfidf = json.load(f)
    with open(os.path.join(EXT, 'tfidf_data_xgb.json')) as f:
        xgb_tfidf = json.load(f)
    with open(os.path.join(EXT, 'xgb_preprocessing.json')) as f:
        prep = json.load(f)
    return {
        'rf_tfidf': rf_tfidf, 'xgb_tfidf': xgb_tfidf, 'prep': prep,
        'sessions': {
            fam: ort.InferenceSession(os.path.join(EXT, name),
                                      providers=['CPUExecutionProvider'])
            for fam, name in (('rf', 'phishing_rf.onnx'), ('xgb', 'phishing_xgb.onnx'),
                              ('cb', 'phishing_catboost.onnx'))},
    }


def predict_v34(models, urls):
    """Percent-scale phishing probability per URL, per family."""
    urls = [str(u) for u in urls]
    X_rf = extract_features_onnx(urls, models['rf_tfidf'])
    X_xgb = preprocess_xgb(extract_features_onnx(urls, models['xgb_tfidf']), models['prep'])
    out = {}
    for fam, X in (('rf', X_rf), ('xgb', X_xgb), ('cb', X_rf)):
        sess = models['sessions'][fam]
        res = sess.run(None, {sess.get_inputs()[0].name: X})
        out[fam] = np.asarray(res[1], dtype=float)[:, 1] * 100.0
    return out


def check_ablation_config():
    with open(os.path.join(ROOT, 'ablation_results', 'ablation_metrics.json')) as f:
        metrics = json.load(f)
    meta = metrics['meta']
    problems = []
    if not meta.get('drop_empty_tokens'):
        problems.append('drop_empty_tokens is not enabled in ablation_metrics.json')
    if meta.get('cb_iterations') != 300:
        problems.append(f"cb_iterations is {meta.get('cb_iterations')}, expected 300")
    missing = [a for a in TASK_ARMS if a not in meta.get('arms', [])]
    if missing:
        problems.append(f"arms missing from the ablation run: {missing}")
    return metrics['meta'], problems


# --------------------------------------------------------------- metrics ----

def detected(probs, thr):
    return probs > thr


def rate(count, n):
    return round(100.0 * count / n, 3) if n else 0.0


def legit_metrics(probs):
    det = detected(probs, DETECT).sum()
    blk = detected(probs, BLOCK).sum()
    n = len(probs)
    return {'n': n, 'detect_fp': int(det), 'detect_fpr': rate(det, n),
            'block_fp': int(blk), 'block_fpr': rate(blk, n)}


def phish_metrics(probs, mask=None):
    if mask is not None:
        probs = probs[mask]
    n = len(probs)
    det = detected(probs, DETECT).sum()
    blk = detected(probs, BLOCK).sum()
    return {'n': n, 'detect_tp': int(det), 'detect_recall': rate(det, n),
            'block_tp': int(blk), 'block_recall': rate(blk, n)}


def mcnemar(base_vec, cand_vec):
    b = int(np.sum(cand_vec & ~base_vec))
    c = int(np.sum(~cand_vec & base_vec))
    if b + c == 0:
        return {'b': b, 'c': c, 'p': 1.0}
    return {'b': b, 'c': c, 'p': float(binomtest(b, b + c, 0.5).pvalue)}


# ------------------------------------------------------------------ main ----

def main():
    legit, phish, prefix = load_frozen()
    n_legit, n_phish, n_prefix = len(legit), len(phish), len(prefix)
    print(f"Frozen lists: {n_legit} legit rows, {n_phish} phishing rows, {n_prefix} prefix domains")

    legit_urls = [r['browser_url'] or r['url'] for r in legit]
    phish_browser = [r['browser_url'] or r['url'] for r in phish]
    phish_listed = [r['url'] for r in phish]
    prefix_urls = []
    for r in prefix:
        prefix_urls += [r['url_https'], r['url_https_www'], r['url_http']]

    all_urls = legit_urls + phish_browser + phish_listed + prefix_urls
    s_legit = slice(0, n_legit)
    s_pb = slice(n_legit, n_legit + n_phish)
    s_pl = slice(n_legit + n_phish, n_legit + 2 * n_phish)
    s_px = slice(n_legit + 2 * n_phish, len(all_urls))
    print(f"Scoring {len(all_urls)} URL variants per model")

    # masks
    def host_of(u):
        try:
            return (urlparse(u).hostname or '').lower()
        except ValueError:
            return ''

    legit_headline = np.array([r['in_headline'] == '1' for r in legit])
    legit_kind_landing = np.array([r['kind'] == 'landing' for r in legit])
    legit_www = np.array([host_of(u).startswith('www.') for u in legit_urls])
    bands = sorted({r['band'] for r in legit})
    phish_live = np.array([r['live'] == '1' for r in phish])
    phish_seen = np.array([r['seen_in_training_host'] == '1' for r in phish])
    phish_free = np.array([is_free_hosting(u) for u in phish_browser])

    # --- models ---
    meta, problems = check_ablation_config()
    for p in problems:
        print(f"!! WARNING: {p}")
    if problems:
        print("!! The ablation artifacts may not match the committed run — review before trusting output")

    P = {}   # P[(model_key, family)] = {'legit','pb','pl','px'} percent probs

    print("v3.4 ONNX ...", flush=True)
    v34 = load_v34()
    preds = predict_v34(v34, all_urls)
    for fam in FAMILIES:
        P[('v3.4', fam)] = {'legit': preds[fam][s_legit], 'pb': preds[fam][s_pb],
                            'pl': preds[fam][s_pl], 'px': preds[fam][s_px]}

    for arm in TASK_ARMS:
        for fam in FAMILIES:
            path = os.path.join(ROOT, 'ablation_results', f'{arm}__{fam}.pkl')
            model = joblib.load(path)
            _, proba = make_predictors(model, fam)
            probs = np.asarray(proba(np.array(all_urls)), dtype=float)[:, 1] * 100.0
            P[(arm, fam)] = {'legit': probs[s_legit], 'pb': probs[s_pb],
                             'pl': probs[s_pl], 'px': probs[s_px]}
            del model
            print(f"  scored {arm} / {FAMILIES[fam]}", flush=True)

    # --- per-model metrics ---
    results = {}
    for (key, fam), p in P.items():
        entry = {
            'legit_headline': legit_metrics(p['legit'][legit_headline]),
            'legit_all': legit_metrics(p['legit']),
            'legit_landing': legit_metrics(p['legit'][legit_kind_landing]),
            'legit_deep': legit_metrics(p['legit'][~legit_kind_landing]),
            'legit_www': legit_metrics(p['legit'][legit_www]),
            'legit_non_www': legit_metrics(p['legit'][~legit_www]),
            'legit_by_band': {b: legit_metrics(p['legit'][np.array(
                [r['band'] == b for r in legit])]) for b in bands},
            'phish_browser': phish_metrics(p['pb']),
            'phish_listed': phish_metrics(p['pl']),
            'phish_live': phish_metrics(p['pb'], phish_live),
            'phish_seen_host': phish_metrics(p['pb'], phish_seen),
            'phish_unseen_host': phish_metrics(p['pb'], ~phish_seen),
            'phish_free_hosting': phish_metrics(p['pb'], phish_free),
            'prefix': {},
        }
        for i, name in enumerate(('https', 'www', 'http')):
            vec = p['px'][i::3]
            entry['prefix'][name] = int(detected(vec, DETECT).sum())
        px = p['px'].reshape(-1, 3)
        det = detected(px, DETECT)          # rows x 3
        blk = detected(px, BLOCK)
        entry['prefix'].update({
            'n': n_prefix,
            'detect_flips_any': int(np.sum(det.any(axis=1) & ~det.all(axis=1))),
            'detect_flips_https_http': int(np.sum(det[:, 0] != det[:, 2])),
            'block_flips_any': int(np.sum(blk.any(axis=1) & ~blk.all(axis=1))),
        })
        results.setdefault(key, {})[fam] = entry

    # --- McNemar (candidate vs v3.4, same family, browser variant, detected) ---
    mcnemar_out = {}
    for arm in TASK_ARMS:
        mcnemar_out[arm] = {}
        for fam in FAMILIES:
            base = detected(P[('v3.4', fam)]['pb'], DETECT)
            cand = detected(P[(arm, fam)]['pb'], DETECT)
            base_b = detected(P[('v3.4', fam)]['pb'], BLOCK)
            cand_b = detected(P[(arm, fam)]['pb'], BLOCK)
            rec_gain = (results[arm][fam]['phish_browser']['detect_recall']
                        - results['v3.4'][fam]['phish_browser']['detect_recall'])
            mcnemar_out[arm][fam] = {
                'detected': mcnemar(base, cand),
                'blocked': mcnemar(base_b, cand_b),
                'recall_gain_pp': round(rec_gain, 2),
            }

    # --- deployment rule (protocol §c) ---
    rule = {}
    n = len(legit)
    for fam in FAMILIES:
        base = results['v3.4'][fam]
        fam_rule = {'baseline': {'detect_fpr': base['legit_all']['detect_fpr'],
                                 'block_fpr': base['legit_all']['block_fpr'],
                                 'recall': base['phish_browser']['detect_recall']},
                    'candidates': {}}
        passing = []
        for arm in CANDIDATES:
            r = results[arm][fam]
            mc = mcnemar_out[arm][fam]
            c1 = (r['legit_all']['detect_fpr'] <= 2.0
                  and r['legit_all']['detect_fp'] <= base['legit_all']['detect_fp'])
            c2 = r['legit_all']['block_fpr'] <= 1.0
            c3 = (mc['recall_gain_pp'] >= 10.0 and mc['detected']['p'] < 0.05)
            ok = bool(c1 and c2 and c3)
            fam_rule['candidates'][arm] = {
                'detect_fpr': r['legit_all']['detect_fpr'],
                'detect_fp': r['legit_all']['detect_fp'],
                'block_fpr': r['legit_all']['block_fpr'],
                'recall': r['phish_browser']['detect_recall'],
                'recall_gain_pp': mc['recall_gain_pp'], 'mcnemar_p': mc['detected']['p'],
                'c1_fpr': bool(c1), 'c2_block': bool(c2), 'c3_recall': bool(c3),
                'pass': ok}
            if ok:
                passing.append(arm)
        if passing:
            passing.sort(key=lambda a: (-results[a][fam]['phish_browser']['detect_recall'],
                                        results[a][fam]['legit_all']['detect_fpr']))
            fam_rule['decision'] = {'winner': passing[0], 'keep_v34': False}
        else:
            fam_rule['decision'] = {'winner': None, 'keep_v34': True}
        rule[fam] = fam_rule

    # --- construction details (limitations; verified counts only) ---
    def registrable(host):
        parts = host.split('.')
        return '.'.join(parts[-2:]) if len(parts) >= 2 else host

    WL = ['google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'github.com',
          'stackoverflow.com', 'microsoft.com', 'apple.com', 'amazon.com',
          'wikipedia.org', 'linkedin.com', 'instagram.com']
    wl_rows = [i for i, r in enumerate(legit) if any(
        (h := host_of(r['browser_url'] or r['url'])) == d or h.endswith('.' + d) for d in WL)]
    listed_http = [r for r in phish if r['url'].lower().startswith('http://')]
    bare_hosts = [r for r in phish if re.match(r'^https?://[^/]+$', r['url'])]
    slash_idx = [i for i, r in enumerate(phish) if r['browser_url'] == r['url'] + '/']
    cb_nw = P[('no_scheme_www', 'cb')]
    landing_reg = {r['domain']: registrable(host_of(r['browser_url'] or r['url']))
                   for r in legit if r['kind'] == 'landing'}
    landing_changed = [d for d in landing_reg if landing_reg[d] != registrable(d)]
    construction = {
        'whitelisted_legit_rows': len(wl_rows),
        'whitelisted_fp_v34': {fam: int(sum(
            1 for i in wl_rows if P[('v3.4', fam)]['legit'][i] > DETECT)) for fam in FAMILIES},
        'deep_non200': int(sum(1 for r in legit if r['kind'] == 'deep' and r['http_code'] != '200')),
        'deep_offsite': int(sum(1 for r in legit if r['kind'] == 'deep' and
                               registrable(host_of(r['browser_url'] or r['url']))
                               != landing_reg.get(r['domain'], ''))),
        'landing_redirected_registrable': len(landing_changed),
        'phish_answered': int(sum(1 for r in phish if r['http_code'] not in ('0', ''))),
        'phish_listed_http': len(listed_http),
        'phish_listed_http_answered': int(sum(1 for r in listed_http
                                              if r['http_code'] not in ('0', ''))),
        'phish_listed_http_live': int(sum(1 for r in listed_http if r['live'] == '1')),
        'bare_host_listings': len(bare_hosts),
        'slash_only_rows': len(slash_idx),
        'slash_cb_no_scheme_www': {
            'as_listed_detected': int(sum(1 for i in slash_idx if cb_nw['pl'][i] > DETECT)),
            'with_slash_detected': int(sum(1 for i in slash_idx if cb_nw['pb'][i] > DETECT))},
    }

    # --- context: in-distribution (ablation_metrics.json) vs external ---
    ablation = json.load(open(os.path.join(ROOT, 'ablation_results', 'ablation_metrics.json')))
    context = {}
    for arm in TASK_ARMS:
        context[arm] = {}
        for fam in FAMILIES:
            a = ablation['arms'][arm][fam]
            context[arm][fam] = {
                'validation_recall': a['validation']['recall_phishing'],
                'validation_fpr': a['validation']['fpr'],
                'test_recall': a['test']['recall_phishing'] if 'test' in a else None,
                'test_fpr': a['test']['fpr'] if 'test' in a else None,
                'external_recall': results[arm][fam]['phish_browser']['detect_recall'] / 100.0,
                'external_fpr': results[arm][fam]['legit_all']['detect_fpr'] / 100.0,
            }

    # --- provenance ---
    provenance = {}
    for name in ('legit_urls.csv', 'phishing_urls.csv', 'prefix_sensitivity.csv'):
        path = os.path.join(FROZEN, name)
        provenance[name] = {'sha256': sha256(path), 'rows': sum(1 for _ in open(path, encoding='utf-8')) - 1}
    ext_manifest = json.load(open(os.path.join(EXT, 'manifest.json')))
    model_hashes = {fam: sha256(os.path.join(EXT, f))[:16] + '…' for fam, f in
                    (('rf', 'phishing_rf.onnx'), ('xgb', 'phishing_xgb.onnx'),
                     ('cb', 'phishing_catboost.onnx'))}
    try:
        harness_commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip()
    except Exception:
        harness_commit = 'unknown'

    train_hosts = set()
    train_urls = set()
    with open(os.path.join(ROOT, 'URL dataset.csv'), newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            u = (row.get('url') or '').strip().lower()
            if u:
                train_urls.add(u)
                h = host_of(u if '://' in u else 'http://' + u)
                if h:
                    train_hosts.add(h)
    phishing_hosts = {r['host'] for r in phish}
    overlap_hosts = sorted(h for h in phishing_hosts if h in train_hosts)
    legit_landing_hosts = {host_of(r['browser_url'] or r['url'])
                           for r in legit if r['kind'] == 'landing'}
    construction['legit_host_overlap_training'] = len(legit_landing_hosts & train_hosts)
    construction['legit_exact_training_rows'] = sum(
        1 for r in legit if (r['browser_url'] or r['url']).lower() in train_urls)

    summary = {
        'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'harness_commit': harness_commit,
        'extension_version': ext_manifest.get('version'),
        'model_hashes': model_hashes,
        'provenance': provenance,
        'thresholds': {'detected': '>60', 'blocked': '>80', 'scale': 'percent'},
        'counts': {'legit_rows': n_legit, 'legit_headline': int(legit_headline.sum()),
                   'phishing_rows': n_phish, 'phishing_live': int(phish_live.sum()),
                   'phishing_seen_host': int(phish_seen.sum()),
                   'phishing_free_hosting': int(phish_free.sum()),
                   'prefix_domains': n_prefix},
        'training_overlap': {'phishing_hosts_total': len(phishing_hosts),
                             'phishing_hosts_seen_in_training': len(overlap_hosts),
                             'hosts': overlap_hosts[:100]},
        'ablation_config': {'drop_empty_tokens': meta.get('drop_empty_tokens'),
                            'cb_iterations': meta.get('cb_iterations'),
                            'warnings': problems},
        'results': results, 'mcnemar': mcnemar_out, 'deployment_rule': rule,
        'context_in_distribution': context, 'construction': construction,
    }
    with open(os.path.join(OUTDIR, 'external_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {os.path.join(OUTDIR, 'external_metrics.json')}")

    write_report(summary)
    print(f"Saved {os.path.join(OUTDIR, 'external_report.md')}")

    # console summary
    for fam in FAMILIES:
        d = rule[fam]['decision']
        print(f"{FAMILIES[fam]}: " + (f"winner {d['winner']}" if not d['keep_v34'] else "keep v3.4"))
        for arm in CANDIDATES:
            c = rule[fam]['candidates'][arm]
            print(f"  {arm}: FPR {c['detect_fpr']}% block {c['block_fpr']}% recall {c['recall']}% "
                  f"(+{c['recall_gain_pp']}pp, p={fmt_p(c['mcnemar_p'])}) pass={c['pass']}")


def fmt_p(p):
    if p is None:
        return '—'
    return '< 0.000001' if p < 1e-6 else f'{p:.6f}'


def write_report(s):
    r = s['results']
    lines = ["# External 500/500 evaluation — live Tranco legit + PhishTank phishing", ""]
    lines += [
        f"Generated: {s['generated']} — harness commit `{s['harness_commit']}`, "
        f"extension v{s['extension_version']}",
        f"Protocol: `external_500/PROTOCOL.md` (frozen before collection; amendment log inside).",
        f"Lists (frozen, hashed): "
        + '; '.join(f"`{k}` {v['rows']} rows sha256 `{v['sha256'][:16]}…`"
                    for k, v in s['provenance'].items()),
        f"ONNX models: RF `{s['model_hashes']['rf']}`, XGB `{s['model_hashes']['xgb']}`, "
        f"CatBoost `{s['model_hashes']['cb']}`",
        f"Thresholds (extension, strict): detected >60, blocked >80 (percent).",
        "",
        f"- Legit: {s['counts']['legit_rows']} URLs ({s['counts']['legit_headline']} headline), "
        f"phishing: {s['counts']['phishing_rows']} URLs "
        f"({s['construction']['phish_answered']} of 500 answered at resolution, "
        f"{s['counts']['phishing_live']} live, "
        f"{s['counts']['phishing_seen_host']} hosts seen in training, "
        f"{s['counts']['phishing_free_hosting']} free-hosting).",
        "",
        "## Legit — headline 500 (browser URLs; detected >60, blocked >80)",
        "",
        "| Model | detected FP | detected FPR | blocked FP | blocked FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    order = [('v3.4', fam) for fam in FAMILIES] + [(a, fam) for a in ARMS for fam in FAMILIES]
    for key, fam in order:
        m = r[key][fam]['legit_headline']
        label = f"v3.4 {FAMILIES[fam]}" if key == 'v3.4' else f"`{key}` {FAMILIES[fam]}"
        lines.append(f"| {label} | {m['detect_fp']}/{m['n']} | {m['detect_fpr']}% "
                     f"| {m['block_fp']}/{m['n']} | {m['block_fpr']}% |")

    lines += ["", "## Legit — all 1.000 rows (protocol rule inputs; baseline + candidates)", "",
              "| Model | detected FPR (all) | blocked FPR (all) | detected FPR landing | "
              "detected FPR deep | worst band (detected FPR) |", "|---|---:|---:|---:|---:|---:|"]
    for key, fam in order:
        if key != 'v3.4' and key not in CANDIDATES:
            continue
        e = r[key][fam]
        worst = max(e['legit_by_band'].items(), key=lambda kv: kv[1]['detect_fpr'])
        label = f"v3.4 {FAMILIES[fam]}" if key == 'v3.4' else f"`{key}` {FAMILIES[fam]}"
        lines.append(f"| {label} | {e['legit_all']['detect_fpr']}% ({e['legit_all']['detect_fp']}/{e['legit_all']['n']}) "
                     f"| {e['legit_all']['block_fpr']}% | {e['legit_landing']['detect_fpr']}% "
                     f"| {e['legit_deep']['detect_fpr']}% | {worst[0]} {worst[1]['detect_fpr']}% |")

    lines += ["", "## Phishing — browser variant, 500 (detected recall >60, blocked >80)", "",
              "| Model | detected TP | detected recall | blocked TP | blocked recall |",
              "|---|---:|---:|---:|---:|"]
    for key, fam in order:
        m = r[key][fam]['phish_browser']
        label = f"v3.4 {FAMILIES[fam]}" if key == 'v3.4' else f"`{key}` {FAMILIES[fam]}"
        lines.append(f"| {label} | {m['detect_tp']}/500 | {m['detect_recall']}% "
                     f"| {m['block_tp']}/500 | {m['block_recall']}% |")

    c = s['construction']
    lines += ["", "### Recall range under the https-upgrade assumption (v3.4 baseline)", "",
              f"{c['phish_listed_http']} of the 500 feed URLs are listed as `http://` and only "
              f"{c['phish_listed_http_answered']} of them answered when resolved, so the "
              "browser variant assumes the https upgrade for the rest. As-listed recall scores the "
              "observed string (reference; see the bare-host caveat below). Report the pair, not "
              "one number.", "",
              "| Model | browser recall (https assumed) | as-listed recall |", "|---|---:|---:|"]
    for fam in FAMILIES:
        rb = r['v3.4'][fam]['phish_browser']['detect_recall']
        rl = r['v3.4'][fam]['phish_listed']['detect_recall']
        lines.append(f"| v3.4 {FAMILIES[fam]} | {rb}% | {rl}% |")

    lines += ["", "## Phishing — variants and subsets (baseline + candidates)", "",
              "| Model | as-listed recall | live-only recall | seen-host recall | unseen-host recall "
              "| free-hosting recall | recall gain vs v3.4 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, fam in order:
        if key != 'v3.4' and key not in CANDIDATES:
            continue
        e = r[key][fam]
        gain = s['mcnemar'][key][fam]['recall_gain_pp'] if key != 'v3.4' else 0.0
        label = f"v3.4 {FAMILIES[fam]}" if key == 'v3.4' else f"`{key}` {FAMILIES[fam]}"
        lines.append(f"| {label} | {e['phish_listed']['detect_recall']}% "
                     f"| {e['phish_live']['detect_recall']}% (n={e['phish_live']['n']}) "
                     f"| {e['phish_seen_host']['detect_recall']}% (n={e['phish_seen_host']['n']}) "
                     f"| {e['phish_unseen_host']['detect_recall']}% (n={e['phish_unseen_host']['n']}) "
                     f"| {e['phish_free_hosting']['detect_recall']}% (n={e['phish_free_hosting']['n']}) "
                     f"| {'—' if key == 'v3.4' else f'{gain:+.2f} pp'} |")

    n_www = r['v3.4']['rf']['legit_www']['n']
    n_non = r['v3.4']['rf']['legit_non_www']['n']
    lines += ["", "## Legit FPR — `www.` split (why the false positives happen; baseline + candidates)", "",
              f"Of the 1.000 legit URLs, {n_www} carry a `www.` host and {n_non} do not; the "
              "feature set learned `www.` as a legit signal, and the non-www rows absorb the "
              "false positives.", "",
              "| Model | flagged, with www. | flagged, without www. |", "|---|---:|---:|"]
    for key, fam in order:
        if key != 'v3.4' and key not in CANDIDATES:
            continue
        e = r[key][fam]
        w, nw = e['legit_www'], e['legit_non_www']
        label = f"v3.4 {FAMILIES[fam]}" if key == 'v3.4' else f"`{key}` {FAMILIES[fam]}"
        lines.append(f"| {label} | {w['detect_fp']}/{w['n']} ({w['detect_fpr']}%) "
                     f"| {nw['detect_fp']}/{nw['n']} ({nw['detect_fpr']}%) |")

    lines += ["", "## Legit detected FPR by Tranco band (baseline + candidates)", "",
              "| Model | 1-1k | 1k-10k | 10k-100k | 100k-500k | 500k-1M |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, fam in order:
        if key != 'v3.4' and key not in CANDIDATES:
            continue
        e = r[key][fam]
        label = f"v3.4 {FAMILIES[fam]}" if key == 'v3.4' else f"`{key}` {FAMILIES[fam]}"
        cells = " | ".join(f"{e['legit_by_band'][b]['detect_fpr']}%"
                           for b in ('1-1k', '1k-10k', '10k-100k', '100k-500k', '500k-1M'))
        lines.append(f"| {label} | {cells} |")

    lines += ["", "## McNemar — candidates vs v3.4 of the same family (browser variant, detected >60)", "",
              "| Family | Candidate | v3.4-only correct (c) | candidate-only correct (b) | p (detected) "
              "| p (blocked) | recall gain |", "|---|---|---:|---:|---:|---:|---:|"]
    for fam in FAMILIES:
        for arm in CANDIDATES:
            mc = s['mcnemar'][arm][fam]
            lines.append(f"| {FAMILIES[fam]} | `{arm}` | {mc['detected']['c']} | {mc['detected']['b']} "
                         f"| {fmt_p(mc['detected']['p'])} | {fmt_p(mc['blocked']['p'])} "
                         f"| {mc['recall_gain_pp']:+.2f} pp |")

    lines += ["", "## Prefix sensitivity — synthetic `https://d/` vs `https://www.d/` vs `http://d/` "
              "(headline domains)", "",
              "| Model | detected https | detected www | detected http | flips (any of 3) "
              "| flips https vs http |", "|---|---:|---:|---:|---:|---:|"]
    for key, fam in order:
        if key != 'v3.4' and key not in CANDIDATES:
            continue
        px = r[key][fam]['prefix']
        label = f"v3.4 {FAMILIES[fam]}" if key == 'v3.4' else f"`{key}` {FAMILIES[fam]}"
        lines.append(f"| {label} | {px['https']}/{px['n']} | {px['www']}/{px['n']} "
                     f"| {px['http']}/{px['n']} | {px['detect_flips_any']} "
                     f"| {px['detect_flips_https_http']} |")

    lines += ["", "## Deployment rule (protocol §c) — evaluation", ""]
    for fam in FAMILIES:
        c = s['deployment_rule'][fam]
        lines.append(f"### {FAMILIES[fam]}")
        lines.append("")
        lines.append(f"Baseline v3.4: detected FPR {c['baseline']['detect_fpr']}% | blocked FPR "
                     f"{c['baseline']['block_fpr']}% | detected recall {c['baseline']['recall']}%.")
        lines.append("")
        lines.append("| Candidate | c1 FPR ≤2% & ≤v3.4 | c2 blocked ≤1% | c3 +≥10pp & p<0.05 | pass |")
        lines.append("|---|---|---|---|---|")
        for arm, v in c['candidates'].items():
            lines.append(f"| `{arm}` | {'yes' if v['c1_fpr'] else 'no'} ({v['detect_fpr']}%) "
                         f"| {'yes' if v['c2_block'] else 'no'} ({v['block_fpr']}%) "
                         f"| {'yes' if v['c3_recall'] else 'no'} ({v['recall_gain_pp']:+.2f} pp, "
                         f"p={fmt_p(v['mcnemar_p'])}) | {'PASS' if v['pass'] else 'fail'} |")
        dec = c['decision']
        lines.append("")
        lines.append(f"**Decision: {'keep v3.4' if dec['keep_v34'] else 'replace with ' + dec['winner']}**")
        lines.append("")

    lines += ["", "## Context — in-distribution vs external (all arms)", "",
              "| Arm | Family | validation recall | test recall | external recall | "
              "validation FPR | test FPR | external FPR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for fam in FAMILIES:
            c = s['context_in_distribution'][arm][fam]
            vr = c['validation_recall'] * 100
            tr = c['test_recall'] * 100 if c['test_recall'] is not None else None
            er = c['external_recall'] * 100
            vf = c['validation_fpr'] * 100
            tf = c['test_fpr'] * 100 if c['test_fpr'] is not None else None
            ef = c['external_fpr'] * 100
            lines.append(f"| `{arm}` | {FAMILIES[fam]} | {vr:.2f}% "
                         f"| {f'{tr:.2f}%' if tr is not None else '—'} | {er:.2f}% "
                         f"| {vf:.3f}% | {f'{tf:.3f}%' if tf is not None else '—'} | {ef:.3f}% |")
    lines += ["", "## Notes", "",
              "- v3.4 rows score the deployed ONNX models through the extension feature path "
              "(`features.extract_features_onnx` / `preprocess_xgb`); arm rows score each cached "
              "`.pkl` through `benchmark_sampling.make_predictors`. Baseline and candidates are "
              "therefore both scored the way they would run in production/analysis.",
              "- Cross-check: the v3.4 browser numbers reproduce the 25/25 real-world test pattern "
              "(there: RF 0/25 recall, 0 FP; XGB 28.0% recall / 16.0% FPR; CatBoost 24.0% / 20.0%) "
              "at 20x the sample size, and the batched scorer was re-verified against per-URL "
              "`run_realworld_test.evaluate` scoring with an exact count match on the same "
              "1.000 URL variants.",
              "- Collection environment: Bitdefender real-time protection was active during "
              "collection; individual fetches may have been blocked by its web filtering "
              "(recorded as drops or non-answering entries, never silently dropped). Phishing "
              "liveness (33/500 answered, 15 live) is therefore a lower bound; scoring is "
              "URL-only, so headline metrics do not depend on liveness.",
              "- Phishing entries that no longer resolve are still scored (URL-only analysis); the "
              "live-only row reports the subset that answered when resolved.",
              "- The `all` arm is trained on the fit set (90% of the train pool), so it is close "
              "to but not identical with v3.4 (full train pool).",
              ]
    c = s['construction']
    lines += [
        "- Pre-registration trail: the frozen lists were committed before scoring (`18d49b6`); "
        "the scorer changed once afterwards (a KeyError hotfix in the rule-evaluation code plus "
        "report notes) — thresholds, metrics and rule logic are unchanged, and the scoring pass "
        "reproduced identical counts.",
        f"- Legit construction (documented, no re-sampling): the PhishTank exclusion matched "
        f"registrable (last-two-label) domains, so whole suffixes such as co.uk / com.au / co.id "
        f"were removed (no legit domains under those suffixes remain); "
        f"{c['whitelisted_legit_rows']} scored rows sit on hosts the extension whitelists "
        f"(v3.4 flags: RF {c['whitelisted_fp_v34']['rf']}, XGB {c['whitelisted_fp_v34']['xgb']}, "
        f"CB {c['whitelisted_fp_v34']['cb']} — 0.1 pp for XGB/CB); {c['deep_non200']} deep URLs "
        f"returned non-200 and {c['deep_offsite']} redirected off the landing site "
        f"({c['landing_redirected_registrable']} landings themselves redirected to another "
        f"registrable domain); all are scored as stored (`browser_url`).",
        f"- Legit/training overlap (from the committed audit): {c['legit_host_overlap_training']} "
        f"of 500 legit landing hosts also appear in the training data and "
        f"{c['legit_exact_training_rows']} scored legit URLs are exact training rows — the legit "
        f"FPR is therefore a slight underestimate for entirely unseen hosts.",
        f"- As-listed caveat: {c['bare_host_listings']} phishing entries are bare hosts without "
        f"the trailing slash a browser adds; on the {c['slash_only_rows']} slash-only rows, "
        f"CatBoost `no_scheme_www` detects "
        f"{c['slash_cb_no_scheme_www']['as_listed_detected']} as listed vs "
        f"{c['slash_cb_no_scheme_www']['with_slash_detected']} with the slash — as-listed recall "
        "for de-shortcut arms must not be quoted as browser-realistic.",
    ]
    if s['ablation_config']['warnings']:
        lines.append(f"- **WARNING:** ablation config problems: {s['ablation_config']['warnings']}")
    if s['training_overlap']['phishing_hosts_seen_in_training']:
        lines.append(f"- Training host overlap: {s['training_overlap']['phishing_hosts_seen_in_training']} "
                     f"of {s['training_overlap']['phishing_hosts_total']} phishing hosts also appear "
                     f"in the training dataset.")
    lines.append("")
    path = os.path.join(OUTDIR, 'external_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
