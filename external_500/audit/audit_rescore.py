"""Independent re-score of the frozen external lists (does not import score_external.py
or run_realworld_test.py). Compares every count with external_500/external_metrics.json.

Usage: audit_rescore.py <repo root> <scratch dir> [js]
  js  = also check the Python feature emulation against the real utils.js (needs node)
"""
import csv
import json
import math
import os
import subprocess
import sys

import joblib
import numpy as np
import onnxruntime as ort

ROOT, SCRATCH = sys.argv[1], sys.argv[2]
sys.path.insert(0, ROOT)
import features  # noqa: E402
import ml_pipelines  # noqa: E402

EXT = os.path.join(ROOT, 'phishingdetectorExt')
FROZEN = os.path.join(ROOT, 'external_500', 'frozen')
ARMS = ['all', 'no_scheme', 'no_scheme_www', 'tfidf_only', 'struct_only',
        'tfidf_only_no_scheme_www', 'struct_only_no_scheme_www']
FAMS = ['rf', 'xgb', 'cb']


def rows(name):
    with open(os.path.join(FROZEN, name), newline='', encoding='utf-8') as f:
        return [{k: v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


legit, phish, prefix = rows('legit_urls.csv'), rows('phishing_urls.csv'), rows('prefix_sensitivity.csv')
U = {
    'legit': [r['browser_url'] or r['url'] for r in legit],
    'pb': [r['browser_url'] or r['url'] for r in phish],
    'pl': [r['url'] for r in phish],
    'px': [u for r in prefix for u in (r['url_https'], r['url_https_www'], r['url_http'])],
}
all_urls = U['legit'] + U['pb'] + U['pl'] + U['px']
cuts = np.cumsum([0] + [len(U[k]) for k in ('legit', 'pb', 'pl', 'px')])
sl = {k: slice(cuts[i], cuts[i + 1]) for i, k in enumerate(('legit', 'pb', 'pl', 'px'))}
headline = np.array([r['in_headline'] == '1' for r in legit])

# ---------------- v3.4: per-URL features, one ONNX call per URL ----------------
rf_tfidf = json.load(open(os.path.join(EXT, 'tfidf_data.json')))
xgb_tfidf = json.load(open(os.path.join(EXT, 'tfidf_data_xgb.json')))
prep = json.load(open(os.path.join(EXT, 'xgb_preprocessing.json')))
sess = {f: ort.InferenceSession(os.path.join(EXT, n), providers=['CPUExecutionProvider'])
        for f, n in (('rf', 'phishing_rf.onnx'), ('xgb', 'phishing_xgb.onnx'), ('cb', 'phishing_catboost.onnx'))}

feat_rf = np.stack([np.concatenate([features.build_tfidf_vector(u, rf_tfidf), features.structural_features(u)])
                    for u in all_urls]).astype(np.float32)
feat_xgb_raw = np.stack([np.concatenate([features.build_tfidf_vector(u, xgb_tfidf), features.structural_features(u)])
                         for u in all_urls]).astype(np.float32)
feat_xgb = features.preprocess_xgb(feat_xgb_raw, prep)
print('v3.4 feature shapes:', feat_rf.shape, feat_xgb.shape)


def run_one_by_one(s, X):
    name = s.get_inputs()[0].name
    probs, labels = [], []
    for row in X:
        out = s.run(None, {name: row[None, :]})
        labels.append(int(np.asarray(out[0]).ravel()[0]))
        p = out[1]
        p = p[0] if isinstance(p, list) else np.asarray(p)[0]
        probs.append(float(p[1]) if not isinstance(p, dict) else float(p[1]))
    return np.array(probs) * 100.0, np.array(labels)


P, LABEL = {}, {}
for fam, X in (('rf', feat_rf), ('xgb', feat_xgb), ('cb', feat_rf)):
    P[('v3.4', fam)], LABEL[fam] = run_one_by_one(sess[fam], X)
    # the extension only acts when label == 1 AND probability > 60
    assert np.all(LABEL[fam][P[('v3.4', fam)] > 60] == 1), f'{fam}: score > 60 with label 0'

# ---------------- optional: Python emulation vs the real utils.js ----------------
if len(sys.argv) > 3 and sys.argv[3] == 'js':
    urls_json = os.path.join(SCRATCH, 'audit_urls.json')
    out_jsonl = os.path.join(SCRATCH, 'audit_js_features.jsonl')
    json.dump(all_urls, open(urls_json, 'w', encoding='utf-8'), ensure_ascii=False)
    subprocess.run(['node', os.path.join(SCRATCH, 'js_features.mjs'), EXT, urls_json, out_jsonl], check=True)
    worst = {'struct': 0.0, 'rf_tfidf': 0.0, 'xgb_in': 0.0}
    bad = 0
    with open(out_jsonl, encoding='utf-8') as f:
        for i, line in enumerate(f):
            js = json.loads(line)
            d_struct = float(np.max(np.abs(np.array(js['struct'], dtype=np.float64) - feat_rf[i, -9:])))
            dense = np.zeros(feat_rf.shape[1] - 9)
            for idx, val in js['rf']:
                dense[idx] = val
            d_rf = float(np.max(np.abs(dense - feat_rf[i, :-9])))
            d_x = float(np.max(np.abs(np.array(js['xgb_in']) - feat_xgb[i])))
            worst = {'struct': max(worst['struct'], d_struct), 'rf_tfidf': max(worst['rf_tfidf'], d_rf),
                     'xgb_in': max(worst['xgb_in'], d_x)}
            if d_struct > 1e-3 or d_rf > 1e-5 or d_x > 1e-2:
                bad += 1
                if bad <= 5:
                    print('  JS/Python feature mismatch:', all_urls[i][:100], d_struct, d_rf, d_x)
    print(f'JS vs Python features over {len(all_urls)} URL variants: max abs diff {worst} | rows over tolerance: {bad}')

# ---------------- ablation arms: direct pipeline calls ----------------
arr = np.array(all_urls)
for arm in ARMS:
    for fam in FAMS:
        m = joblib.load(os.path.join(ROOT, 'ablation_results', f'{arm}__{fam}.pkl'))
        if fam == 'rf':
            pr = m.predict_proba(arr)
        elif fam == 'xgb':
            pr = ml_pipelines.predict_proba_with_xgb(m, arr)
        else:
            pr = m['model'].predict_proba(ml_pipelines._to_dense(m['feature_extractor'].transform(arr)))
        P[(arm, fam)] = np.asarray(pr, dtype=float)[:, 1] * 100.0
        del m
print('scored', len(P), 'models on', len(all_urls), 'URL variants')
np.save(os.path.join(SCRATCH, 'audit_probs.npy'), {f'{k[0]}|{k[1]}': v for k, v in P.items()}, allow_pickle=True)


# ---------------- compare with the committed metrics ----------------
def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


ref = json.load(open(os.path.join(ROOT, 'external_500', 'external_metrics.json'), encoding='utf-8'))
diffs, checked = [], 0


def cmp(label, mine, theirs):
    global checked
    checked += 1
    if mine != theirs:
        diffs.append((label, mine, theirs))


for (key, fam), p in P.items():
    r = ref['results'][key][fam]
    lg, pb, pl, px = p[sl['legit']], p[sl['pb']], p[sl['pl']], p[sl['px']].reshape(-1, 3)
    cmp(f'{key}/{fam} legit_all detect', int((lg > 60).sum()), r['legit_all']['detect_fp'])
    cmp(f'{key}/{fam} legit_all block', int((lg > 80).sum()), r['legit_all']['block_fp'])
    cmp(f'{key}/{fam} legit_headline detect', int((lg[headline] > 60).sum()), r['legit_headline']['detect_fp'])
    cmp(f'{key}/{fam} legit_headline block', int((lg[headline] > 80).sum()), r['legit_headline']['block_fp'])
    cmp(f'{key}/{fam} phish_browser detect', int((pb > 60).sum()), r['phish_browser']['detect_tp'])
    cmp(f'{key}/{fam} phish_browser block', int((pb > 80).sum()), r['phish_browser']['block_tp'])
    cmp(f'{key}/{fam} phish_listed detect', int((pl > 60).sum()), r['phish_listed']['detect_tp'])
    for i, n in enumerate(('https', 'www', 'http')):
        cmp(f'{key}/{fam} prefix {n}', int((px[:, i] > 60).sum()), r['prefix'][n])
    if key != 'v3.4':
        base = P[('v3.4', fam)][sl['pb']] > 60
        cand = pb > 60
        b, c = int((cand & ~base).sum()), int((~cand & base).sum())
        mc = ref['mcnemar'][key][fam]['detected']
        cmp(f'{key}/{fam} mcnemar b', b, mc['b'])
        cmp(f'{key}/{fam} mcnemar c', c, mc['c'])
        if abs(exact_mcnemar(b, c) - mc['p']) > 1e-5:
            diffs.append((f'{key}/{fam} mcnemar p', exact_mcnemar(b, c), mc['p']))
        checked += 1

print(f'compared {checked} values with external_metrics.json -> {len(diffs)} differences')
for d in diffs[:20]:
    print('  DIFF', d)

# ---------------- deployment rule, recomputed from my own numbers ----------------
print('\nDeployment rule (recomputed):')
for fam in FAMS:
    base = P[('v3.4', fam)]
    b_fp = int((base[sl['legit']] > 60).sum())
    b_rec = (base[sl['pb']] > 60).mean() * 100
    for arm in ('no_scheme', 'no_scheme_www'):
        p = P[(arm, fam)]
        fp, blk = int((p[sl['legit']] > 60).sum()), int((p[sl['legit']] > 80).sum())
        rec = (p[sl['pb']] > 60).mean() * 100
        bb = int(((p[sl['pb']] > 60) & ~(base[sl['pb']] > 60)).sum())
        cc = int((~(p[sl['pb']] > 60) & (base[sl['pb']] > 60)).sum())
        c1 = fp / 10 <= 2.0 and fp <= b_fp
        c2 = blk / 10 <= 1.0
        c3 = rec - b_rec >= 10 and exact_mcnemar(bb, cc) < 0.05
        print(f'  {fam:3} {arm:14} FPR {fp/10:5.1f}% (v3.4 {b_fp/10:.1f}%) block {blk/10:5.1f}% recall {rec:5.1f}% '
              f'(v3.4 {b_rec:.1f}%, p={exact_mcnemar(bb, cc):.3g}) c1={c1} c2={c2} c3={c3} -> {"PASS" if c1 and c2 and c3 else "fail"}')
        ref_pass = ref['deployment_rule'][fam]['candidates'][arm]['pass']
        assert ref_pass == bool(c1 and c2 and c3)

# ---------------- diagnostics the report does not show ----------------
print('\nBy band, detected FPR on all legit rows (v3.4):')
for band in ('1-1k', '1k-10k', '10k-100k', '100k-500k', '500k-1M'):
    m = np.array([r['band'] == band for r in legit])
    print(f'  {band:10}', ' '.join(f"{f}:{(P[('v3.4', f)][sl['legit']][m] > 60).mean()*100:5.1f}%" for f in FAMS))

www = np.array([features.normalize_url(u, True).startswith('www.') for u in U['legit']])
print('\nLegit detected FPR split by leading www. (n www = %d, non-www = %d):' % (www.sum(), (~www).sum()))
for key in ('v3.4', 'no_scheme', 'no_scheme_www'):
    print(f'  {key:14}', ' '.join(
        f"{f}: www {(P[(key, f)][sl['legit']][www] > 60).mean()*100:5.1f}% / non-www {(P[(key, f)][sl['legit']][~www] > 60).mean()*100:5.1f}%"
        for f in FAMS))

listed_http = np.array([r['url'].lower().startswith('http://') for r in phish])
slash_only = np.array([r['url'] != r['browser_url'] and r['url'].lower().startswith('https://')
                       and r['url'] + '/' == r['browser_url'] for r in phish])
print(f'\nPhishing listed as http: {listed_http.sum()} | listed https, browser differs only by an added trailing slash: {slash_only.sum()}')
for key in ('v3.4', 'no_scheme', 'no_scheme_www'):
    for f in FAMS:
        pb, pl = P[(key, f)][sl['pb']] > 60, P[(key, f)][sl['pl']] > 60
        print(f'  {key:14} {f:3} detected among http-listed: as-listed {pl[listed_http].sum():3d} -> browser {pb[listed_http].sum():3d} of {listed_http.sum()}'
              f' | slash-only rows: as-listed {pl[slash_only].sum():3d} -> browser {pb[slash_only].sum():3d} of {slash_only.sum()}')

wl = ['google.com', 'facebook.com']
print('\nLegit rows on statically whitelisted hosts and their v3.4 verdicts:')
for i, u in enumerate(U['legit']):
    h = u.split('//')[1].split('/')[0].lower()
    if any(h == d or h.endswith('.' + d) for d in wl):
        print('  ', u[:70], {f: round(float(P[('v3.4', f)][sl['legit']][i]), 1) for f in FAMS})
