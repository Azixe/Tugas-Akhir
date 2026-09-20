"""Real-world validation on live PhishTank URLs + legitimate sites.

Pak Rahmad's request: compare the models on real-world data with at least
20 legitimate and 20 phishing URLs. The phishing sample comes from the
PhishTank live feed (online-valid), collected 2026-09-20, sampled across
unique domains (seed 42).

Two URL variants are scored for every entry:

  * as-listed   — the exact string in the PhishTank feed
  * browser     — what the browser actually scans: http:// entries are
                  upgraded to https:// (Chrome HTTPS-First / HSTS), and
                  curl-resolved redirects are applied

The "browser" variant is the primary summary; "as-listed" is kept as a
reference. URL-only analysis: liveness does not affect scoring.

Usage (run from the repo root):

    python run_realworld_test.py

Reads : realworld_test/urls_phishing.csv (+ urls_phishing_resolved.csv if present),
        realworld_test/urls_legit.csv
Writes: realworld_test/realworld_results.csv, realworld_test/realworld_report.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features_onnx, preprocess_xgb  # noqa: E402

EXT = 'phishingdetectorExt'
OUTDIR = 'realworld_test'

# Free-hosting / CDN platforms frequently abused for phishing: the URL string
# carries little signal (the brand appears only on the page), so URL-only
# models are expected to struggle here.
FREE_HOSTING_SUFFIXES = (
    'pages.dev', 'web.app', 'netlify.app', 'vercel.app', 'weebly.com',
    'weeblysite.com', 'wixsite.com', 'wixstudio.com', 'framer.app', 'r2.dev',
    'dweb.link', 'webcindario.com', 'blogspot.com', 'github.io', 'glitch.me',
    'repl.co', 'firebaseapp.com', 'herokuapp.com', '000webhostapp.com',
    'canva.site', 'notion.site', 'sharepoint.com', 'forms.gle', 'sites.google.com',
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def is_free_hosting(url):
    host = url.split('//')[-1].split('/')[0].lower()
    return any(host == s or host.endswith('.' + s) or host.endswith(s)
               for s in FREE_HOSTING_SUFFIXES)


def https_variant(url):
    if url.startswith('http://'):
        return 'https://' + url[len('http://'):]
    return url


def load_models():
    with open(os.path.join(EXT, 'tfidf_data.json')) as f:
        rf_tfidf = json.load(f)
    with open(os.path.join(EXT, 'tfidf_data_xgb.json')) as f:
        xgb_tfidf = json.load(f)
    with open(os.path.join(EXT, 'xgb_preprocessing.json')) as f:
        xgb_prep = json.load(f)
    return {
        'rf_sess': ort.InferenceSession(os.path.join(EXT, 'phishing_rf.onnx'),
                                        providers=['CPUExecutionProvider']),
        'xgb_sess': ort.InferenceSession(os.path.join(EXT, 'phishing_xgb.onnx'),
                                         providers=['CPUExecutionProvider']),
        'rf_tfidf': rf_tfidf,
        'xgb_tfidf': xgb_tfidf,
        'xgb_prep': xgb_prep,
    }


def onnx_predict(session, X):
    res = session.run(None, {session.get_inputs()[0].name: X})
    label = int(res[0].flatten()[0])
    probs = np.asarray(res[1]).flatten()
    return label, float(probs[1]) if len(probs) > 1 else float(probs[0])


def verdict(label, conf_pct):
    if label == 1 and conf_pct > 80:
        return 'PHISHING'
    if label == 1 and conf_pct > 60:
        return 'SUSPICIOUS'
    return 'SAFE'


def evaluate(models, url):
    rf_label, rf_p = onnx_predict(
        models['rf_sess'], extract_features_onnx([url], models['rf_tfidf']))
    xgb_label, xgb_p = onnx_predict(
        models['xgb_sess'],
        preprocess_xgb(extract_features_onnx([url], models['xgb_tfidf']), models['xgb_prep']))
    return {
        'rf_label': rf_label, 'rf_conf': round(rf_p * 100, 2),
        'rf_verdict': verdict(rf_label, rf_p * 100),
        'xgb_label': xgb_label, 'xgb_conf': round(xgb_p * 100, 2),
        'xgb_verdict': verdict(xgb_label, xgb_p * 100),
    }


def read_urls(path, label):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            url = (r.get('url') or '').strip()
            if url:
                rows.append({'url': url, 'label': label,
                             'note': (r.get('note') or r.get('target') or '').strip()})
    return rows


def read_resolved():
    path = os.path.join(OUTDIR, 'urls_phishing_resolved.csv')
    mapping = {}
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                final = (r.get('final_url') or '').strip()
                if r.get('url') and final:
                    mapping[r['url']] = final
    return mapping


def browser_url(url, resolved):
    return https_variant(resolved.get(url, url))


def summarize(results, variant):
    """variant: 'listed' or 'browser'."""
    def pick(r, field):
        return r[f'{field}_browser'] if variant == 'browser' else r[field]

    pred = [pick(r, f'{m}_label') for r in results for m in ('rf', 'xgb')]
    out = {}
    true = [r['label'] for r in results]
    for model in ('rf', 'xgb'):
        pred = [pick(r, f'{model}_label') for r in results]
        tp = sum(1 for p, t in zip(pred, true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred, true) if p == 1 and t == 0)
        tn = sum(1 for p, t in zip(pred, true) if p == 0 and t == 0)
        fn = sum(1 for p, t in zip(pred, true) if p == 0 and t == 1)
        acc = (tp + tn) / len(true)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        out[model] = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                      'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'fpr': fpr}
    return out


def dataset_context():
    """Scheme distribution of the training set + top RF features (if available)."""
    stats = None
    try:
        from collections import Counter
        c = {'phishing': Counter(), 'legitimate': Counter()}
        with open('URL dataset.csv', newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                t = (r.get('type') or '').strip()
                u = (r.get('url') or '').strip().lower()
                if t in c:
                    c[t]['https' if u.startswith('https://') else
                         ('http' if u.startswith('http://') else 'other')] += 1
        stats = c
    except FileNotFoundError:
        pass

    top_features = None
    try:
        import joblib
        pipe = joblib.load(os.path.join('sampling_results', 'rf_full.pkl'))
        clf = pipe.named_steps['clf']
        vocab = pipe.named_steps['features'].transformer_list[0][1].vocabulary_
        inv = {v: k for k, v in vocab.items()}
        imp = clf.feature_importances_
        top_features = [(inv.get(i, f'structural#{i - 1500}'), float(imp[i]))
                        for i in np.argsort(imp)[-2:][::-1]]
    except Exception:
        pass
    return stats, top_features


def main():
    phish = read_urls(os.path.join(OUTDIR, 'urls_phishing.csv'), 1)
    legit = read_urls(os.path.join(OUTDIR, 'urls_legit.csv'), 0)
    resolved = read_resolved()
    print(f"URLs: {len(phish)} phishing + {len(legit)} legitimate = {len(phish) + len(legit)}")
    print(f"Resolved final URLs available for {len(resolved)} phishing entries")

    models = load_models()
    results = []
    for row in phish + legit:
        url = row['url']
        burl = browser_url(url, resolved)
        listed = evaluate(models, url)
        browser = evaluate(models, burl) if burl != url else dict(listed)
        results.append({
            **row,
            'url_browser': burl,
            'group': 'free-hosting' if (row['label'] == 1 and is_free_hosting(burl)) else
                     ('owned-domain' if row['label'] == 1 else 'legit'),
            **listed,
            **{f'{k}_browser': v for k, v in browser.items()},
            'rf_correct': int(listed['rf_label'] == row['label']),
            'xgb_correct': int(listed['xgb_label'] == row['label']),
            'rf_correct_browser': int(browser['rf_label'] == row['label']),
            'xgb_correct_browser': int(browser['xgb_label'] == row['label']),
            'scheme_changed': int(burl != url),
        })
        flag = ' [scheme/redirect variant]' if burl != url else ''
        print(f"  {'phish' if row['label'] else 'legit'} | RF {listed['rf_label']}/"
              f"{listed['rf_conf']:5.1f}% (browser {browser['rf_label']}/{browser['rf_conf']:5.1f}%) | "
              f"XGB {listed['xgb_label']}/{listed['xgb_conf']:5.1f}% "
              f"(browser {browser['xgb_label']}/{browser['xgb_conf']:5.1f}%) | {url[:60]}{flag}")

    listed_sum = summarize(results, 'listed')
    browser_sum = summarize(results, 'browser')
    agreement = np.mean([r['rf_label_browser'] == r['xgb_label_browser'] for r in results]) * 100
    changed = [r for r in results if r['scheme_changed']]

    # ---- per-URL CSV ----
    csv_path = os.path.join(OUTDIR, 'realworld_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ---- Markdown report ----
    stats, top_features = dataset_context()
    ext_manifest = json.load(open(os.path.join(EXT, 'manifest.json')))
    md = []
    md.append("# Real-World Validation — Live PhishTank + Legitimate URLs")
    md.append("")
    md.append(f"- Phishing source: PhishTank `online-valid` (76,683 URLs with unique domains), "
              f"sampled {len(phish)} across unique domains, seed 42, collected 2026-09-20")
    md.append(f"- Legitimate: {len(legit)} manual URLs (FP-prone sites: Steam, Reddit, Discord, ...)")
    md.append(f"- Extension model version: v{ext_manifest['version']}  |  "
              f"RF sha256 `{sha256(os.path.join(EXT, 'phishing_rf.onnx'))[:16]}…`  |  "
              f"XGB sha256 `{sha256(os.path.join(EXT, 'phishing_xgb.onnx'))[:16]}…`")
    md.append(f"- Two variants scored per URL: **as-listed** (raw feed string) and **browser** "
              f"(http→https upgrade + curl-resolved redirects; {len(changed)} URLs differ). "
              f"The browser variant is primary.")
    md.append("- URL-only analysis (same input the extension sees); liveness does not affect scoring")
    if stats:
        ph = stats['phishing']; lg = stats['legitimate']
        tot_p = sum(ph.values()); tot_l = sum(lg.values())
        md.append(f"- Training set (Mendeley 2024): phishing {ph['http']/tot_p*100:.1f}% http / "
                  f"{ph['https']/tot_p*100:.1f}% https, legitimate {lg['http']/tot_l*100:.1f}% http / "
                  f"{lg['https']/tot_l*100:.1f}% https — the scheme alone separates ~94% of the data")
    if top_features:
        md.append(f"- RF's top-2 features by importance: `{top_features[0][0]}` "
                  f"({top_features[0][1]*100:.1f}%) and `{top_features[1][0]}` "
                  f"({top_features[1][1]*100:.1f}%) — a dataset shortcut now obsolete (modern phishing is https)")
    md.append("")
    md.append("## Summary — browser-observed (primary)")
    md.append("")
    md.append("| Metric | Random Forest | XGBoost |")
    md.append("|---|---:|---:|")
    for label, key, fmt in [
            ('Accuracy', 'acc', '{:.2f}%'), ('Precision (phishing)', 'prec', '{:.2f}%'),
            ('Recall (phishing)', 'rec', '{:.2f}%'), ('F1 (phishing)', 'f1', '{:.2f}%'),
            ('False Positive Rate', 'fpr', '{:.2f}%')]:
        md.append(f"| {label} | " +
                  " | ".join(fmt.format(browser_sum[m][key] * 100) for m in ('rf', 'xgb')) + " |")
    md.append(f"| TP / FP / TN / FN | " +
              " | ".join(f"{browser_sum[m]['tp']} / {browser_sum[m]['fp']} / "
                         f"{browser_sum[m]['tn']} / {browser_sum[m]['fn']}" for m in ('rf', 'xgb')) + " |")
    md.append("")
    md.append(f"RF and XGBoost agree on {agreement:.1f}% of URLs.")
    md.append("")
    md.append("## Summary — as-listed (reference)")
    md.append("")
    md.append("| Metric | Random Forest | XGBoost |")
    md.append("|---|---:|---:|")
    for label, key, fmt in [
            ('Accuracy', 'acc', '{:.2f}%'), ('Recall (phishing)', 'rec', '{:.2f}%'),
            ('False Positive Rate', 'fpr', '{:.2f}%')]:
        md.append(f"| {label} | " +
                  " | ".join(fmt.format(listed_sum[m][key] * 100) for m in ('rf', 'xgb')) + " |")
    md.append(f"| TP / FP / TN / FN | " +
              " | ".join(f"{listed_sum[m]['tp']} / {listed_sum[m]['fp']} / "
                         f"{listed_sum[m]['tn']} / {listed_sum[m]['fn']}" for m in ('rf', 'xgb')) + " |")
    md.append("")
    md.append("## Scheme sensitivity (why one letter flips the model)")
    md.append("")
    md.append("| URL (as-listed) | RF as-listed | RF browser | XGB as-listed | XGB browser |")
    md.append("|---|---:|---:|---:|---:|")
    for r in changed:
        md.append(f"| {r['url']} | {r['rf_label']} / {r['rf_conf']:.1f}% "
                  f"| {r['rf_label_browser']} / {r['rf_conf_browser']:.1f}% "
                  f"| {r['xgb_label']} / {r['xgb_conf']:.1f}% "
                  f"| {r['xgb_label_browser']} / {r['xgb_conf_browser']:.1f}% |")
    md.append("")
    md.append("## Phishing sample by domain type (browser variant)")
    md.append("")
    md.append("| Domain type | n | RF caught | XGBoost caught |")
    md.append("|---|---:|---:|---:|")
    for group in ('free-hosting', 'owned-domain'):
        rows = [r for r in results if r['label'] == 1 and r['group'] == group]
        md.append(f"| {group} | {len(rows)} | {sum(r['rf_label_browser'] for r in rows)}/{len(rows)} "
                  f"| {sum(r['xgb_label_browser'] for r in rows)}/{len(rows)} |")
    md.append("")
    md.append("## Extension behavior (decision thresholds, browser variant)")
    md.append("")
    md.append("| Model | Blocked (>80%) | Warned (60-80%) | SAFE (<60%) |")
    md.append("|---|---|---|---|")
    for model, label in [('rf', 'Random Forest'), ('xgb', 'XGBoost')]:
        blocked_tp = sum(1 for r in results if r['label'] == 1
                         and r[f'{model}_verdict_browser'] == 'PHISHING')
        blocked_fp = sum(1 for r in results if r['label'] == 0
                         and r[f'{model}_verdict_browser'] == 'PHISHING')
        warned = sum(1 for r in results if r[f'{model}_verdict_browser'] == 'SUSPICIOUS')
        safe = sum(1 for r in results if r[f'{model}_verdict_browser'] == 'SAFE')
        md.append(f"| {label} | {blocked_tp + blocked_fp} ({blocked_tp} TP / {blocked_fp} FP) "
                  f"| {warned} | {safe} |")
    md.append("")
    md.append("## Per-URL results (browser variant)")
    md.append("")
    md.append("Format: ✓/✗ = raw model label vs expected; the word is the extension verdict "
              "at its thresholds (>80% PHISHING, 60–80% SUSPICIOUS, <60% SAFE).")
    md.append("")
    md.append("| # | Expected | URL (browser) | RF | XGB |")
    md.append("|---:|---|---|---|---|")
    for i, r in enumerate(results, 1):
        exp = 'phish' if r['label'] == 1 else 'legit'
        rf_mark = '✓' if r['rf_correct_browser'] else '✗'
        xgb_mark = '✓' if r['xgb_correct_browser'] else '✗'
        url = r['url_browser'].replace('|', '%7C')
        md.append(f"| {i} | {exp} | {url} | {rf_mark} {r['rf_verdict_browser']} "
                  f"({r['rf_conf_browser']:.1f}%) | {xgb_mark} {r['xgb_verdict_browser']} "
                  f"({r['xgb_conf_browser']:.1f}%) |")
    md.append("")
    md.append("## Errors (browser variant)")
    md.append("")
    for model, label in [('rf', 'Random Forest'), ('xgb', 'XGBoost')]:
        fps = [r['url_browser'] for r in results if r['label'] == 0 and r[f'{model}_label_browser'] == 1]
        fns = [r['url_browser'] for r in results if r['label'] == 1 and r[f'{model}_label_browser'] == 0]
        md.append(f"**{label}** — false positives: {len(fps)}, false negatives: {len(fns)}")
        for u in fps:
            md.append(f"- FP: {u}")
        for u in fns:
            md.append(f"- FN: {u}")
        md.append("")
    md.append("> Thresholds mirror the extension: >80% PHISHING, 60–80% SUSPICIOUS, <60% SAFE. "
              "Network reachability of the sample is logged in `urls_phishing_resolved.csv` "
              "and does not affect scoring.")
    md_path = os.path.join(OUTDIR, 'realworld_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md) + "\n")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")
    for m, label in [('rf', 'RF '), ('xgb', 'XGB')]:
        b, l = browser_sum[m], listed_sum[m]
        print(f"{label} browser: acc {b['acc']*100:.2f}% | recall {b['rec']*100:.2f}% | "
              f"FPR {b['fpr']*100:.2f}% | FP {b['fp']} FN {b['fn']}   "
              f"(as-listed: acc {l['acc']*100:.2f}% | recall {l['rec']*100:.2f}%)")


if __name__ == "__main__":
    main()