"""Real-world validation on live PhishTank URLs + legitimate sites.

Pak Rahmad's request: compare the models on real-world data with at least
20 legitimate and 20 phishing URLs. The phishing sample comes from the
PhishTank live feed (online-valid), collected 2026-09-20, sampled across
unique domains (seed 42).

URL-only analysis: the models see exactly the same string the extension
scans, so page liveness does not affect scoring.

Usage (run from the repo root):

    python run_realworld_test.py

Reads : realworld_test/urls_phishing.csv, realworld_test/urls_legit.csv
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


def is_free_hosting(url):
    host = url.split('//')[-1].split('/')[0].lower()
    return any(host == s or host.endswith('.' + s) or host.endswith(s) for s in FREE_HOSTING_SUFFIXES)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


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


def read_urls(path, label):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            url = (r.get('url') or '').strip()
            if url:
                rows.append({
                    'url': url,
                    'label': label,
                    'note': (r.get('note') or r.get('target') or '').strip(),
                })
    return rows


def summarize(results, model):
    pred = [r[f'{model}_label'] for r in results]
    true = [r['label'] for r in results]
    tp = sum(1 for p, t in zip(pred, true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, true) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(pred, true) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(pred, true) if p == 0 and t == 1)
    acc = (tp + tn) / len(true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'fpr': fpr}


def main():
    phish = read_urls(os.path.join(OUTDIR, 'urls_phishing.csv'), 1)
    legit = read_urls(os.path.join(OUTDIR, 'urls_legit.csv'), 0)
    print(f"URLs: {len(phish)} phishing + {len(legit)} legitimate = {len(phish) + len(legit)}")
    if len(phish) < 20 or len(legit) < 20:
        print("WARNING: Pak Rahmad's minimum is 20 per class")

    models = load_models()
    results = []
    sanity = {'rf': 0, 'xgb': 0}
    for row in phish + legit:
        url = row['url']

        rf_label, rf_p = onnx_predict(
            models['rf_sess'], extract_features_onnx([url], models['rf_tfidf']))
        xgb_feats = preprocess_xgb(
            extract_features_onnx([url], models['xgb_tfidf']), models['xgb_prep'])
        xgb_label, xgb_p = onnx_predict(models['xgb_sess'], xgb_feats)

        # sanity: predicted label should agree with the probability ordering
        sanity['rf'] += int((rf_label == 1) != (rf_p > 0.5))
        sanity['xgb'] += int((xgb_label == 1) != (xgb_p > 0.5))

        results.append({
            **row,
            'group': 'free-hosting' if (row['label'] == 1 and is_free_hosting(url)) else
                     ('owned-domain' if row['label'] == 1 else 'legit'),
            'rf_label': rf_label, 'rf_conf': round(rf_p * 100, 2),
            'rf_verdict': verdict(rf_label, rf_p * 100),
            'xgb_label': xgb_label, 'xgb_conf': round(xgb_p * 100, 2),
            'xgb_verdict': verdict(xgb_label, xgb_p * 100),
            'rf_correct': int(rf_label == row['label']),
            'xgb_correct': int(xgb_label == row['label']),
        })
        print(f"  {'phish' if row['label'] else 'legit'} | RF {rf_label}/{rf_p*100:5.1f}% "
              f"| XGB {xgb_label}/{xgb_p*100:5.1f}% | {url[:70]}")

    print(f"\nlabel/probability-order sanity mismatches: {sanity}")

    # ---- training-data context: share of free-hosting phishing in the Mendeley set ----
    dataset_note = None
    try:
        with open('URL dataset.csv', newline='', encoding='utf-8') as f:
            ds_phish = [r['url'] for r in csv.DictReader(f)
                        if (r.get('type') or '').strip() == 'phishing' and r.get('url')]
        if ds_phish:
            free_share = sum(is_free_hosting(u) for u in ds_phish) / len(ds_phish) * 100
            live_free = [r for r in results if r['label'] == 1 and r['group'] == 'free-hosting']
            live_total = [r for r in results if r['label'] == 1]
            dataset_note = (f"Training set (Mendeley 2024): {free_share:.1f}% of phishing URLs are on "
                            f"free-hosting platforms vs {len(live_free)}/{len(live_total)} "
                            f"({len(live_free)/len(live_total)*100:.0f}%) in this live sample.")
    except FileNotFoundError:
        pass
    if dataset_note:
        print("\n" + dataset_note)

    rf_sum = summarize(results, 'rf')
    xgb_sum = summarize(results, 'xgb')
    agreement = np.mean([r['rf_label'] == r['xgb_label'] for r in results]) * 100

    # ---- per-URL CSV ----
    csv_path = os.path.join(OUTDIR, 'realworld_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ---- Markdown report ----
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
    md.append("- URL-only analysis (same input the extension sees); liveness does not affect scoring")
    if dataset_note:
        md.append(f"- {dataset_note}")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| Metric | Random Forest | XGBoost |")
    md.append("|---|---:|---:|")
    md.append(f"| Accuracy | {rf_sum['acc']*100:.2f}% | {xgb_sum['acc']*100:.2f}% |")
    md.append(f"| Precision (phishing) | {rf_sum['prec']*100:.2f}% | {xgb_sum['prec']*100:.2f}% |")
    md.append(f"| Recall (phishing) | {rf_sum['rec']*100:.2f}% | {xgb_sum['rec']*100:.2f}% |")
    md.append(f"| F1 (phishing) | {rf_sum['f1']*100:.2f}% | {xgb_sum['f1']*100:.2f}% |")
    md.append(f"| False Positive Rate | {rf_sum['fpr']*100:.2f}% | {xgb_sum['fpr']*100:.2f}% |")
    md.append(f"| TP / FP / TN / FN | {rf_sum['tp']} / {rf_sum['fp']} / {rf_sum['tn']} / {rf_sum['fn']} "
              f"| {xgb_sum['tp']} / {xgb_sum['fp']} / {xgb_sum['tn']} / {xgb_sum['fn']} |")
    md.append("")
    md.append(f"RF and XGBoost agree on {agreement:.1f}% of URLs.")
    md.append("")
    md.append("## Phishing sample by domain type")
    md.append("")
    md.append("| Domain type | n | RF caught | XGBoost caught |")
    md.append("|---|---:|---:|---:|")
    for group in ('free-hosting', 'owned-domain'):
        rows = [r for r in results if r['label'] == 1 and r['group'] == group]
        rf_c = sum(r['rf_label'] for r in rows)
        xgb_c = sum(r['xgb_label'] for r in rows)
        md.append(f"| {group} | {len(rows)} | {rf_c}/{len(rows)} | {xgb_c}/{len(rows)} |")
    md.append("")
    md.append("## Extension behavior (decision thresholds)")
    md.append("")
    md.append("| Model | Blocked (>80%) | Warned (60-80%) | SAFE (<60%) |")
    md.append("|---|---|---|---|")
    for model, label in [('rf', 'Random Forest'), ('xgb', 'XGBoost')]:
        blocked_tp = sum(1 for r in results if r['label'] == 1 and r[f'{model}_verdict'] == 'PHISHING')
        blocked_fp = sum(1 for r in results if r['label'] == 0 and r[f'{model}_verdict'] == 'PHISHING')
        warned = sum(1 for r in results if r[f'{model}_verdict'] == 'SUSPICIOUS')
        safe = sum(1 for r in results if r[f'{model}_verdict'] == 'SAFE')
        md.append(f"| {label} | {blocked_tp + blocked_fp} ({blocked_tp} TP / {blocked_fp} FP) "
                  f"| {warned} | {safe} |")
    md.append("")
    md.append("## Per-URL results")
    md.append("")
    md.append("Format: ✓/✗ = raw model label vs expected; the word shown is the extension "
              "verdict at its thresholds (>80% PHISHING, 60–80% SUSPICIOUS, <60% SAFE).")
    md.append("")
    md.append("| # | Expected | URL | RF | XGB |")
    md.append("|---:|---|---|---|---|")
    for i, r in enumerate(results, 1):
        exp = 'phish' if r['label'] == 1 else 'legit'
        rf_mark = '✓' if r['rf_correct'] else '✗'
        xgb_mark = '✓' if r['xgb_correct'] else '✗'
        url = r['url'].replace('|', '%7C')
        md.append(f"| {i} | {exp} | {url} | {rf_mark} {r['rf_verdict']} ({r['rf_conf']:.1f}%) "
                  f"| {xgb_mark} {r['xgb_verdict']} ({r['xgb_conf']:.1f}%) |")
    md.append("")
    md.append("## Errors")
    md.append("")
    for model, label in [('rf', 'Random Forest'), ('xgb', 'XGBoost')]:
        fps = [r['url'] for r in results if r['label'] == 0 and r[f'{model}_label'] == 1]
        fns = [r['url'] for r in results if r['label'] == 1 and r[f'{model}_label'] == 0]
        md.append(f"**{label}** — false positives: {len(fps)}, false negatives: {len(fns)}")
        for u in fps:
            md.append(f"- FP: {u}")
        for u in fns:
            md.append(f"- FN: {u}")
        md.append("")
    md.append("> Thresholds mirror the extension: >80% PHISHING, 60–80% SUSPICIOUS, <60% SAFE.")
    md_path = os.path.join(OUTDIR, 'realworld_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md) + "\n")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"RF  accuracy {rf_sum['acc']*100:.2f}% | FPR {rf_sum['fpr']*100:.2f}% | FP {rf_sum['fp']} | FN {rf_sum['fn']}")
    print(f"XGB accuracy {xgb_sum['acc']*100:.2f}% | FPR {xgb_sum['fpr']*100:.2f}% | FP {xgb_sum['fp']} | FN {xgb_sum['fn']}")


if __name__ == "__main__":
    main()