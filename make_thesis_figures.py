"""Draw the Bab 4 charts from the committed report files.

Every value is read from a report JSON; nothing is typed in by hand:

    sampling_results/overfitting.json      (analyze_overfitting.py)
    sampling_results/thresholds.json       (analyze_thresholds.py)
    ablation_results/ablation_metrics.json (benchmark_ablation.py)
    external_500/external_metrics.json     (external_500/score_external.py)

Usage (repo root):

    python make_thesis_figures.py            # writes docs/figures/*.png (300 dpi)

Style: one colour per model family in every chart (Random Forest blue,
XGBoost orange, CatBoost green) plus a diagonal hatch on two of them so the
series stay distinguishable in greyscale print; Times New Roman; Indonesian
labels with decimal commas.
"""

from __future__ import annotations

import json
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

OUTDIR = os.path.join('docs', 'figures')
WIDTH_IN = 5.6            # ~14.2 cm, the thesis text width
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e3e2de'
FAMILIES = ['rf', 'xgb', 'cb']
NAMES = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'cb': 'CatBoost'}
# Categorical slots 1-3 of the validated palette (colour-blind safe as a set).
COLORS = {'rf': '#2a78d6', 'xgb': '#eb6834', 'cb': '#1baf7a'}
HATCH = {'rf': None, 'xgb': '////', 'cb': '\\\\\\\\'}

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 9, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'hatch.linewidth': 0.5, 'hatch.color': '#ffffff', 'legend.frameon': False,
    'figure.dpi': 100, 'savefig.dpi': 300,
})


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def num(value, digits=1):
    """Indonesian decimal comma; rounds half up so labels match the report tables."""
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(round(value, 6))).quantize(quantum, rounding=ROUND_HALF_UP)).replace('.', ',')


def style_axis(ax, horizontal=False):
    ax.set_axisbelow(True)
    if horizontal:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    else:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(length=0)


def grouped_bars(ax, groups, values, fmt=lambda v: num(v), ylabel=None, ymax=None, label_size=7):
    """values[family] = one number per group. Bars grow from zero."""
    width = 0.24
    for i, fam in enumerate(FAMILIES):
        xs = [g + (i - 1) * (width + 0.02) for g in range(len(groups))]
        bars = ax.bar(xs, values[fam], width, color=COLORS[fam], hatch=HATCH[fam],
                      edgecolor='#ffffff', linewidth=0.4, label=NAMES[fam])
        for rect, v in zip(bars, values[fam]):
            ax.annotate(fmt(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 2), textcoords='offset points', ha='center', va='bottom',
                        fontsize=label_size, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    if ylabel:
        ax.set_ylabel(ylabel)
    top = ymax if ymax is not None else max(max(v) for v in values.values()) * 1.15
    ax.set_ylim(0, top)
    style_axis(ax)


def legend_below(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=ncol, fontsize=8.5,
               handlelength=1.6, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))


def save(fig, name, bottom=0.2):
    fig.tight_layout(rect=(0, bottom, 1, 1))
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, facecolor='#ffffff')
    plt.close(fig)
    print('wrote', path)


def fig_overfitting_bars(over):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.9))
    values = {f: [over['models'][f]['train']['f1'] * 100, over['models'][f]['test']['f1'] * 100]
              for f in FAMILIES}
    n_train, n_test = over['meta']['train']['total'], over['meta']['test']['total']
    groups = [f'Data latih\n({n_train:,} URL)'.replace(',', '.'),
              f'Data uji\n({n_test:,} URL)'.replace(',', '.')]
    grouped_bars(ax, groups, values, fmt=lambda v: num(v, 2), ylabel='F1-Score (%)', ymax=112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    legend_below(fig, ax)
    save(fig, 'gambar_overfitting_latih_uji.png', bottom=0.1)


def fig_overfitting_curves(over):
    # Boosting models only: they are fitted iteratively, so this is where a test curve
    # turning back up would show. Random Forest is covered by the train/test bars.
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.7))
    xlabel = {'xgb': 'Jumlah pohon', 'cb': 'Jumlah iterasi'}
    for ax, fam in zip(axes, ('xgb', 'cb')):
        curve = over['models'][fam]['curve']
        xs = [c['size'] for c in curve]
        for split, style, name in (('train', '-', 'Data latih'), ('test', '--', 'Data uji')):
            ys = [c[split]['log_loss'] for c in curve]
            ax.plot(xs, ys, style, color=COLORS[fam], linewidth=1.6, marker='o', markersize=2.6,
                    label=name)
        ax.set_title(NAMES[fam], fontsize=9)
        ax.set_xlabel(xlabel[fam], fontsize=8.5)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: num(v)))
        style_axis(ax)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('Log loss')
    handles = [plt.Line2D([], [], color=MUTED, linestyle='-', linewidth=1.6, label='Data latih'),
               plt.Line2D([], [], color=MUTED, linestyle='--', linewidth=1.6, label='Data uji')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, 'gambar_overfitting_kurva.png', bottom=0.1)


def fig_ablation(abl):
    arms = [('all', 'Seluruh fitur'), ('no_scheme', 'Tanpa skema'),
            ('no_scheme_www', 'Tanpa skema dan www.'), ('tfidf_only', 'Hanya TF-IDF'),
            ('struct_only', 'Hanya struktural'),
            ('tfidf_only_no_scheme_www', 'Hanya TF-IDF,\ntanpa skema dan www.'),
            ('struct_only_no_scheme_www', 'Hanya struktural,\ntanpa skema dan www.')]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.0))
    height = 0.25
    for i, fam in enumerate(FAMILIES):
        vals = [abl['arms'][a][fam]['test']['recall_phishing'] * 100 for a, _ in arms]
        ys = [g + (i - 1) * (height + 0.02) for g in range(len(arms))]
        bars = ax.barh(ys, vals, height, color=COLORS[fam], hatch=HATCH[fam],
                       edgecolor='#ffffff', linewidth=0.4, label=NAMES[fam])
        for rect, v in zip(bars, vals):
            ax.annotate(num(v), (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                        xytext=(2, 0), textcoords='offset points', ha='left', va='center', fontsize=7)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([label for _, label in arms])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Recall phishing pada test set (%)')
    style_axis(ax, horizontal=True)
    legend_below(fig, ax)
    save(fig, 'gambar_ablasi_recall.png', bottom=0.07)


def fig_external_candidates(ext):
    keys = [('v3.4', 'Model dasar'), ('no_scheme', 'Tanpa skema'), ('no_scheme_www', 'Tanpa skema\ndan www.')]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0))
    recall = {f: [ext['results'][k][f]['phish_browser']['detect_recall'] for k, _ in keys] for f in FAMILIES}
    fpr = {f: [ext['results'][k][f]['legit_all']['detect_fpr'] for k, _ in keys] for f in FAMILIES}
    grouped_bars(axes[0], [g for _, g in keys], recall, ylabel='URL phishing terdeteksi (%)', ymax=75, label_size=6.5)
    grouped_bars(axes[1], [g for _, g in keys], fpr, ylabel='URL legitimate terdeteksi (%)', ymax=75, label_size=6.5)
    axes[0].set_title('(a) Recall, 500 URL phishing', fontsize=9)
    axes[1].set_title('(b) FPR, 1.000 URL legitimate', fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=8)
    legend_below(fig, axes[0])
    save(fig, 'gambar_eksternal_kandidat.png', bottom=0.1)


def fig_www_split(ext):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.9))
    base = ext['results']['v3.4']
    n_www, n_non = base['rf']['legit_www']['n'], base['rf']['legit_non_www']['n']
    values = {f: [base[f]['legit_www']['detect_fpr'], base[f]['legit_non_www']['detect_fpr']] for f in FAMILIES}
    grouped_bars(ax, [f'Host berawalan www.\n({n_www} URL)', f'Host tanpa www.\n({n_non} URL)'], values,
                 ylabel='URL legitimate terdeteksi (%)', ymax=75)
    legend_below(fig, ax)
    save(fig, 'gambar_fpr_awalan_www.png', bottom=0.1)


def fig_prefix(ext):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.9))
    base = ext['results']['v3.4']
    values = {f: [base[f]['prefix']['https'], base[f]['prefix']['www'], base[f]['prefix']['http']] for f in FAMILIES}
    grouped_bars(ax, ['https://domain/', 'https://www.domain/', 'http://domain/'], values,
                 fmt=lambda v: str(int(v)), ylabel='Domain terdeteksi (dari 500)', ymax=560)
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    legend_below(fig, ax)
    save(fig, 'gambar_sensitivitas_awalan.png', bottom=0.1)


def fig_test_vs_external(ext, thr):
    at60 = {f: next(t for t in thr['models'][f]['by_threshold'] if abs(t['threshold'] - 0.6) < 1e-9)
            for f in FAMILIES}
    base = ext['results']['v3.4']
    recall = {f: [at60[f]['recall'] * 100, base[f]['phish_listed']['detect_recall'],
                  base[f]['phish_browser']['detect_recall']] for f in FAMILIES}
    fpr = {f: [at60[f]['fpr'] * 100, base[f]['legit_all']['detect_fpr']] for f in FAMILIES}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0), gridspec_kw={'width_ratios': [3, 2]})
    grouped_bars(axes[0], ['Test set', 'Eksternal,\nsesuai daftar', 'Eksternal,\nvarian browser'], recall,
                 ylabel='URL phishing terdeteksi (%)', ymax=112, label_size=6.5)
    grouped_bars(axes[1], ['Test set', 'Eksternal'], fpr, fmt=lambda v: num(v, 1 if v >= 1 else 2),
                 ylabel='URL legitimate terdeteksi (%)', ymax=112, label_size=6.5)
    axes[0].set_title('(a) Recall', fontsize=9)
    axes[1].set_title('(b) FPR', fontsize=9)
    for ax in axes:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(labelsize=8)
    legend_below(fig, axes[0])
    save(fig, 'gambar_test_vs_eksternal.png', bottom=0.1)


def fig_char_external(char):
    """Exploratory: ablation arms with and without the character n-gram block on the frozen lists."""
    arms = [('all', 'Seluruh fitur'), ('char', 'Seluruh fitur\n+ n-gram'),
            ('no_scheme_www', 'Tanpa skema\ndan www.'), ('char_no_scheme_www', 'Tanpa skema\ndan www.\n+ n-gram')]
    n_p, n_l = char['meta']['phishing'], char['meta']['legit']
    res = char['results']
    recall = {f: [100 * res[f'{a}__{f}']['phish_detected_browser'] / n_p for a, _ in arms] for f in FAMILIES}
    fpr = {f: [100 * res[f'{a}__{f}']['legit_detected'] / n_l for a, _ in arms] for f in FAMILIES}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.2))
    grouped_bars(axes[0], [g for _, g in arms], recall, ylabel='URL phishing terdeteksi (%)', ymax=100, label_size=5.5)
    grouped_bars(axes[1], [g for _, g in arms], fpr, ylabel='URL legitimate terdeteksi (%)', ymax=100, label_size=5.5)
    axes[0].set_title('(a) Recall, 500 URL phishing', fontsize=9)
    axes[1].set_title('(b) FPR, 1.000 URL legitimate', fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=6.5)
    legend_below(fig, axes[0])
    save(fig, 'gambar_ngram_eksternal.png', bottom=0.1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    over = load('sampling_results/overfitting.json')
    thr = load('sampling_results/thresholds.json')
    abl = load('ablation_results/ablation_metrics.json')
    ext = load('external_500/external_metrics.json')
    fig_overfitting_bars(over)
    fig_overfitting_curves(over)
    fig_ablation(abl)
    fig_external_candidates(ext)
    fig_www_split(ext)
    fig_prefix(ext)
    fig_test_vs_external(ext, thr)
    char_path = 'ablation_char_results/external_exploratory.json'
    if os.path.exists(char_path):
        fig_char_external(load(char_path))


if __name__ == '__main__':
    main()
