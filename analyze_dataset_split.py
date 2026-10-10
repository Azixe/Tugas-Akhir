"""Quantify train/test host overlap on the shared protocol (limitation analysis).

A random row-level 80/20 split can place URLs from the *same host* (or the same
site infrastructure) on both sides of the split, which inflates in-distribution
metrics. This script measures how often that happens in the 450,176-URL dataset.

It also measures two prefix shortcuts per class on the whole cleaned dataset:
the URL scheme and a leading ``www.`` label. Both are properties of how the
dataset was collected rather than of phishing itself. The share of host-only
URLs (no path) is reported next to them as a check; it is not a shortcut.

Definitions:
- host: lowercase hostname (no port/userinfo/scheme). IP literals kept as-is.
- registrable domain (approximation): the last two dot-labels of the host
  (e.g. ``login.example.co.id`` -> ``co.id``). This is deliberately coarse
  (no PSL); the exact-host number is the headline, the 2-label number is an
  upper bound on "same site family" overlap.

Outputs:
    sampling_results/dataset_split_report.md
    sampling_results/dataset_split.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampling  # noqa: E402


def host_of(url: str) -> str:
    s = str(url).strip().lower()
    if '://' in s:
        s = s.split('://', 1)[1]
    s = s.split('/', 1)[0]
    s = s.split('@')[-1]
    s = s.split(':', 1)[0]
    return s


SCHEME_WWW_RE = re.compile(r'^[a-z][a-z0-9+.\-]*://www\.')


def path_of(url: str) -> str:
    """Everything after the host, without a bare trailing slash."""
    s = str(url).strip().lower()
    if '://' in s:
        s = s.split('://', 1)[1]
    return s.split('/', 1)[1] if '/' in s else ''


def scheme_of(url: str) -> str:
    s = str(url).strip().lower()
    return s.split('://', 1)[0] if '://' in s else '(none)'


def prefix_stats(urls) -> dict:
    """Scheme and leading-``www.`` shares for one group of URLs."""
    n = len(urls)
    lowered = [str(u).strip().lower() for u in urls]
    schemes = Counter(scheme_of(u) for u in lowered)
    scheme_www = sum(1 for u in lowered if SCHEME_WWW_RE.match(u))
    host_www = sum(1 for u in lowered if host_of(u).startswith('www.'))
    no_path = sum(1 for u in lowered if not path_of(u))
    return {
        'rows': int(n),
        'schemes': {k: int(v) for k, v in schemes.most_common()},
        'schemes_pct': {k: round(100.0 * v / n, 2) for k, v in schemes.most_common()},
        'scheme_www_prefix': int(scheme_www),
        'scheme_www_prefix_pct': round(100.0 * scheme_www / n, 2),
        'host_starts_with_www': int(host_www),
        'host_starts_with_www_pct': round(100.0 * host_www / n, 2),
        'host_only_no_path': int(no_path),
        'host_only_no_path_pct': round(100.0 * no_path / n, 2),
    }


def registrable_approx(host: str) -> str:
    if not host:
        return ''
    parts = host.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else host


def main():
    df = sampling.load_clean_dataset()
    train_pool, test = sampling.common_split(df)
    print(f"Train {len(train_pool)} / test {len(test)}")

    train_hosts = Counter(host_of(u) for u in train_pool['url'].values)
    train_regs = {registrable_approx(h) for h in train_hosts}

    test_hosts = [host_of(u) for u in test['url'].values]
    n = len(test_hosts)
    host_hits = sum(1 for h in test_hosts if h in train_hosts)
    reg_hits = sum(1 for h in test_hosts if registrable_approx(h) in train_regs)

    top_overlap = [(h, c) for h, c in Counter(h for h in test_hosts if h in train_hosts).most_common(10)]

    prefixes = {
        name: prefix_stats(df.loc[df['label'] == label, 'url'].values)
        for name, label in (('legitimate', 0), ('phishing', 1))
    }

    res = {
        'train_rows': int(len(train_pool)),
        'test_rows': int(n),
        'train_unique_hosts': int(len(train_hosts)),
        'test_unique_hosts': int(len(set(test_hosts))),
        'test_rows_host_in_train': int(host_hits),
        'test_rows_host_in_train_pct': round(100.0 * host_hits / n, 2),
        'test_rows_registrable_in_train': int(reg_hits),
        'test_rows_registrable_in_train_pct': round(100.0 * reg_hits / n, 2),
        'top_overlapping_hosts': top_overlap,
        'prefix_shortcuts': prefixes,
    }

    lines = [
        "# Train/Test Host Overlap — random split limitation",
        "",
        f"Common protocol (stratified 80/20, seed {sampling.SEED}); rows deduplicated by URL.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Test rows whose exact host appears in train | {res['test_rows_host_in_train']} "
        f"({res['test_rows_host_in_train_pct']}%) |",
        f"| Test rows whose 2-label domain appears in train | {res['test_rows_registrable_in_train']} "
        f"({res['test_rows_registrable_in_train_pct']}%) |",
        f"| Unique hosts — train / test | {res['train_unique_hosts']:,} / {res['test_unique_hosts']:,} |",
        "",
        "## Top overlapping exact hosts (test rows)",
        "",
        "| Host | Test rows |", "|---|---:|",
    ]
    lines += [f"| `{h}` | {c} |" for h, c in top_overlap]
    lines += ["",
              "> Reading: a row-level random split measures in-distribution generalization; the overlap above "
              "quantifies how much of it comes from the same hosts being seen during training.",
              ""]

    scheme_names = sorted({k for p in prefixes.values() for k in p['schemes']},
                          key=lambda k: -sum(p['schemes'].get(k, 0) for p in prefixes.values()))
    lines += [
        "## Prefix shortcuts by class (whole cleaned dataset)",
        "",
        "| Metric | Legitimate | Phishing |",
        "|---|---:|---:|",
        f"| Rows | {prefixes['legitimate']['rows']:,} | {prefixes['phishing']['rows']:,} |",
    ]
    for k in scheme_names:
        cells = [f"{p['schemes'].get(k, 0):,} ({p['schemes_pct'].get(k, 0.0):.2f}%)" for p in prefixes.values()]
        lines.append(f"| Scheme `{k}` | {cells[0]} | {cells[1]} |")
    for key, label in (('scheme_www_prefix', 'URL starts with scheme + `www.`'),
                       ('host_starts_with_www', 'Host starts with `www.`'),
                       ('host_only_no_path', 'Host only (no path after the host)')):
        cells = [f"{p[key]:,} ({p[key + '_pct']:.2f}%)" for p in prefixes.values()]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines += ["",
              "> Reading: the scheme and a leading `www.` label each separate the two classes almost on their "
              "own; the share of host-only URLs does not. The two prefixes reach the models through tokens "
              "(`http:`, `https:`, whole hostnames) and through the structural features that measure length "
              "and count dots and host labels.",
              ""]

    with open('sampling_results/dataset_split.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    with open('sampling_results/dataset_split_report.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("Saved: dataset_split_report.md / .json")
    print(f"  host overlap: {res['test_rows_host_in_train_pct']}% | 2-label domain overlap: "
          f"{res['test_rows_registrable_in_train_pct']}%")
    for name, p in prefixes.items():
        print(f"  {name}: scheme+www. {p['scheme_www_prefix_pct']}% | schemes {p['schemes_pct']}")


if __name__ == '__main__':
    main()
