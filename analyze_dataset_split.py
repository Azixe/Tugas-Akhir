"""Quantify train/test host overlap on the shared protocol (limitation analysis).

A random row-level 80/20 split can place URLs from the *same host* (or the same
site infrastructure) on both sides of the split, which inflates in-distribution
metrics. This script measures how often that happens in the 450,176-URL dataset.

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

    with open('sampling_results/dataset_split.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    with open('sampling_results/dataset_split_report.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("Saved: dataset_split_report.md / .json")
    print(f"  host overlap: {res['test_rows_host_in_train_pct']}% | 2-label domain overlap: "
          f"{res['test_rows_registrable_in_train_pct']}%")


if __name__ == '__main__':
    main()
