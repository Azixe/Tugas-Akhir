"""Build the frozen PhishTank list for the external 500/500 evaluation.

Protocol: external_500/PROTOCOL.md §(b) and §(c). Steps:

1. Fetch the `online-valid` feed (curl), record provenance (timestamp, sha-256, rows).
2. Filters: online+verified, submitted within RECENCY_DAYS; widen by steps only if fewer
   than MIN_HOSTS unique hosts remain.
3. Drop exact URL matches with the training dataset; dedup by hostname (1 URL per host).
4. Cap: at most HOSTING_CAP entries sharing the same registrable (2-label) suffix.
5. Seeded sample of PHISH_TARGET hosts (seed 42).
6. Resolve each entry with curl (https upgrade -> final URL + HTTP code) = browser variant.
7. Freeze: frozen/phishing_urls.csv + frozen/phishtank_domains.csv (all feed domains,
   for the Tranco eligibility check) + provenance entry.

Usage (repo root):
    python external_500/build_phishtank_list.py [--target 500] [--resolve-max 0]

--resolve-max caps how many entries get the (slow) curl resolution, for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEED_URL = 'https://data.phishtank.com/data/online-valid.csv'
RAW_DIR = os.path.join(ROOT, 'external_500', 'raw')
FROZEN = os.path.join(ROOT, 'external_500', 'frozen')
DATASET = os.path.join(ROOT, 'URL dataset.csv')

SEED = 42
RECENCY_DAYS = 60
RECENCY_STEPS = [60, 90, 120, 180]
MIN_HOSTS = 1500
HOSTING_CAP = 50
PHISH_TARGET = 500
CURL = 'curl'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36'


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def host_of(url: str) -> str:
    s = str(url).strip().lower()
    if '://' not in s:
        s = 'http://' + s
    try:
        h = urlparse(s).hostname or ''
    except ValueError:
        h = ''
    return h


def registrable(host: str) -> str:
    parts = host.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else host


def curl_fetch(url: str, out_path: str, max_time: int = 60) -> tuple[int, str]:
    """Fetch url to out_path; return (http_code, final_url)."""
    cmd = [CURL, '-sS', '-L', '--max-time', str(max_time), '--connect-timeout', '10',
           '-A', UA, '-o', out_path, '-w', '%{http_code}\t%{url_effective}', url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=max_time + 20)
        code, _, eff = r.stdout.partition('\t')
        return int(code or 0), eff.strip()
    except Exception:
        return 0, ''


def resolve_phishing(url: str, max_time: int = 20) -> tuple[str, int]:
    """https-upgrade then follow redirects; returns (final_url, http_code)."""
    candidate = re.sub(r'^http://', 'https://', url, flags=re.I)
    code, final = curl_fetch(candidate, os.devnull, max_time)
    return (final or candidate), code


def main():
    ap = argparse.ArgumentParser(description="Freeze the PhishTank list (protocol §b).")
    ap.add_argument('--target', type=int, default=PHISH_TARGET)
    ap.add_argument('--resolve-max', type=int, default=0, help='smoke test: resolve only N')
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FROZEN, exist_ok=True)

    raw_path = os.path.join(RAW_DIR, 'phishtank_online_valid.csv')
    fetched_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    print(f"fetching {FEED_URL} ...", flush=True)
    code, _ = curl_fetch(FEED_URL, raw_path, max_time=120)
    if code != 200:
        raise SystemExit(f"feed download failed (HTTP {code})")
    digest = sha256(raw_path)
    print(f"  saved {raw_path} ({os.path.getsize(raw_path)/1e6:.1f} MB, sha256 {digest[:16]}…)")

    # ---- parse ----
    rows = list(csv.DictReader(open(raw_path, newline='', encoding='utf-8')))
    print(f"  feed rows: {len(rows)}")

    # all feed domains (for Tranco eligibility)
    feed_hosts = set()
    for r in rows:
        h = host_of(r.get('url', ''))
        if h:
            feed_hosts.add(h)
    with open(os.path.join(FROZEN, 'phishtank_domains.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['host', 'registrable'])
        for h in sorted(feed_hosts):
            w.writerow([h, registrable(h)])
    print(f"  unique feed hosts written: {len(feed_hosts)}")

    # ---- training URLs ----
    train_urls = set()
    with open(DATASET, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            u = (r.get('url') or '').strip().lower()
            if u:
                train_urls.add(u)
    print(f"  training URLs loaded: {len(train_urls)}")

    # ---- recency widening ----
    now = datetime.now(timezone.utc)
    selected = None
    used_days = None
    for days in RECENCY_STEPS:
        cutoff = now - timedelta(days=days)
        pool = []
        seen_hosts = set()
        for r in rows:
            if (r.get('online') or '').strip().lower() != 'yes':
                continue
            if (r.get('verified') or '').strip().lower() != 'yes':
                continue
            try:
                sub = datetime.fromisoformat((r.get('submission_time') or '').replace('Z', '+00:00'))
            except ValueError:
                continue
            if sub < cutoff:
                continue
            url = (r.get('url') or '').strip()
            if url.lower() in train_urls:          # exact-URL training match
                continue
            h = host_of(url)
            if not h or h in seen_hosts:           # one URL per host
                continue
            seen_hosts.add(h)
            pool.append({'phish_id': r.get('phish_id'), 'url': url, 'host': h,
                         'suffix': registrable(h), 'submission_time': sub.isoformat(timespec='seconds'),
                         'target': (r.get('target') or '').strip()})
        print(f"  recency {days}d: {len(pool)} unique-host candidates")
        if len(pool) >= MIN_HOSTS or days == RECENCY_STEPS[-1]:
            selected, used_days = pool, days
            break

    # ---- seeded sample with hosting cap ----
    rng = random.Random(SEED)
    rng.shuffle(selected)
    taken, per_suffix = [], {}
    for row in selected:
        s = row['suffix']
        if per_suffix.get(s, 0) >= HOSTING_CAP:
            continue
        per_suffix[s] = per_suffix.get(s, 0) + 1
        taken.append(row)
        if len(taken) >= args.target:
            break
    print(f"  sampled: {len(taken)} (recency {used_days}d, cap {HOSTING_CAP}/suffix, "
          f"{len(per_suffix)} suffixes)")

    # ---- resolve (browser variant) ----
    limit = args.resolve_max or len(taken)
    out_rows = []
    t0 = time.time()
    for i, row in enumerate(taken):
        if i < limit:
            browser_url, http_code = resolve_phishing(row['url'])
        else:
            browser_url, http_code = '', 0
        out_rows.append({**row, 'browser_url': browser_url, 'http_code': http_code})
        if (i + 1) % 25 == 0 or (i + 1) == len(taken):
            print(f"  resolved {i+1}/{len(taken)} ({time.time()-t0:.0f}s)", flush=True)

    # host-level training flag
    train_hosts = {host_of(u) for u in train_urls}
    for row in out_rows:
        row['seen_in_training_host'] = int(row['host'] in train_hosts)
        row['live'] = int(200 <= row['http_code'] < 400)

    out_path = os.path.join(FROZEN, 'phishing_urls.csv')
    cols = ['phish_id', 'url', 'browser_url', 'http_code', 'host', 'suffix',
            'seen_in_training_host', 'live', 'submission_time', 'target']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row[k] for k in cols})
    print(f"saved {out_path} ({len(out_rows)} rows)")

    prov = os.path.join(FROZEN, 'PROVENANCE.md')
    with open(prov, 'a', encoding='utf-8') as f:
        f.write(f"\n## PhishTank feed (fetched {fetched_at})\n\n"
                f"- URL: {FEED_URL}\n- rows in feed: {len(rows)}\n- sha256: `{digest}`\n"
                f"- filters: online+verified, recency {used_days}d, exact-URL minus training, "
                f"1 per host, cap {HOSTING_CAP} per suffix\n"
                f"- sampled: {len(out_rows)} hosts, seed {SEED}, min-hosts trigger {MIN_HOSTS}\n"
                f"- resolution: {sum(1 for r in out_rows if r['http_code'])} of {len(out_rows)} "
                f"attempted answered (non-zero HTTP code); "
                f"{sum(1 for r in out_rows if r['live'])} live (2xx/3xx)\n")
    print(f"provenance appended: {prov}")


if __name__ == '__main__':
    main()
