"""Build the frozen Tranco legit list for the external 500/500 evaluation.

Protocol: external_500/PROTOCOL.md §(a). Steps:

1. Fetch the latest Tranco list (API -> list_id/download URL), record provenance.
2. Five rank bands of 100 domains each (seed 42): 1-1k, 1k-10k, 10k-100k, 100k-500k, 500k-1M.
3. Per band: draw ~POOL domains in seeded order; keep the first PER_BAND that pass:
   - answers over https with an HTML page (curl -L);
   - not in the extension's static whitelist (`WHITELIST` in utils.js, mirrored);
   - registrable domain not in the PhishTank feed (frozen/phishtank_domains.csv).
4. Two real URLs per passing domain: *landing* = final URL after redirects; *deep* = one
   seeded same-site link (non-empty path) from the landing HTML, resolved with curl.
5. Headline: 500 URLs, one per domain, exactly 50/50 landing/deep per band (seeded shuffle
   of the 100 domains, first 50 -> landing, last 50 -> deep).
6. Prefix-sensitivity list: https://d/, https://www.d/, http://d/ for the 500 headline domains.
7. Freeze: frozen/legit_urls.csv, frozen/prefix_sensitivity.csv, PROVENANCE.md.

Usage (repo root):
    python external_500/build_legit_list.py                     # full run (~30-50 min)
    python external_500/build_legit_list.py --per-band 2 --pool 4 --tag _smoke
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
import tempfile
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAW_DIR = os.path.join(ROOT, 'external_500', 'raw')
FROZEN = os.path.join(ROOT, 'external_500', 'frozen')
SEED = 42
CURL = 'curl'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36'
BANDS = [('1-1k', 1, 1000), ('1k-10k', 1001, 10000), ('10k-100k', 10001, 100000),
         ('100k-500k', 100001, 500000), ('500k-1M', 500001, 1000000)]
WHITELIST = ['google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'github.com',
             'stackoverflow.com', 'microsoft.com', 'apple.com', 'amazon.com',
             'wikipedia.org', 'linkedin.com', 'instagram.com']


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_url(url: str) -> str:
    """Drop a redundant default port (browsers normalise these away)."""
    return re.sub(r'^(https://[^/:]+):443(?=/|$)', r'\1', url, flags=re.I)\
            .replace(':80/', '/', 1) if url.startswith('http://') else \
        re.sub(r'^(https://[^/:]+):443(?=/|$)', r'\1', url, flags=re.I)


def registrable(host: str) -> str:
    parts = (host or '').lower().split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else host


def whitelisted(host: str) -> bool:
    host = (host or '').lower()
    return any(host == d or host.endswith('.' + d) for d in WHITELIST)


def curl_fetch(url: str, out_path: str | None = None, max_time: int = 25) -> tuple[int, str, str]:
    """Fetch url; returns (http_code, content_type, final_url)."""
    cmd = [CURL, '-sS', '-L', '--max-time', str(max_time), '--connect-timeout', '10', '-A', UA]
    cmd += ['-o', out_path or os.devnull]
    cmd += ['-w', '%{http_code}\t%{content_type}\t%{url_effective}', url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=max_time + 20)
        code, ctype, eff = (r.stdout.split('\t') + ['', ''])[:3]
        return int(code or 0), ctype.strip(), eff.strip()
    except Exception:
        return 0, '', ''


HREF_RE = re.compile(r'href=["\']([^"\'#]+)["\']', re.I)
ASSET_EXT = {'.css', '.js', '.mjs', '.cjs', '.map', '.png', '.jpg', '.jpeg', '.gif', '.svg',
             '.ico', '.webp', '.avif', '.bmp', '.woff', '.woff2', '.ttf', '.otf', '.eot',
             '.mp4', '.webm', '.mp3', '.wav', '.ogg', '.zip', '.rar', '.7z', '.gz', '.exe',
             '.dmg', '.apk'}


def deep_links(html_path: str, base_url: str, registrable_domain: str) -> list[str]:
    """Same-site links with a non-empty path, excluding the landing page itself."""
    try:
        html = open(html_path, encoding='utf-8', errors='ignore').read()
    except OSError:
        return []
    base = base_url.rstrip('/')
    out = []
    for m in HREF_RE.finditer(html):
        href = m.group(1).strip()
        if href.lower().startswith(('mailto:', 'javascript:', 'tel:', 'data:')):
            continue
        try:
            u = urljoin(base_url, href)
            p = urlparse(u)
        except ValueError:
            continue
        if p.scheme != 'https' or not p.hostname:
            continue
        if registrable(p.hostname) != registrable_domain:
            continue
        path = p.path or ''
        if path in ('', '/') or path == urlparse(base_url).path.rstrip('/') + '/':
            continue
        clean = f"{p.scheme}://{p.netloc}{p.path}"
        if clean.rstrip('/') == base:
            continue
        if os.path.splitext(p.path.lower())[1] in ASSET_EXT:
            continue
        out.append(clean)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="Freeze the Tranco legit list (protocol §a).")
    ap.add_argument('--per-band', type=int, default=100)
    ap.add_argument('--pool', type=int, default=300, help='candidates drawn per band')
    ap.add_argument('--tag', default='', help="suffix for output files (smoke tests)")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FROZEN, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec='seconds')

    # ---- fetch Tranco ----
    api_path = os.path.join(RAW_DIR, 'tranco_api.json')
    code, _, _ = curl_fetch('https://tranco-list.eu/api/lists/date/latest', api_path, 60)
    info = json.load(open(api_path, encoding='utf-8'))
    list_id, download = info['list_id'], info['download']
    raw_path = os.path.join(RAW_DIR, f'tranco_{list_id}.csv')
    print(f"Tranco list {list_id} (created {info.get('created_on')}) -> {download}")
    code, _, _ = curl_fetch(download, raw_path, 180)
    if code != 200:
        raise SystemExit(f"Tranco download failed (HTTP {code})")
    digest = sha256(raw_path)
    print(f"  saved {raw_path} ({os.path.getsize(raw_path)/1e6:.1f} MB, sha256 {digest[:16]}…)")

    domains = []
    with open(raw_path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].isdigit():
                domains.append((int(row[0]), row[1].strip().lower()))
    print(f"  parsed {len(domains)} ranked domains")

    phishtank_regs = set()
    pt_path = os.path.join(FROZEN, 'phishtank_domains.csv')
    if os.path.exists(pt_path):
        with open(pt_path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                phishtank_regs.add(r['registrable'])
    print(f"  PhishTank registrable domains for exclusion: {len(phishtank_regs)}")

    rng = random.Random(SEED)
    rows, headline_domains = [], []
    tmpdir = tempfile.mkdtemp(prefix='tranco_body_')
    t0 = time.time()

    for band_label, lo, hi in BANDS:
        band_domains = [d for r, d in domains if lo <= r <= hi]
        rng.shuffle(band_domains)
        pool = band_domains[:args.pool]
        passed, drops = [], {'fetch': 0, 'whitelist': 0, 'phishtank': 0, 'no_deep': 0}
        for d in pool:
            if len(passed) >= args.per_band:
                break
            body = os.path.join(tmpdir, f"{d}.html")
            code, ctype, final = curl_fetch(f"https://{d}/", body)
            if code != 200 or 'text/html' not in ctype or not final.lower().startswith('https://'):
                drops['fetch'] += 1
                continue
            fhost = (urlparse(final).hostname or '').lower()
            if whitelisted(fhost):
                drops['whitelist'] += 1
                continue
            if registrable(fhost) in phishtank_regs:
                drops['phishtank'] += 1
                continue
            links = deep_links(body, final, registrable(fhost))
            if not links:
                drops['no_deep'] += 1
                continue
            deep = rng.choice(links)
            dcode, _, dfinal = curl_fetch(deep, None, 25)
            passed.append({'domain': d, 'band': band_label, 'host': fhost,
                           'landing': final, 'landing_code': code,
                           'deep': deep, 'deep_final': dfinal or deep, 'deep_code': dcode})
            print(f"  [{band_label}] {len(passed)}/{args.per_band} {d}", flush=True)
        print(f"  [{band_label}] done: {len(passed)} pass | drops {drops} "
              f"({time.time()-t0:.0f}s)", flush=True)

        # headline: seeded 50/50 landing/deep per band
        shuffle = list(passed)
        rng.shuffle(shuffle)
        half = len(shuffle) // 2
        for i, p in enumerate(shuffle):
            kind = 'landing' if i < half else 'deep'
            rows.append({'domain': p['domain'], 'band': band_label, 'kind': 'landing',
                         'url': clean_url(p['landing']), 'browser_url': clean_url(p['landing']),
                         'http_code': p['landing_code'], 'in_headline': int(kind == 'landing')})
            rows.append({'domain': p['domain'], 'band': band_label, 'kind': 'deep',
                         'url': clean_url(p['deep']), 'browser_url': clean_url(p['deep_final']),
                         'http_code': p['deep_code'], 'in_headline': int(kind == 'deep')})
            headline_domains.append({'domain': p['domain'], 'band': band_label,
                                     'url_https': f"https://{p['domain']}/",
                                     'url_https_www': f"https://www.{p['domain']}/",
                                     'url_http': f"http://{p['domain']}/"})

    suffix = args.tag
    legit_path = os.path.join(FROZEN, f'legit_urls{suffix}.csv')
    with open(legit_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['domain', 'band', 'kind', 'url', 'browser_url',
                                          'http_code', 'in_headline'])
        w.writeheader()
        w.writerows(rows)
    head_path = os.path.join(FROZEN, f'prefix_sensitivity{suffix}.csv')
    with open(head_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['domain', 'band', 'url_https', 'url_https_www', 'url_http'])
        w.writeheader()
        w.writerows(headline_domains)
    print(f"saved {legit_path} ({len(rows)} rows, {sum(r['in_headline'] for r in rows)} headline)")
    print(f"saved {head_path} ({len(headline_domains)} domains)")

    if not args.tag:
        with open(os.path.join(FROZEN, 'PROVENANCE.md'), 'a', encoding='utf-8') as f:
            f.write(f"\n## Tranco list (fetched {fetched_at})\n\n"
                    f"- list id: `{list_id}` (created {info.get('created_on')})\n"
                    f"- download: {download}\n- sha256: `{digest}`\n"
                    f"- bands: {', '.join(b[0] for b in BANDS)} x {args.per_band} domains, "
                    f"pool {args.pool}/band, seed {SEED}\n"
                    f"- frozen rows: {len(rows)} ({sum(r['in_headline'] for r in rows)} headline)\n")


if __name__ == '__main__':
    main()
