"""Audit the frozen external lists against external_500/PROTOCOL.md (read-only)."""
import csv
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

ROOT = sys.argv[1]
FROZEN = os.path.join(ROOT, 'external_500', 'frozen')
WHITELIST = ['google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'github.com', 'stackoverflow.com',
             'microsoft.com', 'apple.com', 'amazon.com', 'wikipedia.org', 'linkedin.com', 'instagram.com']


def rows(name):
    with open(os.path.join(FROZEN, name), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def host(u):
    try:
        return (urlparse(u).hostname or '').lower()
    except ValueError:
        return ''


def reg2(h):
    p = h.split('.')
    return '.'.join(p[-2:]) if len(p) >= 2 else h


def wl(h):
    return any(h == d or h.endswith('.' + d) for d in WHITELIST)


legit, phish, prefix = rows('legit_urls.csv'), rows('phishing_urls.csv'), rows('prefix_sensitivity.csv')

print('=== LEGIT ===')
print('rows', len(legit), '| domains', len({r['domain'] for r in legit}), '| headline', sum(r['in_headline'] == '1' for r in legit))
print('per band domains:', dict(Counter(r['band'] for r in legit if r['kind'] == 'landing')))
print('headline kind per band:', dict(Counter((r['band'], r['kind']) for r in legit if r['in_headline'] == '1')))
per_dom = Counter(r['domain'] for r in legit if r['in_headline'] == '1')
print('domains with exactly one headline row:', sum(v == 1 for v in per_dom.values()), 'of', len(per_dom))
print('http codes landing:', dict(Counter(r['http_code'] for r in legit if r['kind'] == 'landing')))
print('http codes deep   :', dict(Counter(r['http_code'] for r in legit if r['kind'] == 'deep').most_common(8)))
bu = [r['browser_url'] or r['url'] for r in legit]
print('scheme of scored URL:', dict(Counter(u.split('://')[0] for u in bu)))
print('scored host starts with www.:', sum(host(u).startswith('www.') for u in bu), '| landing only:',
      sum(host(r['browser_url']).startswith('www.') for r in legit if r['kind'] == 'landing'))
print('scored URL on a statically whitelisted host:', sum(wl(host(u)) for u in bu),
      [u for u in bu if wl(host(u))][:5])
off = [(r['domain'], r['kind'], r['browser_url']) for r in legit
       if reg2(host(r['browser_url'])) != reg2(host(next(x['browser_url'] for x in legit if x['domain'] == r['domain'] and x['kind'] == 'landing')))]
print('deep rows whose final URL left the landing site (2-label):', len(off), off[:4])
print('landing final host differs from Tranco domain (2-label):',
      sum(reg2(host(r['browser_url'])) != reg2(r['domain']) for r in legit if r['kind'] == 'landing'))
print('deep url != deep browser_url (redirected):', sum(r['url'] != r['browser_url'] for r in legit if r['kind'] == 'deep'))
print('scored URLs with a query string:', sum('?' in u for u in bu), '| non-ASCII:', sum(not u.isascii() for u in bu),
      '| with uppercase:', sum(u != u.lower() for u in bu), '| with fragment:', sum('#' in u for u in bu))
print('landing with non-root path:', sum(urlparse(r['browser_url']).path not in ('', '/') for r in legit if r['kind'] == 'landing'))
pt = {r['registrable'] for r in csv.DictReader(open(os.path.join(FROZEN, 'phishtank_domains.csv'), encoding='utf-8'))}
print('landing hosts whose 2-label domain is in the PhishTank feed:', sum(reg2(host(r['browser_url'])) in pt for r in legit if r['kind'] == 'landing'))
multi = [s for s in pt if s in ('co.uk', 'com.br', 'com.au', 'co.id', 'co.jp', 'com.cn', 'co.in', 'com.mx', 'github.io', 'web.app', 'pages.dev', 'blogspot.com')]
print('public-suffix-like entries inside the PhishTank exclusion set:', sorted(multi))
tld = Counter(reg2(r['domain']).split('.')[-1] for r in legit if r['kind'] == 'landing')
print('top TLDs in legit set:', tld.most_common(8))

print('\n=== PHISHING ===')
print('rows', len(phish), '| unique hosts', len({r['host'] for r in phish}), '| unique phish_id', len({r['phish_id'] for r in phish}))
print('max per suffix:', Counter(r['suffix'] for r in phish).most_common(5))
subs = [datetime.fromisoformat(r['submission_time']) for r in phish]
fetch = datetime(2026, 10, 10, 13, 18, 50, tzinfo=timezone.utc)
print('submission range:', min(subs).date(), '..', max(subs).date(), '| older than 60 d before fetch:',
      sum(s < fetch - timedelta(days=60) for s in subs))
print('http codes:', dict(Counter(r['http_code'] for r in phish).most_common()))
print('live flag:', sum(r['live'] == '1' for r in phish), '| seen host:', sum(r['seen_in_training_host'] == '1' for r in phish))
print('as-listed scheme:', dict(Counter(r['url'].split('://')[0].lower() for r in phish)))
print('browser scheme  :', dict(Counter(r['browser_url'].split('://')[0].lower() for r in phish)))
same = sum(r['url'] == r['browser_url'] for r in phish)
only_scheme = sum(r['url'] != r['browser_url'] and re.sub(r'^http://', 'https://', r['url'], flags=re.I) == r['browser_url'] for r in phish)
slash = sum(re.sub(r'^http://', 'https://', r['url'], flags=re.I) + '/' == r['browser_url'] for r in phish)
other = [(r['url'], r['browser_url'], r['http_code']) for r in phish
         if r['url'] != r['browser_url'] and re.sub(r'^http://', 'https://', r['url'], flags=re.I) not in (r['browser_url'], r['browser_url'].rstrip('/'))
         and re.sub(r'^http://', 'https://', r['url'], flags=re.I) + '/' != r['browser_url']]
print(f'url == browser_url: {same} | differs by scheme only: {only_scheme} | by scheme and/or added trailing slash: {slash} | other: {len(other)}')
for o in other[:12]:
    print('   ', o)
print('host changed between listed and browser:', sum(host(r['url']) != host(r['browser_url']) for r in phish))
print('browser URL on whitelisted host:', [r['browser_url'] for r in phish if wl(host(r['browser_url']))][:5])
print('non-ASCII browser URLs:', sum(not r['browser_url'].isascii() for r in phish), '| as-listed with fragment:',
      sum('#' in r['url'] for r in phish), '| browser with fragment:', sum('#' in r['browser_url'] for r in phish),
      '| uppercase in browser URL:', sum(r['browser_url'] != r['browser_url'].lower() for r in phish))
print('www. hosts (browser):', sum(host(r['browser_url']).startswith('www.') for r in phish),
      '| host-only path:', sum(urlparse(r['browser_url']).path in ('', '/') and not urlparse(r['browser_url']).query for r in phish))

train = set()
with open(os.path.join(ROOT, 'URL dataset.csv'), newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        u = (r.get('url') or '').strip().lower()
        if u:
            train.add(u)
train_hosts = {host(u if '://' in u else 'http://' + u) for u in train}
print('exact training matches (as-listed / browser):', sum(r['url'].lower() in train for r in phish),
      sum(r['browser_url'].lower() in train for r in phish))
print('phishing hosts in training (recomputed):', sum(r['host'] in train_hosts for r in phish))
print('legit scored URLs that are exact training rows:', sum(u.lower() in train for u in bu),
      '| legit landing hosts present in training:', sum(host(r['browser_url']) in train_hosts for r in legit if r['kind'] == 'landing'))

print('\n=== PREFIX ===')
print('rows', len(prefix), '| same domains as legit:', {r['domain'] for r in prefix} == {r['domain'] for r in legit},
      '| well-formed:', all(r['url_https'] == f"https://{r['domain']}/" and r['url_https_www'] == f"https://www.{r['domain']}/"
                            and r['url_http'] == f"http://{r['domain']}/" for r in prefix))
print('Tranco domains that already start with www.:', sum(r['domain'].startswith('www.') for r in prefix))
