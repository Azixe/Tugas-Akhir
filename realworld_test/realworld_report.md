# Real-World Validation — Live PhishTank + Legitimate URLs

- Phishing source: PhishTank `online-valid` (76,683 URLs with unique domains), sampled 25 across unique domains, seed 42, collected 2026-09-20
- Legitimate: 20 manual URLs (FP-prone sites: Steam, Reddit, Discord, ...)
- Extension model version: v3.1  |  RF sha256 `625c9333151667c4…`  |  XGB sha256 `daa8c8d49a200ac3…`
- URL-only analysis (same input the extension sees); liveness does not affect scoring
- Training set (Mendeley 2024): 2.6% of phishing URLs are on free-hosting platforms vs 16/25 (64%) in this live sample.

## Summary

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 51.11% | 64.44% |
| Precision (phishing) | 100.00% | 84.62% |
| Recall (phishing) | 12.00% | 44.00% |
| F1 (phishing) | 21.43% | 57.89% |
| False Positive Rate | 0.00% | 10.00% |
| TP / FP / TN / FN | 3 / 0 / 20 / 22 | 11 / 2 / 18 / 14 |

RF and XGBoost agree on 77.8% of URLs.

## Phishing sample by domain type

| Domain type | n | RF caught | XGBoost caught |
|---|---:|---:|---:|
| free-hosting | 16 | 3/16 | 8/16 |
| owned-domain | 9 | 0/9 | 3/9 |

## Extension behavior (decision thresholds)

| Model | Blocked (>80%) | Warned (60-80%) | SAFE (<60%) |
|---|---|---|---|
| Random Forest | 0 (0 TP / 0 FP) | 1 | 44 |
| XGBoost | 11 (9 TP / 2 FP) | 1 | 33 |

## Per-URL results

Format: ✓/✗ = raw model label vs expected; the word shown is the extension verdict at its thresholds (>80% PHISHING, 60–80% SUSPICIOUS, <60% SAFE).

| # | Expected | URL | RF | XGB |
|---:|---|---|---|---|
| 1 | phish | https://allegrolokalnie.oferta839174.click/milwaukee-m18-fuel-szlifierka-/14759 | ✗ SAFE (7.2%) | ✗ SAFE (0.0%) |
| 2 | phish | https://bafkreicsgdat4hmyhxfwmqjpak66skttjj4o2d2bennbpjwfmfi4ukl5fa.ipfs.dweb.link/ | ✗ SAFE (9.1%) | ✓ SAFE (51.0%) |
| 3 | phish | http://bntp3725nhw-yfhcnfyn-7d5e0f-xk266a.pages.dev/ | ✓ SAFE (56.0%) | ✓ PHISHING (100.0%) |
| 4 | phish | https://cdcinforma001020.web.app/ | ✗ SAFE (12.4%) | ✓ PHISHING (97.3%) |
| 5 | phish | https://choisir-horaire.com/ | ✗ SAFE (44.3%) | ✗ SAFE (4.2%) |
| 6 | phish | https://comcast-xfiintybkjhm.weeblysite.com/ | ✗ SAFE (8.1%) | ✗ SAFE (0.0%) |
| 7 | phish | http://crypto-ah4.netlify.app/ | ✓ SUSPICIOUS (61.9%) | ✓ PHISHING (100.0%) |
| 8 | phish | https://discreet-mountain-005558.framer.app/ | ✗ SAFE (9.6%) | ✗ SAFE (34.7%) |
| 9 | phish | https://germanshepherddatabase.org/pp_pedigree.php?id=%22%2F%3E%3Cimg%20src%3D%22https%3A%2F%2Fgoogle.com%2FMGSwmyb3v2ATI.jpg%22%20onerror%3D%22window.location%3DdecodeURIComponent%28atob%28%27Njg3NDc0NzA3MzNhMmYyZjY2NjI3YTY2NmY2ZDJlNjM2MTM2NjEzODYyMzczNDJkMzMzMjYxNjMyZDM0MzU2NTYxMmQzODM0NjYzMDJkNjY2MjM2MzAzNzYyMzgzNzM4Mzk2MTYyMmU2MzZjNjk2MzZiMmYzNjM0NjM2NDY1MzAzNDMwMmQzMjMyMzMzNTJkMzQzNTYyNjUyZDYxMzM2NjM2MmQzNDY0NjM2NDM2NjMzODM4MzY2NDYyNjUyZTcwNjg3MA%3D%3D%27%29.replace%28%2F%28..%29%2Fg%2C%20%27%25%241%27%29%29%3B%22%3E | ✗ SAFE (39.5%) | ✓ PHISHING (95.6%) |
| 10 | phish | https://itdkbgit8.web.app/ | ✗ SAFE (12.5%) | ✓ PHISHING (98.8%) |
| 11 | phish | https://lecaikejiao.com/uid_4673027932?ticket=ofuT1ALEqM | ✗ SAFE (46.9%) | ✗ SAFE (2.6%) |
| 12 | phish | https://live-desktophelp.wixstudio.com/en-us | ✗ SAFE (7.1%) | ✗ SAFE (3.2%) |
| 13 | phish | https://lolafry11.wixsite.com/my-site | ✗ SAFE (20.3%) | ✗ SAFE (42.0%) |
| 14 | phish | http://norzeta-gld-fentela-r5t2hp76.pages.dev/ | ✓ SAFE (56.0%) | ✓ PHISHING (100.0%) |
| 15 | phish | https://official-rabbycdn.wixstudio.com/us-en | ✗ SAFE (7.1%) | ✗ SAFE (5.2%) |
| 16 | phish | https://pay-interbank.webcindario.com/ | ✗ SAFE (8.1%) | ✗ SAFE (0.0%) |
| 17 | phish | https://pub-fbcc9cd30d2f4f4793500c39a15307b1.r2.dev/link.html | ✗ SAFE (20.1%) | ✓ SUSPICIOUS (67.3%) |
| 18 | phish | https://settallupservices.com/ | ✗ SAFE (44.2%) | ✗ SAFE (5.4%) |
| 19 | phish | https://shodbj.com/view/radfTprBTd | ✗ SAFE (48.0%) | ✗ SAFE (32.8%) |
| 20 | phish | https://siggnonnatto-mygovv.web.app/ | ✗ SAFE (9.8%) | ✓ PHISHING (97.5%) |
| 21 | phish | https://viaverde-seguranca.com/steps/billing.php | ✗ SAFE (38.5%) | ✓ PHISHING (99.2%) |
| 22 | phish | https://web.tegtdi.top | ✗ SAFE (12.3%) | ✗ SAFE (39.8%) |
| 23 | phish | https://webservice000auth-xfinity.weebly.com/ | ✗ SAFE (8.7%) | ✗ SAFE (0.0%) |
| 24 | phish | https://www.auberge-lorraine-levaltin.fr/webspace/portal/clients/login.php?verification#_login&amp;appIdKey=91bfa3054d6407b&amp;country=RO | ✗ SAFE (42.8%) | ✓ PHISHING (99.3%) |
| 25 | phish | https://xfinitymail2026.weebly.com/ | ✗ SAFE (9.0%) | ✗ SAFE (0.0%) |
| 26 | legit | https://steamcommunity.com/ | ✓ SAFE (44.3%) | ✓ SAFE (5.0%) |
| 27 | legit | https://steamcommunity.com/id/WhyIzMyLifeLikeDiz/ | ✓ SAFE (47.6%) | ✗ PHISHING (89.9%) |
| 28 | legit | https://steamcommunity.com/market/listings/730/AK-47%20%7C%20Redline%20%28Field-Tested%29 | ✓ SAFE (45.8%) | ✗ PHISHING (94.6%) |
| 29 | legit | https://www.reddit.com/r/programming/ | ✓ SAFE (8.8%) | ✓ SAFE (0.0%) |
| 30 | legit | https://www.reddit.com/r/ProgrammerHumor/comments/1f8k2zp/some_very_long_post_title_with_many_words_and/ | ✓ SAFE (10.2%) | ✓ SAFE (0.0%) |
| 31 | legit | https://discord.com/ | ✓ SAFE (44.3%) | ✓ SAFE (1.4%) |
| 32 | legit | https://www.kaskus.co.id/ | ✓ SAFE (12.6%) | ✓ SAFE (6.9%) |
| 33 | legit | https://www.tokopedia.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) |
| 34 | legit | https://www.detik.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) |
| 35 | legit | https://www.kompas.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) |
| 36 | legit | https://open.spotify.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.1%) |
| 37 | legit | https://www.figma.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) |
| 38 | legit | https://gitlab.com/explore | ✓ SAFE (46.2%) | ✓ SAFE (0.0%) |
| 39 | legit | https://www.npmjs.com/package/react | ✓ SAFE (9.1%) | ✓ SAFE (0.0%) |
| 40 | legit | https://docs.python.org/3/library/csv.html | ✓ SAFE (8.3%) | ✓ SAFE (0.1%) |
| 41 | legit | https://arxiv.org/abs/2409.19825 | ✓ SAFE (41.2%) | ✓ SAFE (10.0%) |
| 42 | legit | https://www.bbc.com/news | ✓ SAFE (11.2%) | ✓ SAFE (0.0%) |
| 43 | legit | https://www.cloudflare.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) |
| 44 | legit | https://web.whatsapp.com/ | ✓ SAFE (9.9%) | ✓ SAFE (2.2%) |
| 45 | legit | https://www.tiktok.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) |

## Errors

**Random Forest** — false positives: 0, false negatives: 22
- FN: https://allegrolokalnie.oferta839174.click/milwaukee-m18-fuel-szlifierka-/14759
- FN: https://bafkreicsgdat4hmyhxfwmqjpak66skttjj4o2d2bennbpjwfmfi4ukl5fa.ipfs.dweb.link/
- FN: https://cdcinforma001020.web.app/
- FN: https://choisir-horaire.com/
- FN: https://comcast-xfiintybkjhm.weeblysite.com/
- FN: https://discreet-mountain-005558.framer.app/
- FN: https://germanshepherddatabase.org/pp_pedigree.php?id=%22%2F%3E%3Cimg%20src%3D%22https%3A%2F%2Fgoogle.com%2FMGSwmyb3v2ATI.jpg%22%20onerror%3D%22window.location%3DdecodeURIComponent%28atob%28%27Njg3NDc0NzA3MzNhMmYyZjY2NjI3YTY2NmY2ZDJlNjM2MTM2NjEzODYyMzczNDJkMzMzMjYxNjMyZDM0MzU2NTYxMmQzODM0NjYzMDJkNjY2MjM2MzAzNzYyMzgzNzM4Mzk2MTYyMmU2MzZjNjk2MzZiMmYzNjM0NjM2NDY1MzAzNDMwMmQzMjMyMzMzNTJkMzQzNTYyNjUyZDYxMzM2NjM2MmQzNDY0NjM2NDM2NjMzODM4MzY2NDYyNjUyZTcwNjg3MA%3D%3D%27%29.replace%28%2F%28..%29%2Fg%2C%20%27%25%241%27%29%29%3B%22%3E
- FN: https://itdkbgit8.web.app/
- FN: https://lecaikejiao.com/uid_4673027932?ticket=ofuT1ALEqM
- FN: https://live-desktophelp.wixstudio.com/en-us
- FN: https://lolafry11.wixsite.com/my-site
- FN: https://official-rabbycdn.wixstudio.com/us-en
- FN: https://pay-interbank.webcindario.com/
- FN: https://pub-fbcc9cd30d2f4f4793500c39a15307b1.r2.dev/link.html
- FN: https://settallupservices.com/
- FN: https://shodbj.com/view/radfTprBTd
- FN: https://siggnonnatto-mygovv.web.app/
- FN: https://viaverde-seguranca.com/steps/billing.php
- FN: https://web.tegtdi.top
- FN: https://webservice000auth-xfinity.weebly.com/
- FN: https://www.auberge-lorraine-levaltin.fr/webspace/portal/clients/login.php?verification#_login&amp;appIdKey=91bfa3054d6407b&amp;country=RO
- FN: https://xfinitymail2026.weebly.com/

**XGBoost** — false positives: 2, false negatives: 14
- FP: https://steamcommunity.com/id/WhyIzMyLifeLikeDiz/
- FP: https://steamcommunity.com/market/listings/730/AK-47%20%7C%20Redline%20%28Field-Tested%29
- FN: https://allegrolokalnie.oferta839174.click/milwaukee-m18-fuel-szlifierka-/14759
- FN: https://choisir-horaire.com/
- FN: https://comcast-xfiintybkjhm.weeblysite.com/
- FN: https://discreet-mountain-005558.framer.app/
- FN: https://lecaikejiao.com/uid_4673027932?ticket=ofuT1ALEqM
- FN: https://live-desktophelp.wixstudio.com/en-us
- FN: https://lolafry11.wixsite.com/my-site
- FN: https://official-rabbycdn.wixstudio.com/us-en
- FN: https://pay-interbank.webcindario.com/
- FN: https://settallupservices.com/
- FN: https://shodbj.com/view/radfTprBTd
- FN: https://web.tegtdi.top
- FN: https://webservice000auth-xfinity.weebly.com/
- FN: https://xfinitymail2026.weebly.com/

> Thresholds mirror the extension: >80% PHISHING, 60–80% SUSPICIOUS, <60% SAFE.
