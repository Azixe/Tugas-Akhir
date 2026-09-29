# Real-World Validation — Live PhishTank + Legitimate URLs

- Phishing source: PhishTank `online-valid` (76,683 URLs with unique domains), sampled 25 across unique domains, seed 42, collected 2026-09-20
- Legitimate: 25 manual URLs (FP-prone sites: Steam, Reddit, Discord, ...)
- Extension model version: v3.4  |  RF sha256 `625c9333151667c4…`  |  XGB sha256 `4efd7ead699770a0…`  |  CatBoost sha256 `791a503e72426426…`
- Two variants scored per URL: **as-listed** (raw feed string) and **browser** (http→https upgrade + curl-resolved redirects; 5 URLs differ). The browser variant is primary.
- URL-only analysis (same input the extension sees); liveness does not affect scoring
- Training set (Mendeley 2024): phishing 93.8% http / 6.2% https, legitimate 0.0% http / 100.0% https — the scheme alone separates ~94% of the data
- RF's top-2 features by importance: `https:` (23.4%) and `http:` (20.4%) — a dataset shortcut now obsolete (modern phishing is https)

## Summary — browser-observed (primary)

| Metric | Random Forest | XGBoost | CatBoost |
|---|---:|---:|---:|
| Accuracy | 50.00% | 56.00% | 52.00% |
| Precision (phishing) | 0.00% | 63.64% | 54.55% |
| Recall (phishing) | 0.00% | 28.00% | 24.00% |
| F1 (phishing) | 0.00% | 38.89% | 33.33% |
| False Positive Rate | 0.00% | 16.00% | 20.00% |
| TP / FP / TN / FN | 0 / 0 / 25 / 25 | 7 / 4 / 21 / 18 | 6 / 5 / 20 / 19 |

Model agreement (browser variant): RF↔XGB 78.0% | RF↔CatBoost 78.0% | XGB↔CatBoost 80.0%.

## Summary — as-listed (reference)

| Metric | Random Forest | XGBoost | CatBoost |
|---|---:|---:|---:|
| Accuracy | 56.00% | 62.00% | 58.00% |
| Recall (phishing) | 12.00% | 40.00% | 36.00% |
| False Positive Rate | 0.00% | 16.00% | 20.00% |
| TP / FP / TN / FN | 3 / 0 / 25 / 22 | 10 / 4 / 21 / 15 | 9 / 5 / 20 / 16 |

## Scheme sensitivity (why one letter flips the model)

| URL (as-listed) | RF as-listed | RF browser | XGB as-listed | XGB browser | CatBoost as-listed | CatBoost browser |
|---|---:|---:|---:|---:|---:|---:|
| http://bntp3725nhw-yfhcnfyn-7d5e0f-xk266a.pages.dev/ | 1 / 56.0% | 0 / 6.5% | 1 / 97.5% | 0 / 0.0% | 1 / 100.0% | 0 / 0.6% |
| http://crypto-ah4.netlify.app/ | 1 / 61.9% | 0 / 12.4% | 1 / 100.0% | 1 / 89.9% | 1 / 100.0% | 1 / 69.1% |
| http://norzeta-gld-fentela-r5t2hp76.pages.dev/ | 1 / 56.0% | 0 / 6.5% | 1 / 99.5% | 0 / 0.0% | 1 / 100.0% | 0 / 0.4% |
| https://web.tegtdi.top | 0 / 12.3% | 0 / 11.5% | 0 / 33.8% | 0 / 41.5% | 0 / 1.1% | 0 / 0.8% |
| https://www.auberge-lorraine-levaltin.fr/webspace/portal/clients/login.php?verification#_login&amp;appIdKey=91bfa3054d6407b&amp;country=RO | 0 / 42.8% | 0 / 8.0% | 1 / 97.6% | 0 / 2.5% | 1 / 79.6% | 0 / 0.4% |

## Phishing sample by domain type (browser variant)

| Domain type | n | RF caught | XGBoost caught | CatBoost caught |
|---|---:|---:|---:|---:|
| free-hosting | 16 | 0/16 | 5/16 | 2/16 |
| owned-domain | 9 | 0/9 | 2/9 | 4/9 |

## Extension behavior (decision thresholds, browser variant)

| Model | Blocked (>80%) | Warned (60-80%) | SAFE (<60%) |
|---|---|---|---|
| Random Forest | 0 (0 TP / 0 FP) | 0 | 50 |
| XGBoost | 8 (6 TP / 2 FP) | 2 | 40 |
| CatBoost | 9 (5 TP / 4 FP) | 1 | 40 |

## Per-URL results (browser variant)

Format: ✓/✗ = raw model label vs expected; the word is the extension verdict at its thresholds (>80% PHISHING, 60–80% SUSPICIOUS, <60% SAFE). Percentages are the model's phishing probability (skor), the same quantity the thresholds and the extension UI use.

| # | Expected | URL (browser) | RF | XGB | CatBoost |
|---:|---|---|---|---|---|
| 1 | phish | https://allegrolokalnie.oferta839174.click/milwaukee-m18-fuel-szlifierka-/14759 | ✗ SAFE (7.2%) | ✗ SAFE (0.1%) | ✗ SAFE (1.1%) |
| 2 | phish | https://bafkreicsgdat4hmyhxfwmqjpak66skttjj4o2d2bennbpjwfmfi4ukl5fa.ipfs.dweb.link/ | ✗ SAFE (9.1%) | ✗ SAFE (3.4%) | ✗ SAFE (1.2%) |
| 3 | phish | https://bntp3725nhw-yfhcnfyn-7d5e0f-xk266a.pages.dev/ | ✗ SAFE (6.5%) | ✗ SAFE (0.0%) | ✗ SAFE (0.6%) |
| 4 | phish | https://cdcinforma001020.web.app/ | ✗ SAFE (12.4%) | ✓ PHISHING (95.0%) | ✗ SAFE (14.5%) |
| 5 | phish | https://choisir-horaire.com/ | ✗ SAFE (44.3%) | ✗ SAFE (0.1%) | ✗ SAFE (40.1%) |
| 6 | phish | https://comcast-xfiintybkjhm.weeblysite.com/ | ✗ SAFE (8.1%) | ✗ SAFE (0.0%) | ✗ SAFE (0.9%) |
| 7 | phish | https://crypto-ah4.netlify.app/ | ✗ SAFE (12.4%) | ✓ PHISHING (89.9%) | ✓ SUSPICIOUS (69.1%) |
| 8 | phish | https://discreet-mountain-005558.framer.app/ | ✗ SAFE (9.6%) | ✓ SAFE (51.5%) | ✗ SAFE (44.2%) |
| 9 | phish | https://germanshepherddatabase.org/pp_pedigree.php?id=%22%2F%3E%3Cimg%20src%3D%22https%3A%2F%2Fgoogle.com%2FMGSwmyb3v2ATI.jpg%22%20onerror%3D%22window.location%3DdecodeURIComponent%28atob%28%27Njg3NDc0NzA3MzNhMmYyZjY2NjI3YTY2NmY2ZDJlNjM2MTM2NjEzODYyMzczNDJkMzMzMjYxNjMyZDM0MzU2NTYxMmQzODM0NjYzMDJkNjY2MjM2MzAzNzYyMzgzNzM4Mzk2MTYyMmU2MzZjNjk2MzZiMmYzNjM0NjM2NDY1MzAzNDMwMmQzMjMyMzMzNTJkMzQzNTYyNjUyZDYxMzM2NjM2MmQzNDY0NjM2NDM2NjMzODM4MzY2NDYyNjUyZTcwNjg3MA%3D%3D%27%29.replace%28%2F%28..%29%2Fg%2C%20%27%25%241%27%29%29%3B%22%3E | ✗ SAFE (39.5%) | ✓ PHISHING (95.6%) | ✓ PHISHING (99.7%) |
| 10 | phish | https://itdkbgit8.web.app/ | ✗ SAFE (12.5%) | ✓ PHISHING (99.2%) | ✗ SAFE (22.2%) |
| 11 | phish | https://lecaikejiao.com/uid_4673027932?ticket=ofuT1ALEqM | ✗ SAFE (46.9%) | ✗ SAFE (10.2%) | ✓ PHISHING (95.6%) |
| 12 | phish | https://live-desktophelp.wixstudio.com/en-us | ✗ SAFE (7.1%) | ✗ SAFE (2.3%) | ✗ SAFE (0.3%) |
| 13 | phish | https://lolafry11.wixsite.com/my-site | ✗ SAFE (20.3%) | ✗ SAFE (29.0%) | ✓ PHISHING (87.8%) |
| 14 | phish | https://norzeta-gld-fentela-r5t2hp76.pages.dev/ | ✗ SAFE (6.5%) | ✗ SAFE (0.0%) | ✗ SAFE (0.4%) |
| 15 | phish | https://official-rabbycdn.wixstudio.com/us-en | ✗ SAFE (7.1%) | ✗ SAFE (1.5%) | ✗ SAFE (0.3%) |
| 16 | phish | https://pay-interbank.webcindario.com/ | ✗ SAFE (8.1%) | ✗ SAFE (0.0%) | ✗ SAFE (0.9%) |
| 17 | phish | https://pub-fbcc9cd30d2f4f4793500c39a15307b1.r2.dev/link.html | ✗ SAFE (20.1%) | ✗ SAFE (24.9%) | ✗ SAFE (14.4%) |
| 18 | phish | https://settallupservices.com/ | ✗ SAFE (44.2%) | ✗ SAFE (0.1%) | ✗ SAFE (36.0%) |
| 19 | phish | https://shodbj.com/view/radfTprBTd | ✗ SAFE (48.0%) | ✗ SAFE (30.5%) | ✓ PHISHING (89.5%) |
| 20 | phish | https://siggnonnatto-mygovv.web.app/ | ✗ SAFE (9.8%) | ✓ PHISHING (97.8%) | ✗ SAFE (12.4%) |
| 21 | phish | https://viaverde-seguranca.com/steps/billing.php | ✗ SAFE (38.5%) | ✓ PHISHING (96.8%) | ✓ PHISHING (96.8%) |
| 22 | phish | https://web.tegtdi.top/ | ✗ SAFE (11.5%) | ✗ SAFE (41.5%) | ✗ SAFE (0.8%) |
| 23 | phish | https://webservice000auth-xfinity.weebly.com/ | ✗ SAFE (8.7%) | ✗ SAFE (0.0%) | ✗ SAFE (1.6%) |
| 24 | phish | https://www.auberge-lorraine-levaltin.fr/ | ✗ SAFE (8.0%) | ✗ SAFE (2.5%) | ✗ SAFE (0.4%) |
| 25 | phish | https://xfinitymail2026.weebly.com/ | ✗ SAFE (9.0%) | ✗ SAFE (0.0%) | ✗ SAFE (1.4%) |
| 26 | legit | https://steamcommunity.com/ | ✓ SAFE (44.3%) | ✓ SAFE (0.2%) | ✓ SAFE (34.4%) |
| 27 | legit | https://steamcommunity.com/id/WhyIzMyLifeLikeDiz/ | ✓ SAFE (47.6%) | ✗ SUSPICIOUS (77.6%) | ✗ PHISHING (98.5%) |
| 28 | legit | https://steamcommunity.com/market/listings/730/AK-47%20%7C%20Redline%20%28Field-Tested%29 | ✓ SAFE (45.8%) | ✗ PHISHING (99.6%) | ✗ PHISHING (93.7%) |
| 29 | legit | https://www.reddit.com/r/programming/ | ✓ SAFE (8.8%) | ✓ SAFE (0.0%) | ✓ SAFE (0.5%) |
| 30 | legit | https://www.reddit.com/r/ProgrammerHumor/comments/1f8k2zp/some_very_long_post_title_with_many_words_and/ | ✓ SAFE (10.2%) | ✓ SAFE (0.1%) | ✓ SAFE (1.4%) |
| 31 | legit | https://discord.com/ | ✓ SAFE (44.3%) | ✓ SAFE (0.2%) | ✓ SAFE (32.6%) |
| 32 | legit | https://www.kaskus.co.id/ | ✓ SAFE (12.6%) | ✓ SAFE (4.2%) | ✓ SAFE (0.8%) |
| 33 | legit | https://www.tokopedia.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) | ✓ SAFE (0.4%) |
| 34 | legit | https://www.detik.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) | ✓ SAFE (0.4%) |
| 35 | legit | https://www.kompas.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) | ✓ SAFE (0.4%) |
| 36 | legit | https://open.spotify.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.1%) | ✓ SAFE (0.5%) |
| 37 | legit | https://www.figma.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) | ✓ SAFE (0.4%) |
| 38 | legit | https://gitlab.com/explore | ✓ SAFE (46.2%) | ✓ SAFE (0.3%) | ✗ PHISHING (88.2%) |
| 39 | legit | https://www.npmjs.com/package/react | ✓ SAFE (9.1%) | ✓ SAFE (0.0%) | ✓ SAFE (0.4%) |
| 40 | legit | https://docs.python.org/3/library/csv.html | ✓ SAFE (8.3%) | ✓ SAFE (0.1%) | ✓ SAFE (0.5%) |
| 41 | legit | https://arxiv.org/abs/2409.19825 | ✓ SAFE (41.2%) | ✓ SAFE (12.1%) | ✗ PHISHING (91.1%) |
| 42 | legit | https://www.bbc.com/news | ✓ SAFE (11.2%) | ✓ SAFE (0.1%) | ✓ SAFE (0.5%) |
| 43 | legit | https://www.cloudflare.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) | ✓ SAFE (0.4%) |
| 44 | legit | https://web.whatsapp.com/ | ✓ SAFE (9.9%) | ✓ SAFE (3.7%) | ✓ SAFE (0.9%) |
| 45 | legit | https://www.tiktok.com/ | ✓ SAFE (9.0%) | ✓ SAFE (0.0%) | ✓ SAFE (0.4%) |
| 46 | legit | https://www.paypal.com/ | ✓ SAFE (26.6%) | ✗ PHISHING (99.3%) | ✗ SAFE (50.4%) |
| 47 | legit | https://www.bca.co.id/ | ✓ SAFE (12.6%) | ✓ SAFE (2.9%) | ✓ SAFE (0.8%) |
| 48 | legit | https://www.netflix.com/login | ✓ SAFE (35.3%) | ✗ SUSPICIOUS (68.5%) | ✓ SAFE (41.2%) |
| 49 | legit | https://www.lazada.co.id/ | ✓ SAFE (12.6%) | ✓ SAFE (4.2%) | ✓ SAFE (0.8%) |
| 50 | legit | https://www.bing.com/search?q=cara+cek+url+phishing | ✓ SAFE (8.7%) | ✓ SAFE (0.0%) | ✓ SAFE (0.4%) |

## Errors (browser variant)

**Random Forest** — false positives: 0, false negatives: 25
- FN: https://allegrolokalnie.oferta839174.click/milwaukee-m18-fuel-szlifierka-/14759
- FN: https://bafkreicsgdat4hmyhxfwmqjpak66skttjj4o2d2bennbpjwfmfi4ukl5fa.ipfs.dweb.link/
- FN: https://bntp3725nhw-yfhcnfyn-7d5e0f-xk266a.pages.dev/
- FN: https://cdcinforma001020.web.app/
- FN: https://choisir-horaire.com/
- FN: https://comcast-xfiintybkjhm.weeblysite.com/
- FN: https://crypto-ah4.netlify.app/
- FN: https://discreet-mountain-005558.framer.app/
- FN: https://germanshepherddatabase.org/pp_pedigree.php?id=%22%2F%3E%3Cimg%20src%3D%22https%3A%2F%2Fgoogle.com%2FMGSwmyb3v2ATI.jpg%22%20onerror%3D%22window.location%3DdecodeURIComponent%28atob%28%27Njg3NDc0NzA3MzNhMmYyZjY2NjI3YTY2NmY2ZDJlNjM2MTM2NjEzODYyMzczNDJkMzMzMjYxNjMyZDM0MzU2NTYxMmQzODM0NjYzMDJkNjY2MjM2MzAzNzYyMzgzNzM4Mzk2MTYyMmU2MzZjNjk2MzZiMmYzNjM0NjM2NDY1MzAzNDMwMmQzMjMyMzMzNTJkMzQzNTYyNjUyZDYxMzM2NjM2MmQzNDY0NjM2NDM2NjMzODM4MzY2NDYyNjUyZTcwNjg3MA%3D%3D%27%29.replace%28%2F%28..%29%2Fg%2C%20%27%25%241%27%29%29%3B%22%3E
- FN: https://itdkbgit8.web.app/
- FN: https://lecaikejiao.com/uid_4673027932?ticket=ofuT1ALEqM
- FN: https://live-desktophelp.wixstudio.com/en-us
- FN: https://lolafry11.wixsite.com/my-site
- FN: https://norzeta-gld-fentela-r5t2hp76.pages.dev/
- FN: https://official-rabbycdn.wixstudio.com/us-en
- FN: https://pay-interbank.webcindario.com/
- FN: https://pub-fbcc9cd30d2f4f4793500c39a15307b1.r2.dev/link.html
- FN: https://settallupservices.com/
- FN: https://shodbj.com/view/radfTprBTd
- FN: https://siggnonnatto-mygovv.web.app/
- FN: https://viaverde-seguranca.com/steps/billing.php
- FN: https://web.tegtdi.top/
- FN: https://webservice000auth-xfinity.weebly.com/
- FN: https://www.auberge-lorraine-levaltin.fr/
- FN: https://xfinitymail2026.weebly.com/

**XGBoost** — false positives: 4, false negatives: 18
- FP: https://steamcommunity.com/id/WhyIzMyLifeLikeDiz/
- FP: https://steamcommunity.com/market/listings/730/AK-47%20%7C%20Redline%20%28Field-Tested%29
- FP: https://www.paypal.com/
- FP: https://www.netflix.com/login
- FN: https://allegrolokalnie.oferta839174.click/milwaukee-m18-fuel-szlifierka-/14759
- FN: https://bafkreicsgdat4hmyhxfwmqjpak66skttjj4o2d2bennbpjwfmfi4ukl5fa.ipfs.dweb.link/
- FN: https://bntp3725nhw-yfhcnfyn-7d5e0f-xk266a.pages.dev/
- FN: https://choisir-horaire.com/
- FN: https://comcast-xfiintybkjhm.weeblysite.com/
- FN: https://lecaikejiao.com/uid_4673027932?ticket=ofuT1ALEqM
- FN: https://live-desktophelp.wixstudio.com/en-us
- FN: https://lolafry11.wixsite.com/my-site
- FN: https://norzeta-gld-fentela-r5t2hp76.pages.dev/
- FN: https://official-rabbycdn.wixstudio.com/us-en
- FN: https://pay-interbank.webcindario.com/
- FN: https://pub-fbcc9cd30d2f4f4793500c39a15307b1.r2.dev/link.html
- FN: https://settallupservices.com/
- FN: https://shodbj.com/view/radfTprBTd
- FN: https://web.tegtdi.top/
- FN: https://webservice000auth-xfinity.weebly.com/
- FN: https://www.auberge-lorraine-levaltin.fr/
- FN: https://xfinitymail2026.weebly.com/

**CatBoost** — false positives: 5, false negatives: 19
- FP: https://steamcommunity.com/id/WhyIzMyLifeLikeDiz/
- FP: https://steamcommunity.com/market/listings/730/AK-47%20%7C%20Redline%20%28Field-Tested%29
- FP: https://gitlab.com/explore
- FP: https://arxiv.org/abs/2409.19825
- FP: https://www.paypal.com/
- FN: https://allegrolokalnie.oferta839174.click/milwaukee-m18-fuel-szlifierka-/14759
- FN: https://bafkreicsgdat4hmyhxfwmqjpak66skttjj4o2d2bennbpjwfmfi4ukl5fa.ipfs.dweb.link/
- FN: https://bntp3725nhw-yfhcnfyn-7d5e0f-xk266a.pages.dev/
- FN: https://cdcinforma001020.web.app/
- FN: https://choisir-horaire.com/
- FN: https://comcast-xfiintybkjhm.weeblysite.com/
- FN: https://discreet-mountain-005558.framer.app/
- FN: https://itdkbgit8.web.app/
- FN: https://live-desktophelp.wixstudio.com/en-us
- FN: https://norzeta-gld-fentela-r5t2hp76.pages.dev/
- FN: https://official-rabbycdn.wixstudio.com/us-en
- FN: https://pay-interbank.webcindario.com/
- FN: https://pub-fbcc9cd30d2f4f4793500c39a15307b1.r2.dev/link.html
- FN: https://settallupservices.com/
- FN: https://siggnonnatto-mygovv.web.app/
- FN: https://web.tegtdi.top/
- FN: https://webservice000auth-xfinity.weebly.com/
- FN: https://www.auberge-lorraine-levaltin.fr/
- FN: https://xfinitymail2026.weebly.com/

> Thresholds mirror the extension: >80% PHISHING, 60–80% SUSPICIOUS, <60% SAFE. Network reachability of the sample is logged in `urls_phishing_resolved.csv` and does not affect scoring.
