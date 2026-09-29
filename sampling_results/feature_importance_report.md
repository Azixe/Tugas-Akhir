# Feature Importance (RQ2) — Random Forest & XGBoost

Train split only (common protocol, seed 42); the test set is never used here.

## Random Forest — top 20 (MDI, raw 1,509 features)

| Rank | Feature | Type | Importance |
|---:|---|---|---:|
| 1 | `https:` | tfidf | 0.234057 |
| 2 | `http:` | tfidf | 0.204493 |
| 3 | `subdomain_level` | structural | 0.103111 |
| 4 | `dot_count` | structural | 0.068895 |
| 5 | `wp` | tfidf | 0.031099 |
| 6 | `digit_ratio` | structural | 0.024937 |
| 7 | `url_length` | structural | 0.019809 |
| 8 | `php` | tfidf | 0.019537 |
| 9 | `login` | tfidf | 0.016961 |
| 10 | `entropy` | structural | 0.015072 |
| 11 | `` | tfidf | 0.014717 |
| 12 | `br` | tfidf | 0.014440 |
| 13 | `includes` | tfidf | 0.013253 |
| 14 | `admin` | tfidf | 0.012694 |
| 15 | `dash_count` | structural | 0.012122 |
| 16 | `js` | tfidf | 0.011117 |
| 17 | `images` | tfidf | 0.011096 |
| 18 | `ru` | tfidf | 0.010580 |
| 19 | `content` | tfidf | 0.007199 |
| 20 | `index` | tfidf | 0.007197 |

### Nine structural features (RF)

| Feature | Importance | Rank (of 1,509) |
|---|---:|---:|
| `url_length` | 0.019809 | 7 |
| `dot_count` | 0.068895 | 4 |
| `slash_count` | 0.006871 | 21 |
| `dash_count` | 0.012122 | 15 |
| `at_count` | 0.005897 | 23 |
| `digit_ratio` | 0.024937 | 6 |
| `entropy` | 0.015072 | 10 |
| `is_common_tld` | 0.004267 | 27 |
| `subdomain_level` | 0.103111 | 3 |

## XGBoost — top 20 (importance model, 200 trees / depth 7)

| Rank | Feature | Type | Importance |
|---:|---|---|---:|
| 1 | `http:` | tfidf | 0.795185 |
| 2 | `sites.google.com` | tfidf | 0.029437 |
| 3 | `google` | tfidf | 0.017106 |
| 4 | `subdomain_level` | structural | 0.011442 |
| 5 | `000webhostapp` | tfidf | 0.007163 |
| 6 | `wixsite` | tfidf | 0.006870 |
| 7 | `sharepoint` | tfidf | 0.004674 |
| 8 | `runescape` | tfidf | 0.004629 |
| 9 | `login` | tfidf | 0.004008 |
| 10 | `m=weblogin` | tfidf | 0.003861 |
| 11 | `wp` | tfidf | 0.002891 |
| 12 | `br` | tfidf | 0.002773 |
| 13 | `preview` | tfidf | 0.002674 |
| 14 | `webapps` | tfidf | 0.002500 |
| 15 | `paypal` | tfidf | 0.002389 |
| 16 | `redirect` | tfidf | 0.002099 |
| 17 | `blogspot` | tfidf | 0.002060 |
| 18 | `tumblr` | tfidf | 0.002040 |
| 19 | `secure` | tfidf | 0.002015 |
| 20 | `is_common_tld` | structural | 0.001947 |

### Nine structural features (XGB, mask = importance > median = 0.0)

| Feature | Importance | Rank (of 1,509) | Survived pruning? |
|---|---:|---:|:--:|
| `url_length` | 0.000687 | 64 | yes |
| `dot_count` | 0.000674 | 67 | yes |
| `slash_count` | 0.000418 | 107 | yes |
| `dash_count` | 0.000471 | 99 | yes |
| `at_count` | 0.000698 | 61 | yes |
| `digit_ratio` | 0.000500 | 88 | yes |
| `entropy` | 0.000890 | 48 | yes |
| `is_common_tld` | 0.001947 | 20 | yes |
| `subdomain_level` | 0.011442 | 4 | yes |

Pruning mask: 177 of 1509 features survive (9/9 structural).
> TF-IDF importances come from the exact pipeline steps used at training time (saved extractor + saved scaler + SelectKBest k='all').
