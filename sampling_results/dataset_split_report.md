# Train/Test Host Overlap — random split limitation

Common protocol (stratified 80/20, seed 42); rows deduplicated by URL.

| Metric | Value |
|---|---:|
| Test rows whose exact host appears in train | 63066 (70.05%) |
| Test rows whose 2-label domain appears in train | 70838 (78.68%) |
| Unique hosts — train / test | 141,040 / 46,048 |

## Top overlapping exact hosts (test rows)

| Host | Test rows |
|---|---:|
| `www.en.wikipedia.org` | 2533 |
| `www.youtube.com` | 1667 |
| `www.facebook.com` | 1525 |
| `www.amazon.com` | 914 |
| `www.imdb.com` | 719 |
| `www.linkedin.com` | 680 |
| `www.myspace.com` | 586 |
| `www.mylife.com` | 583 |
| `www.absoluteastronomy.com` | 395 |
| `www.manta.com` | 381 |

> Reading: a row-level random split measures in-distribution generalization; the overlap above quantifies how much of it comes from the same hosts being seen during training.
