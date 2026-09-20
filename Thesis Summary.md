Here's a clean summary of your thesis proposal.

---

**Title:** *Perancangan Ekstensi Browser Client-Side untuk Deteksi URL Phishing Real-Time Berbasis Algoritma Random Forest dan XGBoost*

---

**What you're building**

A standalone Google Chrome extension that detects phishing URLs in real-time, entirely on the client side — no external server, no API calls, no blacklist dependency. The ML model runs locally inside the browser via ONNX Runtime.

---

**The problem being solved**

Conventional browser defenses (Google Safe Browsing, AV signature DBs) rely on blacklists with an average detection lag of ~4.5 days. Modern phishing sites have a median lifespan of only ~5.46 hours — meaning ~83.9% are already dead before they're ever flagged. Existing client-side solutions still depend on a local Python backend, which isn't truly standalone. Server-side approaches introduce network latency and privacy leakage. Your thesis fills that gap.

---

**Technical specs**

| Layer | Stack |
|---|---|
| ML Training | Python, scikit-learn (Random Forest), XGBoost |
| Data | 450,176 URLs (Mendeley, 2024); balanced via undersampling to ~104k/class |
| Features | 9 lexical features: URL length, dot count, slash count, dash count, @ count, digit ratio, Shannon entropy, TLD type, subdomain depth |
| Model format | Converted .pkl → ONNX via `skl2onnx` |
| Inference runtime | ONNX Runtime Web (`ort-wasm`) running in JS |
| Extension architecture | Manifest V3; Service Worker + Content Script + Popup UI |
| Platform | Google Chrome desktop only |

---

**System workflow**

1. User navigates to a URL → browser fires a navigation event
2. Service Worker intercepts → checks local whitelist first (bypass if found)
3. If not whitelisted → extracts 9 lexical features from the URL string in JS
4. Feeds feature vector into ONNX model → gets probability score
5. Decision logic:
   - Score > 80% → block page (`blocked.html`)
   - Score 60–80% → inject warning banner via Content Script
   - Score < 60% → no action

---

**Expected outcomes / success criteria**

| Metric | Target |
|---|---|
| F1-Score | > 90% |
| Inference latency | < 100ms |
| Idle RAM | < 50 MB |
| Peak RAM | < 150 MB |
| CPU spike | < 30% during scan |
| Extension package size | < 30 MB |

The core research question is also a **comparative study** between Random Forest vs. XGBoost — specifically which algorithm achieves the better balance between detection accuracy, false positive rate, and runtime efficiency in a resource-constrained browser environment.

---

One note: this is a proposal, not a completed thesis — so the model hasn't been trained yet and the performance figures above are targets/hypotheses, not measured results.








1. Functional Testing (Black-box)
Tests whether the extension works as intended behaviorally, without inspecting internals. Test cases include:

Automatic detection triggering on navigation
Whitelist feature (bypass for trusted domains)
Blocked page rendering (blocked.html)
Warning banner injection for suspicious URLs

Sample URLs sourced from PhishTank (live phishing URLs).

2. System Performance Testing
Conducted via Chrome DevTools Performance Profiler. Four metrics:
IDMetricDescriptionPass CriteriaP-01Inference LatencyTime from URL input to prediction output< 100ms averageP-02Memory FootprintService Worker RAM usage< 50MB idle, < 150MB peakP-03CPU UsageCPU spike during feature extraction + inference< 30% on standard processorP-04Extension SizeTotal .crx package including ONNX model< 30 MB

3. ML Model Performance Testing
Evaluated on the 20% held-out test split using Confusion Matrix derived metrics:
MetricTargetAccuracyHigh overall classification correctnessPrecisionMinimize false positives (legitimate sites wrongly blocked)RecallMinimize false negatives (phishing sites that slip through)F1-Score> 90% (hard pass/fail gate)
If F1 < 90%, the pipeline loops back to hyperparameter tuning or feature re-engineering before proceeding.

4. Comparative Analysis (RF vs. XGBoost)
There's also an implicit fourth testing layer — the head-to-head comparison between the two algorithms across all the above metrics. The goal isn't just "which is more accurate" but which achieves the optimal accuracy vs. computational cost tradeoff for a browser-constrained environment. This is effectively the core research contribution.