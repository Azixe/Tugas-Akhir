// === BACKGROUND SERVICE WORKER ===
importScripts('utils.js', 'libs/ort.min.js');

// Model state
let currentModel = 'rf';  // 'rf' or 'xgb'
let models = {
    rf:  { session: null, tfidfData: null, ready: false, initPromise: null },
    xgb: { session: null, tfidfData: null, prepData: null, ready: false, initPromise: null }
};

// Load saved model preference
chrome.storage.local.get('selectedModel', ({ selectedModel }) => {
    if (selectedModel === 'rf' || selectedModel === 'xgb') currentModel = selectedModel;
    console.log('[BG] Selected model:', currentModel);
});

async function initModel(modelType) {
    const m = models[modelType];
    if (!m) return false;
    if (m.ready) return true;
    if (m.initPromise) return m.initPromise;   // dedupe concurrent init calls

    m.initPromise = (async () => {
        try {
            ort.env.wasm.wasmPaths = chrome.runtime.getURL('libs/');
            ort.env.wasm.numThreads = 1;

            if (modelType === 'rf') {
                const [tRes, mRes] = await Promise.all([
                    fetch(chrome.runtime.getURL('tfidf_data.json')),
                    fetch(chrome.runtime.getURL('phishing_rf.onnx'))
                ]);
                m.tfidfData = await tRes.json();
                m.session = await ort.InferenceSession.create(await mRes.arrayBuffer(), {
                    executionProviders: ['wasm']
                });
            } else {
                const [tRes, mRes, pRes] = await Promise.all([
                    fetch(chrome.runtime.getURL('tfidf_data_xgb.json')),
                    fetch(chrome.runtime.getURL('phishing_xgb.onnx')),
                    fetch(chrome.runtime.getURL('xgb_preprocessing.json'))
                ]);
                m.tfidfData = await tRes.json();
                m.session = await ort.InferenceSession.create(await mRes.arrayBuffer(), {
                    executionProviders: ['wasm']
                });
                m.prepData = await pRes.json();
            }

            m.ready = true;
            console.log(`[BG] ${modelType.toUpperCase()} model ready`);
            return true;
        } catch (e) {
            console.error(`[BG] ${modelType.toUpperCase()} init error:`, e);
            return false;
        } finally {
            m.initPromise = null;
        }
    })();
    return m.initPromise;
}

async function predict(url) {
    const startTime = performance.now();
    // Snapshot the model: a switchModel call while this scan runs must not
    // leave us reading a half-initialized session.
    const modelType = currentModel;
    const m = models[modelType];
    if (!m || !m.ready || !m.session) throw new Error(`${modelType.toUpperCase()} model not ready`);

    // Extract raw features (TF-IDF + structural = 1509)
    const rawFeatures = [...tfidf(url.toLowerCase(), m.tfidfData), ...structural(url)];

    let inputFeatures;
    if (modelType === 'xgb') {
        // XGBoost: scale → select (179) → PCA → 151 features
        inputFeatures = preprocessXgb(rawFeatures, m.prepData);
    } else {
        // RF: use raw 1509 features directly
        inputFeatures = rawFeatures;
    }

    const input = new ort.Tensor('float32', Float32Array.from(inputFeatures), [1, inputFeatures.length]);
    const results = await m.session.run({ [m.session.inputNames[0]]: input });

    const label = Number(results[m.session.outputNames[0]].data[0]);
    const probs = results[m.session.outputNames[1]].data;
    const phishingProb = probs[1] * 100;
    const legitProb = probs[0] * 100;
    const conf = (label === 1 ? phishingProb : legitProb);

    const inferenceTime = performance.now() - startTime;
    console.log(`[BG] [${modelType.toUpperCase()}] ${label === 1 ? 'PHISHING' : 'SAFE'} ${url} ${conf.toFixed(1)}% (${inferenceTime.toFixed(2)}ms)`);

    return {
        isPhishing: label === 1,
        phishingProbability: phishingProb,
        legitimateProbability: legitProb,
        confidence: conf,
        url,
        inferenceTime,
        model: modelType
    };
}

// Check if URL is in user whitelist
async function isUserWhitelisted(url) {
    try {
        const { userWhitelist = [] } = await chrome.storage.local.get('userWhitelist');
        const host = new URL(url).hostname.toLowerCase();
        return isDomainWhitelisted(host, userWhitelist);
    } catch { return false; }
}

// Add domain to whitelist (normalized to a bare lowercase hostname);
// also cleans legacy entries that carry ports/userinfo.
async function addToWhitelist(domain) {
    const { userWhitelist = [] } = await chrome.storage.local.get('userWhitelist');
    const normalized = normalizeDomain(domain);
    if (!normalized) return userWhitelist;
    const clean = [...new Set(userWhitelist.map(normalizeDomain).filter(Boolean))];
    if (!clean.includes(normalized)) clean.push(normalized);
    await chrome.storage.local.set({ userWhitelist: clean });
    return clean;
}

// --- Navigation scanning state ---

// One-time continuations from blocked.html: url -> expiry timestamp.
// Service-worker lifetime is enough because the user navigates right after
// clicking "Continue"; the TTL bounds entries the user never follows through.
const sessionBypass = new Map();
const BYPASS_TTL_MS = 60 * 1000;

// tabId -> { url, data, delivered } for URLs in the 60-80% warning band
const pendingWarnings = new Map();
// tabId -> url currently being scored (dedupes repeated onBeforeNavigate)
const scansInFlight = new Map();

const WARN_THRESHOLD = 60;
const BLOCK_THRESHOLD = 80;

function addBypass(url) {
    if (sessionBypass.size > 100) sessionBypass.clear();
    sessionBypass.set(url, Date.now() + BYPASS_TTL_MS);
}

function consumeBypass(url) {
    const expiresAt = sessionBypass.get(url);
    if (expiresAt === undefined) return false;
    sessionBypass.delete(url);
    return expiresAt > Date.now();
}

async function isTabStillOn(tabId, url) {
    try {
        const tab = await chrome.tabs.get(tabId);
        return tab.url === url || tab.pendingUrl === url;
    } catch { return false; }
}

function blockedPageUrl(r) {
    return chrome.runtime.getURL('blocked.html') +
        `?url=${encodeURIComponent(r.url)}&conf=${r.phishingProbability.toFixed(1)}`;
}

function tryDeliverWarning(tabId) {
    const pending = pendingWarnings.get(tabId);
    if (!pending) return;
    chrome.tabs.sendMessage(tabId, { action: 'warn', ...pending.data })
        .then(() => { pending.delivered = true; })
        .catch(() => {});   // content script not ready yet: onCompleted retries
}

async function handleNavigationScan(tabId, url) {
    if (!shouldScan(url)) return;
    try {
        if (await isUserWhitelisted(url)) {
            console.log('[BG] User whitelisted:', url);
            return;
        }
        if (!await initModel(currentModel)) return;

        const r = await predict(url);
        if (!r.isPhishing || r.phishingProbability <= WARN_THRESHOLD) return;

        if (r.phishingProbability > BLOCK_THRESHOLD) {
            // The user may have moved on while the model was running: never
            // replace a tab that is no longer showing the scanned URL.
            if (!await isTabStillOn(tabId, r.url)) return;
            await chrome.tabs.update(tabId, { url: blockedPageUrl(r) }).catch(() => {});
            return;
        }

        pendingWarnings.set(tabId, { url: r.url, data: r, delivered: false });
        if (await isTabStillOn(tabId, r.url)) tryDeliverWarning(tabId);
    } catch (e) {
        console.error('[BG] Navigation scan error:', e);
    }
}

// Navigation listeners:
// 1. Intercept before the navigation commits. This is best-effort: the page
//    may already start loading while the model initializes and scores; a >80%
//    verdict replaces the tab once the result is ready (it does not cancel
//    the in-flight request).
chrome.webNavigation.onBeforeNavigate.addListener(({ frameId, tabId, url }) => {
    if (frameId !== 0 || !shouldScan(url)) return;

    pendingWarnings.delete(tabId);   // a new navigation invalidates old warnings

    if (consumeBypass(url)) return;

    // Redirects and process swaps can raise repeated onBeforeNavigate events
    // for one logical navigation; score each tab+URL only once.
    if (scansInFlight.get(tabId) === url) return;
    scansInFlight.set(tabId, url);
    handleNavigationScan(tabId, url).finally(() => {
        if (scansInFlight.get(tabId) === url) scansInFlight.delete(tabId);
    });
});

// 2. On load completion: deliver pending warnings; if the final URL differs
//    from the scanned one (server redirect), score the final URL as well.
chrome.webNavigation.onCompleted.addListener(async ({ frameId, tabId, url }) => {
    if (frameId !== 0) return;

    const pending = pendingWarnings.get(tabId);
    if (pending) pendingWarnings.delete(tabId);

    if (pending && pending.url === url) {
        if (!pending.delivered) {
            chrome.tabs.sendMessage(tabId, { action: 'warn', ...pending.data }).catch(() => {});
        }
        return;
    }
    if (!pending) return;   // already scored in onBeforeNavigate

    await handleNavigationScan(tabId, url);
});

// 3. Drop per-tab state when a tab goes away.
chrome.tabs.onRemoved.addListener(tabId => {
    pendingWarnings.delete(tabId);
    scansInFlight.delete(tabId);
});

function isBlockedPageSender(sender) {
    return typeof sender?.url === 'string' &&
        sender.url.startsWith(chrome.runtime.getURL('blocked.html'));
}

// Message handler for popup, content script, and blocked page
chrome.runtime.onMessage.addListener((req, sender, res) => {
    if (!req || typeof req.action !== 'string') return;

    if (req.action === 'scan') {
        if (typeof req.url !== 'string' || !req.url) { res({ error: 'Invalid URL' }); return; }
        initModel(currentModel).then(ok =>
            ok ? predict(req.url).then(res).catch(e => res({ error: e.message }))
               : res({ error: 'Model not ready' })
        );
        return true;
    }
    if (req.action === 'switchModel') {
        if (req.model !== 'rf' && req.model !== 'xgb') {
            res({ success: false, error: 'Invalid model' });
            return;
        }
        currentModel = req.model;
        chrome.storage.local.set({ selectedModel: currentModel });
        initModel(currentModel).then(ok => res({ success: ok, model: currentModel }));
        return true;
    }
    if (req.action === 'getModel') {
        res({ model: currentModel });
        return;
    }
    if (req.action === 'addWhitelist') {
        if (typeof req.domain !== 'string' || !normalizeDomain(req.domain)) {
            res({ success: false, error: 'Invalid domain' });
            return;
        }
        addToWhitelist(req.domain).then(list => res({ success: true, list }));
        return true;
    }
    if (req.action === 'bypassUrl') {
        if (!isBlockedPageSender(sender)) { res({ success: false, error: 'Forbidden' }); return; }
        if (typeof req.url !== 'string' || !req.url) { res({ success: false, error: 'Invalid URL' }); return; }
        addBypass(req.url);
        res({ success: true });
        return;
    }
});

console.log('[BG] Service worker loaded (RF + XGBoost)');
