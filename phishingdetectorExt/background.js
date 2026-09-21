// === BACKGROUND SERVICE WORKER ===
importScripts('utils.js', 'libs/ort.min.js');

// Model state
let currentModel = 'rf';  // 'rf' or 'xgb'
let models = {
    rf:  { session: null, tfidfData: null, ready: false },
    xgb: { session: null, tfidfData: null, prepData: null, ready: false }
};

// Load saved model preference
chrome.storage.local.get('selectedModel', ({ selectedModel }) => {
    if (selectedModel) currentModel = selectedModel;
    console.log('[BG] Selected model:', currentModel);
});

async function initModel(modelType) {
    const m = models[modelType];
    if (m.ready) return true;

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
        } else if (modelType === 'xgb') {
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
    }
}

async function predict(url) {
    const startTime = performance.now();
    const m = models[currentModel];

    // Extract raw features (TF-IDF + structural = 1509)
    const rawFeatures = [...tfidf(url.toLowerCase(), m.tfidfData), ...structural(url)];

    let inputFeatures;
    if (currentModel === 'xgb') {
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
    console.log(`[BG] [${currentModel.toUpperCase()}] ${label === 1 ? 'PHISHING' : 'SAFE'} ${url} ${conf.toFixed(1)}% (${inferenceTime.toFixed(2)}ms)`);

    return {
        isPhishing: label === 1,
        phishingProbability: phishingProb,
        legitimateProbability: legitProb,
        confidence: conf,
        url,
        inferenceTime,
        model: currentModel
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

// Add domain to whitelist
async function addToWhitelist(domain) {
    const { userWhitelist = [] } = await chrome.storage.local.get('userWhitelist');
    domain = domain.toLowerCase().replace(/^https?:\/\//, '').split('/')[0].trim();
    if (!userWhitelist.includes(domain) && domain) {
        userWhitelist.push(domain);
        await chrome.storage.local.set({ userWhitelist });
    }
    return userWhitelist;
}

// Session bypass set for one-time continuation from blocked page
const sessionBypass = new Set();
const pendingWarnings = new Map();

// Navigation listeners:
// 1. Intercept before navigation starts (blocks high-confidence phishing before scripts run)
chrome.webNavigation.onBeforeNavigate.addListener(async ({ frameId, tabId, url }) => {
    if (frameId !== 0 || !shouldScan(url)) return;

    // Check one-time session bypass (e.g. user clicked "Continue" on blocked.html)
    if (sessionBypass.has(url)) {
        sessionBypass.delete(url);
        return;
    }

    // Check user whitelist
    if (await isUserWhitelisted(url)) {
        console.log('[BG] User whitelisted:', url);
        return;
    }

    if (!await initModel(currentModel)) return;

    try {
        const r = await predict(url);

        if (r.isPhishing && r.phishingProbability > 80) {
            chrome.tabs.update(tabId, {
                url: chrome.runtime.getURL('blocked.html') +
                    `?url=${encodeURIComponent(url)}&conf=${r.phishingProbability.toFixed(1)}`
            });
        } else if (r.isPhishing && r.phishingProbability > 60) {
            pendingWarnings.set(tabId, { url, ...r });
        }
    } catch (e) {
        console.error('[BG] Navigation intercept error:', e);
    }
});

// 2. Deliver warning banner to content script once the DOM has loaded
chrome.webNavigation.onCompleted.addListener(async ({ frameId, tabId, url }) => {
    if (frameId !== 0) return;
    if (pendingWarnings.has(tabId)) {
        const warnData = pendingWarnings.get(tabId);
        if (warnData.url === url) {
            chrome.tabs.sendMessage(tabId, { action: 'warn', ...warnData }).catch(() => {});
        }
        pendingWarnings.delete(tabId);
    }
});

// Message handler for popup, content script, and blocked page
chrome.runtime.onMessage.addListener((req, _, res) => {
    if (req.action === 'scan') {
        initModel(currentModel).then(ok =>
            ok ? predict(req.url).then(res).catch(e => res({ error: e.message }))
               : res({ error: 'Model not ready' })
        );
        return true;
    }
    if (req.action === 'switchModel') {
        currentModel = req.model;
        chrome.storage.local.set({ selectedModel: currentModel });
        initModel(currentModel).then(ok => {
            res({ success: ok, model: currentModel });
        });
        return true;
    }
    if (req.action === 'getModel') {
        res({ model: currentModel });
        return true;
    }
    if (req.action === 'addWhitelist') {
        addToWhitelist(req.domain).then(list => res({ success: true, list }));
        return true;
    }
    if (req.action === 'bypassUrl') {
        sessionBypass.add(req.url);
        res({ success: true });
        return true;
    }
});

console.log('[BG] Service worker loaded (RF + XGBoost)');
