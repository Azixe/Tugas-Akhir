// === BACKGROUND SERVICE WORKER ===
importScripts('utils.js', 'libs/ort.min.js');

let session = null, tfidfData = null, ready = false;

async function init() {
    if (ready) return true;
    try {
        ort.env.wasm.wasmPaths = chrome.runtime.getURL('libs/');
        ort.env.wasm.numThreads = 1;
        
        const [tRes, mRes] = await Promise.all([
            fetch(chrome.runtime.getURL('tfidf_data.json')),
            fetch(chrome.runtime.getURL('phishing_rf.onnx'))
        ]);
        tfidfData = await tRes.json();
        session = await ort.InferenceSession.create(await mRes.arrayBuffer(), {
            executionProviders: ['wasm']
        });
        ready = true;
        console.log('[BG] Ready');
        return true;
    } catch (e) {
        console.error('[BG] Init error:', e);
        return false;
    }
}

async function predict(url) {
    const startTime = performance.now();
    
    const features = [...tfidf(url.toLowerCase(), tfidfData), ...structural(url)];
    const input = new ort.Tensor('float32', Float32Array.from(features), [1, features.length]);
    const results = await session.run({ [session.inputNames[0]]: input });
    
    const label = Number(results[session.outputNames[0]].data[0]);
    const probs = results[session.outputNames[1]].data;
    const conf = (label === 1 ? probs[1] : probs[0]) * 100;
    
    const inferenceTime = performance.now() - startTime;
    console.log(`[BG] ⏱️ Inference: ${inferenceTime.toFixed(2)}ms`);
    
    return { isPhishing: label === 1, confidence: conf, url, inferenceTime };
}

// Check if URL is in user whitelist
async function isUserWhitelisted(url) {
    try {
        const { userWhitelist = [] } = await chrome.storage.local.get('userWhitelist');
        const host = new URL(url).hostname.toLowerCase();
        return userWhitelist.some(d => host.includes(d) || d.includes(host));
    } catch { return false; }
}

// Add domain to whitelist
async function addToWhitelist(domain) {
    const { userWhitelist = [] } = await chrome.storage.local.get('userWhitelist');
    domain = domain.toLowerCase().replace(/^https?:\/\//, '').split('/')[0];
    if (!userWhitelist.includes(domain) && domain) {
        userWhitelist.push(domain);
        await chrome.storage.local.set({ userWhitelist });
    }
    return userWhitelist;
}

// Navigation listener
chrome.webNavigation.onCompleted.addListener(async ({ frameId, tabId, url }) => {
    if (frameId !== 0 || !shouldScan(url)) return;
    
    // Check user whitelist
    if (await isUserWhitelisted(url)) {
        console.log('[BG] User whitelisted:', url);
        return;
    }
    
    if (!await init()) return;
    
    try {
        const r = await predict(url);
        console.log('[BG]', r.isPhishing ? '⚠️' : '✅', url, r.confidence.toFixed(1) + '%');
        
        if (r.isPhishing && r.confidence > 80) {
            chrome.tabs.update(tabId, {
                url: chrome.runtime.getURL('blocked.html') + 
                    `?url=${encodeURIComponent(url)}&conf=${r.confidence.toFixed(1)}`
            });
        } else if (r.isPhishing && r.confidence > 60) {
            chrome.tabs.sendMessage(tabId, { action: 'warn', ...r }).catch(() => {});
        }
    } catch (e) {
        console.error('[BG] Error:', e);
    }
});

// Message handler for popup and content script
chrome.runtime.onMessage.addListener((req, _, res) => {
    if (req.action === 'scan') {
        init().then(ok => ok ? predict(req.url).then(res).catch(e => res({ error: e.message })) : res({ error: 'Not ready' }));
        return true;
    }
    if (req.action === 'addWhitelist') {
        addToWhitelist(req.domain).then(list => res({ success: true, list }));
        return true;
    }
});

console.log('[BG] Service worker loaded');
