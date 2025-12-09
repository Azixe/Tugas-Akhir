// === BACKGROUND SERVICE WORKER ===
// Auto-detect phishing URLs pada setiap navigasi

importScripts('libs/ort.min.js');

let session = null;
let tfidfData = null;
let isInitialized = false;

// --- INITIALIZATION ---
async function initialize() {
    if (isInitialized) return true;
    
    try {
        console.log("[BG] Initializing ONNX Runtime...");
        
        // Configure WASM
        ort.env.wasm.wasmPaths = chrome.runtime.getURL("libs/");
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
        
        // Load TF-IDF data
        const tfidfResponse = await fetch(chrome.runtime.getURL("tfidf_data.json"));
        tfidfData = await tfidfResponse.json();
        console.log("[BG] TF-IDF loaded. Vocab size:", Object.keys(tfidfData.vocabulary).length);
        
        // Load ONNX model
        const modelUrl = chrome.runtime.getURL("phishing_rf.onnx");
        const modelResponse = await fetch(modelUrl);
        const modelBuffer = await modelResponse.arrayBuffer();
        
        session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        // DEBUG: Print input/output names
        console.log("[BG] Model Input Names:", session.inputNames);
        console.log("[BG] Model Output Names:", session.outputNames);
        
        console.log("[BG] Model loaded successfully!");
        isInitialized = true;
        return true;
        
    } catch (e) {
        console.error("[BG] Init Error:", e);
        return false;
    }
}

// --- URL PREDICTION (FIXED) ---
async function predictUrl(url) {
    if (!session || !tfidfData) {
        throw new Error("Model not initialized");
    }
    
    const s_url = url.toLowerCase();
    const textVector = computeTfidf(s_url, tfidfData);
    const structVector = computeStructuralFeatures(url);
    const combinedFeatures = [...textVector, ...structVector];
    
    // Use actual input name from model
    const inputName = session.inputNames[0];
    const inputTensor = new ort.Tensor('float32', Float32Array.from(combinedFeatures), [1, combinedFeatures.length]);
    const feeds = { [inputName]: inputTensor };
    
    const results = await session.run(feeds);
    
    // DEBUG: Print all outputs
    console.log("[BG] Raw outputs:", Object.keys(results));
    for (const [name, tensor] of Object.entries(results)) {
        console.log(`[BG] Output "${name}":`, tensor.data);
    }
    
    // Get output names dynamically
    const outputNames = session.outputNames;
    
    // Usually: outputNames[0] = label, outputNames[1] = probabilities
    const labelOutput = results[outputNames[0]];
    const probOutput = results[outputNames[1]];
    
    let prediction = 0;
    let confidence = 0;
    
    if (labelOutput) {
        prediction = Number(labelOutput.data[0]);
    }
    
    if (probOutput) {
        // probOutput bisa berupa array atau Map-like structure
        const probData = probOutput.data;
        
        if (probData instanceof Map) {
            // ZipMap output
            confidence = (probData.get(1) || 0) * 100;
        } else if (Array.isArray(probData) || probData.length) {
            // Array output [prob_0, prob_1]
            confidence = (prediction === 1 ? probData[1] : probData[0]) * 100;
        } else if (typeof probData === 'object') {
            // Object output {0: prob_0, 1: prob_1}
            confidence = (prediction === 1 ? (probData[1] || 0) : (probData[0] || 0)) * 100;
        }
    }
    
    // Fallback: jika tidak ada probability, gunakan prediction saja
    if (confidence === 0 && prediction === 1) {
        confidence = 100;
    }
    
    return {
        isPhishing: prediction === 1,
        confidence: confidence,
        url: url
    };
}

// --- TF-IDF VECTORIZER ---
function computeTfidf(url, data) {
    const vocabSize = Object.keys(data.vocabulary).length;
    const tfVector = new Array(vocabSize).fill(0);
    const tokens = makeTokens(url);
    
    tokens.forEach(token => {
        if (data.vocabulary.hasOwnProperty(token)) {
            const idx = data.vocabulary[token];
            tfVector[idx] += 1;
        }
    });
    
    const tfidfVector = tfVector.map((tf, i) => {
        if (tf > 0 && data.sublinear_tf) {
            tf = 1 + Math.log(tf);
        }
        return tf * data.idf[i];
    });
    
    if (data.norm === 'l2') {
        const norm = Math.sqrt(tfidfVector.reduce((sum, val) => sum + val * val, 0));
        if (norm > 0) {
            return tfidfVector.map(val => val / norm);
        }
    }
    return tfidfVector;
}

// --- TOKENIZER ---
function makeTokens(url) {
    const partsBySlash = url.split('/');
    let totalTokens = [];
    
    partsBySlash.forEach(part => {
        const tokensByDash = part.split('-');
        let tokensDot = [];
        
        tokensByDash.forEach(token => {
            const tempTokens = token.split('.');
            tokensDot = tokensDot.concat(tempTokens);
        });
        
        totalTokens = totalTokens.concat(tokensByDash).concat(tokensDot);
    });
    
    const uniqueTokens = [...new Set(totalTokens)];
    return uniqueTokens.filter(t => t && t !== 'com' && t !== 'www');
}

// --- STRUCTURAL FEATURES ---
function computeStructuralFeatures(originalUrl) {
    const s_url = originalUrl.toLowerCase();
    const features = [];
    
    features.push(s_url.length);
    features.push((s_url.match(/\./g) || []).length);
    features.push((s_url.match(/\//g) || []).length);
    features.push((s_url.match(/-/g) || []).length);
    features.push((s_url.match(/@/g) || []).length);
    
    const digitCount = (s_url.match(/\d/g) || []).length;
    features.push(s_url.length > 0 ? digitCount / s_url.length : 0);
    
    features.push(calculateEntropy(s_url));
    
    const commonTlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id'];
    features.push(commonTlds.some(tld => s_url.endsWith(tld)) ? 1 : 0);
    
    try {
        let domain = s_url.replace(/^https?:\/\//, '').split('/')[0];
        features.push(Math.max(0, (domain.match(/\./g) || []).length - 1));
    } catch (e) {
        features.push(0);
    }
    
    return features;
}

function calculateEntropy(str) {
    if (!str || str.length === 0) return 0;
    
    const freq = {};
    for (const char of str) {
        freq[char] = (freq[char] || 0) + 1;
    }
    
    let entropy = 0;
    const len = str.length;
    for (const count of Object.values(freq)) {
        const p = count / len;
        entropy -= p * Math.log2(p);
    }
    return entropy;
}

// --- WHITELIST ---
const WHITELIST = [
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'github.com', 'stackoverflow.com', 'microsoft.com', 'apple.com',
    'amazon.com', 'wikipedia.org', 'linkedin.com', 'reddit.com'
];

function isWhitelisted(url) {
    try {
        const urlObj = new URL(url);
        const hostname = urlObj.hostname.toLowerCase();
        return WHITELIST.some(domain => hostname.includes(domain));
    } catch (e) {
        return true;
    }
}

function shouldScan(url) {
    if (!url) return false;
    if (!url.startsWith('http://') && !url.startsWith('https://')) return false;
    if (url.startsWith('chrome://') || url.startsWith('chrome-extension://')) return false;
    if (isWhitelisted(url)) return false;
    return true;
}

// --- MAIN: Listen for Navigation ---
chrome.webNavigation.onCompleted.addListener(async (details) => {
    if (details.frameId !== 0) return;
    
    const url = details.url;
    
    if (!shouldScan(url)) {
        console.log("[BG] Skipped:", url);
        return;
    }
    
    console.log("[BG] Scanning:", url);
    
    const ready = await initialize();
    if (!ready) {
        console.error("[BG] Failed to initialize");
        return;
    }
    
    try {
        const result = await predictUrl(url);
        console.log("[BG] Result:", result);
        
        if (result.isPhishing && result.confidence > 80) {
            // HIGH RISK: Block and redirect
            const blockedUrl = chrome.runtime.getURL("blocked.html") + 
                `?url=${encodeURIComponent(url)}` +
                `&confidence=${result.confidence.toFixed(1)}`;
            
            chrome.tabs.update(details.tabId, { url: blockedUrl });
            console.log("[BG] 🚫 BLOCKED:", url);
            
        } else if (result.isPhishing && result.confidence > 60) {
            chrome.tabs.sendMessage(details.tabId, {
                action: "showWarning",
                confidence: result.confidence,
                url: url
            }).catch(err => console.log("[BG] Content script not ready:", err));
            console.log("[BG] ⚠️ WARNING:", url);
            
        } else {
            // SAFE: Optional - show safe badge
            chrome.tabs.sendMessage(details.tabId, {
                action: "showSafe"
            }).catch(err => {});
            console.log("[BG] ✅ SAFE:", url);
        }
        
    } catch (e) {
        console.error("[BG] Prediction error:", e);
    }
});

// --- Listen for messages from popup ---
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "scanUrl") {
        initialize().then(ready => {
            if (!ready) {
                sendResponse({ error: "Model not ready" });
                return;
            }
            predictUrl(request.url).then(result => {
                sendResponse(result);
            }).catch(e => {
                sendResponse({ error: e.message });
            });
        });
        return true;
    }
});

console.log("[BG] Phishing Detector Service Worker loaded!");