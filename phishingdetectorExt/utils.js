// === SHARED UTILITIES ===
// Common functions used by both background.js and popup.js

const WHITELIST = ['google.com','youtube.com','facebook.com','twitter.com','github.com','stackoverflow.com','microsoft.com','apple.com','amazon.com','wikipedia.org','linkedin.com','instagram.com'];
const COMMON_TLDS = ['.com','.org','.net','.edu','.gov','.id','.co.id'];

// Shannon Entropy
function entropy(s) {
    if (!s) return 0;
    const freq = {};
    for (const c of s) freq[c] = (freq[c]||0) + 1;
    let e = 0;
    for (const v of Object.values(freq)) {
        const p = v / s.length;
        e -= p * Math.log2(p);
    }
    return e;
}

// Tokenizer (must match Python)
function tokenize(url) {
    let tokens = [];
    url.split('/').forEach(p => {
        const byDash = p.split('-');
        let byDot = [];
        byDash.forEach(t => byDot = byDot.concat(t.split('.')));
        tokens = tokens.concat(byDash, byDot);
    });
    return [...new Set(tokens)].filter(t => t && t !== 'com' && t !== 'www');
}

// TF-IDF Vector
function tfidf(url, data) {
    const tokens = tokenize(url);
    const vec = new Array(Object.keys(data.vocabulary).length).fill(0);
    tokens.forEach(t => {
        if (data.vocabulary[t] !== undefined) vec[data.vocabulary[t]]++;
    });
    let sumSq = 0;
    for (let i = 0; i < vec.length; i++) {
        if (vec[i] > 0 && data.sublinear_tf) vec[i] = 1 + Math.log(vec[i]);
        vec[i] *= data.idf[i];
        sumSq += vec[i] * vec[i];
    }
    if (sumSq > 0) {
        const norm = Math.sqrt(sumSq);
        for (let i = 0; i < vec.length; i++) vec[i] /= norm;
    }
    return vec;
}

// Structural Features (9 features)
function structural(url) {
    const s = url.toLowerCase();
    const len = s.length;
    const dots = (s.match(/\./g)||[]).length;
    const slashes = (s.match(/\//g)||[]).length;
    const dashes = (s.match(/-/g)||[]).length;
    const ats = (s.match(/@/g)||[]).length;
    const digits = (s.match(/\d/g)||[]).length;
    const domain = s.replace(/^https?:\/\//,'').split('/')[0];
    const subLevel = (domain.match(/\./g)||[]).length;
    const isTld = COMMON_TLDS.some(t => s.endsWith(t)) ? 1 : 0;
    return [len, dots, slashes, dashes, ats, len > 0 ? digits/len : 0, entropy(s), isTld, subLevel];
}

// Check if URL should be scanned
function shouldScan(url) {
    if (!url || !url.startsWith('http')) return false;
    if (url.startsWith('chrome://') || url.startsWith('chrome-extension://')) return false;
    try {
        const host = new URL(url).hostname.toLowerCase();
        return !WHITELIST.some(d => host.includes(d));
    } catch { return false; }
}

// === XGBoost Preprocessing Functions ===

// StandardScaler: (x - mean) / scale
function scaleFeatures(features, mean, scale) {
    const result = new Array(features.length);
    for (let i = 0; i < features.length; i++) {
        result[i] = (features[i] - mean[i]) / scale[i];
    }
    return result;
}

// Select specific feature indices from a vector
function selectFeatures(features, indices) {
    const result = new Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
        result[i] = features[indices[i]];
    }
    return result;
}

// PCA transform: (X - mean) @ components.T
function pcaTransform(features, pcaMean, pcaComponents) {
    // pcaComponents shape: [n_components, n_features]
    const nComponents = pcaComponents.length;
    const result = new Array(nComponents);
    for (let i = 0; i < nComponents; i++) {
        let sum = 0;
        for (let j = 0; j < features.length; j++) {
            sum += (features[j] - pcaMean[j]) * pcaComponents[i][j];
        }
        result[i] = sum;
    }
    return result;
}

// Full XGBoost preprocessing pipeline: scale → select → PCA
function preprocessXgb(rawFeatures, prepData) {
    const scaled = scaleFeatures(rawFeatures, prepData.scaler_mean, prepData.scaler_scale);
    const selected = selectFeatures(scaled, prepData.selected_feature_indices);
    const pcaResult = pcaTransform(selected, prepData.pca_mean, prepData.pca_components);
    return pcaResult;
}
