// === POPUP SCRIPT ===
ort.env.wasm.wasmPaths = chrome.runtime.getURL('libs/');
ort.env.wasm.numThreads = 1;

const $ = id => document.getElementById(id);

// Whitelist Management
async function getWhitelist() {
    const { userWhitelist = [] } = await chrome.storage.local.get('userWhitelist');
    return userWhitelist;
}

async function addToWhitelist(domain) {
    const list = await getWhitelist();
    domain = domain.toLowerCase().replace(/^https?:\/\//, '').split('/')[0];
    if (!list.includes(domain) && domain) {
        list.push(domain);
        await chrome.storage.local.set({ userWhitelist: list });
    }
    return list;
}

async function removeFromWhitelist(domain) {
    let list = await getWhitelist();
    list = list.filter(d => d !== domain);
    await chrome.storage.local.set({ userWhitelist: list });
    return list;
}

function renderWhitelist(list) {
    const container = $('wl-list');
    if (list.length === 0) {
        container.innerHTML = '<div class="empty">No whitelisted domains</div>';
        return;
    }
    container.innerHTML = list.map(d => `
        <div class="wl-item">
            <span>${d}</span>
            <button class="del" data-domain="${d}">&times;</button>
        </div>
    `).join('');
    
    container.querySelectorAll('.del').forEach(btn => {
        btn.onclick = async () => {
            const newList = await removeFromWhitelist(btn.dataset.domain);
            renderWhitelist(newList);
        };
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    const status = $('status'), result = $('result'), verdict = $('verdict');
    const conf = $('confidence'), btn = $('btn-scan'), urlDiv = $('url-display');
    
    let session, data, currentUrl = '';
    
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.onclick = () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            $('panel-' + tab.dataset.tab).classList.add('active');
            if (tab.dataset.tab === 'whitelist') loadWhitelist();
        };
    });
    
    // Load whitelist
    async function loadWhitelist() {
        const list = await getWhitelist();
        renderWhitelist(list);
    }
    
    // Add domain button
    $('btn-add').onclick = async () => {
        const input = $('wl-input');
        if (input.value.trim()) {
            const list = await addToWhitelist(input.value.trim());
            renderWhitelist(list);
            input.value = '';
        }
    };
    
    // Add current site button
    $('btn-add-current').onclick = async () => {
        if (currentUrl) {
            const list = await addToWhitelist(currentUrl);
            renderWhitelist(list);
            status.textContent = 'Added to whitelist!';
            setTimeout(() => status.textContent = '', 2000);
        }
    };
    
    // Get current tab URL
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    currentUrl = tab?.url || '';
    urlDiv.textContent = currentUrl || 'No URL';
    
    // Load model
    try {
        status.textContent = 'Loading...';
        const [tRes, mRes] = await Promise.all([
            fetch(chrome.runtime.getURL('tfidf_data.json')),
            fetch(chrome.runtime.getURL('phishing_rf.onnx'))
        ]);
        data = await tRes.json();
        session = await ort.InferenceSession.create(await mRes.arrayBuffer(), { executionProviders: ['wasm'] });
        status.textContent = 'Ready';
        btn.disabled = false;
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
        return;
    }
    
    // Scan button
    btn.onclick = async () => {
        const url = currentUrl;
        
        if (!url?.startsWith('http')) {
            status.textContent = 'Invalid URL';
            return;
        }
        
        btn.disabled = true;
        status.textContent = 'Scanning...';
        result.style.display = 'none';
        
        try {
            // Extract features using inline functions (same as utils.js)
            const s = url.toLowerCase();
            
            // Tokenize
            let tokens = [];
            s.split('/').forEach(p => {
                const byDash = p.split('-');
                let byDot = [];
                byDash.forEach(t => byDot = byDot.concat(t.split('.')));
                tokens = tokens.concat(byDash, byDot);
            });
            tokens = [...new Set(tokens)].filter(t => t && t !== 'com' && t !== 'www');
            
            // TF-IDF
            const vec = new Array(Object.keys(data.vocabulary).length).fill(0);
            tokens.forEach(t => { if (data.vocabulary[t] !== undefined) vec[data.vocabulary[t]]++; });
            let sumSq = 0;
            for (let i = 0; i < vec.length; i++) {
                if (vec[i] > 0 && data.sublinear_tf) vec[i] = 1 + Math.log(vec[i]);
                vec[i] *= data.idf[i];
                sumSq += vec[i] * vec[i];
            }
            if (sumSq > 0) { const n = Math.sqrt(sumSq); for (let i = 0; i < vec.length; i++) vec[i] /= n; }
            
            // Structural
            const len = s.length;
            const cnt = p => (s.match(p) || []).length;
            const domain = s.replace(/^https?:\/\//, '').split('/')[0];
            const freq = {};
            for (const c of s) freq[c] = (freq[c]||0) + 1;
            let ent = 0;
            for (const v of Object.values(freq)) { const p = v/len; ent -= p * Math.log2(p); }
            const tlds = ['.com','.org','.net','.edu','.gov','.id','.co.id'];
            //const struct = [len, cnt(/\./g), cnt(/\//g), cnt(/-/g), cnt(/@/g), cnt(/\d/g)/len, ent, tlds.some(t=>s.endsWith(t))?1:0, Math.max(0,(domain.match(/\./g)||[]).length-1)];
            const struct = [len, cnt(/\./g), cnt(/\//g), cnt(/-/g), cnt(/@/g), cnt(/\d/g)/len, ent, tlds.some(t=>s.endsWith(t))?1:0, (domain.match(/\./g)||[]).length];
            
            // Predict
            const features = [...vec, ...struct];
            const input = new ort.Tensor('float32', Float32Array.from(features), [1, features.length]);
            const results = await session.run({ [session.inputNames[0]]: input });
            const label = Number(results[session.outputNames[0]].data[0]);
            const probs = results[session.outputNames[1]].data;
            const prob = (label === 1 ? probs[1] : probs[0]) * 100;
            
            // Display
            result.style.display = 'block';
            if (label === 1 && prob > 80) {
                verdict.textContent = ' PHISHING';
                result.className = 'result-box danger';
                conf.textContent = `High Risk (${prob.toFixed(1)}%)`;
            } else if (label === 1 && prob > 60) {
                verdict.textContent = ' SUSPICIOUS';
                result.className = 'result-box warning';
                conf.textContent = `Medium Risk (${prob.toFixed(1)}%)`;
            } else {
                verdict.textContent = ' SAFE';
                result.className = 'result-box safe';
                conf.textContent = `Confidence: ${(100-prob).toFixed(1)}%`;
            }
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
        }
        
        btn.disabled = false;
        status.textContent = '';
    };
});
