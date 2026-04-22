// === POPUP SCRIPT ===
// Delegates all inference to background.js via message passing

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
    const modelSelect = $('model-select');
    const modelTag = $('model-tag');
    
    let currentUrl = '';
    
    // --- Model Selector ---
    // Get current model from background
    chrome.runtime.sendMessage({ action: 'getModel' }, (response) => {
        if (response && response.model) {
            modelSelect.value = response.model;
        }
    });
    
    // Handle model switch
    modelSelect.onchange = () => {
        const newModel = modelSelect.value;
        status.textContent = `Switching to ${newModel === 'rf' ? 'Random Forest' : 'XGBoost'}...`;
        btn.disabled = true;
        result.style.display = 'none';
        
        chrome.runtime.sendMessage({ action: 'switchModel', model: newModel }, (response) => {
            if (response && response.success) {
                status.textContent = `${newModel === 'rf' ? 'Random Forest' : 'XGBoost'} ready`;
                btn.disabled = false;
                setTimeout(() => { if (status.textContent.includes('ready')) status.textContent = ''; }, 2000);
            } else {
                status.textContent = 'Error switching model';
            }
        });
    };
    
    // --- Tab Switching ---
    document.querySelectorAll('.tab').forEach(tab => {
        tab.onclick = () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            $('panel-' + tab.dataset.tab).classList.add('active');
            if (tab.dataset.tab === 'whitelist') loadWhitelist();
        };
    });
    
    // --- Whitelist ---
    async function loadWhitelist() {
        const list = await getWhitelist();
        renderWhitelist(list);
    }
    
    $('btn-add').onclick = async () => {
        const input = $('wl-input');
        if (input.value.trim()) {
            const list = await addToWhitelist(input.value.trim());
            renderWhitelist(list);
            input.value = '';
        }
    };
    
    $('btn-add-current').onclick = async () => {
        if (currentUrl) {
            const list = await addToWhitelist(currentUrl);
            renderWhitelist(list);
            status.textContent = 'Added to whitelist!';
            setTimeout(() => status.textContent = '', 2000);
        }
    };
    
    // --- Get Current Tab URL ---
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    currentUrl = tab?.url || '';
    urlDiv.textContent = currentUrl || 'No URL';
    
    // Enable scan button (background handles model loading)
    if (currentUrl && currentUrl.startsWith('http')) {
        btn.disabled = false;
        status.textContent = '';
    } else {
        status.textContent = 'No scannable URL';
    }
    
    // --- Scan Button ---
    btn.onclick = async () => {
        const url = currentUrl;
        
        if (!url?.startsWith('http')) {
            status.textContent = 'Invalid URL';
            return;
        }
        
        btn.disabled = true;
        status.textContent = 'Scanning...';
        result.style.display = 'none';
        
        // Send scan request to background.js (which handles ONNX inference)
        chrome.runtime.sendMessage({ action: 'scan', url }, (r) => {
            btn.disabled = false;
            status.textContent = '';
            
            if (!r || r.error) {
                status.textContent = 'Error: ' + (r?.error || 'No response');
                return;
            }
            
            // Display result
            const label = r.isPhishing;
            const prob = r.confidence;
            const modelName = r.model === 'xgb' ? 'XGBoost' : 'Random Forest';
            const verdictIcon = $('verdict-icon');
            
            result.style.display = 'block';
            if (label && prob > 80) {
                verdictIcon.textContent = '🚨';
                verdict.textContent = 'PHISHING';
                result.className = 'result-box danger';
                conf.textContent = `High Risk (${prob.toFixed(1)}%)`;
            } else if (label && prob > 60) {
                verdictIcon.textContent = '⚠️';
                verdict.textContent = 'SUSPICIOUS';
                result.className = 'result-box warning';
                conf.textContent = `Medium Risk (${prob.toFixed(1)}%)`;
            } else {
                verdictIcon.textContent = '✅';
                verdict.textContent = 'SAFE';
                result.className = 'result-box safe';
                conf.textContent = `Confidence: ${(100-prob).toFixed(1)}%`;
            }
            
            modelTag.textContent = `${modelName} · ${r.inferenceTime.toFixed(1)}ms`;
        });
    };
});
