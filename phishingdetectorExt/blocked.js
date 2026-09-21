// === BLOCKED PAGE ===
// External file because the extension CSP (`script-src 'self'`) blocks inline
// scripts and inline event handlers in Manifest V3.

const params = new URLSearchParams(location.search);
const blockedUrl = params.get('url') || 'Unknown';
const confidence = parseFloat(params.get('conf')) || 0;
const blockedDomain = blockedUrl.replace(/^https?:\/\//, '').split('/')[0];

function renderReport() {
    document.getElementById('url').textContent = blockedUrl;
    document.getElementById('conf-val').textContent = confidence.toFixed(1) + '%';
    const pct = Math.max(0, Math.min(100, confidence));
    document.getElementById('conf-bar').style.width = pct + '%';
}

function goBack() {
    if (history.length > 1) history.back();
    else location.href = 'https://google.com';
}

function proceed() {
    if (!confirm('⚠️ Are you sure? This site may steal your data.')) return;
    // One-time bypass so the service worker lets this URL through once
    chrome.runtime.sendMessage({ action: 'bypassUrl', url: blockedUrl }, () => {
        location.href = blockedUrl;
    });
}

function trustSite() {
    if (!blockedDomain || blockedDomain === 'Unknown') return;
    chrome.runtime.sendMessage({ action: 'addWhitelist', domain: blockedDomain }, (response) => {
        if (!response || !response.success) return;
        document.getElementById('header').style.background = 'linear-gradient(135deg,#27ae60,#2ecc71)';
        document.getElementById('header').querySelector('h1').textContent = '✓ Site Trusted';
        document.getElementById('header').querySelector('p').textContent = 'Added to whitelist';
        document.getElementById('alert').className = 'alert success';
        document.getElementById('alert').textContent = '✓ ' + blockedDomain + ' has been added to your whitelist.';

        // Rebuild the button row with a real listener (no inline handler, no HTML injection)
        const buttons = document.getElementById('buttons');
        buttons.textContent = '';
        const btn = document.createElement('button');
        btn.className = 'btn btn-safe';
        btn.style.width = '100%';
        btn.textContent = 'Continue to Site →';
        btn.addEventListener('click', () => { location.href = blockedUrl; });
        buttons.appendChild(btn);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    renderReport();
    document.getElementById('btn-back').addEventListener('click', goBack);
    document.getElementById('btn-trust').addEventListener('click', trustSite);
    document.getElementById('btn-continue').addEventListener('click', proceed);
});
