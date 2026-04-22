// === CONTENT SCRIPT - Warning Banner ===
chrome.runtime.onMessage.addListener((msg, _, res) => {
    if (msg.action === 'warn') showBanner(msg.confidence, msg.url);
    res({ ok: true });
});

function showBanner(conf, url) {
    if (document.getElementById('pw-banner')) return;
    
    const domain = url.replace(/^https?:\/\//, '').split('/')[0];
    
    const banner = document.createElement('div');
    banner.id = 'pw-banner';
    banner.innerHTML = `
        <style>
            #pw-banner{position:fixed;top:0;left:0;right:0;background:linear-gradient(90deg,#f39c12,#e74c3c);color:#fff;z-index:2147483647;font-family:system-ui,sans-serif;box-shadow:0 2px 10px rgba(0,0,0,.3);animation:slideIn .3s}
            @keyframes slideIn{from{transform:translateY(-100%)}to{transform:translateY(0)}}
            @keyframes slideOut{from{transform:translateY(0)}to{transform:translateY(-100%)}}
            #pw-banner .wrap{max-width:1200px;margin:auto;padding:10px 16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
            #pw-banner .icon{font-size:24px}
            #pw-banner .text{flex:1;min-width:200px}
            #pw-banner .text b{display:block;font-size:13px}
            #pw-banner .text span{font-size:11px;opacity:.9}
            #pw-banner button{padding:6px 14px;border:none;border-radius:4px;font-size:12px;font-weight:600;cursor:pointer}
            #pw-banner .trust{background:#27ae60;color:#fff}
            #pw-banner .detail{background:rgba(255,255,255,.2);color:#fff;border:1px solid rgba(255,255,255,.3)}
            #pw-banner .dismiss{background:#fff;color:#e74c3c}
        </style>
        <div class="wrap">
            <span class="icon">⚠️</span>
            <div class="text">
                <b>Suspicious Website Detected</b>
                <span>${conf.toFixed(1)}% phishing probability. Proceed with caution.</span>
            </div>
            <button class="trust" id="pw-trust">✓ Trust Site</button>
            <button class="detail" id="pw-detail">Details</button>
            <button class="dismiss" id="pw-dismiss">Dismiss</button>
        </div>
    `;
    document.body.prepend(banner);
    document.body.style.marginTop = '50px';
    
    // Dismiss
    document.getElementById('pw-dismiss').onclick = () => {
        banner.style.animation = 'slideOut .3s forwards';
        setTimeout(() => { banner.remove(); document.body.style.marginTop = ''; }, 300);
    };
    
    // Trust site - add to whitelist
    document.getElementById('pw-trust').onclick = () => {
        chrome.runtime.sendMessage({ action: 'addWhitelist', domain }, (response) => {
            if (response?.success) {
                banner.style.background = 'linear-gradient(90deg,#27ae60,#2ecc71)';
                banner.querySelector('.text b').textContent = '✓ Site Added to Whitelist';
                banner.querySelector('.text span').textContent = domain + ' will no longer be flagged.';
                banner.querySelectorAll('button').forEach(b => b.style.display = 'none');
                setTimeout(() => {
                    banner.style.animation = 'slideOut .3s forwards';
                    setTimeout(() => { banner.remove(); document.body.style.marginTop = ''; }, 300);
                }, 2000);
            }
        });
    };
    
    // Details modal
    document.getElementById('pw-detail').onclick = () => showModal(conf, url, domain);
}

function showModal(conf, url, domain) {
    if (document.getElementById('pw-modal')) return;
    const modal = document.createElement('div');
    modal.id = 'pw-modal';
    modal.innerHTML = `
        <style>
            #pw-modal{position:fixed;inset:0;z-index:2147483647;font-family:system-ui,sans-serif}
            #pw-modal .overlay{position:absolute;inset:0;background:rgba(0,0,0,.6);backdrop-filter:blur(3px)}
            #pw-modal .box{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;border-radius:12px;width:90%;max-width:400px;box-shadow:0 10px 40px rgba(0,0,0,.3)}
            #pw-modal .header{padding:16px;border-bottom:1px solid #eee;display:flex;justify-content:space-between;align-items:center}
            #pw-modal h3{margin:0;color:#e74c3c;font-size:16px}
            #pw-modal .close{background:none;border:none;font-size:24px;cursor:pointer;color:#999}
            #pw-modal .body{padding:16px}
            #pw-modal .bar{height:8px;background:#eee;border-radius:4px;overflow:hidden;margin:8px 0}
            #pw-modal .fill{height:100%;background:linear-gradient(90deg,#f39c12,#e74c3c);width:${conf}%}
            #pw-modal .url{background:#f5f5f5;padding:10px;border-radius:6px;word-break:break-all;font-size:11px;color:#e74c3c;margin:12px 0}
            #pw-modal .footer{padding:16px;border-top:1px solid #eee;display:flex;gap:10px}
            #pw-modal .btn{flex:1;padding:10px;border:none;border-radius:6px;font-weight:600;cursor:pointer;font-size:12px}
            #pw-modal .leave{background:#27ae60;color:#fff}
            #pw-modal .trust{background:#3498db;color:#fff}
            #pw-modal .stay{background:#f5f5f5;color:#666}
        </style>
        <div class="overlay" id="pm-overlay"></div>
        <div class="box">
            <div class="header"><h3>⚠️ Security Warning</h3><button class="close" id="pm-close">&times;</button></div>
            <div class="body">
                <div><b>Risk Level</b><div class="bar"><div class="fill"></div></div><span style="color:#e74c3c;font-size:13px">${conf.toFixed(1)}% Phishing Probability</span></div>
                <div class="url"><b>URL:</b> ${url}</div>
            </div>
            <div class="footer">
                <button class="btn leave" id="pm-leave">Leave</button>
                <button class="btn trust" id="pm-trust">Trust Site</button>
                <button class="btn stay" id="pm-stay">Continue</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    
    const closeModal = () => modal.remove();
    document.getElementById('pm-overlay').onclick = closeModal;
    document.getElementById('pm-close').onclick = closeModal;
    document.getElementById('pm-stay').onclick = closeModal;
    document.getElementById('pm-leave').onclick = () => location.href = 'https://google.com';
    document.getElementById('pm-trust').onclick = () => {
        chrome.runtime.sendMessage({ action: 'addWhitelist', domain }, (response) => {
            if (response?.success) {
                closeModal();
                document.getElementById('pw-banner')?.remove();
                document.body.style.marginTop = '';
            }
        });
    };
}

