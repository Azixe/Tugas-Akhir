// Content Script - Inject warning banner ke halaman web

// Listen for messages from background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "showWarning") {
        showWarningBanner(request.confidence, request.url);
        sendResponse({ success: true });
    }
    if (request.action === "showSafe") {
        showSafeBadge();
        sendResponse({ success: true });
    }
});

function showWarningBanner(confidence, url) {
    // Remove existing banner if any
    const existing = document.getElementById('phishing-warning-banner');
    if (existing) existing.remove();
    
    // Create banner
    const banner = document.createElement('div');
    banner.id = 'phishing-warning-banner';
    banner.innerHTML = `
        <div class="pw-content">
            <span class="pw-icon">⚠️</span>
            <div class="pw-text">
                <strong>Suspicious Website Detected!</strong>
                <span>This URL has ${confidence.toFixed(1)}% probability of being a phishing site. Proceed with caution.</span>
            </div>
            <div class="pw-buttons">
                <button class="pw-btn pw-btn-details" id="pw-details">Details</button>
                <button class="pw-btn pw-btn-dismiss" id="pw-dismiss">Dismiss</button>
            </div>
        </div>
    `;
    
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        #phishing-warning-banner {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #f39c12 0%, #e74c3c 100%);
            color: white;
            z-index: 2147483647;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            animation: pw-slideDown 0.3s ease-out;
        }
        
        @keyframes pw-slideDown {
            from { transform: translateY(-100%); }
            to { transform: translateY(0); }
        }
        
        @keyframes pw-slideUp {
            from { transform: translateY(0); }
            to { transform: translateY(-100%); }
        }
        
        .pw-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .pw-icon {
            font-size: 28px;
            flex-shrink: 0;
        }
        
        .pw-text {
            flex: 1;
            min-width: 200px;
        }
        
        .pw-text strong {
            display: block;
            font-size: 14px;
            margin-bottom: 2px;
        }
        
        .pw-text span {
            font-size: 12px;
            opacity: 0.95;
        }
        
        .pw-buttons {
            display: flex;
            gap: 10px;
            flex-shrink: 0;
        }
        
        .pw-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .pw-btn-details {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .pw-btn-details:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .pw-btn-dismiss {
            background: white;
            color: #e74c3c;
        }
        
        .pw-btn-dismiss:hover {
            background: #f8f9fa;
            transform: scale(1.05);
        }
        
        /* Push page content down */
        #phishing-warning-banner + * {
            margin-top: 60px;
        }
    `;
    
    document.head.appendChild(style);
    document.body.insertBefore(banner, document.body.firstChild);
    
    // Add body padding to prevent content overlap
    document.body.style.marginTop = '60px';
    
    // Event listeners
    document.getElementById('pw-dismiss').addEventListener('click', () => {
        banner.style.animation = 'pw-slideUp 0.3s ease-out forwards';
        setTimeout(() => {
            banner.remove();
            document.body.style.marginTop = '';
        }, 300);
    });
    
    document.getElementById('pw-details').addEventListener('click', () => {
        showDetailsModal(confidence, url);
    });
}

function showDetailsModal(confidence, url) {
    // Remove existing modal if any
    const existing = document.getElementById('phishing-modal');
    if (existing) existing.remove();
    
    const modal = document.createElement('div');
    modal.id = 'phishing-modal';
    modal.innerHTML = `
        <div class="pm-overlay"></div>
        <div class="pm-content">
            <div class="pm-header">
                <h2>⚠️ Security Warning</h2>
                <button class="pm-close" id="pm-close">&times;</button>
            </div>
            <div class="pm-body">
                <div class="pm-risk">
                    <div class="pm-risk-label">Risk Level</div>
                    <div class="pm-risk-bar">
                        <div class="pm-risk-fill" style="width: ${confidence}%"></div>
                    </div>
                    <div class="pm-risk-value">${confidence.toFixed(1)}% Phishing Probability</div>
                </div>
                
                <div class="pm-url">
                    <strong>URL:</strong>
                    <code>${url}</code>
                </div>
                
                <div class="pm-tips">
                    <h4>🛡️ Safety Tips:</h4>
                    <ul>
                        <li>Don't enter passwords or personal information</li>
                        <li>Check if the domain name is correct</li>
                        <li>Look for HTTPS and valid certificates</li>
                        <li>When in doubt, navigate directly to the official website</li>
                    </ul>
                </div>
            </div>
            <div class="pm-footer">
                <button class="pm-btn pm-btn-leave" id="pm-leave">Leave This Site</button>
                <button class="pm-btn pm-btn-stay" id="pm-stay">I Understand, Continue</button>
            </div>
        </div>
    `;
    
    const style = document.createElement('style');
    style.id = 'phishing-modal-style';
    style.textContent = `
        #phishing-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 2147483647;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .pm-overlay {
            position: absolute;
            inset: 0;
            background: rgba(0,0,0,0.6);
            backdrop-filter: blur(4px);
        }
        
        .pm-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 16px;
            width: 90%;
            max-width: 500px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: pm-popup 0.3s ease-out;
        }
        
        @keyframes pm-popup {
            from { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
            to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        }
        
        .pm-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #eee;
        }
        
        .pm-header h2 {
            margin: 0;
            font-size: 20px;
            color: #e74c3c;
        }
        
        .pm-close {
            background: none;
            border: none;
            font-size: 28px;
            color: #999;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }
        
        .pm-close:hover { color: #333; }
        
        .pm-body {
            padding: 20px;
        }
        
        .pm-risk {
            margin-bottom: 20px;
        }
        
        .pm-risk-label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        .pm-risk-bar {
            height: 12px;
            background: #e9ecef;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .pm-risk-fill {
            height: 100%;
            background: linear-gradient(90deg, #f39c12, #e74c3c);
            border-radius: 6px;
        }
        
        .pm-risk-value {
            margin-top: 8px;
            font-size: 14px;
            color: #e74c3c;
            font-weight: 600;
        }
        
        .pm-url {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            word-break: break-all;
        }
        
        .pm-url code {
            color: #e74c3c;
            font-size: 12px;
        }
        
        .pm-tips h4 {
            margin: 0 0 10px 0;
            color: #333;
        }
        
        .pm-tips ul {
            margin: 0;
            padding-left: 20px;
            color: #666;
            font-size: 14px;
        }
        
        .pm-tips li {
            margin-bottom: 6px;
        }
        
        .pm-footer {
            display: flex;
            gap: 12px;
            padding: 20px;
            border-top: 1px solid #eee;
        }
        
        .pm-btn {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .pm-btn-leave {
            background: #27ae60;
            color: white;
        }
        
        .pm-btn-leave:hover {
            background: #219a52;
        }
        
        .pm-btn-stay {
            background: #f8f9fa;
            color: #666;
            border: 1px solid #ddd;
        }
        
        .pm-btn-stay:hover {
            background: #e9ecef;
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(modal);
    
    // Events
    document.getElementById('pm-close').addEventListener('click', closeModal);
    document.querySelector('.pm-overlay').addEventListener('click', closeModal);
    document.getElementById('pm-stay').addEventListener('click', closeModal);
    document.getElementById('pm-leave').addEventListener('click', () => {
        window.location.href = 'https://www.google.com';
    });
    
    function closeModal() {
        modal.remove();
        document.getElementById('phishing-modal-style')?.remove();
    }
}

function showSafeBadge() {
    // Optional: Show small safe indicator
    const existing = document.getElementById('phishing-safe-badge');
    if (existing) return;
    
    const badge = document.createElement('div');
    badge.id = 'phishing-safe-badge';
    badge.innerHTML = '✓ Verified Safe';
    badge.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #27ae60;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-family: -apple-system, sans-serif;
        font-size: 12px;
        font-weight: 600;
        z-index: 2147483647;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        animation: fadeInOut 3s ease-in-out forwards;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeInOut {
            0% { opacity: 0; transform: translateY(20px); }
            15% { opacity: 1; transform: translateY(0); }
            85% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(-20px); }
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(badge);
    
    setTimeout(() => badge.remove(), 3000);
}