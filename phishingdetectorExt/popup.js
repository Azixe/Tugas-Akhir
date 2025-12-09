// Konfigurasi ONNX Runtime - FIXED
// Harus pakai folder path dengan trailing slash!
ort.env.wasm.wasmPaths = chrome.runtime.getURL("libs/");

// Disable threading (tidak support di extension popup)
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

document.addEventListener('DOMContentLoaded', async () => {
    const statusDiv = document.getElementById('status');
    const resultDiv = document.getElementById('result');
    const verdictH2 = document.getElementById('verdict');
    const confDiv = document.getElementById('confidence');
    const btnScan = document.getElementById('btn-scan');
    const urlDisplay = document.getElementById('url-display');

    let session = null;
    let tfidfData = null;

    // --- 1. Load Model & Vocabulary ---
    try {
        statusDiv.innerText = "Loading model...";
        
        // Load TF-IDF data
        const tfidfResponse = await fetch(chrome.runtime.getURL("tfidf_data.json"));
        tfidfData = await tfidfResponse.json();
        
        // Load ONNX model dengan explicit options
        const modelUrl = chrome.runtime.getURL("phishing_rf.onnx");
        const modelResponse = await fetch(modelUrl);
        const modelBuffer = await modelResponse.arrayBuffer();
        
        session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        statusDiv.innerText = "Ready to scan!";
        btnScan.disabled = false;
        
    } catch (e) {
        statusDiv.innerText = "Error Init: " + e.message;
        console.error("Init Error:", e);
        btnScan.disabled = true;
    }

    // --- 2. Fungsi Tombol Scan ---
    btnScan.addEventListener('click', async () => {
        // Ambil URL dari tab aktif
        chrome.tabs.query({active: true, currentWindow: true}, async (tabs) => {
            const currentUrl = tabs[0].url;
            urlDisplay.innerText = currentUrl;
            
            // Validasi URL
            if (!currentUrl || !currentUrl.startsWith('http')) {
                statusDiv.innerText = "Hanya bisa scan URL http/https";
                statusDiv.style.color = "orange";
                return;
            }
            
            // UI Loading
            btnScan.disabled = true;
            statusDiv.innerText = "Menganalisis...";
            resultDiv.style.display = "none";

            try {
                // Proses Prediksi
                const prediction = await predictUrl(currentUrl, session, tfidfData);
                
                // Tampilkan Hasil
                displayResult(prediction);
            } catch (e) {
                statusDiv.innerText = `Error: ${e.message}`;
            } finally {
                btnScan.disabled = false;
                statusDiv.innerText = "";
            }
        });
    });

    // Helper: Tampilkan Hasil
    function displayResult(pred) {
        const { label, probability } = pred;
        resultDiv.style.display = "block";
        
        if (probability > 80) {
            verdictH2.innerText = "PHISHING DETECTED";
            verdictH2.style.color = "#721c24";
            resultDiv.className = "result-box phishing";
            confDiv.innerText = `Bahaya Tinggi (${probability.toFixed(2)}%)`;
        } else if (probability > 50) {
            verdictH2.innerText = "SUSPICIOUS";
            verdictH2.style.color = "#856404";
            resultDiv.className = "result-box"; // Bisa tambah class kuning
            resultDiv.style.backgroundColor = "#fff3cd";
            resultDiv.style.borderColor = "#ffeeba";
            confDiv.innerText = `Mencurigakan (${probability.toFixed(2)}%)`;
        } else {
            verdictH2.innerText = "SAFE";
            verdictH2.style.color = "#155724";
            resultDiv.className = "result-box safe";
            confDiv.innerText = `Aman (Confidence: ${(100 - probability).toFixed(2)}%)`;
        }
    }
});

// ==========================================
// --- LOGIC ENGINE (PYTHON TRANSLATION) ---
// ==========================================

async function predictUrl(url, session, tfidfData) {
    // 1. Preprocessing
    const s_url = url.toLowerCase();

    // 2. Ekstraksi Fitur
    const textVector = computeTfidf(s_url, tfidfData);
    const structVector = computeStructuralFeatures(url); 

    // 3. Gabungkan
    const combinedFeatures = [...textVector, ...structVector];
    
    // 4. Buat Tensor
    const inputTensor = new ort.Tensor('float32', Float32Array.from(combinedFeatures), [1, combinedFeatures.length]);

    // 5. Jalankan Inference
    const feeds = { float_input: inputTensor };
    
    // Debugging: Cek dulu nama output model Anda di console
    // console.log("Starting inference...");
    
    const results = await session.run(feeds);
    
    // console.log("Raw ONNX Results:", results); 

    // 6. Parsing Output (KHUSUS ZIPMAP: FALSE)
    // Output biasanya bernama 'output_label' (Int64) dan 'output_probability' (Float32)
    // Tapi kadang scikit-learn menamakannya 'label' dan 'probabilities'
    
    // Kita cari output yang merupakan probabilitas (biasanya index ke-2 di object keys, atau cari nama yang mengandung 'prob')
    let probTensor = results.probabilities || results.output_probability;
    let labelTensor = results.label || results.output_label;

    // Fallback: Jika nama tidak ketemu, ambil berdasarkan urutan keys
    if (!probTensor) {
        const keys = Object.keys(results);
        // Biasanya keys[1] adalah probabilitas
        probTensor = results[keys[1]];
        labelTensor = results[keys[0]];
    }

    // Ambil Label (0 atau 1)
    const label = Number(labelTensor.data[0]); 
    
    // Ambil Probabilitas Phishing
    // Karena ZipMap False, data berbentuk Flat Array: [Prob_Kelas0, Prob_Kelas1]
    // Kita mau Prob_Kelas1 (Phishing)
    const probPhishing = probTensor.data[1] * 100; 

    return { label: label, probability: probPhishing };
}

// --- LOGIC 1: TF-IDF VECTORIZER (Manual Re-implementation) ---
function computeTfidf(url, data) {
    const vocab = data.vocabulary;
    const idf = data.idf;
    const tokens = makeTokens(url);
    const vectorSize = Object.keys(vocab).length;
    const tfVector = new Array(vectorSize).fill(0);

    // Hitung Term Frequency (TF)
    const termCounts = {};
    tokens.forEach(token => {
        if (vocab.hasOwnProperty(token)) {
            termCounts[token] = (termCounts[token] || 0) + 1;
        }
    });

    // Hitung TF-IDF
    // Rumus sklearn default: tf * idf (dengan sublinear_tf=False, norm='l2' default)
    // Cek tfidf_data.json parameter Anda. Asumsi standard L2 norm.
    
    let sumSquares = 0;

    for (const [token, count] of Object.entries(termCounts)) {
        const idx = vocab[token];
        const tf = count; // Jika sublinear_tf=True -> 1 + Math.log(count)
        const val = tf * idf[idx];
        tfVector[idx] = val;
        sumSquares += val * val;
    }

    // L2 Normalization (Wajib agar sama dengan Python)
    if (sumSquares > 0) {
        const norm = Math.sqrt(sumSquares);
        for (let i = 0; i < vectorSize; i++) {
            if (tfVector[i] !== 0) {
                tfVector[i] /= norm;
            }
        }
    }

    return tfVector;
}

// --- LOGIC 2: TOKENIZER (FIXED) ---
function makeTokens(url) {
    // Harus PERSIS sama dengan Python training
    const partsBySlash = url.split('/');
    let totalTokens = [];
    
    partsBySlash.forEach(part => {
        const tokensByDash = part.split('-');
        let tokensDot = [];
        
        tokensByDash.forEach(token => {
            const tempTokens = token.split('.');
            tokensDot = tokensDot.concat(tempTokens);
        });
        
        // Python logic: total_tokens = total_tokens + tokens + tokens_dot
        totalTokens = totalTokens.concat(tokensByDash).concat(tokensDot);
    });
    
    // Unique + filter empty strings + remove 'com' & 'www'
    const uniqueTokens = [...new Set(totalTokens)];
    return uniqueTokens.filter(t => t && t !== 'com' && t !== 'www');
}

// --- LOGIC 3: STRUCTURAL FEATURES ---
function computeStructuralFeatures(originalUrl) {
    const s_url = originalUrl.toLowerCase();
    
    // 1. Length
    const length = s_url.length;
    
    // 2. Counts
    const dot_count = (s_url.match(/\./g) || []).length;
    const slash_count = (s_url.match(/\//g) || []).length;
    const dash_count = (s_url.match(/-/g) || []).length;
    const at_count = (s_url.match(/@/g) || []).length;
    
    // 3. Digit Ratio
    const digit_count = (s_url.match(/\d/g) || []).length;
    const digit_ratio = length > 0 ? digit_count / length : 0;
    
    // 4. Shannon Entropy
    const entropy = calculateEntropy(s_url);
    
    // 5. Common TLD
    const common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.id', '.co.id'];
    let is_common_tld = 0;
    for (const tld of common_tlds) {
        if (s_url.endsWith(tld) || s_url.endsWith(tld + '/')) {
            is_common_tld = 1;
            break;
        }
    }
    
    // 6. Subdomain Level
    let subdomain_level = 0;
    try {
        let clean = s_url.replace("https://", "").replace("http://", "").split('/')[0];
        subdomain_level = (clean.match(/\./g) || []).length;
    } catch (e) { subdomain_level = 0; }
    
    // 7. Has IP (Simple Check)
    let has_ip = 0;
    try {
        const domain = s_url.replace("https://", "").replace("http://", "").split('/')[0];
        const parts = domain.split('.');
        if (parts.length === 4 && parts.every(p => !isNaN(parseInt(p)))) {
            has_ip = 1;
        }
    } catch (e) { has_ip = 0; }

    // Return Array (URUTAN WAJIB SAMA DENGAN PYTHON)
    // [length, dot_count, slash_count, dash_count, at_count, digit_ratio, entropy, is_common_tld, subdomain_level]
    // Perhatikan: di export_to_web.py jumlah fitur manual adalah 9.
    // Di class StructuralFeatureExtractor Python Anda, urutannya:
    // length, dot, slash, dash, at, digit_ratio, entropy, is_common, subdomain.
    // (has_ip sepertinya Anda hapus di versi "Optimized" terakhir? atau masih ada?)
    
    // CEK ULANG CODE PYTHON OPTIMIZED ANDA TERAKHIR:
    // features.append([length, dot_count, slash_count, dash_count, at_count, digit_ratio, entropy, is_common_tld, subdomain_level])
    // Ternyata 'has_ip' TIDAK ADA di versi optimized terakhir yang saya berikan.
    // JADI SAYA HAPUS JUGA DI SINI.
    
    return [length, dot_count, slash_count, dash_count, at_count, digit_ratio, entropy, is_common_tld, subdomain_level];
}

function calculateEntropy(str) {
    if (!str) return 0;
    const len = str.length;
    const frequencies = {};
    for (let char of str) {
        frequencies[char] = (frequencies[char] || 0) + 1;
    }
    
    let entropy = 0;
    for (let count of Object.values(frequencies)) {
        const p = count / len;
        entropy -= p * Math.log2(p);
    }
    return entropy;
}