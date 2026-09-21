// === BENCHMARK PAGE SCRIPT (P-01 latency) ===
// Drives the extension's own scan pipeline through the background service
// worker (the same "scan" message path the popup uses) over the shared
// 1000-URL test set, for each model. Results are downloadable as JSON.

const MODELS = ['rf', 'xgb'];
const WARMUP_COUNT = 5;
const MAX_CONSECUTIVE_ERRORS = 10;

const state = {
    urls: [],
    running: false,
    cancel: false,
    results: {},          // model -> { initMs, wallMs, aborted, samples }
};

const $ = id => document.getElementById(id);
const round = (x, d = 2) => Math.round(x * 10 ** d) / 10 ** d;

function percentile(sorted, p) {
    if (!sorted.length) return 0;
    const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
    return sorted[idx];
}

function stats(values) {
    if (!values.length) return { n: 0, min: 0, median: 0, avg: 0, p95: 0, max: 0 };
    const sorted = [...values].sort((a, b) => a - b);
    return {
        n: values.length,
        min: round(sorted[0]),
        median: round(percentile(sorted, 50)),
        avg: round(values.reduce((a, b) => a + b, 0) / values.length),
        p95: round(percentile(sorted, 95)),
        max: round(sorted[sorted.length - 1]),
    };
}

function verdictCounts(samples) {
    const counts = { PHISHING: 0, SUSPICIOUS: 0, SAFE: 0 };
    for (const s of samples) {
        if (s.error || typeof s.phishingProbability !== 'number') continue;
        if (s.isPhishing && s.phishingProbability > 80) counts.PHISHING++;
        else if (s.isPhishing && s.phishingProbability > 60) counts.SUSPICIOUS++;
        else counts.SAFE++;
    }
    return counts;
}

function log(message) {
    const line = `[${new Date().toLocaleTimeString()}] ${message}`;
    const box = $('log');
    box.textContent += line + '\n';
    box.scrollTop = box.scrollHeight;
}

function setControls(enabled) {
    for (const id of ['run-rf', 'run-xgb', 'run-both', 'url-count']) $(id).disabled = !enabled;
    $('stop').disabled = enabled;
}

function setProgress(done, total) {
    $('bar-fill').style.width = total ? `${(done / total) * 100}%` : '0%';
}

function renderSummary(model) {
    const run = state.results[model];
    if (!run) return;
    const ok = run.samples.filter(s => !s.error);
    const rt = stats(ok.map(s => s.roundTripMs));
    const pipe = stats(ok.map(s => s.inferenceTimeMs).filter(v => typeof v === 'number'));
    const verdicts = verdictCounts(run.samples);

    const tr = $('row-' + model);
    tr.children[1].textContent = run.initMs.toFixed(1);
    tr.children[2].textContent = String(run.samples.length);
    tr.children[3].textContent = String(rt.min);
    tr.children[4].textContent = String(rt.median);
    tr.children[5].textContent = String(rt.avg);
    tr.children[6].textContent = String(rt.p95);
    tr.children[7].textContent = String(rt.max);
    tr.children[8].textContent = String(pipe.avg);
    tr.children[9].textContent = String(pipe.p95);
    tr.children[10].textContent = String(run.samples.length - ok.length);
    $('verdicts-' + model).textContent =
        `PHISHING ${verdicts.PHISHING} · SUSPICIOUS ${verdicts.SUSPICIOUS} · SAFE ${verdicts.SAFE}` +
        (run.aborted ? ' · run aborted early' : '');
}

async function runModel(model, count) {
    log(`Switching to ${model.toUpperCase()} (init timing starts)...`);
    const initT0 = performance.now();
    const sw = await chrome.runtime.sendMessage({ action: 'switchModel', model });
    const initMs = performance.now() - initT0;
    if (!sw || !sw.success) throw new Error(`${model} model failed to initialize`);
    log(`${model.toUpperCase()} ready, init = ${initMs.toFixed(1)} ms`);

    const warm = Math.min(WARMUP_COUNT, count);
    for (let i = 0; i < warm && !state.cancel; i++) {
        await chrome.runtime.sendMessage({ action: 'scan', url: state.urls[i] }).catch(() => {});
    }
    if (warm) log(`${warm} warm-up scans done`);

    const samples = [];
    const wallT0 = performance.now();
    let consecutiveErrors = 0;
    let aborted = false;

    for (let i = 0; i < count; i++) {
        if (state.cancel) { aborted = true; break; }
        const url = state.urls[i];
        const start = performance.now();
        let res;
        try {
            res = await chrome.runtime.sendMessage({ action: 'scan', url });
        } catch (e) {
            res = { error: e && e.message ? e.message : String(e) };
        }
        const roundTripMs = performance.now() - start;

        if (res && res.error) {
            consecutiveErrors++;
            if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) aborted = true;
        } else {
            consecutiveErrors = 0;
        }

        samples.push({
            i,
            url,
            roundTripMs: round(roundTripMs, 3),
            inferenceTimeMs: res && typeof res.inferenceTime === 'number' ? round(res.inferenceTime, 3) : null,
            isPhishing: res ? res.isPhishing ?? null : null,
            phishingProbability: res ? res.phishingProbability ?? null : null,
            legitimateProbability: res ? res.legitimateProbability ?? null : null,
            model: res ? res.model ?? null : null,
            error: res ? res.error ?? null : null,
        });

        if (i % 10 === 0 || i === count - 1) {
            setProgress(i + 1, count);
            $('status').textContent = `${model.toUpperCase()}: ${i + 1} / ${count}`;
        }
        if (aborted) { log(`Aborting ${model.toUpperCase()} after ${consecutiveErrors} consecutive errors`); break; }
    }

    const wallMs = performance.now() - wallT0;
    state.results[model] = { initMs, wallMs, aborted, samples };
    renderSummary(model);
    log(`${model.toUpperCase()} done: ${samples.length} samples in ${(wallMs / 1000).toFixed(1)} s`);
}

async function runModels(models) {
    if (state.running) return;
    if (!state.urls.length) { $('status').textContent = 'URL list not loaded'; return; }

    state.running = true;
    state.cancel = false;
    setControls(false);
    $('download').disabled = true;

    const requested = parseInt($('url-count').value, 10) || state.urls.length;
    const count = Math.max(1, Math.min(requested, state.urls.length));

    try {
        for (const model of models) {
            if (state.cancel) break;
            await runModel(model, count);
        }
    } catch (e) {
        log(`ERROR: ${e.message}`);
    }

    const restore = await chrome.runtime.sendMessage({ action: 'switchModel', model: 'rf' }).catch(() => null);
    log(restore && restore.success ? 'Model restored to RF' : 'WARNING: could not restore RF');

    state.running = false;
    setControls(true);
    $('download').disabled = Object.keys(state.results).length === 0;
    $('status').textContent = state.cancel ? 'Cancelled' : 'Finished';
    setProgress(0, 1);
    log('All runs complete. Use "Download JSON" and send the file.');
}

async function sha256Hex(path) {
    const res = await fetch(chrome.runtime.getURL(path));
    const buf = await res.arrayBuffer();
    const digest = await crypto.subtle.digest('SHA-256', buf);
    const hex = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
    return { sha256: hex, bytes: buf.byteLength };
}

async function buildResults() {
    const manifest = chrome.runtime.getManifest();
    const files = {};
    for (const path of ['phishing_rf.onnx', 'phishing_xgb.onnx', 'tfidf_data.json', 'tfidf_data_xgb.json', 'xgb_preprocessing.json']) {
        files[path] = await sha256Hex(path);
    }
    const runs = {};
    for (const model of Object.keys(state.results)) {
        const run = state.results[model];
        const ok = run.samples.filter(s => !s.error);
        runs[model] = {
            initMs: round(run.initMs, 3),
            wallMs: round(run.wallMs, 3),
            aborted: run.aborted,
            summary: {
                samples: run.samples.length,
                errors: run.samples.length - ok.length,
                verdicts: verdictCounts(run.samples),
                roundTripMs: stats(ok.map(s => s.roundTripMs)),
                inferenceTimeMs: stats(ok.map(s => s.inferenceTimeMs).filter(v => typeof v === 'number')),
            },
            samples: run.samples,
        };
    }
    return {
        meta: {
            generatedAt: new Date().toISOString(),
            extensionVersion: manifest.version,
            userAgent: navigator.userAgent,
            hardwareConcurrency: navigator.hardwareConcurrency,
            deviceMemoryGB: navigator.deviceMemory ?? null,
            urlSource: 'bench_urls.json (common test split, seed 42, same as the Python benchmark)',
            urlsInFile: state.urls.length,
            urlsRequestedPerModel: parseInt($('url-count').value, 10) || state.urls.length,
            warmupScans: WARMUP_COUNT,
        },
        modelFiles: files,
        runs,
    };
}

function downloadResults() {
    buildResults().then(data => {
        const ts = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
        const name = `bench_browser_v${data.meta.extensionVersion}_${ts}.json`;
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 5000);
        log(`Downloaded ${name} (Chrome Downloads folder)`);
    }).catch(e => log(`Download failed: ${e.message}`));
}

async function init() {
    if (!chrome.runtime || !chrome.runtime.getManifest) {
        $('status').textContent = 'Open this page as an extension page (chrome-extension://.../bench.html)';
        return;
    }
    try {
        state.urls = await (await fetch(chrome.runtime.getURL('bench_urls.json'))).json();
    } catch (e) {
        $('status').textContent = 'Failed to load bench_urls.json: ' + e.message;
        return;
    }
    $('url-total').textContent = String(state.urls.length);
    $('url-count').value = String(state.urls.length);
    $('status').textContent = `Ready: ${state.urls.length} URLs, extension v${chrome.runtime.getManifest().version}`;
    log(`Loaded ${state.urls.length} URLs`);
    log('For the cleanest cold-start numbers: reload the extension, then open this page and press Run (do not open the service worker DevTools first).');

    $('run-rf').addEventListener('click', () => runModels(['rf']));
    $('run-xgb').addEventListener('click', () => runModels(['xgb']));
    $('run-both').addEventListener('click', () => runModels(['rf', 'xgb']));
    $('stop').addEventListener('click', () => { state.cancel = true; log('Stop requested...'); });
    $('download').addEventListener('click', downloadResults);
}

document.addEventListener('DOMContentLoaded', init);
