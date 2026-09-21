// Regression tests for the extension navigation pipeline (background.js).
//
// Chrome APIs are stubbed, so the real background.js source runs unchanged in
// Node: message validation, block/warn flows, stale-tab guards, redirect
// re-scans and the one-time bypass are all exercised.
//
// Run: node --test tests/extension_navigation.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const utilsSrc = readFileSync(path.join(root, 'phishingdetectorExt/utils.js'), 'utf8');
const backgroundSrc = readFileSync(path.join(root, 'phishingdetectorExt/background.js'), 'utf8')
    .replace("importScripts('utils.js', 'libs/ort.min.js');", '');
const blockedPage = 'chrome-extension://test/blocked.html';

const silentConsole = { log() {}, error() {} };

function createHarness() {
    const state = {
        storage: { selectedModel: 'rf', userWhitelist: [] },
        tabs: new Map(),
        updates: [],
        sent: [],
        sendAttempts: 0,
        failSend: 0,
        nextProb: [0.1, 0.9],           // [legit, phishing]
        onBeforeNavigate: [],
        onCompleted: [],
        onMessage: [],
        onRemoved: [],
    };

    const chrome = {
        runtime: {
            getURL: p => `chrome-extension://test/${p}`,
            onMessage: { addListener: fn => state.onMessage.push(fn) },
        },
        storage: {
            local: {
                get: async key => (typeof key === 'string' ? { [key]: state.storage[key] } : { ...state.storage }),
                set: async obj => Object.assign(state.storage, obj),
            },
        },
        tabs: {
            get: async id => {
                const tab = state.tabs.get(id);
                if (!tab) throw new Error('No tab with id ' + id);
                return tab;
            },
            update: async (id, opts) => { state.updates.push({ id, ...opts }); },
            sendMessage: async (id, msg) => {
                state.sendAttempts++;
                if (state.failSend > 0) { state.failSend--; throw new Error('Receiving end does not exist'); }
                state.sent.push({ id, msg });
            },
            onRemoved: { addListener: fn => state.onRemoved.push(fn) },
        },
        webNavigation: {
            onBeforeNavigate: { addListener: fn => state.onBeforeNavigate.push(fn) },
            onCompleted: { addListener: fn => state.onCompleted.push(fn) },
        },
    };

    const ort = {
        env: { wasm: {} },
        Tensor: class { constructor(type, data, dims) { this.data = data; this.dims = dims; } },
        InferenceSession: {
            create: async () => ({
                inputNames: ['input'],
                outputNames: ['label', 'probabilities'],
                run: async () => ({
                    label: { data: [state.nextProb[1] > 0.5 ? 1 : 0] },
                    probabilities: { data: state.nextProb },
                }),
            }),
        },
    };

    const fetchStub = async () => ({
        json: async () => ({
            vocabulary: {}, idf: [], sublinear_tf: false,
            scaler_mean: [], scaler_scale: [], selected_feature_indices: [],
            pca_mean: [], pca_components: [],
        }),
        arrayBuffer: async () => new ArrayBuffer(8),
    });

    const factory = new Function('chrome', 'ort', 'performance', 'fetch', 'console', `
        ${utilsSrc}
        ${backgroundSrc}
        return {};
    `);
    factory(chrome, ort, performance, fetchStub, silentConsole);

    const fireBefore = (tabId, url) => state.onBeforeNavigate.forEach(fn => fn({ frameId: 0, tabId, url }));
    const fireCompleted = (tabId, url) => state.onCompleted.forEach(fn => fn({ frameId: 0, tabId, url }));
    const sendMessage = (msg, sender = {}) => new Promise(resolve => {
        for (const fn of state.onMessage) {
            const ret = fn(msg, sender, resolve);
            if (ret === true) return;   // async response pending
        }
        resolve(undefined);
    });

    return { state, fireBefore, fireCompleted, sendMessage };
}

const flush = async (n = 25) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };

test('block band: tabs.update points at blocked.html with the scanned URL', async () => {
    const h = createHarness();
    h.state.tabs.set(1, { url: 'https://phish.example/login', pendingUrl: 'https://phish.example/login' });
    h.state.nextProb = [0.05, 0.95];
    h.fireBefore(1, 'https://phish.example/login');
    await flush();

    assert.equal(h.state.updates.length, 1);
    assert.match(h.state.updates[0].url, /^chrome-extension:\/\/test\/blocked\.html/);
    assert.match(h.state.updates[0].url, /url=https%3A%2F%2Fphish\.example%2Flogin/);
    assert.match(h.state.updates[0].url, /conf=95\.0/);
});

test('stale guard: a tab that moved on is never replaced by a late verdict', async () => {
    const h = createHarness();
    h.state.tabs.set(1, { url: 'https://phish.example/login' });
    h.state.nextProb = [0.05, 0.95];
    h.fireBefore(1, 'https://phish.example/login');
    h.state.tabs.set(1, { url: 'https://already-moved.example/' });   // user left mid-scan
    await flush();

    assert.equal(h.state.updates.length, 0);
});

test('warning band: banner delivered once, dedup across onCompleted', async () => {
    const h = createHarness();
    h.state.tabs.set(1, { url: 'https://sus.example/', pendingUrl: 'https://sus.example/' });
    h.state.nextProb = [0.3, 0.7];
    h.fireBefore(1, 'https://sus.example/');
    await flush();
    assert.equal(h.state.sent.length, 1, 'warning delivered while page loads');

    h.fireCompleted(1, 'https://sus.example/');
    await flush();
    assert.equal(h.state.sent.length, 1, 'no duplicate banner after load completes');
});

test('warning band: retries onCompleted when the content script was not ready', async () => {
    const h = createHarness();
    h.state.tabs.set(1, { url: 'https://sus.example/', pendingUrl: 'https://sus.example/' });
    h.state.nextProb = [0.3, 0.7];
    h.state.failSend = 1;   // immediate delivery fails (no listener yet)

    h.fireBefore(1, 'https://sus.example/');
    await flush();
    assert.equal(h.state.sent.length, 0);
    assert.equal(h.state.sendAttempts, 1);

    h.fireCompleted(1, 'https://sus.example/');
    await flush();
    assert.equal(h.state.sent.length, 1, 'retried after load');
    assert.equal(h.state.sendAttempts, 2);
});

test('redirect: the final URL is scored when it differs from the scanned one', async () => {
    const h = createHarness();
    h.state.tabs.set(1, { url: 'https://short.example/', pendingUrl: 'https://short.example/' });
    h.state.nextProb = [0.3, 0.7];
    h.fireBefore(1, 'https://short.example/');
    await flush();

    // Server redirect to a blocked-band destination; tab now shows the target.
    h.state.nextProb = [0.02, 0.98];
    h.state.tabs.set(1, { url: 'https://evil.example/login' });
    h.fireCompleted(1, 'https://evil.example/login');
    await flush();

    assert.equal(h.state.updates.length, 1);
    assert.match(h.state.updates[0].url, /evil\.example%2Flogin/);
});

test('dedup: repeated onBeforeNavigate for one navigation scores once', async () => {
    const h = createHarness();
    h.state.tabs.set(1, { url: 'https://phish.example/', pendingUrl: 'https://phish.example/' });
    h.state.nextProb = [0.05, 0.95];
    h.fireBefore(1, 'https://phish.example/');
    h.fireBefore(1, 'https://phish.example/');   // redirect process swap
    await flush();
    assert.equal(h.state.updates.length, 1);
});

test('bypass: blocked-page Continue survives exactly one navigation', async () => {
    const h = createHarness();
    h.state.nextProb = [0.05, 0.95];
    h.state.tabs.set(1, { url: 'https://phish.example/', pendingUrl: 'https://phish.example/' });

    const res = await h.sendMessage(
        { action: 'bypassUrl', url: 'https://phish.example/' },
        { url: blockedPage + '?url=https%3A%2F%2Fphish.example%2F&conf=95' }
    );
    assert.equal(res.success, true);

    h.fireBefore(1, 'https://phish.example/');
    await flush();
    assert.equal(h.state.updates.length, 0, 'bypassed navigation is not blocked');

    h.fireBefore(1, 'https://phish.example/');
    await flush();
    assert.equal(h.state.updates.length, 1, 'second navigation is blocked again');
});

test('message validation: bad input and wrong senders are rejected', async () => {
    const h = createHarness();

    assert.deepEqual(await h.sendMessage({ action: 'switchModel', model: 'evil' }), { success: false, error: 'Invalid model' });
    assert.deepEqual(await h.sendMessage({ action: 'scan', url: 42 }), { error: 'Invalid URL' });
    assert.deepEqual(await h.sendMessage({ action: 'addWhitelist', domain: '   ' }), { success: false, error: 'Invalid domain' });
    assert.deepEqual(
        await h.sendMessage({ action: 'bypassUrl', url: 'https://phish.example/' }, { url: 'https://evil.example/' }),
        { success: false, error: 'Forbidden' }
    );
    assert.deepEqual(await h.sendMessage({ action: 'switchModel', model: 'xgb' }), { success: true, model: 'xgb' });
});

test('whitelist: stored entries are normalized and matched by hostname', async () => {
    const h = createHarness();
    h.state.storage.userWhitelist = ['EXAMPLE.com:8443'];

    const res = await h.sendMessage({ action: 'addWhitelist', domain: 'https://user:pw@example.com:8443/path' });
    assert.equal(res.success, true);
    assert.deepEqual(res.list, ['example.com'], 'legacy port entry cleaned');
    assert.deepEqual(h.state.storage.userWhitelist, ['example.com']);

    h.state.nextProb = [0.05, 0.95];
    h.state.tabs.set(1, { url: 'https://example.com/', pendingUrl: 'https://example.com/' });
    h.fireBefore(1, 'https://example.com/');
    await flush();
    assert.equal(h.state.updates.length, 0, 'whitelisted navigation is not blocked');
});
