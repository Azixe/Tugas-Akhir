# Extension — Known Issues (Deferred)

> Findings from the 2026-09-21 code review of the extension audit patch
> (`fix(extension): consistency corrections...` + `fix(extension): harden
> navigation...`). Fixed items are in the commits; the items below were
> deliberately deferred. Revisit before the thesis defense / final build.

## 1. Pre-emptive blocking is best-effort only

`webNavigation.onBeforeNavigate` does not cancel the in-flight request: the
page may start loading (and run scripts) while the model initializes and
scores, and a >80% verdict then replaces the tab. Truly blocking before any
page code runs needs `declarativeNetRequest` rules or a cached verdict.
The comment in `background.js` states this limitation.

## 2. Service-worker lifetime loses per-tab state

`pendingWarnings`, `scansInFlight` and `sessionBypass` live in the MV3
service-worker memory. Chrome terminates the worker after ~30s idle, so a
pending warning or a bypass added just before termination can be lost.
Mitigation if it matters: `chrome.storage.session`. In practice the window
is small (the blocked-page message wakes the worker; the user navigates
immediately).

## 3. Bypass keys on the exact URL string

`sessionBypass` matches the URL string exactly. If the browser upgrades
`http` to `https`, or a redirect hop changes scheme/trailing slash between
the blocked URL and the next navigation, the bypass is not consumed and the
redirect can be blocked again. Also, a bypass is consumed by whichever tab
navigates to that URL first, not necessarily the tab that requested it.

## 4. Redirect targets get scored twice

A warning-band URL whose final URL differs (server redirect) is scored once
in `onBeforeNavigate` and once in `onCompleted`. This is intentional
coverage (the final URL was previously never scored) but doubles the cost
for the warming path. A small verdict cache would remove the duplicate.

## 5. Whitelist is static + user list only

The static `WHITELIST` in `utils.js` covers popular domains. `Trust Site`
adds a bare hostname to `chrome.storage.local`; there is no expiry or
review UI beyond the popup list.

## Verification

- `node --test tests/extension_navigation.test.mjs` (9 tests, stubbed Chrome)
- Python feature parity: `pytest tests/test_features_parity.py` (passes, 5 tests)
- Manual Chrome checks (pending): blocked-page buttons, warning banner,
  model switcher, popup SAFE-confidence display.
