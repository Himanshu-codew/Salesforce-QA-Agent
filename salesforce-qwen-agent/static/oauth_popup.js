// OAuth popup completion (same-origin static file, allowed by script-src 'self').
// Hands the result back ONLY to the window that opened this popup, and only to
// our own origin — never a wildcard — so a hostile page cannot steal the handshake.
(function () {
    'use strict';
    var resultEl = document.getElementById('oauth-result');
    var sessionId = resultEl ? (resultEl.getAttribute('data-session-id') || '') : '';
    if (window.opener) {
        window.opener.postMessage({ type: 'oauth_success', session_id: sessionId }, window.location.origin);
        window.close();
    } else {
        window.location.href = '/';
    }
})();