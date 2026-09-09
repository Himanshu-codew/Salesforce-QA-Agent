// Frontend WebSocket reconnect behavior tests for static/script.js.
//
// Loads the real script.js inside a vm sandbox with a controllable WebSocket
// mock and deterministic timer stubs, then drives the full
// connect -> open -> close -> reconnect -> stale-handler lifecycle and asserts:
//  1. exactly one socket exists at a time (no duplicate sockets)
//  2. reconnect schedules exactly one timer (no storm)
//  3. the ping interval is created exactly once and cleared on close
//  4. an in-flight request ("isProcessing") is unlocked when the socket dies
//  5. a stale socket's onclose/onerror can never replace a newer socket
//  6. visibilitychange resumes immediately when the tab comes back
//  7. no auto-resend (mutation safety) after reconnect
'use strict';

const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

const SCRIPT = fs.readFileSync(
  path.join(__dirname, '..', 'static', 'script.js'),
  'utf8'
);

// --- deterministic timer stubs ----------------------------------------------
const timers = { timeouts: [], intervals: [], seq: 1 };
function stubSetTimeout(fn, ms) {
  const id = timers.seq++;
  timers.timeouts.push({ id, fn, ms });
  return id;
}
function stubClearTimeout(id) {
  timers.timeouts = timers.timeouts.filter((t) => t.id !== id);
}
function stubSetInterval(fn, ms) {
  const id = timers.seq++;
  timers.intervals.push({ id, fn, ms });
  return id;
}
function stubClearInterval(id) {
  timers.intervals = timers.intervals.filter((t) => t.id !== id);
}

// --- WebSocket mock ---------------------------------------------------------
const sockets = [];
class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    this.sent = [];
    this.closeCount = 0;
    this.onopen = null;
    this.onclose = null;
    this.onerror = null;
    this.onmessage = null;
    sockets.push(this);
  }
  send(data) {
    this.sent.push(String(data));
  }
  close() {
    this.closeCount += 1;
    this.readyState = FakeWebSocket.CLOSED;
  }
}

// --- DOM/document stubs -----------------------------------------------------
const listeners = {};
function makeEl() {
  return {
    textContent: '',
    innerHTML: '',
    className: '',
    style: {},
    remove() {},
    focus() {},
    addEventListener() {},
    appendChild() {},
    setAttribute() {},
    querySelector() {
      return { className: '', setAttribute() {} };
    },
    classList: { add() {}, remove() {}, contains() { return false; } },
  };
}

// --- sandbox -----------------------------------------------------------------
const sandbox = {
  console,
  crypto: { randomUUID: () => 'session-test-001' },
  WebSocket: FakeWebSocket,
  setTimeout: stubSetTimeout,
  clearTimeout: stubClearTimeout,
  setInterval: stubSetInterval,
  clearInterval: stubClearInterval,
  window: {
    location: { protocol: 'http:', host: 'example.test' },
    wsPingTimer: null,
    matchMedia: () => null,
    addEventListener() {},
  },
  document: {
    visibilityState: 'visible',
    getElementById: (id) => (id === 'thinkingIndicator' ? null : makeEl()),
    addEventListener: (type, fn) => {
      (listeners[type] = listeners[type] || []).push(fn);
    },
  },
  localStorage: {
    getItem: () => null,
    setItem() {},
  },
};
vm.createContext(sandbox);
vm.runInContext(SCRIPT, sandbox, { filename: 'static/script.js' });

function run(expr) {
  return vm.runInContext(expr, sandbox);
}

// --- helpers ---------------------------------------------------------------
function connect() {
  run('connectWebSocket()');
}
function fireReconnectTimer() {
  assert.strictEqual(timers.timeouts.length, 1, 'exactly one reconnect timer');
  const t = timers.timeouts[0];
  timers.timeouts = [];
  t.fn();
}

// --- scenario: single socket, ordered lifecycle ------------------------------
connect();
assert.strictEqual(sockets.length, 1, 'exactly one socket created');
assert.strictEqual(run('ws'), sockets[0], 'global ws points at the new socket');
assert.strictEqual(run('isConnected'), false);

// duplicate connect while CONNECTING must not create a second socket
connect();
connect();
assert.strictEqual(sockets.length, 1, 'no duplicate sockets while CONNECTING');

// open: exactly one ping interval, reconnectAttempts reset
sockets[0].readyState = FakeWebSocket.OPEN;
sockets[0].onopen();
assert.strictEqual(run('isConnected'), true, 'connected after open');
assert.strictEqual(timers.intervals.length, 1, 'ping interval started exactly once');
sockets[0].onopen(); // duplicate open
assert.strictEqual(timers.intervals.length, 1, 'duplicate open does not stack intervals');

// connect while OPEN must not create a second socket
connect();
assert.strictEqual(sockets.length, 1, 'no duplicate socket while OPEN');

// mark an in-flight request, then let the socket die unexpectedly
run('isProcessing = true');
sockets[0].readyState = FakeWebSocket.CLOSED;
sockets[0].onclose();

assert.strictEqual(run('ws'), null, 'socket reference cleared on close');
assert.strictEqual(run('isConnected'), false, 'disconnected after close');
assert.strictEqual(
  run('isProcessing'),
  false,
  'in-flight request unlocked on unexpected close (no dead send button)'
);
assert.strictEqual(timers.intervals.length, 0, 'ping interval cleared on close');
assert.strictEqual(timers.timeouts.length, 1, 'exactly one reconnect timer scheduled');

// reconnect: a single new socket replaces the old one
fireReconnectTimer();
assert.strictEqual(sockets.length, 2, 'reconnect created exactly one new socket');
const s1 = sockets[1];
assert.strictEqual(run('ws'), s1, 'global ws points at the newest socket');

// --- stale-handler regression -------------------------------------------------
// The OLD socket fires close/error AFTER the new socket exists: it must be
// ignored — no ws clobbering, no extra reconnect timer, no status flood.
const timerCountBefore = timers.timeouts.length;
sockets[0].onclose();
sockets[0].onerror({ message: 'stale' });
assert.strictEqual(run('ws'), s1, 'stale onclose cannot replace the newer socket');
assert.strictEqual(
  timers.timeouts.length,
  timerCountBefore,
  'stale onclose cannot schedule a second reconnect'
);
assert.strictEqual(run('isConnected'), false, 'stale handler did not flip state up');

// the new socket opens normally
s1.readyState = FakeWebSocket.OPEN;
s1.onopen();
assert.strictEqual(run('isConnected'), true, 'reconnected after open');
assert.strictEqual(timers.intervals.length, 1, 'new ping interval is single');

// --- visibilitychange ---------------------------------------------------------
// visible + open socket -> no action (no duplicate socket, no extra timer)
run("document.visibilityState = 'visible'");
listeners.visibilitychange.forEach((fn) => fn());
assert.strictEqual(sockets.length, 2, 'visibility with an open socket does nothing');

// visible + closed socket + pending reconnect -> clear timer, connect NOW
s1.readyState = FakeWebSocket.CLOSED;
s1.onclose();
assert.strictEqual(timers.timeouts.length, 1, 'close scheduled one reconnect');
listeners.visibilitychange.forEach((fn) => fn());
assert.strictEqual(timers.timeouts.length, 0, 'pending reconnect timer cancelled on resume');
assert.strictEqual(sockets.length, 3, 'visibility resume reconnected immediately');
assert.strictEqual(run('ws'), sockets[2], 'newest socket active after resume');

// --- no auto-resend (mutation safety) ----------------------------------------
// None of the sockets may have transmitted a user "message" after the
// disconnects above; only pings are allowed.
for (const s of sockets) {
  for (const frame of s.sent) {
    assert.ok(
      frame.includes('"ping"'),
      `no message auto-resend after reconnect (got: ${frame.slice(0, 60)})`
    );
  }
}

console.log('ws-reconnect-puppet: ALL ASSERTIONS PASSED');