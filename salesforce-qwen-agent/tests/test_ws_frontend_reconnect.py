"""
Frontend WebSocket reconnect behavior tests.

Runs the real ``static/script.js`` inside a Node vm sandbox (a controllable
WebSocket mock + deterministic timer stubs) and drives the full
connect -> open -> close -> reconnect -> stale-handler lifecycle.

Regressions covered (Phase 3 frontend audit):

- exactly one socket exists at a time (duplicate-socket guard)
- reconnect schedules exactly one timer (no storm) with exponential backoff
- the ping/heartbeat interval starts exactly once and is cleared on close
- an in-flight request's "isProcessing" lock is released on unexpected close
  (previously a mid-flight disconnect left the send button disabled forever)
- a stale socket's onclose/onerror can never replace a newer socket or
  schedule a second reconnect
- visibilitychange resumes the connection immediately (background-tab
  throttling + Render cold-boot recovery)
- no auto-resend of user messages after reconnect (mutation safety)

Skips if Node.js is not installed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PUPPET = Path(__file__).with_name("puppet_ws_reconnect.js")


def test_frontend_reconnect_behavior():
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed; skipping frontend reconnect tests")

    result = subprocess.run(
        ["node", str(PUPPET)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"frontend reconnect puppet failed:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "ALL ASSERTIONS PASSED" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))