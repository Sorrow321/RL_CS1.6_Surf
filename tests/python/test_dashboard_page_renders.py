"""The dashboard's runs page must RENDER, not just parse on the server.

Why this test exists: on 2026-09-12 one unescaped apostrophe inside a
description string in viewer/runs.js stopped the whole script from parsing,
and the page showed nothing but "select a run" for an hour while every
server endpoint answered 200. No Python-side test can see that; a browser
can. This starts the dashboard on a spare port, renders runs.html in
headless Chrome/Edge (skipped when neither is installed) and asserts that
the run list the API returns actually appears in the DOM. Any future
JavaScript syntax slip in viewer/runs.js fails here, before a launch.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


def _browser():
    for b in BROWSERS:
        if os.path.isabs(b):
            if Path(b).exists():
                return b
        elif shutil.which(b):
            return shutil.which(b)
    return None


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.mark.skipif(_browser() is None, reason="no headless-capable browser installed")
def test_runs_page_renders_the_run_list():
    if not (ROOT / "runs").is_dir():
        pytest.skip("no runs/ directory here")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "dashboard.py"), "--port", str(port)],
                            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        runs = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(base + "/api/runs", timeout=5) as r:
                    runs = json.loads(r.read().decode("utf-8"))["runs"]
                break
            except Exception:
                time.sleep(0.5)
        assert runs is not None, "the dashboard did not answer /api/runs"
        if not runs:
            pytest.skip("no runs to render")
        dom = subprocess.run([_browser(), "--headless=new", "--disable-gpu", "--no-sandbox",
                              "--virtual-time-budget=8000", "--dump-dom",
                              base + "/viewer/runs.html"],
                             capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
        assert len(dom) > 2000, "the browser returned no page"
        names = [r["name"] for r in runs[:5]]
        shown = [n for n in names if n in dom]
        assert shown, (f"none of the API's runs {names} appears in the rendered page: "
                       "viewer/runs.js most likely does not parse (check the browser console)")
    finally:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
