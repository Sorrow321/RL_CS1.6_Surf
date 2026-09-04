"""The runs page must be SERVED, with the chart library that draws it.

Two failures this file gates, both of which look like "the dashboard is
broken" and neither of which any logic test can see:

  * The charts are uPlot, vendored at viewer/vendor/. A CDN would leave a
    rented box behind an ssh tunnel with a blank Metrics section, so the
    file has to be in the repo AND come back over HTTP with a content type
    the browser will execute. Windows serves .js from the REGISTRY by
    default, where a stray entry hands out text/plain and Chrome refuses
    the script - dashboard.Handler pins the type for exactly that reason.

  * A run directory relaunched under the same name holds two lives in one
    progress.csv (the step folds back to 0). _metrics_from_csv keeps only
    the LAST monotone segment; if that guard regresses, every chart draws
    two lines and the overlay feature would silently compare a run against
    its own previous life. The alternate X axes (iteration, wall-clock)
    are computed over that same segment and must not cross the fold
    either.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import dashboard                                          # noqa: E402


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=20) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, "", b""


# --------------------------------------------------------------- serving ---
def test_the_gate_can_fail(server):
    """A vendor file that does not exist must 404, or every assertion below
    is vacuous."""
    code, _, _ = _get(server, "/viewer/vendor/this_library_does_not_exist.js")
    assert code == 404


def test_runs_page_is_served(server):
    code, ctype, body = _get(server, "/viewer/runs.html")
    assert code == 200
    assert ctype == "text/html"
    text = body.decode("utf-8")
    assert 'src="vendor/uPlot.iife.min.js"' in text, "page does not load uPlot"
    assert 'href="vendor/uPlot.min.css"' in text, "page does not load uPlot css"
    assert 'src="runs.js"' in text


def test_runs_js_is_served_as_javascript(server):
    code, ctype, body = _get(server, "/viewer/runs.js")
    assert code == 200
    assert ctype == "text/javascript", (
        f"served as {ctype!r}; a browser will not execute that")
    assert b"uPlot" in body


@pytest.mark.parametrize("path,ctype,marker", [
    ("/viewer/vendor/uPlot.iife.min.js", "text/javascript", b"uPlot"),
    ("/viewer/vendor/uPlot.min.css", "text/css", b".uplot"),
])
def test_vendored_chart_library_is_served(server, path, ctype, marker):
    code, got, body = _get(server, path)
    assert code == 200, f"{path} is not in the repo (no CDN fallback exists)"
    assert got == ctype
    assert marker in body
    assert len(body) > 500


def test_vendored_uplot_is_the_pinned_version():
    """The version is pinned in a comment in the file as distributed; the
    page's own comment must not drift away from it."""
    js = (ROOT / "viewer" / "vendor" / "uPlot.iife.min.js").read_text(
        encoding="utf-8", errors="replace")
    assert "(v1.6.32)" in js.splitlines()[0]
    html = (ROOT / "viewer" / "runs.html").read_text(encoding="utf-8")
    assert "uPlot v1.6.32" in html


# ------------------------------------------------------- the two-life guard --
TWO_LIVES = """time/total_timesteps,rollout/ep_rew_mean,time/fps
1048576,-9.0,50000
2097152,-8.0,60000
3145728,-7.0,70000
4194304,-6.0,80000
1048576,1.0,100000
2097152,2.0,110000
3145728,3.0,120000
4194304,4.0,130000
5242880,5.0,140000
"""


@pytest.fixture()
def two_life_run(tmp_path, monkeypatch):
    d = tmp_path / "reused_name"
    d.mkdir()
    (d / "progress.csv").write_text(TWO_LIVES, encoding="utf-8")
    monkeypatch.setattr(dashboard, "RUNS", tmp_path)
    return d.name


def test_metrics_endpoint_returns_the_last_life_only(server, two_life_run):
    code, ctype, body = _get(server, "/api/metrics?run=" + two_life_run)
    assert code == 200
    assert ctype == "application/json"
    doc = json.loads(body)
    s = doc["series"]["rollout/ep_rew_mean"]
    assert s["values"] == [1.0, 2.0, 3.0, 4.0, 5.0], (
        "the first life leaked in - the chart would draw two lines")
    assert s["steps"] == [1048576, 2097152, 3145728, 4194304, 5242880]
    assert all(b > a for a, b in zip(s["steps"], s["steps"][1:])), \
        "x folds backwards"


def test_alternate_axes_do_not_cross_the_fold(server, two_life_run):
    doc = json.loads(_get(server, "/api/metrics?run=" + two_life_run)[2])
    assert doc["axes"] == ["steps", "iter", "wall"]
    s = doc["series"]["rollout/ep_rew_mean"]
    # the iteration index restarts with the live segment, not at row 5
    assert s["iter"] == [1, 2, 3, 4, 5]
    assert len(s["wall"]) == len(s["steps"])
    assert all(b >= a for a, b in zip(s["wall"], s["wall"][1:])), \
        "wall-clock runs backwards"
    assert s["wall"][0] >= 0.0


def test_wall_axis_is_absent_without_an_fps_column(tmp_path):
    """time/fps is the only clock progress.csv has. Without it the page must
    be told the axis does not exist rather than shown a fabricated one."""
    p = tmp_path / "progress.csv"
    p.write_text("time/total_timesteps,eval/path\n1,10\n2,20\n3,30\n",
                 encoding="utf-8")
    out = dashboard._metrics_from_csv(p)
    assert "wall" not in out["eval/path"]
    assert out["eval/path"]["iter"] == [1, 2, 3]


def test_wall_hours_matches_the_cumulative_fps_definition():
    """train_fast.py logs fps as (step - step_start) / (now - t_start) with
    t_start fixed before the loop, so hours = (x - x0) / fps. x0 is
    recovered from the first two rows' step delta - here a resume at 1000."""
    xs = [1100.0, 1200.0, 1300.0]
    fps = [10.0, 10.0, 5.0]           # 100 steps in 10 s, then a slowdown
    got = dashboard._wall_hours(xs, fps)
    # hours are rounded to 5 dp (0.036 s) so the JSON stays small
    assert got == [pytest.approx(10 / 3600, abs=1e-5),
                   pytest.approx(20 / 3600, abs=1e-5),
                   pytest.approx(60 / 3600, abs=1e-5)]
    assert dashboard._wall_hours([1.0], [1.0]) is None
