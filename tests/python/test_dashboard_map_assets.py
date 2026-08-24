""""The logic is right" and "the page renders" are different claims.

`test_viewer_map_resolution.py` asserts the viewer RESOLVES a trajectory's
map from the trajectory's own header rather than from run.json. It passed
throughout the incident that motivated this file - and the dashboard still
rendered an empty scene for every pool map, because
`viewer/assets/<map>.mesh.json` did not exist. The viewer asked for the right
geometry and got a 404, which looks identical to "wrong map".

So this file gates the OTHER half: for every map a run's trajectories name,
the mesh asset the viewer will request must actually be served with a 200 by
the dashboard that is serving that run. It starts a real server on an
ephemeral port and makes real requests, because the failure being gated was
a missing FILE, and no amount of path-resolution logic can see that.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.request
import urllib.error
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
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def _maps_named_by(run_dir: Path) -> set:
    """Every map any of this run's trajectories names in its own header -
    which is exactly the set the viewer will request geometry for."""
    out = set()
    for p in sorted(run_dir.glob("traj_*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if isinstance(row, dict) and "map" in row:
                    out.add(row["map"])
                    break
    return out


def test_the_gate_can_fail(server):
    """A mesh that does not exist must 404, or every assertion below is
    vacuous and the whole file proves nothing."""
    code, _ = _get(server, "/viewer/assets/surf_this_map_does_not_exist"
                           ".mesh.json")
    assert code == 404


def test_a_known_map_mesh_is_served(server):
    """The repo ships cannonball's mesh; it must come back as real JSON with
    the geometry the viewer needs, not an error page with a 200 on it."""
    code, body = _get(server, "/viewer/assets/surf_src_cannonball.mesh.json")
    assert code == 200
    doc = json.loads(body)
    assert doc.get("map") == "surf_src_cannonball"
    assert doc.get("world", {}).get("positions"), "no world geometry"


@pytest.mark.parametrize("run_dir", sorted(
    (p for p in (ROOT / "runs").glob("*")
     if p.is_dir() and p.name != "tb" and list(p.glob("traj_*.jsonl"))),
    key=lambda p: p.stat().st_mtime, reverse=True)[:3],
    ids=lambda p: p.name)
def test_every_map_a_run_records_has_servable_geometry(server, run_dir):
    """The real gate, over the most recent runs that have recordings.

    A multi-map run writes one trajectory per map per eval, each naming its
    own map. If any of those maps has no mesh asset the dashboard renders an
    empty scene for it - the exact symptom the user reported twice, and the
    one the resolution test cannot see.
    """
    named = _maps_named_by(run_dir)
    if not named:
        pytest.skip("no trajectory headers")
    missing = []
    for m in sorted(named):
        code, body = _get(server, f"/viewer/assets/{m}.mesh.json")
        if code != 200 or not body:
            missing.append(f"{m} -> HTTP {code}")
    assert not missing, (
        f"{run_dir.name} records {len(named)} map(s) whose geometry the "
        f"dashboard cannot serve: {missing}. The viewer resolves the map "
        f"correctly and then 404s, which renders an empty scene. Export "
        f"them with tools/export_map.py (fetch_pool.sh does it for pool "
        f"maps).")
