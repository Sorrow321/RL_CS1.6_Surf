"""--ge-arc U (zlr1b2, 2026-10-31): Go-Explore from the true spawn becomes the arc route.

(a) the startup refusals (no --ge-minutes, the ratchet on, an explicit --race-arc);
(b) on surf_za_loop: a 3-second energy-cell search at 7 ms with exact chains, then
    train_fast._ge_arc_build writes the route (128 u spacing, starting at the spawn) and returns
    spawn rows = the replayed line (views along the velocity) + forward archive cells.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
MAP = ROOT / "maps" / "surf_za_loop.bsp"
FIELD = ROOT / "maps" / "surf_za_loop.goalc_32.npz"
DLL = ROOT / "build" / "libsurfcore.so"
needs_map = pytest.mark.skipif(not (MAP.exists() and FIELD.exists() and DLL.exists()),
                               reason="needs the built core + surf_za_loop and its goalc field")
TRAIN = ROOT / "python" / "train_fast.py"


def _refusal(extra):
    r = subprocess.run([sys.executable, "-u", str(TRAIN), "--map", str(MAP), "--run", "geArcRefuse",
                        "--reward", "race"] + extra, capture_output=True, text=True, timeout=600,
                       cwd=str(ROOT))
    return r.returncode, (r.stdout + r.stderr)


@needs_map
def test_the_startup_refusals():
    rc, out = _refusal(["--ge-arc", "1500"])
    assert rc != 0 and "--ge-minutes" in out
    rc, out = _refusal(["--ge-arc", "1500", "--ge-minutes", "1", "--race-ratchet"])
    assert rc != 0 and "--no-race-ratchet" in out
    rc, out = _refusal(["--ge-arc", "1500", "--ge-minutes", "1", "--race-arc", "x.npz"])
    assert rc != 0 and "not with --race-arc" in out


@needs_map
def test_the_search_becomes_a_route_and_spawn_rows(tmp_path):
    import train_fast as tf
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "explore_phase1.py"),
                        "--map", str(MAP), "--out", str(out / "ge"), "--envs", "64",
                        "--cell", "64", "--cell-energy", "32", "--cell-heading", "1",
                        "--view-follow=-90,-60,-30,-15,0,15,30,60,90", "--select-energy", "3",
                        "--exact-chains", "--decisions", "100", "--act-every", "4",
                        "--maxvel", "3500", "--tick-ms", "7", "--max-minutes", "0.05",
                        "--threads", "2", "--seed", "0", "--goals", "1"],
                       capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    assert r.returncode in (0, 1), r.stdout[-2000:] + r.stderr[-2000:]
    z = np.load(out / "ge" / "archive.npz", allow_pickle=True)
    args = SimpleNamespace(map=str(MAP), race_no_corner_cut=1, race_kill_aware=None,
                           goal_cell=None, lidar_cell=32.0,
                           race_arc=str(out / "ge_route.npz"))
    assert tf._ge_field_path(args) == FIELD
    rows = tf._ge_arc_build(args, out, z)
    route = np.load(out / "ge_route.npz")
    assert float(route["spacing"]) == 128.0 and len(route["route"]) >= 2
    seg = np.linalg.norm(np.diff(route["route"].astype(np.float64), axis=0), axis=1)
    assert np.all(seg <= 128.0 + 1e-2)
    line = np.load(out / "ge_line.npy")
    assert np.allclose(route["route"][0], line["origin"][0], atol=1e-2)   # starts at the spawn
    assert len(rows) >= len(line)
    v = rows["velocity"].astype(np.float64)
    fast = np.hypot(v[:, 0], v[:, 1]) >= 100.0
    hd = np.degrees(np.arctan2(v[fast, 1], v[fast, 0])) % 360.0
    dyaw = (rows["yaw"][fast].astype(np.float64) - hd + 180.0) % 360.0 - 180.0
    assert np.all(np.abs(dyaw) < 1e-3)                                    # views along v


def test_the_recorder_lists_the_keys_as_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt as rc
    assert {"ge_arc", "ge_arc_minutes"} <= set(rc.TRAIN_ONLY)


def test_no_race_ratchet_leaves_the_default_unset():
    """--no-race-ratchet shares race_ratchet's dest: registered with default None after the
    original, the default stays None (a store_false registered first would turn the ratchet ON
    for every run that passes neither flag)."""
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    a = src.index('"--race-ratchet", action="store_true", default=None')
    b = src.index('"--no-race-ratchet", dest="race_ratchet", action="store_false"')
    assert a < b
    blk = src[b: b + 200]
    assert "default=None" in blk


@needs_map
def test_ge_arc_without_the_ratchet_flag_is_not_refused_for_the_ratchet():
    # refused LATER (an invalid --ge-frac), so a ratchet ON by default would show up first
    rc, out = _refusal(["--ge-arc", "1500", "--ge-minutes", "1", "--ge-frac", "2"])
    assert rc != 0 and "--ge-frac must be in (0, 1)" in out
    assert "--no-race-ratchet" not in out
