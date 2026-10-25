"""explore_phase1 --view-follow (pbr4b3, 2026-10-25).

Each burst holds a view OFFSET from the velocity heading (the policy's --view-absolute velocity
frame) and the view is steered to it every tick above 100 u/s, so the explorer's keys mean what
they would mean to the policy.

(a) flag off: archive and meta bit-identical to the explorer before the change;
(b) flag on: at speed the view stays near heading + one of the offsets, the meta says so, and
    ge_replay replays the deepest chain exactly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402

GE = ROOT / "tools" / "explore_phase1.py"
FIELD = ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"
SPINE = ROOT / "autoresearch" / "spines" / "pbr4b3_beam_g11.npy"
ARGS = ["--map", str(PB), "--envs", "128", "--cell", "64", "--cell-energy", "32",
        "--decisions", "50", "--act-every", "4", "--maxvel", "3500", "--max-iters", "3",
        "--threads", "2", "--exact-chains"]


def _roots(tmp_path) -> Path:
    s = np.load(SPINE)
    r = s[1900:2100:25].copy()                   # fast states of the policy's own line
    r["tick"] = 0
    r["stuck_ticks"] = 0
    p = tmp_path / "roots.npy"
    np.save(p, r)
    return p


def _pre_change(dst: Path):
    s = subprocess.run(["git", "show", "HEAD:tools/explore_phase1.py"], capture_output=True,
                       cwd=str(ROOT))
    if s.returncode != 0 or b'"--view-follow"' in s.stdout:
        return None
    dst.write_bytes(s.stdout)
    return True


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    roots = _roots(tmp_path)
    pre = ROOT / "tools" / "_explore_phase1_pre_vf.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("HEAD's explorer already has --view-follow (run before committing)")
        for name, script in (("new", GE), ("old", pre)):
            r = _run([sys.executable, "-u", str(script), "--out", str(tmp_path / name),
                      "--roots-spine", str(roots)] + ARGS, timeout=900)
            assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    finally:
        pre.unlink(missing_ok=True)
    a = np.load(tmp_path / "new" / "archive.npz")
    b = np.load(tmp_path / "old" / "archive.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k
    ma = json.loads((tmp_path / "new" / "archive.meta.json").read_text())
    mb = json.loads((tmp_path / "old" / "archive.meta.json").read_text())
    assert "view_follow" not in ma and ma.keys() == mb.keys()


@needs_pb
def test_view_follows_the_velocity_and_replays(tmp_path):
    roots = _roots(tmp_path)
    out = tmp_path / "vf"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--roots-spine", str(roots),
              "--view-follow=-30,0,30"] + ARGS, timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["view_follow"] == "-30,0,30"
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
              "--archive", str(out / "archive.npz"), "--field", str(FIELD), "--out", str(line)],
             timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
    st = np.load(line, allow_pickle=False)
    v = st["velocity"].astype(np.float64)
    fast = np.hypot(v[:, 0], v[:, 1]) >= 150.0
    assert fast.sum() > 50
    off = (st["yaw"].astype(np.float64) - np.degrees(np.arctan2(v[:, 1], v[:, 0]))
           + 180.0) % 360.0 - 180.0
    # the view sits within 30 deg (+ a steering lag of a few 10 deg/tick ticks) of the velocity
    assert np.mean(np.abs(off[fast]) <= 45.0) >= 0.9, np.percentile(np.abs(off[fast]), [50, 90])
