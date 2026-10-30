"""explore_phase1 --view-follow-nobrake (uf2r1b2, 2026-10-30).

Under --view-follow a burst holds a view offset from the velocity heading, so its keys keep one
wish angle relative to the motion; with uniform keys about half of the draws wish backward, and
GoldSrc's air acceleration turns a held backward wish into a brake of up to 250 u/s per tick. The
flag presses the opposite keys for those draws.

(a) the helper: after it no wish points more than 90 deg from the velocity, the strafe (90) and
    the forward/coast draws are kept, a released pair stays released;
(b) the premise, in the core: a held wish at 180 deg brakes a 1,000 u/s flight by hundreds of
    u/s in a few ticks, one at 90 deg (the side the helper keeps) does not lose speed;
(c) flag off: archive and meta bit-identical to HEAD's explorer;
(d) flag on: the meta says so and ge_replay replays the deepest chain exactly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from explore_phase1 import nobrake_keys  # noqa: E402

GE = ROOT / "tools" / "explore_phase1.py"
MAP = ROOT / "maps" / "surf_unitfarmer2.bsp"
FIELD = ROOT / "maps" / "surf_unitfarmer2.goalc_32.npz"
needs_map = pytest.mark.skipif(not (MAP.exists() and FIELD.exists()),
                               reason="surf_unitfarmer2 bsp + goalc field not here")
ARGS = ["--map", str(MAP), "--envs", "64", "--cell", "64", "--cell-energy", "32",
        "--decisions", "25", "--act-every", "4", "--maxvel", "3500", "--max-iters", "2",
        "--threads", "2", "--exact-chains", "--seed", "3"]


def _wish_beta(fwd, side, off):
    f = np.asarray(fwd) - 1
    s = np.asarray(side) - 1
    return (np.asarray(off, np.float64) + np.degrees(np.arctan2(-s, f)) + 180.0) % 360.0 - 180.0


def test_helper_never_brakes_and_keeps_the_rest():
    offs = np.array([-90, -60, -30, -15, 0, 15, 30, 60, 90], np.float64)
    F, S, O = np.meshgrid(np.arange(3), np.arange(3), offs, indexing="ij")
    F, S, O = F.ravel(), S.ravel(), O.ravel()
    fo, so, flip = nobrake_keys(F, S, O)
    wish = (fo != 1) | (so != 1)
    assert np.all(np.abs(_wish_beta(fo, so, O))[wish] <= 90.0 + 1e-9)
    # a draw that did not brake is untouched; a released pair stays released
    keep = (np.abs(_wish_beta(F, S, O)) <= 90.0 + 1e-9) | ((F == 1) & (S == 1))
    assert np.array_equal(fo[keep], F[keep]) and np.array_equal(so[keep], S[keep])
    assert not flip[keep].any() and flip[~keep].all()
    # the mirror is the opposite key pair: back -> forward, left -> right
    assert nobrake_keys(np.array([0]), np.array([1]), np.array([0.0]))[:2] == (2, 1)
    assert nobrake_keys(np.array([1]), np.array([0]), np.array([90.0]))[:2] == (1, 2)
    # the air strafe at 90 deg is kept: left with the view along v, right with the view 90 left
    assert nobrake_keys(np.array([1]), np.array([0]), np.array([0.0]))[:2] == (1, 0)
    assert nobrake_keys(np.array([1]), np.array([2]), np.array([0.0]))[:2] == (1, 2)


def _air_state(core, speed=1000.0):
    s = core.get_states()[:1].copy()
    s["origin"] = (-2176.0, -3400.0, 900.0)      # open air south of the start platform
    s["velocity"] = (speed, 0.0, 0.0)
    s["yaw"] = 0.0
    s["tick"] = 0
    s["stuck_ticks"] = 0
    return s


@needs_map
def test_premise_back_wish_brakes_strafe_does_not():
    from surfgym.core import SurfCore, default_config
    cfg = default_config(num_envs=1, spawn_mode=2, max_episode_ticks=6000, water_fail=1,
                         yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    cfg.phys.sv_maxvelocity = 3500.0
    core = SurfCore(str(MAP), cfg)
    out = {}
    for name, fwd, side in (("back", 0, 1), ("left", 1, 0), ("none", 1, 1)):
        core.reset(0)
        core.set_state(0, _air_state(core))
        a = np.array([[7, 3, fwd, side, 0, 0]], np.int32)     # yaw bin 7 = 0 deg/tick
        for _ in range(4):
            core.step(a)
        v = np.asarray(core.get_states()[0]["velocity"], np.float64)
        out[name] = float(np.hypot(v[0], v[1]))
    assert out["back"] < 1000.0 - 300.0, out          # ~250 u/s per tick
    assert out["left"] >= out["none"] - 1e-3, out      # the strafe adds speed, never brakes
    assert abs(out["none"] - 1000.0) < 1.0, out


def _pre_change(dst: Path):
    s = subprocess.run(["git", "show", "HEAD:tools/explore_phase1.py"], capture_output=True,
                       cwd=str(ROOT))
    if s.returncode != 0 or b'"--view-follow-nobrake"' in s.stdout:
        return None
    dst.write_bytes(s.stdout)
    return True


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))


@needs_map
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "tools" / "_explore_phase1_pre_nb.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("HEAD's explorer already has --view-follow-nobrake (run before committing)")
        for name, script in (("new", GE), ("old", pre)):
            r = _run([sys.executable, "-u", str(script), "--out", str(tmp_path / name),
                      "--view-follow=-90,-30,0,30,90"] + ARGS, timeout=900)
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
    assert "view_follow_nobrake" not in ma and ma.keys() == mb.keys()


@needs_map
def test_flag_on_records_and_replays(tmp_path):
    out = tmp_path / "nb"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--view-follow=-90,-30,0,30,90",
              "--view-follow-nobrake"] + ARGS, timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["view_follow_nobrake"] is True
    z = np.load(out / "archive.npz")
    acts = z["link_acts"]
    assert len(acts) > 0
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(MAP),
              "--archive", str(out / "archive.npz"), "--field", str(FIELD), "--out", str(line)],
             timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout


def test_refused_without_view_follow(tmp_path):
    r = _run([sys.executable, "-u", str(GE), "--out", str(tmp_path / "x"),
              "--view-follow-nobrake"] + ARGS, timeout=300)
    assert r.returncode != 0
    assert "--view-follow-nobrake" in (r.stdout + r.stderr)
