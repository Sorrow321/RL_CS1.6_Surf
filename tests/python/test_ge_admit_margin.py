"""explore_phase1 --admit-margin (uf2r1b2, 2026-10-30).

A visit opens a new archive cell only if its run lived SECS past it or was still alive at the
chunk's end - the trainer's --respawn-margin rule, applied to the archive - so a surf pit does not
fill with falls that are already doomed (and an energy ranking does not prefer them).

(a) admit_mask on a hand-built rollout: runs that die keep only the visits >= margin before
    their last row; a run alive at the chunk end keeps everything;
(b) flag off: archive and meta bit-identical to HEAD's explorer;
(c) flag on: the meta says so, fewer cells open than without it on the same seed, and
    ge_replay still replays the deepest chain exactly.
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

from explore_phase1 import admit_mask  # noqa: E402

GE = ROOT / "tools" / "explore_phase1.py"
MAP = ROOT / "maps" / "surf_unitfarmer2.bsp"
FIELD = ROOT / "maps" / "surf_unitfarmer2.goalc_32.npz"
needs_map = pytest.mark.skipif(not (MAP.exists() and FIELD.exists()),
                               reason="surf_unitfarmer2 bsp + goalc field not here")
ARGS = ["--map", str(MAP), "--envs", "64", "--cell", "64", "--cell-energy", "32",
        "--decisions", "25", "--act-every", "4", "--maxvel", "3500", "--max-iters", "3",
        "--threads", "2", "--exact-chains", "--seed", "5", "--view-follow=-90,-30,0,30,90"]


def test_admit_mask_on_a_hand_built_rollout():
    # 10 ticks, 2 envs. env 0: run 0 rows 0-3 (dies after row 3), run 2 rows 4-9 (alive at the
    # end). env 1: run 1 rows 0-9 (alive at the end).
    rid = np.full((10, 2), -1, np.int32)
    rid[0:4, 0] = 0
    rid[4:10, 0] = 2
    rid[:, 1] = 1
    g_run = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    g_tick = np.array([0, 1, 2, 3, 0, 9, 4, 9])
    m = admit_mask(rid, 3, g_run, g_tick, margin_ticks=2)
    # run 0's last row is 3: visits at 0 and 1 are >= 2 rows before it, 2 and 3 are not
    assert m.tolist() == [True, True, False, False, True, True, True, True]
    # a zero margin admits everything
    assert admit_mask(rid, 3, g_run, g_tick, margin_ticks=0).all()


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))


def _pre_change(dst: Path):
    s = subprocess.run(["git", "show", "HEAD:tools/explore_phase1.py"], capture_output=True,
                       cwd=str(ROOT))
    if s.returncode != 0 or b'"--admit-margin"' in s.stdout:
        return None
    dst.write_bytes(s.stdout)
    return True


@needs_map
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "tools" / "_explore_phase1_pre_am.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("HEAD's explorer already has --admit-margin (run before committing)")
        for name, script in (("new", GE), ("old", pre)):
            r = _run([sys.executable, "-u", str(script), "--out", str(tmp_path / name)] + ARGS,
                     timeout=900)
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
    assert "admit_margin" not in ma and ma.keys() == mb.keys()


@needs_map
def test_flag_on_opens_fewer_cells_and_replays(tmp_path):
    outs = {}
    # 4 s bursts, so runs reach the platform's edges and die inside a chunk
    longer = ARGS + ["--decisions", "100", "--max-iters", "4"]
    for name, extra in (("off", []), ("on", ["--admit-margin", "1.0"])):
        out = tmp_path / name
        r = _run([sys.executable, "-u", str(GE), "--out", str(out)] + longer + extra, timeout=900)
        assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
        outs[name] = out
    meta = json.loads((outs["on"] / "archive.meta.json").read_text())
    assert meta["admit_margin"] == 1.0
    n_off = len(np.load(outs["off"] / "archive.npz")["state"])
    n_on = len(np.load(outs["on"] / "archive.npz")["state"])
    assert 1 < n_on < n_off, (n_on, n_off)
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(MAP),
              "--archive", str(outs["on"] / "archive.npz"), "--field", str(FIELD),
              "--out", str(line)], timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
