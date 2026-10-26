"""explore_phase1 --tick-pattern / ge_replay / auto_spine survival (pbr5b1, 2026-10-26).

A 7.63 ms checkpoint flies the integer-ms PATTERN 8,8,7, and a precision contact the explorer
finds at the rounded 8 ms may not exist there (gate 19's edge-hop died at 8,8,7 in every phase).
The explorer now cycles the pattern (phase reset at every iteration), stores each exact-chain
link's start phase, and ge_replay replays it.

(a) --tick-pattern 8,8,7: the meta and the archive say so, and the chains replay exactly;
(b) flag off: archive and meta bit-identical to the explorer before the change;
(c) auto_spine.survives: a state falling into a kill volume does not survive, a standing one does.
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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402

GE = ROOT / "tools" / "explore_phase1.py"
FIELD = ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"
ARGS = ["--map", str(PB), "--envs", "128", "--cell", "64", "--cell-energy", "32",
        "--decisions", "50", "--act-every", "4", "--maxvel", "3500", "--max-iters", "3",
        "--threads", "2", "--exact-chains"]


@needs_pb
def test_tick_pattern_chains_replay_exactly(tmp_path):
    r = _run([sys.executable, "-u", str(GE), "--out", str(tmp_path / "tp")] + ARGS
             + ["--tick-pattern", "8,8,7"], timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.loads((tmp_path / "tp" / "archive.meta.json").read_text())
    assert meta["tick_pattern"] == [8, 8, 7]
    z = np.load(tmp_path / "tp" / "archive.npz")
    assert "link_phase" in z.files and set(np.unique(z["link_phase"])) <= {0, 1, 2}
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
              "--archive", str(tmp_path / "tp" / "archive.npz"), "--field", str(FIELD),
              "--out", str(line)], timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
    assert len(np.load(line, allow_pickle=False)) > 1


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:tools/explore_phase1.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b'"--tick-pattern"' not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "tools" / "_explore_phase1_pre_pattern.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change explorer in the first-parent history")
        for name, script in (("new", GE), ("old", pre)):
            r = _run([sys.executable, "-u", str(script), "--out", str(tmp_path / name)] + ARGS,
                     timeout=900)
            assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    finally:
        pre.unlink(missing_ok=True)
    a = np.load(tmp_path / "new" / "archive.npz")
    b = np.load(tmp_path / "old" / "archive.npz")
    assert sorted(a.files) == sorted(b.files) and "link_phase" not in a.files
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k
    ma = json.loads((tmp_path / "new" / "archive.meta.json").read_text())
    mb = json.loads((tmp_path / "old" / "archive.meta.json").read_text())
    for m in (ma, mb):
        m.pop("seconds", None), m.pop("out", None)
    assert "tick_pattern" not in ma
    assert {k: v for k, v in ma.items() if "time" not in k and "sec" not in k} == \
        {k: v for k, v in mb.items() if "time" not in k and "sec" not in k}


@needs_pb
def test_survives_rejects_a_fall_into_a_kill_volume():
    import auto_spine as A
    from surfgym.core import STATE_DTYPE
    from surfgym.zones import kill_zones
    # a state just above a kill volume, falling fast into it, cannot survive half a second;
    # the same position at rest on top of the volume's footprint is not tested (no floor
    # guaranteed) - instead a fresh map spawn (standing in the start room) must survive
    k = [z for z in kill_zones(str(PB)) if (np.array(z["maxs"]) - np.array(z["mins"]))[2] < 40]
    assert k, "no thin kill sheet on the map"
    mn, mx = np.array(k[0]["mins"], float), np.array(k[0]["maxs"], float)
    st = np.zeros(1, STATE_DTYPE)
    st["origin"][0] = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mx[2] + 60.0]
    st["velocity"][0] = [0.0, 0.0, -1500.0]
    st["onground"][0] = -1
    assert not A.survives(str(PB), st[0], 3500, 7.667, [8, 8, 7])
    zones = json.loads(Path(str(PB)).with_suffix(".zones.json").read_text())
    sp = zones.get("spawn") or zones.get("spawns")
    if isinstance(sp, list):
        sp = sp[0]
    origin = sp.get("origin") if isinstance(sp, dict) else None
    if origin is None:
        pytest.skip("no spawn origin in the zones file")
    st2 = np.zeros(1, STATE_DTYPE)
    st2["origin"][0] = origin
    st2["onground"][0] = 0
    assert A.survives(str(PB), st2[0], 3500, 7.667, [8, 8, 7])
