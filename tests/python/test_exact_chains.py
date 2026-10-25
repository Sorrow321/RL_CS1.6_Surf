"""explore_phase1 --exact-chains + tools/ge_replay.py (pbr1b1, 2026-10-23).

(a) with the flag every cell's parent is the cell its run was restored from, states are never
    replaced, and the saved link actions replay every link of the deepest chain exactly
    (tools/ge_replay.py checks each link's end against the child's archived state);
(b) flag off: the archive and its meta are bit-identical to the explorer before the change.
"""
from __future__ import annotations

import json
import os
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
ARGS = ["--map", str(PB), "--envs", "128", "--cell", "64", "--cell-energy", "32",
        "--decisions", "50", "--act-every", "4", "--maxvel", "3500", "--max-iters", "3",
        "--threads", "2"]


@needs_pb
def test_exact_chains_replay_exactly(tmp_path):
    out = tmp_path / "ge"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--exact-chains"] + ARGS,
             timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    z = np.load(out / "archive.npz", allow_pickle=False)
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["exact_chains"] is True and meta["maxvel"] == 3500.0
    par, off, dep = z["parent"], z["link_off"], z["depth"]
    assert len(off) == len(par) + 1
    roots = par < 0
    lens = np.diff(off)
    assert (lens[roots] == 0).all() and (lens[~roots] > 0).all()
    # depth = parent's depth + the link's ticks, exactly (the parent is the restore cell)
    assert (dep[~roots] == dep[par[~roots]] + lens[~roots]).all()
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
              "--archive", str(out / "archive.npz"), "--field", str(FIELD), "--out", str(line)],
             timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
    st = np.load(line, allow_pickle=False)
    assert len(st) > 1 and (st["tick"] == 0).all()


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:tools/explore_phase1.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--exact-chains" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "tools" / "_explore_phase1_pre_exact.py"
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
    assert a.files == b.files and "link_acts" not in a.files
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k
    assert (json.loads((tmp_path / "new" / "archive.meta.json").read_text())
            == json.loads((tmp_path / "old" / "archive.meta.json").read_text()))


@needs_pb
def test_exact_chains_at_a_tick_pattern(tmp_path):
    """--tick-ms 7.63 ([8, 8, 7] ms): the archive stores each link's pattern phase and
    ge_replay replays the deepest chain exactly at that tick (pbr4b2)."""
    out = tmp_path / "ge"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--exact-chains",
              "--tick-ms", "7.63"] + ARGS, timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    z = np.load(out / "archive.npz", allow_pickle=False)
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["tick_ms"] == 7.63
    assert "link_phase" in z.files and len(z["link_phase"]) == len(z["parent"])
    assert set(np.unique(z["link_phase"]).tolist()) <= {0, 1, 2}
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
              "--archive", str(out / "archive.npz"), "--field", str(FIELD), "--out", str(line)],
             timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
