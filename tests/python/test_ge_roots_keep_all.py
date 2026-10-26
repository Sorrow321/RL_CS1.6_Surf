"""explore_phase1 --roots-keep-all (pbr5b2, 2026-10-26): every root gets its own cell even when two
share a key - a boosted copy of a state (auto_spine's --ge-boost) usually shares its original's
64 u x energy cell and was dropped as a duplicate before any exploration (pbr5b2's GE-H: 144 roots
-> 48 cells). (a) Archive.add_root on its own; (b) the explorer on princessburglar: flag off
archive + meta bit-identical to the explorer before it, flag on all roots kept and the meta says so;
ge_replay still replays its deepest chain exactly."""
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
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402

GE = ROOT / "tools" / "explore_phase1.py"
SPINE = ROOT / "autoresearch" / "spines" / "pbr4b3_beam_g11.npy"
FIELD = ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"
BASE = "a480992"      # the commit before the flag
ARGS = ["--map", str(PB), "--envs", "128", "--cell", "64", "--cell-energy", "32",
        "--decisions", "50", "--act-every", "4", "--maxvel", "3500", "--max-iters", "3",
        "--threads", "2", "--exact-chains", "--seed", "3"]


def test_add_root_keeps_duplicates():
    sys.path.insert(0, str(ROOT / "python"))
    import explore_phase1 as ep
    from surfgym.core import STATE_DTYPE
    h = ep.CellHash(np.zeros(3), np.full(3, 1000.0), cell=64.0, energy=32.0)
    a = ep.Archive(h)
    s = np.zeros(3, STATE_DTYPE)
    s["origin"][:] = 100.0
    s["velocity"][:, 0] = [480.0, 485.0, 900.0]      # 0 and 1 share an energy bucket, 2 does not
    k = h.keys(s)
    assert k[0] == k[1] != k[2]
    idx = [a.add_root(int(k[j]), s[j]) for j in range(3)]
    assert idx == [0, 1, 2] and a.size == 3
    assert a.index[int(k[0])] == 0                   # the index keeps the first
    assert (a.parent[:3] == -1).all() and (a.depth[:3] == 0).all()
    b = ep.Archive(h)
    assert [b.add(int(k[j]), s[j]) for j in range(3)] == [0, 0, 1] and b.size == 2


def _roots(tmp_path) -> Path:
    s = np.load(SPINE)
    r = s[1900:2100:25].copy()
    r["tick"] = 0
    r["stuck_ticks"] = 0
    r2 = r.copy()
    r2["velocity"] = (r2["velocity"].astype(np.float64) * 1.003).astype(r2["velocity"].dtype)
    p = tmp_path / "roots.npy"
    np.save(p, np.concatenate([r, r2]))
    return p


@needs_pb
def test_explorer_flag_off_bit_identical_and_on_keeps_all(tmp_path):
    roots = _roots(tmp_path)
    pre = ROOT / "tools" / "_explore_phase1_pre_rka.py"
    src = subprocess.run(["git", "show", f"{BASE}:tools/explore_phase1.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--roots-keep-all" not in src
    pre.write_bytes(src)
    try:
        for name, script, extra in (("new", GE, []), ("old", pre, []), ("on", GE, ["--roots-keep-all"])):
            r = _run([sys.executable, "-u", str(script), "--out", str(tmp_path / name),
                      "--roots-spine", str(roots)] + ARGS + extra)
            # the explorer exits 1 when no run reached the goal (accepted, as everywhere)
            assert r.returncode in (0, 1) and (tmp_path / name / "archive.npz").exists(), \
                r.stdout[-3000:] + r.stderr[-3000:]
    finally:
        pre.unlink(missing_ok=True)
    za, zb = np.load(tmp_path / "new" / "archive.npz"), np.load(tmp_path / "old" / "archive.npz")
    assert za.files == zb.files
    for k in za.files:
        assert np.array_equal(za[k], zb[k]), k
    ma = json.loads((tmp_path / "new" / "archive.meta.json").read_text())
    mb = json.loads((tmp_path / "old" / "archive.meta.json").read_text())
    for m in (ma, mb):
        for k in ("elapsed_s", "ticks_per_s", "out", "argv", "time"):
            m.pop(k, None)
    assert ma == mb and "roots_keep_all" not in ma
    zo = np.load(tmp_path / "on" / "archive.npz")
    mo = json.loads((tmp_path / "on" / "archive.meta.json").read_text())
    assert mo.get("roots_keep_all") is True
    n_roots = len(np.load(roots))
    assert int((zo["parent"] == -1).sum()) == n_roots > int((za["parent"] == -1).sum())
    out = tmp_path / "on" / "line.npy"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
                        "--archive", str(tmp_path / "on" / "archive.npz"), "--field", str(FIELD),
                        "--out", str(out)], capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "links" in r.stdout and out.exists()
