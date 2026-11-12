"""explore_phase1 --cell-held + ge_replay --held (fx/sib, 2026-11-12): the PLAIN archive - every
state is a cell as before, so flight cells stay stepping stones - plus one bool per cell: does the
map hold the cell's state (contact_rows: walkable ground or a surfable face's push; a root: its
ground flag). ge_replay --held ends the chain at the deepest held cell, so a line handed to the
trainer never ends in a flight (the trainer's banked frontier and the tree's hand-back are read
the same way). A contact-ONLY archive (--cell-contact) drops the stepping stones instead and
saturated on princessburglar (5.2k cells, no crossing where the plain key crossed).

On princessburglar from the spawn: flag off = the explorer before the flag, bit for bit; flag on =
the same archive plus the column; --held replays to a held cell; the refusals."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
sys.path.insert(0, str(ROOT / "tools"))
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402

GE = ROOT / "tools" / "explore_phase1.py"
REPLAY = ROOT / "tools" / "ge_replay.py"
FIELD = ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"
BASE = "624fc14f"     # the commit before the flag
ARGS = ["--map", str(PB), "--envs", "128", "--cell", "64", "--cell-energy", "32",
        "--decisions", "50", "--act-every", "4", "--maxvel", "3500", "--max-iters", "4",
        "--threads", "2", "--seed", "3"]


def _replay(arch, out, extra):
    return subprocess.run([sys.executable, str(REPLAY), "--map", str(PB), "--archive", str(arch),
                           "--field", str(FIELD), "--out", str(out)] + extra,
                          capture_output=True, text=True, cwd=str(ROOT), timeout=600)


@needs_pb
def test_the_plain_archive_plus_one_column_and_a_line_that_ends_held(tmp_path):
    pre = ROOT / "tools" / "_explore_phase1_pre_held.py"
    src = subprocess.run(["git", "show", f"{BASE}:tools/explore_phase1.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--cell-held" not in src
    pre.write_bytes(src)
    try:
        for name, script, extra in (("new", GE, []), ("old", pre, []), ("on", GE, ["--cell-held"])):
            r = _run([sys.executable, "-u", str(script), "--out", str(tmp_path / name),
                      "--exact-chains"] + ARGS + extra)
            # the explorer exits 1 when no run reached the goal (accepted, as everywhere)
            assert r.returncode in (0, 1) and (tmp_path / name / "archive.npz").exists(), \
                r.stdout[-3000:] + r.stderr[-3000:]
    finally:
        pre.unlink(missing_ok=True)
    za, zb = np.load(tmp_path / "new" / "archive.npz"), np.load(tmp_path / "old" / "archive.npz")
    zo = np.load(tmp_path / "on" / "archive.npz")
    assert za.files == zb.files and "held" not in za.files
    assert sorted(zo.files) == sorted(za.files + ["held"])
    for k in za.files:
        assert np.array_equal(za[k], zb[k]), k
        assert np.array_equal(za[k], zo[k]), k                 # the search itself is unchanged
    metas = [json.loads((tmp_path / n / "archive.meta.json").read_text()) for n in ("new", "old", "on")]
    for m in metas:
        for k in ("elapsed_s", "ticks_per_s", "out", "argv", "time"):
            m.pop(k, None)
    assert metas[0] == metas[1] and "cell_held" not in metas[0]
    assert metas[2].pop("cell_held") == 1 and metas[2] == metas[0]
    held = zo["held"]
    og = zo["state"]["onground"] != -1
    assert held.dtype == bool and len(held) == len(zo["state"]) > 100
    assert og.any() and held[og].all()                         # the ground always holds
    assert 0 < int(held.sum()) < len(held)                     # flight cells are kept, unmarked

    out = tmp_path / "line.npy"
    r = _replay(tmp_path / "on" / "archive.npz", out, ["--held"])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    end = int(re.search(r"deepest held cell (\d+):", r.stdout).group(1))
    assert held[end] and "links" in r.stdout
    from ge_replay import GoalField, deepest_cell
    zf = np.load(FIELD, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    assert deepest_cell(zo, f, held=True)[0] == end
    plain, d = deepest_cell(zo, f)
    assert d[plain] <= d[end]
    line = np.load(out)
    assert np.abs(line[-1]["origin"] - zo["state"][end]["origin"]).max() <= 0.5

    # an archive without the column is refused, never read as "all held"
    r = _replay(tmp_path / "new" / "archive.npz", tmp_path / "x.npy", ["--held"])
    assert r.returncode != 0 and "no held column" in r.stdout + r.stderr


@needs_pb
def test_it_needs_exact_chains(tmp_path):
    r = _run([sys.executable, "-u", str(GE), "--out", str(tmp_path / "o"), "--cell-held"] + ARGS)
    assert r.returncode != 0 and "give --exact-chains" in r.stdout + r.stderr
    assert not (tmp_path / "o" / "archive.npz").exists()
