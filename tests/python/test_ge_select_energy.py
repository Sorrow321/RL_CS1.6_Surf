"""explore_phase1 --select-energy K (pbr5b2, 2026-10-26): the most energetic cell (z + |v|^2 / 2g) of
each --cell place is drawn (1 + K) times as often as W = 1/sqrt(C_seen+1) says - count-based
selection expanded GE-G's one fast cell in the lane gate 19 needs no more often than the slow ones.
(a) the weights on a synthetic archive (K = 0: the Nature cumsum bit for bit); (b) the explorer on
princessburglar: flag off archive + meta bit-identical to the explorer before it, flag on the meta
says so and ge_replay still replays its deepest chain exactly; (c) auto_spine passes the flag only
when it is set."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
sys.path.insert(0, str(ROOT / "tools"))
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402
from test_ge_roots_keep_all import ARGS, FIELD, GE, _roots  # noqa: E402

BASE = "3a7d523"      # the commit before the flag


def test_select_energy_weights():
    sys.path.insert(0, str(ROOT / "python"))
    import explore_phase1 as ep
    from surfgym.core import STATE_DTYPE
    h = ep.CellHash(np.zeros(3), np.full(3, 1000.0), cell=64.0, energy=32.0)
    s = np.zeros(4, STATE_DTYPE)
    s["origin"][:] = [[100, 100, 100], [110, 100, 100], [120, 100, 100], [500, 100, 100]]
    s["velocity"][:, 0] = [300.0, 600.0, 450.0, 200.0]   # 0-2 share a place, 1 is its fastest
    k = h.keys(s)
    pk = h.pos_keys(s)
    assert len(set(k[:3].tolist())) == 3 and len(set(pk[:3].tolist())) == 1 and pk[3] != pk[0]
    seen = np.array([0, 3, 0, 8], np.int64)
    base = 1.0 / np.sqrt(seen.astype(np.float64) + 1.0)
    for K, mult in ((0.0, [1, 1, 1, 1]), (3.0, [1, 4, 1, 4])):
        a = ep.Archive(h)
        a.select_energy = K
        for j in range(4):
            a.add(int(k[j]), s[j])
        a.seen[:4] = seen
        a.build_sampler()
        if K == 0.0:
            assert np.array_equal(a._cum, np.cumsum(base))
        w = np.diff(np.concatenate([[0.0], a._cum]))
        assert np.allclose(w, base * np.asarray(mult, np.float64))
    assert sorted(a.energy_tops().tolist()) == [1, 3]


@needs_pb
def test_explorer_flag_off_bit_identical_and_on_runs(tmp_path):
    roots = _roots(tmp_path)
    pre = ROOT / "tools" / "_explore_phase1_pre_sel.py"
    src = subprocess.run(["git", "show", f"{BASE}:tools/explore_phase1.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--select-energy" not in src
    pre.write_bytes(src)
    try:
        for name, script, extra in (("new", GE, []), ("old", pre, []),
                                    ("on", GE, ["--select-energy", "3"])):
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
    assert ma == mb and "select_energy" not in ma
    mo = json.loads((tmp_path / "on" / "archive.meta.json").read_text())
    assert mo.get("select_energy") == 3.0
    zo = np.load(tmp_path / "on" / "archive.npz")
    assert not all(np.array_equal(za[k], zo[k]) for k in za.files)    # the draws moved
    out = tmp_path / "on" / "line.npy"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
                        "--archive", str(tmp_path / "on" / "archive.npz"), "--field", str(FIELD),
                        "--out", str(out)], capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "links" in r.stdout and out.exists()


def test_auto_spine_passes_select_energy_only_when_set(tmp_path, monkeypatch):
    import auto_spine
    from surfgym.core import STATE_DTYPE
    seen = []
    monkeypatch.setattr(auto_spine, "run", lambda cmd, log, timeout: seen.append(
        [str(c) for c in cmd]) or 1)
    line = np.zeros(200, STATE_DTYPE)
    line["origin"][:, 0] = np.arange(200.0)
    ns = auto_spine.main.__globals__["argparse"].Namespace
    base = dict(map="m.bsp", envs=64, ge_cell=64.0, ge_energy=32.0, act_every=4, ge_minutes=1.0,
                seed=0, maxvel=None, threads=0, tick=10.0, ge_view_follow="none", roots_secs=1.0,
                roots_n=8, ge_boost="1.0", ge_cell_heading=1, ge_keep_all=1, field="f.npz")
    auto_spine.go_explore(ns(**base), tmp_path / "off", line, 150, boosts=[1.0])
    auto_spine.go_explore(ns(**base, ge_select_energy=0.0), tmp_path / "zero", line, 150,
                          boosts=[1.0])
    auto_spine.go_explore(ns(**base, ge_select_energy=3.0), tmp_path / "on", line, 150,
                          boosts=[1.0])
    assert "--select-energy" not in seen[0] and "--select-energy" not in seen[1]
    i = seen[2].index("--select-energy")
    assert float(seen[2][i + 1]) == 3.0
