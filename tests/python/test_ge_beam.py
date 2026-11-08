"""tools/ge_beam.py (unr1b2): the archive's bookkeeping, exact chain replay, and the trainer flag.

* link_row_index: a run that began mid-hold applies row 0 for h0 ticks, then one row per K;
* Archive: a replacement makes a NEW node (the old one stays for its children's chains), the
  cell's statistics move with the cell, save/load round-trips;
* Explorer on the real map with random runs only (no network): every archived cell's chain
  replays link by link onto its archived state (the restore is the core's own reset with yaw
  jitter 0), the line has one state per tick of the chain's depth;
* train_fast: --ge-beam is off by default and every config key it writes is TRAIN_ONLY in
  record_ckpt (the record gate refuses an undeclared key).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import ge_beam as gb                                                     # noqa: E402
from surfgym.core import STATE_DTYPE                                     # noqa: E402

MAP = ROOT / "maps" / "surf_src_sidistic.bsp"


def test_link_row_index():
    t = np.arange(12)
    assert list(gb.link_row_index(t, 0, 4)) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert list(gb.link_row_index(t, 3, 4)) == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]


def test_archive_replace_and_roundtrip(tmp_path):
    A = gb.Archive(cap=4)
    s = np.zeros(1, STATE_DTYPE)[0]
    r = A.add(7, s, -1, 0, 100.0, 0, 0, 0, gb.MODE_ROOT, 0)
    a = A.add(11, s, r, 50, 90.0, 50, 0, 13, gb.MODE_POLICY, 1)
    A.seen[a] = 5
    b = A.add(11, s, r, 30, 90.0, 30, 2, 8, gb.MODE_RANDOM, 2)      # a sooner arrival
    assert A.index[11] == b and A.cur[b] and not A.cur[a]
    assert A.seen[b] == 5                                             # the cell's count moved
    c = A.add(13, s, a, 80, 80.0, 30, 0, 8, gb.MODE_POLICY, 3)        # a child of the old node
    assert A.chain(c) == [r, a, c]
    for k in range(10):                                               # growth past cap
        A.add(100 + k, s, r, 1, 1.0, 1, 0, 1, gb.MODE_POLICY, 4)
    off = A.store_link(np.ones((5, 6), np.int8), np.full((5, 2), 2.0, np.float32))
    A.link[c] = off
    A.save(tmp_path / "a.npz")
    B = gb.Archive.load(tmp_path / "a.npz")
    assert B.n == A.n and B.index == A.index and B.nlink == A.nlink
    assert np.array_equal(B.parent[:B.n], A.parent[:A.n])
    assert np.array_equal(B.rows[off:off + 5], A.rows[off:off + 5])
    rng = np.random.default_rng(0)
    sel = B.select(rng, 32, 0.25, 4)
    assert len(sel) == 32 and B.cur[sel].all()


@pytest.mark.skipif(not MAP.exists(), reason="the map is not in this checkout")
def test_random_explorer_chains_replay_exactly(tmp_path):
    pytest.importorskip("torch")
    from surfgym import SurfCore  # noqa: F401  (the core library must load)
    cfg = {"reward": "race", "act_every": 4, "view_continuous": 1, "view_absolute": "velocity",
           "keys_hold": True, "maxvel": 4000.0, "ep_ticks": 30000, "lidar_cell": 32.0,
           "race_no_corner_cut": 1, "tick_ms": 10.0, "pitch_rate": 1.33, "yaw_adaptive": 1}
    import torch
    ck = tmp_path / "cfg.pt"
    torch.save({"config": cfg, "global_step": 0}, ck)
    ctx = gb.setup(str(ck), str(MAP), 24, torch.device("cpu"), with_policy=False)
    P = SimpleNamespace(cell=64.0, cell_energy=128.0, cell_heading=4, cells=6, decisions=40,
                        random_frac=1.0, repeat=0.95, keys_temp=1.0, view_temp=1.0,
                        view_hold=[0.0, 25.0], front_frac=0.25, front_k=8, replace_ticks=25,
                        viable_ticks=10, gain=1.0, out=str(tmp_path), resume=False, burst=0,
                        max_nodes=10**7)
    ex = gb.Explorer(ctx, P, np.random.default_rng(1))
    for _ in range(6):
        ex.step()
    A = ex.A
    assert A.n > 20
    assert (A.link[1:A.n] >= 0).all()                     # every link was stored at run end
    depth_links = [len(A.chain(i)) - 1 for i in range(A.n)]
    deep = int(np.argmax(depth_links))
    assert depth_links[deep] >= 2                          # a chain through restored cells
    for node in {deep, int(np.argmax(A.depth[:A.n]))}:
        line = ex.replay_line(node)                        # raises if any link is off
        assert len(line) == int(A.depth[node]) + 1
        assert np.allclose(line[-1]["origin"], A.state[node]["origin"], atol=1e-3)
        assert (line["tick"] == 0).all()
    info = ex.maybe_export(tmp_path)
    assert info is None or Path(info["line"]).exists()


def test_trainer_flag_off_and_train_only():
    import re
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    keys = set(re.findall(r"\b(ge_beam(?:_[a-z]+)*)=", src.split("--ge-beam (unr1b2): written "
                                                                 "ONLY when on")[1][:1500]))
    assert "ge_beam" in keys and len(keys) >= 10
    import record_ckpt
    missing = sorted(k for k in keys if k not in record_ckpt.TRAIN_ONLY)
    assert not missing, missing
    import train_fast
    ap_src = src[src.index('ap.add_argument("--ge-beam", action="store_true"'):][:200]
    assert "store_true" in ap_src
    assert not hasattr(train_fast, "GEB")                 # the state lives inside train()


@pytest.mark.skipif(not MAP.exists(), reason="the map is not in this checkout")
def test_burst_runs_replay_exactly(tmp_path):
    """--burst: held random rows at a run's start (no network: the 'policy' rows are the neutral
    row the no-policy explorer emits), the burst rows and the rest replay exactly, and the burst
    kind is recorded on its cells."""
    pytest.importorskip("torch")
    import torch
    cfg = {"reward": "race", "act_every": 4, "view_continuous": 1, "view_absolute": "velocity",
           "keys_hold": True, "maxvel": 4000.0, "ep_ticks": 30000, "lidar_cell": 32.0,
           "race_no_corner_cut": 1, "tick_ms": 10.0, "pitch_rate": 1.33, "yaw_adaptive": 1}
    ck = tmp_path / "cfg.pt"
    torch.save({"config": cfg, "global_step": 0}, ck)
    ctx = gb.setup(str(ck), str(MAP), 16, torch.device("cpu"), with_policy=False)
    P = SimpleNamespace(cell=64.0, cell_energy=128.0, cell_heading=4, cells=4, decisions=30,
                        random_frac=0.0, repeat=0.9, keys_temp=1.0, view_temp=1.0,
                        view_hold=[0.0, 25.0], front_frac=0.25, front_k=8, replace_ticks=25,
                        viable_ticks=10, gain=1.0, out=str(tmp_path), resume=False, burst=10,
                        max_nodes=10**7)
    ex = gb.Explorer(ctx, P, np.random.default_rng(2))
    for _ in range(5):
        ex.step()
    A = ex.A
    assert (A.mode[1:A.n] == gb.MODE_BURST).all()
    deep = int(np.argmax([len(A.chain(i)) for i in range(A.n)]))
    line = ex.replay_line(deep)
    assert len(line) == int(A.depth[deep]) + 1


def test_frontier_skips_doomed_cells():
    """--front-skip-doomed: the frontier share never draws a cell restored >= 2 times from which
    no run lived the viable time (the fatal flight past a wall), and falls back to them only
    when nothing else is left."""
    A = gb.Archive()
    s = np.zeros(1, STATE_DTYPE)[0]
    for i in range(300):
        A.add(i, s, -1, 0, float(i), 0, 0, 0, gb.MODE_POLICY, 0)
    A.chosen[:50] = 3
    A.surv[:50] = 5                 # the 50 deepest: explored, every run died at once
    A.surv[50:] = 1000
    A.chosen[50:60] = 1             # explored once: not yet judged
    rng = np.random.default_rng(0)
    x = A.select(rng, 400, 1.0, 16, 0.0, 100)
    assert x.min() >= 50 and x.max() < 66
    x = A.select(rng, 400, 1.0, 16, 0.0, 0)
    assert x.max() < 16             # the old rule draws the doomed cells
    A2 = gb.Archive()
    for i in range(3):
        A2.add(i, s, -1, 0, float(i), 0, 0, 0, gb.MODE_POLICY, 0)
    A2.chosen[:3] = 5
    A2.surv[:3] = 1
    assert len(A2.select(rng, 10, 1.0, 2, 0.0, 100)) == 10      # all doomed: still a selection


@pytest.mark.skipif(not MAP.exists(), reason="the map is not in this checkout")
def test_chain_roots(tmp_path):
    """--chain-frac: the roots are nodes on the deepest viable cell's own chain, within the last
    --chain-secs of its depth."""
    pytest.importorskip("torch")
    import torch
    cfg = {"reward": "race", "act_every": 4, "view_continuous": 1, "view_absolute": "velocity",
           "keys_hold": True, "maxvel": 4000.0, "ep_ticks": 30000, "lidar_cell": 32.0,
           "race_no_corner_cut": 1, "tick_ms": 10.0, "pitch_rate": 1.33, "yaw_adaptive": 1}
    ck = tmp_path / "cfg.pt"
    torch.save({"config": cfg, "global_step": 0}, ck)
    ctx = gb.setup(str(ck), str(MAP), 16, torch.device("cpu"), with_policy=False)
    P = SimpleNamespace(cell=64.0, cell_energy=128.0, cell_heading=4, cells=4, decisions=30,
                        random_frac=1.0, repeat=0.9, keys_temp=1.0, view_temp=1.0,
                        view_hold=[0.0, 25.0], front_frac=0.25, front_k=8, replace_ticks=25,
                        viable_ticks=10, gain=1.0, out=str(tmp_path), resume=False, burst=0,
                        max_nodes=10**7, chain_frac=0.5, chain_secs=0.5)
    ex = gb.Explorer(ctx, P, np.random.default_rng(3))
    for _ in range(5):
        ex.step()
    b = ex.best_viable()
    roots = ex.chain_roots(0.5)
    if b >= 0:
        chain = set(ex.A.chain(b))
        assert len(roots) and set(roots.tolist()) <= chain
        assert (ex.A.depth[roots] >= ex.A.depth[b] - 50).all()


@pytest.mark.skipif(not MAP.exists(), reason="the map is not in this checkout")
def test_trace_step_starts_at_the_root(tmp_path):
    """--trace-every: a trace step restores only the root cells (every run is a chain from the
    true start) and runs --trace-decisions; its cells replay exactly like any other."""
    pytest.importorskip("torch")
    import torch
    cfg = {"reward": "race", "act_every": 4, "view_continuous": 1, "view_absolute": "velocity",
           "keys_hold": True, "maxvel": 4000.0, "ep_ticks": 30000, "lidar_cell": 32.0,
           "race_no_corner_cut": 1, "tick_ms": 10.0, "pitch_rate": 1.33, "yaw_adaptive": 1}
    ck = tmp_path / "cfg.pt"
    torch.save({"config": cfg, "global_step": 0}, ck)
    ctx = gb.setup(str(ck), str(MAP), 16, torch.device("cpu"), with_policy=False)
    P = SimpleNamespace(cell=64.0, cell_energy=128.0, cell_heading=4, cells=4, decisions=10,
                        random_frac=1.0, repeat=0.9, keys_temp=1.0, view_temp=1.0,
                        view_hold=[0.0, 25.0], front_frac=0.25, front_k=8, replace_ticks=25,
                        viable_ticks=10, gain=1.0, out=str(tmp_path), resume=False, burst=0,
                        max_nodes=10**7, chain_frac=0.5, chain_secs=0.5, trace_decisions=60)
    ex = gb.Explorer(ctx, P, np.random.default_rng(4))
    ex.step()
    n0 = ex.A.n
    chosen0 = ex.A.chosen[:n0].copy()
    ex.step(trace=True)
    A = ex.A
    root = np.flatnonzero(A.mode[:A.n] == gb.MODE_ROOT)
    grew = np.flatnonzero(A.chosen[:n0] > chosen0)
    assert set(grew.tolist()) <= set(root.tolist())          # only the roots were restored
    new = np.arange(n0, A.n)
    assert len(new) and set(A.parent[new].tolist()) <= set(root.tolist())
    deep = new[int(np.argmax(A.depth[new]))]
    assert A.depth[deep] > 10 * 4                             # longer than a normal step
    line = ex.replay_line(deep)
    assert len(line) == int(A.depth[deep]) + 1


def test_write_spawns_band(tmp_path):
    """--spawn-file: only viable, non-root current cells, the deepest --spawn-band of them, with
    tick / stuck zeroed."""
    A = gb.Archive()
    s = np.zeros(1, STATE_DTYPE)[0]
    for i in range(200):
        st = np.array(s, copy=True)
        st["tick"] = 7
        st["origin"] = [i, 0, 0]
        A.add(i, st, -1, 0, float(i), 0, 0, 0, gb.MODE_ROOT if i == 0 else gb.MODE_POLICY, 0)
    A.surv[:200] = 1000
    A.surv[:20] = 0                                   # the 20 deepest are not viable
    ex = gb.Explorer.__new__(gb.Explorer)
    ex.A = A
    ex.P = SimpleNamespace(viable_ticks=100)
    ex.rng = np.random.default_rng(0)
    f = tmp_path / "sp.npy"
    n = ex.write_spawns(f, 0.1, 10_000)
    rows = np.load(f)
    assert n == len(rows) == 18                       # ceil(0.1 x 180 viable)
    xs = sorted(int(x) for x in rows["origin"][:, 0])
    assert xs == list(range(20, 38))
    assert (rows["tick"] == 0).all()
