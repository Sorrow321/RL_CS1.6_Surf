"""tools/frontier_archive.py: roots from a greedy eval, cell keys, a few Go-Explore batches on the
real core, the atomic write the trainer's --spawn-states re-read relies on."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import frontier_archive as fa  # noqa: E402

MAP = ROOT / "maps" / "surf_src_sidistic.bsp"
FIELD = ROOT / "maps" / "surf_src_sidistic.goal_32.npz"


def _write_traj(path, episodes, tick_ms=10):
    with open(path, "w", encoding="utf-8") as fh:
        for e, rows in enumerate(episodes):
            fh.write(json.dumps({"map": "surf_src_sidistic", "tick_ms": tick_ms, "episode": e}) + "\n")
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.write(json.dumps({"end": "fail", "ticks": len(rows), "best_progress": 0.0}) + "\n")


def _row(t, p, v, yaw=0.0, buttons=0, onground=0):
    return [t, p[0], p[1], p[2], v[0], v[1], v[2], yaw, buttons, onground, 0.0, 0.0, -5.0, 1, 1]


def _template():
    from surfgym.core import STATE_DTYPE
    return np.zeros(1, STATE_DTYPE)[0]


def test_roots_are_the_last_secs_of_every_episode(tmp_path):
    p = tmp_path / "traj_0000000100.jsonl"
    e1 = [_row(t, (t, 0, 100), (10, 0, 0)) for t in range(1000)]
    e2 = [_row(t, (0, t, 200), (0, 20, 0), buttons=4) for t in range(300)]
    _write_traj(p, [e1, e2])
    r = fa.roots_from_traj(p, _template(), secs=6.0, every=10)
    # episode 1: ticks 400..990 step 10 (60 rows); episode 2 is shorter than 6 s: all of it (30)
    assert len(r) == 90
    assert np.allclose(r["origin"][:60, 0], np.arange(400, 1000, 10))
    assert np.allclose(r["origin"][60:, 1], np.arange(0, 300, 10))
    assert (r["ducked"][60:] == 1).all() and (r["ducked"][:60] == 0).all()
    assert (r["tick"] == 0).all() and (r["onground"] == -1).all()
    # the time base comes from the header: at 5 ms, 6 s is 1,200 ticks
    p2 = tmp_path / "traj_0000000200.jsonl"
    _write_traj(p2, [e1], tick_ms=5)
    assert len(fa.roots_from_traj(p2, _template(), secs=6.0, every=10)) == 100


def test_cell_keys_split_position_and_speed():
    o = np.array([[0, 0, 0], [127, 5, 64], [128, 0, 0], [0, 0, 0], [-1, 0, 0]], np.float64)
    v = np.array([[0, 0, 0], [399, 0, 0], [0, 0, 0], [0, 401, 0], [0, 0, 0]], np.float64)
    k = fa.cell_keys(o, v, 128.0)
    assert k[0] == k[1]                 # same cell, same speed bucket
    assert len({k[0], k[2], k[3], k[4]}) == 4
    vv = np.array([[1e5, 0, 0]], np.float64)
    assert fa.cell_keys(o[:1], vv, 128.0)[0] >> 48 == fa.SPD_MAX_BUCKET


def test_write_atomic_leaves_no_partial_file(tmp_path):
    a = np.zeros(7, np.dtype([("x", np.float32)]))
    out = tmp_path / "s" / "states.npy"
    fa.write_atomic(out, a)
    assert np.load(out).shape == (7,)
    assert not list(out.parent.glob("*.tmp*"))


@pytest.mark.skipif(not MAP.exists() or not FIELD.exists(), reason="sidistic map/field absent")
def test_explore_grows_the_archive_with_valid_states(tmp_path):
    core, field = fa.build(MAP, FIELD, envs=16, threads=1)
    arch = fa.FrontierArchive(core, field, cell=128.0, horizon=120, seed=1)
    # the map's true start (the first row of every greedy eval), on the start platform
    rows = [_row(t, (7837.0, 0.0, 15042.0), (0, 0, 0), yaw=348.9, onground=1) for t in range(50)]
    p = tmp_path / "traj_0000000300.jsonl"
    _write_traj(p, [rows])
    n0 = arch.add_roots(fa.roots_from_traj(p, arch.template, 6.0, 10))
    assert n0 >= 1 and arch.is_root.all()
    for _ in range(4):
        arch.explore()
    assert len(arch.states) > n0
    out = arch.output(cap=10 ** 6)
    n_cells = int((~arch.is_root).sum())
    assert out.dtype == arch.template.dtype and len(out) == n_cells
    # half the rows come from the deepest quarter by d
    dd = field.sample(out["origin"].astype(np.float64))
    q25 = np.quantile(arch.d[~arch.is_root], 0.25)
    assert (dd <= q25 + 1e-6).mean() >= 0.5
    d = field.sample(out["origin"].astype(np.float64))
    assert (d < field.reach_max - 0.5 * field.cell).all()
    for s in out[:20]:
        o = tuple(float(x) for x in s["origin"])
        assert not core.trace(o, o, hull=1 if s["ducked"] else 0).startsolid
    small = arch.output(cap=4)
    assert len(small) == min(4, n_cells)
    fa.write_atomic(tmp_path / "a.npy", out)
    assert np.load(tmp_path / "a.npy").dtype == out.dtype


@pytest.mark.skipif(not MAP.exists() or not FIELD.exists(), reason="sidistic map/field absent")
def test_survivor_gating_admits_nothing_from_a_doomed_root(tmp_path):
    # a root falling at 1,300 u/s 130 u above the lower level's fail floor (z 9,536-9,544) dies
    # within ~0.1 s whatever the keys: with a 0.5 s hold nothing it reaches may become a cell
    core, field = fa.build(MAP, FIELD, envs=16, threads=1)
    arch = fa.FrontierArchive(core, field, cell=128.0, horizon=80, seed=2, hold_secs=0.5)
    assert arch.hold_checks == 13            # 0.5 s at 10 ms = 50 ticks, in 4-tick checks
    rows = [_row(t, (2800.0, 0.0, 9700.0), (0, 0, -1300)) for t in range(10)]
    p = tmp_path / "traj_0000000400.jsonl"
    _write_traj(p, [rows])
    n0 = arch.add_roots(fa.roots_from_traj(p, arch.template, 6.0, 10))
    assert n0 >= 1
    for _ in range(3):
        assert arch.explore() == 0
    assert len(arch.states) == n0
    # without the hold (one check) the same flights do admit their last instants
    arch0 = fa.FrontierArchive(core, field, cell=128.0, horizon=80, seed=2, hold_secs=0.0)
    arch0.add_roots(fa.roots_from_traj(p, arch0.template, 6.0, 10))
    assert arch0.hold_checks == 1
