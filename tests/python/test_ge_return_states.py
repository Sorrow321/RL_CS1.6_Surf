"""tools/ge_return_states.py (uf2r1b2, 2026-10-30): the archive's Go-Explore return weight as a
spawn file - W = 1/sqrt(seen + 1), x (1 + K) on each place's most energetic cell."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import ge_return_states as G  # noqa: E402


def _archive(tmp_path):
    st = np.zeros(4, STATE_DTYPE)
    # cells 0 and 1 share one 64 u place (cell 1 faster = more energetic); 2, 3 are elsewhere
    st["origin"] = [[10, 10, 10], [20, 20, 20], [500, 10, 10], [10, 500, 10]]
    st["velocity"] = [[100, 0, 0], [900, 0, 0], [0, 0, 0], [0, 0, 0]]
    st["tick"] = 77
    seen = np.array([0, 0, 3, 0], np.int64)
    p = tmp_path / "archive.npz"
    np.savez(p, state=st, seen=seen)
    meta = {"cell": 64.0, "grid_mins": [0.0, 0.0, 0.0], "grid_dims": [16, 16, 16],
            "select_energy": 3.0}
    p.with_suffix(".meta.json").write_text(json.dumps(meta))
    return p


def test_weights_follow_the_explorer_rule(tmp_path):
    from explore_phase1 import CellHash
    p = _archive(tmp_path)
    z = np.load(p)
    h = CellHash(mins=[0, 0, 0], maxs=[1e9] * 3, cell=64.0)
    h.dims = (16, 16, 16)
    w = G.return_weights(z["state"], z["seen"], h, 3.0)
    # cell 1 tops its place (x4); cell 0 does not; 2 and 3 top their own places (x4)
    assert np.allclose(w, [1.0, 4.0, 4.0 / 2.0, 4.0])


def test_file_draws_with_that_weight(tmp_path):
    p = _archive(tmp_path)
    out = tmp_path / "ret.npy"
    assert G.main(["--archive", str(p), "--out", str(out), "--n", "40000", "--seed", "1"]) == 0
    r = np.load(out)
    assert r.dtype == STATE_DTYPE and len(r) == 40000
    assert np.all(r["tick"] == 0) and np.all(r["stuck_ticks"] == 0)
    x = r["origin"][:, 0].astype(int)
    y = r["origin"][:, 1].astype(int)
    frac = np.array([np.mean((x == 10) & (y == 10)), np.mean(x == 20), np.mean(x == 500),
                     np.mean(y == 500)])
    want = np.array([1.0, 4.0, 2.0, 4.0]) / 11.0
    assert np.allclose(frac, want, atol=0.01), frac


def test_tops_writes_each_places_most_energetic_cell_once(tmp_path):
    p = _archive(tmp_path)
    out = tmp_path / "tops.npy"
    assert G.main(["--archive", str(p), "--out", str(out), "--tops"]) == 0
    r = np.load(out)
    # three places: cell 1 (the faster of the shared place), cells 2 and 3
    assert len(r) == 3
    assert sorted(r["origin"][:, 0].astype(int).tolist()) == [10, 20, 500]
    # the slower cell of the shared place (cell 0, at (10, 10, 10)) is not written
    assert not np.any((r["origin"][:, 0] == 10.0) & (r["origin"][:, 1] == 10.0))


def test_not_behind_roots_drops_rows_behind_the_roots(tmp_path):
    # a 1-D toy field along x: d = 1000 - x on a 16-cell grid of 64 u (GoalField grids are
    # (nz, ny, nx)); the root is at x 300 (d 700)
    st = np.zeros(4, STATE_DTYPE)
    st["origin"] = [[300, 32, 32], [500, 32, 32], [100, 32, 32], [700, 32, 32]]
    st["velocity"] = [[10, 0, 0], [10, 0, 0], [10, 0, 0], [10, 0, 0]]
    p = tmp_path / "archive.npz"
    np.savez(p, state=st, seen=np.zeros(4, np.int64), parent=np.array([-1, 0, 0, 1]))
    p.with_suffix(".meta.json").write_text(json.dumps(
        {"cell": 64.0, "grid_mins": [0.0, 0.0, 0.0], "grid_dims": [16, 16, 16]}))
    xs = (np.arange(16) + 0.5) * 64.0
    grid = np.ascontiguousarray(np.broadcast_to((1000.0 - xs)[None, None, :], (4, 4, 16)),
                                np.float32)
    fp = tmp_path / "field.npz"
    np.savez(fp, grid=grid, quant=np.float32(1.0), mins=np.zeros(3, np.float32),
             cell=np.float32(64.0), reach_max=np.float32(1e6))
    out = tmp_path / "tops.npy"
    assert G.main(["--archive", str(p), "--out", str(out), "--tops",
                   "--not-behind-roots", str(fp)]) == 0
    xs_out = sorted(np.load(out)["origin"][:, 0].astype(int).tolist())
    assert xs_out == [300, 500, 700]          # x 100 lies behind the root (d 900 > 700)
