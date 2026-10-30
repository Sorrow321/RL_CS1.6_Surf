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
