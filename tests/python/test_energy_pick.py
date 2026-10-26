"""tools/energy_pick.py (pbr5b2, 2026-10-26): the exact-chains archive cell that keeps the most
energy height over the policy's reference line at matched progress, among non-plunging on-route
cells whose chains are not slower than the line. Synthetic archive on a field whose d falls along x."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402


class F:                                   # d = 10,000 - x, reachable everywhere
    reach_max, cell = 1e9, 32.0

    def sample(self, o):
        return 10000.0 - np.asarray(o, np.float64)[:, 0]


def _line(n=1000):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float64)          # 1 u per tick, progress 0..10%
    s["origin"][:, 2] = 2000.0
    s["velocity"][:, 0] = 400.0
    return s


def test_pick_prefers_energy_and_rejects_slow_plunging_and_offroute():
    import energy_pick
    line = _line()
    st = np.zeros(6, STATE_DTYPE)
    parent = np.array([-1, 0, 0, 0, 0, 0])
    depth = np.array([0, 100, 100, 900, 100, 100])
    st[0] = line[200]                                            # the root: line row 200
    for i, (x, vx, vz, y) in enumerate([(300, 600, 0, 0),         # 1: +energy, on time
                                        (300, 500, 0, 0),         # 2: less energy
                                        (300, 900, 0, 0),         # 3: most energy, but SLOW (900 ticks)
                                        (300, 900, -800, 0),      # 4: most energy, plunging
                                        (300, 900, 0, 500)],      # 5: most energy, off the route
                                       start=1):
        st[i]["origin"] = (x, y, 2000.0)
        st[i]["velocity"] = (vx, 0.0, vz)
    best, cands = energy_pick.pick(st, parent, depth, line, F(), 10000.0, 2.5, 3.5,
                                   near=150.0, slack=130)
    assert best["cell"] == 1 and [c["cell"] for c in cands] == [1, 2]
    assert best["gain"] > 0 and abs(best["h_line"] - (2000 + 400 ** 2 / 1600)) < 1e-6
    assert best["line_ticks"] == 100


def test_cli_writes_the_winner(tmp_path):
    import json
    import energy_pick
    line = _line()
    np.save(tmp_path / "line.npy", line)
    st = np.zeros(2, STATE_DTYPE)
    st[0] = line[200]
    st[1]["origin"] = (300, 0, 2000.0)
    st[1]["velocity"] = (700, 0.0, 0.0)
    np.savez(tmp_path / "arch.npz", state=st, parent=np.array([-1, 0]), depth=np.array([0, 100]),
             link_acts=np.zeros((100, 6), np.int64), link_off=np.array([0, 0, 100]))
    nx, ny, nz, cell = 64, 4, 4, 32.0
    grid = np.zeros((nz, ny, nx), np.float32)
    grid[:] = (10000.0 - (np.arange(nx) + 0.5) * cell)[None, None, :]
    np.savez(tmp_path / "field.npz", grid=grid, quant=np.float64(1.0), mins=np.zeros(3),
             cell=np.float64(cell), reach_max=np.float64(20000.0))
    rc = energy_pick.main(["--archive", str(tmp_path / "arch.npz"), "--line", str(tmp_path / "line.npy"),
                           "--field", str(tmp_path / "field.npz"), "--lo", "2.5", "--hi", "3.5",
                           "--d0", "10000", "--out", str(tmp_path / "pick.json")])
    assert rc == 0
    assert json.loads((tmp_path / "pick.json").read_text())["cell"] == 1
