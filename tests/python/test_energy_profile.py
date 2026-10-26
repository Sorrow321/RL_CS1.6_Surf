"""tools/energy_profile.py: energy height at first arrival per progress level, evals vs a spine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import energy_profile  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402
from test_auto_spine_energy import _field  # noqa: E402


def _eval(path, eps):
    """record.py's format: a header, rows [t, x, y, z, vx, vy, vz, ...], a trailer."""
    with open(path, "w", encoding="utf-8") as fh:
        for k, (xs, vx) in enumerate(eps):
            fh.write(json.dumps({"map": "m", "tick_ms": 10, "episode": k}) + "\n")
            for t, (x, v) in enumerate(zip(xs, vx)):
                fh.write(json.dumps([t, float(x), 50.0, 50.0, float(v), 0.0, 0.0, 0, 0]) + "\n")
            fh.write(json.dumps({"end": "fail"}) + "\n")


def test_first_arrival_profile(tmp_path, capsys):
    fp = _field(tmp_path)                     # d = 10000 - x; start at x 0 -> d0 = 10000 - 16
    xs = np.linspace(0.0, 1000.0, 101)
    _eval(tmp_path / "traj_1.jsonl", [(xs, np.full(101, 400.0)), (xs[:51], np.full(51, 800.0)),
                                      (xs, np.where(xs < 600, 800.0, 400.0))])
    s = np.zeros(101, STATE_DTYPE)
    s["origin"][:, 0] = xs
    s["origin"][:, 1:] = 50.0
    s["velocity"][:, 0] = 600.0
    np.save(tmp_path / "ref.npy", s)
    rc = energy_profile.main([str(tmp_path / "traj_1.jsonl"), "--field", str(fp), "--levels",
                              "3,8", "--ref", str(tmp_path / "ref.npy")])
    out = capsys.readouterr().out
    assert rc == 0
    l3 = [ln for ln in out.splitlines() if ln.startswith("  3.0")][0]
    l8 = [ln for ln in out.splitlines() if ln.startswith("  8.0")][0]
    # h = 50 + v^2/1600: 400 u/s -> 150, 800 -> 450, the reference's 600 -> 275
    assert " 275 " in l3 and "3/3 h  150- 450" in l3, l3
    # 8% of d0 ~ x 800: the 51-row episode never got there; the third slowed to 400 at x 600
    assert "2/3 h  150- 150" in l8, l8
    h = energy_profile.energy([[0, 0, 10.0]], [[40.0, 0, 0]])
    assert abs(float(h[0]) - 11.0) < 1e-9
