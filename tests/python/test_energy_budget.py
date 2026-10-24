"""tools/energy_budget.py (pbr2b1, 2026-10-24): h = z + |v|^2/2g and its loss stretches."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import energy_budget  # noqa: E402


def test_free_fall_keeps_h_and_a_crash_is_a_loss():
    g, dt = 800.0, 0.01
    rows, z, vz = [], 1000.0, 0.0
    for t in range(100):                       # free fall: h constant
        rows.append([t, 0, 0, z, 300.0, 0, vz])
        z += vz * dt - 0.5 * g * dt * dt
        vz -= g * dt
    for t in range(100, 110):                  # a crash: speed gone at once
        rows.append([t, 0, 0, z, 0.0, 0, 0.0])
    r = np.asarray(rows, np.float64)
    h = energy_budget.energy_height(r, g)
    assert np.allclose(h[:100], h[0], atol=1.0)
    ls = energy_budget.losses(h, window=20, loss=60.0)
    assert len(ls) == 1 and ls[0][1] == 100 and ls[0][2] < -60.0
