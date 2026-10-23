"""tools/archive_to_spawn.py: keep the archive rows ahead of the start on the field, zero clocks."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import archive_to_spawn as A  # noqa: E402


def test_select_ahead_of_start():
    st = np.zeros(5, STATE_DTYPE)
    st["origin"][:, 0] = np.arange(5)
    st["tick"] = 99
    st["stuck_ticks"] = 3
    d = np.array([1000.0, 999.0, 900.0, 1100.0, 5e9])
    out, dk = A.select(st, d, 1000.0, 0.0, reach_max=1e9)
    assert out["origin"][:, 0].tolist() == [1.0, 2.0]
    assert (out["tick"] == 0).all() and (out["stuck_ticks"] == 0).all()
    assert st["tick"][1] == 99                       # the input is not modified
    out2, _ = A.select(st, d, 1000.0, 0.05, reach_max=1e9)
    assert out2["origin"][:, 0].tolist() == [2.0]
