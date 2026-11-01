"""tools/pool_add_lines.py: lines replicated to the target share, base rows untouched."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import pool_add_lines  # noqa: E402


def test_share_and_tiling():
    base = np.zeros(700, STATE_DTYPE)
    base["origin"][:, 0] = np.arange(700)
    l1 = np.zeros(10, STATE_DTYPE)
    l1["origin"][:, 0] = 1000 + np.arange(10)
    l2 = np.zeros(40, STATE_DTYPE)
    l2["origin"][:, 0] = 2000 + np.arange(40)
    l2["tick"] = 7
    out = pool_add_lines.build(base, [l1, l2], 0.3)
    assert np.array_equal(out[:700]["origin"], base["origin"])
    lines = out[700:]
    assert abs(len(lines) / len(out) - 0.3) < 0.01
    x = lines["origin"][:, 0]
    assert ((x >= 1000) & (x < 1010)).sum() == ((x >= 2000) & (x < 2040)).sum()   # equal weight
    assert set(np.unique(x[(x >= 1000) & (x < 1010)])) == set(1000 + np.arange(10))
    assert (out["tick"] == 0).all()


def test_no_lines_is_the_base():
    base = np.zeros(5, STATE_DTYPE)
    assert np.array_equal(pool_add_lines.build(base, [], 0.3), base)
