"""tools/splice_line.py (pbr2b1, 2026-10-24): head[:k] + tail where tail[0] sits on head[k]."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
from surfgym.core import STATE_DTYPE  # noqa: E402
import splice_line  # noqa: E402


def test_splice_at_the_root():
    head = np.zeros(10, STATE_DTYPE)
    head["origin"][:, 0] = np.arange(10) * 10.0
    head["tick"] = 7
    tail = np.zeros(4, STATE_DTYPE)
    tail["origin"][:, 0] = [40.2, 100.0, 200.0, 300.0]
    out, k = splice_line.splice(head, tail)
    assert k == 4 and len(out) == 8
    assert np.allclose(out["origin"][:, 0], [0, 10, 20, 30, 40.2, 100, 200, 300])
    assert (out["tick"] == 0).all()
    with pytest.raises(ValueError):
        splice_line.splice(head, tail[1:])
