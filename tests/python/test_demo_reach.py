"""--demo-reach: a spine episode wins on a finish OR on coming within the win line of the spine's
last row; off (thr None) it is the paper's finish-only outcome, unchanged."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.respawn import demo_reach_wins  # noqa: E402


def test_off_is_finish_only():
    g = np.array([True, False, False])
    out = demo_reach_wins(g, np.array([0.0, 0.0, 0.0]), None)
    assert out.tolist() == [True, False, False]


def test_reach_counts_as_win():
    g = np.array([False, False, True])
    out = demo_reach_wins(g, np.array([100.0, 99.0, 5000.0]), 99.5)
    assert out.tolist() == [False, True, True]


def test_flag_parses_default_off():
    sys.path.insert(0, str(ROOT / "python"))
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--demo-reach", type=float, default=None' in src
    assert "if demo is not None and float(args.demo_reach or 0.0) > 0.0:" in src
