"""tools/archive_chain.py: the dense chain keeps every state once, at its first occurrence, so a
burst prefix shared by two chain cells is not repeated (DemoCurriculum.match would attribute the
first copy's episodes to the last copy)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import archive_chain as C  # noqa: E402


def _rows(xs):
    r = np.zeros(len(xs), STATE_DTYPE)
    r["origin"][:, 0] = xs
    return r


def test_parent_chain():
    assert C.parent_chain(np.array([-1, 0, 1, 1]), 3) == [0, 1, 3]


def test_shared_burst_prefix_is_kept_once_in_order():
    # one burst entered cell 1 after rows 0..2 and cell 2 after rows 0..4: the raw concatenation
    # repeats 0..2; the dedup gives 0..4 in order
    paths = _rows([0, 1, 2, 0, 1, 2, 3, 4])
    cells = np.array([1, 1, 1, 2, 2, 2, 2, 2])
    raw = C.dense_rows(paths, cells, [1, 2])
    assert raw["origin"][:, 0].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    out = C.dedup_first(raw)
    assert out["origin"][:, 0].tolist() == [0, 1, 2, 3, 4]
