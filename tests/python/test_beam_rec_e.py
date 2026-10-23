"""tools/beam_tas.py --score rec_e: elites by the lineage's record (min field d so far) in bins,
then by energy height; --dump-elite-states writes STATE_DTYPE rows for --spawn-states."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import beam_tas as B  # noqa: E402


def _states(z, v):
    st = np.zeros(len(z), STATE_DTYPE)
    st["origin"][:, 2] = z
    st["velocity"][:, 0] = v
    st["tick"] = 77
    return st


def test_energy_height():
    st = _states([100.0, 0.0], [0.0, 400.0])
    np.testing.assert_allclose(B.energy_height(st, 800.0), [100.0, 100.0])


def test_rank_record_then_energy():
    # records 1000 / 1010 / 990 / 1200 share bins of 256 except the last; inside the bin the
    # higher energy wins whatever its instantaneous distance; the invalid row is last
    rec = np.array([1000.0, 1010.0, 990.0, 1200.0, 500.0])
    h = np.array([10.0, 50.0, 30.0, 999.0, 1e9])
    valid = np.array([True, True, True, True, False])
    order = B.rank_rec_e(rec, h, valid, 256.0)
    assert order.tolist()[:4] == [1, 2, 0, 3]
    assert order[-1] == 4
    # a better record bin beats any energy
    rec2 = np.array([700.0, 1000.0])
    assert B.rank_rec_e(rec2, np.array([0.0, 1e6]), np.array([True, True]), 256.0)[0] == 0
    # quant 0: the raw record
    assert B.rank_rec_e(np.array([2.0, 1.0]), np.zeros(2), np.ones(2, bool), 0.0)[0] == 1


def test_elite_archive_roundtrip(tmp_path):
    arch = B.EliteArchive(per_gen=2)
    st = _states([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    arch.add(st, np.array([2, 0, 1]), np.array([9.0, 8.0, 7.0]), np.array([1.0, 2.0, 3.0]),
             gen=1, tick=100)
    arch.add(st, np.array([1]), np.array([9.0, 8.0, 7.0]), np.array([1.0, 2.0, 3.0]),
             gen=2, tick=200)
    out = tmp_path / "el.npy"
    assert arch.save(out, {"provenance": "test"}) == 3
    rows = np.load(out)
    assert rows.dtype == STATE_DTYPE and len(rows) == 3
    np.testing.assert_allclose(rows["origin"][:, 2], [3.0, 1.0, 2.0])
    assert (rows["tick"] == 0).all()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["rows"] == 3 and meta["gens"][0]["rec_d_best"] == 7.0
    assert meta["provenance"] == "test"
