"""The extra scalar blocks of a BC file (surfgym.bc XBLOCKS: the compass, the ratchet gap, the
surf flag, the pot rate) - written by tools/plan_to_bc.py, spliced by BCDataset into the row
exactly where the policy wrapper's _obs puts them: [15 core | latch | aux | T | keys | ratchet |
surf | prate]. A file without them, read by a trainer that needs none, is the row it always was."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import bc  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402

N = 4
LAYOUT = {"aux": 5, "ratchet": 1, "surf": 1, "prate": 1}


def _write(path, xcols=None, keys=True):
    meta = {"n_latch": 0, "obs_reward": False}
    if keys:
        meta["keys_hold"] = 1
    if xcols:
        meta["xcols"] = [[k, int(np.asarray(v).reshape(N, -1).shape[1])]
                         for k, v in xcols.items()]
    bc.save_bc_dataset(path, np.zeros(N, STATE_DTYPE), np.full((N, 15), 1.0, np.float32),
                       np.zeros(N, np.float32), np.zeros((N, 6), np.int64),
                       np.ones(N, np.float32), np.zeros(N, np.int32), meta,
                       keys=(np.full((N, 7), 7.0, np.float32) if keys else None),
                       xcols=xcols)


def _x():
    return {"aux": np.full((N, 5), 2.0), "ratchet": np.full((N, 1), 3.0),
            "surf": np.full((N, 1), 4.0), "prate": np.full((N, 1), 5.0)}


def test_row_order_matches_the_wrapper(tmp_path):
    pytest.importorskip("torch")
    p = tmp_path / "bc.npz"
    _write(p, _x())
    ds = bc.BCDataset(p, "cpu", n_latch=0, obs_reward=False, keys_hold=True, xlayout=LAYOUT)
    want = [1.0] * 15 + [2.0] * 5 + [7.0] * 7 + [3.0, 4.0, 5.0]
    assert ds.scal.shape == (N, len(want))
    assert ds.scal[0].tolist() == want
    assert ds.xlayout == [["aux", 5], ["ratchet", 1], ["surf", 1], ["prate", 1]]


def test_the_file_must_carry_exactly_what_the_policy_reads(tmp_path):
    pytest.importorskip("torch")
    p = tmp_path / "bc.npz"
    _write(p, _x())
    with pytest.raises(SystemExit):        # the trainer reads no ratchet gap
        bc.BCDataset(p, "cpu", n_latch=0, obs_reward=False, keys_hold=True,
                     xlayout={"aux": 5, "surf": 1, "prate": 1})
    q = tmp_path / "plain.npz"
    _write(q, None)
    with pytest.raises(SystemExit):        # the policy reads blocks the file does not have
        bc.BCDataset(q, "cpu", n_latch=0, obs_reward=False, keys_hold=True, xlayout=LAYOUT)


def test_a_file_without_blocks_is_the_old_row(tmp_path):
    pytest.importorskip("torch")
    q = tmp_path / "plain.npz"
    _write(q, None)
    ds = bc.BCDataset(q, "cpu", n_latch=0, obs_reward=False, keys_hold=True,
                      xlayout={"aux": 0, "ratchet": 0, "surf": 0, "prate": 0})
    assert ds.scal[0].tolist() == [1.0] * 15 + [7.0] * 7
    assert "x_aux" not in np.load(q).files


def test_save_validates_names_and_meta(tmp_path):
    with pytest.raises(ValueError):
        _write(tmp_path / "a.npz", {"compass": np.zeros((N, 5))})
    with pytest.raises(ValueError):        # meta says 5 wide, the column is 4
        meta = {"n_latch": 0, "obs_reward": False, "xcols": [["aux", 5]]}
        bc.save_bc_dataset(tmp_path / "b.npz", np.zeros(N, STATE_DTYPE),
                           np.zeros((N, 15), np.float32), np.zeros(N, np.float32),
                           np.zeros((N, 6), np.int64), np.ones(N, np.float32),
                           np.zeros(N, np.int32), meta, xcols={"aux": np.zeros((N, 4))})


def test_merge_carries_the_blocks(tmp_path):
    a, b, m = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "m.npz"
    _write(a, _x())
    _write(b, _x())
    bc.merge_bc_files([a, b], m)
    arr = bc.load_bc_arrays(m)
    assert arr["xcols"]["aux"].shape == (2 * N, 5)
    assert arr["meta"]["xcols"] == [["aux", 5], ["ratchet", 1], ["surf", 1], ["prate", 1]]
    c = tmp_path / "c.npz"
    _write(c, None)
    with pytest.raises(ValueError):
        bc.merge_bc_files([a, c], tmp_path / "bad.npz")
