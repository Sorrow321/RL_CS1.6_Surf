"""Expert iteration with a --keys-hold policy (skate_laby, 2026-10-04): a planner line's ENGINE
rows become POLICY-space targets plus the held-key observation columns (surfgym.bc
keys_policy_rows / keys_policy_probs), and the BC file round-trips through BCDataset only in the
mode it was built for.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.bc import keys_policy_probs, keys_policy_rows      # noqa: E402
from surfgym.keyshold import HELD_HEADS, KEEP, NEUTRAL, KeysHold  # noqa: E402


def _engine_line(n=60, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, 6), np.int64)
    a[:, 0] = rng.integers(0, 15, n)          # yaw
    a[:, 1] = rng.integers(0, 7, n)           # pitch
    # held heads change rarely, like a real line
    for h, hi in ((2, 3), (3, 3), (5, 2)):
        v = int(NEUTRAL[HELD_HEADS.index(h)])
        for t in range(n):
            if rng.random() < 0.15:
                v = int(rng.integers(0, hi))
            a[t, h] = v
    a[:, 4] = rng.integers(0, 2, n)           # jump
    return a


def test_policy_rows_resolve_back_to_the_engine_rows():
    eng = _engine_line()
    pol, held, feat = keys_policy_rows(eng)
    k = KeysHold(1)
    for t in range(len(eng)):
        np.testing.assert_array_equal(k.features()[0], feat[t])     # what the policy SAW
        row = pol[t:t + 1].astype(np.int32).copy()
        k.resolve(row)
        np.testing.assert_array_equal(row[0], eng[t])                # what the engine gets


def test_unchanged_keys_are_keep_and_changes_are_shifted():
    eng = _engine_line(seed=3)
    pol, held, _ = keys_policy_rows(eng)
    for c, h in enumerate(HELD_HEADS):
        same = eng[:, h] == held[:, c]
        assert (pol[same, h] == KEEP).all()
        assert (pol[~same, h] == eng[~same, h] + 1).all()
    for h in (0, 1, 4):                                              # yaw, pitch, jump
        np.testing.assert_array_equal(pol[:, h], eng[:, h])
    np.testing.assert_array_equal(held[0], NEUTRAL)                  # a line starts NEUTRAL


def test_distribution_target_moves_with_the_keep_rule():
    rng = np.random.default_rng(1)
    p = rng.random((5, 6, 15)).astype(np.float32)
    for h, nb in enumerate((15, 7, 3, 3, 2, 2)):      # mass only on each head's ENGINE bins
        p[:, h, nb:] = 0.0
    p /= p.sum(-1, keepdims=True)
    held = np.array([[0, 1, 0], [1, 2, 1], [2, 0, 0], [1, 1, 1], [0, 0, 1]])
    q = keys_policy_probs(p, held)
    np.testing.assert_allclose(q.sum(-1), 1.0, atol=1e-5)             # mass conserved
    r = np.arange(5)
    for c, h in enumerate(HELD_HEADS):
        np.testing.assert_allclose(q[r, h, 0], p[r, h, held[:, c]])  # held value -> KEEP
        np.testing.assert_allclose(q[r, h, held[:, c] + 1], 0.0)     # its old slot emptied
    for h in (0, 1, 4):
        np.testing.assert_array_equal(q[:, h], p[:, h])


def test_the_file_round_trips_only_in_its_own_mode(tmp_path):
    import torch
    from surfgym.bc import BCDataset, save_bc_dataset
    from surfgym.core import STATE_DTYPE
    eng = _engine_line(n=12)
    pol, held, feat = keys_policy_rows(eng)
    n = len(eng)
    states = np.zeros(n, STATE_DTYPE)
    f = tmp_path / "bc.npz"
    meta = {"obs_reward": False, "n_latch": 0, "keys_hold": 1}
    save_bc_dataset(f, states, np.zeros((n, 15), np.float32), np.zeros(n, np.float32),
                    pol, np.ones(n, np.float32), np.zeros(n, np.int32), meta, keys=feat)
    ds = BCDataset(f, torch.device("cpu"), n_latch=0, obs_reward=False, keys_hold=True)
    assert tuple(ds.scal.shape) == (n, 15 + 7)
    np.testing.assert_array_equal(ds.scal[:, 15:].numpy(), feat)
    with pytest.raises(SystemExit):
        BCDataset(f, torch.device("cpu"), n_latch=0, obs_reward=False, keys_hold=False)
    with pytest.raises(ValueError):                    # keys meta without the columns
        save_bc_dataset(tmp_path / "x.npz", states, np.zeros((n, 15), np.float32),
                        np.zeros(n, np.float32), eng, np.ones(n, np.float32),
                        np.zeros(n, np.int32), meta)
