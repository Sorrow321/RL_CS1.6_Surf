"""beam_tas --novelty (zlr2b2, 2026-11-01): the count-based novelty selection's own rules.

NoveltyCount keys a lineage's (x, y, z) // CELL x energy height // EBIN cell, counts every live
lineage at every resample boundary, and splits the elite into the --score leaders followed by the
rarest of the rest. Off (no --novelty) the resample block is the old one: `nov` is None and
`keep = elig[:n_elite]` is the only line that runs.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import beam_tas as B  # noqa: E402


def _states(o, v):
    return {"origin": np.asarray(o, np.float32), "velocity": np.asarray(v, np.float32)}


def test_keys_split_position_and_energy():
    nc = B.NoveltyCount(128.0, 256.0, 0.5)
    st = _states([[0, 0, 0], [10, 10, 10], [130, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0], [800, 0, 0]])
    k = nc.keys(st, np.arange(4))
    # same cell, same energy bin
    assert k[0] == k[1]
    # next cell along x
    assert k[2] != k[0]
    # same place, 800 u/s: h = 400 u -> energy bin 1 instead of 0
    assert k[3] != k[0]
    # position only: the energy no longer separates them
    nc0 = B.NoveltyCount(128.0, 0.0, 0.5)
    k0 = nc0.keys(st, np.arange(4))
    assert k0[3] == k0[0] and k0[2] != k0[0]


def test_negative_coordinates_do_not_collide():
    nc = B.NoveltyCount(128.0, 256.0, 0.5)
    st = _states([[-1, -1, -1], [1, 1, 1], [-129, 0, 0], [-3000, 2500, -4000]],
                 [[0, 0, 0]] * 4)
    k = nc.keys(st, np.arange(4))
    assert len(set(k.tolist())) == 4


def test_visit_counts_accumulate_over_boundaries():
    nc = B.NoveltyCount(128.0, 0.0, 0.5)
    st = _states([[0, 0, 0], [5, 5, 5], [500, 0, 0]], [[0, 0, 0]] * 3)
    c1 = nc.visit(st, np.arange(3))
    assert c1.tolist() == [2, 2, 1]
    c2 = nc.visit(st, np.array([2]))
    assert c2.tolist() == [2]
    assert nc.boundaries == 2
    assert sorted(nc.counts.values()) == [2, 2]


def test_split_keeps_score_leaders_then_rarest():
    nc = B.NoveltyCount(128.0, 0.0, 0.5)
    elig = np.array([7, 3, 5, 1, 0, 2])            # --score order, best first
    counts = np.array([9, 9, 4, 1, 1, 3])          # aligned with elig
    dgeo = np.zeros(8)
    dgeo[[1, 0]] = [50.0, 40.0]                    # tie on count 1: the smaller d first
    keep, n_nov = nc.split(elig, counts, dgeo, 4)
    # 2 by score (7, 3), then the 2 rarest of the rest: count 1 x2 (0 has the smaller d, then 1)
    assert keep.tolist() == [7, 3, 0, 1]
    assert n_nov == 2
    # keep[0] is the score leader whatever the counts say
    assert keep[0] == elig[0]


def test_split_frac_one_still_keeps_the_leader():
    nc = B.NoveltyCount(128.0, 0.0, 1.0)
    elig = np.array([4, 2, 0, 1, 3])
    counts = np.array([5, 1, 2, 3, 4])
    keep, n_nov = nc.split(elig, counts, np.zeros(5), 2)
    assert keep[0] == 4
    assert n_nov == 2
    assert keep.tolist() == [4, 2, 0]


def test_split_short_population():
    nc = B.NoveltyCount(128.0, 0.0, 0.5)
    keep, n_nov = nc.split(np.array([3]), np.array([1]), np.zeros(4), 4)
    assert keep.tolist() == [3] and n_nov == 0
    keep, n_nov = nc.split(np.array([], np.int64), np.array([], np.int64), np.zeros(4), 4)
    assert len(keep) == 0 and n_nov == 0


def test_bad_constants_refused():
    import pytest
    for args in [(0.0, 0.0, 0.5), (128.0, -1.0, 0.5), (128.0, 0.0, 0.0), (128.0, 0.0, 1.5)]:
        with pytest.raises(SystemExit):
            B.NoveltyCount(*args)
