"""Gate F (docs/multimap-ddp-plan.md): the aggregated multi-map metrics must
read correctly FROM EVERY RANK, not from rank 0's slice.

The rig the plan asks for: two maps, one solved and one at zero, evaluated on
two different ranks. Every rank must come out of the eval holding the whole
table and the same two headline numbers.

Two levels:

* :func:`test_aggregate_*` exercise the shipped :func:`train_fast.eval_aggregate`
  directly - the same reason ``adv_moments64`` is module-level.
* :func:`test_two_rank_allreduce_agrees` spawns a REAL two-process gloo group,
  has each process fill only the row it owns, all-reduces, and asserts both
  processes report identical aggregates. NCCL is Linux-only, but the reduction
  under test is an integer/float SUM over a fixed-shape tensor, which gloo
  performs identically - what is being gated is the ownership/partition logic,
  not the transport.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import EVAL_K, eval_aggregate            # noqa: E402


# ---------------------------------------------------------------- helpers
def row(*, pct_sum, n_eps, n_box, prog_u_sum=0.0, finish_s_sum=0.0,
        n_finish_geo=0.0, fwd=0.0, path=0.0, speed=0.0, evaluated=1.0):
    r = np.zeros(EVAL_K, np.float64)
    r[0] = prog_u_sum
    r[1] = finish_s_sum
    r[2] = n_finish_geo
    r[3] = pct_sum
    r[4] = n_eps
    r[5] = n_box
    r[6] = fwd
    r[7] = path
    r[8] = speed
    r[9] = evaluated
    return r


# ------------------------------------------------------------ the rig
# map 0: SOLVED - 3 of 3 greedy episodes cross the finish box, 100% covered.
# map 1: ZERO   - 3 episodes, 0% covered, no finish.
SOLVED = row(pct_sum=300.0, n_eps=3, n_box=3, prog_u_sum=3 * 35_637.0,
             finish_s_sum=3 * 21.5, n_finish_geo=3, fwd=9_000.0,
             path=40_000.0, speed=2_800.0)
ZERO = row(pct_sum=0.0, n_eps=3, n_box=0, prog_u_sum=0.0, fwd=10.0,
           path=120.0, speed=280.0)


def test_aggregate_one_solved_one_zero():
    agg = eval_aggregate(np.stack([SOLVED, ZERO]), ["trigger", "trigger"], 3)
    assert agg["map_pct"] == pytest.approx(50.0)
    assert agg["maps_finished"] == pytest.approx(0.5)
    assert agg["n_maps_scored"] == 2
    assert agg["n_maps_finished"] == 1
    assert agg["pct"].tolist() == pytest.approx([100.0, 0.0])
    # the mean is over MAPS, not over episodes: a map with 30 eval episodes
    # must not out-vote one with 3
    many = row(pct_sum=0.0, n_eps=30, n_box=0)
    agg2 = eval_aggregate(np.stack([SOLVED, many]), ["trigger"] * 2, 3)
    assert agg2["map_pct"] == pytest.approx(50.0)


def test_aggregate_is_a_percentage_not_map_units():
    """A long map covered 10% and a short map covered 90% must average 50%.

    ``race/eval_progress`` would say (0.10*198,380 + 0.90*35,637)/2 = 25,957
    units, which is 84% the long map's opinion. The pool spans a 5x range of
    route length; the aggregate has to be scale-free or it is a weighted vote.
    """
    long_map = row(pct_sum=10.0, n_eps=1, n_box=0,
                   prog_u_sum=0.10 * 198_380)
    short_map = row(pct_sum=90.0, n_eps=1, n_box=0,
                    prog_u_sum=0.90 * 35_637)
    agg = eval_aggregate(np.stack([long_map, short_map]), ["trigger"] * 2, 1)
    assert agg["map_pct"] == pytest.approx(50.0)
    # and the units number really is dominated by the long map
    assert agg["eval_prog"] == pytest.approx((0.10 * 198_380
                                              + 0.90 * 35_637) / 2)


def test_trigger_and_button_are_reported_split():
    """A null on a button map is weaker evidence: the two must not be pooled
    into one headline without the split beside it (CLAUDE.md 4b)."""
    agg = eval_aggregate(np.stack([SOLVED, ZERO]), ["button", "trigger"], 3)
    assert agg["map_pct"] == pytest.approx(50.0)
    assert agg["maps_finished"] == pytest.approx(0.5)
    # ... but the trigger-only view sees only the map that is at zero
    assert agg["map_pct_trigger"] == pytest.approx(0.0)
    assert agg["maps_finished_trigger"] == pytest.approx(0.0)
    assert agg["n_trigger"] == 1


def test_unevaluated_map_is_detectable():
    """A map no rank evaluated must be visible, not silently averaged out."""
    missing = row(pct_sum=0.0, n_eps=0, n_box=0, evaluated=0.0)
    agg = eval_aggregate(np.stack([SOLVED, missing]), ["trigger"] * 2, 3)
    assert not agg["evaluated"].all()
    # the surviving map is scored on its own, NOT counted as a 0% map
    assert agg["n_maps_scored"] == 1
    assert agg["map_pct"] == pytest.approx(100.0)


def test_row_width_is_pinned():
    with pytest.raises(ValueError):
        eval_aggregate(np.zeros((2, EVAL_K - 1)), ["trigger"] * 2, 3)
    with pytest.raises(ValueError):
        eval_aggregate(np.zeros((2, EVAL_K)), ["trigger"], 3)


# ------------------------------------------------- real two-process gather
def _worker(rank, world, port, out_q):
    import torch
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        # exactly what the trainer does: a zero table, and this rank fills
        # ONLY the maps it owns (round-robin i % world == rank)
        tab = torch.zeros((2, EVAL_K), dtype=torch.float64)
        rows = [SOLVED, ZERO]
        for i in range(2):
            if i % world == rank:
                tab[i] = torch.from_numpy(rows[i])
        dist.all_reduce(tab)
        agg = eval_aggregate(tab.numpy(), ["trigger", "trigger"], 3)
        out_q.put((rank, agg["map_pct"], agg["maps_finished"],
                   agg["pct"].tolist(), agg["n_maps_finished"],
                   bool(agg["evaluated"].all())))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not __import__("torch").distributed.is_gloo_available(),
                    reason="no gloo")
def test_two_rank_allreduce_agrees():
    """THE GATE. One map solved, one at zero, on two different ranks: both
    ranks must read map_pct 50% and maps_finished 0.5, and both must hold the
    complete per-map table. Before the all-reduce each rank holds a table
    half of which is zeros, so a rank reporting its own slice would read 50%
    / 0% on rank 0 and 0% / 50% on rank 1 - which is the bug this test
    exists to make impossible."""
    import multiprocessing as mp
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(r, 2, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    got = [q.get(timeout=180) for _ in range(2)]
    for p in ps:
        p.join(timeout=60)
        assert p.exitcode == 0, f"rank died: exitcode {p.exitcode}"
    assert len(got) == 2
    got.sort()
    for rank, pct, fin, per_map, n_fin, complete in got:
        assert pct == pytest.approx(50.0), f"rank {rank} read {pct}"
        assert fin == pytest.approx(0.5), f"rank {rank} read {fin}"
        assert per_map == pytest.approx([100.0, 0.0]), f"rank {rank}"
        assert n_fin == 1
        assert complete, f"rank {rank} is missing a map's row"
    # and the two ranks agree exactly, not just approximately
    assert got[0][1:] == got[1][1:]


# ------------------------------------------------- the novelty-count fan-out
# The novelty table is PER MAP (its keys are that map's cells), so the DDP
# sync has to be a FLEET operation. The merge's central hazard was that
# train_fast keeps `reward_fn` as an alias onto SLOT 0, so the auto-merged
# sync compiled, ran, and would have synced one map while the other four
# diverged - with no logged number changing.
from surfgym.mapfleet import MapFleet                      # noqa: E402
from surfgym.rewards import RaceReward                     # noqa: E402

CNT_DT = np.dtype([("slot", np.int32), ("cell", np.int64), ("inc", np.int32)])


class _FakeSlot:
    """Just enough of MapSlot for the counts helpers (MapFleet's __init__
    validates env ranges and obs_dim, which these tests do not exercise)."""

    def __init__(self, reward_fn):
        self.reward_fn = reward_fn


def _mk_race(ncells):
    r = RaceReward(field=None, scale=1.0, int_coef=0.25)
    r._counts = np.zeros(ncells, np.int64)
    r._counts_base = np.zeros(ncells, np.int64)     # what on_reset does
    r.track_touched = True
    return r


def _fleet(reward_fns):
    f = MapFleet.__new__(MapFleet)                  # bypass the env-range checks
    f.slots = [_FakeSlot(r) for r in reward_fns]
    f.single = len(f.slots) == 1
    f.n_maps = len(f.slots)
    return f


def test_counts_sync_is_per_map_and_batched():
    """Three maps, two ranks, one gather. Every rank must end with every
    map's table equal to what a single process would have counted, and no
    map may receive another map's cells."""
    NC, NMAPS, W = 500, 3, 2
    rng = np.random.default_rng(7)
    ranks = [[_mk_race(NC) for _ in range(NMAPS)] for _ in range(W)]
    fleets = [_fleet(rs) for rs in ranks]
    single = [np.zeros(NC, np.int64) for _ in range(NMAPS)]

    for _ in range(3):                              # multi-round: base advances
        for rs in ranks:
            for m, r in enumerate(rs):
                v = rng.integers(0, NC, 200)
                np.add.at(r._counts, v, 1)
                r._touched.append(v.copy())
                np.add.at(single[m], v, 1)
        # the collective: one gather of every rank's every slot
        wire = np.concatenate([f.counts_delta_sparse(CNT_DT) for f in fleets])
        assert set(np.unique(wire["slot"]).tolist()) <= set(range(NMAPS))
        for f in fleets:
            f.apply_counts_delta_sparse(wire)
        for rs in ranks:
            for m, r in enumerate(rs):
                assert np.array_equal(r._counts, single[m]), f"map {m}"
                assert np.array_equal(r._counts, r._counts_base)


def test_slot_0_only_sync_would_be_caught():
    """The negative control. If the sync ran on slot 0 alone - the shape the
    aliased merge would have produced - maps 1..n stay rank-divergent. This
    test asserts the bug is detectable, so the test above is not vacuous."""
    NC, W = 200, 2
    rng = np.random.default_rng(11)
    ranks = [[_mk_race(NC), _mk_race(NC)] for _ in range(W)]
    for rs in ranks:
        for r in rs:
            v = rng.integers(0, NC, 100)
            np.add.at(r._counts, v, 1)
            r._touched.append(v.copy())
    # the WRONG sync: slot 0 only
    deltas = [rs[0].counts_delta_sparse() for rs in ranks]
    cells = np.concatenate([d[0] for d in deltas])
    incs = np.concatenate([d[1] for d in deltas])
    for rs in ranks:
        rs[0].apply_counts_delta_sparse(cells, incs)
    assert np.array_equal(ranks[0][0]._counts, ranks[1][0]._counts)
    assert not np.array_equal(ranks[0][1]._counts, ranks[1][1]._counts), (
        "map 1 must still be rank-divergent, or this control proves nothing")


def test_empty_window_exchanges_nothing_and_changes_nothing():
    """A no-visit iteration still has to participate: the collective is
    unconditional on every rank, so an empty local batch must be a legal,
    zero-length contribution rather than a skipped call."""
    f = _fleet([_mk_race(64), _mk_race(64)])
    wire = f.counts_delta_sparse(CNT_DT)
    assert len(wire) == 0 and wire.dtype == CNT_DT
    before = [r.reward_fn._counts.copy() for r in f.slots]
    f.apply_counts_delta_sparse(wire)
    for r, b in zip(f.slots, before):
        assert np.array_equal(r.reward_fn._counts, b)
