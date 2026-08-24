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
