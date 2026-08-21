"""Tier A property tests for the DDP path (docs/ddp-plan.md §5).

CPU-only; the gloo pair-harness tests spawn two processes and need no GPU
box. Every test pins an EXACT equivalence — "the curves look similar" is
not a gate in this repo.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import distributed                       # noqa: E402
from surfgym.core import STATE_DTYPE                  # noqa: E402
from surfgym.respawn import RespawnBuffer             # noqa: E402
from surfgym.rewards import RaceReward                # noqa: E402


# ---------------------------------------------------------------------------
# 1. advantage moments: pooled f64 sum/sumsq with ddof=1 == torch std/mean
# ---------------------------------------------------------------------------
def test_adv_moments_ddof():
    torch.manual_seed(7)
    shards = [torch.randn(4096) * (1.0 + r) + r for r in range(4)]
    full = torch.cat(shards)
    n_g = float(full.numel())
    s1 = sum(s.double().sum() for s in shards)
    s2 = sum(s.double().pow(2).sum() for s in shards)
    m64 = s1 / n_g
    v64 = (s2 - s1 * m64) / (n_g - 1.0)               # ddof=1 == torch.std
    assert abs(float(m64) - float(full.mean())) <= 1e-6
    assert abs(float(v64.clamp_min(0).sqrt()) - float(full.std())) <= 1e-5
    # the ddof slip MUST be caught: /n_g differs by a detectable margin
    v_wrong = (s2 - s1 * m64) / n_g
    assert abs(float(v_wrong.sqrt()) - float(full.std())) > 1e-5


# ---------------------------------------------------------------------------
# 2. push_many == a loop of _push (store/head/size/harvested byte-identical)
# ---------------------------------------------------------------------------
def _rand_states(n, seed):
    rng = np.random.default_rng(seed)
    rows = np.zeros(n, dtype=STATE_DTYPE)
    raw = rng.integers(0, 255, size=(n, STATE_DTYPE.itemsize),
                       dtype=np.uint8).astype(np.uint8)
    rows.view(np.uint8).reshape(n, -1)[:] = raw
    return rows


@pytest.mark.parametrize("cap,total,chunk", [(50, 120, 17), (64, 64, 64),
                                             (100, 37, 5)])
def test_push_many_equals_push_loop(cap, total, chunk):
    rows = _rand_states(total, seed=cap)
    a = RespawnBuffer(4, reservoir=cap)
    b = RespawnBuffer(4, reservoir=cap)
    for r in rows:
        a._push(r)
    for s0 in range(0, total, chunk):
        b.push_many(rows[s0:s0 + chunk])
    assert a._head == b._head and a._size == b._size
    assert a.harvested == b.harvested
    assert a._store.tobytes() == b._store.tobytes()


def test_push_many_with_distance_column():
    rows = _rand_states(40, seed=3)
    kw = dict(dist_fn=lambda o: np.linalg.norm(o, axis=1), dist_max=1e4)
    a = RespawnBuffer(4, reservoir=32, **kw)
    b = RespawnBuffer(4, reservoir=32, **kw)
    ds = a._dists(list(rows))
    for r, d in zip(rows, ds):
        a._push(r, float(d))
    b.push_many(rows)
    assert a._store.tobytes() == b._store.tobytes()
    assert np.allclose(a._d, b._d)


# ---------------------------------------------------------------------------
# 3. counts delta round-trip: merged table == single-process table; a
#    resume neither divides history by R nor multiplies it by R
# ---------------------------------------------------------------------------
def _mk_race(ncells, restored=None):
    r = RaceReward(field=None, scale=1.0, int_coef=0.25)
    base = (np.zeros(ncells, np.int64) if restored is None
            else np.asarray(restored, np.int64).copy())
    r._counts = base.copy()
    r._counts_base = base.copy()                      # what on_reset does
    return r


def test_counts_delta_roundtrip():
    ncells = 1000
    rng = np.random.default_rng(0)
    visits = [rng.integers(0, ncells, 500), rng.integers(0, ncells, 700)]
    ranks = [_mk_race(ncells) for _ in range(2)]
    single = np.zeros(ncells, np.int64)
    for r, v in zip(ranks, visits):
        np.add.at(r._counts, v, 1)
        np.add.at(single, v, 1)
    fleet = sum(r.counts_delta().astype(np.int64) for r in ranks)
    for r in ranks:
        r.apply_counts_delta(fleet)
    for r in ranks:
        assert np.array_equal(r._counts, single)
    # second round: deltas start from the synced base, not from zero
    v2 = rng.integers(0, ncells, 300)
    np.add.at(ranks[0]._counts, v2, 1)
    np.add.at(single, v2, 1)
    fleet2 = sum(r.counts_delta().astype(np.int64) for r in ranks)
    for r in ranks:
        r.apply_counts_delta(fleet2)
        assert np.array_equal(r._counts, single)


def test_counts_resume_neither_divides_nor_multiplies():
    ncells = 64
    restored = np.arange(ncells, dtype=np.int64) * 3
    ranks = [_mk_race(ncells, restored) for _ in range(4)]
    # no new visits: the first sync must be a no-op, NOT +3 x history
    fleet = sum(r.counts_delta().astype(np.int64) for r in ranks)
    assert fleet.sum() == 0
    for r in ranks:
        r.apply_counts_delta(fleet)
        assert np.array_equal(r._counts, restored)


# ---------------------------------------------------------------------------
# 4. gather ordering: stable sort by (tick, global env) reproduces the
#    single-process append order exactly, snapshot order included
# ---------------------------------------------------------------------------
def test_gather_ordering():
    rng = np.random.default_rng(5)
    n_global, ws = 8, 2
    n_local = n_global // ws
    # single-process event stream: per tick, ended envs ascending, and each
    # env contributes its pending rows in snapshot order
    events = []                                       # (tick, genv, payload)
    payload = 0
    for tick in range(1, 20):
        for genv in sorted(rng.choice(n_global, rng.integers(0, 4),
                                      replace=False)):
            for _snap in range(rng.integers(1, 3)):
                events.append((tick, int(genv), payload))
                payload += 1
    single_order = [p for _, _, p in events]
    # shard by env ownership, preserving per-rank arrival order
    shards = [[(t, g, p) for t, g, p in events if g // n_local == r]
              for r in range(ws)]
    merged = np.array([e for s in shards for e in s],
                      dtype=[("tick", np.int32), ("env", np.int32),
                             ("p", np.int64)])
    key = merged["tick"].astype(np.int64) * (n_global + 1) + merged["env"]
    got = merged["p"][np.argsort(key, kind="stable")].tolist()
    assert got == single_order


# ---------------------------------------------------------------------------
# 5/6. gloo pair harness: all_gather_var_bytes round-trip; the flat grad
#      all-reduce equals the single-process concat-batch gradient; params
#      stay EXACTLY equal across ranks through 20 Adam steps
# ---------------------------------------------------------------------------
def _mk_facade(rank, ws):
    d = distributed.Dist()
    d.rank, d.world_size, d.local_rank = rank, ws, rank
    d.is_main, d.enabled = rank == 0, True
    d.device = torch.device("cpu")
    return d


def _tiny_net(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(),
                               torch.nn.Linear(16, 1))


def _pair_worker(rank, ws, tmpfile):
    import torch.distributed as dist
    dist.init_process_group(
        "gloo", init_method=f"file:///{Path(tmpfile).as_posix()}",
        rank=rank, world_size=ws)
    try:
        D = _mk_facade(rank, ws)

        # -- all_gather_var_bytes: rank-ordered, empties survive ------------
        payload = b"rank%d" % rank * (rank + 1) if rank else b""
        parts = D.all_gather_var_bytes(payload)
        assert parts[0] == b"" and parts[1] == b"rank1rank1"

        # -- gradient equivalence -------------------------------------------
        torch.manual_seed(99)
        X = torch.randn(64, 6)
        Y = torch.randn(64, 1)
        net = _tiny_net(seed=1)                       # identical init
        ref = _tiny_net(seed=1)                       # single-process twin
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        ropt = torch.optim.Adam(ref.parameters(), lr=1e-2)

        n_par = sum(p.numel() for p in net.parameters())
        flat = torch.zeros(n_par)
        views, off = [], 0
        for p in net.parameters():
            views.append(torch.as_strided(flat, p.shape, p.stride(), off))
            off += p.numel()

        sh = 64 // ws
        for step in range(20):
            xb, yb = X[rank * sh:(rank + 1) * sh], Y[rank * sh:(rank + 1) * sh]
            opt.zero_grad(set_to_none=True)
            ((net(xb) - yb) ** 2).mean().backward()
            grads = [p.grad for p in net.parameters()]   # RE-READ each step
            torch._foreach_copy_(views, grads)
            D.all_reduce_sum_(flat)
            flat.mul_(1.0 / ws)
            torch._foreach_copy_(grads, views)
            ropt.zero_grad(set_to_none=True)
            ((ref(X) - Y) ** 2).mean().backward()
            if step == 0:
                for p, rp in zip(net.parameters(), ref.parameters()):
                    assert torch.allclose(p.grad, rp.grad, atol=1e-6), \
                        "averaged shard grad != concat-batch grad"
            opt.step()
            ropt.step()

        c = torch.stack([sum(p.double().sum() for p in net.parameters())])
        D.assert_equal("pair_params", c)              # EXACT across ranks
        for p, rp in zip(net.parameters(), ref.parameters()):
            assert torch.allclose(p, rp, atol=1e-4), \
                "20-step DDP trajectory left the single-process one"

        # -- assert_distinct fires on collapsed streams ---------------------
        D.assert_distinct("streams_ok", 1000 + rank)
        try:
            D.assert_distinct("streams_bad", 42)
            raise AssertionError("assert_distinct failed to fire")
        except RuntimeError:
            pass
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available()
                    or not torch.distributed.is_gloo_available(),
                    reason="gloo backend unavailable")
def test_gloo_pair_harness():
    with tempfile.TemporaryDirectory() as td:
        tmpfile = os.path.join(td, "pg_init")
        torch.multiprocessing.spawn(_pair_worker, args=(2, tmpfile),
                                    nprocs=2, join=True)


# ---------------------------------------------------------------------------
# 7. checkpoint interchange: no wrapper means no 'module.' prefix — pin it
#    so nobody wraps the policy later
# ---------------------------------------------------------------------------
def test_state_dict_has_no_module_prefix():
    from train_fast import N_SCALAR, Policy
    p = Policy(N_SCALAR + 64 * 32, 64, 32)
    assert all(not k.startswith("module.") for k in p.state_dict())
