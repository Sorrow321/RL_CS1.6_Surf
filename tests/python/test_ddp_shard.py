"""--ddp-maps shard (mm/ddp, 2026-11-17): map-sharded DDP pieces, on CPU with gloo.

Every gloo test spawns two real processes (torch.multiprocessing, a file:// rendezvous) and runs
the SHIPPED code - surfgym.distributed.FlatGradSync (the trainer's gradient all-reduce),
train_fast.adv_moments64 and train_fast.adv_norm_per_map - on two ranks holding UNEQUAL row
counts, the case map sharding creates (7 maps on 2 ranks = 4 + 3 maps). Each pins an exact
equivalence with one process holding the union of the rows:

* the sample-weighted gradient == the single-process gradient of the mean loss over the union
  (and the replicated design's 1/world_size mean is shown to be wrong there);
* pooled advantage moments (all-reduced sums over MB_ALL rows) == the union's mean / std;
* per-map normalisation on the owner rank == per-map normalisation of the union, the
  under-ADV_MAP_MIN_ROWS fallback included (it takes the union's pooled moments);
* all_gather_object, shard_maps, RankPrefix.

The checkpoint union / resume-by-tag round trips run the real trainer:
tests/python/test_ddp_shard_resume.py.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import distributed                       # noqa: E402

needs_gloo = pytest.mark.skipif(not torch.distributed.is_available()
                                or not torch.distributed.is_gloo_available(),
                                reason="gloo backend unavailable")


def _spawn(fn, *args, ws=2):
    with tempfile.TemporaryDirectory() as td:
        init = Path(td, "pg_init").as_uri()
        torch.multiprocessing.spawn(fn, args=(ws, init) + tuple(args), nprocs=ws, join=True)


def _facade(rank, ws, init):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=ws)
    d = distributed.Dist()
    d.rank, d.world_size, d.local_rank = rank, ws, rank
    d.is_main, d.enabled = rank == 0, True
    d.device = torch.device("cpu")
    d.backend = "gloo"
    return d


# ---------------------------------------------------------------- pure pieces
def test_shard_maps_is_round_robin_and_a_partition():
    assert distributed.shard_maps(7, 2, 0) == [0, 2, 4, 6]
    assert distributed.shard_maps(7, 2, 1) == [1, 3, 5]
    assert distributed.shard_maps(7, 4, 3) == [3]
    for n in range(1, 12):
        for ws in range(1, n + 1):
            got = sorted(g for r in range(ws) for g in distributed.shard_maps(n, ws, r))
            assert got == list(range(n))
    with pytest.raises(ValueError):
        distributed.shard_maps(7, 2, 2)


def test_rank_prefix_writes_whole_prefixed_lines():
    buf = io.StringIO()
    w = distributed.RankPrefix(buf, "[r1] ")
    print("a", 1, file=w)
    w.write("half")
    assert buf.getvalue() == "[r1] a 1\n"        # a partial line waits for its newline
    w.write(" line\nnext\n")
    assert buf.getvalue() == "[r1] a 1\n[r1] half line\n[r1] next\n"
    w.write("tail")
    w.close()
    assert buf.getvalue().endswith("[r1] tail\n")
    plain = io.StringIO()
    p0 = distributed.RankPrefix(plain, "")
    print("step 1", file=p0)
    assert plain.getvalue() == "step 1\n"         # rank 0: unprefixed, "^step " still parses


def test_world_size_one_facade_is_a_no_op():
    d = distributed.Dist()
    assert d.all_gather_object({"x": 1}) == [{"x": 1}]
    with d.rank0_first_unless(True):
        pass
    with d.rank0_first_unless(False):
        pass


# ---------------------------------------------------------------- gloo pair: gradients
def _rows(seed):
    """The union of one minibatch's rows: 3 maps, unequal counts (40 + 48 on rank 0's maps 0
    and 2, 32 on rank 1's map 1 - every map at or over ADV_MAP_MIN_ROWS), features, a
    target, an advantage, a map id."""
    g = torch.Generator().manual_seed(seed)
    n = {0: 40, 1: 32, 2: 48}
    m = torch.cat([torch.full((k,), i, dtype=torch.long) for i, k in n.items()])
    x = torch.randn(len(m), 6, generator=g)
    y = torch.randn(len(m), 1, generator=g)
    # very different advantage scales per map: what --adv-norm-per-map exists for
    scale = torch.tensor([0.05, 3.0, 0.4])[m]
    a = torch.randn(len(m), generator=g) * scale + 0.1 * m.double().float()
    return x, y, a, m


def _net(seed=1):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(),
                               torch.nn.Linear(16, 2))


def _loss(net, x, y, a_n):
    """mb_step's shape: a policy term weighted by the normalised advantage + a value MSE, each a
    MEAN over this process's rows"""
    out = net(x)
    return (-(a_n * out[:, 0]).mean() + 0.5 * (out[:, 1:] - y).pow(2).mean())


def _grad_pair(rank, ws, init, overlap, out_path):
    D = _facade(rank, ws, init)
    try:
        from train_fast import adv_norm_per_map
        x, y, a, m = _rows(5)
        mine = (m == 1) if rank == 1 else (m != 1)          # rank 1 owns map 1 alone
        n_all = len(m)
        mb = int(mine.sum())
        net, ref = _net(), _net()
        buckets = ([list(net[2].parameters()), list(net[0].parameters())] if overlap
                   else [list(net.parameters())])
        gs = distributed.FlatGradSync(buckets, D, torch.device("cpu"), weight=mb / n_all)
        if overlap:
            gs.register_hooks()
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        # the local map ids, as the trainer's (idx % N) // PER gives them on this rank
        loc = torch.unique(m[mine], return_inverse=True)[1]
        for step in range(12):
            opt.zero_grad(set_to_none=True)
            a_loc = adv_norm_per_map(a[mine], loc, int(loc.max()) + 1)
            _loss(net, x[mine], y[mine], a_loc).backward()
            gs.sync()
            # the union's gradient AT THESE PARAMETERS (one process, every row): equal at
            # every step of a real Adam trajectory. (Two separate Adam runs are not compared:
            # the policy term's output bias has a gradient of exactly mean(A_norm) = 0, and
            # Adam turns float noise there into a full lr step either way)
            ref.load_state_dict(net.state_dict())
            ref.zero_grad(set_to_none=True)
            _loss(ref, x, y, adv_norm_per_map(a, m, 3)).backward()
            for p, rp in zip(net.parameters(), ref.parameters()):
                assert torch.allclose(p.grad, rp.grad, atol=1e-6, rtol=1e-5), \
                    f"step {step}: sample-weighted shard gradient != the union's gradient"
            if step == 0 and rank == 0:
                torch.save([rp.grad.clone() for rp in ref.parameters()], out_path)
            opt.step()
        c = torch.stack([sum(p.double().sum() for p in net.parameters())])
        D.assert_equal("shard_params", c)                     # EXACT across ranks
        # the replicated design's mean of the two ranks' means is NOT the union's gradient
        # when the ranks hold different row counts: the test can tell the two apart
        net2 = _net()                                         # step 0's parameters again
        gs2 = distributed.FlatGradSync([list(net2.parameters())], D, torch.device("cpu"))
        _loss(net2, x[mine], y[mine],
              adv_norm_per_map(a[mine], loc, int(loc.max()) + 1)).backward()
        gs2.sync()
        ref_g = torch.load(out_path) if rank == 0 else None
        if rank == 0:
            diff = max(float((p.grad - g).abs().max()) for p, g in zip(net2.parameters(), ref_g))
            assert diff > 1e-4, f"1/world_size averaging matched the union ({diff})"
    finally:
        import torch.distributed as dist
        dist.destroy_process_group()


@needs_gloo
@pytest.mark.parametrize("overlap", [False, True])
def test_sample_weighted_gradient_is_the_union_gradient(overlap, tmp_path):
    _spawn(_grad_pair, overlap, str(tmp_path / "ref.pt"))


# ---------------------------------------------------------------- gloo pair: advantages
def _adv_pair(rank, ws, init):
    D = _facade(rank, ws, init)
    try:
        from train_fast import adv_moments64, adv_norm_per_map
        x, y, a, m = _rows(11)
        # one minibatch split over the ranks by map ownership; make map 2 a fallback map (< 32
        # rows) by keeping only 10 of its rows, so the pooled moments are exercised too
        keep = torch.ones(len(m), dtype=torch.bool)
        keep[torch.nonzero(m == 2).squeeze(1)[10:]] = False
        a, m = a[keep], m[keep]
        mine = (m == 1) if rank == 1 else (m != 1)
        n_all = len(m)
        # pooled (flag off): the trainer's per-epoch (2, n_mb) stack, here one minibatch
        ap = a[mine].double().view(1, -1)
        st = torch.stack([ap.sum(1), ap.pow(2).sum(1)])
        D.all_reduce_sum_(st)
        mu, sd = adv_moments64(st, float(n_all))
        assert abs(float(mu[0]) - float(a.double().mean())) < 1e-6
        assert abs(float(sd[0]) - float(a.double().std())) < 1e-5
        pooled_union = (a - a.mean()) / (a.std() + 1e-8)
        assert torch.allclose(((a[mine] - mu[0]) / (sd[0] + 1e-8)), pooled_union[mine],
                              atol=1e-5)
        # per map: the owner rank's rows give the union's per-map moments, and the fallback
        # map (10 rows) gets the union's pooled moments through pooled=
        loc = torch.unique(m[mine], return_inverse=True)[1]
        got = adv_norm_per_map(a[mine], loc, int(loc.max()) + 1, pooled=(mu[0], sd[0]))
        want = adv_norm_per_map(a, m, 3)[mine]
        assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
        # without pooled= the fallback would use THIS rank's moments: different on rank 0
        if rank == 0:
            alone = adv_norm_per_map(a[mine], loc, int(loc.max()) + 1)
            fb = (m[mine] == 2)
            assert float((alone[fb] - want[fb]).abs().max()) > 1e-3
        # every map but the fallback one comes out at unit std, whichever rank owns it
        for k in torch.unique(m[mine]).tolist():
            if k != 2:
                assert abs(float(got[m[mine] == k].std()) - 1.0) < 1e-4
    finally:
        import torch.distributed as dist
        dist.destroy_process_group()


@needs_gloo
def test_advantage_moments_pooled_and_per_map_equal_one_process():
    _spawn(_adv_pair)


# ---------------------------------------------------------------- gloo pair: objects
def _obj_pair(rank, ws, init):
    D = _facade(rank, ws, init)
    try:
        part = None if rank == 0 else {"unstuck": {"celestial": {"T": 0.25}},
                                       "respawn": {"surf_src_celestial": np.arange(5)}}
        got = D.all_gather_object(part)
        assert got[0] is None
        assert got[1]["unstuck"] == {"celestial": {"T": 0.25}}
        assert np.array_equal(got[1]["respawn"]["surf_src_celestial"], np.arange(5))
        rows = D.all_gather_object([(rank, "x" * (1000 * rank))])
        assert [r[0][0] for r in rows] == [0, 1] and len(rows[1][0][1]) == 1000
        with D.rank0_first_unless(True):              # sharded: no barrier per map
            pass
    finally:
        import torch.distributed as dist
        dist.destroy_process_group()


@needs_gloo
def test_all_gather_object_carries_unequal_payloads():
    _spawn(_obj_pair)


# ---------------------------------------------------------------- the child GPU choice
INV = [{"index": 0, "uuid": "GPU-aaa", "free_mib": 3000.0, "total_mib": 16376.0},
       {"index": 1, "uuid": "GPU-bbb", "free_mib": 11800.0, "total_mib": 16376.0}]


def test_gpu_spec_resolves_to_a_card_uuid():
    from surfgym import gpupick
    assert gpupick.visible_value("inherit", inv=INV)[0] is None
    assert gpupick.visible_value("", inv=INV)[0] is None
    assert gpupick.visible_value("free", inv=INV)[0] == "GPU-bbb"      # the most free memory
    assert gpupick.visible_value("0", inv=INV)[0] == "GPU-aaa"
    assert gpupick.visible_value("own", own_uuid="GPU-aaa", inv=INV)[0] == "GPU-aaa"
    assert gpupick.visible_value("GPU-zzz", inv=INV)[0] == "GPU-zzz"
    with pytest.raises(ValueError):
        gpupick.visible_value("7", inv=INV)
    with pytest.raises(ValueError):
        gpupick.visible_value("fastest", inv=INV)
    # no card here (the droplet): every spec falls back to inherit
    assert gpupick.visible_value("free", inv=[])[0] is None
    assert gpupick.visible_value("own", own_uuid=None, inv=[])[0] is None


def test_child_env_pins_the_card_and_drops_torchruns_variables():
    from surfgym import gpupick
    base = {"PATH": "/bin", "RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1",
            "MASTER_PORT": "29500", "TORCHELASTIC_RUN_ID": "x", "CUDA_VISIBLE_DEVICES": "0,1"}
    env, note = gpupick.child_env("free", base=base, drop_torchrun=True, inv=INV)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-bbb" and env["PATH"] == "/bin"
    assert not any(k in env for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_PORT",
                                      "TORCHELASTIC_RUN_ID"))
    assert "card 1" in note
    assert gpupick.child_env("inherit", base=base, inv=INV)[0] is None   # what it always did
    env, _ = gpupick.child_env("inherit", base=base, drop_torchrun=True, inv=INV)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and "RANK" not in env
