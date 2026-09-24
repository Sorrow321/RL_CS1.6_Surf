"""--plan-vocab proposals: the learned planner over per-decision TRAJECTORY
PROPOSALS (surfgym/goalprop.py), its goal-system, trainer and recorder
wiring.

(a) CPU only, no DLL: the spec and its base vocabulary; the batched
    arc-fraction resampling against np.interp; the hindsight bank (rows,
    padding, the (position, velocity x 1 s) query) and its sources (the live
    reservoir's goal rows, a checkpoint's reservoir, the map guard, the
    refresh rule); hindsight candidates are the policy's own segments
    re-anchored at the agent, and their plan lines and budgets are the
    diet's; the perturbations move along exactly the stated axes (take-off
    0.5 s earlier / later, arc height and time x 1.25 / x 0.75, 192 u
    lateral at the end); eps-NMS keeps only candidates more than 64 u apart
    in mean point distance, with the informed and total caps and the
    seed rule; candidate counts, sources and masks; the pointer head (the
    stored log-prob is recomputed exactly, masked slots have probability 0,
    permutation equivariance, the first PPO ratio is 1, a finite update that
    moves only the planner); completion counted by source; the state round
    trip and the refusals of a mismatched spec; the eval hooks (greedy =
    argmax over the set, repeatable per seed, re-planned on the training
    clock, JSON-safe headers).
(b) the built core + maps: with none of the new flags the trainer is
    bit-identical to the base commit (plain race, --goals, --goal-planner
    bfs and vocab, and learned walk / surf warm resumes); the CPU smoke on
    surf_edgeflow_blue050 - a tiny --goal-planner vocab executor, then a
    warm resume with --goal-planner learned --plan-vocab proposals
    --freeze-policy 1 - with tools/record_gate.py on its checkpoint; the
    same on labyrinth_left100 (walking shapes as the uninformed half); the
    refusals.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import torch                                                    # noqa: E402

from surfgym.goallearn import (LearnedPlanner, PlanVocab,        # noqa: E402
                               _spec_default, planner_from_state)
from surfgym.goalplan import BFSPlanner                          # noqa: E402
from surfgym.goalprop import (AXES, SRC_HS,                      # noqa: E402
                              SRC_PERT, SRC_UNIF, HindsightBank,
                              PointerNet, ProposalMaker, ProposalPlanner,
                              arc_samples, base_vocab_spec,
                              make_proposal_hooks, nms_select, perturb,
                              pointer_logp, prop_cols, prop_spec,
                              proposal_planner_from_state)
from surfgym.goalsurf import SurfVocab, build_obs_surf, surf_spec  # noqa: E402

_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
POOL = Path(os.environ.get("SURF_TEST_POOL") or (ROOT / "maps_pool"))
LAB100 = POOL / "labyrinth_left100.bsp"
EF050 = POOL / "surf_edgeflow_blue050.bsp"
needs_lab = pytest.mark.skipif(
    not (DLL.exists() and LAB100.exists()),
    reason="needs the built core + maps_pool/labyrinth_left100.bsp "
           "(SURF_TEST_POOL / SURFCORE_DLL from a worktree)")
needs_ef = pytest.mark.skipif(
    not (DLL.exists() and EF050.exists()
         and EF050.with_suffix(".zones.json").exists()),
    reason="needs the built core + maps_pool/surf_edgeflow_blue050.{bsp,"
           "zones.json} (SURF_TEST_POOL / SURFCORE_DLL from a worktree)")
TRAIN = ROOT / "python" / "train_fast.py"
CELL = 32.0
MINS = np.zeros(3)
SNAP = 0.25


# ==========================================================================
# (a) CPU only
# ==========================================================================
def _toy_graph(n=48):
    """An open box of n x n x 12 voxels at 32 u with a floor, a finish box
    at the far corner."""
    occ = np.ones((12, n, n), np.uint8)
    occ[1:11, 1:n - 1, 1:n - 1] = 0
    box = {"mins": [(n - 6) * CELL, (n - 6) * CELL, 32.0],
           "maxs": [(n - 2) * CELL, (n - 2) * CELL, 96.0]}
    return BFSPlanner(occ, MINS, CELL, finish_box=box, n_targets=0)


def _bank_rows(n, rng, speed=600.0, arc=150.0, grid=False):
    """Reservoir-layout rows: states in the toy box (or on a 400 u grid, so
    no two rows are within the 192 u hindsight reach), each with a segment
    of its own straight flight at 0.25 s snapshots and a vertical arc of
    ``arc`` u over its length (5-12 valid points, zero beyond)."""
    from surfgym.core import STATE_DTYPE
    st = np.zeros(n, STATE_DTYPE)
    if grid:
        i = np.arange(n)
        st["origin"] = np.column_stack([300.0 + 400.0 * (i % 8),
                                        300.0 + 400.0 * (i // 8),
                                        np.full(n, 68.0)])
    else:
        st["origin"] = np.column_stack([rng.uniform(300, 1200, n),
                                        rng.uniform(300, 1200, n),
                                        np.full(n, 68.0)])
    ang = rng.uniform(0, 2 * np.pi, n)
    st["velocity"] = np.column_stack([speed * np.cos(ang),
                                      speed * np.sin(ang), np.zeros(n)])
    segs = np.zeros((n, 64, 3), np.float32)
    seglen = rng.integers(5, 13, n).astype(np.int32)
    for i in range(n):
        k = int(seglen[i])
        t = np.arange(k)[:, None] * SNAP
        s = st["origin"][i] + st["velocity"][i] * t
        s[:, 2] += arc * np.sin(np.pi * np.arange(k) / (k - 1))
        segs[i, :k] = s
    goals = segs[np.arange(n), seglen - 1].copy()
    return st, goals, segs, seglen


def _bank(n=300, seed=0, **kw):
    return HindsightBank.from_arrays(*_bank_rows(n, np.random.default_rng(seed),
                                                 **kw))


def test_spec_and_base_vocabulary():
    s = prop_spec("surf", 32, SNAP)
    assert s["vocab"] == "proposals" and s["base"] == "surf"
    assert s["k"] == 32 and s["in_ch"] == 4 and s["snap_secs"] == 0.25
    assert s["hs_radius"] == 192.0 and s["nms_u"] == 64.0
    assert s["lateral_u"] == 192.0 and s["shift_s"] == 0.5
    assert s["vscale"] == 0.25 and s["n_pts"] == 8
    assert base_vocab_spec(s) == surf_spec()
    w = prop_spec("walk", 16)
    assert w["in_ch"] == 2 and w["base"] == "walk" and w["k"] == 16
    bw = base_vocab_spec(w)
    bw.pop("in_ch")
    assert bw == _spec_default()
    with pytest.raises(ValueError):
        prop_spec("proposals")
    with pytest.raises(ValueError):
        prop_spec("surf", 1)


def test_arc_samples_is_the_polyline_interpolation():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(2, 12))
        q = np.cumsum(rng.normal(0, 100, (n, 3)), axis=0)
        pad = np.vstack([q, np.repeat(q[-1:], 5, axis=0)])
        f = np.sort(rng.uniform(0, 1, 7))
        f = np.concatenate([[0.0], f, [1.0]])
        pts, tot = arc_samples(pad[None, None], f)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
            np.diff(q, axis=0), axis=1))])
        want = np.stack([np.interp(f * cum[-1], cum, q[:, a])
                         for a in range(3)], axis=1)
        assert pts.shape == (1, 1, len(f), 3)
        assert np.allclose(pts[0, 0], want, atol=1e-8)
        assert tot[0, 0] == pytest.approx(cum[-1])
        assert np.allclose(pts[0, 0, -1], q[-1])


def test_hindsight_bank_rows_padding_and_query():
    rng = np.random.default_rng(1)
    st, goals, segs, seglen = _bank_rows(60, rng)
    goals[3] = np.nan            # a row the harvest gave no goal
    seglen[4] = 1                # a degenerate segment
    bank = HindsightBank.from_arrays(st, goals, segs, seglen)
    assert bank.n == 58 and bank.width == 64
    keep = np.setdiff1d(np.arange(60), [3, 4])
    assert np.array_equal(bank.org, st["origin"][keep].astype(np.float64))
    for j, i in enumerate(keep[:10]):
        k = int(seglen[i])
        assert np.array_equal(bank.segs[j, :k], segs[i, :k])
        # padded with the last valid point, not zeros
        assert np.all(bank.segs[j, k:] == segs[i, k - 1])
    # the query: nearest in (position, velocity x 1 s), within the radius
    p = st["origin"][keep[:5]].astype(np.float64) + [50.0, 0.0, 0.0]
    v = st["velocity"][keep[:5]].astype(np.float64) + [0.0, 40.0, 0.0]
    d, r = bank.query(p, v, 3, 192.0)
    keys = np.hstack([bank.org, bank.vel])
    for b in range(5):
        dd = np.linalg.norm(keys - np.hstack([p[b], v[b]])[None], axis=1)
        order = np.argsort(dd)[:3]
        ok = dd[order] <= 192.0
        assert np.array_equal(r[b][ok], order[ok])
        assert np.allclose(d[b][ok], dd[order][ok])
        assert np.all(r[b][~ok] == -1) and np.all(np.isinf(d[b][~ok]))
    assert r[0, 0] == 0 and d[0, 0] == pytest.approx(math.hypot(50, 40))
    # nothing within reach -> all misses
    d, r = bank.query(np.array([[5000.0, 5000.0, 5000.0]]),
                      np.zeros((1, 3)), 4, 192.0)
    assert np.all(r == -1)


def test_bank_sources_reservoir_checkpoint_and_refresh():
    from surfgym.respawn import RespawnBuffer
    rng = np.random.default_rng(2)
    st, goals, segs, seglen = _bank_rows(40, rng)
    rb = RespawnBuffer(4, reservoir=100, margin_ticks=0, snap_every=25,
                       goal_k=(100, 500), map_id="toy")
    assert rb.goal_rows() is None                     # empty
    goals[:10] = np.nan                               # 10 rows carry none
    rb.push_many(st, goals=goals, segs=segs, seglen=seglen)
    gr = rb.goal_rows()
    assert len(gr[0]) == rb.size == 40
    bank = HindsightBank.from_reservoir(rb)
    assert bank.n == 30
    assert np.array_equal(bank.org, st["origin"][10:].astype(np.float64))
    sd = rb.state_dict()
    assert HindsightBank.from_state(sd, map_id="toy").n == 30
    assert HindsightBank.from_state(sd, map_id="another_map") is None
    assert HindsightBank.from_state({"states": None}) is None
    # a buffer without goal columns has no rows to offer
    assert RespawnBuffer(4, reservoir=10).goal_rows() is None
    # the planner's refresh rule: built once, rebuilt only after 10% of the
    # reservoir was harvested anew (or on force)
    P = _toy_graph()
    pl = ProposalPlanner(P, 2, "cpu", spec=prop_spec("surf", 8))
    pl.set_reservoir(rb)
    b0 = pl.bank
    assert b0 is not None and pl.bank_n == 30
    rb.push_many(st[:2], goals=goals[10:12], segs=segs[:2],
                 seglen=seglen[:2])
    pl.set_reservoir(rb)
    assert pl.bank is b0                               # 2 < 10% of 42
    rb.push_many(st[:5], goals=goals[10:15], segs=segs[:5],
                 seglen=seglen[:5])
    pl.set_reservoir(rb)
    assert pl.bank is not b0 and pl.bank_n == 37
    b1 = pl.bank
    pl.set_reservoir(rb, force=True)
    assert pl.bank is not b1 and pl.bank_n == 37


def test_hindsight_candidates_are_the_policys_segments_reanchored():
    from surfgym.goals import resample_polyline_np
    rng = np.random.default_rng(3)
    st, goals, segs, seglen = _bank_rows(40, rng, grid=True)
    bank = HindsightBank.from_arrays(st, goals, segs, seglen)
    mk = ProposalMaker(SurfVocab(), prop_spec("surf", 32, SNAP))
    r = 7
    # the agent 60 u from row 7's state with its velocity: row 7 is the
    # nearest (distance 60 < 192), its segment re-anchored at the agent
    p = st["origin"][r:r + 1].astype(np.float64) + [60.0, 0.0, 0.0]
    v = st["velocity"][r:r + 1].astype(np.float64)
    pr = mk.make(bank, p, v, np.zeros(1), np.random.default_rng(0))
    hs = np.flatnonzero((pr.src[0] == SRC_HS) & pr.mask[0])
    assert len(hs) == 1                     # no other row within reach
    c = int(hs[0])
    k = int(seglen[r])
    want = segs[r, :k].astype(np.float64) - segs[r, 0] + p[0]
    assert int(pr.nq[0, c]) == k
    assert np.allclose(pr.Q[0, c, :k], want)
    assert np.allclose(pr.Q[0, c, 0], p[0])            # starts at the agent
    assert pr.T[0, c] == pytest.approx((k - 1) * SNAP)
    assert pr.match[0, c] == pytest.approx(60.0 / 192.0)
    ln, secs, arc = pr.line(0, c, mk.vocab, True)
    assert np.array_equal(ln, resample_polyline_np(want))   # the diet's rule
    assert secs == pytest.approx((k - 1) * SNAP)
    # through the planner: the line goes on the fan as is, and the budget is
    # 1.5 x the flight time (the diet's hindsight budget)
    P = _toy_graph()
    pl = ProposalPlanner(P, 1, "cpu", spec=prop_spec("surf", 32, SNAP),
                         seed=1)
    pl.use_bank(bank)
    for _ in range(200):
        pl.st.need[:] = True
        idx, lines, _ = pl.plan(p, v, np.zeros(1))
        if pl.o_src[0] == SRC_HS:
            break
    assert pl.o_src[0] == SRC_HS and pl.st.shape[0] == -1
    assert np.array_equal(lines[0], resample_polyline_np(want))
    assert pl.st.budget[0] == math.ceil(1.5 * (k - 1) * SNAP * 100.0 - 1e-6)


def _seed(n=9, W=12, p=(100.0, 200.0, 68.0), step=(100.0, 0.0, 0.0),
          arc=150.0):
    p = np.asarray(p, np.float64)
    i = np.arange(n)
    pts = p + np.outer(i, step)
    pts[:, 2] += arc * np.sin(np.pi * i / (n - 1))
    A = np.repeat(pts[-1:], W, axis=0)
    A[:n] = pts
    return A[None, None], p


def _fr(A, n):
    q = A[:n]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
        np.diff(q, axis=0), axis=1))])
    return cum / cum[-1]


def test_perturbations_move_along_the_stated_axes():
    n = 9
    A, p = _seed(n)
    T = (n - 1) * SNAP
    Q, nq, T6, ok = perturb(A, [[n]], [[T]], p[None], [0.0], 2, SNAP)
    assert Q.shape == (1, 1, 6, 14, 3) and list(AXES) == [
        "earlier", "later", "higher", "flatter", "left", "right"]
    a = A[0, 0]
    # earlier: the first 0.5 s (2 snapshots) dropped, re-anchored at p
    e = Q[0, 0, 0]
    assert nq[0, 0, 0] == n - 2 and ok[0, 0, 0]
    assert np.allclose(e[:n - 2], a[2:n] - a[2] + p)
    assert T6[0, 0, 0] == pytest.approx(T - 0.5)
    # later: 2 snapshots on the initial velocity, then the seed translated
    lt = Q[0, 0, 1]
    v0 = a[1] - a[0]
    want = np.vstack([p, p + v0, p + 2 * v0, a[1:n] + 2 * v0])
    assert nq[0, 0, 1] == n + 2
    assert np.allclose(lt[:n + 2], want)
    assert T6[0, 0, 1] == pytest.approx(T + 0.5)
    # higher / flatter: the arc's height over the start->end chord x 1.25 /
    # x 0.75, the time too; horizontal and both ends unchanged
    fr = _fr(a, n)
    chord = a[0, 2] + fr * (a[n - 1, 2] - a[0, 2])
    for ax, sc in ((2, 1.25), (3, 0.75)):
        h = Q[0, 0, ax, :n]
        assert np.allclose(h[:, :2], a[:n, :2])
        assert np.allclose(h[:, 2] - chord, sc * (a[:n, 2] - chord))
        assert np.allclose(h[[0, -1]], a[[0, n - 1]])
        assert T6[0, 0, ax] == pytest.approx(T * sc)
        assert nq[0, 0, ax] == n
    assert (Q[0, 0, 2, n // 2, 2] - Q[0, 0, 3, n // 2, 2]) == pytest.approx(
        0.5 * 150.0)
    # left / right: the chord runs +x, so left is +y; 192 u x arc fraction
    for ax, sg in ((4, 1.0), (5, -1.0)):
        off = Q[0, 0, ax, :n] - a[:n]
        assert np.allclose(off[:, [0, 2]], 0.0)
        assert np.allclose(off[:, 1], sg * 192.0 * fr)
        assert off[-1, 1] == pytest.approx(sg * 192.0)
        assert T6[0, 0, ax] == pytest.approx(T)
    # padding repeats each variant's last point
    for ax in range(6):
        k = int(nq[0, 0, ax])
        assert np.allclose(Q[0, 0, ax, k:], Q[0, 0, ax, k - 1])
    # a closed loop (end == start): the lateral axis falls back to the
    # seed's first step (+y here: left is -x); a 3-point seed cannot be
    # taken off 2 snapshots earlier
    B, _ = _seed(n=5, step=(0.0, 100.0, 0.0), arc=0.0)
    B = B.copy()
    B[0, 0, 4:] = B[0, 0, 0]
    B[0, 0, 3] = B[0, 0, 0] + [50.0, 100.0, 0.0]
    Q2, nq2, _, ok2 = perturb(B, [[5]], [[1.0]], B[0, 0, :1], [0.0], 2,
                              SNAP)
    off = Q2[0, 0, 4, :5] - B[0, 0, :5]
    assert np.all(off[1:, 0] < 0.0) and np.allclose(off[:, 1], 0.0)
    C, pc = _seed(n=3)
    _, _, _, ok3 = perturb(C, [[3]], [[0.5]], pc[None], [0.0], 2, SNAP)
    assert not ok3[0, 0, 0] and ok3[0, 0, 1:].all()


def test_nms_keeps_only_candidates_different_enough():
    base = np.zeros((8, 3))
    base[:, 0] = np.linspace(100.0, 800.0, 8)

    def at(dy):
        return base + [0.0, dy, 0.0]
    pts = np.stack([at(0.0), at(30.0), at(100.0), at(300.0), at(500.0),
                    at(-40.0)])[None]
    informed = np.array([1, 1, 1, 1, 1, 0], bool)
    seed = np.array([-1, -1, -1, 1, 0, -1])
    valid = np.ones((1, 6), bool)
    kept, nk = nms_select(pts, valid, informed, seed, 6, 6, 64.0)
    # 1: 30 u from 0; 3: its seed (1) was dropped; 5: 40 u from 0
    assert kept[0, :nk[0]].tolist() == [0, 2, 4] and nk[0] == 3
    assert np.all(kept[0, 3:] == -1)
    # the informed cap, then the total cap
    kept, nk = nms_select(pts, valid, informed, seed, 6, 2, 64.0)
    assert kept[0, :nk[0]].tolist() == [0, 2]
    kept, nk = nms_select(pts, valid, np.zeros(6, bool),
                          np.full(6, -1), 2, 2, 64.0)
    assert kept[0, :nk[0]].tolist() == [0, 2]
    # an invalid column is never admitted
    v2 = valid.copy()
    v2[0, 0] = False
    kept, _ = nms_select(pts, v2, informed, seed, 6, 6, 64.0)
    assert 0 not in kept[0] and kept[0, 0] == 1
    # the metric is the MEAN point distance: one point 500 u off is 62.5 u
    # (a duplicate), 520 u off is 65 u (different enough)
    for dz, keep in ((500.0, False), (520.0, True)):
        q = base.copy()
        q[-1, 2] += dz
        k2, n2 = nms_select(np.stack([base, q])[None], np.ones((1, 2), bool),
                            np.zeros(2, bool), np.full(2, -1), 2, 2, 64.0)
        assert (n2[0] == 2) == keep


def _check_set(pr, b, K):
    m = pr.mask[b]
    assert int(m.sum()) == int(pr.n_valid[b]) <= K
    src = pr.src[b]
    assert np.all(src[~m] == -1) and np.all(pr.feats[b][~m] == 0.0)
    assert np.all(np.isin(src[m], [SRC_HS, SRC_PERT, SRC_UNIF]))
    inf = int(((src == SRC_HS) | (src == SRC_PERT)).sum())
    assert inf <= K // 2
    # every admitted pair is more than eps apart (mean point distance)
    P = pr.pts[b][m]
    d = np.linalg.norm(P[:, None] - P[None], axis=-1).mean(-1)
    np.fill_diagonal(d, np.inf)
    assert np.all(d > 64.0)
    # a perturbation's seed is one of this set's hindsight candidates
    hs_org = {tuple(np.round(pr.seed_org[b, c], 3))
              for c in np.flatnonzero(m & (src == SRC_HS))}
    for c in np.flatnonzero(m & (src == SRC_PERT)):
        assert tuple(np.round(pr.seed_org[b, c], 3)) in hs_org
        assert 0 <= pr.axis[b, c] < 6
    # every candidate starts at the agent and is a real plan (>= 64 u)
    assert np.allclose(pr.Q[b][m][:, 0], pr.p[b])
    assert np.all(pr.L[b][m] >= 64.0)
    return src[m]


def test_candidate_sets_counts_sources_and_masks():
    rng = np.random.default_rng(4)
    st, goals, segs, seglen = _bank_rows(400, rng)
    bank = HindsightBank.from_arrays(st, goals, segs, seglen)
    for base, K in (("surf", 32), ("walk", 32), ("surf", 8)):
        vocab = SurfVocab() if base == "surf" else PlanVocab()
        mk = ProposalMaker(vocab, prop_spec(base, K, SNAP))
        n = 40
        p = st["origin"][:n].astype(np.float64)
        v = st["velocity"][:n].astype(np.float64)
        pr = mk.make(bank, p, v, rng.uniform(0, 360, n),
                     np.random.default_rng(5))
        assert pr.feats.shape == (n, K, 54) and pr.feats.dtype == np.float32
        n_hs = n_pert = 0
        for b in range(n):
            s = _check_set(pr, b, K)
            n_hs += int((s == SRC_HS).sum())
            n_pert += int((s == SRC_PERT).sum())
            # the agent stands on a bank row: its own segment is a candidate
            assert (s == SRC_HS).sum() >= 1
        assert n_pert > 0 and n_hs > 0
        # at the bank's own states and with plenty of shapes, the sets fill
        assert np.mean(pr.n_valid) >= K - 1
        # no bank: shapes only
        pr0 = mk.make(None, p, v, np.zeros(n), np.random.default_rng(5))
        for b in range(n):
            s = _check_set(pr0, b, K)
            assert np.all(s == SRC_UNIF)
        assert np.all(pr0.n_valid == K)
        # a shape is its vocabulary's own anchored line
        c = 0
        ln, secs, _ = pr0.line(0, c, vocab, base == "surf")
        k = int(pr0.shape[0, c])
        want = (vocab.anchor(k, p[0], float(pr0.Lu[0])) if base == "surf"
                else vocab.anchor(k, p[0]))
        assert secs is None and np.array_equal(ln, want)


@pytest.mark.parametrize("route", ["numba", "numpy"])
def test_fast_path_builds_exactly_the_reference_sets(route, monkeypatch):
    """ProposalMaker.make (the greedy over informed columns only, the shapes
    by precomputed geometry; its numba kernel, or with SURFGYM_NO_NUMBA its
    numpy fallback) == make_reference (every column through nms_select), on
    surf (no two shapes within eps) and walk (near-duplicate shapes exist:
    the shapes' loop)."""
    import surfgym.goalprop as gp
    if route == "numpy":
        monkeypatch.setattr(gp, "_FAST_SELECT", None)
    elif gp._FAST_SELECT is None:
        pytest.skip("numba unavailable (or SURFGYM_NO_NUMBA=1)")
    rng = np.random.default_rng(12)
    st, goals, segs, seglen = _bank_rows(400, rng)
    bank = HindsightBank.from_arrays(st, goals, segs, seglen)
    for base, K in (("surf", 32), ("walk", 32), ("walk", 8), ("surf", 4)):
        vocab = SurfVocab() if base == "surf" else PlanVocab()
        mk = ProposalMaker(vocab, prop_spec(base, K, SNAP))
        n = 50
        p = st["origin"][:n].astype(np.float64)
        p[n // 2:] += 5000.0                  # half the envs: no hindsight
        v = st["velocity"][:n].astype(np.float64)
        v[:n // 4] *= 0.05                    # slow: the length floor
        yaw = rng.uniform(0, 360, n)
        a = mk.make(bank, p, v, yaw, np.random.default_rng(3))
        b = mk.make_reference(bank, p, v, yaw, np.random.default_rng(3))
        for f in ("mask", "src", "shape", "axis", "n_valid", "nq"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), (base, K, f)
        m = a.mask
        assert np.allclose(a.pts[m], b.pts[m])
        assert np.allclose(a.L[m], b.L[m]) and np.allclose(a.T[m], b.T[m])
        assert np.allclose(a.match[m], b.match[m])
        assert np.allclose(a.seed_org[m], b.seed_org[m])
        assert np.allclose(a.feats, b.feats, atol=1e-6)
        for bi, ci in list(zip(*np.nonzero(m)))[::37]:
            la = a.line(bi, ci, vocab, base == "surf")
            lb = b.line(bi, ci, vocab, base == "surf")
            assert np.allclose(la[0], lb[0], atol=1e-3)
            assert la[1] == lb[1] or abs(la[1] - lb[1]) < 1e-9
        if base == "walk" and K == 32:
            # walking shapes that lie within eps of each other exist, so the
            # shapes' greedy loop (not the shortcut) ran
            nu = min(2 * K, 80)
            sh = np.argsort(np.random.default_rng(3).random((n, 80)),
                            axis=1)[:, :nu]
            dd = 800.0 * mk.shape_dunit[sh[:, :, None], sh[:, None, :]]
            dd[:, np.arange(nu), np.arange(nu)] = np.inf
            assert (dd <= 64.0).any()
            assert np.all(a.n_valid[n // 2:] == K)


def _planner(n=8, K=16, bank=None, seed=3, **cfg):
    P = _toy_graph()
    pl = ProposalPlanner(P, n, "cpu", tick_ms=10.0, act_every=4,
                         cfg=dict({"plan_batch": 8, "plan_epochs": 2}, **cfg),
                         seed=seed, spec=prop_spec("surf", K, SNAP))
    pl.use_bank(bank)
    return pl


def test_pointer_head_logprob_ratio_and_masking():
    rng = np.random.default_rng(6)
    st, goals, segs, seglen = _bank_rows(300, rng)
    bank = HindsightBank.from_arrays(st, goals, segs, seglen)
    n, K = 8, 16
    pl = _planner(n, K, bank)
    pos = st["origin"][:n].astype(np.float64)
    vel = st["velocity"][:n].astype(np.float64)
    idx, lines, fresh = pl.plan(pos, vel, np.zeros(n))
    assert idx.tolist() == list(range(n)) and fresh.all()
    img = torch.as_tensor(pl.o_img)
    scal = torch.as_tensor(pl.o_scal)
    cand = torch.as_tensor(pl.o_cand)
    mask = torch.as_tensor(pl.o_mask)
    act = torch.as_tensor(pl.o_act)
    with torch.no_grad():
        lp, lp_all, v = pointer_logp(pl.net, img, scal, cand, mask, act)
    # the stored log-prob IS the recomputed one (PPO's ratio starts at 1)
    assert np.allclose(lp.numpy(), pl.o_logp, atol=1e-6)
    pr = lp_all.exp()
    assert torch.all(pr[~mask] == 0.0)
    assert torch.allclose(pr.sum(-1), torch.ones(n), atol=1e-5)
    assert torch.all(mask.gather(1, act[:, None]))       # a valid choice
    # permutation equivariance: shuffled candidates, shuffled logits, the
    # same value
    perm = torch.as_tensor(rng.permutation(K))
    with torch.no_grad():
        l0, v0 = pl.net(img, scal, cand, mask)
        l1, v1 = pl.net(img, scal, cand[:, perm], mask[:, perm])
    assert torch.allclose(l1, l0[:, perm], atol=1e-5)
    assert torch.allclose(v1, v0, atol=1e-5)
    # a PPO update over closed plans: ratio 1 before the first step, finite
    # losses, the planner's weights move
    no = np.zeros(n, bool)
    before = {k: t.clone() for k, t in pl.net.state_dict().items()}
    for _ in range(2):
        for _ in range(460):
            pos[:, :2] += rng.normal(0, 4, (n, 2))
            pl.on_tick(pos, no, no, no)
        pl.plan(pos, vel, np.zeros(n))
    assert pl.n_ready() >= 8
    assert [len(b) for b in pl.buf] == [len(b) for b in pl.buf_x]
    up = pl.update()
    assert up is not None and up["ratio0"] < 1e-5
    for k in ("loss_pi", "loss_v", "entropy", "kl"):
        assert np.isfinite(up[k]), k
    assert 0.0 < up["entropy"] <= math.log(K) + 1e-6
    assert any(not torch.equal(before[k], t)
               for k, t in pl.net.state_dict().items())
    assert pl.n_ready() == 0 and all(not b for b in pl.buf_x)
    txt, row = pl.note_and_row()
    assert len(row) == len(prop_cols("surf")) and "PROP cand" in txt
    assert "distinct" in txt


def test_completion_is_counted_by_source():
    rng = np.random.default_rng(7)
    st, goals, segs, seglen = _bank_rows(300, rng)
    bank = HindsightBank.from_arrays(st, goals, segs, seglen)
    n = 6
    pl = _planner(n, 16, bank, plan_novelty=0.0)
    pos = st["origin"][:n].astype(np.float64)
    vel = st["velocity"][:n].astype(np.float64)
    _, lines, _ = pl.plan(pos, vel, np.zeros(n))
    src = pl.o_src.copy()
    no = np.zeros(n, bool)
    # env 0 flies ITS line to the end (completes); env 1 dies; the rest
    # stand still to their budgets
    ln = lines[0].astype(np.float64)
    cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
        np.diff(ln, axis=0), axis=1))))
    cur = pos.copy()
    for t in range(1, 400):
        s = min(cum[-1], 25.0 * t)
        cur[0] = [np.interp(s, cum, ln[:, a]) for a in range(3)]
        pl.on_tick(cur, no, no, no)
        if not pl.st.active[0]:
            break
    assert not pl.st.active[0]
    died = np.zeros(n, bool)
    died[1] = True
    pl.on_tick(cur, died, no, died, term_pos=cur)
    for _ in range(1000):
        if not pl.st.active.any():
            break
        pl.on_tick(cur, no, no, no)
    assert not pl.st.active.any()
    w = pl.w
    assert int(w["closed_src"].sum()) == n
    assert np.array_equal(w["closed_src"], np.bincount(src, minlength=3))
    assert int(w["complete_src"].sum()) == 1
    assert w["complete_src"][src[0]] == 1
    # the transitions and their candidate sets stay in step
    assert [len(b) for b in pl.buf] == [len(b) for b in pl.buf_x] == [1] * n
    for i in range(n):
        assert np.array_equal(pl.buf_x[i][0][1], pl.o_mask[i])
    e = pl.pop_window()
    cs = e[f"complete_{('hs', 'pert', 'unif')[src[0]]}"]
    assert cs == pytest.approx(1.0 / int((src == src[0]).sum()))


def test_state_round_trip_and_spec_refusals():
    rng = np.random.default_rng(8)
    bank = _bank(200, 8)
    pl = _planner(4, 16, bank)
    st, *_ = _bank_rows(4, rng)
    pl.plan(st["origin"].astype(np.float64), st["velocity"].astype(np.float64),
            np.zeros(4))
    sd = pl.state_dict_all()
    assert sd["spec"]["vocab"] == "proposals" and sd["spec"]["k"] == 16
    net, mk, spec = proposal_planner_from_state(sd)
    assert isinstance(net, PointerNet) and mk.k == 16 and mk.surf
    img = torch.rand(2, 4, 32, 32)
    sc = torch.rand(2, 9)
    cand = torch.rand(2, 16, 54)
    mask = torch.rand(2, 16) > 0.3
    mask[:, 0] = True
    with torch.no_grad():
        assert torch.equal(net(img, sc, cand, mask)[0],
                           pl.net(img, sc, cand, mask)[0])
    pl2 = _planner(4, 16, None, seed=9)
    pl2.load_state_dict_all(sd)
    with torch.no_grad():
        assert torch.equal(pl2.net(img, sc, cand, mask)[0],
                           pl.net(img, sc, cand, mask)[0])
    # another K, a categorical surf planner, the recorder's categorical
    # loader: all refuse the proposals state (and back)
    with pytest.raises(ValueError):
        _planner(4, 8, None).load_state_dict_all(sd)
    surf = LearnedPlanner(_toy_graph(), 4, "cpu", spec=surf_spec())
    with pytest.raises(ValueError):
        surf.load_state_dict_all(sd)
    with pytest.raises(ValueError):
        pl2.load_state_dict_all(surf.state_dict_all())
    with pytest.raises(ValueError):
        planner_from_state(sd)
    with pytest.raises(ValueError):
        proposal_planner_from_state(surf.state_dict_all())
    with pytest.raises(ValueError):
        ProposalPlanner(_toy_graph(), 2, "cpu", spec=surf_spec())


class _FakeCore:
    def __init__(self, n=1):
        from surfgym.core import STATE_DTYPE
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)


def test_eval_hooks_are_greedy_repeatable_and_replan_on_the_clock():
    from surfgym.goallearn import VisitGrid
    from surfgym.goals import MultiLine
    rng = np.random.default_rng(9)
    st, goals, segs, seglen = _bank_rows(300, rng)
    bank = HindsightBank.from_arrays(st, goals, segs, seglen)
    pl = _planner(2, 16, bank)
    core = _FakeCore()
    p0 = st["origin"][5].astype(np.float64)
    core.states_view["origin"][0] = p0
    core.states_view["velocity"][0] = st["velocity"][5]
    heads = []
    for rep in range(2):
        ev = {}
        line = MultiLine(1, device="cpu")
        meta, tick = make_proposal_hooks(
            pl.net, pl.maker, bank, pl.graph, core, ev, line=line,
            act_every=4, spec=pl.spec, seed=11)
        h = meta(0)
        heads.append((h, line.pts[0, :int(line.length[0])].numpy().copy()))
    (h1, l1), (h2, l2) = heads
    assert h1 == h2 and np.array_equal(l1, l2)            # repeatable
    assert json.loads(json.dumps(h1)) == h1               # JSON-safe
    assert h1["plan"]["proposals"] and h1["plan"]["candidates"] == 16
    assert h1["plan"]["source"] in ("hindsight", "perturbed", "uninformed")
    # greedy: the argmax of the network over the same candidate set
    p = p0[None]
    v = st["velocity"][5:6].astype(np.float64)
    pr = pl.maker.make(bank, p, v, np.zeros(1), np.random.default_rng(11))
    vg = VisitGrid(1, pl.wm)
    vg.update(p)
    img, sc = build_obs_surf(pl.wm, vg, np.zeros(1, np.int64), p, v,
                             np.zeros(1), pl.finish)
    with torch.no_grad():
        lg, _ = pl.net(torch.as_tensor(img), torch.as_tensor(sc),
                       torch.as_tensor(pr.feats), torch.as_tensor(pr.mask))
    c = int(lg.argmax(-1)[0])
    want, _, _ = pr.line(0, c, pl.maker.vocab, True)
    assert np.allclose(l2, want, atol=1e-4)
    # standing still: the plan closes on its budget and the next one is
    # chosen at the next decision boundary; a finish is counted
    zero = np.zeros(1, np.uint8)
    one = np.ones(1, np.uint8)
    t = 0
    for t in range(2000):
        tick(t, None, None, zero, zero)
        if ev["plans"] == 2:
            break
    assert ev["plans"] == 2 and ev["closed"] == 1 and (t + 1) % 4 == 0
    assert sum(ev["src"]) == 2 and len(ev["ncand"]) == 2 and "void" in ev
    core.goal_hits[0] = 1
    tick(t + 1, None, None, one, zero)
    assert ev["succ"] == 1 and ev["n"] == 1
    # no bank (another map): shapes only
    ev3 = {}
    meta3, _ = make_proposal_hooks(pl.net, pl.maker, None, pl.graph, core,
                                   ev3, act_every=4, spec=pl.spec, seed=11)
    assert meta3(0)["plan"]["source"] == "uninformed"


def test_walk_base_planner_plans_and_updates():
    """The walking base: the patch observation, the 80 walking shapes as
    the uninformed half, the off-graph diagnostic, no void column."""
    P = BFSPlanner(np.pad(np.zeros((2, 62, 62), np.uint8), 1,
                          constant_values=1), MINS, CELL,
                   finish_box={"mins": [58 * CELL, 58 * CELL, 32.0],
                               "maxs": [62 * CELL, 62 * CELL, 96.0]},
                   n_targets=0)
    rng = np.random.default_rng(10)
    st, goals, segs, seglen = _bank_rows(200, rng, speed=250.0, arc=0.0)
    bank = HindsightBank.from_arrays(st, goals, segs, seglen)
    pl = ProposalPlanner(P, 6, "cpu", act_every=4, seed=2,
                         cfg={"plan_batch": 6, "plan_epochs": 1},
                         spec=prop_spec("walk", 16, SNAP))
    pl.use_bank(bank)
    assert pl.in_ch == 2 and not pl.surf and pl.st.budget_ticks == 480
    pos = st["origin"][:6].astype(np.float64)
    vel = st["velocity"][:6].astype(np.float64)
    idx, lines, _ = pl.plan(pos, vel, np.zeros(6))
    for i, ln in enumerate(lines):
        assert np.allclose(ln[0], pos[i], atol=1e-3)
    no = np.zeros(6, bool)
    for _ in range(500):
        pl.on_tick(pos, no, no, no)
    pl.plan(pos, vel, np.zeros(6))
    assert pl.update() is not None
    txt, row = pl.note_and_row()
    assert len(row) == len(prop_cols("walk")) and "void" not in txt
    # flat walking segments: the higher / flatter variants coincide with
    # their seeds and are deduplicated away
    pr = pl.maker.make(bank, pos, vel, np.zeros(6), np.random.default_rng(0))
    ax = pr.axis[pr.src == SRC_PERT]
    assert not np.isin(ax, [2, 3]).any()


# ==========================================================================
# (b) the built core + maps
# ==========================================================================
def _env():
    # -1, not "": on Windows an EMPTY value UNSETS the variable and the GPU
    # stays visible (memory: windows-empty-env-var-unsets)
    # SURFGYM_LEGACY_SPAWNS: these runs are compared bit for bit with a commit that predates
    # map_spawn_pool lifting embedded spawns (edgeflow's front row) clear
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4", SURFGYM_LEGACY_SPAWNS="1")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _run(cmd, timeout=2400):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(ROOT), timeout=timeout, encoding="utf-8",
                          errors="replace")


# the stage-1 test set (tests/python/test_goal_planner.py)
LAB_FLAGS = ["--map", str(LAB100), "--reward", "race", "--envs", "64",
             "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
             "--lidar-cell", "32", "--goal-cell", "32",
             "--lidar-range", "11500", "--lidar-near", "2000",
             "--emb", "64", "--hidden", "64", "--act-every", "4",
             "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
             "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2",
             "--vf", "0.5", "--ent", "0.005", "--n-steps", "8",
             "--epochs", "1", "--minibatches", "2", "--ep-ticks", "96",
             "--time-pen", "0.005", "--success-bonus", "50",
             "--finish-k", "0", "--stall-secs", "30", "--maxvel", "4000",
             "--train-stride", "1", "--yaw-adaptive", "--respawn-frac", "0.9",
             "--respawn-margin", "0.1", "--respawn-reservoir", "1000",
             "--int-coef", "0.25", "--int-view", "8", "--int-speed", "3",
             "--ckpt-every", "1e9", "--record-every", "4096",
             "--eval-eps", "2", "--eval-greedy-only", "--seed", "7"]
MODERN = ["--view-continuous", "--view-absolute", "velocity", "--keys-hold"]
PLAN = ["--goals", "1", "--goal-obs", "fan", "--goal-planner", "bfs",
        "--goal-fan-offsets", "0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0"]
# the surf smoke of tests/python/test_goal_surf.py: drop spawns, so an
# untrained executor moves and the reservoir harvests segments
EF_FLAGS = ["--map", str(EF050), "--reward", "race", "--envs", "128",
            "--spawn", "mixed", "--lidar-w", "16", "--lidar-h", "8",
            "--lidar-cell", "32", "--goal-cell", "32",
            "--lidar-range", "11500", "--lidar-near", "2000",
            "--emb", "64", "--hidden", "64", "--act-every", "4",
            "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
            "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2",
            "--vf", "0.5", "--ent", "0.005", "--n-steps", "16",
            "--epochs", "1", "--minibatches", "2", "--ep-ticks", "1000",
            "--time-pen", "0.005", "--success-bonus", "50",
            "--finish-k", "0", "--stall-secs", "0", "--maxvel", "4000",
            "--train-stride", "1", "--yaw-adaptive", "--respawn-frac", "0.9",
            "--respawn-margin", "0.5", "--respawn-reservoir", "20000",
            "--int-coef", "0.25", "--int-view", "8", "--int-speed", "3",
            "--ckpt-every", "1e9", "--record-every", "65536",
            "--eval-eps", "2", "--eval-greedy-only", "--seed", "7"]
VOCAB = ["--goals", "1", "--goal-obs", "fan", "--goal-planner", "vocab",
         "--plan-vocab", "surf", "--goal-reward", "arc",
         "--race-dist", "euclid"]
EF_SRC_STEPS = 393216
EF_RES_STEPS = 196608
NEW_KEYS = ("plan_k", "plan_base")


def _train(run, flags, extra, script=TRAIN, steps="12288", runs_dir=None):
    shutil.rmtree((runs_dir or ROOT / "runs") / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + flags
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _cfg(d: Path) -> dict:
    return json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]


def _rows(d: Path):
    rows = (d / "progress.csv").read_text(encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


def _f(x):
    return float(x) if x not in ("", None) else float("nan")


def _base_tree(dst: Path):
    """The python/ tree of the last first-parent commit WITHOUT the proposal
    planner (HEAD itself while it is uncommitted), extracted under ``dst``;
    returns the ref, or None."""
    try:
        r = subprocess.run(["git", "rev-list", "--first-parent",
                            "--max-count=200", "HEAD"], capture_output=True,
                           text=True, cwd=str(ROOT), timeout=60)
        refs = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        refs = []
    for ref in refs:
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode != 0 or b"--plan-k" in r.stdout:
            continue
        z = dst / "base_python.zip"
        subprocess.run(["git", "archive", "--format=zip", "-o", str(z), ref,
                        "python"], check=True, cwd=str(ROOT), timeout=120)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
        z.unlink()
        return ref
    return None


def _ckpt(p: Path):
    return torch.load(p, map_location="cpu", weights_only=False)


def _same_policy_and_adam(sa, sb):
    assert set(sa["policy"]) == set(sb["policy"])
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    assert set(oa) == set(ob)
    for i in oa:
        for k in oa[i]:
            if torch.is_tensor(oa[i][k]):
                assert torch.equal(oa[i][k], ob[i][k]), (i, k)


def _assert_identical(a: Path, b: Path):
    ca, cb = _cfg(a), _cfg(b)
    assert not any(k in ca for k in NEW_KEYS)
    assert ca.get("plan_vocab") != "proposals"
    assert ca == cb
    ra, rb = _rows(a), _rows(b)
    assert len(ra) == len(rb) >= 3
    assert list(ra[0]) == list(rb[0])
    # the proposals-only columns (the diet has its own plan/complete_hs)
    assert not any(k in ra[0] for k in ("plan/cand", "plan/cand_pert",
                                        "plan/choose_pert", "plan/bank"))
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], (k, x[k], y[k])
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert ta and [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    for extra in ("goals.csv", "plan.csv"):
        assert (a / extra).exists() == (b / extra).exists(), extra
        if (a / extra).exists():
            assert (a / extra).read_bytes() == (b / extra).read_bytes(), extra
    sa, sb = _ckpt(a / "ckpt_final.pt"), _ckpt(b / "ckpt_final.pt")
    _same_policy_and_adam(sa, sb)
    assert ("planner" in sa) == ("planner" in sb)
    if "planner" in sa:
        pa, pb = sa["planner"], sb["planner"]
        assert pa["spec"] == pb["spec"]
        for k in pa["net"]:
            assert torch.equal(pa["net"][k], pb["net"][k]), k
        assert np.array_equal(pa["nov_count"], pb["nov_count"])
        assert pa["updates"] == pb["updates"]


@pytest.fixture(scope="module")
def ef_src():
    """A tiny --goal-planner vocab surf executor on blue050 (drop spawns),
    shared by the surf flag-off resume and the proposals smoke."""
    if not (DLL.exists() and EF050.exists()):
        pytest.skip("needs the built core + surf_edgeflow_blue050")
    src = ROOT / "runs" / "pprop_ef_src"
    _train(src.name, EF_FLAGS, MODERN + VOCAB, steps=str(EF_SRC_STEPS))
    yield src
    shutil.rmtree(src, ignore_errors=True)


@pytest.fixture(scope="module")
def lab_src():
    """A tiny stage-1 bfs executor on lab100."""
    if not (DLL.exists() and LAB100.exists()):
        pytest.skip("needs the built core + labyrinth_left100")
    src = ROOT / "runs" / "pprop_lab_src"
    _train(src.name, LAB_FLAGS, MODERN + PLAN + ["--goal-reward", "arc"],
           steps="8192")
    yield src
    shutil.rmtree(src, ignore_errors=True)


@needs_lab
@pytest.mark.parametrize("mode", ["race", "goals", "bfs"])
def test_flag_off_is_bit_identical_to_the_base_commit(mode, tmp_path):
    """None of the new flags: this branch's trainer against the base
    commit's whole python/ tree (config, progress.csv minus fps, the eval
    trajectories, goals.csv / plan.csv, weights, Adam moments)."""
    ref = _base_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-proposals python/ tree in the history")
    old = tmp_path / "python" / "train_fast.py"
    extra = MODERN + {"race": [],
                      "goals": ["--goals", "1", "--goal-obs", "fan"],
                      "bfs": PLAN + ["--goal-reward", "arc"]}[mode]
    new_run = ROOT / "runs" / f"pprop_ctl_new_{mode}"
    old_run = tmp_path / "runs" / f"pprop_ctl_old_{mode}"
    _train(new_run.name, LAB_FLAGS, extra)
    _train(old_run.name, LAB_FLAGS, extra, script=old,
           runs_dir=tmp_path / "runs")
    assert old_run.exists(), "the base trainer writes under its own tree"
    _assert_identical(new_run, old_run)
    shutil.rmtree(new_run, ignore_errors=True)


@needs_ef
def test_flag_off_vocab_executor_is_bit_identical(tmp_path):
    """--goal-planner vocab (the diet, whose goal-system iterate() gained a
    branch) from scratch: bit-identical to the base commit."""
    ref = _base_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-proposals python/ tree in the history")
    old = tmp_path / "python" / "train_fast.py"
    new_run = ROOT / "runs" / "pprop_ctl_new_vocab"
    old_run = tmp_path / "runs" / "pprop_ctl_old_vocab"
    _train(new_run.name, EF_FLAGS, MODERN + VOCAB, steps="98304")
    _train(old_run.name, EF_FLAGS, MODERN + VOCAB, script=old,
           runs_dir=tmp_path / "runs", steps="98304")
    _assert_identical(new_run, old_run)
    shutil.rmtree(new_run, ignore_errors=True)


@needs_lab
def test_flag_off_learned_walk_resume_is_bit_identical(lab_src, tmp_path):
    ref = _base_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-proposals python/ tree in the history")
    old = tmp_path / "python" / "train_fast.py"
    res = ["--ckpt", str(lab_src / "ckpt_final.pt"), "--goal-planner",
           "learned", "--freeze-policy", "1", "--ep-ticks", "1200",
           "--plan-batch", "32", "--record-every", "16384"]
    new_run = ROOT / "runs" / "pprop_lrn_new"
    old_run = tmp_path / "runs" / "pprop_lrn_old"
    _train(new_run.name, LAB_FLAGS, res, steps=str(8192 + 32768))
    _train(old_run.name, LAB_FLAGS, res, script=old,
           runs_dir=tmp_path / "runs", steps=str(8192 + 32768))
    _assert_identical(new_run, old_run)
    assert "planner" in _ckpt(new_run / "ckpt_final.pt")
    shutil.rmtree(new_run, ignore_errors=True)


@needs_ef
def test_flag_off_learned_surf_resume_is_bit_identical(ef_src, tmp_path):
    """--goal-planner learned over the surf vocab executor (plan_vocab surf
    restored - the resume path this branch changed): bit-identical."""
    ref = _base_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-proposals python/ tree in the history")
    old = tmp_path / "python" / "train_fast.py"
    res = ["--ckpt", str(ef_src / "ckpt_final.pt"), "--goal-planner",
           "learned", "--freeze-policy", "1", "--plan-batch", "64"]
    steps = str(EF_SRC_STEPS + 98304)
    new_run = ROOT / "runs" / "pprop_lsrf_new"
    old_run = tmp_path / "runs" / "pprop_lsrf_old"
    _train(new_run.name, EF_FLAGS, res, steps=steps)
    _train(old_run.name, EF_FLAGS, res, script=old,
           runs_dir=tmp_path / "runs", steps=steps)
    _assert_identical(new_run, old_run)
    assert _cfg(new_run)["plan_vocab"] == "surf"
    shutil.rmtree(new_run, ignore_errors=True)


def _gate(run, ck):
    g = _run([sys.executable, str(ROOT / "tools" / "record_gate.py"), run,
              "--ckpt", str(ck), "--no-pov"])
    assert g.returncode == 0, g.stdout[-3000:] + g.stderr[-3000:]
    assert "record gate PASSED: 3 recording(s)" in g.stdout


@needs_ef
def test_surf_smoke_proposals_over_the_vocab_executor(ef_src):
    """The warm resume of a --goal-planner vocab executor with --goal-planner
    learned --plan-vocab proposals --freeze-policy 1: the base vocabulary is
    restored as surf, the hindsight bank is the reservoir's, candidate sets
    of all three sources are built and completion is counted by source, the
    pointer planner updates with finite losses, the executor and its Adam
    stay bit-identical, the eval runs the proposal planner greedy from the
    map start, and the record gate passes."""
    run = "pprop_ef_prop"
    d = ROOT / "runs" / run
    shutil.rmtree(d, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt",
              str(ef_src / "ckpt_final.pt")] + EF_FLAGS
             + ["--steps", str(EF_SRC_STEPS + EF_RES_STEPS),
                "--goal-planner", "learned", "--plan-vocab", "proposals",
                "--freeze-policy", "1", "--plan-batch", "64"])
    out = r.stdout
    assert r.returncode == 0, out[-5000:] + r.stderr[-4000:]
    assert "plan_base=surf" in out and "planner: FRESH" in out
    assert "planner LEARNED over TRAJECTORY PROPOSALS" in out
    assert "goals: LEARNED PLANNER over TRAJECTORY PROPOSALS" in out
    assert "proposal planner greedy" in out
    cfg = _cfg(d)
    assert cfg["goal_planner"] == "learned"
    assert cfg["plan_vocab"] == "proposals" and cfg["plan_base"] == "surf"
    assert cfg["plan_k"] == 32 and cfg["freeze_policy"] == 1
    assert "plan_hindsight" not in cfg
    rows = _rows(d)
    cols = prop_cols("surf")
    assert list(rows[0])[-len(cols):] == cols
    upd = [x for x in rows if x["plan/loss_pi"] != ""]
    assert upd, "the planner never updated"
    for x in upd:
        assert np.isfinite(float(x["plan/loss_pi"]))
        assert np.isfinite(float(x["plan/loss_v"]))
    ent = [_f(x["plan/entropy"]) for x in rows if x["plan/entropy"]]
    assert ent and all(0.0 < e <= math.log(32) + 1e-6 for e in ent)
    cand = [_f(x["plan/cand"]) for x in rows if x["plan/cand"]]
    assert cand and all(0.0 < c <= 32.0 for c in cand)
    inf = [_f(x["plan/cand_hs"]) + _f(x["plan/cand_pert"]) for x in rows
           if x["plan/cand_hs"]]
    assert max(inf) > 0.0, "no hindsight or perturbed candidate was built"
    assert max(int(x["plan/bank"]) for x in rows) > 0
    for col in ("plan/wall", "plan/wall_base", "plan/void", "plan/void_base",
                "plan/choose_hs", "plan/choose_pert", "plan/choose_unif"):
        v = [_f(x[col]) for x in rows if x[col]]
        assert v and all(0.0 <= a <= 1.0 for a in v), col
    s_src = _ckpt(ef_src / "ckpt_final.pt")
    s_new = _ckpt(d / "ckpt_final.pt")
    _same_policy_and_adam(s_src, s_new)
    assert s_new["planner"]["spec"]["vocab"] == "proposals"
    assert s_new["planner"]["spec"]["base"] == "surf"
    assert s_new["planner"]["updates"] >= 1
    tr = sorted(d.glob("traj_*.jsonl"))
    txt = tr[-1].read_text(encoding="utf-8")
    assert "Infinity" not in txt and "NaN" not in txt
    h = json.loads(txt.splitlines()[0])
    assert h["plan"]["planner"] == "learned" and h["plan"]["proposals"]
    _gate(run, d / "ckpt_final.pt")
    # the smoke's numbers, for the report
    print("\nSMOKE surf proposals (step, closed, cand, cand hs/pert/unif, "
          "choose hs/pert/unif, complete hs/pert/unif, bank, entropy)")
    for x in rows:
        if x["plan/cand"]:
            print("  ", tuple(x[k] for k in (
                "time/total_timesteps", "plan/closed", "plan/cand",
                "plan/cand_hs", "plan/cand_pert", "plan/cand_unif",
                "plan/choose_hs", "plan/choose_pert", "plan/choose_unif",
                "plan/complete_hs", "plan/complete_pert",
                "plan/complete_unif", "plan/bank", "plan/entropy")))
    shutil.rmtree(d, ignore_errors=True)


@needs_lab
def test_lab_smoke_proposals_walk_base(lab_src):
    """The same on labyrinth_left100 over a stage-1 bfs executor: the base
    vocabulary is the walking one (80 shapes of 800 u), the planner's input
    the walkable patch; the record gate passes."""
    run = "pprop_lab_prop"
    d = ROOT / "runs" / run
    shutil.rmtree(d, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt",
              str(lab_src / "ckpt_final.pt")] + LAB_FLAGS
             + ["--steps", str(8192 + 81920), "--goal-planner", "learned",
                "--plan-vocab", "proposals", "--freeze-policy", "1",
                "--ep-ticks", "1200", "--plan-batch", "64",
                "--record-every", "32768"])
    out = r.stdout
    assert r.returncode == 0, out[-5000:] + r.stderr[-4000:]
    assert "plan_base=walk" in out and "walking vocabulary shapes" in out
    cfg = _cfg(d)
    assert cfg["plan_vocab"] == "proposals" and cfg["plan_base"] == "walk"
    rows = _rows(d)
    cols = prop_cols("walk")
    assert list(rows[0])[-len(cols):] == cols
    assert "plan/void" not in rows[0]
    assert [x for x in rows if x["plan/loss_pi"] != ""]
    s_new = _ckpt(d / "ckpt_final.pt")
    _same_policy_and_adam(_ckpt(lab_src / "ckpt_final.pt"), s_new)
    assert s_new["planner"]["spec"]["base"] == "walk"
    _gate(run, d / "ckpt_final.pt")
    last = [x for x in rows if x["plan/cand"]][-1]
    print("\nSMOKE lab proposals", {k: last[k] for k in last
                                    if k.startswith("plan/")})
    shutil.rmtree(d, ignore_errors=True)


@needs_lab
def test_refusals(lab_src):
    base = [sys.executable, "-u", str(TRAIN), "--run", "pprop_bad"] \
        + LAB_FLAGS + ["--steps", "12288"]
    goals = ["--goals", "1", "--goal-obs", "fan"]
    ck = ["--ckpt", str(lab_src / "ckpt_final.pt")]
    # a categorical learned planner's checkpoint, to resume from
    lrn = ROOT / "runs" / "pprop_lrn_src"
    _train(lrn.name, LAB_FLAGS, ck + ["--goal-planner", "learned",
                                      "--freeze-policy", "1",
                                      "--plan-batch", "16"], steps="12288")
    cases = [
        (MODERN + goals + ["--goal-planner", "bfs", "--plan-vocab",
                           "proposals"],
         "--plan-vocab without --goal-planner learned or vocab"),
        (MODERN + goals + ["--goal-planner", "vocab", "--plan-vocab",
                           "proposals", "--goal-reward", "arc"],
         "--plan-vocab proposals is the LEARNED planner's candidate set"),
        (MODERN + goals + ["--goal-planner", "bfs", "--plan-k", "16"],
         "--plan-k without --plan-vocab proposals"),
        (ck + ["--goal-planner", "learned", "--plan-vocab", "proposals",
               "--freeze-policy", "1", "--plan-k", "1"],
         "--plan-k: at least 2 candidates"),
        (["--ckpt", str(lrn / "ckpt_final.pt"), "--goal-planner", "learned",
          "--plan-vocab", "proposals", "--freeze-policy", "1"],
         "--plan-vocab proposals on a learned-planner checkpoint"),
    ]
    for extra, msg in cases:
        r = _run(base + extra)
        assert r.returncode != 0, extra
        assert msg in r.stdout + r.stderr, (extra,
                                            (r.stdout + r.stderr)[-1500:])
    shutil.rmtree(ROOT / "runs" / "pprop_bad", ignore_errors=True)
    shutil.rmtree(lrn, ignore_errors=True)
