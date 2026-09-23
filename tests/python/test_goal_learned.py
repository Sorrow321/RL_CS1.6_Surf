"""--goal-planner learned: the learned planner (surfgym/goallearn.py), its
goal-system wiring, --freeze-policy, the recorder and the trainer.

(a) CPU only, no DLL: the vocabulary (80 shapes, 800 u each, headings and
    turns as specified) and plans anchored at the agent; the walkable patch
    and the wall-crossing diagnostic on a hand-made room with a wall; the
    visit channel; the macro-step bookkeeping (a plan ends on completion, on
    its budget and on the episode's end; rewards accumulated as specified;
    an end while waiting for the next plan); the planner's GAE on a hand
    case; a PPO update is finite and moves only the planner; the state
    round trip; the eval hooks on a fake core.
(b) the trainer (CPU, the built core + maps_pool/labyrinth_left100.bsp):
    with none of the new flags it is bit-identical to the base commit
    (plain race, --goals, --goal-planner bfs); the smoke - a tiny stage-1
    bfs executor, then a warm resume with --goal-planner learned
    --freeze-policy 1 - updates the planner with finite loss and entropy,
    leaves the executor and its Adam state bit-identical, and
    tools/record_gate.py passes on the learned checkpoint; the refusals.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import torch                                                    # noqa: E402

from surfgym.goallearn import (COMPLETE_FRAC, R_EXEC_FAIL,       # noqa: E402
                               R_EXEC_OK, LearnedPlanner, PlanVocab,
                               VisitGrid, WalkMap, make_learned_hooks,
                               plan_gae, planner_from_state)
from surfgym.goalplan import BFSPlanner                          # noqa: E402

_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
POOL = Path(os.environ.get("SURF_TEST_POOL") or (ROOT / "maps_pool"))
LAB100 = POOL / "labyrinth_left100.bsp"
LAB200 = POOL / "labyrinth_left200.bsp"
needs_lab = pytest.mark.skipif(
    not (DLL.exists() and LAB100.exists() and LAB200.exists()),
    reason="needs the built core + maps_pool/labyrinth_left{100,200}.bsp "
           "(SURF_TEST_POOL / SURFCORE_DLL from a worktree)")
TRAIN = ROOT / "python" / "train_fast.py"
CELL = 32.0
MINS = np.zeros(3)


# ==========================================================================
# (a) the planner, CPU only
# ==========================================================================
def test_vocabulary_is_80_shapes_of_800u():
    V = PlanVocab()
    assert V.K == 80 and V.raw.shape == (80, 9, 3)
    seg = np.linalg.norm(np.diff(V.raw, axis=1), axis=2)
    assert np.allclose(seg, 100.0) and np.allclose(seg.sum(1), 800.0)
    assert np.all(V.raw[:, 0] == 0.0) and np.all(V.raw[:, :, 2] == 0.0)
    # 16 world headings x (straight, +45, -45, +90, -90)
    assert sorted(set(np.round(V.heading_deg, 6))) == \
        sorted(set(np.round(np.arange(16) * 22.5, 6)))
    assert list(V.turn_deg[:5]) == [0.0, 45.0, -45.0, 90.0, -90.0]
    for k in range(V.K):
        h = math.radians(V.heading_deg[k])
        th = math.radians(V.turn_deg[k])
        d = np.diff(V.raw[k], axis=0)[:, :2]
        a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        # segment i points along h + theta (i + 0.5) / 8
        want = h + th * (np.arange(8) + 0.5) / 8.0
        assert np.allclose(np.cos(a), np.cos(want), atol=1e-9)
        assert np.allclose(np.sin(a), np.sin(want), atol=1e-9)
        # a LEFT turn ends to the left of the start heading
        end = V.raw[k, -1, :2]
        cross = math.cos(h) * end[1] - math.sin(h) * end[0]
        assert np.sign(round(cross, 6)) == np.sign(V.turn_deg[k])
    assert np.allclose(V.raw[0, -1], [800.0, 0.0, 0.0])      # +x straight
    assert np.allclose(V.raw[4 * 5, -1], [0.0, 800.0, 0.0], atol=1e-9)
    # all 80 PATHS are distinct; their END points are not: (h, +45) and
    # (h + 45, -45) use the same eight segment directions in reverse order
    # (and likewise the 90s), so 16 x 5 shapes reach 48 distinct ends
    paths = np.round(V.raw[:, 1:, :2].reshape(V.K, -1), 3)
    assert len({tuple(e) for e in paths}) == V.K
    ends = np.round(V.raw[:, -1, :2], 3)
    assert len({tuple(e) for e in ends}) == 48


def test_plans_are_anchored_at_the_agent_and_lifted():
    V = PlanVocab()
    o = np.array([-437.5, 1210.25, 68.0])
    for k in (0, 3, 17, 79):
        ln = V.anchor(k, o)
        assert ln.dtype == np.float32 and ln.shape == (V.n_line, 3)
        assert np.allclose(ln[0], o, atol=1e-3)                # starts there
        assert np.allclose(ln[:, 2], o[2])                     # its height
        assert np.allclose(ln[-1], V.raw[k, -1] + o, atol=1e-2)
        # resampled at (about) the fan spacing: 800 u of arc in 6 steps;
        # a chord across a bend is shorter, never longer
        seg = np.linalg.norm(np.diff(ln.astype(np.float64), axis=0), axis=1)
        assert np.all(seg <= 800.0 / 6.0 + 1e-3)
        if V.turn_deg[k] == 0.0:
            assert np.allclose(seg, 800.0 / 6.0, atol=1e-3)
    assert V.n_line == 7


def _room(nz=4, ny=40, nx=40):
    occ = np.ones((nz, ny, nx), np.uint8)
    occ[1:nz - 1, 1:ny - 1, 1:nx - 1] = 0
    return occ


def _walled():
    """A 40 x 40-cell room at 32 u (1,280 u) with a full-height wall at x
    cells 20-21 for y cells 1..27 (a gap above), and a finish box top-right."""
    occ = _room()
    occ[1:3, 1:28, 20:22] = 1
    box = {"mins": [30 * CELL, 30 * CELL, 32.0],
           "maxs": [34 * CELL, 34 * CELL, 96.0]}
    return BFSPlanner(occ, MINS, CELL, finish_box=box, n_targets=0)


def test_walk_patch_sees_the_wall():
    P = _walled()
    wm = WalkMap(P)
    assert wm.s == 2 and wm.pc == 64.0 and wm.n == 32
    o = np.array([[5.5 * CELL, 6.5 * CELL, 68.0]])      # left of the wall
    iz = wm.layer(o)
    cy, cx = wm.cells(o)
    pt = wm.patch(iz, cy, cx)[0]
    assert pt.shape == (32, 32)
    assert pt[16, 16] == 1.0                            # the agent's cell
    # the wall occupies graph columns 20, 21 = patch column 10 (of 20);
    # relative to the agent's patch column (cx = 2) that is window col 24
    assert cx[0] == 2 and cy[0] == 3
    assert pt[16, 16 + 8] == 0.0                        # the wall's column
    assert pt[16, 16 + 7] == 1.0 and pt[16, 16 + 9] == 1.0
    assert pt[16, 16 - 2] == 0.5                        # the room's rim
    assert pt[0, 0] == 0.0                              # off the map


def test_wall_crossing_diagnostic_on_a_hand_made_wall():
    P = _walled()
    lp = LearnedPlanner(P, 2, "cpu", tick_ms=10.0)
    V = lp.vocab
    o = np.array([[10.5 * CELL, 4.5 * CELL, 68.0]])
    iz = lp.wm.layer(o)
    cross = lp.wall_all(iz, o)[0]
    east = 0 * 5          # heading 0 (+x), straight: into the wall at x 640
    north = 4 * 5         # heading 90 (+y), straight: along the wall, 800 u
    west = 8 * 5          # heading 180: 336 u to the room's west wall
    assert cross[east]                                  # through the wall
    assert not cross[north]                             # parallel to it
    assert cross[west]                                  # into the outer wall
    # the base rate over the vocabulary is strictly between 0 and 1 here
    assert 0.0 < cross.mean() < 1.0
    assert V.K == len(cross)
    # the per-choice form agrees with the whole-vocabulary matrix
    from surfgym.goallearn import vocab_crossings
    o2 = np.vstack([o, [[20.5 * CELL + 400.0, 30.5 * CELL, 68.0]]])
    iz2 = lp.wm.layer(o2)
    full = vocab_crossings(V, lp.wm, iz2, o2)
    sh = np.array([east, north])
    got = vocab_crossings(V, lp.wm, iz2, o2, shapes=sh)
    assert got.tolist() == full[np.arange(2), sh].tolist()
    # the continuous companion: the share of the polyline off the graph
    from surfgym.goallearn import vocab_offgraph
    anyx, frac = vocab_offgraph(V, lp.wm, iz, o)
    assert np.array_equal(anyx[0], cross)
    assert frac[0, north] == 0.0                        # all on the graph
    # east: 800 u from x 336, sampled every 16 u from 48 u on (48 samples);
    # only the wall (x 640..704: 4 samples) is off-graph
    assert frac[0, east] == pytest.approx(4.0 / 48.0)
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    assert np.all(frac[0][~cross] == 0.0) and np.all(frac[0][cross] > 0.0)


def test_visit_grid_counts_entries_and_resets():
    P = _walled()
    wm = WalkMap(P)
    vg = VisitGrid(2, wm)
    p = np.array([[100.0, 100.0, 68.0], [300.0, 500.0, 68.0]])
    vg.update(p)
    vg.update(p)                                        # same cells: no entry
    cy, cx = wm.cells(p)
    assert vg.patch([0, 1], cy, cx, cap=4)[:, 16, 16].tolist() == [0.25, 0.25]
    q = p.copy()
    q[0, 0] += 64.0                                     # env 0 moves a cell
    vg.update(q)
    vg.update(p)                                        # and back: 2nd entry
    assert vg.patch([0], cy[:1], cx[:1], cap=4)[0, 16, 16] == 0.5
    # an episode end clears the env's grid; the spawn is the first visit
    vg.update(p, ended=np.array([True, False]))
    assert vg.patch([0], cy[:1], cx[:1], cap=4)[0, 16, 16] == 0.25
    assert vg.cnt[0].sum() == 1


def _lp(n=4, **cfg):
    """An open 64 x 64-cell room (2,048 u), finish box top-right."""
    P = BFSPlanner(_room(ny=64, nx=64), MINS, CELL,
                   finish_box={"mins": [58 * CELL, 58 * CELL, 32.0],
                               "maxs": [62 * CELL, 62 * CELL, 96.0]},
                   n_targets=0)
    return LearnedPlanner(P, n, "cpu", tick_ms=10.0, act_every=4,
                          cfg=cfg, seed=3)


def _plan_all(lp, pos):
    n = len(pos)
    return lp.plan(pos, np.zeros((n, 3)), np.zeros(n))


def test_macro_steps_close_on_completion_budget_and_episode_end():
    lp = _lp(n=4, plan_novelty=0.5)
    # env 0's plan ends within 800 u of (400, 400); the others' end cells
    # are their own positions, far from it and from each other
    pos = np.array([[400.0, 400.0, 68.0], [1500.0, 400.0, 68.0],
                    [1800.0, 1800.0, 68.0], [1800.0, 1400.0, 68.0]])
    idx, lines, fresh = _plan_all(lp, pos)
    assert idx.tolist() == [0, 1, 2, 3] and fresh.all()
    for i, ln in enumerate(lines):
        assert np.allclose(ln[0], pos[i], atol=1e-3)    # anchored at the agent
    assert not lp.st.need.any() and lp.st.active.all()
    budget = lp.st.budget_ticks
    assert budget == 480                  # 800 u / 250 u/s x 1.5 at 100 Hz
    n = 4
    no = np.zeros(n, bool)
    # env 0 walks ITS line at 8 u/tick; the others stand still
    ln0 = lines[0].astype(np.float64)
    seg = np.linalg.norm(np.diff(ln0, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    cur = pos.copy()
    t_done = None
    for t in range(1, 200):
        s = min(cum[-1], 8.0 * t)
        cur[0] = [np.interp(s, cum, ln0[:, a]) for a in range(3)]
        lp.on_tick(cur, no, no, no)
        if not lp.st.active[0]:
            t_done = t
            break
    assert t_done is not None
    # it closed at 90% of the line's arc, completed, before its budget
    arc_frac = (8.0 * t_done) / cum[-1]
    assert COMPLETE_FRAC - 0.05 <= arc_frac <= COMPLETE_FRAC + 0.05
    assert len(lp.buf[0]) == 1
    tr = lp.buf[0][0]
    assert tr[6] is False                               # not terminal
    assert tr[5] == pytest.approx(R_EXEC_OK + 0.5)      # + first-visit novelty
    assert lp.st.need[0] and not lp.st.active[0]
    # env 2's episode ENDS by death at the next tick: -0.3, no novelty,
    # terminal; env 3's ends in the finish box: -0.3 + 10, terminal
    ended = np.array([False, False, True, True])
    fin = np.array([False, False, False, True])
    died = ended & ~fin
    term = cur.copy()
    lp.on_tick(cur, ended, fin, died, term_pos=term)
    tr2, tr3 = lp.buf[2][0], lp.buf[3][0]
    assert tr2[6] and tr2[5] == pytest.approx(R_EXEC_FAIL)
    # (env 3's terminal position is a fresh cell: +0.5 novelty too)
    assert tr3[6] and tr3[5] == pytest.approx(R_EXEC_FAIL + 10.0 + 0.5)
    # env 1 stands still until its budget runs out: -0.3 + novelty (its end
    # cell is new too)
    for t in range(budget):
        lp.on_tick(cur, no, no, no)
        if not lp.st.active[1]:
            break
    assert not lp.st.active[1] and len(lp.buf[1]) == 1
    tr1 = lp.buf[1][0]
    assert tr1[6] is False and tr1[5] == pytest.approx(R_EXEC_FAIL + 0.5)
    assert lp.st.elapsed[1] == budget
    w = lp.pop_window()
    assert w["closed"] == 4 and w["complete"] == pytest.approx(0.25)
    assert w["ep"] == 2 and w["finish"] == pytest.approx(0.5)
    # the next decision boundary re-plans exactly the four waiting envs
    idx2, _, fresh2 = _plan_all(lp, cur)
    assert idx2.tolist() == [0, 1, 2, 3]
    assert fresh2.tolist() == [False, False, False, False]


def test_novelty_counts_are_global_and_batch_safe():
    lp = _lp(n=3, plan_novelty=0.5)
    pos = np.array([[600.0, 600.0, 68.0]] * 3)
    _plan_all(lp, pos)
    ended = np.array([True, True, True])
    trunc_only = np.zeros(3, bool)
    # three truncations (alive) ending in ONE cell on one tick: 1st, 2nd,
    # 3rd visit of that cell
    lp.on_tick(pos, ended, trunc_only, trunc_only, term_pos=pos)
    r = sorted(b[0][5] for b in lp.buf)
    want = sorted(R_EXEC_FAIL + 0.5 / math.sqrt(k) for k in (1, 2, 3))
    assert np.allclose(r, want)
    cx, cy, cz = lp._cells(pos[:1])
    assert lp.nov_count[cx[0], cy[0], cz[0]] == 3


def test_an_episode_end_while_waiting_marks_the_last_plan_terminal():
    lp = _lp(n=1, plan_novelty=0.0)
    pos = np.array([[600.0, 600.0, 68.0]])
    _, lines, _ = _plan_all(lp, pos)
    no = np.zeros(1, bool)
    end = lines[0][-1].astype(np.float64)[None, :]
    lp.on_tick(end, no, no, no)                        # completes at once
    assert len(lp.buf[0]) == 1 and lp.buf[0][0][6] is False
    yes = np.ones(1, bool)
    lp.on_tick(end, yes, yes, no, term_pos=end)        # finishes, waiting
    tr = lp.buf[0][0]
    assert tr[6] is True and tr[5] == pytest.approx(R_EXEC_OK + 10.0)
    assert len(lp.buf[0]) == 1
    # the NEXT episode ends before its first plan (a spawn that finishes in
    # the same decision): the previous episode's terminal plan is untouched
    lp.request([0], end)
    lp.on_tick(end, yes, yes, no, term_pos=end)
    assert len(lp.buf[0]) == 1
    assert lp.buf[0][0][5] == pytest.approx(R_EXEC_OK + 10.0)


def test_plan_gae_hand_case():
    # two plans, not terminal, then a terminal one; bootstrap 2.0 unused
    r, v, d = [1.0, 0.0, 5.0], [0.5, 1.0, 2.0], [False, False, True]
    g, lam = 0.95, 0.95
    adv, ret = plan_gae(r, v, d, 2.0, g, lam)
    d2 = 5.0 - 2.0
    d1 = 0.0 + g * 2.0 - 1.0
    d0 = 1.0 + g * 1.0 - 0.5
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    assert np.allclose(adv, [a0, a1, a2]) and np.allclose(ret, adv + v)
    # not terminal: the open plan's value is bootstrapped
    adv2, _ = plan_gae([1.0], [0.0], [False], 3.0, g, lam)
    assert np.allclose(adv2, [1.0 + g * 3.0])
    # a terminal in the middle cuts the chain
    adv3, _ = plan_gae([0.0, 1.0], [0.0, 0.0], [True, False], 10.0, g, lam)
    assert np.allclose(adv3, [0.0, 1.0 + g * 10.0])


def test_update_is_finite_and_moves_only_the_planner():
    lp = _lp(n=8, plan_batch=16, plan_epochs=2)
    rng = np.random.default_rng(0)
    pos = rng.uniform(200, 1000, (8, 3))
    pos[:, 2] = 68.0
    no = np.zeros(8, bool)
    before = {k: v.clone() for k, v in lp.net.state_dict().items()}
    for _ in range(3):
        _plan_all(lp, pos)
        for _ in range(lp.st.budget_ticks):
            pos[:, :2] += rng.normal(0, 4, (8, 2))
            lp.on_tick(pos, no, no, no)
    assert lp.n_ready() >= 16
    st = lp.update()
    assert st is not None and st["n"] >= 16
    for k in ("loss_pi", "loss_v", "entropy", "kl"):
        assert np.isfinite(st[k]), k
    assert 0.0 < st["entropy"] <= math.log(lp.K) + 1e-6
    assert any(not torch.equal(before[k], v)
               for k, v in lp.net.state_dict().items())
    assert lp.n_ready() == 0 and lp.updates == 1
    txt, row = lp.note_and_row()
    assert "PLAN chosen" in txt and "upd 1" in txt and len(row) == 18
    # the state round trip: the same logits from a fresh planner
    sd = lp.state_dict_all()
    lp2 = _lp(n=8)
    lp2.load_state_dict_all(sd)
    img = torch.rand(3, 2, 32, 32)
    sc = torch.rand(3, 9)
    assert torch.equal(lp.net(img, sc)[0], lp2.net(img, sc)[0])
    assert lp2.updates == 1 and np.array_equal(lp2.nov_count, lp.nov_count)
    net, voc, spec = planner_from_state(sd)
    assert torch.equal(net(img, sc)[0], lp.net(img, sc)[0]) and voc.K == 80


class _FakeCore:
    def __init__(self, n=1):
        from surfgym.core import STATE_DTYPE
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)


def test_eval_hooks_plan_greedily_and_count_the_box():
    from surfgym.goals import MultiLine
    lp = _lp(n=2)
    core = _FakeCore()
    core.states_view["origin"][0] = [600.0, 600.0, 68.0]
    ev = {}
    line = MultiLine(1, device="cpu")
    meta, tick = lp.eval_hooks(core, ev, line=line)
    h = meta(0)
    assert h["plan"]["planner"] == "learned" and ev["plans"] == 1
    k = h["plan"]["shape"]
    # greedy: the argmax of the network at that state (the spawn cell is the
    # episode's first visit)
    from surfgym.goallearn import build_obs
    p0 = core.states_view["origin"][0:1].astype(np.float64)
    vg = VisitGrid(1, lp.wm)
    vg.update(p0)
    img, sc = build_obs(lp.wm, vg, np.zeros(1, np.int64), p0,
                        np.zeros((1, 3)), np.zeros(1), lp.finish)
    with torch.no_grad():
        lg, _ = lp.net(torch.as_tensor(img), torch.as_tensor(sc))
    assert int(lg.argmax(-1)[0]) == k
    assert np.allclose(line.pts[0, 0].numpy(), [600.0, 600.0, 68.0])
    assert np.allclose(line.pts[0, int(line.length[0]) - 1].numpy(),
                       lp.vocab.raw[k, -1] + p0[0], atol=1e-2)
    zero = np.zeros(1, np.uint8)
    one = np.ones(1, np.uint8)
    # stand still to the budget: the plan closes, and a new one is chosen
    # at the next decision boundary (t + 1) % 4 == 0
    t = 0
    for t in range(lp.st.budget_ticks + 8):
        tick(t, None, None, zero, zero)
        if ev["plans"] == 2:
            break
    assert ev["plans"] == 2 and ev["closed"] == 1 and ev["complete"] == 0
    assert (t + 1) % 4 == 0
    core.goal_hits[0] = 1
    tick(t + 1, None, None, one, zero)
    assert ev["succ"] == 1 and ev["n"] == 1
    assert isinstance(k, int)


# ==========================================================================
# (b) the trainer
# ==========================================================================
def _env():
    # -1, not "": on Windows an EMPTY value UNSETS the variable and the GPU
    # stays visible (memory: windows-empty-env-var-unsets)
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _run(cmd, timeout=2400):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(ROOT), timeout=timeout, encoding="utf-8",
                          errors="replace")


# the stage-1 test set (tests/python/test_goal_planner.py): 64 envs, 16x8
# lidar, 64-wide nets, 0.96 s episodes for the flag-off comparison
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
NEW_KEYS = ("freeze_policy", "plan_lr", "plan_ent", "plan_batch",
            "plan_epochs", "plan_novelty", "plan_progress",
            "plan_finish_bonus")


def _train(run, extra, script=TRAIN, steps="12288", runs_dir=None):
    shutil.rmtree((runs_dir or ROOT / "runs") / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + LAB_FLAGS
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _cfg(d: Path) -> dict:
    return json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]


def _rows(d: Path):
    rows = (d / "progress.csv").read_text(encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


def _base_tree(dst: Path):
    """The python/ tree of the last first-parent commit WITHOUT the learned
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
        if r.returncode != 0 or b"--freeze-policy" in r.stdout:
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
    assert ca == cb
    ra, rb = _rows(a), _rows(b)
    assert len(ra) == len(rb) >= 3
    assert list(ra[0]) == list(rb[0])
    assert not any(k.startswith("plan/") for k in ra[0])
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
    _same_policy_and_adam(_ckpt(a / "ckpt_final.pt"),
                          _ckpt(b / "ckpt_final.pt"))
    assert "planner" not in _ckpt(a / "ckpt_final.pt")


@needs_lab
@pytest.mark.parametrize("mode", ["race", "goals", "bfs"])
def test_flag_off_is_bit_identical_to_the_base_commit(mode, tmp_path):
    """None of the new flags: this branch's trainer against the base
    commit's whole python/ tree (config, progress.csv minus fps, the eval
    trajectories, goals.csv / plan.csv, weights, Adam moments). 'bfs' is the
    stage-1 planner path, which goalsys.py's new branches sit beside."""
    ref = _base_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-learned-planner python/ tree in the history")
    old = tmp_path / "python" / "train_fast.py"
    extra = MODERN + {"race": [],
                      "goals": ["--goals", "1", "--goal-obs", "fan"],
                      "bfs": PLAN + ["--goal-reward", "arc"]}[mode]
    new_run = ROOT / "runs" / f"plrn_ctl_new_{mode}"
    old_run = tmp_path / "runs" / f"plrn_ctl_old_{mode}"
    _train(new_run.name, extra)
    _train(old_run.name, extra, script=old, runs_dir=tmp_path / "runs")
    assert old_run.exists(), "the base trainer writes under its own tree"
    _assert_identical(new_run, old_run)
    shutil.rmtree(new_run, ignore_errors=True)


SMOKE_SRC_STEPS = 8192
SMOKE_STEPS = 81920


@needs_lab
def test_learned_planner_smoke_freeze_and_record_gate():
    """A tiny stage-1 bfs executor, then a warm resume with --goal-planner
    learned --freeze-policy 1: the planner updates with finite losses and
    entropy, the executor and its Adam state stay bit-identical to the
    source checkpoint, the checkpoint carries the planner, and the
    launcher's record gate passes on it."""
    src = ROOT / "runs" / "plrn_src_t"
    # the source carries lab200 as a HELD-OUT map, like plLAB100a: its
    # config stores the STEM, which a bare resume must resolve under
    # maps_pool/ (it died on 'no such BSP' before), and the learned
    # planner's held-out eval plans on lab200's own graph
    _train(src.name, MODERN + PLAN + ["--goal-reward", "arc",
                                      "--heldout-maps", str(LAB200),
                                      "--heldout-goal-cell", "32"],
           steps=str(SMOKE_SRC_STEPS))
    assert _cfg(src)["heldout_maps"] == ["labyrinth_left200"]
    run = "plrn_smoke_t"
    d = ROOT / "runs" / run
    shutil.rmtree(d, ignore_errors=True)
    # a BARE resume of the goal plumbing (goals / fan / arc / the view and
    # keys modes are restored from the checkpoint); only the planner mode
    # is overridden
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt",
              str(src / "ckpt_final.pt")] + LAB_FLAGS
             + ["--steps", str(SMOKE_SRC_STEPS + SMOKE_STEPS),
                "--goal-planner", "learned", "--freeze-policy", "1",
                "--ep-ticks", "1200", "--plan-batch", "64",
                "--record-every", "32768"])
    out = r.stdout
    assert r.returncode == 0, out[-5000:] + r.stderr[-4000:]
    assert "planner: FRESH" in out and "planner LEARNED:" in out
    assert "goals: LEARNED PLANNER" in out
    assert "plan-eval finish" in out and "learned planner greedy" in out
    assert "greedy[HELDOUT labyrinth_left200]" in out
    ht = sorted(d.glob("traj_*_labyrinth_left200.jsonl"))
    assert ht
    h0 = json.loads(ht[-1].read_text().splitlines()[0])
    assert h0["map"] == "labyrinth_left200"
    assert h0["plan"]["planner"] == "learned"
    assert h0["plan"]["graph_dist"] > 5000.0         # lab200's own graph
    cfg = _cfg(d)
    assert cfg["goal_planner"] == "learned" and cfg["freeze_policy"] == 1
    assert cfg["goal_reward"] == "arc" and cfg["goals"] == 1
    assert cfg["plan_batch"] == 64 and cfg["plan_novelty"] == 0.5
    rows = _rows(d)
    assert rows and all(k in rows[0] for k in
                        ("plan/closed", "plan/wall", "plan/updates"))
    upd = [x for x in rows if x["plan/loss_pi"] != ""]
    assert upd, "the planner never updated"
    for x in upd:
        assert np.isfinite(float(x["plan/loss_pi"]))
        assert np.isfinite(float(x["plan/loss_v"]))
    ent = [float(x["plan/entropy"]) for x in rows if x["plan/entropy"]]
    assert ent and all(0.0 < e <= math.log(80) + 1e-6 for e in ent)
    assert int(rows[-1]["plan/updates"]) >= 1
    closed = sum(int(x["plan/closed"]) for x in rows)
    assert closed >= 64
    walls = [float(x["plan/wall"]) for x in rows if x["plan/wall"]]
    assert walls and all(0.0 <= w <= 1.0 for w in walls)
    wlen = [float(x["plan/wall_len"]) for x in rows if x["plan/wall_len"]]
    assert wlen and all(0.0 <= w <= 1.0 for w in wlen)
    # the executor is FROZEN: policy + Adam bit-identical to the source
    s_src, s_new = _ckpt(src / "ckpt_final.pt"), _ckpt(d / "ckpt_final.pt")
    _same_policy_and_adam(s_src, s_new)
    assert "planner" not in s_src and "planner" in s_new
    assert s_new["planner"]["updates"] >= 1
    assert s_new["global_step"] == SMOKE_SRC_STEPS + SMOKE_STEPS
    # the launcher's record gate on the learned checkpoint (greedy, stoch,
    # mixed), the dashboard buttons' backend
    g = _run([sys.executable, str(ROOT / "tools" / "record_gate.py"), run,
              "--ckpt", str(d / "ckpt_final.pt"), "--no-pov"])
    assert g.returncode == 0, g.stdout[-3000:] + g.stderr[-3000:]
    assert "record gate PASSED: 3 recording(s)" in g.stdout
    # the smoke's numbers, for the report
    last = rows[-1]
    print("\nSMOKE", {k: last[k] for k in last if k.startswith("plan/")})
    print("SMOKE upd rows", [(x["plan/loss_pi"], x["plan/loss_v"],
                              x["plan/entropy"], x["plan/complete"],
                              x["plan/wall"], x["plan/wall_base"],
                              x["plan/wall_len"], x["plan/wall_len_base"])
                             for x in upd])
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(src, ignore_errors=True)


@needs_lab
def test_refusals():
    base = [sys.executable, "-u", str(TRAIN), "--run", "plrn_bad"] \
        + LAB_FLAGS + ["--steps", "2048"] + MODERN
    cases = [
        (["--goals", "1", "--goal-planner", "learned"],
         "--goal-planner learned drives a TRAINED executor"),
        (["--goals", "1", "--goal-planner", "bfs", "--plan-lr", "1e-3"],
         "--plan-lr without --goal-planner learned"),
        (["--freeze-policy", "1"], "--freeze-policy freezes a TRAINED"),
    ]
    for extra, msg in cases:
        r = _run(base + extra)
        assert r.returncode != 0, extra
        assert msg in r.stdout + r.stderr, (extra, (r.stdout + r.stderr)[-1500:])
    shutil.rmtree(ROOT / "runs" / "plrn_bad", ignore_errors=True)
