"""--plan-vocab surf and --goal-planner vocab: the surf vocabulary, the
occupancy-slab observation, the executor's plan diet (surfgym/goalsurf.py),
their wiring into the learned planner (surfgym/goallearn.py), the goal
system, the trainer and the recorder.

(a) CPU only, no DLL: the 144 speed-scaled 3-D shapes (headings, turns,
    descents, the length clamp, anchored at the agent, the direct resample
    equal to the generic one); the slabs on a hand-made grid (exact solid
    fractions, outside = solid, the visit channel on the same lattice); the
    3-D solid-crossing and void diagnostics on a hand-made wall; the diet
    mixes uniform shapes and hindsight segments at the requested rate, and a
    hindsight line IS the policy's own segment (the spawn row's own exactly,
    a matched pool row's displacement anchored at the agent, a miss falls
    back to a shape); plans close on completion, budget and episode end, and
    completion is counted by source; the learned planner under the surf spec
    (speed-scaled shapes, 4-channel input, a finite PPO update, the state
    round trip); the eval hooks of both.
(b) the built core + maps: the slabs on surf_edgeflow_blue050 against a
    brute-force voxel count; with none of the new flags the trainer is
    bit-identical to the base commit (plain race, --goals, --goal-planner
    bfs, and a --goal-planner learned warm resume) on labyrinth_left100; the
    CPU smoke on surf_edgeflow_blue050 - a from-scratch --goal-planner vocab
    executor, then a warm resume with --goal-planner learned
    --freeze-policy 1 - and tools/record_gate.py on both checkpoints; the
    refusals.
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

from surfgym.goalplan import BFSPlanner                          # noqa: E402
from surfgym.goallearn import (LearnedPlanner, VisitGrid,        # noqa: E402
                               planner_from_state)
from surfgym.goalsurf import (DIET_COLS, SlabMap, SurfVocab,     # noqa: E402
                              VocabDiet, build_obs_surf, line_crossing,
                              make_vocab_hooks, surf_spec,
                              surf_vocab_crossing)

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


# ==========================================================================
# (a) CPU only
# ==========================================================================
def test_surf_vocabulary_is_144_speed_scaled_3d_shapes():
    V = SurfVocab()
    assert V.K == 144 and V.unit.shape == (144, 9, 3)
    seg = np.linalg.norm(np.diff(V.unit, axis=1), axis=2)
    assert np.allclose(seg, 1.0 / 8.0) and np.allclose(seg.sum(1), 1.0)
    assert np.all(V.unit[:, 0] == 0.0)
    assert sorted(set(np.round(V.heading_deg, 6))) == \
        sorted(set(np.round(np.arange(16) * 22.5, 6)))
    assert sorted(set(V.turn_deg)) == [-45.0, 0.0, 45.0]
    assert sorted(set(V.pitch_deg)) == [-40.0, -20.0, 0.0]
    # the index order: k = (heading * 3 + turn) * 3 + pitch
    assert list(V.turn_deg[:9]) == [0.0] * 3 + [45.0] * 3 + [-45.0] * 3
    assert list(V.pitch_deg[:3]) == [0.0, -20.0, -40.0]
    for k in range(V.K):
        d = np.diff(V.unit[k], axis=0)
        ph = math.radians(V.pitch_deg[k])
        # a constant descent: every segment at the shape's pitch
        assert np.allclose(d[:, 2], math.sin(ph) / 8.0)
        assert np.allclose(np.hypot(d[:, 0], d[:, 1]), math.cos(ph) / 8.0)
        h = math.radians(V.heading_deg[k])
        th = math.radians(V.turn_deg[k])
        a = np.arctan2(d[:, 1], d[:, 0])
        want = h + th * (np.arange(8) + 0.5) / 8.0
        assert np.allclose(np.cos(a), np.cos(want), atol=1e-9)
        assert np.allclose(np.sin(a), np.sin(want), atol=1e-9)
        # a LEFT turn ends left of its start heading
        e = V.unit[k, -1, :2]
        cross = math.cos(h) * e[1] - math.sin(h) * e[0]
        assert np.sign(round(cross, 9)) == np.sign(V.turn_deg[k])
    # length = clamp(3 s x max(|v_xy|, 500 u/s), 800, 6000)
    assert np.allclose(V.length_for([0.0, 250.0, 500.0, 1000.0, 1999.0,
                                     2000.0, 3500.0]),
                       [1500.0, 1500.0, 1500.0, 3000.0, 5997.0, 6000.0,
                        6000.0])
    V2 = SurfVocab(t_plan=1.0)
    assert np.allclose(V2.length_for([0.0, 700.0]), [800.0, 800.0])
    assert V.budget_secs == pytest.approx(4.5)
    assert V.n_line == 48                     # 6000 u at 128 u
    # the ends are the scaled unit ends
    o = np.array([[10.0, -20.0, 600.0]])
    e = V.ends(o, [3000.0])[0]
    assert np.allclose(e, o + 3000.0 * V.unit[:, -1])
    # a -40 deg shape of 3000 u drops 3000 sin 40 = 1928 u
    k40 = int(np.flatnonzero((V.pitch_deg == -40.0)
                             & (V.turn_deg == 0.0))[0])
    assert e[k40, 2] - 600.0 == pytest.approx(-3000.0 * math.sin(
        math.radians(40.0)))


def test_surf_anchor_is_the_generic_resample_and_starts_at_the_agent():
    from surfgym.route import resample_polyline
    V = SurfVocab()
    rng = np.random.default_rng(0)
    for k in (0, 7, 50, 143):
        for L in (800.0, 1500.0, 2345.6, 6000.0):
            o = rng.uniform(-3000.0, 3000.0, 3)
            ln = V.anchor(k, o, L)
            ref = resample_polyline(o[None, :] + L * V.unit[k],
                                    V.spacing)[0]
            assert ln.dtype == np.float32 and ln.shape == ref.shape
            assert np.array_equal(ln, ref)
            assert np.allclose(ln[0], o, atol=1e-3)
            arc = np.linalg.norm(np.diff(ln.astype(np.float64), axis=0),
                                 axis=1).sum()
            # the resample cuts a turn's chords a little, never lengthens
            assert L * 0.98 <= arc <= L + 1e-2
            assert len(ln) == max(2, int(round(L / 128.0)) + 1)


def _grid():
    """20 x 20 x 20 voxels of 32 u (640 u cube): a solid floor (iz 0-1), a
    wall at ix 12-13 up to iz 9, everything else free."""
    s = np.zeros((20, 20, 20), bool)
    s[0:2] = True
    s[:10, :, 12:14] = True
    return s


def test_slabs_on_a_hand_made_grid_are_exact_solid_fractions():
    s = _grid()
    sm = SlabMap(s, MINS, CELL, patch_cell_u=64.0, n=8,
                 offsets=(-128.0, 128.0), slab_u=256.0)
    assert sm.s == 2 and sm.pc == 64.0 and sm.tz == 8
    assert (sm.gy, sm.gx) == (10, 10)
    p = np.array([[5.0 * 64 + 32.0, 5.0 * 64 + 32.0, 200.0]])   # cell (5,5)
    cy, cx = sm.cells(p)
    assert (int(cy[0]), int(cx[0])) == (5, 5)
    sl = sm.slabs(p, cy, cx)[0]
    assert sl.shape == (2, 8, 8)

    def brute(z0, rows, cols):
        # the solid fraction of layers z0 .. z0 + 7 over 2 x 2 columns,
        # anything outside the grid solid
        out = np.zeros((len(rows), len(cols)))
        for a, r in enumerate(rows):
            for b, c in enumerate(cols):
                tot = 0
                for iz in range(z0, z0 + 8):
                    for iy in (2 * r, 2 * r + 1):
                        for ix in (2 * c, 2 * c + 1):
                            ok = (0 <= iz < 20 and 0 <= iy < 20
                                  and 0 <= ix < 20)
                            tot += 1 if not ok else int(s[iz, iy, ix])
                out[a, b] = tot / 32.0
        return out
    rows = np.arange(5 - 4, 5 + 4)
    cols = np.arange(5 - 4, 5 + 4)
    # slab 0: [200 - 128 - 128, 200 - 128 + 128) = [-56, 200) -> iz -2 .. 5
    assert np.allclose(sl[0], brute(-2, rows, cols))
    # slab 1: [200, 456) -> iz 6 .. 13
    assert np.allclose(sl[1], brute(6, rows, cols))
    # the wall column (patch col 6 = ix 12-13) is solid up to iz 9: the
    # upper slab holds 4 of its 8 layers
    assert sl[1, 4, 4 + 1] == pytest.approx(0.5)
    # the floor: 2 real layers + 2 below the grid of the lower slab
    assert sl[0, 4, 4] == pytest.approx(4.0 / 8.0)
    # outside the map (window rows off the grid) reads solid
    q = np.array([[32.0, 32.0, 200.0]])
    cy, cx = sm.cells(q)
    assert np.all(sm.slabs(q, cy, cx)[0][:, 0, :] == 1.0)
    # a point far above the grid: both slabs entirely outside -> solid
    r = np.array([[320.0, 320.0, 5000.0]])
    cy, cx = sm.cells(r)
    assert np.all(sm.slabs(r, cy, cx) == 1.0)
    # solid_at: voxels, and outside = solid
    assert sm.solid_at(np.array([[400.0, 100.0, 100.0]]))[0]      # wall
    assert not sm.solid_at(np.array([[100.0, 100.0, 100.0]]))[0]  # air
    assert sm.solid_at(np.array([[-1.0, 100.0, 100.0]]))[0]       # outside
    # the visit channel rides the same lattice
    vg = VisitGrid(1, sm)
    vg.update(p)
    img, sc = build_obs_surf(sm, vg, np.zeros(1, np.int64), p,
                             np.array([[300.0, 0.0, 0.0]]), np.zeros(1),
                             np.array([600.0, 320.0, 100.0]))
    assert img.shape == (1, 3, 8, 8) and sc.shape == (1, 9)
    assert img[0, 2, 4, 4] == pytest.approx(0.25)
    assert img[0, 2].sum() == pytest.approx(0.25)
    assert np.allclose(img[0, :2], sl)
    # the scalars are the walking planner's: finish direction, log dist,
    # velocity / 1000, sin / cos yaw
    g = np.array([600.0, 320.0, 100.0]) - p[0]
    assert np.allclose(sc[0, :3], g / np.linalg.norm(g), atol=1e-6)
    assert sc[0, 4] == pytest.approx(0.3) and sc[0, 8] == pytest.approx(1.0)


def test_surf_crossing_and_void_on_a_hand_made_wall():
    s = _grid()
    sm = SlabMap(s, MINS, CELL, n=8)
    V = SurfVocab()
    o = np.array([[5.5 * 32, 10.5 * 32, 7.5 * 32]])     # x 176, z 240
    L = np.array([800.0])
    east = int(np.flatnonzero((V.heading_deg == 0.0) & (V.turn_deg == 0.0)
                              & (V.pitch_deg == 0.0))[0])
    north = int(np.flatnonzero((V.heading_deg == 90.0) & (V.turn_deg == 0.0)
                               & (V.pitch_deg == 0.0))[0])
    cx, fr, vd = surf_vocab_crossing(V, sm, o, L, shapes=[east],
                                     kill_z=100.0)
    assert cx[0] and 0.0 < fr[0] < 1.0 and not vd[0]  # through the wall
    cx, fr, vd = surf_vocab_crossing(V, sm, o, L, shapes=[north],
                                     kill_z=100.0)
    # north runs through free air up to y 640, then leaves the grid
    # (outside = solid): some crossing, never the first 64 u
    assert cx[0]
    # the whole-vocabulary form agrees with the per-shape form
    ax, af, av = surf_vocab_crossing(V, sm, o, L, kill_z=100.0)
    assert ax.shape == (1, 144)
    for k in (east, north, 5, 100):
        c1, f1, v1 = surf_vocab_crossing(V, sm, o, L, shapes=[k],
                                         kill_z=100.0)
        assert (ax[0, k], af[0, k], av[0, k]) == (c1[0], f1[0], v1[0])
    # void: a shape's END below the kill plane (240 + 800 sin(-40) = -274)
    down = int(np.flatnonzero((V.pitch_deg == -40.0) & (V.turn_deg == 0.0)
                              & (V.heading_deg == 90.0))[0])
    assert av[0, down] and not av[0, north]
    # line_crossing on the anchored line agrees with the shape form
    ln = V.anchor(east, o[0], 800.0)
    c2, f2, v2 = line_crossing(sm, ln, kill_z=100.0)
    assert c2 and not v2 and f2 == pytest.approx(
        surf_vocab_crossing(V, sm, o, L, shapes=[east])[1][0], abs=0.05)


def _toy_graph(n=48):
    """An open box of n x n x 12 voxels at 32 u with a floor, a finish box
    at the far corner (the planner needs only the graph, the solid grid,
    the finish and the kill ceiling)."""
    occ = np.ones((12, n, n), np.uint8)
    occ[1:11, 1:n - 1, 1:n - 1] = 0
    box = {"mins": [(n - 6) * CELL, (n - 6) * CELL, 32.0],
           "maxs": [(n - 2) * CELL, (n - 2) * CELL, 96.0]}
    return BFSPlanner(occ, MINS, CELL, finish_box=box, n_targets=0)


def _pool_rows(P, n, rng, speed=600.0):
    from surfgym.core import STATE_DTYPE
    pool = np.zeros(n, STATE_DTYPE)
    pool["origin"] = np.column_stack([rng.uniform(300, 1200, n),
                                      rng.uniform(300, 1200, n),
                                      np.full(n, 68.0)])
    ang = rng.uniform(0, 2 * np.pi, n)
    pool["velocity"] = np.column_stack([speed * np.cos(ang),
                                        speed * np.sin(ang), np.zeros(n)])
    segs = np.zeros((n, 64, 3), np.float32)
    seglen = rng.integers(5, 13, n).astype(np.int32)
    t = np.arange(64)[:, None] * 0.25
    for i in range(n):
        segs[i] = pool["origin"][i] + pool["velocity"][i] * t
    goals = segs[np.arange(n), seglen - 1].copy()
    return pool, goals, segs, seglen


def test_diet_mixes_the_two_sources_at_the_requested_rate():
    P = _toy_graph()
    rng = np.random.default_rng(1)
    for h in (0.0, 0.3, 0.5, 1.0):
        n = 400
        d = VocabDiet(P, n, hindsight=h, seed=11, snap_secs=0.25)
        pool, goals, segs, seglen = _pool_rows(P, n, rng)
        d.set_bank(pool, goals, segs, seglen)
        assert d.bank_n == n
        pos = pool["origin"].astype(np.float64)
        vel = pool["velocity"].astype(np.float64)
        tot = hs = 0
        for rep in range(5):
            # mid-episode re-plans at the bank's own states: a hindsight
            # request always finds its row (distance 0), so the realised
            # share IS the coin's
            d.st.need[:] = True
            d.fresh[:] = False
            idx, lines, fresh = d.plan(pos, vel, np.zeros(n))
            assert len(idx) == n and not fresh.any()
            tot += n
            hs += int(d.src[idx].sum())
        w = d.pop_window()
        assert w["hs_miss"] == 0.0 or h == 0.0
        share = hs / tot
        sd = math.sqrt(max(h * (1 - h), 1e-9) / tot)
        assert abs(share - h) <= 4.0 * sd + 1e-9, (h, share)
        assert w["hs_share"] == pytest.approx(share)
        assert w["hs_req"] == pytest.approx(share)


def test_hindsight_lines_are_the_policys_own_segments():
    from surfgym.goals import resample_polyline_np
    P = _toy_graph()
    rng = np.random.default_rng(2)
    n = 6
    d = VocabDiet(P, n, hindsight=1.0, seed=3, snap_secs=0.25)
    pool, goals, segs, seglen = _pool_rows(P, 50, rng)
    d.set_bank(pool, goals, segs, seglen)
    # env 0: a fresh spawn ON pool row 7 with its own segment -> exactly it
    # env 1: a fresh spawn with no own segment, at row 9's state -> row 9's
    #        displacement anchored at the agent (NN, distance 0)
    # env 2: mid-episode, 60 u and 50 u/s off row 12 -> row 12's
    #        displacement anchored at the agent (60^2 + 50^2 < 192^2)
    # env 3: mid-episode, 400 u off every row -> a miss: a vocabulary shape
    # env 4: fresh, own segment given, but the coin (h = 1) -> own segment
    # env 5: mid-episode at row 20 with a 300 u/s velocity mismatch -> miss
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    for e, r in ((0, 7), (1, 9), (2, 12), (4, 30), (5, 20)):
        pos[e] = pool["origin"][r]
        vel[e] = pool["velocity"][r]
    pos[2] += [60.0, 0.0, 0.0]
    vel[2] += [0.0, 50.0, 0.0]
    pos[3] = [3000.0, 3000.0, 68.0]
    vel[5] += [300.0, 0.0, 0.0]
    own = [segs[7, :seglen[7]].astype(np.float64), None, None, None,
           segs[30, :seglen[30]].astype(np.float64), None]
    d.request(np.arange(n), pos, segs=own)
    d.fresh[[2, 3, 5]] = False
    idx, lines, _ = d.plan(pos, vel, np.zeros(n))
    assert idx.tolist() == list(range(n))
    assert d.src.tolist() == [1, 1, 1, 0, 1, 0]

    def want(r, p):
        s = segs[r, :seglen[r]].astype(np.float64)
        return resample_polyline_np(s - s[0] + p)
    assert np.array_equal(lines[0], resample_polyline_np(own[0]))
    assert np.allclose(lines[0][0], pool["origin"][7])
    assert np.array_equal(lines[1], want(9, pos[1]))
    assert np.array_equal(lines[2], want(12, pos[2]))
    assert np.allclose(lines[2][0], pos[2], atol=1e-3)
    assert np.array_equal(lines[4], resample_polyline_np(own[4]))
    # the misses are vocabulary shapes anchored at the agent, sized for the
    # current speed
    k3 = int(d.st.shape[3])
    assert 0 <= k3 < 144
    assert np.array_equal(lines[3], d.vocab.anchor(k3, pos[3], 1500.0))
    k5 = int(d.st.shape[5])
    L5 = d.vocab.length_for([np.hypot(vel[5, 0], vel[5, 1])])[0]
    assert np.array_equal(lines[5], d.vocab.anchor(k5, pos[5], L5))
    # hindsight rows carry no shape; budgets: 1.5 x the flight time
    assert d.st.shape[[0, 1, 2, 4]].tolist() == [-1] * 4
    for e, r in ((0, 7), (1, 9), (2, 12)):
        secs = (int(seglen[r]) - 1) * 0.25
        assert d.st.budget[e] == int(math.ceil(1.5 * secs * 100.0 - 1e-6))
    assert d.st.budget[3] == 450 and d.st.budget[5] == 450
    # a spawn's own segment is consumed by its first plan
    assert all(s is None for s in d.spawn_seg)
    w = d.pop_window()
    assert w["hs_req"] == 1.0 and w["hs_miss"] == pytest.approx(2.0 / 6.0)
    assert w["hs_share"] == pytest.approx(4.0 / 6.0)
    # no bank and no own segment: every hindsight request misses
    d2 = VocabDiet(P, 3, hindsight=1.0, seed=3)
    d2.request(np.arange(3), pos[:3])
    d2.plan(pos[:3], vel[:3], np.zeros(3))
    assert d2.src.tolist() == [0, 0, 0]
    assert d2.pop_window()["hs_miss"] == 1.0


def test_diet_closes_plans_and_counts_completion_by_source():
    P = _toy_graph()
    rng = np.random.default_rng(4)
    n = 4
    d = VocabDiet(P, n, hindsight=0.0, seed=5,
                  start_pts=np.array([[400.0, 400.0, 68.0]]))
    pos = np.array([[400.0, 400.0, 68.0], [800.0, 800.0, 68.0],
                    [900.0, 500.0, 68.0], [500.0, 900.0, 68.0]])
    d.request(np.arange(n), pos)
    idx, lines, fresh = d.plan(pos, np.zeros((n, 3)), np.zeros(n))
    assert fresh.all() and d.src.tolist() == [0] * 4
    assert d.st.budget_ticks == 450
    # env 0 flies ITS line at 20 u/tick: it completes (90% of the arc)
    ln = lines[0].astype(np.float64)
    cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
        np.diff(ln, axis=0), axis=1))))
    cur = pos.copy()
    no = np.zeros(n, bool)
    t_done = None
    for t in range(1, 200):
        s = min(cum[-1], 20.0 * t)
        cur[0] = [np.interp(s, cum, ln[:, a]) for a in range(3)]
        d.on_tick(cur, no, no, no)
        if not d.st.active[0]:
            t_done = t
            break
    assert t_done is not None
    assert 0.85 <= 20.0 * t_done / cum[-1] <= 0.95
    # env 1 dies, env 2 finishes (both close uncompleted); env 3 stands
    # still until its budget
    ended = np.array([False, True, True, False])
    fin = np.array([False, False, True, False])
    d.on_tick(cur, ended, fin, ended & ~fin)
    for _ in range(460):
        d.on_tick(cur, no, no, no)
        if not d.st.active[3]:
            break
    assert not d.st.active.any() and d.st.need.all()
    w = d.pop_window()
    assert w["closed"] == 4 and w["complete"] == pytest.approx(0.25)
    assert w["complete_rand"] == pytest.approx(0.25)
    assert w["complete_hs"] != w["complete_hs"]          # no hindsight: NaN
    assert w["ep"] == 2 and w["finish"] == pytest.approx(0.5)
    # a hindsight plan closed on completion counts on its own side
    d.h = 1.0
    pool, goals, segs, seglen = _pool_rows(P, 20, rng)
    d.set_bank(pool, goals, segs, seglen)
    p1 = pool["origin"][:1].astype(np.float64)
    v1 = pool["velocity"][:1].astype(np.float64)
    d.st.need[:] = False
    d.st.need[1] = True
    posb = cur.copy()
    posb[1] = p1[0]
    velb = np.zeros((n, 3))
    velb[1] = v1[0]
    idx, lines, _ = d.plan(posb, velb, np.zeros(n))
    assert idx.tolist() == [1] and d.src[1] == 1
    end = lines[0][-1].astype(np.float64)
    posb[1] = end
    d.on_tick(posb, no, no, no)
    w = d.pop_window()
    assert w["closed"] == 1 and w["complete_hs"] == 1.0
    txt, row = d.note_and_row()
    assert len(row) == len(DIET_COLS) and "DIET chosen" in txt


def test_learned_planner_under_the_surf_spec():
    P = _toy_graph()
    lp = LearnedPlanner(P, 6, "cpu", tick_ms=10.0, act_every=4,
                        cfg={"plan_batch": 8, "plan_epochs": 2}, seed=3,
                        spec=surf_spec())
    assert lp.surf and lp.K == 144 and lp.in_ch == 4
    assert lp.o_img.shape == (6, 4, 32, 32)
    assert lp.st.budget_ticks == 450
    rng = np.random.default_rng(0)
    pos = np.column_stack([rng.uniform(300, 1200, 6),
                           rng.uniform(300, 1200, 6), np.full(6, 68.0)])
    vel = np.zeros((6, 3))
    vel[:, 0] = [0.0, 400.0, 800.0, 1200.0, 2500.0, 100.0]
    idx, lines, fresh = lp.plan(pos, vel, np.zeros(6))
    assert idx.tolist() == list(range(6)) and fresh.all()
    L = lp.vocab.length_for(np.abs(vel[:, 0]))
    for i, ln in enumerate(lines):
        k = int(lp.st.shape[i])
        assert np.array_equal(ln, lp.vocab.anchor(k, pos[i], L[i]))
    assert [len(x) for x in lines] == [13, 13, 20, 29, 48, 13]
    no = np.zeros(6, bool)
    for _ in range(3):
        for _ in range(lp.st.budget_ticks):
            pos[:, :2] += rng.normal(0, 4, (6, 2))
            lp.on_tick(pos, no, no, no)
        lp.plan(pos, vel, np.zeros(6))
    st = lp.update()
    assert st is not None
    for k in ("loss_pi", "loss_v", "entropy", "kl"):
        assert np.isfinite(st[k]), k
    assert 0.0 < st["entropy"] <= math.log(144) + 1e-6
    txt, row = lp.note_and_row()
    from surfgym.goallearn import PLAN_COLS_SURF
    assert len(row) == len(PLAN_COLS_SURF) and "void" in txt
    sd = lp.state_dict_all()
    assert sd["spec"]["vocab"] == "surf"
    net, voc, spec = planner_from_state(sd)
    img = torch.rand(2, 4, 32, 32)
    sc = torch.rand(2, 9)
    assert voc.K == 144 and torch.equal(net(img, sc)[0], lp.net(img, sc)[0])
    lp2 = LearnedPlanner(P, 6, "cpu", spec=surf_spec())
    lp2.load_state_dict_all(sd)
    assert torch.equal(lp2.net(img, sc)[0], lp.net(img, sc)[0])
    # a walking planner refuses a surf planner's weights (and back)
    walk = LearnedPlanner(P, 6, "cpu")
    with pytest.raises(ValueError):
        walk.load_state_dict_all(sd)
    with pytest.raises(ValueError):
        lp2.load_state_dict_all(walk.state_dict_all())


class _FakeCore:
    def __init__(self, n=1):
        from surfgym.core import STATE_DTYPE
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)


def test_vocab_eval_picks_the_shape_ending_nearest_the_finish():
    from surfgym.goals import MultiLine
    P = _toy_graph()
    V = SurfVocab()
    core = _FakeCore()
    core.states_view["origin"][0] = [400.0, 400.0, 68.0]
    core.states_view["velocity"][0] = [900.0, 0.0, 0.0]
    ev = {}
    line = MultiLine(1, device="cpu")
    meta, tick = make_vocab_hooks(V, P, core, ev, line=line, act_every=4)
    h = meta(0)
    k = h["plan"]["shape"]
    p0 = np.array([400.0, 400.0, 68.0])
    ends = V.ends(p0[None], [2700.0])[0]
    assert k == int(np.argmin(np.linalg.norm(ends - P.finish_center[None],
                                             axis=1)))
    assert h["plan"]["planner"] == "vocab" and h["plan"]["length"] == 2700.0
    # the toy box is fully walkable, so the graph distance is finite here;
    # a surf map's is null (never Infinity) - json round trip either way
    assert json.loads(json.dumps(h))["plan"]["shape"] == k
    assert np.allclose(line.pts[0, 0].numpy(), p0)
    zero = np.zeros(1, np.uint8)
    one = np.ones(1, np.uint8)
    t = 0
    for t in range(460):
        tick(t, None, None, zero, zero)
        if ev["plans"] == 2:
            break
    assert ev["plans"] == 2 and ev["closed"] == 1 and (t + 1) % 4 == 0
    core.goal_hits[0] = 1
    tick(t + 1, None, None, one, zero)
    assert ev["succ"] == 1 and ev["n"] == 1 and "void" in ev


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


@needs_ef
def test_slabs_on_edgeflow_match_a_brute_force_voxel_count():
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.rewards import map_spawn_pool
    from surfgym.zones import load_zones
    core = SurfCore(str(EF050), SurfEnvConfig(num_envs=1))
    P = BFSPlanner.for_core(core, CELL, load_zones(str(EF050))["end"],
                            n_targets=0)
    assert P.kill_z == pytest.approx(256.0)
    sm = SlabMap.for_graph(P, surf_spec())
    solid = P.solid
    nz, ny, nx = solid.shape
    rng = np.random.default_rng(0)
    spawns = map_spawn_pool(core)["origin"].astype(np.float64)
    pts = np.vstack([spawns[:4], np.column_stack([
        rng.uniform(-1000, 1000, 6), rng.uniform(-1700, 1700, 6),
        rng.uniform(0, 1500, 6)])])
    cy, cx = sm.cells(pts)
    sl = sm.slabs(pts, cy, cx)
    assert sl.shape == (10, 3, 32, 32)
    assert np.all((sl >= 0.0) & (sl <= 1.0))
    for i in range(len(pts)):
        for c, off in enumerate((-384.0, -128.0, 128.0)):
            z0 = int(np.floor((pts[i, 2] + off - 128.0 - P.mins[2]) / CELL))
            for (wr, wc) in ((16, 16), (3, 29), (28, 5), (0, 0)):
                r, col = int(cy[i]) - 16 + wr, int(cx[i]) - 16 + wc
                tot = 0
                for iz in range(z0, z0 + 8):
                    for iy in (2 * r, 2 * r + 1):
                        for ix in (2 * col, 2 * col + 1):
                            ok = (0 <= iz < nz and 0 <= iy < ny
                                  and 0 <= ix < nx)
                            tot += 1 if not ok else int(solid[iz, iy, ix])
                assert sl[i, c, wr, wc] == pytest.approx(tot / 32.0), \
                    (i, c, wr, wc)
    # at a spawn: the slab just below holds the platform, the slab above
    # is open air
    assert sl[0, 1, 16, 16] > 0.0 and sl[0, 2, 16, 16] == 0.0


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
NEW_KEYS = ("plan_vocab", "plan_hindsight")


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


def _base_tree(dst: Path):
    """The python/ tree of the last first-parent commit WITHOUT the surf
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
        if r.returncode != 0 or b"--plan-vocab" in r.stdout:
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
        assert pa["spec"] == pb["spec"] and "vocab" not in pa["spec"]
        for k in pa["net"]:
            assert torch.equal(pa["net"][k], pb["net"][k]), k
        assert np.array_equal(pa["nov_count"], pb["nov_count"])
        assert pa["updates"] == pb["updates"]


@needs_lab
@pytest.mark.parametrize("mode", ["race", "goals", "bfs"])
def test_flag_off_is_bit_identical_to_the_base_commit(mode, tmp_path):
    """None of the new flags: this branch's trainer against the base
    commit's whole python/ tree (config, progress.csv minus fps, the eval
    trajectories, goals.csv / plan.csv, weights, Adam moments)."""
    ref = _base_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-surf-planner python/ tree in the history")
    old = tmp_path / "python" / "train_fast.py"
    extra = MODERN + {"race": [],
                      "goals": ["--goals", "1", "--goal-obs", "fan"],
                      "bfs": PLAN + ["--goal-reward", "arc"]}[mode]
    new_run = ROOT / "runs" / f"psrf_ctl_new_{mode}"
    old_run = tmp_path / "runs" / f"psrf_ctl_old_{mode}"
    _train(new_run.name, LAB_FLAGS, extra)
    _train(old_run.name, LAB_FLAGS, extra, script=old,
           runs_dir=tmp_path / "runs")
    assert old_run.exists(), "the base trainer writes under its own tree"
    _assert_identical(new_run, old_run)
    shutil.rmtree(new_run, ignore_errors=True)


@needs_lab
def test_flag_off_learned_walk_resume_is_bit_identical(tmp_path):
    """--goal-planner learned WITHOUT --plan-vocab (the walking planner of
    stage 3): a warm resume of the same bfs checkpoint by this branch's
    trainer and by the base commit's is bit-identical - the planner's
    network, its counts and updates, the executor, every log."""
    ref = _base_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-surf-planner python/ tree in the history")
    old = tmp_path / "python" / "train_fast.py"
    src = ROOT / "runs" / "psrf_lsrc"
    _train(src.name, LAB_FLAGS, MODERN + PLAN + ["--goal-reward", "arc"],
           steps="8192")
    res = ["--ckpt", str(src / "ckpt_final.pt"), "--goal-planner",
           "learned", "--freeze-policy", "1", "--ep-ticks", "1200",
           "--plan-batch", "32", "--record-every", "16384"]
    new_run = ROOT / "runs" / "psrf_lrn_new"
    old_run = tmp_path / "runs" / "psrf_lrn_old"
    _train(new_run.name, LAB_FLAGS, res, steps=str(8192 + 32768))
    _train(old_run.name, LAB_FLAGS, res, script=old,
           runs_dir=tmp_path / "runs", steps=str(8192 + 32768))
    _assert_identical(new_run, old_run)
    assert "planner" in _ckpt(new_run / "ckpt_final.pt")
    shutil.rmtree(new_run, ignore_errors=True)
    shutil.rmtree(src, ignore_errors=True)


# the surf smoke: 128 envs, tiny nets and lidar, 10 s episodes. DROP spawns
# (--spawn mixed: live surf-catch starts above ramp faces) so an untrained
# executor MOVES - the reservoir's reached-state goals need 480 u of travel,
# which an untrained walker on the spawn platform never makes - and the
# hindsight half of the diet is exercised; evals still start at the map
# start. A smoke of the machinery, not a recipe.
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
EF_LRN_STEPS = 196608


def _f(x):
    return float(x) if x not in ("", None) else float("nan")


@needs_ef
def test_surf_smoke_vocab_executor_then_learned_planner_and_record_gate():
    """From scratch: --goal-planner vocab --plan-vocab surf trains the
    executor on the diet (both sources used, completion counted by source,
    the executor's optimizer stepped); then a warm resume with
    --goal-planner learned --freeze-policy 1 (plan_vocab restored) trains
    the surf planner with finite losses while the executor and its Adam
    stay bit-identical; the record gate passes on both checkpoints."""
    src = ROOT / "runs" / "psrf_ef_src"
    _train(src.name, EF_FLAGS, MODERN + VOCAB, steps=str(EF_SRC_STEPS))
    cfg = _cfg(src)
    assert cfg["goal_planner"] == "vocab" and cfg["plan_vocab"] == "surf"
    assert cfg["plan_hindsight"] == 0.5 and cfg["goal_reward"] == "arc"
    assert "freeze_policy" not in cfg and "plan_lr" not in cfg
    rows = _rows(src)
    assert all(k in rows[0] for k in DIET_COLS)
    assert list(rows[0])[-len(DIET_COLS):] == DIET_COLS
    closed = sum(int(x["plan/closed"]) for x in rows)
    assert closed >= 100
    hs = [_f(x["plan/hs_share"]) for x in rows if x["plan/hs_share"]]
    assert hs and max(hs) > 0.0, "no hindsight plan was ever issued"
    chs = [_f(x["plan/complete_hs"]) for x in rows if x["plan/complete_hs"]]
    crd = [_f(x["plan/complete_rand"]) for x in rows
           if x["plan/complete_rand"]]
    assert chs and crd
    for col in ("plan/wall", "plan/wall_base", "plan/wall_len",
                "plan/void", "plan/void_base", "plan/hs_miss"):
        v = [_f(x[col]) for x in rows if x[col]]
        assert v and all(0.0 <= a <= 1.0 for a in v), col
    lens = [_f(x["plan/len"]) for x in rows if x["plan/len"]]
    assert lens and all(800.0 <= a <= 20000.0 for a in lens)
    ev = [x["plan/eval_finish"] for x in rows if x["plan/eval_finish"]]
    assert ev
    # the executor TRAINS: its Adam has stepped
    s_src = _ckpt(src / "ckpt_final.pt")
    steps = [float(v["step"]) for v in s_src["optimizer"]["state"].values()
             if "step" in v]
    assert steps and min(steps) > 0
    assert "planner" not in s_src
    # the eval: the vocab rule from the map start, JSON-safe headers
    tr = sorted(src.glob("traj_*.jsonl"))
    assert tr
    txt = tr[-1].read_text(encoding="utf-8")
    assert "Infinity" not in txt and "NaN" not in txt
    h0 = json.loads(txt.splitlines()[0])
    assert h0["plan"]["planner"] == "vocab" and h0["plan"]["graph_dist"] is None
    assert h0["plan"]["length"] >= 1500.0
    # --- stage 3: the learned surf planner over the frozen executor
    run = "psrf_ef_lrn"
    d = ROOT / "runs" / run
    shutil.rmtree(d, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt",
              str(src / "ckpt_final.pt")] + EF_FLAGS
             + ["--steps", str(EF_SRC_STEPS + EF_LRN_STEPS),
                "--goal-planner", "learned", "--freeze-policy", "1",
                "--plan-batch", "64"])
    out = r.stdout
    assert r.returncode == 0, out[-5000:] + r.stderr[-4000:]
    assert "plan_vocab=surf" in out and "planner: FRESH" in out
    assert "plan vocabulary SURF: 144 shapes" in out
    cfg = _cfg(d)
    assert cfg["goal_planner"] == "learned" and cfg["plan_vocab"] == "surf"
    assert cfg["freeze_policy"] == 1 and "plan_hindsight" not in cfg
    rows = _rows(d)
    from surfgym.goallearn import PLAN_COLS_SURF
    assert list(rows[0])[-len(PLAN_COLS_SURF):] == PLAN_COLS_SURF
    upd = [x for x in rows if x["plan/loss_pi"] != ""]
    assert upd, "the planner never updated"
    for x in upd:
        assert np.isfinite(float(x["plan/loss_pi"]))
        assert np.isfinite(float(x["plan/loss_v"]))
    ent = [float(x["plan/entropy"]) for x in rows if x["plan/entropy"]]
    assert ent and all(0.0 < e <= math.log(144) + 1e-6 for e in ent)
    s_new = _ckpt(d / "ckpt_final.pt")
    _same_policy_and_adam(s_src, s_new)
    assert s_new["planner"]["spec"]["vocab"] == "surf"
    assert s_new["planner"]["updates"] >= 1
    tr = sorted(d.glob("traj_*.jsonl"))
    h1 = json.loads(tr[-1].read_text(encoding="utf-8").splitlines()[0])
    assert h1["plan"]["planner"] == "learned" and "pitch" in h1["plan"]
    # the launcher's record gate on both checkpoints (greedy, stoch, mixed)
    for rn, ck in ((src.name, src / "ckpt_final.pt"),
                   (run, d / "ckpt_final.pt")):
        g = _run([sys.executable, str(ROOT / "tools" / "record_gate.py"), rn,
                  "--ckpt", str(ck), "--no-pov"])
        assert g.returncode == 0, g.stdout[-3000:] + g.stderr[-3000:]
        assert "record gate PASSED: 3 recording(s)" in g.stdout
    # the smoke's numbers, for the report: per log window with closed
    # plans (step, closed, completion random / hindsight, hindsight share,
    # miss rate, solid crossing vs base, void vs base)
    print("\nSMOKE vocab (step, closed, cmpl_rand, cmpl_hs, hs_share, "
          "hs_miss, wall, wall_base, void, void_base)")
    for x in _rows(src):
        if int(x["plan/closed"]):
            print("  ", tuple(x[k] for k in (
                "time/total_timesteps", "plan/closed", "plan/complete_rand",
                "plan/complete_hs", "plan/hs_share", "plan/hs_miss",
                "plan/wall", "plan/wall_base", "plan/void",
                "plan/void_base")))
    print("SMOKE learned (loss_pi, loss_v, entropy, complete, wall, "
          "wall_base, void, void_base)")
    for x in upd:
        print("  ", tuple(x[k] for k in (
            "plan/loss_pi", "plan/loss_v", "plan/entropy", "plan/complete",
            "plan/wall", "plan/wall_base", "plan/void", "plan/void_base")))
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(src, ignore_errors=True)


@needs_lab
def test_refusals():
    base = [sys.executable, "-u", str(TRAIN), "--run", "psrf_bad"] \
        + LAB_FLAGS + ["--steps", "2048"] + MODERN
    goals = ["--goals", "1", "--goal-obs", "fan"]
    cases = [
        (goals + ["--goal-planner", "vocab", "--goal-reward", "arc"],
         "it needs --plan-vocab surf"),
        (goals + ["--goal-planner", "vocab", "--plan-vocab", "surf"],
         "--goal-reward arc"),
        (goals + ["--goal-planner", "vocab", "--plan-vocab", "surf",
                  "--goal-reward", "arc", "--plan-hindsight", "1.5"],
         "--plan-hindsight is a probability"),
        (goals + ["--goal-planner", "bfs", "--plan-hindsight", "0.3"],
         "--plan-hindsight without --goal-planner vocab"),
        (goals + ["--goal-planner", "bfs", "--plan-vocab", "surf"],
         "--plan-vocab without --goal-planner learned or vocab"),
    ]
    for extra, msg in cases:
        r = _run(base + extra)
        assert r.returncode != 0, extra
        assert msg in r.stdout + r.stderr, (extra,
                                            (r.stdout + r.stderr)[-1500:])
    shutil.rmtree(ROOT / "runs" / "psrf_bad", ignore_errors=True)
