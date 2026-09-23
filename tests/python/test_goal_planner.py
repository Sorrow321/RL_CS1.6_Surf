"""--goal-planner bfs: the deterministic BFS planner (surfgym/goalplan.py),
its goal-system wiring (surfgym/goalsys.py), the plan reward, the fan
offsets, the recorder and the trainer.

(a) the planner on toy occupancies, CPU only, no DLL: the walkable rule
    (support within 64 u below, a dead floor supports nothing, the kill
    ceiling), the edges (26-neighbourhood, no diagonal squeeze past a solid
    corner, Euclidean costs), Dijkstra against scipy's, the descent ends at
    its target with the field's own length, the target draw's band and its
    fallbacks, the plan lifted to the start's height above the floor, the
    plan potential (exact on nodes, nearest node + offset off the graph).
(b) the real labyrinths (the built core + maps_pool/labyrinth_left{100,
    200}.bsp; SURF_TEST_POOL / SURFCORE_DLL from a worktree): the graph
    connects every map spawn to the finish box, the spawn's plan goes LEFT
    around the wall, plans from random starts end at their targets, the
    goal system hands out kind-3 goals with the box (not a sphere) as the
    finish, and the plan potential falls along a plan.
(c) the trainer (CPU): with the planner flags OFF it is bit-identical to the
    base commit (config, progress.csv minus fps, the eval trajectories,
    weights, Adam moments), plain and with --goals; a smoke on
    labyrinth_left100 with --goal-reward arc and plan trains and writes the
    planner's config keys; tools/record_gate.py passes on that checkpoint
    (greedy, stoch, mixed); the zero-shot recording on labyrinth_left200
    plans with lab200's own graph; and the refusals.
"""
from __future__ import annotations

import json
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

from surfgym.goalplan import (BFSPlanner, PlanDistField,         # noqa: E402
                              make_plan_hooks, parse_fan_offsets,
                              walkable_mask)

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


# ==========================================================================
# (a) the planner on toy occupancies
# ==========================================================================
def _room(nz=6, ny=12, nx=12):
    """A closed box: solid floor iz=0, ceiling iz=nz-1, walls on the rim;
    free inside."""
    occ = np.ones((nz, ny, nx), np.uint8)
    occ[1:nz - 1, 1:ny - 1, 1:nx - 1] = 0
    return occ


MINS = np.zeros(3)


def test_walkable_rule_support_and_kill_ceiling():
    occ = _room()
    walk, r, floor = walkable_mask(occ != 0, MINS, CELL)
    assert r == 2                                   # 64 u at cell 32
    # the two layers above the floor are walkable, nothing higher
    assert walk[1, 1:-1, 1:-1].all() and walk[2, 1:-1, 1:-1].all()
    assert not walk[3:].any() and not walk[0].any()
    # the floor height is the TOP face of the supporting solid, per layer
    assert np.all(floor[1][walk[1]] == CELL) and np.all(floor[2][walk[2]] == CELL)
    # a platform (solid block at iz=2) supports the cells above it
    occ2 = occ.copy()
    occ2[1:3, 4:6, 4:6] = 1
    w2, _, f2 = walkable_mask(occ2 != 0, MINS, CELL)
    assert w2[3, 4:6, 4:6].all() and w2[4, 4:6, 4:6].all()
    assert np.all(f2[3, 4:6, 4:6] == 3 * CELL)
    # a KILL CEILING at the platform's base: the floor is dead, supports
    # nothing, and the layers at or below the ceiling are not nodes; the
    # platform top (above the ceiling) still is
    kz = 1.5 * CELL                                 # centre of iz=1 is 48
    w3, _, _ = walkable_mask(occ2 != 0, MINS, CELL, kill_z=kz)
    assert not w3[1].any() and not w3[2].any()
    assert w3[3, 4:6, 4:6].all()
    assert int(w3.sum()) == 8


def test_edges_are_26_connected_without_corner_cutting():
    occ = _room(nz=4)
    occ[1, 3, 3] = 1                                # two solid cells meeting
    occ[1, 4, 4] = 1                                # at one corner
    P = BFSPlanner(occ, MINS, CELL, n_targets=0)
    a = P.node_of[1, 3, 4]
    b = P.node_of[1, 4, 3]
    assert a >= 0 and b >= 0
    assert b not in set(P.nbr[a].tolist())          # no squeeze between them
    c = P.node_of[1, 2, 5]
    assert c in set(P.nbr[a].tolist())              # a free diagonal is fine
    # the costs are the Euclidean step lengths
    assert sorted(set(np.round(P.wk / CELL, 6).tolist())) == \
        sorted(set(np.round([1.0, np.sqrt(2), np.sqrt(3)], 6).tolist()))
    # every edge is symmetric (the graph is undirected)
    for i in range(P.n_nodes):
        for j in P.nbr[i]:
            if j >= 0:
                assert i in set(P.nbr[j].tolist())


def test_dijkstra_matches_scipy_and_descent_ends_at_the_target():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    occ = _room(nz=5, ny=14, nx=14)
    occ[1:4, 7, 1:10] = 1                           # a wall with a gap
    occ[1:3, 3, 4:13] = 1
    P = BFSPlanner(occ, MINS, CELL, n_targets=12, seed=3)
    rows, cols, w = [], [], []
    for i in range(P.n_nodes):
        for k, j in enumerate(P.nbr[i]):
            if j >= 0:
                rows.append(i)
                cols.append(int(j))
                w.append(P.wk[k])
    g = csr_matrix((w, (rows, cols)), shape=(P.n_nodes, P.n_nodes))
    ref = dijkstra(g, directed=True, indices=P.targets)
    assert np.allclose(P.dist[:P.n_rand], ref, rtol=1e-6, atol=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(60):
        t = int(rng.integers(P.n_rand))
        s = int(rng.integers(P.n_nodes))
        if not np.isfinite(P.dist[t, s]):
            continue
        path = np.asarray(P._descend(P.dist[t], s, P.nbr, P.wk))
        assert path[0] == s and path[-1] == P.targets[t]
        # each step is an edge and the steps add up to the field's length
        steps = np.linalg.norm(np.diff(P.xyz[path], axis=0), axis=1)
        assert np.isclose(steps.sum(), P.dist[t, s], rtol=1e-4, atol=1e-2)
        assert np.all(np.diff(P.dist[t][path]) < 0)


def test_choose_draws_the_band_then_falls_back():
    occ = _room(nz=4, ny=20, nx=20)
    box = {"mins": [16 * CELL, 16 * CELL, 0.0],
           "maxs": [19 * CELL, 19 * CELL, 3 * CELL]}
    P = BFSPlanner(occ, MINS, CELL, finish_box=box, n_targets=64, seed=1)
    assert P.fin == P.n_rand and len(P.finish_nodes) > 0
    # random targets never sit inside the finish box
    assert not set(P.targets.tolist()) & set(P.finish_nodes.tolist())
    s = int(P.node_of[1, 2, 2])
    rng = np.random.default_rng(5)
    got = [P.choose(s, rng, 0.0, 128.0, 256.0) for _ in range(200)]
    d = P.dist[np.asarray(got), s]
    assert np.all((d >= 128.0) & (d <= 256.0))      # inside the band
    assert len(set(got)) > 1                        # uniform among eligible
    fin = [P.choose(s, rng, 1.0, 128.0, 256.0) for _ in range(20)]
    assert set(fin) == {P.fin}
    # an impossible band: the reachable target nearest to it
    t = P.choose(s, rng, 0.0, 1e6, 2e6)
    assert t == int(np.argmax(P.dist[:P.n_rand, s]))


def test_plan_is_lifted_to_the_start_height_and_ends_at_the_goal():
    occ = _room(nz=4, ny=20, nx=20)
    occ[1:3, 10, 1:17] = 1                          # force a detour
    box = {"mins": [2 * CELL, 16 * CELL, 0.0], "maxs": [5 * CELL, 18 * CELL, 3 * CELL]}
    P = BFSPlanner(occ, MINS, CELL, finish_box=box, n_targets=8, seed=2)
    o = np.array([3.5 * CELL, 2.5 * CELL, CELL + 36.0])   # 36 u above the floor
    pl = P.plan(o, P.fin)
    assert pl is not None and pl.finish
    assert np.allclose(pl.raw[0], o)
    assert np.allclose(pl.raw[:, 2], o[2])          # flat floor: one height
    assert np.allclose(pl.line[0], o, atol=1e-3)
    assert np.allclose(pl.line[-1][:2], P.finish_center[:2], atol=1e-3)
    assert pl.raw[:, 0].max() > 16 * CELL           # the detour around x=17
    # resampled at the fan spacing (equal ARC steps; a chord across a
    # corner is shorter, never longer)
    seg = np.linalg.norm(np.diff(pl.line, axis=0), axis=1)
    assert seg.max() <= 1.5 * 128.0 and abs(np.median(seg) - 128.0) < 64.0
    # a random target: the plan ends ON it, at the start's height
    t = 3
    pt = P.plan(o, t)
    tgt = P.xyz[P.targets[t]]
    assert np.allclose(pt.goal[:2], tgt[:2]) and np.isclose(pt.goal[2], o[2])
    assert np.allclose(pt.line[-1], pt.goal, atol=1e-3)


def test_plan_potential_is_the_field_on_nodes_and_nearest_plus_offset_off():
    occ = _room(nz=4, ny=16, nx=16)
    P = BFSPlanner(occ, MINS, CELL, n_targets=4, seed=4)
    f = PlanDistField(P, 3)
    f.set_targets([0, 1, 2], [0, 1, -1])
    nodes = np.array([5, 40, 77])
    got = f.sample(P.xyz[nodes])
    assert np.isclose(got[0], P.dist[0, 5], atol=1e-3)
    assert np.isclose(got[1], P.dist[1, 40], atol=1e-3)
    assert got[2] == 0.0                            # no target -> 0
    # far above the graph (no walkable corner): nearest node + offset
    q = P.xyz[[5, 40, 77]].copy()
    q[:, 2] += 1000.0
    nn = P.snap(q)
    exp = P.dist[[0, 1], nn[:2]] + np.linalg.norm(q[:2] - P.xyz[nn[:2]], axis=1)
    assert np.allclose(f.sample(q)[:2], exp, rtol=1e-5)
    with pytest.raises(ValueError, match="row-aligned"):
        f.sample(q[:2])


def test_parse_fan_offsets():
    assert parse_fan_offsets(None) is None
    assert parse_fan_offsets("0.25,0.5, 1") == (0.25, 0.5, 1.0)
    assert parse_fan_offsets([0.25, 2.0]) == (0.25, 2.0)
    for bad in ("", "1,1", "0.5,0.25", "-1,2", "a,b"):
        with pytest.raises(ValueError):
            parse_fan_offsets(bad)


# ==========================================================================
# (b) the real labyrinths
# ==========================================================================
_PLANNERS = {}


def _lab(path, n_targets=256, seed=9173):
    key = (str(path), n_targets, seed)
    if key not in _PLANNERS:
        from surfgym.core import SurfCore, SurfEnvConfig
        from surfgym.rewards import map_spawn_pool
        from surfgym.zones import load_zones
        core = SurfCore(str(path), SurfEnvConfig(num_envs=1))
        P = BFSPlanner.for_core(core, CELL, load_zones(str(path))["end"],
                                n_targets=n_targets, seed=seed)
        spawns = map_spawn_pool(core)["origin"].astype(np.float64)
        _PLANNERS[key] = (P, spawns, core)
    return _PLANNERS[key]


@needs_lab
@pytest.mark.parametrize("path,wall_x", [(LAB100, -400.0), (LAB200, -1400.0)])
def test_labyrinth_spawn_reaches_the_finish_by_going_left(path, wall_x):
    P, spawns, _ = _lab(path)
    print("\n" + P.describe())
    assert P.fin is not None and len(P.finish_nodes) > 0
    s = P.snap(spawns)
    assert np.all(np.isfinite(P.dist[P.fin, s]))    # every spawn connects
    for o in spawns:
        pl = P.plan(o, P.fin)
        assert pl is not None and pl.finish
        # the wall forces a LEFT detour: the path's minimum x is far below
        # the start's (lab100 turns at x ~ -464, lab200 at ~ -1488)
        assert pl.raw[:, 0].min() < min(o[0] - 800.0, wall_x)
        assert pl.line[:, 0].min() < min(o[0] - 800.0, wall_x)
        # ... and the path ends inside the finish box
        last = pl.raw[-2]                           # before the box centre
        box_lo = np.array([368.0, 816.0]) - 24.0
        box_hi = np.array([656.0, 1008.0]) + 24.0
        assert np.all(last[:2] >= box_lo) and np.all(last[:2] <= box_hi)
        # the plan is flat on this flat maze, at the spawn's height
        assert np.allclose(pl.line[:, 2], o[2], atol=1e-3)
    # the table is tiny here; the report quotes these
    assert P.dist.nbytes < 4 * 2**20 and P.build_secs < 60.0


@needs_lab
def test_labyrinth_plans_from_random_starts_end_at_their_targets():
    P, spawns, _ = _lab(LAB100)
    rng = np.random.default_rng(11)
    n_fin = 0
    for _ in range(300):
        s0 = int(rng.integers(P.n_nodes))
        o = P.xyz[s0] + np.r_[rng.uniform(-14, 14, 2), 0.0]
        o[2] = P.floor[s0] + 36.0 + rng.uniform(0.0, 30.0)
        s = int(P.snap(o[None])[0])
        t = P.choose(s, rng, 0.2, 256.0, 4096.0)
        assert t >= 0
        pl = P.plan(o, t)
        assert pl is not None
        path = np.asarray(P._descend(P.dist[t], s, P.nbr, P.wk))
        if pl.finish:
            n_fin += 1
            assert path[-1] in set(P.finish_nodes.tolist())
            assert np.allclose(pl.goal[:2], P.finish_center[:2])
        else:
            assert path[-1] == P.targets[t]
            assert 256.0 <= pl.length <= 4096.0
            assert np.allclose(pl.goal[:2], P.xyz[P.targets[t], :2])
        # the line starts at the start and ends at the goal
        assert np.allclose(pl.line[0], o, atol=1e-3)
        assert np.allclose(pl.line[-1], pl.goal, atol=1e-3)
    assert 20 < n_fin < 120                          # ~20% finish draws


class _FakeCore:
    """The goal system's view of a core: states, bounds, goal hits."""

    def __init__(self, n):
        from surfgym.core import STATE_DTYPE
        self.num_envs = int(n)
        self.states_view = np.zeros(self.num_envs, STATE_DTYPE)
        self.goal_hits = np.zeros(self.num_envs, np.uint8)
        self.config = types.SimpleNamespace(
            phys=types.SimpleNamespace(sv_gravity=800.0))
        self.failed = np.zeros(self.num_envs, np.uint8)

    def map_bounds(self):
        return (np.asarray([-2e3, -2e3, -200.0], np.float32),
                np.asarray([2e3, 2e3, 400.0], np.float32))

    def force_fail(self, m):
        self.failed |= np.asarray(m, np.uint8)


class _Reach:
    reach_max = 1e9

    def sample(self, pos):
        return np.zeros(len(np.atleast_2d(pos)), np.float32)

    def reachable(self, pos):
        return np.ones(len(np.atleast_2d(pos)), bool)


def _goal_args(**over):
    a = types.SimpleNamespace(
        goal_radius=192.0, goal_holdout=None, goal_air_frac=0.0,
        goal_kmin=1.0, goal_kmax=5.0, goal_kcap=60.0, goal_curriculum=0,
        goal_frontier=0, goal_route_uniform=0, goal_route=None,
        goal_route_frac=0.0, goal_fixed=0, goal_plan_finish=0.2,
        goal_plan_dmin=256.0, goal_plan_dmax=4096.0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


@needs_lab
def test_goal_system_hands_out_planned_goals_with_the_box_as_the_finish(tmp_path):
    from surfgym.goals import MultiLine
    from surfgym.goalsys import GoalSystem
    P, spawns, _ = _lab(LAB100)
    n = 64
    core = _FakeCore(n)
    core.states_view["origin"][:] = spawns[np.arange(n) % len(spawns)]
    offs = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    line = MultiLine(n, offsets=offs, device="cpu")
    field = PlanDistField(P, n)
    gs = GoalSystem(core, n, line, _Reach(), 4000.0, _goal_args(), "cpu",
                    str(tmp_path), seed=0, dist_field=field, planner=P)
    gs.set_finish([368.0, 816.0, 8.0], [656.0, 1008.0, 160.0])
    assert gs.eval_line.offsets == offs            # the eval fan is the fan
    gs.assign(np.arange(n))
    assert np.all(gs.kind == 3) and gs.n_assigned[3] == n
    fin = gs.plan_fin
    assert 0 < fin.sum() < n
    # the finish is the BOX: no sphere is armed for it; targets get one
    assert not gs.sphere.active[fin].any() and gs.sphere.active[~fin].all()
    assert np.allclose(gs.sphere.radius[~fin], 192.0)
    # the plan field follows the targets, the finish row for finish envs
    assert np.all(field.tgt[fin] == P.fin) and np.all(field.tgt[~fin] < P.fin)
    # each env's fan line starts at its spawn
    L = line.length.numpy()
    first = line.pts.numpy()[np.arange(n), 0]
    assert np.allclose(first, core.states_view["origin"], atol=1e-3)
    last = line.pts.numpy()[np.arange(n), L - 1]
    assert np.allclose(last[~fin, :2], gs.sphere.center[~fin, :2], atol=1e-3)
    # the box counts as the finish's success (and only for finish envs)
    done = np.zeros(n, np.uint8)
    i_fin, i_tgt = int(np.flatnonzero(fin)[0]), int(np.flatnonzero(~fin)[0])
    done[[i_fin, i_tgt]] = 1
    core.goal_hits[[i_fin, i_tgt]] = 1
    g = gs.on_step(done, np.zeros(n, np.uint8), np.full(n, 10))
    assert g[i_fin] and not g[i_tgt]
    assert gs.plan_n.tolist() == [1, 1] and gs.plan_ok.tolist() == [1, 0]
    note = gs.note(1000)
    assert "plan fin 100.0%/1 tgt  0.0%/1" in note
    rows = (tmp_path / "plan.csv").read_text().splitlines()
    assert rows[0].startswith("step,n,succ,n_finish") and len(rows) == 2
    # the eval: the training map's tally lands in goals.csv / plan.csv; a
    # held-out map's (planned on its own graph) does not overwrite it
    ecore = _FakeCore(1)
    ecore.states_view["origin"][0] = spawns[0]
    meta, tick = gs.eval_hooks(ecore, seed=1)
    meta(0)
    ecore.goal_hits[0] = 1
    tick(10, None, None, np.ones(1, np.uint8), np.zeros(1, np.uint8))
    assert "plan-eval finish 1/1" in gs.eval_note()
    assert gs._last_eval == (1.0, 1)
    P200, sp200, _ = _lab(LAB200)
    ecore.states_view["origin"][0] = sp200[0]
    meta, tick = gs.eval_hooks(ecore, seed=1, planner=P200)
    h = meta(0)
    assert h["plan"]["length"] > 5000.0            # lab200's own plan
    assert min(p[0] for p in h["line"]) < -1400.0
    gs.eval_note()
    assert gs._last_eval == (1.0, 1)                # untouched


@needs_lab
def test_plan_potential_falls_along_the_spawn_plan():
    P, spawns, _ = _lab(LAB100)
    f = PlanDistField(P, 1)
    f.set_targets([0], [P.fin])
    pl = P.plan(spawns[0], P.fin)
    # walk the plan at 8 u steps: the potential falls by ~ the distance
    # walked (corners cut by the resample cost a little of the slope)
    pts = []
    for a, b in zip(pl.line[:-1], pl.line[1:]):
        k = max(1, int(np.linalg.norm(b - a) // 8))
        pts += [a + (b - a) * (j / k) for j in range(k)]
    d = np.array([f.sample(np.asarray(p, np.float64)[None])[0] for p in pts])
    walked = np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1)
    drop = -np.diff(d)
    assert drop.sum() > 0.9 * walked.sum() - 200.0
    assert (drop < -8.5).mean() < 0.02             # (almost) never uphill
    assert d[0] == pytest.approx(pl.length, abs=40.0)


@needs_lab
def test_eval_hooks_score_the_finish_by_the_box():
    from surfgym.goals import MultiLine
    P, spawns, _ = _lab(LAB100)
    core = _FakeCore(1)
    core.states_view["origin"][0] = spawns[0]
    ev = {}
    line = MultiLine(1, device="cpu")
    meta, tick = make_plan_hooks(P, core, ev, line=line, radius=192.0,
                                 finish_radius=192.0)
    h = meta(0)
    assert h["plan"]["target"] == "finish" and ev["box"] and ev["center"] is None
    assert len(h["line"]) >= 2
    one = np.ones(1, np.uint8)
    zero = np.zeros(1, np.uint8)
    core.goal_hits[0] = 1                          # the box ended it: a win
    tick(50, None, None, one, zero)
    assert ev["succ"] == 1 and ev["ticks"] == [50]
    meta(1)
    core.goal_hits[0] = 0                          # a death is not
    tick(90, None, None, one, zero)
    assert ev["succ"] == 1 and ev["n"] == 2
    # the secondary eval: a random target with a sphere
    ev2 = {}
    meta2, tick2 = make_plan_hooks(P, core, ev2, radius=192.0,
                                   random_targets=True,
                                   rng=np.random.default_rng(3))
    h2 = meta2(0)
    assert isinstance(h2["plan"]["target"], int) and ev2["center"] is not None
    core.states_view["origin"][0] = ev2["center"]
    tick2(5, None, None, zero, zero)               # inside the sphere
    assert core.failed[0] == 1 and ev2["pending"]


# ==========================================================================
# (c) the trainer
# ==========================================================================
def _env():
    # -1, not "": on Windows an EMPTY value UNSETS the variable and the GPU
    # stays visible (memory: windows-empty-env-var-unsets)
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(ROOT), timeout=timeout, encoding="utf-8",
                          errors="replace")


# the view_continuous smoke set, on the small maze: 64 envs, 16x8 lidar,
# 64-wide nets, 1.0 s episodes so assignment and resets run many times
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
PLAN_KEYS = ("goal_planner", "goal_plan_targets", "goal_plan_finish",
             "goal_plan_dmin", "goal_plan_dmax", "goal_fan_offsets")


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
    """The python/ tree of the last first-parent commit WITHOUT
    --goal-planner (HEAD itself while the flag is uncommitted), extracted
    under ``dst``; returns the ref, or None."""
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
        if r.returncode != 0 or b"--goal-planner" in r.stdout:
            continue
        z = dst / "base_python.zip"
        subprocess.run(["git", "archive", "--format=zip", "-o", str(z), ref,
                        "python"], check=True, cwd=str(ROOT), timeout=120)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
        z.unlink()
        return ref
    return None


def _assert_identical(a: Path, b: Path):
    ca, cb = _cfg(a), _cfg(b)
    assert not any(k in ca for k in PLAN_KEYS)
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
    for extra in ("goals.csv",):
        if (a / extra).exists() or (b / extra).exists():
            assert (a / extra).read_bytes() == (b / extra).read_bytes(), extra
    assert not (a / "plan.csv").exists()
    import torch
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert set(sa["policy"]) == set(sb["policy"])
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    assert set(oa) == set(ob)
    for i in oa:
        for k in oa[i]:
            if torch.is_tensor(oa[i][k]):
                assert torch.equal(oa[i][k], ob[i][k]), (i, k)


@needs_lab
@pytest.mark.parametrize("mode", ["race", "goals"])
def test_flag_off_is_bit_identical_to_the_base_commit(mode, tmp_path):
    """No planner flag: the trainer of this branch against the base
    commit's (its whole python/ tree runs, so the old trainer imports its
    own surfgym): the config gains no key and every number is the same.
    'goals' runs the goal system without the planner (reached-state/air
    goals, the fan) - goalsys.py, goals.py and the fan construction all
    changed on this branch."""
    ref = _base_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-planner python/ tree in the first-parent history")
    old = tmp_path / "python" / "train_fast.py"
    extra = MODERN + (["--goals", "1", "--goal-obs", "fan"]
                      if mode == "goals" else [])
    new_run = ROOT / "runs" / f"plan_ctl_new_{mode}"
    old_run = tmp_path / "runs" / f"plan_ctl_old_{mode}"
    _train(new_run.name, extra)
    _train(old_run.name, extra, script=old, runs_dir=tmp_path / "runs")
    assert old_run.exists(), "the base trainer writes under its own tree"
    _assert_identical(new_run, old_run)
    shutil.rmtree(new_run, ignore_errors=True)


@needs_lab
@pytest.mark.parametrize("reward", ["plan", "arc"])
def test_planner_smoke_and_record_gate(reward):
    run = f"plan_smoke_{reward}_t"
    d = ROOT / "runs" / run
    # the plan arm also carries lab200 as a HELD-OUT map: the in-trainer
    # zero-shot eval, planned on lab200's own graph
    held = (["--heldout-maps", str(LAB200), "--heldout-goal-cell", "32"]
            if reward == "plan" else [])
    r = _train(run, MODERN + PLAN + ["--goal-reward", reward] + held,
               steps="16384")
    out = r.stdout
    if held:
        assert "greedy[HELDOUT labyrinth_left200]" in out
        ht = sorted(d.glob("traj_*_labyrinth_left200.jsonl"))
        assert ht
        h0 = json.loads(ht[-1].read_text().splitlines()[0])
        assert h0["map"] == "labyrinth_left200" and h0["plan"]["length"] > 5000.0
        assert min(p[0] for p in h0["line"]) < -1400.0
    assert "planner bfs:" in out and "goals: PLANNED" in out
    assert "plan-eval finish" in out
    assert "8 horizons 0.25-2s" in out
    cfg = _cfg(d)
    assert cfg["goal_planner"] == "bfs" and cfg["goal_reward"] == reward
    assert cfg["goal_fan_offsets"] == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    assert cfg["goal_plan_targets"] == 256 and cfg["goal_plan_finish"] == 0.2
    rows = _rows(d)
    assert len(rows) >= 4
    plan = (d / "plan.csv").read_text().splitlines()
    assert len(plan) == len(rows) + 1
    n_ended = sum(int(x.split(",")[1]) for x in plan[1:])
    assert n_ended >= 64                            # reassignment ran
    if reward == "arc":
        assert "multiarc" in out and "goal arc shaping" in out
    else:
        assert "PLANNER's own cost-to-go" in out
    ck = d / "ckpt_final.pt"
    assert ck.exists()
    # the launcher's record gate (the dashboard buttons' backend), from
    # this checkout - it resolves the map by stem under maps/ + maps_pool/
    if (ROOT / "maps_pool" / LAB100.name).exists():
        g = _run([sys.executable, str(ROOT / "tools" / "record_gate.py"), run,
                  "--ckpt", str(ck), "--no-pov"])
        assert g.returncode == 0, g.stdout[-3000:] + g.stderr[-3000:]
        assert "record gate PASSED: 3 recording(s)" in g.stdout
    # the zero-shot probe: the lab100 policy on lab200, planning with
    # lab200's own graph
    rec = d / "zs200.jsonl"
    z = _run([sys.executable, str(ROOT / "tools" / "record_ckpt.py"), str(ck),
              "--map", str(LAB200), "--episodes", "1", "--ep-ticks", "200",
              "--out", str(rec)])
    assert z.returncode == 0, z.stdout[-3000:] + z.stderr[-3000:]
    hdr = json.loads(rec.read_text().splitlines()[0])
    assert hdr["map"] == "labyrinth_left200"
    assert hdr["plan"]["target"] == "finish"
    assert min(p[0] for p in hdr["line"]) < -1400.0  # lab200's detour
    shutil.rmtree(d, ignore_errors=True)


@needs_lab
def test_refusals():
    base = [sys.executable, "-u", str(TRAIN), "--run", "plan_bad"] + LAB_FLAGS \
        + ["--steps", "2048"] + MODERN
    cases = [
        (["--goal-planner", "bfs"], "--goal-planner plans GOALS"),
        (["--goals", "1", "--goal-reward", "plan"],
         "--goal-reward plan shapes on the planner's field"),
        (["--goals", "1", "--goal-plan-targets", "8"],
         "--goal-plan-targets without --goal-planner bfs"),
        (["--goal-fan-offsets", "0.5,1"], "--goal-fan-offsets sets the horizons"),
        (["--goals", "1", "--goal-fan-offsets", "1,0.5"], "strictly increasing"),
        (PLAN + ["--obs-compass", "1"], "is not implemented"),
    ]
    for extra, msg in cases:
        r = _run(base + extra)
        assert r.returncode != 0, extra
        assert msg in r.stdout + r.stderr, (extra, (r.stdout + r.stderr)[-1500:])
    shutil.rmtree(ROOT / "runs" / "plan_bad", ignore_errors=True)
