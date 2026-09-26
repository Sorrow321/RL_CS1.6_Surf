"""--goal-planner primlearn (surfgym/goalprimplan.py): the learned primitive planner (step 2).

(a) the mixture: its log density is the Gaussian one for a single component, the squash stays in
    goalprim's ranges, greedy takes the heaviest component's mean;
(b) the planner cycle on a real core: the rays are in [0, 1], a planned env gets a primitive,
    closed primitives become transitions, a PPO update runs, the state saves and loads;
(c) the trainer + recorder on the small maze (CPU): a primlearn run trains its planner, dumps its
    knobs, and record_ckpt.py records the checkpoint with the stored planner;
(d) --plan-joint: planner + executor on ONE reward - the planner's transitions are the executor's
    per-tick reward, SMDP-discounted on its clock, and the trainer runs, records and refuses;
(e) --plan-az (AlphaZero-style expert iteration): the policy target pulls the mixture toward the
    candidates the search visited most; the target replay reads each file once and keeps the
    newest; the update is unchanged with the flag off and logs the targets with it on;
    tools/az_worker.py writes well-formed targets from a checkpoint, and the trainer resumed with
    --plan-az fits them (finite, logged, restored, recordable); the refusals.
"""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from surfgym.goalprim import PrimitivePlanner                                  # noqa: E402
from surfgym.goalprimplan import (N_OBS, N_RAYS, PrimLearnedPlanner, RayCaster,  # noqa: E402
                                  mix_logp, mix_sample, observe, squash)

LAB = ROOT / "maps_pool" / "labyrinth_left100.bsp"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_core = pytest.mark.skipif(not (DLL.exists() and LAB.exists()),
                                reason="needs the built core + maps_pool/labyrinth_left100")


def test_single_component_logp_is_gaussian():
    mu = torch.tensor([[[0.3, -0.2]]])
    ls = torch.tensor([[[-0.5, 0.1]]])
    lg = torch.zeros(1, 1)
    u = torch.tensor([[0.7, 0.4]])
    want = sum(-0.5 * ((u[0, d] - mu[0, 0, d]) / math.exp(ls[0, 0, d])) ** 2 - ls[0, 0, d]
               - 0.5 * math.log(2 * math.pi) for d in range(2))
    assert abs(float(mix_logp(lg, mu, ls, u)[0]) - float(want)) < 1e-5


def test_squash_ranges_and_greedy():
    P = PrimitivePlanner(side=100.0, down=50.0, up=20.0)
    s = squash(P, np.random.default_rng(0).normal(0, 4, (2000, 6)))
    assert s[:, :3].min() >= -100 and s[:, :3].max() <= 100
    assert s[:, 3:].min() >= -50 and s[:, 3:].max() <= 20
    lg = torch.tensor([[0.1, 2.0, -1.0]])
    mu = torch.arange(3 * 6, dtype=torch.float32).view(1, 3, 6)
    g = mix_sample(lg, mu, torch.zeros(1, 3, 6), torch.Generator(), greedy=True)
    assert torch.equal(g[0], mu[0, 1])


@needs_core
def test_planner_cycle_on_a_core():
    from surfgym.core import SurfCore, SurfEnvConfig
    n = 16
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64)
    x = observe(RayCaster(core), pos, sv["velocity"], sv["yaw"], pos[0] + [3000.0, 0.0, 0.0])
    assert x.shape == (n, N_OBS)
    assert (x[:, :N_RAYS] >= 0).all() and (x[:, :N_RAYS] <= 1.0 + 1e-6).all()
    assert abs(float(x[0, N_RAYS + 3]) - math.log1p(3.0)) < 1e-4    # log1p(dist / 1000)
    assert abs(float(np.hypot(x[0, N_RAYS], x[0, N_RAYS + 1])) - 1.0) < 1e-4   # level, unit
    prim = PrimitivePlanner(secs=0.5, n_envs=n)
    P = PrimLearnedPlanner(prim, core, n, "cpu", finish=pos[0] + [3000.0, 0.0, 0.0],
                           bounds=core.map_bounds(), act_every=4,
                           cfg={"plan_uniform": 0.0, "plan_batch": 8, "plan_novelty": 0.0, "plan_shaping": "pbrs",
                                "plan_r_fail": -0.5})
    P.request(np.arange(n), pos)
    idx, lines, fresh = P.plan(pos, sv["velocity"], sv["yaw"])
    assert len(idx) == n and fresh.all() and P.decided.all()        # plan_uniform 0: all chosen
    # stand still: every primitive times out (budget 1.5 x 0.5 s) and is a failed transition
    for _ in range(P.budget_ticks + 1):
        P.on_tick(pos, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))
    assert P.n_ready() == n and all(t[4] == -0.5 for b in P.buf for t in b)   # r_fail, no progress
    P.plan(pos, sv["velocity"], sv["yaw"])
    u = P.update()
    assert u is not None and u["n"] == n and P.n_ready() == 0
    Q = PrimLearnedPlanner(prim, core, n, "cpu", finish=P.finish, bounds=core.map_bounds())
    Q.load_state_dict_all(P.state_dict_all())
    for a, b in zip(P.net.parameters(), Q.net.parameters()):
        assert torch.equal(a, b)
    assert Q.updates == 1
    # progress pays a SURVIVOR, is BANKED, and a death charges the bank back: every env moves
    # 1,000 u (a primitive closes alive by time-out), then half of them die on the next one
    P.plan(pos, sv["velocity"], sv["yaw"])            # every env waits after the update above
    moved = pos + np.array([1000.0, 0.0, 0.0])
    d_gain = (np.linalg.norm(pos - P.finish, axis=1) - np.linalg.norm(moved - P.finish, axis=1))
    P.elapsed[:] = P.budget_ticks                     # the next tick times every primitive out
    P.on_tick(moved, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))
    r = np.array([P.buf[i][-1][4] for i in range(n)])
    # exact shaping from an empty bank: gamma * Phi(s') - 0; r_ok 0 (a completion is free)
    assert np.allclose(r, -0.5 + 0.95 * d_gain / 1000.0)
    assert np.allclose(P.bank, d_gain / 1000.0)
    P.plan(moved, sv["velocity"], sv["yaw"])
    # every episode ends on the next tick: a third finish, a third die, a third hit the time cap
    ended = np.ones(n, bool)
    fin = np.arange(n) % 3 == 0
    died = np.arange(n) % 3 == 1
    further = moved + np.array([500.0, 0.0, 0.0])
    P.on_tick(further, ended, fin, died, further)
    r = np.array([P.buf[i][-1][4] for i in range(n)])
    d2 = (np.linalg.norm(moved - P.finish, axis=1) - np.linalg.norm(further - P.finish, axis=1))
    # EVERY end is a terminal at potential 0: the bank is refunded (-Phi(s)), the last
    # primitive's own progress is not paid, and the finish adds its bonus - so the discounted
    # shaping of every episode telescopes to 0 and only reaching the finish counts
    assert np.allclose(r[fin], -0.5 - d_gain[fin] / 1000.0 + 10.0)
    failed = ~fin
    assert np.allclose(r[failed], -0.5 - d_gain[failed] / 1000.0)
    assert (d_gain[died] > 0).any() and (d2[died] > 0).any()   # a dive that would have paid
    assert np.all(P.bank[failed] == 0.0)


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


FLAGS = ["--map", str(LAB), "--reward", "race", "--envs", "64", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32", "--goal-cell", "32",
         "--lidar-range", "11500", "--lidar-near", "2000", "--emb", "64", "--hidden", "64",
         "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
         "--n-steps", "8", "--epochs", "1", "--minibatches", "2", "--ep-ticks", "300",
         "--stall-secs", "30", "--maxvel", "4000", "--yaw-adaptive", "--respawn-frac", "0.9",
         "--respawn-margin", "0.1", "--respawn-reservoir", "1000", "--ckpt-every", "1e9",
         "--record-every", "8192", "--eval-eps", "2", "--eval-greedy-only", "--seed", "7",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold",
         "--goals", "1", "--goal-obs", "fan", "--goal-planner", "primlearn",
         "--goal-reward", "arc", "--goal-fan-offsets", "0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0",
         "--goal-kcap", "1", "--prim-secs", "0.5", "--plan-batch", "64"]


@needs_core
def test_trainer_and_recorder_run_primlearn():
    run = "primlearn_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "24576"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = r.stdout
    assert "LEARNED PRIMITIVES" in out and "EXEC" in out and "PLAN adv" in out, out[-2000:]
    assert "plan-eval finish" in out and "prim planner greedy" in out, out[-2000:]
    assert " upd " in out, out[-2000:]                   # the planner's PPO ran
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["goal_planner"] == "primlearn" and cfg["plan_uniform"] == 0.5
    assert cfg["prim_secs"] == 0.5 and cfg["plan_batch"] == 64
    assert cfg["exec_cut"] == 1                          # primlearn cuts the executor's return
    head = (d / "progress.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "exec/complete" in head and "plan/adv_plan" in head and "plan/ep_prog_start" in head
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["planner"]["primlearn"] and ck["planner"]["updates"] >= 1
    rec = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "2"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    head = json.loads(rec.read_text(encoding="utf-8").splitlines()[0])
    assert head["plan"]["planner"] == "primlearn" and len(head["plan"]["numbers"]) == 6
    # --plan-search: every choice is the best of 4 candidates simulated with the executor
    r3 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "2",
                         "--plan-search", "4"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r3.returncode == 0, r3.stdout[-3000:] + r3.stderr[-3000:]
    assert "search: " in r3.stdout and "candidates died in simulation" in r3.stdout, r3.stdout[-2000:]
    # --plan-mcts: a tree over primitives from the exact state, and its fidelity report
    r4 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "2",
                         "--plan-mcts", "3", "--plan-mcts-k", "3", "--plan-mcts-depth", "2"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r4.returncode == 0, r4.stdout[-3000:] + r4.stderr[-3000:]
    assert "MCTS: 3 expansions" in r4.stdout and "mcts: " in r4.stdout, r4.stdout[-2000:]
    assert "mcts fidelity" in r4.stdout, r4.stdout[-2000:]
    # the search continues the real executor exactly (held action / view / keys / phase,
    # the real observation): the committed primitive's simulated end lands on the real one.
    # A rename of the wrapper's private attributes would silently restart the simulation on
    # released keys and a fresh decision (0/9 with search vs 2/9 without, 82c955a)
    import re as _re
    mm = _re.search(r"matched the real one (\d+)/(\d+)", r4.stdout)
    assert mm and int(mm.group(1)) == int(mm.group(2)), r4.stdout[-2000:]
    me = _re.search(r"simulated end median (\d+) u", r4.stdout)
    assert me is None or int(me.group(1)) <= 16, r4.stdout[-2000:]
    # --plan-mcts-leaf zero + --plan-mcts-dump: an unexpanded, live leaf is worth exactly its own
    # reward (no value head), and every edge carries its planned curve and its flown path
    dump = d / "mcts_dump.json"
    r5 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1",
                         "--plan-mcts", "3", "--plan-mcts-k", "3", "--plan-mcts-depth", "2",
                         "--plan-mcts-leaf", "zero", "--plan-mcts-dump", str(dump)],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r5.returncode == 0, r5.stdout[-3000:] + r5.stderr[-3000:]
    assert "leaf = NOTHING" in r5.stdout and "mcts dump:" in r5.stdout, r5.stdout[-2000:]
    dd = json.loads(dump.read_text(encoding="utf-8"))["decisions"]
    assert dd and all(q_["tree"] for q_ in dd)
    edges = [e for q_ in dd for e in q_["tree"]]
    assert all(e["line"] and e["path"] and len(e["path"][0]) == 3 for e in edges)
    kids = {e["parent"] for e in edges}
    leaves = [e for q_ in dd for e in q_["tree"]
              if e["id"] not in {x["parent"] for x in q_["tree"]} and not e["died"] and not e["fin"]]
    assert all(abs(e["q"] - e["r"]) < 1e-3 for e in leaves)
    assert sum(e["root_choice"] for q_ in dd for e in q_["tree"]) == len(dd) and -1 in kids
    shutil.rmtree(d, ignore_errors=True)


def test_mcts_edges_back_up_by_max():
    """A deterministic model: an edge is worth its BEST continuation (not the mean of the ones
    tried), a terminal edge its own reward, an unexpanded one r + gamma x the value head."""
    from surfgym.goalsearch import _Edge
    g = 0.95
    leaf = _Edge(None, 1.0, 4.0, False, False, None, None, None, 0.0, 10)
    assert np.isclose(leaf.q(g), 1.0 + g * 4.0)
    dead = _Edge(None, -2.0, 99.0, True, False, None, None, None, 0.0, 10)
    fin = _Edge(None, 10.5, 99.0, False, True, None, None, None, 0.0, 10)
    assert dead.term and fin.term and dead.q(g) == -2.0 and fin.q(g) == 10.5
    leaf.child = [dead, fin, _Edge(None, 0.2, 1.0, False, False, None, None, None, 0.0, 5)]
    assert np.isclose(leaf.q(g), 1.0 + g * 10.5)            # the finish below, not the mean


@needs_core
def test_obedience_gate_credits_what_was_flown():
    """--plan-obey: forward progress is credited in proportion to the share of the primitive the
    executor flew (f = arc / 0.9, capped at 1); backward progress counts in full; the row has one
    value per progress.csv column."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goalprimplan import PRIMLEARN_COLS
    n = 8
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64)
    fin = pos.mean(0) + np.array([3000.0, 0.0, 0.0])
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu", finish=fin,
                           bounds=core.map_bounds(), act_every=4,
                           cfg={"plan_uniform": 0.0, "plan_novelty": 0.0, "plan_obey": 1,
                                "plan_shaping": "pbrs"})
    P.request(np.arange(n), pos)
    P.plan(pos, sv["velocity"], sv["yaw"])
    step = np.where((np.arange(n) % 2 == 0)[:, None], [[600.0, 0.0, 0.0]], [[-600.0, 0.0, 0.0]])
    moved = pos + step                                # even envs toward the finish, odd away
    P.track.arc[:] = 0.45 * P.track.total_arc()       # the executor flew half of each primitive
    P.track.advance = lambda o: (np.zeros(n), np.ones(n, bool))   # keep that arc this tick
    P.elapsed[:] = P.budget_ticks
    P.on_tick(moved, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))
    r = np.array([P.buf[i][-1][4] for i in range(n)])
    p = (np.linalg.norm(pos - fin, axis=1) - np.linalg.norm(moved - fin, axis=1)) / 1000.0
    fwd = p > 0
    assert fwd.any() and (~fwd).any()
    assert np.allclose(r[fwd], 0.95 * 0.5 * p[fwd])   # f = 0.45 / 0.9, shaped from bank 0
    assert np.allclose(r[~fwd], 0.95 * p[~fwd])       # backward: in full
    assert np.allclose(P.bank, np.where(fwd, 0.5 * p, p))
    txt, row = P.note_and_row()
    assert len(row) == len(PRIMLEARN_COLS) and "EXEC" in txt and "cmpl" in txt


@needs_core
def test_episodic_coverage_pays_new_ground_to_survivors():
    """--plan-cover C: a planner primitive that ends alive earns C per 128 u cell it added to what
    its episode had visited; a death earns none; ground visited earlier in the episode pays 0."""
    from surfgym.core import SurfCore, SurfEnvConfig
    n = 4
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = np.tile(sv["origin"][0].astype(np.float64), (n, 1))
    lo, hi = (np.asarray(b, np.float64) for b in core.map_bounds())
    d = (lo + hi) / 2.0 - pos[0]
    d[2] = 0.0
    step = 128.0 * d / np.linalg.norm(d)                # toward the box centre: stays inside
    fin = pos[0] + np.array([0.0, 0.0, 5000.0])         # progress is off below anyway
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu", finish=fin,
                           bounds=core.map_bounds(), act_every=4,
                           cfg={"plan_uniform": 0.0, "plan_novelty": 0.0, "plan_cover": 0.1,
                                "plan_progress": 0.0})
    P.request(np.arange(n), pos)
    P.plan(pos, sv["velocity"], sv["yaw"])
    P.track.advance = lambda o: (np.zeros(n), np.ones(n, bool))   # close only at the budget
    no = np.zeros(n, bool)
    p = pos.copy()
    assert np.all((pos[0] + 5 * step > lo) & (pos[0] + 5 * step < hi))
    for _ in range(5):                                  # 5 new 128 u cells
        p = p + step
        P.on_tick(p, no, no, no)
    died = np.array([False, True, False, False])
    P.elapsed[:] = P.budget_ticks
    P.on_tick(p, died, no, died, p)                     # env 1 dies; the rest time out
    r = np.array([P.buf[i][-1][4] for i in range(n)])
    added = P.ep_cov[0] - 0                             # the spawn cell + the new ones
    assert added >= 5 and np.allclose(r[~died], 0.1 * added), (added, r)
    assert r[1] == 0.0
    # back over the same ground: nothing new
    P.plan(p, sv["velocity"][:n], sv["yaw"][:n])
    for _ in range(4):                                  # back over the cells it covered
        p = p - step
        P.on_tick(p, no, no, no)
    P.elapsed[:] = P.budget_ticks
    P.on_tick(p, no, no, no)
    r2 = np.array([P.buf[i][-1][4] for i in (0, 2, 3)])
    assert np.allclose(r2, 0.0), r2


@needs_core
def test_coverage_is_count_weighted_across_episodes():
    """--plan-cover pays C / sqrt(1 + N) per cell, N = episodes that covered it before: a second
    episode over the ground 4 envs covered pays C / sqrt(5) per cell."""
    from surfgym.core import SurfCore, SurfEnvConfig
    n = 4
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = np.tile(sv["origin"][0].astype(np.float64), (n, 1))
    lo, hi = (np.asarray(b, np.float64) for b in core.map_bounds())
    d = (lo + hi) / 2.0 - pos[0]
    d[2] = 0.0
    step = 128.0 * d / np.linalg.norm(d)
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                           finish=pos[0] + np.array([0.0, 0.0, 5000.0]),
                           bounds=core.map_bounds(), act_every=4,
                           cfg={"plan_uniform": 0.0, "plan_novelty": 0.0, "plan_cover": 0.1,
                                "plan_progress": 0.0})
    no = np.zeros(n, bool)
    rewards = []
    for ep in range(2):
        P.request(np.arange(n), pos)
        P.plan(pos, sv["velocity"], sv["yaw"])
        P.track.advance = lambda o: (np.zeros(n), np.ones(n, bool))
        p = pos.copy()
        for _ in range(5):
            p = p + step
            P.on_tick(p, no, no, no)
        P.elapsed[:] = P.budget_ticks
        P.on_tick(p, no, no, no)
        rewards.append(np.array([P.buf[i][-1][4] for i in range(n)]))
        added = P.ep_cov.copy()
    assert np.allclose(rewards[0], 0.1 * added)
    assert np.allclose(rewards[1], 0.1 / np.sqrt(1.0 + n) * added)


@needs_core
def test_shaping_telescopes_to_zero_over_an_episode():
    """Exact potential-based shaping: whatever the primitives do (forward, back) and however the
    episode ends, the planner's discounted progress rewards sum to 0 - only the finish counts."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goallearn import PLAN_GAMMA
    rng = np.random.default_rng(5)
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=1))
    core.reset(0)
    sv = core.states_view
    for ending in ("died", "cap", "finished"):
        pos = sv["origin"].astype(np.float64).copy()
        P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=1), core, 1, "cpu",
                               finish=pos[0] + np.array([3000.0, 500.0, 0.0]),
                               bounds=core.map_bounds(), act_every=4,
                               cfg={"plan_uniform": 0.0, "plan_novelty": 0.0,
                                    "plan_finish_bonus": 0.0, "plan_shaping": "pbrs"})
        P.request(np.arange(1), pos)
        no = np.zeros(1, bool)
        for k in range(4):
            P.plan(pos, sv["velocity"], sv["yaw"])
            pos = pos + rng.normal(0.0, 400.0, (1, 3))
            P.elapsed[:] = P.budget_ticks
            last = k == 3
            end = np.array([last])
            P.on_tick(pos, end, end & (ending == "finished"), end & (ending == "died"), pos)
        r = np.array([t[4] for t in P.buf[0]])
        assert len(r) == 4 and P.buf[0][-1][5]
        assert abs(float(np.sum(r * PLAN_GAMMA ** np.arange(len(r))))) < 1e-9, (ending, r)


def test_return_weights_steer_the_reservoir_draw():
    """--plan-return: RespawnBuffer.build_pool draws reservoir states in proportion to weight_fn;
    a zero-weight state is never drawn, and without weight_fn the draw is the one that shipped."""
    from surfgym.core import STATE_DTYPE
    from surfgym.respawn import RespawnBuffer
    rb = RespawnBuffer(n_envs=2, reservoir=1000, seed=3)
    rows = np.zeros(200, dtype=STATE_DTYPE)
    rows["origin"][:100] = [0.0, 0.0, 0.0]
    rows["origin"][100:] = [5000.0, 0.0, 0.0]
    rb.push_many(rows)
    start = np.zeros(4, dtype=STATE_DTYPE)
    start["origin"][:] = [-1.0, -1.0, -1.0]
    rb.weight_fn = lambda o: (np.asarray(o)[:, 0] > 1000.0).astype(np.float64)
    pool = rb.build_pool(start, pool_size=400, fresh_frac=0.1)
    re = pool[pool["origin"][:, 0] > -0.5]
    assert len(re) > 0 and np.all(re["origin"][:, 0] > 1000.0)
    rb.weight_fn = None
    pool2 = rb.build_pool(start, pool_size=400, fresh_frac=0.1)
    re2 = pool2[pool2["origin"][:, 0] > -0.5]
    assert (re2["origin"][:, 0] < 1000.0).any() and (re2["origin"][:, 0] > 1000.0).any()


def test_flat_lines_measure_the_horizontal_plane():
    """--prim-flat: a horizontal plan says nothing about height. A flat line at the spawn's
    height must keep paying arc progress to an agent 300 u below it (descending a ramp), and
    the fan must show it no vertical offset; with the flag off the 3D corridor rejects it."""
    from surfgym.goalarc import MultiArcProgress
    from surfgym.goals import MultiLine
    line = np.array([[0, 0, 600], [128, 0, 600], [256, 0, 600], [384, 0, 600]], np.float32)
    a3 = MultiArcProgress(1, l_max=8, spacing=128.0, corridor=192.0, window=4)
    af = MultiArcProgress(1, l_max=8, spacing=128.0, corridor=192.0, window=4)
    af.set_flat(True)
    for t in (a3, af):
        t.set_lines(np.array([0]), [line])
    p = np.array([[200.0, 0.0, 300.0]])
    d3, in3 = a3.advance(p)
    df, inf_ = af.advance(p)
    assert not in3[0] and d3[0] == 0.0
    assert inf_[0] and np.isclose(df[0], 200.0)
    assert np.isclose(af.total_arc()[0], 384.0)
    ml = MultiLine(1, l_max=8, spacing=128.0)
    ml.set_flat(True)
    ml.set_lines(np.array([0]), [line])
    f = ml.features_np(np.array([[0.0, 0.0, 300.0]]), np.array([0.0]), np.array([500.0]))
    up = f.reshape(1, -1, 3)[0, :, 2].cpu().numpy()
    fwd = f.reshape(1, -1, 3)[0, :, 0].cpu().numpy()
    assert np.allclose(up, 0.0) and fwd[-1] > 0
    m3 = MultiLine(1, l_max=8, spacing=128.0)
    m3.set_lines(np.array([0]), [line])
    u3 = m3.features_np(np.array([[0.0, 0.0, 300.0]]), np.array([0.0]),
                        np.array([500.0])).reshape(1, -1, 3)[0, :, 2].cpu().numpy()
    assert (u3 > 0).all()


def test_squashed_entropy_penalises_saturation():
    """--plan-ent-squash: the pre-squash entropy + E[log(1 - tanh(u)^2)] is the entropy of the
    SQUASHED action. It peaks near sigma 1 and falls as mass piles at the bounds (a wide spread,
    an off-centre mean), where the pre-squash proxy keeps rising (one dimension: 0.50 / 0.67 /
    0.33 / -1.28 at (mu, sigma) = (0, .5) (0, 1) (0, 1.65) (2, 1.65), checked by quadrature)."""
    import torch
    from surfgym.goalprimplan import mix_entropy, mix_logjac
    gen = torch.Generator().manual_seed(0)
    lg = torch.zeros(1, 1)

    def h(mu, sigma, n=20000):
        m = torch.full((1, 1, 1), float(mu))
        ls = torch.full((1, 1, 1), float(np.log(sigma)))
        return float(mix_entropy(lg, ls)[0] + mix_logjac(lg, m, ls, gen, n=n)[0])

    def exact(mu, sigma):
        # H(tanh u) = H(u) + E[log(1 - tanh(u)^2)], the expectation by quadrature
        u = np.linspace(mu - 12 * sigma, mu + 12 * sigma, 200001)
        pdf = np.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        lj = 2.0 * (np.log(2.0) - u - np.logaddexp(0.0, -2.0 * u))
        return 0.5 * np.log(2 * np.pi * np.e * sigma ** 2) + np.trapz(pdf * lj, u)

    cases = [(0, 0.5), (0, 1.0), (0, 1.65), (2, 1.65)]
    got = [h(m, s_) for m, s_ in cases]
    want = [exact(m, s_) for m, s_ in cases]
    assert np.allclose(got, want, atol=0.03), (got, want)
    assert got[1] > got[2] > got[3]            # past sigma ~1 it falls; an off-centre mean more


def test_map_frame_squash_is_linear_wrapped_and_round_trips():
    """--prim-frame map: the heading dims are LINEAR in u and wrapped (u = 1 is 180 deg), so no
    tanh bound sits on any direction; the vertical dims are squashed as before; unsquash inverts."""
    from surfgym.goalsearch import unsquash
    P = PrimitivePlanner(n_envs=1, frame="map")
    u = np.array([[0.5, 1.0, -1.2, 0.3, -0.4, 2.0]])
    nums = squash(P, u)
    assert np.allclose(nums[0, :3], [90.0, -180.0, 144.0])
    assert np.allclose(nums[0, 3:], squash(PrimitivePlanner(n_envs=1), u)[0, 3:])
    back = squash(P, unsquash(P, nums))
    dh = (back[0, :3] - nums[0, :3] + 180.0) % 360.0 - 180.0
    assert np.allclose(dh, 0.0, atol=1e-9) and np.allclose(back[0, 3:], nums[0, 3:], atol=1e-6)
    # the default frame is untouched: its unsquash is still arctanh of the scaled numbers
    V = PrimitivePlanner(n_envs=1)
    nv = np.array([[90.0, -45.0, 0.0, 30.0, -60.0, 0.0]])
    assert np.allclose(unsquash(V, nv)[0, :3], np.arctanh([0.5, -0.25, 0.0]))


def test_logjac_mask_leaves_the_linear_dims_out():
    """--plan-ent-squash under the map frame: the heading dims are linear, so their mean must not
    move the squashed-action entropy (unmasked, a far heading mean would be 'saturation')."""
    from surfgym.goalprimplan import mix_logjac
    lg = torch.zeros(1, 1)
    ls = torch.zeros(1, 1, 2)
    mask = torch.tensor([0.0, 1.0])

    def lj(mu0, m):
        gen = torch.Generator().manual_seed(3)
        mu = torch.tensor([[[float(mu0), 0.2]]])
        return float(mix_logjac(lg, mu, ls, gen, n=64, mask=m)[0])

    assert lj(0.0, mask) == lj(5.0, mask)
    assert lj(5.0, None) < lj(0.0, None) - 1.0


@needs_core
def test_map_frame_observation_is_in_world_axes():
    """observe(frame='map'): rays at fixed map azimuths, the finish and the velocity in world
    x / y / z. Moving along +x the motion frame IS the world frame, so the two agree except the
    velocity slots (vx, vy instead of the two speeds); moving along +y the map frame's rays and
    finish components do not change at all."""
    from surfgym.core import SurfCore, SurfEnvConfig
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=1))
    core.reset(0)
    pos = core.states_view["origin"].astype(np.float64)
    fin = pos[0] + [2000.0, 1500.0, -100.0]
    c = RayCaster(core)
    vx, vy = np.array([[400.0, 0.0, -50.0]]), np.array([[0.0, 400.0, -50.0]])
    a = observe(c, pos, vx, np.zeros(1), fin, np.zeros(1))
    b = observe(c, pos, vx, np.zeros(1), fin, np.zeros(1), frame="map")
    d = observe(c, pos, vy, np.zeros(1), fin, np.zeros(1), frame="map")
    assert np.allclose(a[:, :N_RAYS + 4], b[:, :N_RAYS + 4], atol=1e-6)
    assert np.allclose(b[0, N_RAYS + 4:N_RAYS + 7], [0.4, 0.0, -0.05], atol=1e-6)
    assert np.allclose(d[:, :N_RAYS + 4], b[:, :N_RAYS + 4], atol=1e-6)
    assert np.allclose(d[0, N_RAYS + 4:N_RAYS + 7], [0.0, 0.4, -0.05], atol=1e-6)
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=1, frame="map"), core, 1, "cpu",
                           finish=fin, bounds=core.map_bounds(), cfg={"plan_ent_squash": 1})
    assert P.frame == "map" and P.sq_mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=1, frame="map"), core, 1, "cpu",
                           finish=fin, bounds=core.map_bounds(), cfg={"plan_fixed": "straight"})


@needs_core
def test_trainer_and_recorder_run_the_map_frame():
    """--prim-frame map end to end: the trainer dumps and prints it, the planner trains, and
    record_ckpt.py mirrors it - greedy, with the search (unsquash, the world-axes observation of
    simulated ends) and with the random override."""
    run = "primlearn_map_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "24576", "--prim-frame", "map",
                        "--plan-ent-squash", "1"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "MAP frame" in r.stdout and " upd " in r.stdout, r.stdout[-2000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["prim_frame"] == "map"
    rec = d / "rec.jsonl"
    for extra in ([], ["--plan-mcts", "3", "--plan-mcts-k", "3", "--plan-mcts-depth", "2",
                       "--plan-mcts-uniform", "0.5"], ["--plan-override", "random"]):
        r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                             str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1"]
                            + extra,
                            capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                            timeout=900, encoding="utf-8", errors="replace")
        assert r2.returncode == 0, (extra, r2.stdout[-3000:] + r2.stderr[-3000:])
        assert "MAP frame" in r2.stdout, r2.stdout[-2000:]
    shutil.rmtree(d, ignore_errors=True)


@needs_core
def test_prev_block_reads_the_previous_primitive_in_the_observation_frame():
    """--plan-prev: observe() appends the previous primitive - has, its numbers (map: the headings
    as cos / sin), its vertical rates / range, its end relative to the agent and its end
    direction (world axes under map, the motion frame else), the share flown, completed; all
    zero without one."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goalprimplan import n_prev
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=1))
    core.reset(0)
    pos = core.states_view["origin"].astype(np.float64)
    fin = pos[0] + [2000.0, 0.0, 0.0]
    c = RayCaster(core)
    vy = np.array([[0.0, 400.0, 0.0]])                        # moving due +y
    prev = {"has": np.array([True]), "nums": np.array([[90.0, 180.0, -90.0, 45.0, -60.0, 0.0]]),
            "end": pos + [300.0, 1200.0, -100.0], "tan": np.array([[0.0, 1.0, 0.0]]),
            "frac": np.array([0.8]), "done": np.array([False])}
    M = PrimitivePlanner(n_envs=1, frame="map")
    x = observe(c, pos, vy, np.zeros(1), fin, np.zeros(1), frame="map", prev=prev, prim=M)
    assert x.shape == (1, N_OBS + n_prev(3)) and n_prev(3) == 18
    f = x[0, N_OBS:]
    assert f[0] == 1.0
    assert np.allclose(f[1:7], [0.0, 1.0, -1.0, 0.0, 0.0, -1.0], atol=1e-6)   # cos/sin 90, 180, -90
    assert np.allclose(f[7:10], [45.0 / 90.0, -60.0 / 120.0, 0.0])
    assert np.allclose(f[10:13], [0.3, 1.2, -0.1], atol=1e-6)                  # world x, y, z
    assert np.allclose(f[13:16], [0.0, 1.0, 0.0]) and np.isclose(f[16], 0.8) and f[17] == 0.0
    # the level frame: the same end seen from the motion frame (forward = +y, left = -x)
    L = PrimitivePlanner(n_envs=1, frame="level")
    y = observe(c, pos, vy, np.zeros(1), fin, np.zeros(1), frame="level", prev=prev, prim=L)
    g = y[0, N_OBS:]
    assert np.allclose(g[1:4], [0.5, 1.0, -0.5]) and np.allclose(g[4:7], 0.0)  # rates / 180
    assert np.allclose(g[10:13], [1.2, -0.3, -0.1], atol=1e-6)
    assert np.allclose(g[13:16], [1.0, 0.0, 0.0], atol=1e-6)
    # none yet: the block is all zero
    none = dict(prev, has=np.array([False]))
    z = observe(c, pos, vy, np.zeros(1), fin, np.zeros(1), frame="map", prev=none, prim=M)
    assert np.all(z[0, N_OBS:] == 0.0)
    assert np.array_equal(z[:, :N_OBS], x[:, :N_OBS])


@needs_core
def test_plan_prev_carries_the_closed_primitive_and_forgets_it_at_an_episode_end():
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goalprimplan import n_prev
    n = 4
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64)
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n, frame="map"), core, n, "cpu",
                           finish=pos[0] + [3000.0, 0.0, 0.0], bounds=core.map_bounds(),
                           act_every=4, cfg={"plan_uniform": 0.0, "plan_prev": 1,
                                             "plan_shaping": "plain"})
    assert P.d_in == N_OBS + n_prev(3) and P.net.body[0].in_features == P.d_in
    P.request(np.arange(n), pos)
    P.plan(pos, sv["velocity"], sv["yaw"])
    assert not P.pv_has.any() and np.all(P.o_x[:, N_OBS:] == 0.0)     # no previous yet
    first = P.cu_nums.copy()
    P.elapsed[:] = P.budget_ticks                                      # time every primitive out
    ended = np.array([False, False, True, True])
    P.on_tick(pos, ended, np.zeros(n, bool), ended.copy(), pos)
    assert P.pv_has.tolist() == [True, True, False, False]
    assert np.array_equal(P.pv_nums[:2], first[:2]) and not P.pv_done.any()
    P.request(np.array([2, 3]), pos[2:])                               # the ended two respawn
    P.plan(pos, sv["velocity"], sv["yaw"])
    f = P.o_x[:, N_OBS:]
    assert np.all(f[:2, 0] == 1.0) and np.all(f[2:] == 0.0)
    h = np.degrees(np.arctan2(f[:2, 2], f[:2, 1]))                     # knot 1 back from cos/sin
    assert np.allclose(((h - first[:2, 0]) + 180.0) % 360.0 - 180.0, 0.0, atol=1e-3)
    sd = P.state_dict_all()
    Q = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n, frame="map"), core, n, "cpu",
                           finish=P.finish, bounds=core.map_bounds(), cfg={"plan_prev": 0})
    with pytest.raises(ValueError):
        Q.load_state_dict_all(sd)                                      # the width differs
    with pytest.raises(ValueError):
        PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                           finish=P.finish, bounds=core.map_bounds(),
                           cfg={"plan_prev": 1, "plan_az": 0.5})


@needs_core
def test_trainer_and_recorder_run_plan_prev():
    """--plan-prev end to end (map frame): the trainer dumps and prints it, the planner trains,
    record_ckpt.py mirrors it (greedy), and the search refuses it cleanly; the trainer refuses
    the combinations that do not carry the previous primitive."""
    run = "primlearn_prev_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "24576", "--prim-frame", "map",
                        "--plan-ent-squash", "1", "--plan-prev", "1", "--prim-pitch-max", "30",
                        "--plan-replan", "0.5", "--plan-smdp", "1"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "PREVIOUS primitive in the observation" in r.stdout and " upd " in r.stdout, \
        r.stdout[-2000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["plan_prev"] == 1 and cfg["prim_frame"] == "map"
    # --prim-pitch-max / --plan-replan ride along: dumped, printed and mirrored below
    assert cfg["prim_pitch_max"] == 30.0 and cfg["plan_replan"] == 0.5
    assert "closes on arc >= 0.45" in r.stdout, r.stdout[-2000:]
    rec = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "2"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    r3 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1",
                         "--plan-mcts", "2"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r3.returncode != 0 and "--plan-prev" in (r3.stdout + r3.stderr)
    shutil.rmtree(d, ignore_errors=True)
    for extra in (["--plan-az", "0.5"], ["--plan-cap", "bootstrap", "--plan-shaping", "refund"],
                  ["--plan-joint", "1"]):
        rr = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                             "--run", run + "_refuse", "--steps", "4096", "--plan-prev", "1"]
                            + FLAGS + extra,
                            capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                            timeout=600, encoding="utf-8", errors="replace")
        assert rr.returncode != 0 and "--plan-prev with" in (rr.stdout + rr.stderr), \
            (extra, rr.stdout[-1500:] + rr.stderr[-1500:])
    shutil.rmtree(ROOT / "runs" / (run + "_refuse"), ignore_errors=True)


@needs_core
def test_plan_replan_closes_at_half_the_arc_and_half_the_budget():
    """--plan-replan 0.5: a primitive closes after half its completion arc or half its time
    budget - the curve keeps its length; 1 is today's rule."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goallearn import COMPLETE_FRAC
    n = 2
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64)
    fin = pos[0] + [3000.0, 0.0, 0.0]

    def planner(f):
        return PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                                  finish=fin, bounds=core.map_bounds(), act_every=4,
                                  cfg={"plan_uniform": 0.0, "plan_shaping": "plain",
                                       "plan_replan": f})
    A, B = planner(1.0), planner(0.5)
    assert A.close_frac == COMPLETE_FRAC and B.close_frac == 0.5 * COMPLETE_FRAC
    assert A.budget_ticks == 75 and B.budget_ticks == 38          # ceil(1.5 x 0.5 s x F / 10 ms)
    for P in (A, B):
        P.request(np.arange(n), pos)
        P.plan(pos, sv["velocity"], sv["yaw"])
        assert P.ncurve == 51                                      # the curve keeps 0.5 s
    # standing still: B's primitives time out at tick 38, A's are still open
    for _ in range(38):
        for P in (A, B):
            P.on_tick(pos, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))
    assert B.need.all() and not A.need.any()
    with pytest.raises(ValueError):
        planner(0.0)


@needs_core
def test_mcts_edge_reward_follows_the_checkpoints_shaping():
    """The search scores edges with the PLANNER's reward: record_ckpt.py mirrors --plan-shaping
    into the planner it builds, so a plain checkpoint's tree keeps a death's progress (it once
    fell back to the refund rule and charged every death the episode's bank)."""
    run = "primlearn_plain_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "12288", "--plan-shaping", "plain"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = ROOT / "runs" / run
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(d / "rec.jsonl"), "--episodes",
                         "1", "--plan-mcts", "2", "--plan-mcts-k", "3", "--plan-mcts-depth", "2"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "a death keeps the progress it made" in r2.stdout, r2.stdout[-2000:]
    # the overrides: the planner proposes nothing, and the refund rule is forced on this plain
    # checkpoint
    r3 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(d / "rec.jsonl"), "--episodes",
                         "1", "--plan-mcts", "2", "--plan-mcts-k", "3", "--plan-mcts-depth", "2",
                         "--plan-mcts-no-planner", "--plan-mcts-leaf", "zero",
                         "--plan-mcts-reward", "refund"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r3.returncode == 0, r3.stdout[-3000:] + r3.stderr[-3000:]
    assert "UNIFORM primitives" in r3.stdout and "failed-end refund" in r3.stdout,         r3.stdout[-2000:]
    shutil.rmtree(d, ignore_errors=True)


@needs_core
def test_plan_reset_actor_reinitialises_only_the_actor_heads():
    """--plan-reset-actor: at load, the mixture heads go back to their initialisation; the body and
    the value head keep the loaded weights; without the flag the load is exact."""
    from surfgym.core import SurfCore, SurfEnvConfig
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=1))
    core.reset(0)
    fin = core.states_view["origin"][0].astype(np.float64) + [2000.0, 0.0, 0.0]

    def planner(reset):
        return PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=1), core, 1, "cpu", finish=fin,
                                  bounds=core.map_bounds(), cfg={"plan_reset_actor": reset})
    A = planner(0)
    with torch.no_grad():
        for prm in A.net.parameters():
            prm.add_(torch.randn_like(prm) * 0.3)          # a "trained" planner
    sd = A.state_dict_all()
    B, C = planner(0), planner(1)
    B.load_state_dict_all(sd)
    C.load_state_dict_all(sd)
    fresh = planner(0)
    for (n, a), (_, b), (_, c), (_, f) in zip(A.net.named_parameters(), B.net.named_parameters(),
                                              C.net.named_parameters(), fresh.net.named_parameters()):
        assert torch.equal(a, b), n                        # no flag: exact
        if n.startswith(("logits", "mu", "log_std")):
            if n.endswith("bias"):
                assert torch.equal(c, f), n                # the heads' biases back to init
            else:
                assert not torch.equal(c, a) and float(c.norm()) < float(a.norm()), n
        else:
            assert torch.equal(c, a), n                    # body and value head kept


def test_plan_gae_smdp_discount_and_truncation_bootstrap():
    """plan_gae's per-plan discount (--plan-smdp) and the time cap's bootstrap (--plan-cap
    bootstrap): a truncated plan bootstraps V(s_T) instead of 0 and nothing crosses the episode
    boundary; with neither, the output is the one that shipped."""
    from surfgym.goallearn import plan_gae
    r, v, d = [1.0, 2.0, 0.5], [0.3, 0.2, 0.1], [False, True, False]
    a0, _ = plan_gae(r, v, d, 0.7, gamma=0.9, lam=0.8)
    a1, _ = plan_gae(r, v, d, 0.7, gamma=0.9, lam=0.8, gammas=[0.9, 0.9, 0.9],
                     boots=[None, None, None])
    assert np.allclose(a0, a1)
    g = [0.9, 0.81, 0.95]
    a2, ret2 = plan_gae(r, v, d, 0.7, gamma=0.9, lam=0.8, gammas=g, boots=[None, 4.0, None])
    d2 = 0.5 + 0.95 * 0.7 - 0.1                    # plan 2 bootstraps the open plan
    d1 = 2.0 + 0.81 * 4.0 - 0.2                    # plan 1: truncated, V(s_T) = 4
    d0 = 1.0 + 0.9 * 0.2 - 0.3                     # plan 0 -> plan 1 inside the episode
    assert np.allclose(a2, [d0 + 0.9 * 0.8 * d1, d1, d2])
    assert np.allclose(ret2, a2 + np.asarray(v))


def test_cap_bootstrap_keeps_the_bank_and_bootstraps():
    """--plan-cap bootstrap: an episode stopped by the time cap refunds nothing and its last
    transition carries V(s_T) as a bootstrap (terminal for the recursion); a death still refunds
    the bank. The default (refund) charges the bank at the cap."""
    from surfgym.core import SurfCore, SurfEnvConfig
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=1))
    core.reset(0)
    sv = core.states_view
    out = {}
    for cap in ("refund", "bootstrap"):
        for ending in ("cap", "died"):
            pos = sv["origin"].astype(np.float64).copy()
            fin = pos[0] + np.array([3000.0, 0.0, 0.0])
            P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=1), core, 1, "cpu",
                                   finish=fin, bounds=core.map_bounds(), act_every=4,
                                   cfg={"plan_uniform": 0.0, "plan_novelty": 0.0,
                                        "plan_finish_bonus": 0.0, "plan_shaping": "refund",
                                        "plan_cap": cap})
            P.request(np.arange(1), pos)
            for k in range(3):
                P.plan(pos, sv["velocity"], sv["yaw"])
                pos = pos + np.array([[200.0, 0.0, 0.0]])     # 200 u closer every primitive
                P.elapsed[:] = P.budget_ticks
                end = np.array([k == 2])
                P.on_tick(pos, end, np.zeros(1, bool), end & (ending == "died"), pos,
                          term_vel=np.zeros((1, 3)), term_yaw=np.zeros(1))
            out[(cap, ending)] = list(P.buf[0])
    for cap in ("refund", "bootstrap"):
        died = out[(cap, "died")]
        assert died[-1][5] and died[-1][7] is None
        assert np.isclose(died[-1][4], -0.4)                # the bank (2 x 0.2) refunded
    capped = out[("refund", "cap")]
    assert np.isclose(capped[-1][4], -0.4) and capped[-1][7] is None
    boot = out[("bootstrap", "cap")]
    assert boot[-1][5] and boot[-1][7] is not None          # truncated: V(s_T) bootstrapped
    assert np.isclose(boot[-1][4], 0.2)                     # its own progress, no refund
    assert all(len(t) == 8 and t[6] == boot[0][6] for t in boot)


def test_route_units_and_the_start_decision():
    """--plan-units route: the map start's route pays ROUTE_PAY whatever its length (the bank the
    planner sees stays inside the observation's clip on a long map); --plan-uniform-start 0: a
    map-start spawn never opens with a uniform primitive."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goalprimplan import ROUTE_PAY
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=4))
    core.reset(0)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64).copy()
    for d in (3000.0, 20000.0):
        fin = pos[0] + np.array([d, 0.0, 0.0])
        P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=4), core, 4, "cpu",
                               finish=fin, bounds=core.map_bounds(), act_every=4,
                               start_pts=pos[:1], cfg={"plan_units": "route",
                                                       "plan_uniform": 1.0,
                                                       "plan_uniform_start": 0})
        assert np.isclose(d / P.unit, ROUTE_PAY)
        P.request(np.arange(4), np.vstack([pos[:1], pos[:1] + 500.0, pos[:1] + 900.0,
                                           pos[:1]]))
        P.plan(pos, sv["velocity"], sv["yaw"])
        # uniform share 1.0: every fresh episode opens uniform - except the two map-start ones
        assert P.decided.tolist() == [True, False, False, True]


@needs_core
def test_trainer_runs_the_review_fixes():
    """The adversarial review's fixes on, together: --plan-ent-squash, --plan-smdp, --plan-cap
    bootstrap (3 s episodes: the cap fires, and goalsys hands the planner the terminal velocity
    and heading), --plan-units route, --plan-uniform-start 0. The config records each; the
    recorder mirrors the unit and records."""
    run = "primlearn_v2_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    extra = ["--plan-shaping", "refund", "--plan-cap", "bootstrap", "--plan-smdp", "1",
             "--plan-ent-squash", "1", "--plan-units", "route", "--plan-uniform-start", "0"]
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "24576"] + FLAGS + extra,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = r.stdout
    assert "SQUASHED action" in out and "SMDP discount" in out and "TRUNCATION" in out, out[-3000:]
    assert " upd " in out, out[-2000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["plan_cap"] == "bootstrap" and cfg["plan_smdp"] == 1
    assert cfg["plan_ent_squash"] == 1 and cfg["plan_units"] == "route"
    assert cfg["plan_uniform_start"] == 0
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["planner"]["unit"] != 1000.0
    rec = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    shutil.rmtree(d, ignore_errors=True)


@needs_core
def test_plan_fixed_is_a_no_planner_control():
    """--plan-fixed straight: every primitive is straight along the motion, the planner collects
    nothing and never updates, and the recorder's eval uses the same rule."""
    run = "primlearn_fixed_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "24576", "--plan-fixed", "straight"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["plan_fixed"] == "straight"
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["planner"]["updates"] == 0
    rec = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    head = json.loads(rec.read_text(encoding="utf-8").splitlines()[0])
    assert all(abs(z) < 1e-6 for z in head["plan"]["numbers"])      # straight = all zeros
    shutil.rmtree(d, ignore_errors=True)


def test_refund_with_interest_nets_every_failure_to_zero():
    """--plan-shaping refund_i: the bank grows by 1/gamma per primitive, so a failed episode's
    DISCOUNTED planner return is exactly 0 however late it fails (death or cap) and whichever way
    it moved - hiding pays nothing - while a finish keeps its credits. Also under --plan-smdp,
    where each primitive's gamma follows its duration."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goallearn import PLAN_GAMMA
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=1))
    core.reset(0)
    sv = core.states_view
    for smdp in (0, 1):
        for ending, n_prim in (("died", 2), ("cap", 5), ("finished", 4), ("late_cap", 3)):
            pos = sv["origin"].astype(np.float64).copy()
            fin = pos[0] + np.array([4000.0, 0.0, 0.0])
            P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=1), core, 1, "cpu",
                                   finish=fin, bounds=core.map_bounds(), act_every=4,
                                   cfg={"plan_uniform": 0.0, "plan_novelty": 0.0,
                                        "plan_finish_bonus": 0.0, "plan_shaping": "refund_i",
                                        "plan_smdp": smdp})
            P.request(np.arange(1), pos)
            steps = [150.0, -80.0, 220.0, 60.0, 90.0]
            durs = [P.budget_ticks, P.budget_ticks // 2, P.budget_ticks, 7, P.budget_ticks]
            for k in range(n_prim):
                P.plan(pos, sv["velocity"], sv["yaw"])
                pos = pos + np.array([[steps[k], 0.0, 0.0]])
                P.elapsed[:] = durs[k]
                last = k == n_prim - 1
                if ending == "late_cap" and last:
                    # this primitive closes ALIVE; the cap then ends the episode while the env
                    # waits for its next decision (the late path)
                    P.on_tick(pos, np.zeros(1, bool), np.zeros(1, bool), np.zeros(1, bool), pos)
                    P.on_tick(pos, np.ones(1, bool), np.zeros(1, bool), np.zeros(1, bool), pos)
                    break
                end = np.array([last])
                P.on_tick(pos, end, end & (ending == "finished"), end & (ending == "died"), pos)
            b = P.buf[0]
            r = np.array([t[4] for t in b])
            g = np.array([(PLAN_GAMMA ** (t[6] / P.nominal_ticks)) if smdp else PLAN_GAMMA
                          for t in b])
            disc = np.concatenate([[1.0], np.cumprod(g)[:-1]])
            ret = float(np.sum(r * disc))
            assert b[-1][5], (smdp, ending)
            if ending == "finished":
                assert ret > 0.1, (smdp, ending, r)
            else:
                assert abs(ret) < 1e-9, (smdp, ending, r, ret)


@needs_core
def test_joint_transitions_are_the_executors_reward():
    """--plan-joint 1: a planner transition's reward is the executor's per-tick rewards handed to
    add_reward, discounted by the executor's per-tick gamma from the primitive's start tick and
    summed until the next decision (the ticks the env waits for it included) or the episode's end;
    its duration counts those ticks and update() discounts it gamma ** ticks (an SMDP). No refund,
    progress, r_fail, finish bonus, novelty or coverage of the planner's own - all switched on here
    and all absent; the bank stays an observation; only the time cap bootstraps V(s_T)."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goallearn import plan_gae
    from surfgym.tick import TickClock
    from surfgym.goalprimplan import JOINT_REWARD_SCALE
    n = 4
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = np.tile(sv["origin"][0].astype(np.float64), (n, 1))
    lo, hi = (np.asarray(b, np.float64) for b in core.map_bounds())
    d = (lo + hi) / 2.0 - pos[0]
    d[2] = 0.0
    step = 64.0 * d / np.linalg.norm(d)                 # toward the box centre: new cells, inside
    assert np.all((pos[0] + 13 * step > lo) & (pos[0] + 13 * step < hi))
    g = 0.9995
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                           finish=pos[0] + 40.0 * step, bounds=core.map_bounds(), act_every=4,
                           exec_gamma=g,
                           cfg={"plan_joint": 1, "plan_uniform": 0.0, "plan_novelty": 0.5,
                                "plan_cover": 0.1, "plan_finish_bonus": 10.0,
                                "plan_r_fail": -0.5, "plan_shaping": "refund"})
    assert P.joint and P.gamma_tick == g and "JOINT with the executor" in P.describe()
    rng = np.random.default_rng(11)
    no = np.zeros(n, bool)
    handed = [[] for _ in range(n)]

    def tick(p, ended=no, finished=no, died=no, term=None):
        P.on_tick(p, ended, finished, died, term, term_vel=np.zeros((n, 3)),
                  term_yaw=np.zeros(n))
        r = rng.normal(0.0, 1.0, n)                     # the executor's reward for this tick
        P.add_reward(r)
        for i in range(n):
            handed[i].append(float(r[i]))

    P.request(np.arange(n), pos)
    P.plan(pos, sv["velocity"], sv["yaw"])
    assert P.j_act.all()
    # primitive 1: 10 ticks toward the finish; the last one times every primitive out
    p = pos.copy()
    for k in range(10):
        p = p + step
        if k == 9:
            P.elapsed[:] = P.budget_ticks
        tick(p)
    assert P.n_ready() == 0 and not P.active.any()     # closed, not pushed: open until a decision
    assert np.all(P.bank > 0.0)                         # progress still banked - an observation
    for _ in range(2):                                  # the rest of the executor's decision
        tick(p)
    P.plan(p, sv["velocity"], sv["yaw"])                # the next decision pushes it
    assert P.n_ready() == n
    for i in range(n):
        t = P.buf[i][-1]
        want = JOINT_REWARD_SCALE * sum(g ** j * x for j, x in enumerate(handed[i]))
        assert np.isclose(t[4], want) and not t[5] and t[6] == 12 and t[7] is None, (i, t[4:])
    # primitive 2: the episode ends on its 3rd tick - env 0 finishes, env 1 dies, env 2 hits the
    # time cap; env 3's primitive timed out on tick 1 and it dies on tick 3 while waiting
    handed = [[] for _ in range(n)]
    ids = np.arange(n)
    for k in range(3):
        p = p + step
        if k == 0:
            P.elapsed[3] = P.budget_ticks
        end = np.full(n, k == 2)
        tick(p, end, end & (ids == 0), end & np.isin(ids, (1, 3)), p)
    assert P.n_ready() == 2 * n and not P.j_act.any()
    for i in range(n):
        t = P.buf[i][-1]
        want = JOINT_REWARD_SCALE * sum(g ** j * x for j, x in enumerate(handed[i]))
        assert np.isclose(t[4], want), (i, t[4], want)  # no refund / bonus / novelty / coverage
        assert t[5] and t[6] == 3
        assert (t[7] is not None) == (i == 2)           # only the time cap bootstraps V(s_T)
    # update(): transition k is discounted g ** its ticks (the SMDP on the executor's clock)
    rets = []
    for i in range(n):
        b = P.buf[i]
        rets.append(plan_gae([t[4] for t in b], [t[3] for t in b], [t[5] for t in b], 0.0,
                             gammas=[g ** t[6] for t in b], boots=[t[7] for t in b])[1])
    u = P.update(force=True)
    assert u is not None and u["n"] == 2 * n
    assert np.isclose(u["ret_mean"], float(np.concatenate(rets).mean()), rtol=1e-5, atol=1e-6)
    # another tick: the executor's own conversion (TickClock.gamma), exactly
    tc = TickClock(7.63)
    P.set_tick_ms(tc.ms)
    assert P.gamma_tick == tc.gamma(g)


@needs_core
def test_trainer_runs_plan_joint():
    """--plan-joint 1 (docs/planner-design.md section 9): the executor trains on the race reward
    toward the finish (no goal arc is built, no --exec-cut) and the planner on the same reward
    summed over its primitives. The config records the flag, the planner updates, the recorder
    records the checkpoint - and refuses a search that would score it in the planner's units."""
    run = "primlearn_joint_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "24576", "--plan-joint", "1"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = r.stdout
    assert "JOINT with the executor" in out and "RACE reward toward the finish" in out, out[-3000:]
    assert "goal arc shaping scale" not in out, out[-3000:]      # no arc pay is built
    assert " upd " in out, out[-2000:]                           # the planner's PPO ran
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["plan_joint"] == 1 and "exec_cut" not in cfg and cfg["race_arc"] is None
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["planner"]["cfg"]["plan_joint"] == 1 and ck["planner"]["updates"] >= 1
    rec = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    head = json.loads(rec.read_text(encoding="utf-8").splitlines()[0])
    assert head["plan"]["planner"] == "primlearn"
    r3 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1",
                         "--plan-search", "4"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r3.returncode != 0 and "--plan-joint checkpoint" in r3.stdout + r3.stderr, \
        r3.stdout[-2000:] + r3.stderr[-2000:]


# ==========================================================================================
# (d) --plan-az: AlphaZero-style expert iteration (tools/az_worker.py -> the planner's update)
# ==========================================================================================
def _az_logp(net, x, u):
    """log pi_theta(u_i | x) of every candidate, (B, K)."""
    with torch.no_grad():
        lg, mu, ls, v = net(x)
        b, k, d = u.shape
        lp = mix_logp(lg.repeat_interleave(k, 0), mu.repeat_interleave(k, 0),
                      ls.repeat_interleave(k, 0), u.reshape(b * k, d)).view(b, k)
    return lp, v


def test_az_policy_target_pulls_toward_the_most_visited_candidates():
    """The policy term -sum_i pi_i log pi_theta(u_i | x): a few steps on it raise the log density
    of the candidate the search visited most, relative to the ones it visited less; a padded
    candidate (mask False, pi 0) changes neither term; the value term pulls V(x) toward z."""
    from surfgym.goalprimplan import PrimPlannerNet, az_losses
    torch.manual_seed(0)
    net = PrimPlannerNet(N_OBS, 6)
    g = torch.Generator().manual_seed(1)
    b = 8
    x = torch.rand(b, N_OBS, generator=g)
    u = 0.7 * torch.randn(b, 3, 6, generator=g)
    pi = torch.tensor([[0.8, 0.15, 0.05]]).repeat(b, 1)
    mask = torch.ones(b, 3, dtype=torch.bool)
    z = torch.full((b,), 2.0)
    lp0, v0 = _az_logp(net, x, u)
    a_pi, a_v = az_losses(net, x, u, pi, mask, z)
    assert torch.isclose(a_pi, -(pi * lp0).sum(1).mean(), atol=1e-5)
    assert torch.isclose(a_v, ((v0 - z) ** 2).mean(), atol=1e-5)
    u4 = torch.cat([u, torch.full((b, 1, 6), 3.0)], 1)
    pi4 = torch.cat([pi, torch.zeros(b, 1)], 1)
    m4 = torch.cat([mask, torch.zeros(b, 1, dtype=torch.bool)], 1)
    p4, v4 = az_losses(net, x, u4, pi4, m4, z)
    assert torch.allclose(a_pi, p4) and torch.allclose(a_v, v4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(60):
        l_pi, l_v = az_losses(net, x, u, pi, mask, z)
        opt.zero_grad()
        (l_pi + l_v).backward()
        opt.step()
    lp1, v1 = _az_logp(net, x, u)
    gain = (lp1 - lp0).numpy()
    assert (gain[:, 0] > gain[:, 2]).all(), gain          # every target: most- over least-visited
    assert gain[:, 0].mean() > gain[:, 1].mean() > gain[:, 2].mean(), gain.mean(0)
    assert float(l_pi.detach()) < float(a_pi.detach())
    assert ((v1 - z).abs() < (v0 - z).abs()).all()


def test_az_replay_reads_each_target_once_and_keeps_the_newest(tmp_path):
    """The trainer's replay of tools/az_worker.py's files: each file is read once; a malformed
    file, a half-written temp file and a foreign file are skipped; the FIFO keeps the newest
    ``cap``; a minibatch pads every target to the batch's largest K (pi 0, masked)."""
    import os
    from surfgym.goalprimplan import AZReplay
    d = tmp_path / "az"
    d.mkdir()
    rng = np.random.default_rng(0)
    clock = [1_700_000_000_000_000_000]

    def put(name, k=3, **over):
        t = {"x": rng.random(N_OBS).astype(np.float32),
             "u": rng.normal(size=(k, 6)).astype(np.float32),
             "pi": np.full(k, 1.0 / k, np.float32), "z": np.float32(0.5)}
        t.update(over)
        np.savez(d / name, **t)
        clock[0] += 10_000_000                        # distinct, increasing mtimes
        os.utime(d / name, ns=(clock[0], clock[0]))

    put("targets_a_1_000001.npz", k=2)
    put("targets_a_1_000002.npz", k=4)
    put("targets_a_1_000003.npz", x=np.zeros(5, np.float32))      # malformed: x too short
    (d / "targets_a_1_000004.npz.tmp").write_bytes(b"half a file")  # a write in progress
    (d / "notes.txt").write_text("x", encoding="utf-8")
    R = AZReplay(d, 6, cap=2, seed=0)
    assert R.load_new() == 2 and len(R) == 2 and R.bad == 1
    assert [len(r[2]) for r in R.rows] == [2, 4]                   # oldest first
    assert R.load_new() == 0 and R.bad == 1                         # each file once
    put("targets_a_1_000005.npz", k=3, z=np.float32(-1.0))
    assert R.load_new() == 1 and len(R) == 2
    assert [len(r[2]) for r in R.rows] == [4, 3]                   # FIFO: the newest two
    x, u, pi, m, z = R.sample(32, "cpu")
    assert x.shape == (32, N_OBS) and u.shape == (32, 4, 6) and pi.shape == m.shape == (32, 4)
    assert torch.allclose(pi.sum(1), torch.ones(32))
    three = m.sum(1) == 3
    assert three.any() and (~three).any()
    assert torch.equal(three, pi[:, 3] == 0.0) and torch.equal(three, z == -1.0)


@needs_core
def test_az_update_off_unchanged_on_fits_and_only_drops_the_surrogate(tmp_path, monkeypatch):
    """PrimLearnedPlanner.update with search targets on disk: --plan-az 0 is the update that
    shipped even with a replay attached; --plan-az 1 reads the 6 targets, fits them (finite terms,
    other weights) and logs them on the planner's line and progress.csv row; --plan-az-only drops
    the PPO surrogate - with the AlphaZero terms zeroed (or no target on disk yet) and no entropy
    bonus the policy heads do not move at all, while the plain --plan-az update moves them."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym import goalprimplan as gp
    n = 8
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64)
    fin = pos[0] + np.array([3000.0, 0.0, 0.0])
    d = tmp_path / "az"
    d.mkdir()
    rng = np.random.default_rng(3)
    for i in range(6):
        k = 2 + i % 3
        w = rng.random(k).astype(np.float32)
        np.savez(d / f"targets_t_0_{i:06d}.npz", x=rng.random(N_OBS).astype(np.float32),
                 u=rng.normal(size=(k, 6)).astype(np.float32), pi=w / w.sum(),
                 z=np.float32(rng.normal()))

    def run(cfg, attach=True, path=d):
        torch.manual_seed(0)
        P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu", finish=fin,
                               bounds=core.map_bounds(), act_every=4,
                               cfg={"plan_uniform": 0.0, "plan_batch": 8, "plan_novelty": 0.0,
                                    **cfg}, seed=0)
        if attach:
            P.attach_az(path, seed=1)
        P.request(np.arange(n), pos)
        P.plan(pos, sv["velocity"], sv["yaw"])
        for _ in range(P.budget_ticks + 1):
            P.on_tick(pos, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))
        P.plan(pos, sv["velocity"], sv["yaw"])
        return P, P.update()

    P0, u0 = run({}, attach=False)
    P1, u1 = run({})
    assert u0 is not None and "az_targets" not in u1
    assert all(torch.equal(a, b) for a, b in zip(P0.net.parameters(), P1.net.parameters()))
    P2, u2 = run({"plan_az": 1.0})
    assert u2["az_targets"] == 6 and u2["az_new"] == 6
    assert np.isfinite(u2["az_pi"]) and np.isfinite(u2["az_v"])
    assert any(not torch.equal(a, b) for a, b in zip(P0.net.parameters(), P2.net.parameters()))
    txt, row = P2.note_and_row()
    assert len(row) == len(gp.PRIMLEARN_COLS) and " az 6 (+6) pi " in txt, txt
    assert gp.PRIMLEARN_COLS[-4:] == ["plan/az_targets", "plan/az_new", "plan/az_loss_pi",
                                      "plan/az_loss_v"]
    assert row[-4:-2] == [6, 6] and all(isinstance(v, float) for v in row[-2:])
    _, row0 = P0.note_and_row()
    assert row0[-4:] == ["", "", "", ""]

    def zero_az(net, x, u, pi, mask, z):
        return torch.zeros(()), torch.zeros(())
    monkeypatch.setattr(gp, "az_losses", zero_az)
    heads = ("logits", "mu", "log_std")
    empty = tmp_path / "empty"
    empty.mkdir()
    for only, moves, path in ((1, False, d), (1, False, empty), (0, True, d)):
        Pa, ua = run({"plan_az": 1.0, "plan_az_only": only, "plan_ent": 0.0}, path=path)
        assert ua["az_targets"] == (0 if path == empty else 6)
        torch.manual_seed(0)
        fresh = gp.PrimPlannerNet(N_OBS, 6)          # the planner's initial weights (same seed)
        same = all(torch.equal(getattr(Pa.net, h).weight, getattr(fresh, h).weight)
                   and torch.equal(getattr(Pa.net, h).bias, getattr(fresh, h).bias)
                   for h in heads)
        assert Pa.updates == 1 and same != moves, (only, moves)
        assert not torch.equal(Pa.net.v.weight, fresh.v.weight)    # the value still regresses


@needs_core
def test_az_worker_targets_feed_the_trainer():
    """tools/az_worker.py on a trained checkpoint writes one well-formed target per search (the
    root's planner observation with bank 0, every root candidate, visit fractions summing to 1,
    z = the visit-weighted Q); the trainer resumed with --plan-az reads them, logs finite AlphaZero
    terms on its planner line and in progress.csv, records the flag, restores it on a bare resume,
    and the recorder still records the checkpoint (TRAIN_ONLY)."""
    import csv
    run = "primlearn_az_smoke"
    d = ROOT / "runs" / run
    shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "24576"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    w = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "az_worker.py"), str(d),
                        "--ckpt", str(d / "ckpt_final.pt"), "--searches", "3", "--sims", "2",
                        "--k", "2", "--device", "cpu", "--settle", "0", "--seed", "3"],
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=900, encoding="utf-8", errors="replace")
    assert w.returncode == 0, w.stdout[-3000:] + w.stderr[-3000:]
    assert "roots = 25% map starts + 75% of the reservoir" in w.stdout, w.stdout[-2000:]
    assert "stop (3 searches)" in w.stdout, w.stdout[-2000:]
    files = sorted((d / "az").glob("targets_*.npz"))
    assert len(files) == 3 and not list((d / "az").glob("*.tmp"))
    for p in files:
        with np.load(p) as f:
            x, u, pi, q, z = f["x"], f["u"], f["pi"], f["q"], float(f["z"])
            assert x.shape == (N_OBS,) and u.ndim == 2 and u.shape[1] == 6 and len(u) >= 2
            assert pi.shape == q.shape == (len(u),) and np.isfinite(u).all()
            assert (pi >= 0).all() and np.isclose(pi.sum(), 1.0)
            assert np.isfinite(z) and np.isclose(z, float((pi * q).sum()), atol=1e-5)
            assert (x[:N_RAYS] >= 0).all() and (x[:N_RAYS] <= 1.0 + 1e-6).all()
            assert x[N_OBS - 1] == 0.0                       # bank 0: a fresh root
            assert int(f["step"]) == 24576 and int(f["sims"]) == 2
    # the trainer, resumed with --plan-az, fits them
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                         "--run", run, "--ckpt", str(d / "ckpt_final.pt"), "--steps", "40960",
                         "--plan-az", "1.0"] + FLAGS,
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=1800, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "planner: --plan-az 1 - AlphaZero targets" in r2.stdout, r2.stdout[-3000:]
    assert " az 3 (+3) pi " in r2.stdout, r2.stdout[-3000:]
    with open(d / "progress.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    az = [x for x in rows if x["plan/az_targets"] != ""]
    assert az and int(az[0]["plan/az_targets"]) == 3 and int(az[0]["plan/az_new"]) == 3
    for x in az:
        assert np.isfinite(float(x["plan/az_loss_pi"])) and np.isfinite(float(x["plan/az_loss_v"]))
    assert all(x["plan/az_targets"] == "" for x in rows
               if int(x["time/total_timesteps"]) <= 24576)       # the run before the flag
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["plan_az"] == 1.0 and "plan_az_only" not in cfg
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["plan_az"] == 1.0 and ck["planner"]["cfg"]["plan_az"] == 1.0
    # a bare resume keeps it on (restored like every planner knob)
    r3 = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                         "--run", run, "--ckpt", str(d / "ckpt_final.pt"), "--steps", "43008"]
                        + FLAGS, capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=1800, encoding="utf-8", errors="replace")
    assert r3.returncode == 0, r3.stdout[-3000:] + r3.stderr[-3000:]
    assert "planner: --plan-az 1 - AlphaZero targets" in r3.stdout, r3.stdout[-3000:]
    rec = d / "rec.jsonl"
    r4 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r4.returncode == 0, r4.stdout[-3000:] + r4.stderr[-3000:]
    shutil.rmtree(d, ignore_errors=True)


@needs_core
def test_plan_joint_refusals():
    """--plan-joint keeps ONE return: an explicit --exec-cut 1 is refused, and so is
    --reward-per-decision (the planner is handed the executor's reward tick by tick)."""
    run = "primlearn_joint_bad"
    base = [sys.executable, "-u", str(ROOT / "python" / "train_fast.py"), "--run", run,
            "--steps", "2048", "--plan-joint", "1"] + FLAGS
    for extra, msg in ((["--exec-cut", "1"], "--exec-cut 1 with --plan-joint 1"),
                       (["--reward-per-decision"], "--plan-joint hands the planner")):
        r = subprocess.run(base + extra, capture_output=True, text=True, env=_env(),
                           cwd=str(ROOT), timeout=900, encoding="utf-8", errors="replace")
        assert r.returncode != 0 and msg in r.stdout + r.stderr, \
            (extra, (r.stdout + r.stderr)[-1500:])
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def test_plan_az_refusals():
    """--plan-az outside primlearn, --plan-az-only without --plan-az, --plan-az on the no-planner
    control and a negative weight are refused before anything trains."""
    base = [sys.executable, "-u", str(ROOT / "python" / "train_fast.py"), "--run", "az_bad",
            "--steps", "2048"]
    prim = list(FLAGS)
    prim[prim.index("--goal-planner") + 1] = "prim"
    i = prim.index("--plan-batch")
    del prim[i:i + 2]
    cases = [(prim + ["--plan-az", "1"], "--plan-az without --goal-planner primlearn"),
             (FLAGS + ["--plan-az-only", "1"], "--plan-az-only 1 trains the planner's policy"),
             (FLAGS + ["--plan-az", "1", "--plan-fixed", "straight"],
              "--plan-az with --plan-fixed"),
             (FLAGS + ["--plan-az", "-1"], "--plan-az >= 0")]
    for extra, msg in cases:
        r = subprocess.run(base + extra, capture_output=True, text=True, env=_env(),
                           cwd=str(ROOT), timeout=600, encoding="utf-8", errors="replace")
        assert r.returncode != 0 and msg in r.stdout + r.stderr, (msg, (r.stdout + r.stderr)[-800:])
    shutil.rmtree(ROOT / "runs" / "az_bad", ignore_errors=True)


def test_plain_shaping_pays_progress_and_charges_nothing():
    """--plan-shaping plain: each primitive is paid the Euclidean progress it made toward the
    finish; a death or a time-out takes nothing back (no refund, no bank charge)."""
    from surfgym.core import SurfCore, SurfEnvConfig
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=1))
    core.reset(0)
    sv = core.states_view
    for ending in ("died", "cap"):
        pos = sv["origin"].astype(np.float64).copy()
        fin = pos[0] + np.array([4000.0, 0.0, 0.0])
        P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=1), core, 1, "cpu",
                               finish=fin, bounds=core.map_bounds(), act_every=4,
                               cfg={"plan_uniform": 0.0, "plan_novelty": 0.0,
                                    "plan_finish_bonus": 10.0, "plan_shaping": "plain"})
        P.request(np.arange(1), pos)
        steps = [300.0, -100.0, 200.0]
        for k, st in enumerate(steps):
            P.plan(pos, sv["velocity"], sv["yaw"])
            pos = pos + np.array([[st, 0.0, 0.0]])
            P.elapsed[:] = P.budget_ticks
            end = np.array([k == len(steps) - 1])
            P.on_tick(pos, end, np.zeros(1, bool), end & (ending == "died"), pos)
        r = np.array([t[4] for t in P.buf[0]])
        assert np.allclose(r, np.array(steps) / 1000.0), (ending, r)   # per 1000 u, no charge
        assert P.buf[0][-1][5]


@needs_core
def test_plan_return_2_weights_the_frontier():
    """--plan-return 2 (the reverse curriculum's frontier): a reservoir state's weight is its
    cell's Go-Explore weight times p (1 - p) + 0.01, p = the finish rate of the episodes that
    SPAWNED in the cell. A cell the policy always finishes from and one it never does weigh less
    than one where it sometimes does; an untried cell counts as p = 0.5; the counts decay per
    update, save and load; mode 1 is unchanged."""
    from surfgym.core import SurfCore, SurfEnvConfig
    n = 30
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    base = core.states_view["origin"][0].astype(np.float64)
    lo, hi = (np.asarray(b, np.float64) for b in core.map_bounds())
    d = (lo + hi) / 2.0 - base
    d[2] = 0.0
    step = 256.0 * d / np.linalg.norm(d)               # toward the box centre: four cells inside
    cells = [base + k * step for k in range(4)]
    mk = lambda mode: PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                                         finish=base + [0.0, 0.0, 5000.0],
                                         bounds=core.map_bounds(), act_every=4,
                                         cfg={"plan_uniform": 0.0, "plan_cover": 0.1,
                                              "plan_return": mode})
    P = mk(2)
    # 10 episodes spawn in each of cells 0 (always finish), 1 (3 of 10 finish), 2 (never)
    org = np.repeat(np.array(cells[:3]), 10, axis=0)
    P.request(np.arange(n), org)
    fin = np.zeros(n, bool)
    fin[:10] = True
    fin[10:13] = True
    P.on_tick(org, np.ones(n, bool), fin, ~fin)
    w = P.return_weights(np.array(cells))
    assert w[1] > w[0] and w[1] > w[2], w
    assert w[3] > w[1]                     # untried: p = 0.5, the most uncertain
    assert (P.sp_cell == -1).all()          # charged once, at the episode's end
    k1 = P._cells(np.array(cells[1:2]))
    k1 = (k1[0] * P.nov_shape[1] + k1[1]) * P.nov_shape[2] + k1[2]
    assert P.sp_n[k1][0] == pytest.approx(10.0) and P.sp_f[k1][0] == pytest.approx(3.0)
    Q = mk(2)
    Q.load_state_dict_all(P.state_dict_all())
    assert np.array_equal(Q.sp_n, P.sp_n) and np.array_equal(Q.sp_f, P.sp_f)
    assert "frontier cells" in P.goid_summary()
    # mode 1: the coverage weight alone, no spawn statistics kept
    R = mk(1)
    assert R.sp_n is None and R.goid_summary() == ""
    R.request(np.arange(n), org)
    R.on_tick(org, np.ones(n, bool), fin, ~fin)
    assert np.allclose(R.return_weights(np.array(cells)), 1.0 / np.sqrt(1.0 + np.array(
        [R.cov_n[(c[0] * R.nov_shape[1] + c[1]) * R.nov_shape[2] + c[2]]
         for c in zip(*R._cells(np.array(cells)))], np.float64)))


@needs_core
def test_plan_return_3_selects_per_cell():
    """--plan-return 3 (Go-Explore's selection per CELL): a cell's total weight is 1/sqrt(1 + N)
    whatever the number of reservoir states standing in it; mode 1 weighs every state."""
    from surfgym.core import SurfCore, SurfEnvConfig
    n = 4
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    base = core.states_view["origin"][0].astype(np.float64)
    lo, hi = (np.asarray(b, np.float64) for b in core.map_bounds())
    d = (lo + hi) / 2.0 - base
    d[2] = 0.0
    far = base + 256.0 * d / np.linalg.norm(d)
    org = np.vstack([np.repeat(base[None, :], 10, 0), far[None, :]])     # 10 states vs 1

    def mk(mode):
        return PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                                  finish=base + [0.0, 0.0, 5000.0], bounds=core.map_bounds(),
                                  act_every=4, cfg={"plan_uniform": 0.0, "plan_cover": 0.1,
                                                    "plan_return": mode})
    w3, w1 = mk(3).return_weights(org), mk(1).return_weights(org)
    assert w3[:10].sum() == pytest.approx(w3[10])            # N = 0 everywhere: equal cells
    assert w1[:10].sum() == pytest.approx(10.0 * w1[10])     # per state


@needs_core
def test_plan_sil_replays_finished_episodes_with_mc_returns():
    """--plan-sil: the decisions of a FINISHED episode enter the replay with their discounted
    Monte-Carlo returns; a failed episode's and an open episode's do not; the SIL term raises the
    replayed primitive's log density."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goallearn import PLAN_GAMMA
    n = 2
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    base = core.states_view["origin"][0].astype(np.float64)
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                           finish=base + [0.0, 0.0, 5000.0], bounds=core.map_bounds(),
                           act_every=4, cfg={"plan_uniform": 0.0, "plan_sil": 1.0,
                                             "plan_batch": 1, "plan_shaping": "refund_i"})
    x = np.zeros(P.net.d_in if hasattr(P.net, "d_in") else N_OBS, np.float32)
    u = np.full(P.d_act, 0.7, np.float32)
    # env 0: three decisions, the last one FINISHES; env 1: two decisions, dies; then one open
    P.buf[0] = [(x, u, 0.0, 0.0, 1.0, False, 50, None), (x, u, 0.0, 0.0, 2.0, False, 50, None),
                (x, u, 0.0, 0.0, 10.0, True, 50, None)]
    P.buf_fin[0] = [False, False, True]
    P.buf[1] = [(x, u, 0.0, 0.0, 1.0, False, 50, None), (x, u, 0.0, 0.0, -1.0, True, 50, None),
                (x, u, 0.0, 0.0, 0.5, False, 50, None)]
    P.buf_fin[1] = [False, False, False]
    before = float(mix_logp(*P.net(torch.as_tensor(x[None, :]))[:3],
                            torch.as_tensor(u[None, :])))
    out = P.update(force=True)
    assert P.sil_len() == 3 and P.sil_n == 3
    g = PLAN_GAMMA
    assert np.allclose(sorted(P.sil["R"][:3]), sorted([10.0, 2.0 + g * 10.0,
                                                       1.0 + g * (2.0 + g * 10.0)]), atol=1e-4)
    assert "sil_pi" in out and out["sil_len"] == 3
    assert all(len(f) == 0 for f in P.buf_fin)
    for _ in range(5):
        P.buf[0] = [(x, u, 0.0, 0.0, 0.0, False, 50, None)]
        P.buf_fin[0] = [False]
        P.update(force=True)
    after = float(mix_logp(*P.net(torch.as_tensor(x[None, :]))[:3],
                           torch.as_tensor(u[None, :])))
    assert after > before


@needs_core
def test_plan_sil_uniform_replays_the_opener_of_a_finished_episode():
    """--plan-sil-uniform 1: a uniform opening primitive is never a PPO transition, but when its
    episode finishes it joins the SIL replay with its progress + gamma x the return of the
    planner's primitives that followed; an unfinished episode's opener is dropped."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goallearn import PLAN_GAMMA
    n = 2
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64)
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                           finish=pos[0] + [0.0, 0.0, 5000.0], bounds=core.map_bounds(),
                           act_every=4, cfg={"plan_uniform": 1.0, "plan_sil": 1.0,
                                             "plan_sil_uniform": 1, "plan_novelty": 0.0,
                                             "plan_batch": 1, "plan_shaping": "refund_i"})
    P.request(np.arange(n), pos)
    P.plan(pos, sv["velocity"], sv["yaw"])
    assert (P.uh_state == 1).all() and not P.decided.any() and P.n_ready() == 0
    no = np.zeros(n, bool)
    for _ in range(P.budget_ticks + 1):                 # stand still: the opener times out alive
        P.on_tick(pos, no, no, no)
    assert (P.uh_state == 2).all() and np.allclose(P.uh_prog, 0.0)
    P.plan(pos, sv["velocity"], sv["yaw"])              # the planner's own primitive
    assert P.decided.all()
    fin = np.array([True, False])
    P.on_tick(pos, np.ones(n, bool), fin, ~fin)         # env 0 finishes, env 1 dies
    assert (P.uh_state == 0).all() and P.sil_uni_n == 1
    fb = float(P.cfg["plan_finish_bonus"])
    assert float(P.sil["R"][0]) == pytest.approx(PLAN_GAMMA * fb)
    P.update(force=True)                                # the finished decided transition joins
    assert P.sil_len() == 2
    assert sorted(float(r) for r in P.sil["R"][:2]) == pytest.approx(sorted([PLAN_GAMMA * fb, fb]))


def test_plan_straight_component_is_floored_fixed_and_loads_onto_an_old_planner():
    """--plan-straight EPS: one more mixture component with mean 0 and a fixed spread, whose
    weight never falls below EPS; the greedy choice is all-zero numbers when it is the heaviest;
    a planner stored without it gains it at load (the old components untouched)."""
    from surfgym.goalprimplan import PrimPlannerNet, STRAIGHT_SD, mix_entropy
    torch.manual_seed(0)
    eps = 0.1
    net = PrimPlannerNet(N_OBS, 6, straight=eps)
    x = torch.randn(64, N_OBS)
    lg, mu, ls, v = net(x)
    assert lg.shape == (64, 4) and mu.shape == (64, 4, 6) and ls.shape == (64, 4, 6)
    w = torch.softmax(lg, -1)
    assert torch.allclose(w.sum(-1), torch.ones(64), atol=1e-5)
    assert (w[:, -1] >= eps - 1e-6).all()
    assert torch.equal(mu[:, -1], torch.zeros(64, 6))
    assert torch.allclose(ls[:, -1], torch.full((64, 6), math.log(STRAIGHT_SD)))
    u = mix_sample(lg, mu, ls, torch.Generator().manual_seed(1))
    assert torch.isfinite(mix_logp(lg, mu, ls, u)).all() and torch.isfinite(mix_entropy(lg, ls)).all()
    # greedy = the heaviest component's mean: force 'keep going' heaviest -> all zeros
    with torch.no_grad():
        net.logits.bias[-1] = 20.0
    lg2, mu2, ls2, _ = net(x)
    assert torch.equal(mix_sample(lg2, mu2, ls2, None, greedy=True), torch.zeros(64, 6))
    # an old planner (no component) loads into a straight one: the old logits rows are kept
    old = PrimPlannerNet(N_OBS, 6)
    sd = {k: t.clone() for k, t in old.state_dict().items()}
    sd["logits.bias"][:] = torch.tensor([0.5, -0.5, 1.0])
    from surfgym.core import SurfCore, SurfEnvConfig
    if DLL.exists() and LAB.exists():
        core = SurfCore(str(LAB), SurfEnvConfig(num_envs=1))
        core.reset(0)
        pos = core.states_view["origin"].astype(np.float64)
        mk = lambda s: PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=1), core, 1, "cpu",
                                          finish=pos[0] + [3000.0, 0.0, 0.0],
                                          bounds=core.map_bounds(), cfg={"plan_straight": s})
        P_old = mk(0.0)
        P_old.net.load_state_dict(sd)
        xo = np.zeros(N_OBS, np.float32)
        uo = np.zeros(6, np.float32)
        P_old.buf[0] = [(xo, uo, 0.0, 0.0, 1.0, True, 50, None)]
        P_old.buf_fin[0] = [False]
        P_old.update(force=True)                      # Adam now holds moments for 3 logits
        sd["logits.bias"] = P_old.net.logits.bias.detach().clone()
        P_new = mk(eps)
        P_new.load_state_dict_all(P_old.state_dict_all())
        assert torch.equal(P_new.net.logits.bias[:3], sd["logits.bias"])
        # the stored moments do not fit the 4-logit layer: the load starts Adam fresh, and the
        # first update runs (it crashed in torch's foreach lerp, 2026-09-26)
        P_new.buf[0] = [(xo, uo, 0.0, 0.0, 1.0, True, 50, None)]
        P_new.buf_fin[0] = [False]
        assert P_new.update(force=True) is not None
        with pytest.raises(ValueError):
            mk(0.0).load_state_dict_all(P_new.state_dict_all())


def test_fit_flown_primitive_recovers_the_knots_of_a_flown_curve():
    """--plan-sil-flown: fitting the primitive to a path that IS a primitive's curve (level frame,
    one point per 10 ms) gives back its knots; the SIL replay takes the fitted numbers."""
    from surfgym.goalprim import curve
    from surfgym.goalprimplan import fit_flown_primitive
    nums = np.array([60.0, -30.0, 90.0, 20.0, -10.0, 5.0])
    pts = curve(np.zeros(3), np.array([700.0, 0.0, 0.0]), 0.0, nums, 2.0, 3, 300.0,
                frame="level")
    fit = fit_flown_primitive(pts, 0.01, 2.0, 3, 180.0, 120.0, 90.0)
    assert fit is not None
    assert np.allclose(fit[:3], nums[:3], atol=8.0), fit
    assert np.allclose(fit[3:], nums[3:], atol=8.0), fit
    assert fit_flown_primitive(pts[:5], 0.01, 2.0, 3, 180.0, 120.0, 90.0) is None   # too short
    still = np.zeros((200, 3))
    assert fit_flown_primitive(still, 0.01, 2.0, 3, 180.0, 120.0, 90.0) is None     # no motion
    # the replay: a finished episode's transition is pushed with the flown numbers
    P = PrimLearnedPlanner.__new__(PrimLearnedPlanner)
    P.sil, P.sil_n = None, 0
    x = np.zeros(N_OBS, np.float32)
    chosen, flown = np.full(6, 0.9, np.float32), np.full(6, -0.2, np.float32)
    P._sil_push([(x, chosen, 0.0, 0.0, 10.0, True, 50, None)], [True], None, [flown])
    assert P.sil_n == 1 and np.allclose(P.sil["u"][0], flown)
    P._sil_push([(x, chosen, 0.0, 0.0, 10.0, True, 50, None)], [True], None, [None])
    assert np.allclose(P.sil["u"][1], chosen)


@needs_core
def test_plan_sil_flown_records_the_terminal_position_not_the_autoreset_spawn():
    """--plan-sil-flown: on the tick an episode ends, ``pos`` is already the auto-reset spawn; the
    flown path must end at ``term_pos``, so a finishing primitive flown straight fits as straight
    (with the spawn recorded, the path ended in a teleport - Codex's review, 2026-09-26)."""
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goalprim import curve
    n = 1
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    sv = core.states_view
    start = sv["origin"].astype(np.float64)
    P = PrimLearnedPlanner(PrimitivePlanner(secs=2.0, n_envs=n), core, n, "cpu",
                           finish=start[0] + [5000.0, 0.0, 0.0], bounds=core.map_bounds(),
                           act_every=4, cfg={"plan_uniform": 0.0, "plan_sil": 1.0,
                                             "plan_sil_flown": 1, "plan_novelty": 0.0,
                                             "plan_shaping": "refund_i"})
    P.request(np.arange(n), start)
    P.plan(start, np.array([[700.0, 0.0, 0.0]]), np.array([0.0]))
    P.track.advance = lambda o: (np.zeros(n), np.ones(n, bool))    # close only at the end
    no = np.zeros(n, bool)
    p = start.copy()
    for _ in range(80):                                   # 0.8 s straight along +x at 700 u/s
        p = p + np.array([[7.0, 0.0, 0.0]])
        P.on_tick(p, no, no, no)
    spawn = start + np.array([[-3000.0, 2000.0, 0.0]])    # the auto-reset spawn, far away
    term = p + np.array([[7.0, 0.0, 0.0]])
    yes = np.ones(n, bool)
    P.on_tick(spawn, yes, yes, no, term_pos=term, term_vel=np.array([[700.0, 0.0, 0.0]]),
              term_yaw=np.array([0.0]))
    k = int(P.hs_n[0])
    assert np.allclose(P.hs_pos[0, k - 1], term[0])        # the terminal position, not the spawn
    assert P.buf_fin[0] == [True] and P.buf_hs[0][0] is not None
    nums = squash(P.prim, np.asarray(P.buf_hs[0][0], np.float64)[None, :])[0]
    assert np.all(np.abs(nums[:3]) < 20.0), nums            # flown straight -> fits straight


@needs_core
def test_plan_sil_ext_reads_search_expert_episodes_once(tmp_path):
    """--plan-sil-ext: the update's SIL replay takes the search-expert records written under
    <run>/sil_ext (x, u, R per decision of a finished search-driven episode), each file once, and
    skips a file of the wrong width."""
    from surfgym.core import SurfCore, SurfEnvConfig
    n = 1
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    pos = core.states_view["origin"].astype(np.float64)
    P = PrimLearnedPlanner(PrimitivePlanner(secs=0.5, n_envs=n), core, n, "cpu",
                           finish=pos[0] + [3000.0, 0.0, 0.0], bounds=core.map_bounds(),
                           cfg={"plan_sil": 1.0})
    P.attach_sil_ext(tmp_path / "sil_ext")
    x = np.zeros((3, P.d_in), np.float32)
    u = np.zeros((3, P.d_act), np.float32)
    np.savez(tmp_path / "sil_ext" / "a.npz", x=x, u=u, R=np.array([1.0, 2.0, 3.0], np.float32))
    np.savez(tmp_path / "sil_ext" / "bad.npz", x=np.zeros((2, 5), np.float32), u=u[:2],
             R=np.ones(2, np.float32))
    assert P._sil_ext_load() == 3 and P.sil_ext["n"] == 3
    assert P._sil_ext_load() == 0                       # each file once
    assert sorted(P.sil_ext["R"][:3].tolist()) == [1.0, 2.0, 3.0]
    assert P.sil_len() == 0                             # its own replay, not the trainer's
    P.buf[0] = [(np.zeros(P.d_in, np.float32), np.zeros(P.d_act, np.float32), 0.0, 0.0, 1.0,
                 True, 50, None)]
    P.buf_fin[0] = [False]
    out = P.update(force=True)                          # the ext term runs (no crash)
    assert out is not None
