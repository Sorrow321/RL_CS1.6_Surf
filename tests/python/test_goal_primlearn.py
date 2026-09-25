"""--goal-planner primlearn (surfgym/goalprimplan.py): the learned primitive planner (step 2).

(a) the mixture: its log density is the Gaussian one for a single component, the squash stays in
    goalprim's ranges, greedy takes the heaviest component's mean;
(b) the planner cycle on a real core: the rays are in [0, 1], a planned env gets a primitive,
    closed primitives become transitions, a PPO update runs, the state saves and loads;
(c) the trainer + recorder on the small maze (CPU): a primlearn run trains its planner, dumps its
    knobs, and record_ckpt.py records the checkpoint with the stored planner.
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
                           cfg={"plan_uniform": 0.0, "plan_batch": 8, "plan_novelty": 0.0,
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
    assert np.allclose(r, -0.5 + d_gain / 1000.0)     # r_ok 0: completion itself is free
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
    # the finish keeps its progress and earns the bonus
    assert np.allclose(r[fin], -0.5 + d2[fin] / 1000.0 + 10.0)
    # a death AND a time-out: no progress of their own, and the 1,000 u banked before are charged
    # back - the shaped return of a failed episode is 0
    failed = ~fin
    assert np.allclose(r[failed], -0.5 - np.maximum(d_gain[failed], 0.0) / 1000.0)
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
    assert "LEARNED PRIMITIVES" in out and "EXEC cmpl" in out and "PLAN adv" in out, out[-2000:]
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
    shutil.rmtree(d, ignore_errors=True)
