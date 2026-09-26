"""--plan-choices 3 / --plan-turn / --plan-close-corridor / --plan-fail-secs (the user,
2026-09-26): the planner CHOOSES among three fixed primitives (forward, left, right) instead of
drawing a primitive's numbers, and the judge that closes a primitive is a family from lenient to
strict (the completion corridor; a fail rule that re-plans once the executor has left it).

(a) the table: forward is all zeros, left / right turn +-turn deg/s sideways and never climb;
(b) on a real core: the chosen numbers are rows of the table, the lines go straight / bend left /
    bend right of the motion, a PPO update on the categorical runs, the state round-trips and a
    mixture planner refuses it;
(c) the judge: an executor far off the line never completes inside a narrow corridor but does
    inside a wide one; with --plan-fail-secs it is closed after exactly that long outside;
(d) the trainer + recorder run the flags end to end (CPU, the small maze), and the search refuses
    a --plan-choices checkpoint.
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
from surfgym.goalprimplan import (PrimChoiceNet, PrimLearnedPlanner,           # noqa: E402
                                  choice_table)

LAB = ROOT / "maps_pool" / "labyrinth_left100.bsp"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_core = pytest.mark.skipif(not (DLL.exists() and LAB.exists()),
                                reason="needs the built core + maps_pool/labyrinth_left100")


def test_choice_table_forward_left_right():
    t = choice_table(3, 3, 45.0)
    assert t.shape == (3, 6)
    assert np.all(t[0] == 0.0)                                   # forward: straight, level
    assert np.all(t[1, :3] == 45.0) and np.all(t[2, :3] == -45.0)
    assert np.all(t[:, 3:] == 0.0)                               # nobody climbs or dives
    with pytest.raises(ValueError):
        choice_table(5, 3, 45.0)
    net = PrimChoiceNet(32, 3)
    lg, v = net(torch.zeros(4, 32))
    assert lg.shape == (4, 3) and v.shape == (4,)
    assert torch.allclose(lg, torch.zeros_like(lg))              # the choice starts uniform


def _core(n):
    from surfgym.core import SurfCore, SurfEnvConfig
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    return core


def _mk(core, n, **cfg):
    fin = core.states_view["origin"][0].astype(np.float64) + [3000.0, 0.0, 0.0]
    prim = PrimitivePlanner(secs=0.5, n_envs=n, frame="level", flat=True)
    base = {"plan_uniform": 0.0, "plan_batch": 8, "plan_novelty": 0.0, "plan_choices": 3,
            "plan_turn": 45.0}
    base.update(cfg)
    return PrimLearnedPlanner(prim, core, n, "cpu", finish=fin, bounds=core.map_bounds(),
                              act_every=4, cfg=base)


@needs_core
def test_choice_planner_cycle_on_a_core():
    n = 24
    core = _core(n)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64)
    P = _mk(core, n)
    assert "LEARNED CHOICES" in P.describe()
    P.request(np.arange(n), pos)
    idx, lines, fresh = P.plan(pos, sv["velocity"], sv["yaw"])
    assert len(idx) == n and P.decided.all()
    k = np.rint(P.o_u[:, 0]).astype(int)
    assert set(k.tolist()) <= {0, 1, 2} and np.all(P.o_u[:, 1:] == 0.0)
    assert int(P.w_pick.sum()) == n
    # the line bends the way the choice says, relative to the heading it leaves along (the view
    # yaw: the agents stand still at the spawn), and stays level
    for j, i in enumerate(idx):
        ln = np.asarray(lines[j], np.float64)
        y = math.radians(float(sv["yaw"][i]))
        d = ln[-1] - ln[0]
        fwd = d[0] * math.cos(y) + d[1] * math.sin(y)
        left = -d[0] * math.sin(y) + d[1] * math.cos(y)
        assert fwd > 0.0 and abs(d[2]) < 1e-3
        if k[i] == 0:
            assert abs(left) < 1e-3 * max(1.0, fwd)
        elif k[i] == 1:
            assert left > 0.05 * fwd
        else:
            assert left < -0.05 * fwd
    # greedy picks the most likely choice; nums_of maps an index to its table row
    x, u, lp, v, ent = P.choose(P.caster, pos, sv["velocity"], sv["yaw"], greedy=True)
    # a near-uniform head at init (orthogonal weights of gain 0.01, zero bias)
    assert np.allclose(lp, math.log(1.0 / 3.0), atol=0.01)
    assert np.allclose(ent, math.log(3.0), atol=0.01)
    assert np.allclose(P.nums_of(np.array([[2.0, 0, 0, 0, 0, 0]])), choice_table(3, 3, 45.0)[2])
    # stand still: every primitive times out and becomes a transition; PPO on the categorical
    for _ in range(P.budget_ticks + 1):
        P.on_tick(pos, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))
    assert P.n_ready() == n
    P.plan(pos, sv["velocity"], sv["yaw"])
    out = P.update()
    assert out is not None and out["n"] == n and math.isfinite(out["loss_pi"])
    txt, row = P.note_and_row()
    assert "F/L/R" in txt
    Q = _mk(core, n)
    Q.load_state_dict_all(P.state_dict_all())
    for a, b in zip(P.net.parameters(), Q.net.parameters()):
        assert torch.equal(a, b)
    M = _mk(core, n, plan_choices=0)
    with pytest.raises(ValueError, match="plan-choices"):
        M.load_state_dict_all(P.state_dict_all())
    with pytest.raises(ValueError, match="plan-straight"):
        _mk(core, n, plan_straight=0.1)


@needs_core
def test_close_corridor_and_the_fail_rule():
    n = 4
    core = _core(n)
    sv = core.states_view
    pos = sv["origin"].astype(np.float64)

    def fly(P, lateral, ticks):
        """Every env flies ALONG its line, ``lateral`` u to the side of it, for ``ticks``."""
        P.request(np.arange(n), pos)
        idx, lines, _ = P.plan(pos, sv["velocity"], sv["yaw"])
        closed_at = np.full(n, -1)
        for t in range(ticks):
            # one physics tick: every env moves together
            q = pos.copy()
            s = min(1.0, (t + 1) / (0.8 * P.budget_ticks))
            for j, i in enumerate(idx):
                ln = np.asarray(lines[j], np.float64)
                y = math.radians(float(sv["yaw"][i]))
                side = np.array([-math.sin(y), math.cos(y), 0.0])
                q[i] = ln[0] + s * (ln[-1] - ln[0]) * 1.02 + lateral * side
            P.on_tick(q, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))
            newly = (closed_at < 0) & ~P.active
            closed_at[newly] = t
        return closed_at

    # 300 u to the side: outside the recipe's 192 u corridor, inside a 1,024 u one
    narrow = _mk(core, n, plan_choices=3)
    assert narrow.corridor == 192.0 and narrow.fail_ticks == 0
    wide = _mk(core, n, plan_choices=3, plan_close_corridor=1024.0)
    assert wide.corridor == 1024.0
    fly(narrow, 300.0, narrow.budget_ticks + 2)
    fly(wide, 300.0, wide.budget_ticks + 2)
    assert narrow.w["complete"] == 0                 # timed out: never inside the corridor
    assert wide.w["complete"] > 0                    # completed along the line from 300 u away
    # the strict judge: 0.05 s outside the corridor (5 ticks at 10 ms) closes the primitive
    strict = _mk(core, n, plan_choices=3, plan_fail_secs=0.05)
    assert strict.fail_ticks == 5 and "FAILS" in strict.describe()
    at = fly(strict, 300.0, strict.budget_ticks + 2)
    assert strict.w["failc"] == strict.w["closed"] > 0
    assert np.all(at[at >= 0] < strict.budget_ticks // 2)
    txt, row = strict.note_and_row()
    assert "failed" in txt and row[-1] != ""


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
         "--goal-kcap", "1", "--prim-secs", "0.5", "--plan-batch", "64",
         "--prim-frame", "level", "--prim-flat", "1", "--plan-shaping", "refund_i",
         "--plan-choices", "3", "--plan-turn", "45", "--race-arc-corridor", "1024",
         "--plan-close-corridor", "1024", "--plan-fail-secs", "0.5"]


@needs_core
def test_trainer_and_recorder_run_plan_choices():
    run = "plan_choices_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "24576"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = r.stdout
    assert "LEARNED CHOICES" in out and "F/L/R" in out and " upd " in out, out[-2000:]
    assert "completion corridor is 1024 u" in out, out[-2000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["plan_choices"] == 3 and cfg["plan_turn"] == 45.0
    assert cfg["plan_close_corridor"] == 1024.0 and cfg["plan_fail_secs"] == 0.5
    head = (d / "progress.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert head[-4:] == ["plan/pick_f", "plan/pick_l", "plan/pick_r", "plan/fail_close"]
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["planner"]["obs"]["choices"] == 3 and ck["planner"]["updates"] >= 1
    rec = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "2"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    head = json.loads(rec.read_text(encoding="utf-8").splitlines()[0])
    nums = head["plan"]["numbers"]
    assert [round(z, 1) for z in nums] in [[round(z, 1) for z in row]
                                          for row in choice_table(3, 3, 45.0).tolist()]
    r3 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(rec), "--episodes", "1",
                         "--plan-mcts", "3"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r3.returncode != 0 and "plan-choices" in (r3.stdout + r3.stderr)
    # the refusals
    r4 = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                         "--run", run + "_bad", "--steps", "8192"] + FLAGS
                        + ["--plan-straight", "0.1"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=600, encoding="utf-8", errors="replace")
    assert r4.returncode != 0 and "--plan-choices with --plan-straight" in (r4.stdout + r4.stderr)
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(ROOT / "runs" / (run + "_bad"), ignore_errors=True)
