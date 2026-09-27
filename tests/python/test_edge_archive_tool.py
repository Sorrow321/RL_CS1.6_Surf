"""tools/edge_archive.py (2026-09-27): the planner-free move operator and the terminal-complete path.

(a) RayOperator (--rays 3): its constants, and on a real core its lines are exactly a
    --plan-shape ray planner's (the lines every archive of 2026-09-26/27 flew) - so an archive
    with step 1's primitive follower, or an untrained policy, flies the same moves;
(b) Flyer.fly on a fake core: a flight that FINISHES keeps its last pre-step position and the
    swept tick's end (Codex 2026-09-27: ended rows were cleared before the 10-tick path sampler,
    so a finishing flight's path stopped up to 10 ticks short and the self-route ~23 u before the
    box - the core autoresets a finished row, so the terminal state itself is gone after the step).
"""
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
import edge_archive as ea                                                       # noqa: E402

LAB = ROOT / "maps_pool" / "labyrinth_left100.bsp"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_core = pytest.mark.skipif(not (DLL.exists() and LAB.exists()),
                                reason="needs the built core + maps_pool/labyrinth_left100")


def test_ray_operator_constants():
    op = ea.RayOperator(10.0, [1.0, 2.0, 3.0])
    assert op.n_choice == 3 and op.commit_ticks == 200 and op.budget_ticks == 200
    assert "no planner" in op.describe()
    assert ea.RayOperator(7.667, [0, 0, 0]).commit_ticks == int(round(2000.0 / 7.667))
    for k, sign in ((0, 0.0), (1, 1.0), (2, -1.0)):
        ln, pts = op.line_and_curve_of([0.0, 0.0, 0.0], [600.0, 0.0, 50.0], 0.0, None, k)
        d = np.asarray(pts[-1]) - np.asarray(pts[0])
        assert abs(math.degrees(math.atan2(d[1], d[0])) - 45.0 * sign) < 1e-6
        assert abs(d[2]) < 1e-9                                      # level
        assert abs(np.linalg.norm(d) - 600.0 * 2.0 * 1.5) < 7.0     # speed x 1.5 x 2 s
        sl = np.linalg.norm(np.diff(np.asarray(ln, np.float64), axis=0), axis=1)
        assert np.allclose(sl, sl[0], atol=1e-3) and abs(sl[0] - 128.0) < 64.0


@needs_core
def test_ray_operator_draws_the_ray_planners_lines():
    from surfgym.core import SurfCore, SurfEnvConfig
    from surfgym.goalprim import PrimitivePlanner
    from surfgym.goalprimplan import PrimLearnedPlanner
    n = 4
    core = SurfCore(str(LAB), SurfEnvConfig(num_envs=n))
    core.reset(0)
    pos = core.states_view["origin"].astype(np.float64)
    fin = pos[0] + [3000.0, 0.0, 0.0]
    prim = PrimitivePlanner(secs=2.0, n_envs=n, frame="level", flat=True)
    P = PrimLearnedPlanner(prim, core, n, "cpu", finish=fin, bounds=core.map_bounds(),
                           act_every=4, cfg={"plan_uniform": 0.0, "plan_batch": 8,
                                             "plan_novelty": 0.0, "plan_choices": 3,
                                             "plan_turn": 45.0, "plan_shape": "ray",
                                             "plan_close": "commit"})
    op = ea.RayOperator(10.0, fin)
    for vel, yaw in (([700.0, -300.0, 20.0], 10.0), ([0.0, 0.0, 0.0], 37.0),
                     ([50.0, 40.0, -900.0], -120.0)):
        for k in range(3):
            l1, p1 = P.line_and_curve_of(pos[0], np.asarray(vel), yaw, P.choice_nums[k], k)
            l2, p2 = op.line_and_curve_of(pos[0], np.asarray(vel), yaw, None, k)
            assert np.array_equal(np.asarray(l1), np.asarray(l2))
            assert np.array_equal(np.asarray(p1), np.asarray(p2))
    assert op.commit_ticks == P.commit_ticks


# --------------------------------------------------------------- (b) the terminal on a fake core
_DT = np.dtype([("origin", "<f4", 3), ("velocity", "<f4", 3), ("yaw", "<f4"), ("tick", "<i4"),
                ("stuck_ticks", "<i4"), ("onground", "<i4")])


class _FakeCore:
    """Every env flies +x at 1,000 u/s (10 u per 10 ms tick); env ``fin_env`` crosses the goal on
    batch tick ``fin_tick`` and is AUTORESET inside that step, like the real core."""

    def __init__(self, n, fin_env=0, fin_tick=25):
        self.num_envs = n
        self.s = np.zeros(n, _DT)
        self.s["onground"] = -1
        self.t = 0
        self.fin_env, self.fin_tick = fin_env, fin_tick
        self._hits = np.zeros(n, np.uint8)

    @property
    def states_view(self):
        return self.s

    @property
    def goal_hits(self):
        return self._hits

    def get_states(self):
        return self.s.copy()

    def set_state(self, i, st):
        self.s[i] = st
        self.s["velocity"][i] = (1000.0, 0.0, 0.0)

    def step(self, acts, view=None):
        self.t += 1
        self.s["origin"] += self.s["velocity"] * np.float32(0.01)
        self.s["tick"] += 1
        done = np.zeros(self.num_envs, bool)
        self._hits[:] = 0
        if self.t == self.fin_tick:
            done[self.fin_env] = True
            self._hits[self.fin_env] = 1
            self.s["origin"][self.fin_env] = (-5000.0, 0.0, 0.0)    # the spawn: terminal gone
        obs = np.zeros((self.num_envs, 4), np.float32)
        return obs, np.zeros(self.num_envs, np.float32), done, np.zeros(self.num_envs, bool), obs


class _FakePol:
    _k = 4
    keys_hold = False
    keys = None

    def __init__(self):
        self._tick = 0

    def act(self, obs):
        self._tick += 1
        return np.zeros((len(obs), 6), np.int32)


def test_fly_keeps_a_finishing_flights_terminal_position():
    core = _FakeCore(2)
    sc = SimpleNamespace(core=core, line=SimpleNamespace(set_lines=lambda idx, lines: None),
                         make_policy=lambda c, l: _FakePol())
    ctx = SimpleNamespace(planner=ea.RayOperator(10.0, [0.0, 0.0, 0.0]), scratch=sc,
                          tick=SimpleNamespace(ms=10.0))
    fl = ea.Flyer(ctx)
    arch = ea.Archive()
    st0 = core.get_states()[0]
    nid = arch.add(st0, None, np.zeros(4, np.float32), ("ROOT",), -1, -1, 0, 0, None)
    r_fin, r_live = fl.fly([(nid, 0), (nid, 1)], arch, None)
    assert r_fin["fin"] and not r_fin["died"] and r_fin["ticks"] == 25
    # samples every 10 ticks (100, 200), then the last pre-step position (240) and the swept
    # tick's end (250) - not the autoreset spawn (-5000)
    xs = [p[0] for p in r_fin["path"]]
    assert xs == [0.0, 100.0, 200.0, 240.0, 250.0]
    assert r_fin["terminal"] == [250.0, 0.0, 0.0]
    # a flight that reaches the plan's end is unchanged: no terminal, its end state is the core's
    assert not r_live["fin"] and r_live["terminal"] is None and r_live["ticks"] == 200
    assert abs(float(r_live["end"]["origin"][0]) - 2000.0) < 1e-2
