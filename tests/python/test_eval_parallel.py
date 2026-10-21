"""--eval-parallel 1 (2026-10-21, autoresearch r4b3): the greedy eval's episodes run SIDE BY
SIDE on an eval core of --eval-eps envs (surfgym.record.record_rollout_parallel), one batched
forward per decision instead of one per episode.

(a) The recorder on the real core:
    * one episode on a one-env core writes exactly the bytes record_rollout writes;
    * N episodes on N envs: N episodes in env order, every one from the map start, each env's
      FIRST episode only (rows numbered from 0, trailer ticks = rows), the core's own cap.
(b) The greedy-line pool's side-by-side capture is episode i's recorded rows, row for row.
(c) The side-by-side patience hook: per-env cumulative rule, one kill per window, an env is
    let go once its first episode ends.
(d) The trainer (CPU, sidistic): the eval runs side by side and says so (header, config), a
    bare resume restores it, the refusals; the recorder lists the flag as training-only.
    (Flag off: tests/python/test_eval_patience.py's bit-identity test runs the trainer
    before both eval changes against this one.)
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.greedy_spawn import GreedyLinePool                    # noqa: E402

from test_ez_viewc import DLL, SID, SMOKE, TRAIN, _run, _train, needs_run  # noqa: E402

needs_core = pytest.mark.skipif(not (SID.exists() and DLL.exists()),
                                reason="needs the core + sidistic")


class _Pol:
    """A scripted policy: random yaw bin, forward held (record_rollout's act contract)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        a = np.zeros((obs.shape[0], 6), np.int32)
        a[:, 0] = self.rng.integers(0, 15, obs.shape[0])
        a[:, 2] = 2
        return a


def _core(n, cap):
    """An eval core as the trainer builds one: the map's own spawn points as the pool."""
    from surfgym import SurfCore, default_config
    from surfgym.rewards import map_spawn_pool
    c = SurfCore(str(SID), default_config(num_envs=n, spawn_mode=2, max_episode_ticks=cap,
                                          sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))
    c.set_spawn_pool(map_spawn_pool(c))
    return c


def _episodes(path):
    out, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        if line.startswith("{"):
            d = json.loads(line)
            if "end" in d:
                out.append((hdr, np.asarray(cur, np.float64), d))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return out


# ==========================================================================
# (a) the recorder
# ==========================================================================
@needs_core
def test_one_episode_on_one_env_is_record_rollout_byte_for_byte(tmp_path):
    from surfgym.record import record_rollout, record_rollout_parallel
    a, b = tmp_path / "seq.jsonl", tmp_path / "par.jsonl"
    sa = record_rollout(_core(1, 300), _Pol(), a, episodes=1, max_ticks=300, seed=5,
                        header_extra={"eval_stall": 0})
    sb = record_rollout_parallel(_core(1, 300), _Pol(), b, 1, seed=5,
                                 header_extra={"eval_stall": 0})
    assert a.read_bytes() == b.read_bytes()
    assert sa == sb


@needs_core
def test_n_episodes_side_by_side_each_from_the_start_first_episode_only(tmp_path):
    from surfgym.record import record_rollout_parallel
    core = _core(4, 400)
    path = tmp_path / "par.jsonl"
    summ = record_rollout_parallel(core, _Pol(1), path, 3, seed=7,
                                   header_extra={"eval_parallel": 1})
    eps = _episodes(path)
    assert len(eps) == len(summ) == 3
    starts = []
    for i, (hdr, rows, tr) in enumerate(eps):
        assert hdr["episode"] == i and hdr["eval_parallel"] == 1 and hdr["map"]
        assert tr["ticks"] == len(rows) == summ[i]["ticks"] <= 400
        assert list(rows[:, 0]) == list(range(len(rows)))       # one episode, numbered from 0
        assert tr["end"] in ("fail", "trunc", "done")
        if tr["end"] == "trunc":
            assert len(rows) == 400                              # the core's own cap
        starts.append(rows[0, 1:4])
    # every env starts at one of the map's own spawn points (the eval's spawn pool)
    from surfgym.rewards import map_spawn_pool
    pool = np.asarray(map_spawn_pool(core)["origin"], np.float64)
    for q in starts:
        assert np.min(np.abs(pool - q).max(axis=1)) < 1.0, (q, pool)
    # a cap on the loop ends the episodes still running as truncated
    summ2 = record_rollout_parallel(_core(3, 400), _Pol(1), tmp_path / "cut.jsonl", 3,
                                    seed=7, max_ticks=50)
    assert all(s["ticks"] <= 50 for s in summ2)
    with pytest.raises(ValueError):
        record_rollout_parallel(_core(2, 50), _Pol(), tmp_path / "x.jsonl", 3, seed=1)


# ==========================================================================
# (b) the greedy-line pool
# ==========================================================================
@needs_core
def test_the_side_by_side_capture_is_each_episodes_recorded_rows(tmp_path):
    from surfgym.record import record_rollout_parallel
    p = GreedyLinePool(0.2, window=(0.0, 1.0))
    path = tmp_path / "par.jsonl"
    summ = record_rollout_parallel(_core(3, 250), _Pol(2), path, 3, seed=9,
                                   on_tick=p.hook(3))
    eps_cap = [e for e in p._par[0]]
    rows = _episodes(path)
    assert len(eps_cap) == len(rows) == 3
    for ep, (hdr, js, tr), s in zip(eps_cap, rows, summ):
        assert ep is not None and len(ep) == tr["ticks"] == len(js) == s["ticks"]
        np.testing.assert_allclose(ep["origin"], js[:, 1:4], atol=0.006)
        np.testing.assert_allclose(ep["velocity"], js[:, 4:7], atol=0.006)
    st = p.build(summ, 10.0)
    assert st["eps"] == 3 and p._par is None


def test_the_side_by_side_capture_keeps_a_running_episode_and_ignores_later_ones():
    from surfgym.core import STATE_DTYPE
    p = GreedyLinePool(0.2, window=(0.0, 1.0))
    hook = p.hook(2)
    for t in range(10):
        st = np.zeros(2, STATE_DTYPE)
        st["origin"][:, 0] = [t, 100 + t]
        done = np.array([t == 3, False])          # env 0 ends at tick 3, env 1 never
        hook(t, st, np.zeros(2), done, np.zeros(2, bool))
    eps, cur = p._par
    assert [int(x) for x in eps[0]["origin"][:, 0]] == [0, 1, 2, 3]   # nothing after its end
    assert eps[1] is None and len(cur[1]) == 10
    p.build([{"end": "fail"}, {"end": "trunc"}], 10.0)
    assert p.info["eps"] == 2


# ==========================================================================
# (c) the side-by-side patience hook
# ==========================================================================
class _FakeCore:
    def __init__(self, n):
        self.num_envs = n
        self.failed = np.zeros(n, np.uint8)
        self.states_view = np.zeros(n, dtype=[("origin", np.float64, (3,))])

    def force_fail(self, mask):
        self.failed |= np.asarray(mask, np.uint8)


class _LineField:
    reach_max = 1e6 + 16.0
    cell = 32.0

    def sample(self, pts):
        x = np.asarray(pts, np.float64)[:, 0].copy()
        return np.where(x >= 1e6, 1e6 + 20.0, x)


def test_the_side_by_side_patience_rule_is_per_env():
    from train_fast import make_eval_patience_hook_par
    core = _FakeCore(4)                 # 3 eval envs on a 4-env core: env 3 is never touched
    hook = make_eval_patience_hook_par(core, _LineField(), 50, 3)
    kills = {0: [], 1: [], 2: []}
    for t in range(200):
        core.states_view["origin"][:, 0] = [1000.0,                  # env 0: stuck
                                           1000.0 - 1.0 * t,        # env 1: gaining
                                           2e6,                     # env 2: off the field
                                           1000.0]                  # env 3: not an eval env
        core.failed[:] = 0
        done = np.array([False, False, t == 120, False])
        hook(t, None, None, done, np.zeros(4, bool))
        for i in range(3):
            if core.failed[i]:
                kills[i].append(t)
        assert not core.failed[3]
    assert kills[0] == [50, 100, 150]   # armed at tick 0, one kill per 50-tick window
    assert kills[1] == []
    assert kills[2] == [50, 100]        # let go once its first episode ended at tick 120
    assert hook.state["killed"] == [0, 2]


def test_the_side_by_side_patience_rule_with_a_zero_window_is_off():
    from train_fast import make_eval_patience_hook_par
    core = _FakeCore(2)
    hook = make_eval_patience_hook_par(core, _LineField(), 0, 2)
    for t in range(300):
        core.states_view["origin"][:, 0] = 1000.0
        hook(t, None, None, np.zeros(2, bool), np.zeros(2, bool))
    assert not core.failed.any() and hook.state["killed"] == []


# ==========================================================================
# (d) the trainer
# ==========================================================================
def test_the_recorder_treats_the_flag_as_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert "eval_parallel" in record_ckpt.TRAIN_ONLY


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


PAR_SMOKE = ["--record-every", "2048", "--eval-eps", "3", "--eval-greedy-only"]


@needs_run
def test_the_trainer_runs_its_eval_side_by_side_and_a_resume_restores_it():
    run = "evpar_on"
    r = _train(run, PAR_SMOKE + ["--eval-parallel", "1", "--eval-patience", "3"],
               steps="4096")
    cfg = _cfg(run)
    assert cfg["eval_parallel"] == 1 and cfg["eval_patience"] == 3.0
    trajs = sorted((ROOT / "runs" / run).glob("traj_*.jsonl"))
    assert trajs
    for p in trajs:
        eps = _episodes(p)
        assert len(eps) == 3
        for i, (hdr, rows, tr) in enumerate(eps):
            assert hdr["episode"] == i and hdr["eval_parallel"] == 1
            assert hdr["eval_patience"] == 3.0
            assert tr["ticks"] == len(rows) and rows[0, 0] == 0
    assert re.search(r"greedy: .* -> traj_", r.stdout)
    run2 = "evpar_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + PAR_SMOKE
              + ["--steps", "6144"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "eval_parallel=1" in r2.stdout
    assert _cfg(run2)["eval_parallel"] == 1
    for p in sorted((ROOT / "runs" / run2).glob("traj_*.jsonl")):
        assert all(h["eval_parallel"] == 1 for h, _, _ in _episodes(p))
    for dd in (run, run2):
        shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--eval-parallel", "1", "--record-every", "2048"], "evals without --eval-greedy-only"),
    (["--eval-parallel", "1", "--eval-greedy-only", "--eval-stall", "1"], "--eval-stall"),
])
def test_bad_settings_are_refused(extra, msg):
    smoke = [a for a in SMOKE if a != "--eval-greedy-only"]   # the preset carries it
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "evpar_bad"] + smoke
             + ["--steps", "2048"] + extra)
    shutil.rmtree(ROOT / "runs" / "evpar_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout[-2000:]
