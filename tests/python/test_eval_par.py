"""--eval-par: the greedy eval's episodes side by side (one per env of the eval core).

(a) record_rollout_par on the real core: one episode per env, written in record_rollout's
    format (score.py's reader parses it, episode i = env i), and GreedyLinePool.hook_par
    captures env i's rows exactly as they were recorded;
(b) par_stall_hook: the per-env stall hooks read as one hook over the recorded episodes;
(c) the trainer (CPU smoke): --eval-par 1 with --eval-stall 1 and --respawn-greedy - the
    evals are valid eval files from the true start, the pool is built, the config carries
    eval_par and a bare resume restores it; bad combinations are refused.
Flags off are covered by test_pitch_std_min.py's bit-identity test (the trainer before
this round's changes).

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/python/test_eval_par.py -p no:cacheprovider
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
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool                    # noqa: E402

from test_ez_viewc import DLL, SID, SMOKE, TRAIN, _run, needs_run  # noqa: E402


class _Pol:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def act(self, obs):
        a = np.zeros((obs.shape[0], 6), np.int32)
        a[:, 0] = self.rng.integers(0, 15, obs.shape[0])
        a[:, 2] = 2
        return a


@pytest.mark.skipif(not (SID.exists() and DLL.exists()), reason="needs the core + sidistic")
def test_side_by_side_recording_format_and_capture(tmp_path):
    from surfgym import SurfCore, default_config
    from surfgym.record import record_rollout_par
    import score

    core = SurfCore(str(SID), default_config(num_envs=3, spawn_mode=2, max_episode_ticks=150,
                                             sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))
    p = GreedyLinePool(0.2, window=(0.0, 1.0))
    path = tmp_path / "t.jsonl"
    summ = record_rollout_par(core, _Pol(), path, 3, max_ticks=400, seed=3,
                              on_tick=p.hook_par(3), header_extra={"eval_stall": 0})
    eps = score.episodes(path)
    assert len(eps) == len(summ) == 3
    for i, (hdr, rows, tr) in enumerate(eps):
        assert hdr["episode"] == i and hdr["map"] == SID.stem and hdr["eval_stall"] == 0
        assert tr["end"] in ("fail", "trunc", "done") and tr["ticks"] == len(rows)
        assert summ[i]["end"] == tr["end"] and summ[i]["ticks"] == tr["ticks"]
        assert np.array_equal(rows[:, 0], np.arange(len(rows)))      # per-episode tick
    # the capture is the recording, env i = episode i, row for row
    cap = [np.array(c, dtype=STATE_DTYPE) for c in p._par[0]]
    for (hdr, rows, tr), ep in zip(eps, cap):
        assert len(ep) == len(rows)
        np.testing.assert_allclose(ep["origin"], rows[:, 1:4], atol=0.006)
        np.testing.assert_allclose(ep["velocity"], rows[:, 4:7], atol=0.006)
    # build() reads the parallel capture in env order and clears it
    st = p.build(summ, tick_ms=10.0)
    assert st["eps"] == 3 and p._par is None
    # a 150-tick cap: every episode is a 150-tick trunc or ended earlier
    assert all(len(r) <= 150 for _, r, _ in eps)


def test_par_stall_hook_reads_like_one_hook_over_the_recorded_episodes():
    import train_fast as tf

    class _Core:
        num_envs = 2

        def __init__(self):
            self.states_view = np.zeros(2, STATE_DTYPE)
            self.fails = np.zeros(2, int)

        def force_fail(self, m):
            self.fails += np.asarray(m, int)

    class _Field:
        def sample(self, q):
            return np.asarray(q, np.float64)[:, 0]

    core = _Core()
    hk = tf.par_stall_hook([tf.make_eval_stall_hook(core, _Field(), 5, 1.0, 1, env=e)
                            for e in range(2)])
    no, yes = np.zeros(2, bool), np.array([False, True])
    # env 0 progresses, env 1 is stuck -> env 1's first episode is killed
    for t in range(8):
        core.states_view["origin"][0, 0] = 1000.0 - 10.0 * t
        core.states_view["origin"][1, 0] = 500.0
        hk(t, None, None, no, no)
    assert hk.state["killed"] == [1] and hk.state["n"] == 1
    # env 1's episode ends (the kill); its NEXT episode is stuck too: not counted
    hk(8, None, None, yes, no)
    for t in range(8):
        core.states_view["origin"][0, 0] = 900.0 - 10.0 * t
        hk(9 + t, None, None, no, no)
    assert hk.state["killed"] == [1] and hk.state["n"] == 1


@needs_run
def test_trainer_smoke_eval_par_with_stall_and_greedy_spawns():
    import torch
    import score
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("epar_base")
    run = "epar_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "14336", "--respawn-greedy", "0.25",
                                       "--eval-par", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "--eval-par: the greedy eval flies its 2 episodes side by side" in out
    built = re.findall(r"greedy-line pool: ([0-9,]+) own states from (\d+)/(\d+) failed", out)
    assert len(built) >= 2 and all(int(e) == 2 for _, _, e in built), out[-3000:]
    trajs = sorted((ROOT / "runs" / run).glob("traj_*.jsonl"))
    assert trajs
    for t in trajs:
        eps = score.episodes(t)
        assert len(eps) == 2 and [h["episode"] for h, _, _ in eps] == [0, 1]
        assert all(h["eval_stall"] == 1 for h, _, _ in eps)
    cfg = json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["eval_par"] == 1 and cfg["eval_stall"] == 1
    ck = torch.load(ROOT / "runs" / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert ck["config"]["eval_par"] == 1
    # a bare resume restores it
    run2 = "epar_bare"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE
              + ["--steps", "16384", "--record-every", "4096", "--eval-eps", "2"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "eval_par=1" in r2.stdout and "side by side" in r2.stdout
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
def test_eval_par_refuses_what_it_does_not_run():
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "epar_bad"] + SMOKE
             + ["--steps", "2048", "--eval-par", "1", "--race-latch", "0.5"])
    shutil.rmtree(ROOT / "runs" / "epar_bad", ignore_errors=True)
    assert r.returncode != 0 and "--eval-par runs the plain single-map greedy eval" in (
        r.stdout + r.stderr), r.stdout[-2000:]
