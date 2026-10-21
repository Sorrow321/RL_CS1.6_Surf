"""--eval-stall + --respawn-greedy: a stall kill is a stop, not a death.

(a) the eval stall hook lists the episodes it ended (counted from 0 in the recording);
(b) GreedyLinePool.build keeps only the best-progress window of a summary marked
    ``stall`` (never the seconds before the kill), and a real fail keeps both anchors;
(c) a sidistic CPU smoke with both flags: the pool line is printed and the run completes.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/python/test_greedy_stall.py -p no:cacheprovider
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool                    # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run           # noqa: E402


class _Core:
    """force_fail + states_view, one env, the origin set by the test."""

    def __init__(self):
        self.num_envs = 1
        self.states_view = np.zeros(1, STATE_DTYPE)
        self.fails = 0

    def force_fail(self, m):
        self.fails += int(m[0])


class _Field:
    def sample(self, p):
        return np.asarray(p, np.float64)[:, 0]          # d = x


def test_the_hook_lists_the_episodes_it_killed():
    import train_fast as tf
    core = _Core()
    hook = tf.make_eval_stall_hook(core, _Field(), stall_ticks=5, stall_eps=1.0, every=1)
    no = np.zeros(1, bool)
    yes = np.ones(1, bool)
    # episode 0: progresses every tick (d falls by 10), ends by done -> no kill
    for t in range(20):
        core.states_view["origin"][0, 0] = 1000.0 - 10.0 * t
        hook(t, None, None, no, no)
    hook(20, None, None, yes, no)
    # episode 1: stuck at d 500 -> killed after 5 ticks; then the trainer's core ends it
    for t in range(8):
        core.states_view["origin"][0, 0] = 500.0
        hook(21 + t, None, None, no, no)
    hook(29, None, None, yes, no)
    # episode 2: stuck again -> killed
    for t in range(8):
        core.states_view["origin"][0, 0] = 400.0
        hook(30 + t, None, None, no, no)
    assert hook.state["killed"] == [1, 2]
    assert core.fails == hook.state["n"] == 2


def _ep(n, d_best_at):
    """n states, origin x = the field distance: falls to its minimum at d_best_at, then
    rises (the line turned back) - so the best-progress tick is d_best_at."""
    rows = np.zeros(n, STATE_DTYPE)
    x = np.abs(np.arange(n) - d_best_at).astype(np.float32) + 100.0
    rows["origin"][:, 0] = x
    rows["tick"] = np.arange(n) + 3
    return rows


def _feed(pool, eps):
    hook = pool.hook()
    t = 0
    for ep in eps:
        for i in range(len(ep)):
            last = i == len(ep) - 1
            hook(t, ep[i:i + 1], np.zeros(1), np.array([last]), np.array([False]))
            t += 1


def test_a_stall_fail_keeps_only_its_best_window_a_death_keeps_both():
    d_fn = lambda o: np.asarray(o, np.float64)[:, 0]   # noqa: E731
    p = GreedyLinePool(0.3, window=(0.25, 2.0))
    a, b = _ep(2000, 300), _ep(2000, 300)              # best at 3.0 s, end at 20 s
    _feed(p, [a, b])
    st = p.build([{"end": "fail", "stall": True}, {"end": "fail"}], tick_ms=10.0, d_fn=d_fn)
    # a: [100, 300] (2 s up to the best, inclusive); b: the same + [1800, 1975)
    want_a = a[100:301]
    want_b = b[np.unique(np.concatenate([np.arange(1800, 1975), np.arange(100, 301)]))]
    assert st["rows"] == len(want_a) + len(want_b)
    assert np.array_equal(p.rows["origin"][:len(want_a)], want_a["origin"])
    assert np.array_equal(p.rows["origin"][len(want_a):], want_b["origin"])
    assert st["deaths_s"] == [20.0] and st["stops_s"] == [3.0, 3.0]
    # without the mark both episodes are deaths (the pre-change rule, unchanged)
    p2 = GreedyLinePool(0.3, window=(0.25, 2.0))
    _feed(p2, [a, b])
    st2 = p2.build([{"end": "fail"}, {"end": "fail"}], tick_ms=10.0, d_fn=d_fn)
    assert st2["rows"] == 2 * len(want_b) and st2["deaths_s"] == [20.0, 20.0]


@needs_run
def test_smoke_eval_stall_with_greedy_spawns():
    import torch
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("gstall_base")
    run = "gstall_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "14336", "--respawn-greedy", "0.25"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    built = re.findall(r"greedy-line pool: ([0-9,]+) own states from (\d+)/(\d+) failed", r.stdout)
    assert len(built) >= 2, r.stdout[-3000:]
    assert "stallkill" in r.stdout
    ck = torch.load(ROOT / "runs" / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert ck["config"]["eval_stall"] == 1
    for dd in (d0, ROOT / "runs" / run):
        shutil.rmtree(dd, ignore_errors=True)
