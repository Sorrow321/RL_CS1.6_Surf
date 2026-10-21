"""--respawn-greedy-stuck N (2026-10-21, autoresearch r4b3): the greedy-line pool also keeps N
evenly spaced states of each greedy episode's STALLED stretch - from its best-progress tick to its
end - so training returns to where the deployed line wanders once it stops gaining.

(a) The pool: a stuck episode adds its pre-stop window AND N states spread over the stalled stretch
    (first = the best tick, last = the final tick); N = 0 is the pool as before; an episode whose best
    is its end adds no stalled stretch; a negative N is refused.
(b) The trainer (CPU, sidistic): the flag reaches the pool (startup line), the config and a bare
    resume; it is refused without --respawn-greedy.
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

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool                    # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run              # noqa: E402


def _ep(n, best_at):
    """n states, origin x = tick except the best tick (the field minimum) at x = -1."""
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"][:, 0] = np.arange(n, dtype=np.float32) + 1000.0
    rows["origin"][best_at, 0] = -1.0
    rows["tick"] = np.arange(n)
    return rows


def _build(pool, eps, ends):
    hook = pool.hook()
    t = 0
    for ep in eps:
        for i in range(len(ep)):
            hook(t, ep[i:i + 1], np.zeros(1), np.array([i == len(ep) - 1]), np.array([False]))
            t += 1
    return pool.build([{"end": e} for e in ends], 10.0,
                      d_fn=lambda o: np.asarray(o, np.float64)[:, 0])


def _ticks(pool):
    x = np.asarray(pool.rows["origin"][:, 0], np.float64)
    return np.sort(np.where(x < 0, -1, x - 1000.0)).astype(int)


def test_a_stuck_episode_adds_n_states_of_its_stalled_stretch():
    ep = _ep(3000, 1000)
    off = GreedyLinePool(0.3, window=(0.25, 2.0))
    _build(off, [ep], ["trunc"])
    on = GreedyLinePool(0.3, window=(0.25, 2.0), stuck=50)
    _build(on, [ep], ["trunc"])
    t_off, t_on = _ticks(off), _ticks(on)
    assert len(t_off) == 201                       # ticks 800..1000 (the best is tick 1000)
    extra = sorted(set(t_on.tolist()) - set(t_off.tolist()))
    assert len(extra) == 49                        # 50 spread over 1000..2999, 1000 already in
    assert extra[-1] == 2999 and extra[0] > 1000
    gaps = np.diff([1000] + extra)
    assert gaps.max() - gaps.min() <= 1            # evenly spaced
    assert len(on.rows) == len(off.rows) + 49


def test_no_stalled_stretch_when_the_best_is_the_end_or_the_flag_is_off():
    ep = _ep(3000, 2999)                           # still gaining when it died
    on = GreedyLinePool(0.3, window=(0.25, 2.0), stuck=50)
    _build(on, [ep], ["fail"])
    assert len(on.rows) == 175                     # only the death window (2.0..0.25 s)
    off = GreedyLinePool(0.3, window=(0.25, 2.0), stuck=0)
    _build(off, [_ep(3000, 1000)], ["trunc"])
    assert len(off.rows) == 201
    with pytest.raises(ValueError):
        GreedyLinePool(0.3, window=(0.25, 2.0), stuck=-1)


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_wires_restores_and_refuses_it():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("gstuck_base")
    run = "gstuck_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "10240", "--respawn-greedy", "0.25",
                                       "--respawn-greedy-stuck", "40"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "+ 40 evenly spaced states of each episode's stalled stretch" in r.stdout
    assert re.search(r"greedy-line pool: [0-9,]+ own states", r.stdout)
    assert _cfg(run)["respawn_greedy_stuck"] == 40
    run2 = "gstuck_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_greedy_stuck=40" in r2.stdout
    assert _cfg(run2)["respawn_greedy_stuck"] == 40
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "gstuck_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-greedy-stuck", "10"])
    assert r3.returncode != 0 and "needs --respawn-greedy F > 0" in r3.stdout + r3.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "gstuck_bad"):
        shutil.rmtree(dd, ignore_errors=True)
