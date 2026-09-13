"""--unstuck-reach alive (2026-09-13): the plateau detector's progress
measure becomes the deepest geodesic an episode reached AND was still alive
`hold` ticks later, so a dive's last seconds and a deep spawn that dies at
once cannot pin T at its cap (docs/unstuck.md, "The alive reach").

(a) AliveReach: a death cuts the last `hold` ticks off (a dive reaching the
    field's minimum at impact is credited only up to `hold` ticks before
    it); a timeout counts its whole trajectory; a finish counts as d0; an
    episode that dies within `hold` ticks of its spawn contributes nothing;
    `spawn_dmin` keeps only start-anchored episodes; rows reset between
    episodes and the shared ring pointer serves envs of different ages.
(b) trainer smoke (CPU, cannonball toy set): `--unstuck-reach alive` runs,
    the config carries the three keys, progress.csv gains unstuck/reach and
    unstuck/reach_n, the step line prints the reach, and a flagless resume
    restores the mode.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_fast import AliveReach                              # noqa: E402
from test_unstuck import ABS, TRAIN, _csv, _run, _train         # noqa: E402
from test_view_continuous import SMOKE_FLAGS, needs_run         # noqa: E402

D0 = 10000.0


def _play(ar, d_by_env, ends):
    """Drive one env-set through AliveReach. d_by_env: list of per-env d
    sequences (all the same length); ends: {tick: (env, died, goal)}."""
    n = len(d_by_env)
    T = len(d_by_env[0])
    for t in range(T):
        d = np.array([seq[t] for seq in d_by_env], np.float64)
        ar.tick(d, np.ones(n, bool))
        if t in ends:
            i, died, goal = ends[t]
            ended = np.zeros(n, bool); ended[i] = True
            dd = np.zeros(n, bool); dd[i] = died
            gg = np.zeros(n, bool); gg[i] = goal
            ar.end(ended, dd, gg)


def test_death_cuts_the_last_hold_ticks_off():
    # d falls 100 u per tick from 9,000 for 20 ticks then the env dies at
    # d = 7,100. With hold 5 the deepest VERIFIED point is 5 ticks before
    # the last written one: d(14) = 7,600 -> reach 2,400, not 2,900.
    ar = AliveReach(1, hold_ticks=5, d0=D0)
    d = [9000.0 - 100.0 * t for t in range(20)]
    _play(ar, [d], {19: (0, True, False)})
    best, n = ar.pop()
    assert n == 1 and best == D0 - 7600.0
    # and the accumulator is empty again
    b, n = ar.pop()
    assert n == 0 and b != b


def test_timeout_counts_the_whole_trajectory_and_finish_counts_d0():
    ar = AliveReach(2, hold_ticks=5, d0=D0)
    d0s = [9000.0 - 100.0 * t for t in range(20)]      # env 0: times out at 7,100
    d1s = [9000.0 - 300.0 * t for t in range(20)]      # env 1: finishes
    _play(ar, [d0s, d1s], {18: (0, False, False), 19: (1, True, True)})
    best, n = ar.pop()
    assert n == 2 and best == D0                        # the finish
    ar2 = AliveReach(1, hold_ticks=5, d0=D0)
    _play(ar2, [d0s], {19: (0, False, False)})
    best, n = ar2.pop()
    assert n == 1 and best == D0 - 7100.0               # the timeout's last tick counts


def test_short_death_contributes_nothing_and_spawn_filter_anchors():
    ar = AliveReach(1, hold_ticks=50, d0=D0)
    d = [5000.0 - 100.0 * t for t in range(10)]        # deep spawn, dead in 10 ticks
    _play(ar, [d], {9: (0, True, False)})
    best, n = ar.pop()
    assert n == 0 and best != best
    # start-anchored: a window spawn at d 5,000 that survives is ignored
    # when spawn_dmin = 8,000; a start spawn at 9,000 counts
    ar = AliveReach(2, hold_ticks=3, d0=D0, spawn_dmin=8000.0)
    w = [5000.0 - 10.0 * t for t in range(30)]
    st = [9000.0 - 10.0 * t for t in range(30)]
    _play(ar, [w, st], {29: (0, False, False)})
    ar.end(np.array([False, True]), np.zeros(2, bool), np.zeros(2, bool))
    best, n = ar.pop()
    assert n == 1 and best == D0 - (9000.0 - 290.0)


def test_rows_reset_between_episodes_and_ages_differ():
    ar = AliveReach(2, hold_ticks=4, d0=D0)
    n = 2
    # env 0 runs a deep episode that dies at tick 9 (verified min d(5) = 4,500)
    # then a shallow one; env 1 starts its episode 3 ticks later than env 0
    for t in range(24):
        d = np.array([5000.0 - 100.0 * t if t < 10 else 9000.0 - 10.0 * (t - 10),
                      8000.0 - 50.0 * t], np.float64)
        live = np.array([True, t >= 3])
        ar.tick(d, live)
        if t == 9:
            ar.end(np.array([True, False]), np.array([True, False]), np.zeros(n, bool))
            b, k = ar.pop()
            assert k == 1 and b == D0 - 4500.0
    # env 0's second episode times out: its own ring only (7,000 was the
    # first episode's); env 1 has been alive 21 ticks, verified 17 of them
    ar.end(np.array([True, True]), np.zeros(n, bool), np.zeros(n, bool))
    b, k = ar.pop()
    assert k == 2
    # env 1 (timeout: every tick) reached d = 8000 - 50*23 = 6,850
    assert b == D0 - 6850.0
    assert np.isinf(ar.vmin).all() and (ar.age == 0).all() and np.isnan(ar.spawn_d).all()


@needs_run
def test_alive_reach_smoke_runs_logs_and_resumes():
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    run = "cya_reach"
    r = _train(run, ABS + ["--unstuck", "--unstuck-reach", "alive",
                           "--unstuck-hold", "0.05", "--unstuck-patience", "0",
                           "--unstuck-period", "2048", "--unstuck-rate", "1",
                           "--unstuck-max", "2",
                           # a step is one env TICK: 2,048 steps / 64 envs =
                           # 32 ticks per iteration, 160 over the run, and
                           # envs standing on the platform never fall - a
                           # 48-tick cap times each one out three times so
                           # the reach gets readings
                           "--ep-ticks", "48"], steps="10240")
    assert "--unstuck-reach alive: a point counts when the episode is still alive 0.05 s (5 ticks) later" in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["unstuck_reach"] == "alive" and cfg["unstuck_hold"] == 0.05
    assert cfg["unstuck_reach_spawn_d"] is None
    rows = _csv(run)
    assert len(rows) == 5
    head = list(rows[0])
    i = head.index("unstuck/T")
    assert head[i:i + 5] == ["unstuck/T", "unstuck/stuck_steps",
                             "unstuck/best", "unstuck/reach", "unstuck/reach_n"]
    reach = [x["unstuck/reach"] for x in rows]
    rn = [int(x["unstuck/reach_n"]) for x in rows]
    # 64 envs x 3 timeouts (ticks 48, 96, 144)
    assert any(v != "" for v in reach) and sum(rn) == 192
    assert all(np.isfinite(float(v)) for v in reach if v != "")
    assert " reach " in r.stdout
    best = [x["unstuck/best"] for x in rows]
    assert any(v != "" for v in best)
    # a flagless resume restores the mode with the checkpoint's knobs
    run2 = "cya_reach_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "12288"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "unstuck_reach=alive" in r2.stdout and "unstuck_hold=0.05" in r2.stdout
    rows2 = _csv(run2)
    assert len(rows2) == 1 and "unstuck/reach" in rows2[0]
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)
