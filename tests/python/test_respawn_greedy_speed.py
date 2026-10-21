"""--respawn-greedy-speed LO HI (2026-10-21, autoresearch r4b4): the greedy-line spawn rows'
velocities scaled by U(LO, HI), on their own RNG.

Why: on sidistic the CP4 window is energy-gated. From the policy's own states before its
greedy death, 0/50 stochastic episodes got through; the same states with velocity x1.25 got
through 1/16 (pre-gate) and 4/16 (approach). The boosted rows are the ones that succeed, so
the critic sees what arriving faster is worth there (respawn.py's --respawn-speed argument,
applied to the states the greedy line fails from).

(a) The pool: the mixed rows are EXACTLY the unscaled pool's rows (same positions, same RNG
    stream) with only the velocity scaled, inside [LO, HI]; speed None mixes what the pool
    always mixed; bad ranges are refused; the recorder lists the key as training-only.
(b) The trainer (CPU, sidistic): the flag needs --respawn-greedy; on a resume onto a filled
    reservoir it is announced, dumped in the config and restored by a bare resume.
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

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run           # noqa: E402


def _filled(pool):
    """A pool built from one synthetic failed episode of 400 distinct states."""
    rows = np.zeros(400, STATE_DTYPE)
    rows["origin"][:, 0] = np.arange(400)
    rows["velocity"][:, 0] = 1000.0 + np.arange(400)
    rows["velocity"][:, 2] = -50.0
    hook = pool.hook()
    for i in range(400):
        hook(i, rows[i:i + 1], np.zeros(1), np.array([i == 399]), np.array([False]))
    pool.build([{"end": "fail"}], tick_ms=10.0)
    return pool


def _base_pool(n=1000):
    p = np.zeros(n, STATE_DTYPE)
    p["origin"][:, 1] = -1.0
    return p


def test_the_boost_scales_only_the_velocity_of_the_rows_the_plain_pool_draws():
    plain = _filled(GreedyLinePool(0.3, window=(0.0, 3.0), seed=5))
    boost = _filled(GreedyLinePool(0.3, window=(0.0, 3.0), seed=5, speed=(1.1, 1.3)))
    base = _base_pool()
    for _ in range(3):
        a = plain.mix(base, 100)
        b = boost.mix(base, 100)
        n = plain.mixed
        assert n == boost.mixed == int(round(900 * 0.3))
        assert np.array_equal(a["origin"], b["origin"])          # the same rows were drawn
        assert np.array_equal(a[:-n], b[:-n])                    # the rest is untouched
        ratio = b["velocity"][-n:] / a["velocity"][-n:]
        assert np.allclose(ratio[:, 0], ratio[:, 2])             # one scale per row
        assert (ratio[:, 0] >= 1.1 - 1e-6).all() and (ratio[:, 0] <= 1.3 + 1e-6).all()
        assert ratio[:, 0].std() > 0.01
    assert "velocity x U(1.1, 1.3)" in boost.line()
    assert "velocity x" not in plain.line()


def test_without_the_flag_the_pool_mixes_what_it_always_mixed():
    a = _filled(GreedyLinePool(0.25, window=(0.5, 2.0), seed=11))
    b = _filled(GreedyLinePool(0.25, window=(0.5, 2.0), seed=11, speed=None))
    base = _base_pool(512)
    assert a.speed is None and b.speed is None
    for _ in range(2):
        assert a.mix(base, 50).tobytes() == b.mix(base, 50).tobytes()


@pytest.mark.parametrize("speed", [(0.0, 1.0), (1.3, 1.1), (-1.0, 2.0)])
def test_bad_ranges_are_refused(speed):
    with pytest.raises(ValueError):
        GreedyLinePool(0.2, speed=speed)


def test_the_recorder_treats_the_key_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"respawn_greedy_speed"' in src


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_needs_respawn_greedy_for_it():
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "gspeed_bad"] + SMOKE
             + ["--steps", "2048", "--respawn-greedy-speed", "1.0", "1.3"])
    shutil.rmtree(ROOT / "runs" / "gspeed_bad", ignore_errors=True)
    assert r.returncode != 0
    assert "--respawn-greedy-speed needs --respawn-greedy" in r.stdout + r.stderr


@needs_run
def test_a_resume_boosts_dumps_and_restores_it():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("gspeed_base")
    run = "gspeed_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "14336", "--respawn-greedy", "0.25",
                                       "--respawn-greedy-speed", "1.0", "1.3"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "; velocity x U(1, 1.3)" in r.stdout
    assert re.search(r"greedy-line pool: [0-9,]+ own states .*velocity x U\(1, 1\.3\)", r.stdout)
    assert _cfg(run)["respawn_greedy_speed"] == [1.0, 1.3]
    run2 = "gspeed_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "18432"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_greedy_speed=1-1.3" in r2.stdout
    assert _cfg(run2)["respawn_greedy_speed"] == [1.0, 1.3]
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)
