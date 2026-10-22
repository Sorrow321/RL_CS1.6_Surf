"""--respawn-greedy-speed LO HI (2026-10-22, autoresearch r5b1): each greedy-line row mixed into
the spawn pool has its velocity scaled by U(LO, HI) - the deployed line's own pre-failure states
replayed faster (or slower) than it flew them.

(a) The pool: off (None) is the shipped mix byte for byte (rows and the row generator's state);
    on, the rows drawn are exactly the unscaled ones (its own generator), only their velocity
    changes, each by one factor in [LO, HI]; the head of the pool is untouched; bad ranges are
    refused; the pool's log line says so only when on.
(b) The trainer (CPU, sidistic): the flag reaches the pool (startup line), the config and a bare
    resume; it is refused without --respawn-greedy and with LO > HI.
"""
from __future__ import annotations

import json
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


def _filled(**kw):
    """A pool holding 500 distinct rows (origin x = row id, velocity (100 + id, -50, 7))."""
    pool = GreedyLinePool(0.3, window=(0.25, 2.0), seed=11, **kw)
    rows = np.zeros(500, STATE_DTYPE)
    rows["origin"][:, 0] = np.arange(500, dtype=np.float32)
    rows["velocity"][:, 0] = 100.0 + np.arange(500, dtype=np.float32)
    rows["velocity"][:, 1] = -50.0
    rows["velocity"][:, 2] = 7.0
    pool.rows = rows
    return pool


def _base_pool(n=4096):
    p = np.zeros(n, STATE_DTYPE)
    p["origin"][:, 0] = -1.0 - np.arange(n, dtype=np.float32)
    p["velocity"][:, 0] = 3.0
    return p


def test_off_is_the_shipped_mix_byte_for_byte():
    a, b = _filled(), _filled(speed=None)
    for _ in range(3):
        pa, pb = a.mix(_base_pool(), 410), b.mix(_base_pool(), 410)
        assert pa.tobytes() == pb.tobytes()
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert b.speed is None and b.speed_rng is None
    assert "speed" not in b.line()


def test_on_scales_only_the_velocity_of_the_rows_it_mixed_in():
    off, on = _filled(), _filled(speed=(1.0, 1.5))
    for _ in range(3):
        po, pn = off.mix(_base_pool(), 410), on.mix(_base_pool(), 410)
        n = off.mixed
        assert n == on.mixed == int(round((4096 - 410) * 0.3))
        head, tail = slice(0, 4096 - n), slice(4096 - n, 4096)
        assert po[head].tobytes() == pn[head].tobytes()          # start + reservoir rows
        # the same rows were drawn (its own generator): every field but the velocity agrees
        for f in STATE_DTYPE.names:
            if f != "velocity":
                assert po[tail][f].tobytes() == pn[tail][f].tobytes(), f
        k = pn["velocity"][tail] / po["velocity"][tail]
        assert np.allclose(k, k[:, :1], rtol=1e-6)                # one factor per row
        assert k.min() >= 1.0 - 1e-6 and k.max() <= 1.5 + 1e-6
        assert k[:, 0].std() > 0.05                               # it really draws
    assert off.rng.bit_generator.state == on.rng.bit_generator.state
    assert "at x1..1.5 speed" in on.line()


def test_a_bad_range_is_refused():
    for bad in ((0.0, 1.5), (-1.0, 1.0), (1.5, 1.0)):
        with pytest.raises(ValueError):
            GreedyLinePool(0.3, speed=bad)
    GreedyLinePool(0.3, speed=(1.2, 1.2))                         # a fixed factor is fine


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_wires_restores_and_refuses_it():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("gspeed_base")
    run = "gspeed_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "10240", "--respawn-greedy", "0.25",
                                       "--respawn-greedy-speed", "1.0", "1.5"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "velocity x U(1, 1.5) per row (--respawn-greedy-speed)" in r.stdout
    assert _cfg(run)["respawn_greedy_speed"] == [1.0, 1.5]
    run2 = "gspeed_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_greedy_speed=1-1.5" in r2.stdout
    assert _cfg(run2)["respawn_greedy_speed"] == [1.0, 1.5]
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "gspeed_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-greedy-speed", "1.0", "1.5"])
    assert r3.returncode != 0 and "needs --respawn-greedy F > 0" in r3.stdout + r3.stderr
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "gspeed_bad2"] + SMOKE
              + ["--steps", "2048", "--respawn-greedy", "0.25",
                 "--respawn-greedy-speed", "1.5", "1.0"])
    assert r4.returncode != 0 and "needs 0 < LO <= HI" in r4.stdout + r4.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "gspeed_bad",
               ROOT / "runs" / "gspeed_bad2"):
        shutil.rmtree(dd, ignore_errors=True)
