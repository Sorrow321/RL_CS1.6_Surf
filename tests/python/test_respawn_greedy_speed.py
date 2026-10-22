"""--respawn-greedy-speed LO HI (2026-10-22, autoresearch r6b3): each greedy-line spawn row's
velocity scaled by U(LO, HI), direction kept - the reservoir's --respawn-speed curriculum applied
where the greedy line FAILS (sidistic's finish: a climb out of a basement that the line reaches
too slowly).

(a) The pool: the mixed rows are exactly the unscaled pool's rows with the velocity times a factor
    in [LO, HI] (same rows drawn: the scale has its own generator); the head of the pool is
    untouched; (1, 1) is byte-identical to a pool built without the option; bad ranges are refused;
    the recorder lists the flag as training-only.
(b) The trainer (CPU, sidistic): the flag reaches the pool (startup line), the config and a bare
    resume; it is refused without --respawn-greedy.
(c) Flag off: the trainer before this change and after it, both with --respawn-greedy on, give the
    same config, progress.csv (fps aside), eval trajectory bytes and weights.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool                    # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _csv, _run, needs_run       # noqa: E402


def _filled(pool, n=600):
    """Fill the pool's rows directly: distinct origins, velocity (100 + i, -50, 30)."""
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"][:, 0] = np.arange(n, dtype=np.float32)
    rows["velocity"][:, 0] = 100.0 + np.arange(n, dtype=np.float32)
    rows["velocity"][:, 1] = -50.0
    rows["velocity"][:, 2] = 30.0
    pool.rows = rows
    return pool


def _base_pool():
    p = np.zeros(4096, STATE_DTYPE)
    p["origin"][:, 1] = np.arange(4096, dtype=np.float32)
    p["velocity"][:, 0] = 7.0
    return p


# ==========================================================================
# (a) the pool
# ==========================================================================
def test_the_scaled_rows_are_the_unscaled_rows_times_a_factor_in_range():
    off = _filled(GreedyLinePool(0.3, seed=5))
    on = _filled(GreedyLinePool(0.3, seed=5, speed=(1.5, 2.0)))
    a, b = off.mix(_base_pool(), 410), on.mix(_base_pool(), 410)
    n = off.mixed
    assert n == on.mixed == int(round((4096 - 410) * 0.3)) and n > 0
    # the head is untouched by either
    assert a[:4096 - n].tobytes() == b[:4096 - n].tobytes()
    ta, tb = a[4096 - n:], b[4096 - n:]
    assert np.array_equal(ta["origin"], tb["origin"])          # the same rows drawn
    va, vb = ta["velocity"].astype(np.float64), tb["velocity"].astype(np.float64)
    k = vb[:, 0] / va[:, 0]
    assert (k >= 1.5 - 1e-6).all() and (k <= 2.0 + 1e-6).all()
    assert k.std() > 0.05                                      # a factor per row
    assert np.allclose(vb, va * k[:, None], rtol=1e-5)          # direction kept
    for f in ("yaw", "pitch", "onground", "tick"):
        assert np.array_equal(ta[f], tb[f]), f
    assert "speed x U(1.5, 2)" in on.line() and "speed x" not in off.line()


def test_one_one_is_the_pool_without_the_option_byte_for_byte():
    a = _filled(GreedyLinePool(0.3, seed=11))
    b = _filled(GreedyLinePool(0.3, seed=11, speed=(1.0, 1.0)))
    assert b.speed_rng is None
    for _ in range(3):
        assert a.mix(_base_pool(), 410).tobytes() == b.mix(_base_pool(), 410).tobytes()
    assert a.line() == b.line()


@pytest.mark.parametrize("speed", [(0.0, 1.0), (-1.0, 2.0), (2.0, 1.5)])
def test_bad_ranges_are_refused(speed):
    with pytest.raises(ValueError):
        GreedyLinePool(0.3, speed=speed)


def test_the_recorder_treats_the_flag_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"respawn_greedy_speed"' in src


# ==========================================================================
# (b) the trainer
# ==========================================================================
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
                                       "--respawn-greedy-speed", "1.2", "1.8"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "velocity x U(1.2, 1.8) per row" in r.stdout
    assert "speed x U(1.2, 1.8)" in r.stdout
    assert _cfg(run)["respawn_greedy_speed"] == [1.2, 1.8]
    run2 = "gspeed_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_greedy_speed=1.2-1.8" in r2.stdout
    assert _cfg(run2)["respawn_greedy_speed"] == [1.2, 1.8]
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "gspeed_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-greedy-speed", "1.0", "1.5"])
    assert r3.returncode != 0 and "needs --respawn-greedy F > 0" in r3.stdout + r3.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "gspeed_bad"):
        shutil.rmtree(dd, ignore_errors=True)


# ==========================================================================
# (c) flag off
# ==========================================================================
def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change (it imports only
    GreedyLinePool from surfgym.greedy_spawn, whose default this change leaves as it was)."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-greedy-speed" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    pre = ROOT / "python" / "_train_fast_pre_gspeed.py"
    d0, seeded = _seeded_ckpt("gspeed_off_base")
    runs = {"gspeed_off_new": TRAIN, "gspeed_off_old": pre}
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in runs.items():
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run, "--ckpt", str(seeded)]
                     + SMOKE + GREEDY_SMOKE + ["--steps", "14336", "--respawn-greedy", "0.25"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "gspeed_off_new", ROOT / "runs" / "gspeed_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_greedy_speed" not in ca and ca["respawn_greedy"] == 0.25
    ra, rb = _csv(a.name), _csv(b.name)
    assert ra and len(ra) == len(rb) and list(ra[0]) == list(rb[0])
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert ta and [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    assert sa["respawn_greedy"]["rows"].tobytes() == sb["respawn_greedy"]["rows"].tobytes()
    for dd in (d0, a, b):
        shutil.rmtree(dd, ignore_errors=True)
