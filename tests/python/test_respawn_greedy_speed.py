"""--respawn-greedy-speed LO HI (2026-10-22, autoresearch r6b4): each greedy-line spawn row's
velocity is scaled by U(LO, HI) when it is mixed into the training pool - what --respawn-speed
does to the reservoir rows, which never reached the greedy rows (they are mixed in AFTER the
reservoir's build_pool).

(a) The pool: with the flag the rows drawn are exactly the rows drawn without it (own RNG), only
    their velocity is scaled, inside [LO, HI], per row; without it nothing is scaled and no draw is
    made; bad ranges are refused.
(b) The trainer (CPU, sidistic): the flag reaches the pool (startup line), the config and a bare
    resume; it is refused without --respawn-greedy; flag off is bit-identical to the trainer
    before the change (config, progress.csv, eval bytes, weights).
"""
from __future__ import annotations

import json
import re
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

from test_ez_viewc import SMOKE, TRAIN, _csv, _run, _train, needs_run  # noqa: E402


def _filled(pool, n=500, seed=3):
    rng = np.random.default_rng(seed)
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"] = rng.uniform(-1000, 1000, (n, 3)).astype(np.float32)
    rows["velocity"] = rng.uniform(-1500, 1500, (n, 3)).astype(np.float32)
    rows["yaw"] = rng.uniform(0, 360, n).astype(np.float32)
    pool.rows = rows
    return rows


def _base(n=4096):
    p = np.zeros(n, STATE_DTYPE)
    p["origin"][:, 0] = np.arange(n, dtype=np.float32)
    p["velocity"][:, 1] = 7.0
    return p


def test_the_same_rows_are_drawn_and_only_their_velocity_is_scaled():
    off = GreedyLinePool(0.3, window=(0.25, 2.0), seed=11)
    on = GreedyLinePool(0.3, window=(0.25, 2.0), seed=11, speed=(1.0, 1.5))
    src = _filled(off)
    _filled(on)
    for _ in range(3):                      # several iterations: the streams stay aligned
        a, b = off.mix(_base(), 410), on.mix(_base(), 410)
        n = off.mixed
        assert n == on.mixed == int(round((4096 - 410) * 0.3))
        assert np.array_equal(a[:-n], b[:-n])                 # head untouched by both
        assert np.array_equal(a["origin"], b["origin"])        # the same rows were drawn
        assert np.array_equal(a["yaw"], b["yaw"])
        va, vb = a["velocity"][-n:].astype(np.float64), b["velocity"][-n:].astype(np.float64)
        # off: the rows are the pool's own states, velocity untouched
        vel = {tuple(v) for v in src["velocity"].tolist()}
        assert all(tuple(v) in vel for v in a["velocity"][-n:].tolist())
        na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
        k = na > 1.0
        ratio = nb[k] / na[k]
        assert ratio.min() >= 1.0 - 1e-5 and ratio.max() <= 1.5 + 1e-5
        assert ratio.std() > 0.05                              # per row, not one global factor
        # the direction is kept
        cos = (va[k] * vb[k]).sum(1) / (na[k] * nb[k])
        assert cos.min() > 1.0 - 1e-6


def test_off_scales_nothing_and_makes_no_extra_draw():
    p = GreedyLinePool(0.3, window=(0.25, 2.0), seed=5)
    assert p.speed is None and p.srng is None
    src = _filled(p)
    out = p.mix(_base(), 410)
    n = p.mixed
    vel = {tuple(v) for v in src["velocity"].tolist()}
    assert all(tuple(v) in vel for v in out["velocity"][-n:].tolist())
    assert "velocity x" not in p.line()


@pytest.mark.parametrize("speed", [(0.0, 1.2), (1.5, 1.0), (-1.0, 1.0)])
def test_bad_ranges_are_refused(speed):
    with pytest.raises(ValueError):
        GreedyLinePool(0.3, window=(0.25, 2.0), speed=speed)


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
                                       "--respawn-greedy-speed", "1.1", "1.4"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "each row's velocity x U(1.1, 1.4)" in r.stdout
    assert re.search(r"greedy-line pool: [0-9,]+ own states", r.stdout)
    assert _cfg(run)["respawn_greedy_speed"] == [1.1, 1.4]
    run2 = "gspeed_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_greedy_speed=1.1-1.4" in r2.stdout
    assert _cfg(run2)["respawn_greedy_speed"] == [1.1, 1.4]
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "gspeed_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-greedy-speed", "1.0", "1.5"])
    assert r3.returncode != 0 and "needs --respawn-greedy F > 0" in r3.stdout + r3.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "gspeed_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this flag."""
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
    pre = ROOT / "python" / "_train_fast_pre_gspeed.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("gspeed_off_new", [])
        _train("gspeed_off_old", [], script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "gspeed_off_new", ROOT / "runs" / "gspeed_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb and "respawn_greedy_speed" not in ca
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
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
