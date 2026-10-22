"""--respawn-greedy-turn T / --respawn-greedy-speed LO HI (2026-10-22, autoresearch r6b2): the mixed
greedy-line rows (the policy's own frontier states) nudged in heading and energy - each row's horizontal
velocity and view yaw turned together by U(-T, T) deg, its velocity scaled by U(LO, HI).

(a) The pool: the turn rotates velocity and yaw by the same angle and keeps the speed; the scale stays
    in [LO, HI]; the rows DRAWN are exactly the unperturbed pool's (own RNG); both off returns the rows
    untouched; bad settings are refused.
(b) The trainer (CPU, sidistic, a seeded reservoir): the startup line, the config keys and a bare
    resume; refused without --respawn-greedy; flag off is bit-identical to the trainer before the change
    with --respawn-greedy on.
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

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run              # noqa: E402


def _pool(**kw):
    p = GreedyLinePool(0.5, window=(0.25, 2.0), seed=5, **kw)
    rows = np.zeros(200, STATE_DTYPE)
    rng = np.random.default_rng(0)
    rows["origin"] = rng.uniform(-1000, 1000, (200, 3)).astype(np.float32)
    rows["velocity"] = rng.normal(0.0, 900.0, (200, 3)).astype(np.float32)
    rows["yaw"] = rng.uniform(-180.0, 180.0, 200).astype(np.float32)
    p.rows = rows
    return p


def _start(n=40):
    return np.zeros(n, STATE_DTYPE)


def test_the_turn_rotates_velocity_and_yaw_together_and_keeps_the_speed():
    off, on = _pool(), _pool(turn=90.0)
    a, b = off.mix(_start(), 4), on.mix(_start(), 4)
    n = off.mixed
    assert n == on.mixed == 18
    ra, rb = a[-n:], b[-n:]
    assert np.array_equal(ra["origin"], rb["origin"])            # the same rows were drawn
    va, vb = ra["velocity"].astype(np.float64), rb["velocity"].astype(np.float64)
    ang_v = np.degrees(np.arctan2(vb[:, 1], vb[:, 0]) - np.arctan2(va[:, 1], va[:, 0]))
    ang_y = rb["yaw"].astype(np.float64) - ra["yaw"].astype(np.float64)
    wrap = lambda x: (x + 180.0) % 360.0 - 180.0
    assert np.allclose(wrap(ang_v - ang_y), 0.0, atol=1e-2)
    assert np.all(np.abs(wrap(ang_y)) <= 90.0 + 1e-3) and np.abs(wrap(ang_y)).max() > 30.0
    assert np.allclose(np.hypot(va[:, 0], va[:, 1]), np.hypot(vb[:, 0], vb[:, 1]), rtol=1e-4)
    assert np.allclose(va[:, 2], vb[:, 2])


def test_the_speed_scale_stays_in_its_range_and_off_is_untouched():
    off, on = _pool(), _pool(speed=(1.0, 1.5))
    a, b = off.mix(_start(), 4), on.mix(_start(), 4)
    n = off.mixed
    sa = np.linalg.norm(a[-n:]["velocity"].astype(np.float64), axis=1)
    sb = np.linalg.norm(b[-n:]["velocity"].astype(np.float64), axis=1)
    r = sb / sa
    assert r.min() >= 1.0 - 1e-5 and r.max() <= 1.5 + 1e-5 and r.max() > 1.2
    assert np.array_equal(a[-n:]["yaw"], b[-n:]["yaw"])
    rows = off.rows[:7]
    assert off.perturb(rows) is rows


@pytest.mark.parametrize("kw,msg", [
    ({"turn": 200.0}, "--respawn-greedy-turn T needs 0 <= T <= 180"),
    ({"speed": (1.5, 1.0)}, "--respawn-greedy-speed LO HI needs 0 < LO <= HI"),
    ({"speed": (0.0, 1.0)}, "--respawn-greedy-speed LO HI needs 0 < LO <= HI"),
])
def test_bad_pool_settings_are_refused(kw, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        GreedyLinePool(0.3, **kw)


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_wires_restores_and_refuses_it():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("gpert_base")
    run = "gpert_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "10240", "--respawn-greedy", "0.25",
                                       "--respawn-greedy-turn", "60",
                                       "--respawn-greedy-speed", "1.0", "1.4"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "heading (velocity and view yaw) turned by U(-60, 60) deg" in r.stdout
    assert "velocity scaled by U(1, 1.4)" in r.stdout
    cfg = _cfg(run)
    assert cfg["respawn_greedy_turn"] == 60.0 and cfg["respawn_greedy_speed"] == [1.0, 1.4]
    run2 = "gpert_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_greedy_turn=60" in r2.stdout and "respawn_greedy_speed=1-1.4" in r2.stdout
    assert _cfg(run2)["respawn_greedy_speed"] == [1.0, 1.4]
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "gpert_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-greedy-turn", "30"])
    assert r3.returncode != 0 and "need --respawn-greedy F > 0" in r3.stdout + r3.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "gpert_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-greedy-turn" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_backward_mix import _csv
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    pre = ROOT / "python" / "_train_fast_pre_gpert.py"
    d0, seeded = _seeded_ckpt("gpert_off_base")
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("gpert_off_new", TRAIN), ("gpert_off_old", pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run, "--ckpt", str(seeded)]
                     + SMOKE + GREEDY_SMOKE + ["--steps", "10240", "--respawn-greedy", "0.25"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "gpert_off_new", ROOT / "runs" / "gpert_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_greedy_turn" not in ca and "respawn_greedy_speed" not in ca
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
    for dd in (d0, a, b):
        shutil.rmtree(dd, ignore_errors=True)
