"""--respawn-vel-rot: a share of the spawn pool's reservoir rows inside the frontier zone
get a fresh velocity DIRECTION (uniform heading, elevation U(-45, 45) deg), speed kept,
the view yaw turned by the same heading change.

(a) the rotation math (a replica of vel_rot_rows' arithmetic, pinned to the trainer's
    source): speed kept exactly, elevation bounded, yaw follows the heading;
(b) the refusals;
(c) flag-off identity: the trainer without the flag reproduces the newest committed
    trainer that has no such flag (config dump, progress.csv, eval trajectory bytes).
The flow itself (the share printed as `vrot` once the reservoir passes the pool's
2,000-state floor) was smoked by hand on sidistic (2026-10-20: 0% below the floor,
49-51% after it at --respawn-vel-rot 0.5) - 3.5 min of CPU, too slow for the suite.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_view_ou_temp import ROOT, SMOKE, TRAIN, _csv, _run, _train, needs_run  # noqa: E402


def _rotate(vel, yaw, rng):
    """The trainer's arithmetic, verbatim."""
    v = vel.astype(np.float64)
    spd = np.linalg.norm(v, axis=1)
    head0 = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    head = rng.uniform(-180.0, 180.0, len(v))
    elev = np.radians(rng.uniform(-45.0, 45.0, len(v)))
    hr = np.radians(head)
    nv = np.stack([np.cos(elev) * np.cos(hr), np.cos(elev) * np.sin(hr),
                   np.sin(elev)], axis=1) * spd[:, None]
    ny = np.mod(yaw.astype(np.float64) + (head - head0), 360.0)
    return nv.astype(np.float32), ny.astype(np.float32), head


def test_rotation_keeps_speed_bounds_elevation_and_turns_the_view_with_it():
    rng = np.random.default_rng(0)
    vel = rng.normal(0, 900, (4000, 3)).astype(np.float32)
    yaw = rng.uniform(0, 360, 4000).astype(np.float32)
    nv, ny, head = _rotate(vel, yaw, np.random.default_rng(1))
    s0, s1 = np.linalg.norm(vel, axis=1), np.linalg.norm(nv, axis=1)
    assert np.allclose(s0, s1, rtol=1e-4, atol=1e-2)
    elev = np.degrees(np.arcsin(np.clip(nv[:, 2] / np.maximum(s1, 1e-6), -1, 1)))
    assert elev.min() >= -45.01 and elev.max() <= 45.01
    assert elev.std() > 20.0                                  # actually spread out
    # the view-minus-heading angle is preserved
    off0 = np.mod(yaw - np.degrees(np.arctan2(vel[:, 1], vel[:, 0])), 360.0)
    off1 = np.mod(ny - np.degrees(np.arctan2(nv[:, 1], nv[:, 0])), 360.0)
    d = np.abs(((off1 - off0) + 180.0) % 360.0 - 180.0)
    assert d.max() < 0.05
    # headings uniform: every 90-deg quadrant gets ~a quarter
    q = np.histogram(head, bins=[-180, -90, 0, 90, 180])[0] / len(head)
    assert np.all(np.abs(q - 0.25) < 0.03)


def test_the_trainer_carries_that_arithmetic_and_the_keys():
    src = TRAIN.read_text(encoding="utf-8")
    for frag in ('ap.add_argument("--respawn-vel-rot", type=float, default=None',
                 'elev = np.radians(vr_rng.uniform(-45.0, 45.0, len(pick)))',
                 'rp["velocity"][rows] = nv.astype(np.float32)',
                 '+ (head - head0), 360.0).astype(np.float32)',
                 "if VEL_ROT > 0.0:\n                        _rp = vel_rot_rows(_s, _rp)"):
        assert frag in src, frag
    rec = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"respawn_vel_rot", "respawn_vel_rot_front",' in rec


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--respawn-vel-rot", "0.5", "--respawn-frac", "0"], "it needs --respawn-frac > 0"),
    (["--respawn-vel-rot", "1.5"], "--respawn-vel-rot must be in [0, 1]"),
    (["--respawn-vel-rot", "0.5", "--respawn-novel"], "not implemented with --respawn-novel"),
])
def test_refusals(extra, msg):
    shutil.rmtree(ROOT / "runs" / "r1b3_vrot_refuse", ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "r1b3_vrot_refuse"] + SMOKE
             + ["--steps", "1024", "--record-every", "1e12"] + extra, timeout=600)
    shutil.rmtree(ROOT / "runs" / "r1b3_vrot_refuse", ignore_errors=True)
    assert r.returncode != 0
    assert msg in (r.stdout + r.stderr)


def _parent_trainer() -> Path | None:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--first-parent",
                        "--max-count=60", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    for ref in r.stdout.split():
        g = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if g.returncode != 0 or not g.stdout or "--respawn-vel-rot" in g.stdout:
            continue
        p = ROOT / "python" / "_train_fast_parent_for_vrot_test.py"
        p.write_text(g.stdout, encoding="utf-8")
        return p
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_parent_trainer():
    parent = _parent_trainer()
    if parent is None:
        pytest.skip("git could not serve a pre-flag train_fast.py")
    a, b = "r1b3_vrot_off_now", "r1b3_vrot_off_par"
    try:
        _train(a, [], record="2048")
        _train(b, [], script=parent, record="2048")
        ca = json.loads((ROOT / "runs" / a / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((ROOT / "runs" / b / "run.json").read_text(encoding="utf-8"))["config"]
        assert ca == cb, "the flag-off config dump changed"
        ra, rb = _csv(a), _csv(b)
        assert list(ra[0]) == list(rb[0]) and len(ra) == len(rb)
        for xa, xb in zip(ra, rb):
            for k in xa:
                if not k.startswith("time/"):
                    assert xa[k] == xb[k], (k, xa[k], xb[k])
        ta = sorted((ROOT / "runs" / a).glob("traj_*.jsonl"))
        tb = sorted((ROOT / "runs" / b).glob("traj_*.jsonl"))
        assert ta and [x.name for x in ta] == [x.name for x in tb]
        for xa, xb in zip(ta, tb):
            assert xa.read_bytes() == xb.read_bytes(), xa.name
    finally:
        parent.unlink(missing_ok=True)
        for dd in (a, b):
            shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)
