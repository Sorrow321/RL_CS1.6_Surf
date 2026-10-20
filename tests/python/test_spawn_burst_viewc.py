"""--spawn-burst under --view-continuous (autoresearch r1b4, 2026-10-20).

Go-Explore's "explore after return" (1901.10995 sec 2.1.4): for a burst after
a respawn the env takes random HELD actions that leave the PPO pool (b_ez),
so the burst only shapes learning through the states it reaches. Until now the
flag was refused under --view-continuous (the scratch default) and --unstuck.
Now the burst also holds a random view command, a uniform draw in the view
head's tanh space turned into the core's view row by view_from_z_t, and two
knobs say how often and how long: --spawn-burst-frac (the share of respawns
that get a burst, default 1.0 = every respawn as before) and
--spawn-burst-uniform (length ~ U{1..K} instead of K).

(a) flag OFF: the trainer is the parent commit's, bit for bit (config dump,
    progress.csv without fps, the eval trajectory bytes, every weight, the
    Adam moments) on the continuous-view + keys-hold + potential + unstuck
    path the incumbent recipe runs.
(b) flag ON under --view-continuous + --unstuck: the run trains, the bursts
    fire (the step line's burst share is > 0 once episodes end), the config
    dump carries the two knobs, record_ckpt records the checkpoint (the record
    gate's own backend), and a knob without --spawn-burst is refused.

Runs on sidistic (its caches ship with the autoresearch maps); skipped where
the map, its goal field or the core are missing.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "python" / "train_fast.py"
MAPS = Path(os.environ.get("SURF_TEST_MAPS") or (ROOT / "maps"))
MAP = MAPS / "surf_src_sidistic.bsp"
FIELD = MAPS / "surf_src_sidistic.goal_32.npz"
DLL = ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so")
needs = pytest.mark.skipif(
    not (MAP.exists() and FIELD.exists() and DLL.exists()),
    reason="needs the built core + sidistic + its prebaked goal field")

FLAGS = ["--map", str(MAP), "--reward", "race", "--envs", "16",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--lidar-cell", "32", "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
         "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2", "--vf", "0.5",
         "--ent", "0.005", "--n-steps", "8", "--epochs", "1",
         "--minibatches", "2", "--ep-ticks", "120", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "4000", "--train-stride", "1",
         "--yaw-adaptive", "--respawn-frac", "0.9", "--respawn-margin", "2",
         "--respawn-reservoir", "1000", "--int-coef", "0.25", "--int-view", "8",
         "--int-speed", "3", "--ckpt-every", "1e9", "--record-every", "1e12",
         "--eval-eps", "1", "--eval-greedy-only", "--seed", "7",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold",
         "--obs-potential", "norm", "--obs-potential-curtain",
         "--unstuck", "--unstuck-reach", "eval", "--unstuck-max", "2"]


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2")


def _run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(ROOT), timeout=timeout, encoding="utf-8",
                          errors="replace")


def _train(run, extra, script=TRAIN, steps="3072"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + FLAGS
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _parent_trainer(dst: Path):
    """train_fast.py as it was before --spawn-burst-frac existed: the parent
    of the commit that added it, or HEAD while the change is uncommitted."""
    log = subprocess.run(["git", "log", "--format=%H", "-S--spawn-burst-frac",
                          "--", "python/train_fast.py"], cwd=str(ROOT),
                         capture_output=True, text=True)
    revs = log.stdout.split()
    rev = (revs[-1] + "~1") if revs else "HEAD"
    src = subprocess.run(["git", "show", f"{rev}:python/train_fast.py"],
                         cwd=str(ROOT), capture_output=True, text=True)
    if src.returncode != 0 or "--spawn-burst-frac" in src.stdout:
        return None
    dst.write_text(src.stdout, encoding="utf-8")
    return dst


@needs
def test_flag_off_is_bit_identical_to_the_parent_trainer():
    import torch
    old = ROOT / "python" / "train_fast_pre_sbviewc.py"
    if _parent_trainer(old) is None:
        pytest.skip("no pre-flag train_fast.py in the history")
    try:
        _train("sbv_pre", [], script=old)
        _train("sbv_post", [])
        a, b = ROOT / "runs" / "sbv_pre", ROOT / "runs" / "sbv_post"
        ca = json.loads((a / "run.json").read_text(encoding="utf-8"))
        cb = json.loads((b / "run.json").read_text(encoding="utf-8"))
        ca, cb = ca.get("config", ca), cb.get("config", cb)
        assert {k for k in set(ca) | set(cb)
                if k != "run" and ca.get(k) != cb.get(k)} == set()
        ra = list(csv.reader((a / "progress.csv").open(encoding="utf-8")))
        rb = list(csv.reader((b / "progress.csv").open(encoding="utf-8")))
        assert ra[0] == rb[0] and len(ra) == len(rb)
        fi = ra[0].index("time/fps")
        for x, y in zip(ra[1:], rb[1:]):
            assert [v for j, v in enumerate(x) if j != fi] == \
                   [v for j, v in enumerate(y) if j != fi]
        ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
        assert [p.name for p in ta] == [p.name for p in tb]
        for p, q in zip(ta, tb):
            assert p.read_bytes() == q.read_bytes()
        ka = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
        kb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in ka["policy"]:
            assert torch.equal(ka["policy"][k], kb["policy"][k]), k
        sa, sb = ka["optimizer"]["state"], kb["optimizer"]["state"]
        for k in sa:
            for m in sa[k]:
                if torch.is_tensor(sa[k][m]):
                    assert torch.equal(sa[k][m], sb[k][m]), (k, m)
    finally:
        old.unlink(missing_ok=True)
        for r in ("sbv_pre", "sbv_post"):
            shutil.rmtree(ROOT / "runs" / r, ignore_errors=True)


@needs
def test_burst_on_trains_fires_dumps_records_and_refuses():
    run = "sbv_on"
    try:
        r = _train(run, ["--spawn-burst", "12", "--spawn-burst-frac", "0.5",
                         "--spawn-burst-uniform"])
        shares = [float(l.rsplit("burst ", 1)[1].rstrip("%")) for l in
                  r.stdout.splitlines() if l.startswith("step ") and " burst " in l]
        assert shares and max(shares) > 0.0, r.stdout[-3000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))
        cfg = cfg.get("config", cfg)
        assert cfg["spawn_burst"] == 12
        assert cfg["spawn_burst_frac"] == 0.5
        assert cfg["spawn_burst_uniform"] == 1
        out = ROOT / "runs" / run / "rec_sbv.jsonl"
        r2 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                   str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(out),
                   "--episodes", "1", "--ep-ticks", "200"])
        assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
        assert out.exists() and out.stat().st_size > 0
        r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "sbv_bad"] + FLAGS
                  + ["--steps", "512", "--spawn-burst-frac", "0.5"])
        assert r3.returncode != 0
        assert "need --spawn-burst K > 0" in r3.stdout + r3.stderr
    finally:
        for r in (run, "sbv_bad"):
            shutil.rmtree(ROOT / "runs" / r, ignore_errors=True)
