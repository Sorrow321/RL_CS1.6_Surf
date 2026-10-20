"""--ez-eps under --view-continuous (2026-10-20, autoresearch r1b1).

ez-greedy (Dabney et al. 2021) was refused under --view-continuous and under
--unstuck. It now runs there: a burst holds a uniform draw of the yaw z
(tanh(z) uniform on (-1, 1), so the yaw offset is warped like the policy's
own) and of the four key heads for n ~ zeta(mu) decisions; the pitch stays
the policy's draw; the rows are dropped from the PPO pool by b_ez, exactly
as on the bins.

(a) CPU smoke on sidistic (abs view, keys-hold, unstuck at its defaults):
    bursts run, the step line reports a non-zero burst share, the config
    carries ez_eps, and a flagless resume restores it.
(c) --ez-front B: bursts start only inside the frontier band (progress within
    the top B of the reservoir's deepest), re-read once per iteration.
(b) flag off: the trainer before this change and after it give the same
    config, progress.csv (fps aside), eval trajectory bytes and weights on
    the same smoke - ez_eps 0 never enters the burst code.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "python" / "train_fast.py"
SID = ROOT / "maps" / "surf_src_sidistic.bsp"
SID_FIELD = ROOT / "maps" / "surf_src_sidistic.goal_32.npz"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_run = pytest.mark.skipif(
    not (SID.exists() and SID_FIELD.exists() and DLL.exists()),
    reason="needs the built core + sidistic and its prebaked goal field")

SMOKE = ["--map", str(SID), "--reward", "race", "--envs", "64",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--lidar-cell", "32", "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
         "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2", "--vf", "0.5",
         "--ent", "0.005", "--n-steps", "8", "--epochs", "1",
         "--minibatches", "2", "--ep-ticks", "3000", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "4000", "--train-stride", "1",
         "--yaw-adaptive", "--respawn-frac", "0.9", "--respawn-margin", "2",
         "--respawn-reservoir", "1000", "--int-coef", "0.25", "--int-view", "8",
         "--int-speed", "3", "--ckpt-every", "1e9", "--record-every", "1e12",
         "--eval-eps", "1", "--eval-greedy-only", "--seed", "7",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold"]


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(ROOT), timeout=timeout, encoding="utf-8",
                          errors="replace")


def _train(run, extra, script=TRAIN, steps="6144"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


@needs_run
def test_ez_bursts_run_under_the_abs_view_and_unstuck_and_resume():
    run = "ezviewc_on"
    r = _train(run, ["--ez-eps", "0.5", "--ez-max", "8", "--unstuck",
                     "--unstuck-reach", "eval"])
    shares = [float(m) for m in re.findall(r"  ez +([0-9.]+)%", r.stdout)]
    assert shares and all(s > 0.0 for s in shares), r.stdout[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["ez_eps"] == 0.5 and cfg["ez_max"] == 8
    assert cfg["view_continuous"] and cfg["unstuck"]
    # a flagless resume restores the burst rate and keeps bursting
    run2 = "ezviewc_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "ez_eps=0.5" in r2.stdout
    assert re.search(r"  ez +[0-9.]+%", r2.stdout)
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
def test_ez_front_bands_the_burst_starts_and_resumes():
    """--ez-front 0.5: bursts start only in the top half of the reservoir's
    reach. A 4 s episode cap and a 0.1 s harvest margin fill the reservoir
    (1 s snapshots) inside the smoke, so the band is live (the step line prints
    where it starts), the config carries the key and a flagless resume
    restores it; --ez-front without --ez-eps is refused."""
    run = "ezviewc_front"
    r = _train(run, ["--ez-eps", "0.5", "--ez-max", "8", "--ez-front", "0.5",
                     "--ep-ticks", "400", "--respawn-margin", "0.1"],
               steps="40960")
    assert re.search(r"  ez +[0-9.]+% front>[0-9.]+%", r.stdout), r.stdout[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["ez_front"] == 0.5 and cfg["ez_eps"] == 0.5
    run2 = "ezviewc_front_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "ez_front=0.5" in r2.stdout
    bad = _run([sys.executable, "-u", str(TRAIN), "--run", "ezviewc_front_bad"]
               + SMOKE + ["--steps", "2048", "--ez-front", "0.5"])
    assert bad.returncode != 0 and "needs --ez-eps > 0" in (bad.stdout + bad.stderr)
    for dd in (d, ROOT / "runs" / run2, ROOT / "runs" / "ezviewc_front_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200",
                        "HEAD"], capture_output=True, text=True, cwd=str(ROOT),
                       timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"ez_z = (" not in s.stdout and b"--ez-front" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_ezviewc.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("ezviewc_off_new", [])
        _train("ezviewc_off_old", [], script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "ezviewc_off_new", ROOT / "runs" / "ezviewc_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb and ca["ez_eps"] == 0.0
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
