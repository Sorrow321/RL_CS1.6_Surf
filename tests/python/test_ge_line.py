"""--ge-line (unr1b1, 2026-11-08): the Go-Explore -> backward-curriculum loop with the Nature
paper's explore step. The trainer hands its best greedy episode to tools/auto_spine.py --line
(no beam, no policy: the uniform explorer alone), beside the training, and a line that beats
the greedy becomes a DemoCurriculum on its own share of the spawn pool.

CPU, surf_src_sidistic: auto_spine --line on a scripted line (no beam directory, the line+ge
source); the trainer's refusals; a smoke run in which a search starts, ends and its outcome is
applied; TRAIN_ONLY in record_ckpt; flag off bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
TRAIN = ROOT / "python" / "train_fast.py"
SD = ROOT / "maps" / "surf_src_sidistic.bsp"
SD_FIELD = ROOT / "maps" / "surf_src_sidistic.goalc_32.npz"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_sd = pytest.mark.skipif(
    not (SD.exists() and DLL.exists() and SD_FIELD.exists()
         and (ROOT / "maps" / "surf_src_sidistic.goal_32.npz").exists()),
    reason="needs the built core + surf_src_sidistic and its prebaked goal fields")

SMOKE = ["--map", str(SD), "--reward", "race", "--envs", "64",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--lidar-cell", "32", "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
         "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2", "--vf", "0.5",
         "--ent", "0.005", "--n-steps", "8", "--epochs", "1",
         "--minibatches", "2", "--ep-ticks", "3000", "--time-pen", "0.001",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--race-no-corner-cut", "1", "--maxvel", "4000",
         "--train-stride", "1", "--yaw-adaptive", "--respawn-frac", "0.9",
         "--respawn-margin", "2", "--respawn-reservoir", "1000", "--ckpt-every", "1e9",
         "--eval-eps", "2", "--eval-greedy-only",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold",
         "--no-compile", "--fp32", "--seed", "0",
         "--race-ratchet", "--respawn-greedy", "0.25", "--record-every", "2048"]


def _env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
               PYTHONIOENCODING="utf-8")
    env.pop("SELF_STATES", None)
    return env


def _train(run, extra, steps=8192, script=TRAIN):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    return subprocess.run([sys.executable, "-u", str(script), "--run", run] + SMOKE
                          + ["--steps", str(steps)] + list(extra), capture_output=True,
                          text=True, env=_env(), cwd=str(ROOT), timeout=2400,
                          encoding="utf-8", errors="replace")


def _csv(run):
    import csv
    p = ROOT / "runs" / run / "progress.csv"
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _walk_line(path, n=400):
    """The true spawn walked forward (no turn, forward held) until it dies: a real line."""
    from surfgym.core import SurfCore, default_config
    from surfgym.rewards import map_spawn_pool
    cfg = default_config(num_envs=1, lidar_w=0, lidar_h=0)
    cfg.phys.sv_maxvelocity = 4000.0
    core = SurfCore(str(SD), cfg)
    core.set_teleport_fail(True)
    core.reset(0)
    core.set_state(0, map_spawn_pool(core)[0].copy())
    rows, a = [], np.zeros((1, 6), np.int32)
    a[0, 0], a[0, 1], a[0, 2] = 7, 3, 2
    for _ in range(n):
        rows.append(core.get_states()[0].copy())
        _, _, d, tr, _ = core.step(a)
        if d[0] or tr[0]:
            break
    core.close()
    np.save(path, np.array(rows))
    return len(rows)


def test_train_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    for k in ("ge_line", "ge_line_minutes", "ge_line_gain", "ge_line_success",
              "ge_line_threads"):
        assert k in record_ckpt.TRAIN_ONLY, k


@needs_sd
def test_auto_spine_line_mode_runs_the_explorer_alone(tmp_path):
    line = tmp_path / "line.npy"
    n = _walk_line(line)
    assert n >= 50
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "auto_spine.py"),
                        "--line", str(line), "--map", str(SD), "--field", str(SD_FIELD),
                        "--out", str(out), "--greedy-d", "99000", "--gain", "200",
                        "--envs", "64", "--ge-minutes", "0.1", "--ge-boost", "1.0",
                        "--ge-select-energy", "1", "--act-every", "4", "--threads", "2",
                        "--maxvel", "4000"], capture_output=True, text=True, env=_env(),
                       cwd=str(ROOT), timeout=900, encoding="utf-8", errors="replace")
    txt = r.stdout + r.stderr
    m = [ln for ln in r.stdout.splitlines() if ln.startswith("AUTO_SPINE ")]
    assert m, txt[-4000:]
    res = json.loads(m[-1][len("AUTO_SPINE "):])
    assert res["line"] == str(line)
    assert not (out / "beam").exists() and not (out / "beam_br").exists()   # no policy search
    assert "explore_phase1.py" in txt and "--view-follow" in txt and "--select-energy" in txt
    if res.get("spine"):
        from surfgym.core import STATE_DTYPE
        sp = np.load(res["spine"])
        assert sp.dtype == STATE_DTYPE and res["source"] == "line+ge"
        # the spine starts with the line itself: the policy's own states up to the root
        assert np.allclose(sp[0]["origin"], np.load(line)[0]["origin"])
        assert r.returncode == 0
    else:
        assert r.returncode == 2


@needs_sd
def test_refusals():
    bad = _train("gel_bad1", ["--ge-line", "1.5"], steps=512)
    assert bad.returncode != 0 and "--ge-line must be in (0, 1)" in bad.stdout + bad.stderr
    flags = [f for f in SMOKE if f != "--race-ratchet"]
    shutil.rmtree(ROOT / "runs" / "gel_bad2", ignore_errors=True)
    bad = subprocess.run([sys.executable, "-u", str(TRAIN), "--run", "gel_bad2"] + flags
                         + ["--steps", "512", "--ge-line", "0.2"], capture_output=True,
                         text=True, env=_env(), cwd=str(ROOT), timeout=900,
                         encoding="utf-8", errors="replace")
    assert bad.returncode != 0 and "no --race-ratchet" in bad.stdout + bad.stderr
    for run in ("gel_bad1", "gel_bad2"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_sd
def test_the_search_runs_beside_the_training_and_its_line_is_applied():
    run = "gel_smoke"
    try:
        r = _train(run, ["--ge-line", "0.2", "--ge-line-minutes", "0.15"], steps=16384)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--ge-line 0.2: after every greedy eval" in out
        m = re.search(r"\[\s*([\d,]+)\] --ge-line search 1: Go-Explore from episode", out)
        assert m, out[-6000:]
        started = int(m.group(1).replace(",", ""))
        e = re.search(r"\[\s*([\d,]+)\] --ge-line search 1: (NEW LINE|no line|line NOT)", out)
        assert e, out[-6000:]
        assert int(e.group(1).replace(",", "")) > started        # the training went on
        g1 = ROOT / "runs" / run / "gel" / "g001"
        assert (g1 / "line.npy").exists() and (g1 / "auto_spine.log").exists()
        log = (g1 / "auto_spine.log").read_text()
        assert "AUTO_SPINE " in log and '"line": ' in log
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["ge_line"] == 0.2 and cfg["ge_line_minutes"] == 0.15
        if "NEW LINE" in e.group(2):
            cur = json.loads((ROOT / "runs" / run / "gel" / "current.json").read_text())
            assert cur["end_d"] < cur["greedy_best_d"] - 200.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--ge-line" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_sd
def test_flag_off_is_bit_identical():
    """Flag off: the config and every logged number match the trainer before the change."""
    pre = ROOT / "python" / "_train_fast_pre_gel.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("gel_off_new", TRAIN), ("gel_off_old", pre)):
            r = _train(run, [], steps=8192, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "gel_off_new", ROOT / "runs" / "gel_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "ge_line" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        assert not (a / "gel").exists()
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
