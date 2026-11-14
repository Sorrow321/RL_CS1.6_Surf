"""--int-coef-progress HI:LO:P (ab/intp, 2026-11-14): the count novelty's coefficient follows the
run's own greedy frontier.

Why: the novelty weight conflicts between maps (ledger f26bef37, the uf2 arm ab6u_2). Celestial
passes its fork and finishes with --int-coef 0.25 (906.5M) and stalls at the fork with 2.5;
unitfarmer2 dives into its pit with 2.5 (9/9 at 151-227M in four runs) and does not with 0.25
(0/9 by 453.5M). Where each run is stuck differs: unitfarmer2 at the start (its greedy frontier
at 1-4% of the spawn's d for billions of steps, the pit a dip BEHIND it), celestial at the fork
(25-30%). One setting for every map: 2.5 at progress 0, falling linearly to 0.25 at 10%.

The measure is the greedy eval's frontier - 1 - the median over its episodes of the CONTACT-BANKED
deepest d / the spawn's d - not the reservoir's 'mind': the reservoir holds curriculum and
greedy-row spawns and death dives (ab5c_1 read 'mind' 2.1% of d0, i.e. 97.9% progress, while
its greedy sat at the fork at ~28%), and banking drops a fatal dive after the last contact.

1. The pieces: the spec, the interpolation, the frontier reading (median, banked, a dip behind
   the frontier reads as no progress), the checkpoint state.
2. The wiring in train_fast (source).
3. CPU runs (the built core + a map with its prebaked goal field; INTP_TEST_MAP / SURFCORE_DLL
   point at them): the flag OFF is bit-identical to the parent commit; the flag ON reads every
   greedy eval, sets the coefficient, checkpoints it, and a resume restores it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

G_STEP = -800.0 * 0.01         # vz's change per 10 ms tick in free flight (sv_gravity 800)


# --- 1. the pieces --------------------------------------------------------------------------------

def test_the_spec_parses_and_bad_ones_are_refused():
    from train_fast import parse_int_coef_progress
    assert parse_int_coef_progress("2.5:0.25:0.1") == (2.5, 0.25, 0.1)
    for bad in ("2.5:0.25", "a:b:c", "2.5:0.25:0", "2.5:0.25:1.5", "-1:0.25:0.1",
                "2.5:-0.1:0.1"):
        with pytest.raises(SystemExit):
            parse_int_coef_progress(bad)


def test_the_interpolation():
    from train_fast import int_coef_at
    hi, lo, p = 2.5, 0.25, 0.10
    assert int_coef_at(0.0, hi, lo, p) == 2.5
    assert int_coef_at(-0.08, hi, lo, p) == 2.5            # a dip below the spawn's d
    assert int_coef_at(float("nan"), hi, lo, p) == 2.5     # no reading yet
    assert int_coef_at(0.05, hi, lo, p) == pytest.approx(1.375)
    assert int_coef_at(0.10, hi, lo, p) == pytest.approx(0.25)
    assert int_coef_at(0.28, hi, lo, p) == pytest.approx(0.25)   # celestial at its fork
    assert int_coef_at(0.016, hi, lo, p) == pytest.approx(2.14)  # unitfarmer2 diving


class _Field:
    """d = x (reachable everywhere)."""
    reach_max, cell = 1e9, 32.0

    def sample(self, o):
        return np.asarray(o, np.float64)[:, 0]


def _episode(d, contact):
    """Rows [tick, x, y, z, vx, vy, vz]: x carries d; vz falls by one gravity step per tick in
    free flight and holds (the map pushed back) where ``contact`` is True."""
    n = len(d)
    vz = np.zeros(n)
    for i in range(1, n):
        vz[i] = 0.0 if contact[i] else vz[i - 1] + G_STEP
    e = np.zeros((n, 7))
    e[:, 0] = np.arange(n)
    e[:, 1] = d
    e[:, 6] = vz
    return e


def test_the_frontier_is_the_median_banked_depth():
    from train_fast import greedy_frontier_progress
    d0 = 30000.0
    # ride to 29,400 on the map (contact), then a fatal dive that falls to 20,000 in the air
    ride = np.r_[np.linspace(d0, 29400.0, 50), np.linspace(29400.0, 20000.0, 50)]
    dive = _episode(ride, np.r_[np.ones(50, bool), np.zeros(50, bool)])
    # the dive does not count: the frontier is the last contact's 29,400 -> 2%
    assert greedy_frontier_progress([dive], _Field(), d0, G_STEP, 1e9) == pytest.approx(0.02)
    # a pit BEHIND the frontier: d rises above the spawn's after a platform at 29,600, on the
    # map all the way - the deepest contact is the platform, the pit reads as no progress
    pit = _episode(np.r_[np.linspace(d0, 29600.0, 40), np.linspace(29600.0, 32800.0, 60)],
                   np.ones(100, bool))
    assert greedy_frontier_progress([pit], _Field(), d0, G_STEP, 1e9) == pytest.approx(
        400.0 / d0)
    # the median of the episodes, not the best one
    far = _episode(np.linspace(d0, 21000.0, 100), np.ones(100, bool))           # 30%
    assert greedy_frontier_progress([pit, pit, far], _Field(), d0, G_STEP, 1e9) \
        == pytest.approx(400.0 / d0)
    assert greedy_frontier_progress([far, far, pit], _Field(), d0, G_STEP, 1e9) \
        == pytest.approx(0.30)
    none = greedy_frontier_progress([], _Field(), d0, G_STEP, 1e9)
    assert none != none                                                  # NaN: no reading


# --- 2. the wiring --------------------------------------------------------------------------------

def test_train_only():
    import record_ckpt
    assert "int_coef_progress" in record_ckpt.TRAIN_ONLY


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--int-coef-progress"' in src
    assert 'meta["config"]["int_coef_progress"] = [float(v) for v in INTP]' in src
    # refused where its reading or its coefficient would not be the run's one
    assert 'raise SystemExit("--int-coef-progress and --int-match both set the novelty "' in src
    assert 'raise SystemExit("--int-coef-progress reads ONE map\'s greedy frontier on the main "' \
        in src
    # the coefficient starts from the checkpoint (or HI) BEFORE --unstuck takes its base
    i_rest = src.index('if ck is not None and ck.get("int_coef_progress") is not None:')
    i_set = src.index('_s.reward_fn.int_coef = float(INTP_ST["coef"])')
    i_base = src.index("INT_BASE = [float(_s.reward_fn.int_coef) for _s in slots]")
    assert i_rest < i_set < i_base
    # every greedy eval of the first map is read after the eval block; --unstuck's base moves too
    assert 'INTP_ST["eval"] = path  # --int-coef-progress: read after the eval block' in src
    assert "_int_progress_after_eval(_ipp)" in src
    assert "INT_BASE[:] = [c] * len(INT_BASE)" in src
    # and the checkpoint carries the coefficient with its reading
    assert 'state["int_coef_progress"] = {"progress": float(INTP_ST["progress"]),' in src


# --- 3. CPU runs ----------------------------------------------------------------------------------

MAP = Path(os.environ.get("INTP_TEST_MAP", str(ROOT / "maps" / "surf_unitfarmer2.bsp")))
DLL = Path(os.environ.get("SURFCORE_DLL", str(
    ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))))
needs_run = pytest.mark.skipif(
    not (MAP.exists() and DLL.exists() and MAP.with_suffix(".goal_32.npz").exists()),
    reason="needs the built core + a map with its prebaked goal field (INTP_TEST_MAP, "
           "SURFCORE_DLL)")
SMOKE = ["--map", str(MAP), "--reward", "race", "--envs", "64", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32", "--lidar-range", "11500",
         "--lidar-near", "2000", "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4", "--gamma", "0.9995",
         "--gae", "0.95", "--clip", "0.2", "--vf", "0.5", "--ent", "0.005", "--n-steps", "8",
         "--epochs", "1", "--minibatches", "2", "--ep-ticks", "600", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "3500", "--train-stride", "1", "--yaw-adaptive",
         "--respawn-frac", "0.9", "--respawn-margin", "10", "--respawn-reservoir", "1000",
         "--int-coef", "2.5", "--int-view", "0", "--int-speed", "8", "--ckpt-every", "1e9",
         "--eval-eps", "3", "--eval-greedy-only", "--unstuck", "--unstuck-max", "0.5",
         "--unstuck-patience", "4096", "--unstuck-period", "4096", "--no-compile", "--fp32",
         "--seed", "0"]


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", SURFCORE_DLL=str(DLL))


def _train(root, run, extra, timeout=1800):
    shutil.rmtree(Path(root) / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(Path(root) / "python" / "train_fast.py"), "--run", run] \
        + SMOKE + extra
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=str(root),
                          timeout=timeout)


def _base_tree(tmp_path):
    """<tmp>/base/python as it was before --int-coef-progress: the parent of the first commit
    that mentions the flag (INTP_BASE_REF overrides)."""
    ref = os.environ.get("INTP_BASE_REF")
    if not ref:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-S",
                            "int-coef-progress", "--", "python/train_fast.py"],
                           capture_output=True, text=True)
        shas = r.stdout.split()
        if r.returncode != 0 or not shas:
            pytest.skip("the flag is not committed yet; set INTP_BASE_REF")
        ref = shas[-1] + "^"
    base = tmp_path / "base"
    base.mkdir(parents=True, exist_ok=True)
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(tar), ref,
                        "python"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git archive {ref} failed: {r.stderr[-400:]}")
    with tarfile.open(tar) as tf:
        tf.extractall(base)
    assert "int-coef-progress" not in (base / "python" / "train_fast.py").read_text(
        encoding="utf-8")
    return base


def _rows(path, drop=("time/fps",)):
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    keep = [i for i, h in enumerate(head) if h not in drop]
    return [[r.split(",")[i] for i in keep] for r in lines[1:]]


@needs_run
def test_the_flag_off_is_bit_identical_to_the_parent_commit(tmp_path):
    base = _base_tree(tmp_path)
    extra = ["--steps", "6144", "--record-every", "2048"]
    r_old = _train(base, "intp_ctl", extra)
    assert r_old.returncode == 0, r_old.stdout[-3000:] + r_old.stderr[-3000:]
    r_new = _train(ROOT, "intp_ctl", extra)
    assert r_new.returncode == 0, r_new.stdout[-3000:] + r_new.stderr[-3000:]
    d_old, d_new = base / "runs" / "intp_ctl", ROOT / "runs" / "intp_ctl"
    c_old = json.loads((d_old / "run.json").read_text(encoding="utf-8"))["config"]
    c_new = json.loads((d_new / "run.json").read_text(encoding="utf-8"))["config"]
    assert "int_coef_progress" not in c_new and c_new == c_old
    t_old, t_new = sorted(d_old.glob("traj_*.jsonl")), sorted(d_new.glob("traj_*.jsonl"))
    assert len(t_old) >= 3 and [p.name for p in t_old] == [p.name for p in t_new]
    for a, b in zip(t_old, t_new):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs"
    assert _rows(d_old / "progress.csv") == _rows(d_new / "progress.csv")
    shutil.rmtree(d_new, ignore_errors=True)


@needs_run
def test_the_flag_on_reads_every_eval_checkpoints_and_restores():
    from train_fast import int_coef_at
    r = _train(ROOT, "intp_on", ["--steps", "6144", "--record-every", "2048",
                                 "--int-coef-progress", "2.5:0.25:0.1"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    reads = [ln for ln in r.stdout.splitlines() if "--int-coef-progress: greedy frontier" in ln]
    assert len(reads) >= 3, r.stdout[-3000:]
    for ln in reads:
        prog = float(ln.split("greedy frontier ")[1].split("%")[0]) / 100.0
        coef = float(ln.split("novelty coefficient ")[1].split()[0])
        assert coef == pytest.approx(int_coef_at(prog, 2.5, 0.25, 0.1), rel=1e-3)
    d = ROOT / "runs" / "intp_on"
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["int_coef_progress"] == [2.5, 0.25, 0.1] and c["int_coef"] == 2.5
    import torch
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    st = ck["int_coef_progress"]
    last = float(reads[-1].split("novelty coefficient ")[1].split()[0])
    assert st["coef"] == pytest.approx(last, rel=1e-3)
    # a resume restores the coefficient before its first eval; without the flag it is not used
    r2 = _train(ROOT, "intp_on2", ["--steps", "8192", "--record-every", "1e12",
                                   "--no-eval-at-start", "--ckpt", str(d / "ckpt_final.pt"),
                                   "--int-coef-progress", "2.5:0.25:0.1"])
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert (f"--int-coef-progress: restored the novelty coefficient {st['coef']:.4g}"
            in r2.stdout), r2.stdout[-3000:]
    for run in ("intp_on", "intp_on2"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
