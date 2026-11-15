"""--int-share-max S (ab/share, 2026-11-15): a ONE-SIDED cap on the count novelty's share of the
return past --int-coef-progress's knee.

Why: the novelty coefficient means different things on different maps. At 0.25 the novelty is
16-23% of celestial's training return and 84-86% of cannonball's; at 0.05 it is 1-3% and
17-20%. The runs that moved past the start sat at 15-23% on both maps (celestial's fork at
0.25; cannonball's wall at 0.05), the stuck ones at <= 3% or >= 84%. A cap on the share keeps
the gate's coefficient where the novelty is moderate or quiet and pulls it down where it
dominates; below the knee (unitfarmer2's dive needs the novelty at 92-99% of the return) it does
nothing.

1. The pieces: the update (one-sided, rate-limited, clamped; nothing paid leaves it), its fixed
   point, the knee of either gate form.
2. record_ckpt and the wiring in train_fast (source).
3. CPU runs (the built core + a map with its prebaked goal field; INTP_TEST_MAP / SURFCORE_DLL):
   the flag OFF is bit-identical to the parent commit with the gate on; the flag needs the
   gate; ON past a knee at 0 it moves the multiplier every iteration, checkpoints it, and a
   resume restores it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


# --- 1. the pieces --------------------------------------------------------------------------------

def test_the_update_is_one_sided_rate_limited_and_clamped():
    from train_fast import share_cap_step
    s = 0.25                                       # novelty <= |rest| / 3
    # over the cap: down by the square root of the factor, at most x0.25 per iteration
    assert share_cap_step(1.0, 12.0, 9.0, s) == pytest.approx((3.0 / 12.0) ** 0.5)
    assert share_cap_step(1.0, 1e6, 3.0, s) == 0.25
    # under the cap: back up, at most x4, never above 1 (the gate's own coefficient)
    assert share_cap_step(0.1, 1.0, 30.0, s) == pytest.approx(0.1 * 10.0 ** 0.5)
    assert share_cap_step(0.1, 1.0, 300.0, s) == pytest.approx(0.4)
    assert share_cap_step(0.5, 1.0, 30.0, s) == 1.0
    assert share_cap_step(1.0, 1.0, 30.0, s) == 1.0
    # clamped below
    assert share_cap_step(2e-4, 1e6, 1.0, s) == 1e-4
    # an iteration with nothing paid on one side leaves it
    assert share_cap_step(0.3, 0.0, 10.0, s) == 0.3
    assert share_cap_step(0.3, 10.0, 0.0, s) == 0.3
    # a negative rest counts by its size (the time penalty can dominate a short episode)
    assert share_cap_step(1.0, 12.0, -9.0, s) == pytest.approx(0.5)


@pytest.mark.parametrize("n0, ext, capped", [
    (37.0, 6.0, True),      # cannonball at 0.25 (ab8n_r: 84-86% of the return)
    (33.6, 3.1, True),      # unitfarmer2's dive at 2.5 (92%) - below the knee in a run
    (8.7, 29.0, False),     # celestial at 0.25 (ab4u_c5: 23%)
    (0.4, 22.0, False),     # petrus at 0.25 (2%)
])
def test_the_fixed_point_is_the_share(n0, ext, capped):
    from train_fast import share_cap_step
    s, m = 0.25, 1.0
    for _ in range(60):                         # the novelty paid scales with the multiplier
        m = share_cap_step(m, n0 * m, ext, s)
    share = n0 * m / (n0 * m + abs(ext))
    if capped:
        assert m < 1.0 and share == pytest.approx(s, abs=1e-3)
    else:
        assert m == 1.0 and share == pytest.approx(n0 / (n0 + ext))


def test_the_knee_of_either_gate_form():
    from train_fast import int_coef_progress_knee, parse_int_coef_progress
    assert int_coef_progress_knee(parse_int_coef_progress("2.5:0.25:0.1")) == 0.1
    assert int_coef_progress_knee(parse_int_coef_progress("0:2.5,0.1:0.25,0.9:0.05")) == 0.1
    assert int_coef_progress_knee(parse_int_coef_progress("0:2.5")) == 0.0


# --- 2. record_ckpt and the wiring ----------------------------------------------------------------

def test_record_ckpt_treats_it_as_train_only():
    import record_ckpt
    assert "int_share_max" in record_ckpt.TRAIN_ONLY
    record_ckpt.audit_cfg({"int_share_max": 0.25, "int_coef_progress": [2.5, 0.25, 0.1]})


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--int-share-max"' in src
    assert 'meta["config"]["int_share_max"] = ISM' in src
    for refusal in ('"--int-share-max caps the novelty past --int-coef-progress\'s knee: it "',
                    '"--int-share-max measures the novelty IN the reward; --int-split "',
                    '"--int-share-max sums the reward per physics tick; "'):
        assert refusal in src
    # the rollout sums the reward paid (novelty included), the iteration measures and moves m
    assert "ism_rsum += float(r.sum())" in src
    i_upd = src.index('ISM_ST["m"] = share_cap_step(ISM_ST["m"], _ip, _ext, ISM)')
    i_base = src.index("INT_BASE[:] = [_cb] * len(INT_BASE)")
    # ... BEFORE --unstuck's block multiplies the base by (1 + T) for the next rollout
    i_unst = src.index("_s.reward_fn.int_coef = _b * (1.0 + _T_next)")
    assert i_upd < i_base < i_unst
    # the gate's reading at every eval carries the multiplier past the knee, and only there
    assert "if _ism_active():" in src and "c = c * ISM_ST[\"m\"]" in src
    assert "return bool(ISM) and p == p and p >= ISM_KNEE" in src
    # checkpointed and restored
    assert 'state["int_share_max"] = {"m": float(ISM_ST["m"])}' in src
    assert 'ISM_ST["m"] = float(ck["int_share_max"]["m"])' in src


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
    """<tmp>/base/python as it was before --int-share-max: the parent of the first commit that
    mentions it (SHARE_BASE_REF overrides)."""
    ref = os.environ.get("SHARE_BASE_REF")
    if not ref:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-S",
                            "int-share-max", "--", "python/train_fast.py"],
                           capture_output=True, text=True)
        shas = r.stdout.split()
        if r.returncode != 0 or not shas:
            pytest.skip("the flag is not committed yet; set SHARE_BASE_REF")
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
    assert "int-share-max" not in (base / "python" / "train_fast.py").read_text(encoding="utf-8")
    return base


def _rows(path, drop=("time/fps",)):
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    keep = [i for i, h in enumerate(head) if h not in drop]
    return [[r.split(",")[i] for i in keep] for r in lines[1:]]


@needs_run
def test_the_flag_off_is_bit_identical_to_the_parent_commit(tmp_path):
    base = _base_tree(tmp_path)
    extra = ["--steps", "6144", "--record-every", "2048", "--int-coef-progress", "2.5:0.25:0.1"]
    r_old = _train(base, "share_ctl", extra)
    assert r_old.returncode == 0, r_old.stdout[-3000:] + r_old.stderr[-3000:]
    r_new = _train(ROOT, "share_ctl", extra)
    assert r_new.returncode == 0, r_new.stdout[-3000:] + r_new.stderr[-3000:]
    d_old, d_new = base / "runs" / "share_ctl", ROOT / "runs" / "share_ctl"
    c_old = json.loads((d_old / "run.json").read_text(encoding="utf-8"))["config"]
    c_new = json.loads((d_new / "run.json").read_text(encoding="utf-8"))["config"]
    assert "int_share_max" not in c_new and c_new == c_old
    t_old, t_new = sorted(d_old.glob("traj_*.jsonl")), sorted(d_new.glob("traj_*.jsonl"))
    assert len(t_old) >= 3 and [p.name for p in t_old] == [p.name for p in t_new]
    for a, b in zip(t_old, t_new):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs"
    assert _rows(d_old / "progress.csv") == _rows(d_new / "progress.csv")
    shutil.rmtree(d_new, ignore_errors=True)


@needs_run
def test_the_flag_needs_the_gate():
    r = _train(ROOT, "share_nogate", ["--steps", "2048", "--int-share-max", "0.25"], timeout=600)
    assert r.returncode != 0
    assert "--int-share-max caps the novelty past --int-coef-progress's knee" in (
        r.stdout + r.stderr)
    shutil.rmtree(ROOT / "runs" / "share_nogate", ignore_errors=True)


@needs_run
def test_the_cap_moves_every_iteration_checkpoints_and_restores():
    from train_fast import share_cap_step
    # a knee at 0: the cap acts at every reading at or past the spawn's d (a smoke run's frontier
    # stays near 0% and dips just below it at times); 64-tick episodes end every other iteration,
    # so the iterations have novelty to measure, and at 2.5 the novelty dominates their return
    short = ["--ep-ticks", "64", "--stall-secs", "60"]
    r = _train(ROOT, "share_on", ["--steps", "20480", "--record-every", "2048",
                                  "--int-coef-progress", "0:2.5", "--int-share-max", "0.25"]
               + short)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "--int-share-max 0.25: past frontier progress 0% (the gate's knee)" in r.stdout
    notes = re.findall(r"share-cap m ([\d.e+-]+) \(novelty ([\d,.-]+) vs rest ([\d,.-]+): "
                       r"(\d+)% this iter\)", r.stdout)
    assert len(notes) >= 3, r.stdout[-3000:]
    ms = [float(n[0]) for n in notes]
    assert all(0.0 < m <= 1.0 for m in ms) and min(ms) < 1.0      # the cap did pull it down
    # each step is share_cap_step's on the iteration's own numbers (the note rounds them; an
    # iteration below the knee leaves m where it was)
    prev = 1.0
    for (m, ip, ext, _sh) in notes:
        ip, ext = float(ip.replace(",", "")), abs(float(ext.replace(",", "")))
        # the note prints both sums to the unit: the step is decreasing in the novelty and
        # increasing in the rest, so the rounding box bounds it
        lo = share_cap_step(prev, ip + 0.5, max(ext - 0.5, 1e-6), 0.25)
        hi = share_cap_step(prev, max(ip - 0.5, 1e-6), ext + 0.5, 0.25)
        within = lo * 0.995 - 1e-3 <= float(m) <= hi * 1.005 + 1e-3
        # a printed 0 may be no novelty at all (no episode ended): m then stays where it was
        assert within or (ip == 0.0 and float(m) == pytest.approx(prev, abs=1e-3)), \
            (prev, ip, ext, m)
        prev = float(m)
    d = ROOT / "runs" / "share_on"
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["int_share_max"] == 0.25 and c["int_coef_progress"] == [[0.0, 2.5]]
    import torch
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    m_ck = float(ck["int_share_max"]["m"])
    assert m_ck == pytest.approx(ms[-1], rel=1e-2, abs=1e-3)
    r2 = _train(ROOT, "share_on2", ["--steps", "22528", "--record-every", "1e12",
                                    "--no-eval-at-start", "--ckpt", str(d / "ckpt_final.pt"),
                                    "--int-coef-progress", "0:2.5", "--int-share-max", "0.25"]
                + short)
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert f"--int-share-max: restored the cap's multiplier {m_ck:.4g}" in r2.stdout, \
        r2.stdout[-3000:]
    for run in ("share_on", "share_on2"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
