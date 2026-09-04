"""--priv-critic: the asymmetric actor-critic (Pinto et al. 2017).

The critic gets a privileged state block the actor never sees
(surfgym/privfeat.py), concatenated to the value tower's output right before
``value_head``. What is pinned here:

1. The BLOCK: ten named columns, the documented normalisation, and the fact
   that ``velocity_from_obs`` inverts src/env.c's ``write_obs`` exactly -
   which is what lets the truncation bootstrap rebuild s_T's block after the
   core has autoreset off it.
2. ACTOR INVARIANCE: given the same actor weights, the logits are the same
   function of the observation with the flag and without it, and are
   unchanged by what the privileged block contains. That is the whole claim
   of the method - the deployed policy is identical in form.
3. OFF is byte-identical: priv_dim 0 builds no module, draws no RNG and adds
   no state_dict key, and ``value_head`` is the same ``Linear(hidden, 1)``.
   The end-to-end half of this (a scratch rollout+update against the
   pre-flag trainer, comparing progress.csv and every checkpoint tensor)
   runs as ``test_no_flag_is_bit_identical_to_the_pre_flag_trainer`` and
   skips when no pre-flag git ref is reachable.
4. Warm start: ``widen_for_priv`` zero-pads a plain checkpoint's value_head
   and appends a fresh priv MLP, so the ACTOR is bit-identical and V(s) is
   the checkpoint's own function to ~1 ulp of fp32 (the wider GEMM's
   summation order). The trainer prints that as a notice.
5. The reverse is refused: a privileged checkpoint resumed WITHOUT the flag
   exits with a message, rather than silently dropping the critic's input.
6. A tiny CPU scratch run with the flag trains, evaluates, logs
   ``train/explained_var`` and writes the block into run.json and the ckpt.

    python -m pytest tests/python/test_priv_critic.py -q
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.privfeat import (PRIV_DIM, PRIV_FEATURES, VEL_SCALE,  # noqa: E402
                              PrivFeat, velocity_from_obs)
from train_fast import N_SCALAR, Policy, widen_for_priv         # noqa: E402

_DLL_NAME = "surfcore.dll" if os.name == "nt" else "libsurfcore.so"
DLL = Path(os.environ.get("SURFCORE_DLL") or (ROOT / "build" / _DLL_NAME))


def _maps_dir():
    """Where to take cannonball AND its prebaked goal field from.

    Both, from the SAME directory: the caches key on the bsp's size +
    mtime_ns, so a copy of the map with a different mtime is a cache MISS
    and the trainer silently starts a 30-minute bake (CLAUDE.md, the
    worktree rule). ``RL_SURF_MAPS`` is how a worktree points at the main
    checkout's maps; without a cache these tests SKIP rather than bake.
    """
    env = os.environ.get("RL_SURF_MAPS")
    for d in ([Path(env)] if env else []) + [ROOT / "maps"]:
        if ((d / "surf_src_cannonball.bsp").exists()
                and (d / "surf_src_cannonball.goal_32.npz").exists()):
            return d
    return None


MAPS = _maps_dir()
CANNONBALL = (MAPS or (ROOT / "maps")) / "surf_src_cannonball.bsp"

needs_core = pytest.mark.skipif(not (CANNONBALL.exists() and DLL.exists()),
                                reason="needs the built core + cannonball")

W, H = 8, 4
OBS = N_SCALAR + W * H


def _pol(**kw):
    return Policy(OBS, W, H, emb=32, hidden=24, **kw)


# ==========================================================================
# 1. the block
# ==========================================================================
def test_block_is_ten_named_columns():
    assert PRIV_DIM == 10 == len(PRIV_FEATURES)
    assert PRIV_FEATURES == ("pos_x", "pos_y", "pos_z",
                             "vel_x", "vel_y", "vel_z",
                             "d_frac", "arc_frac", "t_frac", "latch")
    assert VEL_SCALE == 4000.0


def test_normalisation_is_the_documented_one():
    pf = PrivFeat(map_center=(100.0, -200.0, 50.0), map_scale=1000.0,
                  d0=200.0, ep_ticks=3000, arc_len=800.0)
    out = np.zeros((2, PRIV_DIM), np.float32)
    pos = np.array([[1100.0, -200.0, 50.0], [100.0, 800.0, -950.0]])
    vel = np.array([[4000.0, 0.0, -2000.0], [0.0, -4000.0, 0.0]])
    pf.fill(out, pos, vel, d=np.array([200.0, 50.0]),
            tick=np.array([0, 1500]), arc=np.array([0.0, 400.0]),
            latch=np.array([False, True]))
    # position: (p - centre) / ONE scale, so the three columns share it
    assert np.allclose(out[0, 0:3], [1.0, 0.0, 0.0])
    assert np.allclose(out[1, 0:3], [0.0, 1.0, -1.0])
    assert np.allclose(out[:, 3:6], vel / 4000.0)
    assert np.allclose(out[:, 6], [1.0, 0.25])        # d / d0
    assert np.allclose(out[:, 7], [0.0, 0.5])         # arc / length
    assert np.allclose(out[:, 8], [0.0, 0.5])         # tick / ep_ticks
    assert np.allclose(out[:, 9], [0.0, 1.0])         # latch
    assert out.dtype == np.float32


def test_absent_arc_and_latch_are_a_constant_zero_column():
    # a run without --race-arc / --race-latch must feed 0, not garbage: the
    # column is then constant and the critic can only ignore it
    pf = PrivFeat((0.0, 0.0, 0.0), 1000.0, 100.0, 1000)     # arc_len 0
    out = np.full((1, PRIV_DIM), 7.0, np.float32)
    pf.fill(out, np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1),
            np.zeros(1), arc=np.array([500.0]), latch=None)
    assert out[0, 7] == 0.0 and out[0, 9] == 0.0


def test_nothing_is_clipped():
    """A long fall leaves the [-1, 1] box; the block must SAY so."""
    pf = PrivFeat((0.0, 0.0, 0.0), 1000.0, 100.0, 1000)
    out = np.zeros((1, PRIV_DIM), np.float32)
    pf.fill(out, np.array([[9000.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, -9000.0]]), np.array([500.0]),
            np.array([4000]))
    assert out[0, 0] == 9.0 and out[0, 5] == -2.25
    assert out[0, 6] == 5.0 and out[0, 8] == 4.0


def test_fill_refuses_a_block_of_the_wrong_width():
    pf = PrivFeat((0.0, 0.0, 0.0), 1000.0, 100.0, 1000)
    with pytest.raises(ValueError):
        pf.fill(np.zeros((1, PRIV_DIM - 1), np.float32), np.zeros((1, 3)),
                np.zeros((1, 3)), np.zeros(1), np.zeros(1))


def test_velocity_from_obs_inverts_the_ego_rotation():
    """Pure algebra against write_obs (src/env.c) - no core needed."""
    rng = np.random.default_rng(3)
    v = rng.normal(0.0, 900.0, (64, 3))
    yaw = rng.uniform(-180.0, 180.0, 64)
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    o = np.zeros((64, N_SCALAR), np.float32)
    o[:, 0] = (v[:, 0] * cy + v[:, 1] * sy) / 1000.0
    o[:, 1] = (-v[:, 0] * sy + v[:, 1] * cy) / 1000.0
    o[:, 2] = v[:, 2] / 1000.0
    o[:, 7], o[:, 8] = sy, cy
    assert np.allclose(velocity_from_obs(o), v, atol=1e-2)


@needs_core
def test_velocity_from_obs_matches_the_live_core():
    """The same inversion on the REAL obs the engine writes."""
    from surfgym import SurfCore, default_config
    core = SurfCore(str(CANNONBALL), default_config(num_envs=16))
    obs = core.reset(11)
    rng = np.random.default_rng(1)
    for _ in range(40):
        act = np.stack([rng.integers(0, n, 16)
                        for n in (15, 7, 3, 3, 2, 2)], 1).astype(np.int32)
        obs = core.step(act)[0]
    got = velocity_from_obs(obs)
    want = np.asarray(core.states_view["velocity"], np.float64)
    assert np.abs(got - want).max() < 0.5, np.abs(got - want).max()


# ==========================================================================
# 2. actor invariance
# ==========================================================================
def _copy_shared(dst: Policy, src: Policy) -> None:
    """Give `dst` every tensor `src` has at the same shape (i.e. everything
    but value_head.weight and the priv MLP)."""
    sd, ss = dst.state_dict(), src.state_dict()
    for k, v in ss.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v.clone()
    dst.load_state_dict(sd)


def test_actor_is_the_same_function_with_and_without_the_flag():
    torch.manual_seed(4)
    plain = _pol().eval()
    priv = _pol(priv_dim=PRIV_DIM).eval()
    _copy_shared(priv, plain)
    x = torch.randn(6, OBS)
    la, va = plain(x)
    lb, vb = priv(x, priv=torch.randn(6, PRIV_DIM))
    assert torch.equal(la, lb)                    # BIT-identical logits
    assert not torch.allclose(va, vb)             # the critic did change


def test_logits_do_not_move_when_the_privileged_block_does():
    torch.manual_seed(5)
    p = _pol(priv_dim=PRIV_DIM).eval()
    x = torch.randn(4, OBS)
    l1, v1 = p(x, priv=torch.zeros(4, PRIV_DIM))
    l2, v2 = p(x, priv=torch.full((4, PRIV_DIM), 5.0))
    assert torch.equal(l1, l2)
    assert not torch.allclose(v1, v2)


def test_no_actor_tensor_grows():
    """The privileged block must be structurally unable to reach pi."""
    torch.manual_seed(6)
    a, b = _pol().state_dict(), _pol(priv_dim=PRIV_DIM).state_dict()
    grew = {k for k in a if k in b and a[k].shape != b[k].shape}
    assert grew == {"value_head.weight"}, grew
    assert set(b) - set(a) == {f"priv_mlp.{i}.{w}"
                              for i in (0, 1, 3, 4) for w in ("weight", "bias")}


def test_value_is_nan_without_a_privileged_row():
    """Actor-only callers (the recorder) work; a caller that USES V cannot
    silently get value_head over a zero block."""
    torch.manual_seed(7)
    p = _pol(priv_dim=PRIV_DIM).eval()
    x = torch.randn(3, OBS)
    logits, value = p(x)
    assert torch.isnan(value).all()
    assert torch.equal(logits, p(x, priv=torch.randn(3, PRIV_DIM))[0])


# ==========================================================================
# 3. off is byte-identical
# ==========================================================================
def test_priv_dim_zero_is_the_pre_flag_policy():
    torch.manual_seed(11)
    a = _pol()
    torch.manual_seed(11)
    b = _pol(priv_dim=0, priv_hidden=128)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    assert b.priv_mlp is None and b.priv_hidden == 0
    assert tuple(sb["value_head.weight"].shape) == (1, 24)
    x = torch.randn(5, OBS)
    a.eval(), b.eval()
    la, va = a(x)
    lb, vb = b(x)
    assert torch.equal(la, lb) and torch.equal(va, vb)


def test_flag_is_recorded_and_mirrored_by_the_recorder():
    train = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    rec = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"--priv-critic"' in train and '"priv_critic": int(' in train
    # record_ckpt's audit refuses any config key it never mentions
    assert "priv_critic" in rec and "priv_hidden" in rec
    assert "priv_features" in rec
    assert "priv_dim=(PRIV_DIM if cfg.get" in rec       # MIRRORED ...
    # ... and the load stays STRICT, which is the only thing that has
    # ever caught an unmirrored flag (the --obs-reward 523-vs-522 bug)
    assert 'policy.load_state_dict(ck["policy"])' in rec


# ==========================================================================
# 4. --rnn / --fp32-heads paths
# ==========================================================================
def test_rnn_concat_happens_after_the_gru():
    torch.manual_seed(12)
    p = _pol(rnn="gru", rnn_size=8, priv_dim=PRIV_DIM).eval()
    x = torch.randn(3, OBS)
    h = torch.randn(3, 8)
    pv = torch.randn(3, PRIV_DIM)
    l1, v1, h1 = p(x, h, priv=pv)
    l2, v2, h2 = p(x, h, priv=torch.randn(3, PRIV_DIM))
    # the GRU is upstream of the concat: the state it emits, and the logits
    # that read it, cannot depend on the privileged block
    assert torch.equal(h1, h2) and torch.equal(l1, l2)
    assert not torch.allclose(v1, v2)


def test_fp32_heads_path_runs_with_the_block():
    torch.manual_seed(13)
    a = _pol(priv_dim=PRIV_DIM).eval()
    b = _pol(priv_dim=PRIV_DIM, fp32_heads=True).eval()
    b.load_state_dict(a.state_dict())
    x, pv = torch.randn(4, OBS), torch.randn(4, PRIV_DIM)
    la, va = a(x, priv=pv)
    lb, vb = b(x, priv=pv)
    # outside autocast the two are the same arithmetic
    assert torch.allclose(la, lb) and torch.allclose(va, vb)


# ==========================================================================
# 5. the warm start
# ==========================================================================
def test_widen_for_priv_keeps_the_actor_and_the_value():
    torch.manual_seed(14)
    plain = _pol().eval()
    n_par = len(list(plain.parameters()))
    ck = {"policy": {k: v.clone() for k, v in plain.state_dict().items()},
          "optimizer": {"state": {}, "param_groups":
                        [{"params": list(range(n_par))}]}}
    torch.manual_seed(99)                       # a DIFFERENT draw
    wide = _pol(priv_dim=PRIV_DIM)
    n = widen_for_priv(ck, wide)
    assert n == 1 + 8                           # value_head.weight + the MLP
    wide.load_state_dict(ck["policy"])          # strict
    wide.eval()
    x, pv = torch.randn(5, OBS), torch.randn(5, PRIV_DIM)
    la, va = plain(x)
    lb, vb = wide(x, priv=pv)
    assert torch.equal(la, lb)                  # the actor: BIT-identical
    assert torch.allclose(va, vb, atol=1e-6, rtol=0)    # V: ~1 ulp of fp32
    assert (ck["policy"]["value_head.weight"][:, 24:] == 0).all()
    # every fresh parameter has an optimizer slot and no moments
    assert (sorted(int(i) for i in ck["optimizer"]["param_groups"][0]["params"])
            == list(range(len(list(wide.parameters())))))
    assert widen_for_priv(ck, wide) == 0        # idempotent


def test_widen_pads_the_adam_moments_too():
    torch.manual_seed(15)
    plain = _pol()
    params = list(plain.parameters())
    vh = [i for i, p in enumerate(params)
          if p is plain.value_head.weight][0]
    ck = {"policy": {k: v.clone() for k, v in plain.state_dict().items()},
          "optimizer": {"state": {vh: {"exp_avg": torch.randn(1, 24),
                                       "exp_avg_sq": torch.rand(1, 24)}},
                        "param_groups": [{"params": list(range(len(params)))}]}}
    wide = _pol(priv_dim=PRIV_DIM)
    widen_for_priv(ck, wide)
    for k in ("exp_avg", "exp_avg_sq"):
        t = ck["optimizer"]["state"][vh][k]
        assert tuple(t.shape) == (1, 24 + 128)
        assert (t[:, 24:] == 0).all()


# ==========================================================================
# 6. end to end on the CPU (tiny nets, prebaked field)
# ==========================================================================
SMOKE = ["--reward", "race", "--envs", "64", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32",
         "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64",
         "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
         "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2",
         "--vf", "0.5", "--ent", "0.005",
         "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
         "--ep-ticks", "3000", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "4000",
         "--train-stride", "1", "--yaw-adaptive",
         "--respawn-frac", "0.9", "--respawn-margin", "10",
         "--respawn-reservoir", "1000", "--int-coef", "0.25",
         "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9",
         "--eval-eps", "1", "--eval-greedy-only", "--no-eval-at-start"]

needs_field = pytest.mark.skipif(
    MAPS is None,
    reason="needs the prebaked cannonball goal field (set RL_SURF_MAPS to "
           "the main checkout's maps/ when running from a worktree)")


def _env():
    # CUDA_VISIBLE_DEVICES="-1", never "": Windows deletes an empty var
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1",
                PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="8",
                NUMBA_NUM_THREADS="8", SURFCORE_DLL=str(DLL))


def _train(root: Path, run: str, extra, timeout=1800):
    """One tiny trainer process. ``out = ROOT/runs/<run>``, and ROOT is
    derived from train_fast.py's own path - which is what lets the pre-flag
    comparison below run an extracted tree without touching this one."""
    shutil.rmtree(root / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(root / "python" / "train_fast.py"),
           "--map", str(CANNONBALL), "--run", run] + SMOKE + extra
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       cwd=str(root), timeout=timeout)
    return r, root / "runs" / run


@needs_core
@needs_field
def test_scratch_run_with_the_flag_trains_evaluates_and_logs_ev():
    r, rd = _train(ROOT, "priv_test_on", ["--priv-critic", "1",
                                          "--steps", "8192",
                                          "--record-every", "8192"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "priv-critic: 10 columns" in out
    assert "'d_frac', 'arc_frac', 't_frac', 'latch'" in out
    assert "no arc line (column 7 is 0)" in out
    assert "greedy:" in out          # the eval ran with the priv feed wired
    cfg = json.loads((rd / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["priv_critic"] == 1 and cfg["priv_hidden"] == 128
    assert cfg["priv_features"] == list(PRIV_FEATURES)
    # explained variance is the direct read-out of a better critic
    rows = list(csv.DictReader(
        (rd / "progress.csv").open(newline="", encoding="utf-8")))
    assert rows and "train/explained_var" in rows[0]
    ev = [float(x["train/explained_var"]) for x in rows]
    assert all(v == v for v in ev), ev            # finite, never NaN
    ck = torch.load(rd / "ckpt_latest.pt", map_location="cpu",
                    weights_only=False)
    assert tuple(ck["policy"]["value_head.weight"].shape) == (1, 64 + 128)
    assert any(k.startswith("priv_mlp.") for k in ck["policy"])
    assert ck["config"]["priv_critic"] == 1
    shutil.rmtree(rd, ignore_errors=True)


@needs_core
@needs_field
def test_resume_notice_and_the_refusal():
    r, plain = _train(ROOT, "priv_test_ctl",
                      ["--steps", "4096", "--record-every", "1e12"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    ck = plain / "ckpt_latest.pt"
    assert ck.exists()
    assert not any(k.startswith("priv_mlp.") for k in
                   torch.load(ck, map_location="cpu",
                              weights_only=False)["policy"])

    # a plain checkpoint + the flag: widened, announced, and it trains
    r, warm = _train(ROOT, "priv_test_warm",
                     ["--priv-critic", "1", "--ckpt", str(ck),
                      "--steps", "6144", "--record-every", "1e12"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--priv-critic: this checkpoint has no privileged critic" in r.stdout
    assert "TRAILING" in r.stdout and "grows from zero" in r.stdout
    assert "resumed" in r.stdout

    # the reverse is refused rather than silently dropping the critic's input
    r, back = _train(ROOT, "priv_test_back",
                     ["--ckpt", str(warm / "ckpt_latest.pt"),
                      "--steps", "8192", "--record-every", "1e12"])
    assert r.returncode != 0
    msg = r.stdout + r.stderr
    assert "trained with --priv-critic" in msg
    assert "Pass --priv-critic 1" in msg
    for d in (plain, warm, back):
        shutil.rmtree(d, ignore_errors=True)


@needs_core
@needs_field
def test_recorder_builds_the_same_shape_from_the_config(tmp_path):
    r, rd = _train(ROOT, "priv_test_rec",
                   ["--priv-critic", "1", "--steps", "4096",
                    "--record-every", "1e12"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    traj = tmp_path / "rec.jsonl"
    rr = subprocess.run(
        [sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
         str(rd / "ckpt_latest.pt"), "--map", str(CANNONBALL),
         "--episodes", "1", "--out", str(traj)],
        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
        timeout=1200)
    # a STRICT load of a privileged checkpoint: the recorder has to have
    # rebuilt value_head at the right width and created the priv MLP
    assert rr.returncode == 0, rr.stdout[-4000:] + rr.stderr[-4000:]
    assert traj.exists() and traj.stat().st_size > 0
    shutil.rmtree(rd, ignore_errors=True)


# --------------------------------------------------------------------------
# the end-to-end half of "off is byte-identical": the same scratch
# rollout+update run by the PRE-FLAG trainer, compared row by row and tensor
# by tensor. Skipped when no reachable git ref predates the flag (after this
# lands on the integration branch, every candidate has it - and then there is
# nothing left to compare against, which is the honest outcome).
# --------------------------------------------------------------------------
# progress.csv columns added by OTHER features merged onto the integration
# branch after the reference commit. The air-key diagnostics are written by
# every arm (no flag gates them), so a pre-flag tree simply has four fewer
# columns; nothing about their VALUES is claimed here.
INERT_COLS_SINCE = {"act/fwd_air", "act/strafe_flip", "act/jump_air",
                    "act/duck_air"}



def _preflag_candidates():
    """Newest first-parent ancestors of HEAD, newest first: the right reference
    for "no flag == pre-flag code" is the closest commit that lacks the flag,
    never an old integration branch such as main (whose trainer does not even
    accept today's flags)."""
    try:
        r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=60)
        refs = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        refs = []
    return tuple(refs[1:]) or ("HEAD^",)      # [0] is HEAD itself

def _preflag_tree(dest: Path):
    for ref in _preflag_candidates():
        try:
            src = subprocess.run(
                ["git", "show", f"{ref}:python/train_fast.py"],
                capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        except (OSError, subprocess.SubprocessError):        # no git
            return None
        if src.returncode != 0 or "priv_critic" in src.stdout:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        tar = dest / "py.tar"
        a = subprocess.run(["git", "archive", ref, "python", "-o", str(tar)],
                           capture_output=True, text=True, cwd=str(ROOT),
                           timeout=300)
        if a.returncode != 0:
            continue
        subprocess.run(["tar", "-xf", str(tar), "-C", str(dest)],
                       capture_output=True, text=True, timeout=300)
        return dest if (dest / "python" / "train_fast.py").exists() else None
    return None


@needs_core
@needs_field
def test_no_flag_is_bit_identical_to_the_pre_flag_trainer(tmp_path):
    old = _preflag_tree(tmp_path / "pre")
    if old is None:
        pytest.skip("no reachable git ref predates --priv-critic")
    ra, a = _train(old, "priv_bit_old",
                   ["--steps", "8192", "--record-every", "1e12"])
    assert ra.returncode == 0, ra.stdout[-4000:] + ra.stderr[-4000:]
    rb, b = _train(ROOT, "priv_bit_new",
                   ["--steps", "8192", "--record-every", "1e12"])
    assert rb.returncode == 0, rb.stdout[-4000:] + rb.stderr[-4000:]

    def rows(p):
        return list(csv.DictReader(
            (p / "progress.csv").open(newline="", encoding="utf-8")))

    ra_, rb_ = rows(a), rows(b)
    assert len(ra_) == len(rb_) > 0
    # The pre-flag tree also predates the OTHER opt-in features merged onto
    # the integration branch since. The air-key masks append four
    # diagnostic columns unconditionally (they need no flag), so those are
    # the one permitted extra; every column the two files share is still
    # compared value for value below.
    added_cols = set(rb_[0]) - set(ra_[0])
    assert not set(ra_[0]) - set(rb_[0]), set(ra_[0]) - set(rb_[0])
    assert added_cols <= INERT_COLS_SINCE, added_cols
    for i, (x, y) in enumerate(zip(ra_, rb_)):
        for c in x:
            if c.startswith("time/") or "fps" in c or "wall" in c:
                continue        # wall-clock is not a property of the code
            assert x[c] == y[c], (i, c, x[c], y[c])
    ca = torch.load(a / "ckpt_latest.pt", map_location="cpu",
                    weights_only=False)
    cb = torch.load(b / "ckpt_latest.pt", map_location="cpu",
                    weights_only=False)
    assert set(ca["policy"]) == set(cb["policy"])
    for k in ca["policy"]:
        assert torch.equal(ca["policy"][k], cb["policy"][k]), k
    sa, sb = ca["optimizer"]["state"], cb["optimizer"]["state"]
    assert set(sa) == set(sb)
    for i in sa:
        for k, v in sa[i].items():
            if torch.is_tensor(v):
                assert torch.equal(v, sb[i][k]), (i, k)
    assert ca["global_step"] == cb["global_step"]
    # the config dump gains exactly the three new keys, all at their off value
    assert set(cb["config"]) - set(ca["config"]) == {
        "priv_critic", "priv_features", "priv_hidden"}
    assert not set(ca["config"]) - set(cb["config"])
    assert cb["config"]["priv_critic"] == 0
    assert cb["config"]["priv_features"] is None
    assert cb["config"]["priv_hidden"] is None
    for k in ca["config"]:
        assert ca["config"][k] == cb["config"][k], k
    shutil.rmtree(b, ignore_errors=True)
