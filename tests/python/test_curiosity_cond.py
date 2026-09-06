"""--curiosity-cond (docs/curiosity_cond.md): a T-conditioned family.

(a) the draw: T = 0 with probability p0, else log-uniform on [tmin, tmax];
    a pure function of (seed, env, episode) whatever the batch; the
    encoding t = log1p(T)/log1p(T_max) and its inverse.
(b) the advantage buckets: bucket 0 is T = 0, the rest equal-probability
    slices of the continuum; numpy searchsorted == torch.bucketize.
(c) RaceReward: the flag OFF is the control bit for bit; with per-env T
    the shaping term is x (1 - T/T_max), the count bonus x T, the success
    bonus and the fail penalty unchanged - against a hand computation.
(d) T is redrawn at every episode end for the ended envs only; cc_boot
    keeps the T of the episode that ended; cc_obs is the encoded column.
(e) the eval wrappers append the column LAST on the scalar side, with the
    encoded value; a policy built one column wider reads it.
(f) per-bucket normalisation equals per-bucket (a - mean)/(std + eps) with
    torch.std's estimator; a one-row bucket normalises to 0.
(g) the per-env keys temperature: cc_temp_block pins the two view bins at
    1 on the bins and is (n, 1) on the continuous view; the tempered log-
    prob with a per-row block equals the per-row scalar computation; the
    TemperedTorchPolicy at a per-env keys_temp is sample_view at that
    block under the same seed.
(h) the trainer's rollout draws through temp_t and the update scores each
    row at its own recorded temperature (source-level pin).
(i) trainer smokes (CPU, the toy scratch set): the flag OFF is bit-identical
    to the trainer of the last commit without the flag (bins and the
    absolute view); the flag ON trains with finite losses and sane kl,
    resamples T at episode ends (short episodes, so the truncation
    bootstrap's terminal row carries the column), logs the T distribution
    and per-bucket episode stats, evals the T = 0 member, writes the config
    and the checkpoint; record_ckpt --cc-T records another member (and the
    stochastic one at the trained keys temperature) and refuses a plain
    checkpoint; a flagless resume restores the flag; a plain checkpoint is
    widened onto the family; beam_tas --cc-T mix plans with the family;
    BCDataset synthesises the T = 0 column; the refusals.
"""
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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                                   # noqa: E402

from surfgym.core import STATE_DTYPE                           # noqa: E402
from surfgym.goalfield import EuclidField                      # noqa: E402
from surfgym.rewards import (CC_TMAX, RaceReward, cc_bucket_edges,   # noqa: E402
                             cc_bucket_of, cc_decode, cc_draw, cc_encode,
                             cc_uniform)
from train_fast import (N_VIEW, NACT, NVEC, GreedyTorchPolicy,  # noqa: E402
                        HeadPacker, Policy, SampledTorchPolicy,
                        TemperedTorchPolicy, cc_bucket_normalize,
                        cc_keys_temp, cc_temp_block, logprob_entropy_padded,
                        make_cc_feed, sample_view, split_view, view_from_z_t)
from test_view_absolute import RATE, _tiny                     # noqa: E402
from test_view_continuous import CANNONBALL, SMOKE_FLAGS, needs_run  # noqa: E402

CPU = torch.device("cpu")
TRAIN = ROOT / "python" / "train_fast.py"
RECORD = ROOT / "tools" / "record_ckpt.py"
BEAM = ROOT / "tools" / "beam_tas.py"
ROUTE = CANNONBALL.parent / "surf_src_cannonball.route.npz"
ABS = ["--view-continuous", "--view-absolute", "velocity"]


# ==========================================================================
# (a) the draw and the encoding
# ==========================================================================
def test_draw_is_the_mixture_and_a_pure_function_of_seed_env_episode():
    T = cc_draw(7, np.arange(100000), 0, 0.5, 0.05, 2.0)
    assert abs(float((T == 0.0).mean()) - 0.5) < 0.01
    pos = T[T > 0.0]
    assert pos.min() >= 0.05 and pos.max() <= 2.0
    # log-uniform: the mean of log T is the midpoint of the log range
    assert abs(float(np.log(pos).mean()) - 0.5 * (np.log(0.05) + np.log(2.0))) < 0.02
    # the batch does not matter: env 3's episode 1 is the same draw alone
    assert np.array_equal(cc_draw(7, [3, 4], [1, 1], 0.5, 0.05, 2.0),
                          cc_draw(7, np.arange(10), 1, 0.5, 0.05, 2.0)[3:5])
    # and the episode index / the seed do
    a = cc_uniform(7, np.arange(1000), 0, 1)
    b = cc_uniform(7, np.arange(1000), 1, 1)
    c = cc_uniform(8, np.arange(1000), 0, 1)
    assert not np.any(a == b) and not np.any(a == c)
    assert a.min() >= 0.0 and a.max() < 1.0
    # p0 0 never draws a zero, p0 1 always does
    assert np.all(cc_draw(1, np.arange(500), 0, 0.0, 0.05, 2.0) > 0.0)
    assert np.all(cc_draw(1, np.arange(500), 0, 1.0, 0.05, 2.0) == 0.0)
    # the encoding: 0 at 0, 1 at T_max, monotone, invertible
    assert float(cc_encode(0.0, 2.0)) == 0.0 and float(cc_encode(2.0, 2.0)) == 1.0
    ts = cc_encode(np.linspace(0.0, 2.0, 50), 2.0)
    assert np.all(np.diff(ts) > 0.0)
    assert np.allclose(cc_decode(ts, 2.0), np.linspace(0.0, 2.0, 50))
    assert CC_TMAX == 2.0


# ==========================================================================
# (b) the buckets
# ==========================================================================
def test_bucket_edges_are_equal_probability_and_torch_bucketize_agrees():
    e = cc_bucket_edges(0.5, 0.05, 2.0, 4)
    assert e.shape == (3,) and np.all(np.diff(e) > 0.0)
    assert 0.0 < e[0] < float(cc_encode(0.05, 2.0))      # T = 0 alone in bucket 0
    T = cc_draw(3, np.arange(200000), 0, 0.5, 0.05, 2.0)
    bk = cc_bucket_of(cc_encode(T, 2.0), e)
    frac = np.bincount(bk, minlength=4) / len(T)
    assert abs(frac[0] - 0.5) < 0.01
    assert np.all(np.abs(frac[1:] - 0.5 / 3.0) < 0.01)
    assert np.all(bk[T == 0.0] == 0) and np.all(bk[T > 0.0] >= 1)
    tb = torch.bucketize(torch.as_tensor(cc_encode(T, 2.0)),
                         torch.as_tensor(e), right=True)
    assert np.array_equal(tb.numpy(), bk)
    # p0 = 0: every bucket is a slice of the continuum
    e0 = cc_bucket_edges(0.0, 0.05, 2.0, 4)
    T0 = cc_draw(3, np.arange(200000), 0, 0.0, 0.05, 2.0)
    f0 = np.bincount(cc_bucket_of(cc_encode(T0, 2.0), e0), minlength=4) / len(T0)
    assert np.all(np.abs(f0 - 0.25) < 0.01)
    assert cc_bucket_edges(0.5, 0.05, 2.0, 1).shape == (0,)
    assert cc_bucket_edges(0.5, 0.05, 2.0, 2).shape == (1,)
    with pytest.raises(ValueError):
        cc_bucket_edges(0.5, 0.05, 2.0, 0)


# ==========================================================================
# (c) the reward's per-env mix
# ==========================================================================
class FakeCore:
    """Only what RaceReward reads: the states view, goal_hits, map_bounds."""

    def __init__(self, n=1, bounds=((-3e4, -3e4, -3e4), (3e4, 3e4, 3e4))):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self._bounds = (np.asarray(bounds[0], np.float32),
                        np.asarray(bounds[1], np.float32))

    def map_bounds(self):
        return self._bounds


GOAL = {"mins": [-14720.0, 7487.0, -1824.0], "maxs": [-8064.0, 7488.0, -352.0]}
SCALE = 100.0 / 198380.0


def _race(**kw):
    kw.setdefault("scale", SCALE)
    kw.setdefault("time_pen", 0.005)
    kw.setdefault("success_bonus", 50.0)
    return RaceReward(EuclidField(GOAL), **kw)


def _place(core, d, i=0):
    """Put env `i` exactly `d` units from the box face at y = 7487."""
    core.states_view["origin"][i] = (-11000.0 + 300.0 * i, 7487.0 + float(d),
                                     -1000.0)


def _step(rw, core, done=None, trunc=None):
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    done = z if done is None else np.asarray(done, np.uint8)
    trunc = z if trunc is None else np.asarray(trunc, np.uint8)
    obs = np.zeros((n, 15), np.float32)
    return rw(obs, obs, obs, np.zeros(n, np.float32), done, trunc, core)


def test_flag_off_is_the_control_bit_for_bit():
    a = _race(int_coef=0.25)
    b = _race(int_coef=0.25, cc_tmax=0.0)
    ca, cb = FakeCore(), FakeCore()
    out = []
    for rw, c in ((a, ca), (b, cb)):
        rs = []
        for d in list(range(20000, 3000, -137)) + list(range(3000, 20000, 311)):
            _place(c, d)
            rs.append(_step(rw, c).copy())
        out.append(np.concatenate(rs))
    assert out[0].dtype == out[1].dtype == np.float32
    assert np.array_equal(out[0], out[1])
    assert b.cc_T() is None and b.cc_obs() is None


def test_per_env_mix_against_a_hand_computation():
    n = 4
    # max_step 1000: the 300 u move below must count in full (the default
    # 100 u clip is the teleport guard, not part of what is tested here)
    rw = _race(int_coef=0.25, fail_pen=1.0, cc_tmax=2.0, max_step=1000.0)
    core = FakeCore(n)
    for i in range(n):
        _place(core, 20000.0, i)
    _step(rw, core)                                   # on_reset
    rw.set_cc_T([0.0, 0.5, 1.0, 2.0])
    assert np.array_equal(rw.cc_T(), [0.0, 0.5, 1.0, 2.0])
    assert np.allclose(rw.cc_obs(), cc_encode([0.0, 0.5, 1.0, 2.0], 2.0))
    # every env moves 300 u closer: a new 256 u cell for every env, count 0
    for i in range(n):
        _place(core, 19700.0, i)
    r = _step(rw, core)
    shaping = 300.0 * SCALE - 0.005
    want = [(1.0 - T / 2.0) * shaping + T * 0.25 / np.sqrt(1.0)
            for T in (0.0, 0.5, 1.0, 2.0)]
    assert np.allclose(r, want, atol=1e-6), (r, want)
    # T = 0 pays NO novelty, T_max pays no shaping (only the bonus x T_max)
    assert abs(float(r[0]) - shaping) < 1e-6
    assert abs(float(r[3]) - 2.0 * 0.25) < 1e-6
    # the outcome terms are paid in full to every member: the success bonus
    # to env 1 (goal), the fail penalty to env 2 (done, no goal)
    core.goal_hits[:] = 0
    core.goal_hits[1] = 1
    done = np.zeros(n, np.uint8)
    done[1] = done[2] = 1
    r2 = _step(rw, core, done=done)
    assert abs(float(r2[1]) - 50.0) < 1e-6
    assert abs(float(r2[2]) + 1.0) < 1e-6
    assert float(r2[0]) == pytest.approx(-0.005, abs=1e-7)   # sat still: time pen
    assert float(r2[3]) == pytest.approx(0.0, abs=1e-7)      # (1 - 1) x time pen
    with pytest.raises(ValueError):
        rw.set_cc_T([0.0, 0.0, 0.0, 3.0])
    with pytest.raises(ValueError):
        _race(int_coef=0.25, cc_tmax=2.0, cc_p0=1.5)
    with pytest.raises(ValueError):
        _race(int_coef=0.25, cc_tmax=2.0, cc_tmin=3.0)
    with pytest.raises(ValueError):
        _race(int_coef=0.25, cc_tmax=2.0, speed_equiv=0.1)


# ==========================================================================
# (d) resampling at episode end
# ==========================================================================
def test_T_is_redrawn_for_the_ended_envs_and_boot_keeps_the_old_one():
    n = 8
    rw = _race(int_coef=0.25, cc_tmax=2.0, cc_p0=0.5, cc_tmin=0.05, cc_seed=11)
    core = FakeCore(n)
    for i in range(n):
        _place(core, 20000.0, i)
    _step(rw, core)
    T0 = rw.cc_T().copy()
    assert np.array_equal(T0, cc_draw(11, np.arange(n), 0, 0.5, 0.05, 2.0))
    assert np.array_equal(rw.cc_boot(), T0)
    done = np.zeros(n, np.uint8)
    done[1] = done[5] = 1
    _step(rw, core, done=done)
    T1 = rw.cc_T()
    keep = np.ones(n, bool)
    keep[[1, 5]] = False
    assert np.array_equal(T1[keep], T0[keep])
    assert np.array_equal(T1[[1, 5]], cc_draw(11, [1, 5], [1, 1], 0.5, 0.05, 2.0))
    assert np.array_equal(rw.cc_boot()[[1, 5]], T0[[1, 5]])
    assert np.allclose(rw.cc_obs_boot([1, 5]), cc_encode(T0[[1, 5]], 2.0))
    assert np.allclose(rw.cc_obs_boot([1, 5], live=True), cc_encode(T1[[1, 5]], 2.0))
    assert np.allclose(rw.cc_obs(), cc_encode(T1, 2.0))
    # a truncation is an episode end too
    trunc = np.zeros(n, np.uint8)
    trunc[1] = 1
    _step(rw, core, trunc=trunc)
    assert np.array_equal(rw.cc_T()[1:2], cc_draw(11, [1], [2], 0.5, 0.05, 2.0))
    s = rw.cc_summary()
    assert 0.0 <= s["frac0"] <= 1.0 and s["T_max"] <= 2.0


# ==========================================================================
# (e) the eval wrappers
# ==========================================================================
class _EvalCore:
    class _Cfg:
        pitch_rate_max_deg = RATE
    config = _Cfg()

    def __init__(self, n):
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)


class _Lidar:
    def render(self, o, yw, pt, dk):
        return torch.zeros(o.shape[0], 32)


def test_eval_wrappers_append_the_column_last_with_the_encoded_value():
    torch.manual_seed(0)
    p = Policy(15 + 1 + 32, 8, 4, emb=16, hidden=16, route_dim=1,
               view_continuous=True, view_absolute="velocity")
    p.eval()
    pk = HeadPacker(CPU)
    obs = np.random.default_rng(0).standard_normal((5, 15)).astype(np.float32)
    feed = make_cc_feed(0.7, 2.0)
    assert float(feed.t) == pytest.approx(float(cc_encode(0.7, 2.0)))
    g = GreedyTorchPolicy(p, pk, CPU, _Lidar(), _EvalCore(5), 4, cc_fn=feed)
    row = g._obs(obs)
    assert row.shape == (5, 48)
    assert torch.allclose(row[:, 15], torch.full((5,), float(cc_encode(0.7, 2.0))))
    assert torch.equal(row[:, :15], torch.as_tensor(obs))
    assert torch.all(row[:, 16:] == 0.0)
    a = g.act(obs)
    assert a.shape == (5, 6) and g.view.shape == (5, 2)
    # per-env T: one value per env
    fv = make_cc_feed([0.0, 0.5, 1.0, 1.5, 2.0], 2.0)
    s = SampledTorchPolicy(p, pk, CPU, _Lidar(), _EvalCore(5), 1, cc_fn=fv)
    row = s._obs(obs)
    assert np.allclose(row[:, 15].numpy(), cc_encode([0.0, 0.5, 1.0, 1.5, 2.0], 2.0))
    # without the feed the row is the plain one
    p0 = _tiny(absolute="velocity")
    g0 = GreedyTorchPolicy(p0, pk, CPU, _Lidar(), _EvalCore(5), 4)
    assert g0._obs(obs).shape == (5, 47)
    with pytest.raises(ValueError):
        make_cc_feed(2.5, 2.0)
    with pytest.raises(ValueError):
        fv(_EvalCore(3), 3)


# ==========================================================================
# (f) per-bucket normalisation
# ==========================================================================
def test_bucket_normalize_matches_per_bucket_moments():
    g = torch.Generator().manual_seed(4)
    a = torch.randn(300, generator=g) * torch.tensor([1.0, 10.0, 0.1, 3.0]).repeat(75)
    bk = torch.randint(0, 4, (300,), generator=g)
    bk[0] = 3
    bk[bk == 3] = 0
    bk[0] = 3                                     # bucket 3 has ONE row
    out = cc_bucket_normalize(a, bk, 4)
    want = torch.zeros_like(a)
    for b in range(4):
        m = bk == b
        if int(m.sum()) >= 2:
            want[m] = (a[m] - a[m].mean()) / (a[m].std() + 1e-8)
    assert torch.allclose(out, want, atol=1e-5)
    assert float(out[0]) == 0.0
    # one bucket = the shipped per-minibatch estimator
    one = cc_bucket_normalize(a, torch.zeros(300, dtype=torch.long), 1)
    assert torch.allclose(one, (a - a.mean()) / (a.std() + 1e-8), atol=1e-6)


# ==========================================================================
# (g) the per-env keys temperature
# ==========================================================================
def test_temp_block_pins_the_view_bins_and_the_tempered_logprob_is_per_row():
    kt = cc_keys_temp([0.0, 2.0, 1.0], 0.25)
    assert np.allclose(kt, [1.0, 1.5, 1.25])
    blk = cc_temp_block(kt, view_continuous=False)
    assert blk.shape == (3, NACT) and blk.dtype == np.float32
    assert np.all(blk[:, :N_VIEW] == 1.0)
    assert np.allclose(blk[:, N_VIEW:], kt[:, None])
    blv = cc_temp_block(kt, view_continuous=True)
    assert blv.shape == (3, 1) and np.allclose(blv[:, 0], kt)
    # the tempered log-prob with the per-row block == per-row scalar temps on
    # the keys plus the untempered view bins
    torch.manual_seed(5)
    logits = torch.randn(3, sum(NVEC)) * 2.0
    pk = HeadPacker(CPU)
    padded = pk.pad(logits)
    acts = torch.stack([torch.randint(0, n, (3,)) for n in NVEC], 1)
    lp, ent = logprob_entropy_padded(padded, acts, torch.as_tensor(blk).unsqueeze(-1))
    for i in range(3):
        lv, ev = logprob_entropy_padded(padded[i:i + 1, :N_VIEW], acts[i:i + 1, :N_VIEW])
        lk, ek = logprob_entropy_padded(padded[i:i + 1, N_VIEW:], acts[i:i + 1, N_VIEW:],
                                        float(kt[i]))
        assert torch.allclose(lp[i], lv[0] + lk[0], atol=1e-5)
        assert torch.allclose(ent[i], ev[0] + ek[0], atol=1e-5)


def test_tempered_wrapper_at_per_env_keys_temp_is_sample_view_at_that_block():
    p = _tiny(absolute="velocity")
    p.eval()
    pk = HeadPacker(CPU)
    obs = np.random.default_rng(1).standard_normal((5, 15 + 32)).astype(np.float32)
    kt = cc_keys_temp([0.0, 0.5, 1.0, 1.5, 2.0], 0.25)
    torch.manual_seed(12)
    tpol = TemperedTorchPolicy(p, pk, CPU, None, _EvalCore(5), 1, keys_temp=kt)
    assert torch.is_tensor(tpol.keys_temp) and tuple(tpol.keys_temp.shape) == (5, 1, 1)
    assert tpol._temp_view == 1.0
    act = tpol.act(obs)
    view = tpol.view.copy()
    torch.manual_seed(12)
    with torch.no_grad():
        lg, _ = p(torch.as_tensor(obs))
        cat, mu = split_view(lg.float())
        a2, z2, _ = sample_view(pk.pad(cat), mu, p.log_std(),
                                torch.as_tensor(cc_temp_block(kt, True)).unsqueeze(-1),
                                1.0)
    assert np.array_equal(act[:, N_VIEW:], a2[:, N_VIEW:].numpy())
    assert np.allclose(view, view_from_z_t(z2, RATE, "velocity").numpy(), atol=1e-6)
    # the Gaussian heads are untouched by a keys temperature: same z as the
    # untempered draw under the same seed
    torch.manual_seed(12)
    with torch.no_grad():
        _, z1, _ = sample_view(pk.pad(cat), mu, p.log_std(), None)
    assert np.allclose(z1.numpy(), z2.numpy())
    with pytest.raises(ValueError):
        TemperedTorchPolicy(p, pk, CPU, None, _EvalCore(5), 1, keys_temp=[1.0, 0.0])


# ==========================================================================
# (h) source-level pin: the trainer draws and scores per env
# ==========================================================================
def test_trainer_draws_through_temp_t_and_scores_each_row_at_its_own_temperature():
    src = TRAIN.read_text(encoding="utf-8")
    assert "temp_t = torch.ones((N, CC_TEMP_S, 1), device=device)" in src
    assert "b_cct[t].copy_(temp_t)" in src
    assert "f_temp = f_cct[idx]" in src
    assert "a = cc_bucket_normalize(a, f_bkt[idx], CC_B)" in src
    assert re.search(r"f_bkt = \(torch\.bucketize\(f_scal\[:, CC_COL\]", src)
    assert "reward_fn.cc_obs_boot(ti, live=rpd)" in src
    assert "cc_np[:] = reward_fn.cc_obs()" in src
    # the unstuck pins still hold (the same helpers, the same literal calls)
    assert re.search(r"sample_view\(padded, mu, policy\.log_std\(\), temp_t,"
                     r"\s+tempv_t\)", src)
    assert len(re.findall(r"f_temp=temp_t,\s*f_tempv=tempv_t", src)) == 2


# ==========================================================================
# (i) trainer smokes
# ==========================================================================
_env_dll = os.environ.get("SURFCORE_DLL")


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _run(cmd, timeout=1800):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          env=_env(), cwd=str(ROOT), timeout=timeout,
                          encoding="utf-8", errors="replace")


def _train(run, extra, script=TRAIN, steps="6144"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE_FLAGS
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


def _pre_trainer(dst: Path):
    """The train_fast.py of the last first-parent commit WITHOUT the flag
    (HEAD itself while it is uncommitted), or None. BYTES: the console is
    cp1251 and the file is UTF-8."""
    try:
        r = subprocess.run(["git", "rev-list", "--first-parent",
                            "--max-count=200", "HEAD"],
                           capture_output=True, text=True, cwd=str(ROOT),
                           timeout=60)
        refs = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        refs = []
    for ref in (tuple(refs) or ("HEAD", "HEAD^")):
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode != 0:
            continue
        if b"--curiosity-cond" in r.stdout:
            continue
        if b"--unstuck" not in r.stdout:
            return None
        dst.write_bytes(r.stdout)
        return ref
    return None


def _assert_runs_identical(a: Path, b: Path):
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert "curiosity_cond" not in ca
    assert ca == cb
    ra, rb = _csv(a.name), _csv(b.name)
    assert len(ra) == len(rb) == 3
    assert list(ra[0]) == list(rb[0])
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
    assert set(sa["policy"]) == set(sb["policy"])
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    assert set(oa) == set(ob)
    for i in oa:
        for k in oa[i]:
            if torch.is_tensor(oa[i][k]):
                assert torch.equal(oa[i][k], ob[i][k]), (i, k)


@needs_run
@pytest.mark.parametrize("mode", ["bins", "abs"])
def test_flag_off_is_bit_identical_to_the_trainer_before_curiosity_cond(mode):
    """Both sampling paths and the update were touched (the temperature
    slot, mb_step's advantage branch, the obs layout): the flag OFF must
    reproduce the pre-flag trainer - config, progress.csv (fps excluded,
    header included), eval trajectory, weights, Adam moments."""
    old = ROOT / "python" / "train_fast_precc.py"
    ref = _pre_trainer(old)
    if ref is None:
        pytest.skip("no pre-curiosity-cond train_fast.py in the first-parent history")
    flags = ABS if mode == "abs" else []
    try:
        _train(f"cya_cc_new_{mode}", flags)
        _train(f"cya_cc_old_{mode}", flags, script=old)
    finally:
        old.unlink(missing_ok=True)
    a, b = ROOT / "runs" / f"cya_cc_new_{mode}", ROOT / "runs" / f"cya_cc_old_{mode}"
    _assert_runs_identical(a, b)
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)


@needs_run
def test_curiosity_cond_smoke_trains_evals_records_resumes_and_widens():
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    run = "cya_cc_smoke"
    # 64-tick episodes: every env ends (truncates) five times in 20,480
    # steps, so T is redrawn, the bootstrap rebuilds terminal rows with the
    # column, and every bucket sees episodes
    r = _train(run, ABS + ["--curiosity-cond", "--ep-ticks", "64"],
               steps="20480")
    assert "--curiosity-cond: per-env T = 0 with p 0.5, else log-uniform [0.05, 2]" in r.stdout
    assert "obs column 15" in r.stdout and "keys heads sample at 1 + 0.25 T" in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["curiosity_cond"] == 1 and cfg["cc_p0"] == 0.5
    assert cfg["cc_tmin"] == 0.05 and cfg["cc_tmax"] == 2.0
    assert cfg["cc_buckets"] == 4 and cfg["cc_temp_scale"] == 1
    assert cfg["cc_temp_gain"] == 0.25
    rows = _csv(run)
    assert len(rows) == 10
    head = list(rows[0])
    assert head[-14:] == ["cc/frac0", "cc/T_mean", "cc/n_b0", "cc/len_b0",
                          "cc/rew_b0", "cc/n_b1", "cc/len_b1", "cc/rew_b1",
                          "cc/n_b2", "cc/len_b2", "cc/rew_b2", "cc/n_b3",
                          "cc/len_b3", "cc/rew_b3"]
    for x in rows:
        assert np.isfinite(float(x["train/loss"])) and np.isfinite(float(x["train/value_loss"]))
        assert abs(float(x["train/approx_kl"])) < 0.05
        assert 0.25 <= float(x["cc/frac0"]) <= 0.75
        assert 0.0 < float(x["cc/T_mean"]) < 1.0
    # T is REDRAWN: the share at 0 moves between iterations that ended episodes
    fr = [float(x["cc/frac0"]) for x in rows]
    assert len(set(fr)) > 1
    ended = [x for x in rows if x["cc/n_b0"] not in ("", "0")]
    assert len(ended) >= 4
    for x in ended:
        assert sum(int(x[f"cc/n_b{b}"]) for b in range(4)) == 64
        assert float(x["cc/len_b0"]) == 64.0
        # bucket 0 (T = 0) is paid the full time penalty of 64 ticks
        # (-0.32) and no novelty; the top bucket's mix is far from it
        assert -0.34 < float(x["cc/rew_b0"]) < -0.28
        if x["cc/n_b3"] not in ("", "0"):
            assert float(x["cc/rew_b3"]) > float(x["cc/rew_b0"]) + 0.1
    assert "cc T0" in r.stdout and "len 64/64/64/64" in r.stdout
    assert (d / "traj_0000002048.jsonl").exists()
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["curiosity_cond"] == 1
    # 10 no-GPS scalars + emb 64 + the T column
    assert tuple(ck["policy"]["pi.0.weight"].shape)[1] == 10 + 64 + 1

    # record_ckpt: another member of the family, greedy and stochastic
    out = d / "rec_T1.jsonl"
    rr = _run([sys.executable, "-u", RECORD, d / "ckpt_final.pt",
               "--map", CANNONBALL, "--episodes", "1", "--cc-T", "1.0",
               "--out", out], timeout=900)
    assert rr.returncode == 0, rr.stdout[-3000:] + rr.stderr[-3000:]
    assert "recording the member at T = 1 (t = 0.6309, obs column 15)" in rr.stdout
    hdr = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert hdr["cc_T"] == 1.0
    rs = _run([sys.executable, "-u", RECORD, d / "ckpt_final.pt",
               "--map", CANNONBALL, "--episodes", "1", "--cc-T", "1.5",
               "--stochastic", "--out", d / "rec_T15s.jsonl"], timeout=900)
    assert rs.returncode == 0, rs.stdout[-3000:] + rs.stderr[-3000:]
    assert "samples the keys at temperature 1.375" in rs.stdout
    r0 = _run([sys.executable, "-u", RECORD, d / "ckpt_final.pt",
               "--map", CANNONBALL, "--episodes", "1",
               "--out", d / "rec_T0.jsonl"], timeout=900)
    assert r0.returncode == 0 and "member at T = 0 (t = 0.0000" in r0.stdout

    # a flagless resume restores the flag and every knob
    run2 = run + "_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", TRAIN, "--run", run2, "--ckpt",
               d / "ckpt_final.pt"] + SMOKE_FLAGS + ["--steps", "22528"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "curiosity_cond=1" in r2.stdout and "cc_temp_gain=0.25" in r2.stdout
    rows2 = _csv(run2)
    assert len(rows2) == 1 and 0.0 <= float(rows2[0]["cc/frac0"]) <= 1.0

    # a plain checkpoint is widened onto the family (the column is LAST)
    plain = "cya_cc_plain"
    _train(plain, ABS, steps="2048")
    dp = ROOT / "runs" / plain
    run3 = run + "_widen"
    shutil.rmtree(ROOT / "runs" / run3, ignore_errors=True)
    r3 = _run([sys.executable, "-u", TRAIN, "--run", run3, "--ckpt",
               dp / "ckpt_final.pt"] + SMOKE_FLAGS
              + ["--steps", "4096", "--curiosity-cond"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    assert ("--curiosity-cond: this checkpoint's towers read 0 of this run's "
            "1 scalar-side columns; widened 6 tensors") in r3.stdout
    # --cc-T on a plain checkpoint is refused
    rb = _run([sys.executable, "-u", RECORD, dp / "ckpt_final.pt",
               "--map", CANNONBALL, "--episodes", "1", "--cc-T", "0.5",
               "--out", dp / "rec_bad.jsonl"], timeout=900)
    assert rb.returncode != 0
    assert "this checkpoint was not trained with it" in rb.stdout + rb.stderr

    # beam_tas: the proposal envs as the training mixture, the greedy
    # gate/replay core the T = 0 member, the trained keys temperatures
    if ROUTE.exists():
        bo = d / "beam_mix"
        rp = _run([sys.executable, "-u", BEAM, d / "ckpt_final.pt",
                   "--map", CANNONBALL, "--envs", "16", "--max-ticks", "300",
                   "--objective", "auto", "--skip-gate", "--allow-nonfinisher",
                   "--greedy-eps", "1", "--route-file", ROUTE,
                   "--out-dir", bo, "--cc-T", "mix", "--seed", "0"],
                  timeout=900)
        assert rp.returncode == 0, rp.stdout[-3000:] + rp.stderr[-3000:]
        assert "proposal envs at the training mixture (p0 0.5, [0.05, 2])" in rp.stdout
        assert "--cc-temp trained: proposal keys temperatures 1.000.." in rp.stdout
        assert "replay exact" in rp.stdout
        z = np.load(bo / "beam_best.npz", allow_pickle=False)
        assert str(z["cc_T"]) == "mix" and int(z["view_mode"]) == 1
        assert json.loads((bo / "summary.json").read_text(encoding="utf-8"))["cc_T"] == "mix"
        rq = _run([sys.executable, "-u", BEAM, dp / "ckpt_final.pt",
                   "--map", CANNONBALL, "--envs", "16", "--max-ticks", "300",
                   "--objective", "auto", "--skip-gate", "--allow-nonfinisher",
                   "--greedy-eps", "1", "--route-file", ROUTE,
                   "--out-dir", dp / "beam_bad", "--cc-T", "0.5"], timeout=900)
        assert rq.returncode != 0
        assert "this checkpoint was not trained with it" in rq.stdout + rq.stderr

    # expert_dagger refuses the family with the reason
    import expert_dagger
    with pytest.raises(SystemExit, match="does not relabel a --curiosity-cond"):
        expert_dagger.load_bundle(str(d / "ckpt_final.pt"), str(CANNONBALL), "cpu")
    for dd in (d, ROOT / "runs" / run2, ROOT / "runs" / run3, dp):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
def test_curiosity_cond_is_refused_where_it_cannot_be_right():
    for extra, msg in ((["--unstuck"], "not implemented with --unstuck"),
                       (["--obs-reward"], "not implemented with --obs-reward"),
                       (["--rnn", "gru"], "not implemented with --rnn"),
                       (["--int-coef", "0"], "needs --int-coef > 0"),
                       (["--cc-tmin", "3"], "--cc-tmin must be in (0, --cc-tmax]")):
        r = _run([sys.executable, "-u", TRAIN, "--run", "cya_cc_refuse"]
                 + SMOKE_FLAGS + ["--steps", "2048", "--curiosity-cond"] + extra)
        assert r.returncode != 0, extra
        assert msg in r.stdout + r.stderr, (extra, (r.stdout + r.stderr)[-1500:])
    shutil.rmtree(ROOT / "runs" / "cya_cc_refuse", ignore_errors=True)


def test_bc_dataset_synthesises_the_T_column_at_zero(tmp_path):
    from surfgym.bc import BCDataset
    from test_view_absolute import _write_bc
    fp = tmp_path / "bc.npz"
    _write_bc(fp, n=8, view_absolute="velocity")
    bc = BCDataset(fp, CPU, n_latch=0, obs_reward=False, view_continuous=True,
                   yaw_adaptive=True, pitch_rate_max_deg=RATE,
                   view_absolute="velocity", n_cc=1)
    assert tuple(bc.scal.shape) == (8, 16)
    assert torch.all(bc.scal[:, 15] == 0.0)
    assert "T column synthesised at 0" in bc.describe()
    bc0 = BCDataset(fp, CPU, n_latch=0, obs_reward=False, view_continuous=True,
                    yaw_adaptive=True, pitch_rate_max_deg=RATE,
                    view_absolute="velocity")
    assert tuple(bc0.scal.shape) == (8, 15) and "T column" not in bc0.describe()
