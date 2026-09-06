"""--unstuck (docs/unstuck.md): the sampling temperature, the eval wrapper
that carries it (TemperedTorchPolicy), the plateau schedule, the novelty
count decay, tools/diversity_bench.py's metric helpers, and the trainer.

(a) tempering: temp None is the shipped expression op for op (same draws
    under the same seed); a temperature scales the Gaussian residual by
    exactly temp and flattens a categorical head towards softmax(logits /
    temp); the log-prob sample_view returns IS logprob_entropy_view's at
    the same temp (the ratio is 1 at the rollout) and equals a hand
    computation; a float and a 0-d tensor temp agree.
(b) TemperedTorchPolicy: temp 1 / eps 0 is SampledTorchPolicy byte for
    byte; at temp 3 its decision is sample_view(.., temp=3) under the same
    seed (the same code path the trainer's rollout takes); eps 1 makes
    every head uniform.
(c) UnstuckSchedule: 0 while improving, rises after patience at the rate,
    caps, halves per period after a new best (or resets), pulses the
    count decay once per period at T > 0, round-trips its state.
(d) RaceReward.decay_counts: halves (>> 1) or floors (x f), decays a
    pending table too, refuses DDP sharing and a factor outside [0, 1].
(e) diversity_bench helpers: pos_cells == RaceReward._cells (and the
    view/speed-keyed key // 24), the batched order-only progress ==
    eval_honesty.corridor_progress_ordered per rollout, single linkage
    and the medoid RMS on hand cases.
(f) trainer smokes (CPU, the toy scratch set): the flag OFF is bit-identical
    to the trainer of the last commit without --unstuck (bins and the
    absolute view); the flag ON with patience 0 and a 2,048-step period
    shows T rising to the cap, the entropy coefficient scaled, count
    decays pulsing, the config / checkpoint carrying the state, and a
    resume continuing from the stored T.
"""
from __future__ import annotations

import json
import os
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
import torch.nn.functional as F                                # noqa: E402

from surfgym.core import STATE_DTYPE                           # noqa: E402
from surfgym.rewards import RaceReward                         # noqa: E402
from train_fast import (N_VIEW, NVEC, HeadPacker,              # noqa: E402
                        SampledTorchPolicy, TemperedTorchPolicy,
                        UnstuckSchedule, gauss_logp, logprob_entropy_padded,
                        logprob_entropy_view, sample_padded, sample_view,
                        split_view, view_from_z_t)
import diversity_bench as db                                   # noqa: E402
from eval_honesty import corridor_progress_ordered             # noqa: E402
from test_view_absolute import RATE, _Core, _tiny              # noqa: E402
from test_view_continuous import SMOKE_FLAGS, needs_run        # noqa: E402

CPU = torch.device("cpu")


def _padded(B=6, seed=0, scale=2.0):
    torch.manual_seed(seed)
    logits = torch.randn(B, sum(NVEC)) * scale
    return HeadPacker(CPU).pad(logits), logits


# ==========================================================================
# (a) tempering
# ==========================================================================
def test_temp_none_is_the_shipped_draw_op_for_op():
    padded, _ = _padded()
    mu = torch.randn(6, 2)
    ls = torch.tensor([-1.2, -0.7])
    torch.manual_seed(1)
    a0, l0 = sample_padded(padded)
    torch.manual_seed(1)
    a1, l1 = sample_padded(padded, None)
    assert torch.equal(a0, a1) and torch.equal(l0, l1)
    torch.manual_seed(2)
    r0 = sample_view(padded, mu, ls)
    torch.manual_seed(2)
    r1 = sample_view(padded, mu, ls, None)
    assert all(torch.equal(x, y) for x, y in zip(r0, r1))
    lp0, e0 = logprob_entropy_view(padded, r0[0], mu, ls, r0[1], 0.0)
    lp1, e1 = logprob_entropy_view(padded, r0[0], mu, ls, r0[1], 0.0, None)
    assert torch.equal(lp0, lp1) and torch.equal(e0, e1)
    lp0, e0 = logprob_entropy_padded(padded, a0)
    lp1, e1 = logprob_entropy_padded(padded, a0, None)
    assert torch.equal(lp0, lp1) and torch.equal(e0, e1)


def test_temp_scales_the_gaussian_residual_and_flattens_the_categoricals():
    padded, logits = _padded(B=1, scale=3.0)
    mu = torch.zeros(1, 2)
    ls = torch.tensor([-1.2, -0.7])
    torch.manual_seed(3)
    z1 = sample_view(padded, mu, ls, 1.0)[1]
    torch.manual_seed(3)
    z3 = sample_view(padded, mu, ls, 3.0)[1]
    torch.manual_seed(3)
    z3t = sample_view(padded, mu, ls, torch.tensor(3.0))[1]
    assert torch.allclose(z3 - mu, 3.0 * (z1 - mu), rtol=1e-5)
    assert torch.allclose(z3, z3t)
    # the yaw head (15 bins): the empirical law at temp t is softmax(l / t)
    n = 20000
    for t in (1.0, 4.0):
        torch.manual_seed(4)
        acts = torch.stack([sample_padded(padded, t)[0][0, 0]
                            for _ in range(n)])
        freq = torch.bincount(acts, minlength=15).float() / n
        want = F.softmax(logits[0, :15] / t, dim=0)
        assert (freq - want).abs().max() < 0.02, (t, freq, want)
    # and it IS flatter: the entropy of the tempered head grows with t
    def ent(t):
        p = F.softmax(logits[0, :15] / t, dim=0)
        return float(-(p * p.log()).sum())
    assert ent(1.0) < ent(2.0) < ent(4.0)


def test_tempered_logprob_is_the_behaviour_density():
    padded, logits = _padded(B=8, seed=5)
    mu = torch.randn(8, 2)
    ls = torch.tensor([-1.2, -0.7])
    for t in (1.7, torch.tensor(2.5)):
        torch.manual_seed(6)
        act, z, lp = sample_view(padded, mu, ls, t)
        lp2, ent = logprob_entropy_view(padded, act, mu, ls, z, 1.0, t)
        assert torch.allclose(lp, lp2, atol=1e-5)
        tf = float(t)
        # by hand: tempered softmax over the four categorical heads + the
        # Gaussian at sigma * t
        lsm = F.log_softmax(padded[:, N_VIEW:] / tf, dim=-1)
        hand_c = lsm.gather(-1, act[:, N_VIEW:].unsqueeze(-1)).squeeze(-1).sum(-1)
        hand = hand_c + gauss_logp(z, mu, ls + np.log(tf))
        assert torch.allclose(lp, hand, atol=1e-5)
        # the entropy is the tempered distribution's: Gaussian part grows by
        # 2 log t, categorical part is flatter
        _, ent1 = logprob_entropy_view(padded, act, mu, ls, z, 1.0, None)
        assert torch.all(ent > ent1)


# ==========================================================================
# (b) the wrapper
# ==========================================================================
def _obs(n=5, seed=0):
    return np.random.default_rng(seed).standard_normal(
        (n, 15 + 32)).astype(np.float32)


def test_tempered_wrapper_at_T0_is_the_sampled_policy():
    p = _tiny(absolute="velocity")
    p.eval()
    pk = HeadPacker(CPU)
    obs = _obs()
    torch.manual_seed(11)
    s = SampledTorchPolicy(p, pk, CPU, None, _Core(), 1)
    a0 = s.act(obs).copy()
    v0 = s.view.copy()
    torch.manual_seed(11)
    tpol = TemperedTorchPolicy(p, pk, CPU, None, _Core(), 1, temp=1.0, eps=0.0)
    a1 = tpol.act(obs)
    assert np.array_equal(a0, a1) and np.array_equal(v0, tpol.view)
    assert tpol.temp is None and tpol.eps == 0.0


def test_tempered_wrapper_is_sample_view_at_the_same_temp():
    p = _tiny(absolute="velocity")
    p.eval()
    pk = HeadPacker(CPU)
    obs = _obs()
    torch.manual_seed(12)
    tpol = TemperedTorchPolicy(p, pk, CPU, None, _Core(), 1, temp=3.0)
    act = tpol.act(obs)
    view = tpol.view.copy()
    torch.manual_seed(12)
    with torch.no_grad():
        lg, _ = p(torch.as_tensor(obs))
        cat, mu = split_view(lg.float())
        a2, z2, _ = sample_view(pk.pad(cat), mu, p.log_std(), 3.0)
    assert np.array_equal(act[:, N_VIEW:], a2[:, N_VIEW:].numpy())
    assert np.allclose(view, view_from_z_t(z2, RATE, "velocity").numpy(),
                       atol=1e-6)
    # and NOT the untempered draw
    torch.manual_seed(12)
    with torch.no_grad():
        _, z1, _ = sample_view(pk.pad(cat), mu, p.log_std(), None)
    assert not np.allclose(view, view_from_z_t(z1, RATE, "velocity").numpy())


def test_per_component_temperature_reaches_only_its_component():
    """temp_view overrides temp for the Gaussian heads only; the wrapper's
    view_scale / keys_temp go through the same call. Under one seed the
    categorical draw is untouched by a view-only temperature (same Gumbel
    noise, same logits) and the Gaussian residual is untouched by a
    keys-only one; the log-prob follows the same split."""
    padded, _ = _padded(B=8, seed=7)
    mu = torch.randn(8, 2)
    ls = torch.tensor([-1.2, -0.7])
    torch.manual_seed(8)
    a0, z0, lp0 = sample_view(padded, mu, ls, None)
    torch.manual_seed(8)
    a1, z1, lp1 = sample_view(padded, mu, ls, None, [3.0, 1.0])   # yaw only
    assert torch.equal(a0, a1)
    assert torch.allclose(z1[:, 0] - mu[:, 0], 3.0 * (z0[:, 0] - mu[:, 0]))
    assert torch.allclose(z1[:, 1], z0[:, 1])
    lp1b, _ = logprob_entropy_view(padded, a1, mu, ls, z1, 1.0, None, [3.0, 1.0])
    assert torch.allclose(lp1, lp1b, atol=1e-5)
    torch.manual_seed(8)
    a2, z2, lp2 = sample_view(padded, mu, ls, 4.0, torch.tensor([1.0, 1.0]))
    assert torch.allclose(z2, z0)                     # keys only: z untouched
    lp2b, _ = logprob_entropy_view(padded, a2, mu, ls, z2, 1.0, 4.0,
                                   torch.tensor([1.0, 1.0]))
    assert torch.allclose(lp2, lp2b, atol=1e-5)
    # the wrapper: view_scale (1, 3) leaves the keys as SampledTorchPolicy
    # draws them and scales the pitch residual by 3 - the same call
    p = _tiny(absolute="velocity")
    p.eval()
    pk = HeadPacker(CPU)
    obs = _obs()
    torch.manual_seed(9)
    s = SampledTorchPolicy(p, pk, CPU, None, _Core(), 1)
    a_s = s.act(obs).copy()
    v_s = s.view.copy()
    torch.manual_seed(9)
    w = TemperedTorchPolicy(p, pk, CPU, None, _Core(), 1, view_scale=(1.0, 3.0))
    a_w = w.act(obs)
    assert np.array_equal(a_s, a_w)
    assert np.allclose(w.view[:, 0], v_s[:, 0], atol=1e-5)   # yaw untouched
    assert not np.allclose(w.view[:, 1], v_s[:, 1])          # pitch scaled
    torch.manual_seed(9)
    k = TemperedTorchPolicy(p, pk, CPU, None, _Core(), 1, keys_temp=5.0)
    k.act(obs)
    assert np.allclose(k.view, v_s, atol=1e-5)               # view untouched
    with pytest.raises(ValueError):
        TemperedTorchPolicy(p, pk, CPU, None, _Core(), 1, view_scale=(1.0,))


def test_eps_one_makes_every_head_uniform():
    p = _tiny(absolute="velocity")
    p.eval()
    pk = HeadPacker(CPU)
    obs = _obs(n=1)
    torch.manual_seed(13)
    tpol = TemperedTorchPolicy(p, pk, CPU, None, _Core(), 1, eps=1.0)
    acts, yaws = [], []
    for _ in range(3000):
        acts.append(tpol.act(obs)[0].copy())
        yaws.append(float(tpol.view[0, 0]))
    acts = np.array(acts)
    for h in range(N_VIEW, 6):
        f = np.bincount(acts[:, h], minlength=NVEC[h]) / len(acts)
        assert np.abs(f - 1.0 / NVEC[h]).max() < 0.05, (h, f)
    # the view head is a uniform u in (-1, 1): the target reaches far past
    # anything a 0.3-sigma policy would draw (off_warp(0.75) ~ 60 deg)
    yaws = np.abs(np.array(yaws))
    assert yaws.max() > 150.0 and (yaws > 60.0).mean() > 0.15
    with pytest.raises(ValueError):
        TemperedTorchPolicy(p, pk, CPU, None, _Core(), 1, temp=0.0)
    with pytest.raises(ValueError):
        TemperedTorchPolicy(p, pk, CPU, None, _Core(), 1, eps=1.5)


def test_eps_of_schedule():
    assert db.eps_of(0.0) == 0.0
    assert abs(db.eps_of(1.0) - 0.05) < 1e-12
    assert db.eps_of(4.0) == 0.2
    assert db.eps_of(100.0) == 0.5


# ==========================================================================
# (c) the schedule
# ==========================================================================
def test_schedule_rises_after_patience_caps_and_halves_after_a_best():
    s = UnstuckSchedule(eps=500, patience=1000, rate=1.0, tmax=4.0,
                        period=1000, step=0)
    seen = []
    for st in range(0, 13001, 1000):
        prog = 1000.0 if st < 9000 else (3000.0 if st == 9000 else 3000.0)
        T, dec = s.observe(st, res_prog=prog)
        seen.append((st, round(T, 3), dec, s.stuck_steps))
    assert seen[0] == (0, 0.0, False, 0)          # first reading = a best
    assert seen[1] == (1000, 0.0, False, 1000)    # not yet past patience
    assert seen[2] == (2000, 1.0, True, 2000)     # rising; decay pulses
    assert seen[5] == (5000, 4.0, True, 5000)     # capped
    assert seen[8] == (8000, 4.0, True, 8000)
    assert seen[9] == (9000, 2.0, True, 0)        # new best: halved
    assert seen[10] == (10000, 1.0, True, 1000)   # still within patience
    assert seen[11] == (11000, 2.0, True, 2000)   # stuck again: rising
    assert s.n_improve == 2 and s.n_decays == 12   # one pulse per period at T > 0
    assert s.best == 3000.0


def test_schedule_improvement_needs_eps_and_reset_mode_resets():
    s = UnstuckSchedule(eps=500, patience=0, rate=1.0, tmax=10.0,
                        period=1000, reset=True, step=0)
    s.observe(0, res_prog=1000.0)
    s.observe(1000, res_prog=1400.0)       # +400 < eps: not an improvement
    assert s.best == 1000.0 and s.T == 1.0
    s.observe(2000, res_prog=1600.0)       # +600 > eps: reset
    assert s.best == 1600.0 and s.T == 0.0 and s.stuck_steps == 0
    # the arc reading counts too, and a NaN reading is no reading
    s.observe(3000, res_prog=float("nan"), arc_prog=50.0)
    assert s.best_arc == 50.0 and s.T == 0.0
    s.observe(4000)
    assert s.T == 1.0
    # T never goes negative and decays to exactly 0 once tiny
    s2 = UnstuckSchedule(patience=0, rate=1.0, tmax=1.0, period=100, step=0)
    s2.observe(0, res_prog=1.0)
    s2.observe(100)
    assert s2.T == 1.0
    for k in range(2, 40):
        s2.observe(100 * k, res_prog=1e6 * k)   # improving every call
    assert s2.T == 0.0 and s2.decay_acc == 0.0


def test_schedule_state_round_trips():
    s = UnstuckSchedule(patience=0, rate=0.5, tmax=4.0, period=1000, step=0)
    s.observe(0, res_prog=10.0)
    s.observe(2000, res_prog=10.0)
    d = s.state_dict()
    assert set(d) == set(UnstuckSchedule.KEYS)
    d2 = json.loads(json.dumps(d))          # checkpoint-shaped
    s2 = UnstuckSchedule(patience=0, rate=0.5, tmax=4.0, period=1000, step=0)
    s2.load_state_dict(d2)
    d3 = s2.state_dict()
    for k in d:                              # best_arc is NaN: compare as such
        assert (d3[k] == d[k]) or (d3[k] != d3[k] and d[k] != d[k]), k
    assert s2.observe(3000, res_prog=10.0) == s.observe(3000, res_prog=10.0)
    with pytest.raises(ValueError):
        UnstuckSchedule(period=0.0)


# ==========================================================================
# (d) the count decay
# ==========================================================================
class _Flat:
    def sample(self, pos):
        return np.zeros(len(np.atleast_2d(pos)))


def test_decay_counts_halves_or_floors_and_refuses_ddp():
    rr = RaceReward(_Flat(), scale=1.0, int_coef=0.1, int_view=8, int_speed=3)
    assert rr.decay_counts(0.5) == 0            # no table yet: a no-op
    rr._counts = np.array([0, 1, 2, 3, 1000, 2_000_001], np.int64)
    rr._pending_counts = np.array([7, 8], np.uint32)
    assert rr.decay_counts(0.5) == 4
    assert rr._counts.tolist() == [0, 0, 1, 1, 500, 1_000_000]
    assert rr._pending_counts.tolist() == [3, 4]
    assert rr._counts.dtype == np.int64
    assert rr.decay_counts(0.3) == 2
    assert rr._counts.tolist() == [0, 0, 0, 0, 150, 300_000]
    assert rr.decay_counts(1.0) == 2 and rr._counts.tolist() == [0, 0, 0, 0, 150, 300_000]
    with pytest.raises(ValueError):
        rr.decay_counts(1.5)
    rr.track_touched = True
    with pytest.raises(RuntimeError):
        rr.decay_counts(0.5)


# ==========================================================================
# (e) diversity_bench helpers
# ==========================================================================
def test_pos_cells_is_the_rewards_position_key():
    mins = np.array([-16384.0, -16384.0, -16384.0])
    dims = (129, 129, 129)
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-17000, 17000, (500, 3))
    st = np.zeros(500, STATE_DTYPE)
    st["origin"] = xyz.astype(np.float32)
    st["yaw"] = rng.uniform(-180, 180, 500).astype(np.float32)
    st["velocity"][:, 0] = rng.uniform(0, 3500, 500).astype(np.float32)
    rr0 = RaceReward(_Flat(), scale=1.0, int_coef=0.1)
    rr0._mins, rr0._dims = mins, dims
    key0 = rr0._cells(st)
    mine = db.pos_cells(st["origin"], mins, dims, 256.0)
    assert np.array_equal(key0, mine)
    rr = RaceReward(_Flat(), scale=1.0, int_coef=0.1, int_view=8, int_speed=3)
    rr._mins, rr._dims = mins, dims
    key = rr._cells(st)
    assert np.array_equal(key // 24, mine)      # (pos * 8 + view) * 3 + speed


def _route():
    a = np.stack([np.arange(100) * 128.0, np.zeros(100), np.zeros(100)], 1)
    b = np.stack([np.full(60, 99 * 128.0), (np.arange(60) + 1) * 128.0,
                  np.zeros(60)], 1)
    return np.concatenate([a, b]).astype(np.float32), 128.0


def test_batched_order_only_progress_matches_eval_honesty_per_rollout():
    pts, spacing = _route()
    T = 600
    t = np.arange(T)
    # A rides the line at 30 u/tick and turns the corner; B rides it, then
    # leaves the corridor sideways and keeps going; C ends early (NaN after)
    A = np.stack([np.minimum(t * 30.0, 99 * 128.0),
                  np.maximum(t * 30.0 - 99 * 128.0, 0.0), np.zeros(T)], 1)
    B = A.copy()
    B[300:, 1] += np.arange(300) * 12.0 + 1600.0
    C = A.copy()
    C[200:] = np.nan
    pos = np.stack([A, B, C], 1).astype(np.float32)
    posf = db.forward_fill(pos)
    got = db.order_only_progress(posf, pts, spacing)
    want = []
    for i, L in enumerate((T, T, 200)):
        q, _ = corridor_progress_ordered(pos[:L, i], pts, spacing, 1500.0, 16)
        want.append(q)
    assert np.allclose(got, want, atol=1e-6), (got, want)
    assert got[0] > got[1] > got[2] > 0.0
    assert abs(got[0] - 599 * 30.0) < 200.0


def test_single_linkage_medoid_and_alive_masks():
    P = np.array([[0, 0, 0], [100, 0, 0], [1000, 0, 0], [1100, 0, 0]], float)
    assert db.single_linkage(P, 512.0) == 2
    assert db.single_linkage(P, 50.0) == 4
    assert db.single_linkage(P, 5000.0) == 1
    assert db.single_linkage(P[:0], 512.0) == 0
    assert db.medoid_rms(P[:1]) == 0.0
    assert abs(db.medoid_rms(P[:2]) - 100.0 / np.sqrt(2)) < 1e-9
    end = np.array([-1, 250, 99])
    assert db.alive_at(end, 100, 300).tolist() == [True, True, False]
    assert db.alive_at(end, 250, 300).tolist() == [True, True, False]
    assert db.alive_at(end, 251, 300).tolist() == [True, False, False]
    assert not db.alive_at(end, 300, 300).any()
    pos = np.full((300, 3, 3), np.nan, np.float32)
    pos[:, 0] = [0, 0, 0]
    pos[:251, 1] = [300, 0, 0]
    pos[:100, 2] = [0, 400, 0]
    ticks, rms, nal = db.spread_curve(pos, end, every=100)
    assert ticks.tolist() == [0, 100, 200] and nal.tolist() == [3, 2, 2]
    assert rms[2] > 0.0 and rms[0] > rms[2]
    ends = db.end_positions(db.forward_fill(pos), end)
    assert ends.tolist() == [[0, 0, 0], [300, 0, 0], [0, 400, 0]]


# ==========================================================================
# (f) trainer smokes
# ==========================================================================
TRAIN = ROOT / "python" / "train_fast.py"
_env_dll = os.environ.get("SURFCORE_DLL")


def _env():
    # -1, not "": on Windows an EMPTY value UNSETS the variable and the GPU
    # stays visible (memory: windows-empty-env-var-unsets)
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
    r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE_FLAGS
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


def _preunstuck_trainer(dst: Path):
    """The train_fast.py of the last first-parent commit WITHOUT --unstuck
    (HEAD itself while the flag is uncommitted), or None. BYTES: the
    console is cp1251 and the file is UTF-8."""
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
        if b"--unstuck" in r.stdout:
            continue
        if b"--view-absolute" not in r.stdout:
            return None
        dst.write_bytes(r.stdout)
        return ref
    return None


def _assert_runs_identical(a: Path, b: Path):
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert "unstuck" not in ca
    assert ca == cb
    ra, rb = _csv(a.name), _csv(b.name)
    assert len(ra) == len(rb) == 3
    assert list(ra[0]) == list(rb[0])            # the same header
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
    assert "unstuck" not in sa and "unstuck" not in sb
    assert set(sa["policy"]) == set(sb["policy"])
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    assert set(oa) == set(ob)
    for i in oa:
        for k in oa[i]:
            if torch.is_tensor(oa[i][k]):
                assert torch.equal(oa[i][k], ob[i][k]), (i, k)


ABS = ["--view-continuous", "--view-absolute", "velocity"]


@needs_run
@pytest.mark.parametrize("mode", ["bins", "abs"])
def test_flag_off_is_bit_identical_to_the_trainer_before_unstuck(mode):
    """Both sampling paths were touched (sample_padded for the bins,
    sample_view for the continuous heads): the flag OFF must reproduce
    the pre-flag trainer on each - config, progress.csv (fps excluded,
    header included), eval trajectory, weights, Adam moments."""
    old = ROOT / "python" / "train_fast_preunstuck.py"
    ref = _preunstuck_trainer(old)
    if ref is None:
        pytest.skip("no pre-unstuck train_fast.py in the first-parent history")
    flags = ABS if mode == "abs" else []
    try:
        _train(f"cya_us_new_{mode}", flags)
        _train(f"cya_us_old_{mode}", flags, script=old)
    finally:
        old.unlink(missing_ok=True)
    a, b = ROOT / "runs" / f"cya_us_new_{mode}", ROOT / "runs" / f"cya_us_old_{mode}"
    _assert_runs_identical(a, b)
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)


@needs_run
def test_unstuck_smoke_T_rises_scales_the_coefficients_and_resumes():
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    run = "cya_unstuck"
    r = _train(run, ABS + ["--unstuck", "--unstuck-patience", "0",
                           "--unstuck-period", "2048", "--unstuck-rate", "1",
                           "--unstuck-max", "3"], steps="10240")
    assert "--unstuck: T rises 1 per 2,048 steps" in r.stdout
    assert "--unstuck: novelty counts x 0.5" in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["unstuck"] == 1 and cfg["unstuck_patience"] == 0.0
    assert cfg["unstuck_period"] == 2048.0 and cfg["unstuck_max"] == 3.0
    assert cfg["unstuck_temp"] == 1 and cfg["unstuck_ent"] == 1 \
        and cfg["unstuck_int"] == 1
    rows = _csv(run)
    assert len(rows) == 5
    assert list(rows[0])[-3:] == ["unstuck/T", "unstuck/stuck_steps",
                                  "unstuck/best"]
    T = [float(x["unstuck/T"]) for x in rows]
    stuck = [int(x["unstuck/stuck_steps"]) for x in rows]
    # T is the temperature the iteration's rollout RAN at: 0 on the first,
    # rising by 1 per 2,048-step iteration once stuck (the first reading of
    # a non-empty reservoir counts as a best and delays it by one), capped
    assert T[0] == 0.0 and T[-1] == 3.0
    assert all(b >= a for a, b in zip(T, T[1:]))
    assert 2.0 in T and stuck[-1] > 0
    # the step line's entropy coefficient is 0.005 x (1 + T): 0.02 at the cap
    assert "ent 0.0200" in r.stdout
    assert "  T 3.00" in r.stdout
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    st = ck["unstuck"]
    assert st["T"] == 3.0 and st["n_decays"] >= 1
    # a flagless resume restores the flag, its knobs and the state, and the
    # resumed run's first iteration RUNS at the stored T
    run2 = "cya_unstuck_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "12288"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "unstuck=1" in r2.stdout and "unstuck_period=2048" in r2.stdout
    assert "restored --unstuck state: T 3.000" in r2.stdout
    rows2 = _csv(run2)
    assert len(rows2) == 1 and float(rows2[0]["unstuck/T"]) == 3.0
    cfg2 = json.loads((ROOT / "runs" / run2 / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert cfg2["unstuck"] == 1 and cfg2["unstuck_period"] == 2048.0
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


def test_trainer_rollout_and_update_share_the_wrappers_tempered_helpers():
    """The proof that the trainer's behaviour policy at T IS the benchmark's
    sigma knob at T: the rollout draws through sample_view / sample_padded
    with the static temperature tensor, the update recomputes through
    logprob_entropy_view / logprob_entropy_padded with the same tensor, and
    TemperedTorchPolicy calls the same two draw helpers (test (b) above
    pins that its decision equals sample_view(.., temp))."""
    import re
    src = TRAIN.read_text(encoding="utf-8")
    assert re.search(r"sample_view\(padded, mu, policy\.log_std\(\), temp_t,"
                     r"\s+tempv_t\)", src)
    assert "act, logp = sample_padded(padded, temp_t)" in src
    assert "PITCH_ENT, f_temp, f_tempv)" in src
    assert re.search(r"logprob_entropy_padded\(padded, f_act\[idx\],"
                     r"\s+f_temp\)", src)
    assert len(re.findall(r"f_temp=temp_t,\s*f_tempv=tempv_t", src)) == 2
    assert "set_unstuck_temp(_T_next)" in src


@needs_run
def test_unstuck_is_refused_where_it_cannot_be_right():
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_us_refuse"]
             + SMOKE_FLAGS + ["--steps", "2048", "--unstuck", "--rnn", "gru"])
    assert r.returncode != 0
    assert "--unstuck is not implemented with --rnn" in r.stdout + r.stderr
    shutil.rmtree(ROOT / "runs" / "cya_us_refuse", ignore_errors=True)
