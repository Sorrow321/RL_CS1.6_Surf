"""--view-continuous: yaw and pitch as squashed Gaussians (docs/contyaw.md,
surfgym/view.py, src/env.c surf_step_view, train_fast.py).

What is pinned here:

(a) THE DISCRETE PATH IS UNTOUCHED. The ABI-8 core's surf_step reproduces
    the ABI-7 core's trajectories byte for byte: tests/python/data/
    view_golden_abi7.npz holds states, the final obs and a SHA-256 over
    every obs/reward/done/trunc buffer of 600 scripted ticks on
    surf_ski_2, generated with the DLL built from the commit BEFORE this
    change, under four configs (fixed / adaptive yaw x plain /
    yaw_blend+side_hold).
(b) THE CONTINUOUS PATH REPRODUCES THE BINS. Feeding surf_step_view the
    bins' own table values (surfgym.view.bin_to_view) gives the same
    trajectories, bit for bit, under the same four configs - K_BINS[b] as
    the yaw command IS bin b under --yaw-adaptive, 2*YAW_BINS[b] with
    fixed bins, PITCH_BINS[b]*(rate/10) for the pitch.
(c) The warp: warp(+-0.5) = +-1, warp(+-1) = +-20, odd, monotone, and
    warp_inv is its inverse; the torch copy in train_fast agrees with the
    numpy one.
(d) The mixed distribution: sample_view's log-prob equals a hand
    computation (four categorical terms + two Gaussian densities), the
    update's logprob_entropy_view recomputes the rollout's log-prob bit
    for bit from the stored z, and the entropy is the categorical sum plus
    the Gaussian entropy. Greedy/Sampled eval wrappers publish a view.
(e) The Policy under the flag: the discrete state_dict is a strict subset,
    the three new tensors come LAST in parameters() (Adam's index rule),
    and the flag off draws no RNG and changes no tensor.
(f) The transplant (tools/transplant_view.py) on a synthetic discrete
    policy: shared tensors bit-identical, the fitted mean reproduces a
    linear target, the optimizer groups grow by exactly the new indices.
(g) BC targets: bin_to_view / z_from_view round trip, bin_view_moments on
    a one-hot is the bin's own z with zero spread, BCDataset derives the
    view targets from a discrete file and reads them from a continuous one.
(h) A planner line round trip: a scripted continuous line written as
    beam_best.npz with `view`, read back and replayed through
    surfgym.bc.replay_line, ends on the same tick in the same state.

    python -m pytest tests/python/test_view_continuous.py -q

CPU only. (a), (b), (h) need the built core (SURFCORE_DLL from a worktree).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
SKI = ROOT / "maps" / "surf_ski_2.bsp"
GOLDEN = ROOT / "tests" / "python" / "data" / "view_golden_abi7.npz"
needs_core = pytest.mark.skipif(not (DLL.exists() and SKI.exists()),
                                reason="needs the built core + surf_ski_2")

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402

from surfgym.core import PITCH_BINS, YAW_BINS                  # noqa: E402
from surfgym.view import (K_BINS, K_MAX, LOG_STD_INIT, U_CLIP,  # noqa: E402
                          bin_to_view, bin_view_moments, view_from_z,
                          warp, warp_inv, z_from_view)
from train_fast import (LOG2PI, N_VIEW, NACT, NEUTRAL_ACT, NVEC,  # noqa: E402
                        GreedyTorchPolicy, HeadPacker, Policy,
                        SampledTorchPolicy, gauss_entropy, gauss_logp,
                        logprob_entropy_padded, logprob_entropy_view,
                        sample_view, split_view, view_from_z_t, warp_t)


# ==========================================================================
# (a) + (b): the core
# ==========================================================================
def _scripted(core, adaptive, mode, n=32, ticks=600):
    obs = core.reset(1234)
    rng = np.random.default_rng(7)
    h = hashlib.sha256()
    for _ in range(ticks):
        a = np.stack([rng.integers(0, 15, n), rng.integers(0, 7, n),
                      rng.integers(0, 3, n), rng.integers(0, 3, n),
                      rng.integers(0, 2, n), rng.integers(0, 2, n)],
                     1).astype(np.int32)
        a = np.ascontiguousarray(a)
        if mode == "discrete":
            obs, rew, done, trunc, _t = core.step(a)
        else:
            v = bin_to_view(a, bool(adaptive), 1.33)
            a2 = a.copy()
            a2[:, 0], a2[:, 1] = NEUTRAL_ACT[0], NEUTRAL_ACT[1]
            obs, rew, done, trunc, _t = core.step(a2, view=v)
        for buf in (obs, rew, done, trunc):
            h.update(buf.tobytes())
    return h.hexdigest(), core.get_states().copy(), obs.copy()


@needs_core
@pytest.mark.parametrize("adaptive,blend,hold",
                         [(0, 1.0, 0), (0, 0.5, 3), (1, 1.0, 0), (1, 0.5, 3)])
def test_discrete_path_matches_the_abi7_golden_and_the_view_path_matches_the_bins(
        adaptive, blend, hold):
    from surfgym import SurfCore, default_config
    if not GOLDEN.exists():
        pytest.skip("golden file missing")
    g = np.load(GOLDEN)
    key = f"a{adaptive}_b{blend}_h{hold}"
    for mode in ("discrete", "view"):
        cfg = default_config(num_envs=32, spawn_mode=0, max_episode_ticks=400,
                             water_fail=1, yaw_adaptive=adaptive,
                             yaw_blend=blend, side_hold_ticks=hold,
                             lidar_w=8, lidar_h=4, pitch_rate_max_deg=1.33)
        core = SurfCore(str(SKI), cfg, dll_path=str(DLL))
        hsh, st, obs = _scripted(core, adaptive, mode)
        assert hsh == str(g[key + "_hash"]), (key, mode)
        assert st.tobytes() == g[key + "_states"].tobytes(), (key, mode)
        assert np.array_equal(obs, g[key + "_obs"]), (key, mode)


@needs_core
def test_view_validation_and_nan_guard():
    from surfgym import SurfCore, default_config
    cfg = default_config(num_envs=4, spawn_mode=0, max_episode_ticks=400,
                         yaw_adaptive=1, lidar_w=0, lidar_h=0)
    core = SurfCore(str(SKI), cfg, dll_path=str(DLL))
    core.reset(1)
    a = np.zeros((4, 6), np.int32)
    a[:, 0], a[:, 1] = 7, 3
    with pytest.raises(ValueError):
        core.step(a, view=np.zeros((4, 2), np.float64))
    with pytest.raises(ValueError):
        core.step(a, view=np.zeros((3, 2), np.float32))
    # a NaN command applies 0: the yaw is unchanged by the step
    y0 = core.states_view["yaw"].copy()
    v = np.full((4, 2), np.nan, np.float32)
    core.step(a, view=v)
    y1 = core.states_view["yaw"].copy()
    assert np.allclose(y1, y0)


# ==========================================================================
# (c): the warp
# ==========================================================================
def test_warp_anchors_and_inverse():
    assert warp(0.5) == pytest.approx(1.0, abs=1e-12)
    assert warp(-0.5) == pytest.approx(-1.0, abs=1e-12)
    assert warp(1.0) == pytest.approx(K_MAX, abs=1e-12)
    assert warp(0.0) == 0.0
    u = np.linspace(-1, 1, 2001)
    k = warp(u)
    assert np.all(np.diff(k) > 0)                         # monotone
    assert np.allclose(warp(-u), -k)                      # odd
    assert np.allclose(warp_inv(k), u, atol=1e-9)         # inverse
    assert np.allclose(warp_inv(warp(K_BINS / 20.0)), K_BINS / 20.0)
    # every K_BINS value is reachable: its u lies strictly inside (-1, 1)
    # except the two ceilings, which sit exactly at +-1
    ub = warp_inv(K_BINS)
    assert ub[0] == -1.0 and ub[-1] == 1.0
    assert np.all(np.abs(ub[1:-1]) < 1.0)
    # resolution: fine near zero, coarse at the ceiling
    d0 = (warp(0.01) - warp(0.0)) / 0.01
    d1 = (warp(1.0) - warp(0.99)) / 0.01
    assert 0.3 < d0 < 0.35 and d1 > 100


def test_torch_warp_matches_numpy():
    u = torch.linspace(-1, 1, 4001, dtype=torch.float64)
    assert torch.allclose(warp_t(u), torch.as_tensor(warp(u.numpy())),
                          atol=1e-12)
    z = torch.randn(64, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    got = view_from_z_t(z, 1.33).numpy()
    want = view_from_z(z.numpy(), 1.33)
    assert np.allclose(got, want, atol=1e-5)
    assert got.dtype == np.float32 and want.dtype == np.float32


# ==========================================================================
# (d): the mixed distribution
# ==========================================================================
def _hand_logp(padded, act, mu, log_std, z):
    lsm = F.log_softmax(padded, dim=-1)
    lp = 0.0
    for h in range(N_VIEW, NACT):
        lp = lp + lsm[torch.arange(len(act)), h, act[:, h]]
    sig = log_std.exp()
    for j in range(N_VIEW):
        lp = lp - 0.5 * ((z[:, j] - mu[:, j]) / sig[j]) ** 2 - log_std[j] \
            - 0.5 * LOG2PI
    return lp


def test_sample_view_logp_matches_a_hand_computation_and_the_update():
    torch.manual_seed(3)
    pk = HeadPacker(torch.device("cpu"))
    B = 257
    logits = torch.randn(B, sum(NVEC) + N_VIEW)
    cat, mu = split_view(logits)
    assert cat.shape == (B, sum(NVEC)) and mu.shape == (B, N_VIEW)
    padded = pk.pad(cat)
    log_std = torch.tensor([-1.2, -0.7])
    act, z, logp = sample_view(padded, mu, log_std)
    assert act.shape == (B, NACT) and z.shape == (B, N_VIEW)
    assert torch.all(act[:, 0] == NEUTRAL_ACT[0])
    assert torch.all(act[:, 1] == NEUTRAL_ACT[1])
    for h in range(N_VIEW, NACT):
        assert int(act[:, h].max()) < NVEC[h]
    want = _hand_logp(padded, act, mu, log_std, z)
    assert torch.allclose(logp, want, atol=1e-5)
    # the update's recomputation from the STORED z closes the ratio
    logp2, ent = logprob_entropy_view(padded, act, mu, log_std, z)
    assert torch.equal(logp2, logprob_entropy_padded(padded[:, N_VIEW:],
                                                     act[:, N_VIEW:])[0]
                       + gauss_logp(z, mu, log_std))
    assert torch.allclose(logp2, logp, atol=1e-5)
    _, ent_c = logprob_entropy_padded(padded[:, N_VIEW:], act[:, N_VIEW:])
    assert torch.allclose(ent, ent_c + gauss_entropy(log_std))
    # Gaussian entropy: 0.5 log(2 pi e sigma^2) per head
    want_h = sum(0.5 * np.log(2 * np.pi * np.e * float(s) ** 2)
                 for s in log_std.exp())
    assert float(gauss_entropy(log_std)) == pytest.approx(want_h, abs=1e-6)
    # the draw is z = mu + sigma eps: its moments over many rows
    z_big = sample_view(pk.pad(torch.zeros(20000, sum(NVEC))),
                        torch.zeros(20000, N_VIEW), log_std)[1]
    assert z_big.std(0).numpy() == pytest.approx(log_std.exp().numpy(), rel=0.05)


def _tiny(view=True, seed=0):
    torch.manual_seed(seed)
    return Policy(15 + 8 * 4, 8, 4, emb=16, hidden=16, view_continuous=view)


class _Core:
    """A stub core: enough for the eval wrappers (no lidar attached)."""
    class _Cfg:
        pitch_rate_max_deg = 1.33
    config = _Cfg()


def test_eval_wrappers_publish_a_view_and_greedy_is_the_mean():
    p = _tiny()
    p.eval()
    pk = HeadPacker(torch.device("cpu"))
    obs = np.random.default_rng(0).standard_normal((5, 15 + 32)).astype(np.float32)
    g = GreedyTorchPolicy(p, pk, torch.device("cpu"), None, _Core(), 4)
    a = g.act(obs)
    assert a.shape == (5, 6) and a.dtype == np.int32
    assert np.all(a[:, 0] == NEUTRAL_ACT[0]) and np.all(a[:, 1] == NEUTRAL_ACT[1])
    assert g.view is not None and g.view.shape == (5, 2) \
        and g.view.dtype == np.float32
    with torch.no_grad():
        lg, _ = p(torch.as_tensor(obs))
    _, mu = split_view(lg)
    want = view_from_z_t(mu, 1.33).numpy()
    assert np.allclose(g.view, want, atol=1e-6)
    # the act_every hold keeps the same view for K ticks
    v0 = g.view.copy()
    g.act(obs)
    assert np.array_equal(g.view, v0)
    s = SampledTorchPolicy(p, pk, torch.device("cpu"), None, _Core(), 1)
    s.act(obs)
    assert s.view.shape == (5, 2) and np.isfinite(s.view).all()
    assert np.all(np.abs(s.view[:, 0]) <= K_MAX) \
        and np.all(np.abs(s.view[:, 1]) <= 1.33 + 1e-6)
    # a discrete policy publishes none: record_rollout keeps core.step(acts)
    d = GreedyTorchPolicy(_tiny(view=False), pk, torch.device("cpu"), None,
                          _Core(), 1)
    d.act(obs)
    assert d.view is None


# ==========================================================================
# (e): the Policy
# ==========================================================================
def test_the_flag_adds_three_tensors_last_and_off_draws_no_rng():
    d = _tiny(view=False, seed=5)
    c = _tiny(view=True, seed=5)
    nd = [n for n, _ in d.named_parameters()]
    nc = [n for n, _ in c.named_parameters()]
    assert nc[:len(nd)] == nd
    assert nc[len(nd):] == ["view_head.weight", "view_head.bias",
                            "view_std.log_std"]
    for k, v in d.state_dict().items():
        assert torch.equal(v, c.state_dict()[k]), k
    assert set(c.state_dict()) - set(d.state_dict()) == {
        "view_head.weight", "view_head.bias", "view_std.log_std"}
    assert torch.allclose(c.view_std.log_std,
                          torch.full((2,), float(LOG_STD_INIT)))
    # the flag off is the pre-flag model: a second draw with the same seed
    # matches tensor for tensor (no RNG consumed by the absent module)
    d2 = _tiny(view=False, seed=5)
    for k, v in d.state_dict().items():
        assert torch.equal(v, d2.state_dict()[k])
    x = torch.randn(3, 15 + 32)
    ld, vd = d(x)
    lc, vc = c(x)
    assert ld.shape == (3, sum(NVEC)) and lc.shape == (3, sum(NVEC) + N_VIEW)
    assert torch.equal(ld, lc[:, :sum(NVEC)]) and torch.equal(vd, vc)
    with pytest.raises(SystemExit):
        Policy(15 + 8 * 4, 8, 4, emb=16, hidden=16, view_continuous=True,
               yaw_cond=True)
    with pytest.raises(SystemExit):
        Policy(15 + 8 * 4, 8, 4, emb=16, hidden=16, view_continuous=True,
               n_codes=4, chunk=2)


# ==========================================================================
# (f): the transplant's arithmetic
# ==========================================================================
def test_transplant_fit_and_optimizer_groups():
    from transplant_view import fit_ridge
    torch.manual_seed(0)
    T = torch.randn(4096, 16)
    W0 = torch.randn(2, 16)
    b0 = torch.tensor([0.3, -0.2])
    Z = T @ W0.T + b0
    W, b = fit_ridge(T, Z.double(), 1e-9)
    assert torch.allclose(W.float(), W0, atol=1e-4)
    assert torch.allclose(b.float(), b0, atol=1e-4)
    d = _tiny(view=False)
    c = _tiny(view=True)
    sd = {k: v.clone() for k, v in d.state_dict().items()}
    missing, unexpected = c.load_state_dict(sd, strict=False)
    assert set(missing) == {"view_head.weight", "view_head.bias",
                            "view_std.log_std"} and not unexpected
    for k in sd:
        assert torch.equal(c.state_dict()[k], sd[k])
    opt = torch.optim.Adam(d.parameters(), lr=1e-3)
    st = opt.state_dict()
    n_c = len(list(c.parameters()))
    have = list(st["param_groups"][0]["params"])
    st["param_groups"][0]["params"] = have + [i for i in range(n_c)
                                              if i not in set(have)]
    opt_c = torch.optim.Adam(c.parameters(), lr=1e-3)
    opt_c.load_state_dict(st)                    # the group sizes agree
    assert len(opt_c.param_groups[0]["params"]) == n_c


# ==========================================================================
# (g): BC targets
# ==========================================================================
def test_bin_to_view_and_z_round_trip():
    acts = np.array([[b, p, 1, 1, 0, 0] for b in range(15) for p in range(7)],
                    np.int64)
    v = bin_to_view(acts, True, 1.33)
    assert v.dtype == np.float32 and v.shape == (105, 2)
    assert np.array_equal(v[:, 0], K_BINS[acts[:, 0]])
    assert np.array_equal(v[:, 1], (PITCH_BINS[acts[:, 1]]
                                    * (np.float32(1.33) / np.float32(10))))
    v2 = bin_to_view(acts, False, 1.33)
    assert np.array_equal(v2[:, 0], (np.float32(2) * YAW_BINS)[acts[:, 0]])
    z = z_from_view(v, 1.33)
    back = view_from_z(z, 1.33)
    # inside the clip the round trip is exact; the ceilings map to
    # tanh(atanh(0.999)) = 0.999 of the ceiling
    inner = (np.abs(v[:, 0]) < K_MAX) & (np.abs(v[:, 1]) < 1.33 - 1e-6)
    assert np.allclose(back[inner], v[inner], atol=1e-4)
    assert np.allclose(np.abs(back[~inner, 0][np.abs(v[~inner, 0]) >= K_MAX]),
                       warp(U_CLIP), atol=1e-4)
    # a frozen gaze: z_pitch 0
    assert np.all(z_from_view(v, 0.0)[:, 1] == 0.0)


def test_bin_view_moments_onehot_is_the_bin_with_zero_spread():
    n = 15
    probs = np.zeros((n, 6, 15), np.float32)
    probs[np.arange(n), 0, np.arange(n)] = 1.0
    probs[:, 1, 3] = 1.0
    mu, sd = bin_view_moments(probs, True, 1.33)
    want = z_from_view(bin_to_view(np.stack([np.arange(n), np.full(n, 3),
                                             np.ones(n), np.ones(n),
                                             np.zeros(n), np.zeros(n)], 1),
                                   True, 1.33), 1.33)
    assert np.allclose(mu, want, atol=1e-5) and np.allclose(sd, 0.0)
    # a two-bin mixture has the spread of the two z's
    p2 = np.zeros((1, 6, 15), np.float32)
    p2[0, 0, 7] = 0.5
    p2[0, 0, 10] = 0.5
    p2[0, 1, 3] = 1.0
    mu2, sd2 = bin_view_moments(p2, True, 1.33)
    z7, z10 = want[7, 0], want[10, 0]
    assert mu2[0, 0] == pytest.approx((z7 + z10) / 2, abs=1e-5)
    assert sd2[0, 0] == pytest.approx(abs(z10 - z7) / 2, abs=1e-5)


def test_bcdataset_reads_and_derives_view_targets(tmp_path):
    from surfgym.bc import BCDataset, save_bc_dataset
    from surfgym.core import STATE_DTYPE
    n = 12
    states = np.zeros(n, STATE_DTYPE)
    scal = np.zeros((n, 15), np.float32)
    latch = np.zeros(n, np.float32)
    acts = np.zeros((n, 6), np.int64)
    acts[:, 0] = np.arange(n) % 15
    acts[:, 1] = np.arange(n) % 7
    acts[:, 2:4] = 1
    w = np.ones(n, np.float32)
    lid = np.zeros(n, np.int32)
    meta = {"obs_reward": False, "n_latch": 0, "act_every": 4}
    f1 = tmp_path / "discrete.npz"
    save_bc_dataset(f1, states, scal, latch, acts, w, lid, meta)
    dev = torch.device("cpu")
    bc = BCDataset(f1, dev, n_latch=0, obs_reward=False, view_continuous=True,
                   yaw_adaptive=True, pitch_rate_max_deg=1.33)
    assert "DERIVED from the bins" in bc.view_note
    want = z_from_view(bin_to_view(acts, True, 1.33), 1.33)
    assert np.allclose(bc.vz.numpy(), want, atol=1e-6)
    assert np.allclose(bc.vzmu.numpy(), want) and np.all(bc.vzsd.numpy() == 0)
    out = bc.sample_all(5, view=True)
    assert len(out) == 11 and out[8].shape == (5, 2)
    assert len(bc.sample_all(5)) == 8
    # a discrete trainer never builds them
    bd = BCDataset(f1, dev, n_latch=0, obs_reward=False)
    assert bd.vz is None
    with pytest.raises(RuntimeError):
        bd.sample_all(3, view=True)
    # a continuous file: the columns are read, not derived
    view = np.random.default_rng(1).uniform(-1, 1, (n, 2)).astype(np.float32)
    zmu = np.random.default_rng(2).standard_normal((n, 2)).astype(np.float32)
    zsd = np.abs(np.random.default_rng(3).standard_normal((n, 2))).astype(np.float32)
    f2 = tmp_path / "cont.npz"
    save_bc_dataset(f2, states, scal, latch, acts, w, lid,
                    dict(meta, view_continuous=1), view=view, view_zmu=zmu,
                    view_zsd=zsd)
    bc2 = BCDataset(f2, dev, n_latch=0, obs_reward=False, view_continuous=True,
                    yaw_adaptive=True, pitch_rate_max_deg=1.33)
    assert "from the file" in bc2.view_note and "moments from the file" in bc2.view_note
    assert np.allclose(bc2.vz.numpy(), z_from_view(view, 1.33), atol=1e-6)
    assert np.array_equal(bc2.vzmu.numpy(), zmu)
    assert np.array_equal(bc2.vzsd.numpy(), zsd)
    with pytest.raises(ValueError):
        save_bc_dataset(tmp_path / "bad.npz", states, scal, latch, acts, w,
                        lid, meta, view_zmu=zmu, view_zsd=zsd)


# ==========================================================================
# (h): a planner line round trip through replay_line
# ==========================================================================
@needs_core
def test_continuous_line_round_trip_replays_to_the_same_tick(tmp_path):
    from surfgym import SurfCore, default_config
    from surfgym.bc import replay_line
    cfg = default_config(num_envs=1, spawn_mode=0, max_episode_ticks=900,
                         water_fail=1, yaw_adaptive=1, lidar_w=0, lidar_h=0,
                         pitch_rate_max_deg=1.33)
    core = SurfCore(str(SKI), cfg, dll_path=str(DLL))
    k = 4
    obs0 = core.reset(11).copy()
    spawn = core.get_states()[0].copy()
    rng = np.random.default_rng(5)
    D = 120
    acts = np.zeros((D, 6), np.int64)
    acts[:, 0], acts[:, 1] = NEUTRAL_ACT[0], NEUTRAL_ACT[1]
    acts[:, 2] = 2
    acts[:, 3] = rng.integers(0, 3, D)
    view = np.stack([rng.uniform(-3, 3, D), rng.uniform(-1.33, 1.33, D)],
                    1).astype(np.float32)
    # the line: step it once, remember where it ends
    core.set_state(0, spawn)
    end = None
    for d in range(D):
        a = np.ascontiguousarray(acts[d].reshape(1, 6), np.int32)
        v = np.ascontiguousarray(view[d].reshape(1, 2), np.float32)
        for _ in range(k):
            _o, _r, done, trunc, _t = core.step(a, view=v)
            if done[0] or trunc[0]:
                end = d
                break
        if end is not None:
            break
    st_end = core.get_states()[0].copy()
    npz = tmp_path / "beam_best.npz"
    np.savez(npz, acts=acts.astype(np.int8), view=view, act_every=np.int32(k),
             spawn_state=spawn.reshape(1), obs_start=obs0[0],
             view_continuous=np.int32(1))
    z = np.load(npz, allow_pickle=False)
    assert int(z["view_continuous"]) == 1 and z["view"].dtype == np.float32
    views_out = []
    rows, ticks_states, finished, ticks = replay_line(
        core, z["spawn_state"][0], z["obs_start"], z["acts"], k,
        view=z["view"], views_out=views_out)
    assert len(rows) == len(views_out) and np.allclose(views_out[0], view[0])
    if end is None:
        assert ticks == D * k and len(rows) == D
        assert core.get_states()[0].tobytes() == st_end.tobytes()
    else:
        assert len(rows) == end + 1
    # the discrete call is untouched: no view, four-tuple rows
    r2, *_ = replay_line(core, z["spawn_state"][0], z["obs_start"], z["acts"], k)
    assert len(r2[0]) == 4


# ==========================================================================
# (i): the trainer with the flag OFF is the pre-flag trainer, bit for bit
# ==========================================================================
import json                                                    # noqa: E402
import shutil                                                  # noqa: E402
import subprocess                                              # noqa: E402

MAPS = Path(os.environ.get("SURF_TEST_MAPS") or (ROOT / "maps"))
CANNONBALL = MAPS / "surf_src_cannonball.bsp"
GOALFIELD = MAPS / "surf_src_cannonball.goal_32.npz"
needs_run = pytest.mark.skipif(
    not (CANNONBALL.exists() and DLL.exists() and GOALFIELD.exists()),
    reason="needs the built core + cannonball + its prebaked goal field "
           "(SURF_TEST_MAPS / SURFCORE_DLL from a worktree)")

SMOKE_FLAGS = ["--map", str(CANNONBALL), "--reward", "race", "--envs", "64",
               "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
               "--lidar-cell", "32", "--lidar-range", "11500",
               "--lidar-near", "2000", "--emb", "64", "--hidden", "64",
               "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
               "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95",
               "--clip", "0.2", "--vf", "0.5", "--ent", "0.005",
               "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
               "--ep-ticks", "3000", "--time-pen", "0.005",
               "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
               "--race-dist", "geodesic", "--maxvel", "4000",
               "--train-stride", "1", "--yaw-adaptive",
               "--respawn-frac", "0.9", "--respawn-margin", "10",
               "--respawn-reservoir", "1000", "--int-coef", "0.25",
               "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9",
               "--record-every", "1e12", "--eval-eps", "1",
               "--eval-greedy-only", "--seed", "7"]


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _train(run, extra, script="train_fast.py", steps="6144", timeout=1800):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(ROOT / "python" / script),
           "--run", run] + SMOKE_FLAGS + ["--steps", steps] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       cwd=str(ROOT), timeout=timeout,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


def _preflag_candidates():
    try:
        r = subprocess.run(["git", "rev-list", "--first-parent",
                            "--max-count=200", "HEAD"],
                           capture_output=True, text=True, cwd=str(ROOT),
                           timeout=60)
        refs = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        refs = []
    return tuple(refs[1:]) or ("HEAD^",)


def _unpatched_trainer(dst: Path):
    """The pre-flag train_fast.py out of git, or None. BYTES: the console
    is cp1251 and the file is UTF-8."""
    for ref in _preflag_candidates():
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode == 0 and b"--view-continuous" not in r.stdout:
            dst.write_bytes(r.stdout)
            return ref
    return None


@needs_run
def test_no_flag_is_bit_identical_to_the_unpatched_trainer():
    """The control run is the run that shipped: same config dump, same
    progress.csv (fps excluded), same eval trajectory, same weights, same
    Adam moments. The pre-flag copy lives in python/ so its ROOT and every
    import resolve exactly as the patched file's do; the surfgym package
    underneath is the SAME (its view=None path is byte-identical to the
    pre-change core, test (a) above)."""
    old = ROOT / "python" / "train_fast_preview.py"
    ref = _unpatched_trainer(old)
    if ref is None:
        pytest.skip("no pre-flag train_fast.py in the first-parent history")
    try:
        _train("cy_view_ctl_new", [])
        _train("cy_view_ctl_old", [], script="train_fast_preview.py")
    finally:
        old.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "cy_view_ctl_new", ROOT / "runs" / "cy_view_ctl_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert "view_continuous" not in ca
    assert ca == cb
    ra, rb = _csv("cy_view_ctl_new"), _csv("cy_view_ctl_old")
    assert len(ra) == len(rb) == 3
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta = sorted(a.glob("traj_*.jsonl"))
    tb = sorted(b.glob("traj_*.jsonl"))
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
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)


@needs_run
def test_the_flag_trains_and_the_eval_publishes_a_view():
    """A tiny CPU scratch run with the flag: finite losses, the config
    carries the key, the checkpoint the three tensors, the greedy eval ran
    through record_rollout's view path (a trajectory exists and its rows
    are finite), sigma stays where it started to within a few percent."""
    r = _train("cy_view_flag", ["--view-continuous"])
    assert "--view-continuous: yaw and pitch are squashed Gaussians" in r.stdout
    assert "sig 0.3" in r.stdout
    d = ROOT / "runs" / "cy_view_flag"
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["view_continuous"] == 1
    rows = _csv("cy_view_flag")
    for x in rows:
        for k in ("train/loss", "train/value_loss", "train/approx_kl"):
            assert np.isfinite(float(x[k])), k
    sd = torch.load(d / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)["policy"]
    assert {"view_head.weight", "view_head.bias", "view_std.log_std"} <= set(sd)
    assert torch.all((sd["view_std.log_std"].exp() - 0.3).abs() < 0.03)
    tr = sorted(d.glob("traj_*.jsonl"))
    assert tr and len(tr[0].read_text(encoding="utf-8").splitlines()) > 100
    # the flag is REFUSED with the combinations that read the yaw bin
    cmd = [sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
           "--run", "cy_view_bad"] + SMOKE_FLAGS + \
        ["--steps", "2048", "--view-continuous", "--yaw-cond"]
    rr = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                        cwd=str(ROOT), timeout=600, encoding="utf-8",
                        errors="replace")
    assert rr.returncode != 0 and "--yaw-cond" in (rr.stdout + rr.stderr)
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(ROOT / "runs" / "cy_view_bad", ignore_errors=True)
