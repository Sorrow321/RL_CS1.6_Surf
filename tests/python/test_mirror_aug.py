"""``--aug-mirror`` / ``--eval-mirror`` / ``record_ckpt --mirror`` (surfgym/mirror.py) and
the DrAC update augmentation ``--aug-drop`` / ``--aug-noise`` / ``--aug-shift``
(surfgym/augment.py).

What could be silently wrong, and would make a whole arm unreadable:

  * the reflection of a row is not an involution, or flips the wrong scalar columns or
    image channels;
  * the engine-row reflection swaps the wrong side bin or leaves the yaw alone;
  * THE PHYSICS CLAIM the mirror rests on: from one state, the reflected commands fly the
    reflected trajectory - checked here on the real C core (cannonball, open air);
  * the eval wrapper does not reflect what it reads or what it emits;
  * the shift is not an edge-replicated translation, the drop splits a pixel's channels
    or zeroes the wrong share, the noise has the wrong scale;
  * DrAC's terms are not zero on identical outputs, or leak gradient into their target
    (the clean frame's policy / value, and the view's log sigma).

    python -m pytest tests/python/test_mirror_aug.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.augment import augment_image, drac_terms        # noqa: E402
from surfgym.core import STATE_DTYPE                          # noqa: E402
from surfgym.mirror import (H_SIDE, SCAL_NEG, SCAL_NEG_GPS,   # noqa: E402
                            Mirror, layout_for, mirror_actions_)

MAPS = ROOT / "maps"


# ==========================================================================
# (a) the row reflection
# ==========================================================================
def _rows(b=4, ns=20, h=4, w=6, c=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, ns + h * w * c, generator=g)


def test_row_reflection_is_an_involution_and_reflects_exactly_what_it_says():
    h, w, c, ns = 4, 6, 5, 20
    m = Mirror(h, w, c, neg_cols=(1, 10, 17), neg_ch=(3,))
    x = _rows(ns=ns, h=h, w=w, c=c)
    y = m.rows(x)
    assert torch.equal(m.rows(y), x)                          # exactly, not approximately
    img, imy = (x[:, ns:].reshape(-1, h, w, c), y[:, ns:].reshape(-1, h, w, c))
    for j in range(w):
        for ch in range(c):
            s = -1.0 if ch == 3 else 1.0
            assert torch.equal(imy[:, :, j, ch], s * img[:, :, w - 1 - j, ch])
    for col in range(ns):
        s = -1.0 if col in (1, 10, 17) else 1.0
        assert torch.equal(y[:, col], s * x[:, col])


def test_masked_reflection_touches_only_the_masked_rows_in_place():
    m = Mirror(4, 6, 5, neg_cols=SCAL_NEG, neg_ch=(3,))
    x = _rows()
    ref = m.rows(x)
    mask = torch.tensor([True, False, True, False])
    z = x.clone()
    out = m.obs_(z, mask)
    assert out is z
    assert torch.equal(z[mask], ref[mask]) and torch.equal(z[~mask], x[~mask])
    z2 = x.clone()
    m.obs_(z2)
    assert torch.equal(z2, ref)


def test_layout_reads_the_normal_channel_the_gps_columns_and_the_arrow():
    lid = SimpleNamespace(H=32, W=64, channels=5, views=1, normals=False,
                          texmap=object(), texture_mode="normal", tex_channel=2)
    m = layout_for(lid)
    # depth, potential, (nx fwd, ny LEFT, nz up): the left component is channel 3
    assert m.neg_ch == (3,) and m.neg_cols == SCAL_NEG and m.frame == 32 * 64 * 5
    assert layout_for(lid, gps=True).neg_cols == tuple(sorted(SCAL_NEG + SCAL_NEG_GPS))
    assert 23 in layout_for(lid, rvec0=22).neg_cols          # (forward, LEFT, up, length)
    plain = SimpleNamespace(H=32, W=64, channels=2, views=1, normals=False, texmap=None)
    assert layout_for(plain).neg_ch == ()
    with pytest.raises(ValueError):
        layout_for(SimpleNamespace(H=32, W=64, channels=5, views=4))
    with pytest.raises(ValueError):
        Mirror(4, 6, 2, neg_ch=(2,))


# ==========================================================================
# (b) the engine row
# ==========================================================================
def test_engine_row_swaps_the_side_key_and_negates_the_yaw_only():
    act = np.array([[7, 3, 1, 0, 0, 0], [7, 3, 2, 1, 1, 0], [7, 3, 0, 2, 0, 1]], np.int32)
    view = np.array([[10.0, -20.0], [-3.5, 5.0], [0.0, 30.0]], np.float32)
    a, v = act.copy(), view.copy()
    mirror_actions_(a, v)
    assert H_SIDE == 3
    assert a[:, H_SIDE].tolist() == [2, 1, 0]
    assert np.array_equal(np.delete(a, H_SIDE, 1), np.delete(act, H_SIDE, 1))
    assert np.array_equal(v[:, 0], -view[:, 0]) and np.array_equal(v[:, 1], view[:, 1])
    mirror_actions_(a, v)
    assert np.array_equal(a, act) and np.array_equal(v, view)
    b, w = act.copy(), view.copy()
    mirror_actions_(b, w, np.array([False, False, True]))
    assert np.array_equal(b[:2], act[:2]) and b[2, H_SIDE] == 0 and w[2, 0] == -0.0
    mirror_actions_(b, None, np.zeros(3, bool))              # an all-False mask is a no-op
    assert b[2, H_SIDE] == 0


# ==========================================================================
# (c) the physics claim, on the real core
# ==========================================================================
@pytest.fixture(scope="module")
def core():
    bsp = MAPS / "surf_src_cannonball.bsp"
    if not bsp.exists():
        pytest.skip("cannonball bsp not present")
    from surfgym import SurfCore, default_config
    c = SurfCore(str(bsp), default_config(num_envs=2, lidar_w=0, lidar_h=0, view_mode=1,
                                          lookahead_k=0, sv_maxvelocity=4000.0,
                                          max_episode_ticks=100000))
    c.reset(0)
    return c


def _open_air(core, clear=900.0):
    """A point whose hull can travel `clear` units in 26 directions without touching
    anything: the flight below stays in free air, where the claim is pure physics."""
    mins, maxs = core.map_bounds()
    rng = np.random.default_rng(3)
    dirs = [np.array(d, np.float64) - 1.0 for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)]
    dirs = [d / np.linalg.norm(d) for d in dirs]
    for _ in range(20000):
        p = mins + (maxs - mins) * rng.random(3)
        if core.point_contents(p) != -1:
            continue
        if all(core.trace(p, p + clear * d, 0).fraction >= 1.0 for d in dirs):
            return p
    pytest.skip("no open-air point found")


def test_reflected_commands_fly_the_reflected_trajectory(core):
    p = _open_air(core)
    yaw = 37.0
    psi = math.radians(yaw)
    st = core.get_states()[0].copy()
    st["origin"] = p
    st["velocity"] = (900.0 * math.cos(psi), 900.0 * math.sin(psi), 120.0)
    st["basevelocity"] = (0.0, 0.0, 0.0)
    st["yaw"] = yaw
    st["pitch"] = -10.0
    st["onground"] = -1
    st["oldbuttons"] = 0
    st["ducked"] = 0
    st["stuck_ticks"] = 0
    core.set_state(0, st)
    core.set_state(1, st)
    # the reflection plane: vertical, through p, along the shared heading
    d = np.array([math.cos(psi), math.sin(psi), 0.0])
    n = np.array([-math.sin(psi), math.cos(psi), 0.0])

    def refl(v):
        return v - 2.0 * np.dot(v, n) * n

    rng = np.random.default_rng(11)
    for t in range(80):
        act = np.zeros((2, 6), np.int32)
        act[:, 0], act[:, 1] = 7, 3
        view = np.zeros((2, 2), np.float32)
        if t < 40:
            # random commands: none / forward (S would brake the flight to a crawl),
            # A / none / D, any view target
            act[0, 2] = rng.integers(1, 3)
            act[0, 3] = rng.integers(0, 3)
            view[0] = (rng.uniform(-60.0, 60.0), rng.uniform(-40.0, 20.0))
        else:
            # then a held air strafe: D with the view 15 deg right of the velocity -
            # a sustained turn, so the two flights part company for real
            act[0, 2], act[0, 3] = 1, 2
            view[0] = (-15.0, -10.0)
        act[1] = act[0]
        view[1] = view[0]
        mirror_actions_(act[1:], view[1:])
        obs, _, done, trunc, _ = core.step(act, view=view)
        assert not done.any() and not trunc.any(), f"an episode ended at tick {t}"
        s = core.get_states()
        assert s["onground"][0] == -1 and s["onground"][1] == -1, "touched something"
        o0, o1 = obs[0], obs[1]
        for k in (0, 2, 3, 9, 11):                   # forward, vertical, speed, pitch, its rate
            assert o1[k] == pytest.approx(o0[k], abs=1e-4), (t, k)
        for k in SCAL_NEG:                           # lateral velocity, yaw rate
            assert o1[k] == pytest.approx(-o0[k], abs=1e-4), (t, k)
        dp0 = s["origin"][0].astype(np.float64) - p
        dp1 = s["origin"][1].astype(np.float64) - p
        assert np.allclose(dp1, refl(dp0), atol=2e-2), (t, dp0, dp1)
        assert np.allclose(s["velocity"][1], refl(s["velocity"][0].astype(np.float64)),
                           atol=2e-2)
    # and the flight was not trivial: it went somewhere, and it turned off the plane
    # (the two envs end 2x that far apart)
    assert np.linalg.norm(dp0) > 300.0 and abs(np.dot(dp0, n)) > 50.0, dp0


# ==========================================================================
# (d) the eval wrapper
# ==========================================================================
RATE = 1.33


class _EvalCore:
    class _Cfg:
        pitch_rate_max_deg = RATE
    config = _Cfg()

    def __init__(self, n):
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)


class _Lidar:
    """An 8x4x2 'image' that is different in every column, channel and env."""
    H, W, channels, views, normals, texmap = 4, 8, 2, 1, False, None

    def render(self, o, yw, pt, dk):
        n = o.shape[0]
        base = torch.arange(4 * 8 * 2, dtype=torch.float32).reshape(1, -1)
        return (base + 100.0 * torch.arange(n, dtype=torch.float32).reshape(-1, 1)) / 64.0


def test_eval_wrapper_reflects_the_row_it_reads_and_the_row_it_emits():
    import train_fast
    from train_fast import GreedyTorchPolicy, HeadPacker, Policy
    torch.manual_seed(0)
    p = Policy(15 + 64, 8, 4, emb=16, hidden=16, in_ch=2, view_continuous=True,
               view_absolute="velocity")
    p.eval()
    pk = HeadPacker("cpu")
    obs = np.random.default_rng(0).standard_normal((5, 15)).astype(np.float32)
    lid = _Lidar()
    m = layout_for(lid)
    g0 = GreedyTorchPolicy(p, pk, "cpu", lid, _EvalCore(5), 4)
    gm = GreedyTorchPolicy(p, pk, "cpu", lid, _EvalCore(5), 4, mirror=m)
    r0, rm = g0._obs(obs), gm._obs(obs)
    assert torch.equal(rm, m.rows(r0))
    # what the mirrored wrapper emits = the reflection of what the plain policy emits
    # on the reflected row: same weights, greedy, so the draw is deterministic
    a_m = gm.act(obs).copy()
    v_m = gm.view.copy()
    logits, _ = p(m.rows(r0))
    cat, mu = train_fast.split_view(logits.float())
    a_ref = pk.pad(cat)[:, train_fast.N_VIEW:].argmax(-1).numpy()
    side_ref = a_ref[:, H_SIDE - train_fast.N_VIEW]
    assert np.array_equal(a_m[:, H_SIDE], 2 - side_ref)
    yaw_ref = train_fast.view_from_z_t(mu, RATE, "velocity")[:, 0].detach().numpy()
    assert np.allclose(v_m[:, 0], -yaw_ref, atol=1e-5)
    with pytest.raises(ValueError):
        GreedyTorchPolicy(p, pk, "cpu", lid, _EvalCore(5), 4, mirror=m,
                          route=object())
    pd = Policy(15 + 64, 8, 4, emb=16, hidden=16, in_ch=2)
    with pytest.raises(ValueError):
        GreedyTorchPolicy(pd, pk, "cpu", lid, _EvalCore(5), 4, mirror=m)


# ==========================================================================
# (e) the augmentations
# ==========================================================================
def test_shift_is_an_edge_replicated_translation_within_k():
    h, w, c, k = 6, 10, 3, 2
    r = torch.arange(h).view(1, h, 1, 1).expand(1, h, w, c)
    q = torch.arange(w).view(1, 1, w, 1).expand(1, h, w, c)
    ch = torch.arange(c).view(1, 1, 1, c).expand(1, h, w, c)
    code = (r * 1000 + q * 10 + ch).float().reshape(1, -1).repeat(200, 1)
    torch.manual_seed(5)
    out = augment_image(code, h, w, c, shift=k).reshape(200, h, w, c)
    seen = set()
    for b in range(200):
        o = out[b]
        dy = int(o[h // 2, w // 2, 0].item()) // 1000 - h // 2
        dx = (int(o[h // 2, w // 2, 0].item()) % 1000) // 10 - w // 2
        assert -k <= dy <= k and -k <= dx <= k
        seen.add((dy, dx))
        ri = torch.arange(h).add(dy).clamp(0, h - 1)
        ci = torch.arange(w).add(dx).clamp(0, w - 1)
        want = (ri.view(h, 1, 1) * 1000 + ci.view(1, w, 1) * 10
                + torch.arange(c).view(1, 1, c)).float()
        assert torch.equal(o, want)
    assert len(seen) == (2 * k + 1) ** 2                      # every offset is reachable


def test_drop_zeroes_whole_pixels_at_the_asked_share_and_noise_has_its_scale():
    h, w, c = 32, 64, 5
    x = torch.ones(64, h * w * c)
    torch.manual_seed(1)
    y = augment_image(x, h, w, c, drop=0.25).reshape(64, h, w, c)
    zero = (y == 0)
    assert torch.equal(zero.all(-1), zero.any(-1))            # a pixel goes as a whole
    assert abs(zero[..., 0].float().mean().item() - 0.25) < 0.01
    assert torch.equal(y[~zero], torch.ones_like(y[~zero]))
    z = augment_image(torch.zeros(64, h * w * c), h, w, c, noise=0.05)
    assert abs(z.mean().item()) < 2e-3 and abs(z.std().item() - 0.05) < 2e-3
    xb = torch.ones(4, h * w * c, dtype=torch.bfloat16)
    assert augment_image(xb, h, w, c, drop=0.3, noise=0.1, shift=3).dtype == torch.bfloat16
    assert torch.equal(augment_image(x, h, w, c), x)          # nothing asked, nothing done


def test_the_update_branch_runs_on_a_real_policy_and_reaches_only_the_weights():
    """mb_step's DrAC branch, expression for expression, on a small continuous-view
    policy (the trainer's own Policy, HeadPacker and split_view): finite terms, a
    gradient on the network, none on the stored frame."""
    import train_fast
    from train_fast import HeadPacker, Policy, split_view
    torch.manual_seed(0)
    h, w, c = 4, 8, 5
    p = Policy(15 + h * w * c, w, h, emb=16, hidden=16, in_ch=c, view_continuous=True,
               view_absolute="velocity", simba=True)
    pk = HeadPacker("cpu")
    scal = torch.randn(32, 15)
    img = torch.rand(32, h * w * c).to(torch.bfloat16)     # the update's bf16 frame buffer
    with torch.autocast("cpu", dtype=torch.bfloat16):      # the trainer's `amp`
        logits, value = p.forward_split(scal, img)
        logits_a, value_a = p.forward_split(
            scal, augment_image(img, h, w, c, 0.25, 0.05, 2))
        _cs, _ms = split_view(logits.float())
        _ca, _ma = split_view(logits_a.float())
        kl, vm = drac_terms(pk.pad(_cs)[:, train_fast.N_VIEW:],
                            pk.pad(_ca)[:, train_fast.N_VIEW:],
                            value, value_a, _ms, _ma, p.log_std())
    assert torch.isfinite(kl) and torch.isfinite(vm) and kl.item() >= 0.0
    (0.1 * (kl + vm)).backward()
    g = [q.grad for q in p.parameters() if q.grad is not None]
    assert g and any(float(x.abs().sum()) > 0.0 for x in g)
    assert not img.requires_grad


def test_drac_terms_are_zero_on_identical_outputs_and_train_only_the_augmented_side():
    g = torch.Generator().manual_seed(2)
    neg = -1e30
    ps = torch.randn(6, 4, 5, generator=g)
    ps[:, 2, 3:] = neg                                         # a padded head
    pa = ps.clone()
    vs = torch.randn(6, generator=g)
    kl, vm = drac_terms(ps, pa, vs, vs.clone())
    assert kl.item() == pytest.approx(0.0, abs=1e-6) and vm.item() == 0.0
    ps.requires_grad_(True)
    vs.requires_grad_(True)
    pa2 = (pa + 0.3 * torch.randn(pa.shape, generator=g)).requires_grad_(True)
    va = (vs.detach() + 0.5).requires_grad_(True)
    mus = torch.randn(6, 2, generator=g).requires_grad_(True)
    mua = (mus.detach() + 0.2).requires_grad_(True)
    log_std = torch.tensor([math.log(0.4), math.log(0.5)], requires_grad=True)
    kl, vm = drac_terms(ps, pa2, vs, va, mus, mua, log_std)
    assert torch.isfinite(kl) and kl.item() > 0.0
    assert vm.item() == pytest.approx(0.25, rel=1e-5)
    (kl + vm).backward()
    assert ps.grad is None and vs.grad is None and mus.grad is None
    assert log_std.grad is None                               # sigma is not the regulariser's
    assert pa2.grad is not None and va.grad is not None and mua.grad is not None
    # the Gaussian part is exactly (mu_s - mu_a)^2 / (2 sigma^2)
    k0, _ = drac_terms(ps, ps.detach(), vs, vs.detach(), mus, mua, log_std)
    want = (0.2 ** 2 / (2 * 0.4 ** 2) + 0.2 ** 2 / (2 * 0.5 ** 2))
    assert k0.item() == pytest.approx(want, rel=1e-4)
