"""--goal-planner ramps (surfgym/goalramps.py + surfgym/rampvocab.py, 2026-09-28): the ramp-window
task's pieces, on synthetic data (no map files, no GPU).

1. The warm start: widen_for_target_channel grows a checkpoint's first conv by ONE zero input
   slice, so the resumed policy computes the checkpoint's own logits and value WHATEVER the new
   channel holds (Codex 23:16Z: identity for logits, values, greedy actions); the slice then
   receives gradient; the Adam moments are padded; anything but one added channel is refused.
2. reset_critic_tower replaces the value tower only.
3. MultiArcProgress.set_lines(keep_bank=True) keeps the episode's death-bond bank.
4. RampVocab: a touch on a validated contact plane is its surface, off the plane / with another
   normal / from the other hull it is not; a vocabulary is refused on another map signature.
5. RampWindows: inside T1's piece BOX enters it, leaving the box after that passes it (the window
   shifts T2 -> T1), a hop inside the box is still the ride, inside T2's box before T1 was entered
   skips T1; a wedge (sides + end cap) is ONE piece that enters the window as a side (never the
   cap) and whose box covers all of it; a standing spawn rides its first target down the
   potential; the channel's values move continuously per surface (at most 1 / fade_ticks per tick,
   a second shift inside a fade included) and rebuild exactly from the recorded events; a ramp
   contact outside the pieces the env may ride is off-target (going back included).
6. TargetLidar: one channel more than the wrapped lidar; mode "off" holds it at zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (N_SCALAR, Policy, reset_critic_tower,     # noqa: E402
                        widen_for_target_channel)
from surfgym.goalarc import MultiArcProgress                      # noqa: E402
from surfgym import goalramps as gr                               # noqa: E402
from surfgym.rampvocab import RampVocab, bsp_signature            # noqa: E402

W, H = 8, 4


def _pol(in_ch, seed):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + W * H * in_ch, W, H, emb=32, hidden=24, in_ch=in_ch)


def _ck(policy):
    params = list(policy.parameters())
    st = {}
    for i, (name, p) in enumerate(policy.named_parameters()):
        st[i] = {"exp_avg": torch.randn(*p.shape) + 1.0, "exp_avg_sq": torch.rand(*p.shape) + 1.0,
                 "step": torch.tensor(7.0)}
    return {"policy": {k: v.clone() for k, v in policy.state_dict().items()},
            "optimizer": {"state": st, "param_groups": [{"params": list(range(len(params)))}]}}


def _rows(n, in_ch, target=None, seed=0):
    """scalars + a channel-fastest (H, W, in_ch) image: channel 0 the same depth every time, the
    target channel (in_ch 2) = `target` ((n, H*W) or None -> zeros)"""
    g = torch.Generator().manual_seed(seed)
    scal = torch.randn(n, N_SCALAR, generator=g)
    depth = torch.rand(n, H * W, generator=g)
    if in_ch == 1:
        return torch.cat([scal, depth], 1)
    t = torch.zeros(n, H * W) if target is None else target
    img = torch.stack([depth, t], dim=-1).reshape(n, -1)
    return torch.cat([scal, img], 1)


def test_the_widened_policy_is_the_checkpoints_function_whatever_the_channel_holds():
    plain = _pol(1, 0).eval()
    ck = _ck(plain)
    wide = _pol(2, 99)
    assert widen_for_target_channel(ck, wide) == 3          # the weight + both moments
    wide.load_state_dict(ck["policy"])                      # strict
    wide.eval()
    assert float(wide.state_dict()["conv.0.weight"][:, 1].abs().max()) == 0.0
    la, va = plain(_rows(8, 1, seed=3))
    for fill in (None, torch.rand(8, H * W), torch.full((8, H * W), 1.0)):
        lb, vb = wide(_rows(8, 2, target=fill, seed=3))
        assert torch.allclose(la, lb, atol=1e-6, rtol=0), (la - lb).abs().max().item()
        assert torch.allclose(va, vb, atol=1e-6, rtol=0), (va - vb).abs().max().item()
        assert torch.equal(la.argmax(-1), lb.argmax(-1))    # the greedy action


def test_the_new_slice_receives_gradient_and_the_moments_are_padded():
    plain = _pol(1, 0)
    ck = _ck(plain)
    old_m = ck["optimizer"]["state"][0]["exp_avg"].clone()
    wide = _pol(2, 5)
    widen_for_target_channel(ck, wide)
    m = ck["optimizer"]["state"][0]["exp_avg"]
    assert m.shape[1] == 2 and torch.equal(m[:, :1], old_m) and float(m[:, 1].abs().max()) == 0.0
    wide.load_state_dict(ck["policy"])
    logits, v = wide(_rows(4, 2, target=torch.rand(4, H * W), seed=1))
    (logits.sum() + v.sum()).backward()
    g = wide.conv[0].weight.grad
    assert float(g[:, 1].abs().max()) > 0.0


def test_added_target_channels_widen_as_the_checkpoints_function_and_fewer_are_refused():
    """--target-views 6 grows a one-view checkpoint (depth + target) by five more zero input
    slices: the widened policy computes the checkpoint's logits and value whatever the new
    channels hold; a run with FEWER image channels than its checkpoint is refused"""
    assert widen_for_target_channel(_ck(_pol(2, 1)), _pol(2, 2)) == 0     # already has them
    plain = _pol(2, 0).eval()
    ck = _ck(plain)
    wide = _pol(7, 5)
    assert widen_for_target_channel(ck, wide) == 3          # the weight + both moments
    wide.load_state_dict(ck["policy"])
    wide.eval()
    assert float(wide.state_dict()["conv.0.weight"][:, 2:].abs().max()) == 0.0
    g = torch.Generator().manual_seed(8)
    scal = torch.randn(6, N_SCALAR, generator=g)
    base = torch.rand(6, H * W, 2, generator=g)
    la, va = plain(torch.cat([scal, base.reshape(6, -1)], 1))
    for extra in (torch.zeros(6, H * W, 5), torch.rand(6, H * W, 5, generator=g)):
        lb, vb = wide(torch.cat([scal, torch.cat([base, extra], -1).reshape(6, -1)], 1))
        assert torch.allclose(la, lb, atol=1e-6, rtol=0) and torch.allclose(va, vb, atol=1e-6, rtol=0)
    with pytest.raises(SystemExit):
        widen_for_target_channel(_ck(_pol(3, 1)), _pol(2, 2))


def test_reset_critic_replaces_the_value_tower_only():
    old = _pol(1, 0)
    ck = _ck(old)
    fresh = _pol(1, 42)
    n = reset_critic_tower(ck, fresh)
    assert n == 6                                  # vf.0 w/b, vf.2 w/b, value_head w/b
    sd, fs, os_ = ck["policy"], fresh.state_dict(), old.state_dict()
    for k in sd:
        if k.startswith("vf.") or k.startswith("value_head."):
            assert torch.equal(sd[k], fs[k])
        else:
            assert torch.equal(sd[k], os_[k])


def test_the_arc_bank_survives_a_window_shift():
    arc = MultiArcProgress(2, l_max=16, spacing=128.0)
    line = np.stack([np.arange(8) * 128.0, np.zeros(8), np.zeros(8)], 1).astype(np.float32)
    arc.set_lines(np.array([0, 1]), [line, line])
    arc.bank[:] = 123.0
    arc.set_lines(np.array([0]), [line], keep_bank=True)
    arc.set_lines(np.array([1]), [line])
    assert arc.bank[0] == 123.0 and arc.bank[1] == 0.0


# ---------------------------------------------------------------- a synthetic vocabulary
def _vocab(tmp_path, bsp):
    """three target ramps: planes y = 0 (x 0..2000), y = 3000 and y = 6000 (the same extent),
    their outward normal -y; validated origins 20 u out (standing) and 12 u out (ducked)"""
    rng = np.random.default_rng(0)
    pts, nrm, sid, dpts, dsid = [], [], [], [], []
    for s, y in enumerate((0.0, 3000.0, 6000.0)):
        p = np.stack([rng.uniform(0, 2000, 300), np.full(300, y), rng.uniform(0, 500, 300)], 1)
        n = np.tile([0.0, -1.0, 0.0], (300, 1))
        pts.append(p + n * 20.0)
        dpts.append(p + n * 12.0)
        nrm.append(n)
        sid.append(np.full(300, s))
        dsid.append(np.full(300, s))
    f = tmp_path / "vocab.npz"
    np.savez(f, version=4, map=Path(bsp).stem, bsp_sig=bsp_signature(bsp), mesh_sha1="x",
             cat=np.array([1, 1, 1]), normal=np.tile([0.0, -1.0, 0.0], (3, 1)),
             area=np.full(3, 1e6), touchable=np.ones(3),
             points=np.concatenate(pts), normals=np.concatenate(nrm),
             surf=np.concatenate(sid), duck_points=np.concatenate(dpts),
             duck_normals=np.concatenate(nrm), duck_surf=np.concatenate(dsid))
    return f


@pytest.fixture
def voc(tmp_path):
    bsp = tmp_path / "surf_fake.bsp"
    bsp.write_bytes(b"x" * 64)
    return RampVocab(_vocab(tmp_path, bsp), bsp), bsp, tmp_path


def _touch(o, n):
    cnt = np.array([1])
    nrm = np.zeros((1, 8, 3), np.float32)
    pts = np.zeros((1, 8, 3), np.float32)
    nrm[0, 0], pts[0, 0] = n, o
    return cnt, nrm, pts


def test_classify_is_the_plane_the_normal_and_the_hull(voc):
    v, _bsp, _ = voc
    n = [0.0, -1.0, 0.0]
    assert v.classify(*_touch([500, -20.0, 200], n), np.array([0]))[0, 0] == 0
    assert v.classify(*_touch([500, 2980.0, 200], n), np.array([0]))[0, 0] == 1
    assert v.classify(*_touch([500, -12.0, 200], n), np.array([1]))[0, 0] == 0     # ducked
    assert v.classify(*_touch([500, -12.0, 200], n), np.array([0]))[0, 0] == -1    # off-plane
    assert v.classify(*_touch([500, -20.0, 200], [0.0, 0.0, 1.0]), np.array([0]))[0, 0] == -1
    assert v.classify(*_touch([9000, -20.0, 200], n), np.array([0]))[0, 0] == -1   # far off it


def test_a_vocabulary_is_bound_to_its_map(voc):
    _v, bsp, tmp = voc
    other = tmp / "surf_fake2.bsp"
    other.write_bytes(b"y" * 65)
    with pytest.raises(ValueError):
        RampVocab(_vocab(tmp, bsp), other)
    bsp.write_bytes(b"x" * 70)                        # the same map, recompiled
    with pytest.raises(ValueError):
        RampVocab(_vocab(tmp, tmp / "surf_fake.bsp").parent / "vocab.npz", tmp / "surf_fake2.bsp")


def _windows(v, n=2, goal_field=None):
    return gr.RampWindows(v, n, ((900, 8000, 0), (1100, 8200, 300)), 10.0, topk=1, horizon=6.0,
                          fade=0.3, deterministic=True, goal_field=goal_field)


# the synthetic ramps stand in the planes y = 0 / 3000 / 6000 (facing -y, x 0..2000, z 0..500);
# their validated contact origins lie at y = -20 / 2980 / 5980, so each ramp's BOX spans x
# -48..2048, y (its origins' y) +-48, z -48..548. APPROACH is outside every box and meets ramp 0
# at a glancing angle; ON0 / ON1 / ON2 are inside ramp 0 / 1 / 2's box (on its contact plane);
# HOP is inside ramp 0's box but off its plane (airborne over it); AWAY has left ramp 0's box
APPROACH = (np.array([-1000.0, -600.0, 400.0]), np.array([1500.0, 600.0, 0.0]))
ON0 = (np.array([1000.0, -20.0, 250.0]), np.array([1500.0, 0.0, 0.0]))
ON1 = (np.array([1000.0, 2980.0, 250.0]), np.array([1500.0, 0.0, 0.0]))
ON2 = (np.array([1000.0, 5980.0, 250.0]), np.array([1500.0, 0.0, 0.0]))
HOP = (np.array([700.0, -60.0, 400.0]), np.array([1500.0, 100.0, 0.0]))
AWAY = (np.array([2600.0, -400.0, 300.0]), np.array([1500.0, -300.0, 0.0]))


def _state(s, n=2):
    return np.tile(s[0], (n, 1)), np.tile(s[1], (n, 1))


def _tick(w, st0, st1=APPROACH, ended=None):
    """one tick of a 2-env window set: env 0 at state st0, env 1 at st1"""
    return w.on_tick(None, np.stack([st0[0], st1[0]]), np.stack([st0[1], st1[1]]),
                     np.zeros(2, bool) if ended is None else ended)


def test_enter_pass_and_the_window_shift(voc):
    """the boxes (the user, 2026-09-28): inside T1's box ENTERS it (the line is rebuilt), still
    inside changes nothing, leaving it PASSES it and the window shifts; inside T2's box before T1
    was entered SKIPS T1"""
    v, _bsp, _ = voc
    w = _windows(v)
    lines = w.spawn([0, 1], *_state(APPROACH))
    assert len(lines) == 2 and w.t1[0] == 0 and w.t2[0] == 1
    idx, _ = _tick(w, ON0)                                  # env 0 enters T1
    assert list(idx) == [0] and w.entered[0] and not w.entered[1]
    for _ in range(5):
        idx, _ = _tick(w, ON0)
        assert len(idx) == 0
    idx, _ = _tick(w, AWAY)                                  # out of T1's box: passed
    assert list(idx) == [0] and w.t1[0] == 1 and w.t2[0] == 2 and w.prev[0] == 0
    assert w.n_capt[0] == 1 and w.tau[0] == 0 and not w.entered[0]
    idx, _ = _tick(w, AWAY, ON1)                             # env 1 skips T1 = 0
    assert list(idx) == [1] and w.t1[1] == 1 and w.entered[1] and w.n_skip[1] == 1


def test_a_hop_inside_the_box_is_still_the_ride(voc):
    """airborne over T1 but inside its box - a hop - is not a pass, however long (the finisher's
    49 hops all stayed inside); leaving the box is"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.spawn([0, 1], *_state(APPROACH))
    _tick(w, ON0)
    for _ in range(60):
        idx, _ = _tick(w, HOP)
        assert 0 not in idx
    assert w.t1[0] == 0 and w.entered[0] and w.n_capt[0] == 0
    idx, _ = _tick(w, AWAY)
    assert 0 in idx and w.t1[0] == 1 and w.prev[0] == 0 and w.n_capt[0] == 1


def test_leaving_t1_into_t2s_box_passes_t1_then_enters_t2(voc):
    v, _bsp, _ = voc
    w = _windows(v)
    w.spawn([0, 1], *_state(APPROACH))
    _tick(w, ON0)                                            # T1 = 0 entered
    idx, _ = _tick(w, ON1)                                   # in T2's box, out of T1's: passed
    assert list(idx) == [0] and w.prev[0] == 0 and w.t1[0] == 1 and w.n_capt[0] == 1
    idx, _ = _tick(w, ON1)                                   # the new T1 entered
    assert list(idx) == [0] and w.entered[0] and w.n_skip[0] == 0


def _wedge_vocab(tmp_path, bsp):
    """one WEDGE piece along +x - side A (facing -y) and side B (facing +y) meeting at a ridge
    (y 0, z 1000, x 0..3600) and an end CAP (0.93, 0, 0.37) closing them at x 3600..4000 - and a
    separate ramp R further along (x 6000..9000, facing -y); contact origins 20 u out"""
    rng = np.random.default_rng(1)
    k = 400

    def side(sy):                  # ridge (x 0..3600, y 0, z 1000) -> base (x 0..4000, y sy*750, z 0)
        u = rng.uniform(0, 1, k)
        x = rng.uniform(0, 1, k) * (3600.0 + 400.0 * u)
        return np.stack([x, sy * 750.0 * u, 1000.0 * (1.0 - u)], 1)
    a_, b_ = rng.uniform(0, 1, (2, k))
    fl = a_ + b_ > 1
    a_[fl], b_[fl] = 1 - a_[fl], 1 - b_[fl]
    c0, c1, c2 = np.array([4000.0, -750, 0]), np.array([4000.0, 750, 0]), np.array([3600.0, 0, 1000])
    cap = c0 + a_[:, None] * (c1 - c0) + b_[:, None] * (c2 - c0)
    ur, xr = rng.uniform(0, 1, k), rng.uniform(6000, 9000, k)
    ramp = np.stack([xr, -2000.0 - 750.0 * ur, 1000.0 * (1.0 - ur)], 1)
    nA, nB = np.array([0.0, -0.8, 0.6]), np.array([0.0, 0.8, 0.6])
    nC = np.array([0.928, 0.0, 0.371]) / np.linalg.norm([0.928, 0.0, 0.371])
    pts, nrm, sid = [], [], []
    for s_, (P, n) in enumerate(((side(-1.0), nA), (side(1.0), nB), (cap, nC), (ramp, nA))):
        pts.append(P + n * 20.0)
        nrm.append(np.tile(n, (k, 1)))
        sid.append(np.full(k, s_))
    R0, R1 = np.array([0.0, 0, 1000]), np.array([3600.0, 0, 1000])
    A0, A1 = np.array([0.0, -750, 0]), np.array([4000.0, -750, 0])
    B0, B1 = np.array([0.0, 750, 0]), np.array([4000.0, 750, 0])
    Q = [np.array([6000.0, -2000, 1000]), np.array([9000.0, -2000, 1000]),
         np.array([6000.0, -2750, 0]), np.array([9000.0, -2750, 0])]
    tris = np.array([[R0, A0, A1], [R0, A1, R1], [R0, B1, B0], [R0, R1, B1], [A1, B1, R1],
                     [Q[0], Q[2], Q[3]], [Q[0], Q[3], Q[1]]])
    f = tmp_path / "wedge.npz"
    P = np.concatenate(pts)
    N = np.concatenate(nrm)
    S = np.concatenate(sid)
    np.savez(f, version=4, map=Path(bsp).stem, bsp_sig=bsp_signature(bsp), mesh_sha1="x",
             cat=np.array([1, 1, 1, 1]), normal=np.stack([nA, nB, nC, nA]),
             area=np.array([3.8e6, 3.8e6, 3.0e5, 3.75e6]), touchable=np.ones(4),
             points=P, normals=N, surf=S, duck_points=P - N * 8.0, duck_normals=N, duck_surf=S,
             tris=tris, tri_surf=np.array([0, 0, 1, 1, 2, 3, 3]))
    return RampVocab(f, bsp)


A_, B_, CAP, R_ = 0, 1, 2, 3


def test_the_wedge_is_one_piece_and_enters_the_window_as_a_side(voc):
    """a wedge's sides and its end cap are ONE piece; the window takes it as the surface that
    runs furthest along the travel direction - a side, never the cap - whether the flight comes
    along the wedge or at its end cap head-on; of the two sides, the nearer"""
    _v, bsp, tmp = voc
    v = _wedge_vocab(tmp, bsp)
    assert v.piece[A_] == v.piece[B_] == v.piece[CAP] != v.piece[R_]
    fin = ((20000, 0, 0), (20100, 100, 100))
    for o, vel, want in (([-1500.0, -500.0, 900.0], [1500.0, 0.0, 0.0], {A_}),     # along, A's side
                         ([-1500.0, 500.0, 900.0], [1500.0, 0.0, 0.0], {B_}),      # along, B's side
                         ([5000.0, 0.0, 600.0], [-1500.0, 0.0, 0.0], {A_, B_})):   # at the cap
        w = gr.RampWindows(v, 1, fin, 10.0, topk=1, horizon=3.0, fade=0.3, deterministic=True)
        w.spawn([0], np.array([o]), np.array([vel]))
        assert int(w.t1[0]) in want, (o, vel, w.t1[0])
        assert int(w.t2[0]) != CAP


def test_anywhere_in_the_pieces_box_is_on_its_target(voc):
    """T1 is a side of the wedge: being over the OTHER side - inside the piece's box - enters it
    (the same piece, no skip), and T2 is on another piece"""
    _v, bsp, tmp = voc
    v = _wedge_vocab(tmp, bsp)
    w = gr.RampWindows(v, 1, ((20000, 0, 0), (20100, 100, 100)), 10.0, topk=1, horizon=3.0,
                       fade=0.3, deterministic=True)
    o, vel = np.array([[-1500.0, -500.0, 900.0]]), np.array([[1500.0, 0.0, 0.0]])
    w.spawn([0], o, vel)
    assert w.t1[0] == A_ and w.t2[0] == R_
    idx, _ = w.on_tick(None, np.array([[1500.0, 400.0, 700.0]]), vel, np.zeros(1, bool))
    assert list(idx) == [0] and w.entered[0] and w.t1[0] == A_ and w.n_skip[0] == 0


def _linear_field(fx, fy):
    """a real GoalField (a voxel grid, so the compiled window samples it too) holding the linear
    potential 10 000 - (fx x + fy y) at its voxel centres - trilinear sampling reproduces it"""
    from surfgym.goalfield import GoalField
    cell = 200.0
    mins = np.array([-6000.0, -6000.0, -4000.0])
    nx, ny, nz = 100, 100, 40
    xs = mins[0] + (np.arange(nx) + 0.5) * cell
    ys = mins[1] + (np.arange(ny) + 0.5) * cell
    lay = 10000.0 - (fx * xs[None, :] + fy * ys[:, None])
    grid = np.broadcast_to(lay[None], (nz, ny, nx)).astype(np.float32).copy()
    return GoalField(grid, mins, cell, 1e9)


def _DescentField():
    """a geodesic potential falling along +x (10 000 - x): its descent direction is yaw 0"""
    return _linear_field(1.0, 0.0)


def test_a_standing_spawn_rides_its_first_target_down_the_potential(voc):
    """a standing spawn's window is laid along the descent direction its T1 draw used: with its
    own zero velocity the arrival fell straight onto T1 and its ride ran either way (utopia's
    start: back toward the start)"""
    v, _bsp, _ = voc
    w = _windows(v, n=1, goal_field=_DescentField())
    o = np.array([[-2000.0, -600.0, 400.0]])
    ln = w.spawn([0], o, np.zeros((1, 3)))[0]
    assert w.t1[0] == 0
    # the ride along T1: RAMP_PRESS inside its contact plane (y = -20)
    on_ramp = ln[np.abs(ln[:, 1] - (-20.0 + gr.RAMP_PRESS)) < 1.0]
    assert len(on_ramp) >= 2 and on_ramp[-1, 0] > on_ramp[0, 0] + 500.0


def _AlongY():
    """a geodesic potential falling along +y (10 000 - y): the ramps y = 0 / 3000 / 6000 are
    consecutive, a surface left behind never comes back as a target"""
    return _linear_field(0.0, 1.0)


def _channel_by_surface(w):
    ids, vals = w.slots([0])
    return {int(s): float(x) for s, x in zip(ids[0], vals[0]) if s != gr.NONE}


def test_the_channel_values_move_continuously(voc):
    """per SURFACE the channel moves at most 1 / fade_ticks per tick - through a takeoff, and
    through a second shift inside the fade (only the oldest surface, dropped from the three
    slots, may go to zero at once)"""
    v, _bsp, _ = voc
    w = _windows(v, goal_field=_AlongY())
    w.spawn([0, 1], *_state(APPROACH))
    ids, vals = w.slots()
    assert w.t1[0] == 0 and w.t2[0] == 1
    assert np.allclose(vals[0], [0.0, 1.0, 0.5])                  # before any takeoff
    # enter 0, pass it; enter 1, pass it into 2's box and enter 2, pass it - three shifts, the
    # last two inside one fade
    script = [ON0] * 3 + [AWAY] * 2 + [ON1] * 2 + [ON2] * 2 + [AWAY] * 80
    prev = _channel_by_surface(w)
    shifts = 0
    for st in script:
        idx, _ = _tick(w, st)
        shifts += int(0 in idx and w.tau[0] == 0)
        cur = _channel_by_surface(w)
        for s_ in set(prev) & set(cur):
            assert abs(cur[s_] - prev[s_]) <= 1.0 / w.fade_ticks + 1e-6, (s_, prev, cur)
        assert all(-1e-6 <= x <= 1.0 + 1e-6 for x in cur.values())
        prev = cur
    assert shifts >= 2
    ids, vals = w.slots([0])
    assert ids[0, 0] == 2 and ids[0, 1] == gr.FIN                 # 0 -> 1 -> 2 -> the finish
    assert np.allclose(vals[0, :2], [0.0, 1.0])                   # the fades have finished


class _FakeLidar:
    def __init__(self, n_ch=1):
        self.W, self.H, self.channels = W, H, n_ch
        self.device = torch.device("cpu")
        self.near, self.range = 2000.0, 11500.0
        self.yoff = torch.linspace(-0.9, 0.9, W)
        self.poff = torch.linspace(-0.6, 0.6, H)
        self.pinhole = False

    def render(self, origin, yaw, pitch, ducked):
        return torch.full((origin.shape[0], H, W), 0.5)


def test_the_target_lidar_adds_one_channel_and_off_holds_it_at_zero(voc):
    v, _bsp, _ = voc
    lid = gr.TargetLidar(_FakeLidar(), tm=None, windows=_windows(v), mode="off")
    assert lid.channels == 2
    o = torch.zeros(2, 3)
    img = lid.render(o, torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.int64))
    assert img.shape == (2, H, W, 2)
    assert float(img[..., 1].abs().max()) == 0.0 and float(img[..., 0].min()) == 0.5


def test_the_recorded_window_events_rebuild_the_channel_exactly(voc):
    """make_ramp_hooks records one event per window change ([row, prev, T1, T2, tau, fade
    starts]); the POV render rebuilds every row's slots from them with slot_values - which must
    equal what the windows showed the policy at that row (RampWindows.slots after the previous
    tick), a second shift inside a fade included; an old 5-field event list still reads"""
    v, _bsp, _ = voc
    w = _windows(v, goal_field=_AlongY())
    w.spawn([0, 1], *_state(APPROACH))
    script = [ON0] * 3 + [AWAY] * 40 + [ON1] * 2 + [ON2] * 2 + [AWAY] * 60
    seen_ids, seen_vals, events = [], [], []
    gr._snap_event(w, 0, events)
    for k, st in enumerate(script):
        si, sv = w.slots([0])
        seen_ids.append(si[0])
        seen_vals.append(sv[0])
        idx, _ = _tick(w, st)
        if 0 in idx:
            gr._snap_event(w, k + 1, events)
    got_i, got_v = gr.slot_values(events, len(script), w.fade_ticks)
    assert len(events) >= 4                            # spawn + three shifts
    assert any(abs(e[5] - 1.0) > 1e-3 for e in events)  # one shift started mid-fade
    assert np.array_equal(got_i, np.asarray(seen_ids))
    assert np.allclose(got_v, np.asarray(seen_vals), atol=1e-5)
    n_old = int(events[1][0]) + int(w.fade_ticks) + 5
    old_i, old_v = gr.slot_values([e[:5] for e in events[:2]], n_old, w.fade_ticks)
    assert np.allclose(old_v[-1], [0.0, 1.0, 0.5]) and old_i[-1, 1] == events[1][2]


def test_a_recording_without_logged_windows_is_replayed_from_its_states(voc):
    """replay_events: states inside T1's box enter it, the first state out of it passes it -
    exactly what the live hooks do (the boxes need positions only)"""
    v, _bsp, _ = voc
    w = _windows(v)
    rows = []
    for k in range(80):
        if k < 5:
            o, vel = list(APPROACH[0]), list(APPROACH[1])  # in the air, approaching
        elif k < 25:
            o, vel = [1000.0 + k, -20.0, 250.0], [1500.0, 0.0, 0.0]   # on ramp 0's plane
        else:
            o, vel = list(AWAY[0]), list(AWAY[1])          # left it
        rows.append([k] + o + vel + [0.0, 0])
    ev = gr.replay_events(w, v, np.asarray(rows, float))
    assert ev[0][2] == 0                                # T1 = ramp 0 at the spawn
    # the pass: the first row out of the box (row 25), the window shifts
    assert any(e[2] == 1 and e[1] == 0 and e[0] == 25 for e in ev), ev


def test_the_pass_reward_counts_window_shifts_and_nothing_else(voc):
    """--ramp-reward pass: tick_pass is +1 on the tick a window SHIFTS (a pass, a skip) and 0
    on every other tick - an entry, a hop inside the box, plain flight, an ended row"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.spawn([0, 1], *_state(APPROACH))
    paid = []

    def tick(st0, st1=APPROACH):
        _tick(w, st0, st1)
        paid.append(int(w.tick_pass[0]))
    tick(ON0)                                               # the entry: 0
    for _ in range(30):
        tick(HOP)                                           # hops inside the box: 0
    assert sum(paid) == 0 and w.t1[0] == 0
    tick(AWAY)                                              # the pass: +1, once
    tick(AWAY)
    assert sum(paid) == 1 and w.t1[0] == 1
    tick(ON1)                                               # the new T1 entered: 0
    assert sum(paid) == 1
    _tick(w, ON1, ON1)                                      # env 1 skips T1 = 0 for T2 = 1
    assert w.tick_pass[1] == 1 and w.tick_pass[0] == 0
    _tick(w, AWAY, AWAY, ended=np.array([True, True]))      # ended rows never pay
    assert w.tick_pass.sum() == 0


def test_the_numba_search_answers_exactly_what_the_kd_trees_do(voc):
    """RampWindows' compiled search and window (surfgym.rampfast: brute-force minima over the
    same subsampled origins, window_line step for step) against its Python / KD-tree path - the
    same T1 / T2 and the same line from every spawn state"""
    if gr._FAST_SEARCH is None:
        pytest.skip("numba unavailable (or SURFGYM_NO_NUMBA=1): only the KD path exists")
    _v, bsp, tmp = voc
    v = _wedge_vocab(tmp, bsp)
    fin = ((20000, 0, 0), (20100, 100, 100))
    rng = np.random.default_rng(5)
    o = np.column_stack([rng.uniform(-3000, 6000, 40), rng.uniform(-2500, 1500, 40),
                         rng.uniform(200, 1500, 40)])
    vel = np.column_stack([rng.uniform(-1500, 2500, 40), rng.uniform(-800, 800, 40),
                           rng.uniform(-300, 300, 40)])
    got = []
    for fast in (True, False):
        w = gr.RampWindows(v, 40, fin, 10.0, topk=2, horizon=3.0, fade=0.3,
                           rng=np.random.default_rng(0))
        if not fast:
            w._fast = None
            w._fast_win = False
        lines = w.spawn(np.arange(40), o, vel)
        got.append((w.t1.copy(), w.t2.copy(), lines))
    assert np.array_equal(got[0][0], got[1][0]) and np.array_equal(got[0][1], got[1][1])
    assert all(a.shape == b.shape and np.allclose(a, b, atol=1e-3)
               for a, b in zip(got[0][2], got[1][2]))


def test_the_finish_box_is_a_candidate_only_within_the_reach_cap(voc):
    """FIN's geodesic distance is 0: it is eligible only when the band's lower edge reaches 0
    (it used to be appended always - training drew it as T1 from 10.4% of real utopia states, a
    median 90k u of geodesic away)"""
    v, _bsp, _ = voc
    w = _windows(v, n=1, goal_field=_AlongY())
    p, vel = APPROACH
    far = w._candidates(p, vel, set(), d_max=10350.0, d_min=100.0)
    near = w._candidates(p, vel, set(), d_max=10350.0, d_min=-10.0)
    assert all(s != gr.FIN for _d, s in far) and any(s != gr.FIN for _d, s in far)
    assert any(s == gr.FIN for _d, s in near)


def test_a_piece_is_eligible_as_a_whole_never_through_its_end_cap(voc):
    """eligibility is the PIECE's median geodesic distance: a band that only the end cap of a
    wedge lies in (the cap nearer the finish than the sides) does not make the wedge a candidate
    as its cap (utopia's [28, 29, 30] below 155k)"""
    _v, bsp, tmp = voc
    v = _wedge_vocab(tmp, bsp)
    w = gr.RampWindows(v, 1, ((20000, 0, 0), (20100, 100, 100)), 10.0, topk=1, horizon=3.0,
                       fade=0.3, deterministic=True, goal_field=_DescentField())
    d_cap = float(np.median(w.gf.sample(w.tp[CAP])))
    d_side = float(np.median(w.gf.sample(w.tp[A_])))
    assert d_cap < d_side
    band = w._candidates(np.array([-1500.0, -500.0, 900.0]), np.array([1500.0, 0.0, 0.0]),
                         set(), d_max=0.5 * (d_cap + d_side), d_min=0.0)
    assert all(s not in (A_, B_, CAP) for _d, s in band)


def test_the_ride_surface_follows_the_pieces_axis_whatever_the_heading(voc):
    """the ride surface is the one running furthest along the piece's main axis (its largest
    surface's level line), so a flight heading ACROSS a wedge still gets a side, never the cap
    (along the travel direction the cap won from 38 of 360 headings)"""
    _v, bsp, tmp = voc
    v = _wedge_vocab(tmp, bsp)
    w = gr.RampWindows(v, 1, ((20000, 0, 0), (20100, 100, 100)), 10.0, topk=1, horizon=3.0,
                       fade=0.3, deterministic=True)
    q = int(v.piece[A_])
    for hd in range(0, 360, 15):
        vh = np.array([np.cos(np.radians(hd)), np.sin(np.radians(hd)), 0.0])
        p0 = np.array([3800.0, 0.0, 1400.0]) - vh * 1500.0
        path = w._arc(p0, vh * 1500.0, 3.0)
        assert w._ride_face(q, path, vh, -np.inf, np.inf, {}) in (A_, B_), hd


def test_a_piece_left_behind_is_never_a_target_again(voc):
    """the pieces an episode has left (and the one it spawned on) are excluded for the rest of
    it - no cycles"""
    _v, bsp, tmp = voc
    v = _wedge_vocab(tmp, bsp)
    w = gr.RampWindows(v, 1, ((20000, 0, 0), (20100, 100, 100)), 10.0, topk=1, horizon=3.0,
                       fade=0.3, deterministic=True)
    o, vel = np.array([[-1500.0, -500.0, 900.0]]), np.array([[1500.0, 0.0, 0.0]])
    w.spawn([0], o, vel)
    assert w.t1[0] in (A_, B_) and w.t2[0] == R_
    wedge = int(v.piece[A_])
    away = (np.array([[9500.0, -2500.0, 1500.0]]), np.array([[-1500.0, 0.0, 0.0]]))
    w._shift(0, away[0][0], away[1][0], riding=False)       # A left: R is T1, heading back
    assert wedge in w.visited[0] and w.t1[0] == R_
    assert w.t2[0] not in (A_, B_, CAP)                      # the wedge is not next, ever


def test_off_target_is_a_ramp_contact_outside_the_pieces_it_may_ride(voc):
    """--ramp-offtarget-pen: a contact on a RAMP-like plane outside the boxes of T1, T2 (and the
    spawn's piece before the first pass) is off-target - so is going back to a piece already
    passed; a contact inside T1's box, a wall, a floor are not"""
    v, _bsp, _ = voc
    w = _windows(v, n=4)
    w.spawn(np.arange(4), *_state(APPROACH, 4))
    assert w.t1[0] == 0 and w.t2[0] == 1
    cnt = np.ones(4, np.int32)
    nrm = np.zeros((4, 8, 3), np.float32)
    nrm[0, 0] = [0.0, -0.8, 0.6]           # a ramp, far from T1 / T2
    nrm[1, 0] = [0.0, -0.8, 0.6]           # a ramp, inside T1's box
    nrm[2, 0] = [0.0, -1.0, 0.0]           # a wall
    nrm[3, 0] = [0.0, 0.0, 1.0]            # a floor
    far = np.array([1000.0, 9000.0, 250.0])
    o = np.stack([far, ON0[0], far, far])
    assert w.offtarget(cnt, nrm, o).tolist() == [True, False, False, False]
    w1 = _windows(v, n=1)
    w1.spawn([0], APPROACH[0][None], APPROACH[1][None])
    w1.on_tick(None, ON0[0][None], ON0[1][None], np.zeros(1, bool))     # enter 0
    w1.on_tick(None, AWAY[0][None], AWAY[1][None], np.zeros(1, bool))   # pass it: T1 = 1
    assert w1.prev[0] == 0 and w1.t1[0] == 1
    back = w1.offtarget(np.ones(1, np.int32), nrm[:1], ON0[0][None])     # back on ramp 0
    assert back[0]


def test_the_first_capture_keeps_riding_down_the_potential(voc):
    """Codex's case: a standing spawn's line rides T1 down the potential (+x), and the FIRST
    ENTRY - at zero velocity - rebuilds the ride from the state: laid with the descent direction
    too, it still runs +x (from the state's own velocity it ran back)"""
    v, _bsp, _ = voc
    w = _windows(v, n=1, goal_field=_DescentField())
    o = np.array([[-2000.0, -600.0, 400.0]])
    w.spawn([0], o, np.zeros((1, 3)))
    assert w.t1[0] == 0
    at = np.array([[1000.0, -20.0, 250.0]])
    idx, lines = w.on_tick(None, at, np.zeros((1, 3)), np.zeros(1, bool))   # the entry
    assert list(idx) == [0] and w.entered[0]
    ride = lines[0][np.abs(lines[0][:, 1] + 20.0) < 1.0]
    assert len(ride) >= 2 and ride[-1, 0] > ride[0, 0] + 500.0



def _short(v, n=1, goal_field=None, fast=True):
    """a 0.8 s horizon: the reach cap ((|v| + 940 u/s) x 0.8 s) spans one of the synthetic
    ramps, never two - so a band can be empty (under _AlongY: ramps at d 10,020 / 7,020 /
    4,020, the finish at ~1,900)"""
    w = gr.RampWindows(v, n, ((900, 8000, 0), (1100, 8200, 300)), 10.0, topk=1, horizon=0.8,
                       fade=0.3, deterministic=True, goal_field=goal_field)
    if not fast:
        w._fast = None
        w._fast_win = False
    return w


def test_an_empty_band_holds_and_redraws_never_an_unreachable_finish(voc):
    """Codex 2026-09-28: with no ramp in the band and the finish beyond the reach cap the window
    HOLDS - T1 NONE, nothing in the channel, a line that runs on ahead - and redraws every
    REPLAN_SECS until a target is in reach; it used to take the finish, a map away"""
    v, _bsp, _ = voc
    w = _short(v, goal_field=_AlongY())
    far = (np.array([1000.0, -3000.0, 400.0]), np.array([0.0, 1500.0, 0.0]))    # d 13,000
    ln = w.spawn([0], far[0][None], far[1][None])[0]
    assert w.t1[0] == gr.NONE and w.t2[0] == gr.NONE and w.stats["holds"] == 1
    assert np.allclose(ln[0], far[0]) and ln[-1, 1] > far[0][1] + 200.0      # runs on ahead
    assert float(w.slots([0])[1].max()) == 0.0                                 # no target shown
    near = (np.array([1000.0, -1500.0, 400.0]), np.array([0.0, 1500.0, 0.0]))  # ramp 0 in reach
    got = None
    for k in range(1, w.replan_ticks + 1):
        idx, _ = w.on_tick(None, near[0][None], near[1][None], np.zeros(1, bool))
        if len(idx):
            got = k
            break
    assert got is not None and w.t1[0] == 0                   # redrawn within REPLAN_SECS


def test_passing_the_last_target_in_reach_holds_then_redraws_continuously(voc):
    """T1 passed with no T2 (nothing was in reach past its ride) and nothing in reach from the
    exit either: the window holds (the pass still counts), the channel keeps fading the ramp
    just left, and once a ramp comes into reach it is drawn as T1 - every channel value moving
    at most 1 / fade_ticks per tick throughout"""
    v, _bsp, _ = voc
    w = _short(v, goal_field=_AlongY())
    w.spawn([0], APPROACH[0][None], APPROACH[1][None])
    assert w.t1[0] == 0 and w.t2[0] == gr.NONE              # ramp 1 is past T1's reach
    mid = (np.array([1000.0, 1500.0, 400.0]), np.array([0.0, 1500.0, 0.0]))  # ramp 1 in reach
    script = [ON0] * 3 + [AWAY] * 25 + [mid] * 40
    prev = _channel_by_surface(w)
    held = False
    for st in script:
        w.on_tick(None, st[0][None], st[1][None], np.zeros(1, bool))
        if w.t1[0] == gr.NONE:
            held = True
            assert w.prev[0] == 0 and w.n_capt[0] == 1
        cur = _channel_by_surface(w)
        for s_ in set(prev) & set(cur):
            assert abs(cur[s_] - prev[s_]) <= 1.0 / w.fade_ticks + 1e-6, (s_, prev, cur)
        prev = cur
    assert held and w.stats["holds"] == 1
    assert w.t1[0] == 1 and w.n_capt[0] == 1                 # redrawn: ramp 1


def test_the_compiled_window_holds_exactly_where_the_python_one_does(voc):
    """the empty band in the compiled search (rampfast.next_target NONE, the ride's lookahead)
    against the Python path: the same T1 / T2 - NONE included - and the same lines"""
    if gr._FAST_SEARCH is None:
        pytest.skip("numba unavailable (or SURFGYM_NO_NUMBA=1): only the KD path exists")
    v, _bsp, _ = voc
    rng = np.random.default_rng(3)
    o = np.column_stack([rng.uniform(-1500, 2500, 60), rng.uniform(-4000, 7000, 60),
                         rng.uniform(100, 900, 60)])
    vel = np.column_stack([rng.uniform(-800, 800, 60), rng.uniform(-500, 2500, 60),
                           rng.uniform(-300, 300, 60)])
    got = []
    for fast in (True, False):
        w = _short(v, n=60, goal_field=_AlongY(), fast=fast)
        lines = w.spawn(np.arange(60), o, vel)
        got.append((w.t1.copy(), w.t2.copy(), lines))
    assert (got[0][0] == gr.NONE).any() and (got[0][1] == gr.NONE).any()
    assert ((got[0][0] >= 0) & (got[0][1] == gr.NONE)).any()     # a ride with nothing next
    assert np.array_equal(got[0][0], got[1][0]) and np.array_equal(got[0][1], got[1][1])
    assert all(a.shape == b.shape and np.allclose(a, b, atol=1e-3)
               for a, b in zip(got[0][2], got[1][2]))


def test_the_steered_velocity_is_continuous_at_the_ray_floor(voc):
    """_steer blends the state's own horizontal direction into the descent direction below
    RAY_FLOOR (weight |v_h| / RAY_FLOOR): no jump at RAY_FLOOR (the hard switch turned two B7
    shifts by 45 and 61 deg - Codex 2026-09-28), idempotent, its own vertical kept, all descent
    at a standstill"""
    v, _bsp, _ = voc
    w = _windows(v, n=1, goal_field=_DescentField())          # descent: +x
    p = np.array([0.0, 0.0, 400.0])

    def ang(a, b):
        c = float(a[:2] @ b[:2]) / (np.linalg.norm(a[:2]) * np.linalg.norm(b[:2]))
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    for deg in (0.0, 45.0, 90.0, 135.0, 170.0, 180.0):
        u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        lo = w._steer(p, np.array([*(u * (gr.RAY_FLOOR - 0.1)), 50.0]))
        hi = w._steer(p, np.array([*(u * (gr.RAY_FLOOR + 0.1)), 50.0]))
        assert ang(lo, hi) < 0.1 and abs(np.linalg.norm(lo[:2]) - np.linalg.norm(hi[:2])) < 0.5
        assert lo[2] == 50.0 and np.allclose(w._steer(p, lo), lo)
    assert np.allclose(w._steer(p, np.array([0.0, 0.0, -100.0])), [gr.RAY_FLOOR, 0.0, -100.0])
    prev = None
    for s in np.linspace(0.0, 400.0, 801):                    # across: a smooth turn
        cur = w._steer(p, np.array([0.0, s, 0.0]))
        if prev is not None:
            assert ang(prev, cur) < 1.0, s
        prev = cur


def test_the_riding_first_line_counts_its_lookahead_against_the_buffer(voc):
    """rampfast.window's riding-first ride reserves room for the 8 lookahead points that may
    follow it (it reserved 1: a ride filling the buffer wrote past its end - Codex 2026-09-28);
    a full buffer hands the window to the Python path, which lays the same line"""
    if gr._FAST_SEARCH is None:
        pytest.skip("numba unavailable (or SURFGYM_NO_NUMBA=1): only the KD path exists")
    v, _bsp, _ = voc
    w = _windows(v, n=1)
    p, vel = np.array([1000.0, -20.0, 250.0]), np.array([1500.0, 0.0, 0.0])
    m = max(2, int((float(w.tp[0][:, 0].max()) - 1000.0) / 15.0))   # its ride, 15 u a point
    w._buf = np.empty((m + 6, 3))            # room for the ride and 4 more: not the lookahead
    assert w._fast_window(p, vel, 0, gr.NONE, True, set()) is None
    w._buf = np.empty((100_000, 3))
    fast = w._fast_window(p, vel, 0, gr.NONE, True, set())[0]
    py, _raw = gr.window_line(w.tp, w.tn, w.tt, w.finish, p, vel, [0], fin=gr.FIN,
                              riding_first=True, gravity=w.gravity, dt_path=w.dt_path,
                              line_cap=w.line_cap)
    assert fast.shape == py.shape and np.allclose(fast, py, atol=1e-3)



def test_the_pass_flag_is_one_on_the_decision_after_a_shift_and_zero_otherwise(voc):
    """--ramp-obs-pass (the user, 2026-09-28): take_passes is 1 where the window shifted since the
    last read - a pass, a skip - and 0 through an entry, hops inside the box and plain flight;
    one read restarts it, the bootstrap's pass_flags does not, a respawn does"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.spawn([0, 1], *_state(APPROACH))
    assert w.take_passes().tolist() == [0.0, 0.0]
    _tick(w, ON0)                                            # the entry: no pass
    for _ in range(5):
        _tick(w, HOP)
    assert w.take_passes().tolist() == [0.0, 0.0]
    _tick(w, AWAY)                                           # env 0 passes T1
    _tick(w, AWAY)
    assert w.take_passes().tolist() == [1.0, 0.0]
    assert w.take_passes().tolist() == [0.0, 0.0]            # read once
    _tick(w, AWAY, ON1)                                      # env 1 skips T1 for T2
    assert w.pass_flags([1]).tolist() == [1.0] and w.pass_flags([1]).tolist() == [1.0]
    w.spawn([1], APPROACH[0][None], APPROACH[1][None])       # a new episode: nothing passed
    assert w.take_passes().tolist() == [0.0, 0.0]


def test_the_eval_pass_feed_reads_env_0_once_per_decision(voc):
    """make_pass_feed: the eval / recording column - env 0's flag off its 1-env windows, 0 for
    any other env of the core, restarted by the read"""
    v, _bsp, _ = voc
    w = _windows(v, n=1)
    w.spawn([0], APPROACH[0][None], APPROACH[1][None])
    feed = gr.make_pass_feed(w)

    class _Core:
        states_view = {"origin": np.zeros((3, 3))}
    for st in (ON0, ON0, AWAY):
        w.on_tick(None, st[0][None], st[1][None], np.zeros(1, bool))
    assert feed(_Core(), 3).tolist() == [1.0, 0.0, 0.0]
    assert feed(_Core()).tolist() == [0.0, 0.0, 0.0]


def test_a_checkpoint_grows_onto_the_pass_column_as_its_own_function():
    """--ramp-obs-pass is the LAST scalar-side column, so widen_for_obs' zero-pad grows a
    checkpoint onto it: the resumed policy computes the checkpoint's logits and value whatever
    the flag holds, and the flag's weights then learn from zero"""
    from train_fast import widen_for_obs
    R = 5
    torch.manual_seed(0)
    old = Policy(N_SCALAR + R + W * H, W, H, emb=32, hidden=24, route_dim=R).eval()
    ck = _ck(old)
    torch.manual_seed(1)
    new = Policy(N_SCALAR + R + 1 + W * H, W, H, emb=32, hidden=24, route_dim=R + 1)
    assert widen_for_obs(ck, new, R + 1, flag="--ramp-obs-pass 1") > 0
    new.load_state_dict(ck["policy"])
    new.eval()
    g = torch.Generator().manual_seed(4)
    scal = torch.randn(6, N_SCALAR + R, generator=g)
    img = torch.rand(6, W * H, generator=g)
    la, va = old(torch.cat([scal, img], 1))
    for flag in (0.0, 1.0):
        lb, vb = new(torch.cat([scal, torch.full((6, 1), flag), img], 1))
        assert torch.allclose(la, lb, atol=1e-6, rtol=0), (la - lb).abs().max().item()
        assert torch.allclose(va, vb, atol=1e-6, rtol=0), (va - vb).abs().max().item()
    lb, vb = new(torch.cat([scal, torch.ones(6, 1), img], 1))
    (lb.sum() + vb.sum()).backward()
    col = N_SCALAR + R - N_SCALAR                 # the flag's column in the towers' input
    grads = [p.grad for n_, p in new.named_parameters() if n_ in ("pi.0.weight", "vf.0.weight")]
    feat = int(new.feat_dim)
    assert all(float(gr_[:, feat + col].abs().max()) > 0.0 for gr_ in grads)



MID = (np.array([1000.0, 1500.0, 400.0]), np.array([1500.0, 0.0, 0.0]))   # between ramps 0 and 1


def test_a_predefined_sequence_replaces_the_planner_and_may_revisit_a_ramp(voc):
    """--ramp-sequence (the user, 2026-09-28): T1 / T2 are the list's next two, a pass advances
    it - back to a ramp already left included (unitfarmer2's record lands on its first ramp again),
    whatever the geodesic order says - entering the LAST target completes it, and past the end
    the window holds with no redraw"""
    v, _bsp, _ = voc
    w = _windows(v, n=1, goal_field=_AlongY())
    w.set_sequence([0, 1, 0])
    w.spawn([0], APPROACH[0][None], APPROACH[1][None])
    assert (w.t1[0], w.t2[0], w.seq_k[0]) == (0, 1, 0)

    def tick(st):
        return w.on_tick(None, st[0][None], st[1][None], np.zeros(1, bool))
    tick(ON0)
    tick(AWAY)                                                # 0 passed
    assert (w.t1[0], w.t2[0], w.seq_k[0]) == (1, 0, 1) and not w.seq_done[0]
    tick(ON1)
    tick(MID)                                                 # 1 passed: back to 0, the last
    assert (w.t1[0], w.t2[0], w.seq_k[0]) == (0, gr.NONE, 2) and not w.seq_done[0]
    tick(ON0)                                                 # the last target entered
    assert w.seq_done[0] and w.seq_stage(0) == 3
    tick(AWAY)                                                # left it: the window holds
    assert w.t1[0] == gr.NONE and not w.holding[0]
    for _ in range(3 * w.replan_ticks):
        idx, _ = tick(MID)
        assert len(idx) == 0 and w.t1[0] == gr.NONE           # no redraw after the sequence
    w.settle([0], [True])
    assert w.pop_stats()["seq_done"] == 1
    w.spawn([0], APPROACH[0][None], APPROACH[1][None])        # a new episode starts it over
    assert (w.t1[0], w.seq_k[0], bool(w.seq_done[0])) == (0, 0, False)


def test_a_skip_into_the_last_target_completes_the_sequence(voc):
    v, _bsp, _ = voc
    w = _windows(v, n=1)
    w.set_sequence([0, 1])
    w.spawn([0], APPROACH[0][None], APPROACH[1][None])
    w.on_tick(None, ON1[0][None], ON1[1][None], np.zeros(1, bool))   # into T2's box first
    assert w.t1[0] == 1 and w.seq_k[0] == 1 and w.seq_done[0] and w.n_skip[0] == 1


def test_a_sequence_must_name_target_surfaces(voc):
    v, _bsp, _ = voc
    w = _windows(v, n=1)
    with pytest.raises(ValueError):
        w.set_sequence([0, 99])
    with pytest.raises(ValueError):
        w.set_sequence([])


class _SeqCore:
    """the few core calls make_ramp_hooks makes, one env"""
    num_envs = 1

    def __init__(self):
        self.killed = 0
        self.goal_hits = np.zeros(1, bool)
        self.states_view = {"origin": APPROACH[0][None].astype(np.float32),
                            "velocity": APPROACH[1][None].astype(np.float32),
                            "ducked": np.zeros(1, np.int64)}

    def get_touch(self):
        return (np.zeros(1, np.int32), np.zeros((1, 8, 3), np.float32),
                np.zeros((1, 8, 3), np.float32))

    def force_fail(self, m):
        self.killed += int(np.asarray(m)[0])

    def at(self, st):
        self.states_view["origin"] = st[0][None].astype(np.float32)
        self.states_view["velocity"] = st[1][None].astype(np.float32)


def test_the_eval_ends_a_completed_sequence_as_a_success(voc):
    """make_ramp_hooks under --ramp-sequence: the tick the last target is entered the episode is
    ended (force_fail) and it settles as a SUCCESS, with its stage in the record"""
    v, _bsp, _ = voc
    w = _windows(v, n=1)
    w.set_sequence([0, 1])
    core = _SeqCore()
    ev = {}
    meta, on_tick = gr.make_ramp_hooks(w, v, core, ev)
    meta(0)
    no = np.zeros(1, bool)
    for k, st in enumerate((ON0, ON0, AWAY, ON1)):
        core.at(st)
        on_tick(k, None, None, no, no)
    assert core.killed == 1 and w.seq_done[0]
    on_tick(4, None, None, np.ones(1, bool), no)              # the kill lands: the episode ends
    assert ev["succ"] == 1 and ev["stages"] == [2]
    rec = meta.episode_end(0)["targets"]
    assert rec["seq_done"] and rec["seq_stage"] == 2 and rec["sequence"] == [0, 1]



@pytest.mark.parametrize("pinhole", [False, True])
def test_six_target_views_each_see_their_own_side(tmp_path, pinhole):
    """--target-views 6: the target channel from the view's own direction and from up, down,
    back, left, right. A box of six target surfaces around the eye - a wall ahead, a ceiling, a
    floor, a wall behind, one on each side - each lights exactly its own view's channel"""
    from surfgym.targetmask import TargetMask

    def quad(p0, p1, p2, p3):
        a, b, c, d = (np.asarray(x, float) for x in (p0, p1, p2, p3))
        return np.array([[a, b, c], [a, c, d]])
    e = 17.0                                                   # the standing eye height
    faces = [quad([1000, -300, e - 300], [1000, 300, e - 300], [1000, 300, e + 300],
                  [1000, -300, e + 300]),                      # 0: ahead (+x)
             quad([-400, -400, e + 1000], [400, -400, e + 1000], [400, 400, e + 1000],
                  [-400, 400, e + 1000]),                      # 1: above
             quad([-400, -400, e - 1000], [400, -400, e - 1000], [400, 400, e - 1000],
                  [-400, 400, e - 1000]),                      # 2: below
             quad([-1000, -300, e - 300], [-1000, 300, e - 300], [-1000, 300, e + 300],
                  [-1000, -300, e + 300]),                     # 3: behind
             quad([-300, 1000, e - 300], [300, 1000, e - 300], [300, 1000, e + 300],
                  [-300, 1000, e + 300]),                      # 4: left (+y)
             quad([-300, -1000, e - 300], [300, -1000, e - 300], [300, -1000, e + 300],
                  [-300, -1000, e + 300])]                     # 5: right (-y)
    mesh = tmp_path / "box.npz"
    np.savez(mesh, tris=np.concatenate(faces).astype(np.float32),
             tri_surf=np.repeat(np.arange(6), 2), cat=np.ones(6, np.int64))
    tm = TargetMask(str(mesh), None, "cpu")

    class _Lid:
        channels, near, range = 1, 2000.0, 11500.0

        def __init__(self):
            self.H, self.W, self.device = 24, 48, torch.device("cpu")
            self.pinhole = pinhole
            self.yoff = torch.linspace(0.9, -0.9, self.W)        # col 0 looks left, like the lidar
            self.poff = torch.linspace(0.6, -0.6, self.H)        # row 0 looks up
            # --pinhole: GpuLidar's tangent-plane offsets (120 x 90 deg)
            self.uoff = torch.as_tensor(np.tan(np.radians(60.0))
                                        * (2.0 * np.arange(self.W) / (self.W - 1) - 1.0),
                                        dtype=torch.float32)
            self.voff = torch.as_tensor(np.tan(np.radians(45.0))
                                        * (1.0 - 2.0 * np.arange(self.H) / (self.H - 1)),
                                        dtype=torch.float32)

        def _dirs_pinhole(self, N, yaw_deg, pitch_deg, d2r):
            yw = yaw_deg.view(N, 1, 1) * d2r
            pt = pitch_deg.view(N, 1, 1) * d2r
            cy, sy, cp, sp = torch.cos(yw), torch.sin(yw), torch.cos(pt), torch.sin(pt)
            u = self.uoff.view(1, 1, self.W)
            v = self.voff.view(1, self.H, 1)
            dx = cp * cy + u * sy - v * sp * cy
            dy = cp * sy - u * cy - v * sp * sy
            dz = sp + v * cp
            inv = torch.rsqrt(dx * dx + dy * dy + dz * dz)
            self._dx.copy_(dx * inv)
            self._dy.copy_(dy * inv)
            self._dz.copy_(dz * inv)

        def render(self, origin, yaw, pitch, ducked, **kw):
            return torch.zeros(origin.shape[0], self.H, self.W)

        def _ensure_buffers(self, n):
            self._dx = torch.zeros(n, self.H, self.W)
            self._dy = torch.zeros_like(self._dx)
            self._dz = torch.zeros_like(self._dx)

        def _dirs_equiangular(self, N, yaw_deg, pitch_deg, d2r):
            p = pitch_deg.view(N, 1, 1) * d2r + self.poff.view(1, self.H, 1)
            y = yaw_deg.view(N, 1, 1) * d2r + self.yoff.view(1, 1, self.W)
            cp = torch.cos(p)
            self._dx.copy_(cp * torch.cos(y))
            self._dy.copy_(cp * torch.sin(y))
            self._dz.copy_(torch.sin(p).expand_as(self._dz))

    class _Win:
        target = 0

        def slots(self, idx=None):
            return (np.array([[gr.NONE, self.target, gr.NONE]]),
                    np.array([[0.0, 1.0, 0.0]], np.float32))
    win = _Win()
    lid = gr.TargetLidar(_Lid(), tm, win, views=6)
    assert lid.channels == 7
    names = ["ahead"] + [v[0] for v in gr.TARGET_VIEWS]
    for s_ in range(6):
        win.target = s_
        img = lid.render(torch.zeros(1, 3), torch.zeros(1), torch.zeros(1),
                         torch.zeros(1, dtype=torch.int64))
        assert img.shape == (1, 24, 48, 7)
        lit = [names[c] for c in range(6) if float(img[0, :, :, 1 + c].abs().max()) > 0.0]
        assert lit == [names[s_]], (s_, lit)



def test_a_pass_pays_the_exits_energy_height_and_nothing_else(voc):
    """--ramp-exit-bonus: tick_exit_h is, on the tick of a PASS, the exit's energy as the height it
    could climb to - z + |v|^2 / 2g - above the left ramp's lowest validated contact; 0 on an
    entry, on hops inside the box, on a skip and on every other tick"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.spawn([0, 1], *_state(APPROACH))
    zlow = float(w.tp[0][:, 2].min())

    def tick(st0, st1=APPROACH):
        _tick(w, st0, st1)
        return w.tick_exit_h.copy()
    assert tick(ON0).sum() == 0.0                                  # the entry
    for _ in range(5):
        assert tick(HOP).sum() == 0.0                              # hops inside the box
    h = tick(AWAY)                                                 # the pass
    want = AWAY[0][2] + float(AWAY[1] @ AWAY[1]) / (2.0 * w.gravity) - zlow
    assert abs(h[0] - want) < 1e-6 and h[1] == 0.0
    assert tick(AWAY).sum() == 0.0                                 # once
    assert tick(AWAY, ON1)[1] == 0.0 and w.n_skip[1] == 1          # a skip pays nothing



def test_ramp_pairs_give_each_spawn_its_own_pair_and_end_on_the_second(voc):
    """--ramp-pairs: a spawn at a pair's state is shown THAT pair [T1, T2] and entering T2 after
    T1 completes it (the episode's success); a spawn at no pair's state falls back to the
    planner - the global sequence stays None"""
    from surfgym.core import STATE_DTYPE
    v, _bsp, _ = voc
    st = np.zeros(2, STATE_DTYPE)
    st["origin"][0] = APPROACH[0]
    st["origin"][1] = MID[0]
    pairs = {"states": st, "t1": np.array([0, 1]), "t2": np.array([1, 2]), "source": "self:test",
             "map": "surf_fake", "path": "mem"}
    pl = gr.RampPlanner(v, 3, ((900, 8000, 0), (1100, 8200, 300)), 10.0, topk=1, horizon=6.0,
                        fade=0.3, pairs=pairs)
    w = pl.windows
    w.spawn([0, 1, 2], np.stack([APPROACH[0], MID[0], np.array([50.0, -3000.0, 400.0])]),
            np.stack([APPROACH[1], MID[1], APPROACH[1]]))
    assert w.env_seq[0] == [0, 1] and (w.t1[0], w.t2[0]) == (0, 1)
    assert w.env_seq[1] == [1, 2] and (w.t1[1], w.t2[1]) == (1, 2)
    assert w.env_seq[2] is None and w.seq is None
    o = np.stack([ON0[0], MID[0], MID[0]])
    vel = np.stack([ON0[1], MID[1], MID[1]])
    w.on_tick(None, o, vel, np.zeros(3, bool))                    # env 0 enters 0
    o[0] = AWAY[0]
    w.on_tick(None, o, vel, np.zeros(3, bool))                    # passes 0: T1 = 1
    assert w.t1[0] == 1 and not w.seq_done[0]
    o[0] = ON1[0]
    w.on_tick(None, o, vel, np.zeros(3, bool))                    # enters 1: the pair is done
    assert w.seq_done[0] and w.seq_stage(0) == 2


def test_ramp_file_for_map_picks_each_maps_own_file(tmp_path):
    """--ramp-vocab / --ramp-pairs on a --maps run: one npz per map, picked by its own "map"
    field; a single file is returned as it is (its consumer checks the map); none -> None"""
    for m in ("surf_a", "surf_b"):
        np.savez(tmp_path / f"{m}.npz", map=m)
    both = f"{tmp_path / 'surf_a.npz'},{tmp_path / 'surf_b.npz'}"
    assert gr.ramp_file_for_map(both, "surf_b") == tmp_path / "surf_b.npz"
    assert gr.ramp_file_for_map(both, "surf_a") == tmp_path / "surf_a.npz"
    assert gr.ramp_file_for_map(both, "surf_c") is None
    assert gr.ramp_file_for_map("surf_a.npz", "surf_zzz", tmp_path) == tmp_path / "surf_a.npz"


class _SlotCore:
    """a map slot's core: n envs, the calls the ramp goal system makes"""

    class config:
        class phys:
            sv_gravity = 800.0

    def __init__(self, n):
        self.n = n
        self.killed = np.zeros(n, np.int64)
        self.goal_hits = np.zeros(n, bool)
        self.states_view = {"origin": np.zeros((n, 3), np.float32),
                            "velocity": np.zeros((n, 3), np.float32),
                            "ducked": np.zeros(n, np.int64)}

    def map_bounds(self):
        return np.array([-9e3, -9e3, -9e3]), np.array([9e3, 9e3, 9e3])

    def get_touch(self):
        return (np.zeros(self.n, np.int32), np.zeros((self.n, 8, 3), np.float32),
                np.zeros((self.n, 8, 3), np.float32))

    def force_fail(self, m):
        self.killed += np.asarray(m, bool)

    def at(self, i, st):
        self.states_view["origin"][i] = st[0]
        self.states_view["velocity"][i] = st[1]


def test_one_goal_system_runs_each_map_slot_on_its_own_core_planner_and_rows(voc, tmp_path):
    """--goal-planner ramps on --maps (the user, 2026-09-28: joint training on many maps' pairs):
    envs [0, 2) are map A, [2, 4) map B. A spawn reads ITS map's core and pairs; a completed pair
    kills the env on ITS map's core in its LOCAL row and settles as a success on the next tick;
    the log counts each map's completions"""
    from types import SimpleNamespace
    from surfgym.core import STATE_DTYPE
    from surfgym.goalsys import GoalSystem, RampSlot
    v, _bsp, _ = voc

    def pairs(o1, o2, t1, t2, name):
        st = np.zeros(2, STATE_DTYPE)
        st["origin"][0], st["origin"][1] = o1, o2
        return {"states": st, "t1": np.array(t1), "t2": np.array(t2), "source": "self:test",
                "map": name, "path": "mem"}
    box = ((900, 8000, 0), (1100, 8200, 300))
    pa = gr.RampPlanner(v, 2, box, 10.0, topk=1, horizon=6.0, fade=0.3,
                        pairs=pairs(APPROACH[0], MID[0], [0, 1], [1, 2], "surf_a"))
    pb = gr.RampPlanner(v, 2, box, 10.0, topk=1, horizon=6.0, fade=0.3,
                        pairs=pairs(MID[0], APPROACH[0], [1, 0], [2, 1], "surf_b"))
    ca, cb = _SlotCore(2), _SlotCore(2)
    ca.at(0, APPROACH)
    ca.at(1, MID)
    cb.at(0, MID)
    cb.at(1, APPROACH)
    args = SimpleNamespace(goal_radius=192.0, goal_holdout=None, goal_air_frac=0.0,
                           goal_kmin=1.0, goal_kmax=5.0, goal_kcap=1.0, goal_curriculum=0)
    gs = GoalSystem(ca, 4, None, SimpleNamespace(reachable=lambda p: np.ones(len(p), bool)),
                    1.0, args, "cpu", tmp_path, planner=pa,
                    ramp_slots=[RampSlot(0, 2, ca, pa, name="surf_a"),
                                RampSlot(2, 4, cb, pb, name="surf_b")])
    gs.assign(np.arange(4))
    assert list(gs.plan_tgt) == [0, 1, 1, 0]                  # each map's own pairs
    assert pb.windows.env_seq == [[1, 2], [0, 1]]
    no = np.zeros(4, bool)
    ln = np.ones(4, np.int64)
    # map B's local env 1 (global 3): enter 0, pass it, enter 1 - its pair is done
    for st in (ON0, AWAY, ON1):
        cb.at(1, st)
        fin = gs.on_step(no, no, ln)
        assert not fin.any()
    assert list(cb.killed) == [0, 1] and not ca.killed.any()  # killed on B's core, row 1
    assert list(gs.pending) == [False, False, False, True]
    done = np.array([False, False, False, True])
    cb.at(1, APPROACH)                                        # the core respawned it
    fin = gs.on_step(done, no, ln)
    assert list(fin) == [False, False, False, True]           # settled as the success
    gs.assign(np.array([3]))
    assert pb.windows.env_seq[1] == [0, 1] and not pb.windows.seq_done[1]
    note = gs.note(0)
    assert "per map done a 0/0 b 1/1" in note, note
    assert gs.ramp_slot("surf_b").planner is pb and gs.ramp_slot("surf_a").planner is pa


def test_t1_dist_and_the_reach_event(voc):
    """--ramp-reward dist (the user, 2026-10-06): t1_dist is the Euclidean distance to the
    nearest validated contact origin of T1's piece (ramp 0's origins lie at y = -20, x 0..2000,
    z 0..500), it follows the window to the next target after a pass, and a box entry raises
    tick_enter for exactly one tick; FIN measures to the finish box (0 inside)"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.spawn([0, 1], *_state(APPROACH))
    d = w.t1_dist(np.stack([APPROACH[0], APPROACH[0]]))
    want = float(np.hypot(1000.0, 580.0))              # to the ramp's x = 0 edge at z ~400
    assert abs(d[0] - want) < 80.0 and abs(d[1] - want) < 80.0
    _tick(w, ON0)                                       # env 0 enters T1
    assert list(w.tick_enter) == [1, 0]
    assert w.t1_dist(np.stack([ON0[0], APPROACH[0]]))[0] < 60.0
    _tick(w, ON0)
    assert list(w.tick_enter) == [0, 0]                 # one tick only
    _tick(w, AWAY)                                      # passed: T1 is ramp 1 (origins y 2980)
    assert w.t1[0] == 1
    d1 = w.t1_dist(np.stack([AWAY[0], APPROACH[0]]))[0]
    assert abs(d1 - float(np.hypot(600.0, 3380.0))) < 60.0   # to ramp 1's x = 2000 edge
    w.t1[0] = gr.FIN                                    # the finish box (900..1100, 8000..8200)
    assert w.t1_dist(np.stack([np.array([1000.0, 8100.0, 100.0]), APPROACH[0]]))[0] == 0.0
    assert abs(w.t1_dist(np.stack([np.array([1000.0, 7000.0, 100.0]), APPROACH[0]]))[0]
               - 1000.0) < 1e-6
    w.t1[1] = gr.NONE
    assert np.isnan(w.t1_dist(np.stack([APPROACH[0], APPROACH[0]]))[1])


def test_target_vec_points_at_the_next_ramp_in_the_views_frame(voc):
    """--ramp-obs-vec (the user, 2026-10-06: "the vector ... that points to the next ramp"): the
    unit vector to T1's nearest validated contact origin in the VIEW's ego frame (forward, left,
    up) and its length / 2,000 u; once T1 is entered it points at T2; FIN at the finish box's
    nearest point; zeros for NONE; the length column is capped at 4"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.spawn([0, 1], *_state(APPROACH))
    o = np.stack([APPROACH[0], APPROACH[0]])
    u = np.array([1000.0, 580.0, 0.0]) / np.hypot(1000.0, 580.0)   # to ramp 0's x = 0 edge
    a = w.target_vec(o, np.array([0.0, 90.0]))
    assert a.shape == (2, 4) and a.dtype == np.float32
    # the vocabulary's origins are 300 random samples per ramp (~58 u apart), so the nearest one
    # is near the edge, not on it: the direction to within ~0.15; the FRAME exactly
    assert np.allclose(a[0, :3], u, atol=0.15)                      # yaw 0: forward is +x
    assert np.allclose(a[1, :3], [a[0, 1], -a[0, 0], a[0, 2]], atol=1e-5)   # yaw 90: +y
    assert np.allclose(np.linalg.norm(a[:, :3], axis=1), 1.0, atol=1e-5)
    assert abs(a[0, 3] - np.hypot(1000.0, 580.0) / gr.VEC_D_SCALE) < 0.04
    _tick(w, ON0)                                                   # env 0 enters T1 ...
    assert w.entered[0] and w.t2[0] == 1
    b = w.target_vec(np.stack([ON0[0], APPROACH[0]]), np.zeros(2))
    assert np.allclose(b[0, :3], [0.0, 1.0, 0.0], atol=0.15)        # ... so it points at T2
    assert abs(b[0, 3] - 3000.0 / gr.VEC_D_SCALE) < 0.04
    assert np.allclose(b[1], a[0])                                  # env 1 untouched
    sub = w.target_vec(APPROACH[0][None], np.array([0.0]), idx=[1])  # one env by index
    assert np.allclose(sub[0], a[0])
    w.t1[1], w.entered[1] = gr.FIN, False                           # the finish box
    f = w.target_vec(np.stack([ON0[0], np.array([1000.0, 7000.0, 100.0])]), np.full(2, 90.0))
    assert np.allclose(f[1], [1.0, 0.0, 0.0, 1000.0 / gr.VEC_D_SCALE], atol=1e-6)
    far = w.target_vec(np.stack([ON0[0], np.array([1000.0, -60000.0, 100.0])]), np.zeros(2))
    assert far[1, 3] == gr.VEC_D_CAP                                # 68,000 u: capped
    w.t1[1] = gr.NONE
    assert np.all(w.target_vec(o, np.zeros(2))[1] == 0.0)


def test_the_eval_vec_feed_is_env_0s_arrow_and_zeros_elsewhere(voc):
    """the recorder's / eval's vec_fn: env 0's arrow off the eval windows from the core's live
    origin and view yaw, zeros for every other env of the core"""
    v, _bsp, _ = voc
    w = _windows(v, n=1)
    w.spawn([0], *_state(APPROACH, n=1))

    class _Core:
        states_view = {"origin": np.stack([APPROACH[0], ON0[0], ON1[0]]),
                       "yaw": np.array([0.0, 45.0, 90.0])}

    out = gr.make_vec_feed(w)(_Core(), 3)
    assert out.shape == (3, 4)
    assert np.allclose(out[0], w.target_vec(APPROACH[0][None], np.zeros(1))[0])
    assert np.all(out[1:] == 0.0)


def _touches(rows):
    """per env (origin, normal) or None -> SurfCore.get_touch's (counts, normals, points)"""
    n = len(rows)
    cnt = np.zeros(n, np.int32)
    nrm = np.zeros((n, 8, 3), np.float32)
    pts = np.zeros((n, 8, 3), np.float32)
    for i, r in enumerate(rows):
        if r is not None:
            cnt[i] = 1
            pts[i, 0], nrm[i, 0] = r
    return cnt, nrm, pts


def test_ramp_touch_reaches_on_contact_not_on_the_box(voc):
    """--ramp-touch (2026-10-06: uf2's sequence was 'completed' 9/9 by clipping S19's padded box
    without ever touching S19): inside a target's box WITHOUT a contact reaches nothing - no
    tick_touch, the arrow still points at it, the LAST target does not complete; the first
    contact with its own surface does, once; a contact with another ramp never counts"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.set_sequence([0, 1])
    w.set_touch(True)
    w.spawn([0, 1], *_state(APPROACH))
    n_ = np.array([0.0, -1.0, 0.0])
    duck = np.zeros(2, np.int64)
    _tick(w, ON0)                                        # env 0 inside ramp 0's box ...
    assert w.entered[0] and not w.touched[0]
    w.note_touch(*_touches([None, None]), duck)
    assert list(w.tick_touch) == [0, 0]
    a = w.target_vec(np.stack([ON0[0], APPROACH[0]]), np.zeros(2))
    assert a[0, 3] < 0.05                                # ... the arrow still on ramp 0
    # env 1 touches ramp 1 while its T1 is ramp 0: not a reach
    w.note_touch(*_touches([(ON0[0], n_), (ON1[0], n_)]), duck)
    assert list(w.tick_touch) == [1, 0] and w.touched[0] and not w.touched[1]
    assert not w.seq_done[0]                             # ramp 0 is not the last target
    b = w.target_vec(np.stack([ON0[0], APPROACH[0]]), np.zeros(2))
    assert b[0, 1] > 0.9                                 # touched: the arrow turns to ramp 1
    w.note_touch(*_touches([(ON0[0], n_), None]), duck)
    assert list(w.tick_touch) == [0, 0]                  # once per target
    _tick(w, AWAY)                                       # passed: T1 = ramp 1, the LAST
    assert w.t1[0] == 1 and not w.touched[0]
    _tick(w, ON1)                                        # inside its box: NOT complete
    assert w.entered[0] and not w.seq_done[0]
    w.note_touch(*_touches([None, None]), duck)
    assert not w.seq_done[0]
    w.note_touch(*_touches([(ON1[0], n_), None]), duck)  # the contact completes it
    assert w.seq_done[0] and w.tick_touch[0] == 1


def test_without_ramp_touch_the_box_completes_as_before(voc):
    """the default: entering the LAST target's box completes the sequence (the pre-flag rule)"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.set_sequence([0, 1])
    w.spawn([0, 1], *_state(APPROACH))
    _tick(w, ON0)
    _tick(w, AWAY)
    _tick(w, ON1)
    assert w.seq_done[0] and not w.touch
    w.note_touch(*_touches([None, None]), np.zeros(2, np.int64))   # a no-op without the flag
    assert list(w.tick_touch) == [0, 0]


def test_ramp_touch_a_box_graze_is_not_a_pass(voc):
    """--ramp-touch: leaving T1's box WITHOUT having touched it is no pass - T1 stays the target
    and its box can be entered again; touched, the same exit passes it as before"""
    v, _bsp, _ = voc
    w = _windows(v)
    w.set_sequence([0, 1, 2])
    w.set_touch(True)
    w.spawn([0, 1], *_state(APPROACH))
    n_ = np.array([0.0, -1.0, 0.0])
    duck = np.zeros(2, np.int64)
    _tick(w, HOP)                                        # inside ramp 0's box, off its plane
    w.note_touch(*_touches([None, None]), duck)
    assert w.entered[0] and not w.touched[0]
    _tick(w, AWAY)                                       # left it untouched: no pass
    assert w.t1[0] == 0 and not w.entered[0] and w.n_capt[0] == 0 and w.seq_k[0] == 0
    _tick(w, ON0)                                        # back in, and this time on it
    w.note_touch(*_touches([(ON0[0], n_), None]), duck)
    assert w.touched[0]
    _tick(w, AWAY)                                       # touched, then left: PASSED
    assert w.t1[0] == 1 and w.n_capt[0] == 1 and w.seq_k[0] == 1
