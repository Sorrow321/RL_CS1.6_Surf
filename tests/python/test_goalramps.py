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


def test_only_one_added_channel_is_a_supported_warm_start():
    ck = _ck(_pol(1, 0))
    assert widen_for_target_channel(_ck(_pol(2, 1)), _pol(2, 2)) == 0     # already has it
    with pytest.raises(SystemExit):
        widen_for_target_channel(ck, _pol(3, 3))


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
