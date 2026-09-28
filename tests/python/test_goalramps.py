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
5. RampWindows: capture at the first T1 contact, takeoff after DEPART_TICKS contact-free ticks
   (the window shifts T2 -> T1), a T2 contact before T1 skips T1, and the channel's values move
   continuously (at most 1 / fade_ticks per tick).
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


def _windows(v):
    return gr.RampWindows(v, 2, ((900, 8000, 0), (1100, 8200, 300)), 10.0, topk=1, horizon=6.0,
                          fade=0.3, deterministic=True)


def test_capture_takeoff_and_the_window_shift(voc):
    v, _bsp, _ = voc
    w = _windows(v)
    o = np.array([[1000.0, -1500.0, 250.0], [1000.0, -1500.0, 250.0]])
    vel = np.array([[0.0, 1500.0, 0.0], [0.0, 1500.0, 0.0]])
    lines = w.spawn([0, 1], o, vel)
    assert len(lines) == 2 and w.t1[0] == 0 and w.t2[0] == 1
    none = -np.ones((2, 8), np.int64)
    on0 = none.copy()
    on0[0, 0] = 0
    idx, _ = w.on_tick(on0, o, vel, np.zeros(2, bool))              # capture of T1 by env 0
    assert list(idx) == [0] and w.captured[0] and not w.captured[1]
    for _ in range(gr.DEPART_TICKS - 1):
        idx, _ = w.on_tick(none, o, vel, np.zeros(2, bool))
        assert len(idx) == 0
    idx, _ = w.on_tick(none, o, vel, np.zeros(2, bool))             # the takeoff
    assert list(idx) == [0] and w.t1[0] == 1 and w.t2[0] == 2 and w.prev[0] == 0
    assert w.n_capt[0] == 1 and w.tau[0] == 0
    on2 = none.copy()
    on2[1, 0] = 1                                                   # env 1 skips T1 = 0
    idx, _ = w.on_tick(on2, o, vel, np.zeros(2, bool))
    assert list(idx) == [1] and w.t1[1] == 1 and w.captured[1] and w.n_skip[1] == 1


def test_the_channel_values_move_continuously(voc):
    v, _bsp, _ = voc
    w = _windows(v)
    o = np.array([[1000.0, -1500.0, 250.0]] * 2)
    vel = np.array([[0.0, 1500.0, 0.0]] * 2)
    w.spawn([0, 1], o, vel)
    ids, vals = w.slots()
    assert np.allclose(vals[0], [0.0, 1.0, 0.5])                  # before any takeoff
    none = -np.ones((2, 8), np.int64)
    on0 = none.copy()
    on0[0, 0] = 0
    prev = w.slots()[1]
    for t in range(80):
        w.on_tick(on0 if t < 3 else none, o, vel, np.zeros(2, bool))
        cur = w.slots()[1]
        # a slot that changed target id starts from where the SAME surface's value was
        assert np.all(cur >= -1e-6) and np.all(cur <= 1.0 + 1e-6)
        prev = cur
    ids, vals = w.slots()
    assert np.allclose(vals[0], [0.0, 1.0, 0.5])                  # the fade has finished
    assert ids[0, 0] == 0 and ids[0, 1] == 1 and ids[0, 2] == 2


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
    """make_ramp_hooks records one event per window change ([row, prev, T1, T2, tau]); the POV
    render rebuilds every row's slots from them with slot_values - which must equal what the
    windows showed the policy at that row (RampWindows.slots after the previous tick)"""
    v, _bsp, _ = voc
    w = _windows(v)
    o = np.array([[1000.0, -1500.0, 250.0]] * 2)
    vel = np.array([[0.0, 1500.0, 0.0]] * 2)
    w.spawn([0, 1], o, vel)
    none = -np.ones((2, 8), np.int64)
    on0 = none.copy()
    on0[0, 0] = 0
    on1 = none.copy()
    on1[0, 0] = 1
    script = [on0] * 3 + [none] * 40 + [on1] * 5 + [none] * 60
    seen_ids, seen_vals, events = [], [], []

    def snap(row):
        e = [row, int(w.prev[0]), int(w.t1[0]), int(w.t2[0]), int(min(w.tau[0], 1 << 20))]
        if not events or events[-1][1:4] != e[1:4]:
            events.append(e)
    snap(0)
    for k, ids in enumerate(script):
        si, sv = w.slots([0])
        seen_ids.append(si[0])
        seen_vals.append(sv[0])
        idx, _ = w.on_tick(ids, o, vel, np.zeros(2, bool))
        if 0 in idx:
            snap(k + 1)
    got_i, got_v = gr.slot_values(events, len(script), w.fade_ticks)
    assert len(events) >= 3                            # spawn + two takeoffs
    assert np.array_equal(got_i, np.asarray(seen_ids))
    assert np.allclose(got_v, np.asarray(seen_vals), atol=1e-6)


def test_a_recording_without_logged_windows_is_replayed_from_its_states(voc):
    """replay_events: states resting on T1's validated contact plane capture it; leaving it for
    DEPART_TICKS takes off - the same events the live hooks write (from positions)"""
    v, _bsp, _ = voc
    w = _windows(v)
    rows = []
    for k in range(80):
        if k < 5:
            o = [1000.0, -1500.0, 250.0]                 # in the air, approaching
        elif k < 25:
            o = [1000.0 + k, -20.0, 250.0]              # on ramp 0's standing contact plane
        else:
            o = [1000.0, 1000.0, 250.0]                 # left it
        rows.append([k] + o + [0.0, 1500.0, 0.0, 0.0, 0])
    ev = gr.replay_events(w, v, np.asarray(rows, float))
    assert ev[0][2] == 0                                # T1 = ramp 0 at the spawn
    # the takeoff: DEPART_TICKS rows after the last contact (row 24), the window shifts
    assert any(e[2] == 1 and e[1] == 0 and e[0] == 25 + gr.DEPART_TICKS - 1 for e in ev), ev
