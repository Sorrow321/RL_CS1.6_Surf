"""--view-absolute {velocity,world}: ABSOLUTE view targets on top of
--view-continuous (docs/contyaw.md "Absolute targets", surfgym/view.py,
src/env.c surf_yaw_delta_abs / surf_pitch_delta_abs, train_fast.py).

What is pinned here:

(a) THE CORE, modes 1 / 2 (ABI 9, cfg.view_mode): a target is reached
    within the per-tick clamp in the expected number of ticks and then
    held; the +-180 seam takes the short way (yaw 170 -> target -170 turns
    +20, never -340); the velocity-frame base is atan2(vy, vx) in the ego
    frame write_obs uses, tracks a ROTATING velocity tick by tick under a
    row held constant, and falls back to the current yaw below 100 u/s
    (the command is then a bounded delta); the pitch target is clamped to
    [-70, 30] and approached at pitch_rate_max_deg per tick; NaN commands
    apply 0; --yaw-blend filters the applied delta and obs slot 10 echoes
    it; view_mode 0 is the delta path (the golden tests in
    test_view_continuous.py cover it on this DLL); an out-of-range mode is
    refused at surf_create.
(b) THE POLICY SIDE: off_warp anchors (+-0.5 -> +-10 deg, +-1 -> +-180),
    odd, monotone, inverse; torch == numpy for both modes; the world
    mode's norm is ignored; the pitch map is the core's [-70, 30]; the
    mixed distribution's log-prob and entropy on the 3-head layout equal a
    hand computation and the update's recomputation from the stored z; the
    Policy's tensors under each mode have the documented shapes and come
    LAST; the eval wrappers publish targets in range and greedy is the
    mean; mode None is the unchanged delta policy tensor for tensor.
(c) TRAINER SMOKES (CPU, the scratch argument set at toy size): each mode
    trains with finite losses and sane kl, evals through record_rollout,
    writes view_absolute into the config and the checkpoint carries the
    mode's tensors; --view-absolute without --view-continuous is refused,
    --bc-file under it is refused, and a checkpoint resumed under the
    wrong mode is refused.

    python -m pytest tests/python/test_view_absolute.py -q

CPU only. (a) needs the built core (SURFCORE_DLL from a worktree); (c)
needs cannonball + its prebaked goal field (SURF_TEST_MAPS=C:/RL_Surf/maps
from a worktree - the caches key on the MAIN checkout's bsp mtime).
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

_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
SKI = ROOT / "maps" / "surf_ski_2.bsp"
needs_core = pytest.mark.skipif(not (DLL.exists() and SKI.exists()),
                                reason="needs the built core + surf_ski_2")

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402

from surfgym.core import SURF_ABI_VERSION, config_to_dict      # noqa: E402
from surfgym.view import (OFF_MAX, PITCH_ABS_HALF, PITCH_ABS_MID,  # noqa: E402
                          VIEW_MODES, n_z, off_warp, off_warp_inv,
                          view_from_z_abs, view_mode_code, z_from_view_abs)
from train_fast import (LOG2PI, N_VIEW, NACT, NEUTRAL_ACT, NVEC,  # noqa: E402
                        GreedyTorchPolicy, HeadPacker, Policy,
                        SampledTorchPolicy, gauss_entropy, gauss_logp,
                        logprob_entropy_padded, logprob_entropy_view,
                        off_warp_t, sample_view, split_view, view_from_z_t)

NEUTRAL = np.array(NEUTRAL_ACT, np.int32)
RATE = 1.33            # the scratch baseline's --pitch-rate
YAW_LIM = 10.0         # yaw_rate_max_deg default


# ==========================================================================
# (a) the core
# ==========================================================================
def _core(mode, n=4, **kw):
    from surfgym import SurfCore, default_config
    cfg = default_config(num_envs=n, spawn_mode=0, max_episode_ticks=4000,
                         water_fail=1, yaw_adaptive=1, lidar_w=0, lidar_h=0,
                         pitch_rate_max_deg=RATE, yaw_jitter_deg=0.0,
                         view_mode=view_mode_code(mode), **kw)
    core = SurfCore(str(SKI), cfg, dll_path=str(DLL))
    core.reset(3)
    return core


def _place(core, yaw, vel, pitch=0.0, z_up=400.0):
    """Every env airborne z_up above the map spawn (checked empty), with
    this yaw (deg), pitch and velocity - no keys are held in the tests, so
    the horizontal velocity is constant in the air and only gravity acts."""
    st = core.get_states()
    for i in range(core.num_envs):
        s = st[i:i + 1].copy()
        s["origin"][0, 2] += z_up
        assert core.point_contents(s["origin"][0]) == -1     # CONTENTS_EMPTY
        s["velocity"][0] = np.asarray(vel, np.float32)
        s["yaw"][0] = float(yaw)
        s["pitch"][0] = float(pitch)
        s["onground"][0] = -1
        core.set_state(i, s)


def _step(core, yaw_cmd, pitch_cmd, ticks=1):
    n = core.num_envs
    a = np.ascontiguousarray(np.tile(NEUTRAL, (n, 1)))
    v = np.ascontiguousarray(np.tile(np.array([yaw_cmd, pitch_cmd], np.float32),
                                     (n, 1)))
    obs = None
    for _ in range(ticks):
        obs, _r, done, trunc, _t = core.step(a, view=v)
        assert not done.any() and not trunc.any()
    return core.states_view["yaw"].copy(), core.states_view["pitch"].copy(), obs


def test_abi_bump_and_the_config_field():
    from surfgym import default_config
    assert SURF_ABI_VERSION == 9
    assert default_config().view_mode == 0
    assert default_config(view_mode=2).view_mode == 2
    assert config_to_dict(default_config(view_mode=1))["view_mode"] == 1
    # the struct field is LAST (the ABI-8 layout with one int appended)
    from surfgym.core import SurfEnvConfig
    assert SurfEnvConfig._fields_[-1][0] == "view_mode"
    assert VIEW_MODES == {None: 0, "delta": 0, "velocity": 1, "world": 2}
    assert n_z("world") == 3 and n_z("velocity") == 2 and n_z(None) == 2


@needs_core
def test_the_dll_reports_abi_9_and_refuses_an_unknown_mode():
    from surfgym import SurfCore, default_config
    core = _core("world")
    assert int(core._lib.surf_abi_version()) == 9
    with pytest.raises(Exception):
        SurfCore(str(SKI), default_config(num_envs=1, view_mode=3),
                 dll_path=str(DLL))


@needs_core
def test_world_target_is_reached_within_the_clamp_and_held():
    core = _core("world")
    _place(core, yaw=0.0, vel=(0.0, 0.0, 0.0))
    y, _p, _o = _step(core, 37.0, 0.0, 3)
    assert np.allclose(y, 30.0)                  # 3 x the 10 deg ceiling
    y, _p, _o = _step(core, 37.0, 0.0, 1)
    assert np.allclose(y, 37.0)                  # the last 7 deg
    y, _p, o = _step(core, 37.0, 0.0, 5)
    assert np.allclose(y, 37.0)                  # held: delta 0 ...
    assert np.allclose(o[:, 10], 0.0)            # ... and slot 10 echoes 0
    # the other way round, through zero: 37 -> -20 (= 340)
    y, _p, o = _step(core, -20.0, 0.0, 5)
    assert np.allclose(y, -13.0 + 360.0)
    assert np.allclose(o[:, 10], -1.0)           # a full negative step
    y, _p, o = _step(core, -20.0, 0.0, 1)
    assert np.allclose(y, 340.0)
    assert np.allclose(o[:, 10], -7.0 / YAW_LIM)


@needs_core
def test_world_target_takes_the_short_way_across_the_seam():
    core = _core("world")
    _place(core, yaw=170.0, vel=(0.0, 0.0, 0.0))
    y, _p, o = _step(core, -170.0, 0.0, 1)       # +20 away, not -340
    assert np.allclose(y, 180.0) and np.allclose(o[:, 10], 1.0)
    y, _p, o = _step(core, -170.0, 0.0, 1)
    assert np.allclose(y, 190.0)                  # -170 stored as 190
    y, _p, o = _step(core, -170.0, 0.0, 3)
    assert np.allclose(y, 190.0)
    # and back: 190 -> 170 is -20
    y, _p, o = _step(core, 170.0, 0.0, 1)
    assert np.allclose(y, 180.0) and np.allclose(o[:, 10], -1.0)
    # through 0/360: yaw 10 -> target 350 is -20
    _place(core, yaw=10.0, vel=(0.0, 0.0, 0.0))
    y, _p, _o = _step(core, 350.0, 0.0, 1)
    assert np.allclose(y, 0.0)
    y, _p, _o = _step(core, 350.0, 0.0, 1)
    assert np.allclose(y, 350.0)
    # a target given past +-180 (e.g. 370) is the same angle as 10
    _place(core, yaw=0.0, vel=(0.0, 0.0, 0.0))
    y, _p, _o = _step(core, 370.0, 0.0, 1)
    assert np.allclose(y, 10.0)
    # exactly opposite (180 away): turns, and by the full ceiling
    _place(core, yaw=0.0, vel=(0.0, 0.0, 0.0))
    y, _p, _o = _step(core, 180.0, 0.0, 1)
    assert np.allclose(np.abs(y - 0.0) % 350.0, 10.0)


@needs_core
def test_velocity_frame_heading_is_the_ego_frame_of_the_observation():
    core = _core("velocity")
    for vel, head in (((1000.0, 0.0, 0.0), 0.0), ((0.0, 1000.0, 0.0), 90.0),
                      ((-1000.0, 0.0, 0.0), 180.0), ((0.0, -1000.0, 0.0), 270.0),
                      ((700.0, 700.0, 0.0), 45.0)):
        _place(core, yaw=head, vel=vel)
        y, _p, o = _step(core, 0.0, PITCH_ABS_MID, 1)
        assert np.allclose(y, head, atol=1e-3), (vel, y)       # offset 0: stay
        assert np.allclose(o[:, 10], 0.0, atol=1e-4)
        # looking along v: the ego-frame velocity is all forward, no left
        assert np.allclose(o[:, 0], np.hypot(vel[0], vel[1]) / 1000.0, atol=1e-3)
        assert np.allclose(o[:, 1], 0.0, atol=1e-3)
    # an offset LEADS the velocity to the left (positive yaw) by that much
    _place(core, yaw=90.0, vel=(0.0, 1000.0, 0.0))
    y, _p, _o = _step(core, 25.0, PITCH_ABS_MID, 2)
    assert np.allclose(y, 110.0)                  # clamped 10 + 10 ...
    y, _p, _o = _step(core, 25.0, PITCH_ABS_MID, 1)
    assert np.allclose(y, 115.0)                  # ... then the last 5
    y, _p, _o = _step(core, 25.0, PITCH_ABS_MID, 4)
    assert np.allclose(y, 115.0)


@needs_core
def test_velocity_frame_tracks_a_rotating_velocity_under_a_held_row():
    """The row is CONSTANT (offset +12) for the whole hold, the velocity
    heading rotates 5 deg per tick: the view must follow the frame every
    tick without a new command - the point of the velocity frame."""
    core = _core("velocity", n=2)
    off = 12.0
    _place(core, yaw=30.0 + off, vel=(1000.0 * np.cos(np.radians(30.0)),
                                      1000.0 * np.sin(np.radians(30.0)), 0.0))
    for k in range(1, 40):
        head = 30.0 + 5.0 * k
        st = core.get_states()
        for i in range(core.num_envs):
            s = st[i:i + 1].copy()
            s["velocity"][0, 0] = 1000.0 * np.cos(np.radians(head))
            s["velocity"][0, 1] = 1000.0 * np.sin(np.radians(head))
            core.set_state(i, s)
        y, _p, o = _step(core, off, PITCH_ABS_MID, 1)
        assert np.allclose(y, (head + off) % 360.0, atol=1e-3), (k, y)
        assert np.allclose(o[:, 10], 5.0 / YAW_LIM, atol=1e-4)   # 5 deg/tick
    # faster than the ceiling: 15 deg/tick of frame rotation, the view
    # lags by 5 per tick and the echo saturates at the clamp
    _place(core, yaw=0.0 + off, vel=(1000.0, 0.0, 0.0))
    head = 0.0
    for k in range(1, 4):
        head += 15.0
        st = core.get_states()
        for i in range(core.num_envs):
            s = st[i:i + 1].copy()
            s["velocity"][0, 0] = 1000.0 * np.cos(np.radians(head))
            s["velocity"][0, 1] = 1000.0 * np.sin(np.radians(head))
            core.set_state(i, s)
        y, _p, o = _step(core, off, PITCH_ABS_MID, 1)
        assert np.allclose(y, off + 10.0 * k, atol=1e-3)
        assert np.allclose(o[:, 10], 1.0)


@needs_core
def test_low_speed_fallback_is_a_bounded_delta():
    core = _core("velocity")
    # 50 u/s: below the 100 u/s floor, the base is the CURRENT yaw, so a
    # +37 command turns +10 per tick indefinitely, +3 turns +3 per tick
    _place(core, yaw=0.0, vel=(50.0, 0.0, 0.0))
    y, _p, _o = _step(core, 37.0, PITCH_ABS_MID, 5)
    assert np.allclose(y, 50.0)
    y, _p, _o = _step(core, 3.0, PITCH_ABS_MID, 4)
    assert np.allclose(y, 62.0)
    y, _p, _o = _step(core, -200.0, PITCH_ABS_MID, 1)    # wrap180(-200) = +160
    assert np.allclose(y, 72.0)
    # stationary on the platform (the map spawn, v = 0): the same
    core2 = _core("velocity")
    y0 = core2.states_view["yaw"].copy()
    y, _p, _o = _step(core2, 37.0, PITCH_ABS_MID, 2)
    assert np.allclose(y, y0 + 20.0)
    # at the floor itself the heading is the base again
    _place(core, yaw=0.0, vel=(0.0, 100.0, 0.0))
    y, _p, _o = _step(core, 0.0, PITCH_ABS_MID, 20)
    assert np.allclose(y, 90.0)
    _place(core, yaw=0.0, vel=(0.0, 99.0, 0.0))
    y, _p, _o = _step(core, 0.0, PITCH_ABS_MID, 20)
    assert np.allclose(y, 0.0)


@needs_core
@pytest.mark.parametrize("mode", ["velocity", "world"])
def test_pitch_target_is_clamped_and_approached_at_the_rate(mode):
    core = _core(mode)
    _place(core, yaw=0.0, vel=(0.0, 0.0, 0.0), pitch=0.0)
    n_up = int(np.ceil(30.0 / RATE))                       # 23 ticks
    _y, p, o = _step(core, 0.0, 30.0, n_up - 1)
    assert np.allclose(p, RATE * (n_up - 1), atol=1e-4)
    assert np.allclose(o[:, 11], 1.0)                      # a full step
    _y, p, o = _step(core, 0.0, 30.0, 1)
    assert np.allclose(p, 30.0, atol=1e-5)
    _y, p, o = _step(core, 0.0, 30.0, 3)
    assert np.allclose(p, 30.0) and np.allclose(o[:, 11], 0.0)
    # above the view's own ceiling: clamped to 30, no push against it
    _y, p, o = _step(core, 0.0, 40.0, 3)
    assert np.allclose(p, 30.0) and np.allclose(o[:, 11], 0.0)
    # all the way down to -70, at most RATE per tick, then held; -80 is -70
    n_dn = int(np.ceil(100.0 / RATE))                      # 76 ticks
    _y, p, o = _step(core, 0.0, -80.0, n_dn - 1)
    assert p.min() > -70.0 and np.allclose(o[:, 11], -1.0)
    _y, p, o = _step(core, 0.0, -80.0, 1)
    assert np.allclose(p, -70.0, atol=1e-4)
    _y, p, o = _step(core, 0.0, -80.0, 2)
    assert np.allclose(p, -70.0) and np.allclose(o[:, 11], 0.0)
    # the policy's map covers exactly that range
    z = np.array([[0.0, 30.0], [0.0, -30.0], [0.0, 0.0]])
    v = view_from_z_abs(z if mode == "velocity" else
                        np.concatenate([z[:, :1], z], 1), mode)
    assert np.allclose(v[:, 1], [30.0, -70.0, -20.0], atol=1e-4)


@needs_core
@pytest.mark.parametrize("mode", ["velocity", "world"])
def test_nan_commands_apply_zero(mode):
    core = _core(mode)
    _place(core, yaw=33.0, vel=(1000.0, 0.0, 0.0), pitch=-5.0)
    y, p, o = _step(core, np.nan, np.nan, 3)
    assert np.allclose(y, 33.0) and np.allclose(p, -5.0)
    assert np.allclose(o[:, 10], 0.0) and np.allclose(o[:, 11], 0.0)
    y, p, o = _step(core, np.inf, -np.inf, 2)
    assert np.allclose(y, 33.0) and np.allclose(p, -5.0 - 2 * RATE, atol=1e-4)


@needs_core
def test_yaw_blend_filters_the_applied_delta_and_slot_10_echoes_it():
    core = _core("world", yaw_blend=0.5)
    _place(core, yaw=0.0, vel=(0.0, 0.0, 0.0))
    y, _p, o = _step(core, 90.0, PITCH_ABS_MID, 1)
    assert np.allclose(y, 5.0) and np.allclose(o[:, 10], 0.5)   # 0.5*10 + 0.5*0
    y, _p, o = _step(core, 90.0, PITCH_ABS_MID, 1)
    assert np.allclose(y, 12.5) and np.allclose(o[:, 10], 0.75)  # 0.5*10 + 0.5*5


@needs_core
def test_mode_zero_is_the_delta_command():
    """view_mode 0 on this DLL is the ABI-8 delta path: a K command of 0
    leaves the yaw alone whatever the velocity, and a 20 deg pitch COMMAND
    is a per-tick rate clamped to the outermost bin - not a target."""
    core = _core(None)
    _place(core, yaw=10.0, vel=(0.0, 1000.0, 0.0), pitch=0.0)
    y, p, _o = _step(core, 0.0, 20.0, 3)
    assert np.allclose(y, 10.0) and np.allclose(p, 3 * RATE, atol=1e-4)


# ==========================================================================
# (b) the policy side
# ==========================================================================
def test_off_warp_anchors_and_inverse():
    assert off_warp(0.5) == pytest.approx(10.0, abs=1e-12)
    assert off_warp(-0.5) == pytest.approx(-10.0, abs=1e-12)
    assert off_warp(1.0) == pytest.approx(OFF_MAX, abs=1e-12)
    assert off_warp(0.0) == 0.0
    u = np.linspace(-1, 1, 2001)
    off = off_warp(u)
    assert np.all(np.diff(off) > 0)                        # monotone
    assert np.allclose(off_warp(-u), -off)                 # odd
    assert np.allclose(off_warp_inv(off), u, atol=1e-9)    # inverse
    # resolution: ~3.5 deg per unit u at zero, 60 at +-10 deg, >1000 at the end
    d0 = (off_warp(0.001) - off_warp(0.0)) / 0.001
    d5 = (off_warp(0.501) - off_warp(0.5)) / 0.001
    d1 = (off_warp(1.0) - off_warp(0.999)) / 0.001
    assert 3.4 < d0 < 3.7 and 55 < d5 < 65 and d1 > 900
    # documented inverse: u = sign(off) ln(1 + |off|/180 (e^b - 1)) / b
    b = 2.0 * np.log(17.0)
    assert off_warp_inv(10.0) == pytest.approx(np.log1p(10.0 / 180.0 * np.expm1(b)) / b)
    assert off_warp_inv(10.0) == pytest.approx(0.5, abs=1e-12)


def test_torch_matches_numpy_for_both_modes_and_the_delta_default_is_unchanged():
    from surfgym.view import view_from_z
    g = torch.Generator().manual_seed(2)
    u = torch.linspace(-1, 1, 4001, dtype=torch.float64)
    assert torch.allclose(off_warp_t(u), torch.as_tensor(off_warp(u.numpy())),
                          atol=1e-12)
    z2 = torch.randn(256, 2, dtype=torch.float64, generator=g) * 2
    z3 = torch.randn(256, 3, dtype=torch.float64, generator=g) * 2
    v = view_from_z_t(z2, RATE, "velocity").numpy()
    assert np.allclose(v, view_from_z_abs(z2.numpy(), "velocity"), atol=1e-4)
    assert v.dtype == np.float32
    w = view_from_z_t(z3, RATE, "world").numpy()
    assert np.allclose(w, view_from_z_abs(z3.numpy(), "world"), atol=1e-4)
    for a in (v, w):
        assert np.all(np.abs(a[:, 0]) <= 180.0 + 1e-4)
        assert np.all(a[:, 1] >= -70.0 - 1e-4) and np.all(a[:, 1] <= 30.0 + 1e-4)
    # the delta default is the pre-change map, bit for bit
    d = view_from_z_t(z2, RATE).numpy()
    assert np.array_equal(d, view_from_z_t(z2, RATE, None).numpy())
    assert np.allclose(d, view_from_z(z2.numpy(), RATE), atol=1e-5)
    # world: the norm is ignored - (2, 2) and (0.5, 0.5) both point at 45 deg,
    # (-1, 0) at 180, (0, -3) at -90; the pitch map is -20 + 50 tanh z
    zz = torch.tensor([[2.0, 2.0, 0.0], [0.5, 0.5, 0.0], [-1.0, 0.0, 30.0],
                       [0.0, -3.0, -30.0]], dtype=torch.float64)
    ww = view_from_z_t(zz, RATE, "world").numpy()
    assert np.allclose(ww[:, 0], [45.0, 45.0, 180.0, -90.0], atol=1e-4)
    assert np.allclose(ww[:, 1], [-20.0, -20.0, 30.0, -70.0], atol=1e-4)
    # velocity: u = +-0.5 (z = atanh 0.5) is +-10 deg, z = 0 is "look along v"
    zv = torch.tensor([[np.arctanh(0.5), 0.0], [-np.arctanh(0.5), 0.0],
                       [0.0, 0.0], [40.0, 0.0]], dtype=torch.float64)
    vv = view_from_z_t(zv, RATE, "velocity").numpy()
    assert np.allclose(vv[:, 0], [10.0, -10.0, 0.0, 180.0], atol=1e-4)
    # z_from_view_abs is a preimage of view_from_z_abs in both modes
    for mode, zz_ in (("velocity", z2.numpy()), ("world", z3.numpy())):
        tgt = view_from_z_abs(zz_, mode)
        back = view_from_z_abs(z_from_view_abs(tgt, mode), mode)
        ok = np.abs(tgt[:, 0]) < 179.0            # the +-180 seam is one angle
        assert np.allclose(back[ok, 0], tgt[ok, 0], atol=0.2)
        inner = (tgt[:, 1] > -69.0) & (tgt[:, 1] < 29.0)
        assert np.allclose(back[inner, 1], tgt[inner, 1], atol=0.2)
    with pytest.raises(ValueError):
        view_from_z_t(z2, RATE, "bogus")


def _hand_logp(padded, act, mu, log_std, z):
    lsm = F.log_softmax(padded, dim=-1)
    lp = 0.0
    for h in range(N_VIEW, NACT):
        lp = lp + lsm[torch.arange(len(act)), h, act[:, h]]
    sig = log_std.exp()
    for j in range(mu.shape[1]):
        lp = lp - 0.5 * ((z[:, j] - mu[:, j]) / sig[j]) ** 2 - log_std[j] \
            - 0.5 * LOG2PI
    return lp


@pytest.mark.parametrize("nz", [2, 3])
def test_mixed_distribution_logp_and_entropy_on_the_new_head_layouts(nz):
    torch.manual_seed(4)
    pk = HeadPacker(torch.device("cpu"))
    B = 193
    logits = torch.randn(B, sum(NVEC) + nz)
    cat, mu = split_view(logits)
    assert cat.shape == (B, sum(NVEC)) and mu.shape == (B, nz)
    padded = pk.pad(cat)
    log_std = torch.tensor([-1.2, -0.7, -0.9][:nz])
    act, z, logp = sample_view(padded, mu, log_std)
    assert act.shape == (B, NACT) and z.shape == (B, nz)
    assert torch.all(act[:, 0] == NEUTRAL_ACT[0]) and torch.all(act[:, 1] == NEUTRAL_ACT[1])
    want = _hand_logp(padded, act, mu, log_std, z)
    assert torch.allclose(logp, want, atol=1e-5)
    logp2, ent = logprob_entropy_view(padded, act, mu, log_std, z)
    assert torch.equal(logp2, logprob_entropy_padded(padded[:, N_VIEW:],
                                                     act[:, N_VIEW:])[0]
                       + gauss_logp(z, mu, log_std))
    assert torch.allclose(logp2, logp, atol=1e-5)
    _, ent_c = logprob_entropy_padded(padded[:, N_VIEW:], act[:, N_VIEW:])
    assert torch.allclose(ent, ent_c + gauss_entropy(log_std))
    want_h = sum(0.5 * np.log(2 * np.pi * np.e * float(s) ** 2)
                 for s in log_std.exp())
    assert float(gauss_entropy(log_std)) == pytest.approx(want_h, abs=1e-6)
    z_big = sample_view(pk.pad(torch.zeros(20000, sum(NVEC))),
                        torch.zeros(20000, nz), log_std)[1]
    assert z_big.std(0).numpy() == pytest.approx(log_std.exp().numpy(), rel=0.05)


def _tiny(view=True, absolute=None, seed=0):
    torch.manual_seed(seed)
    return Policy(15 + 8 * 4, 8, 4, emb=16, hidden=16, view_continuous=view,
                  view_absolute=absolute)


def test_policy_tensors_per_mode_come_last_and_none_is_the_delta_policy():
    d = _tiny(view=True, absolute=None, seed=5)
    v = _tiny(view=True, absolute="velocity", seed=5)
    w = _tiny(view=True, absolute="world", seed=5)
    assert (d.view_absolute, v.view_absolute, w.view_absolute) == (None, "velocity", "world")
    assert (d.n_z, v.n_z, w.n_z) == (2, 2, 3)
    assert _tiny(view=False).n_z == 0
    # velocity mode: the same tensors as the delta policy (same shapes,
    # same RNG draws) - only the env's reading of z differs
    for k, t in d.state_dict().items():
        assert torch.equal(t, v.state_dict()[k]), k
    assert v.view_head.weight.shape == (2, 16) and v.view_std.log_std.shape == (2,)
    # world mode: a third head, last in parameters(), init log 0.3
    assert w.view_head.weight.shape == (3, 16) and w.view_std.log_std.shape == (3,)
    names = [n for n, _ in w.named_parameters()]
    assert names[-3:] == ["view_head.weight", "view_head.bias", "view_std.log_std"]
    assert torch.allclose(w.view_std.log_std, torch.full((3,), float(np.log(0.3))))
    x = torch.randn(3, 15 + 32)
    lw, _ = w(x)
    assert lw.shape == (3, sum(NVEC) + 3)
    assert split_view(lw)[1].shape == (3, 3)
    # the shared tensors are the discrete policy's, key for key
    disc = _tiny(view=False, seed=5)
    for k, t in disc.state_dict().items():
        assert torch.equal(t, w.state_dict()[k]), k
    with pytest.raises(SystemExit):
        _tiny(view=False, absolute="velocity")
    with pytest.raises(SystemExit):
        _tiny(view=True, absolute="polar")


class _Core:
    class _Cfg:
        pitch_rate_max_deg = RATE
    config = _Cfg()


@pytest.mark.parametrize("mode", ["velocity", "world"])
def test_eval_wrappers_publish_targets_in_range_and_greedy_is_the_mean(mode):
    p = _tiny(absolute=mode)
    p.eval()
    pk = HeadPacker(torch.device("cpu"))
    obs = np.random.default_rng(0).standard_normal((5, 15 + 32)).astype(np.float32)
    g = GreedyTorchPolicy(p, pk, torch.device("cpu"), None, _Core(), 4)
    a = g.act(obs)
    assert a.shape == (5, 6) and a.dtype == np.int32
    assert np.all(a[:, 0] == NEUTRAL_ACT[0]) and np.all(a[:, 1] == NEUTRAL_ACT[1])
    assert g.view_absolute == mode
    assert g.view is not None and g.view.shape == (5, 2) and g.view.dtype == np.float32
    with torch.no_grad():
        lg, _ = p(torch.as_tensor(obs))
    _, mu = split_view(lg)
    assert mu.shape == (5, n_z(mode))
    assert np.allclose(g.view, view_from_z_t(mu, RATE, mode).numpy(), atol=1e-6)
    assert np.all(np.abs(g.view[:, 0]) <= 180.0)
    assert np.all(g.view[:, 1] >= -70.0) and np.all(g.view[:, 1] <= 30.0)
    v0 = g.view.copy()
    g.act(obs)
    assert np.array_equal(g.view, v0)                 # the act_every hold
    s = SampledTorchPolicy(p, pk, torch.device("cpu"), None, _Core(), 1)
    s.act(obs)
    assert s.view.shape == (5, 2) and np.isfinite(s.view).all()
    assert np.all(np.abs(s.view[:, 0]) <= 180.0)
    assert np.all(s.view[:, 1] >= -70.0) and np.all(s.view[:, 1] <= 30.0)


# ==========================================================================
# (c) trainer smokes: the scratch argument set at toy size
# ==========================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_view_continuous import SMOKE_FLAGS, CANNONBALL, needs_run  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"
RECORD = ROOT / "tools" / "record_ckpt.py"


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


def _train(run, extra, steps="6144"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE_FLAGS
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


@needs_run
@pytest.mark.parametrize("mode", ["velocity", "world"])
def test_each_mode_trains_evals_and_records_on_cpu(mode):
    # the smoke really runs without the GPU (the live trainer owns it)
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    run = f"cya_smoke_{mode}"
    r = _train(run, ["--view-continuous", "--view-absolute", mode])
    assert f"--view-absolute {mode}: the view row is an ABSOLUTE target" in r.stdout
    assert "sig 0.3" in r.stdout
    assert "bake" not in r.stdout.lower() or "goal_32" in r.stdout   # no re-bake
    d = ROOT / "runs" / run
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["view_continuous"] == 1 and c["view_absolute"] == mode
    rows = _csv(run)
    assert len(rows) == 3
    for x in rows:
        for k in ("train/loss", "train/value_loss", "train/approx_kl",
                  "train/entropy_loss"):
            assert np.isfinite(float(x[k])), k
        # the k1 estimator can read a hair negative on a first iteration
        assert abs(float(x["train/approx_kl"])) < 0.05
    sd = torch.load(d / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    nz = n_z(mode)
    assert sd["config"]["view_absolute"] == mode
    assert sd["policy"]["view_head.weight"].shape == (nz, 64)
    assert sd["policy"]["view_std.log_std"].shape == (nz,)
    assert torch.all((sd["policy"]["view_std.log_std"].exp() - 0.3).abs() < 0.03)
    tr = sorted(d.glob("traj_*.jsonl"))
    assert tr and len(tr[0].read_text(encoding="utf-8").splitlines()) > 100
    # record_ckpt mirrors the mode: the core gets view_mode, the policy the
    # third head in world mode, and the recording runs to a trajectory
    out = d / "rec_smoke.jsonl"
    rr = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
               "--map", str(CANNONBALL), "--episodes", "1", "--out", str(out)],
              timeout=900)
    assert rr.returncode == 0, rr.stdout[-3000:] + rr.stderr[-3000:]
    assert f"--view-absolute {mode}: the view row is an absolute" in rr.stdout
    assert f"core view_mode {n_z(mode) - 1}" in rr.stdout   # 1 velocity, 2 world
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 50
    head = json.loads(lines[0])
    assert head["map"] == "surf_src_cannonball"
    # a resume WITHOUT the flag restores the mode from the checkpoint
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run + "_re"]
              + SMOKE_FLAGS + ["--steps", "8192", "--ckpt",
                               str(d / "ckpt_final.pt")], timeout=1800)
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert f"view_absolute={mode}" in r2.stdout
    c2 = json.loads((ROOT / "runs" / (run + "_re") / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert c2["view_absolute"] == mode
    # ... and a resume under the OTHER mode is refused
    other = "world" if mode == "velocity" else "velocity"
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", run + "_bad"]
              + SMOKE_FLAGS + ["--steps", "8192", "--ckpt",
                               str(d / "ckpt_final.pt"), "--view-continuous",
                               "--view-absolute", other], timeout=900)
    assert r3.returncode != 0 and "view_absolute" in (r3.stdout + r3.stderr)
    for extra in ("_re", "_bad"):
        shutil.rmtree(ROOT / "runs" / (run + extra), ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)


@needs_run
def test_the_flag_is_refused_without_view_continuous_and_with_bc():
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_bad1"]
             + SMOKE_FLAGS + ["--steps", "2048", "--view-absolute", "velocity"],
             timeout=600)
    assert r.returncode != 0 and "needs --view-continuous" in (r.stdout + r.stderr)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_bad2"]
             + SMOKE_FLAGS + ["--steps", "2048", "--view-continuous",
                              "--view-absolute", "world", "--bc-file",
                              "nonexistent.npz"], timeout=600)
    assert r.returncode != 0 and "--bc-file is not implemented under --view-absolute" \
        in (r.stdout + r.stderr)
    for run in ("cya_bad1", "cya_bad2"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
