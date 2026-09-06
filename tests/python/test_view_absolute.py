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
    a delta-mode BC file under it is refused, and a checkpoint resumed
    under the wrong mode is refused.
(d) PITCH HEAD DISCIPLINE (--pitch-entropy, PITCH_LOG_STD_MAX_ABS): the
    entropy of the mixed distribution drops the pitch head's term at 0.0
    and is the old expression op for op at 1.0 (the log-prob is the same
    joint either way); an absolute-mode Policy caps the pitch head's log
    sigma at log 0.5 through a NON-persistent buffer (no new state_dict
    key, a checkpoint from before the cap loads) and project_log_std pulls
    the raw parameter under the cap so its gradient stays live; a CPU
    smoke under a large entropy coefficient shows the yaw sigma climbing
    in both arms while the pitch sigma climbs only with --pitch-entropy 1,
    a longer one drives the pitch sigma INTO the cap (0.500 exactly, the
    raw parameter at log 0.5); and the delta mode of this trainer is
    bit-identical to the trainer before the cap existed.
(e) THE TOOLS UNDER THE ABSOLUTE MODES: an absolute line round-trips
    through surfgym.bc.replay_line on a view_mode-1 core to the same
    state; plan_to_bc reads the npz's view_mode and refuses a mix; the
    BC dataset takes an absolute file's targets (z_from_view_abs, 2- or
    3-wide moments) and refuses a file of another mode or a discrete one;
    the planner's edits (branch draws, grids, the analytic macro) are
    degrees on a target, wrapped, in both absolute modes.

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
from train_fast import (LOG2PI, LOG_STD_MAX, N_VIEW, NACT, NEUTRAL_ACT,  # noqa: E402
                        NVEC, PITCH_LOG_STD_MAX_ABS, GreedyTorchPolicy,
                        HeadPacker, Policy, SampledTorchPolicy,
                        gauss_entropy, gauss_logp, logprob_entropy_padded,
                        logprob_entropy_view, off_warp_t, sample_view,
                        split_view, view_from_z_t)

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
def test_the_flag_is_refused_without_view_continuous_and_a_delta_bc_file_is_refused(tmp_path):
    for run in ("cya_bad1", "cya_bad2"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_bad1"]
             + SMOKE_FLAGS + ["--steps", "2048", "--view-absolute", "velocity"],
             timeout=600)
    assert r.returncode != 0 and "needs --view-continuous" in (r.stdout + r.stderr)
    # --bc-file IS supported under the mode, but a file whose view rows are
    # delta commands (a plan_to_bc run on a delta checkpoint: no
    # view_absolute key) is refused by BCDataset before anything trains
    f = tmp_path / "delta_bc.npz"
    _write_bc(f, n=8, view_absolute=None)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_bad2"]
             + SMOKE_FLAGS + ["--steps", "2048", "--view-continuous",
                              "--view-absolute", "velocity", "--bc-file",
                              str(f)], timeout=600)
    assert r.returncode != 0
    assert "view rows are (K, pitch deg/tick)" in (r.stdout + r.stderr)
    for run in ("cya_bad1", "cya_bad2"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _write_bc(path, n: int = 8, view_absolute=None, nz=None, moments=True):
    """A tiny --view-continuous BC file: zero states, neutral keys, a
    finite view column of the given mode and (n, nz) z moments."""
    from surfgym.bc import save_bc_dataset
    from surfgym.core import STATE_DTYPE
    rng = np.random.default_rng(3)
    states = np.zeros(n, STATE_DTYPE)
    scal = np.zeros((n, 15), np.float32)
    latch = np.zeros(n, np.float32)
    acts = np.zeros((n, 6), np.int64)
    acts[:, 0], acts[:, 1] = NEUTRAL_ACT[0], NEUTRAL_ACT[1]
    acts[:, 2:4] = 1
    if view_absolute is None:
        view = np.stack([rng.uniform(-3, 3, n), rng.uniform(-1.3, 1.3, n)],
                        1).astype(np.float32)
    else:
        view = np.stack([rng.uniform(-40, 40, n), rng.uniform(-60, 20, n)],
                        1).astype(np.float32)
    nz = nz or n_z(view_absolute)
    zmu = rng.standard_normal((n, nz)).astype(np.float32)
    zsd = np.abs(rng.standard_normal((n, nz))).astype(np.float32)
    meta = {"obs_reward": False, "n_latch": 0, "act_every": 4,
            "view_continuous": 1}
    if view_absolute:
        meta["view_absolute"] = view_absolute
    save_bc_dataset(path, states, scal, latch, acts, np.ones(n, np.float32),
                    np.zeros(n, np.int32), meta, view=view,
                    view_zmu=(zmu if moments else None),
                    view_zsd=(zsd if moments else None))
    return view, zmu, zsd


# ==========================================================================
# (d) pitch head discipline: --pitch-entropy and the log sigma cap
# ==========================================================================
@pytest.mark.parametrize("nz", [2, 3])
def test_pitch_entropy_scales_only_the_pitch_heads_entropy(nz):
    torch.manual_seed(6)
    pk = HeadPacker(torch.device("cpu"))
    logits = torch.randn(64, sum(NVEC) + nz)
    cat, mu = split_view(logits)
    padded = pk.pad(cat)
    ls = torch.tensor([-1.2, -0.7, -0.9][:nz])
    act, z, _ = sample_view(padded, mu, ls)
    lp1, e1 = logprob_entropy_view(padded, act, mu, ls, z)
    lp0, e0 = logprob_entropy_view(padded, act, mu, ls, z, 0.0)
    lph, eh = logprob_entropy_view(padded, act, mu, ls, z, 0.5)
    _, ec = logprob_entropy_padded(padded[:, N_VIEW:], act[:, N_VIEW:])
    # the log-prob is the same joint whatever the pitch entropy weight
    assert torch.equal(lp1, lp0) and torch.equal(lp1, lph)
    # 1.0 is the pre-flag expression, op for op
    assert torch.equal(e1, ec + gauss_entropy(ls))
    assert torch.allclose(e0, ec + gauss_entropy(ls[:-1]))
    assert torch.allclose(eh, ec + gauss_entropy(ls[:-1])
                          + 0.5 * gauss_entropy(ls[-1:]))
    # and the entropy's gradient on the pitch log sigma is exactly zero at
    # 0.0 (1 on every yaw head: dH/dlog sigma_j = 1), half at 0.5
    for w, want in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
        lsg = ls.clone().requires_grad_(True)
        _, e = logprob_entropy_view(padded, act, mu, lsg, z, w)
        e.mean().backward()
        assert torch.all(lsg.grad[:-1] == 1.0)
        assert float(lsg.grad[-1]) == pytest.approx(want)


def test_absolute_policy_caps_the_pitch_log_sigma_and_projects_the_raw():
    d = _tiny(view=True, absolute=None, seed=5)
    v = _tiny(view=True, absolute="velocity", seed=5)
    w = _tiny(view=True, absolute="world", seed=5)
    # the cap is a NON-persistent buffer: no new state_dict key, so a
    # checkpoint saved before the cap existed loads key for key
    assert set(v.state_dict()) == set(d.state_dict()) == set(w.state_dict())
    assert not hasattr(d.view_std, "log_std_hi")
    assert torch.allclose(v.view_std.log_std_hi,
                          torch.tensor([LOG_STD_MAX, PITCH_LOG_STD_MAX_ABS]))
    assert torch.allclose(w.view_std.log_std_hi,
                          torch.tensor([LOG_STD_MAX, LOG_STD_MAX,
                                        PITCH_LOG_STD_MAX_ABS]))
    assert PITCH_LOG_STD_MAX_ABS == pytest.approx(np.log(0.5))
    # at init nothing binds: log 0.3 on every head in every mode
    for p in (d, v, w):
        assert torch.allclose(p.log_std(), torch.full((p.n_z,), float(np.log(0.3))))
    with torch.no_grad():
        for p in (d, v, w):
            p.view_std.log_std.fill_(3.0)
    assert torch.equal(d.log_std(), torch.full((2,), LOG_STD_MAX))
    assert torch.allclose(v.log_std(), torch.tensor([LOG_STD_MAX, PITCH_LOG_STD_MAX_ABS]))
    assert torch.allclose(w.log_std(), torch.tensor([LOG_STD_MAX, LOG_STD_MAX,
                                                     PITCH_LOG_STD_MAX_ABS]))
    assert float(v.log_std()[1].detach().exp()) == pytest.approx(0.5)
    # the projection pulls the RAW parameter under every head's ceiling
    # (so its gradient stays alive at the clamp); a no-op in delta mode
    assert d.project_log_std() == 0 and torch.all(d.view_std.log_std == 3.0)
    assert v.project_log_std() == 2
    assert torch.allclose(v.view_std.log_std,
                          torch.tensor([LOG_STD_MAX, PITCH_LOG_STD_MAX_ABS]))
    assert v.project_log_std() == 0
    v.log_std().sum().backward()
    assert torch.equal(v.view_std.log_std.grad, torch.ones(2))   # live AT the cap
    # a checkpoint from before the cap (raw pitch log sigma 3.0) loads into
    # the velocity policy key for key and reads 0.5 through the cap
    v2 = _tiny(view=True, absolute="velocity", seed=1)
    v2.load_state_dict(d.state_dict())
    assert torch.all(v2.view_std.log_std == 3.0)
    assert float(v2.log_std()[1].detach()) == pytest.approx(PITCH_LOG_STD_MAX_ABS)
    assert v2.project_log_std() == 2


def _last_sig(stdout: str):
    """The 'sig y/p' of the LAST step line."""
    sig = None
    for ln in stdout.splitlines():
        if ln.startswith("step ") and "  sig " in ln:
            sig = ln.split("  sig ")[1].split()[0]
    assert sig is not None, stdout[-2000:]
    return [float(x) for x in sig.split("/")]


@needs_run
def test_pitch_sigma_stays_put_without_its_entropy_term_and_the_cap_holds():
    """Velocity-mode smokes under a LARGE entropy coefficient and lr. With
    --pitch-entropy 1 (the old arithmetic) both sigmas climb over 24 Adam
    steps; by default only the yaw sigma climbs and the pitch sigma ends
    within noise of its start. A longer run at a higher lr drives the
    pitch sigma INTO the cap: 0.500 on the step line, and the checkpoint's
    RAW pitch log sigma sits AT log 0.5 (project_log_std), never above."""
    common = ["--view-continuous", "--view-absolute", "velocity",
              "--ent", "0.5", "--lr", "1e-2"]
    r_ctl = _train("cya_pe_ctl", common + ["--pitch-entropy", "1"], steps="24576")
    r_trt = _train("cya_pe_trt", common, steps="24576")
    assert "--pitch-entropy 1: the pitch head's entropy term is scaled by 1" in r_ctl.stdout
    assert "--pitch-entropy 0: the pitch head's entropy term is OFF" in r_trt.stdout
    ls, sig = {}, {}
    for run in ("cya_pe_ctl", "cya_pe_trt"):
        sd = torch.load(ROOT / "runs" / run / "ckpt_final.pt",
                        map_location="cpu", weights_only=False)
        ls[run] = sd["policy"]["view_std.log_std"].clone()
        c = json.loads((ROOT / "runs" / run / "run.json").read_text(
            encoding="utf-8"))["config"]
        assert c["pitch_entropy"] == (1.0 if run.endswith("ctl") else 0.0)
        assert len(_csv(run)) == 12
    sig["ctl"], sig["trt"] = _last_sig(r_ctl.stdout), _last_sig(r_trt.stdout)
    start = float(np.log(0.3))
    d = {k: (v - start).tolist() for k, v in ls.items()}      # log sigma moved
    # 24 Adam steps at lr 1e-2 under a dominant entropy gradient: about
    # +0.2 in log sigma on every head that still has the term (measured
    # +0.196 yaw / +0.209 pitch); the yaw head behaves the same in both
    # arms (+0.196 / +0.196); the pitch head WITHOUT its term still
    # drifts a little from the policy gradient alone (measured +0.085,
    # 2.5x slower) - so the assertion is against the control, not zero
    assert d["cya_pe_ctl"][0] > 0.15 and d["cya_pe_ctl"][1] > 0.15
    assert d["cya_pe_trt"][0] > 0.15                          # yaw: as before
    assert abs(d["cya_pe_trt"][0] - d["cya_pe_ctl"][0]) < 0.03
    assert d["cya_pe_trt"][1] < d["cya_pe_ctl"][1] - 0.08    # pitch: no bonus
    assert d["cya_pe_trt"][1] < 0.12
    assert sig["ctl"][1] > 0.35 and sig["trt"][1] < 0.34
    assert abs(sig["ctl"][0] - sig["trt"][0]) < 0.01
    # the cap: 64 steps at lr 2e-2 with the term on pushes the pitch log
    # sigma +1.28 past its start, i.e. through log 0.5 (needs +0.51)
    r_cap = _train("cya_pe_cap", common + ["--pitch-entropy", "1", "--lr", "2e-2"],
                   steps="65536")
    sd = torch.load(ROOT / "runs" / "cya_pe_cap" / "ckpt_final.pt",
                    map_location="cpu", weights_only=False)
    raw = sd["policy"]["view_std.log_std"]
    assert float(raw[1]) == pytest.approx(PITCH_LOG_STD_MAX_ABS, abs=1e-6)
    assert float(raw[0]) <= LOG_STD_MAX + 1e-6 and float(raw[0]) > start + 0.5
    assert _last_sig(r_cap.stdout)[1] == pytest.approx(0.5, abs=1e-3)
    # a resume of that checkpoint keeps the mode AND the (restored) pitch entropy
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_pe_cap_re"]
              + SMOKE_FLAGS + ["--steps", "67584", "--ckpt",
                               str(ROOT / "runs" / "cya_pe_cap" / "ckpt_final.pt")],
              timeout=1800)
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "pitch_entropy=1" in r2.stdout and "view_absolute=velocity" in r2.stdout
    for run in ("cya_pe_ctl", "cya_pe_trt", "cya_pe_cap", "cya_pe_cap_re"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _prepitch_trainer(dst: Path):
    """The train_fast.py of the last first-parent commit that has
    --view-absolute but not --pitch-entropy (the trainer before the cap),
    or None. BYTES: the console is cp1251 and the file is UTF-8."""
    try:
        r = subprocess.run(["git", "rev-list", "--first-parent",
                            "--max-count=200", "HEAD"],
                           capture_output=True, text=True, cwd=str(ROOT),
                           timeout=60)
        refs = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        refs = []
    # HEAD itself first: with the cap uncommitted, HEAD's trainer IS the
    # pre-cap one; once committed it carries the flag and is skipped
    for ref in (tuple(refs) or ("HEAD", "HEAD^")):
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode != 0:
            continue
        if b"--pitch-entropy" in r.stdout:
            continue
        if b"--view-absolute" not in r.stdout:
            return None
        dst.write_bytes(r.stdout)
        return ref
    return None


def _assert_runs_identical(a: Path, b: Path):
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert "pitch_entropy" not in ca and "view_absolute" not in ca
    assert ca == cb
    ra, rb = _csv(a.name), _csv(b.name)
    assert len(ra) == len(rb) == 3
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
def test_delta_mode_is_bit_identical_to_the_trainer_before_the_pitch_cap():
    """--view-continuous alone (the delta command) on this trainer and on
    the trainer of the commit before --pitch-entropy / the cap: same
    config (no pitch_entropy key), same progress.csv (fps excluded), same
    eval trajectory, same weights, same Adam moments. The pre-cap copy
    lives in python/ so its imports resolve exactly as the patched
    file's do."""
    old = ROOT / "python" / "train_fast_prepitch.py"
    ref = _prepitch_trainer(old)
    if ref is None:
        pytest.skip("no pre-cap train_fast.py in the first-parent history")
    try:
        _train("cya_pe_new", ["--view-continuous"])
        shutil.rmtree(ROOT / "runs" / "cya_pe_old", ignore_errors=True)
        r = _run([sys.executable, "-u", str(old), "--run", "cya_pe_old"]
                 + SMOKE_FLAGS + ["--steps", "6144", "--view-continuous"])
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        old.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "cya_pe_new", ROOT / "runs" / "cya_pe_old"
    _assert_runs_identical(a, b)
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)


# ==========================================================================
# (e) the tools under the absolute modes
# ==========================================================================
@needs_core
def test_absolute_line_round_trips_through_replay_line_on_a_view_mode_core(tmp_path):
    from surfgym.bc import replay_line
    core = _core("velocity", n=1)
    k = 4
    obs0 = core.reset(11).copy()
    spawn = core.get_states()[0].copy()
    rng = np.random.default_rng(5)
    D = 120
    acts = np.zeros((D, 6), np.int64)
    acts[:, 0], acts[:, 1] = NEUTRAL_ACT[0], NEUTRAL_ACT[1]
    acts[:, 2] = 2
    acts[:, 3] = rng.integers(0, 3, D)
    z = rng.standard_normal((D, 2)) * 0.5
    view = view_from_z_abs(z, "velocity")          # targets: offset deg, pitch deg
    assert np.all(np.abs(view[:, 0]) <= 180) and np.all(view[:, 1] >= -70)
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
             view_continuous=np.int32(1), view_mode=np.int32(1),
             finish_ticks=np.int32(0), gate_seed=np.int32(11),
             map=np.str_(str(SKI)), greedy_ticks=np.int32(0))
    zz = np.load(npz, allow_pickle=False)
    assert int(zz["view_mode"]) == 1
    views_out = []
    rows, _ts, _fin, ticks = replay_line(core, zz["spawn_state"][0],
                                         zz["obs_start"], zz["acts"], k,
                                         view=zz["view"], views_out=views_out)
    assert len(rows) == len(views_out) and np.allclose(views_out[0], view[0])
    if end is None:
        assert ticks == D * k and len(rows) == D
        assert core.get_states()[0].tobytes() == st_end.tobytes()
    else:
        assert len(rows) == end + 1
    # plan_to_bc reads the mode and refuses a mix of modes / a wrong checkpoint
    import plan_to_bc as pb
    head = pb.load_plans([npz])
    assert head["view_mode"] == 1 and head["view_continuous"]
    npz2 = tmp_path / "other.npz"
    np.savez(npz2, acts=acts.astype(np.int8), view=view, act_every=np.int32(k),
             spawn_state=spawn.reshape(1), obs_start=obs0[0],
             view_continuous=np.int32(1), view_mode=np.int32(0),
             finish_ticks=np.int32(0), gate_seed=np.int32(11),
             map=np.str_(str(SKI)), greedy_ticks=np.int32(0))
    with pytest.raises(SystemExit, match="view_mode"):
        pb.load_plans([npz, npz2])
    assert pb.load_plans([npz2])["view_mode"] == 0     # no key = delta, too
    # the z targets of the line's rows: the inverse warp, round-tripping
    from surfgym.view import z_from_view_any, view_from_z_any
    zt = z_from_view_any(view, "velocity", RATE)
    assert zt.shape == (D, 2)
    back = view_from_z_any(zt, "velocity", RATE)
    assert np.allclose(back, view, atol=0.05)


def test_bcdataset_takes_absolute_targets_and_refuses_another_mode(tmp_path):
    from surfgym.bc import BCDataset
    dev = torch.device("cpu")
    f = tmp_path / "abs_vel.npz"
    view, zmu, zsd = _write_bc(f, n=10, view_absolute="velocity")
    bc = BCDataset(f, dev, n_latch=0, obs_reward=False, view_continuous=True,
                   yaw_adaptive=True, pitch_rate_max_deg=RATE,
                   view_absolute="velocity")
    assert np.allclose(bc.vz.numpy(), z_from_view_abs(view, "velocity"), atol=1e-6)
    assert np.array_equal(bc.vzmu.numpy(), zmu) and np.array_equal(bc.vzsd.numpy(), zsd)
    assert "[absolute velocity]" in bc.describe() and "from the file" in bc.view_note
    out = bc.sample_all(4, view=True)
    assert out[8].shape == (4, 2) and out[9].shape == (4, 2)
    # the same file under a delta trainer, or under world: refused
    for mode in (None, "world"):
        with pytest.raises(SystemExit, match="view rows are"):
            BCDataset(f, dev, n_latch=0, obs_reward=False, view_continuous=True,
                      yaw_adaptive=True, pitch_rate_max_deg=RATE,
                      view_absolute=mode)
    # a delta file under the absolute trainer: refused; alone: read as before
    fd = tmp_path / "delta.npz"
    _write_bc(fd, n=10, view_absolute=None)
    with pytest.raises(SystemExit, match="view rows are"):
        BCDataset(fd, dev, n_latch=0, obs_reward=False, view_continuous=True,
                  yaw_adaptive=True, pitch_rate_max_deg=RATE,
                  view_absolute="velocity")
    assert BCDataset(fd, dev, n_latch=0, obs_reward=False, view_continuous=True,
                     yaw_adaptive=True, pitch_rate_max_deg=RATE).vz.shape == (10, 2)
    # a DISCRETE file (no view column) cannot be used under an absolute mode
    from surfgym.bc import save_bc_dataset
    from surfgym.core import STATE_DTYPE
    fdisc = tmp_path / "disc.npz"
    n = 6
    acts = np.zeros((n, 6), np.int64)
    acts[:, 0], acts[:, 1], acts[:, 2:4] = 7, 3, 1
    save_bc_dataset(fdisc, np.zeros(n, STATE_DTYPE), np.zeros((n, 15), np.float32),
                    np.zeros(n, np.float32), acts, np.ones(n, np.float32),
                    np.zeros(n, np.int32),
                    {"obs_reward": False, "n_latch": 0, "act_every": 4})
    with pytest.raises(SystemExit, match="discrete BC file"):
        BCDataset(fdisc, dev, n_latch=0, obs_reward=False, view_continuous=True,
                  yaw_adaptive=True, pitch_rate_max_deg=RATE,
                  view_absolute="velocity")
    # world mode: 3-wide moments are read, 2-wide ones refused
    fw = tmp_path / "abs_world.npz"
    view_w, zmu_w, zsd_w = _write_bc(fw, n=10, view_absolute="world")
    bcw = BCDataset(fw, dev, n_latch=0, obs_reward=False, view_continuous=True,
                    yaw_adaptive=True, pitch_rate_max_deg=RATE,
                    view_absolute="world")
    assert bcw.vz.shape == (10, 3) and bcw.vzmu.shape == (10, 3)
    assert np.allclose(bcw.vz.numpy(), z_from_view_abs(view_w, "world"), atol=1e-6)
    fw2 = tmp_path / "abs_world_bad.npz"
    _write_bc(fw2, n=10, view_absolute="world", nz=2)
    with pytest.raises(SystemExit, match="view heads"):
        BCDataset(fw2, dev, n_latch=0, obs_reward=False, view_continuous=True,
                  yaw_adaptive=True, pitch_rate_max_deg=RATE,
                  view_absolute="world")
    # a point-target file (no moments): vzmu is vz with zero spread
    fp = tmp_path / "abs_point.npz"
    view_p, _, _ = _write_bc(fp, n=5, view_absolute="velocity", moments=False)
    bcp = BCDataset(fp, dev, n_latch=0, obs_reward=False, view_continuous=True,
                    yaw_adaptive=True, pitch_rate_max_deg=RATE,
                    view_absolute="velocity")
    assert torch.equal(bcp.vzmu, bcp.vz) and torch.all(bcp.vzsd == 0)


def test_planner_edits_are_degrees_on_a_target_under_the_absolute_modes():
    import beam_tas as bt
    from surfgym.view import wrap180, yaw_limit
    from train_fast import H_SIDE
    act = np.zeros((8, 6), np.int32)
    for mode in ("velocity", "world"):
        # --branch-at draws: jitter 0 is a target in [-180, 180], jitter J
        # an offset in [-J, J] degrees; the same number of variates as the
        # delta draw (the private stream stays aligned)
        y0, s0 = bt.branch_draw_view(np.random.default_rng(1), 64, 0.0, view_absolute=mode)
        assert np.all(np.abs(y0) <= 180.0) and s0.shape == (64,)
        yj, _ = bt.branch_draw_view(np.random.default_rng(1), 64, 5.0, view_absolute=mode)
        assert np.all(np.abs(yj) <= 5.0)
        yd, _ = bt.branch_draw_view(np.random.default_rng(1), 64, 5.0)
        assert np.array_equal(yj, yd)                       # jitter: the same draw
        assert len(np.unique(bt.branch_draw_view(np.random.default_rng(2), 64, 0.0,
                                                 view_absolute=mode)[0])) == 64
        if mode == "velocity":
            # off_warp(u), u uniform: dense near "look along v"
            assert float(np.median(np.abs(y0))) < 45.0
        # apply: wraps, never clips at K_MAX
        view = np.zeros((8, 2), np.float32)
        view[:, 0] = 175.0
        a, v = bt.branch_apply_view(act, view, np.arange(8), np.full(8, 10.0),
                                    s0[:8], 3.0, view_absolute=mode)
        assert np.allclose(v[:, 0], -175.0) and np.array_equal(a[:, H_SIDE], s0[:8])
        a, v = bt.branch_apply_view(act, view, np.arange(8), np.full(8, 190.0),
                                    s0[:8], 0.0, view_absolute=mode)
        assert np.allclose(v[:, 0], -170.0)
        assert yaw_limit(mode) == 180.0
        # --branch-grid: degrees, default -30..30, wrapped on apply
        plans, meta = bt.branch_grid_parse("hold=4:seg=1", 4, view=True,
                                           view_absolute=mode)
        assert meta["yaw"] == [-30.0, -15.0, -5.0, 0.0, 5.0, 15.0, 30.0]
        assert meta["view_absolute"] == mode and meta["plans"] == 21
        assign = np.arange(8) % len(plans)
        a, v = bt.branch_grid_apply_view(act, view, np.arange(8), plans, assign,
                                         0, view_absolute=mode)
        want = [float(wrap180(175.0 + plans[p][0])) for p in assign]
        assert np.allclose(v[:, 0], want)
        assert np.all(np.abs(v[:, 0]) <= 180.0)
    # the delta grid is what it was
    _, meta_d = bt.branch_grid_parse("hold=4", 4, view=True)
    assert meta_d["yaw"] == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    assert meta_d["view_absolute"] is None
    # --macro-yaw track: the analytic TARGET - offset 0 (the view along
    # the velocity) for either key in the velocity frame, the heading of
    # v_h in world mode (NaN below the core's 100 u/s frame floor); NaN
    # where the side key is neutral
    vel = np.array([[1000.0, 0, 0], [0, 1000.0, 0], [50.0, 0, 0], [0, -1000.0, 0]])
    side = np.array([0, 2, 0, 1])
    mv = bt.macro_yaw_abs(vel, side, "velocity")
    assert np.array_equal(np.isnan(mv), [False, False, False, True])
    assert np.all(mv[:3] == 0.0)
    mw = bt.macro_yaw_abs(vel, side, "world")
    assert np.array_equal(np.isnan(mw), [False, False, True, True])
    assert mw[0] == 0.0 and mw[1] == pytest.approx(90.0)
    # MacroHold.decide_view writes it into the view table's yaw column
    mh = bt.MacroHold(4, np.arange(4), 0.2, 0.8, 0.04, np.random.default_rng(0),
                      yaw="track", fwd="none")
    view = np.full((4, 2), 33.0, np.float32)
    a, v = mh.decide_view(act[:4], view, vel, True, 10.0, view_absolute="velocity")
    held = (mh.side[:4] == 0) | (mh.side[:4] == 2)
    assert np.all(v[held, 0] == 0.0) and np.all(v[~held, 0] == 33.0)
    assert np.all(a[:, 0] == NEUTRAL_ACT[0])
    # the prefix hash packs 2 or 3 heads (the 2-head packing unchanged)
    acts = np.ones((5, 6), np.int32)
    h = np.zeros(5, np.uint64)
    z2 = np.array([[0.1, -0.2]] * 5)
    q = (np.clip(np.rint(z2 / 0.05), -2047, 2047) + 2048).astype(np.uint64)
    want = h * np.uint64(1099511628211) ^ (np.uint64(sum(1 << (4 * k) for k in range(6)))
                                            | (q[:, 0] << np.uint64(24))
                                            | (q[:, 1] << np.uint64(36)))
    assert np.array_equal(bt.EpsSampledTorchPolicy._hmix(h, acts, z2), want)
    h3 = bt.EpsSampledTorchPolicy._hmix(h, acts, np.array([[0.1, -0.2, 0.3]] * 5))
    assert h3.dtype == np.uint64 and not np.array_equal(h3, want)
    # build_sim hands the core the checkpoint's view_mode
    if DLL.exists() and SKI.exists():
        c1 = bt.build_sim({"maxvel": 4000.0, "yaw_adaptive": True, "pitch_rate": 1.33,
                           "view_absolute": "velocity"}, str(SKI), 2, 400)
        c0 = bt.build_sim({"maxvel": 4000.0, "yaw_adaptive": True, "pitch_rate": 1.33},
                          str(SKI), 2, 400)
        assert int(c1.config.view_mode) == 1 and int(c0.config.view_mode) == 0
