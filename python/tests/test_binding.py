"""Pure-structure tests for the surfgym binding — run WITHOUT surfcore.dll.

Every struct layout is desk-checked field-by-field against an independent
transcription of src/surfcore.h (name, ctype, byte offset accumulated by
hand: scalars 4 bytes, float[3] arrays 12 bytes, no padding anywhere since
every field is 4-byte aligned).

Runs two ways:
    python -m pytest tests/test_binding.py -q     (from C:\\RL_Surf\\python)
    python tests/test_binding.py                  (plain asserts, no pytest)
"""
from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find surfgym

import surfgym  # noqa: E402  (must import cleanly with no DLL present)
from surfgym.core import (  # noqa: E402
    ACTION_DIM,
    ACTION_NVEC,
    STATE_DTYPE,
    YAW_BINS,
    SurfEnvConfig,
    SurfPhys,
    SurfState,
    SurfTrace,
    _dll_candidates,
    default_config,
    validate_actions,
)

F32, I32 = ctypes.c_float, ctypes.c_int32
F32x3 = ctypes.c_float * 3


@contextmanager
def expect_raises(*exc_types):
    """pytest-free stand-in for pytest.raises."""
    try:
        yield
    except exc_types:
        return
    raise AssertionError(f"expected {' or '.join(e.__name__ for e in exc_types)}")


def _check_layout(struct_cls, expected):
    """expected: list of (name, ctype, size) transcribed from surfcore.h;
    offsets are accumulated independently and compared to ctypes' layout."""
    fields = struct_cls._fields_
    names = [f[0] for f in fields]
    assert names == [e[0] for e in expected], (
        f"{struct_cls.__name__} field order mismatch:\n"
        f"  binding: {names}\n  header:  {[e[0] for e in expected]}"
    )
    offset = 0
    for name, ctype, size in expected:
        fld = getattr(struct_cls, name)
        assert fld.offset == offset, (
            f"{struct_cls.__name__}.{name}: offset {fld.offset} != "
            f"expected {offset}"
        )
        assert fld.size == size, (
            f"{struct_cls.__name__}.{name}: size {fld.size} != expected {size}"
        )
        declared = dict(fields)[name]
        assert declared == ctype, (
            f"{struct_cls.__name__}.{name}: ctype {declared} != {ctype}"
        )
        offset += size
    assert ctypes.sizeof(struct_cls) == offset, (
        f"sizeof({struct_cls.__name__}) = {ctypes.sizeof(struct_cls)} != "
        f"sum of fields {offset} (unexpected padding?)"
    )


# ---------------------------------------------------------------------------
# Struct layouts vs. surfcore.h
# ---------------------------------------------------------------------------

def test_surfphys_layout():
    # surfcore.h: 11 floats then 4 int32 -> 11*4 + 4*4 = 60 bytes.
    expected = [
        ("sv_gravity", F32, 4),
        ("sv_airaccelerate", F32, 4),
        ("sv_accelerate", F32, 4),
        ("sv_friction", F32, 4),
        ("edgefriction", F32, 4),
        ("sv_stopspeed", F32, 4),
        ("sv_maxspeed", F32, 4),
        ("player_maxspeed", F32, 4),
        ("sv_maxvelocity", F32, 4),
        ("sv_stepsize", F32, 4),
        ("sv_bounce", F32, 4),
        ("msec", I32, 4),
        ("enable_stamina", I32, 4),
        ("enable_bhop_cap", I32, 4),
        ("enable_duck", I32, 4),
    ]
    _check_layout(SurfPhys, expected)
    assert ctypes.sizeof(SurfPhys) == 11 * 4 + 4 * 4 == 60


def test_surfenvconfig_layout():
    # 5 int32 + 4 float interleaved (9*4 = 36) + SurfPhys (60) = 96 bytes.
    expected = [
        ("num_envs", I32, 4),
        ("max_episode_ticks", I32, 4),
        ("lookahead_k", I32, 4),
        ("lookahead_dt", F32, 4),
        ("spawn_mode", I32, 4),
        ("spawn_curriculum_frac", F32, 4),
        ("yaw_rate_max_deg", F32, 4),
        ("yaw_jitter_deg", F32, 4),
        ("kill_z", F32, 4),
        ("water_fail", I32, 4),
        ("pitch_rate_max_deg", F32, 4),
        ("lidar_w", I32, 4),
        ("lidar_h", I32, 4),
        ("lidar_hfov_deg", F32, 4),
        ("lidar_vfov_deg", F32, 4),
        ("lidar_range", F32, 4),
        ("phys", SurfPhys, ctypes.sizeof(SurfPhys)),
    ]
    _check_layout(SurfEnvConfig, expected)
    assert ctypes.sizeof(SurfEnvConfig) == 16 * 4 + 60 == 124


def test_surfstate_layout():
    # 3 float[3] (36) + 2 float (8) + 6 int32 (24) + 2 float (8) + 1 int32 (4)
    # + duck: int32 (4) + float (4) = 88 bytes.
    expected = [
        ("origin", F32x3, 12),
        ("velocity", F32x3, 12),
        ("basevelocity", F32x3, 12),
        ("yaw", F32, 4),
        ("pitch", F32, 4),
        ("fuser2", F32, 4),
        ("base_vel_flag", I32, 4),
        ("onground", I32, 4),
        ("oldbuttons", I32, 4),
        ("ducked", I32, 4),
        ("tick", I32, 4),
        ("seg_hint", I32, 4),
        ("progress", F32, 4),
        ("best_progress", F32, 4),
        ("stuck_ticks", I32, 4),
        ("induck", I32, 4),
        ("duck_time", F32, 4),
        ("waterjumptime", F32, 4),
        ("movedir", F32x3, 12),
    ]
    _check_layout(SurfState, expected)
    assert ctypes.sizeof(SurfState) == 108


def test_surftrace_layout():
    # 2 float[3] (24) + 2 float (8) + 3 int32 (12) = 44 bytes.
    expected = [
        ("endpos", F32x3, 12),
        ("normal", F32x3, 12),
        ("dist", F32, 4),
        ("fraction", F32, 4),
        ("startsolid", I32, 4),
        ("allsolid", I32, 4),
        ("ent", I32, 4),
    ]
    _check_layout(SurfTrace, expected)
    assert ctypes.sizeof(SurfTrace) == 2 * 12 + 2 * 4 + 3 * 4 == 44


def test_state_dtype_matches_struct():
    assert STATE_DTYPE.itemsize == ctypes.sizeof(SurfState)
    assert list(STATE_DTYPE.names) == [f[0] for f in SurfState._fields_]
    for name in STATE_DTYPE.names:
        np_off = STATE_DTYPE.fields[name][1]
        ct_off = getattr(SurfState, name).offset
        assert np_off == ct_off, (
            f"STATE_DTYPE.{name} offset {np_off} != ctypes {ct_off}"
        )


def test_state_dtype_roundtrip():
    """Bytes written through the ctypes struct read back identically through
    the numpy dtype (the exact path surf_get_states uses)."""
    st = SurfState()
    st.origin[:] = (1.0, 2.0, 3.0)
    st.velocity[:] = (-4.5, 5.5, -6.5)
    st.basevelocity[:] = (0.0, 0.25, 0.0)
    st.yaw = 123.75
    st.fuser2 = 1315.0
    st.base_vel_flag = 1
    st.onground = -1
    st.oldbuttons = 2
    st.ducked = 0
    st.tick = 4242
    st.seg_hint = 17
    st.progress = 8121.5
    st.best_progress = 9000.25
    st.stuck_ticks = 3
    arr = np.frombuffer(bytes(st), dtype=STATE_DTYPE)[0]
    assert np.allclose(arr["origin"], (1.0, 2.0, 3.0))
    assert np.allclose(arr["velocity"], (-4.5, 5.5, -6.5))
    assert arr["yaw"] == np.float32(123.75)
    assert arr["fuser2"] == np.float32(1315.0)
    assert arr["base_vel_flag"] == 1
    assert arr["onground"] == -1
    assert arr["oldbuttons"] == 2
    assert arr["tick"] == 4242
    assert arr["seg_hint"] == 17
    assert arr["progress"] == np.float32(8121.5)
    assert arr["best_progress"] == np.float32(9000.25)
    assert arr["stuck_ticks"] == 3


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def test_default_config_values():
    cfg = default_config(num_envs=64)
    assert cfg.num_envs == 64
    assert cfg.max_episode_ticks == 6000
    assert cfg.lookahead_k == 8
    assert abs(cfg.lookahead_dt - 0.25) < 1e-6
    assert cfg.spawn_mode == 1
    assert abs(cfg.spawn_curriculum_frac - 0.99) < 1e-6
    assert abs(cfg.yaw_rate_max_deg - 10.0) < 1e-6
    assert abs(cfg.yaw_jitter_deg - 5.0) < 1e-6
    assert cfg.kill_z <= -1e30  # auto kill-z sentinel survives float32
    p = cfg.phys
    assert p.sv_gravity == 800.0
    assert p.sv_airaccelerate == 100.0
    assert p.sv_accelerate == 5.0
    assert p.sv_friction == 4.0
    assert p.edgefriction == 2.0
    assert p.sv_stopspeed == 75.0
    assert p.sv_maxspeed == 320.0
    assert p.player_maxspeed == 250.0
    assert p.sv_maxvelocity == 2000.0
    assert p.sv_stepsize == 18.0
    assert p.sv_bounce == 1.0
    assert p.msec == 10
    assert p.enable_stamina == 1
    assert p.enable_bhop_cap == 1
    assert p.enable_duck == 1


def test_default_config_overrides():
    cfg = default_config(num_envs=4, sv_gravity=790.0, lookahead_k=4,
                         enable_duck=1, spawn_mode=0)
    assert cfg.num_envs == 4
    assert cfg.phys.sv_gravity == 790.0
    assert cfg.lookahead_k == 4
    assert cfg.phys.enable_duck == 1
    assert cfg.spawn_mode == 0
    # untouched defaults remain
    assert cfg.phys.sv_airaccelerate == 100.0
    assert cfg.max_episode_ticks == 6000


def test_default_config_rejects_unknown():
    with expect_raises(TypeError):
        default_config(num_envs=1, gravity=800)  # must be sv_gravity
    with expect_raises(ValueError):
        default_config(num_envs=0)


# ---------------------------------------------------------------------------
# Action validation (the exact checks SurfCore.step performs)
# ---------------------------------------------------------------------------

def test_validate_actions_accepts_good():
    a = np.zeros((16, ACTION_DIM), dtype=np.int32)
    validate_actions(a, 16)  # no raise


def test_validate_actions_rejects_bad():
    n = 16
    with expect_raises(TypeError):
        validate_actions([[0] * ACTION_DIM] * n, n)  # not an ndarray
    with expect_raises(ValueError):
        validate_actions(np.zeros((n, ACTION_DIM), dtype=np.int64), n)  # dtype
    with expect_raises(ValueError):
        validate_actions(np.zeros((n, ACTION_DIM), dtype=np.float32), n)
    with expect_raises(ValueError):
        validate_actions(np.zeros((n + 1, ACTION_DIM), dtype=np.int32), n)
    with expect_raises(ValueError):
        validate_actions(np.zeros((n, ACTION_DIM + 1), dtype=np.int32), n)
    with expect_raises(ValueError):
        validate_actions(np.zeros(n * ACTION_DIM, dtype=np.int32), n)  # 1-D
    noncontig = np.zeros((ACTION_DIM, n), dtype=np.int32).T
    with expect_raises(ValueError):
        validate_actions(noncontig, n)


# ---------------------------------------------------------------------------
# Action / yaw-bin constants
# ---------------------------------------------------------------------------

def test_yaw_bins():
    assert ACTION_NVEC == (15, 7, 3, 3, 2, 2)
    from surfgym import PITCH_BINS
    assert PITCH_BINS.shape == (7,)
    assert np.count_nonzero(PITCH_BINS == 0.0) == 1
    assert np.allclose(np.sort(PITCH_BINS), np.sort(-PITCH_BINS))
    assert YAW_BINS.shape == (15,)
    assert YAW_BINS.dtype == np.float32
    assert np.count_nonzero(YAW_BINS == 0.0) == 1
    assert float(np.abs(YAW_BINS).max()) == 10.0
    # symmetric: every magnitude appears with both signs
    assert np.allclose(np.sort(YAW_BINS), np.sort(-YAW_BINS))
    mags = sorted(set(float(abs(b)) for b in YAW_BINS))
    assert mags == [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]


# ---------------------------------------------------------------------------
# DLL search order (no DLL needed — just the candidate list)
# ---------------------------------------------------------------------------

def test_dll_candidates_order():
    saved = os.environ.pop("SURFCORE_DLL", None)
    try:
        root = Path(surfgym.core.__file__).resolve().parents[2]
        cands = _dll_candidates()
        assert cands == [root / "build" / "surfcore.dll",
                         root / "surfcore.dll"]
        assert all(p.is_absolute() for p in cands)

        cands = _dll_candidates(dll_path=r"X:\elsewhere\surfcore.dll")
        assert cands[0] == Path(r"X:\elsewhere\surfcore.dll")

        os.environ["SURFCORE_DLL"] = r"Y:\env\surfcore.dll"
        cands = _dll_candidates(dll_path=r"X:\arg\surfcore.dll")
        assert cands[0] == Path(r"X:\arg\surfcore.dll")   # arg beats env var
        assert cands[1] == Path(r"Y:\env\surfcore.dll")   # env var beats repo
        assert cands[2:] == [root / "build" / "surfcore.dll",
                             root / "surfcore.dll"]
    finally:
        os.environ.pop("SURFCORE_DLL", None)
        if saved is not None:
            os.environ["SURFCORE_DLL"] = saved


# ---------------------------------------------------------------------------
# Import hygiene: everything imports without the DLL / gymnasium / SB3
# ---------------------------------------------------------------------------

def test_adapter_modules_import_without_deps():
    import importlib
    vec_env = importlib.import_module("surfgym.vec_env")
    gym_env = importlib.import_module("surfgym.gym_env")
    assert hasattr(vec_env, "_build_surf_vec_env")
    assert hasattr(gym_env, "_build")
    # Touching the lazy attribute must raise cleanly (not at import time)
    # when the heavy dependency is absent.
    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        with expect_raises(ImportError):
            vec_env.SurfVecEnv  # noqa: B018
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        with expect_raises(ImportError):
            gym_env.SurfGymEnv  # noqa: B018


def test_package_exports():
    for name in ("SurfCore", "SurfEnvConfig", "SurfPhys", "default_config",
                 "ScriptedStrafer", "RandomPolicy", "record_rollout",
                 "YAW_BINS", "STATE_DTYPE"):
        assert hasattr(surfgym, name), f"surfgym.{name} missing"


def test_random_policy_shape():
    from surfgym import RandomPolicy
    pol = RandomPolicy(seed=7)
    obs = np.zeros((32, 143), dtype=np.float32)
    a = pol.act(obs)
    assert a.shape == (32, ACTION_DIM)
    assert a.dtype == np.int32
    nvec = np.asarray(ACTION_NVEC)
    assert (a >= 0).all() and (a < nvec).all()


# ---------------------------------------------------------------------------
# Plain-python runner (works without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
