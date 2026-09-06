"""ctypes binding to surfcore.dll — an exact mirror of ``src/surfcore.h``.

Design rules (see docs/03-env-design.md, "Python layer"):

* Importing this module never touches the DLL.  The library is located and
  loaded only when :class:`SurfCore` is instantiated, and a missing DLL fails
  with an explicit message listing every path that was tried.
* Every exported function gets explicit ``argtypes``/``restype`` — the classic
  x64 pointer-truncation bug (default int restype chops 64-bit pointers).
* All batch buffers (obs / rewards / done / trunc / terminal_obs / states) are
  allocated once in ``__init__``; their ctypes pointers are cached, so the hot
  loop performs no numpy allocation.  ``step``/``reset`` return *views* of
  those buffers, valid only until the next ``step``/``reset`` call.
* ``ctypes.CDLL`` releases the GIL for the duration of each call, so OpenMP
  inside ``surf_step`` owns all cores.

Struct layouts are transcribed field-for-field from ``surfcore.h``; all fields
are 4-byte scalars or float[3] arrays, so there is no padding and the default
ctypes alignment matches MSVC/clang exactly (verified in tests/test_binding.py).
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "SURF_IN_JUMP",
    "SURF_IN_DUCK",
    "ACTION_DIM",
    "ACTION_NVEC",
    "YAW_BINS",
    "PITCH_BINS",
    "SurfPhys",
    "SurfEnvConfig",
    "SurfState",
    "SurfTrace",
    "STATE_DTYPE",
    "default_config",
    "config_to_dict",
    "phys_to_dict",
    "validate_actions",
    "SurfCore",
]

# ---------------------------------------------------------------------------
# Constants (surfcore.h header comment)
# ---------------------------------------------------------------------------

SURF_IN_JUMP = 2  # usercmd button bits, HLSDK convention
SURF_IN_DUCK = 4

SURF_ABI_VERSION = 7  # must match src/surfcore.h; checked at DLL load

ACTION_DIM = 6
# [yaw_bin, pitch_bin, forward, side, jump, duck]
ACTION_NVEC: Tuple[int, ...] = (15, 7, 3, 3, 2, 2)  # MultiDiscrete nvec (docs/03)

# Yaw bins in degrees *at the default yaw_rate_max_deg = 10*; the core scales
# them so the largest bin equals cfg.yaw_rate_max_deg.  The header lists the
# set {0, ±0.25, ±0.5, ±1, ±2, ±4, ±7, ±10} but does not pin the index->bin
# order; we adopt ascending order (index 7 = 0 deg, index 14 = +10 deg).
# If the DLL uses a different order only ScriptedStrafer cares — flip here.
YAW_BINS = np.array(
    [-10.0, -7.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0,
     0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0],
    dtype=np.float32,
)
assert YAW_BINS.shape == (ACTION_NVEC[0],)

# View-pitch bins (deg at default pitch_rate_max_deg = 10); index 3 = 0,
# positive = look up. Pitch aims the lidar only — no effect on movement.
PITCH_BINS = np.array([-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0], dtype=np.float32)
assert PITCH_BINS.shape == (ACTION_NVEC[1],)

# ---------------------------------------------------------------------------
# Struct mirrors — field order and types must match surfcore.h EXACTLY
# ---------------------------------------------------------------------------

c_float = ctypes.c_float
c_int32 = ctypes.c_int32


class SurfPhys(ctypes.Structure):
    """Mirror of ``SurfPhys`` (surfcore.h) — physics cvars."""

    _fields_ = [
        ("sv_gravity", c_float),          # 800
        ("sv_airaccelerate", c_float),    # 100
        ("sv_accelerate", c_float),       # 5
        ("sv_friction", c_float),         # 4
        ("edgefriction", c_float),        # 2
        ("sv_stopspeed", c_float),        # 75
        ("sv_maxspeed", c_float),         # 320
        ("player_maxspeed", c_float),     # 250
        ("sv_maxvelocity", c_float),      # 2000
        ("sv_stepsize", c_float),         # 18
        ("sv_bounce", c_float),           # 1
        ("msec", c_int32),                # 10
        ("enable_stamina", c_int32),      # 1
        ("enable_bhop_cap", c_int32),     # 1
        ("enable_duck", c_int32),         # 0
    ]


class SurfEnvConfig(ctypes.Structure):
    """Mirror of ``SurfEnvConfig`` (surfcore.h)."""

    _fields_ = [
        ("num_envs", c_int32),
        ("max_episode_ticks", c_int32),
        ("lookahead_k", c_int32),
        ("lookahead_dt", c_float),
        ("spawn_mode", c_int32),
        ("spawn_curriculum_frac", c_float),
        ("yaw_rate_max_deg", c_float),
        ("yaw_jitter_deg", c_float),
        ("kill_z", c_float),
        ("water_fail", c_int32),
        ("pitch_rate_max_deg", c_float),
        ("lidar_w", c_int32),
        ("lidar_h", c_int32),
        ("lidar_hfov_deg", c_float),
        ("lidar_vfov_deg", c_float),
        ("lidar_range", c_float),
        ("phys", SurfPhys),
        ("yaw_adaptive", c_int32),
        ("yaw_blend", c_float),
        ("side_hold_ticks", c_int32),
    ]


class SurfState(ctypes.Structure):
    """Mirror of ``SurfState`` (surfcore.h) — full per-env POD state."""

    _fields_ = [
        ("origin", c_float * 3),
        ("velocity", c_float * 3),
        ("basevelocity", c_float * 3),
        ("yaw", c_float),
        ("pitch", c_float),
        ("fuser2", c_float),
        ("base_vel_flag", c_int32),
        ("onground", c_int32),        # -1 airborne, else solid index
        ("oldbuttons", c_int32),
        ("ducked", c_int32),
        ("tick", c_int32),
        ("seg_hint", c_int32),
        ("progress", c_float),
        ("best_progress", c_float),
        ("stuck_ticks", c_int32),
        ("induck", c_int32),          # bInDuck: duck transition in progress
        ("duck_time", c_float),       # flDuckTime ms countdown
        ("waterjumptime", c_float),   # ms; > 0 while leaping out of water
        ("movedir", c_float * 3),     # waterjump ledge-push direction
    ]


class SurfTrace(ctypes.Structure):
    """Mirror of ``SurfTrace`` (surfcore.h)."""

    _fields_ = [
        ("endpos", c_float * 3),
        ("normal", c_float * 3),
        ("dist", c_float),
        ("fraction", c_float),        # 1.0 = clean
        ("startsolid", c_int32),
        ("allsolid", c_int32),
        ("ent", c_int32),             # -1 none, 0 world, >0 brush ent
    ]


# numpy structured dtype matching SurfState byte-for-byte (packed; the C
# struct has no padding since every field is 4-byte aligned).
STATE_DTYPE = np.dtype(
    [
        ("origin", np.float32, (3,)),
        ("velocity", np.float32, (3,)),
        ("basevelocity", np.float32, (3,)),
        ("yaw", np.float32),
        ("pitch", np.float32),
        ("fuser2", np.float32),
        ("base_vel_flag", np.int32),
        ("onground", np.int32),
        ("oldbuttons", np.int32),
        ("ducked", np.int32),
        ("tick", np.int32),
        ("seg_hint", np.int32),
        ("progress", np.float32),
        ("best_progress", np.float32),
        ("stuck_ticks", np.int32),
        ("induck", np.int32),
        ("duck_time", np.float32),
        ("waterjumptime", np.float32),
        ("movedir", np.float32, (3,)),
    ]
)
assert STATE_DTYPE.itemsize == ctypes.sizeof(SurfState), (
    "STATE_DTYPE / SurfState layout drift: "
    f"{STATE_DTYPE.itemsize} != {ctypes.sizeof(SurfState)}"
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_PHYS_DEFAULTS = {
    "sv_gravity": 800.0,
    "sv_airaccelerate": 100.0,
    "sv_accelerate": 5.0,
    "sv_friction": 4.0,
    "edgefriction": 2.0,
    "sv_stopspeed": 75.0,
    "sv_maxspeed": 320.0,
    "player_maxspeed": 250.0,
    "sv_maxvelocity": 2000.0,
    "sv_stepsize": 18.0,
    "sv_bounce": 1.0,
    "msec": 10,
    "enable_stamina": 1,
    "enable_bhop_cap": 1,
    "enable_duck": 1,
}

_ENV_DEFAULTS = {
    "max_episode_ticks": 6000,
    "lookahead_k": 8,
    "lookahead_dt": 0.25,
    "spawn_mode": 1,
    "spawn_curriculum_frac": 0.99,
    "yaw_rate_max_deg": 10.0,
    "yaw_adaptive": 0,
    "yaw_blend": 1.0,
    "side_hold_ticks": 0,
    "yaw_jitter_deg": 5.0,
    "kill_z": -1e38,  # <= -1e30 -> auto (map min z - 256)
    "water_fail": 1,  # waterlevel>=2 ends the episode; 0 = swimming allowed
    "pitch_rate_max_deg": 10.0,
    "lidar_w": 16,    # 0 disables the depth-image block
    "lidar_h": 8,
    "lidar_hfov_deg": 120.0,
    "lidar_vfov_deg": 90.0,
    "lidar_range": 2000.0,
}


def default_config(num_envs: int = 1, **overrides: Any) -> SurfEnvConfig:
    """Build a :class:`SurfEnvConfig` with the documented defaults.

    ``overrides`` accepts any field of ``SurfEnvConfig`` or ``SurfPhys`` by
    its exact struct name, e.g. ``default_config(num_envs=64, sv_gravity=790,
    lookahead_k=4)``.  Unknown keys raise ``TypeError``.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    env_vals = dict(_ENV_DEFAULTS)
    phys_vals = dict(_PHYS_DEFAULTS)
    for key, val in overrides.items():
        if key in env_vals:
            env_vals[key] = val
        elif key in phys_vals:
            phys_vals[key] = val
        else:
            valid = sorted(env_vals) + sorted(phys_vals)
            raise TypeError(
                f"unknown config field {key!r}; valid overrides: {valid}"
            )
    cfg = SurfEnvConfig()
    cfg.num_envs = int(num_envs)
    cfg.max_episode_ticks = int(env_vals["max_episode_ticks"])
    cfg.lookahead_k = int(env_vals["lookahead_k"])
    cfg.lookahead_dt = float(env_vals["lookahead_dt"])
    cfg.spawn_mode = int(env_vals["spawn_mode"])
    cfg.spawn_curriculum_frac = float(env_vals["spawn_curriculum_frac"])
    cfg.yaw_rate_max_deg = float(env_vals["yaw_rate_max_deg"])
    cfg.yaw_adaptive = int(env_vals["yaw_adaptive"])
    cfg.yaw_blend = float(env_vals.get("yaw_blend", 1.0))
    cfg.side_hold_ticks = int(env_vals.get("side_hold_ticks", 0))
    cfg.yaw_jitter_deg = float(env_vals["yaw_jitter_deg"])
    cfg.kill_z = float(env_vals["kill_z"])
    cfg.water_fail = int(env_vals["water_fail"])
    cfg.pitch_rate_max_deg = float(env_vals["pitch_rate_max_deg"])
    cfg.lidar_w = int(env_vals["lidar_w"])
    cfg.lidar_h = int(env_vals["lidar_h"])
    cfg.lidar_hfov_deg = float(env_vals["lidar_hfov_deg"])
    cfg.lidar_vfov_deg = float(env_vals["lidar_vfov_deg"])
    cfg.lidar_range = float(env_vals["lidar_range"])
    for name, _ in SurfPhys._fields_:
        setattr(cfg.phys, name, phys_vals[name])
    return cfg


def phys_to_dict(phys: SurfPhys) -> dict:
    """SurfPhys -> plain dict (JSONL headers, logging)."""
    out = {}
    for name, ctype in SurfPhys._fields_:
        val = getattr(phys, name)
        out[name] = int(val) if ctype is c_int32 else float(val)
    return out


def config_to_dict(cfg: SurfEnvConfig) -> dict:
    """SurfEnvConfig -> plain nested dict."""
    out = {}
    for name, ctype in SurfEnvConfig._fields_:
        if name == "phys":
            out[name] = phys_to_dict(cfg.phys)
        else:
            val = getattr(cfg, name)
            out[name] = int(val) if ctype is c_int32 else float(val)
    return out


# ---------------------------------------------------------------------------
# Action validation (pure, DLL-free — unit tested without the DLL)
# ---------------------------------------------------------------------------

def validate_actions(actions: np.ndarray, num_envs: int) -> None:
    """Validate an action batch for ``surf_step``.

    Requires a C-contiguous int32 ndarray of shape (num_envs, 6).  Raises
    ``TypeError``/``ValueError`` with a precise message otherwise.
    """
    if not isinstance(actions, np.ndarray):
        raise TypeError(
            f"actions must be a numpy ndarray, got {type(actions).__name__}"
        )
    if actions.dtype != np.int32:
        raise ValueError(f"actions dtype must be int32, got {actions.dtype}")
    if actions.shape != (num_envs, ACTION_DIM):
        raise ValueError(
            f"actions shape must be ({num_envs}, {ACTION_DIM}), "
            f"got {actions.shape}"
        )
    if not actions.flags["C_CONTIGUOUS"]:
        raise ValueError("actions must be C-contiguous")


# ---------------------------------------------------------------------------
# DLL location / loading
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../RL_Surf


def _dll_candidates(dll_path: Optional[Union[str, os.PathLike]] = None) -> List[Path]:
    """Absolute candidate paths for the surfcore library, in search order:
    explicit ``dll_path`` arg, ``SURFCORE_DLL`` env var, ``<repo>/build/``,
    ``<repo>/`` (platform-appropriate names: surfcore.dll on Windows,
    libsurfcore.so on Linux, libsurfcore.dylib on macOS)."""
    cands: List[Path] = []
    if dll_path:
        cands.append(Path(dll_path).resolve())
    env = os.environ.get("SURFCORE_DLL")
    if env:
        cands.append(Path(env).resolve())
    if sys.platform == "win32":
        names = ["surfcore.dll"]
    elif sys.platform == "darwin":
        names = ["libsurfcore.dylib", "libsurfcore.so"]
    else:
        names = ["libsurfcore.so", "surfcore.so"]
    for name in names:
        cands.append(_REPO_ROOT / "build" / name)
    for name in names:
        cands.append(_REPO_ROOT / name)
    return cands


def _load_library(dll_path: Optional[Union[str, os.PathLike]] = None) -> ctypes.CDLL:
    cands = _dll_candidates(dll_path)
    path = next((p for p in cands if p.is_file()), None)
    if path is None:
        tried = "\n  ".join(str(p) for p in cands)
        raise FileNotFoundError(
            "surfcore library not found. Build it first (build.ps1 on Windows, "
            "./build.sh on Linux), or point SURFCORE_DLL / dll_path= at it. "
            f"Paths tried, in order:\n  {tried}"
        )
    # Make dependency DLLs (vcomp140.dll / libomp.dll) resolvable from the
    # same directory; ctypes on Windows needs this explicitly since 3.8.
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path.parent))
    try:
        return ctypes.CDLL(str(path))
    except OSError as exc:
        raise RuntimeError(
            f"failed to load {path}: {exc}\n"
            "WinError 126 here almost always means a *dependency* DLL is "
            "missing (vcomp140.dll / libomp.dll) — check with "
            "'dumpbin /dependents surfcore.dll'."
        ) from exc


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Set argtypes/restype for every exported function (surfcore.h)."""
    P_F32 = ctypes.POINTER(c_float)
    P_I32 = ctypes.POINTER(c_int32)
    P_U8 = ctypes.POINTER(ctypes.c_uint8)
    P_STATE = ctypes.POINTER(SurfState)

    sigs = {
        # lifecycle
        "surf_create": ([ctypes.c_char_p, ctypes.POINTER(SurfEnvConfig),
                         ctypes.c_char_p, c_int32], ctypes.c_void_p),
        "surf_destroy": ([ctypes.c_void_p], None),
        "surf_obs_dim": ([ctypes.c_void_p], c_int32),
        "surf_num_envs": ([ctypes.c_void_p], c_int32),
        "surf_set_waypoints": ([ctypes.c_void_p, P_F32, c_int32], c_int32),
        "surf_abi_version": ([], c_int32),
        "surf_set_spawn_pool": ([ctypes.c_void_p, P_STATE, c_int32], c_int32),
        # hot path
        "surf_reset_all": ([ctypes.c_void_p, ctypes.c_uint64, P_F32], None),
        "surf_step": ([ctypes.c_void_p, P_I32, P_F32, P_F32, P_U8, P_U8,
                       P_F32], None),
        # state access
        "surf_get_states": ([ctypes.c_void_p, P_STATE], None),
        "surf_states_ptr": ([ctypes.c_void_p], ctypes.c_void_p),
        "surf_set_state": ([ctypes.c_void_p, c_int32, P_STATE], None),
        # exposed internals
        "surf_trace": ([ctypes.c_void_p, P_F32, P_F32, c_int32,
                        ctypes.POINTER(SurfTrace)], None),
        "surf_point_contents": ([ctypes.c_void_p, P_F32], c_int32),
        "surf_occupancy_grid": ([ctypes.c_void_p, P_F32, c_float, c_int32,
                                 c_int32, c_int32, P_U8], None),
        "surf_pm_step_usercmd": ([ctypes.c_void_p, P_STATE, c_float, c_float,
                                  c_float, c_float, c_int32, c_int32], None),
        "surf_pm_step_single": ([ctypes.c_void_p, P_STATE, P_I32], None),
        "surf_play_step": ([ctypes.c_void_p, P_STATE, c_float, c_float,
                            c_float, c_float, c_int32, c_int32], c_int32),
        # map info
        "surf_num_spawns": ([ctypes.c_void_p], c_int32),
        "surf_get_spawn": ([ctypes.c_void_p, c_int32, P_F32, P_F32], None),
        "surf_map_bounds": ([ctypes.c_void_p, P_F32, P_F32], None),
    }
    for name, (argtypes, restype) in sigs.items():
        try:
            fn = getattr(lib, name)
        except AttributeError as exc:
            raise RuntimeError(
                f"surfcore.dll does not export {name!r} — DLL out of date "
                "vs. src/surfcore.h? Rebuild the core."
            ) from exc
        fn.argtypes = argtypes
        fn.restype = restype
    abi = int(lib.surf_abi_version())
    if abi != SURF_ABI_VERSION:
        raise RuntimeError(
            f"surfcore ABI mismatch: DLL reports {abi}, this binding needs "
            f"{SURF_ABI_VERSION}. Rebuild the core (build.ps1 / ./build.sh) — "
            "a stale DLL misreads the grown structs (e.g. zero gravity)."
        )
    return lib


# ---------------------------------------------------------------------------
# SurfCore
# ---------------------------------------------------------------------------

class SurfCore:
    """Batch surf simulation — thin, zero-copy wrapper over surfcore.dll.

    Parameters
    ----------
    bsp_path : path to the GoldSrc .bsp map.
    config : :class:`SurfEnvConfig` (see :func:`default_config`); ``None``
        means ``default_config(num_envs=1)``.
    dll_path : explicit path to surfcore.dll; otherwise searched via
        ``SURFCORE_DLL`` env var, ``<repo>/build/surfcore.dll``,
        ``<repo>/surfcore.dll``.
    tick_ms : physics tick length in ms (``--tick-ms``); ``None`` keeps
        ``config.phys.msec`` (10). A non-integer value is realised as the
        shortest repeating integer-ms pattern within 0.05 ms of it
        (:func:`surfgym.tick.tick_pattern`: 7.63 -> ``[8, 8, 7]``), driven
        into the core one batch step at a time through ``surf_set_msec``;
        ``config.phys.msec`` is set to the pattern's first element before
        the core is created. An integer tick just sets ``msec`` and never
        touches the setter, so ``tick_ms=10.0`` is byte-identical to today.

    Buffer ownership: :meth:`reset` and :meth:`step` return views of internal
    preallocated buffers, valid only until the next ``reset``/``step`` call
    on this instance.  Copy (``.copy()``) anything you keep.
    """

    def __init__(
        self,
        bsp_path: Union[str, os.PathLike],
        config: Optional[SurfEnvConfig] = None,
        dll_path: Optional[Union[str, os.PathLike]] = None,
        tick_ms: Optional[float] = None,
    ) -> None:
        self._sim: Optional[int] = None  # set early so __del__ is always safe
        self._tick_pat: Optional[Tuple[int, ...]] = None   # multi-element only
        self._tick_idx = 0
        bsp_path = str(bsp_path)
        if not os.path.isfile(bsp_path):
            raise FileNotFoundError(f"BSP not found: {bsp_path}")
        if config is None:
            config = default_config(num_envs=1)
        if not isinstance(config, SurfEnvConfig):
            raise TypeError("config must be a SurfEnvConfig (see default_config)")
        if config.num_envs < 1:
            raise ValueError(f"config.num_envs must be >= 1, got {config.num_envs}")
        pattern: Optional[list] = None
        if tick_ms is not None:
            from .tick import tick_pattern
            pattern = tick_pattern(tick_ms)
            if int(config.phys.msec) != pattern[0]:
                config.phys.msec = int(pattern[0])
        # the NOMINAL tick: what the core was configured with, latched before
        # any stepping. ``config.phys.msec`` moves with the pattern (step
        # mirrors the tick it just ran into it), so it is the wrong thing to
        # copy into a header or to restore in set_tick_pattern(None).
        self._base_msec = int(config.phys.msec)

        self._lib = _bind(_load_library(dll_path))
        self._cfg = config
        self.bsp_path = bsp_path

        err = ctypes.create_string_buffer(512)
        sim = self._lib.surf_create(
            os.fsencode(bsp_path), ctypes.byref(config), err, c_int32(len(err))
        )
        if not sim:
            msg = err.value.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"surf_create failed: {msg or 'unknown error'}")
        self._sim = sim

        self._num_envs = int(self._lib.surf_num_envs(sim))
        self._obs_dim = int(self._lib.surf_obs_dim(sim))
        n, d = self._num_envs, self._obs_dim

        # Preallocated buffers + cached ctypes pointers: no numpy allocation
        # per step.
        self._obs = np.zeros((n, d), dtype=np.float32)
        self._terminal_obs = np.zeros((n, d), dtype=np.float32)
        self._rewards = np.zeros(n, dtype=np.float32)
        self._done = np.zeros(n, dtype=np.uint8)
        self._trunc = np.zeros(n, dtype=np.uint8)
        self._states_buf = np.zeros(n, dtype=STATE_DTYPE)

        P_F32 = ctypes.POINTER(c_float)
        P_U8 = ctypes.POINTER(ctypes.c_uint8)
        self._P_I32 = ctypes.POINTER(c_int32)
        self._p_obs = self._obs.ctypes.data_as(P_F32)
        self._p_term = self._terminal_obs.ctypes.data_as(P_F32)
        self._p_rew = self._rewards.ctypes.data_as(P_F32)
        self._p_done = self._done.ctypes.data_as(P_U8)
        self._p_trunc = self._trunc.ctypes.data_as(P_U8)
        self._p_states = self._states_buf.ctypes.data_as(
            ctypes.POINTER(SurfState)
        )
        if pattern is not None and len(pattern) > 1:
            self.set_tick_pattern(pattern)

    # -- properties ---------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def tick_pattern(self) -> Tuple[int, ...]:
        """The integer-ms tick sequence the core cycles through per batch
        step: ``(10,)`` by default, ``(8, 8, 7)`` at ``tick_ms=7.63``."""
        if self._tick_pat is not None:
            return self._tick_pat
        return (int(self._cfg.phys.msec),)

    @property
    def tick_ms(self) -> float:
        """Mean physics tick in ms (the value every seconds<->ticks
        conversion must use; ``config.phys.msec`` holds only the pattern's
        first element)."""
        pat = self.tick_pattern
        return float(sum(pat)) / len(pat)

    @property
    def nominal_msec(self) -> int:
        """The core's CONFIGURED tick in ms - the pattern's first element
        under ``--tick-ms``, ``config.phys.msec`` otherwise, and identical
        to it at any fixed tick.

        ``config.phys.msec`` itself is a MOVING value under a pattern:
        :meth:`step` mirrors the tick it just ran into it, so a snapshot of
        it is whatever phase the core happens to be in (three recordings
        from one 7.63 ms core wrote msec 8, 8, 7). Anything that states the
        core's tick once - a trajectory header's ``phys`` block, a log line
        - must read this instead, and take the REALISED sequence from
        :attr:`tick_pattern` / :attr:`tick_ms`."""
        return int(self._base_msec)

    @property
    def tick_phase(self) -> int:
        """Index into :attr:`tick_pattern` of the NEXT step's tick (0 after a
        reset). The recorder writes it into the episode header so an
        episode's duration can be summed exactly."""
        return self._tick_idx if self._tick_pat is not None else 0

    def set_tick_phase(self, phase: int) -> None:
        """Make the NEXT step run ``tick_pattern[phase % len]``. A search
        that re-centres its population on a state captured T ticks into an
        episode (tools/beam_tas.py --commit) restores phase T here, so an
        open-loop replay from tick 0 - which runs the pattern from phase 0
        - meets the same ms sequence the search did. A no-op without a
        pattern (the 10 ms core never touches the setter)."""
        if self._tick_pat is not None:
            self._tick_idx = int(phase) % len(self._tick_pat)

    def set_tick_pattern(self, pattern) -> None:
        """Cycle ``surf_step`` through this integer-ms sequence, one element
        per batch step (all envs step in lockstep, so one tick per batch).
        A single-element pattern only sets ``config.phys.msec`` and the
        per-step setter is never called; ``None`` restores the core's
        NOMINAL tick (:attr:`nominal_msec` - the tick it was configured
        with, NOT the phase ``config.phys.msec`` happens to be sitting in),
        so a 7.63 ms core stopped mid-pattern goes back to a fixed 8 ms
        core and not to whichever of 8 / 8 / 7 ran last.
        Requires a DLL built with ``surf_set_msec`` (additive; older DLLs
        raise here and nothing else changes)."""
        if pattern is None:
            pattern = (int(self._base_msec),)
        pat = tuple(int(v) for v in pattern)
        if not pat or any(v < 1 or v > 50 for v in pat):
            raise ValueError(f"tick pattern must be 1..50 ms each, got {pat}")
        if len(pat) == 1:
            self._set_msec(pat[0])
            self._tick_pat = None
            self._tick_idx = 0
            self._base_msec = int(pat[0])   # a fixed tick IS the config tick
            return
        fn = getattr(self._lib, "surf_set_msec", None)
        if fn is None:
            raise RuntimeError(
                "this surfcore build predates surf_set_msec, which a "
                "non-integer --tick-ms needs - rebuild the core "
                "(build.ps1 / ./build.sh)")
        fn.argtypes = [ctypes.c_void_p, c_int32]
        fn.restype = None
        self._set_msec_fn = fn
        self._tick_pat = pat
        self._tick_idx = 0
        self._set_msec(pat[0])

    def _set_msec(self, msec: int) -> None:
        """Push one tick length (ms) into the core for the next step(s)."""
        if int(self._cfg.phys.msec) == int(msec) and self._tick_pat is None:
            return                      # the config value: nothing to do
        fn = getattr(self, "_set_msec_fn", None)
        if fn is None:
            fn = getattr(self._lib, "surf_set_msec", None)
            if fn is None:
                raise RuntimeError(
                    "this surfcore build predates surf_set_msec - rebuild "
                    "the core (build.ps1 / ./build.sh)")
            fn.argtypes = [ctypes.c_void_p, c_int32]
            fn.restype = None
            self._set_msec_fn = fn
        fn(self._handle(), c_int32(int(msec)))
        self._cfg.phys.msec = int(msec)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def config(self) -> SurfEnvConfig:
        return self._cfg

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Destroy the native sim. Idempotent."""
        if self._sim is not None:
            self._lib.surf_destroy(self._sim)
            self._sim = None

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "SurfCore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _handle(self) -> int:
        if self._sim is None:
            raise RuntimeError("SurfCore is closed")
        return self._sim

    def __repr__(self) -> str:
        state = "closed" if self._sim is None else "open"
        return (
            f"SurfCore({os.path.basename(self.bsp_path)!r}, "
            f"num_envs={self._num_envs}, obs_dim={self._obs_dim}, {state})"
        )

    # -- hot path -----------------------------------------------------------

    def reset(self, seed: int = 0) -> np.ndarray:
        """Reset all envs; returns obs view (N, obs_dim) float32.

        The returned array is an internal buffer, valid until the next
        ``step``/``reset`` call.
        """
        sim = self._handle()
        self._lib.surf_reset_all(sim, int(seed) & 0xFFFFFFFFFFFFFFFF, self._p_obs)
        if self._tick_pat is not None:
            # a reset restarts the tick pattern: same seed, same sequence
            self._tick_idx = 0
            self._set_msec(self._tick_pat[0])
        return self._obs

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance every env one tick.

        actions : int32 (N, 6), C-contiguous — [yaw_bin, pitch_bin, forward,
        side, jump, duck] per surfcore.h.

        Returns ``(obs, rewards, done, trunc, terminal_obs)`` — all views of
        internal buffers, valid only until the next ``step``/``reset`` call.
        Same-step autoreset: where ``done|trunc``, ``obs[i]`` is the *new*
        episode's first obs and ``terminal_obs[i]`` the ended episode's true
        final obs (rows of non-done envs are untouched garbage).
        """
        sim = self._handle()
        validate_actions(actions, self._num_envs)
        if self._tick_pat is not None:
            # --tick-ms pattern: this batch step runs pat[idx] ms, the
            # next one pat[idx+1], ... (8, 8, 7, 8, 8, 7 at 7.63 ms)
            _ms = self._tick_pat[self._tick_idx]
            self._set_msec_fn(sim, c_int32(_ms))
            # mirror: the tick just run - a DIAGNOSTIC that moves with the
            # phase, never the core's tick. Header/log writers want
            # self.nominal_msec (+ tick_pattern / tick_ms).
            self._cfg.phys.msec = _ms
            self._tick_idx = (self._tick_idx + 1) % len(self._tick_pat)
        self._lib.surf_step(
            sim,
            actions.ctypes.data_as(self._P_I32),
            self._p_obs,
            self._p_rew,
            self._p_done,
            self._p_trunc,
            self._p_term,
        )
        return self._obs, self._rewards, self._done, self._trunc, self._terminal_obs

    # -- state access -------------------------------------------------------

    def set_teleport_fail(self, enable: bool = True) -> None:
        """When enabled, any benign map teleport ends the episode as a fail
        in the batch step — surf maps teleport fallers into a jail cell where
        circling farms path-length reward forever. Requires a DLL built with
        the export (additive; older DLLs raise here, nothing else breaks)."""
        fn = getattr(self._lib, "surf_set_teleport_fail", None)
        if fn is None:
            raise RuntimeError(
                "this surfcore build predates surf_set_teleport_fail — "
                "rebuild the core (build.ps1 / ./build.sh)")
        fn.argtypes = [ctypes.c_void_p, c_int32]
        fn.restype = None
        fn(self._handle(), c_int32(1 if enable else 0))

    def set_goal_box(self, mins, maxs) -> None:
        """Race finish zone: an episode completes (done) when the player's
        swept per-tick segment crosses this AABB (hull-inflated; swept, so a
        1u-thin finish curtain still registers at any speed). Requires a DLL
        built with the export (additive; older DLLs raise here)."""
        fn = getattr(self._lib, "surf_set_goal_box", None)
        if fn is None:
            raise RuntimeError(
                "this surfcore build predates surf_set_goal_box — "
                "rebuild the core (build.ps1 / ./build.sh)")
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(c_float),
                       ctypes.POINTER(c_float)]
        fn.restype = None
        lo = [float(v) for v in mins]
        hi = [float(v) for v in maxs]
        if len(lo) != 3 or len(hi) != 3:
            raise ValueError(f"goal box needs 3-vectors, got {lo} / {hi}")
        fn(self._handle(), (c_float * 3)(*lo), (c_float * 3)(*hi))

    @property
    def goal_hits(self) -> np.ndarray:
        """ZERO-COPY read-only uint8 view [num_envs]: 1 exactly on the batch
        tick that env crossed the goal box (episode already autoreset when the
        caller reads it). Mutates every step; never outlives the core."""
        if getattr(self, "_goal_hits_view", None) is None:
            fn = getattr(self._lib, "surf_goal_hits", None)
            if fn is None:
                raise RuntimeError(
                    "this surfcore build predates surf_goal_hits — "
                    "rebuild the core (build.ps1 / ./build.sh)")
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.POINTER(ctypes.c_uint8)
            ptr = fn(self._handle())
            buf = (ctypes.c_uint8 * self.num_envs).from_address(
                ctypes.addressof(ptr.contents))
            view = np.frombuffer(buf, dtype=np.uint8)
            view.flags.writeable = False
            self._goal_hits_view = view
        return self._goal_hits_view

    def force_fail(self, mask: np.ndarray) -> None:
        """Mark envs (mask[i] != 0) to end as a FAIL on their next batch tick
        — the Python-side stagnation kill. Consumed once; a goal crossing on
        the same tick wins over the kill."""
        fn = getattr(self._lib, "surf_force_fail", None)
        if fn is None:
            raise RuntimeError(
                "this surfcore build predates surf_force_fail — "
                "rebuild the core (build.ps1 / ./build.sh)")
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
        fn.restype = None
        arr = np.ascontiguousarray(mask, dtype=np.uint8)
        if arr.shape != (self._num_envs,):
            raise ValueError(f"mask must have shape ({self._num_envs},), "
                             f"got {arr.shape}")
        fn(self._handle(), arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

    def set_spawn_pool(self, states: np.ndarray) -> None:
        """Upload a spawn pool (``STATE_DTYPE`` structured array, >= 1 entry)
        for ``spawn_mode=2`` resets: each reset copies a random entry (author
        origin/yaw/velocity; episode fields are re-zeroed core-side)."""
        arr = np.ascontiguousarray(states, dtype=STATE_DTYPE)
        if arr.ndim != 1 or arr.shape[0] < 1:
            raise ValueError(f"spawn pool must be a 1-D structured array, got {arr.shape}")
        sim = self._handle()
        rc = self._lib.surf_set_spawn_pool(
            sim, arr.ctypes.data_as(ctypes.POINTER(SurfState)), c_int32(arr.shape[0]))
        if rc != 0:
            raise RuntimeError("surf_set_spawn_pool failed")

    @property
    def states_view(self) -> np.ndarray:
        """ZERO-COPY read-only structured view of the internal per-env states.
        Mutates in place every step/reset; never outlives the core. Use for
        hot-loop reward math; use :meth:`get_states` when you need a copy."""
        if getattr(self, "_states_view", None) is None:
            sim = self._handle()
            ptr = self._lib.surf_states_ptr(sim)
            buf = (SurfState * self.num_envs).from_address(ptr)
            view = np.frombuffer(buf, dtype=STATE_DTYPE)
            view.flags.writeable = False
            self._states_view = view
        return self._states_view

    def set_pitch(self, value: float) -> None:
        """Overwrite EVERY env's view pitch in place (deg, + = up).

        A narrow entry point rather than a writable ``states_view``, because
        this is the one state field that is not physics: pitch aims the lidar
        and nothing else (``src/env.c`` passes 0.0 to ``pm_tick``, and the
        state convention is inverted vs GoldSrc angles), so writing it is a
        camera move, not a teleport - no seg_hint to fix up, no velocity to
        keep consistent, nothing the C side has to be told about.

        Used by ``train_fast.py --pitch-fixed`` immediately before every
        lidar render. The value is NOT clamped here: the [-70, +30] clamp
        in env.c applies to the accumulated pitch DELTA, and a caller that
        pins an angle is stating the aim it wants.
        """
        if getattr(self, "_states_rw", None) is None:
            sim = self._handle()
            ptr = self._lib.surf_states_ptr(sim)
            buf = (SurfState * self.num_envs).from_address(ptr)
            self._states_rw = np.frombuffer(buf, dtype=STATE_DTYPE)
        self._states_rw["pitch"][:] = float(value)

    def get_states(self) -> np.ndarray:
        """Full per-env states as a structured array copy, dtype STATE_DTYPE,
        shape (N,).  The copy is yours to keep."""
        sim = self._handle()
        self._lib.surf_get_states(sim, self._p_states)
        return self._states_buf.copy()

    def set_state(self, i: int, state: Union[SurfState, np.void, np.ndarray]) -> None:
        """Teleport env ``i`` to ``state`` (a SurfState ctypes struct, or a
        STATE_DTYPE structured scalar / 1-element array)."""
        sim = self._handle()
        if not 0 <= i < self._num_envs:
            raise IndexError(f"env index {i} out of range [0, {self._num_envs})")
        if isinstance(state, SurfState):
            st = state
        elif isinstance(state, (np.void, np.ndarray)):
            arr = np.asarray(state)
            if arr.dtype != STATE_DTYPE or arr.size != 1:
                raise TypeError(
                    "numpy state must be a single element of STATE_DTYPE"
                )
            st = SurfState.from_buffer_copy(arr.tobytes())
        else:
            raise TypeError(
                f"state must be SurfState or STATE_DTYPE scalar, got "
                f"{type(state).__name__}"
            )
        self._lib.surf_set_state(sim, c_int32(i), ctypes.byref(st))

    # -- track --------------------------------------------------------------

    def set_waypoints(self, points: Union[np.ndarray, Sequence[Sequence[float]]]) -> None:
        """Set the track polyline: float array-like of shape (M, 3), M >= 2,
        map units."""
        sim = self._handle()
        pts = np.ascontiguousarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            raise ValueError(
                f"waypoints must have shape (M>=2, 3), got {pts.shape}"
            )
        rc = self._lib.surf_set_waypoints(
            sim,
            pts.ctypes.data_as(ctypes.POINTER(c_float)),
            c_int32(pts.shape[0]),
        )
        if rc != 0:
            raise ValueError(f"surf_set_waypoints rejected input (rc={rc})")

    # -- exposed internals --------------------------------------------------

    def trace(
        self,
        start: Sequence[float],
        end: Sequence[float],
        hull: int = 0,
    ) -> SurfTrace:
        """Hull trace start->end. hull: 0 stand, 1 duck, 2 point.  Returns a
        :class:`SurfTrace` (fields: endpos, normal, dist, fraction,
        startsolid, allsolid, ent)."""
        sim = self._handle()
        s = (c_float * 3)(*(float(v) for v in start))
        e = (c_float * 3)(*(float(v) for v in end))
        out = SurfTrace()
        self._lib.surf_trace(sim, s, e, c_int32(hull), ctypes.byref(out))
        return out

    def point_contents(self, p: Sequence[float]) -> int:
        """CONTENTS_* at a point (world + contents entities)."""
        sim = self._handle()
        arr = (c_float * 3)(*(float(v) for v in p))
        return int(self._lib.surf_point_contents(sim, arr))

    def occupancy_grid(self, mins: Sequence[float], cell: float,
                       nx: int, ny: int, nz: int) -> np.ndarray:
        """Solid/sky occupancy voxel grid (uint8, shape (nz, ny, nx) C-order
        as [iz, iy, ix]), sampled at voxel centers. For GPU-vision precompute."""
        sim = self._handle()
        out = np.zeros(nx * ny * nz, dtype=np.uint8)
        m = (c_float * 3)(*(float(v) for v in mins))
        self._lib.surf_occupancy_grid(
            sim, m, c_float(cell), c_int32(nx), c_int32(ny), c_int32(nz),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.reshape(nz, ny, nx)

    def pm_step_usercmd(
        self,
        state: SurfState,
        yaw: float,
        pitch: float,
        fmove: float,
        smove: float,
        buttons: int,
        msec: int,
    ) -> None:
        """One physics tick at usercmd level, mutating ``state`` in place
        (parity-harness primitive)."""
        sim = self._handle()
        self._lib.surf_pm_step_usercmd(
            sim, ctypes.byref(state), c_float(yaw), c_float(pitch),
            c_float(fmove), c_float(smove), c_int32(buttons), c_int32(msec),
        )

    # play-step return flags
    PLAY_TELEPORTED = 1
    PLAY_KILLED = 2
    PLAY_IN_WATER = 4
    PLAY_STUCK = 8

    def play_step(
        self,
        state: SurfState,
        yaw: float,
        pitch: float,
        fmove: float,
        smove: float,
        buttons: int,
        msec: int,
    ) -> int:
        """One interactive tick: usercmd physics + live map triggers
        (teleports/boosters/hurt). Mutates ``state``; returns PLAY_* flags."""
        sim = self._handle()
        return int(self._lib.surf_play_step(
            sim, ctypes.byref(state), c_float(yaw), c_float(pitch),
            c_float(fmove), c_float(smove), c_int32(buttons), c_int32(msec),
        ))

    def pm_step_single(self, state: SurfState, action: Sequence[int]) -> None:
        """Discretized single step (action encoding at top of surfcore.h),
        mutating ``state`` in place."""
        sim = self._handle()
        a = np.ascontiguousarray(action, dtype=np.int32)
        if a.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {a.shape}")
        self._lib.surf_pm_step_single(
            sim, ctypes.byref(state), a.ctypes.data_as(self._P_I32)
        )

    # -- map info -----------------------------------------------------------

    def spawns(self) -> List[Tuple[np.ndarray, float]]:
        """All map spawn points as ``[(origin (3,) float32, yaw_deg), ...]``."""
        sim = self._handle()
        n = int(self._lib.surf_num_spawns(sim))
        out: List[Tuple[np.ndarray, float]] = []
        P_F32 = ctypes.POINTER(c_float)
        for i in range(n):
            origin = np.zeros(3, dtype=np.float32)
            yaw = c_float(0.0)
            self._lib.surf_get_spawn(
                sim, c_int32(i), origin.ctypes.data_as(P_F32), ctypes.byref(yaw)
            )
            out.append((origin, float(yaw.value)))
        return out

    def map_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """World AABB as ``(mins (3,), maxs (3,))`` float32 copies."""
        sim = self._handle()
        mins = np.zeros(3, dtype=np.float32)
        maxs = np.zeros(3, dtype=np.float32)
        P_F32 = ctypes.POINTER(c_float)
        self._lib.surf_map_bounds(
            sim, mins.ctypes.data_as(P_F32), maxs.ctypes.data_as(P_F32)
        )
        return mins, maxs
