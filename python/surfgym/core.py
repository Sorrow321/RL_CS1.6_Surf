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
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "SURF_IN_JUMP",
    "SURF_IN_DUCK",
    "ACTION_DIM",
    "ACTION_NVEC",
    "YAW_BINS",
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

ACTION_DIM = 5
ACTION_NVEC: Tuple[int, ...] = (15, 3, 3, 2, 2)  # MultiDiscrete nvec (docs/03)

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
        ("phys", SurfPhys),
    ]


class SurfState(ctypes.Structure):
    """Mirror of ``SurfState`` (surfcore.h) — full per-env POD state."""

    _fields_ = [
        ("origin", c_float * 3),
        ("velocity", c_float * 3),
        ("basevelocity", c_float * 3),
        ("yaw", c_float),
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
    "enable_duck": 0,
}

_ENV_DEFAULTS = {
    "max_episode_ticks": 6000,
    "lookahead_k": 8,
    "lookahead_dt": 0.25,
    "spawn_mode": 1,
    "spawn_curriculum_frac": 0.99,
    "yaw_rate_max_deg": 10.0,
    "yaw_jitter_deg": 5.0,
    "kill_z": -1e38,  # <= -1e30 -> auto (map min z - 256)
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
    cfg.yaw_jitter_deg = float(env_vals["yaw_jitter_deg"])
    cfg.kill_z = float(env_vals["kill_z"])
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

    Requires a C-contiguous int32 ndarray of shape (num_envs, 5).  Raises
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
    """Absolute candidate paths for surfcore.dll, in search order:
    explicit ``dll_path`` arg, ``SURFCORE_DLL`` env var, ``<repo>/build/``,
    ``<repo>/``."""
    cands: List[Path] = []
    if dll_path:
        cands.append(Path(dll_path).resolve())
    env = os.environ.get("SURFCORE_DLL")
    if env:
        cands.append(Path(env).resolve())
    cands.append(_REPO_ROOT / "build" / "surfcore.dll")
    cands.append(_REPO_ROOT / "surfcore.dll")
    return cands


def _load_library(dll_path: Optional[Union[str, os.PathLike]] = None) -> ctypes.CDLL:
    cands = _dll_candidates(dll_path)
    path = next((p for p in cands if p.is_file()), None)
    if path is None:
        tried = "\n  ".join(str(p) for p in cands)
        raise FileNotFoundError(
            "surfcore.dll not found. Build it first (see build.ps1), or point "
            "SURFCORE_DLL / dll_path= at it. Paths tried, in order:\n  "
            f"{tried}"
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
        # hot path
        "surf_reset_all": ([ctypes.c_void_p, ctypes.c_uint64, P_F32], None),
        "surf_step": ([ctypes.c_void_p, P_I32, P_F32, P_F32, P_U8, P_U8,
                       P_F32], None),
        # state access
        "surf_get_states": ([ctypes.c_void_p, P_STATE], None),
        "surf_set_state": ([ctypes.c_void_p, c_int32, P_STATE], None),
        # exposed internals
        "surf_trace": ([ctypes.c_void_p, P_F32, P_F32, c_int32,
                        ctypes.POINTER(SurfTrace)], None),
        "surf_point_contents": ([ctypes.c_void_p, P_F32], c_int32),
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

    Buffer ownership: :meth:`reset` and :meth:`step` return views of internal
    preallocated buffers, valid only until the next ``reset``/``step`` call
    on this instance.  Copy (``.copy()``) anything you keep.
    """

    def __init__(
        self,
        bsp_path: Union[str, os.PathLike],
        config: Optional[SurfEnvConfig] = None,
        dll_path: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        self._sim: Optional[int] = None  # set early so __del__ is always safe
        bsp_path = str(bsp_path)
        if not os.path.isfile(bsp_path):
            raise FileNotFoundError(f"BSP not found: {bsp_path}")
        if config is None:
            config = default_config(num_envs=1)
        if not isinstance(config, SurfEnvConfig):
            raise TypeError("config must be a SurfEnvConfig (see default_config)")
        if config.num_envs < 1:
            raise ValueError(f"config.num_envs must be >= 1, got {config.num_envs}")

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

    # -- properties ---------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

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
        return self._obs

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance every env one tick.

        actions : int32 (N, 5), C-contiguous — [yaw_bin, forward, side, jump,
        duck] per surfcore.h (a[1] is forward, a[2] is side).

        Returns ``(obs, rewards, done, trunc, terminal_obs)`` — all views of
        internal buffers, valid only until the next ``step``/``reset`` call.
        Same-step autoreset: where ``done|trunc``, ``obs[i]`` is the *new*
        episode's first obs and ``terminal_obs[i]`` the ended episode's true
        final obs (rows of non-done envs are untouched garbage).
        """
        sim = self._handle()
        validate_actions(actions, self._num_envs)
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
