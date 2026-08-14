"""surfgym — Python bindings and env adapters for the RL_Surf simulation core.

Core (numpy-only, no DLL touched at import time):

* :class:`SurfCore` — ctypes binding to surfcore.dll (src/surfcore.h).
* :func:`default_config` / :class:`SurfEnvConfig` / :class:`SurfPhys` —
  config helpers with the documented surf defaults.
* :class:`ScriptedStrafer` / :class:`RandomPolicy` — non-learning policies.
* :func:`record_rollout` — trajectory JSONL writer (docs/04).

Adapters (import their heavy deps lazily; not re-exported here so that
``import surfgym`` stays dependency-free):

* ``surfgym.vec_env.SurfVecEnv`` — stable-baselines3 VecEnv.
* ``surfgym.gym_env.SurfGymEnv`` / ``SurfGymVecEnv`` — gymnasium Env /
  vector.VectorEnv.
"""
from . import rewards  # noqa: F401  (numpy-only; safe at import time)
from .core import (
    ACTION_DIM,
    ACTION_NVEC,
    STATE_DTYPE,
    SURF_IN_DUCK,
    SURF_IN_JUMP,
    YAW_BINS,
    SurfCore,
    SurfEnvConfig,
    SurfPhys,
    SurfState,
    SurfTrace,
    config_to_dict,
    default_config,
    phys_to_dict,
    validate_actions,
)
from .policies import RandomPolicy, ScriptedStrafer
from .record import record_rollout

__version__ = "0.1.0"

__all__ = [
    "ACTION_DIM",
    "ACTION_NVEC",
    "STATE_DTYPE",
    "SURF_IN_DUCK",
    "SURF_IN_JUMP",
    "YAW_BINS",
    "SurfCore",
    "SurfEnvConfig",
    "SurfPhys",
    "SurfState",
    "SurfTrace",
    "config_to_dict",
    "default_config",
    "phys_to_dict",
    "validate_actions",
    "RandomPolicy",
    "ScriptedStrafer",
    "record_rollout",
    "__version__",
]
