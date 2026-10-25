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
import os as _os


def _usable_cpus() -> int:
    """CPUs this process may use: the affinity mask capped by the cgroup CFS quota (a
    container's share of a bigger host; train_fast._cgroup_cpu_quota's rule)."""
    try:
        n = len(_os.sched_getaffinity(0))
    except AttributeError:                    # pragma: no cover - Windows
        n = _os.cpu_count() or 8
    try:
        raw = open("/sys/fs/cgroup/cpu.max").read().split()
        if raw[0] != "max":
            n = min(n, max(1, int(float(raw[0]) / float(raw[1]))))
    except (OSError, ValueError, IndexError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            pr = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0 and pr > 0:
                n = min(n, max(1, q // pr))
        except (OSError, ValueError):
            pass
    return n


def default_numba_threads(env=None) -> str:
    """The NUMBA_NUM_THREADS default: OMP_NUM_THREADS when it is set (train_fast sets it before
    it imports this package), else half the usable CPUs, 4..32 (train_fast's OMP rule).

    numba sizes its parallel pool to every VISIBLE core and ignores the CFS quota. On box
    57632609 (pbr4b3, 2026-10-25: a Threadripper with 64 visible threads under a 30.7-CPU
    quota) the goal field's prange sampler ran a 64-thread spinning OpenMP team beside the
    core's own 15-thread team and the trainer fell from ~250k to ~7k steps/s (60 runnable
    threads, load average ~100). The sampler is element-wise, so its result does not depend
    on the thread count."""
    env = _os.environ if env is None else env
    omp = str(env.get("OMP_NUM_THREADS", "")).strip()
    if omp.isdigit() and int(omp) > 0:
        return omp
    return str(max(4, min(32, _usable_cpus() // 2)))


# numba reads NUMBA_NUM_THREADS when it is first imported, and goalfield (through rewards)
# is the first numba user in every trainer and tool process: set before that import
_os.environ.setdefault("NUMBA_NUM_THREADS", default_numba_threads())

from . import rewards  # noqa: F401,E402  (numpy-only; safe at import time)
from .core import (
    ACTION_DIM,
    ACTION_NVEC,
    STATE_DTYPE,
    SURF_IN_DUCK,
    SURF_IN_JUMP,
    YAW_BINS,
    PITCH_BINS,
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
    "PITCH_BINS",
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
