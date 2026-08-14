"""stable-baselines3 VecEnv adapter.

The C core's same-step autoreset (surfcore.h: a done env's ``obs`` row is the
new episode's first obs, ``terminal_obs`` its true final obs) maps *directly*
onto SB3's VecEnv contract — SB3 expects the post-reset obs from ``step`` and
the real final obs in ``infos[i]["terminal_observation"]``.

stable_baselines3 (and gymnasium, for the spaces) are imported lazily on
first access of ``SurfVecEnv`` — ``import surfgym.vec_env`` works without
either installed.  Requires SB3 >= 2.0 (gymnasium-based spaces).

Usage::

    from surfgym import SurfCore, default_config
    from surfgym.vec_env import SurfVecEnv          # triggers the lazy import

    core = SurfCore("maps/surf_ski_2.bsp", default_config(num_envs=1024))
    venv = SurfVecEnv(core)
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["SurfVecEnv"]


def _build_surf_vec_env() -> type:
    from stable_baselines3.common.vec_env import VecEnv  # lazy import
    import gymnasium as gym  # SB3 >= 2.0 speaks gymnasium spaces

    from .core import ACTION_NVEC, SurfCore

    class SurfVecEnv(VecEnv):
        """SB3 VecEnv over a :class:`SurfCore` batch sim.

        The adapter does not own the core: ``close()`` leaves it alive unless
        constructed with ``own_core=True``.  Returned arrays are copies (the
        core's buffers are reused every step).
        """

        def __init__(self, core: SurfCore, seed: int = 0,
                     own_core: bool = False) -> None:
            self.core = core
            self._seed = int(seed)
            self._own_core = bool(own_core)
            self._actions: np.ndarray | None = None
            observation_space = gym.spaces.Box(
                -np.inf, np.inf, (core.obs_dim,), dtype=np.float32
            )
            action_space = gym.spaces.MultiDiscrete(list(ACTION_NVEC))
            super().__init__(core.num_envs, observation_space, action_space)
            self.render_mode = None

        # -- VecEnv API ----------------------------------------------------

        def reset(self) -> np.ndarray:
            return self.core.reset(self._seed).copy()

        def step_async(self, actions: np.ndarray) -> None:
            self._actions = np.ascontiguousarray(actions, dtype=np.int32)

        def step_wait(self):
            assert self._actions is not None, "step_async not called"
            obs, rewards, done, trunc, terminal_obs = self.core.step(self._actions)
            ended = (done | trunc).astype(bool)
            infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
            for i in np.flatnonzero(ended):
                infos[i]["terminal_observation"] = terminal_obs[i].copy()
                if trunc[i] and not done[i]:
                    infos[i]["TimeLimit.truncated"] = True
            return obs.copy(), rewards.copy(), ended, infos

        def close(self) -> None:
            if self._own_core:
                self.core.close()

        def seed(self, seed: int | None = None):
            if seed is not None:
                self._seed = int(seed)
            return [self._seed] * self.num_envs

        # -- attribute plumbing (no per-env Python objects exist) ----------

        def _indices(self, indices) -> list[int]:
            if indices is None:
                return list(range(self.num_envs))
            if isinstance(indices, int):
                return [indices]
            return list(indices)

        def get_attr(self, attr_name: str, indices=None) -> list:
            target = self.core if hasattr(self.core, attr_name) else self
            value = getattr(target, attr_name)
            return [value for _ in self._indices(indices)]

        def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
            setattr(self, attr_name, value)

        def env_method(self, method_name: str, *method_args, indices=None,
                       **method_kwargs) -> list:
            raise NotImplementedError(
                "SurfVecEnv has no per-env Python objects to call methods on"
            )

        def env_is_wrapped(self, wrapper_class, indices=None) -> list:
            return [False for _ in self._indices(indices)]

        def get_images(self):
            raise NotImplementedError("SurfVecEnv does not render")

    return SurfVecEnv


def __getattr__(name: str):  # PEP 562 — lazy class construction
    if name == "SurfVecEnv":
        cls = _build_surf_vec_env()
        globals()["SurfVecEnv"] = cls  # cache for subsequent lookups
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
