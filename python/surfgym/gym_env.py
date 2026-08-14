"""gymnasium adapters: single Env and vector.VectorEnv.

Both classes are built lazily on first attribute access (PEP 562), so
``import surfgym.gym_env`` succeeds without gymnasium installed.

Autoreset shims — the C core resets done envs in place the *same* step
(surfcore.h): ``obs[i]`` already holds the new episode's first obs and
``terminal_obs[i]`` the ended episode's true final obs.

* ``SurfGymEnv`` (single env): a done ``step`` returns the *terminal* obs
  with the flags, and stashes the core's in-place reset obs; the following
  ``reset()`` (called without a seed) hands that stashed obs back instead of
  re-resetting — matching the gymnasium episode boundary exactly.

* ``SurfGymVecEnv`` (gymnasium >= 1.0 next-step autoreset convention): the
  stored reset obs is delayed one step.  At the step where env ``i`` ends we
  report ``terminal_obs[i]`` with terminated/truncated set and stash the
  core's reset obs; on the *next* step we report that stashed obs with
  reward 0 and both flags False.  Two documented distortions vs. the letter
  of the convention: (1) the action submitted on the reset step cannot be
  ignored — the core steps all envs in lockstep — so it becomes the new
  episode's real first action and (2) the reward that first action earned is
  reported as 0 (convention) and dropped; compliant training loops (e.g.
  CleanRL) skip the reset-step transition anyway.  If an episode ends *on*
  its own reset step (a 1-tick episode) it is folded away entirely — its
  termination is never reported.  With max_episode_ticks in the thousands
  this is unobservable in practice.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = ["SurfGymEnv", "SurfGymVecEnv"]


def _build() -> dict:
    import gymnasium as gym  # lazy import
    from gymnasium import spaces
    from gymnasium.vector import VectorEnv
    from gymnasium.vector.utils import batch_space

    from .core import ACTION_DIM, ACTION_NVEC, SurfCore, default_config

    def _single_spaces(obs_dim: int):
        obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        act_space = spaces.MultiDiscrete(list(ACTION_NVEC))
        return obs_space, act_space

    class SurfGymEnv(gym.Env):
        """Single-env gymnasium.Env over a ``num_envs=1`` :class:`SurfCore`.

        Build from a map (``SurfGymEnv("maps/x.bsp")`` — owns its core) or
        wrap an existing 1-env core (``SurfGymEnv(core=core)``).
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            bsp_path: Optional[str] = None,
            config=None,
            dll_path: Optional[str] = None,
            core: Optional[SurfCore] = None,
        ) -> None:
            if core is None:
                if bsp_path is None:
                    raise TypeError("SurfGymEnv needs bsp_path= or core=")
                if config is None:
                    config = default_config(num_envs=1)
                if config.num_envs != 1:
                    raise ValueError(
                        f"SurfGymEnv is single-env; config.num_envs="
                        f"{config.num_envs}"
                    )
                core = SurfCore(bsp_path, config, dll_path=dll_path)
                self._own_core = True
            else:
                if core.num_envs != 1:
                    raise ValueError(
                        f"SurfGymEnv needs a num_envs=1 core, got "
                        f"{core.num_envs}"
                    )
                self._own_core = False
            self.core = core
            self.observation_space, self.action_space = _single_spaces(core.obs_dim)
            self.render_mode = None
            self._pending_reset_obs: Optional[np.ndarray] = None
            self._needs_reset = True

        def reset(self, *, seed: Optional[int] = None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                obs = self.core.reset(seed)[0].copy()
            elif self._pending_reset_obs is not None:
                # Same-step autoreset already happened inside the core; hand
                # back the stashed first obs of the new episode.
                obs = self._pending_reset_obs
            else:
                obs = self.core.reset(int(self.np_random.integers(2 ** 63)))[0].copy()
            self._pending_reset_obs = None
            self._needs_reset = False
            return obs, {}

        def step(self, action):
            if self._needs_reset:
                raise RuntimeError("call reset() before step()")
            a = np.ascontiguousarray(action, dtype=np.int32).reshape(1, ACTION_DIM)
            obs, rewards, done, trunc, terminal_obs = self.core.step(a)
            terminated = bool(done[0])
            truncated = bool(trunc[0])
            if terminated or truncated:
                # obs[0] is already the NEXT episode's first obs (in-place
                # autoreset) — stash it for reset(), return the true final obs.
                self._pending_reset_obs = obs[0].copy()
                self._needs_reset = True
                out = terminal_obs[0].copy()
            else:
                out = obs[0].copy()
            return out, float(rewards[0]), terminated, truncated, {}

        def close(self) -> None:
            if self._own_core:
                self.core.close()

    class SurfGymVecEnv(VectorEnv):
        """gymnasium.vector.VectorEnv over a :class:`SurfCore` batch sim,
        shimming the core's same-step autoreset to gymnasium >= 1.0's
        next-step convention (see module docstring for the shim contract).
        """

        def __init__(self, core: SurfCore, own_core: bool = False) -> None:
            self.core = core
            self._own_core = bool(own_core)
            single_obs, single_act = _single_spaces(core.obs_dim)
            try:
                super().__init__()  # gymnasium >= 1.0: no-arg init
            except TypeError:  # gymnasium 0.2x: (num_envs, obs_space, act_space)
                super().__init__(core.num_envs, single_obs, single_act)
            self.num_envs = core.num_envs
            self.single_observation_space = single_obs
            self.single_action_space = single_act
            self.observation_space = batch_space(single_obs, core.num_envs)
            self.action_space = batch_space(single_act, core.num_envs)
            self.render_mode = None
            try:  # advertise the convention where gymnasium understands it
                from gymnasium.vector import AutoresetMode
                self.metadata = {**self.metadata,
                                 "autoreset_mode": AutoresetMode.NEXT_STEP}
            except ImportError:
                pass
            self._pending = np.zeros(core.num_envs, dtype=bool)
            self._stored = np.zeros((core.num_envs, core.obs_dim),
                                    dtype=np.float32)

        def reset(self, *, seed: Optional[int] = None, options=None):
            obs = self.core.reset(0 if seed is None else int(seed))
            self._pending[:] = False
            return obs.copy(), {}

        def step(self, actions):
            a = np.ascontiguousarray(actions, dtype=np.int32)
            prev_pending = self._pending
            obs, rewards, done, trunc, terminal_obs = self.core.step(a)
            ended = (done | trunc).astype(bool)

            out_obs = obs.copy()
            out_rew = rewards.astype(np.float32, copy=True)
            out_term = done.astype(bool)
            out_trunc = trunc.astype(bool)

            # Envs that ended this tick: report the true final obs now...
            end_fresh = ended & ~prev_pending
            out_obs[end_fresh] = terminal_obs[end_fresh]
            # ...except envs owed a reset obs from last step: those emit the
            # delayed reset obs with reward 0 and clear flags (next-step
            # convention).  Read _stored BEFORE overwriting it below.
            out_obs[prev_pending] = self._stored[prev_pending]
            out_rew[prev_pending] = 0.0
            out_term[prev_pending] = False
            out_trunc[prev_pending] = False

            # Stash the in-place reset's first obs for every env that ended
            # (a pending env that ended again is folded — see module doc).
            self._stored[ended] = obs[ended]
            self._pending = ended
            return out_obs, out_rew, out_term, out_trunc, {}

        def close_extras(self, **kwargs) -> None:
            if self._own_core:
                self.core.close()

    return {"SurfGymEnv": SurfGymEnv, "SurfGymVecEnv": SurfGymVecEnv}


def __getattr__(name: str):  # PEP 562 — lazy class construction
    if name in ("SurfGymEnv", "SurfGymVecEnv"):
        classes = _build()
        globals().update(classes)  # cache both for subsequent lookups
        return classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
