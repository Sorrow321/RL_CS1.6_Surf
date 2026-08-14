"""Non-learning policies: the physics smoke test and a random baseline.

``ScriptedStrafer`` is *not* RL — it is the port's pulse check (docs/03): a
sinusoidal strafe-sync bot.  If it gains speed on a ramp, the air-accelerate
port breathes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import ACTION_DIM, ACTION_NVEC, YAW_BINS

__all__ = ["ScriptedStrafer", "RandomPolicy"]

# Side bins per surfcore.h: a[2] side 0..2 -> sidemove {-400, 0, +400}.
# -400 = moveleft (A), +400 = moveright (D), HLSDK sign convention.
_SIDE_LEFT = 0
_SIDE_NEUTRAL = 1
_SIDE_RIGHT = 2
_FORWARD_NEUTRAL = 1


class ScriptedStrafer:
    """Sinusoidal strafe-sync bot (physics smoke test, obs-independent).

    Each tick the desired yaw delta follows one sine cycle per
    ``period_ticks``:  ``delta = yaw_rate_max_deg * sin(2*pi*t/period)`` —
    the yaw bin swings smoothly through the negative and positive halves of
    the sine (the non-uniform bins give gentle deltas near the zero
    crossings).  The sidemove sign is matched to the sign of the current yaw
    delta — strafe-sync: a positive yaw delta turns *left* (GoldSrc yaw is
    CCW-positive), paired with moveleft (sidemove -400, side bin 0).

    ``side_hold`` — ticks to keep holding the previous strafe direction while
    the chosen yaw bin is exactly 0 (the sine's zero crossings); after that
    the side key is released.  With the default 1 the bot effectively never
    lets go mid-oscillation.

    forwardmove stays neutral; jump and duck are never pressed.
    ``act(obs) -> (N, 5) int32`` returns an internal buffer reused across
    calls (copy if you keep it).
    """

    def __init__(self, core, period_ticks: int = 100, side_hold: int = 1) -> None:
        if period_ticks < 2:
            raise ValueError(f"period_ticks must be >= 2, got {period_ticks}")
        self._n = int(core.num_envs)
        self._period = int(period_ticks)
        self._side_hold = int(side_hold)
        # Bins scaled so the largest equals yaw_rate_max_deg (surfcore.h).
        rate = float(core.config.yaw_rate_max_deg)
        self._bins_deg = YAW_BINS * (rate / 10.0)
        self._rate = rate
        self._t = 0
        self._last_side = _SIDE_LEFT  # sine goes positive first -> moveleft
        self._zero_ticks = 0
        self._actions = np.empty((self._n, ACTION_DIM), dtype=np.int32)
        self._actions[:, 1] = _FORWARD_NEUTRAL
        self._actions[:, 3] = 0  # jump
        self._actions[:, 4] = 0  # duck

    def reset(self) -> None:
        """Restart the oscillation phase."""
        self._t = 0
        self._last_side = _SIDE_LEFT
        self._zero_ticks = 0

    def act(self, obs: np.ndarray) -> np.ndarray:  # obs unused (scripted)
        desired = self._rate * np.sin(2.0 * np.pi * self._t / self._period)
        self._t += 1
        yaw_bin = int(np.abs(self._bins_deg - desired).argmin())
        delta = float(self._bins_deg[yaw_bin])
        if delta > 0.0:  # turning left -> hold moveleft
            side = _SIDE_LEFT
            self._last_side = side
            self._zero_ticks = 0
        elif delta < 0.0:  # turning right -> hold moveright
            side = _SIDE_RIGHT
            self._last_side = side
            self._zero_ticks = 0
        else:  # zero crossing: hold the previous key for side_hold ticks
            self._zero_ticks += 1
            side = self._last_side if self._zero_ticks <= self._side_hold else _SIDE_NEUTRAL
        self._actions[:, 0] = yaw_bin
        self._actions[:, 2] = side
        return self._actions


class RandomPolicy:
    """Uniform random actions over MultiDiscrete([15, 3, 3, 2, 2]).

    The batch size is inferred from ``obs.shape[0]`` each call, so one
    instance works with any core.
    """

    def __init__(self, seed: Optional[int] = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._nvec = np.asarray(ACTION_NVEC, dtype=np.int32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        n = int(np.asarray(obs).shape[0])
        return self._rng.integers(
            0, self._nvec, size=(n, ACTION_DIM), dtype=np.int32
        )
