"""--obs-surf: one observation column, 1 while the player rides a surfable face.

The 15 core scalars carry "on walkable ground" (slot 4, ``onground != -1``)
but nothing that separates riding a ramp from free fall: both are AIRBORNE
in the engine (pm.c sets onground only at plane normal z >= 0.7), so slot 4
reads 0 in both, and the policy had to infer surfing from the image and from
vz across decisions. This module supplies that bit (user, 2026-10-19: "an
additional observation that is a binary flag, 1 when the agent is surfing").

The test is the contact test --surf-bonus pays on (surfgym.rewards.
RaceReward), taken over one DECISION (K physics ticks) instead of one tick:

    airborne now  AND  airborne at the previous decision
    AND  (vz - vz_prev) + g * dt * K > dvz           (0 in exact free fall)
    AND  vz above the -sv_maxvelocity clamp

* the map pushed the player back UP during the decision - a ramp under it;
* a near-vertical wall clips only horizontal velocity, so it reads 0;
* a ceiling pushes down (negative), so it reads 0;
* the previous-decision airborne test drops the take-off decision, where a
  jump's impulse would otherwise read as support;
* pm.c clamps every velocity axis to +-sv_maxvelocity, so a fall pinned at
  the clamp stops losing vz and would read as support - excluded;
* the first decision of an episode reads 0 (nothing to compare with).

ONE class serves the rollout and the evals (the ObsAux argument,
surfgym/obsaux.py: two implementations of one feature drift, and the drift
shows up only as recordings that disagree with training). The trainer calls
``features`` once per decision, after ``reset(ended)``; an eval wrapper calls
``eval_features``, which finds episode starts off the core's per-env tick
counter the way the --rnn state and --keys-hold do: between two decisions
exactly K ticks elapse, so a counter that advanced by less was reset.
"""
from __future__ import annotations

import numpy as np

# u/s of vertical give-back over a decision before it counts as support: the
# reward's per-tick threshold (RaceReward.surf_dvz). Free fall gives exactly 0
# and a ramp gives far more, so the value sits on a wide plateau.
SURF_DVZ = 1.0


def g_step_of(sv_gravity: float, tick_ms: float, ticks: int) -> float:
    """The free-fall vz loss over one decision, u/s: g * dt * K."""
    return float(sv_gravity) * float(tick_ms) * 1e-3 * int(ticks)


class SurfFlag:
    """The --obs-surf column for one fleet of envs (see the module doc).

    Parameters
    ----------
    n_envs : int
    g_step : float
        ``g_step_of(sv_gravity, tick_ms, K)``.
    ticks : int
        K, physics ticks per decision (the eval path's episode-start rule).
    vmax : float
        The core's sv_maxvelocity; 0 = no clamp exclusion.
    dvz : float
        Give-back threshold, u/s.
    """

    def __init__(self, n_envs: int, g_step: float, ticks: int,
                 vmax: float = 0.0, dvz: float = SURF_DVZ):
        self.n = int(n_envs)
        self.g_step = float(g_step)
        self.ticks = max(1, int(ticks))
        self.vmax = float(vmax)
        self.dvz = float(dvz)
        self._vz = np.full(self.n, np.nan, np.float64)   # vz at the previous decision
        self._air = np.zeros(self.n, bool)                # airborne then
        self._prev_tick = None
        self.flag = np.zeros(self.n, np.float32)          # the value last computed

    def reset(self, mask) -> None:
        """Forget the marked envs' previous decision (an episode started).

        `mask` is a boolean (n,) array or an index array - whatever the
        caller already has for `ended`."""
        m = np.asarray(mask)
        if m.dtype == bool and not m.any():
            return
        self._vz[m] = np.nan
        self._air[m] = False
        self.flag[m] = 0.0

    def features(self, onground, vz, out=None) -> np.ndarray:
        """(n,) float32 in {0, 1} for the state the policy is about to act
        on; advances the tracker. One call per DECISION."""
        air = np.asarray(onground) == -1
        vz = np.asarray(vz, np.float64)
        with np.errstate(invalid="ignore"):
            dev = (vz - self._vz) + self.g_step          # NaN after a reset
            sup = air & self._air & (dev > self.dvz)     # NaN compares False
        if self.vmax > 0.0:
            sup &= vz > -(self.vmax - 0.5)
        self.flag[:] = sup
        self._vz[:] = vz
        self._air[:] = air
        if out is None:
            return self.flag
        out[:] = self.flag
        return out

    def note_ticks(self, tick) -> None:
        """Episode starts read off the core's per-env tick counter (the eval
        path's only episode signal; see the module doc)."""
        t = np.asarray(tick, np.int64)
        if self._prev_tick is not None and len(self._prev_tick) == len(t):
            started = t < self._prev_tick + self.ticks
            if started.any():
                self.reset(started)
        self._prev_tick = t.copy()

    def eval_features(self, onground, vz, tick) -> np.ndarray:
        """``features`` with the eval path's episode-start detection."""
        self.note_ticks(tick)
        return self.features(onground, vz)


def make_surf_feed(sv_gravity: float, tick_ms: float, ticks: int,
                   vmax: float = 0.0):
    """The eval wrappers' ``surf_fn(core, n)``. Build one per recording: the
    tracker it closes over is that recording's (its first decision reads 0)."""
    st = {"f": None}

    def feed(core, n):
        f = st["f"]
        if f is None or f.n != n:
            f = st["f"] = SurfFlag(n, g_step_of(sv_gravity, tick_ms, ticks),
                                   ticks, vmax=vmax)
        sv = core.states_view
        return f.eval_features(sv["onground"],
                               np.asarray(sv["velocity"])[:, 2], sv["tick"])
    return feed
