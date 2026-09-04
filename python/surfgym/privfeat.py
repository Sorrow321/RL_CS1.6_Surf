"""--priv-critic: the privileged state block the asymmetric critic reads.

Asymmetric actor-critic (Pinto et al. 2017, "Asymmetric Actor Critic for
Image-Based Robot Learning"; the same arrangement OpenAI's Dactyl and the
Isaac Gym legged-locomotion stack use): the critic is a TRAINING-TIME
object, so it may read the simulator's state directly while the actor keeps
the deployable observation.  The actor's input is untouched by everything in
this file - the deployed policy is identical in form to the control's.

Everything the critic is given here is state the trainer ALREADY has in
hand once per decision: the core's own ``states_view`` (position, velocity,
tick) and the numbers ``RaceReward`` computed for this very tick (the
geodesic distance the shaping term is built from, the arc anchor, the
--race-latch flag).  Nothing new is simulated, traced or rendered.

The block is TEN columns, in this order, per env:

===  ==========  ===================================================
  0  ``pos_x``   (origin.x - map_center.x) / map_scale
  1  ``pos_y``   (origin.y - map_center.y) / map_scale
  2  ``pos_z``   (origin.z - map_center.z) / map_scale
  3  ``vel_x``   velocity.x / 4000
  4  ``vel_y``   velocity.y / 4000
  5  ``vel_z``   velocity.z / 4000
  6  ``d_frac``  d / d0 - the geodesic distance to the goal that
                 ``RaceReward`` shapes on, over the map's START geodesic
                 (``rf_d0``, the mean over the spawn pool)
  7  ``arc_frac``arc / route_length under --race-arc, else a constant 0
  8  ``t_frac``  states_view.tick / ep_ticks (1.0 at a truncation)
  9  ``latch``   1.0 once --race-latch has fired this episode, else 0.0
===  ==========  ===================================================

``map_center`` is the BSP bounds' midpoint (the same centre obs slots 12..14
use) and ``map_scale`` is the LARGEST half-extent of those bounds, so the
three position columns share ONE scale - a per-axis normalisation would
make the block anisotropic and a 8192 constant would put a small map in a
corner of its range.  Positions land in [-1, 1] inside the map; nothing here
is clipped, because a critic feature that silently saturates hides exactly
the states (a long fall, a spawn further out than the pool mean) that the
value function is hardest on.

The columns are held in ONE implementation on purpose.  The trainer feeds
this block from three places - the rollout, the truncation bootstrap's
V(s_T) and the eval recording - and two of the three are reconstructions;
CLAUDE.md's frame-ring and route-fan notes are both about the same failure,
a second copy of an observation drifting out of step with the first.
"""
from __future__ import annotations

import numpy as np

#: column names, in order; ``len`` is the width of the block
PRIV_FEATURES = ("pos_x", "pos_y", "pos_z",
                 "vel_x", "vel_y", "vel_z",
                 "d_frac", "arc_frac", "t_frac", "latch")
PRIV_DIM = len(PRIV_FEATURES)

#: velocity normaliser.  A FIXED constant, not --maxvel: the block has to
#: mean the same thing across runs for two arms to be comparable, and 4000
#: is the horizontal cap every arm in this project has used.
VEL_SCALE = 4000.0


class PrivFeat:
    """Per-map normalisation for the privileged block.

    One instance per map slot (the centre, the scale and d0 are all that
    map's).  Stateless apart from those constants - the per-env numbers are
    passed in by whichever caller has them.
    """

    dim = PRIV_DIM
    names = PRIV_FEATURES

    def __init__(self, map_center, map_scale: float, d0: float,
                 ep_ticks: int, arc_len: float = 0.0):
        self.center = np.asarray(map_center, np.float64).reshape(3)
        self.scale = float(max(float(map_scale), 1.0))
        self.d0 = float(max(float(d0), 1.0))
        self.ep_ticks = float(max(int(ep_ticks), 1))
        self.arc_len = float(arc_len or 0.0)

    def describe(self) -> str:
        return (f"priv-critic: {self.dim} columns {list(self.names)} | "
                f"centre {np.round(self.center, 1).tolist()} scale "
                f"{self.scale:,.0f}u | d0 {self.d0:,.0f}u | ep_ticks "
                f"{self.ep_ticks:,.0f}"
                + (f" | arc {self.arc_len:,.0f}u" if self.arc_len > 0.0
                   else " | no arc line (column 7 is 0)"))

    # -- the one implementation ---------------------------------------------
    def fill(self, out, pos, vel, d, tick, arc=None, latch=None) -> None:
        """Write the block for these rows into ``out`` (n, 10) float32.

        ``pos`` (n, 3), ``vel`` (n, 3), ``d`` (n,), ``tick`` (n,) are
        required; ``arc`` (n,) is the arc anchor under --race-arc and
        ``latch`` (n,) the --race-latch flag, either of which may be None
        (the column is then left at zero, which is what the run without
        that flag has in it for the whole run).
        """
        n = out.shape[0]
        if out.shape[1] != self.dim:
            raise ValueError(f"priv block is {self.dim} wide, got "
                             f"{out.shape[1]}")
        p = np.asarray(pos, np.float64).reshape(n, 3)
        v = np.asarray(vel, np.float64).reshape(n, 3)
        out[:, 0:3] = ((p - self.center) / self.scale).astype(np.float32)
        out[:, 3:6] = (v / VEL_SCALE).astype(np.float32)
        out[:, 6] = (np.asarray(d, np.float64).reshape(n)
                     / self.d0).astype(np.float32)
        if arc is not None and self.arc_len > 0.0:
            out[:, 7] = (np.asarray(arc, np.float64).reshape(n)
                         / self.arc_len).astype(np.float32)
        else:
            out[:, 7] = 0.0
        out[:, 8] = (np.asarray(tick, np.float64).reshape(n)
                     / self.ep_ticks).astype(np.float32)
        if latch is None:
            out[:, 9] = 0.0
        else:
            out[:, 9] = np.asarray(latch, bool).reshape(n).astype(np.float32)

    def fill_live(self, out, states, reward_fn) -> None:
        """The rollout's call: everything comes off the live core state and
        the reward object that just ran on it.

        Called AFTER the reward call and after the autoreset, so ``_d``,
        the arc anchor and the latch flag all describe the state the policy
        is about to act on - the same instant ``fill_vision`` renders.
        """
        d = reward_fn.dist_now()
        if d is None:                       # before the first on_reset
            d = np.zeros(out.shape[0], np.float64)
        arc = None if reward_fn.arc is None else reward_fn.arc.arc
        self.fill(out, states["origin"], states["velocity"], d,
                  states["tick"], arc=arc, latch=reward_fn.latch_flags())


def velocity_from_obs(obs) -> np.ndarray:
    """World velocity out of a scalar obs row, exactly (n, 3) float64.

    ``write_obs`` (src/env.c) stores the velocity in the EGO frame:
    ``o0 = ( vx*cy + vy*sy)/1000``, ``o1 = (-vx*sy + vy*cy)/1000``,
    ``o2 = vz/1000`` with ``o7 = sin(yaw)``, ``o8 = cos(yaw)``.  That
    rotation is orthonormal, so inverting it recovers the world vector with
    no information lost - which is what lets the truncation bootstrap build
    the privileged block for a terminal state the core has already reset
    away from.
    """
    o = np.asarray(obs, np.float64)
    sy, cy = o[:, 7], o[:, 8]
    return np.stack([(o[:, 0] * cy - o[:, 1] * sy) * 1000.0,
                     (o[:, 0] * sy + o[:, 1] * cy) * 1000.0,
                     o[:, 2] * 1000.0], axis=1)
