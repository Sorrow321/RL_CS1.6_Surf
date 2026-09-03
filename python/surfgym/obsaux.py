"""Handcrafted scalar-side observation blocks: ``--act-hist`` and
``--obs-compass``.

Both live in ONE class, and ``python/train_fast.py`` and
``tools/record_ckpt.py`` instantiate the SAME class, for the reason
``tests/python/test_framestack.py`` gives about the frame ring: two
implementations of one feature drift, and the drift only shows up as eval
recordings that disagree with training - on ``race/eval_progress``, which is
the number every arm is judged by.

Row layout of the scalar half (train_fast.py builds it in this order)::

    [ 15 core | N_FAN route fan | N_LATCH | 6*K act-hist | 5 compass | image ]

The two blocks are concatenated AFTER the latch, i.e. they are the TRAILING
part of the scalar half. That is deliberate: ``train_fast.widen_for_route``
warm-starts a checkpoint by zero-padding the first Linear's TRAILING columns,
so growing these blocks onto a checkpoint that has never seen them is
function-identical at step 0. It also means the growth is only safe while the
old block stays a PREFIX of the new one - train_fast.py refuses the rest.

--act-hist K: the last K DECISIONS, MOST RECENT FIRST
-----------------------------------------------------
One decision is the six-tuple the engine receives (``surfcore.h`` action
encoding): ``[yaw bin, pitch bin, forward, side, jump, duck]``.  It is encoded
scale-free into six numbers in [-1, 1], one per head:

===  =====================================================================
0    signed yaw delta / max yaw delta = ``YAW_BINS[a0] / 10``  (index 7 = 0).
     Under ``--yaw-adaptive`` the bin means a MULTIPLE of the optimal-strafe
     rate instead (``src/env.c`` ``K_BINS``), so the table is ``K_BINS / 20``
     - still the bin's own share of full turn authority, still monotone, and
     still in [-1, 1].
1    signed pitch delta / max pitch delta = ``PITCH_BINS[a1] / 10``
     (index 3 = 0, positive = look up).
2    forwardmove in {-1, 0, +1} = ``a2 - 1``  (engine {-400, 0, +400}).
3    sidemove    in {-1, 0, +1} = ``a3 - 1``.
4    jump held   in {0, 1}.
5    duck held   in {0, 1}.
===  =====================================================================

Slot 0 duplicates the core's own scalar 10 ("previous yaw action delta /
yaw_rate_max_deg") at K=1 BY CONSTRUCTION - that is the encoding this file
adopts, not an accident, and it is what makes K>1 a straight extension of a
feature the policy already reads rather than a second convention.

The hypothesis this feature exists to test: air-strafing needs the strafe key
and the yaw change PHASE-LOCKED, and a memoryless policy cannot see the phase
it is in.

--obs-compass: five numbers off the shaping distance field
----------------------------------------------------------
0..2  the field's DOWNHILL direction at the agent, a unit vector in the EGO
      frame ``(forward, left, up)`` - the same frame and the same sign
      convention as the velocity scalars and the route fan
      (``src/env.c``: forward = ``(cos yaw, sin yaw)``, left =
      ``(-sin yaw, cos yaw)``), so "the finish is to my left" is the same
      number whichever way the map faces.
3     ``d / d0`` - the distance the field reports over the distance it
      reported at this episode's start, clipped to [0, 4].
4     ``|grad d|`` clipped to [0, 1].  A geodesic field descends at exactly
      1 unit per unit along a real path, so 1 = "there is a way down here"
      and 0 = flat, undefined, or off the reachable set.

Where the field holds its unreachable SENTINEL all five are zero: a sentinel
differenced against a real distance is not a gradient, it is the edge of the
bake.  ``GoalField`` marks that with ``sample() >= reach_max - cell/2`` (the
same test its own ``reachable()`` uses); ``EuclidField`` and
``GoalDistField`` have no unreachable set and read as valid everywhere.

The field is whichever one ``RaceReward`` actually shapes on - the map's
geodesic ``reward_field``, or the per-env ``GoalDistField`` under
``--goal-reward euclid/geo``, whose ``sample()`` is row-aligned to envs (row i
is measured against env i's own goal).  That row alignment is why every probe
below is sampled at FULL fleet width and sliced afterwards, never as a short
array.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ACT_FEAT", "CMP_FEAT", "ObsAux", "yaw_hist_table",
           "pitch_hist_table"]

ACT_FEAT = 6          # one decision -> six numbers, one per action head
CMP_FEAT = 5          # ego downhill (fwd, left, up) + d/d0 + |grad|

# Multiples of the optimal-strafe turn rate, used when cfg.yaw_adaptive.
# Mirrors src/env.c K_BINS exactly (the C table is the authority; this is a
# normalisation table, not a second definition of the action space).
_K_BINS = np.array([-20.0, -8.0, -3.0, -1.5, -1.0, -0.75, -0.5, 0.0,
                    0.5, 0.75, 1.0, 1.5, 3.0, 8.0, 20.0], np.float32)

_EPS = 1e-9


def yaw_hist_table(yaw_adaptive: bool = False) -> np.ndarray:
    """Yaw bin -> [-1, 1].  ``YAW_BINS / 10`` normally, ``K_BINS / 20`` under
    ``--yaw-adaptive``; both are the bin's share of full turn authority."""
    from surfgym.core import YAW_BINS
    tab = (_K_BINS / 20.0) if yaw_adaptive else (np.asarray(YAW_BINS,
                                                            np.float32) / 10.0)
    return np.ascontiguousarray(tab, np.float32)


def pitch_hist_table() -> np.ndarray:
    """Pitch bin -> [-1, 1] (``PITCH_BINS / 10``; index 3 = 0, + = up)."""
    from surfgym.core import PITCH_BINS
    return np.ascontiguousarray(np.asarray(PITCH_BINS, np.float32) / 10.0,
                                np.float32)


class ObsAux:
    """The ``--act-hist`` + ``--obs-compass`` block for one fleet of envs.

    Parameters
    ----------
    n_envs : int
    k : int
        ``--act-hist``: how many past decisions to carry.  0 = the block is
        absent and ``n_features`` loses ``6*k``.
    field : None | object | sequence of (slice, object)
        ``--obs-compass``: the distance field the reward shapes on.  A
        sequence of ``(slice, field)`` gives one field per map slot (the
        multi-map fleet); a bare object covers every env.  None = no compass
        block.
    yaw_adaptive : bool
        Which yaw table to encode with (see the module docstring).
    d_clip : float
        Ceiling on ``d/d0``.  4.0 matches ``RouteLine.clamp``.
    """

    def __init__(self, n_envs: int, k: int = 0, field=None,
                 yaw_adaptive: bool = False, d_clip: float = 4.0):
        self.n = int(n_envs)
        self.k = max(0, int(k))
        self.d_clip = float(d_clip)
        self.yaw_tab = yaw_hist_table(yaw_adaptive)
        self.pitch_tab = pitch_hist_table()
        self.hist = (np.zeros((self.n, self.k, ACT_FEAT), np.float32)
                     if self.k else None)
        # blocks: [(slice, field, threshold, cell)] - one per map slot, so a
        # multi-map fleet samples each env against ITS OWN map's field
        self.blocks = []
        if field is not None:
            pairs = (list(field) if isinstance(field, (list, tuple))
                     and field and isinstance(field[0], tuple)
                     else [(slice(0, self.n), field)])
            for sl, f in pairs:
                self.blocks.append((sl, f, _valid_threshold(f), _cell_of(f)))
        self.compass = bool(self.blocks)
        # d0 per env, NaN = "latch it at the next valid sample".  Per env
        # rather than the map's start geodesic because a reservoir respawn
        # starts an episode mid-map, and the ratio has to mean "how far
        # through THIS episode am I".
        self._d0 = np.full(self.n, np.nan, np.float64)
        self._prev_tick = None
        # scratch, so the per-decision path allocates nothing
        self._probe = np.zeros((self.n, 3), np.float64)
        self._out = np.zeros((self.n, self.n_features), np.float32)

    # ---- shape ---------------------------------------------------------
    @property
    def n_hist(self) -> int:
        return ACT_FEAT * self.k

    @property
    def n_compass(self) -> int:
        return CMP_FEAT if self.compass else 0

    @property
    def n_features(self) -> int:
        return self.n_hist + self.n_compass

    def describe(self) -> str:
        bits = []
        if self.k:
            bits.append(f"act-hist {self.k} decisions ({self.n_hist} cols, "
                        f"most recent first)")
        if self.compass:
            bits.append(f"compass ({CMP_FEAT} cols, "
                        f"{len(self.blocks)} field"
                        f"{'s' if len(self.blocks) > 1 else ''})")
        return "obs aux: " + (", ".join(bits) if bits else "off")

    # ---- --act-hist ----------------------------------------------------
    def encode(self, act) -> np.ndarray:
        """(M, 6) action indices -> (M, 6) float32 in [-1, 1]."""
        a = np.asarray(act)
        if a.ndim != 2 or a.shape[1] != ACT_FEAT:
            raise ValueError(f"actions must be (M, {ACT_FEAT}), got {a.shape}")
        out = np.empty((a.shape[0], ACT_FEAT), np.float32)
        out[:, 0] = self.yaw_tab[a[:, 0]]
        out[:, 1] = self.pitch_tab[a[:, 1]]
        out[:, 2] = a[:, 2].astype(np.float32) - 1.0
        out[:, 3] = a[:, 3].astype(np.float32) - 1.0
        out[:, 4] = a[:, 4].astype(np.float32)
        out[:, 5] = a[:, 5].astype(np.float32)
        return out

    def push(self, act) -> None:
        """Record the decision the engine is about to receive.  Most recent
        first, so slot 0 is always "what I just did"."""
        if self.hist is None:
            return
        if self.k > 1:
            # .copy() on purpose: an overlapping slice assignment into the
            # same array is not defined to be a shift
            self.hist[:, 1:] = self.hist[:, :-1].copy()
        self.hist[:, 0] = self.encode(act)

    # ---- episode boundaries -------------------------------------------
    def reset(self, mask) -> None:
        """Zero the history and re-arm the d0 latch for the marked envs.

        `mask` is a boolean (n,) array or an index array - whatever the
        caller already has for `ended`."""
        m = np.asarray(mask)
        if m.dtype == bool and not m.any():
            return
        if self.hist is not None:
            self.hist[m] = 0.0
        self._d0[m] = np.nan

    def note_ticks(self, tick) -> None:
        """Episode starts read off the core's per-env tick counter, which
        ``reset_env`` zeroes (``src/env.c``).

        This is the eval path's only episode signal: ``record_rollout`` never
        tells a policy an episode ended, which is the same reason
        ``_push_frame`` reads this counter for the frame ring.  Inference has
        to collapse its history at a spawn exactly like training does, or the
        recorded policy is not the trained one."""
        t = np.asarray(tick, np.int64)
        if self._prev_tick is not None and len(self._prev_tick) == len(t):
            started = t <= self._prev_tick
            if started.any():
                self.reset(started)
        self._prev_tick = t.copy()

    # ---- --obs-compass -------------------------------------------------
    def _sample(self, pos: np.ndarray, out: np.ndarray) -> None:
        """Every block's field sampled at FULL fleet width, then sliced.

        GoalDistField.sample is row-aligned to envs, so handing it a short
        array silently measures env i against env j's goal."""
        for sl, f, _thr, _h in self.blocks:
            out[sl] = np.asarray(f.sample(pos), np.float64)[sl]

    def compass_features(self, pos, yaw_deg, out=None,
                         latch: bool = True) -> np.ndarray:
        """(n, 3) world positions + (n,) yaw in DEGREES -> (n, 5) float32.

        `pos` must be the WHOLE fleet: see ``_sample``.  ``latch=False``
        reads the d0 anchors without writing them, which is what the
        truncation bootstrap needs - it evaluates a RECONSTRUCTED terminal
        pose for rows whose live state has already moved on, and must not
        re-anchor anyone."""
        p = np.ascontiguousarray(np.asarray(pos, np.float64).reshape(-1, 3))
        n = p.shape[0]
        if n != self.n:
            raise ValueError(f"compass wants all {self.n} envs, got {n}")
        res = np.zeros((n, CMP_FEAT), np.float32) if out is None else out
        res[:] = 0.0

        d = np.empty(n, np.float64)
        self._sample(p, d)
        thr = np.empty(n, np.float64)
        h = np.empty(n, np.float64)
        for sl, _f, t, c in self.blocks:
            thr[sl] = t
            h[sl] = c
        ok = d < thr
        if not ok.any():
            return res

        grad = np.zeros((n, 3), np.float64)
        probe = self._probe
        dp = np.empty(n, np.float64)
        dm = np.empty(n, np.float64)
        for ax in range(3):
            probe[:] = p
            probe[:, ax] += h
            self._sample(probe, dp)
            probe[:, ax] = p[:, ax] - h
            self._sample(probe, dm)
            axok = (dp < thr) & (dm < thr)
            # an axis that ran off the reachable set contributes nothing;
            # a sentinel differenced against a distance is not a gradient
            grad[:, ax] = np.where(axok, (dp - dm) / (2.0 * h), 0.0)

        nrm = np.linalg.norm(grad, axis=1)
        # downhill = -grad, made a unit vector; a flat or undefined field
        # has no direction and reads as the zero vector
        live = ok & (nrm > _EPS)
        u = np.zeros((n, 3), np.float64)
        u[live] = -grad[live] / nrm[live, None]

        yr = np.asarray(yaw_deg, np.float64).reshape(-1) * (np.pi / 180.0)
        cy, sy = np.cos(yr), np.sin(yr)
        # ego frame, sign-identical to route.RouteLine.features and
        # src/env.c: forward = (cos yaw, sin yaw), left = (-sin yaw, cos yaw)
        res[:, 0] = np.where(ok, u[:, 0] * cy + u[:, 1] * sy, 0.0)
        res[:, 1] = np.where(ok, -u[:, 0] * sy + u[:, 1] * cy, 0.0)
        res[:, 2] = np.where(ok, u[:, 2], 0.0)

        need = ok & ~np.isfinite(self._d0)
        if latch and need.any():
            self._d0[need] = np.maximum(d[need], 1.0)
        have = ok & np.isfinite(self._d0)
        res[:, 3] = np.where(have,
                             np.clip(d / np.where(have, self._d0, 1.0),
                                     0.0, self.d_clip), 0.0)
        res[:, 4] = np.where(ok, np.clip(nrm, 0.0, 1.0), 0.0)
        return res

    # ---- the whole block ----------------------------------------------
    def features(self, pos=None, yaw_deg=None, out=None,
                 latch: bool = True) -> np.ndarray:
        """(n, n_features) float32: [ 6*K history | 5 compass ].

        `out` may be a pinned host array of exactly that shape, which is what
        the trainer passes so the per-decision path stays allocation-free."""
        res = self._out if out is None else out
        if self.hist is not None:
            res[:, :self.n_hist] = self.hist.reshape(self.n, self.n_hist)
        if self.compass:
            self.compass_features(pos, yaw_deg, out=res[:, self.n_hist:],
                                  latch=latch)
        return res

    def eval_features(self, pos, yaw_deg, tick) -> np.ndarray:
        """``features`` with the eval path's episode-start detection folded
        in (see ``note_ticks``).  One call per DECISION."""
        self.note_ticks(tick)
        return self.features(pos, yaw_deg)


def _cell_of(field) -> float:
    """The finite-difference step: one cell of the field's own grid.

    GoalDistField composes a baked field, so its cell is that field's; a
    field with no grid at all (EuclidField, pure Euclidean goals) is smooth
    and 32u - the default bake cell for a source-port map - is as good a step
    as any."""
    c = getattr(field, "cell", None)
    if c is None:
        c = getattr(getattr(field, "geo", None), "cell", None)
    return float(c) if c else 32.0


def _valid_threshold(field) -> float:
    """Distances at or above this are the field's unreachable sentinel.

    Exactly ``GoalField.reachable``'s own test (``reach_max - cell/2``), so
    the compass and the reachability guards agree by construction.  A field
    with no unreachable set reports +inf and every sample is valid."""
    lim = float(getattr(field, "reach_max", np.inf))
    if not np.isfinite(lim):
        return np.inf
    return lim - 0.5 * float(getattr(field, "cell", 0.0))
