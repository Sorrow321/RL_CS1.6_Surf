"""``--respawn-front F``: spawn a share of the training episodes on the approach to the deepest
places REAL training episodes have reached - the seconds before each new near-record, kept even
when the episode dies right after.

Why. The reservoir keeps an episode's once-a-second snapshots only from ``--respawn-margin``
(2 s) before its end, so the moments before a death never become spawns. That is right for most
of the map and wrong at the frontier: an episode that reaches a new depth and dies within 2 s
leaves nothing behind, so training never starts THERE and the critic only ever sees "this far,
then death". On sidistic (round 6) stochastic training episodes climb out of the basement into
the finish room up to the funnel rim's level (the novel ceiling, d_best 1,415 u) while the
reservoir's deepest state stays in the basement below (d ~2,000): every room entry dies on the
room's kill floor within ~1 s. Go-Explore (Ecoffet et al. 2021) returns to the frontier cell and
explores from it whether or not the trajectory that found it survived.

What. Every ``every`` ticks the trainer hands the pool each env's pre-step state, its field
distance ``d`` and whether its episode is REAL (spawned at the map start, the reservoir or the
greedy line - never a novel row). The pool keeps a per-env ring of the last ``hi`` seconds. When a
real env sets a new best ``d`` for its episode that lies within ``band`` x d0 of the ceiling (the
deepest ``d`` any real episode has reached, the novel sampler's ``d_best`` when it runs), the
ring's states from ``hi`` to ``lo`` seconds before that tick join a FIFO pool (``cap`` states),
at most once per ``cooldown`` seconds per env. Each iteration ``frac`` of the pool's non-start
rows - right before the greedy-line tail, so the map-start rows, the novel rows and the greedy rows
are kept - are replaced by uniform draws from it, clocks zeroed, once it holds ``min_rows``.
Episodes from these rows are REAL (harvested, they move the ceiling): the states were reached by
the policy itself.

Generic. The policy's own states, one band in units of the map's own start distance, two
windows in seconds, no map quantity (CLAUDE.md sections 0 / 0b).
"""
from __future__ import annotations

import numpy as np

from .core import STATE_DTYPE


class FrontierRowPool:
    def __init__(self, n_envs: int, frac: float, band: float, d0: float, tick_ms: float,
                 every: int = 4, window=(0.0, 2.0), cooldown: float = 0.5, cap: int = 20000,
                 min_rows: int = 200, seed: int = 0, advance: float = 0.0, cross: bool = False,
                 finish: bool = False, speed=(1.0, 1.0)):
        frac = float(frac)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"--respawn-front F needs 0 < F < 1, got {frac!r}")
        if not float(band) > 0.0:
            raise ValueError(f"--respawn-front-band B needs B > 0, got {band!r}")
        lo, hi = (float(v) for v in window)
        if not 0.0 <= lo < hi:
            raise ValueError(f"--respawn-front-window LO HI needs 0 <= LO < HI, got {lo!r} {hi!r}")
        if not float(d0) > 0.0:
            raise ValueError("--respawn-front needs the map's start distance d0 > 0")
        if float(advance) < 0.0:
            raise ValueError(f"--respawn-front-advance A needs A >= 0, got {advance!r}")
        # --respawn-front-advance A: a capture also needs the episode to have ADVANCED >= A x d0
        # in field distance over the window (d HI s ago - d now): an arrival at the frontier, not
        # a crawl along it. Without it a policy that creeps at the frontier (round 6: agents
        # hanging at ~110 u/s on the finish room's east slope) re-captures its own crawl every
        # cooldown and floods the pool with one dead-end spot. 0 = off
        self.advance = float(advance)
        # --respawn-front-cross 1: a capture also needs the window to START outside the band (d HI s
        # ago > the band's edge): the arrival INTO the frontier, not what follows inside it. Once
        # real episodes finish, the band holds the easy part (round 6: every finishing episode gave
        # ~4 "fall down the finish shaft" windows, flooding the pool); crossings keep the climb
        self.cross = bool(cross)
        # --respawn-front-finish 1: a captured window is held per env and joins the pool only if
        # that episode then FINISHES (the last capture before the finish); otherwise it is dropped.
        # The rows are then approaches known to lead to the finish, not every lucky arrival
        # (round 6: replaying all crossings into the finish room, the policy - greedy or sampled -
        # entered the room 0 times in 24 tries from their starts)
        self.finish = bool(finish)
        self.pending = [None] * int(n_envs)
        # --respawn-front-speed LO HI: each mixed row's velocity x U(LO, HI), direction kept - the
        # --respawn-speed curriculum at the training frontier (round 6: the arrivals at the finish
        # room's east slope carry E ~1,450-1,620 where the funnel rim needs ~1,600-1,700+). Its own
        # generator, so the rows drawn are exactly the unscaled ones; (1, 1) = off, no draw
        s_lo, s_hi = (float(v) for v in speed)
        if not 0.0 < s_lo <= s_hi:
            raise ValueError(f"--respawn-front-speed LO HI needs 0 < LO <= HI, got {s_lo!r} {s_hi!r}")
        self.speed = (s_lo, s_hi)
        self.speed_rng = (np.random.default_rng([int(seed), 5303])
                          if (s_lo, s_hi) != (1.0, 1.0) else None)
        self.n = int(n_envs)
        self.frac, self.band, self.d0 = frac, float(band), float(d0)
        self.lo, self.hi = lo, hi
        self.every = max(1, int(every))
        per_s = 1000.0 / float(tick_ms) / self.every          # snapshots per second
        self.k_hi = max(1, int(round(hi * per_s)))             # ring slots back to HI
        self.k_lo = int(round(lo * per_s))                     # ... and to LO
        self.R = self.k_hi + 1
        self.cool = max(1, int(round(float(cooldown) * per_s)))
        self.cap, self.min_rows = int(cap), int(min_rows)
        self.rng = np.random.default_rng([int(seed), 4243])
        self.ring = np.zeros((self.n, self.R), STATE_DTYPE)
        self.ring_d = np.full((self.n, self.R), np.inf)
        self.fill = np.zeros(self.n, np.int64)                 # snapshots this episode
        self.ep_best = np.full(self.n, np.inf)
        self.last_cap = np.full(self.n, -10 ** 9, np.int64)
        self.ceiling = np.inf                                  # own min over real envs
        self.rows = np.zeros(0, STATE_DTYPE)
        self._head = 0
        self._tick = 0
        self.captured = 0          # captures, cumulative
        self.added = 0             # states added, cumulative
        self.mixed = 0

    @property
    def size(self) -> int:
        return int(len(self.rows))

    # -- per tick --------------------------------------------------------------
    def observe(self, states, d_fn, real, ended, ceiling=None, finished=None) -> int:
        """Once per physics tick, AFTER the step: ``states`` the (n,) post-step STATE_DTYPE view
        (an ended env's row is already its next episode's spawn), ``d_fn`` origins (n, 3) -> the
        field distance (non-finite = off the field; called on snapshot ticks only), ``real`` (n,)
        bool whether the env's CURRENT episode is real, ``ended`` (n,) bool the episodes that
        ended this tick, ``ceiling`` the deepest d a real episode has reached (None: the pool's
        own running min). -> states added."""
        ended = np.asarray(ended, bool)
        if ended.any():
            if self.finish:
                fin = (np.zeros(self.n, bool) if finished is None
                       else np.asarray(finished, bool))
                for i in np.flatnonzero(ended):
                    if fin[i] and self.pending[i] is not None:
                        self._push(self.pending[i])
                        self.captured += 1
                    self.pending[i] = None
            self.fill[ended] = 0
            self.ep_best[ended] = np.inf
        self._tick += 1
        if self._tick % self.every:
            return 0
        d = np.asarray(d_fn(np.asarray(states["origin"], np.float64)), np.float64)
        d = np.where(np.isfinite(d), d, np.inf)
        real = np.asarray(real, bool)
        slot = self.fill % self.R
        self.ring[np.arange(self.n), slot] = states
        self.ring_d[np.arange(self.n), slot] = d
        self.fill += 1
        if real.any():
            self.ceiling = min(self.ceiling, float(d[real].min()))
        ceil = float(ceiling) if ceiling is not None and np.isfinite(ceiling) else self.ceiling
        rec = real & (d < self.ep_best)
        self.ep_best = np.where(real, np.minimum(self.ep_best, d), self.ep_best)
        snap_no = self._tick // self.every
        # a full window only: the episode has flown >= HI s, so a row spawned inside the band
        # cannot re-capture its own spawn (a frontier row recaptures only a DEEPER record it
        # flew to)
        go = (rec & (d <= ceil + self.band * self.d0) & (self.fill >= self.R)
              & (snap_no - self.last_cap >= self.cool))
        if (self.advance > 0.0 or self.cross) and go.any():
            # the oldest snapshot of the window is the one HI s ago (the ring holds R = k_hi + 1)
            d_old = self.ring_d[np.arange(self.n), self.fill % self.R]
            if self.advance > 0.0:
                go &= (d_old - d) >= self.advance * self.d0
            if self.cross:
                go &= d_old > ceil + self.band * self.d0
        added = 0
        for i in np.flatnonzero(go):
            back = np.arange(self.k_hi, self.k_lo - 1, -1)     # oldest first
            idx = (int(self.fill[i]) - 1 - back) % self.R
            self.last_cap[i] = snap_no
            if self.finish:
                # held until the episode ends: committed on a finish, dropped otherwise
                self.pending[i] = self.ring[i, idx].copy()
                continue
            self._push(self.ring[i, idx])
            self.captured += 1
            added += len(idx)
        return added

    def _push(self, rows) -> None:
        rows = np.array(rows, dtype=STATE_DTYPE, copy=True)
        rows["tick"] = 0
        rows["stuck_ticks"] = 0
        self.added += len(rows)
        if self.size < self.cap:
            take = min(len(rows), self.cap - self.size)
            self.rows = np.concatenate([self.rows, rows[:take]])
            rows = rows[take:]
        for r in rows:                                         # FIFO once full
            self.rows[self._head] = r
            self._head = (self._head + 1) % self.cap

    # -- the pool ----------------------------------------------------------------
    def mix(self, pool: np.ndarray, n_fresh: int, n_tail: int = 0) -> np.ndarray:
        """Overwrite ``frac`` of the pool's non-start rows, the block right before its last
        ``n_tail`` rows (where --respawn-greedy writes), with uniform draws from the frontier
        states; a no-op until the pool holds ``min_rows``."""
        n = int(round((len(pool) - int(n_fresh)) * self.frac))
        n = min(n, len(pool) - int(n_fresh) - int(n_tail))
        if n <= 0 or self.size < self.min_rows:
            self.mixed = 0
            return pool
        out = np.array(pool, dtype=STATE_DTYPE, copy=True)
        end = len(out) - int(n_tail)
        out[end - n:end] = self.rows[self.rng.integers(0, self.size, n)]
        if self.speed_rng is not None:
            sc = self.speed_rng.uniform(self.speed[0], self.speed[1], n).astype(np.float32)
            out["velocity"][end - n:end] *= sc[:, None]
        self.mixed = n
        return out

    def line(self) -> str:
        return ("frontier rows: {:,} own states ({:,} captures; ceiling d {:,.0f} u, band {:g} x d0 "
                "= {:,.0f} u, {:g}..{:g} s before each record{}), {:.0%} of the non-start pool"
                .format(self.size, self.captured, self.ceiling, self.band, self.band * self.d0,
                        self.hi, self.lo,
                        (", advance >= {:,.0f} u".format(self.advance * self.d0)
                         if self.advance > 0.0 else "")
                        + (", crossings into the band only" if self.cross else "")
                        + (", finishing episodes only" if self.finish else "")
                        + (", speed x U({:g}, {:g})".format(*self.speed)
                           if self.speed_rng is not None else ""), self.frac))

    # -- persistence -------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"rows": self.rows.copy(), "head": int(self._head), "ceiling": float(self.ceiling),
                "captured": int(self.captured), "added": int(self.added)}

    def load_state_dict(self, st) -> bool:
        rows = st.get("rows") if isinstance(st, dict) else None
        if rows is None or getattr(rows, "dtype", None) != STATE_DTYPE:
            return False
        self.rows = np.array(rows[:self.cap], dtype=STATE_DTYPE, copy=True)
        self._head = int(st.get("head", 0)) % max(1, self.cap)
        self.ceiling = float(st.get("ceiling", np.inf))
        self.captured = int(st.get("captured", 0))
        self.added = int(st.get("added", 0))
        return True
