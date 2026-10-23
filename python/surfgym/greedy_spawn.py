"""``--respawn-greedy F``: spawn a share of the training episodes on the policy's OWN
greedy line, in the seconds before each place it dies.

Why. A run is judged on GREEDY evals from the true start, and the greedy policy (the
mean view, the argmax keys) is never executed in training: the behaviour policy samples
at the unstuck temperature, and the reservoir is harvested from those stochastic
episodes. So the states the greedy line reaches just before it dies can be states that
training seldom visits with that position AND velocity, and nothing corrects the action
there. On sidistic (round 2) the stochastic policy kept crossing the CP2 pit (training
ceiling 29.76%) while every greedy eval for ~1.5B steps dived into it (27.4%). DAgger's
point (Ross, Gordon, Bagnell 2011): train on the states the DEPLOYED policy visits.

What. Every greedy eval's episodes are captured through the eval's ``on_tick`` hook (the
eval core's pre-step STATE_DTYPE snapshot of env 0, exactly what ``record_ckpt.py
--dump-states`` writes). For each episode that ended in a FAIL, the states from ``hi``
down to ``lo`` seconds before its last tick are kept (default 5.0 .. 0.5 s; the last
half second is mostly past saving); and for a FAIL or a truncated episode whose best
field progress came more than ``hi`` s before its end (a line that got stuck - evals
never stall-kill - or lost ground long before dying), the ``hi`` seconds up to that
best tick. Each iteration ``frac`` of the pool's NON-start rows
- the reservoir TAIL, so the map-start rows at the head and the novel rows right after
them are untouched - are replaced by uniform draws from those states, clocks zeroed
(``tick``, ``stuck_ticks``) as ``--spawn-states`` does. The pool is rebuilt at every
eval, so it follows the greedy line wherever it now dies; an eval with no failed
episode empties it (nothing to correct).

Generic. The policy's own states from the true start (CLAUDE.md section 0 allows
"windows cut from its own greedy line"), two constants in seconds, no map quantity
(section 0b). Episodes spawned from these rows are REAL episodes (harvested into the
reservoir, they move the novel ceiling): they are states the policy reached on its own
from the map start.
"""
from __future__ import annotations

import numpy as np

from .core import STATE_DTYPE


def as_truncated(summaries, idx) -> list:
    """``summaries`` (record_rollout's per-episode trailers) with the episodes at ``idx``
    marked ``end = "trunc"``. --eval-patience ends a greedy eval episode that stopped
    gaining as a FAIL (core.force_fail); for the pool that episode is the STUCK line it
    would have been without the rule (a truncated episode: only its best-progress anchor
    counts), not a death with decisions to correct in the seconds before it."""
    idx = set(int(i) for i in idx)
    return [dict(s, end="trunc") if i in idx else s for i, s in enumerate(summaries)]


class GreedyLinePool:
    def __init__(self, frac: float, window=(0.5, 5.0), seed: int = 0, stuck: int = 0,
                 turn: float = 0.0, speed=None, keep: int = 0):
        frac = float(frac)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"--respawn-greedy F needs 0 < F < 1, got {frac!r}")
        lo, hi = (float(v) for v in window)
        if not 0.0 <= lo < hi:
            raise ValueError(f"--respawn-greedy-window LO HI needs 0 <= LO < HI, got {lo!r} {hi!r}")
        if int(stuck) < 0:
            raise ValueError(f"--respawn-greedy-stuck N needs N >= 0, got {stuck!r}")
        if int(keep) < 0:
            raise ValueError(f"--respawn-greedy-keep K needs K >= 0, got {keep!r}")
        # --respawn-greedy-keep K: the windows of the K DEEPEST failed greedy episodes seen so
        # far (ranked by the episode's best field progress) are kept ACROSS evals and drawn
        # beside the last eval's own windows. The deployed line reaches its frontier in a
        # few episodes of a few evals (sidistic round 6: 10 chamber entries in 2,376 greedy
        # episodes), and a pool rebuilt at every eval forgets them at the next one. 0 = off:
        # the pool is the last eval's windows only, as before
        self.keep = int(keep)
        self._arch: list = []          # [(best d, eval number, rows)], deepest first
        self._n_eval = 0
        self.n_arch = 0                # pool rows (at its END) from earlier evals' kept episodes
        self.last_arch = np.zeros(0, STATE_DTYPE)   # the kept-episode rows the last mix drew
        if not 0.0 <= float(turn) <= 180.0:
            raise ValueError(f"--respawn-greedy-turn T needs 0 <= T <= 180, got {turn!r}")
        if speed is not None:
            speed = tuple(float(v) for v in speed)
            if not (len(speed) == 2 and 0.0 < speed[0] <= speed[1]):
                raise ValueError(f"--respawn-greedy-speed LO HI needs 0 < LO <= HI, got {speed!r}")
        self.frac = frac
        self.lo, self.hi = lo, hi
        # --respawn-greedy-stuck N: also N evenly spaced states of each episode's STALLED
        # stretch (its best-progress tick to its end) - where the deployed line wanders once it
        # stops gaining. 0 = off: the pool is the two anchors' windows, as before
        self.stuck = int(stuck)
        # --respawn-greedy-turn T / --respawn-greedy-speed LO HI: each mixed row's horizontal
        # velocity and view yaw turned together by U(-T, T) deg and its velocity scaled by
        # U(LO, HI) - the policy's own frontier states, nudged in heading and energy. Their
        # own RNG, so the rows DRAWN are exactly the unperturbed ones; off = no draw at all
        self.turn = float(turn)
        self.speed = speed
        self._prng = np.random.default_rng(int(seed) + 7717)
        self.rng = np.random.default_rng(int(seed))
        self.rows = np.zeros(0, STATE_DTYPE)
        self.info = {"eps": 0, "fail": 0, "deaths_s": []}
        self.mixed = 0
        self._eps: list = []
        self._cur: list = []

    @property
    def size(self) -> int:
        return int(len(self.rows))

    # -- capture -------------------------------------------------------------
    def hook(self, n_envs: int = 0):
        """A fresh ``on_tick(t, states, rewards, done, trunc)`` for ONE eval recording:
        it keeps env 0's pre-step snapshot every tick and closes an episode on done or
        trunc (record_rollout's own rule), so episode i aligns with trailer i.

        ``n_envs`` > 0: the recording runs its episodes side by side
        (record_rollout_parallel) - env i's FIRST episode is episode i, captured until
        it ends; the env's later ticks (the core autoresets it in place) are ignored."""
        self._eps, self._cur = [], []
        if n_envs > 0:
            cur = [[] for _ in range(int(n_envs))]
            eps = [None] * int(n_envs)
            live = np.ones(int(n_envs), bool)
            self._par = (eps, cur)

            def on_tick_par(t, states, rewards, done, trunc):
                for i in np.flatnonzero(live):
                    cur[i].append(states[i].copy())
                    if bool(done[i]) or bool(trunc[i]):
                        eps[i] = np.array(cur[i], dtype=STATE_DTYPE)
                        cur[i] = []
                        live[i] = False
            return on_tick_par
        self._par = None

        def on_tick(t, states, rewards, done, trunc):
            self._cur.append(states[0].copy())
            if bool(done[0]) or bool(trunc[0]):
                self._eps.append(np.array(self._cur, dtype=STATE_DTYPE))
                self._cur = []
        return on_tick

    def build(self, summaries, tick_ms: float, d_fn=None, rank_fn=None) -> dict:
        """Rebuild the pool from the recording just captured. ``summaries`` is
        record_rollout's return (one trailer per episode, in order); ``tick_ms`` the eval
        core's realised mean tick. -> the build's stats.

        Two anchors per episode that did not finish:
        * a FAIL's last tick: the states HI..LO s before it (the decisions that led to
          the death; the last LO s are mostly past saving);
        * with ``d_fn`` (origins (n, 3) -> field distance, non-finite = off the field),
          the tick of the episode's BEST progress (its minimum d) when that lies more than
          HI s before the end: the states HI..0 s before it. That is where a line that
          got STUCK (a truncated episode: evals never stall-kill) or that lost ground
          long before dying stopped making progress.

        ``rank_fn`` (same signature as ``d_fn``; default ``d_fn``) ranks the episodes for
        --respawn-greedy-keep by their minimum distance; the windows themselves follow
        ``d_fn`` either way."""
        par = getattr(self, "_par", None)
        if par is not None:
            # side by side: episode i is env i's first episode (one still running when the
            # recording stopped is what it captured so far)
            eps = [e if e is not None else np.array(c, dtype=STATE_DTYPE)
                   for e, c in zip(*par)]
            self._par = None
        else:
            eps = list(self._eps)
            if self._cur:                      # an episode cut by the recording budget
                eps.append(np.array(self._cur, dtype=STATE_DTYPE))
        self._eps, self._cur = [], []
        per_s = 1000.0 / float(tick_ms)
        lo_t, hi_t = int(round(self.lo * per_s)), int(round(self.hi * per_s))
        keep, deaths, stops, n_used, best = [], [], [], 0, []
        for i, ep in enumerate(eps):
            end = summaries[i].get("end") if i < len(summaries) else None
            n = len(ep)
            if end not in ("fail", "trunc") or not n:
                continue
            idx = []
            d_best = np.inf
            if end == "fail" and n - lo_t > max(0, n - hi_t):
                idx.append(np.arange(max(0, n - hi_t), n - lo_t))
                deaths.append(round(n / per_s, 2))
            if d_fn is not None:
                d = np.asarray(d_fn(np.asarray(ep["origin"], np.float64)), np.float64)
                d = np.where(np.isfinite(d), d, np.inf)
                if np.isfinite(d).any():
                    tb = int(np.argmin(d))
                    d_best = float(d[tb])
                    if tb < n - hi_t and tb > 0:
                        idx.append(np.arange(max(0, tb - hi_t), tb + 1))
                        stops.append(round(tb / per_s, 2))
                        if self.stuck > 0:
                            # the stalled stretch after the stop, evenly spaced
                            idx.append(np.unique(np.linspace(
                                tb, n - 1, num=min(self.stuck, n - tb)).astype(np.int64)))
            if idx:
                keep.append(ep[np.unique(np.concatenate(idx))])
                if self.keep > 0 and rank_fn is not None:
                    dr = np.asarray(rank_fn(np.asarray(ep["origin"], np.float64)), np.float64)
                    dr = dr[np.isfinite(dr)]
                    d_best = float(dr.min()) if len(dr) else np.inf
                best.append(d_best)
                n_used += 1
        if keep:
            rows = np.concatenate(keep).astype(STATE_DTYPE, copy=True)
            rows["tick"] = 0
            rows["stuck_ticks"] = 0
        else:
            rows = np.zeros(0, STATE_DTYPE)
        self.info = {"eps": len(eps), "fail": n_used, "deaths_s": deaths, "stops_s": stops}
        if self.keep > 0:
            # --respawn-greedy-keep: this eval's episodes compete with the kept ones on their
            # best progress; the kept entries from EARLIER evals join the pool after this
            # eval's own rows (an entry from this eval is in them already)
            cand = []
            for r_, d_ in zip(keep, best):
                if np.isfinite(d_):
                    r_ = np.array(r_, dtype=STATE_DTYPE, copy=True)
                    r_["tick"] = 0
                    r_["stuck_ticks"] = 0
                    cand.append((float(d_), self._n_eval, r_))
            self._arch = sorted(self._arch + cand, key=lambda e: (e[0], -e[1]))[:self.keep]
            old = [r_ for d_, e_, r_ in self._arch if e_ != self._n_eval]
            self.n_arch = int(sum(len(r_) for r_ in old))
            if old:
                rows = np.concatenate([rows] + old).astype(STATE_DTYPE, copy=False)
            self.info["kept_d"] = [round(d_, 1) for d_, _, _ in self._arch]
            self.info["kept_rows"] = self.n_arch
        self._n_eval += 1
        self.rows = rows
        return dict(self.info, rows=self.size)

    # -- the pool ------------------------------------------------------------
    def mix(self, pool: np.ndarray, n_fresh: int) -> np.ndarray:
        """Overwrite ``frac`` of the pool's non-start rows (the TAIL of
        ``[fresh (n_fresh) | novel | reservoir]``; the env resets by uniform pool draw, so
        counts are the probabilities) with uniform draws from the greedy-line states."""
        n = int(round((len(pool) - int(n_fresh)) * self.frac))
        if n <= 0 or not self.size:
            self.mixed = 0
            self.last_arch = np.zeros(0, STATE_DTYPE)
            return pool
        out = np.array(pool, dtype=STATE_DTYPE, copy=True)
        pick = self.rng.integers(0, self.size, n)
        out[len(out) - n:] = self.perturb(self.rows[pick])
        self.mixed = n
        if self.keep > 0:
            # --respawn-greedy-keep: the drawn rows that came from earlier evals' kept episodes
            # (the pool's last n_arch rows), for the trainer's outcome counter
            self.last_arch = out[len(out) - n:][pick >= self.size - self.n_arch].copy()
        return out

    def perturb(self, rows: np.ndarray) -> np.ndarray:
        """``rows`` with --respawn-greedy-turn / --respawn-greedy-speed applied (a copy);
        the rows themselves when both are off."""
        if self.turn <= 0.0 and self.speed is None:
            return rows
        rows = np.array(rows, dtype=STATE_DTYPE, copy=True)
        n = len(rows)
        v = rows["velocity"].astype(np.float64)
        if self.turn > 0.0:
            a = self._prng.uniform(-self.turn, self.turn, n)
            c, s_ = np.cos(np.radians(a)), np.sin(np.radians(a))
            vx, vy = v[:, 0].copy(), v[:, 1].copy()
            v[:, 0] = c * vx - s_ * vy
            v[:, 1] = s_ * vx + c * vy
            rows["yaw"] = (rows["yaw"].astype(np.float64) + a).astype(np.float32)
        if self.speed is not None:
            v *= self._prng.uniform(self.speed[0], self.speed[1], n)[:, None]
        rows["velocity"] = v.astype(np.float32)
        return rows

    def line(self) -> str:
        d, b = self.info["deaths_s"], self.info.get("stops_s", [])
        kd = self.info.get("kept_d")
        return ("greedy-line pool: {:,} own states from {}/{} failed greedy episodes "
                "(deaths at {} s, {:g}..{:g} s before each; best-progress stops at {} s), "
                "{:.0%} of the non-start pool"
                .format(self.size, self.info["fail"], self.info["eps"],
                        ", ".join(f"{v:.1f}" for v in d) if d else "-", self.hi, self.lo,
                        ", ".join(f"{v:.1f}" for v in b) if b else "-", self.frac)
                + (f"; kept {len(kd)}/{self.keep} deepest (best d {kd[0]:,.0f}..{kd[-1]:,.0f} u), "
                   f"{self.info.get('kept_rows', 0):,} rows from earlier evals"
                   if self.keep > 0 and kd else ""))

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> dict:
        st = {"rows": self.rows.copy(), "info": dict(self.info)}
        if self.keep > 0:
            st["kept"] = [(float(d_), int(e_), r_.copy()) for d_, e_, r_ in self._arch]
            st["n_eval"] = int(self._n_eval)
            st["n_arch"] = int(self.n_arch)
        return st

    def load_state_dict(self, st) -> bool:
        rows = st.get("rows") if isinstance(st, dict) else None
        if rows is None or getattr(rows, "dtype", None) != STATE_DTYPE:
            return False
        self.rows = np.array(rows, dtype=STATE_DTYPE, copy=True)
        self.info = dict(st.get("info") or self.info)
        if self.keep > 0 and st.get("kept") is not None:
            self._arch = [(float(d_), int(e_), np.array(r_, dtype=STATE_DTYPE, copy=True))
                          for d_, e_, r_ in st["kept"]][:self.keep]
            self._n_eval = int(st.get("n_eval", 0))
            self.n_arch = min(int(st.get("n_arch", 0)), self.size)
        else:
            # a checkpoint saved without the kept episodes (or a K that is now off): the
            # restored rows are one eval's own windows
            self.n_arch = 0
        return True


class SpawnOutcomeCounter:
    """How the episodes that started from a given set of pool rows end: the rows' origins are
    remembered per pool build, the fresh spawns are tagged by origin (the novel sampler's and
    --respawn-front-perturb's rule), and each tagged episode that ends is counted, with its
    finishes and the deepest field distance any of them reached. Counters only."""

    def __init__(self, field) -> None:
        self.field = field
        self._lim = float(field.reach_max) - 0.5 * float(field.cell)
        self._keys = frozenset()
        self._keys_prev = frozenset()
        self.env = None
        self.spawned = 0
        self._ended = 0
        self._won = 0
        self._best_d = float("inf")

    def remember(self, rows) -> None:
        org = np.ascontiguousarray(np.asarray(rows["origin"]), np.float32).reshape(-1, 3)
        self._keys_prev = self._keys
        self._keys = frozenset(org[i].tobytes() for i in range(len(org)))

    def note_outcomes(self, ended, success) -> None:
        """Call BEFORE :meth:`note_spawns` of the same tick."""
        if self.env is None:
            return
        e = np.flatnonzero(np.asarray(ended, bool))
        b = e[self.env[e]] if len(e) else e
        if len(b):
            self._ended += int(len(b))
            self._won += int(np.asarray(success, bool)[b].sum())

    def note_spawns(self, states, ended) -> None:
        e = np.flatnonzero(np.asarray(ended, bool))
        n = len(np.asarray(ended))
        if self.env is None or len(self.env) != n:
            self.env = np.zeros(n, bool)
        if not len(e):
            return
        self.env[e] = False
        if not self._keys and not self._keys_prev:
            return
        org = np.ascontiguousarray(np.asarray(states["origin"])[e], np.float32)
        for j, i in enumerate(e):
            k = org[j].tobytes()
            self.env[i] = (k in self._keys) or (k in self._keys_prev)
        self.spawned += int(self.env[e].sum())

    def observe(self, pos) -> None:
        if self.env is None or not self.env.any():
            return
        p = np.asarray(pos, np.float64).reshape(-1, 3)[self.env]
        d = np.asarray(self.field.sample(p), np.float64)
        d = d[d < self._lim]
        if len(d):
            self._best_d = min(self._best_d, float(d.min()))

    def pop_outcomes(self) -> tuple:
        """(ended, finished, deepest d reached) since the last call."""
        out = (self._ended, self._won, self._best_d)
        self._ended = self._won = 0
        self._best_d = float("inf")
        return out
