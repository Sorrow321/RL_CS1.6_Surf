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


class GreedyLinePool:
    def __init__(self, frac: float, window=(0.5, 5.0), seed: int = 0):
        frac = float(frac)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"--respawn-greedy F needs 0 < F < 1, got {frac!r}")
        lo, hi = (float(v) for v in window)
        if not 0.0 <= lo < hi:
            raise ValueError(f"--respawn-greedy-window LO HI needs 0 <= LO < HI, got {lo!r} {hi!r}")
        self.frac = frac
        self.lo, self.hi = lo, hi
        self.rng = np.random.default_rng(int(seed))
        self.rows = np.zeros(0, STATE_DTYPE)
        self.info = {"eps": 0, "fail": 0, "deaths_s": []}
        self.mixed = 0
        self._eps: list = []
        self._cur: list = []
        # --eval-parallel: per-env capture (None = the sequential hook's lists above)
        self._par = None

    @property
    def size(self) -> int:
        return int(len(self.rows))

    # -- capture -------------------------------------------------------------
    def hook(self):
        """A fresh ``on_tick(t, states, rewards, done, trunc)`` for ONE eval recording:
        it keeps env 0's pre-step snapshot every tick and closes an episode on done or
        trunc (record_rollout's own rule), so episode i aligns with trailer i."""
        self._eps, self._cur = [], []
        self._par = None

        def on_tick(t, states, rewards, done, trunc):
            self._cur.append(states[0].copy())
            if bool(done[0]) or bool(trunc[0]):
                self._eps.append(np.array(self._cur, dtype=STATE_DTYPE))
                self._cur = []
        return on_tick

    def hook_parallel(self, n: int):
        """The capture for ``record_rollout_parallel`` (``--eval-parallel``): env ``i``'s
        pre-step snapshots until ITS first episode ends (done or trunc), so episode ``i``
        of the capture aligns with trailer ``i`` of that recording, exactly as ``hook``
        aligns them for the one-env sequential recording. An episode still open when
        ``build`` runs (a recording budget cut it) is kept as it stands."""
        n = int(n)
        self._eps, self._cur = [], []
        cur = [[] for _ in range(n)]
        fin = [None] * n
        self._par = (cur, fin)

        def on_tick(t, states, rewards, done, trunc):
            end = np.asarray(done, bool)[:n] | np.asarray(trunc, bool)[:n]
            for i in range(n):
                if fin[i] is not None:
                    continue
                cur[i].append(states[i].copy())
                if end[i]:
                    fin[i] = np.array(cur[i], dtype=STATE_DTYPE)
                    cur[i] = []
        return on_tick

    def build(self, summaries, tick_ms: float, d_fn=None) -> dict:
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
          long before dying stopped making progress."""
        if self._par is not None:
            # --eval-parallel: one episode per env, in env order (= the trailers' order)
            cur, fin = self._par
            eps = [f if f is not None else np.array(c, dtype=STATE_DTYPE)
                   for c, f in zip(cur, fin)]
            self._par = None
        else:
            eps = list(self._eps)
            if self._cur:                      # an episode cut by the recording budget
                eps.append(np.array(self._cur, dtype=STATE_DTYPE))
        self._eps, self._cur = [], []
        per_s = 1000.0 / float(tick_ms)
        lo_t, hi_t = int(round(self.lo * per_s)), int(round(self.hi * per_s))
        keep, deaths, stops, n_used = [], [], [], 0
        for i, ep in enumerate(eps):
            end = summaries[i].get("end") if i < len(summaries) else None
            n = len(ep)
            if end not in ("fail", "trunc") or not n:
                continue
            idx = []
            if end == "fail" and n - lo_t > max(0, n - hi_t):
                idx.append(np.arange(max(0, n - hi_t), n - lo_t))
                deaths.append(round(n / per_s, 2))
            if d_fn is not None:
                d = np.asarray(d_fn(np.asarray(ep["origin"], np.float64)), np.float64)
                d = np.where(np.isfinite(d), d, np.inf)
                if np.isfinite(d).any():
                    tb = int(np.argmin(d))
                    if tb < n - hi_t and tb > 0:
                        idx.append(np.arange(max(0, tb - hi_t), tb + 1))
                        stops.append(round(tb / per_s, 2))
            if idx:
                keep.append(ep[np.unique(np.concatenate(idx))])
                n_used += 1
        if keep:
            rows = np.concatenate(keep).astype(STATE_DTYPE, copy=True)
            rows["tick"] = 0
            rows["stuck_ticks"] = 0
        else:
            rows = np.zeros(0, STATE_DTYPE)
        self.rows = rows
        self.info = {"eps": len(eps), "fail": n_used, "deaths_s": deaths, "stops_s": stops}
        return dict(self.info, rows=self.size)

    # -- the pool ------------------------------------------------------------
    def mix(self, pool: np.ndarray, n_fresh: int) -> np.ndarray:
        """Overwrite ``frac`` of the pool's non-start rows (the TAIL of
        ``[fresh (n_fresh) | novel | reservoir]``; the env resets by uniform pool draw, so
        counts are the probabilities) with uniform draws from the greedy-line states."""
        n = int(round((len(pool) - int(n_fresh)) * self.frac))
        if n <= 0 or not self.size:
            self.mixed = 0
            return pool
        out = np.array(pool, dtype=STATE_DTYPE, copy=True)
        out[len(out) - n:] = self.rows[self.rng.integers(0, self.size, n)]
        self.mixed = n
        return out

    def line(self) -> str:
        d, b = self.info["deaths_s"], self.info.get("stops_s", [])
        return ("greedy-line pool: {:,} own states from {}/{} failed greedy episodes "
                "(deaths at {} s, {:g}..{:g} s before each; best-progress stops at {} s), "
                "{:.0%} of the non-start pool"
                .format(self.size, self.info["fail"], self.info["eps"],
                        ", ".join(f"{v:.1f}" for v in d) if d else "-", self.hi, self.lo,
                        ", ".join(f"{v:.1f}" for v in b) if b else "-", self.frac))

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> dict:
        return {"rows": self.rows.copy(), "info": dict(self.info)}

    def load_state_dict(self, st) -> bool:
        rows = st.get("rows") if isinstance(st, dict) else None
        if rows is None or getattr(rows, "dtype", None) != STATE_DTYPE:
            return False
        self.rows = np.array(rows, dtype=STATE_DTYPE, copy=True)
        self.info = dict(st.get("info") or self.info)
        return True
