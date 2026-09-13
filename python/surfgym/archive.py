"""PredecessorArchive - the survivor-gated predecessor archive.

Mechanism 2 of the 2026-09-13 cross-review, built for the unitfarmer2
exploration benchmark: when a policy-own episode takes a RARE transition
(an edge of the position-cell graph the fleet has crossed fewer than
``--int-rare`` times, reported by RaceReward.rare_entry) and then SURVIVES
``hold`` more ticks, the ``window`` of its own states BEFORE that transition
- the causal prefix, where the decision was made - is committed to a small
archive. Each iteration a fixed fraction of the spawn pool is replaced by
archive rows, so the fleet keeps practising the run-up to every discovery
that paid off, with no map boxes, no demo and no hand-cut window. An
episode that dies inside the hold discards its prefix: the archive holds
the run-ups to survivable discoveries only.

Cadence: ``observe`` once per physics tick after the reward (the ``states``
view is post-step; ended rows already hold the NEW episode's spawn and are
skipped), ``mix_pool`` once per iteration on the freshly built pool,
``pop_stats`` once per iteration. Rows are STATE_DTYPE, exactly what the
core's spawn pool takes. FIFO store; checkpointed through ``state_dict``.
"""
from __future__ import annotations

import numpy as np

from .core import STATE_DTYPE


class PredecessorArchive:
    def __init__(self, n_envs: int, window_ticks: int, hold_ticks: int,
                 capacity: int = 20_000, snap_every: int = 4, seed: int = 0):
        self.n = int(n_envs)
        self.W = max(1, int(window_ticks))
        self.H = max(1, int(hold_ticks))
        self.cap = max(1, int(capacity))
        self.every = max(1, int(snap_every))
        self.slots = max(1, self.W // self.every)
        self.ring = np.zeros((self.n, self.slots), dtype=STATE_DTYPE)
        self.ring_n = np.zeros(self.n, np.int64)      # valid snapshots per env
        self.ptr = np.zeros(self.n, np.int64)          # next write slot per env
        self.age = np.zeros(self.n, np.int64)          # ticks since the episode began
        self.rare_tick = np.full(self.n, -1, np.int64)  # age at the last rare entry
        self.pending: dict[int, np.ndarray] = {}        # env -> prefix rows awaiting the hold
        self.store = np.zeros(self.cap, dtype=STATE_DTYPE)
        self.size = 0
        self.head = 0
        self.rng = np.random.default_rng(seed)
        self.n_rare = 0
        self.n_commit = 0
        self.n_discard = 0
        self.rows_committed = 0
        self.total_commits = 0

    # ---------------------------------------------------------------- per tick
    def _prefix(self, i: int) -> np.ndarray:
        k = int(self.ring_n[i])
        if k == 0:
            return self.ring[i, :0].copy()
        if k < self.slots:
            return self.ring[i, :k].copy()
        p = int(self.ptr[i])
        return np.concatenate([self.ring[i, p:], self.ring[i, :p]]).copy()

    def _commit(self, rows: np.ndarray) -> None:
        for r in rows:
            self.store[self.head] = r
            self.head = (self.head + 1) % self.cap
            self.size = min(self.size + 1, self.cap)
        self.rows_committed += len(rows)
        self.total_commits += 1

    def observe(self, states: np.ndarray, rare: np.ndarray, ended: np.ndarray,
                died: np.ndarray) -> None:
        ended = np.asarray(ended, bool)
        died = np.asarray(died, bool)
        rare = np.asarray(rare, bool)
        live = ~ended
        # 1. prefixes whose hold has elapsed while the env is still alive
        if self.pending:
            for i in list(self.pending):
                if live[i] and self.age[i] - self.rare_tick[i] >= self.H:
                    self._commit(self.pending.pop(i))
                    self.n_commit += 1
        # 2. new rare entries on live envs: freeze the run-up (a newer rare
        #    entry replaces an older pending prefix that has not yet survived)
        ri = np.flatnonzero(rare & live)
        for i in ri:
            pref = self._prefix(int(i))
            if len(pref):
                self.pending[int(i)] = pref
                self.rare_tick[i] = self.age[i]
                self.n_rare += 1
        # 3. snapshot the live envs every `every` ticks
        snap = live & (self.age % self.every == 0)
        idx = np.flatnonzero(snap)
        if len(idx):
            self.ring[idx, self.ptr[idx]] = states[idx]
            self.ptr[idx] = (self.ptr[idx] + 1) % self.slots
            self.ring_n[idx] = np.minimum(self.ring_n[idx] + 1, self.slots)
        self.age[live] += 1
        # 4. episode ends: a death inside the hold discards the prefix, a
        #    finish / timeout after the rare entry is a survivor
        for i in np.flatnonzero(ended):
            i = int(i)
            if i in self.pending:
                if died[i]:
                    self.n_discard += 1
                    self.pending.pop(i)
                else:
                    self._commit(self.pending.pop(i))
                    self.n_commit += 1
            self.age[i] = 0
            self.rare_tick[i] = -1
            self.ring_n[i] = 0
            self.ptr[i] = 0

    # ------------------------------------------------------------ per iteration
    def mix_pool(self, pool: np.ndarray, frac: float) -> np.ndarray:
        """Replace a `frac` share of the pool's rows (uniformly chosen
        positions) by uniformly drawn archive rows; untouched while empty."""
        m = int(round(len(pool) * float(frac)))
        if self.size == 0 or m <= 0:
            return pool
        m = min(m, len(pool))
        pos = self.rng.choice(len(pool), m, replace=False)
        out = pool.copy()
        out[pos] = self.store[self.rng.integers(0, self.size, m)]
        return out

    def pop_stats(self) -> dict:
        d = dict(rows=int(self.size), commits=int(self.n_commit), rare=int(self.n_rare),
                 discards=int(self.n_discard), pending=int(len(self.pending)),
                 rows_committed=int(self.rows_committed))
        self.n_rare = self.n_commit = self.n_discard = self.rows_committed = 0
        return d

    # --------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        if self.size < self.cap:
            rows = self.store[:self.size].copy()
        else:
            rows = np.concatenate([self.store[self.head:], self.store[:self.head]])
        return {"rows": rows, "total_commits": int(self.total_commits),
                "W": self.W, "H": self.H, "every": self.every}

    def load_state_dict(self, d: dict) -> None:
        rows = np.asarray(d.get("rows"), dtype=STATE_DTYPE)
        rows = rows[-self.cap:]
        self.store[:len(rows)] = rows
        self.size = len(rows)
        self.head = len(rows) % self.cap
        self.total_commits = int(d.get("total_commits", 0))
