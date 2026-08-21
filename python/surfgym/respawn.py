"""respawn.py — reset-to-state exploring starts from the agent's own runs.

Start-only spawns make the agent replay the early map forever to touch the
frontier once; uniformly random spawns are worse (no momentum, off-track —
this map needs carried speed). The middle path (Go-Explore style): snapshot
full physics states along live episodes, and respawn most new episodes from
recent snapshots — same position, same velocity, same stance — perturbed
just enough (view, speed scale) to decorrelate.

The 10-seconds-before-death rule: a state 1s before a fatal mistake is
usually unrecoverable — the commitment was earlier. Harvest only snapshots
at least ``margin_ticks`` before the episode ended (any ending: death,
stall-kill, truncation, or finish — the last also avoids farming trivial
spawn-next-to-the-finish wins).

Position is never perturbed (a nudge can embed a state in a ramp/wall);
view yaw gets the env's own reset jitter, pitch and speed are jittered at
pool-refresh time, re-rolled every refresh.
"""
from __future__ import annotations

import numpy as np

from .core import STATE_DTYPE

__all__ = ["RespawnBuffer"]


class RespawnBuffer:
    """Per-env snapshot collector + FIFO state reservoir.

    Call :meth:`observe` once per physics tick with the post-step state view
    and the ended mask; call :meth:`build_pool` once per training iteration
    to get a fresh spawn pool for ``core.set_spawn_pool``.
    """

    def __init__(self, n_envs: int, reservoir: int = 100_000,
                 margin_ticks: int = 1000, snap_every: int = 100,
                 map_id: str = "", seed: int = 23,
                 dist_fn=None, dist_max: float | None = None,
                 dist_valid_max: float | None = None,
                 bins: int = 16) -> None:
        self.n = int(n_envs)
        self.map_id = str(map_id)     # reservoir states are map coordinates
        self.margin = int(margin_ticks)
        self.snap_every = int(snap_every)
        self.cap = int(reservoir)
        # progress-binned sampling (Go-Explore cell selection): dist_fn maps
        # origins (n,3) -> goal distance (n,). Uniform-over-states sampling
        # mirrors visitation density — the mastered early track dominates and
        # the frontier is starved; binning by distance flattens the
        # curriculum over the track instead.
        if dist_fn is not None and not dist_max:
            raise ValueError("dist_fn needs dist_max (the start distance d0)")
        self.dist_fn = dist_fn
        self.dist_max = float(dist_max) if dist_max else None
        # states whose distance is a field sentinel (unreachable under a
        # masked field) must not be sampleable via the binned path — they
        # would pool with genuine near-start states in the last bin and
        # earn zero shaping until they exit the masked region
        self.dist_valid_max = (float(dist_valid_max)
                               if dist_valid_max else None)
        self.bins = int(bins)
        self._d = np.zeros(self.cap, np.float32) if dist_fn is not None else None
        self._store = np.zeros(self.cap, dtype=STATE_DTYPE)
        self._size = 0
        self._head = 0
        self._tick = np.zeros(self.n, np.int64)      # episode tick per env
        self._last_snap = np.zeros(self.n, np.int64)
        # per-env pending snapshots: list of (tick, state row) — harvested or
        # discarded when the episode ends
        self._pend: list[list] = [[] for _ in range(self.n)]
        self.rng = np.random.default_rng(seed)
        self.harvested = 0                            # lifetime, for logging
        # harvest outbox (docs/ddp-plan.md §3b): observe() defers pushes
        # here; the trainer drains once per iteration and (under DDP)
        # all-gathers before pushing, so every rank's ring stays
        # byte-identical. (tick_in_iteration, env, state row) triples —
        # the sort key that reproduces single-GPU push order exactly.
        self._out: list = []
        self._iter_tick = 0

    # -- collection ---------------------------------------------------------
    def observe(self, states: np.ndarray, ended: np.ndarray,
                stagnant: np.ndarray | None = None) -> None:
        """states: post-step STATE_DTYPE view (ended rows are the NEW
        episode's spawn); ended: bool mask of episodes that ended this tick;
        stagnant: optional mask of envs currently making no progress — their
        states are never snapshotted (the 10s pre-END margin alone still
        admits mid-stall states, because a stall-KILL fires 15s after the
        stall began: end-relative margins cannot see the onset)."""
        self._tick += 1
        self._iter_tick += 1
        # harvest FIRST: pending snapshots belong to the episode that just
        # ended; the state rows of ended envs are already next-episode.
        # Deferred to the outbox, not pushed: build_pool runs at the TOP of
        # the iteration, so an end-of-iteration push_many sees exactly what
        # per-tick pushes would have (semantics-free, plan §3b), and one
        # code path serves both the single-GPU and the DDP trainer.
        if ended.any():
            for i in np.flatnonzero(ended):
                cutoff = self._tick[i] - self.margin
                self._out.extend((self._iter_tick, int(i), row)
                                 for t, row in self._pend[i] if t <= cutoff)
                self._pend[i].clear()
                self._tick[i] = 0
                self._last_snap[i] = 0
        snap = (~ended) & (self._tick - self._last_snap >= self.snap_every)
        if stagnant is not None:
            snap &= ~stagnant
        if snap.any():
            idx = np.flatnonzero(snap)
            rows = states[idx].copy()                 # detach from the view
            for j, i in enumerate(idx):
                self._pend[i].append((int(self._tick[i]), rows[j]))
            self._last_snap[idx] = self._tick[idx]

    def _dists(self, rows) -> np.ndarray | None:
        if self.dist_fn is None:
            return None
        org = np.stack([np.asarray(r["origin"], np.float32) for r in rows])
        return np.asarray(self.dist_fn(org), np.float32)

    def _push(self, row, d: float | None = None) -> None:
        self._store[self._head] = row
        if self._d is not None:
            self._d[self._head] = 0.0 if d is None else float(d)
        self._head = (self._head + 1) % self.cap
        self._size = min(self._size + 1, self.cap)
        self.harvested += 1

    def drain_harvest(self):
        """This iteration's deferred harvest: ``(rows, ticks, envs)`` with
        rows a (k,) STATE_DTYPE array, ticks/envs int32. Resets the outbox
        and the iteration tick counter."""
        k = len(self._out)
        rows = np.zeros(k, dtype=STATE_DTYPE)
        ticks = np.zeros(k, np.int32)
        envs = np.zeros(k, np.int32)
        for j, (t, i, row) in enumerate(self._out):
            rows[j] = row
            ticks[j] = t
            envs[j] = i
        self._out.clear()
        self._iter_tick = 0
        return rows, ticks, envs

    def flush_harvest(self) -> None:
        """Drain the outbox straight into the ring — the single-process
        path (the DDP trainer all-gathers between drain and push)."""
        rows, _, _ = self.drain_harvest()
        self.push_many(rows)

    def push_many(self, rows: np.ndarray) -> None:
        """Vectorised wrap-aware ring write, byte-identical to a loop of
        ``_push`` (tests pin this). Required under DDP: every rank pushes
        ALL fleet rows, and the per-row Python path would cost 10-20 ms."""
        rows = np.ascontiguousarray(rows)
        k = len(rows)
        if k == 0:
            return
        if k > self.cap:
            # only the last cap rows survive, laid out exactly where a loop
            # of _push would have left them: the write effectively starts
            # k-cap slots further around the ring
            self._head = (self._head + k - self.cap) % self.cap
            self._size = self.cap             # min() below keeps it capped
            self.harvested += k - self.cap    # _push would have counted them
            rows = rows[-self.cap:]
            k = self.cap
        ds = None
        if self._d is not None and self.dist_fn is not None:
            ds = np.asarray(self.dist_fn(
                rows["origin"].astype(np.float32)), np.float32)
        end = self._head + k
        if end <= self.cap:
            self._store[self._head:end] = rows
            if ds is not None:
                self._d[self._head:end] = ds
        else:
            n1 = self.cap - self._head
            self._store[self._head:] = rows[:n1]
            self._store[:end - self.cap] = rows[n1:]
            if ds is not None:
                self._d[self._head:] = ds[:n1]
                self._d[:end - self.cap] = ds[n1:]
        self._head = end % self.cap
        self._size = min(self._size + k, self.cap)
        self.harvested += k

    def _binned_pick(self, n: int) -> np.ndarray:
        """Draw n reservoir indices uniformly over occupied distance bins
        (then uniformly within a bin), capping each bin's draws at 4x its
        population so a 50-state frontier bin is not cloned into a quarter
        of the fleet — the degenerate self-reinforcing correlation the
        2000-state pool floor exists to prevent. Any residual demand the
        caps cannot absorb (tiny reservoir) tops up with plain uniform
        draws, which is the safe (visitation-shaped) distribution."""
        d = self._d[:self._size]
        valid = (np.flatnonzero(d < self.dist_valid_max)
                 if self.dist_valid_max is not None
                 else np.arange(self._size))
        if len(valid) == 0:
            return self.rng.integers(0, self._size, n)
        edges = np.linspace(0.0, self.dist_max, self.bins + 1)
        which = np.clip(np.digitize(d[valid], edges) - 1, 0, self.bins - 1)
        groups = [valid[g] for b in range(self.bins)
                  if len(g := np.flatnonzero(which == b))]
        caps = np.array([4 * len(g) for g in groups], np.int64)
        alloc = np.zeros(len(groups), np.int64)
        left = int(n)
        while left > 0:
            open_ = np.flatnonzero(alloc < caps)
            if len(open_) == 0:
                break
            share = np.zeros(len(groups), np.int64)
            base, rem = divmod(left, len(open_))
            share[open_] = base
            share[open_[:rem]] += 1
            share = np.minimum(share, caps - alloc)
            if share.sum() == 0:
                break
            alloc += share
            left -= int(share.sum())
        picks = [self.rng.choice(g, size=int(a), replace=True)
                 for g, a in zip(groups, alloc) if a > 0]
        out = (np.concatenate(picks) if picks
               else np.empty(0, np.int64))
        if left > 0:
            out = np.concatenate(
                [out, self.rng.integers(0, self._size, left)])
        self.rng.shuffle(out)
        return out

    # -- pool building ------------------------------------------------------
    def build_pool(self, start_pool: np.ndarray, pool_size: int = 4096,
                   fresh_frac: float = 0.10,
                   vel_scale: tuple[float, float] = (0.9, 1.1),
                   pitch_jitter: float = 5.0) -> np.ndarray:
        """Mix map-start entries with perturbed reservoir samples. The env
        resets by uniform pool draw, so entry counts ARE the probabilities.

        ``vel_scale`` is the spawn speed multiplier range. Above-1 ranges are
        a deliberate curriculum tool: speed-gated jumps can be practiced at
        make-it speed before the policy has learned to CARRY that speed —
        the value of the boosted states then pulls the upstream line faster."""
        n_fresh = max(1, int(round(pool_size * fresh_frac)))
        if self._size == 0:
            return start_pool
        n_re = pool_size - n_fresh
        idx = (self._binned_pick(n_re) if self._d is not None
               else self.rng.integers(0, self._size, n_re))
        re = self._store[idx].copy()
        # perturb: speed scale (never direction — that IS the run), view
        # pitch; yaw gets the env's own reset jitter on top
        scale = self.rng.uniform(vel_scale[0], vel_scale[1],
                                 n_re).astype(np.float32)
        re["velocity"] = re["velocity"] * scale[:, None]
        re["pitch"] = np.clip(re["pitch"] + self.rng.uniform(
            -pitch_jitter, pitch_jitter, n_re).astype(np.float32), -70.0, 30.0)
        fresh = start_pool[self.rng.integers(0, len(start_pool), n_fresh)]
        return np.concatenate([fresh, re])

    # -- persistence --------------------------------------------------------
    def state_dict(self, max_states: int = 20_000) -> dict:
        """Checkpoint payload (a recent subsample — same lesson as the
        novelty counts: cross-episode state must survive resumes)."""
        if self._size == 0:
            return {"states": None, "map_id": self.map_id}
        take = min(self._size, max_states)
        # the ring's newest `take` entries, oldest-first
        pos = (self._head - take) % self.cap
        idx = (pos + np.arange(take)) % self.cap
        return {"states": self._store[idx].copy(), "map_id": self.map_id}

    def load_state_dict(self, d) -> None:
        arr = d.get("states") if isinstance(d, dict) else None
        if arr is None or len(arr) == 0:
            return
        if d.get("map_id") != self.map_id:
            # states are raw map coordinates: a cross-map (or legacy,
            # unlabeled) payload would spawn 90% of episodes in solid/void
            print(f"respawn reservoir dropped: ckpt map "
                  f"{d.get('map_id')!r} != {self.map_id!r}")
            return
        arr = np.asarray(arr, dtype=STATE_DTYPE)[-self.cap:]
        # payloads predating the distance column (every F'/F2-era ckpt) get
        # their d recomputed here, or the whole restore lands in one bin
        ds = self._dists(list(arr))
        for k, row in enumerate(arr):
            self._push(row, None if ds is None else float(ds[k]))
        self.harvested -= len(arr)               # loading is not harvesting

    @property
    def size(self) -> int:
        return self._size
