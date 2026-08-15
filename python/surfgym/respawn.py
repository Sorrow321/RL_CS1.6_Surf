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
                 map_id: str = "", seed: int = 23) -> None:
        self.n = int(n_envs)
        self.map_id = str(map_id)     # reservoir states are map coordinates
        self.margin = int(margin_ticks)
        self.snap_every = int(snap_every)
        self.cap = int(reservoir)
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
        # harvest FIRST: pending snapshots belong to the episode that just
        # ended; the state rows of ended envs are already next-episode
        if ended.any():
            for i in np.flatnonzero(ended):
                cutoff = self._tick[i] - self.margin
                for t, row in self._pend[i]:
                    if t <= cutoff:
                        self._push(row)
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

    def _push(self, row) -> None:
        self._store[self._head] = row
        self._head = (self._head + 1) % self.cap
        self._size = min(self._size + 1, self.cap)
        self.harvested += 1

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
        idx = self.rng.integers(0, self._size, n_re)
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
        arr = np.asarray(arr, dtype=STATE_DTYPE)
        for row in arr[-self.cap:]:
            self._push(row)
        self.harvested -= len(arr[-self.cap:])   # loading is not harvesting

    @property
    def size(self) -> int:
        return self._size
