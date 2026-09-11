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

__all__ = ["RespawnBuffer", "DemoCurriculum", "RandomSpawnSampler"]


class DemoCurriculum:
    """Salimans & Chen (1812.03381) backward curriculum over a demo spine.

    ``states``: STATE_DTYPE array, TIME-ordered (index 0 = earliest, the
    last = just before the goal). The curriculum coordinate is demo time,
    as in the paper — necessary here because the goal field is blind along
    the winning route. tau starts at the demo's end; episodes start
    uniformly from the window {tau-D+1 .. tau}; when the finish rate from
    window starts reaches ``rate`` (paper: rho = 0.2), tau moves one state
    earlier; the reference implementation's forward backoff moves it back
    toward the end when the window stops succeeding. The C reset draws
    uniformly from the pool and never reports an index, so realized spawns
    are re-identified by matching origins against the demo rows."""

    def __init__(self, states, window: int = 10, rate: float = 0.2,
                 min_ep: float = 50.0, seed: int = 29,
                 grow: int = 0) -> None:
        self.S = np.asarray(states, STATE_DTYPE)
        self.n = len(self.S)
        self.D = int(window)
        self.rate = float(rate)
        self.min_ep = float(min_ep)
        # --demo-grow S (0 = off, byte-identical to the sliding paper
        # rule): the spawn range is ANCHORED at the goal end and WIDENS
        # backward - draws are uniform over [tau, n-1] rather than the
        # sliding [tau-D+1, tau], and tau retreats S states per advance
        # instead of 1. Two reasons this exists as an option:
        #   * the sliding window eventually stops sampling the end of the
        #     spine, so the policy can forget the part it already had; the
        #     growing one only ever adds, and at tau = 0 it degenerates to
        #     "uniform over the whole spine";
        #   * one state per advance with the hardcoded 20-iteration
        #     cooldown covers ~400 of 9,138 states in a 2-hour run, i.e.
        #     the curriculum could not reach the map start inside the
        #     budget at all.
        # The ADVANCE CRITERION stays local either way: the finish rate is
        # measured over the FRONTIER BAND [tau, tau+D-1] (the newest
        # states), never over the whole widened range, which would be
        # dominated by the easy goal-adjacent part and would advance on
        # its own success.
        self.grow = int(grow)
        self.tau = self.n - 1
        self.ep = np.zeros(self.n, np.float64)
        self.win = np.zeros(self.n, np.float64)
        self.decay = 0.99
        self._cool = 0
        self.rng = np.random.default_rng(seed)
        self._key = {tuple(np.round(np.asarray(r["origin"], np.float64), 1)): i
                     for i, r in enumerate(self.S)}
        self.last_info = ""

    def _band(self) -> tuple[int, int]:
        """Inclusive [lo, hi] of the frontier band scoring the advance."""
        if self.grow:
            return self.tau, min(self.n - 1, self.tau + self.D - 1)
        return max(0, self.tau - self.D + 1), self.tau

    def _draw(self) -> tuple[int, int]:
        """Inclusive [lo, hi] of the spine range episodes start from."""
        if self.grow:
            return self.tau, self.n - 1
        return max(0, self.tau - self.D + 1), self.tau

    def _lo(self) -> int:
        return self._band()[0]

    def build_pool(self, start_pool: np.ndarray, pool_size: int = 4096,
                   fresh_frac: float = 0.10) -> np.ndarray:
        n_fresh = max(1, int(round(pool_size * fresh_frac)))
        n_demo = pool_size - n_fresh
        self._move()
        d_lo, d_hi = self._draw()
        idx = self.rng.integers(d_lo, d_hi + 1, n_demo)
        fresh = start_pool[self.rng.integers(0, len(start_pool), n_fresh)]
        return np.concatenate([fresh, self.S[idx].copy()])

    def match(self, origins: np.ndarray) -> np.ndarray:
        """Demo index of each position, -1 if not a demo spawn."""
        out = np.full(len(origins), -1, np.int64)
        for j, o in enumerate(np.asarray(origins, np.float64)):
            out[j] = self._key.get(tuple(np.round(o, 1)), -1)
        return out

    def note_outcomes(self, idxs: np.ndarray, wins: np.ndarray) -> None:
        ok = idxs >= 0
        for i, w in zip(idxs[ok], np.asarray(wins)[ok]):
            self.ep[i] = self.ep[i] * self.decay + 1.0
            self.win[i] = self.win[i] * self.decay + float(w)

    def _move(self) -> None:
        lo, hi = self._band()
        ep = float(self.ep[lo:hi + 1].sum())
        win = float(self.win[lo:hi + 1].sum())
        r = win / max(ep, 1e-9)
        self._cool = max(0, self._cool - 1)
        step = self.grow if self.grow else 1
        moved = 0
        if ep >= self.min_ep and self._cool == 0:
            if r >= self.rate and self.tau > 0:
                self.tau = max(0, self.tau - step)
                moved = -1
            elif r < self.rate and self.tau < self.n - 1:
                self.tau = min(self.n - 1, self.tau + step)
                moved = +1
            if moved:
                self._cool = 20
        d_lo, d_hi = self._draw()
        self.last_info = (f"demo starts [{d_lo},{d_hi}]/{self.n} "
                          f"(frontier band [{lo},{hi}]) "
                          f"success {r:.1%} over {ep:.0f} eps")
        if moved:
            d_lo, d_hi = self._draw()
            print(f"demo curriculum: tau {'<- earlier' if moved < 0 else '-> later (backoff)'}"
                  f" starts [{d_lo},{d_hi}] (success {r:.1%} "
                  f"over {ep:.0f} eps)")

    def region_report(self, bins: int = 10) -> str:
        """Per-region episodes and finish rate over the whole spine.

        The curriculum's own record of WHERE episodes started and how
        often they reached the goal from there - the evidence a ledger
        needs to show the curriculum moving (and to separate a training
        win rate that is the harvest from one that is the policy)."""
        edges = np.linspace(0, self.n, bins + 1).astype(int)
        out = []
        for b in range(bins):
            i, j = edges[b], edges[b + 1]
            e = float(self.ep[i:j].sum())
            w = float(self.win[i:j].sum())
            out.append(f"{b * 100 // bins:d}%:{e:.0f}/"
                       f"{(w / e if e > 0 else 0.0):.0%}")
        return "demo regions (eps/finish, decile of spine) " + " ".join(out)


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
                 bins: int = 16, mode: str = "uniform",
                 goal_k: tuple[int, int] | None = None,
                 seg_max: int = 64, goal_min_dist: float = 0.0,
                 min_speed: float = 0.0) -> None:
        self.n = int(n_envs)
        # --respawn-min-speed: a snapshot slower than this (u/s, full 3-D
        # speed) is never taken. The deep bins of the goal runs held the
        # agent's own stalled arrivals (2026-09-02: 75% of a fresh band
        # below 200 u/s), and a surfer restarted without speed is stuck.
        # 0 = off, byte-identical to before.
        self.min_speed = float(min_speed)
        # --goals: every harvested snapshot also carries a GOAL - the
        # origin this very episode reached k ticks later (k drawn in
        # goal_k) - and the SEGMENT of snapshots between them. A goal
        # drawn from the agent's own future is reachable by construction
        # and its difficulty is k (research-plan-goalcond.md, variable
        # 2: HER's "future" strategy applied to the goal DISTRIBUTION,
        # which is what composes with on-policy PPO). None = off, and off
        # allocates nothing and changes no byte of the control path.
        self.goal_k = (None if goal_k is None
                       else (int(goal_k[0]), int(goal_k[1])))
        self.seg_max = int(seg_max)
        # a goal inside the start's own sphere is not a goal (a policy
        # that barely moves would otherwise harvest instant successes -
        # measured: 100% "reached" at 0.0 s on the first smoke). Later
        # snapshots closer than this are skipped; none left -> NaN.
        self.goal_min_dist = float(goal_min_dist)
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
        # start-state selection mode over the distance bins (all verbatim
        # implementations from docs/research-litsurvey.md section 6):
        #   uniform  — equal share per occupied bin (the original binned mode)
        #   goex     — Go-Explore (1901.10995): bin weight 1/sqrt(chosen+1)
        #   florensa — Reverse Curriculum (1707.05300): sample bins whose
        #              estimated success rate lies in (0.1, 0.9), with a
        #              reserved share of draws from mastered bins
        #   backward — Salimans & Chen (1812.03381): a moving window of bins
        #              nearest the goal; advances backward on a success rule
        self.mode = str(mode)
        if self.mode != "uniform" and dist_fn is None:
            raise ValueError(f"mode {self.mode!r} needs dist_fn/dist_max")
        self.bin_chosen = np.zeros(self.bins, np.int64)   # realized spawns
        # Go-Explore C_seen: number of EPISODES that visited the bin (Nature
        # 2004.12919: +1 per exploration run that touches the cell, however
        # many times it does). _ep_bins marks the bins the current episode
        # has touched; folded into bin_seen at episode end.
        self.bin_seen = np.zeros(self.bins, np.int64)
        self._ep_bins = np.zeros((self.n, self.bins), bool)
        self.bin_ep = np.zeros(self.bins, np.float64)     # decayed episodes
        self.bin_win = np.zeros(self.bins, np.float64)    # decayed finishes
        self.stat_decay = 0.99          # per-episode-in-bin decay (~100 ep)
        # florensa (1707.05300 App A.1): band R_min/R_max = 0.1/0.9;
        # N_old/(N_new+N_old) = 100/300 of each iteration's starts replay
        # previous good starts (anti-forgetting)
        self.fl_rmin, self.fl_rmax = 0.1, 0.9
        self.fl_reserve = 1.0 / 3.0
        self.fl_min_ep = 5.0            # bins with fewer count as unevaluated
        self.ever_band = np.zeros(self.bins, bool)   # the starts_old analog
        # backward curriculum (1812.03381): rho = 0.2; window width and the
        # retreat step come from the reference code / Go-Explore's reuse
        # (the paper gives no D or Delta)
        self.bw_hi: int | None = None   # window's closest-to-goal bin
        self.bw_init: int | None = None
        self.bw_width = 2               # bins per window
        self.bw_rate = 0.2              # advance when window success >= this
        self.bw_min_ep = 50.0           # ...over at least this many episodes
        self._bw_cool = 0               # rebuilds until the next move check
        self.last_info = ""             # one-line mode diagnostic for logs
        self._d = np.zeros(self.cap, np.float32) if dist_fn is not None else None
        self._store = np.zeros(self.cap, dtype=STATE_DTYPE)
        if self.goal_k is not None:
            self._goal = np.full((self.cap, 3), np.nan, np.float32)
            self._seg = np.zeros((self.cap, self.seg_max, 3), np.float32)
            self._seglen = np.zeros(self.cap, np.int32)
        else:
            self._goal = self._seg = self._seglen = None
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
                stagnant: np.ndarray | None = None,
                success: np.ndarray | None = None) -> None:
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
            ei = np.flatnonzero(ended)
            if self.mode == "goex":
                # fold the ended episodes' visited-bin marks into C_seen
                self.bin_seen += self._ep_bins[ei].sum(0)
                self._ep_bins[ei] = False
            for i in ei:
                # the pre-end margin exists to keep the moments before a
                # DEATH out of the reservoir; a goal-reached ending is not
                # a death, so a successful episode harvests its whole
                # chain - otherwise 2 s goal runs never feed the
                # reservoir and the goal frontier can never move
                # (measured on xsG2: 93% success at 2.1 s, mind 99.1%,
                # k_max pinned)
                cutoff = self._tick[i] - self.margin
                if success is not None and bool(success[i]):
                    cutoff = self._tick[i]
                if self.goal_k is None:
                    self._out.extend((self._iter_tick, int(i), row)
                                     for t, row in self._pend[i]
                                     if t <= cutoff)
                else:
                    self._harvest_with_goals(int(i), cutoff)
                self._pend[i].clear()
                self._tick[i] = 0
                self._last_snap[i] = 0
        snap = (~ended) & (self._tick - self._last_snap >= self.snap_every)
        if stagnant is not None:
            snap &= ~stagnant
        if self.min_speed > 0.0:
            v = np.asarray(states["velocity"], np.float64)
            snap &= np.sqrt((v * v).sum(1)) >= self.min_speed
        if snap.any():
            idx = np.flatnonzero(snap)
            rows = states[idx].copy()                 # detach from the view
            for j, i in enumerate(idx):
                self._pend[i].append((int(self._tick[i]), rows[j]))
            self._last_snap[idx] = self._tick[idx]
            if self.mode == "goex":
                bs = self.bin_of(rows["origin"])
                ok = bs >= 0
                if ok.any():
                    self._ep_bins[idx[ok], bs[ok]] = True

    def _harvest_with_goals(self, i: int, cutoff: int) -> None:
        """Push env i's pending snapshots up to ``cutoff`` with a goal
        each: the snapshot nearest t + k (k ~ U[goal_k]) among the LATER
        snapshots of the same episode - including the ones inside the
        margin, which are never start states but are perfectly good
        goals (the agent was there). The segment is the snapshot origins
        from the start to the goal, subsampled to seg_max. A start with
        no later snapshot gets NaN: the spawn-time assigner gives it a
        random-air goal instead."""
        pend = self._pend[i]
        if not pend:
            return
        ticks = np.asarray([t for t, _ in pend], np.int64)
        origins = np.stack([np.asarray(r["origin"], np.float32)
                            for _, r in pend])
        kmin, kmax = self.goal_k
        for a, (t, row) in enumerate(pend):
            if t > cutoff:
                break
            later = np.flatnonzero(ticks > t)
            if self.goal_min_dist > 0.0 and len(later):
                far = (np.linalg.norm(origins[later] - origins[a], axis=1)
                       >= self.goal_min_dist)
                later = later[far]
            if len(later) == 0:
                self._out.append((self._iter_tick, i, row, None, None))
                continue
            k = int(self.rng.integers(kmin, kmax + 1))
            j = int(later[np.argmin(np.abs(ticks[later] - (t + k)))])
            seg = origins[a:j + 1]
            if len(seg) > self.seg_max:
                pick = np.linspace(0, len(seg) - 1, self.seg_max).round()
                seg = seg[pick.astype(np.int64)]
            self._out.append((self._iter_tick, i, row,
                              origins[j].copy(), seg.copy()))

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
        self._last_goals = None
        if self.goal_k is not None:
            goals = np.full((k, 3), np.nan, np.float32)
            segs = np.zeros((k, self.seg_max, 3), np.float32)
            seglen = np.zeros(k, np.int32)
        for j, item in enumerate(self._out):
            t, i, row = item[0], item[1], item[2]
            rows[j] = row
            ticks[j] = t
            envs[j] = i
            if self.goal_k is not None and item[3] is not None:
                goals[j] = item[3]
                sg = item[4]
                segs[j, :len(sg)] = sg
                seglen[j] = len(sg)
        if self.goal_k is not None:
            self._last_goals = (goals, segs, seglen)
        self._out.clear()
        self._iter_tick = 0
        return rows, ticks, envs

    def flush_harvest(self) -> None:
        """Drain the outbox straight into the ring — the single-process
        path (the DDP trainer all-gathers between drain and push)."""
        rows, _, _ = self.drain_harvest()
        if self._last_goals is not None:
            g, sg, sl = self._last_goals
            self.push_many(rows, goals=g, segs=sg, seglen=sl)
        else:
            self.push_many(rows)

    def push_many(self, rows: np.ndarray, goals=None, segs=None,
                  seglen=None) -> None:
        """Vectorised wrap-aware ring write, byte-identical to a loop of
        ``_push`` (tests pin this). Required under DDP: every rank pushes
        ALL fleet rows, and the per-row Python path would cost 10-20 ms.
        ``goals``/``segs``/``seglen`` are the optional parallel goal
        columns (--goals); rows pushed without them get NaN goals."""
        rows = np.ascontiguousarray(rows)
        k = len(rows)
        if k == 0:
            return
        if self._goal is not None:
            if goals is None:
                goals = np.full((k, 3), np.nan, np.float32)
                segs = np.zeros((k, self.seg_max, 3), np.float32)
                seglen = np.zeros(k, np.int32)
            goals = np.asarray(goals, np.float32)
            segs = np.asarray(segs, np.float32)
            seglen = np.asarray(seglen, np.int32)
        if k > self.cap:
            # only the last cap rows survive, laid out exactly where a loop
            # of _push would have left them: the write effectively starts
            # k-cap slots further around the ring
            self._head = (self._head + k - self.cap) % self.cap
            self._size = self.cap             # min() below keeps it capped
            self.harvested += k - self.cap    # _push would have counted them
            rows = rows[-self.cap:]
            if self._goal is not None:
                goals, segs, seglen = (goals[-self.cap:], segs[-self.cap:],
                                       seglen[-self.cap:])
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
            if self._goal is not None:
                self._goal[self._head:end] = goals
                self._seg[self._head:end] = segs
                self._seglen[self._head:end] = seglen
        else:
            n1 = self.cap - self._head
            self._store[self._head:] = rows[:n1]
            self._store[:end - self.cap] = rows[n1:]
            if ds is not None:
                self._d[self._head:] = ds[:n1]
                self._d[:end - self.cap] = ds[n1:]
            if self._goal is not None:
                self._goal[self._head:] = goals[:n1]
                self._goal[:end - self.cap] = goals[n1:]
                self._seg[self._head:] = segs[:n1]
                self._seg[:end - self.cap] = segs[n1:]
                self._seglen[self._head:] = seglen[:n1]
                self._seglen[:end - self.cap] = seglen[n1:]
        self._head = end % self.cap
        self._size = min(self._size + k, self.cap)
        self.harvested += k

    def _binned_pick(self, n: int) -> np.ndarray:
        """Draw n reservoir indices over occupied distance bins with
        mode-dependent per-bin weights (then uniformly within a bin),
        capping each bin's draws at 4x its population so a 50-state
        frontier bin is not cloned into a quarter of the fleet — the
        degenerate self-reinforcing correlation the 2000-state pool floor
        exists to prevent. Any residual demand the caps cannot absorb
        (tiny reservoir, or zero-weight bins) tops up with plain uniform
        draws, which is the safe (visitation-shaped) distribution."""
        d = self._d[:self._size]
        valid = (np.flatnonzero(d < self.dist_valid_max)
                 if self.dist_valid_max is not None
                 else np.arange(self._size))
        if len(valid) == 0:
            return self.rng.integers(0, self._size, n)
        edges = np.linspace(0.0, self.dist_max, self.bins + 1)
        which = np.clip(np.digitize(d[valid], edges) - 1, 0, self.bins - 1)
        bin_ids, groups = [], []
        for b in range(self.bins):
            g = np.flatnonzero(which == b)
            if len(g):
                bin_ids.append(b)
                groups.append(valid[g])
        weights = self._bin_weights(np.array(bin_ids, np.int64))
        # the 4x anti-cloning cap guards the uniform mode; the paper modes
        # (Go-Explore / Florensa / Salimans-Chen) all deliberately restart
        # whole fleets from rare states, so the cap would defang exactly the
        # bins they exist to oversample. The 2000-state pool floor still
        # applies upstream.
        if self.mode == "uniform":
            caps = np.array([4 * len(g) for g in groups], np.int64)
        else:
            caps = np.full(len(groups), np.iinfo(np.int64).max, np.int64)
        alloc = np.zeros(len(groups), np.int64)
        left = int(n)
        while left > 0:
            open_ = np.flatnonzero((alloc < caps) & (weights > 0))
            if len(open_) == 0:
                break
            w = weights[open_] / weights[open_].sum()
            share = np.zeros(len(groups), np.int64)
            share[open_] = np.floor(w * left).astype(np.int64)
            rem = left - int(share[open_].sum())
            if rem > 0:
                top = open_[np.argsort(-w, kind="stable")[:rem]]
                share[top] += 1
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

    def _bin_weights(self, bin_ids: np.ndarray) -> np.ndarray:
        """Relative draw weight per occupied bin (allocation normalizes,
        so only ratios matter). bin 0 = nearest the finish."""
        k = len(bin_ids)
        if self.mode == "goex":
            # Go-Explore cell selection (Nature 2004.12919, Ext Data Table
            # 1a): W = 1/sqrt(C_seen + 1), C_seen = episodes that visited
            # the cell; the +1 keeps never-seen bins finite
            w = 1.0 / np.sqrt(self.bin_seen[bin_ids] + 1.0)
            top = bin_ids[np.argsort(-w)[:3]]
            self.last_info = (f"goex: seen {self.bin_seen[bin_ids].sum():,}"
                              f" top-W bins {list(top)}")
            return w
        if self.mode == "florensa":
            ep = self.bin_ep[bin_ids]
            sr = np.divide(self.bin_win[bin_ids], ep,
                           out=np.zeros(k), where=ep > 0)
            evaluated = ep >= self.fl_min_ep
            # unevaluated bins count as candidate "good starts" until the
            # training rollouts say otherwise (the paper reads success off
            # the training batch, never dedicated eval rollouts)
            band = ~evaluated | ((sr > self.fl_rmin) & (sr < self.fl_rmax))
            self.ever_band[bin_ids[evaluated & (sr > self.fl_rmin)
                                   & (sr < self.fl_rmax)]] = True
            old = self.ever_band[bin_ids]     # the starts_old replay analog
            w = np.zeros(k)
            if band.any():
                w[band] = (1.0 - self.fl_reserve) / band.sum()
            if old.any():
                w[old] += ((self.fl_reserve if band.any() else 1.0)
                           / old.sum())
            if w.sum() <= 0:      # everything evaluated too hard: fall back
                w = np.ones(k)    # to uniform-over-occupied
            self.last_info = (f"florensa: band bins "
                              f"{list(bin_ids[band & evaluated])} "
                              f"unevaluated {int((~evaluated).sum())} "
                              f"old {list(bin_ids[old])}")
            return w
        if self.mode == "backward":
            if self.bw_hi is None:
                self.bw_hi = self.bw_init = int(bin_ids.min())
            self._bw_move()
            w = ((bin_ids >= self.bw_hi)
                 & (bin_ids < self.bw_hi + self.bw_width)).astype(np.float64)
            if w.sum() <= 0:      # nothing harvested inside the window yet
                w = np.ones(k)
            return w
        return np.ones(k)

    def _bw_move(self) -> None:
        """Salimans-Chen rule (rho = 0.2): once episodes started inside the
        current window finish at >= bw_rate, slide the window one bin away
        from the goal (tau* moves backward along the run). The reference
        implementation also RETREATS toward the goal when success falls
        below threshold everywhere, so an overshot curriculum recovers."""
        a = int(self.bw_hi)
        sel = slice(a, min(self.bins, a + self.bw_width))
        ep = float(self.bin_ep[sel].sum())
        win = float(self.bin_win[sel].sum())
        rate = win / max(ep, 1e-9)
        moved = 0
        self._bw_cool = max(0, self._bw_cool - 1)
        if ep >= self.bw_min_ep and self._bw_cool == 0:
            if rate >= self.bw_rate and a < self.bins - 1:
                self.bw_hi = a + 1
                moved = +1
            elif rate < self.bw_rate and a > self.bw_init:
                self.bw_hi = a - 1
                moved = -1
            if moved:
                # let the moved window collect fresh evidence before the
                # next move (the reference damps via cumulative counts)
                self._bw_cool = 20
        self.last_info = (f"backward: window bins [{self.bw_hi}, "
                          f"{self.bw_hi + self.bw_width - 1}] success "
                          f"{rate:.1%} over {ep:.0f} eps")
        if moved:
            print(f"backward curriculum: window {'-> deeper' if moved > 0 else '<- retreat'}"
                  f" bins [{self.bw_hi}, {self.bw_hi + self.bw_width - 1}]"
                  f" (success {rate:.1%} over {ep:.0f} eps)")

    # -- outcome bookkeeping (goex / florensa / backward modes) -------------
    def bin_of(self, origins: np.ndarray) -> np.ndarray:
        """Distance-bin index of map positions; -1 where the field reads
        invalid/unreachable (those never enter the stats)."""
        d = np.asarray(self.dist_fn(np.asarray(origins, np.float32)
                                    .reshape(-1, 3)), np.float32)
        edges = np.linspace(0.0, self.dist_max, self.bins + 1)
        b = np.clip(np.digitize(d, edges) - 1, 0, self.bins - 1).astype(np.int64)
        if self.dist_valid_max is not None:
            b[d >= self.dist_valid_max] = -1
        return b

    def note_spawns(self, bins: np.ndarray, envs: np.ndarray | None = None) -> None:
        """Count realized episode starts per bin (times-chosen, logged for
        diagnostics), and mark the spawn bin as visited by the new episode
        so short episodes still register in C_seen."""
        ok = bins >= 0
        if ok.any():
            np.add.at(self.bin_chosen, bins[ok], 1)
            if self.mode == "goex" and envs is not None:
                self._ep_bins[np.asarray(envs)[ok], bins[ok]] = True

    def note_outcomes(self, bins: np.ndarray, wins: np.ndarray) -> None:
        """Attribute finished episodes to their start bin. Decayed counts:
        each bin's stats are an EMA over roughly the last
        1/(1-stat_decay) episodes started there."""
        ok = bins >= 0
        for b, w in zip(bins[ok], np.asarray(wins)[ok]):
            self.bin_ep[b] = self.bin_ep[b] * self.stat_decay + 1.0
            self.bin_win[b] = self.bin_win[b] * self.stat_decay + float(w)

    # -- pool building ------------------------------------------------------
    def build_pool(self, start_pool: np.ndarray, pool_size: int = 4096,
                   fresh_frac: float = 0.10,
                   vel_scale: tuple[float, float] = (0.9, 1.1),
                   pitch_jitter: float = 5.0, with_goals: bool = False):
        """Mix map-start entries with perturbed reservoir samples. The env
        resets by uniform pool draw, so entry counts ARE the probabilities.

        ``vel_scale`` is the spawn speed multiplier range. Above-1 ranges are
        a deliberate curriculum tool: speed-gated jumps can be practiced at
        make-it speed before the policy has learned to CARRY that speed —
        the value of the boosted states then pulls the upstream line faster."""
        n_fresh = max(1, int(round(pool_size * fresh_frac)))
        if self._size == 0:
            if with_goals:
                n0 = len(start_pool)
                return (start_pool, np.full((n0, 3), np.nan, np.float32),
                        np.zeros((n0, self.seg_max, 3), np.float32),
                        np.zeros(n0, np.int32))
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
        pool = np.concatenate([fresh, re])
        if not with_goals:
            return pool
        # goal columns parallel to the pool rows: fresh starts carry NaN
        # (the assigner draws a random-air goal), reservoir rows carry
        # the goal harvested with them
        if self._goal is None:
            raise ValueError("build_pool(with_goals=True) needs goal_k")
        goals = np.full((len(pool), 3), np.nan, np.float32)
        segs = np.zeros((len(pool), self.seg_max, 3), np.float32)
        seglen = np.zeros(len(pool), np.int32)
        goals[n_fresh:] = self._goal[idx]
        segs[n_fresh:] = self._seg[idx]
        seglen[n_fresh:] = self._seglen[idx]
        return pool, goals, segs, seglen

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
        d = {"states": self._store[idx].copy(), "map_id": self.map_id}
        if self._goal is not None:
            d["goals"] = self._goal[idx].copy()
            d["segs"] = self._seg[idx].copy()
            d["seglen"] = self._seglen[idx].copy()
        return d

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
        if self._goal is not None and d.get("goals") is not None:
            g = np.asarray(d["goals"], np.float32)[-self.cap:]
            sg = np.asarray(d["segs"], np.float32)[-self.cap:]
            sl = np.asarray(d["seglen"], np.int32)[-self.cap:]
            if len(g) == len(arr) and sg.shape[1] == self.seg_max:
                self.push_many(arr, goals=g, segs=sg, seglen=sl)
                self.harvested -= len(arr)
                return
        ds = self._dists(list(arr))
        for k, row in enumerate(arr):
            self._push(row, None if ds is None else float(ds[k]))
        self.harvested -= len(arr)               # loading is not harvesting

    @property
    def size(self) -> int:
        return self._size


class RandomSpawnSampler:
    """``--respawn-random``: uniform reachable-state exploring starts.

    A spawn SOURCE, not a curriculum: it REPLACES the reservoir. Every
    episode starts either at the map's own start spawn (``start_frac``,
    5% by default - exactly what the evals use) or at a state drawn with
    no reference to what the policy has ever reached:

    * **position** - a uniformly random voxel of the goal field that holds
      a finite potential ("reachable, where we have some potential"),
      jittered uniformly inside its own cell. Airspace included: the field
      is a 3-D BFS over free voxels and most of what it reaches is open
      air, which is where a surfer spends the map. Rejected only when the
      STANDING player hull does not fit at the jittered point
      (``core.trace(p, p, hull=0).startsolid``) or when the trilinear
      potential at that exact point is not finite - a 32 u cell whose
      centre is free can still clip a wall a few units away.
    * **view** - yaw uniform in [-180, 180), pitch uniform in
      ``pitch_range`` (default [-30, 15] deg, the band a surfer looks in).
      The C reset adds the env's own ``yaw_jitter_deg`` on top and wraps
      to [0, 360), exactly as it does for a reservoir row.
    * **velocity** - horizontal speed uniform in ``speed_range``
      (default [1000, 4000] u/s) along ``yaw + N(0, heading_sigma)``
      (default 30 deg): moving roughly where it is looking, as a surfer
      does. Vertical velocity 0, ``onground = -1`` (airborne), like every
      other exploring-start pool here.

    Nothing else about the state is set: the C reset zeroes the whole
    struct first and then copies the pool row, so tick / stuck_ticks /
    progress / ducked start from the same place a reservoir respawn does.
    That is the point of routing this through ``core.set_spawn_pool``
    rather than a bespoke reset: every counter the reward and liveness
    logic keys on an episode start (stall timer ``_since``, ``_best``,
    per-env ``_d0``, latch, arc anchors, novelty, the depth-history ring)
    already treats a pool draw as a true episode start, because that is
    what a reservoir respawn IS.

    Clamping the core applies, for the record: ``sv_maxvelocity`` is a
    PER-AXIS clamp inside PM_CheckVelocity (``src/pm.c``), so at the
    arm's ``--maxvel 4000`` a 4,000 u/s horizontal speed is never clamped
    (its largest component is at most 4,000, and only when the heading is
    axis-aligned). Pitch is not clamped at reset and the sampled band is
    inside the engine's own [-70, 30]. Yaw is wrapped, not clamped.

    ``d_stats`` reports the sampled states' potential distribution - the
    honest replacement for reservoir min-depth, which is meaningless
    without a reservoir.
    """

    def __init__(self, core, field, start_frac: float = 0.05,
                 speed_range: tuple = (1000.0, 4000.0),
                 pitch_range: tuple = (-30.0, 15.0),
                 heading_sigma: float = 30.0, seed: int = 31) -> None:
        self.core = core
        self.field = field
        self.start_frac = float(start_frac)
        self.speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.pitch_range = (float(pitch_range[0]), float(pitch_range[1]))
        self.heading_sigma = float(heading_sigma)
        self.rng = np.random.default_rng(int(seed))
        g = field.grid
        self.nz, self.ny, self.nx = g.shape
        self.cell = float(field.cell)
        self.mins = np.asarray(field.mins, np.float64)
        # the "honest corner" test sample() itself uses: anything that is
        # not the sentinel. The finite-potential guarantee is then
        # re-checked on the jittered point with field.reachable().
        self._vmax = float(getattr(field, "_valid_max",
                                   field.reach_max + 0.5 * field.cell))
        self.drawn = 0          # voxel indices drawn
        self.vox_valid = 0      # ... of which the voxel carries a potential
        self.kept = 0           # states accepted
        self.hull_tried = 0     # jittered points offered to the hull test
        self.hull_rejected = 0  # ... of which the standing hull did not fit
        self._last_d = None     # potentials of the last pool's random half

    # -- sampling -----------------------------------------------------------
    def _positions(self, n: int) -> np.ndarray:
        """(n, 3) float64 accepted positions.

        Two stages, because the second one is 300x the cost of the first:
        drawing a voxel index and reading its grid value is pure numpy, but
        the trilinear potential and the hull trace are per point. So the
        cheap stage over-draws (~10% of cannonball's voxels carry a
        potential), and only as many survivors as are still needed - plus a
        margin for the ~2% the hull rejects - reach the expensive one. The
        survivors are an i.i.d. uniform sequence, so a prefix of them is
        still uniform over the reachable voxels.
        """
        out = []
        got = 0
        while got < n:
            need = n - got
            k = max(4096, int(need * 12))
            ix = self.rng.integers(0, self.nx, k)
            iy = self.rng.integers(0, self.ny, k)
            iz = self.rng.integers(0, self.nz, k)
            self.drawn += k
            ok = self.field.grid[iz, iy, ix] < self._vmax
            self.vox_valid += int(ok.sum())
            if not ok.any():
                continue
            idx = np.stack([ix[ok], iy[ok], iz[ok]], 1).astype(np.float64)
            idx = idx[:int(need * 1.1) + 16]
            p = (self.mins + (idx + 0.5) * self.cell
                 + self.rng.uniform(-0.5 * self.cell, 0.5 * self.cell,
                                    idx.shape))
            # finite potential at the exact jittered point, not merely at
            # the voxel centre
            p = p[self.field.reachable(p)]
            if not len(p):
                continue
            fit = np.fromiter(
                (not self.core.trace(q, q, hull=0).startsolid for q in p),
                bool, len(p))
            self.hull_tried += len(p)
            self.hull_rejected += int((~fit).sum())
            p = p[fit]
            if len(p):
                out.append(p)
                got += len(p)
                self.kept += len(p)
        return np.concatenate(out)[:n]

    def sample_states(self, n: int) -> np.ndarray:
        """(n,) STATE_DTYPE of validated random reachable states."""
        n = int(n)
        rows = np.zeros(n, dtype=STATE_DTYPE)
        if n == 0:
            return rows
        p = self._positions(n)
        yaw = self.rng.uniform(-180.0, 180.0, n)
        pitch = self.rng.uniform(self.pitch_range[0], self.pitch_range[1], n)
        spd = self.rng.uniform(self.speed_range[0], self.speed_range[1], n)
        head = np.radians(yaw + self.rng.normal(0.0, self.heading_sigma, n))
        rows["origin"] = p
        vel = np.zeros((n, 3), np.float64)
        vel[:, 0] = spd * np.cos(head)
        vel[:, 1] = spd * np.sin(head)
        rows["velocity"] = vel
        rows["yaw"] = yaw
        rows["pitch"] = pitch
        rows["onground"] = -1
        self._last_d = np.asarray(self.field.sample(p), np.float64)
        return rows

    def build_pool(self, start_pool: np.ndarray,
                   pool_size: int = 4096) -> np.ndarray:
        """``start_frac`` map-start entries + the rest random. The env
        resets by UNIFORM pool draw, so entry counts ARE the
        probabilities (the same contract RespawnBuffer.build_pool uses)."""
        pool_size = int(pool_size)
        n_start = max(1, int(round(pool_size * self.start_frac)))
        n_rand = max(1, pool_size - n_start)
        fresh = start_pool[self.rng.integers(0, len(start_pool), n_start)]
        return np.concatenate([fresh, self.sample_states(n_rand)])

    # -- diagnostics --------------------------------------------------------
    def d_stats(self) -> dict:
        """Potential distribution of the LAST pool's random states -
        min / p10 / median / p90 / max geodesic distance-to-finish, plus
        the accept rate. Reservoir min-depth is meaningless here (there is
        no reservoir); this is what the ledger reports instead."""
        d = self._last_d
        if d is None or not len(d):
            return {}
        return {"n": int(len(d)), "min": float(d.min()),
                "p10": float(np.percentile(d, 10)),
                "median": float(np.median(d)),
                "p90": float(np.percentile(d, 90)),
                "max": float(d.max()),
                "accept": ((self.vox_valid / self.drawn)
                           if self.drawn else 0.0),
                "hull_reject": ((self.hull_rejected / self.hull_tried)
                                if self.hull_tried else 0.0)}

    def d_line(self, tag: str = "") -> str:
        st = self.d_stats()
        if not st:
            return ""
        return ("randspawn{} d: min {:,.0f}  p10 {:,.0f}  median {:,.0f}"
                "  p90 {:,.0f}  max {:,.0f}  ({:,} states, voxel accept "
                "{:.1%}, hull reject {:.2%})"
                .format(tag, st["min"], st["p10"], st["median"], st["p90"],
                        st["max"], st["n"], st["accept"], st["hull_reject"]))


class FrontierSpawnSampler:
    """``--respawn-frontier``: a FORWARD curriculum on the goal potential.

    The user's ask, verbatim: *"randomize the reservoir more by allowing to
    spawn the agent in places with higher potential compared to where it
    does. For example, if max potential so far is 100, allow it to respawn
    in points with potential up to 20% more than 100. The exact position is
    randomized. Speed taken from reservoir speeds, with up to 5.0 faster
    speed. Or rather let's do the following: when we get stuck, we start
    slowly increasing (linearly with time) the potential where we can
    respawn."*

    So the spawn frontier sits slightly BEYOND what the policy has actually
    reached, and under ``--respawn-frontier-grow`` it creeps further forward
    linearly with time whenever the honest frontier plateaus.

    This is an ADDITION to the reservoir, not a replacement for it (unlike
    :class:`RandomSpawnSampler`): the reservoir keeps harvesting, keeps
    reporting min-depth, and supplies the SPEED distribution the user asked
    for. :meth:`mix` overwrites the first ``n_front`` rows of the reservoir
    half of an already-built pool, so the map-start share (5 %, what the
    evals use) is untouched by construction.

    **progress, not distance.** ``progress = d0 - d`` where ``d0`` is the
    shaping field's geodesic distance at the map's own start spawn. "Up to
    20 % beyond the frontier" is ``progress <= (1 + margin) * P_max``, i.e.
    ``d >= d0 - (1 + margin) * P_max``. The band's far edge is ``d0``
    itself: spawning BEHIND the start is not a curriculum, it is noise.

    **P_max is START-ANCHORED, and that is not a detail.** Take P_max from
    every training episode and the loop is geometric: an episode spawned at
    1.2 x P_max instantly reports 1.2 x P_max, so the cap multiplies itself
    every iteration and reaches the goal in a couple of dozen of them - the
    trivial-win trap CLAUDE.md records (round 19 xPSSR, win rate 0 ->
    18.46 % off a reservoir that had collapsed to 1,485 u from the goal).
    The trainer therefore feeds :meth:`set_cap` a P_max measured ONLY on
    episodes that spawned at the true map start, which no spawn of this
    class can inflate. The plateau term is then the only way the frontier
    outruns real capability, and it is rate-limited in wall-clock.

    **Where inside the cap.** A uniform draw over voxels wastes most spawns
    near the start (that is where the voxels are, and the band always
    contains the whole run-up). Two corrections, both on:

    * every draw is **bin-flattened in d** - ``bins`` equal-width bins over
      the admitted band, a bin picked uniformly among the non-empty ones,
      then a member uniformly - so coverage is uniform in PROGRESS rather
      than in voxel count;
    * ``shell_frac`` of the states come from the **frontier shell**, the
      deepest ``shell_width`` fraction of the band, which is the aggressive
      half of the mixture.

    **Velocity.** Round 31's ``--respawn-random`` was a strong negative and
    a uniformly random heading on an airborne state is a large part of why:
    such a state is unrecoverable by construction. Here the direction comes
    from the field's own local descent, ``-grad d`` by central differences
    with invalid neighbours dropped, perturbed by ``heading_sigma`` in yaw
    and elevation and clamped to a survivable elevation band; the view is
    aimed along that same direction with its own small noise, because a
    surfer who is not looking where it is going cannot steer. The speed
    MAGNITUDE is drawn from the reservoir's own observed horizontal speeds
    and scaled by ``U(*speed_scale)`` - exactly what ``--respawn-speed``
    does to a reservoir row - then clamped to ``maxvel`` so the engine's
    PER-AXIS ``sv_maxvelocity`` clamp can never fire and bend the heading.

    **Clearance** reuses the ``RandomSpawnSampler`` rule: the jittered point
    must carry a finite trilinear potential AND the STANDING player hull
    must fit (``core.trace(p, p, hull=0).startsolid``). Rejection rates are
    counted and reported.
    """

    def __init__(self, core, field, d0: float, margin: float = 0.2,
                 speed_scale: tuple = (0.9, 5.0), shell_frac: float = 0.5,
                 shell_width: float = 0.25, floor: float = 512.0,
                 heading_sigma: float = 15.0, view_sigma: float = 10.0,
                 elev_range: tuple = (-60.0, 30.0), maxvel: float = 4000.0,
                 bins: int = 64, seed: int = 71) -> None:
        self.core = core
        self.field = field
        self.d0 = float(d0)
        self.margin = float(margin)
        self.speed_scale = (float(speed_scale[0]), float(speed_scale[1]))
        self.shell_frac = float(np.clip(shell_frac, 0.0, 1.0))
        self.shell_width = float(np.clip(shell_width, 1e-3, 1.0))
        self.floor = float(floor)
        self.heading_sigma = float(heading_sigma)
        self.view_sigma = float(view_sigma)
        self.elev_range = (float(elev_range[0]), float(elev_range[1]))
        self.maxvel = float(maxvel)
        self.bins = max(2, int(bins))
        self.rng = np.random.default_rng(int(seed))
        g = field.grid
        self.nz, self.ny, self.nx = g.shape
        self.cell = float(field.cell)
        self.mins = np.asarray(field.mins, np.float64)
        self._vmax = float(getattr(field, "_valid_max",
                                   field.reach_max + 0.5 * field.cell))
        # the live cap, in PROGRESS units. Starts at the floor so the very
        # first pool is a narrow band around the start rather than empty.
        self.p_cap = float(floor)
        self.p_max = 0.0
        self.grow = 0.0
        # counters (all cumulative, all reported)
        self.drawn = 0            # voxel indices drawn
        self.band_ok = 0          # ... of which inside the admitted band
        self.hull_tried = 0       # jittered points offered to the hull test
        self.hull_rejected = 0    # ... rejected by the standing hull
        self.cap_rejected = 0     # ... whose trilinear d fell below the cap
        self.grad_rejected = 0    # ... with no usable field gradient
        self.kept = 0
        self.n_shell = 0          # states drawn from the frontier shell
        self.empty_band = 0       # draws that found no candidate at all
        self._last_d = None       # potentials of the last batch
        self._last_spd = None     # speeds of the last batch
        self._last_shell = 0

    # -- the cap ------------------------------------------------------------
    def set_cap(self, p_max: float, grow: float = 0.0) -> float:
        """``p_max`` = the START-ANCHORED frontier progress, ``grow`` = the
        plateau schedule's extra margin. Returns the cap actually used."""
        pm = float(p_max) if p_max == p_max else 0.0
        self.p_max = max(0.0, pm)
        self.grow = max(0.0, float(grow))
        cap = (1.0 + self.margin + self.grow) * self.p_max
        self.p_cap = float(max(self.floor, min(cap, self.d0)))
        return self.p_cap

    @property
    def d_lo(self) -> float:
        """The band's near edge in DISTANCE: the deepest admitted d."""
        return max(0.0, self.d0 - self.p_cap)

    # -- the field's local descent -----------------------------------------
    def descent_dir(self, p: np.ndarray, h: float | None = None):
        """(n, 3) unit ``-grad d`` and (n,) a validity mask.

        Central differences on the trilinear field, one-sided where the far
        side is the sentinel, component zeroed where neither side is valid.
        Invalid = the gradient is degenerate, which happens in a pocket the
        BFS entered from one direction only; those points are dropped rather
        than given an arbitrary heading."""
        p = np.atleast_2d(np.asarray(p, np.float64))
        h = float(self.cell * 2.0 if h is None else h)
        lim = self.field.reach_max - 0.5 * self.cell
        c = np.asarray(self.field.sample(p), np.float64)
        g = np.zeros((len(p), 3), np.float64)
        for ax in range(3):
            off = np.zeros(3, np.float64)
            off[ax] = h
            a = np.asarray(self.field.sample(p + off), np.float64)
            b = np.asarray(self.field.sample(p - off), np.float64)
            va, vb = a < lim, b < lim
            both = va & vb
            g[both, ax] = (a[both] - b[both]) / (2.0 * h)
            only_a = va & ~vb
            g[only_a, ax] = (a[only_a] - c[only_a]) / h
            only_b = vb & ~va
            g[only_b, ax] = (c[only_b] - b[only_b]) / h
        n = np.linalg.norm(g, axis=1)
        ok = n > 1e-6
        d = np.zeros_like(g)
        d[ok] = -g[ok] / n[ok, None]
        return d, ok

    # -- sampling -----------------------------------------------------------
    def _flatten_pick(self, vals: np.ndarray, m: int, lo: float,
                      hi: float) -> np.ndarray:
        """Indices into ``vals`` drawn uniformly in d: a non-empty bin of
        ``self.bins`` over [lo, hi] uniformly, then a member uniformly."""
        if m <= 0 or not len(vals):
            return np.empty(0, np.int64)
        span = max(hi - lo, 1e-6)
        b = np.clip(((vals - lo) / span * self.bins).astype(np.int64),
                    0, self.bins - 1)
        order = np.argsort(b, kind="stable")
        bs = b[order]
        uniq, start, cnt = np.unique(bs, return_index=True, return_counts=True)
        pick_b = self.rng.integers(0, len(uniq), m)
        within = (self.rng.random(m) * cnt[pick_b]).astype(np.int64)
        return order[start[pick_b] + np.minimum(within, cnt[pick_b] - 1)]

    def _accept(self, ix, iy, iz):
        """Voxel indices -> jittered positions that survive every gate."""
        idx = np.stack([ix, iy, iz], 1).astype(np.float64)
        p = (self.mins + (idx + 0.5) * self.cell
             + self.rng.uniform(-0.5 * self.cell, 0.5 * self.cell,
                                idx.shape))
        # the jittered point, not the voxel centre, is what spawns: re-check
        # BOTH the finite potential and the cap there
        dj = np.asarray(self.field.sample(p), np.float64)
        keep = self.field.reachable(p) & (dj >= self.d_lo)
        self.cap_rejected += int((~keep).sum())
        p = p[keep]
        if not len(p):
            return p
        fit = np.fromiter(
            (not self.core.trace(q, q, hull=0).startsolid for q in p),
            bool, len(p))
        self.hull_tried += len(p)
        self.hull_rejected += int((~fit).sum())
        p = p[fit]
        if not len(p):
            return p
        _, gok = self.descent_dir(p)
        self.grad_rejected += int((~gok).sum())
        return p[gok]

    def _positions(self, n: int):
        """(m, 3) accepted positions inside the cap, and how many of them
        came from the frontier shell.

        Shell and body are drawn from DISJOINT d ranges and accepted in
        their own target counts, so ``shell_frac`` is the realised share
        and not merely the share of the candidates offered."""
        d_lo, d_hi = self.d_lo, self.d0
        d_shell = d_lo + self.shell_width * max(d_hi - d_lo, 1e-6)
        want_shell = int(round(n * self.shell_frac))
        got_sh, got_bd = [], []
        n_sh = n_bd = 0
        tries = 0
        while (n_sh + n_bd) < n and tries < 64:
            tries += 1
            need_sh = max(0, want_shell - n_sh)
            need_bd = max(0, (n - want_shell) - n_bd)
            if need_sh + need_bd == 0:
                break
            k = max(8192, int((need_sh + need_bd) * 24))
            ix = self.rng.integers(0, self.nx, k)
            iy = self.rng.integers(0, self.ny, k)
            iz = self.rng.integers(0, self.nz, k)
            self.drawn += k
            gv = self.field.grid[iz, iy, ix].astype(np.float64)
            ok = (gv < self._vmax) & (gv >= d_lo) & (gv <= d_hi)
            self.band_ok += int(ok.sum())
            if not ok.any():
                self.empty_band += 1
                continue
            ix, iy, iz, gv = ix[ok], iy[ok], iz[ok], gv[ok]
            sh = gv <= d_shell
            for mask, need, lo, hi, sink in (
                    (sh, need_sh, d_lo, d_shell, got_sh),
                    (~sh, need_bd, d_shell, d_hi, got_bd)):
                if need <= 0 or not mask.any():
                    continue
                w = np.flatnonzero(mask)
                # 1.4x for the ~10% the cap / hull / gradient gates reject
                j = w[self._flatten_pick(gv[w], int(need * 1.4) + 8, lo, hi)]
                q = self._accept(ix[j], iy[j], iz[j])
                if len(q):
                    sink.append(q[:need])
            n_sh = sum(len(a) for a in got_sh)
            n_bd = sum(len(a) for a in got_bd)
        # a starved half is topped up by the other rather than returned
        # short: the pool size is what fixes the spawn PROBABILITIES
        parts = got_sh + got_bd
        if not parts:
            return np.empty((0, 3), np.float64), 0
        p = np.concatenate(parts)[:n]
        n_sh = min(n_sh, len(p))
        self.kept += len(p)
        return p, n_sh

    def _reservoir_speeds(self, reservoir, n: int) -> np.ndarray:
        """``n`` horizontal speeds drawn from the reservoir's own stored
        velocities - the empirical distribution, not a parametric one."""
        v = reservoir._store[:reservoir.size]["velocity"]
        s = np.hypot(v[:, 0], v[:, 1]).astype(np.float64)
        return s[self.rng.integers(0, len(s), n)]

    def sample_states(self, n: int, reservoir) -> np.ndarray:
        """(m,) STATE_DTYPE, ``m <= n`` (a starved band returns short)."""
        n = int(n)
        if n <= 0 or reservoir is None or reservoir.size == 0:
            return np.zeros(0, dtype=STATE_DTYPE)
        p, n_sh = self._positions(n)
        m = len(p)
        self._last_shell = n_sh
        self.n_shell += n_sh
        if m == 0:
            self._last_d = self._last_spd = None
            return np.zeros(0, dtype=STATE_DTYPE)
        dir3, _ = self.descent_dir(p)
        yaw = np.degrees(np.arctan2(dir3[:, 1], dir3[:, 0]))
        elev = np.degrees(np.arcsin(np.clip(dir3[:, 2], -1.0, 1.0)))
        yaw = yaw + self.rng.normal(0.0, self.heading_sigma, m)
        elev = np.clip(elev + self.rng.normal(0.0, self.heading_sigma, m),
                       self.elev_range[0], self.elev_range[1])
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        ce, se = np.cos(np.radians(elev)), np.sin(np.radians(elev))
        spd = self._reservoir_speeds(reservoir, m) * self.rng.uniform(
            self.speed_scale[0], self.speed_scale[1], m)
        # clamp the MAGNITUDE, so PM_CheckVelocity's per-axis clamp (every
        # component is <= the magnitude) can never fire and bend the heading
        spd = np.minimum(spd, self.maxvel)
        rows = np.zeros(m, dtype=STATE_DTYPE)
        rows["origin"] = p
        vel = np.stack([spd * ce * cy, spd * ce * sy, spd * se], 1)
        rows["velocity"] = vel
        # look where you are going: the same direction, its own small noise
        rows["yaw"] = yaw + self.rng.normal(0.0, self.view_sigma, m)
        rows["pitch"] = np.clip(
            elev + self.rng.normal(0.0, self.view_sigma, m), -70.0, 30.0)
        rows["onground"] = -1
        self._last_d = np.asarray(self.field.sample(p), np.float64)
        self._last_spd = spd
        return rows

    def mix(self, pool: np.ndarray, n_fresh: int, n_front: int,
            reservoir) -> np.ndarray:
        """Overwrite ``n_front`` RESERVOIR rows of an already-built pool.

        The pool ``RespawnBuffer.build_pool`` returns is
        ``[fresh (n_fresh), reservoir (rest)]`` and the env resets by uniform
        pool draw, so entry counts ARE the probabilities: taking the frontier
        share out of the reservoir rows leaves the map-start share exactly
        where it was."""
        n_front = int(min(max(n_front, 0), len(pool) - int(n_fresh)))
        if n_front <= 0:
            return pool
        rows = self.sample_states(n_front, reservoir)
        if not len(rows):
            return pool
        out = np.array(pool, dtype=STATE_DTYPE, copy=True)
        out[int(n_fresh):int(n_fresh) + len(rows)] = rows
        return out

    # -- diagnostics --------------------------------------------------------
    def stats(self) -> dict:
        """The trap guard's half of the ledger line: WHERE the frontier
        states landed on the shaping potential, in PROGRESS units."""
        out = {"p_max": self.p_max, "p_cap": self.p_cap, "grow": self.grow,
               "hull_reject": ((self.hull_rejected / self.hull_tried)
                               if self.hull_tried else 0.0),
               "cap_reject": (self.cap_rejected
                              / max(self.hull_tried + self.cap_rejected, 1)),
               "band_accept": ((self.band_ok / self.drawn)
                               if self.drawn else 0.0),
               "shell_frac": ((self.n_shell / self.kept)
                              if self.kept else 0.0)}
        d = self._last_d
        if d is not None and len(d):
            pr = self.d0 - d
            out.update({"n": int(len(pr)), "min": float(pr.min()),
                        "p10": float(np.percentile(pr, 10)),
                        "median": float(np.median(pr)),
                        "p90": float(np.percentile(pr, 90)),
                        "max": float(pr.max())})
        s = self._last_spd
        if s is not None and len(s):
            out.update({"spd_med": float(np.median(s)),
                        "spd_p90": float(np.percentile(s, 90)),
                        "spd_max": float(s.max())})
        return out

    def line(self, tag: str = "") -> str:
        st = self.stats()
        if "median" not in st:
            return ("frontier{}: no states (cap {:,.0f}u of {:,.0f}u, band "
                    "starved)".format(tag, st["p_cap"], self.d0))
        return ("frontier{} progress: cap {:,.0f}u (Pmax {:,.0f}u, grow "
                "{:+.2f})  spawns min {:,.0f}  p10 {:,.0f}  median {:,.0f}"
                "  p90 {:,.0f}  max {:,.0f}  ({:,} states, shell {:.0%}, "
                "band {:.2%}, hull rej {:.2%}, spd med {:,.0f} p90 {:,.0f})"
                .format(tag, st["p_cap"], st["p_max"], st["grow"],
                        st["min"], st["p10"], st["median"], st["p90"],
                        st["max"], st["n"], st["shell_frac"],
                        st["band_accept"], st["hull_reject"],
                        st.get("spd_med", float("nan")),
                        st.get("spd_p90", float("nan"))))
