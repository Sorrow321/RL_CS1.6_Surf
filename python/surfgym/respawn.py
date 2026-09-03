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

__all__ = ["RespawnBuffer", "DemoCurriculum"]


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
                 min_ep: float = 50.0, seed: int = 29) -> None:
        self.S = np.asarray(states, STATE_DTYPE)
        self.n = len(self.S)
        self.D = int(window)
        self.rate = float(rate)
        self.min_ep = float(min_ep)
        self.tau = self.n - 1
        self.ep = np.zeros(self.n, np.float64)
        self.win = np.zeros(self.n, np.float64)
        self.decay = 0.99
        self._cool = 0
        self.rng = np.random.default_rng(seed)
        self._key = {tuple(np.round(np.asarray(r["origin"], np.float64), 1)): i
                     for i, r in enumerate(self.S)}
        self.last_info = ""

    def _lo(self) -> int:
        return max(0, self.tau - self.D + 1)

    def build_pool(self, start_pool: np.ndarray, pool_size: int = 4096,
                   fresh_frac: float = 0.10) -> np.ndarray:
        n_fresh = max(1, int(round(pool_size * fresh_frac)))
        n_demo = pool_size - n_fresh
        self._move()
        idx = self.rng.integers(self._lo(), self.tau + 1, n_demo)
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
        lo = self._lo()
        ep = float(self.ep[lo:self.tau + 1].sum())
        win = float(self.win[lo:self.tau + 1].sum())
        r = win / max(ep, 1e-9)
        self._cool = max(0, self._cool - 1)
        moved = 0
        if ep >= self.min_ep and self._cool == 0:
            if r >= self.rate and self.tau > 0:
                self.tau -= 1
                moved = -1
            elif r < self.rate and self.tau < self.n - 1:
                self.tau += 1
                moved = +1
            if moved:
                self._cool = 20
        self.last_info = (f"demo window [{self._lo()},{self.tau}]/{self.n} "
                          f"success {r:.1%} over {ep:.0f} eps")
        if moved:
            print(f"demo curriculum: tau {'<- earlier' if moved < 0 else '-> later (backoff)'}"
                  f" window [{self._lo()},{self.tau}] (success {r:.1%} "
                  f"over {ep:.0f} eps)")


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
