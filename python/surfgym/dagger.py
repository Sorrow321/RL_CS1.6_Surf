"""dagger.py - DAgger-style relabelling for expert iteration.

WHY. Expert iteration (tools/expert_loop.py) only ever shows the policy the
planner's ELITE line: every BC row is a (state, action) pair along a line
the policy itself never quite flies. Its own small deviations compound over
~7,500 decisions into states the elite rows never cover, and there it has no
supervision at all - the covariate shift DAgger (Ross, Gordon, Bagnell 2011)
removes by querying the expert AT THE LEARNER'S OWN STATES. Round 30's
exit10/exit30 measured exactly that wall: 98% per-head agreement on the
elite rows, the greedy policy 1.5-2.4 s behind the planner's line, and
doubling the planner and training budgets per round moved nothing.

WHAT. Three steps, all in the real simulator (the C core is the expert's
model as it is the planner's):

1. SAMPLE the current policy's own states (collect_rollout_samples): greedy
   and tempered rollouts from the map start plus rollouts from spawns along
   the elite spine, one state every ``every_ticks`` of each rollout
   (decision_grid), K of them spread evenly over the candidates
   (even_subset). A sample carries the full physics state, the scalar row
   the policy SAW at that decision (the eval wrapper's _obs output: the 15
   core scalars with the --obs-reward slot-12 mirror, plus the --race-latch
   column) and the two feeds' internals from the decision BEFORE, so a
   planner restarted at that state reads the same slot 12 / latch at its
   first decision as the policy did.
2. RELABEL each sample with a short planner window from that exact state
   (relabel_windows): ``copies`` clones of the state in one batched core,
   envs [0, n_greedy) of the group greedy and the rest sampled, elites
   cloned over laggards every ``resample_every`` decisions - beam_tas's
   population search, grouped so one core runs several states at once. The
   objective is the FIRST FINISH inside the window (lockstep: the first
   crossing is the fastest), else the best score at the window's end
   (geodesic d, or the critic near the goal). The LABEL is the winning
   lineage's first decision(s): what the expert does HERE.
3. AGGREGATE (merge_bc_datasets): the relabelled rows join the elite rows
   in one surfgym.bc file, weighted by how far the sampled state sits from
   the elite line (divergence_weights): the states the elite line does not
   cover carry the most weight, and the relabelled rows as a whole get a
   fixed SHARE of the BC loss mass.

The loops here take a ``decider`` (``act(obs) -> (N, 6) int32`` with the
act_every hold, exposing ``last_row`` = the scalar half of the row it just
built and ``last_greedy`` = its argmax actions) and ``feeds`` = (reward_feed,
latch_feed) as surfgym.bc.make_eval_feeds returns them (either may be
None), so they run under a stub policy and a stub core in the CPU tests
(tests/python/test_expert_dagger.py) and under the trainer's own eval
wrappers in tools/expert_dagger.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .bc import N_SCALAR, load_bc_meta, save_bc_dataset
from .core import ACTION_NVEC, STATE_DTYPE

__all__ = ["DAGGER_LINE_ID", "SRC_GREEDY", "SRC_STOCH", "SRC_SPINE",
           "SRC_NAMES", "TICKS_PER_S", "on_grid", "decision_grid",
           "even_subset", "SampleBank", "collect_rollout_samples",
           "relabel_windows", "nearest_distance", "divergence_weights",
           "rows_from_results", "summarize_results", "merge_bc_datasets"]

DAGGER_LINE_ID = -1        # line_id of a relabelled row (elite lines are >= 0)
SRC_GREEDY, SRC_STOCH, SRC_SPINE = 0, 1, 2
SRC_NAMES = ("greedy", "stoch", "spine")
TICKS_PER_S = 100          # 10 ms physics tick


# --------------------------------------------------------------------------
# 1. sampling
# --------------------------------------------------------------------------
def on_grid(d: int, k: int, every_ticks: int) -> bool:
    """Is decision ``d`` (taken at tick d*k) the first decision at or after
    a multiple of ``every_ticks``? Decision 0 always is."""
    every = max(int(every_ticks), 1)
    if d <= 0:
        return True
    return (d * k) // every != ((d - 1) * k) // every


def decision_grid(n_decisions: int, k: int, every_ticks: int) -> np.ndarray:
    """Indices of the decisions on the ``every_ticks`` grid: decision 0 and
    then the first decision at or after each multiple of every_ticks, so
    consecutive picks are every_ticks apart to within k-1 ticks."""
    n = int(n_decisions)
    if n <= 0:
        return np.zeros(0, np.int64)
    d = np.arange(n, dtype=np.int64)
    every = max(int(every_ticks), 1)
    cell = (d * int(k)) // every
    prev = np.concatenate([[-1], cell[:-1]])
    return d[cell != prev]


def even_subset(n: int, m: int, rng=None) -> np.ndarray:
    """``m`` indices spread evenly over ``range(n)`` (all of them when m >=
    n): one per stratum of m equal strata, so every part of the candidate
    pool is represented in proportion - the stratum's centre, or with a
    numpy Generator a uniform pick inside it (a pool whose candidates
    interleave several rollouts aliases with a fixed stride: a stride
    equal to the rollout count picks one rollout only)."""
    n, m = int(n), int(m)
    if n <= 0 or m <= 0:
        return np.zeros(0, np.int64)
    if m >= n:
        return np.arange(n, dtype=np.int64)
    lo = (np.arange(m) * n) // m
    hi = ((np.arange(m) + 1) * n) // m
    if rng is None:
        return ((np.arange(m) + 0.5) * n / m).astype(np.int64)
    return (lo + rng.integers(0, np.maximum(hi - lo, 1))).astype(np.int64)


class SampleBank:
    """The sampled states, with everything a planner needs to restart at
    one: the STATE_DTYPE state, the scalar row the policy saw (``scal``, 15
    columns, slot 12 = the --obs-reward feed value when the checkpoint uses
    it), the latch column, and the feeds' internals from the decision
    before (``d_prev`` NaN / ``tick_prev`` -1 = none: an episode's first
    decision), plus provenance (``src`` SRC_*, ``ep``, ``tick``)."""

    KEYS = ("states", "scal", "latch", "d_prev", "f_prev", "tick_prev",
            "src", "ep", "tick")

    def __init__(self, arrays: dict | None = None):
        self._lists = None if arrays is not None else {k: [] for k in self.KEYS}
        self._arr = None if arrays is None else dict(arrays)

    def add(self, state, scal, latch, d_prev, f_prev, tick_prev, src, ep,
            tick) -> None:
        if self._lists is None:
            raise RuntimeError("a loaded/selected SampleBank is read-only")
        li = self._lists
        li["states"].append(np.asarray(state, STATE_DTYPE).reshape(-1)[0])
        li["scal"].append(np.asarray(scal, np.float32).reshape(N_SCALAR).copy())
        li["latch"].append(float(latch))
        li["d_prev"].append(np.nan if d_prev is None else float(d_prev))
        li["f_prev"].append(0.0 if f_prev is None else float(f_prev))
        li["tick_prev"].append(-1 if tick_prev is None else int(tick_prev))
        li["src"].append(int(src))
        li["ep"].append(int(ep))
        li["tick"].append(int(tick))
        self._arr = None

    def __len__(self) -> int:
        if self._arr is not None:
            return int(len(self._arr["states"]))
        return len(self._lists["states"])

    def arrays(self) -> dict:
        if self._arr is None:
            li = self._lists
            n = len(li["states"])
            self._arr = {
                "states": np.array(li["states"], dtype=STATE_DTYPE).reshape(n),
                "scal": np.array(li["scal"], np.float32).reshape(n, N_SCALAR),
                "latch": np.array(li["latch"], np.float32).reshape(n),
                "d_prev": np.array(li["d_prev"], np.float64).reshape(n),
                "f_prev": np.array(li["f_prev"], np.float32).reshape(n),
                "tick_prev": np.array(li["tick_prev"], np.int64).reshape(n),
                "src": np.array(li["src"], np.int8).reshape(n),
                "ep": np.array(li["ep"], np.int64).reshape(n),
                "tick": np.array(li["tick"], np.int64).reshape(n)}
        return self._arr

    def select(self, idx) -> "SampleBank":
        idx = np.asarray(idx, np.int64)
        a = self.arrays()
        return SampleBank({k: a[k][idx] for k in self.KEYS})

    def save(self, path) -> None:
        np.savez(path, **self.arrays())

    @classmethod
    def load(cls, path) -> "SampleBank":
        z = np.load(path, allow_pickle=False)
        return cls({k: np.asarray(z[k]) for k in cls.KEYS})


def _feed_prev(feed, key):
    """A copy of a feed's per-env internal ``key`` (None when unset)."""
    if feed is None:
        return None
    v = feed.state.get(key)
    return None if v is None else np.array(v)


def collect_rollout_samples(core, decider, feeds, obs, k: int, n_ticks: int,
                            every_ticks: int, src, ep_ids, bank: SampleBank,
                            active=None, skip_first: bool = False) -> dict:
    """Roll every env of ``core`` closed-loop under ``decider`` for up to
    ``n_ticks`` (or until every env has ended), recording into ``bank`` at
    each decision on the every_ticks grid: the full state, the scalar row
    the policy saw, the latch, the feeds' internals from the decision
    before. An env that ends (death or finish) autoresets into a fresh pool
    spawn and is dropped: only the SAME episode's states count. ``src`` is
    an int or a per-env array of SRC_*; ``ep_ids`` a per-env label.

    ``skip_first`` leaves decision 0 out: with FRESH feeds (a map-start
    rollout) the reward feed reads 0 there and only there, which no primed
    restart reproduces - and the spawn states are the elite rows' own
    starts anyway. Rollouts from primed feeds (spine spawns) keep it.

    Returns ``finished`` / ``end_tick`` (-1 = still alive at the cap) /
    ``alive`` per env and the number of rows recorded."""
    rf, lf = feeds
    N = int(core.num_envs)
    alive = (np.ones(N, bool) if active is None
             else np.asarray(active, bool).copy())
    src = np.broadcast_to(np.asarray(src, np.int64), (N,))
    ep_ids = np.broadcast_to(np.asarray(ep_ids, np.int64), (N,))
    finished = np.zeros(N, bool)
    end_tick = np.full(N, -1, np.int64)
    n_rec = 0
    k = max(1, int(k))
    for t in range(int(n_ticks)):
        if not alive.any():
            break
        if t % k == 0:
            d = t // k
            take = on_grid(d, k, every_ticks) and not (skip_first and d == 0)
            d_prev = _feed_prev(rf, "d") if take else None
            f_prev = _feed_prev(lf, "f") if take else None
            t_prev = _feed_prev(lf, "tick") if take else None
            a = decider.act(obs)
            if take:
                st = core.get_states()
                row = np.asarray(decider.last_row, np.float32)
                latch_col = row.shape[1] > N_SCALAR
                for i in np.nonzero(alive)[0]:
                    bank.add(st[i], row[i, :N_SCALAR],
                             float(row[i, N_SCALAR]) if latch_col else 0.0,
                             None if d_prev is None else float(d_prev[i]),
                             None if f_prev is None else float(f_prev[i]),
                             None if t_prev is None else int(t_prev[i]),
                             int(src[i]), int(ep_ids[i]), t)
                    n_rec += 1
        else:
            a = decider.act(obs)
        obs, _rew, done, trunc, _term = core.step(a)
        hit = np.asarray(core.goal_hits, bool)
        finished |= alive & hit
        ended = np.asarray(done, bool) | np.asarray(trunc, bool)
        newly = alive & ended
        end_tick[newly] = t + 1
        alive &= ~ended
    return {"finished": finished, "end_tick": end_tick, "alive": alive,
            "recorded": int(n_rec)}


# --------------------------------------------------------------------------
# 2. the short planner window from each sampled state
# --------------------------------------------------------------------------
def _prime_feeds(feeds, S, idx_s, copies, N, states_now, field, d_floor):
    """Set the feeds' per-env internals so that the window's first decision
    reads what the policy read at that decision of its own rollout: the
    reward feed's previous d, the latch flag and tick of the decision
    before. A sample without a previous decision (an episode's first) gets
    d_prev = d(now) (zero progress) and tick_prev = tick - 1."""
    rf, lf = feeds
    n_act = len(idx_s) * copies
    if rf is not None:
        full = np.zeros(N, np.float64)
        full[:n_act] = np.repeat(S["d_prev"][idx_s], copies)
        miss = np.isnan(full)
        if miss.any():
            dn = np.asarray(field.sample(states_now["origin"]), np.float64)
            if d_floor > 0.0:
                dn = np.maximum(dn, d_floor)
            full[miss] = dn[miss]
        rf.state["d"] = full
    if lf is not None:
        f = np.zeros(N, bool)
        f[:n_act] = np.repeat(S["f_prev"][idx_s] > 0.5, copies)
        tk = np.full(N, -1, np.int64)
        tp = np.repeat(S["tick_prev"][idx_s], copies)
        now = np.asarray(states_now["tick"], np.int64)[:n_act]
        tk[:n_act] = np.where(tp < 0, now - 1, tp)
        lf.state["f"] = f
        lf.state["tick"] = tk


def _clone(core, states, hist, d, obs, st_hist, row_hist, feeds, losers,
           donors):
    """donor -> loser: physics state, action history, obs row, the label
    snapshots and the feeds' per-env internals (beam_tas's resample)."""
    rf, lf = feeds
    for j, don in zip(losers, donors):
        core.set_state(int(j), states[don])
    hist[:d + 1, losers] = hist[:d + 1, donors]
    obs[losers] = obs[donors]
    st_hist[:, losers] = st_hist[:, donors]
    if row_hist is not None:
        row_hist[:, losers] = row_hist[:, donors]
    if rf is not None and rf.state.get("d") is not None:
        rf.state["d"][losers] = rf.state["d"][donors]
    if lf is not None and lf.state.get("f") is not None:
        lf.state["f"][losers] = lf.state["f"][donors]
        lf.state["tick"][losers] = lf.state["tick"][donors]


def relabel_windows(core, make_decider, make_feeds, field, score_fn,
                    samples: SampleBank, k: int, copies: int,
                    window_decisions: int, resample_every: int = 25,
                    elite_frac: float = 0.25, n_greedy: int = 1,
                    label_decisions: int = 1, seed: int = 0,
                    budget_s: float = 0.0, d_floor: float = 0.0,
                    log=None) -> list:
    """The DAgger labels: one short population search per sample.

    ``core`` has N envs; ``copies`` envs per sample, so N // copies samples
    run per chunk in lockstep. Per chunk: every env of a group is set to
    the sample's state and its obs row, the feeds are primed
    (_prime_feeds), and the group runs ``window_decisions`` decisions under
    ``make_decider(feeds, greedy_mask)`` (envs with ``greedy_mask`` act
    greedily: the first ``n_greedy`` of each group, so a group is bounded
    below by the policy's own continuation), cloning its elites over the
    rest every ``resample_every`` decisions (0 = never) by
    ``score_fn(states, obs) -> (score, d)`` (higher is better). The FIRST
    finish inside a group ends its search (lockstep: it is the group's
    fastest); otherwise the best-scoring live env at the window's end
    wins; a group with nothing alive and no finish is ``extinct``.

    Per sample the result dict carries ``label_acts`` ((L, 6) int, the
    winner's first ``label_decisions`` actions), ``label_states`` (the
    STATE_DTYPE state before each of those decisions - row 0 IS the sample
    state) and ``label_rows`` (the scalar row the decider built at each),
    ``finished`` / ``end_tick``, ``end_d``, the greedy env's own outcome
    (``greedy_end_d``, ``greedy_alive``, ``greedy_act0``) and ``disagree``
    (label[0] != the policy's greedy action at that state: the rows that
    carry information). ``budget_s`` > 0 stops issuing chunks once that
    wall clock is spent; results of unprocessed samples are None."""
    N = int(core.num_envs)
    copies = int(copies)
    if copies < 1 or copies > N:
        raise ValueError(f"copies {copies} must be in [1, {N}]")
    G = N // copies
    k = max(1, int(k))
    H = max(1, int(window_decisions))
    R = int(resample_every)
    L = max(1, min(int(label_decisions), H))
    n_elite = max(1, int(round(copies * float(elite_frac))))
    n_greedy = max(0, min(int(n_greedy), copies))
    greedy_mask = (np.arange(N) % copies) < n_greedy
    S = samples.arrays()
    n = len(samples)
    results: list = [None] * n
    t0 = time.time()
    n_chunks = (n + G - 1) // G
    for ci, c0 in enumerate(range(0, n, G)):
        if budget_s > 0.0 and ci > 0 and time.time() - t0 > budget_s:
            if log:
                log(f"relabel: budget {budget_s:.0f}s spent after {c0} of "
                    f"{n} samples - stopping")
            break
        ng = min(G, n - c0)
        idx_s = np.arange(c0, c0 + ng)
        obs = np.array(core.reset(int(seed) + c0))
        feeds = make_feeds()
        rf, lf = feeds
        decider = make_decider(feeds, greedy_mask)
        active = np.zeros(N, bool)
        for g in range(ng):
            s = c0 + g
            lo = g * copies
            for c in range(copies):
                core.set_state(lo + c, S["states"][s])
            obs[lo:lo + copies, :N_SCALAR] = S["scal"][s][None, :]
            active[lo:lo + copies] = True
        valid = active.copy()
        _prime_feeds(feeds, S, idx_s, copies, N, core.get_states(), field,
                     d_floor)
        hist = np.zeros((H, N, 6), np.int8)
        st_hist = np.zeros((L, N), STATE_DTYPE)
        row_hist = None
        greedy0 = None
        fin: list = [None] * ng
        for t in range(H * k):
            d = t // k
            if t % k == 0 and d < L:
                st_hist[d] = core.get_states()
            a = decider.act(obs)
            if t % k == 0:
                hist[d] = a
                if d < L:
                    row = np.asarray(decider.last_row, np.float32)
                    if row_hist is None:
                        row_hist = np.zeros((L, N, row.shape[1]), np.float32)
                    row_hist[d] = row
                if d == 0:
                    greedy0 = np.asarray(decider.last_greedy, np.int32).copy()
            obs, _rew, done, trunc, _term = core.step(a)
            hit = np.asarray(core.goal_hits, bool) & valid & active
            if hit.any():
                for i in np.nonzero(hit)[0]:
                    g = int(i) // copies
                    if fin[g] is None:
                        fin[g] = (t + 1, int(i), hist[:L, i].copy(),
                                  st_hist[:, i].copy(),
                                  None if row_hist is None
                                  else row_hist[:, i].copy())
            ended = np.asarray(done, bool) | np.asarray(trunc, bool)
            valid &= ~ended
            if R > 0 and (t + 1) % (R * k) == 0 and (t + 1) < H * k:
                states = core.get_states()
                sc, _dd = score_fn(states, obs)
                obs = np.array(obs)
                for g in range(ng):
                    if fin[g] is not None:
                        continue          # solved: nothing to search
                    lo, hi = g * copies, (g + 1) * copies
                    v = valid[lo:hi]
                    if not v.any():
                        continue          # extinct: nothing to clone from
                    loc = np.argsort(np.where(v, -sc[lo:hi], np.inf),
                                     kind="stable")
                    elig = lo + loc[v[loc]]
                    keep = elig[:n_elite]
                    keep_set = np.zeros(copies, bool)
                    keep_set[keep - lo] = True
                    if n_greedy > 0 and valid[lo]:
                        keep_set[0] = True   # the untouched greedy line
                    losers = lo + np.nonzero(~keep_set)[0]
                    if len(losers) == 0:
                        continue
                    donors = keep[np.arange(len(losers)) % len(keep)]
                    _clone(core, states, hist, d, obs, st_hist, row_hist,
                           feeds, losers, donors)
                    valid[lo:hi] = True
        states = core.get_states()
        sc, dd = score_fn(states, obs)
        for g in range(ng):
            s = c0 + g
            lo, hi = g * copies, (g + 1) * copies
            res = {"sample": int(s), "finished": False, "end_tick": None,
                   "end_d": float("nan"), "greedy_end_d": float("nan"),
                   "greedy_alive": (bool(valid[lo]) if n_greedy > 0 else None),
                   "greedy_act0": (None if greedy0 is None
                                   else greedy0[lo].copy()),
                   "alive_frac": float(valid[lo:hi].mean()),
                   "extinct": False, "winner": None,
                   "label_acts": None, "label_states": None,
                   "label_rows": None}
            if fin[g] is not None:
                ft, i, acts, sts, rows = fin[g]
                n_lab = max(1, min(L, (ft - 1) // k + 1))
                res.update(finished=True, end_tick=int(ft), winner=int(i),
                           label_acts=acts[:n_lab].astype(np.int64),
                           label_states=sts[:n_lab],
                           label_rows=(None if rows is None
                                       else rows[:n_lab]))
            else:
                v = valid[lo:hi]
                if not v.any():
                    res["extinct"] = True
                    results[s] = res
                    continue
                best = lo + int(np.argmax(np.where(v, sc[lo:hi], -np.inf)))
                res.update(end_tick=H * k, winner=int(best),
                           end_d=float(dd[best]),
                           label_acts=hist[:L, best].astype(np.int64),
                           label_states=st_hist[:, best].copy(),
                           label_rows=(None if row_hist is None
                                       else row_hist[:, best].copy()))
            if n_greedy > 0 and valid[lo]:
                res["greedy_end_d"] = float(dd[lo])
            if res["greedy_act0"] is not None:
                res["disagree"] = bool(np.any(
                    res["label_acts"][0] != res["greedy_act0"]))
            results[s] = res
        if log:
            done_n = sum(1 for r in results[:c0 + ng] if r is not None)
            fins = sum(1 for r in results[:c0 + ng]
                       if r is not None and r["finished"])
            ext = sum(1 for r in results[:c0 + ng]
                      if r is not None and r["extinct"])
            dis = sum(1 for r in results[:c0 + ng]
                      if r is not None and r.get("disagree"))
            log(f"relabel chunk {ci + 1}/{n_chunks}: {done_n}/{n} samples, "
                f"{fins} finished, {ext} extinct, {dis} disagree "
                f"({time.time() - t0:.0f}s)")
    return results


# --------------------------------------------------------------------------
# 3. weights, rows, the merged file
# --------------------------------------------------------------------------
def nearest_distance(points, ref, chunk: int = 128) -> np.ndarray:
    """Euclidean distance from each of ``points`` (n, 3) to the nearest of
    ``ref`` (m, 3), brute force in chunks (m ~ 7,500 spine ticks)."""
    P = np.asarray(points, np.float64).reshape(-1, 3)
    Rf = np.asarray(ref, np.float64).reshape(-1, 3)
    out = np.full(len(P), np.inf)
    if len(Rf) == 0:
        return out
    for i in range(0, len(P), max(1, int(chunk))):
        p = P[i:i + chunk]
        d2 = ((p[:, None, :] - Rf[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def divergence_weights(dist, elite_weight_sum: float, share: float,
                       div_scale: float, div_cap: float) -> np.ndarray:
    """Per-row weights for the relabelled rows.

    Within the set, a row's weight grows with its state's distance to the
    elite line: ``1 + min(dist / div_scale, div_cap)`` (a row ON the line
    counts once, a row div_scale away twice, capped at 1 + div_cap). The
    set as a whole is scaled so it carries ``share`` of the BC loss mass:
    sum(w) = share / (1 - share) * elite_weight_sum. share <= 0 leaves the
    raw factors."""
    dist = np.asarray(dist, np.float64).reshape(-1)
    fac = 1.0 + np.minimum(dist / max(float(div_scale), 1e-9),
                           float(div_cap))
    if len(fac) == 0:
        return fac.astype(np.float32)
    share = float(share)
    if share <= 0.0:
        return fac.astype(np.float32)
    if share >= 1.0:
        raise ValueError("share must be < 1")
    target = share / (1.0 - share) * float(elite_weight_sum)
    return (fac * (target / fac.sum())).astype(np.float32)


def rows_from_results(results, weights, samples: SampleBank | None = None):
    """The relabelled rows in surfgym.bc's column layout, one per labelled
    decision of every labelled sample (the sample's own weight on all of
    its rows). Returns None when nothing was labelled. Also counts, for
    samples that carry a row, whether the window's decision-0 row equals
    the sample's stored row (a faithful restart reads the same slot 12 and
    latch): ``rows['row0_mismatch']``."""
    states, scal, latch, acts, w, sid = [], [], [], [], [], []
    mism = 0
    S = None if samples is None else samples.arrays()
    for r, wt in zip(results, weights):
        if r is None or r.get("label_acts") is None:
            continue
        rows = r["label_rows"]
        if rows is None:
            continue
        if S is not None:
            s = int(r["sample"])
            want = S["scal"][s]
            got = rows[0][:N_SCALAR]
            lat_ok = (rows.shape[1] <= N_SCALAR
                      or abs(float(rows[0][N_SCALAR]) - float(S["latch"][s]))
                      < 1e-6)
            if not (np.allclose(got, want, atol=1e-6) and lat_ok):
                mism += 1
        for j in range(len(r["label_acts"])):
            row = rows[j]
            states.append(r["label_states"][j])
            scal.append(row[:N_SCALAR])
            latch.append(float(row[N_SCALAR]) if len(row) > N_SCALAR else 0.0)
            acts.append(np.asarray(r["label_acts"][j], np.int64))
            w.append(float(wt))
            sid.append(int(r["sample"]))
    if not states:
        return None
    n = len(states)
    return {"states": np.array(states, dtype=STATE_DTYPE).reshape(n),
            "scal": np.array(scal, np.float32).reshape(n, N_SCALAR),
            "latch": np.array(latch, np.float32).reshape(n),
            "actions": np.array(acts, np.int64).reshape(n, 6),
            "weights": np.array(w, np.float32).reshape(n),
            "sample": np.array(sid, np.int64).reshape(n),
            "row0_mismatch": int(mism)}


def summarize_results(results) -> dict:
    """Counts over relabel_windows' results: labelled / finished / extinct
    / unprocessed, the disagreement rate (label != the policy's greedy
    action) and the planner's gain in geodesic d over the policy's own
    greedy continuation where both survived the window."""
    n = len(results)
    done = [r for r in results if r is not None]
    lab = [r for r in done if r.get("label_acts") is not None]
    fin = [r for r in done if r["finished"]]
    ext = [r for r in done if r["extinct"]]
    dis = [r for r in lab if r.get("disagree")]
    gains = [r["greedy_end_d"] - r["end_d"] for r in lab
             if not r["finished"] and r.get("greedy_alive")
             and np.isfinite(r["end_d"]) and np.isfinite(r["greedy_end_d"])]
    gd = [r for r in lab if r.get("greedy_alive") is False]
    out = {"samples": int(n), "processed": int(len(done)),
           "labelled": int(len(lab)), "finished": int(len(fin)),
           "extinct": int(len(ext)), "unprocessed": int(n - len(done)),
           "disagree": int(len(dis)),
           "disagree_rate": (float(len(dis)) / len(lab) if lab else None),
           "greedy_died": int(len(gd)),
           "gain_d_mean": (float(np.mean(gains)) if gains else None),
           "gain_d_median": (float(np.median(gains)) if gains else None),
           "gain_d_max": (float(np.max(gains)) if gains else None),
           "gain_positive": (int(sum(1 for g in gains if g > 0.0))
                             if gains else None),
           "alive_frac_mean": (float(np.mean([r["alive_frac"] for r in done]))
                               if done else None)}
    return out


def merge_bc_datasets(elite_path, rows: dict, out_path, dagger_meta: dict,
                      n_latch: int, obs_reward: bool) -> dict:
    """Elite rows + relabelled rows -> one surfgym.bc file at ``out_path``
    (the elite file's meta, plus ``dagger`` = dagger_meta and the row
    counts). The relabelled rows carry line_id DAGGER_LINE_ID. The elite
    file must have been built for the same checkpoint layout (its
    obs_reward / n_latch must match)."""
    z = np.load(elite_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if bool(meta.get("obs_reward")) != bool(obs_reward) \
            or int(meta.get("n_latch", 0)) != int(n_latch):
        raise SystemExit(f"{elite_path}: built for obs_reward="
                         f"{meta.get('obs_reward')!r} n_latch="
                         f"{meta.get('n_latch')!r}, the relabelled rows for "
                         f"obs_reward={obs_reward!r} n_latch={n_latch}")
    e_states = np.asarray(z["states"], STATE_DTYPE)
    e_scal = np.asarray(z["scal"], np.float32)
    e_latch = np.asarray(z["latch"], np.float32)
    e_act = np.asarray(z["actions"], np.int64)
    e_w = np.asarray(z["weights"], np.float32)
    e_id = np.asarray(z["line_id"], np.int32)
    n_e = int(len(e_states))
    if rows is None:
        n_d = 0
        states, scal, latch, act, w, lid = (e_states, e_scal, e_latch, e_act,
                                            e_w, e_id)
    else:
        n_d = int(len(rows["states"]))
        states = np.concatenate([e_states, rows["states"]])
        scal = np.concatenate([e_scal, rows["scal"]])
        latch = np.concatenate([e_latch, rows["latch"]])
        act = np.concatenate([e_act, rows["actions"]])
        w = np.concatenate([e_w, rows["weights"]])
        lid = np.concatenate([e_id, np.full(n_d, DAGGER_LINE_ID, np.int32)])
    meta = dict(meta)
    meta["elite_file"] = str(elite_path)
    meta["rows_elite"] = n_e
    meta["rows_dagger"] = n_d
    meta["rows"] = n_e + n_d
    meta["weight_elite"] = float(e_w.sum())
    meta["weight_dagger"] = (0.0 if rows is None
                             else float(rows["weights"].sum()))
    meta["dagger"] = dict(dagger_meta)
    save_bc_dataset(out_path, states, scal, latch, act, w, lid, meta)
    return meta


def load_merged_meta(path) -> dict:
    return load_bc_meta(path)


def check_actions(actions) -> None:
    hi = np.asarray(ACTION_NVEC, np.int64)
    a = np.asarray(actions, np.int64).reshape(-1, 6)
    if (a < 0).any() or (a >= hi[None, :]).any():
        raise ValueError("label actions out of range for ACTION_NVEC")


def summary_path_for(out_path) -> Path:
    p = Path(out_path)
    return p.with_name(p.stem + "_summary.json")
