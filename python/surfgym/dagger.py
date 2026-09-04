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

import time
from pathlib import Path

import numpy as np

from .bc import (NACT, NPAD, N_SCALAR, count_probs,
                 gumbel_improved_probs, load_bc_arrays, load_bc_meta,
                 onehot_probs, save_bc_dataset)
from .core import ACTION_NVEC, STATE_DTYPE
from .tick import REFERENCE_TICK_MS, TickClock

__all__ = ["DAGGER_LINE_ID", "SRC_GREEDY", "SRC_STOCH", "SRC_SPINE",
           "SRC_NAMES", "core_clock", "check_core_tick", "on_grid",
           "decision_grid", "even_subset", "SampleBank",
           "collect_rollout_samples",
           "relabel_windows", "nearest_distance", "divergence_weights",
           "rows_from_results", "summarize_results", "merge_bc_datasets",
           "population_probs", "LABEL_TARGETS"]

#: --label-target: what the window's label distribution is built from.
#: "count" = the surviving population's own first decisions (ExIt's TPT with
#: truncation-selection survivors in place of visit counts); "gumbel" =
#: Danihelka 2022's `softmax(logits + sigma(completedQ))` over the same
#: population, which needs the policy's root logits and a normalised Q.
LABEL_TARGETS = ("count", "gumbel")

DAGGER_LINE_ID = -1        # line_id of a relabelled row (elite lines are >= 0)
SRC_GREEDY, SRC_STOCH, SRC_SPINE = 0, 1, 2
SRC_NAMES = ("greedy", "stoch", "spine")


def core_clock(core) -> TickClock:
    """``core``'s physics tick as a :class:`surfgym.tick.TickClock` - the
    one conversion between the SECONDS a caller asks for (tools/
    expert_dagger.py's --every, --rollout-secs, --spine-secs, --window)
    and the TICKS every loop in this module counts.

    This replaces a ``TICKS_PER_S = 100`` constant, which holds only at the
    10 ms reference tick: on a ``--tick-ms 7.63`` core (130.4 Hz) a "3 s"
    planner window was 300 ticks = 2.30 s of physics and a "0.5 s" sampling
    grid was 0.38 s. Read the tick from the core that will actually run the
    ticks, never from a literal. A core that exposes no tick (the stub
    cores in tests/python/test_expert_dagger.py) is the reference, where
    every conversion is the legacy ``* 100`` arithmetic bit for bit.
    """
    return TickClock(float(getattr(core, "tick_ms", REFERENCE_TICK_MS)))


def check_core_tick(core, tick: TickClock, what: str = "core",
                    hint: str = "") -> TickClock:
    """Refuse ``core`` unless it runs the tick ``tick`` converted at.

    Every seconds flag of the relabel phase (--every, --window,
    --rollout-secs, --spine-secs) and the --obs-reward slot-12 mirror are
    converted ONCE, at the checkpoint's clock; a core that then steps at a
    different millisecond makes all of them wrong at once, and silently -
    a 3 s window becomes 2.30 s of physics and every label is planned in
    the wrong dynamics. Called on EVERY core the phase opens, not only the
    first: the 1-env probe core and the 2,048-env search core are separate
    build_sim calls. Returns the core's own clock.
    """
    got = core_clock(core)
    if abs(got.ms - tick.ms) > 1e-9:
        raise SystemExit(
            f"{what}: expected the checkpoint's tick "
            f"{tick.requested_ms:g} ms ({tick.ms:.4f} ms realised, pattern "
            f"{list(tick.pattern)}) but this core runs {got.ms:.4f} ms "
            f"(pattern {list(got.pattern)})."
            + (" " + hint if hint else ""))
    return got


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


def population_probs(hist, alive, winner_local: int, n_label: int,
                     logits=None, q=None, value: float = 0.0,
                     target: str = "count", c_visit: float = 50.0,
                     c_scale: float = 0.1) -> np.ndarray:
    """The window's per-decision policy target, ``(n_label, 6, NPAD)``.

    ``hist`` is the group's action history ``(L, copies, 6)`` AS IT STOOD
    when the group's search ended and ``alive`` ``(copies,)`` which of its
    copies were still in the population then. `_clone` copies a donor's
    whole prefix onto a loser, so the copies whose ``hist[:jd]`` equals the
    winner's are exactly the lineages that stood at the winner's decision-jd
    state, and the share of them taking each action there is that state's
    first-decision distribution - the survivor count standing in for
    AlphaZero's visit count. Decision 0 is the DAgger label proper and sees
    the whole live population; deeper label decisions narrow to the winner's
    own descendants and end at its one-hot, which is the old label exactly.

    The greedy envs are counted like any other copy: they are the policy's
    own continuation, kept precisely so a window can never lose to it, and
    leaving them in puts a floor of prior mass on the policy's own action -
    the trust region a KL toward a point mass does not have.

    ``target='gumbel'`` replaces the count at DECISION 0 with
    ``softmax(logits + sigma(completedQ))`` (surfgym.bc.gumbel_improved_probs)
    using ``q`` (per head, per action, normalised to [0, 1]) and ``value``,
    the root value in the same units. Per HEAD, because the policy is a
    factored categorical over 3,780 joint actions and Sampled MuZero's own
    recommendation for that space is the factored form; a joint-action Q
    would need the joint counts we do not have. Deeper decisions stay counts.
    """
    H = np.asarray(hist)
    al = np.asarray(alive, bool).reshape(-1)
    n_label = max(1, int(n_label))
    out = np.zeros((n_label, NACT, NPAD), np.float32)
    win = H[:, int(winner_local), :]
    match = al.copy()
    for jd in range(n_label):
        idx = np.flatnonzero(match)
        if len(idx) == 0:              # the winner itself already died
            out[jd] = onehot_probs(win[jd][None, :])[0]
        else:
            out[jd] = count_probs(H[jd, idx, :])
        if jd == 0 and target == "gumbel" and logits is not None:
            n_a = np.zeros((NACT, NPAD), np.float64)
            if len(idx):
                for h in range(NACT):
                    n_a[h] = np.bincount(H[0, idx, h], minlength=NPAD)
            out[jd] = gumbel_improved_probs(logits, q, n_a, value,
                                            c_visit=c_visit,
                                            c_scale=c_scale)
        match &= al & np.all(H[jd] == win[jd][None, :], axis=1)
    return out


def relabel_windows(core, make_decider, make_feeds, field, score_fn,
                    samples: SampleBank, k: int, copies: int,
                    window_decisions: int, resample_every: int = 25,
                    elite_frac: float = 0.25, n_greedy: int = 1,
                    label_decisions: int = 1, seed: int = 0,
                    budget_s: float = 0.0, d_floor: float = 0.0,
                    label_target: str = "count", c_visit: float = 50.0,
                    c_scale: float = 0.1, log=None) -> list:
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
    wall clock is spent; results of unprocessed samples are None.

    P2: each result also carries ``label_probs`` ((L, 6, NPAD)), the
    SURVIVING POPULATION's distribution over each labelled decision
    (:func:`population_probs`) rather than only the winner's index, plus
    ``label_alive`` (how many copies that distribution was counted over).
    The population is snapshotted the moment the group's search ends - at
    the first crossing for a finished group, at the window's end otherwise -
    so a finisher's own copy is still in it (the goal tick clears `valid`).
    ``label_target='gumbel'`` needs a decider exposing ``last_logits``
    ((N, 6, NPAD) padded root logits); without one it falls back to the
    counts and says so once."""
    N = int(core.num_envs)
    copies = int(copies)
    if copies < 1 or copies > N:
        raise ValueError(f"copies {copies} must be in [1, {N}]")
    G = N // copies
    k = max(1, int(k))
    H = max(1, int(window_decisions))
    R = int(resample_every)
    L = max(1, min(int(label_decisions), H))
    if str(label_target) not in LABEL_TARGETS:
        raise ValueError(f"label_target must be one of {LABEL_TARGETS}, "
                         f"got {label_target!r}")
    want_gumbel = str(label_target) == "gumbel"
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
        root_logits = None
        fin: list = [None] * ng
        pop: list = [None] * ng
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
                    lg = getattr(decider, "last_logits", None)
                    if lg is not None:
                        root_logits = np.asarray(lg, np.float64).reshape(
                            N, NACT, NPAD).copy()
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
                        # the population AS IT STANDS at the crossing:
                        # `valid` has not been cleared for this tick yet, so
                        # the finisher is still in it. One tick later it is
                        # not, and the group stops searching.
                        lo_g = g * copies
                        pop[g] = (hist[:L, lo_g:lo_g + copies].copy(),
                                  (valid & active)[lo_g:lo_g + copies].copy())
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
                   "label_rows": None, "label_probs": None,
                   "label_alive": 0}
            if fin[g] is not None:
                ft, i, acts, sts, rows = fin[g]
                n_lab = max(1, min(L, (ft - 1) // k + 1))
                res.update(finished=True, end_tick=int(ft), winner=int(i),
                           label_acts=acts[:n_lab].astype(np.int64),
                           label_states=sts[:n_lab],
                           label_rows=(None if rows is None
                                       else rows[:n_lab]))
                ph, pa = pop[g]
                res["label_probs"] = population_probs(
                    ph, pa, int(i) - lo, n_lab, target="count")
                res["label_alive"] = int(pa.sum())
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
                lg = q = None
                v0 = 0.0
                if want_gumbel and root_logits is not None:
                    # Q per (head, first action), normalised INSIDE the group
                    # to [0, 1] - our scores are a geodesic distance, V(s) or
                    # an arc coordinate on unrelated scales, so a Gumbel
                    # c_scale imported from mctx means nothing until they are
                    sg = np.where(v, sc[lo:hi], np.nan)
                    smin = np.nanmin(sg)
                    span = float(np.nanmax(sg) - smin)
                    sn = ((sg - smin) / span if span > 0
                          else np.zeros_like(sg))
                    q = np.zeros((NACT, NPAD), np.float64)
                    h0 = hist[0, lo:hi]
                    for hh in range(NACT):
                        for ii in np.flatnonzero(v):
                            aa = int(h0[ii, hh])
                            q[hh, aa] = max(q[hh, aa], float(sn[ii]))
                    lg = root_logits[best]
                    v0 = float(sn[0]) if v[0] else float(np.nanmean(sn))
                res["label_probs"] = population_probs(
                    hist[:L, lo:hi], v, best - lo, L, logits=lg, q=q,
                    value=v0,
                    target=("gumbel" if lg is not None else "count"),
                    c_visit=c_visit, c_scale=c_scale)
                res["label_alive"] = int(v.sum())
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
    probs = []
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
        lp = r.get("label_probs")
        for j in range(len(r["label_acts"])):
            row = rows[j]
            states.append(r["label_states"][j])
            scal.append(row[:N_SCALAR])
            latch.append(float(row[N_SCALAR]) if len(row) > N_SCALAR else 0.0)
            acts.append(np.asarray(r["label_acts"][j], np.int64))
            w.append(float(wt))
            sid.append(int(r["sample"]))
            # P2: the window population's distribution over this decision;
            # a result from before the target existed (or a relabel that
            # produced none) falls back to the label's own one-hot, which is
            # what the row always meant
            probs.append(None if lp is None or j >= len(lp) else lp[j])
    if not states:
        return None
    n = len(states)
    act_arr = np.array(acts, np.int64).reshape(n, 6)
    oh = onehot_probs(act_arr)
    pr = np.stack([oh[i] if probs[i] is None
                   else np.asarray(probs[i], np.float32).reshape(NACT, NPAD)
                   for i in range(n)])
    return {"states": np.array(states, dtype=STATE_DTYPE).reshape(n),
            "scal": np.array(scal, np.float32).reshape(n, N_SCALAR),
            "latch": np.array(latch, np.float32).reshape(n),
            "actions": act_arr,
            "weights": np.array(w, np.float32).reshape(n),
            "sample": np.array(sid, np.int64).reshape(n),
            "probs": pr.astype(np.float32),
            # the relabel windows plan 3 s ahead and stop; nothing here
            # reaches a terminal from the sampled state, so no row carries a
            # complete return-to-go. P3 lives on the elite lines, which do.
            "zret": np.zeros(n, np.float32),
            "zmask": np.zeros(n, np.float32),
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
    # P2 read-outs: how much of the label is actually a DISTRIBUTION. top1 is
    # the winning action's own share of the surviving population (1.0 = the
    # old one-hot), ent the per-head entropy of the decision-0 target.
    tops, ents, alive = [], [], []
    for r in lab:
        lp = r.get("label_probs")
        if lp is None or not len(lp):
            continue
        p0 = np.asarray(lp[0], np.float64)
        tops.append(float(p0.max(-1).mean()))
        ents.append(float(-(p0 * np.log(np.clip(p0, 1e-12, None))).sum(-1)
                          .mean()))
        alive.append(int(r.get("label_alive") or 0))
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
                               if done else None),
           "target_top1_mean": (float(np.mean(tops)) if tops else None),
           "target_entropy_mean": (float(np.mean(ents)) if ents else None),
           "target_copies_mean": (float(np.mean(alive)) if alive else None)}
    return out


def merge_bc_datasets(elite_path, rows: dict, out_path, dagger_meta: dict,
                      n_latch: int, obs_reward: bool) -> dict:
    """Elite rows + relabelled rows -> one surfgym.bc file at ``out_path``
    (the elite file's meta, plus ``dagger`` = dagger_meta and the row
    counts). The relabelled rows carry line_id DAGGER_LINE_ID. The elite
    file must have been built for the same checkpoint layout (its
    obs_reward / n_latch must match)."""
    z = load_bc_arrays(elite_path)
    meta = z["meta"]
    if bool(meta.get("obs_reward")) != bool(obs_reward) \
            or int(meta.get("n_latch", 0)) != int(n_latch):
        raise SystemExit(f"{elite_path}: built for obs_reward="
                         f"{meta.get('obs_reward')!r} n_latch="
                         f"{meta.get('n_latch')!r}, the relabelled rows for "
                         f"obs_reward={obs_reward!r} n_latch={n_latch}")
    e_states, e_scal, e_latch = z["states"], z["scal"], z["latch"]
    e_act, e_w, e_id = z["actions"], z["weights"], z["line_id"]
    n_e = int(len(e_states))
    # a version-1 elite file has neither target; load_bc_arrays already
    # filled the version-1 meaning (one-hot probs, no value rows), so the
    # merge never has to branch on the version
    e_pr, e_z, e_zm = z["probs"], z["zret"], z["zmask"]
    if rows is None:
        n_d = 0
        states, scal, latch, act, w, lid = (e_states, e_scal, e_latch, e_act,
                                            e_w, e_id)
        pr, zr, zm = e_pr, e_z, e_zm
    else:
        n_d = int(len(rows["states"]))
        states = np.concatenate([e_states, rows["states"]])
        scal = np.concatenate([e_scal, rows["scal"]])
        latch = np.concatenate([e_latch, rows["latch"]])
        act = np.concatenate([e_act, rows["actions"]])
        w = np.concatenate([e_w, rows["weights"]])
        lid = np.concatenate([e_id, np.full(n_d, DAGGER_LINE_ID, np.int32)])
        pr = np.concatenate([e_pr, rows["probs"]])
        zr = np.concatenate([e_z, rows["zret"]])
        zm = np.concatenate([e_zm, rows["zmask"]])
    meta = dict(meta)
    meta["elite_file"] = str(elite_path)
    meta["elite_version"] = int(z["version"])
    meta["rows_elite"] = n_e
    meta["rows_dagger"] = n_d
    meta["rows"] = n_e + n_d
    meta["weight_elite"] = float(e_w.sum())
    meta["weight_dagger"] = (0.0 if rows is None
                             else float(rows["weights"].sum()))
    meta["dagger"] = dict(dagger_meta)
    # keep the merged file at the elite file's version unless the relabel
    # actually brought a distribution: a v1 elite file + one-hot dagger rows
    # merges to a v1 file, byte-identical to what this always wrote
    d_onehot = (rows is None or bool(np.array_equal(
        rows["probs"], onehot_probs(rows["actions"]))))
    if not z["has_probs"] and not z["has_value"] and d_onehot:
        save_bc_dataset(out_path, states, scal, latch, act, w, lid, meta)
    else:
        save_bc_dataset(out_path, states, scal, latch, act, w, lid, meta,
                        probs=pr, zret=zr, zmask=zm)
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
