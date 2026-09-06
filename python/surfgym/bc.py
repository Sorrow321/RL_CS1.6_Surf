"""bc.py - behaviour-cloning data for expert iteration (planner -> policy).

AlphaZero-style expert iteration on the surf trainer: the current policy
proposes, the beam planner (tools/beam_tas.py) improves the line by search
in the real simulator, and the policy is trained to imitate the planner's
(state, action) pairs on top of its ordinary PPO objective. This module is
the data contract between the planner side (tools/plan_to_bc.py writes the
file) and the trainer side (train_fast.py --bc-file consumes it), plus the
two eval-side feeds both of them share with tools/beam_tas.py.

WHAT A ROW IS. One row per policy DECISION (act_every ticks) along a
planner line that finished the map:

* ``states``   STATE_DTYPE - the full physics state the decision was taken
               in (what a spawn pool copies; what the lidar is rendered
               from);
* ``scal``     (R, 15) f32 - the 15 core scalars EXACTLY as the C core
               emitted them at that decision (surf_step's write_obs), with
               slot 12 already replaced by the --obs-reward feed value when
               the checkpoint uses that flag;
* ``latch``    (R,) f32 - the --race-latch observation column (0/1);
* ``actions``  (R, 6) int - the six head indices the planner committed;
* ``weights``  (R,) f32 - per-row loss weight (1.0 = plain).

VERSION 2 adds three OPTIONAL columns; a file without them is a version-1
file and still loads (``BCDataset`` synthesises the version-1 meaning of
each), and ``save_bc_dataset`` writes a byte-identical version-1 file when
none is supplied:

* ``probs``    (R, 6, 15) f32 - the SEARCH-DERIVED policy target: per head,
               a distribution over that head's action at this decision,
               padded to ``NPAD`` slots with zeros (the layout
               ``HeadPacker.pad`` produces). Expert Iteration's TPT target
               (Anthony et al. 2017) in place of CAT: the paper measured
               50 +/- 13 Elo between the two at INDISTINGUISHABLE top-1
               accuracy, so the argmax our loop clones is the weak end of
               the loop, not the search. Rows where only the winner exists
               carry the one-hot of ``actions``, which is exactly the old
               behaviour - the distribution target is a superset;
* ``zret``     (R,) f32 - AlphaZero's ``z``: the DISCOUNTED RETURN-TO-GO of
               the planner line's own rewards from this decision, at
               DECISION granularity under the trainer's own
               ``gamma**act_every`` (:func:`decision_gamma`), computed by
               replaying the line through the trainer's own RaceReward
               (:func:`make_line_reward`). Bootstrap-free on a line that
               reaches a terminal;
* ``zmask``    (R,) f32 - 1.0 where ``zret`` is that complete, terminal-
               backed return and 0.0 where the line was still ALIVE when
               its action table ran out (the tail is missing and the row
               must not enter the value loss).

WHY THE SCALARS ARE STORED AND NOT RE-DERIVED FROM THE STATE. The obvious
design - keep a spare SurfCore, ``set_state`` each row, read its obs - does
not reproduce training's row: ``write_obs`` (src/env.c) reads two per-env
values that live OUTSIDE SurfState, ``last_yaw_delta`` / ``last_pitch_delta``
(obs slots 10 and 11, the previous action's view rates), and ``set_state``
does not carry them. The replay that builds this file steps the very same
core code along the very same action sequence, so the scalars it records at
each decision ARE the scalars the training rollout would have produced for
that decision. 15 floats per row cost nothing; the depth image is the part
that must not be stored, and it is not: the trainer renders it per
minibatch from the stored state with its own lidar, which is a pure function
of (origin, yaw, pitch, ducked) - exactly what fill_vision renders from the
live states_view.

The row the trainer assembles is therefore ``[scal | latch | render(state)]``
= ``[15 core | N_LATCH | image]``, the same layout as ``static_obs`` in the
rollout (route fans and frame stacks are refused: the planner cannot clone
their per-env state either).

--view-continuous (surfgym/view.py) adds three OPTIONAL columns, written
only by a continuous planner/relabel and ignored by a discrete trainer:

* ``view``      (R, 2) f32 - the EXECUTED physical view per decision
                (yaw command K, pitch deg/tick), what the core applied;
* ``view_zmu``  (R, 2) f32 - the mean of the elite copies' pre-tanh z at
* ``view_zsd``  (R, 2) f32 - this decision and its std (moment matching:
                the Gaussian target ``--bc-target dist`` fits); a row only
                one line survived at has zsd 0 and zmu = the row's own z.

A continuous trainer reading a file WITHOUT them derives ``view`` from the
bins exactly as the core would apply them (``surfgym.view.bin_to_view``)
and the moments from ``probs`` (``bin_view_moments``), and says so.

Under ``--view-absolute`` (meta key ``view_absolute`` = "velocity" /
"world") the ``view`` column holds the executed TARGETS (yaw offset deg in
the velocity frame / yaw target deg, pitch target deg), ``view_zmu`` /
``view_zsd`` are (R, n_z) - 3 columns in world mode - and the point target
is ``surfgym.view.z_from_view_abs``. A file of one mode is refused by a
trainer of another (a delta row read as a target, or the other way round,
is a different action), and a discrete file cannot be used under an
absolute mode at all (its bins are per-tick deltas, not targets).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import ACTION_NVEC, STATE_DTYPE
from .tick import REFERENCE_TICK_MS, TickClock
from .view import (bin_to_view, bin_view_moments, n_z, view_desc,
                   z_from_view, z_from_view_any)

__all__ = ["BC_VERSION", "BC_VERSIONS", "NACT", "NPAD", "REWARD_SLOT",
           "make_eval_feeds", "replay_line",
           "save_bc_dataset", "load_bc_meta", "load_bc_arrays", "BCDataset",
           "rank_lineages", "contact_rows", "last_contact_cut",
           "subsample_by_path", "onehot_probs", "check_probs",
           "survivor_probs", "count_probs", "gumbel_improved_probs",
           "decision_gamma", "returns_to_go", "make_line_reward"]

#: 1 = actions only (the CAT target); 2 adds ``probs`` / ``zret`` / ``zmask``.
#: Both load. A writer that is handed none of the three writes 1, so an
#: untreated round's file is byte-identical to the one it always wrote.
BC_VERSION = 2
BC_VERSIONS = (1, 2)
REWARD_SLOT = 12          # train_fast.py REWARD_SLOT: --obs-reward's column
N_SCALAR = 15
NACT = len(ACTION_NVEC)               # 6 factored heads
NPAD = int(max(ACTION_NVEC))          # 15 - train_fast.NPAD, the padded width


# --------------------------------------------------------------------------
# the eval-side feeds (mirrors of train_fast's _make_eval_*_feed)
# --------------------------------------------------------------------------
def _latch_feed(field, d_latch: float):
    """The --race-latch flag for an eval core, per decision (train_fast
    _make_eval_latch_feed, verbatim): sticky once d <= d_latch, cleared when
    the core's per-env tick counter goes backwards (an episode start)."""
    st = {"f": None, "tick": None}

    def feed(core):
        sv = core.states_view
        d = field.sample(sv["origin"]).astype(np.float64)
        tick = np.asarray(sv["tick"], np.int64).copy()
        f, pt = st["f"], st["tick"]
        if f is None or len(f) != len(d) or pt is None:
            f = np.zeros(len(d), bool)
        else:
            f = f & ~(tick <= pt)
        f |= d <= d_latch
        st["f"], st["tick"] = f, tick
        return f.astype(np.float32)

    feed.state = st
    feed.d_latch = float(d_latch)
    return feed


def _reward_feed(field, scale, time_pen, k, d_floor, latch_feed, ng, ng_g,
                 ng_d0):
    """The --obs-reward slot-12 mirror (train_fast _make_eval_reward_feed,
    verbatim): per-decision geodesic progress minus the time cost, the
    latch flag of the PREVIOUS decision zeroing the shaping, squashed."""
    st = {"d": None}

    def feed(core):
        d = field.sample(core.states_view["origin"]).astype(np.float64)
        if d_floor > 0.0:
            d = np.maximum(d, d_floor)
        prev = st["d"]
        st["d"] = d
        if prev is None or len(prev) != len(d):
            return np.zeros(len(d), np.float32)
        delta = np.clip(prev - d, -100.0 * k, 100.0 * k)
        if latch_feed is not None:
            was = latch_feed.state["f"]
            if was is not None and len(was) == len(delta):
                delta = np.where(was, 0.0, delta)
        r = delta * scale - time_pen * k
        if ng:
            r = r - (1.0 - ng_g) * (ng_d0 - d) * scale
        return np.tanh(r / 0.1).astype(np.float32)

    feed.state = st
    feed.time_pen, feed.ng_gamma = float(time_pen), float(ng_g)
    return feed


def _d_latch_of(cfg: dict, d0: float) -> float:
    """The --race-latch threshold in map units (train_fast's ``_s.d_latch``):
    the flag itself, or ``--race-latch-frac`` times the start geodesic. One
    implementation, because the eval mirror and the reward mirror must agree
    on which regime a row is in or the two disagree about its reward."""
    d_latch = float(cfg.get("race_latch") or 0.0)
    frac = float(cfg.get("race_latch_frac") or 0.0)
    if frac > 0.0:
        d_latch = frac * float(d0)
    return d_latch


def make_eval_feeds(cfg: dict, field, d0: float, k: int,
                    tick_ms: float = 10.0):
    """``(reward_slot, reward_feed, latch_fn)`` for a race checkpoint's
    config: the two side-channel observation columns an eval core does not
    produce, computed exactly as train_fast's eval wrappers compute them.
    reward_slot is -1 (and reward_feed None) without --obs-reward; latch_fn
    is None without --race-latch. ``d0`` is the trainer's start distance
    (mean field over the RAW map spawns).

    ``tick_ms`` is the physics tick the core RUNS at (beam_tas / plan_to_bc
    --tick-ms). The trainer feeds its mirror RaceReward's time_pen, which it
    rescales by tick/10 so the penalty per SECOND is unchanged, and the
    --race-ng gamma per tick likewise; handing the 10 ms-referenced values
    to a core at another tick puts a constant offset into the one column
    the policy reads its own reward from (record_ckpt's mirror, the tick-ms
    review). Identity at 10 ms: TickClock.per_tick / gamma return the value
    itself, so the default path is byte-identical."""
    if cfg.get("race_arc"):
        raise SystemExit("make_eval_feeds: --race-arc checkpoints are not "
                         "supported (the slot-12 mirror would be the "
                         "geodesic term, not the arc term)")
    d_latch = _d_latch_of(cfg, d0)
    latch_fn = _latch_feed(field, d_latch) if d_latch > 0.0 else None
    if not cfg.get("obs_reward"):
        return -1, None, latch_fn
    scale = 100.0 / max(float(d0), 1.0) * float(cfg.get("race_shaping") or 1.0)
    tick = TickClock(float(tick_ms))
    tp = tick.per_tick(float(cfg.get("time_pen") or 0.005))
    d_floor = float(cfg.get("race_dfloor") or 0.0)
    ng = int(cfg.get("race_ng") or 0)
    ng_g = tick.gamma(float(cfg.get("gamma", 0.9995))) ** int(k)
    return REWARD_SLOT, _reward_feed(field, scale, tp, int(k), d_floor,
                                     latch_fn, ng, ng_g, float(d0)), latch_fn


# --------------------------------------------------------------------------
# replaying a planner line into rows
# --------------------------------------------------------------------------
def replay_line(core, spawn_state, obs_start, acts, k: int, reward_feed=None,
                latch_fn=None, reward_slot: int = REWARD_SLOT,
                max_ticks: int = 0, keep_final: bool = False,
                reward_fn=None, rewards_out=None, info_out=None,
                view=None, views_out=None):
    """Open-loop replay of one planner line on a 1-env ``core``.

    ``acts`` is the (D, 6) per-decision action table beam_tas committed,
    ``spawn_state`` the STATE_DTYPE row it started from and ``obs_start``
    that state's 15 core scalars (the reset obs). The core must be armed
    (goal box, teleport fail) and reset once already; the feeds must be
    fresh (never called) so their first decision reads 0 / the spawn flag,
    as the planner's own first decision did.

    Returns ``(rows, tick_states, finished, ticks)``: rows is a list of
    ``(state, scal15, latch, action)`` per decision taken while alive,
    tick_states the STATE_DTYPE pre-step state of every tick (the demo
    spine), finished whether the goal box was crossed, ticks the tick
    count at the end.

    ``keep_final``: a line that runs its whole action table WITHOUT ending
    (no finish, no death, no cap - a progress-objective lineage still alive
    when the search stopped) also appends the state AFTER its last action,
    so the spine ends where the planner's line ended and the arc the
    replay reaches is the arc the search credited (which read the post-step
    position every tick). Off by default: byte-identical to before.

    ``reward_fn`` (a :class:`surfgym.rewards.RaceReward` from
    :func:`make_line_reward`) is run alongside the replay at the trainer's
    own cadence and its per-DECISION sum appended to ``rewards_out``, which
    is P3's raw material: ``returns_to_go(rewards_out, decision_gamma(...))``
    is the line's exact discounted return-to-go. ``on_reset`` is called at
    the spawn state before the first step, so the shaping's first delta and
    the latch's arming match the rollout's. Both default None and the
    default path is the loop that shipped, call for call.

    ``info_out`` (a dict) receives HOW the replay stopped: ``terminal`` True
    when the core ended the episode (a goal crossing, a death, a truncation,
    or ``max_ticks``), False when the action table simply ran out with the
    line still alive. P3's ``zmask`` is exactly that flag - only a terminal
    makes the discounted sum from a row that row's COMPLETE return.

    ``view`` (D, 2) float32 is a --view-continuous line's per-decision view
    command, stepped alongside ``acts`` (``core.step(a, view=...)``); each
    decision's row is appended to ``views_out`` next to ``rows``. None is
    the discrete replay, call for call.
    """
    acts = np.asarray(acts)
    view = None if view is None else np.asarray(view, np.float32).reshape(-1, 2)
    core.set_state(0, spawn_state)
    obs = np.array(obs_start, np.float32).reshape(1, -1).copy()
    rows, tick_states = [], []
    t, finished, ended = 0, False, False
    cap = int(max_ticks) if max_ticks else int(len(acts)) * int(k)
    if reward_fn is not None:
        # the state is set, nothing stepped: this is the trainer's on_reset,
        # so the shaping's first delta is measured from the SPAWN and the
        # latch arms on a spawn already inside the shell exactly as the
        # rollout's does
        reward_fn.on_reset(core)
        rew_every = max(1, int(getattr(reward_fn, "every", 1)))
    for d in range(len(acts)):
        a = np.ascontiguousarray(acts[d].reshape(1, 6), dtype=np.int32)
        v = (None if view is None
             else np.ascontiguousarray(view[d].reshape(1, 2), dtype=np.float32))
        # the wrapper's _obs order: the reward feed first (it reads the
        # latch flag as it stood one decision AGO), then the latch
        fv = None if reward_feed is None else float(reward_feed(core)[0])
        lf = 0.0 if latch_fn is None else float(latch_fn(core)[0])
        st = core.get_states()[0].copy()
        scal = obs[0, :N_SCALAR].copy()
        if fv is not None:
            scal[reward_slot] = fv
        rows.append((st, scal, lf, acts[d].astype(np.int64).copy()))
        if views_out is not None:
            views_out.append(None if v is None else v[0].copy())
        r_dec = 0.0
        for _j in range(int(k)):
            tick_states.append(core.get_states()[0].copy())
            prev = obs
            if v is None:
                obs, _rew, done, trunc, _term = core.step(a)
            else:
                obs, _rew, done, trunc, _term = core.step(a, view=v)
            t += 1
            if int(core.goal_hits[0]):
                finished = True
            if reward_fn is not None and (_j + 1) % rew_every == 0:
                # the rollout's own cadence: once per TICK, or once per
                # decision under --reward-per-decision (where the potential
                # telescopes across the K ticks and one call is exact).
                # r_acc in train_fast sums these UNDISCOUNTED inside the
                # decision, which is what makes the per-decision sum the
                # quantity its GAE discounts at gamma**K.
                r_dec += float(reward_fn(prev, obs, obs, _rew, done, trunc,
                                         core)[0])
            if done[0] or trunc[0] or finished or t >= cap:
                ended = True
                break
        if rewards_out is not None:
            rewards_out.append(r_dec)
        if ended:
            break
    if keep_final and not ended and len(acts):
        tick_states.append(core.get_states()[0].copy())
    if info_out is not None:
        info_out["terminal"] = bool(ended)
        info_out["finished"] = bool(finished)
        info_out["decisions"] = int(len(rows))
        info_out["ticks"] = int(t)
    return rows, tick_states, finished, t


# --------------------------------------------------------------------------
# progress-objective lineages: ranking, the fall trim, the spine spacing
# --------------------------------------------------------------------------
def rank_lineages(cands, k: int = 0):
    """Order planner lineages for distillation, distinct, best first.

    A candidate is a dict with ``acts`` ((D, 6) int, the per-decision
    action table), ``finish_tick`` (0 = did not finish), ``best_arc`` (the
    furthest order-only arc it reached, map units), ``arc_tick`` (the tick
    it reached it at) and ``end_tick`` (its last live tick). The order is
    the one both objectives agree on:

    * finishers first, earliest finish tick first (in a lockstep search
      the first crossing IS the fastest run, and under ``--objective auto``
      a finisher outranks every non-finisher whatever its arc);
    * then non-finishers by best arc DESCENDING, ties by the tick the arc
      was reached at ASCENDING (same distance, sooner = faster), then by
      the lineage's end tick (a line that survived longer is the better
      demo of two that got equally far equally fast).

    Duplicates (byte-identical action tables: clones that died on the same
    tick before diverging) are dropped; ``k`` > 0 keeps the first k.
    """
    def key(c):
        ft = int(c.get("finish_tick") or 0)
        if ft > 0:
            return (0, ft, 0.0, 0)
        return (1, -float(c.get("best_arc") or 0.0),
                int(c.get("arc_tick") or 0), int(c.get("end_tick") or 0))

    seen, out = set(), []
    for c in sorted(cands, key=key):
        b = np.ascontiguousarray(np.asarray(c["acts"], np.int8)).tobytes()
        if c.get("view") is not None:
            # a --view-continuous line is its bins AND its float view
            b += np.ascontiguousarray(np.asarray(c["view"], np.float32)).tobytes()
        if b in seen:
            continue
        seen.add(b)
        out.append(c)
        if k > 0 and len(out) >= k:
            break
    return out


def contact_rows(states):
    """Time-ordered STATE_DTYPE states -> the ``[tick, x, y, z, vx, vy, vz,
    yaw]`` rows tools/pick_selfline.py's ``contact_cut`` reads (the
    recorder's 8 columns, built from the real states instead of a lossy
    trajectory line)."""
    s = np.asarray(states, STATE_DTYPE)
    n = len(s)
    t = np.arange(n, dtype=np.float64)
    return np.column_stack([t, np.asarray(s["origin"], np.float64),
                            np.asarray(s["velocity"], np.float64),
                            np.asarray(s["yaw"], np.float64)]).reshape(n, 8)


def last_contact_cut(states, gravity_step=None, tol: float = 1.0):
    """The last tick at which the map pushed back -> ``(cut, g)``.

    tools/pick_selfline.py's rule (CLAUDE.md: "vertical acceleration
    departing from the gravity step"), restated over STATE_DTYPE states so
    the planner side can trim a non-finishing lineage where its track
    physically ended instead of teaching the fall. Between surface contacts
    a Source player is a projectile - ``vz`` drops by exactly the gravity
    step every tick - so a tick whose ``diff(vz)`` departs from that step
    by more than ``tol`` is a tick where geometry acted; the cut keeps that
    tick. ``gravity_step=None`` recovers the step as the median of
    ``diff(vz)`` exactly as pick_selfline does (right whenever most of the
    line is airborne); a caller that knows the engine constant (``-sv_gravity
    * tick_s``) passes it, which is what makes the rule safe on a SHORT
    line that spends most of its ticks on a ramp, where the median would
    be an on-ramp acceleration and the rule would cut nothing.

    An episode with no ballistic tick at all keeps everything (the
    conservative direction), as does one shorter than three ticks.
    """
    s = np.asarray(states, STATE_DTYPE)
    if len(s) < 3:
        return len(s) - 1, 0.0
    vz = np.asarray(s["velocity"], np.float64)[:, 2]
    dvz = np.diff(vz)
    g = float(np.median(dvz)) if gravity_step is None else float(gravity_step)
    hit = np.flatnonzero(np.abs(dvz - g) > float(tol))
    return (int(hit[-1]) + 1 if len(hit) else len(s) - 1), g


def subsample_by_path(states, spacing: float):
    """Indices of a time-ordered spine at (at least) ``spacing`` map units
    of TRAVEL between kept rows - row 0 always, the last row always.

    A per-tick spine drawn uniformly is uniform in TIME, so the slow parts
    of a line (the platform, a stall on a ramp) get most of the spawns and a
    fast flight almost none. Uniform over path length gives every segment
    of the line the same spawn mass, which is what a spawn distribution
    "along the line" means. ``spacing <= 0`` keeps every row.
    """
    s = np.asarray(states, STATE_DTYPE)
    n = len(s)
    if n == 0:
        return np.zeros(0, np.int64)
    if spacing <= 0.0 or n < 3:
        return np.arange(n, dtype=np.int64)
    o = np.asarray(s["origin"], np.float64)
    seg = np.linalg.norm(np.diff(o, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    keep = [0]
    nxt = float(spacing)
    for i in range(1, n):
        if cum[i] >= nxt:
            keep.append(i)
            nxt = cum[i] + float(spacing)
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return np.asarray(keep, np.int64)


# --------------------------------------------------------------------------
# P2: the SEARCH-DERIVED policy target (ExIt's TPT, Gumbel's pi')
# --------------------------------------------------------------------------
# Anthony, Tian & Barber 2017 (arXiv 1705.08439) ran the experiment our loop
# needs: CAT (`-log pi(a*|s)` at the search's argmax) against TPT
# (`-sum_a (n(s,a)/n(s)) log pi(a|s)`, the search's own distribution over
# first decisions). Move-prediction accuracy came out INDISTINGUISHABLE -
# top-1 error 47.0% vs 47.7% - and the TPT network was 50 +/- 13 Elo
# stronger, because "when MCTS is less certain between two moves ... it
# induces the IL agent to trade off accuracy on less important decisions for
# greater accuracy on critical decisions". Our loop clones CAT and reports
# per-head argmax agreement: the target AND the metric are the pair that
# paper measured a 50-Elo gap across.
#
# We have no visit counts, so the empirical stand-in is the population: at a
# state, the share of the SURVIVING search copies whose first decision was
# `a`. Two producers hand us that, and both reduce to the one-hot when only
# the winner survives:
#
#   * tools/plan_to_bc.py - the kept lineages of a beam_tas wave all start
#     from the SAME spawn, so the lineages that agree with line j on
#     decisions 0..d-1 are exactly the copies that stood at line j's
#     decision-d state, and their decision-d actions are that state's
#     empirical first-decision distribution (:func:`survivor_probs`);
#   * surfgym.dagger.relabel_windows - `hist[0]` holds every copy's first
#     decision and `_clone` copies it with the lineage, so counting it over
#     the live envs at the window's end IS the truncation-selection visit
#     count (:func:`count_probs`).


def onehot_probs(actions) -> np.ndarray:
    """(n, 6) action indices -> (n, 6, NPAD) one-hot target: the CAT target
    written in the distribution layout, i.e. what version-1 BC files mean."""
    a = np.asarray(actions, np.int64).reshape(-1, NACT)
    out = np.zeros((len(a), NACT, NPAD), np.float32)
    if len(a):
        out[np.arange(len(a))[:, None], np.arange(NACT)[None, :], a] = 1.0
    return out


def check_probs(probs, actions=None, tol: float = 1e-4) -> None:
    """Raise unless ``probs`` is a valid (n, 6, NPAD) per-head distribution:
    finite, non-negative, zero in every slot past that head's own nvec, and
    each head's row summing to 1. With ``actions``, also that the target
    puts non-zero mass on the stored action (the row the argmax loss uses)."""
    p = np.asarray(probs, np.float64)
    if p.ndim != 3 or p.shape[1] != NACT or p.shape[2] != NPAD:
        raise ValueError(f"probs must be (n, {NACT}, {NPAD}), got {p.shape}")
    if not np.isfinite(p).all() or (p < -tol).any():
        raise ValueError("probs must be finite and non-negative")
    for h, n in enumerate(ACTION_NVEC):
        if n < NPAD and len(p) and np.abs(p[:, h, n:]).max() > tol:
            raise ValueError(f"probs head {h} has mass past its {n} bins")
    if len(p):
        s = p.sum(-1)
        if np.abs(s - 1.0).max() > tol:
            raise ValueError(f"probs rows must sum to 1 (max error "
                             f"{np.abs(s - 1.0).max():.3g})")
    if actions is not None and len(p):
        a = np.asarray(actions, np.int64).reshape(-1, NACT)
        got = p[np.arange(len(a))[:, None], np.arange(NACT)[None, :], a]
        if got.min() <= 0.0:
            raise ValueError("probs put zero mass on the stored action")


def _accumulate(idx_actions, weights) -> np.ndarray:
    """(m, 6) actions + (m,) weights -> the (6, NPAD) weighted histogram,
    normalised per head. An empty set returns zeros (the caller decides)."""
    a = np.asarray(idx_actions, np.int64).reshape(-1, NACT)
    w = np.asarray(weights, np.float64).reshape(-1)
    out = np.zeros((NACT, NPAD), np.float32)
    tot = float(w.sum())
    if len(a) == 0 or tot <= 0.0:
        return out
    flat = (np.arange(NACT)[None, :] * NPAD + a).reshape(-1)
    acc = np.bincount(flat, weights=np.repeat(w, NACT),
                      minlength=NACT * NPAD).reshape(NACT, NPAD)
    return (acc / tot).astype(np.float32)


def count_probs(actions, weights=None) -> np.ndarray:
    """The (6, NPAD) empirical first-decision distribution over a set of
    search copies' actions ((m, 6) indices, optional (m,) weights).

    One copy in gives that copy's one-hot back exactly, which is why the
    argmax target is the special case rather than a different code path."""
    a = np.asarray(actions, np.int64).reshape(-1, NACT)
    w = np.ones(len(a)) if weights is None else weights
    return _accumulate(a, w)


def survivor_probs(tables, weights, ref: int) -> np.ndarray:
    """Per-decision first-decision distributions along lineage ``ref``.

    ``tables`` is a list of (D_i, 6) action tables searched from ONE spawn
    (beam_tas's kept lineages) and ``weights`` their per-line weights. For
    decision d of line ``ref``, the copies that stood at that state are the
    lines whose decisions 0..d-1 are identical to ``ref``'s, so the returned
    (D_ref, 6, NPAD) array holds, per decision, the weighted share of those
    lines that took each action at d.

    The prefix set can only SHRINK with d (a line that has diverged never
    rejoins, and a line that ended is gone), so this is one pass. Where the
    set has collapsed to ``ref`` alone the row is ``ref``'s own one-hot -
    the CAT target, unchanged, which on the measured 95.4%-identical waves
    is most of the line."""
    tabs = [np.asarray(t, np.int64).reshape(-1, NACT) for t in tables]
    M = len(tabs)
    if M == 0:
        raise ValueError("survivor_probs needs at least one table")
    lens = np.array([len(t) for t in tabs], np.int64)
    ref = int(ref)
    D = int(lens[ref])
    out = np.zeros((D, NACT, NPAD), np.float32)
    if D == 0:
        return out
    w = (np.ones(M, np.float64) if weights is None
         else np.asarray(weights, np.float64).reshape(M).astype(np.float64))
    # padded (M, D, 6); rows past a line's own length are zeros and are
    # masked out below, so they never reach a histogram
    A = np.zeros((M, D, NACT), np.int64)
    for i, t in enumerate(tabs):
        n = min(len(t), D)
        A[i, :n] = t[:n]
    long_enough = lens[:, None] > np.arange(D)[None, :]          # (M, D)
    agree = np.all(A == A[ref][None, :, :], axis=2) & long_enough
    # a line is in decision d's survivor set iff it HAS decision d and it
    # agreed with ref on every decision before d; agreement is monotone, so
    # the prefix product is the whole rule (and ref is always in its own set)
    prefix = np.ones((M, D), bool)
    if D > 1:
        prefix[:, 1:] = np.logical_and.accumulate(agree[:, :-1], axis=1)
    W = np.where(prefix & long_enough, w[:, None], 0.0)          # (M, D)
    tot = np.maximum(W.sum(0), 1e-12)
    col = np.arange(D, dtype=np.int64)[None, :] * NPAD
    for h in range(NACT):
        acc = np.bincount((col + A[:, :, h]).reshape(-1), weights=W.reshape(-1),
                          minlength=D * NPAD).reshape(D, NPAD)
        out[:, h, :] = (acc / tot[:, None]).astype(np.float32)
    return out


def gumbel_improved_probs(logits, q, counts, value: float,
                          c_visit: float = 50.0, c_scale: float = 0.1,
                          nvec=ACTION_NVEC) -> np.ndarray:
    """Danihelka et al. 2022's improved policy, per head.

    ``pi' = softmax(logits + sigma(completedQ))`` with
    ``sigma(qhat) = (c_visit + max_b N(b)) * c_scale * qhat`` - the target
    Gumbel MuZero distils by ``KL(pi', pi)`` ("This loss trains all actions,
    not only the action A_{n+1}"). Actions the search never visited are
    COMPLETED with ``v_mix``, the visit-weighted mixture of the root value
    and the visited actions' Q's, so an unvisited action inherits the
    node's value instead of a zero.

    ``logits`` (6, NPAD) the policy's own padded logits at this state, ``q``
    (6, NPAD) the search's Q per root action NORMALISED to [0, 1] (our Q's
    are a geodesic distance, ``V(s)`` or an arc coordinate on unrelated
    scales, so the normalisation has to be built by the caller), ``counts``
    (6, NPAD) how many copies took each action, ``value`` the root value in
    the same normalised units. ``c_scale`` defaults to 0.1, the shipped
    ``mctx`` default (Go/chess use 1.0), because our Q scale is not theirs.

    All-zero counts on a head returns ``softmax(logits)`` for it: with
    nothing visited there is no improvement to apply, which is also what
    makes this reduce to the policy itself at zero search."""
    lg = np.asarray(logits, np.float64).reshape(NACT, NPAD)
    qq = np.asarray(q, np.float64).reshape(NACT, NPAD)
    nn = np.asarray(counts, np.float64).reshape(NACT, NPAD)
    out = np.zeros((NACT, NPAD), np.float32)
    for h, n_h in enumerate(nvec):
        raw = lg[h, :n_h]
        pi = np.exp(raw - raw.max())
        pi = pi / pi.sum()
        N = nn[h, :n_h]
        tot = float(N.sum())
        if tot <= 0.0:
            out[h, :n_h] = pi.astype(np.float32)
            continue
        vis = N > 0
        pi_vis = float(pi[vis].sum())
        qh = qq[h, :n_h]
        v_mix = (float(value) if pi_vis <= 0.0 else
                 (float(value)
                  + tot / pi_vis * float((pi[vis] * qh[vis]).sum()))
                 / (1.0 + tot))
        qc = np.where(vis, qh, v_mix)
        z = raw + (c_visit + float(N.max())) * float(c_scale) * qc
        e = np.exp(z - z.max())
        out[h, :n_h] = (e / e.sum()).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# P3: AlphaZero's z - a value target from the planner's own line
# --------------------------------------------------------------------------
# AlphaZero's loss is `(z - v)^2 - pi^T log p`; ours has neither half. The
# critic is trained only by GAE on the policy's own rollouts, yet `--score
# dv` ranks the endgame with it. On a line that reaches a terminal the exact
# return is free: the planner already replays every kept line tick by tick,
# so running the trainer's own RaceReward alongside gives the reward stream
# the rollout would have produced, and the discounted sum backwards from the
# end is bootstrap-free over the whole line.


def decision_gamma(gamma_ref: float, act_every: int,
                   tick_ms: float = REFERENCE_TICK_MS) -> float:
    """The trainer's ``g_eff``: the discount between two POLICY DECISIONS.

    ``train_fast`` discounts its GAE at ``GAMMA_T ** KH`` where ``GAMMA_T =
    TickClock(tick_ms).gamma(--gamma)`` is the per-TICK discount and KH the
    ticks per decision. ``--gamma`` is defined at the 10 ms reference, so a
    131 Hz arm keeps the same horizon in SECONDS and the exponent changes
    with it; at 10 ms this is exactly ``gamma ** act_every``."""
    return float(TickClock(float(tick_ms)).gamma(float(gamma_ref))
                 ** int(act_every))


def returns_to_go(rewards, gamma_dec: float) -> np.ndarray:
    """``z[d] = sum_{j >= d} gamma_dec**(j - d) * rewards[j]`` - the
    Monte-Carlo return of a line that ENDS at the last element, computed
    backwards in one pass (no bootstrap, no lambda: on a completed line
    GAE(1) with no truncation IS this)."""
    r = np.asarray(rewards, np.float64).reshape(-1)
    z = np.zeros(len(r), np.float64)
    acc = 0.0
    for i in range(len(r) - 1, -1, -1):
        acc = r[i] + float(gamma_dec) * acc
        z[i] = acc
    return z.astype(np.float32)


#: cfg keys whose reward contribution a REPLAY cannot reproduce.
_LINE_REWARD_DROPPED = ("int_coef",)


def make_line_reward(cfg: dict, field, d0: float, k: int,
                     tick_ms: float = REFERENCE_TICK_MS):
    """``(reward_fn, info)``: the trainer's own RaceReward for this
    checkpoint, rebuilt from its config so a replayed planner line earns
    exactly what a training rollout would have earned on the same states.

    Mirrors train_fast's construction term for term (``scale = 100/rf_d0 *
    --race-shaping``, ``TIME_PEN_T``/``SPEED_COEF_T``/``STALL_EPS_T`` per
    tick at the run's tick, ``ng_gamma = GAMMA_T``, ``every = K`` only under
    ``--reward-per-decision``) for the single-map, flat, arc-less race
    policies ``--bc-file`` already restricts itself to.

    ONE term is dropped: ``--int-coef``. Count-based novelty pays
    ``int_coef / sqrt(N(cell))`` off a table shared by the whole training
    fleet across all episodes; a fresh table in a one-env replay would pay
    first-visit novelty for every cell of the line, which is not a reward
    the rollout would ever have seen. ``info['dropped']`` names it so the BC
    file's meta records that the stored ``z`` omits it.

    Refuses ``--race-arc`` and ``--race-kill-aware``: the first replaces the
    geodesic term with a line the BC path has no anchor for (make_eval_feeds
    refuses it too), the second shapes on a DIFFERENT field than the one the
    planner core baked, so ``d0`` and every delta would be another field's."""
    from .rewards import RaceReward
    if cfg.get("race_arc"):
        raise SystemExit("make_line_reward: --race-arc checkpoints are not "
                         "supported (the line's reward would be the geodesic "
                         "term, not the arc term the run trained on)")
    if cfg.get("race_kill_aware"):
        raise SystemExit("make_line_reward: --race-kill-aware shapes on a "
                         "kill-masked field this replay has not baked - its "
                         "d0 and every delta would be the wrong field's")
    tick = TickClock(float(tick_ms))
    every = int(k) if cfg.get("reward_per_decision") else 1
    gamma_t = tick.gamma(float(cfg.get("gamma") or 0.9995))
    d0 = float(max(float(d0), 1.0))
    rf = RaceReward(
        field,
        scale=100.0 / d0 * float(cfg.get("race_shaping") or 1.0),
        time_pen=tick.per_tick(float(cfg.get("time_pen") or 0.0)),
        success_bonus=float(cfg.get("success_bonus") or 0.0),
        stall_ticks=tick.secs_to_ticks(float(cfg.get("stall_secs") or 15.0)),
        stall_eps=tick.per_tick(float(cfg.get("stall_eps") or 32.0)),
        max_step=float(cfg.get("max_step") or 100.0),
        int_coef=0.0,
        speed_equiv=float(cfg.get("speed_equiv") or 0.0),
        fail_pen=float(cfg.get("fail_pen") or 0.0),
        finish_k=float(cfg.get("finish_k") or 0.0),
        finish_tref=float(cfg.get("finish_tref") or 120.0),
        every=every,
        d_floor=float(cfg.get("race_dfloor") or 0.0),
        d_latch=_d_latch_of(cfg, d0),
        ng=int(cfg.get("race_ng") or 0), ng_gamma=gamma_t, ng_d0=d0,
        death_charge=float(cfg.get("death_charge") or 0.0),
        tick_ms=tick.ms)
    rf.speed_coef = tick.per_tick(float(cfg.get("speed_coef") or 0.0))
    info = {"every": every, "gamma_tick": gamma_t,
            "gamma_decision": float(gamma_t ** int(k)),
            "scale": float(rf.scale), "time_pen": float(rf.time_pen),
            "success_bonus": float(rf.success_bonus),
            "d_latch": float(rf.d_latch), "d_floor": float(rf.d_floor),
            "speed_coef": float(rf.speed_coef),
            "dropped": [key for key in _LINE_REWARD_DROPPED
                        if float(cfg.get(key) or 0.0) > 0.0]}
    return rf, info


# --------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------
def save_bc_dataset(path, states, scal, latch, actions, weights, line_id,
                    meta: dict, probs=None, zret=None, zmask=None,
                    view=None, view_zmu=None, view_zsd=None) -> None:
    """Write a BC file. With none of ``probs`` / ``zret`` / ``zmask`` this
    writes VERSION 1 - the same keys, in the same order, that shipped - so
    an untreated round's file is byte-identical to the one it always wrote
    and no existing reader sees a new version number. Any of the three
    promotes the file to version 2 and writes all three (the missing ones
    filled with their version-1 meaning: the one-hot of ``actions``, and a
    zero return with a zero mask, i.e. no value target)."""
    states = np.ascontiguousarray(states, dtype=STATE_DTYPE)
    n = len(states)
    scal = np.ascontiguousarray(scal, dtype=np.float32).reshape(n, N_SCALAR)
    latch = np.ascontiguousarray(latch, dtype=np.float32).reshape(n)
    actions = np.ascontiguousarray(actions, dtype=np.int64).reshape(n, 6)
    weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(n)
    line_id = np.ascontiguousarray(line_id, dtype=np.int32).reshape(n)
    hi = np.asarray(ACTION_NVEC, np.int64)
    if (actions < 0).any() or (actions >= hi[None, :]).any():
        raise ValueError("actions out of range for ACTION_NVEC")
    # --view-continuous columns: written only when given, after every
    # existing key, so a discrete file is byte-identical to before
    vcols = {}
    if view is not None:
        vcols["view"] = np.ascontiguousarray(view, np.float32).reshape(n, 2)
        if not np.isfinite(vcols["view"]).all():
            raise ValueError("view must be finite")
        if (view_zmu is None) != (view_zsd is None):
            raise ValueError("view_zmu and view_zsd come together")
        if view_zmu is not None:
            # (n, 2), or (n, 3) under --view-absolute world (cos, sin, pitch)
            vcols["view_zmu"] = np.ascontiguousarray(view_zmu,
                                                     np.float32).reshape(n, -1)
            vcols["view_zsd"] = np.ascontiguousarray(view_zsd,
                                                     np.float32).reshape(n, -1)
            if vcols["view_zmu"].shape[1] not in (2, 3)                     or vcols["view_zsd"].shape != vcols["view_zmu"].shape:
                raise ValueError("view_zmu / view_zsd must be (n, 2) or (n, 3)")
            if (vcols["view_zsd"] < 0.0).any():
                raise ValueError("view_zsd must be >= 0")
    elif view_zmu is not None or view_zsd is not None:
        raise ValueError("view moments without the executed view")
    if probs is None and zret is None and zmask is None:
        np.savez(path, version=np.int32(1), states=states, scal=scal,
                 latch=latch, actions=actions, weights=weights,
                 line_id=line_id,
                 meta=np.str_(json.dumps(meta, sort_keys=True, default=str)),
                 **vcols)
        return
    probs = (onehot_probs(actions) if probs is None
             else np.ascontiguousarray(probs, np.float32).reshape(n, NACT,
                                                                  NPAD))
    check_probs(probs, actions)
    zret = (np.zeros(n, np.float32) if zret is None
            else np.ascontiguousarray(zret, np.float32).reshape(n))
    zmask = (np.zeros(n, np.float32) if zmask is None
             else np.ascontiguousarray(zmask, np.float32).reshape(n))
    if not np.isfinite(zret).all():
        raise ValueError("zret must be finite")
    np.savez(path, version=np.int32(2), states=states, scal=scal,
             latch=latch, actions=actions, weights=weights, line_id=line_id,
             meta=np.str_(json.dumps(meta, sort_keys=True, default=str)),
             probs=probs, zret=zret, zmask=zmask, **vcols)


def load_bc_meta(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["meta"]))


def load_bc_arrays(path) -> dict:
    """Every column of a BC file of EITHER version, with the version-1
    meaning filled in for the three optional ones (one-hot ``probs``, zero
    ``zret``, zero ``zmask``). ``version`` and ``has_probs`` / ``has_value``
    say what the file itself carried, so a merge can record it."""
    z = np.load(path, allow_pickle=False)
    ver = int(z["version"])
    if ver not in BC_VERSIONS:
        raise SystemExit(f"{path}: BC file version {ver} not in "
                         f"{list(BC_VERSIONS)}")
    act = np.asarray(z["actions"], np.int64)
    n = len(act)
    has_p = "probs" in z.files
    has_v = "zret" in z.files
    return {"version": ver,
            "states": np.asarray(z["states"], STATE_DTYPE),
            "scal": np.asarray(z["scal"], np.float32),
            "latch": np.asarray(z["latch"], np.float32),
            "actions": act,
            "weights": np.asarray(z["weights"], np.float32),
            "line_id": np.asarray(z["line_id"], np.int32),
            "probs": (np.asarray(z["probs"], np.float32) if has_p
                      else onehot_probs(act)),
            "zret": (np.asarray(z["zret"], np.float32) if has_v
                     else np.zeros(n, np.float32)),
            "zmask": (np.asarray(z["zmask"], np.float32) if has_v
                      else np.zeros(n, np.float32)),
            "has_probs": has_p, "has_value": has_v,
            # --view-continuous columns, None when the file has none
            "view": (np.asarray(z["view"], np.float32) if "view" in z.files
                     else None),
            "view_zmu": (np.asarray(z["view_zmu"], np.float32)
                         if "view_zmu" in z.files else None),
            "view_zsd": (np.asarray(z["view_zsd"], np.float32)
                         if "view_zsd" in z.files else None),
            "meta": json.loads(str(z["meta"]))}


class BCDataset:
    """The trainer's view of a BC file: device tensors + a private sampler.

    ``sample(n)`` returns ``(scal, pose, act, w)`` for n random rows drawn
    with the dataset's OWN torch generator, so the PPO permutation stream
    (perm_gen) and the CUDA action noise are untouched by its presence.
    ``sample_all(n)`` is the same draw plus the three version-2 targets -
    ``(scal, pose, act, w, probs, z, zmask)`` - with the version-1 meaning
    synthesised per BATCH when the file does not carry them (the one-hot of
    ``act``, a zero return, a zero mask), so the (B, 6, 15) target never
    exists for a file that has no distribution in it and ``--bc-target
    dist`` on a version-1 file is the argmax loss exactly.
    ``render(lidar, pose, dtype)`` turns the pose columns into the depth
    image the same way the rollout's fill_vision does.

    ``priv_fn`` (--priv-critic) is called ONCE here with the stored states
    and the stored latch column and must return the (n, PRIV_DIM) block the
    critic reads; the trainer passes a closure over its OWN PrivFeat and
    reward field, because privfeat.py holds one implementation of that block
    on purpose and a second copy here would be the drift it warns about."""

    def __init__(self, path, device, n_latch: int, obs_reward: bool,
                 seed: int = 0, priv_fn=None, view_continuous: bool = False,
                 yaw_adaptive: bool = False, pitch_rate_max_deg: float = 10.0,
                 view_absolute=None):
        import torch
        z = np.load(path, allow_pickle=False)
        ver = int(z["version"])
        if ver not in BC_VERSIONS:
            raise SystemExit(f"{path}: BC file version {ver} not in "
                             f"{list(BC_VERSIONS)}")
        self.path = str(path)
        self.version = ver
        self.meta = json.loads(str(z["meta"]))
        states = np.asarray(z["states"], STATE_DTYPE)
        scal = np.asarray(z["scal"], np.float32)
        latch = np.asarray(z["latch"], np.float32)
        act = np.asarray(z["actions"], np.int64)
        w = np.asarray(z["weights"], np.float32)
        self.n = int(len(states))
        if self.n < 1:
            raise SystemExit(f"{path}: empty BC dataset")
        # the file carries the columns the checkpoint it was built for
        # uses; the trainer's own layout must agree or the row is wrong
        if bool(self.meta.get("obs_reward")) != bool(obs_reward):
            raise SystemExit(f"{path}: built for obs_reward="
                             f"{self.meta.get('obs_reward')!r}, trainer has "
                             f"obs_reward={obs_reward!r}")
        if int(self.meta.get("n_latch", 0)) != int(n_latch):
            raise SystemExit(f"{path}: built with n_latch="
                             f"{self.meta.get('n_latch')!r}, trainer has "
                             f"N_LATCH={n_latch}")
        cols = [scal] + ([latch[:, None]] if n_latch else [])
        self.scal = torch.as_tensor(np.concatenate(cols, axis=1),
                                    dtype=torch.float32, device=device)
        pose = np.zeros((self.n, 6), np.float32)
        pose[:, 0:3] = states["origin"]
        pose[:, 3] = states["yaw"]
        pose[:, 4] = states["pitch"]
        pose[:, 5] = states["ducked"]
        self.pose = torch.as_tensor(pose, device=device)
        self.act = torch.as_tensor(act, dtype=torch.long, device=device)
        self.w = torch.as_tensor(w, dtype=torch.float32, device=device)
        # --- the version-2 targets (P2 policy, P3 value) ------------------
        self.has_probs = "probs" in z.files
        self.has_value = "zret" in z.files
        self.probs = None
        if self.has_probs:
            p = np.asarray(z["probs"], np.float32).reshape(self.n, NACT, NPAD)
            check_probs(p, act)
            self.probs = torch.as_tensor(p, dtype=torch.float32,
                                         device=device)
        zr = (np.asarray(z["zret"], np.float32) if self.has_value
              else np.zeros(self.n, np.float32))
        zm = (np.asarray(z["zmask"], np.float32) if self.has_value
              else np.zeros(self.n, np.float32))
        self.z = torch.as_tensor(zr.reshape(self.n), dtype=torch.float32,
                                 device=device)
        self.zmask = torch.as_tensor(zm.reshape(self.n), dtype=torch.float32,
                                     device=device)
        self.value_rows = int((zm > 0.0).sum())
        # --priv-critic: the block is built ONCE, here, from the stored
        # states - never per minibatch, and never from a second copy of the
        # column definitions (privfeat.PrivFeat is the only one)
        self.priv = None
        if priv_fn is not None:
            pv = np.ascontiguousarray(priv_fn(states, latch), np.float32)
            if len(pv) != self.n:
                raise SystemExit(f"{path}: priv block has {len(pv)} rows for "
                                 f"{self.n} BC rows")
            self.priv = torch.as_tensor(pv, dtype=torch.float32,
                                        device=device)
        # --- --view-continuous: the view heads' targets ------------------
        # vz    the executed z per row (from the file's `view`, else from the
        #       bins as the core would apply them) - the argmax-style point
        #       target of the Gaussian NLL;
        # vzmu/vzsd  the elite copies' moment-matched z (the file's own
        #       columns, else the bin distribution's moments, else the point
        #       with std 0) - what --bc-target dist fits.
        self.view_continuous = bool(view_continuous)
        self.view_absolute = (str(view_absolute) if view_absolute else None)
        self.vz = self.vzmu = self.vzsd = None
        self.view_note = ""
        if self.view_continuous:
            # --view-absolute: the file's rows and the trainer's heads must
            # read the view row the same way. A file written by a
            # delta-mode plan_to_bc has no key (= None).
            f_abs = self.meta.get("view_absolute") or None
            if "view" not in z.files and self.view_absolute is not None:
                raise SystemExit(f"{path}: a discrete BC file cannot be used "
                                 f"under --view-absolute {self.view_absolute}"
                                 ": its bins are per-tick deltas, not targets")
            if f_abs != self.view_absolute:
                raise SystemExit(
                    f"{path}: its view rows are {view_desc(f_abs)} "
                    f"(view_absolute={f_abs!r}) but this trainer's heads "
                    f"write {view_desc(self.view_absolute)} "
                    f"(view_absolute={self.view_absolute!r}); rebuild the "
                    "file with tools/plan_to_bc.py from this checkpoint's "
                    "own lines")
            nz = n_z(self.view_absolute)
            if "view" in z.files:
                view = np.asarray(z["view"], np.float32).reshape(self.n, 2)
                self.view_note = "view from the file"
            else:
                view = bin_to_view(act, bool(yaw_adaptive),
                                   float(pitch_rate_max_deg))
                self.view_note = "view DERIVED from the bins"
            vz = z_from_view_any(view, self.view_absolute,
                                 float(pitch_rate_max_deg))
            if "view_zmu" in z.files and "view_zsd" in z.files:
                vmu = np.asarray(z["view_zmu"], np.float32).reshape(self.n, -1)
                vsd = np.asarray(z["view_zsd"], np.float32).reshape(self.n, -1)
                if vmu.shape[1] != nz or vsd.shape[1] != nz:
                    raise SystemExit(f"{path}: view moments are {vmu.shape[1]}"
                                     f" wide, this policy has {nz} view heads")
                self.view_note += ", moments from the file"
            elif self.has_probs and "view" not in z.files:
                vmu, vsd = bin_view_moments(
                    np.asarray(z["probs"], np.float32), bool(yaw_adaptive),
                    float(pitch_rate_max_deg))
                self.view_note += ", moments from the bin distribution"
            else:
                vmu, vsd = vz.copy(), np.zeros_like(vz)
                self.view_note += ", point target (std 0)"
            if not (np.isfinite(vz).all() and np.isfinite(vmu).all()
                    and np.isfinite(vsd).all()):
                raise SystemExit(f"{path}: non-finite view targets")
            self.vz = torch.as_tensor(vz, dtype=torch.float32, device=device)
            self.vzmu = torch.as_tensor(vmu, dtype=torch.float32, device=device)
            self.vzsd = torch.as_tensor(vsd, dtype=torch.float32, device=device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed((int(seed) * 7368787 + 12345) & 0x7FFFFFFFFFFFFFFF)
        self.device = device

    def _draw(self, n: int):
        import torch
        return torch.randint(0, self.n, (int(n),), generator=self.gen,
                             device=self.device)

    def sample(self, n: int):
        idx = self._draw(n)
        return self.scal[idx], self.pose[idx], self.act[idx], self.w[idx]

    def sample_all(self, n: int, view: bool = False):
        """One draw (the same RNG call ``sample`` makes, so the stream is
        unchanged) -> ``(scal, pose, act, w, probs, z, zmask, priv)``;
        with ``view=True`` (--view-continuous) three more: ``(vz, vzmu,
        vzsd)``, the view heads' point target and moment-matched target."""
        import torch
        idx = self._draw(n)
        a = self.act[idx]
        if self.probs is None:
            p = torch.zeros((a.shape[0], NACT, NPAD), dtype=torch.float32,
                            device=self.device)
            p.scatter_(2, a.unsqueeze(-1), 1.0)
        else:
            p = self.probs[idx]
        base = (self.scal[idx], self.pose[idx], a, self.w[idx], p,
                self.z[idx], self.zmask[idx],
                None if self.priv is None else self.priv[idx])
        if not view:
            return base
        if self.vz is None:
            raise RuntimeError("BCDataset built without view_continuous")
        return base + (self.vz[idx], self.vzmu[idx], self.vzsd[idx])

    @staticmethod
    def render(lidar, pose, dtype):
        """(n, FRAME) depth in the rollout buffer's dtype from a (n, 6)
        pose block - fleet.render's single-slot path, verbatim."""
        img = lidar.render(pose[:, 0:3], pose[:, 3], pose[:, 4], pose[:, 5])
        return img.reshape(pose.shape[0], -1).to(dtype)

    def describe(self) -> str:
        m = self.meta
        lines = m.get("lines", "?")
        return (f"bc: {self.n:,} planner decisions from {lines} line(s) "
                f"({self.path}); best {m.get('best_s', '?')}s, "
                f"act_every {m.get('act_every', '?')}, "
                f"obs_reward {bool(m.get('obs_reward'))}, "
                f"n_latch {int(m.get('n_latch', 0))}; v{self.version} "
                f"probs {'yes' if self.has_probs else 'no (one-hot)'}, "
                f"value rows {self.value_rows:,}/{self.n:,}"
                + (f"; view targets: {self.view_note}"
                   + (f" [absolute {self.view_absolute}]"
                      if self.view_absolute else "")
                   if self.view_continuous else ""))
