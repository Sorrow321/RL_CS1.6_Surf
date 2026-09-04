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
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import ACTION_NVEC, STATE_DTYPE

__all__ = ["BC_VERSION", "REWARD_SLOT", "make_eval_feeds", "replay_line",
           "save_bc_dataset", "load_bc_meta", "BCDataset",
           "rank_lineages", "contact_rows", "last_contact_cut",
           "subsample_by_path"]

BC_VERSION = 1
REWARD_SLOT = 12          # train_fast.py REWARD_SLOT: --obs-reward's column
N_SCALAR = 15


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
    return feed


def make_eval_feeds(cfg: dict, field, d0: float, k: int):
    """``(reward_slot, reward_feed, latch_fn)`` for a race checkpoint's
    config: the two side-channel observation columns an eval core does not
    produce, computed exactly as train_fast's eval wrappers compute them.
    reward_slot is -1 (and reward_feed None) without --obs-reward; latch_fn
    is None without --race-latch. ``d0`` is the trainer's start distance
    (mean field over the RAW map spawns)."""
    if cfg.get("race_arc"):
        raise SystemExit("make_eval_feeds: --race-arc checkpoints are not "
                         "supported (the slot-12 mirror would be the "
                         "geodesic term, not the arc term)")
    d_latch = float(cfg.get("race_latch") or 0.0)
    frac = float(cfg.get("race_latch_frac") or 0.0)
    if frac > 0.0:
        d_latch = frac * float(d0)
    latch_fn = _latch_feed(field, d_latch) if d_latch > 0.0 else None
    if not cfg.get("obs_reward"):
        return -1, None, latch_fn
    scale = 100.0 / max(float(d0), 1.0) * float(cfg.get("race_shaping") or 1.0)
    tp = float(cfg.get("time_pen") or 0.005)
    d_floor = float(cfg.get("race_dfloor") or 0.0)
    ng = int(cfg.get("race_ng") or 0)
    ng_g = float(cfg.get("gamma", 0.9995)) ** int(k)
    return REWARD_SLOT, _reward_feed(field, scale, tp, int(k), d_floor,
                                     latch_fn, ng, ng_g, float(d0)), latch_fn


# --------------------------------------------------------------------------
# replaying a planner line into rows
# --------------------------------------------------------------------------
def replay_line(core, spawn_state, obs_start, acts, k: int, reward_feed=None,
                latch_fn=None, reward_slot: int = REWARD_SLOT,
                max_ticks: int = 0, keep_final: bool = False):
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
    """
    acts = np.asarray(acts)
    core.set_state(0, spawn_state)
    obs = np.array(obs_start, np.float32).reshape(1, -1).copy()
    rows, tick_states = [], []
    t, finished, ended = 0, False, False
    cap = int(max_ticks) if max_ticks else int(len(acts)) * int(k)
    for d in range(len(acts)):
        a = np.ascontiguousarray(acts[d].reshape(1, 6), dtype=np.int32)
        # the wrapper's _obs order: the reward feed first (it reads the
        # latch flag as it stood one decision AGO), then the latch
        fv = None if reward_feed is None else float(reward_feed(core)[0])
        lf = 0.0 if latch_fn is None else float(latch_fn(core)[0])
        st = core.get_states()[0].copy()
        scal = obs[0, :N_SCALAR].copy()
        if fv is not None:
            scal[reward_slot] = fv
        rows.append((st, scal, lf, acts[d].astype(np.int64).copy()))
        for _ in range(int(k)):
            tick_states.append(core.get_states()[0].copy())
            obs, _rew, done, trunc, _term = core.step(a)
            t += 1
            if int(core.goal_hits[0]):
                finished = True
            if done[0] or trunc[0] or finished or t >= cap:
                ended = True
                break
        if ended:
            break
    if keep_final and not ended and len(acts):
        tick_states.append(core.get_states()[0].copy())
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
# the file
# --------------------------------------------------------------------------
def save_bc_dataset(path, states, scal, latch, actions, weights, line_id,
                    meta: dict) -> None:
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
    np.savez(path, version=np.int32(BC_VERSION), states=states, scal=scal,
             latch=latch, actions=actions, weights=weights, line_id=line_id,
             meta=np.str_(json.dumps(meta, sort_keys=True, default=str)))


def load_bc_meta(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["meta"]))


class BCDataset:
    """The trainer's view of a BC file: device tensors + a private sampler.

    ``sample(n)`` returns ``(scal, pose, act, w)`` for n random rows drawn
    with the dataset's OWN torch generator, so the PPO permutation stream
    (perm_gen) and the CUDA action noise are untouched by its presence.
    ``render(lidar, pose, dtype)`` turns the pose columns into the depth
    image the same way the rollout's fill_vision does."""

    def __init__(self, path, device, n_latch: int, obs_reward: bool,
                 seed: int = 0):
        import torch
        z = np.load(path, allow_pickle=False)
        if int(z["version"]) != BC_VERSION:
            raise SystemExit(f"{path}: BC file version {int(z['version'])} "
                             f"!= {BC_VERSION}")
        self.path = str(path)
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
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed((int(seed) * 7368787 + 12345) & 0x7FFFFFFFFFFFFFFF)
        self.device = device

    def sample(self, n: int):
        import torch
        idx = torch.randint(0, self.n, (int(n),), generator=self.gen,
                            device=self.device)
        return self.scal[idx], self.pose[idx], self.act[idx], self.w[idx]

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
                f"n_latch {int(m.get('n_latch', 0))}")
