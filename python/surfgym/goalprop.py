"""goalprop.py - TRAJECTORY PROPOSALS: the learned planner chooses among
candidate plans built per decision (``--goal-planner learned --plan-vocab
proposals``); docs/litsurvey-planner-executor.md section 7.2 b4 and 7.3.

Why. The surf executor (stage b, ``--goal-planner vocab``, run psEF050v)
completes 67-79% of HINDSIGHT plans (segments of its own flights) and 0% of
the fixed vocabulary shapes, so a planner restricted to the fixed vocabulary
(stage 3, surfgym/goallearn.py) sees only failures on surf. The user's design
(2026-09-23): "if you randomize the trajectories and take only the ones that
are different enough, these are your ... trajectory proposal ... if you're on
surfing, you can take off earlier, you can take off later, you can take off
flying a bit higher or a bit more flat but faster. So all of these are
reasonable actions. And this is what I want the planner to do." And: "the
planner should ... make sure that these trajectories that it generates are
actually possible to execute ... we should penalize our planner ... We don't
want to enforce it. We can measure it."

THE CANDIDATE SET, built per decision for every env that needs a plan, K =
``--plan-k`` (default 32) polylines, all starting at the agent:

* HINDSIGHT (up to K/2): the reservoir rows nearest the agent in (position,
  velocity x 1 s) within the goal radius (192 u) - goalsurf's hindsight key,
  over the WHOLE reservoir rather than one iteration's ~3.7k-row pool - each
  row's reached-state segment (a piece of the policy's OWN flight, 1-5 s)
  re-anchored to start at the agent. Its time is its flight time.
* PERTURBATIONS of the K/4 nearest of those, along the user's axes, every
  magnitude set once, in seconds, hull widths (32 u) or fractions:
  take off EARLIER (the segment's first 0.5 s dropped) or LATER (0.5 s more
  on the segment's initial velocity first); HIGHER-slower or FLATTER-faster
  (the vertical deviation from the start -> end chord - the arc's height - x
  1.25 or x 0.75, and the time x 1.25 or x 0.75: the same landing, a higher
  or a flatter flight); a LATERAL offset left or right, 6 hull widths (192 u)
  at the end, growing linearly along the arc from 0 at the agent.
* UNINFORMED: the rest, shapes of the executor's own vocabulary drawn
  uniformly without replacement (the 144 surf shapes on a surf executor, the
  80 walking shapes on a walking one), sized for the current speed; up to 2K
  are drawn, so that K survive the deduplication below.
* eps-NMS in plan space ("different enough"): candidates are admitted in
  priority order - the K/4 nearest hindsight segments, their perturbations
  (nearest seed first), the remaining hindsight segments, then the shapes -
  and one is DROPPED when the mean distance between its 8 points at arc
  fractions 1/8 .. 8/8 and those of an already admitted candidate is <= 2 hull
  widths (64 u). A perturbation is admitted only if its seed was; at most K/2
  are informed (hindsight + perturbed: Ichter's lambda = 0.5), the rest
  uninformed; a candidate shorter than 64 u is no plan. Fewer than K
  survivors are padded and MASKED.
* NOTHING filters by geometry: a candidate may run through solid or into the
  pit, and feasibility is learned from the executor's outcomes (the -0.3 of a
  plan it does not complete), measured by the diagnostics below.

THE PLANNER HEAD (:class:`PointerNet`): the stage-3 observation encoder (the
occupancy slabs or the walkable patch, + the visit channel, + 9 scalars) gives
h; a candidate encoder (2 x 128 ReLU) gives e_i from the candidate's features
- its 8 points relative to the agent in the WORLD-aligned frame (the patch's
and the scalars' frame) and in the ego (yaw) frame, / 1000 u; log1p(arc /
1000 u); its time / 3 s; its source one-hot (hindsight / perturbed /
uninformed); the hindsight match distance / 192 u (1 for a shape) - and
logit_i = u . tanh(W_h h + W_e e_i) (the pointer-network score of Vinyals et
al. 2015), a softmax over the VALID candidates. The value head reads h and
the masked mean of the e_i. PPO on the log-prob of the chosen index; the
candidate features and mask are stored per transition, so the ratio is
recomputed on exactly the set the choice was made from.

REWARDS, the plan's lifetime and the PPO are the learned planner's
(goallearn.LearnedPlanner, which this subclasses): --plan-r-ok / --plan-r-fail
per closed plan, +--plan-finish-bonus on a finish, plan-end novelty
0.5/sqrt(n) over 128 u cells (none on a death). A plan closes on 90% arc
completion (corridor 192 u), on its budget (1.5 x its time) or at the
episode's end.

DIAGNOSTICS, never a reward or a filter: the learned planner's columns
(plan/wall and plan/wall_len - 3-D solid crossing on a surf map, off-graph
share on a walking one - with plan/wall_base now the base rate over the
CANDIDATE SET at the same states), and PROP_COLS: the candidate count after
NMS and its split by source, the planner's choice share by source, the
executor's completion rate by source, the hindsight bank's size.

Nothing here reads a demo, a route file or a map-specific constant (CLAUDE.md
rules 0 and 0b): the hindsight segments are the policy's OWN flights (the
reservoir it harvested), the shapes are generic geometry, and every constant
is a time, a hull width, a fraction or a count, the same on every map.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .goallearn import (BUDGET_MULT, BUDGET_SPEED_U, COMPLETE_FRAC,
                        NET_HIDDEN, NOVELTY_CELL_U, N_SCAL, PATCH_N,
                        PLAN_CLIP, PLAN_COLS,
                        PLAN_COLS_SURF, PLAN_DEFAULTS, PLAN_MAX_GRAD, PLAN_MB,
                        PLAN_VF, WALL_BASE_ROWS, LearnedPlanner, PlanState,
                        WalkMap, _entropy, _spec_default, build_obs, plan_gae)
from .goalsurf import (HS_L_MAX, HS_TAU_S, SKIP_CELLS, WALL_BASE_ROWS_SURF,
                       SlabMap, build_obs_surf, make_vocab, surf_spec)

__all__ = ["HindsightBank", "ProposalMaker", "Proposals", "PointerNet",
           "ProposalPlanner", "prop_spec", "base_vocab_spec", "arc_samples",
           "perturb", "nms_select", "proposal_planner_from_state",
           "make_proposal_hooks", "PROP_COLS", "PROP_K", "AXES",
           "SRC_NAMES", "SRC_HS", "SRC_PERT", "SRC_UNIF", "N_FEAT"]

# ------------------------------------------------------------ the candidates
PROP_K = 32                        # candidates per decision (--plan-k)
HULL_U = 32.0                      # the player hull's width (u)
SHIFT_S = 0.5                      # take off earlier / later (s)
VSCALE = 0.25                      # higher-slower / flatter-faster: x (1 +- .25)
LATERAL_U = 6.0 * HULL_U           # lateral offset at the plan's end (192 u)
NMS_U = 2.0 * HULL_U               # "different enough": mean distance (64 u)
MIN_LEN_U = NMS_U                  # a candidate shorter than eps is no plan
AXES = ("earlier", "later", "higher", "flatter", "left", "right")
N_AXES = len(AXES)
SRC_HS, SRC_PERT, SRC_UNIF = 0, 1, 2
SRC_NAMES = ("hindsight", "perturbed", "uninformed")

# ------------------------------------------------------------ the features
N_PTS = 8                          # points per candidate, arc fractions j/8
POS_SCALE_U = 1000.0               # u per feature unit
T_REF_S = 3.0                      # s per feature unit (the surf T_plan)
FEAT_CLIP = 16.0
N_FEAT = 2 * 3 * N_PTS + 6         # world + ego points, arc, time, 3 src, match
CAND_HIDDEN = 128
MASK_NEG = -1e9                    # a masked logit (finite: 0 * logp stays 0)

# ------------------------------------------------------------ diagnostics
DIAG_SAMPLES = 64                  # samples per candidate, every arc/64

# the proposal machine's draws (the uninformed shapes) and the eval's: --seed
# + these offsets (the executor's and the planner's own RNGs are untouched)
PROP_SEED_OFFSET = 5171
PROP_EVAL_SEED_OFFSET = 9173
# the hindsight bank is rebuilt from the reservoir once this share of it has
# been harvested anew since the last build (a KD-tree over up to 100k rows is
# ~0.1 s; the reservoir turns over slowly)
BANK_REFRESH_FRAC = 0.1

# progress.csv columns under --plan-vocab proposals, appended LAST after the
# learned planner's (PLAN_COLS, + PLAN_COLS_SURF's two on a surf base)
PROP_COLS = ["plan/cand", "plan/cand_hs", "plan/cand_pert", "plan/cand_unif",
             "plan/choose_hs", "plan/choose_pert", "plan/choose_unif",
             "plan/complete_hs", "plan/complete_pert", "plan/complete_unif",
             "plan/bank"]

# the spec keys that belong to the proposal machine, not to the base
# vocabulary / observation it wraps
PROP_ONLY_KEYS = ("vocab", "base", "k", "n_pts", "n_feat", "cand_hidden",
                  "hs_radius", "hs_tau_s", "shift_s", "vscale", "lateral_u",
                  "nms_u", "snap_secs")


def prop_spec(base: str = "surf", k: int = PROP_K, snap_secs: float = 0.25,
              hs_radius: float = 192.0) -> dict:
    """Everything that defines what a stored proposal planner MEANS: the base
    vocabulary and observation (the executor's: surf or walk, flattened in),
    the candidate machine's constants and the reservoir's snapshot cadence
    (a hindsight segment's time is its snapshot count x this)."""
    if base not in ("walk", "surf"):
        raise ValueError(f"proposal base vocabulary {base!r}: walk or surf")
    if int(k) < 2:
        raise ValueError(f"--plan-k {k}: at least 2 candidates")
    b = surf_spec() if base == "surf" else _spec_default()
    s = {kk: v for kk, v in b.items() if kk != "vocab"}
    s.setdefault("in_ch", 2)
    s.update({"vocab": "proposals", "base": base, "k": int(k),
              "n_pts": N_PTS, "n_feat": N_FEAT, "cand_hidden": CAND_HIDDEN,
              "hs_radius": float(hs_radius), "hs_tau_s": HS_TAU_S,
              "shift_s": SHIFT_S, "vscale": VSCALE, "lateral_u": LATERAL_U,
              "nms_u": NMS_U, "snap_secs": round(float(snap_secs), 6)})
    return s


def base_vocab_spec(spec: dict) -> dict:
    """A proposals spec -> the spec of its base vocabulary (what make_vocab /
    SlabMap / WalkMap read): the surf spec, or the walking one."""
    s = {k: v for k, v in spec.items() if k not in PROP_ONLY_KEYS}
    if spec.get("base") == "surf":
        s["vocab"] = "surf"
    return s


# ==========================================================================
# the hindsight bank: the policy's own reached-state segments
# ==========================================================================
class HindsightBank:
    """The reservoir rows that carry a reached-state segment (goalsys kind 0:
    a snapshot, the state the same episode reached 1-5 s later and the
    snapshots between), keyed like goalsurf's hindsight source: (position,
    velocity x ``tau_s``). A SNAPSHOT of the rows (the reservoir ring is
    overwritten in place); segments are padded with their last valid point,
    so a gathered segment is a polyline of ``width`` points."""

    def __init__(self, origins, velocities, segs, seglen,
                 tau_s: float = HS_TAU_S):
        from scipy.spatial import cKDTree
        org = np.asarray(origins, np.float64).reshape(-1, 3)
        vel = np.asarray(velocities, np.float64).reshape(-1, 3)
        sg = np.asarray(segs, np.float32)
        sl = np.asarray(seglen, np.int64).reshape(-1)
        if sg.ndim != 3 or sg.shape[0] != len(sl) or sg.shape[2] != 3:
            raise ValueError(f"segments {sg.shape} vs {len(sl)} lengths")
        ok = ((sl >= 2) & np.isfinite(org).all(1) & np.isfinite(vel).all(1)
              & np.isfinite(sg).all(axis=(1, 2)))
        r = np.flatnonzero(ok)
        self.width = int(sg.shape[1])
        self.n = int(len(r))
        self.tau = float(tau_s)
        n = sl[r]
        pad = np.minimum(np.arange(self.width)[None, :], (n - 1)[:, None])
        self.org = org[r]
        self.vel = vel[r]
        self.segs = sg[r[:, None], pad] if self.n else \
            np.zeros((0, self.width, 3), np.float32)
        self.seglen = n
        self.tree = (cKDTree(np.hstack([self.org, self.vel * self.tau]))
                     if self.n else None)

    @classmethod
    def from_arrays(cls, states, goals, segs, seglen,
                    tau_s: float = HS_TAU_S) -> "HindsightBank":
        """Reservoir-layout arrays (STATE_DTYPE rows + the goal columns); a
        row whose goal is NaN carries no segment."""
        g = np.asarray(goals, np.float64).reshape(-1, 3)
        sl = np.array(seglen, np.int64).reshape(-1)
        sl[~np.isfinite(g).all(1)] = 0
        return cls(np.asarray(states["origin"]), np.asarray(states["velocity"]),
                   segs, sl, tau_s)

    @classmethod
    def from_reservoir(cls, respawn, tau_s: float = HS_TAU_S):
        """The live RespawnBuffer's filled ring -> a bank, or None (no goal
        columns, or no row carries a segment)."""
        rows = respawn.goal_rows() if respawn is not None else None
        if rows is None:
            return None
        bank = cls.from_arrays(*rows, tau_s=tau_s)
        return bank if bank.n else None

    @classmethod
    def from_state(cls, sd, map_id: Optional[str] = None,
                   tau_s: float = HS_TAU_S):
        """A checkpoint's ``respawn`` entry (single-map layout) -> a bank, or
        None: no segments, a per-map (multi-map) layout, or another map's
        reservoir (its coordinates mean nothing here)."""
        if not isinstance(sd, dict) or sd.get("states") is None \
                or sd.get("segs") is None:
            return None
        if map_id is not None and sd.get("map_id") not in (None, map_id):
            return None
        bank = cls.from_arrays(sd["states"], sd["goals"], sd["segs"],
                               sd["seglen"], tau_s=tau_s)
        return bank if bank.n else None

    def query(self, p, v, k: int, radius: float):
        """(B, 3) positions and velocities -> (dist (B, k), rows (B, k)): the
        k nearest rows in (position, velocity x tau) within ``radius``,
        nearest first; a miss is (inf, -1)."""
        p = np.atleast_2d(np.asarray(p, np.float64))
        B = len(p)
        k = int(k)
        if self.tree is None or k <= 0 or B == 0:
            return np.full((B, max(k, 0)), np.inf), \
                np.full((B, max(k, 0)), -1, np.int64)
        keys = np.hstack([p, np.atleast_2d(np.asarray(v, np.float64))
                          * self.tau])
        d, r = self.tree.query(keys, k=k, distance_upper_bound=float(radius))
        d = np.asarray(d, np.float64).reshape(B, k)
        r = np.asarray(r, np.int64).reshape(B, k)
        miss = ~np.isfinite(d) | (r >= self.n)
        return np.where(miss, np.inf, d), np.where(miss, -1, r)


# ==========================================================================
# geometry: arc-fraction resampling, perturbations, NMS
# ==========================================================================
def arc_samples(Q, fracs):
    """Polylines ``Q`` (..., N, 3) - padding repeats the last point - and
    arc fractions (M,) -> (points (..., M, 3), arc length (...,)): the point
    at each fraction of each polyline's own 3-D arc length, linear between
    vertices (the batched form of route.resample_polyline's interpolation)."""
    Q = np.asarray(Q, np.float64)
    lead = Q.shape[:-2]
    N = int(Q.shape[-2])
    f = np.asarray(fracs, np.float64).reshape(-1)
    M = len(f)
    q = Q.reshape(-1, N, 3)
    R = q.shape[0]
    if R == 0 or N < 2:
        return np.zeros(lead + (M, 3)), np.zeros(lead)
    seg = np.sqrt(((q[:, 1:] - q[:, :-1]) ** 2).sum(-1))
    cum = np.zeros((R, N))
    np.cumsum(seg, axis=1, out=cum[:, 1:])
    tot = cum[:, -1].copy()
    s = f[None, :] * tot[:, None]
    idx = np.empty((R, M), np.int64)
    step = max(1, (1 << 22) // max(1, M * N))
    for a in range(0, R, step):
        # the LAST vertex whose arc is <= s (zero-length padding segments
        # sit at the end, where the weight below is 0)
        idx[a:a + step] = (cum[a:a + step, None, :]
                           <= s[a:a + step, :, None]).sum(-1) - 1
    np.clip(idx, 0, N - 2, out=idx)
    c0 = np.take_along_axis(cum, idx, 1)
    span = np.take_along_axis(cum, idx + 1, 1) - c0
    w = np.where(span > 0.0, (s - c0) / np.where(span > 0.0, span, 1.0), 0.0)
    rr = np.arange(R)[:, None]
    a0 = q[rr, idx]
    a1 = q[rr, idx + 1]
    pts = a0 + w[..., None] * (a1 - a0)
    return pts.reshape(lead + (M, 3)), tot.reshape(lead)


def _pad_to(X, n):
    """(..., m, 3) -> (..., n, 3), repeating the last point (m <= n)."""
    m = X.shape[-2]
    if m == n:
        return X
    rep = np.repeat(X[..., -1:, :], n - m, axis=-2)
    return np.concatenate([X, rep], axis=-2)


def perturb(A, n, T, p, yaw_deg, nd: int, snap_secs: float,
            vscale: float = VSCALE, lateral_u: float = LATERAL_U):
    """The user's axes on seed segments anchored at the agent.

    ``A`` (B, S, W, 3): seeds, first point == p[b], padded with their last
    valid point; ``n`` (B, S) valid point counts; ``T`` (B, S) flight times
    (s); ``p`` (B, 3); ``yaw_deg`` (B,); ``nd`` = the take-off shift in
    snapshots (SHIFT_S / snap_secs). -> (Q (B, S, 6, W + nd, 3), nq (B, S, 6),
    T6 (B, S, 6), ok (B, S, 6)), the axes in AXES order:

    * earlier: A[nd:] re-anchored at p (the first nd snapshots dropped),
      T - nd x snap; needs n - nd >= 2 points;
    * later: nd more snapshots on the seed's INITIAL velocity (its first
      step), then the seed translated by that lead-in, T + nd x snap;
    * higher / flatter: the vertical deviation from the start -> end chord
      (at each point's arc fraction) x (1 +- vscale), the time x (1 +-
      vscale); the start and the end stay put;
    * left / right: + / - lateral_u x (arc fraction) along the horizontal
      normal of the start -> end chord (the seed's first step if the chord
      is shorter than 1 u, the yaw if that is too): 0 at the agent, the full
      offset at the end; the time unchanged."""
    A = np.asarray(A, np.float64)
    B, S, W, _ = A.shape
    nd = int(nd)
    NQ = W + nd
    n = np.asarray(n, np.int64).reshape(B, S)
    T = np.asarray(T, np.float64).reshape(B, S)
    P = np.asarray(p, np.float64).reshape(B, 1, 1, 3)
    Q = np.empty((B, S, N_AXES, NQ, 3))
    nq = np.empty((B, S, N_AXES), np.int64)
    T6 = np.empty((B, S, N_AXES))
    ok = np.ones((B, S, N_AXES), bool)
    # earlier
    E = np.concatenate([A[:, :, nd:], np.repeat(A[:, :, -1:], nd, axis=2)],
                       axis=2) if nd else A.copy()
    E = E - E[:, :, :1] + P
    Q[:, :, 0] = _pad_to(E, NQ)
    nq[..., 0] = n - nd
    T6[..., 0] = T - nd * float(snap_secs)
    ok[..., 0] = (n - nd) >= 2
    # later
    v0 = A[:, :, 1] - A[:, :, 0]
    lead = P + v0[:, :, None, :] * np.arange(1, nd + 1, dtype=np.float64)[
        None, None, :, None]
    Q[:, :, 1] = np.concatenate([A[:, :, :1], lead,
                                 A[:, :, 1:] + nd * v0[:, :, None, :]], axis=2)
    nq[..., 1] = n + nd
    T6[..., 1] = T + nd * float(snap_secs)
    # the arc fraction of every point (padding: 1)
    seg = np.sqrt(((A[:, :, 1:] - A[:, :, :-1]) ** 2).sum(-1))
    cum = np.concatenate([np.zeros((B, S, 1)), np.cumsum(seg, axis=2)],
                         axis=2)
    tot = cum[..., -1:]
    fr = np.where(tot > 0.0, cum / np.where(tot > 0.0, tot, 1.0), 0.0)
    # higher / flatter: the arc's height over its chord
    z0 = A[:, :, :1, 2]
    chord_z = z0 + fr * (A[:, :, -1:, 2] - z0)
    dev = A[..., 2] - chord_z
    for ax, sc in ((2, 1.0 + float(vscale)), (3, 1.0 - float(vscale))):
        X = A.copy()
        X[..., 2] = chord_z + sc * dev
        Q[:, :, ax] = _pad_to(X, NQ)
        nq[..., ax] = n
        T6[..., ax] = T * sc
    # left / right
    ch = (A[:, :, -1] - A[:, :, 0])[..., :2]
    hl = np.hypot(ch[..., 0], ch[..., 1])
    use_v = hl < 1.0
    vh = v0[..., :2]
    ch = np.where(use_v[..., None], vh, ch)
    hl = np.where(use_v, np.hypot(vh[..., 0], vh[..., 1]), hl)
    yr = np.radians(np.asarray(yaw_deg, np.float64).reshape(B, 1))
    yd = np.stack([np.cos(yr), np.sin(yr)], axis=-1) * np.ones((1, S, 1))
    use_y = hl < 1e-6
    ch = np.where(use_y[..., None], yd, ch)
    hl = np.where(use_y, 1.0, hl)
    u = ch / hl[..., None]
    nrm = np.stack([-u[..., 1], u[..., 0], np.zeros_like(u[..., 0])], axis=-1)
    off = float(lateral_u) * fr[..., None] * nrm[:, :, None, :]
    Q[:, :, 4] = _pad_to(A + off, NQ)
    Q[:, :, 5] = _pad_to(A - off, NQ)
    nq[..., 4] = n
    nq[..., 5] = n
    T6[..., 4] = T
    T6[..., 5] = T
    return Q, nq, T6, ok


def nms_select(pts, valid, informed, seed_col, k: int, k_inf: int,
               eps: float = NMS_U):
    """Greedy eps-NMS over candidates in their (column) priority order - THE
    REFERENCE (ProposalMaker.make's numba kernel and its numpy fallback give
    the same sets, tests/python/test_goal_proposals.py).

    ``pts`` (B, C, M, 3) the candidates' points at matched arc fractions;
    ``valid`` (B, C); ``informed`` (C,) bool; ``seed_col`` (C,) the column of
    a perturbation's seed (-1: none). Column c of env b is ADMITTED when it
    is valid, fewer than k are admitted (fewer than k_inf if it is informed),
    its seed (if any) was admitted, and its mean point distance to EVERY
    admitted candidate is > eps. -> (kept (B, k) column indices, -1 padded,
    n_kept (B,))."""
    pts = np.asarray(pts, np.float64)
    B, C, M, _ = pts.shape
    valid = np.asarray(valid, bool)
    informed = np.asarray(informed, bool).reshape(C)
    seed_col = np.asarray(seed_col, np.int64).reshape(C)
    k = int(k)
    kept = np.full((B, k), -1, np.int64)
    nk = np.zeros(B, np.int64)
    ninf = np.zeros(B, np.int64)
    kp = np.zeros((B, k, M, 3))
    col_in = np.zeros((B, C), bool)
    ar = np.arange(k)
    col_any = valid.any(0)
    for c in range(C):
        if not col_any[c]:
            continue
        cand = valid[:, c] & (nk < k)
        if informed[c]:
            cand &= ninf < int(k_inf)
        if seed_col[c] >= 0:
            cand &= col_in[:, seed_col[c]]
        rows = np.flatnonzero(cand)
        if not len(rows):
            continue
        m = nk[rows]
        mm = int(m.max())
        if mm > 0:
            # only the admitted slots (0 .. mm-1) are compared
            dif = kp[rows, :mm] - pts[rows, c][:, None]
            d = np.sqrt(np.einsum("rkmi,rkmi->rkm", dif, dif)).mean(-1)
            d[ar[None, :mm] >= m[:, None]] = np.inf
            ok = d.min(1) > float(eps)
            rows = rows[ok]
            if not len(rows):
                continue
            m = m[ok]
        kp[rows, m] = pts[rows, c]
        kept[rows, m] = c
        col_in[rows, c] = True
        nk[rows] += 1
        if informed[c]:
            ninf[rows] += 1
    return kept, nk


def _build_fast_select():
    """ProposalMaker.make's selection in ONE numba pass per env: the greedy
    over the informed columns (validity, the seed rule, the informed cap,
    the mean point distance to every admitted one), then the shapes in their
    drawn order against the admitted informed candidates (points) and the
    admitted shapes (L x the unit distance), until K. The same rule as
    nms_select, with early exits; single-threaded (a few hundred candidates,
    and a thread pool would take CPU from the trainer mid-rollout)."""
    try:
        from numba import njit
    except Exception:
        return None

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _select(pts_i, valid_i, seed_col, pts_u, lu, dunit, shapes, k,
                k_inf, eps, kept_i, nk_i, kept_u):
        B = pts_u.shape[0]
        ci_n = pts_i.shape[1]
        nu = pts_u.shape[1]
        M = pts_u.shape[2]
        for b in range(B):
            col_in = np.zeros(ci_n + 1, np.bool_)
            n = 0
            for c in range(ci_n):
                if n >= k_inf:
                    break
                if not valid_i[b, c]:
                    continue
                sc = seed_col[c]
                if sc >= 0 and not col_in[sc]:
                    continue
                ok = True
                for j in range(n):
                    cj = kept_i[b, j]
                    acc = 0.0
                    for m in range(M):
                        dx = pts_i[b, c, m, 0] - pts_i[b, cj, m, 0]
                        dy = pts_i[b, c, m, 1] - pts_i[b, cj, m, 1]
                        dz = pts_i[b, c, m, 2] - pts_i[b, cj, m, 2]
                        acc += np.sqrt(dx * dx + dy * dy + dz * dz)
                    if acc / M <= eps:
                        ok = False
                        break
                if ok:
                    kept_i[b, n] = c
                    col_in[c] = True
                    n += 1
            nk_i[b] = n
            tot = n
            sh = np.empty(nu, np.int64)
            nsh = 0
            for s in range(nu):
                if tot >= k:
                    break
                ok = True
                for j in range(n):
                    cj = kept_i[b, j]
                    acc = 0.0
                    for m in range(M):
                        dx = pts_u[b, s, m, 0] - pts_i[b, cj, m, 0]
                        dy = pts_u[b, s, m, 1] - pts_i[b, cj, m, 1]
                        dz = pts_u[b, s, m, 2] - pts_i[b, cj, m, 2]
                        acc += np.sqrt(dx * dx + dy * dy + dz * dz)
                    if acc / M <= eps:
                        ok = False
                        break
                if ok:
                    for t in range(nsh):
                        if lu[b] * dunit[shapes[b, s], shapes[b, sh[t]]]                                 <= eps:
                            ok = False
                            break
                if ok:
                    kept_u[b, s] = True
                    sh[nsh] = s
                    nsh += 1
                    tot += 1
    return _select


# SURFGYM_NO_NUMBA=1 forces the numpy route (the same switch goalarc.py,
# route.py and goalfield.py use)
_FAST_SELECT = None if os.environ.get("SURFGYM_NO_NUMBA") == "1"     else _build_fast_select()


# ==========================================================================
# the proposal machine
# ==========================================================================
class Proposals:
    """One decision's candidate sets for B envs, K slots each (``mask``
    marks the valid ones). Arrays are (B, K, ...) in admission order."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def B(self) -> int:
        return int(len(self.p))

    def line(self, b: int, c: int, vocab, surf: bool):
        """Candidate c of env b as a plan line -> (line float32 (L, 3), time
        in seconds or None for a shape (its vocabulary's own budget), arc
        length). A shape is its vocabulary's own anchored line (exactly the
        learned planner's); a hindsight or perturbed polyline is resampled at
        the fan spacing like the diet's hindsight lines."""
        from .goals import resample_polyline_np
        b, c = int(b), int(c)
        if not self.mask[b, c]:
            raise ValueError(f"candidate {c} of env {b} is padding")
        if int(self.src[b, c]) == SRC_UNIF:
            k = int(self.shape[b, c])
            if surf:
                ln = vocab.anchor(k, self.p[b], float(self.Lu[b]))
            else:
                ln = vocab.anchor(k, self.p[b])
            return ln, None, float(self.L[b, c])
        q = self.Q[b, c, :int(self.nq[b, c])]
        ln = resample_polyline_np(q)
        if len(ln) > HS_L_MAX:
            ln = ln[:HS_L_MAX]
        arc = float(np.linalg.norm(np.diff(ln.astype(np.float64), axis=0),
                                   axis=1).sum())
        return ln, float(self.T[b, c]), arc

    def ident(self, b: int, c: int) -> tuple:
        """A hashable identity of candidate c of env b (for 'distinct')."""
        s = int(self.src[b, c])
        if s == SRC_UNIF:
            return (s, int(self.shape[b, c]))
        o = np.round(self.seed_org[b, c]).astype(np.int64)
        return (s, int(self.axis[b, c]), int(o[0]), int(o[1]), int(o[2]))


class ProposalMaker:
    """Builds the candidate sets (see the module docstring) for a batch of
    agent states: ``make(bank, p, v, yaw, rng) -> Proposals``.

    ``make`` is the fast path: the greedy NMS loop runs over the INFORMED
    columns only, and the shapes are admitted with precomputed geometry (a
    shape's arc-fraction points are p + L x its unit points, and two shapes
    of one decision share p and L, so their mean distance is L x a constant).
    ``make_reference`` runs every column through :func:`nms_select`; the two
    give the same sets (tests/python/test_goal_proposals.py)."""

    def __init__(self, vocab, spec: dict):
        self.vocab = vocab
        self.spec = dict(spec)
        self.surf = spec.get("base") == "surf"
        self.k = int(spec["k"])
        self.k_hs = max(1, self.k // 2)
        self.k_inf = max(1, self.k // 2)
        self.k_seed = max(1, self.k // 4)
        self.radius = float(spec["hs_radius"])
        self.snap = float(spec["snap_secs"])
        self.nd = max(1, int(round(float(spec["shift_s"]) / self.snap)))
        self.vscale = float(spec["vscale"])
        self.lateral = float(spec["lateral_u"])
        self.eps = float(spec["nms_u"])
        self.n_pts = int(spec["n_pts"])
        self.fracs = (np.arange(1, self.n_pts + 1, dtype=np.float64)
                      / self.n_pts)
        # the base vocabulary's shapes as unit polylines (surf: arc 1, scaled
        # per plan) or fixed ones (walk), and the time a shape stands for
        if self.surf:
            self.shape_pts = np.asarray(vocab.unit, np.float64)
            self.shape_t = float(vocab.t_plan)
            unit = self.shape_pts
        else:
            self.shape_pts = np.asarray(vocab.raw, np.float64)
            self.shape_t = float(vocab.length) / BUDGET_SPEED_U
            unit = self.shape_pts / float(vocab.length)
        # per unit length: the shapes' points at the fractions, and their
        # pairwise mean point distances
        self.shape_u8 = arc_samples(unit, self.fracs)[0]         # (Kv, M, 3)
        u8 = self.shape_u8
        self.shape_dunit = np.sqrt(((u8[:, None] - u8[None]) ** 2).sum(-1)
                                   ).mean(-1)                    # (Kv, Kv)

    def describe(self) -> str:
        return (f"{self.k} candidates per decision: up to {self.k_inf} "
                f"informed - hindsight segments of the policy's own flights "
                f"within {self.radius:g} u of (position, velocity x "
                f"{float(self.spec['hs_tau_s']):g} s), re-anchored at the "
                f"agent, and perturbations of the {self.k_seed} nearest "
                f"(take off {self.nd * self.snap:g} s earlier / later, "
                f"higher-slower / flatter-faster x{1 + self.vscale:g} / "
                f"x{1 - self.vscale:g} arc height and time, lateral "
                f"+-{self.lateral:g} u at the end) - the rest uninformed "
                f"{'surf' if self.surf else 'walking'} vocabulary shapes; "
                f"eps-NMS {self.eps:g} u mean over {self.n_pts} points; "
                "nothing filters by geometry")

    # ------------------------------------------------------------ sources
    def _informed(self, bank: Optional[HindsightBank], p, v, yaw):
        """The informed columns in priority order - the k_seed nearest
        hindsight segments, their perturbations (seed-major), the remaining
        hindsight segments - as a dict of (B, C, ...) arrays with their arc
        samples, or None when no env has a hindsight segment within reach."""
        if bank is None or not bank.n:
            return None
        B = len(p)
        nd = self.nd
        d, r = bank.query(p, v, self.k_hs, self.radius)
        okh = r >= 0
        if not okh.any():
            return None
        rc = np.where(okh, r, 0)
        # the longest segment within reach sets the width (bank segments
        # are padded with their last point to the reservoir's 64, and all
        # below is linear in the width)
        n = bank.seglen[rc]
        W = int(n[okh].max())
        NQ = W + nd
        n = np.where(okh, n, 2)                # a miss: never valid anyway
        Sg = bank.segs[rc, :W].astype(np.float64)
        A = Sg - Sg[:, :, :1] + p[:, None, None, :]
        T = (n - 1) * self.snap
        match = np.where(okh, d / self.radius, 1.0)
        org = bank.org[rc]
        ks = min(self.k_seed, self.k_hs)
        Qp, nqp, Tp, okp = perturb(A[:, :ks], n[:, :ks], T[:, :ks], p, yaw,
                                   nd, self.snap, self.vscale, self.lateral)
        okp &= okh[:, :ks, None]
        nr = self.k_hs - ks
        Q = np.concatenate([_pad_to(A[:, :ks], NQ),
                            Qp.reshape(B, ks * N_AXES, NQ, 3),
                            _pad_to(A[:, ks:], NQ)], axis=1)

        def cat3(a, b, c):
            return np.concatenate([a, b, c], axis=1)
        out = {
            "Q": Q,
            "nq": cat3(n[:, :ks], nqp.reshape(B, -1), n[:, ks:]),
            "T": cat3(T[:, :ks], Tp.reshape(B, -1), T[:, ks:]),
            "ok": cat3(okh[:, :ks], okp.reshape(B, -1), okh[:, ks:]),
            "src": np.concatenate([np.full(ks, SRC_HS),
                                   np.full(ks * N_AXES, SRC_PERT),
                                   np.full(nr, SRC_HS)]).astype(np.int64),
            "axis": np.concatenate([np.full(ks, -1),
                                    np.tile(np.arange(N_AXES), ks),
                                    np.full(nr, -1)]).astype(np.int64),
            # a perturbation's seed is column s of this block (the seeds
            # come first)
            "seed_col": np.concatenate([np.full(ks, -1),
                                        np.repeat(np.arange(ks), N_AXES),
                                        np.full(nr, -1)]).astype(np.int64),
            "match": cat3(match[:, :ks], np.repeat(match[:, :ks], N_AXES,
                                                   axis=1), match[:, ks:]),
            "org": cat3(org[:, :ks], np.repeat(org[:, :ks], N_AXES, axis=1),
                        org[:, ks:])}
        C = Q.shape[1]
        pts = np.zeros((B, C, self.n_pts, 3))
        tot = np.zeros((B, C))
        ok = out["ok"]
        pts[ok], tot[ok] = arc_samples(Q[ok], self.fracs)
        out.update(pts=pts, tot=tot, valid=ok & (tot >= MIN_LEN_U))
        return out

    def _shapes(self, p, v, rng):
        """The uninformed pool: up to 2K distinct base-vocabulary shapes per
        env, uniform without replacement (in the drawn order), sized for the
        current speed -> (shapes (B, nu), lengths (B,), points (B, nu, M,
        3))."""
        B = len(p)
        Kv = int(self.shape_pts.shape[0])
        nu = min(2 * self.k, Kv)
        shapes = np.argsort(rng.random((B, Kv)), axis=1)[:, :nu]
        if self.surf:
            Lu = self.vocab.length_for(np.hypot(v[:, 0], v[:, 1]))
        else:
            Lu = np.full(B, float(self.vocab.length))
        pts = (p[:, None, None, :]
               + Lu[:, None, None, None] * self.shape_u8[shapes])
        return shapes, Lu, pts

    def _shape_poly(self, p, Lu, shapes):
        """(..., 3) origins, (...) lengths, (...) shapes -> the shapes' own
        9-point polylines (..., 9, 3)."""
        if self.surf:
            return (p[..., None, :]
                    + Lu[..., None, None] * self.shape_pts[shapes])
        return p[..., None, :] + self.shape_pts[shapes]

    def _select_numba(self, inf, shapes, Lu, Pu):
        B, nu, M = Pu.shape[0], Pu.shape[1], Pu.shape[2]
        if inf is not None:
            pts_i = np.ascontiguousarray(inf["pts"], np.float64)
            valid_i = np.ascontiguousarray(inf["valid"], np.bool_)
            seed_col = np.ascontiguousarray(inf["seed_col"], np.int64)
        else:
            pts_i = np.zeros((B, 0, M, 3))
            valid_i = np.zeros((B, 0), np.bool_)
            seed_col = np.zeros(0, np.int64)
        kept_i = np.full((B, self.k), -1, np.int64)
        nk_i = np.zeros(B, np.int64)
        kept_u = np.zeros((B, nu), np.bool_)
        _FAST_SELECT(pts_i, valid_i, seed_col,
                     np.ascontiguousarray(Pu, np.float64),
                     np.ascontiguousarray(Lu, np.float64), self.shape_dunit,
                     np.ascontiguousarray(shapes, np.int64), int(self.k),
                     int(self.k_inf), float(self.eps), kept_i, nk_i, kept_u)
        return kept_i, nk_i, kept_u

    def _select_numpy(self, inf, shapes, Lu, Pu):
        B, nu = Pu.shape[0], Pu.shape[1]
        K = self.k
        bb = np.arange(B)[:, None]
        if inf is not None:
            C = len(inf["src"])
            kept_i, nk_i = nms_select(inf["pts"], inf["valid"],
                                      np.ones(C, bool), inf["seed_col"], K,
                                      self.k_inf, self.eps)
        else:
            kept_i = np.full((B, K), -1, np.int64)
            nk_i = np.zeros(B, np.int64)
        blocked = np.zeros((B, nu), bool)
        mi = int(nk_i.max()) if B else 0
        if mi:
            Pk = inf["pts"][bb, np.maximum(kept_i[:, :mi], 0)]  # (B,mi,M,3)
            D = np.sqrt(((Pk[:, :, None] - Pu[:, None]) ** 2).sum(-1)
                        ).mean(-1)                               # (B,mi,nu)
            D[np.arange(mi)[None, :] >= nk_i[:, None]] = np.inf
            blocked = (D <= self.eps).any(1)
        near = (Lu[:, None, None] * self.shape_dunit[shapes[:, :, None],
                                                     shapes[:, None, :]]
                <= self.eps)
        near[:, np.arange(nu), np.arange(nu)] = False
        free = ~blocked
        if not near.any():
            kept_u = free & (nk_i[:, None] + np.cumsum(free, axis=1) <= K)
        else:
            kept_u = np.zeros((B, nu), bool)
            cnt = nk_i.copy()
            for s in range(nu):
                ok = free[:, s] & (cnt < K)
                if s:
                    ok &= ~(near[:, s, :s] & kept_u[:, :s]).any(1)
                kept_u[:, s] = ok
                cnt += ok
        return kept_i, nk_i, kept_u

    # ------------------------------------------------------------- make
    def make(self, bank: Optional[HindsightBank], p, v, yaw_deg,
             rng) -> Proposals:
        p = np.atleast_2d(np.asarray(p, np.float64))
        v = np.atleast_2d(np.asarray(v, np.float64))
        yaw = np.asarray(yaw_deg, np.float64).reshape(-1)
        B, K, M = len(p), self.k, self.n_pts
        inf = self._informed(bank, p, v, yaw)
        shapes, Lu, Pu = self._shapes(p, v, rng)
        # ---- the selection: the informed columns by the greedy rule (seed
        # rule, caps), then the shapes in their drawn order - dropped within
        # eps of an admitted informed candidate or of an admitted EARLIER
        # shape (L x the unit distance), admitted while fewer than K are in
        if _FAST_SELECT is not None:
            kept_i, nk_i, kept_u = self._select_numba(inf, shapes, Lu, Pu)
        else:
            kept_i, nk_i, kept_u = self._select_numpy(inf, shapes, Lu, Pu)
        nk = nk_i + kept_u.sum(1)
        # ---- assemble (B, K) in admission order: informed, then shapes
        NQ = max(inf["Q"].shape[2] if inf is not None else 0,
                 self.shape_pts.shape[1])
        Q = np.zeros((B, K, NQ, 3))
        nq = np.zeros((B, K), np.int64)
        T = np.zeros((B, K))
        L = np.zeros((B, K))
        src = np.full((B, K), -1, np.int64)
        axis = np.full((B, K), -1, np.int64)
        shape = np.full((B, K), -1, np.int64)
        match = np.ones((B, K))
        org = np.zeros((B, K, 3))
        pts = np.zeros((B, K, M, 3))
        if inf is not None:
            bi, ji = np.nonzero(kept_i >= 0)
            ci = kept_i[bi, ji]
            Q[bi, ji] = _pad_to(inf["Q"][bi, ci], NQ)
            nq[bi, ji] = inf["nq"][bi, ci]
            T[bi, ji] = inf["T"][bi, ci]
            L[bi, ji] = inf["tot"][bi, ci]
            src[bi, ji] = inf["src"][ci]
            axis[bi, ji] = inf["axis"][ci]
            match[bi, ji] = inf["match"][bi, ci]
            org[bi, ji] = inf["org"][bi, ci]
            pts[bi, ji] = inf["pts"][bi, ci]
        bu, su = np.nonzero(kept_u)
        ju = (nk_i[:, None] + np.cumsum(kept_u, axis=1) - 1)[bu, su]
        sh = shapes[bu, su]
        Q[bu, ju] = _pad_to(self._shape_poly(p[bu], Lu[bu], sh), NQ)
        nq[bu, ju] = self.shape_pts.shape[1]
        T[bu, ju] = self.shape_t
        L[bu, ju] = Lu[bu]
        src[bu, ju] = SRC_UNIF
        shape[bu, ju] = sh
        org[bu, ju] = p[bu]
        pts[bu, ju] = Pu[bu, su]
        out = Proposals(p=p, yaw=yaw, mask=src >= 0, n_valid=nk, Lu=Lu,
                        Q=Q, nq=nq, T=T, L=L, src=src, axis=axis,
                        shape=shape, match=match, seed_org=org, pts=pts,
                        hs_found=inf is not None)
        out.feats = self.features(out)
        return out

    def make_reference(self, bank: Optional[HindsightBank], p, v, yaw_deg,
                       rng) -> Proposals:
        """The same candidate sets by the generic route: every column (the
        shapes as polylines) arc-sampled and run through nms_select."""
        p = np.atleast_2d(np.asarray(p, np.float64))
        v = np.atleast_2d(np.asarray(v, np.float64))
        yaw = np.asarray(yaw_deg, np.float64).reshape(-1)
        B = len(p)
        inf = self._informed(bank, p, v, yaw)
        shapes, Lu, _ = self._shapes(p, v, rng)
        nu = shapes.shape[1]
        U = self._shape_poly(p[:, None, :], Lu[:, None], shapes)
        NQ = max(inf["Q"].shape[2] if inf is not None else 0, U.shape[2])
        cols = {"Q": [_pad_to(U, NQ)],
                "nq": [np.full((B, nu), U.shape[2], np.int64)],
                "T": [np.full((B, nu), self.shape_t)],
                "ok": [np.ones((B, nu), bool)],
                "src": [np.full(nu, SRC_UNIF, np.int64)],
                "axis": [np.full(nu, -1, np.int64)],
                "seed_col": [np.full(nu, -1, np.int64)],
                "match": [np.ones((B, nu))],
                "org": [np.repeat(p[:, None, :], nu, axis=1)],
                "shape": [shapes]}
        if inf is not None:
            ci = inf["src"].shape[0]
            for k in cols:
                x = (inf[k] if k != "shape" else
                     np.full((B, ci), -1, np.int64))
                if k == "Q":
                    x = _pad_to(x, NQ)
                cols[k].insert(0, x)
        cat = {k: (np.concatenate(x) if k in ("src", "axis", "seed_col")
                   else np.concatenate(x, axis=1)) for k, x in cols.items()}
        C = len(cat["src"])
        pts = np.zeros((B, C, self.n_pts, 3))
        tot = np.zeros((B, C))
        ok = cat["ok"]
        pts[ok], tot[ok] = arc_samples(cat["Q"][ok], self.fracs)
        valid = ok & (tot >= MIN_LEN_U)
        kept, nk = nms_select(pts, valid, cat["src"] != SRC_UNIF,
                              cat["seed_col"], self.k, self.k_inf, self.eps)
        mask = kept >= 0
        ci = np.where(mask, kept, 0)
        bb = np.arange(B)[:, None]
        out = Proposals(
            p=p, yaw=yaw, mask=mask, n_valid=nk, Lu=Lu,
            Q=cat["Q"][bb, ci], nq=cat["nq"][bb, ci], T=cat["T"][bb, ci],
            L=tot[bb, ci], src=np.where(mask, cat["src"][ci], -1),
            axis=np.where(mask, cat["axis"][ci], -1),
            shape=np.where(mask, cat["shape"][bb, ci], -1),
            match=cat["match"][bb, ci], seed_org=cat["org"][bb, ci],
            pts=pts[bb, ci], hs_found=inf is not None)
        out.feats = self.features(out)
        return out

    def features(self, pr: Proposals) -> np.ndarray:
        """(B, K, N_FEAT) float32: the 8 points relative to the agent in the
        world-aligned frame and in the yaw frame (/ 1000 u), log1p(arc /
        1000 u), time / 3 s, the source one-hot, the match distance / radius;
        zero on masked slots."""
        B, K = pr.mask.shape
        rel = (pr.pts - pr.p[:, None, None, :]) / POS_SCALE_U     # (B,K,M,3)
        yr = np.radians(pr.yaw)[:, None, None]
        cy, sy = np.cos(yr), np.sin(yr)
        ego = np.stack([rel[..., 0] * cy + rel[..., 1] * sy,
                        -rel[..., 0] * sy + rel[..., 1] * cy,
                        rel[..., 2]], axis=-1)
        oh = np.zeros((B, K, 3))
        s = np.clip(pr.src, 0, 2)
        np.put_along_axis(oh, s[..., None], 1.0, axis=2)
        f = np.concatenate([rel.reshape(B, K, -1), ego.reshape(B, K, -1),
                            np.log1p(np.maximum(pr.L, 0.0)
                                     / POS_SCALE_U)[..., None],
                            (pr.T / T_REF_S)[..., None], oh,
                            np.clip(pr.match, 0.0, 1.0)[..., None]], axis=-1)
        f = np.clip(f, -FEAT_CLIP, FEAT_CLIP)
        f[~pr.mask] = 0.0
        return np.ascontiguousarray(f, np.float32)


# ==========================================================================
# the network
# ==========================================================================
class PointerNet(nn.Module):
    """The stage-3 observation encoder (patch -> 3 stride-2 convs; scalars
    -> 64; both -> ``hidden``) and a candidate encoder (N_FEAT -> 128 ->
    128); logit_i = u . tanh(W_h h + W_e e_i) over the VALID candidates
    (masked slots get MASK_NEG), value = v([h, masked mean of e_i])."""

    def __init__(self, n_feat: int = N_FEAT, patch_n: int = PATCH_N,
                 n_scal: int = N_SCAL, hidden: int = NET_HIDDEN,
                 in_ch: int = 2, cand_hidden: int = CAND_HIDDEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(int(in_ch), 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        side = int(patch_n)
        for _ in range(3):
            side = (side + 1) // 2
        ch = int(cand_hidden)
        self.scal = nn.Sequential(nn.Linear(int(n_scal), 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(32 * side * side + 64,
                                             int(hidden)), nn.ReLU())
        self.cand = nn.Sequential(nn.Linear(int(n_feat), ch), nn.ReLU(),
                                  nn.Linear(ch, ch), nn.ReLU())
        self.wh = nn.Linear(int(hidden), ch)
        self.we = nn.Linear(ch, ch, bias=False)
        self.u = nn.Linear(ch, 1)
        self.v = nn.Linear(int(hidden) + ch, 1)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # a near-uniform first policy over the valid candidates
        nn.init.orthogonal_(self.u.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, img, scal, cand, mask):
        h = self.trunk(torch.cat([self.conv(img), self.scal(scal)], dim=1))
        e = self.cand(cand)                                      # (B, K, ch)
        z = torch.tanh(self.wh(h)[:, None, :] + self.we(e))
        logits = self.u(z).squeeze(-1)                           # (B, K)
        m = mask.to(torch.bool)
        logits = logits.masked_fill(~m, MASK_NEG)
        mf = m.to(e.dtype)[..., None]
        pooled = (e * mf).sum(1) / mf.sum(1).clamp_min(1.0)
        v = self.v(torch.cat([h, pooled], dim=1)).squeeze(-1)
        return logits, v


def pointer_logp(net, img, scal, cand, mask, act):
    """-> (log-prob of ``act`` (B,), all log-probs (B, K), value (B,)): PPO's
    ratio (update) on the stored candidate set - the arithmetic plan()
    samples the choice and records its log-prob with."""
    logits, v = net(img, scal, cand, mask)
    lp_all = F.log_softmax(logits.float(), dim=-1)
    lp = lp_all.gather(1, act.reshape(-1, 1)).squeeze(1)
    return lp, lp_all, v


# ==========================================================================
# diagnostics
# ==========================================================================
def _cand_diag(pr: Proposals, rows, cols, wm, surf: bool, kill_z: float,
               samples: int = DIAG_SAMPLES):
    """The chosen-plan wall diagnostics on candidates (rows[i], cols[i]) ->
    (crosses, frac, void): surf - a sample inside SOLID (samples within
    SKIP_CELLS cells of arc of the start dropped, goalsurf.line_crossing's
    rule), the share of samples in it, the END below the kill ceiling; walk -
    a sample on a non-walkable graph column at the start's layer (samples
    within one graph cell of arc dropped, goallearn's rule), the share, and
    void False."""
    rows = np.asarray(rows, np.int64).reshape(-1)
    cols = np.asarray(cols, np.int64).reshape(-1)
    if not len(rows):
        z = np.zeros(0, bool)
        return z, np.zeros(0), z
    f = (np.arange(int(samples)) + 1.0) / int(samples)
    Q = pr.Q[rows, cols]
    pts, tot = arc_samples(Q, f)
    arc = f[None, :] * tot[:, None]
    end_z = Q[:, -1, 2]
    if surf:
        keep = arc >= SKIP_CELLS * wm.cell
        bad = wm.solid_at(pts) & keep
        void = end_z < float(kill_z)
    else:
        keep = arc > wm.cell
        iz = wm.layer(pr.p[rows])
        bad = wm.offgraph(iz, pts[..., :2]) & keep
        void = np.zeros(len(rows), bool)
    n = np.maximum(keep.sum(-1), 1)
    return bad.any(-1), bad.sum(-1) / n, void


# ==========================================================================
# the trainable planner over proposals
# ==========================================================================
class ProposalPlanner(LearnedPlanner):
    """--goal-planner learned --plan-vocab proposals: the learned planner
    (goallearn.LearnedPlanner - its plan bookkeeping, rewards, novelty, PPO
    constants, logging and checkpoint format) choosing among per-decision
    candidate sets (:class:`ProposalMaker`) with a pointer head
    (:class:`PointerNet`). ``spec`` is a :func:`prop_spec`."""

    proposals = True

    def __init__(self, graph, n_envs: int, device, *, start_pts=None,
                 tick_ms: float = 10.0, act_every: int = 1,
                 corridor: float = 192.0, cfg: Optional[dict] = None,
                 seed: int = 0, spec: Optional[dict] = None):
        # NOT LearnedPlanner.__init__: the action set, the network and the
        # per-decision record differ; every attribute its reused methods
        # read (request, on_tick, n_ready, set_tick_ms, _wm_for, _cells,
        # state_dict_all, load_state_dict_all, note_and_row) is set here
        if graph.finish_center is None:
            raise ValueError("the learned planner needs the map's finish box")
        if spec is None or spec.get("vocab") != "proposals":
            raise ValueError("ProposalPlanner needs a prop_spec()")
        self.device = torch.device(device)
        self.graph = graph
        self.spec = dict(spec)
        self.cfg = dict(PLAN_DEFAULTS)
        for k, v in (cfg or {}).items():
            if v is not None:
                self.cfg[k] = v
        self.base = str(self.spec["base"])
        self.surf = self.base == "surf"
        bspec = base_vocab_spec(self.spec)
        self.vocab = make_vocab(bspec if self.surf else
                                {k: bspec[k] for k in _spec_default()})
        self.K = int(self.spec["k"])
        if self.surf:
            self.wm = SlabMap.for_graph(graph, self.spec)
        else:
            self.wm = WalkMap(graph, self.spec["patch_cell_u"],
                              self.spec["patch_n"], self.spec["patch_layers"])
        self._wms = {id(graph): self.wm}
        self.in_ch = int(self.spec.get("in_ch", 2))
        self.kill_z = float(getattr(graph, "kill_z", -np.inf))
        self.net = PointerNet(int(self.spec["n_feat"]), self.spec["patch_n"],
                              self.spec["n_scal"], self.spec["hidden"],
                              in_ch=self.in_ch,
                              cand_hidden=int(self.spec["cand_hidden"])
                              ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(),
                                    lr=float(self.cfg["plan_lr"]), eps=1e-5)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))
        self.np_rng = np.random.default_rng(int(seed))
        self.prop_rng = np.random.default_rng(int(seed) + PROP_SEED_OFFSET)
        self.eval_seed = int(seed) + PROP_EVAL_SEED_OFFSET
        self.n = int(n_envs)
        self.act_every = max(1, int(act_every))
        self.corridor = float(corridor)
        self.tick_ms = float(tick_ms)
        # hindsight and perturbed lines are longer than any shape
        self.st = PlanState(self.n, self.wm, self.vocab, tick_ms, corridor,
                            l_max=HS_L_MAX)
        self.finish = np.asarray(graph.finish_center, np.float64)
        self.start_pts = (None if start_pts is None else
                          np.atleast_2d(np.asarray(start_pts, np.float64)))
        pn = int(self.spec["patch_n"])
        nf = int(self.spec["n_feat"])
        self.o_img = np.zeros((self.n, self.in_ch, pn, pn), np.float32)
        self.o_scal = np.zeros((self.n, int(self.spec["n_scal"])), np.float32)
        self.o_cand = np.zeros((self.n, self.K, nf), np.float32)
        self.o_mask = np.zeros((self.n, self.K), bool)
        self.o_act = np.zeros(self.n, np.int64)
        self.o_logp = np.zeros(self.n, np.float32)
        self.o_val = np.zeros(self.n, np.float32)
        self.o_d0 = np.zeros(self.n, np.float64)
        self.o_src = np.zeros(self.n, np.int64)
        self.fresh = np.ones(self.n, bool)
        self.from_start = np.zeros(self.n, bool)
        self.buf = [[] for _ in range(self.n)]
        # the candidate set + mask of every closed transition, in step with
        # self.buf (LearnedPlanner.on_tick appends one tuple per closed plan)
        self.buf_x = [[] for _ in range(self.n)]
        nz, ny, nx = np.asarray(graph.node_of).shape
        ext = np.array([nx, ny, nz], np.float64) * float(graph.cell)
        self.nov_mins = np.asarray(graph.mins, np.float64)
        self.nov_shape = tuple(int(v) for v in
                               np.maximum(1, np.ceil(ext / NOVELTY_CELL_U)))
        self.nov_count = np.zeros(self.nov_shape, np.int32)
        self.cover = np.zeros(self.nov_shape, bool)
        self.updates = 0
        self.maker = ProposalMaker(self.vocab, self.spec)
        self.bank = None
        self._bank_harv = None
        self._reset_window()
        self.last_upd = None
        self.last_eval = None

    # ------------------------------------------------------------ helpers
    @property
    def bank_n(self) -> int:
        return 0 if self.bank is None else int(self.bank.n)

    def describe(self) -> str:
        c = self.cfg
        obs = (f"{self.wm.describe()} + this episode's visits" if self.surf
               else f"walkable patch within +-{self.spec['patch_layers']} "
                    "graph layer(s) + this episode's visits")
        return (f"planner LEARNED over TRAJECTORY PROPOSALS: "
                f"{self.maker.describe()}; POINTER head over the valid "
                f"candidates (features: 8 points world + ego / "
                f"{POS_SCALE_U:g} u, arc, time, source, match), obs {obs}, "
                f"{self.spec['n_scal']} scalars; a plan closes on arc >= "
                f"{COMPLETE_FRAC:g} (corridor {self.corridor:g} u), on its "
                f"budget ({BUDGET_MULT:g} x its time) or on the episode's "
                f"end; reward {float(c['plan_r_ok']):+g}/"
                f"{float(c['plan_r_fail']):+g} executed / not, "
                f"+{c['plan_finish_bonus']:g} finish, novelty "
                f"{c['plan_novelty']:g}/sqrt(n) (none on a death)"
                + (f", progress {c['plan_progress']:g}/1000 u"
                   if float(c["plan_progress"]) else "")
                + f"; PPO lr {float(c['plan_lr']):g} ent "
                f"{float(c['plan_ent']):g} epochs {int(c['plan_epochs'])} "
                f"batch >= {int(c['plan_batch'])} plans, "
                f"{sum(p.numel() for p in self.net.parameters()):,} params; "
                f"hindsight bank {self.bank_n:,} rows")

    def _reset_window(self) -> None:
        super()._reset_window()
        self.w.update({"cand_n": 0, "cand_src": np.zeros(3, np.int64),
                       "chosen_src": np.zeros(3, np.int64),
                       "closed_src": np.zeros(3, np.int64),
                       "complete_src": np.zeros(3, np.int64),
                       "ids": set(), "hs_found": 0})

    # --------------------------------------------------- the hindsight bank
    def use_bank(self, bank: Optional[HindsightBank]) -> None:
        self.bank = bank if (bank is not None and bank.n) else None

    def set_reservoir(self, respawn, force: bool = False) -> None:
        """Rebuild the hindsight bank from the live reservoir (goalsys
        iterate, every iteration) once BANK_REFRESH_FRAC of it has been
        harvested anew, or on the first call."""
        if respawn is None or getattr(respawn, "goal_k", None) is None:
            return
        h = int(getattr(respawn, "harvested", 0))
        if (not force and self._bank_harv is not None
                and h - self._bank_harv
                < BANK_REFRESH_FRAC * max(1, int(respawn.size))):
            return
        self._bank_harv = h
        self.use_bank(HindsightBank.from_reservoir(
            respawn, tau_s=float(self.spec["hs_tau_s"])))

    # --------------------------------------------------------- the fleet
    def _obs(self, wm, visits, idx, p, v, y):
        cap = int(self.spec["visit_cap"])
        if self.surf:
            return build_obs_surf(wm, visits, idx, p, v, y, self.finish, cap)
        return build_obs(wm, visits, idx, p, v, y, self.finish, cap)

    def on_tick(self, pos, ended, finished, died, term_pos=None) -> None:
        was = self.st.active.copy()
        super().on_tick(pos, ended, finished, died, term_pos)
        closed = was & ~self.st.active
        if not closed.any():
            return
        ci = np.flatnonzero(closed)
        # the completion rule of PlanState.tick, re-read (the tracker keeps
        # its arc until the next plan is begun)
        tr = self.st.track
        comp = (~np.asarray(ended, bool)[ci]
                & (tr.arc[ci] >= COMPLETE_FRAC * tr.total_arc()[ci]))
        for i in ci:
            self.buf_x[i].append((self.o_cand[i].copy(),
                                  self.o_mask[i].copy()))
        s = self.o_src[ci]
        np.add.at(self.w["closed_src"], s, 1)
        np.add.at(self.w["complete_src"], s[comp], 1)

    def plan(self, pos, vel, yaw_deg):
        """Candidate sets for every env waiting for a plan, ONE batched
        forward, a sample from the pointer softmax -> (idx, lines, fresh)."""
        idx = np.flatnonzero(self.st.need)
        if not len(idx):
            return idx, [], np.zeros(0, bool)
        p = np.asarray(pos, np.float64)[idx]
        v = np.asarray(vel, np.float64)[idx]
        y = np.asarray(yaw_deg, np.float64).reshape(-1)[idx]
        img, scal = self._obs(self.wm, self.st.visits, idx, p, v, y)
        pr = self.maker.make(self.bank, p, v, y, self.prop_rng)
        dev = self.device
        with torch.no_grad():
            logits, val = self.net(torch.as_tensor(img, device=dev),
                                   torch.as_tensor(scal, device=dev),
                                   torch.as_tensor(pr.feats, device=dev),
                                   torch.as_tensor(pr.mask, device=dev))
            logits = logits.float()
            a = torch.multinomial(torch.softmax(logits, dim=-1), 1,
                                  generator=self.gen).squeeze(1)
            lp = F.log_softmax(logits, dim=-1).gather(
                1, a.unsqueeze(1)).squeeze(1)
            ent = _entropy(logits)
        a_np = a.cpu().numpy().astype(np.int64)
        B = len(idx)
        self.o_img[idx] = img
        self.o_scal[idx] = scal
        self.o_cand[idx] = pr.feats
        self.o_mask[idx] = pr.mask
        self.o_act[idx] = a_np
        self.o_logp[idx] = lp.cpu().numpy()
        self.o_val[idx] = val.float().cpu().numpy()
        self.o_d0[idx] = np.linalg.norm(p - self.finish[None, :], axis=1)
        src = pr.src[np.arange(B), a_np]
        self.o_src[idx] = src
        lines, budgets, shapes = [], np.empty(B, np.int64), \
            np.full(B, -1, np.int64)
        for j in range(B):
            ln, secs, _ = pr.line(j, a_np[j], self.vocab, self.surf)
            lines.append(ln)
            if secs is None:
                budgets[j] = self.st.budget_ticks
                shapes[j] = int(pr.shape[j, a_np[j]])
            else:
                budgets[j] = max(1, int(math.ceil(
                    BUDGET_MULT * secs * self.st.ticks_per_s - 1e-6)))
        self._window_choice(pr, a_np, ent)
        fresh = self.fresh[idx].copy()
        self.fresh[idx] = False
        self.st.begin(idx, shapes, p, lines=lines, budgets=budgets)
        return idx, lines, fresh

    def _window_choice(self, pr: Proposals, a_np, ent) -> None:
        """plan()'s diagnostics: the chosen plans' wall / void (and the
        CANDIDATE SET's base rate at the same states), source shares, the
        candidate counts, distinct identities. Measured, never a reward or a
        filter."""
        w = self.w
        B = pr.B
        ar = np.arange(B)
        w["chosen"] += B
        cx_, fr_, vd_ = _cand_diag(pr, ar, a_np, self.wm, self.surf,
                                   self.kill_z)
        w["wall"] += int(cx_.sum())
        w["wall_len"] += float(fr_.sum())
        nb = min(B, WALL_BASE_ROWS_SURF if self.surf else WALL_BASE_ROWS)
        rr, cc = np.nonzero(pr.mask[:nb])
        bx_, bf_, bv_ = _cand_diag(pr, rr, cc, self.wm, self.surf,
                                   self.kill_z)
        cnt = np.maximum(np.bincount(rr, minlength=nb), 1)

        def per_row(x):
            return float((np.bincount(rr, weights=np.asarray(x, np.float64),
                                      minlength=nb) / cnt).sum())
        w["wall_base"] += per_row(bx_)
        w["wall_len_base"] += per_row(bf_)
        w["wall_base_n"] += nb
        if self.surf:
            w["void"] += int(vd_.sum())
            w["void_base"] += per_row(bv_)
        w["ent"] += float(ent.sum())
        np.add.at(w["shapes"], a_np, 1)
        src = pr.src[ar, a_np]
        np.add.at(w["chosen_src"], src, 1)
        w["cand_n"] += int(pr.n_valid.sum())
        m = pr.mask
        for s in range(3):
            w["cand_src"][s] += int(((pr.src == s) & m).sum())
        w["hs_found"] += int(((pr.src == SRC_HS) & m).any(1).sum())
        for j in range(B):
            w["ids"].add(pr.ident(j, a_np[j]))

    # ------------------------------------------------------------ PPO
    def update(self, force: bool = False):
        """One PPO update over every closed plan since the last one (at least
        --plan-batch), on the candidate set each choice was made from."""
        n = self.n_ready()
        if n == 0 or (n < int(self.cfg["plan_batch"]) and not force):
            return None
        imgs, scals, cands, masks, acts, logps = [], [], [], [], [], []
        advs, rets = [], []
        for i, b in enumerate(self.buf):
            if not b:
                continue
            bx = self.buf_x[i]
            if len(bx) != len(b):
                raise RuntimeError(f"planner env {i}: {len(b)} transitions "
                                   f"but {len(bx)} candidate sets")
            vb = float(self.o_val[i]) if self.st.active[i] else 0.0
            adv, ret = plan_gae([t[5] for t in b], [t[4] for t in b],
                                [t[6] for t in b], vb)
            for t, x in zip(b, bx):
                imgs.append(t[0])
                scals.append(t[1])
                acts.append(t[2])
                logps.append(t[3])
                cands.append(x[0])
                masks.append(x[1])
            advs.append(adv)
            rets.append(ret)
            b.clear()
            bx.clear()
        dev = self.device
        img = torch.as_tensor(np.stack(imgs), device=dev)
        scal = torch.as_tensor(np.stack(scals), device=dev)
        cand = torch.as_tensor(np.stack(cands), device=dev)
        mask = torch.as_tensor(np.stack(masks), device=dev)
        act = torch.as_tensor(np.asarray(acts, np.int64), device=dev)
        old = torch.as_tensor(np.asarray(logps, np.float32), device=dev)
        adv = np.concatenate(advs)
        ret = torch.as_tensor(np.concatenate(rets).astype(np.float32),
                              device=dev)
        adv_n = torch.as_tensor(((adv - adv.mean()) / (adv.std() + 1e-8))
                                .astype(np.float32), device=dev)
        M = len(acts)
        mb = min(PLAN_MB, M)
        ent_c = float(self.cfg["plan_ent"])
        st = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "n_mb": 0,
              "ratio0": None}
        epochs = max(1, int(self.cfg["plan_epochs"]))
        for ep in range(epochs):
            perm = self.np_rng.permutation(M)
            for s in range(0, M, mb):
                j = torch.as_tensor(perm[s:s + mb], device=dev)
                lp, lp_all, v = pointer_logp(self.net, img[j], scal[j],
                                             cand[j], mask[j], act[j])
                ratio = torch.exp(lp - old[j])
                if st["ratio0"] is None:
                    # the first minibatch, before any step: the stored set
                    # and log-prob reproduce (ratio 1 up to float noise)
                    st["ratio0"] = float((ratio - 1.0).abs().max().detach())
                a = adv_n[j]
                pg = -torch.min(ratio * a, torch.clamp(
                    ratio, 1.0 - PLAN_CLIP, 1.0 + PLAN_CLIP) * a).mean()
                vl = ((v.float() - ret[j]) ** 2).mean()
                ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                loss = pg + PLAN_VF * vl - ent_c * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), PLAN_MAX_GRAD)
                self.opt.step()
                if ep == epochs - 1:
                    st["pi"] += float(pg.detach())
                    st["v"] += float(vl.detach())
                    st["ent"] += float(ent.detach())
                    st["kl"] += float((old[j] - lp).mean().detach())
                    st["n_mb"] += 1
        k = max(1, st["n_mb"])
        self.updates += 1
        self.last_upd = {"loss_pi": st["pi"] / k, "loss_v": st["v"] / k,
                         "entropy": st["ent"] / k, "kl": st["kl"] / k,
                         "n": M, "updates": self.updates,
                         "ret_mean": float(ret.mean()),
                         "ratio0": st["ratio0"]}
        return self.last_upd

    # ------------------------------------------------------------ logging
    def pop_window(self) -> dict:
        w = self.w
        rate = (lambda a, b: (a / b) if b else float("nan"))
        ext = {"cand": rate(w["cand_n"], w["chosen"]),
               "bank": self.bank_n, "distinct_ids": len(w["ids"]),
               "hs_found": rate(w["hs_found"], w["chosen"])}
        for s, nm in enumerate(("hs", "pert", "unif")):
            ext[f"cand_{nm}"] = rate(int(w["cand_src"][s]), w["chosen"])
            ext[f"choose_{nm}"] = rate(int(w["chosen_src"][s]), w["chosen"])
            ext[f"complete_{nm}"] = rate(int(w["complete_src"][s]),
                                         int(w["closed_src"][s]))
            ext[f"closed_{nm}"] = int(w["closed_src"][s])
        out = super().pop_window()
        out.update(ext)
        out["distinct"] = ext["distinct_ids"]
        self._last_prop = ext
        return out

    def note_and_row(self):
        """-> (step-line text, progress.csv values: the learned planner's
        columns, then PROP_COLS)."""
        txt, row = super().note_and_row()
        e = self._last_prop
        # "k n/K" reads distinct SHAPES of a K-shape vocabulary; here the
        # count is of distinct chosen candidates (their identities)
        txt = txt.replace(f", k {e['distinct_ids']}/{self.K},",
                          f", distinct {e['distinct_ids']},", 1)

        def f(v, nd):
            return round(float(v), nd) if v == v else ""
        row = list(row) + [f(e["cand"], 3), f(e["cand_hs"], 3),
                           f(e["cand_pert"], 3), f(e["cand_unif"], 3),
                           f(e["choose_hs"], 4), f(e["choose_pert"], 4),
                           f(e["choose_unif"], 4), f(e["complete_hs"], 4),
                           f(e["complete_pert"], 4),
                           f(e["complete_unif"], 4), int(e["bank"])]
        pc = (lambda v: f"{v:.0%}" if v == v else "-")
        num = (lambda v: f"{v:.1f}" if v == v else "-")
        txt += (f" | PROP cand {num(e['cand'])} (hs {num(e['cand_hs'])} "
                f"pert {num(e['cand_pert'])} unif {num(e['cand_unif'])}) "
                f"chose {pc(e['choose_hs'])}/{pc(e['choose_pert'])}/"
                f"{pc(e['choose_unif'])} cmpl {pc(e['complete_hs'])}/"
                f"{pc(e['complete_pert'])}/{pc(e['complete_unif'])} "
                f"(hs/pert/unif) bank {e['bank']:,}")
        return txt, row

    # --------------------------------------------------------------- eval
    def eval_hooks(self, core, ev: dict, *, line=None, graph=None,
                   finish_radius: Optional[float] = None):
        """(episode_meta, on_tick) for record_rollout on a 1-env core: this
        network, GREEDY over this state's candidate set. The hindsight bank
        is the training map's reservoir, so a held-out map gets none."""
        g = graph if graph is not None else self.graph
        return make_proposal_hooks(
            self.net, self.maker, (self.bank if g is self.graph else None),
            g, core, ev, line=line, act_every=self.act_every,
            tick_ms=self.tick_ms, corridor=self.corridor, device=self.device,
            spec=self.spec, wm=self._wm_for(g), finish_radius=finish_radius,
            seed=self.eval_seed)


def proposal_planner_from_state(sd: dict, device="cpu"):
    """A proposals checkpoint's ``planner`` entry -> (net, maker, spec) for
    a recording (tools/record_ckpt.py): the network in eval, no optimizer."""
    if not sd or "net" not in sd:
        raise ValueError("no learned-planner weights in this checkpoint")
    spec = dict(sd.get("spec") or {})
    if spec.get("vocab") != "proposals":
        raise ValueError(f"not a proposals planner (spec vocab "
                         f"{spec.get('vocab')!r})")
    bspec = base_vocab_spec(spec)
    surf = spec.get("base") == "surf"
    vocab = make_vocab(bspec if surf else {k: bspec[k] for k in
                                           _spec_default()})
    net = PointerNet(int(spec["n_feat"]), spec["patch_n"], spec["n_scal"],
                     spec["hidden"], in_ch=int(spec.get("in_ch", 2)),
                     cand_hidden=int(spec["cand_hidden"])).to(
                         torch.device(device))
    net.load_state_dict(sd["net"])
    net.eval()
    return net, ProposalMaker(vocab, spec), spec


def make_proposal_hooks(net, maker: ProposalMaker, bank, graph, core,
                        ev: dict, *, line=None, act_every: int = 1,
                        tick_ms: float = 10.0, corridor: float = 192.0,
                        device="cpu", spec: Optional[dict] = None, wm=None,
                        finish_radius: Optional[float] = None, seed: int = 0):
    """(episode_meta, on_tick) for record_rollout on a core whose env 0 is
    recorded - the ONE implementation of the proposal planner's eval, shared
    by the trainer and tools/record_ckpt.py. From wherever the core spawned
    env 0 the end goal is the finish BOX; at every re-plan (completion,
    budget, then the next executor decision boundary - exactly training's
    clock) the candidate set is built from ``bank`` (None: shapes only) and
    the planner takes its ARGMAX. The uninformed draws come from an RNG
    seeded with ``seed`` per hooks instance, so a recording is repeatable.
    ``ev`` is the caller's tally, updated in place (n, succ, ticks, dists,
    plans, closed, complete, wall, wall_len, void, shapes = chosen
    identities, src = chosen counts by source, ncand)."""
    spec = dict(spec or maker.spec)
    dev = torch.device(device)
    surf = spec.get("base") == "surf"
    if wm is None:
        wm = (SlabMap.for_graph(graph, spec) if surf else
              WalkMap(graph, spec["patch_cell_u"], spec["patch_n"],
                      spec["patch_layers"]))
    st = PlanState(1, wm, maker.vocab, tick_ms, corridor, l_max=HS_L_MAX)
    finish = np.asarray(graph.finish_center, np.float64)
    kill_z = float(getattr(graph, "kill_z", -np.inf))
    K = max(1, int(act_every))
    rng = np.random.default_rng(int(seed))
    cap = int(spec["visit_cap"])
    ev.update({"n": 0, "succ": 0, "pending": False, "center": None,
               "ticks": [], "dists": [], "t0": 0, "box": True, "plans": 0,
               "closed": 0, "complete": 0, "wall": 0, "wall_len": 0.0,
               "shapes": [], "lens": [], "src": [0, 0, 0], "ncand": [],
               "proposals": True})
    if surf:
        ev["void"] = 0
    zero = np.zeros(1, bool)
    last = {}

    def _choose():
        sv = core.states_view
        p = sv["origin"][0:1].astype(np.float64)
        v = sv["velocity"][0:1].astype(np.float64)
        y = sv["yaw"][0:1].astype(np.float64)
        if surf:
            img, scal = build_obs_surf(wm, st.visits, np.zeros(1, np.int64),
                                       p, v, y, finish, cap)
        else:
            img, scal = build_obs(wm, st.visits, np.zeros(1, np.int64), p, v,
                                  y, finish, cap)
        pr = maker.make(bank, p, v, y, rng)
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(img, device=dev),
                            torch.as_tensor(scal, device=dev),
                            torch.as_tensor(pr.feats, device=dev),
                            torch.as_tensor(pr.mask, device=dev))
        c = int(logits.float().argmax(-1)[0])
        ln, secs, arc = pr.line(0, c, maker.vocab, surf)
        bud = (st.budget_ticks if secs is None else max(1, int(math.ceil(
            BUDGET_MULT * secs * st.ticks_per_s - 1e-6))))
        sh = int(pr.shape[0, c]) if secs is None else -1
        st.begin(np.zeros(1, np.int64), [sh], p, lines=[ln],
                 budgets=np.array([bud], np.int64))
        if line is not None:
            line.set_lines(np.array([0]), [ln])
        cx_, fr_, vd_ = _cand_diag(pr, [0], [c], wm, surf, kill_z)
        s = int(pr.src[0, c])
        ev["wall"] += int(cx_[0])
        ev["wall_len"] += float(fr_[0])
        if surf:
            ev["void"] += int(vd_[0])
        ev["plans"] += 1
        ev["shapes"].append(pr.ident(0, c))
        ev["lens"].append(float(arc))
        ev["src"][s] += 1
        ev["ncand"].append(int(pr.n_valid[0]))
        last.update({"src": s, "axis": int(pr.axis[0, c]), "shape": sh,
                     "k": int(pr.n_valid[0]), "length": float(arc),
                     "time": (None if secs is None else float(secs))})
        return ln

    def episode_meta(ep):
        st.visits.reset([0])
        p = core.states_view["origin"][0:1].astype(np.float64)
        st.visits.update(p)
        ln = _choose()
        ev["n"] += 1
        dfin = float("nan")
        if getattr(graph, "fin", None) is not None:
            dfin = float(graph.dist[graph.fin, int(graph.snap(p)[0])])
        ev["dists"].append(dfin)
        thin = ln[:: max(1, len(ln) // 64)]
        rad = float(finish_radius) if finish_radius is not None else 192.0
        plan = {"planner": "learned", "proposals": True,
                "source": SRC_NAMES[last["src"]],
                "axis": (AXES[last["axis"]] if last["axis"] >= 0 else None),
                "shape": last["shape"], "candidates": last["k"],
                "length": round(last["length"], 1),
                "time": (None if last["time"] is None
                         else round(last["time"], 3)),
                # a surf map's walkable graph need not connect the start to
                # the finish: null, never Infinity (the viewer's JSON)
                "graph_dist": (round(dfin, 1) if np.isfinite(dfin)
                               else None)}
        return {"goal": {"center": [float(x) for x in finish],
                         "radius": rad},
                "line": [[float(x) for x in q] for q in thin],
                "plan": plan}

    def on_tick(t, states, rewards, done, trunc):
        if bool(done[0]) or bool(trunc[0]):
            won = bool(done[0]) and bool(np.asarray(core.goal_hits)[0])
            if won:
                ev["succ"] += 1
                ev["ticks"].append(t - ev["t0"])
            if st.active[0]:
                ev["closed"] += 1
            st.active[0] = False
            st.need[0] = False          # episode_meta plans the next one
            ev["t0"] = t + 1
            return
        p = core.states_view["origin"][0:1].astype(np.float64)
        closed, comp = st.tick(p, zero)
        if closed[0]:
            ev["closed"] += 1
            ev["complete"] += int(comp[0])
        if st.need[0] and (t + 1) % K == 0:
            _choose()

    return episode_meta, on_tick


def prop_cols(base: str) -> list:
    """progress.csv columns of a proposals run: the learned planner's (with
    the surf pair on a surf base), then PROP_COLS."""
    return list(PLAN_COLS_SURF if base == "surf" else PLAN_COLS) \
        + list(PROP_COLS)
