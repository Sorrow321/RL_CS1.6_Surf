"""goallearn.py - the LEARNED planner behind ``--goal-planner learned``.

Stage (c) of docs/litsurvey-planner-executor.md section 7.3, step c2: a small
planner network, trained by its own PPO, drives the FROZEN goal-conditioned
executor that stage 1 (``--goal-planner bfs``, surfgym/goalplan.py) trained.
The user's brief (2026-09-23): "make the planner trainable ... the end goal
is to reach the end of the map and we should use the planner sort of to
explore more, to generate reasonable paths around the map", and "if you send
a polyline through the wall, the executor would not be able to execute this
path, and in this case we should penalize our planner ... we don't want to
enforce it. We can measure it".

Nothing here reads a demo, a route file or a map-specific constant (CLAUDE.md
rules 0 and 0b): the vocabulary is generic geometry, the observation is the
stage-1 walkable graph (map geometry) plus this episode's own visits, and
every constant below is set ONCE, in map units that are player constants,
fractions, or counts.

THE SEMI-MDP. Each env runs one PLAN at a time. A plan is CLOSED on the tick
that

* the executor COMPLETES it: arc progress along the plan line >= 90% of the
  line's arc length (a MultiArcProgress tracker whose corridor is the goal
  radius, 192 u - the arrival radius the stage-1 executor was trained on -
  so sliding along the wrong side of a wall does not count);
* its TIME BUDGET runs out: plan length / 250 u/s x 1.5 (250 u/s is
  player_maxspeed, the ground speed cap: a player constant);
* or the EPISODE ENDS (finish, death, stall kill or truncation).

A closed plan is one planner transition; its reward is paid then. The next
plan is chosen at the next executor DECISION boundary (the executor reads the
fan only there, so a plan chosen at the boundary is exactly the plan it
sees), and every env that needs a plan at that boundary is batched into ONE
forward pass.

ACTION: a categorical over a fixed VOCABULARY of polyline shapes, generated
once from generic parameters (:class:`PlanVocab`): 16 world-frame initial
headings x 5 turn profiles (straight, arc 45 deg left / right, arc 90 deg
left / right), each 8 segments x 100 u = 800 u, anchored at the agent and
lifted to its height. NOTHING filters a shape: a plan may run through a
wall, and the only thing that says so is the executor failing to complete
it. The chosen shape becomes the env's line on the MultiLine fan and on the
goal-arc reward (re-anchored at arc 0 at the plan's start).

OBSERVATION: an egocentric, WORLD-ALIGNED top-down patch around the agent,
32 x 32 cells of 64 u (a 2,048 u window), two channels - the fraction of the
cell that is walkable on stage 1's graph within one layer of the agent's own
floor layer (perception of the geometry, not a constraint), and this
episode's visit counts (entries into the cell, saturating at 4) - plus nine
scalars: the unit vector to the finish-box centre and log1p(Euclidean
distance / 1000) (Euclidean, never the geodesic), the velocity / 1000 and
the yaw as (sin, cos).

REWARD per closed plan:

* executability, AMIGo-style (Campero et al. 2021): +0.7 if the executor
  completed the plan within its budget, -0.3 if not - the "penalize the
  planner for plans the executor cannot execute";
* +10 when the episode finishes the map (the finish bonus);
* count-based novelty over where the plan ENDED: beta / sqrt(n) with a
  GLOBAL count n over 128 u cells (3-D), beta = 0.5. Not paid on a death
  (a planner paid for novel deaths learns to find new pits);
* optional, default off: a Euclidean progress-to-finish term, coef per
  1,000 u of distance reduced.

PPO on the planner: gamma 0.95 per plan, GAE lambda 0.95, clip 0.2, value
coef 0.5, entropy bonus (the coverage floor, --plan-ent 0.01), Adam 3e-4,
minibatch 256, 4 epochs, grad clip 0.5. An episode end is terminal for the
planner, TRUNCATION INCLUDED (v1 simplification: the terminal state is not
re-rendered for a bootstrap; with gamma 0.95 per plan the bias is confined to
the last few plans of a truncated episode).

SURF (``--plan-vocab surf``, surfgym/goalsurf.py): the spec carries
"vocab": "surf" and the same network then chooses among 144 speed-scaled
3-D shapes (16 headings x 3 turns x 3 descents, length clamp(3 s x
max(|v_xy|, 500 u/s), 800, 6000) u, budget 4.5 s) from 3 occupancy SLABS +
the visit channel (in_ch 4); its wall diagnostics are the 3-D solid crossing,
plus plans that END below the kill ceiling (plan/void). A walking spec has
no "vocab" key and every branch below is the one that shipped.

DIAGNOSTICS, logged and never in the reward: the share of chosen plans whose
polyline crosses a non-walkable cell of the graph (and the same share over
the whole vocabulary at the same states - the base rate the planner should
fall below as it discovers the geometry); the share of each chosen plan's
LENGTH that lies off the walkable graph (and its base rate) - continuous,
because on a maze of ~130 u corridors almost every 800 u shape crosses
SOME non-walkable column (measured: 97-100% of the vocabulary on
labyrinth_left100) while the executor still completes many of them; the
completion rate, the finish
rate (all episodes and true-start episodes), coverage (distinct 128 u cells
the fleet has stood in), planner entropy, distinct shapes chosen, the mean
plan-end novelty.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PlanVocab", "WalkMap", "VisitGrid", "PlanState", "PlannerNet",
           "LearnedPlanner", "build_obs", "plan_gae", "make_learned_hooks",
           "vocab_crossings", "vocab_offgraph",
           "planner_from_state", "PLAN_KNOBS", "PLAN_DEFAULTS", "PLAN_COLS",
           "PLAN_COLS_SURF", "PLAN_LEARN_SEED_OFFSET"]

# ------------------------------------------------------------ the vocabulary
# user design 2026-09-23; generic geometry, identical on every map
VOCAB_HEADINGS = 16                          # world-frame initial headings
VOCAB_TURNS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0)   # total turn, + = left
VOCAB_SEGS = 8                               # segments per shape
VOCAB_SEG_U = 100.0                          # u per segment -> 800 u shapes

# ---------------------------------------------------------- the observation
PATCH_N = 32                                 # cells per side
PATCH_CELL_U = 64.0                          # u per patch cell (2,048 u)
PATCH_LAYERS = 1                             # +-graph layers around the agent's
VISIT_CAP = 4                                # visit channel saturates here
N_SCAL = 9                                   # goal dir 3, log dist, vel 3, yaw 2
NET_HIDDEN = 256

# ------------------------------------------------------ the plan's lifetime
COMPLETE_FRAC = 0.9                          # arc >= 0.9 x the plan's arc
BUDGET_SPEED_U = 250.0                       # player_maxspeed, u/s
BUDGET_MULT = 1.5

# ----------------------------------------------------------- planner reward
R_EXEC_OK = 0.7                              # AMIGo's teacher, completed
R_EXEC_FAIL = -0.3                           # AMIGo's teacher, not completed
NOVELTY_CELL_U = 128.0                       # global plan-end count cells

# ------------------------------------------------------------- planner PPO
PLAN_GAMMA = 0.95                            # per plan (a semi-MDP step)
PLAN_LAMBDA = 0.95
PLAN_CLIP = 0.2
PLAN_VF = 0.5
PLAN_MB = 256
PLAN_MAX_GRAD = 0.5

# the wall diagnostic samples a shape every 16 u and skips its first graph
# cell (the agent's own cell can read unwalkable while it is mid-jump or
# pressed into a wall - that is not the PLAN crossing anything)
WALL_SAMPLE_U = 16.0
# the base rate (the whole vocabulary's crossing share) is estimated on at
# most this many of the states re-planned in one call
WALL_BASE_ROWS = 16

# the planner's sampling generator is seeded from --seed + this offset (the
# executor's rollout RNG is untouched by the planner's draws)
PLAN_LEARN_SEED_OFFSET = 4561

# the trainer flags (argparse dest -> default). Resolved only under
# --goal-planner learned; TRAIN_ONLY in tools/record_ckpt.py (they shape the
# planner's TRAINING, a recording runs the stored network greedily).
PLAN_KNOBS = ("plan_lr", "plan_ent", "plan_batch", "plan_epochs",
              "plan_novelty", "plan_progress", "plan_finish_bonus",
              "plan_r_ok", "plan_r_fail")
PLAN_DEFAULTS = {"plan_lr": 3e-4, "plan_ent": 0.01, "plan_batch": 512,
                 "plan_epochs": 4, "plan_novelty": 0.5, "plan_progress": 0.0,
                 "plan_finish_bonus": 10.0,
                 # per-plan executability reward (AMIGo's +0.7 / -0.3). With
                 # gamma 0.95 per plan a stream of completed plans is worth
                 # 0.7 / 0.05 = 14 > the finish bonus, i.e. wandering pays;
                 # --plan-r-ok 0 keeps only the penalty (2026-09-23)
                 "plan_r_ok": R_EXEC_OK, "plan_r_fail": R_EXEC_FAIL}

# progress.csv columns, appended LAST and only under --goal-planner learned
PLAN_COLS = ["plan/closed", "plan/complete", "plan/wall", "plan/wall_base",
             "plan/wall_len", "plan/wall_len_base",
             "plan/finish", "plan/finish_start", "plan/eval_finish",
             "plan/coverage", "plan/entropy", "plan/distinct",
             "plan/novelty", "plan/reward", "plan/loss_pi", "plan/loss_v",
             "plan/kl", "plan/updates"]
# --plan-vocab surf: plan/wall and plan/wall_len are the 3-D SOLID crossing
# (surfgym/goalsurf.py) and two columns follow LAST: the share of chosen
# plans that END below the kill ceiling, and the vocabulary's base rate
PLAN_COLS_SURF = PLAN_COLS + ["plan/void", "plan/void_base"]


def _spec_default() -> dict:
    """Everything that defines what a stored planner network MEANS: the
    vocabulary (its action index), the patch and the scalars (its input).
    Stored in the checkpoint, and a recording rebuilds from it."""
    return {"headings": VOCAB_HEADINGS, "turns_deg": list(VOCAB_TURNS_DEG),
            "segs": VOCAB_SEGS, "seg_u": VOCAB_SEG_U,
            "patch_n": PATCH_N, "patch_cell_u": PATCH_CELL_U,
            "patch_layers": PATCH_LAYERS, "visit_cap": VISIT_CAP,
            "n_scal": N_SCAL, "hidden": NET_HIDDEN}


# ==========================================================================
# the vocabulary
# ==========================================================================
class PlanVocab:
    """The planner's action set: K = headings x turns polylines, local
    (anchored at the origin, z = 0), built once from generic parameters.

    Shape k = (heading h, turn theta): segment i (0-based) points along
    ``h + theta * (i + 0.5) / segs`` - the chords of a circular arc whose
    tangent turns by theta over the shape (theta = 0: a straight line). Every
    segment is ``seg_u`` long, so every shape is exactly segs x seg_u long.
    ``h`` is the direction at the START in the world frame (0 = +x, counter-
    clockwise), up to the half-segment chord offset."""

    def __init__(self, headings: int = VOCAB_HEADINGS,
                 turns_deg=VOCAB_TURNS_DEG, segs: int = VOCAB_SEGS,
                 seg_u: float = VOCAB_SEG_U, spacing: Optional[float] = None):
        from .route import DEFAULT_SPACING
        self.headings = int(headings)
        self.turns_deg = tuple(float(t) for t in turns_deg)
        self.segs = int(segs)
        self.seg_u = float(seg_u)
        self.spacing = float(spacing or DEFAULT_SPACING)
        nt = len(self.turns_deg)
        self.K = self.headings * nt
        self.heading_deg = np.zeros(self.K, np.float64)
        self.turn_deg = np.zeros(self.K, np.float64)
        raw = np.zeros((self.K, self.segs + 1, 3), np.float64)
        for hi in range(self.headings):
            h = 2.0 * math.pi * hi / self.headings
            for ti, turn in enumerate(self.turns_deg):
                k = hi * nt + ti
                th = math.radians(turn)
                ang = h + th * (np.arange(self.segs) + 0.5) / self.segs
                steps = self.seg_u * np.stack(
                    [np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
                raw[k, 1:] = np.cumsum(steps, axis=0)
                self.heading_deg[k] = math.degrees(h)
                self.turn_deg[k] = turn
        self.raw = raw
        self.length = float(self.segs * self.seg_u)
        # local wall-diagnostic samples (K, M, 2), every WALL_SAMPLE_U of arc
        # from one sample in (the first sample is skipped by the caller's
        # cell rule, see WalkMap.crosses)
        s = np.arange(WALL_SAMPLE_U, self.length + 1e-9, WALL_SAMPLE_U)
        cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
            np.diff(raw[0], axis=0), axis=1))))
        self.wall_s = s
        self.wall_xy = np.stack([np.stack(
            [np.interp(s, cum, raw[k, :, 0]), np.interp(s, cum, raw[k, :, 1])],
            axis=1) for k in range(self.K)]).astype(np.float64)
        self.n_line = int(len(self.anchor(0, np.zeros(3))))

    def anchor(self, k: int, origin, length=None) -> np.ndarray:
        """Shape ``k`` anchored at ``origin`` and LIFTED to its height, ready
        for MultiLine / MultiArcProgress: resampled at the fan spacing by the
        same helper every other line builder uses (goals.segment_line's
        resample), float32, first point == origin. ``length`` exists for the
        interface the speed-scaled surf vocabulary shares
        (surfgym.goalsurf.SurfVocab); a walking shape has ONE length."""
        from .route import resample_polyline
        if length is not None and abs(float(length) - self.length) > 1e-6:
            raise ValueError(f"the walking vocabulary's shapes are "
                             f"{self.length:g} u, not {float(length):g} u")
        o = np.asarray(origin, np.float64).reshape(3)
        pts = self.raw[int(k)] + o[None, :]
        return resample_polyline(pts, self.spacing)[0]

    def length_for(self, speed) -> np.ndarray:
        """(k,) horizontal speeds -> (k,) plan lengths: one fixed length for
        the walking vocabulary (the surf vocabulary scales it)."""
        return np.full(np.shape(np.atleast_1d(speed)), self.length, np.float64)

    def ends(self, origins, lengths=None) -> np.ndarray:
        """(k, 3) origins -> (k, K, 3) every shape's END anchored there."""
        o = np.atleast_2d(np.asarray(origins, np.float64))
        return o[:, None, :] + self.raw[None, :, -1, :]

    def describe(self) -> str:
        return (f"plan vocabulary: {self.K} shapes = {self.headings} world "
                f"headings x {len(self.turns_deg)} turns "
                f"({', '.join(f'{t:+g}' for t in self.turns_deg)} deg), "
                f"{self.segs} x {self.seg_u:g} u = {self.length:g} u each, "
                f"resampled to {self.n_line} points at {self.spacing:g} u; "
                "nothing filters a shape through a wall")


# ==========================================================================
# perception: the walkable graph as a 2-D patch, and the visit channel
# ==========================================================================
class WalkMap:
    """Stage 1's walkable graph (BFSPlanner), seen from above.

    ``fine[iz]`` (graph resolution) is True where a node exists within
    ``layers`` graph layers of layer iz - "walkable at this floor level", so
    ramps and stairs read as walkable from either end. ``coarse[iz]`` is its
    mean over s x s blocks of graph cells (s = round(64 u / cell)), i.e. the
    walkable FRACTION of each patch cell, zero-padded by half a patch on each
    side so a window slice never leaves the array."""

    def __init__(self, graph, patch_cell_u: float = PATCH_CELL_U,
                 n: int = PATCH_N, layers: int = PATCH_LAYERS):
        self.graph = graph
        self.cell = float(graph.cell)
        self.s = max(1, int(round(float(patch_cell_u) / self.cell)))
        self.pc = self.s * self.cell            # the realised patch cell
        self.n = int(n)
        self.half = self.n // 2
        node = np.asarray(graph.node_of) >= 0
        nz, ny, nx = node.shape
        fine = np.zeros_like(node)
        for d in range(-int(layers), int(layers) + 1):
            lo, hi = max(0, -d), min(nz, nz - d)
            if lo < hi:
                fine[lo:hi] |= node[lo + d:hi + d]
        self.fine = fine
        py, px = (-ny) % self.s, (-nx) % self.s
        wp = np.pad(fine.astype(np.float32), ((0, 0), (0, py), (0, px)))
        gy, gx = (ny + py) // self.s, (nx + px) // self.s
        coarse = wp.reshape(nz, gy, self.s, gx, self.s).mean(axis=(2, 4))
        self.gy, self.gx = int(gy), int(gx)
        h = self.half
        self.coarse = np.pad(coarse.astype(np.float32),
                             ((0, 0), (h, h), (h, h)))
        self.mins = np.asarray(graph.mins, np.float64)[:2]
        self._ar = np.arange(self.n)

    def layer(self, pos) -> np.ndarray:
        """(k, 3) positions -> (k,) the graph layer of the nearest node."""
        nd = self.graph.snap(np.atleast_2d(np.asarray(pos, np.float64)))
        return np.asarray(self.graph.coords[nd, 0], np.int64)

    def cells(self, pos):
        """(k, 3) -> (cy, cx) coarse cell indices, clipped into the lattice
        (an agent outside the graph's box reads the nearest border cell)."""
        p = np.atleast_2d(np.asarray(pos, np.float64))
        cy = np.floor((p[:, 1] - self.mins[1]) / self.pc).astype(np.int64)
        cx = np.floor((p[:, 0] - self.mins[0]) / self.pc).astype(np.int64)
        return (np.clip(cy, 0, self.gy - 1), np.clip(cx, 0, self.gx - 1))

    def window(self, cy, cx):
        """(rows, cols) index grids of the N x N window centred on each
        cell, in PADDED coordinates: row 0 of the window is cy - N/2."""
        rows = np.asarray(cy, np.int64)[:, None] + self._ar[None, :]
        cols = np.asarray(cx, np.int64)[:, None] + self._ar[None, :]
        return rows, cols

    def patch(self, iz, cy, cx) -> np.ndarray:
        """(k,) layers + cells -> (k, N, N) float32 walkable fraction."""
        rows, cols = self.window(cy, cx)
        iz = np.asarray(iz, np.int64)
        return self.coarse[iz[:, None, None], rows[:, :, None],
                           cols[:, None, :]]

    def offgraph(self, iz, xy) -> np.ndarray:
        """(k,) layers + (k, M, 2) world xy samples -> (k, M) bool: the
        sample lands on a non-walkable graph column (off the grid counts).
        Samples inside the start's own graph cell are the caller's to
        drop."""
        xy = np.asarray(xy, np.float64)
        ix = np.floor((xy[..., 0] - self.mins[0]) / self.cell).astype(np.int64)
        iy = np.floor((xy[..., 1] - self.mins[1]) / self.cell).astype(np.int64)
        nz, ny, nx = self.fine.shape
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        izb = np.broadcast_to(np.asarray(iz, np.int64)[:, None], ix.shape)
        ok = np.zeros(ix.shape, bool)
        ok[inb] = self.fine[izb[inb], iy[inb], ix[inb]]
        return ~ok

    def crosses(self, iz, xy) -> np.ndarray:
        """(k,) bool: some sample of the row is off the walkable graph."""
        return self.offgraph(iz, xy).any(axis=-1)


class VisitGrid:
    """This episode's visit counts, per env, on the WalkMap's patch lattice
    (padded like ``WalkMap.coarse`` so a window is a plain slice). A visit is
    an ENTRY into a cell (the cell changed since the previous tick, or the
    episode just started there) - independent of walking speed. uint8,
    saturating at 255; the observation clips at VISIT_CAP anyway."""

    MEM_CAP = 2 << 30

    def __init__(self, n_envs: int, wm: WalkMap):
        self.wm = wm
        self.n = int(n_envs)
        h = wm.half
        shape = (self.n, wm.gy + 2 * h, wm.gx + 2 * h)
        need = int(np.prod(shape))
        if need > self.MEM_CAP:
            raise RuntimeError(
                f"planner visit grid: {self.n} envs x {shape[1]} x "
                f"{shape[2]} cells = {need / 2**30:.1f} GiB, over the "
                f"{self.MEM_CAP / 2**30:.0f} GiB cap - fewer --envs")
        self.cnt = np.zeros(shape, np.uint8)
        self.prev = np.full((self.n, 2), -1, np.int64)

    def reset(self, idx) -> None:
        idx = np.asarray(idx, np.int64).reshape(-1)
        self.cnt[idx] = 0
        self.prev[idx] = -1

    def update(self, pos, ended=None) -> None:
        """Every tick, every env: ``pos`` is the POST-step position (an env
        that ended this tick already stands at its NEW spawn - its grid is
        cleared first and the spawn cell is that episode's first visit)."""
        if ended is not None:
            e = np.flatnonzero(np.asarray(ended, bool))
            if len(e):
                self.reset(e)
        cy, cx = self.wm.cells(pos)
        new = (cy != self.prev[:, 0]) | (cx != self.prev[:, 1])
        i = np.flatnonzero(new)
        if len(i):
            h = self.wm.half
            r, c = cy[i] + h, cx[i] + h
            v = self.cnt[i, r, c].astype(np.int16) + 1
            self.cnt[i, r, c] = np.minimum(v, 255).astype(np.uint8)
        self.prev[:, 0] = cy
        self.prev[:, 1] = cx

    def patch(self, idx, cy, cx, cap: int = VISIT_CAP) -> np.ndarray:
        rows, cols = self.wm.window(cy, cx)
        idx = np.asarray(idx, np.int64)
        c = self.cnt[idx[:, None, None], rows[:, :, None], cols[:, None, :]]
        return (np.minimum(c, cap).astype(np.float32) / float(cap))


def vocab_offgraph(vocab: PlanVocab, wm: WalkMap, iz, p, shapes=None):
    """The wall diagnostics (never a reward, never a filter) of shapes
    anchored at ``p``: -> (crosses, frac), ``crosses`` = some sample of the
    polyline lies on a non-walkable graph column, ``frac`` = the share of
    its samples that do (how MUCH of the plan runs through walls or void).
    ``shapes`` None -> (k, K) arrays over the whole vocabulary; else (k,)
    for shape shapes[i] at p[i]. Samples inside the start's own graph cell
    are skipped (see WALL_SAMPLE_U)."""
    p = np.atleast_2d(np.asarray(p, np.float64))
    keep = vocab.wall_s > wm.cell
    iz = np.asarray(iz, np.int64).reshape(-1)
    if shapes is not None:
        sh = np.asarray(shapes, np.int64).reshape(-1)
        off = wm.offgraph(iz, p[:, None, :2] + vocab.wall_xy[sh][:, keep, :])
        return off.any(axis=-1), off.mean(axis=-1)
    xy = (p[:, None, None, :2] + vocab.wall_xy[None, :, keep, :])
    k, K, M = xy.shape[0], xy.shape[1], xy.shape[2]
    off = wm.offgraph(np.repeat(iz, K), xy.reshape(k * K, M, 2))
    return (off.any(axis=-1).reshape(k, K), off.mean(axis=-1).reshape(k, K))


def vocab_crossings(vocab: PlanVocab, wm: WalkMap, iz, p,
                    shapes=None) -> np.ndarray:
    """vocab_offgraph's ``crosses`` alone: (k, K) or (k,) bool."""
    return vocab_offgraph(vocab, wm, iz, p, shapes)[0]


def build_obs(wm: WalkMap, visits: VisitGrid, idx, pos, vel, yaw_deg,
              finish, cap: int = VISIT_CAP):
    """The planner's observation for envs ``idx`` (``pos``/``vel``/``yaw``
    are the rows of those envs). -> (img (k, 2, N, N), scal (k, 9)) float32.

    scal = [unit vector to the finish-box centre (3), log1p(|g| / 1000),
            velocity / 1000 (3), sin yaw, cos yaw]: world frame throughout,
    like the patch and the vocabulary's headings."""
    p = np.atleast_2d(np.asarray(pos, np.float64))
    iz = wm.layer(p)
    cy, cx = wm.cells(p)
    img = np.stack([wm.patch(iz, cy, cx),
                    visits.patch(idx, cy, cx, cap)], axis=1)
    g = np.asarray(finish, np.float64).reshape(1, 3) - p
    d = np.linalg.norm(g, axis=1)
    u = g / np.maximum(d, 1.0)[:, None]
    v = np.atleast_2d(np.asarray(vel, np.float64)) / 1000.0
    y = np.radians(np.asarray(yaw_deg, np.float64).reshape(-1))
    scal = np.concatenate([u, np.log1p(d / 1000.0)[:, None], v,
                           np.sin(y)[:, None], np.cos(y)[:, None]], axis=1)
    return (np.ascontiguousarray(img, np.float32),
            np.ascontiguousarray(scal, np.float32))


# ==========================================================================
# per-env plan bookkeeping (training fleet and 1-env evals alike)
# ==========================================================================
class PlanState:
    """The open plan of each env, its clock, the completion tracker and the
    episode's visit counts. ``need`` marks the envs waiting for a plan."""

    def __init__(self, n_envs: int, wm: WalkMap, vocab: PlanVocab,
                 tick_ms: float = 10.0, corridor: float = 192.0,
                 visits: bool = True, l_max: Optional[int] = None):
        from .goalarc import MultiArcProgress
        self.n = int(n_envs)
        self.vocab = vocab
        # l_max: the longest line this state will hold (the vocabulary's
        # own by default; the surf diet's hindsight segments need more)
        self.track = MultiArcProgress(
            self.n, l_max=max(2, int(l_max) if l_max else vocab.n_line),
            spacing=vocab.spacing, corridor=float(corridor), window=16)
        # visits=False: no visit channel (the surf DIET has no planner
        # network to show it to)
        self.visits = VisitGrid(self.n, wm) if visits else None
        self.active = np.zeros(self.n, bool)
        self.need = np.ones(self.n, bool)
        self.shape = np.full(self.n, -1, np.int64)
        self.elapsed = np.zeros(self.n, np.int64)
        self.budget = np.zeros(self.n, np.int64)
        self.start = np.zeros((self.n, 3), np.float64)
        self.set_tick_ms(tick_ms)

    def set_tick_ms(self, tick_ms: float) -> None:
        self.ticks_per_s = 1000.0 / float(tick_ms)
        bs = getattr(self.vocab, "budget_secs", None)
        if bs is not None:
            # the surf vocabulary (surfgym/goalsurf.py): a plan is T_plan
            # seconds of travel at the speed it was drawn for, so its
            # budget is BUDGET_MULT x T_plan whatever its length
            self.budget_ticks = int(math.ceil(float(bs) * self.ticks_per_s
                                              - 1e-6))
            return
        # ceil, less a hair: 800 / 250 * 1.5 * 100 is 480.00000000000006 in
        # floating point, and the budget is 480 ticks
        self.budget_ticks = int(math.ceil(self.vocab.length / BUDGET_SPEED_U
                                          * BUDGET_MULT * self.ticks_per_s
                                          - 1e-6))

    def begin(self, idx, shapes, origins, lengths=None, lines=None,
              budgets=None) -> list:
        """Open plans ``shapes`` for envs ``idx`` at ``origins`` -> the lines
        (float32 (L, 3), arc 0 at the origin). ``lengths`` (per env) sizes a
        speed-scaled shape; ``lines`` hands in ready-made lines instead (the
        surf diet's hindsight segments; ``shapes`` is then -1 there) and
        ``budgets`` per-plan budgets in ticks (default: the vocabulary's)."""
        idx = np.asarray(idx, np.int64).reshape(-1)
        shapes = np.asarray(shapes, np.int64).reshape(-1)
        org = np.atleast_2d(np.asarray(origins, np.float64))
        if lines is None:
            if lengths is None:
                lines = [self.vocab.anchor(k, o) for k, o in zip(shapes, org)]
            else:
                ln = np.asarray(lengths, np.float64).reshape(-1)
                lines = [self.vocab.anchor(k, o, L)
                         for k, o, L in zip(shapes, org, ln)]
        else:
            lines = list(lines)
        if len(idx):
            self.track.set_lines(idx, lines)
        self.active[idx] = True
        self.need[idx] = False
        self.shape[idx] = shapes
        self.elapsed[idx] = 0
        self.budget[idx] = (self.budget_ticks if budgets is None
                            else np.asarray(budgets, np.int64).reshape(-1))
        self.start[idx] = org
        return lines

    def tick(self, pos, ended):
        """One physics tick. ``pos`` (n, 3) POST-step positions, ``ended``
        (n,) episode ends on this tick. -> (closed, completed) bool masks.

        An ended row closes its open plan UNcompleted: its post-step
        position is the next episode's spawn, and a plan completed on an
        earlier tick was closed then."""
        ended = np.asarray(ended, bool)
        if self.visits is not None:
            self.visits.update(pos, ended)
        self.track.advance(np.asarray(pos, np.float32))
        act = self.active
        self.elapsed[act] += 1
        comp = act & ~ended & (self.track.arc
                               >= COMPLETE_FRAC * self.track.total_arc())
        tout = act & ~ended & ~comp & (self.elapsed >= self.budget)
        closed = act & (ended | comp | tout)
        self.active[closed] = False
        self.need[closed] = True
        self.need[ended] = True
        return closed, comp


# ==========================================================================
# the network
# ==========================================================================
class PlannerNet(nn.Module):
    """patch (2, N, N) -> 3 stride-2 convs -> flat; scalars -> 64; both ->
    ``hidden`` -> policy logits over the vocabulary + a value."""

    def __init__(self, n_actions: int, patch_n: int = PATCH_N,
                 n_scal: int = N_SCAL, hidden: int = NET_HIDDEN,
                 in_ch: int = 2):
        super().__init__()
        # in_ch: 2 on the walking patch (walkable + visits); the surf
        # observation is 3 occupancy slabs + visits (surfgym/goalsurf.py)
        self.conv = nn.Sequential(
            nn.Conv2d(int(in_ch), 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        side = patch_n
        for _ in range(3):
            side = (side + 1) // 2
        self.scal = nn.Sequential(nn.Linear(n_scal, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(32 * side * side + 64, hidden),
                                   nn.ReLU())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        # a near-uniform first policy (entropy ~ log K): the planner starts
        # by trying every shape, which is the coverage floor's starting point
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, img, scal):
        h = self.trunk(torch.cat([self.conv(img), self.scal(scal)], dim=1))
        return self.pi(h), self.v(h).squeeze(-1)


def plan_gae(rew, val, done, v_boot: float, gamma: float = PLAN_GAMMA,
             lam: float = PLAN_LAMBDA):
    """GAE over ONE env's consecutive plans. ``done[k]`` = the episode ended
    with plan k (terminal: nothing is bootstrapped across it); ``v_boot`` is
    V of the plan that follows the last one (the env's open plan)."""
    r = np.asarray(rew, np.float64)
    v = np.asarray(val, np.float64)
    d = np.asarray(done, bool)
    adv = np.zeros(len(r), np.float64)
    last = 0.0
    nv = float(v_boot)
    for k in range(len(r) - 1, -1, -1):
        nt = 0.0 if d[k] else 1.0
        delta = r[k] + gamma * nv * nt - v[k]
        last = delta + gamma * lam * nt * last
        adv[k] = last
        nv = v[k]
    return adv, adv + v


def _entropy(logits):
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


# ==========================================================================
# the trainable planner
# ==========================================================================
class LearnedPlanner:
    """The network, its optimizer and PPO, the training fleet's PlanState,
    the global plan-end novelty counts and the diagnostics.

    ``graph`` is stage 1's BFSPlanner of the training map (its walkable
    graph; its finish box gives the finish centre). ``start_pts`` are the
    map-start spawn origins (to count finishes FROM THE TRUE START)."""

    def __init__(self, graph, n_envs: int, device, *, start_pts=None,
                 tick_ms: float = 10.0, act_every: int = 1,
                 corridor: float = 192.0, cfg: Optional[dict] = None,
                 seed: int = 0, spec: Optional[dict] = None):
        if graph.finish_center is None:
            raise ValueError("the learned planner needs the map's finish box")
        self.device = torch.device(device)
        self.graph = graph
        self.spec = dict(_spec_default() if spec is None else spec)
        self.cfg = dict(PLAN_DEFAULTS)
        for k, v in (cfg or {}).items():
            if v is not None:
                self.cfg[k] = v
        # --plan-vocab surf (surfgym/goalsurf.py): the speed-scaled 3-D
        # vocabulary and the occupancy-slab observation. The spec says which
        # (a walking spec has no "vocab" key and is exactly what shipped)
        self.surf = self.spec.get("vocab") == "surf"
        if self.surf:
            from .goalsurf import SlabMap, make_vocab
            self.vocab = make_vocab(self.spec)
            self.K = self.vocab.K
            self.wm = SlabMap.for_graph(graph, self.spec)
        else:
            self.vocab = PlanVocab(self.spec["headings"],
                                   self.spec["turns_deg"],
                                   self.spec["segs"], self.spec["seg_u"])
            self.K = self.vocab.K
            self.wm = WalkMap(graph, self.spec["patch_cell_u"],
                              self.spec["patch_n"], self.spec["patch_layers"])
        self._wms = {id(graph): self.wm}
        self.in_ch = int(self.spec.get("in_ch", 2))
        self.kill_z = float(getattr(graph, "kill_z", -np.inf))
        self.net = PlannerNet(self.K, self.spec["patch_n"],
                              self.spec["n_scal"],
                              self.spec["hidden"],
                              in_ch=self.in_ch).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(),
                                    lr=float(self.cfg["plan_lr"]), eps=1e-5)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))
        self.np_rng = np.random.default_rng(int(seed))
        self.n = int(n_envs)
        self.act_every = max(1, int(act_every))
        self.corridor = float(corridor)
        self.tick_ms = float(tick_ms)
        self.st = PlanState(self.n, self.wm, self.vocab, tick_ms, corridor)
        self.finish = np.asarray(graph.finish_center, np.float64)
        self.start_pts = (None if start_pts is None else
                          np.atleast_2d(np.asarray(start_pts, np.float64)))
        # the open plan's decision record, per env
        pn = int(self.spec["patch_n"])
        self.o_img = np.zeros((self.n, self.in_ch, pn, pn), np.float32)
        self.o_scal = np.zeros((self.n, int(self.spec["n_scal"])), np.float32)
        self.o_act = np.zeros(self.n, np.int64)
        self.o_logp = np.zeros(self.n, np.float32)
        self.o_val = np.zeros(self.n, np.float32)
        self.o_d0 = np.zeros(self.n, np.float64)
        # per-env episode facts
        self.fresh = np.ones(self.n, bool)          # at an episode start
        self.from_start = np.zeros(self.n, bool)    # spawned at the map start
        self.buf = [[] for _ in range(self.n)]      # closed transitions
        # global plan-end counts (novelty) and fleet coverage, 128 u cells
        # over the graph's box, dense (a labyrinth is ~10^3 cells, a surf
        # map ~10^6)
        nz, ny, nx = np.asarray(graph.node_of).shape
        ext = np.array([nx, ny, nz], np.float64) * float(graph.cell)
        self.nov_mins = np.asarray(graph.mins, np.float64)
        self.nov_shape = tuple(int(v) for v in
                               np.maximum(1, np.ceil(ext / NOVELTY_CELL_U)))
        self.nov_count = np.zeros(self.nov_shape, np.int32)
        self.cover = np.zeros(self.nov_shape, bool)
        self.updates = 0
        self._reset_window()
        self.last_upd = None
        self.last_eval = None

    # ------------------------------------------------------------ helpers
    def describe(self) -> str:
        c = self.cfg
        if self.surf:
            obs = (f"patch {self.wm.n}x{self.wm.n} cells of {self.wm.pc:g} u "
                   f"({self.wm.describe()} + this episode's visits)")
            bud = (f"{self.st.budget_ticks} ticks (= {BUDGET_MULT:g} x "
                   f"T_plan {self.vocab.t_plan:g} s)")
        else:
            obs = (f"patch {self.wm.n}x{self.wm.n} cells of {self.wm.pc:g} u "
                   f"(walkable within +-{self.spec['patch_layers']} graph "
                   f"layer(s) + this episode's visits)")
            bud = (f"{self.st.budget_ticks} ticks (= {self.vocab.length:g} u "
                   f"/ {BUDGET_SPEED_U:g} u/s x {BUDGET_MULT:g})")
        return (f"planner LEARNED: {self.vocab.describe()}; {obs}, "
                f"{self.spec['n_scal']} "
                f"scalars (finish dir + log dist, Euclidean; vel; yaw); a "
                f"plan closes on arc >= {COMPLETE_FRAC:g} of its length "
                f"(corridor {self.corridor:g} u), on its budget "
                f"{bud} or on the "
                f"episode's end; reward {float(c['plan_r_ok']):+g}/{float(c['plan_r_fail']):+g} "
                f"executed / not, +{c['plan_finish_bonus']:g} finish, "
                f"novelty {c['plan_novelty']:g}/sqrt(n) over "
                f"{NOVELTY_CELL_U:g} u plan-end cells (no novelty on a "
                f"death)"
                + (f", progress {c['plan_progress']:g}/1000 u"
                   if float(c["plan_progress"]) else "")
                + f"; PPO gamma {PLAN_GAMMA:g}/plan lambda {PLAN_LAMBDA:g} "
                f"clip {PLAN_CLIP:g} lr {float(c['plan_lr']):g} ent "
                f"{float(c['plan_ent']):g} epochs {int(c['plan_epochs'])} "
                f"batch >= {int(c['plan_batch'])} plans, "
                f"{sum(p.numel() for p in self.net.parameters()):,} params")

    def set_tick_ms(self, tick_ms: float) -> None:
        self.tick_ms = float(tick_ms)
        self.st.set_tick_ms(tick_ms)

    def _wm_for(self, graph) -> WalkMap:
        wm = self._wms.get(id(graph))
        if wm is None:
            if self.surf:
                from .goalsurf import SlabMap
                wm = SlabMap.for_graph(graph, self.spec)
            else:
                wm = WalkMap(graph, self.spec["patch_cell_u"],
                             self.spec["patch_n"], self.spec["patch_layers"])
            self._wms[id(graph)] = wm
        return wm

    def _cells(self, pos) -> tuple:
        p = np.atleast_2d(np.asarray(pos, np.float64))
        k = np.floor((p - self.nov_mins[None, :]) / NOVELTY_CELL_U).astype(
            np.int64)
        for a in range(3):
            np.clip(k[:, a], 0, self.nov_shape[a] - 1, out=k[:, a])
        return k[:, 0], k[:, 1], k[:, 2]

    def _reset_window(self) -> None:
        self.w = {"closed": 0, "complete": 0, "chosen": 0, "wall": 0,
                  "wall_base": 0.0, "wall_base_n": 0, "wall_len": 0.0,
                  "wall_len_base": 0.0, "ent": 0.0,
                  "nov": 0.0, "nov_n": 0,
                  "rew": 0.0, "ep": 0, "fin": 0, "ep_start": 0,
                  "fin_start": 0,
                  "shapes": np.zeros(self.K, np.int64)}
        if self.surf:
            # the surf spec's extra diagnostic: plans that END below the
            # map's kill ceiling (into the fall net), chosen vs base rate
            self.w["void"] = 0
            self.w["void_base"] = 0.0

    # --------------------------------------------------------- the fleet
    def request(self, idx, origins=None) -> None:
        """Envs ``idx`` start a NEW episode (the goal system's assign): they
        need a plan, and any plan still open is dropped (only possible at
        startup - an episode end closes the open plan first)."""
        idx = np.asarray(idx, np.int64).reshape(-1)
        if not len(idx):
            return
        self.st.active[idx] = False
        self.st.need[idx] = True
        self.fresh[idx] = True
        if origins is not None and self.start_pts is not None:
            o = np.atleast_2d(np.asarray(origins, np.float64))
            d = np.linalg.norm(o[:, None, :] - self.start_pts[None, :, :],
                               axis=2).min(axis=1)
            self.from_start[idx] = d < 1.0

    def on_tick(self, pos, ended, finished, died, term_pos=None) -> None:
        """After every physics tick of the training fleet. ``pos`` (n, 3):
        POST-step positions (ended rows: their new spawn); ``term_pos``: the
        terminal positions of the ended rows (any content elsewhere);
        ``finished`` = ended in the finish box; ``died`` = a terminal that
        is not a finish (death, stall kill); truncation is neither."""
        pos = np.asarray(pos, np.float64)
        ended = np.asarray(ended, bool)
        finished = np.asarray(finished, bool)
        died = np.asarray(died, bool)
        waiting = self.st.need & ~self.st.active
        closed, comp = self.st.tick(pos, ended)
        live = ~ended
        if live.any():
            cx, cy, cz = self._cells(pos[live])
            self.cover[cx, cy, cz] = True
        fb = float(self.cfg["plan_finish_bonus"])
        if closed.any():
            ci = np.flatnonzero(closed)
            endp = pos[ci].copy()
            e = ended[ci]
            if e.any() and term_pos is not None:
                endp[e] = np.asarray(term_pos, np.float64)[ci[e]]
            r = np.where(comp[ci], float(self.cfg["plan_r_ok"]),
                         float(self.cfg["plan_r_fail"])).astype(np.float64)
            r += fb * finished[ci]
            nov = np.zeros(len(ci), np.float64)
            alive = ~died[ci]
            if alive.any():
                ax, ay, az = self._cells(endp[alive])
                # batch-safe: two plans ending in one cell on one tick are
                # the n-th and (n+1)-th visits, like the reward's counts
                keys = (ax * self.nov_shape[1] + ay) * self.nov_shape[2] + az
                flat = self.nov_count.reshape(-1)
                order = np.argsort(keys, kind="stable")
                ks = keys[order]
                first = np.r_[True, ks[1:] != ks[:-1]]
                grp = np.cumsum(first) - 1
                start = np.flatnonzero(first)
                rank = np.arange(len(ks)) - start[grp]
                n_after = flat[ks] + rank + 1
                nv = np.empty(len(ks), np.float64)
                nv[order] = float(self.cfg["plan_novelty"]) / np.sqrt(n_after)
                np.add.at(flat, ks, 1)
                nov[alive] = nv
            r += nov
            coef = float(self.cfg["plan_progress"])
            if coef:
                d1 = np.linalg.norm(endp - self.finish[None, :], axis=1)
                r += coef * (self.o_d0[ci] - d1) / 1000.0
            for j, i in enumerate(ci):
                self.buf[i].append((self.o_img[i].copy(),
                                    self.o_scal[i].copy(),
                                    int(self.o_act[i]), float(self.o_logp[i]),
                                    float(self.o_val[i]), float(r[j]),
                                    bool(e[j])))
            w = self.w
            w["closed"] += len(ci)
            w["complete"] += int(comp[ci].sum())
            w["nov"] += float(nov[alive].sum())
            w["nov_n"] += int(alive.sum())
            w["rew"] += float(r.sum())
        # an episode that ends while its env WAITS for the next plan (the
        # last one closed on an earlier tick of this decision): the last
        # transition becomes terminal and a finish is paid to it. A last
        # transition that is ALREADY terminal belongs to the previous
        # episode (this one ended before its first plan): nothing to credit
        late = waiting & ended
        if late.any():
            for i in np.flatnonzero(late):
                if self.buf[i] and not self.buf[i][-1][6]:
                    t = self.buf[i][-1]
                    self.buf[i][-1] = t[:5] + (t[5] + fb * float(finished[i]),
                                               True)
        if ended.any():
            ei = np.flatnonzero(ended)
            self.w["ep"] += len(ei)
            self.w["fin"] += int(finished[ei].sum())
            fs = self.from_start[ei]
            self.w["ep_start"] += int(fs.sum())
            self.w["fin_start"] += int((finished[ei] & fs).sum())

    def plan(self, pos, vel, yaw_deg):
        """Choose plans for every env waiting for one: ONE batched forward,
        sampled from the categorical. -> (idx, lines, fresh) with ``fresh``
        the rows that are at an episode start."""
        idx = np.flatnonzero(self.st.need)
        if not len(idx):
            return idx, [], np.zeros(0, bool)
        p = np.asarray(pos, np.float64)[idx]
        if self.surf:
            from .goalsurf import build_obs_surf
            img, scal = build_obs_surf(self.wm, self.st.visits, idx, p,
                                       np.asarray(vel, np.float64)[idx],
                                       np.asarray(yaw_deg, np.float64)[idx],
                                       self.finish,
                                       int(self.spec["visit_cap"]))
        else:
            img, scal = build_obs(self.wm, self.st.visits, idx, p,
                                  np.asarray(vel, np.float64)[idx],
                                  np.asarray(yaw_deg, np.float64)[idx],
                                  self.finish, int(self.spec["visit_cap"]))
        with torch.no_grad():
            logits, v = self.net(torch.as_tensor(img, device=self.device),
                                 torch.as_tensor(scal, device=self.device))
            logits = logits.float()
            probs = torch.softmax(logits, dim=-1)
            a = torch.multinomial(probs, 1, generator=self.gen).squeeze(1)
            lp = F.log_softmax(logits, dim=-1).gather(
                1, a.unsqueeze(1)).squeeze(1)
            ent = _entropy(logits)
        a_np = a.cpu().numpy().astype(np.int64)
        self.o_img[idx] = img
        self.o_scal[idx] = scal
        self.o_act[idx] = a_np
        self.o_logp[idx] = lp.cpu().numpy()
        self.o_val[idx] = v.float().cpu().numpy()
        self.o_d0[idx] = np.linalg.norm(p - self.finish[None, :], axis=1)
        if self.surf:
            return self._plan_surf_tail(idx, p, np.asarray(vel, np.float64),
                                        a_np, ent)
        # diagnostics: does the CHOSEN plan cross a wall (every choice), and
        # how often does the whole vocabulary at the same states (the base
        # rate; estimated on at most WALL_BASE_ROWS of this call's states -
        # 80 shapes x ~48 samples each is the expensive part at fleet scale)
        iz = self.wm.layer(p)
        w = self.w
        w["chosen"] += len(idx)
        cx_, fr_ = vocab_offgraph(self.vocab, self.wm, iz, p, shapes=a_np)
        w["wall"] += int(cx_.sum())
        w["wall_len"] += float(fr_.sum())
        nb = min(len(idx), WALL_BASE_ROWS)
        bx_, bf_ = vocab_offgraph(self.vocab, self.wm, iz[:nb], p[:nb])
        w["wall_base"] += float(bx_.mean(1).sum())
        w["wall_len_base"] += float(bf_.mean(1).sum())
        w["wall_base_n"] += nb
        w["ent"] += float(ent.sum())
        np.add.at(w["shapes"], a_np, 1)
        fresh = self.fresh[idx].copy()
        self.fresh[idx] = False
        lines = self.st.begin(idx, a_np, p)
        return idx, lines, fresh

    def _plan_surf_tail(self, idx, p, vel, a_np, ent):
        """plan()'s second half under the surf spec: each shape is sized
        for the speed the env has NOW (surfgym.goalsurf.SurfVocab), and the
        diagnostics are the 3-D ones - the share of chosen plans that cross
        SOLID (and of their length inside it), the share whose END lies
        below the map's kill ceiling, each against the whole vocabulary's
        base rate at the same states. Measured, never a reward or a
        filter."""
        from .goalsurf import WALL_BASE_ROWS_SURF, surf_vocab_crossing
        v = vel[idx]
        L = self.vocab.length_for(np.hypot(v[:, 0], v[:, 1]))
        w = self.w
        w["chosen"] += len(idx)
        cx_, fr_, vd_ = surf_vocab_crossing(self.vocab, self.wm, p, L,
                                            shapes=a_np, kill_z=self.kill_z)
        w["wall"] += int(cx_.sum())
        w["wall_len"] += float(fr_.sum())
        w["void"] += int(vd_.sum())
        nb = min(len(idx), WALL_BASE_ROWS_SURF)
        bx_, bf_, bv_ = surf_vocab_crossing(self.vocab, self.wm, p[:nb],
                                            L[:nb], kill_z=self.kill_z)
        w["wall_base"] += float(bx_.mean(1).sum())
        w["wall_len_base"] += float(bf_.mean(1).sum())
        w["void_base"] += float(bv_.mean(1).sum())
        w["wall_base_n"] += nb
        w["ent"] += float(ent.sum())
        np.add.at(w["shapes"], a_np, 1)
        fresh = self.fresh[idx].copy()
        self.fresh[idx] = False
        lines = self.st.begin(idx, a_np, p, lengths=L)
        return idx, lines, fresh

    def wall_all(self, iz, p, wm: Optional[WalkMap] = None) -> np.ndarray:
        """:func:`vocab_crossings` on this planner's vocabulary."""
        return vocab_crossings(self.vocab, wm or self.wm, iz, p)

    # ------------------------------------------------------------ PPO
    def n_ready(self) -> int:
        return int(sum(len(b) for b in self.buf))

    def update(self, force: bool = False):
        """One PPO update over every closed plan since the last one, when at
        least --plan-batch have closed. -> stats dict, or None."""
        n = self.n_ready()
        if n == 0 or (n < int(self.cfg["plan_batch"]) and not force):
            return None
        imgs, scals, acts, logps, advs, rets = [], [], [], [], [], []
        for i, b in enumerate(self.buf):
            if not b:
                continue
            vb = float(self.o_val[i]) if self.st.active[i] else 0.0
            adv, ret = plan_gae([t[5] for t in b], [t[4] for t in b],
                                [t[6] for t in b], vb)
            for t in b:
                imgs.append(t[0])
                scals.append(t[1])
                acts.append(t[2])
                logps.append(t[3])
            advs.append(adv)
            rets.append(ret)
            b.clear()
        dev = self.device
        img = torch.as_tensor(np.stack(imgs), device=dev)
        scal = torch.as_tensor(np.stack(scals), device=dev)
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
        st = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "n_mb": 0}
        epochs = max(1, int(self.cfg["plan_epochs"]))
        for ep in range(epochs):
            perm = self.np_rng.permutation(M)
            for s in range(0, M, mb):
                j = torch.as_tensor(perm[s:s + mb], device=dev)
                logits, v = self.net(img[j], scal[j])
                lp_all = F.log_softmax(logits.float(), dim=-1)
                lp = lp_all.gather(1, act[j].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(lp - old[j])
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
                         "ret_mean": float(ret.mean())}
        return self.last_upd

    # ------------------------------------------------------------ logging
    def pop_window(self) -> dict:
        w = self.w
        rate = (lambda a, b: (a / b) if b else float("nan"))
        out = {"closed": w["closed"],
               "complete": rate(w["complete"], w["closed"]),
               "chosen": w["chosen"],
               "wall": rate(w["wall"], w["chosen"]),
               "wall_base": rate(w["wall_base"], w["wall_base_n"]),
               "wall_len": rate(w["wall_len"], w["chosen"]),
               "wall_len_base": rate(w["wall_len_base"], w["wall_base_n"]),
               "entropy": rate(w["ent"], w["chosen"]),
               "distinct": int((w["shapes"] > 0).sum()),
               "novelty": rate(w["nov"], w["nov_n"]),
               "reward": rate(w["rew"], w["closed"]),
               "ep": w["ep"], "finish": rate(w["fin"], w["ep"]),
               "ep_start": w["ep_start"],
               "finish_start": rate(w["fin_start"], w["ep_start"]),
               "coverage": int(self.cover.sum())}
        if self.surf:
            out["void"] = rate(w["void"], w["chosen"])
            out["void_base"] = rate(w["void_base"], w["wall_base_n"])
        self._reset_window()
        return out

    def note_and_row(self):
        """-> (step-line text, progress.csv values in PLAN_COLS order);
        resets the window and hands over the last eval + update once."""
        w = self.pop_window()
        u = self.last_upd
        ev = self.last_eval
        self.last_upd = None
        self.last_eval = None

        def f(v, nd):
            return round(float(v), nd) if v == v else ""
        row = [w["closed"], f(w["complete"], 4), f(w["wall"], 4),
               f(w["wall_base"], 4), f(w["wall_len"], 4),
               f(w["wall_len_base"], 4), f(w["finish"], 4),
               f(w["finish_start"], 4),
               (f(ev[0] / ev[1], 4) if ev and ev[1] else ""),
               w["coverage"], f(w["entropy"], 4), w["distinct"],
               f(w["novelty"], 4), f(w["reward"], 4),
               (f(u["loss_pi"], 5) if u else ""),
               (f(u["loss_v"], 5) if u else ""),
               (f(u["kl"], 6) if u else ""), self.updates]
        if self.surf:
            # PLAN_COLS_SURF's two extra columns, LAST
            row += [f(w["void"], 4), f(w["void_base"], 4)]
        pc = (lambda v: f"{v:.1%}" if v == v else "-")
        txt = (f"  PLAN chosen {w['chosen']} (H "
               + (f"{w['entropy']:.2f}" if w["chosen"] else "-")
               + f", k {w['distinct']}/{self.K}, wall {pc(w['wall'])} vs "
               f"base {pc(w['wall_base'])}, "
               + ("solid" if self.surf else "off-graph") + " length "
               f"{pc(w['wall_len'])} vs {pc(w['wall_len_base'])}) closed "
               f"{w['closed']} (cmpl "
               f"{pc(w['complete'])}"
               + (f", nov {w['novelty']:.3f}, r {w['reward']:+.2f}"
                  if w["closed"] else "") + ")"
               + (f" fin {pc(w['finish'])}/{w['ep']} from-start "
                  f"{pc(w['finish_start'])}/{w['ep_start']}"
                  if w["ep"] else "")
               + f" cov {w['coverage']}"
               + (f" void {pc(w['void'])} vs {pc(w['void_base'])}"
                  if self.surf else "")
               + (f" | upd {u['updates']} n {u['n']} pi {u['loss_pi']:+.4f} "
                  f"v {u['loss_v']:.4f} H {u['entropy']:.3f} "
                  f"kl {u['kl']:.4f}" if u else ""))
        return txt, row

    # ----------------------------------------------------- checkpointing
    def state_dict_all(self) -> dict:
        return {"spec": dict(self.spec), "cfg": dict(self.cfg),
                "net": {k: v.detach().cpu() for k, v in
                        self.net.state_dict().items()},
                "opt": self.opt.state_dict(),
                "nov_count": self.nov_count.copy(),
                "cover": self.cover.copy(), "updates": int(self.updates)}

    def load_state_dict_all(self, sd: dict) -> None:
        if dict(sd.get("spec") or {}) != self.spec:
            raise ValueError(f"planner checkpoint spec {sd.get('spec')} != "
                             f"this build's {self.spec}")
        self.net.load_state_dict(sd["net"])
        if sd.get("opt"):
            self.opt.load_state_dict(sd["opt"])
            for g in self.opt.param_groups:
                g["lr"] = float(self.cfg["plan_lr"])
        nc = sd.get("nov_count")
        if nc is not None and tuple(np.shape(nc)) == self.nov_shape:
            self.nov_count[...] = nc
            self.cover[...] = sd.get("cover", self.cover)
        self.updates = int(sd.get("updates", 0))

    # --------------------------------------------------------------- eval
    def eval_hooks(self, core, ev: dict, *, line=None, graph=None,
                   finish_radius: Optional[float] = None):
        """(episode_meta, on_tick) for record_rollout on a 1-env core: this
        network, GREEDY, on ``graph`` (default the training map's)."""
        g = graph if graph is not None else self.graph
        return make_learned_hooks(self.net, self.vocab, g, core, ev,
                                  line=line, act_every=self.act_every,
                                  tick_ms=self.tick_ms,
                                  corridor=self.corridor, device=self.device,
                                  spec=self.spec, wm=self._wm_for(g),
                                  finish_radius=finish_radius)


def planner_from_state(sd: dict, device="cpu"):
    """A checkpoint's ``planner`` entry -> (net, vocab, spec) for a
    recording (tools/record_ckpt.py): the network in eval, no optimizer."""
    if not sd or "net" not in sd:
        raise ValueError("no learned-planner weights in this checkpoint")
    spec = dict(sd.get("spec") or _spec_default())
    if spec.get("vocab") == "surf":
        from .goalsurf import make_vocab
        vocab = make_vocab(spec)
    else:
        vocab = PlanVocab(spec["headings"], spec["turns_deg"], spec["segs"],
                          spec["seg_u"])
    net = PlannerNet(vocab.K, spec["patch_n"], spec["n_scal"],
                     spec["hidden"],
                     in_ch=int(spec.get("in_ch", 2))).to(torch.device(device))
    net.load_state_dict(sd["net"])
    net.eval()
    return net, vocab, spec


def make_learned_hooks(net, vocab: PlanVocab, graph, core, ev: dict, *,
                       line=None, act_every: int = 1, tick_ms: float = 10.0,
                       corridor: float = 192.0, device="cpu",
                       spec: Optional[dict] = None, wm=None,
                       finish_radius: Optional[float] = None):
    """(episode_meta, on_tick) for record_rollout on a core whose env 0 is
    recorded - the ONE implementation of the learned planner's eval, shared
    by the trainer (GoalSystem.eval_hooks) and tools/record_ckpt.py.

    From wherever the core spawned env 0 (the map start on the trainer's
    eval core and in a default recording) the end goal is the finish BOX,
    the planner is GREEDY (argmax) and re-plans exactly like training: when
    the plan completes, when its budget runs out, at the next executor
    decision boundary (the policy wrapper decides on ticks t with
    t % act_every == 0). Success is the core's own box test. ``ev`` is the
    caller's tally dict, updated in place (n, succ, ticks, dists, plans,
    closed, complete, wall)."""
    spec = dict(spec or _spec_default())
    dev = torch.device(device)
    surf = spec.get("vocab") == "surf"
    if surf:
        # --plan-vocab surf: the occupancy-slab observation, shapes sized
        # for the speed at each choice, the 3-D diagnostics
        from .goalsurf import SlabMap, build_obs_surf, surf_vocab_crossing
        wm = wm or SlabMap.for_graph(graph, spec)
    else:
        wm = wm or WalkMap(graph, spec["patch_cell_u"], spec["patch_n"],
                           spec["patch_layers"])
    st = PlanState(1, wm, vocab, tick_ms, corridor)
    finish = np.asarray(graph.finish_center, np.float64)
    kill_z = float(getattr(graph, "kill_z", -np.inf))
    K = max(1, int(act_every))
    ev.update({"n": 0, "succ": 0, "pending": False, "center": None,
               "ticks": [], "dists": [], "t0": 0, "box": True, "plans": 0,
               "closed": 0, "complete": 0, "wall": 0, "wall_len": 0.0,
               "shapes": []})
    if surf:
        ev["void"] = 0
        ev["lens"] = []
    zero = np.zeros(1, bool)

    def _choose():
        sv = core.states_view
        p = sv["origin"][0:1].astype(np.float64)
        if surf:
            v = sv["velocity"][0:1].astype(np.float64)
            img, scal = build_obs_surf(wm, st.visits, np.zeros(1, np.int64),
                                       p, v,
                                       sv["yaw"][0:1].astype(np.float64),
                                       finish, int(spec["visit_cap"]))
        else:
            img, scal = build_obs(wm, st.visits, np.zeros(1, np.int64), p,
                                  sv["velocity"][0:1].astype(np.float64),
                                  sv["yaw"][0:1].astype(np.float64), finish,
                                  int(spec["visit_cap"]))
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(img, device=dev),
                            torch.as_tensor(scal, device=dev))
        k = int(logits.float().argmax(-1)[0])
        if surf:
            L = vocab.length_for(np.hypot(v[:, 0], v[:, 1]))
            lines = st.begin(np.zeros(1, np.int64), [k], p, lengths=L)
        else:
            lines = st.begin(np.zeros(1, np.int64), [k], p)
        if line is not None:
            line.set_lines(np.array([0]), lines)
        if surf:
            cx_, fr_, vd_ = surf_vocab_crossing(vocab, wm, p, L, shapes=[k],
                                                kill_z=kill_z)
            ev["void"] += int(vd_[0])
            ev["lens"].append(float(L[0]))
        else:
            cx_, fr_ = vocab_offgraph(vocab, wm, wm.layer(p), p, shapes=[k])
        ev["wall"] += int(cx_[0])
        ev["wall_len"] += float(fr_[0])
        ev["plans"] += 1
        ev["shapes"].append(k)
        return k, lines[0]

    def episode_meta(ep):
        st.visits.reset([0])
        p = core.states_view["origin"][0:1].astype(np.float64)
        st.visits.update(p)
        k, ln = _choose()
        ev["n"] += 1
        s = int(graph.snap(p)[0])
        dfin = (float(graph.dist[graph.fin, s]) if graph.fin is not None
                else float("nan"))
        ev["dists"].append(dfin)
        thin = ln[:: max(1, len(ln) // 64)]
        rad = float(finish_radius) if finish_radius is not None else 192.0
        plan = {"planner": "learned", "shape": k,
                "heading": float(vocab.heading_deg[k]),
                "turn": float(vocab.turn_deg[k]),
                # a surf map's walkable graph need not connect the start to
                # the finish (edgeflow's does not): null, never Infinity,
                # which a JSON reader in the viewer refuses
                "graph_dist": (round(dfin, 1) if np.isfinite(dfin)
                               else None)}
        if surf:
            plan["pitch"] = float(vocab.pitch_deg[k])
            plan["length"] = round(float(ev["lens"][-1]), 1)
        return {"goal": {"center": [float(v) for v in finish],
                         "radius": rad},
                "line": [[float(v) for v in q] for q in thin],
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
