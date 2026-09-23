"""goalsurf.py - plans for SURF maps: the speed-scaled 3-D vocabulary
(``--plan-vocab surf``), the planner's occupancy-slab observation, and the
executor's PLAN DIET behind ``--goal-planner vocab`` (stage (b) of
docs/litsurvey-planner-executor.md section 7.2).

Why surf needs its own vocabulary. The walking vocabulary (surfgym/goallearn.py
PlanVocab) is 80 flat 800 u shapes: 800 u is 3 s of walking but only 0.2-0.5 s
of surfing, and a flat line through the air is a line the player falls away
from. A surf plan therefore carries a TIME (it is T_plan seconds of travel at
the speed the agent has when the plan is drawn) and a DESCENT.

THE VOCABULARY (:class:`SurfVocab`), generic geometry, identical on every map:
16 world headings x 3 turns (0, +45, -45 deg of total turn, + = left) x 3
descent profiles (0, -20, -40 deg of pitch, constant along the path) = 144
shapes of 8 equal segments. A shape's total (3-D arc) length is

    L = clamp(T_plan x max(|v_xy|, 500 u/s), 800 u, 6000 u),  T_plan = 3 s,

500 u/s being the fan's own speed floor (surfgym.goals.MultiLine). Anchored
at the agent; its time budget is 1.5 x T_plan (goallearn's budget rule -
length / speed x 1.5 - with the speed that sized it). NOTHING filters a
shape: it may run through solid or into the pit, and the only things that
say so are the diagnostics below and the executor failing to complete it.

THE OBSERVATION (:class:`SlabMap`, :func:`build_obs_surf`): the user's "3-D
information ... voxels around the agent", in a sparse form. Three horizontal
SLABS of the goal occupancy grid (the solid grid the goal field and the
walkable graph are built on), 256 u thick, centred 384 u below, 128 u below
and 128 u above the agent - together they tile [-512, +256] u around it -
each seen as a world-aligned 32 x 32 patch of 64 u cells holding the SOLID
FRACTION of that 64 x 64 x 256 u block; plus this episode's visit channel;
plus the walking planner's 9 scalars (finish direction and log distance,
Euclidean; velocity; yaw). Outside the grid reads solid (a BSP's outside is
solid, and the grid's own margin is).

THE DIAGNOSTICS, logged and never a reward or a filter: the share of plans
whose polyline crosses SOLID in 3-D (samples every L/64 along the arc; the
first two cells skipped, since a surfer's own cell can read solid on the
conservative slab-catching grid while it rides a ramp face), the share of
their length inside solid, and the share whose END lies below the map's kill
ceiling (surfgym.goalplan.kill_ceiling: into the fall net) - each against the
whole vocabulary's base rate at the same states.

THE DIET (:class:`VocabDiet`, ``--goal-planner vocab``): the executor trains,
from scratch, on plans from two sources and the goal-arc reward along the
CURRENT plan. Each plan draws its source: with probability 1 - h a shape
uniform over the vocabulary; with probability h = ``--plan-hindsight`` a
HINDSIGHT plan - a reached-state segment of the policy's OWN flight (the
reservoir's goal harvest, goalsys kind 0: a snapshot, the state the same
episode reached k seconds later, and the snapshots between), feasible by
construction:

* at an episode START from a reservoir row that carries one, the plan is that
  row's own segment - the env stands exactly where the flight began;
* otherwise (mid-episode re-plans, map-start spawns) the segment of the pool
  row NEAREST the agent's state, key (position, velocity x 1 s): a velocity
  mismatch counts as the drift it makes in one second, accepted within the
  goal radius (192 u, the arrival radius the executor is trained on), its
  displacement anchored at the agent. None within reach -> a vocabulary shape
  (counted: plan/hs_miss).

A hindsight plan's budget is 1.5 x the time the policy took to fly it. Plans
re-plan exactly like the learned planner's (goallearn.PlanState): a plan
closes on 90% arc completion (corridor = the goal radius), on its budget, or
at the episode's end, and the next is drawn at the next executor decision.

THE EXECUTOR-ONLY EVAL (:func:`make_vocab_hooks`): the real task from the map
spawn, re-planned on the training clock, each plan the vocabulary shape whose
END lies nearest the finish-box centre (Euclidean; sized for the current
speed). This is the planner that knows only where the finish is - the flat
control of section 7.6 - so it is stationary (no reservoir, no network), it
is what a map with no learned prior gets, and on a detour map like edgeflow it
is EXPECTED to fly at the finish across the pit: stage (b)'s verdict is plan
following (completion by source, the tracking diagnostics), and the finish
from the true start is stage (c)'s, the learned planner's.

Nothing here reads a demo, a route file or a map-specific constant (CLAUDE.md
rules 0 and 0b): every constant is a time (T_plan, the 1 s drift), a player or
fan constant (500 u/s, the 192 u goal radius, 64 u cells), an angle or a
count, the same on every map.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .goallearn import (BUDGET_MULT, COMPLETE_FRAC, PlanState, PATCH_CELL_U,
                        PATCH_N, VISIT_CAP, N_SCAL, NET_HIDDEN)

__all__ = ["SurfVocab", "SlabMap", "VocabDiet", "make_vocab", "surf_spec",
           "build_obs_surf", "surf_vocab_crossing", "line_crossing",
           "make_vocab_hooks", "DIET_COLS", "DIET_SEED_OFFSET",
           "WALL_BASE_ROWS_SURF"]

# ------------------------------------------------------------ the vocabulary
SURF_HEADINGS = 16                           # world-frame initial headings
SURF_TURNS_DEG = (0.0, 45.0, -45.0)          # total turn, + = left
SURF_PITCH_DEG = (0.0, -20.0, -40.0)         # constant pitch along the path
SURF_SEGS = 8                                # equal segments per shape
T_PLAN_S = 3.0                               # seconds of travel per plan
PLAN_SPEED_FLOOR = 500.0                     # u/s: MultiLine's speed floor
PLAN_LEN_MIN = 800.0                         # u
PLAN_LEN_MAX = 6000.0                        # u

# ---------------------------------------------------------- the observation
SLAB_OFFSETS_U = (-384.0, -128.0, 128.0)     # slab centres, relative to z
SLAB_U = 256.0                               # slab thickness

# ------------------------------------------------------------ diagnostics
SURF_WALL_SAMPLES = 64                       # samples per shape, every L/64
SKIP_CELLS = 2                               # the start's own cells skipped
WALL_BASE_ROWS_SURF = 4                      # base-rate states per call

# -------------------------------------------------------------- the diet
HS_TAU_S = 1.0                               # velocity mismatch -> drift, s
HS_L_MAX = 768                               # the MultiLine default capacity
DIET_SEED_OFFSET = 6917                      # the diet's RNG: --seed + this

# progress.csv columns under --goal-planner vocab, appended LAST
DIET_COLS = ["plan/closed", "plan/complete", "plan/complete_rand",
             "plan/complete_hs", "plan/hs_share", "plan/hs_miss",
             "plan/wall", "plan/wall_base", "plan/wall_len",
             "plan/wall_len_base", "plan/void", "plan/void_base",
             "plan/len", "plan/finish", "plan/finish_start",
             "plan/eval_finish", "plan/eval_complete"]


def surf_spec() -> dict:
    """Everything that defines a SURF planner network's action index and
    input - stored in a learned checkpoint's spec like the walking one
    (goallearn._spec_default, which has no "vocab" key)."""
    return {"vocab": "surf", "headings": SURF_HEADINGS,
            "turns_deg": list(SURF_TURNS_DEG),
            "pitch_deg": list(SURF_PITCH_DEG), "segs": SURF_SEGS,
            "t_plan": T_PLAN_S, "speed_floor": PLAN_SPEED_FLOOR,
            "len_min": PLAN_LEN_MIN, "len_max": PLAN_LEN_MAX,
            "patch_n": PATCH_N, "patch_cell_u": PATCH_CELL_U,
            "slab_offsets_u": list(SLAB_OFFSETS_U), "slab_u": SLAB_U,
            "visit_cap": VISIT_CAP, "n_scal": N_SCAL, "hidden": NET_HIDDEN,
            "in_ch": len(SLAB_OFFSETS_U) + 1}


def vocab_spec(name) -> Optional[dict]:
    """--plan-vocab -> the learned planner's spec: None for "walk" (the
    walking default, exactly what shipped), surf_spec() for "surf"."""
    if name in (None, "walk"):
        return None
    if name == "surf":
        return surf_spec()
    raise ValueError(f"unknown plan vocabulary {name!r}")


# ==========================================================================
# the vocabulary
# ==========================================================================
class SurfVocab:
    """K = headings x turns x pitches UNIT polylines (3-D arc length 1,
    anchored at the origin), scaled per plan to the length the agent's speed
    asks for. Shape k = (heading h, turn theta, pitch phi), indexed
    ``k = (hi * n_turns + ti) * n_pitch + pi``: segment i points along the
    horizontal angle ``h + theta * (i + 0.5) / segs`` (the chords of an arc,
    as in the walking vocabulary) at a constant pitch phi."""

    kind = "surf"

    def __init__(self, headings: int = SURF_HEADINGS,
                 turns_deg=SURF_TURNS_DEG, pitch_deg=SURF_PITCH_DEG,
                 segs: int = SURF_SEGS, t_plan: float = T_PLAN_S,
                 speed_floor: float = PLAN_SPEED_FLOOR,
                 len_min: float = PLAN_LEN_MIN,
                 len_max: float = PLAN_LEN_MAX,
                 spacing: Optional[float] = None,
                 samples: int = SURF_WALL_SAMPLES):
        from .route import DEFAULT_SPACING, resample_polyline
        self.headings = int(headings)
        self.turns_deg = tuple(float(t) for t in turns_deg)
        self.pitches_deg = tuple(float(p) for p in pitch_deg)
        self.segs = int(segs)
        self.t_plan = float(t_plan)
        self.speed_floor = float(speed_floor)
        self.len_min = float(len_min)
        self.len_max = float(len_max)
        if not 0.0 < self.len_min <= self.len_max:
            raise ValueError(f"plan length clamp [{len_min}, {len_max}]")
        self.spacing = float(spacing or DEFAULT_SPACING)
        nt, npc = len(self.turns_deg), len(self.pitches_deg)
        self.K = self.headings * nt * npc
        self.heading_deg = np.zeros(self.K, np.float64)
        self.turn_deg = np.zeros(self.K, np.float64)
        self.pitch_deg = np.zeros(self.K, np.float64)
        unit = np.zeros((self.K, self.segs + 1, 3), np.float64)
        i = np.arange(self.segs)
        for hi in range(self.headings):
            h = 2.0 * math.pi * hi / self.headings
            for ti, turn in enumerate(self.turns_deg):
                th = math.radians(turn)
                ang = h + th * (i + 0.5) / self.segs
                for pi_, pitch in enumerate(self.pitches_deg):
                    k = (hi * nt + ti) * npc + pi_
                    ph = math.radians(pitch)
                    steps = (1.0 / self.segs) * np.stack(
                        [math.cos(ph) * np.cos(ang),
                         math.cos(ph) * np.sin(ang),
                         np.full(self.segs, math.sin(ph))], axis=1)
                    unit[k, 1:] = np.cumsum(steps, axis=0)
                    self.heading_deg[k] = math.degrees(h)
                    self.turn_deg[k] = turn
                    self.pitch_deg[k] = pitch
        self.unit = unit
        # diagnostic samples at arc fractions (j + 1) / M: the segments are
        # equal, so arc fraction f lies on segment floor(f * segs)
        m = int(samples)
        f = (np.arange(m) + 1.0) / m
        si = np.minimum((f * self.segs).astype(np.int64), self.segs - 1)
        t = (f * self.segs - si)[None, :, None]
        self.sample_f = f
        self.unit_samples = unit[:, si] + t * (unit[:, si + 1] - unit[:, si])
        self.budget_secs = BUDGET_MULT * self.t_plan
        # the longest plan's point count (a straight len_max line)
        self.n_line = int(len(resample_polyline(
            np.array([[0.0, 0.0, 0.0], [self.len_max, 0.0, 0.0]]),
            self.spacing)[0]))
        self.length = self.len_max           # the longest a shape gets

    @classmethod
    def from_spec(cls, spec: dict) -> "SurfVocab":
        return cls(spec["headings"], spec["turns_deg"], spec["pitch_deg"],
                   spec["segs"], spec["t_plan"], spec["speed_floor"],
                   spec["len_min"], spec["len_max"])

    def length_for(self, speed) -> np.ndarray:
        """(k,) horizontal speeds (u/s) -> (k,) plan lengths (u)."""
        s = np.asarray(speed, np.float64).reshape(-1)
        return np.clip(self.t_plan * np.maximum(s, self.speed_floor),
                       self.len_min, self.len_max)

    def anchor(self, k: int, origin, length) -> np.ndarray:
        """Shape ``k``, scaled to ``length`` and anchored at ``origin``,
        resampled at the fan spacing (float32, first point == origin).

        route.resample_polyline's rule - n = round(L / spacing) + 1 points
        at uniform arc fractions - evaluated directly: the shape's segments
        are equal, so arc fraction f lies on segment floor(f * segs). (The
        general helper's per-call diff / norm / cumsum / interp is most of a
        re-plan's cost at fleet scale.)"""
        o = np.asarray(origin, np.float64).reshape(3)
        L = float(length)
        n = max(2, int(round(L / self.spacing)) + 1)
        f = np.linspace(0.0, 1.0, n)
        si = np.minimum((f * self.segs).astype(np.int64), self.segs - 1)
        t = (f * self.segs - si)[:, None]
        u = self.unit[int(k)]
        pts = o[None, :] + L * (u[si] + t * (u[si + 1] - u[si]))
        return pts.astype(np.float32)

    def ends(self, origins, lengths) -> np.ndarray:
        """(k, 3) origins, (k,) lengths -> (k, K, 3) every shape's END."""
        o = np.atleast_2d(np.asarray(origins, np.float64))
        L = np.asarray(lengths, np.float64).reshape(-1)
        return o[:, None, :] + L[:, None, None] * self.unit[None, :, -1, :]

    def describe(self) -> str:
        return (f"plan vocabulary SURF: {self.K} shapes = {self.headings} "
                f"world headings x {len(self.turns_deg)} turns "
                f"({', '.join(f'{t:+g}' for t in self.turns_deg)} deg) x "
                f"{len(self.pitches_deg)} descents "
                f"({', '.join(f'{p:g}' for p in self.pitches_deg)} deg "
                f"pitch), {self.segs} segments; length clamp({self.t_plan:g} "
                f"s x max(|v_xy|, {self.speed_floor:g} u/s), "
                f"{self.len_min:g}, {self.len_max:g}) u, anchored at the "
                f"agent, <= {self.n_line} points at {self.spacing:g} u; budget "
                f"{self.budget_secs:g} s; nothing filters a shape")


def make_vocab(spec: Optional[dict]):
    """A spec -> its vocabulary (SurfVocab for a surf spec, else the walking
    PlanVocab)."""
    if spec is not None and spec.get("vocab") == "surf":
        return SurfVocab.from_spec(spec)
    from .goallearn import PlanVocab, _spec_default
    s = dict(spec or _spec_default())
    return PlanVocab(s["headings"], s["turns_deg"], s["segs"], s["seg_u"])


# ==========================================================================
# perception: occupancy slabs
# ==========================================================================
class SlabMap:
    """The goal occupancy grid seen as horizontal SLABS around the agent.

    ``solid`` is (nz, ny, nx) bool with voxel (iz, iy, ix) centred at
    ``mins + (i + 0.5) * cell``. The patch lattice is WalkMap's: cells of
    s x s voxels (s = round(64 u / cell)) from ``mins``, a window of N x N
    centred on the agent's cell, padded by half a window - with SOLID, since
    outside the grid is outside the world. Slab c covers the voxel layers
    whose span starts at ``z + offset_c - slab_u / 2``, ``tz = round(slab_u /
    cell)`` of them; a cumulative sum over z makes each slab two gathers.
    Layers outside the grid count as solid.

    It carries the interface VisitGrid needs (cells, window, gy, gx, half,
    n), so the visit channel is laid on the same lattice."""

    def __init__(self, solid, mins, cell, patch_cell_u: float = PATCH_CELL_U,
                 n: int = PATCH_N, offsets=SLAB_OFFSETS_U,
                 slab_u: float = SLAB_U):
        solid = np.asarray(solid, bool)
        self.solid = solid
        self.cell = float(cell)
        self.mins3 = np.asarray(mins, np.float64).reshape(3)
        self.mins = self.mins3[:2]
        self.s = max(1, int(round(float(patch_cell_u) / self.cell)))
        self.pc = self.s * self.cell
        self.n = int(n)
        self.half = self.n // 2
        self.offsets = tuple(float(o) for o in offsets)
        self.slab_u = float(slab_u)
        self.tz = max(1, int(round(self.slab_u / self.cell)))
        nz, ny, nx = solid.shape
        self.nz = int(nz)
        py, px = (-ny) % self.s, (-nx) % self.s
        sp = np.pad(solid, ((0, 0), (0, py), (0, px)), constant_values=True)
        gy, gx = (ny + py) // self.s, (nx + px) // self.s
        cnt = sp.reshape(nz, gy, self.s, gx, self.s).sum(
            axis=(2, 4)).astype(np.float32)
        self.gy, self.gx = int(gy), int(gx)
        h = self.half
        full = float(self.s * self.s)
        cnt = np.pad(cnt, ((0, 0), (h, h), (h, h)), constant_values=full)
        cum = np.zeros((nz + 1,) + cnt.shape[1:], np.float32)
        np.cumsum(cnt, axis=0, out=cum[1:])
        self.cum = cum
        self.block = full
        self._ar = np.arange(self.n)

    @classmethod
    def for_graph(cls, graph, spec: dict) -> "SlabMap":
        """The slabs of a BFSPlanner's own occupancy (graph.solid)."""
        return cls(graph.solid, graph.mins, graph.cell,
                   spec.get("patch_cell_u", PATCH_CELL_U),
                   spec.get("patch_n", PATCH_N),
                   spec.get("slab_offsets_u", SLAB_OFFSETS_U),
                   spec.get("slab_u", SLAB_U))

    def describe(self) -> str:
        return (f"{len(self.offsets)} occupancy slabs {self.slab_u:g} u "
                f"thick centred at z "
                f"{', '.join(f'{o:+g}' for o in self.offsets)} u (solid "
                f"fraction of each {self.pc:g} x {self.pc:g} x "
                f"{self.tz * self.cell:g} u block; outside = solid)")

    def cells(self, pos):
        """(k, 3) -> (cy, cx) patch cells, clipped into the lattice."""
        p = np.atleast_2d(np.asarray(pos, np.float64))
        cy = np.floor((p[:, 1] - self.mins[1]) / self.pc).astype(np.int64)
        cx = np.floor((p[:, 0] - self.mins[0]) / self.pc).astype(np.int64)
        return (np.clip(cy, 0, self.gy - 1), np.clip(cx, 0, self.gx - 1))

    def window(self, cy, cx):
        rows = np.asarray(cy, np.int64)[:, None] + self._ar[None, :]
        cols = np.asarray(cx, np.int64)[:, None] + self._ar[None, :]
        return rows, cols

    def slabs(self, pos, cy, cx) -> np.ndarray:
        """(k, 3) positions + cells -> (k, n_slabs, N, N) float32 solid
        fraction."""
        p = np.atleast_2d(np.asarray(pos, np.float64))
        rows, cols = self.window(cy, cx)
        r3, c3 = rows[:, :, None], cols[:, None, :]
        out = np.empty((len(p), len(self.offsets), self.n, self.n),
                       np.float32)
        for c, off in enumerate(self.offsets):
            z0 = p[:, 2] + off - 0.5 * self.slab_u
            i0 = np.floor((z0 - self.mins3[2]) / self.cell).astype(np.int64)
            i1 = i0 + self.tz
            a = np.clip(i0, 0, self.nz)
            b = np.clip(i1, 0, self.nz)
            outside = (self.tz - (b - a)).astype(np.float32) * self.block
            s = (self.cum[b[:, None, None], r3, c3]
                 - self.cum[a[:, None, None], r3, c3]
                 + outside[:, None, None])
            out[:, c] = s / (self.tz * self.block)
        return out

    def solid_at(self, xyz) -> np.ndarray:
        """(..., 3) world points -> (...) bool, True inside a solid voxel or
        outside the grid."""
        q = np.asarray(xyz, np.float64)
        ix = np.floor((q[..., 0] - self.mins3[0]) / self.cell).astype(np.int64)
        iy = np.floor((q[..., 1] - self.mins3[1]) / self.cell).astype(np.int64)
        iz = np.floor((q[..., 2] - self.mins3[2]) / self.cell).astype(np.int64)
        nz, ny, nx = self.solid.shape
        inb = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
               & (iz >= 0) & (iz < nz))
        out = np.ones(ix.shape, bool)
        out[inb] = self.solid[iz[inb], iy[inb], ix[inb]]
        return out


def _plan_scalars(p, vel, yaw_deg, finish) -> np.ndarray:
    """goallearn.build_obs's nine scalars, the same arithmetic: unit vector
    to the finish, log1p(|g| / 1000), velocity / 1000, sin / cos yaw."""
    g = np.asarray(finish, np.float64).reshape(1, 3) - p
    d = np.linalg.norm(g, axis=1)
    u = g / np.maximum(d, 1.0)[:, None]
    v = np.atleast_2d(np.asarray(vel, np.float64)) / 1000.0
    y = np.radians(np.asarray(yaw_deg, np.float64).reshape(-1))
    return np.concatenate([u, np.log1p(d / 1000.0)[:, None], v,
                           np.sin(y)[:, None], np.cos(y)[:, None]], axis=1)


def build_obs_surf(sm: SlabMap, visits, idx, pos, vel, yaw_deg, finish,
                   cap: int = VISIT_CAP):
    """The surf planner's observation for envs ``idx`` -> (img (k, S + 1,
    N, N), scal (k, 9)) float32: the S occupancy slabs, then the visit
    channel (goallearn.VisitGrid on the same lattice)."""
    p = np.atleast_2d(np.asarray(pos, np.float64))
    cy, cx = sm.cells(p)
    sl = sm.slabs(p, cy, cx)
    vis = visits.patch(idx, cy, cx, cap)[:, None]
    img = np.concatenate([sl, vis], axis=1)
    return (np.ascontiguousarray(img, np.float32),
            np.ascontiguousarray(_plan_scalars(p, vel, yaw_deg, finish),
                                 np.float32))


# ==========================================================================
# the 3-D diagnostics
# ==========================================================================
def surf_vocab_crossing(vocab: SurfVocab, sm: SlabMap, p, lengths,
                        shapes=None, kill_z: float = -np.inf):
    """Shapes anchored at ``p`` (k, 3) with lengths (k,) -> (crosses, frac,
    void): some sample inside SOLID, the share of samples inside it, the
    shape's END below ``kill_z``. ``shapes`` None -> (k, K) arrays over the
    whole vocabulary; else (k,) for shape shapes[i] at p[i]. Samples within
    SKIP_CELLS cells of arc from the start are dropped."""
    p = np.atleast_2d(np.asarray(p, np.float64))
    L = np.asarray(lengths, np.float64).reshape(-1)
    skip = SKIP_CELLS * sm.cell
    keep = vocab.sample_f[None, :] * L[:, None] >= skip          # (k, M)
    if shapes is not None:
        sh = np.asarray(shapes, np.int64).reshape(-1)
        xyz = p[:, None, :] + L[:, None, None] * vocab.unit_samples[sh]
        sol = sm.solid_at(xyz) & keep
        n = np.maximum(keep.sum(-1), 1)
        endz = p[:, 2] + L * vocab.unit[sh, -1, 2]
        return sol.any(-1), sol.sum(-1) / n, endz < float(kill_z)
    xyz = (p[:, None, None, :]
           + L[:, None, None, None] * vocab.unit_samples[None])
    sol = sm.solid_at(xyz) & keep[:, None, :]
    n = np.maximum(keep.sum(-1), 1)[:, None]
    endz = p[:, None, 2] + L[:, None] * vocab.unit[None, :, -1, 2]
    return sol.any(-1), sol.sum(-1) / n, endz < float(kill_z)


def line_crossing(sm: SlabMap, line, kill_z: float = -np.inf,
                  samples: int = SURF_WALL_SAMPLES):
    """The same three diagnostics for one arbitrary polyline (a hindsight
    segment): samples at arc fractions (j + 1) / M."""
    q = np.asarray(line, np.float64).reshape(-1, 3)
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    tot = float(cum[-1])
    f = (np.arange(int(samples)) + 1.0) / int(samples)
    s = f * tot
    xyz = np.stack([np.interp(s, cum, q[:, a]) for a in range(3)], axis=1)
    keep = s >= SKIP_CELLS * sm.cell
    sol = sm.solid_at(xyz) & keep
    n = max(int(keep.sum()), 1)
    return bool(sol.any()), float(sol.sum()) / n, bool(q[-1, 2] < kill_z)


def _arc_len(line) -> float:
    q = np.asarray(line, np.float64)
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


# ==========================================================================
# the executor's plan diet (--goal-planner vocab)
# ==========================================================================
class VocabDiet:
    """Stage (b): who writes each env's line while the EXECUTOR trains.

    The interface is the learned planner's (goallearn.LearnedPlanner) so
    the goal system and the trainer drive both the same way: request (new
    episodes), on_tick (every physics tick), plan (every decision boundary,
    for the envs whose plan ended), note_and_row (the log window), eval_hooks,
    set_tick_ms; update is a no-op (nothing here learns)."""

    wants_segments = True          # goalsys hands over the spawns' segments
    eval_label = "vocab eval: the shape ending nearest the finish"
    cols = DIET_COLS

    def __init__(self, graph, n_envs: int, *, spec: Optional[dict] = None,
                 start_pts=None, tick_ms: float = 10.0, act_every: int = 1,
                 corridor: float = 192.0, hindsight: float = 0.5,
                 seed: int = 0, snap_secs: float = 0.25):
        if graph.finish_center is None:
            raise ValueError("the plan diet needs the map's finish box")
        self.graph = graph
        self.spec = dict(spec or surf_spec())
        if self.spec.get("vocab") != "surf":
            raise ValueError("--goal-planner vocab is the SURF diet: it "
                             "needs --plan-vocab surf")
        self.vocab = make_vocab(self.spec)
        self.K = self.vocab.K
        self.sm = SlabMap.for_graph(graph, self.spec)
        self._sms = {id(graph): self.sm}
        self.n = int(n_envs)
        self.act_every = max(1, int(act_every))
        self.corridor = float(corridor)
        self.tick_ms = float(tick_ms)
        self.h = float(hindsight)
        if not 0.0 <= self.h <= 1.0:
            raise ValueError(f"--plan-hindsight is a probability, got {self.h}")
        self.hs_radius = float(corridor)
        self.snap_secs = float(snap_secs)
        self.st = PlanState(self.n, self.sm, self.vocab, tick_ms, corridor,
                            visits=False, l_max=HS_L_MAX)
        self.rng = np.random.default_rng(int(seed))
        self.finish = np.asarray(graph.finish_center, np.float64)
        self.kill_z = float(getattr(graph, "kill_z", -np.inf))
        self.start_pts = (None if start_pts is None else
                          np.atleast_2d(np.asarray(start_pts, np.float64)))
        self.fresh = np.ones(self.n, bool)
        self.from_start = np.zeros(self.n, bool)
        self.src = np.zeros(self.n, np.int8)      # the open plan: 1 = hs
        self.spawn_seg = [None] * self.n
        self.bank = None
        self.bank_n = 0
        self.updates = 0
        self._reset_window()
        self.last_eval = None
        self.last_eval_cmpl = None

    # ------------------------------------------------------------ helpers
    def describe(self) -> str:
        return (f"plan DIET (--goal-planner vocab): {self.vocab.describe()}; "
                f"each plan is a uniform vocabulary shape with p="
                f"{1.0 - self.h:g}, else HINDSIGHT (p={self.h:g}): the "
                f"policy's own reservoir segment - the spawn row's own at an "
                f"episode start, else the pool row nearest (position, "
                f"velocity x {HS_TAU_S:g} s) within {self.hs_radius:g} u, "
                f"anchored at the agent, budget {BUDGET_MULT:g} x its flight "
                f"time (a miss falls back to a shape); a plan closes on arc "
                f">= {COMPLETE_FRAC:g} (corridor {self.corridor:g} u), on its "
                f"budget or at the episode's end; diagnostics (never a "
                f"reward): completion by source, solid crossing and plans "
                f"ending below the kill ceiling vs the vocabulary; "
                f"{self.sm.describe()}")

    def set_tick_ms(self, tick_ms: float) -> None:
        self.tick_ms = float(tick_ms)
        self.st.set_tick_ms(tick_ms)

    def _sm_for(self, graph) -> SlabMap:
        sm = self._sms.get(id(graph))
        if sm is None:
            sm = SlabMap.for_graph(graph, self.spec)
            self._sms[id(graph)] = sm
        return sm

    def _reset_window(self) -> None:
        self.w = {"closed": 0, "complete": 0, "closed_hs": 0,
                  "complete_hs": 0, "chosen": 0, "chosen_hs": 0,
                  "hs_req": 0, "hs_miss": 0, "wall": 0, "wall_len": 0.0,
                  "void": 0, "wall_base": 0.0, "wall_len_base": 0.0,
                  "void_base": 0.0, "base_n": 0, "len": 0.0,
                  "ep": 0, "fin": 0, "ep_start": 0, "fin_start": 0}

    # ------------------------------------------------------ the hindsight
    def set_bank(self, pool, goals, segs, seglen) -> None:
        """This iteration's spawn pool with its goal columns (goalsys
        set_pool): the rows that carry a segment become the hindsight bank,
        keyed by (position, velocity x HS_TAU_S)."""
        from scipy.spatial import cKDTree
        sl = np.asarray(seglen, np.int64).reshape(-1)
        ok = (sl >= 2) & np.all(np.isfinite(np.asarray(goals, np.float64)),
                                axis=1)
        if not ok.any():
            self.bank = None
            self.bank_n = 0
            return
        r = np.flatnonzero(ok)
        org = np.asarray(pool["origin"], np.float64)[r]
        vel = np.asarray(pool["velocity"], np.float64)[r]
        keys = np.hstack([org, vel * HS_TAU_S])
        self.bank = (cKDTree(keys), np.asarray(segs, np.float32)[r].copy(),
                     sl[r].copy())
        self.bank_n = len(r)

    def _hs_line(self, seg, p):
        """A segment -> (line anchored at p, budget ticks, arc length)."""
        from .goals import resample_polyline_np
        s = np.asarray(seg, np.float64)
        line = resample_polyline_np(s - s[0][None, :] + p[None, :])
        if len(line) > HS_L_MAX:
            line = line[:HS_L_MAX]
        secs = max(1, len(s) - 1) * self.snap_secs
        b = int(math.ceil(BUDGET_MULT * secs * self.st.ticks_per_s - 1e-6))
        return line, max(1, b), _arc_len(line)

    # --------------------------------------------------------- the fleet
    def request(self, idx, origins=None, segs=None) -> None:
        """Envs ``idx`` start a NEW episode. ``segs[j]`` is env idx[j]'s
        spawn row's own reached-state segment (None: a map-start row, or a
        row the harvest gave no goal)."""
        idx = np.asarray(idx, np.int64).reshape(-1)
        if not len(idx):
            return
        self.st.active[idx] = False
        self.st.need[idx] = True
        self.fresh[idx] = True
        for j, i in enumerate(idx):
            self.spawn_seg[int(i)] = (None if segs is None else segs[j])
        if origins is not None and self.start_pts is not None:
            o = np.atleast_2d(np.asarray(origins, np.float64))
            d = np.linalg.norm(o[:, None, :] - self.start_pts[None, :, :],
                               axis=2).min(axis=1)
            self.from_start[idx] = d < 1.0

    def on_tick(self, pos, ended, finished, died, term_pos=None) -> None:
        pos = np.asarray(pos, np.float64)
        ended = np.asarray(ended, bool)
        finished = np.asarray(finished, bool)
        closed, comp = self.st.tick(pos, ended)
        w = self.w
        if closed.any():
            ci = np.flatnonzero(closed)
            hs = self.src[ci] == 1
            w["closed"] += len(ci)
            w["complete"] += int(comp[ci].sum())
            w["closed_hs"] += int(hs.sum())
            w["complete_hs"] += int((comp[ci] & hs).sum())
        if ended.any():
            ei = np.flatnonzero(ended)
            w["ep"] += len(ei)
            w["fin"] += int(finished[ei].sum())
            fs = self.from_start[ei]
            w["ep_start"] += int(fs.sum())
            w["fin_start"] += int((finished[ei] & fs).sum())

    def plan(self, pos, vel, yaw_deg):
        """Plans for every env waiting for one -> (idx, lines, fresh)."""
        idx = np.flatnonzero(self.st.need)
        if not len(idx):
            return idx, [], np.zeros(0, bool)
        p = np.asarray(pos, np.float64)[idx]
        v = np.asarray(vel, np.float64)[idx]
        L = self.vocab.length_for(np.hypot(v[:, 0], v[:, 1]))
        n = len(idx)
        coin = self.rng.random(n) < self.h
        fresh = self.fresh[idx].copy()
        segs = [None] * n
        want_nn = []
        for j in np.flatnonzero(coin):
            s = self.spawn_seg[int(idx[j])] if fresh[j] else None
            if s is not None and len(s) >= 2:
                segs[j] = s
            else:
                want_nn.append(j)
        if want_nn and self.bank is not None:
            tree, bsegs, blen = self.bank
            jj = np.asarray(want_nn, np.int64)
            keys = np.hstack([p[jj], v[jj] * HS_TAU_S])
            d, r = tree.query(keys, k=1, distance_upper_bound=self.hs_radius)
            for j, dj, rj in zip(jj, np.atleast_1d(d), np.atleast_1d(r)):
                if np.isfinite(dj) and rj < self.bank_n:
                    segs[int(j)] = bsegs[rj, :int(blen[rj])]
        for i in idx[fresh]:
            self.spawn_seg[int(i)] = None      # a spawn's own: first plan only
        lines, budgets = [None] * n, np.full(n, self.st.budget_ticks, np.int64)
        lens = L.copy()
        for j in range(n):
            if segs[j] is None:
                continue
            try:
                lines[j], budgets[j], lens[j] = self._hs_line(segs[j], p[j])
            except ValueError:
                # a segment with < 2 distinct points (the harvest's goal
                # distance floor makes this unreachable): a miss
                segs[j] = None
                lens[j] = L[j]
        hs = np.array([s is not None for s in segs], bool)
        rnd = np.flatnonzero(~hs)
        shapes = np.full(n, -1, np.int64)
        if len(rnd):
            shapes[rnd] = self.rng.integers(0, self.K, size=len(rnd))
        w = self.w
        for j in range(n):
            if hs[j]:
                cx_, fr_, vd_ = line_crossing(self.sm, lines[j], self.kill_z)
                w["wall"] += int(cx_)
                w["wall_len"] += fr_
                w["void"] += int(vd_)
            else:
                lines[j] = self.vocab.anchor(int(shapes[j]), p[j], L[j])
        if len(rnd):
            cx_, fr_, vd_ = surf_vocab_crossing(self.vocab, self.sm, p[rnd],
                                                L[rnd], shapes=shapes[rnd],
                                                kill_z=self.kill_z)
            w["wall"] += int(cx_.sum())
            w["wall_len"] += float(fr_.sum())
            w["void"] += int(vd_.sum())
        nb = min(n, WALL_BASE_ROWS_SURF)
        bx_, bf_, bv_ = surf_vocab_crossing(self.vocab, self.sm, p[:nb],
                                            L[:nb], kill_z=self.kill_z)
        w["wall_base"] += float(bx_.mean(1).sum())
        w["wall_len_base"] += float(bf_.mean(1).sum())
        w["void_base"] += float(bv_.mean(1).sum())
        w["base_n"] += nb
        w["chosen"] += n
        w["chosen_hs"] += int(hs.sum())
        w["hs_req"] += int(coin.sum())
        w["hs_miss"] += int((coin & ~hs).sum())
        w["len"] += float(lens.sum())
        self.st.begin(idx, shapes, p, lines=lines, budgets=budgets)
        self.src[idx] = hs.astype(np.int8)
        self.fresh[idx] = False
        return idx, lines, fresh

    def update(self, force: bool = False):
        return None                    # the diet learns nothing

    def n_ready(self) -> int:
        return 0

    # ------------------------------------------------------------ logging
    def pop_window(self) -> dict:
        w = self.w
        rate = (lambda a, b: (a / b) if b else float("nan"))
        out = {"closed": w["closed"],
               "complete": rate(w["complete"], w["closed"]),
               "complete_rand": rate(w["complete"] - w["complete_hs"],
                                     w["closed"] - w["closed_hs"]),
               "complete_hs": rate(w["complete_hs"], w["closed_hs"]),
               "closed_hs": w["closed_hs"],
               "chosen": w["chosen"],
               "hs_share": rate(w["chosen_hs"], w["chosen"]),
               "hs_req": rate(w["hs_req"], w["chosen"]),
               "hs_miss": rate(w["hs_miss"], w["hs_req"]),
               "wall": rate(w["wall"], w["chosen"]),
               "wall_base": rate(w["wall_base"], w["base_n"]),
               "wall_len": rate(w["wall_len"], w["chosen"]),
               "wall_len_base": rate(w["wall_len_base"], w["base_n"]),
               "void": rate(w["void"], w["chosen"]),
               "void_base": rate(w["void_base"], w["base_n"]),
               "len": rate(w["len"], w["chosen"]),
               "ep": w["ep"], "finish": rate(w["fin"], w["ep"]),
               "ep_start": w["ep_start"],
               "finish_start": rate(w["fin_start"], w["ep_start"])}
        self._reset_window()
        return out

    def note_and_row(self):
        """-> (step-line text, progress.csv values in DIET_COLS order)."""
        w = self.pop_window()
        ev = self.last_eval
        ec = self.last_eval_cmpl
        self.last_eval = None
        self.last_eval_cmpl = None

        def f(v, nd):
            return round(float(v), nd) if v == v else ""
        row = [w["closed"], f(w["complete"], 4), f(w["complete_rand"], 4),
               f(w["complete_hs"], 4), f(w["hs_share"], 4),
               f(w["hs_miss"], 4), f(w["wall"], 4), f(w["wall_base"], 4),
               f(w["wall_len"], 4), f(w["wall_len_base"], 4),
               f(w["void"], 4), f(w["void_base"], 4), f(w["len"], 1),
               f(w["finish"], 4), f(w["finish_start"], 4),
               (f(ev[0] / ev[1], 4) if ev and ev[1] else ""),
               (f(ec, 4) if ec is not None else "")]
        pc = (lambda v: f"{v:.1%}" if v == v else "-")
        txt = (f"  DIET chosen {w['chosen']} (hs {pc(w['hs_share'])}, miss "
               f"{pc(w['hs_miss'])}, len {w['len']:,.0f}u, solid "
               f"{pc(w['wall'])} vs base {pc(w['wall_base'])}, void "
               f"{pc(w['void'])} vs {pc(w['void_base'])}) closed "
               f"{w['closed']} (cmpl {pc(w['complete'])}: rand "
               f"{pc(w['complete_rand'])} hs {pc(w['complete_hs'])}/"
               f"{w['closed_hs']})"
               + (f" fin {pc(w['finish'])}/{w['ep']} from-start "
                  f"{pc(w['finish_start'])}/{w['ep_start']}"
                  if w["ep"] else "")
               + (f" bank {self.bank_n}" if self.bank is not None else
                  " bank -"))
        return txt, row

    # --------------------------------------------------------------- eval
    def eval_hooks(self, core, ev: dict, *, line=None, graph=None,
                   finish_radius: Optional[float] = None):
        g = graph if graph is not None else self.graph
        return make_vocab_hooks(self.vocab, g, core, ev, line=line,
                                act_every=self.act_every,
                                tick_ms=self.tick_ms, corridor=self.corridor,
                                sm=self._sm_for(g), spec=self.spec,
                                finish_radius=finish_radius)


def make_vocab_hooks(vocab: SurfVocab, graph, core, ev: dict, *, line=None,
                     act_every: int = 1, tick_ms: float = 10.0,
                     corridor: float = 192.0, sm: Optional[SlabMap] = None,
                     spec: Optional[dict] = None,
                     finish_radius: Optional[float] = None):
    """(episode_meta, on_tick) for record_rollout on a core whose env 0 is
    recorded - the executor-only eval of --goal-planner vocab, shared by the
    trainer and tools/record_ckpt.py. From wherever the core spawned env 0
    (the map start in the trainer's eval and a default recording) the end
    goal is the finish BOX; each plan is the vocabulary shape, sized for the
    current speed, whose END lies nearest the finish-box centre (ties: the
    lowest index), re-planned exactly like training (on completion or budget,
    at the next decision boundary). ``ev`` is the caller's tally (n, succ,
    ticks, dists, plans, closed, complete, wall, wall_len, void, shapes)."""
    spec = dict(spec or surf_spec())
    sm = sm or SlabMap.for_graph(graph, spec)
    st = PlanState(1, sm, vocab, tick_ms, corridor, visits=False)
    finish = np.asarray(graph.finish_center, np.float64)
    kill_z = float(getattr(graph, "kill_z", -np.inf))
    K = max(1, int(act_every))
    ev.update({"n": 0, "succ": 0, "pending": False, "center": None,
               "ticks": [], "dists": [], "t0": 0, "box": True, "plans": 0,
               "closed": 0, "complete": 0, "wall": 0, "wall_len": 0.0,
               "void": 0, "shapes": [], "lens": []})
    zero = np.zeros(1, bool)

    def _choose():
        sv = core.states_view
        p = sv["origin"][0:1].astype(np.float64)
        v = sv["velocity"][0:1].astype(np.float64)
        L = vocab.length_for(np.hypot(v[:, 0], v[:, 1]))
        e = vocab.ends(p, L)[0]
        k = int(np.argmin(np.linalg.norm(e - finish[None, :], axis=1)))
        lines = st.begin(np.zeros(1, np.int64), [k], p, lengths=L)
        if line is not None:
            line.set_lines(np.array([0]), lines)
        cx_, fr_, vd_ = surf_vocab_crossing(vocab, sm, p, L, shapes=[k],
                                            kill_z=kill_z)
        ev["wall"] += int(cx_[0])
        ev["wall_len"] += float(fr_[0])
        ev["void"] += int(vd_[0])
        ev["plans"] += 1
        ev["shapes"].append(k)
        ev["lens"].append(float(L[0]))
        return k, float(L[0]), lines[0]

    def episode_meta(ep):
        k, L, ln = _choose()
        ev["n"] += 1
        p = core.states_view["origin"][0:1].astype(np.float64)
        dfin = float("nan")
        if getattr(graph, "fin", None) is not None:
            dfin = float(graph.dist[graph.fin, int(graph.snap(p)[0])])
        ev["dists"].append(dfin)
        thin = ln[:: max(1, len(ln) // 64)]
        rad = float(finish_radius) if finish_radius is not None else 192.0
        return {"goal": {"center": [float(v) for v in finish],
                         "radius": rad},
                "line": [[float(v) for v in q] for q in thin],
                "plan": {"planner": "vocab", "shape": k,
                         "heading": float(vocab.heading_deg[k]),
                         "turn": float(vocab.turn_deg[k]),
                         "pitch": float(vocab.pitch_deg[k]),
                         "length": round(L, 1),
                         # null, never Infinity (a surf map's walkable
                         # graph need not connect start and finish)
                         "graph_dist": (round(dfin, 1) if np.isfinite(dfin)
                                        else None)}}

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
