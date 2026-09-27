"""edge_archive.py - Go-Explore over the learned executor's plans (Codex's v1 spec, agent bus
2026-09-26 21:37; the user's pivot after the 3-choice experiment).

    python tools/edge_archive.py <ckpt> --out runs/research/archive_<name> [--minutes 60]
        [--parents 64] [--seed 0] [--map maps_pool/<map>.bsp]

WHY. Under the planner's refund_i reward every failed episode nets 0, so nothing is learned before
the first finish, and a policy's progress into a detour is not KEPT anywhere (the reservoir keeps
states, never the chain that reached them; SIL keeps only finished episodes). An archive keeps
every new kind of state the agent reaches, and how it got there, whether or not that episode later
dies - progress is kept because it is NEW, not because it is rewarded. No reward, no potential, no
map constant enters the archive.

WHAT. A NODE is an exact simulator state (STATE_DTYPE row + the executor wrapper's held keys + the
core observation), reached from its parent by one PLAN - one of the checkpoint's --plan-choices
rays (forward / left / right) - flown by the checkpoint's own executor, SAMPLING its actions (its
native temperature), for the plan's duration (--plan-close commit: the checkpoint's own period),
closing at the executor's next decision tick like the trainer. A node's KEY (Codex's v1):
  * position: 128 u XYZ cell;
  * horizontal speed: < 128, 128-256, 256-512, 512-1024, 1024-2048, >= 2048 u/s;
  * horizontal velocity azimuth: 8 bins when moving (>= 128 u/s), else one 'stationary' bin;
  * vertical velocity: down / level / up with a 128 u/s dead band;
  * contact: onground != -1.
Held keys, view, yaw are stored in the node, NOT in the key. Two elites per key: the FIRST arrival
(fewest ticks from the root) and the FASTEST (highest speed); a new arrival replaces one only if it
is strictly better on that axis.

LOOP. Pick --parents nodes with weight 1 / sqrt(1 + times selected) (count-only: reward-free,
potential-free), fly ALL C choices from each in one batch on a scratch core (the recorder's own
construction, record_ckpt.build --plan-scratch), admit every live endpoint, repeat. A restored
node's episode clock is zeroed (a deep node must not inherit the cap; the node's own tick count
from the root is kept separately). Stops at the first chain that reaches the finish, --minutes, or
--expansions.

OUTPUT (<out>/): progress.jsonl (one line per report: expansions, nodes, keys, position cells,
admissions / duplicates / replacements, the best distance to the finish, the deepest node, novel-
child yield per choice, simulator steps / s), chain.json (the first finishing chain: every node's
origin, the choice that reached it, its tick from the root, and the flown path points), and a
replay of that chain's choices from the true start, without restores, 32 times (the executor
samples): how often the same plan sequence finishes - a chain that exists only as individually
restored edges is not yet a policy trajectory (Codex).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

POS_CELL = 128.0
HSPD_EDGES = np.array([128.0, 256.0, 512.0, 1024.0, 2048.0])
N_AZ = 8
VZ_DEAD = 128.0
REPLAYS = 32
# --rays 3, the move operator's constants - FIXED, the same on every map and for every executor
# (they are what every archive of 2026-09-26/27 flew through the v1ri planners' --rays path)
RAY_DEG = 45.0          # left / right: +-45 deg from the horizontal velocity's heading
RAY_SECS = 2.0          # a move is COMMITTED for 2 s (200 ticks at 10 ms), closing at the next
                        # executor decision
RAY_FLOOR = 300.0       # a ray is traced at max(horizontal speed, 300 u/s)
RAY_SPACING = 128.0     # the line's vertex spacing (the fan's)


class RayOperator:
    """The planner-free move operator (Codex, agent bus 2026-09-27 01:48): three LEVEL rays -
    forward / 45 deg left / 45 deg right of the horizontal velocity - drawn exactly as a
    --plan-shape ray planner draws them (goalprimplan.ray_curve over BUDGET_MULT x RAY_SECS,
    resampled at RAY_SPACING) and committed for RAY_SECS. No planner network, head or state:
    any executor that follows a line in its fan flies them, step 1's primitive follower (trained
    on uniform random primitives only) included."""

    n_choice = 3
    shape = "ray"

    def __init__(self, tick_ms: float, finish, n: int = 3):
        # n = 4 (--moves rays4, Codex 2026-09-27 04:00): the fourth move CONTINUES along the
        # current 3-D velocity (the all-zero velocity-frame step-1 primitive: no turn, no pitch
        # rate), so a dive or a ramp launch the physics created is kept instead of being flattened
        # onto a level line; no map-derived elevation, the centre of the mover's own vocabulary
        if int(n) not in (3, 4):
            raise ValueError("RayOperator: 3 level rays, or 4 with the 3-D continuation")
        self.n_choice = int(n)
        self.ray_deg = RAY_DEG
        self.secs, self.floor, self.spacing = RAY_SECS, RAY_FLOOR, RAY_SPACING
        self.commit_ticks = int(round(RAY_SECS * 1000.0 / float(tick_ms)))
        self.budget_ticks = self.commit_ticks
        self.choice_nums = np.zeros((self.n_choice, 1), np.float64)   # unused: a ray is its index
        self.finish = np.asarray(finish, np.float64)

    def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
        from surfgym.goalprimplan import BUDGET_MULT, RAY_SIGN, ray_curve
        from surfgym.route import resample_polyline
        if int(k) == 3:
            # the 3-D continuation: straight along the current velocity (its pitch included,
            # clipped at the step-1 +-85 deg), drawn over the same BUDGET_MULT x RAY_SECS
            from surfgym.goalprim import curve
            pts = curve(origin, velocity, yaw_deg, np.zeros(6), self.secs * BUDGET_MULT, 3,
                        self.floor, frame="velocity")
        else:
            pts = ray_curve(origin, velocity, yaw_deg, self.ray_deg * RAY_SIGN[int(k)],
                            self.secs, self.floor, BUDGET_MULT)
        line, _total = resample_polyline(pts, self.spacing)
        if len(line) < 2:
            line = np.vstack([pts[0], pts[-1]])
        return np.asarray(line, np.float32), pts

    def describe(self) -> str:
        return (f"move operator: 3 level rays (0 / +-{self.ray_deg:g} deg from the horizontal "
                f"velocity)" + (" + the 3-D continuation along the velocity"
                                if self.n_choice == 4 else "")
                + f", each committed {self.secs:g} s ({self.commit_ticks} ticks), traced "
                f"at max(speed, {self.floor:g} u/s), line spacing {self.spacing:g} u - no planner")


FRONTIER_TOP = 0.05     # --select-frontier: the top 5% of live nodes by goal-potential progress
SURF_LOOK = 1.0         # --moves surf: seconds of free flight searched for the surface of impact
SURF_DT = 0.05          # ... in 50 ms trace segments


def surf_curve(core, origin, velocity, secs: float, floor: float, gravity: float = 800.0):
    """The SURF LINE: the free-flight arc from ``origin`` (gravity only) until the simulator's own
    hull trace hits a surface within SURF_LOOK s, then straight along that surface's tangent (the
    impact velocity with its normal component removed) for the rest of ``secs``; no impact within
    SURF_LOOK: the free-flight arc for all of ``secs``. Traced at >= ``floor`` u/s. The physically
    natural continuation - land tangentially and ride - from the map's own collision geometry, no
    map constant. -> (T, 3) float64 points, one per 10 ms."""
    o = np.asarray(origin, np.float64).reshape(3)
    v = np.asarray(velocity, np.float64).reshape(3)
    g = np.array([0.0, 0.0, -float(gravity)])
    if np.linalg.norm(v) < float(floor):
        vh = np.array([v[0], v[1], 0.0])
        n_ = np.linalg.norm(vh)
        v = (vh / n_ * float(floor)) if n_ > 1e-6 else np.array([float(floor), 0.0, 0.0])
    pts = [o.copy()]
    t, p, vv, hit = 0.0, o.copy(), v.copy(), None
    while t < min(SURF_LOOK, secs) - 1e-9:
        q = p + vv * SURF_DT + 0.5 * g * SURF_DT ** 2
        tr = core.trace(p.tolist(), q.tolist(), 0)
        if tr.fraction < 1.0 and not tr.startsolid:
            hp = np.array(tr.endpos, np.float64)
            nrm = np.array(tr.normal, np.float64)
            vv = vv + g * SURF_DT * float(tr.fraction)
            hit = (hp, nrm, vv.copy())
            pts.append(hp)
            t += SURF_DT * float(tr.fraction)
            break
        vv = vv + g * SURF_DT
        p = q
        t += SURF_DT
        pts.append(p.copy())
    if hit is not None:
        hp, nrm, vi = hit
        u = vi - float(np.dot(vi, nrm)) * nrm
        spd = max(float(np.linalg.norm(u)), float(floor))
        un = u / max(float(np.linalg.norm(u)), 1e-6)
        rest = max(0.0, secs - t)
        k = int(np.ceil(rest / 0.01))
        for i in range(1, k + 1):
            pts.append(hp + un * spd * (0.01 * i) + nrm * 2.0)
    else:
        while t < secs - 1e-9:
            vv = vv + g * 0.01
            p = p + vv * 0.01
            t += 0.01
            pts.append(p.copy())
    return np.asarray(pts, np.float64)


def tangent_curve(core, origin, velocity, secs: float, floor: float, gravity: float = 800.0):
    """The TANGENT-APPROACH line (Codex 2026-09-27): the free-flight arc to the first surface the
    core's hull trace meets within SURF_LOOK s gives the contact point p, its normal n, the time t
    and the impact velocity v_c. The line is a cubic Hermite curve from ``origin`` (leaving along
    the current velocity) to p + 2 n that ARRIVES along u = v_c with its normal component removed -
    the surface's own plane, so the contact is tangential - then continues straight along u for
    the rest of ``secs`` at |v_c|. No contact within SURF_LOOK: the free-flight arc (surf_curve's
    fallback). Only the simulator's collision geometry; no map constant, no threshold."""
    o = np.asarray(origin, np.float64).reshape(3)
    v = np.asarray(velocity, np.float64).reshape(3)
    g = np.array([0.0, 0.0, -float(gravity)])
    if np.linalg.norm(v) < float(floor):
        vh = np.array([v[0], v[1], 0.0])
        n_ = np.linalg.norm(vh)
        v = (vh / n_ * float(floor)) if n_ > 1e-6 else np.array([float(floor), 0.0, 0.0])
    t, p, vv, hit = 0.0, o.copy(), v.copy(), None
    while t < min(SURF_LOOK, secs) - 1e-9:
        q = p + vv * SURF_DT + 0.5 * g * SURF_DT ** 2
        tr = core.trace(p.tolist(), q.tolist(), 0)
        if tr.fraction < 1.0 and not tr.startsolid:
            t += SURF_DT * float(tr.fraction)
            hit = (np.array(tr.endpos, np.float64), np.array(tr.normal, np.float64),
                   vv + g * SURF_DT * float(tr.fraction), t)
            break
        vv = vv + g * SURF_DT
        p = q
        t += SURF_DT
    if hit is None:
        return surf_curve(core, origin, velocity, secs, floor, gravity)
    pc, nrm, vc, tc = hit
    u = vc - float(np.dot(vc, nrm)) * nrm
    spd = max(float(np.linalg.norm(vc)), float(floor))
    un = u / max(float(np.linalg.norm(u)), 1e-6)
    end = pc + nrm * 2.0
    tc = max(tc, 0.05)
    # Hermite on [0, tc]: position o -> end, derivative v -> un * spd
    k = max(2, int(np.ceil(tc / 0.01)))
    ss = np.linspace(0.0, 1.0, k + 1)
    h00 = 2 * ss ** 3 - 3 * ss ** 2 + 1
    h10 = ss ** 3 - 2 * ss ** 2 + ss
    h01 = -2 * ss ** 3 + 3 * ss ** 2
    h11 = ss ** 3 - ss ** 2
    m0 = v * tc
    m1 = un * spd * tc
    pts = (h00[:, None] * o[None] + h10[:, None] * m0[None] + h01[:, None] * end[None]
           + h11[:, None] * m1[None])
    rest = max(0.0, secs - tc)
    kk = int(np.ceil(rest / 0.01))
    tail = [end + un * spd * (0.01 * i) for i in range(1, kk + 1)]
    return np.vstack([pts] + ([np.asarray(tail)] if tail else []))


class SurfOperator:
    """--moves prim_surf: K random step-1 primitives + ONE surf line (surf_curve) per expansion.
    The surf line needs the scratch core's trace, handed in by the Flyer (``core``)."""

    shape = "prim_surf"

    def __init__(self, tick_ms: float, finish, k: int, seed: int, core, cfg=None,
                 kind: str = "surf"):
        self.prim = PrimOperator(tick_ms, finish, k, seed, cfg=cfg)
        self.core = core
        self.kind = str(kind)           # surf (post-impact tangent) or tangent (approach)
        self.n_choice = int(k) + 1
        self.secs = self.prim.secs
        self.commit_ticks = self.budget_ticks = self.prim.commit_ticks
        self.choice_nums = np.zeros((self.n_choice, 1), np.float64)
        self.finish = self.prim.finish
        self.last = self.prim.last

    def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
        if int(k) < self.n_choice - 1:
            return self.prim.line_and_curve_of(origin, velocity, yaw_deg, nums, int(k))
        from surfgym.route import resample_polyline
        fn = tangent_curve if self.kind == "tangent" else surf_curve
        pts = fn(self.core, origin, velocity, self.secs * 1.5, RAY_FLOOR)
        line, _t = resample_polyline(pts, RAY_SPACING)
        if len(line) < 2:
            line = np.vstack([pts[0], pts[-1]])
        return np.asarray(line, np.float32), pts

    def describe(self) -> str:
        return (f"move operator (prim_surf): {self.n_choice - 1} random step-1 primitives + the "
                f"SURF LINE (free flight to the surface of impact within {SURF_LOOK:g} s, then "
                f"along its tangent), committed {self.secs:g} s ({self.commit_ticks} ticks) - "
                f"no planner")


RAMP_TIMEOUT = 6.0      # --moves ramp: a command's timeout (s) - one constant for every map
RAMP_TAIL = 1.0         # s of slide along the target surface appended to the line (lookahead)
RAMP_COAST = 6.0        # s of no-input coasting that orders a node's candidate surfaces
DEPART_TICKS = 10       # a command DEPARTS its source after this many consecutive contact-free
                        # physics ticks (surf contact flickers tick to tick; one free tick is not
                        # a departure) - one constant for every map


class RampOperator:
    """--moves ramp (the user's ramp graph, 2026-09-27; docs/ramp-search-design.md): a move is the
    COMMAND "go to surface B" (a ramp or floor of tools/ramps.py v2, or the finish box), flown by
    the existing line-following mover as a LINE - a cubic Hermite curve leaving along the current
    velocity and arriving in B's plane at the point of B closest to the node's own no-input coast
    (tangential arrival, as tangent_curve does for the first surface hit), then RAMP_TAIL s of
    slide along B's plane. The Flyer ends the flight at the first NEW surface contact after the
    departure from the source surface (or RAMP_TIMEOUT); a contact with C != B is an outcome of
    command B, never a success of command C (Codex 17:35Z).

    Candidate order per node (progressive widening - every command eventually tried, none
    deleted): the surfaces the node's own coast (RAMP_COAST s of neutral input in the scratch
    core: the physics' natural continuation, sliding on ramps included) touches, in touch order,
    then every other target by its closest approach to the coast, the finish box among them."""

    shape = "ramp"

    def __init__(self, tick_ms: float, finish, rampmap, k: int, timeout: float = RAMP_TIMEOUT,
                 gravity: float = 800.0):
        self.rm = rampmap
        self.gravity = float(gravity)
        self.tick_ms = float(tick_ms)
        self.targets = [int(s) for s in range(rampmap.n_surf) if int(rampmap.cat[s]) in (0, 1)]
        self.FIN = len(self.targets)                  # the finish box is the last command
        self.n_choice = len(self.targets) + 1
        self.k = int(k)
        self.secs = float(timeout)
        self.commit_ticks = self.budget_ticks = int(round(self.secs * 1000.0 / self.tick_ms))
        self.choice_nums = np.zeros((self.n_choice, 1), np.float64)
        self.finish = np.asarray(finish, np.float64)
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(0)
        self.tp, self.tn, self.tt = [], [], []
        for s in self.targets:
            m = rampmap.members[s]
            if len(m) > 400:
                m = np.sort(rng.choice(m, 400, replace=False))
            self.tp.append(rampmap.kp[m])
            self.tn.append(rampmap.kn[m])
            self.tt.append(cKDTree(rampmap.kp[m]))
        self.coast, self.rank, self.ptr = {}, {}, {}
        self.queue = []

    def plan(self, fl, arch, nids):
        """coast every not-yet-planned node in `nids` (batched in the scratch core, neutral input)
        and order its candidate commands"""
        ak = id(arch)
        todo = [p for p in dict.fromkeys(nids) if (ak, p) not in self.rank]
        core = fl.core
        n_t = int(round(RAMP_COAST * 1000.0 / self.tick_ms))
        neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (core.num_envs, 1))
        for c0 in range(0, len(todo), core.num_envs):
            part = todo[c0:c0 + core.num_envs]
            for i in range(core.num_envs):
                st = arch.state[part[min(i, len(part) - 1)]].copy()
                st["tick"] = 0
                st["stuck_ticks"] = 0
                core.set_state(i, st)
            pos = np.zeros((n_t + 1, len(part), 3))
            vel = np.zeros((n_t + 1, len(part), 3))
            alive = np.ones(len(part), bool)
            first = [[] for _ in part]
            sv = core.states_view
            pos[0] = sv["origin"][:len(part)]
            vel[0] = sv["velocity"][:len(part)]
            src = self.rm.contact(pos[0])
            last = np.zeros(len(part), np.int64)
            use_touch = hasattr(core, "get_touch")
            for t in range(n_t):
                # the coast is simulation too: charged to the same budget (Codex 19:14Z)
                fl.live_ticks += int(alive.sum())
                _o, _r, done, trunc, _ = core.step(neutral)
                sv = core.states_view
                ended = (np.asarray(done, bool) | np.asarray(trunc, bool))[:len(part)]
                alive &= ~ended
                pos[t + 1] = np.where(alive[:, None], sv["origin"][:len(part)], pos[t])
                vel[t + 1] = np.where(alive[:, None], sv["velocity"][:len(part)], vel[t])
                last[alive] = t + 1
                if use_touch:
                    cnt, tn_, tp_ = core.get_touch()
                    sets = self.rm.touch_sets(cnt[:len(part)], tn_[:len(part)],
                                              tp_[:len(part)])
                    for j in np.flatnonzero(alive):
                        for s_ in sorted(sets[j]):
                            if s_ >= 0 and s_ != int(src[j]) and s_ not in first[j]:
                                first[j].append(int(s_))
                elif t % 5 == 4:
                    c = self.rm.contact(pos[t + 1], targetable=True)
                    for j in np.flatnonzero(alive & (c >= 0) & (c != src)):
                        if int(c[j]) not in first[j]:
                            first[j].append(int(c[j]))
                if not alive.any():
                    break
            for j, p in enumerate(part):
                # only the SIMULATED part of the coast: no unsimulated tail at the world origin
                path = pos[:last[j] + 1, j][::5]
                self.coast[(ak, p)] = (path, vel[:last[j] + 1, j][::5], src[j])
                touched = [self.targets.index(s) for s in first[j] if s in self.targets]
                dmin = [float(self.tt[ti].query(path, k=1)[0].min())
                        for ti in range(len(self.targets))]
                dmin.append(float(np.linalg.norm(path - self.finish[None], axis=1).min()))
                order = touched + [int(i) for i in np.argsort(dmin) if int(i) not in touched]
                if src[j] >= 0 and src[j] in self.targets:
                    si = self.targets.index(int(src[j]))
                    order = [i for i in order if i != si]
                self.rank[(ak, p)] = order
                self.ptr[(ak, p)] = 0

    def next_jobs(self, arch, p):
        """the node's next K commands (cycling through its whole order: progressive widening)"""
        key = (id(arch), p)
        order = self.rank[key]
        out = []
        for _ in range(min(self.k, len(order))):
            out.append(order[self.ptr[key] % len(order)])
            self.ptr[key] += 1
        return out

    def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
        from surfgym.route import resample_polyline
        key = self.queue.pop(0) if self.queue else None
        o = np.asarray(origin, np.float64).reshape(3)
        v = np.asarray(velocity, np.float64).reshape(3)
        if key is not None and key in self.coast:
            path, pvel, _src = self.coast[key]
        else:                          # no coast (a replay from an unplanned state): the straight arc
            ts = np.arange(0.0, RAMP_COAST, 0.05)
            path = o[None] + v[None] * ts[:, None] + 0.5 * np.array([0, 0, -self.gravity]) * ts[:, None] ** 2
            pvel = v[None] + np.array([0, 0, -self.gravity]) * ts[:, None]
        dt_path = 0.05
        if int(k) == self.FIN:
            j = int(np.argmin(np.linalg.norm(path - self.finish[None], axis=1)))
            pb, nb = self.finish, None
        else:
            tp, tn = self.tp[int(k)], self.tn[int(k)]
            dq, iq = self.tt[int(k)].query(path, k=1)
            j = int(np.argmin(dq))
            pb, nb = tp[int(iq[j])], tn[int(iq[j])]
        # the arrival time: the coast's own time to the closest approach, but never faster than
        # the straight distance at max(current speed, RAY_FLOOR) - from a slow or standing node
        # the coast barely moves, and a 0.2 s curve across the map is not a flyable line
        tc = max(0.2, j * dt_path,
                 float(np.linalg.norm(pb - o)) / max(float(np.linalg.norm(v)), RAY_FLOOR))
        vc = pvel[min(j, len(pvel) - 1)]
        spd = max(float(np.linalg.norm(vc)), RAY_FLOOR)
        if nb is not None:
            u = vc - float(vc @ nb) * nb
            if np.linalg.norm(u) < 1e-3:
                u = (pb - o) - float((pb - o) @ nb) * nb
            end = pb + nb * 2.0
        else:
            u = pb - o
            end = pb
        un = u / max(float(np.linalg.norm(u)), 1e-6)
        vv = v if np.linalg.norm(v) >= 1.0 else un * RAY_FLOOR
        ss = np.linspace(0.0, 1.0, max(2, int(np.ceil(tc / 0.01))) + 1)
        h00 = 2 * ss ** 3 - 3 * ss ** 2 + 1
        h10 = ss ** 3 - 2 * ss ** 2 + ss
        h01 = -2 * ss ** 3 + 3 * ss ** 2
        h11 = ss ** 3 - ss ** 2
        pts = (h00[:, None] * o[None] + h10[:, None] * (vv * tc)[None] + h01[:, None] * end[None]
               + h11[:, None] * (un * spd * tc)[None])
        tail, p, w = [], end.copy(), un * spd
        g = np.array([0.0, 0.0, -self.gravity])
        gt = g - (float(g @ nb) * nb if nb is not None else 0.0)
        for _ in range(int(RAMP_TAIL / 0.01)):
            w = w + gt * 0.01
            p = p + w * 0.01
            tail.append(p.copy())
        pts = np.vstack([pts, np.asarray(tail)])
        line, _t = resample_polyline(pts, RAY_SPACING)
        if len(line) < 2:
            line = np.vstack([pts[0], pts[-1]])
        return np.asarray(line, np.float32), pts

    def describe(self) -> str:
        return (f"move operator (ramp): 'go to surface B' commands over {len(self.targets)} "
                f"surfaces ({int(sum(1 for s in self.targets if self.rm.cat[s] == 1))} ramps, "
                f"{int(sum(1 for s in self.targets if self.rm.cat[s] == 0))} floors) + the finish, "
                f"{self.k} per expansion in the node's coast order (progressive widening), a "
                f"Hermite arrival into B's plane + {RAMP_TAIL:g} s slide; a flight ends at the "
                f"first new contact after departure or {self.secs:g} s - no planner")


class MixOperator:
    """--moves mix: a SUPERSET of the level rays - moves 0-2 are RayOperator's three level rays
    (the basis that finds the edgeflow routes in seconds), moves 3 .. 2+K are K random step-1
    primitives (PrimOperator: the 3-D vocabulary that enters unitfarmer2's pit, which the rays
    and the 3-D continuation do not)."""

    shape = "mix"

    def __init__(self, tick_ms: float, finish, k: int, seed: int, cfg=None):
        self.rays = RayOperator(tick_ms, finish, n=3)
        self.prim = PrimOperator(tick_ms, finish, k, seed, cfg=cfg)
        if self.rays.commit_ticks != self.prim.commit_ticks:
            raise ValueError("--moves mix: the rays and the primitives must commit alike")
        self.n_choice = 3 + int(k)
        self.commit_ticks = self.budget_ticks = self.rays.commit_ticks
        self.choice_nums = np.zeros((self.n_choice, 1), np.float64)
        self.finish = self.rays.finish
        self.last = self.prim.last

    def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
        if int(k) < 3:
            return self.rays.line_and_curve_of(origin, velocity, yaw_deg, nums, int(k))
        return self.prim.line_and_curve_of(origin, velocity, yaw_deg, nums, int(k))

    def describe(self) -> str:
        return (f"move operator (mix): {self.rays.describe().split(': ', 1)[1].split(', each')[0]}"
                f" + {self.n_choice - 3} random step-1 primitives, each committed "
                f"{self.rays.secs:g} s ({self.commit_ticks} ticks) - no planner")


class PrimOperator:
    """--moves prim: K RANDOM primitives per expansion, drawn uniformly from step 1's own
    distribution (goalprim.PrimitivePlanner with its defaults: 3 sideways knots in [-180, 180]
    deg/s, 3 vertical in [-120, 90] deg/s, a 2 s curve along the 3D velocity at max(speed,
    300 u/s), 128 u line spacing) and committed for the curve's 2 s. Each (node, slot) flight
    gets a FRESH draw; the numbers are kept per flight (``last``) so a chain can be written down.
    No planner, no learned choice: the mover's own training vocabulary, climbs and dives
    included."""

    shape = "prim"

    def __init__(self, tick_ms: float, finish, k: int, seed: int, cfg=None):
        from surfgym.goalprim import PRIM_DEFAULTS, PrimitivePlanner
        # the mover's OWN primitive distribution from its checkpoint config (Codex: not the
        # hardcoded defaults), including --prim-turn / --prim-frame
        cfg = cfg or {}
        d = {k_: (cfg.get(k_) if cfg.get(k_) is not None else v_) for k_, v_ in PRIM_DEFAULTS.items()}
        self.prim = PrimitivePlanner(secs=float(d["prim_secs"]), knots=int(d["prim_knots"]),
                                     side=float(d["prim_side"]), down=float(d["prim_down"]),
                                     up=float(d["prim_up"]), floor=float(d["prim_floor"]),
                                     spacing=RAY_SPACING,
                                     flat=bool(cfg.get("prim_flat") or 0),
                                     frame=str(cfg.get("prim_frame") or "velocity"),
                                     pitch_max=float(cfg.get("prim_pitch_max") or 85.0),
                                     turn=str(cfg.get("prim_turn") or "rate"))
        self.n_choice = int(k)
        self.secs = float(d["prim_secs"])
        self.commit_ticks = int(round(self.secs * 1000.0 / float(tick_ms)))
        self.budget_ticks = self.commit_ticks
        self.choice_nums = np.zeros((self.n_choice, 1), np.float64)   # unused: a slot index
        self.finish = np.asarray(finish, np.float64)
        self.rng = np.random.default_rng(int(seed) + 911)
        self.last = []

    def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
        # draw(): step 1's own sampler, with its loop rejection (a curve whose end sphere could be
        # entered long before the curve is done is redrawn), not the raw uniform sample()
        p, _line, _end, _total = self.prim.draw(origin, velocity, yaw_deg, self.rng)
        self.last.append(np.asarray(p, np.float64).tolist())
        return self.prim.line_and_curve(origin, velocity, yaw_deg, p)

    def describe(self) -> str:
        return (f"move operator: {self.n_choice} RANDOM step-1 primitives per expansion (sideways "
                f"+-{self.prim.side:g} deg/s, vertical -{self.prim.down:g}..+{self.prim.up:g} "
                f"deg/s at {self.prim.knots} knots, {self.secs:g} s along the 3D velocity at "
                f"max(speed, {self.prim.floor:g} u/s)), committed {self.secs:g} s "
                f"({self.commit_ticks} ticks) - no planner")


# --speed-bins sqrt2: the horizontal-speed edges at a sqrt(2) ratio instead of 2 (log-uniform either
# way, the same on every map): a faster state in the same cell is a NEW key one bin sooner
HSPD_EDGES_SQRT2 = np.array([128.0, 181.0, 256.0, 362.0, 512.0, 724.0, 1024.0, 1448.0, 2048.0,
                             2896.0])
SPEED_EDGES = {"log2": HSPD_EDGES, "sqrt2": HSPD_EDGES_SQRT2}
_EDGES = HSPD_EDGES


def keys_of(st, mins) -> list:
    """The archive key of each STATE_DTYPE row (Codex's v1 quantiser; --speed-bins picks the
    horizontal-speed edges)."""
    o = np.asarray(st["origin"], np.float64)
    v = np.asarray(st["velocity"], np.float64)
    cell = np.floor((o - mins[None, :]) / POS_CELL).astype(np.int64)
    vh = np.hypot(v[:, 0], v[:, 1])
    sb = np.searchsorted(_EDGES, vh, side="right")
    az = np.where(vh >= HSPD_EDGES[0],
                  np.floor((np.arctan2(v[:, 1], v[:, 0]) + math.pi) / (2.0 * math.pi) * N_AZ)
                  .astype(np.int64) % N_AZ, N_AZ)
    vz = np.where(v[:, 2] > VZ_DEAD, 2, np.where(v[:, 2] < -VZ_DEAD, 0, 1))
    g = (np.asarray(st["onground"]) != -1).astype(np.int64)
    return [(int(cell[i, 0]), int(cell[i, 1]), int(cell[i, 2]), int(sb[i]), int(az[i]),
             int(vz[i]), int(g[i])) for i in range(len(o))]


class Archive:
    """Nodes (parallel lists) + the key index (key -> [first-arrival id, fastest id])."""

    def __init__(self):
        self.state, self.keys_state, self.obs, self.key = [], [], [], []
        self.parent, self.move, self.depth, self.t, self.speed = [], [], [], [], []
        self.path = []
        self.n_sel = []
        self.n_fly = []             # flights launched FROM this node (its expansions)
        self.n_die = []             # ... of which died
        self.index = {}
        self.cells = set()
        self.admit_new = self.admit_dup = self.replaced = 0

    def __len__(self):
        return len(self.state)

    def add(self, st, ks, obs, key, parent, move, depth, t, path):
        nid = len(self.state)
        self.state.append(st.copy())
        self.keys_state.append(ks)
        self.obs.append(np.array(obs, np.float32, copy=True))
        self.key.append(key)
        self.parent.append(int(parent))
        self.move.append(int(move))
        self.depth.append(int(depth))
        self.t.append(int(t))
        self.speed.append(float(np.linalg.norm(np.asarray(st["velocity"], np.float64))))
        self.path.append(path)
        self.n_sel.append(0)
        self.n_fly.append(0)
        self.n_die.append(0)
        return nid

    def admit(self, st, ks, obs, key, parent, move, depth, t, path) -> str:
        """-> 'new' | 'replaced' | 'dup'."""
        slot = self.index.get(key)
        sp = float(np.linalg.norm(np.asarray(st["velocity"], np.float64)))
        if slot is None:
            nid = self.add(st, ks, obs, key, parent, move, depth, t, path)
            self.index[key] = [nid, nid]
            self.cells.add(key[:3])
            self.admit_new += 1
            return "new"
        first, fast = slot
        out = "dup"
        if t < self.t[first]:
            slot[0] = self.add(st, ks, obs, key, parent, move, depth, t, path)
            out = "replaced"
        if sp > self.speed[fast]:
            slot[1] = self.add(st, ks, obs, key, parent, move, depth, t, path)
            out = "replaced"
        if out == "replaced":
            self.replaced += 1
        else:
            self.admit_dup += 1
        return out

    def live_ids(self):
        """The nodes the index currently holds (an elite that was replaced is retired)."""
        return sorted({i for s in self.index.values() for i in s})

    greedy = False              # --greedy: expand each node once
    frontier_frac = 0.0         # --select-frontier
    progress_of = None          # node id -> goal-potential progress (set with --select-frontier)

    key_first = False           # --select-keys: a KEY by count weight, then one of its elites

    def select_keys(self, n, rng):
        """Key-first (Codex 2026-09-27): each key's weight is 1 / sqrt(1 + its selections), so a
        key with separate first / fastest elites is not drawn twice as often; the elite within
        the key is drawn uniformly."""
        keys = list(self.index.keys())
        if not hasattr(self, "_ksel"):
            self._ksel = {}
        w = 1.0 / np.sqrt(1.0 + np.asarray([self._ksel.get(k, 0) for k in keys], np.float64))
        pick = rng.choice(len(keys), size=n, replace=len(keys) < n, p=w / w.sum())
        out = []
        for j in pick:
            k = keys[int(j)]
            self._ksel[k] = self._ksel.get(k, 0) + 1
            first, fast = self.index[k]
            i = first if (first == fast or rng.random() < 0.5) else fast
            self.n_sel[i] += 1
            out.append(int(i))
        return out

    cell_first = False          # --select-cells: a POSITION cell by count weight, then a key

    def select_cells(self, n, rng):
        """Cell-first (Go-Explore's own cell selection): a 128 u position cell by 1 / sqrt(1 + its
        selections), then one of its keys uniformly, then one of that key's elites. Dense regions
        (thousands of keys in few cells) stop swamping spatially rare ones."""
        if not hasattr(self, "_csel"):
            self._csel = {}
        cells = {}
        for k in self.index:
            cells.setdefault(k[:3], []).append(k)
        cl = list(cells.keys())
        w = 1.0 / np.sqrt(1.0 + np.asarray([self._csel.get(c, 0) for c in cl], np.float64))
        pick = rng.choice(len(cl), size=n, replace=True, p=w / w.sum())
        out = []
        for j in pick:
            c = cl[int(j)]
            self._csel[c] = self._csel.get(c, 0) + 1
            ks = cells[c]
            k = ks[int(rng.integers(0, len(ks)))]
            first, fast = self.index[k]
            i = first if (first == fast or rng.random() < 0.5) else fast
            self.n_sel[i] += 1
            out.append(int(i))
        return out

    def select(self, n, rng):
        if self.cell_first and not self.greedy and self.frontier_frac <= 0.0:
            return self.select_cells(n, rng)
        if self.key_first and not self.greedy and self.frontier_frac <= 0.0:
            return self.select_keys(n, rng)
        ids = np.asarray(self.live_ids(), np.int64)
        if self.greedy:
            ids = np.asarray([i for i in ids if self.n_sel[i] == 0], np.int64)
            if not len(ids):
                return []
            pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
            for i in pick:
                self.n_sel[int(i)] += 1
            return [int(i) for i in pick]
        w = 1.0 / np.sqrt(1.0 + np.asarray([self.n_sel[i] for i in ids], np.float64))
        nf = 0
        if self.frontier_frac > 0.0 and self.progress_of is not None and len(ids) > 20:
            # --select-frontier F: F of the parents drawn, with the same count weights, from the
            # top FRONTIER_TOP of live nodes by the map's goal-potential progress (Go-Explore's
            # score-aware cell selection, with the map's own potential as the score); the rest
            # stays count-only, so a dip the potential hides is still explored
            prog = np.asarray([self.progress_of(i) for i in ids], np.float64)
            top = ids[prog >= np.quantile(prog, 1.0 - FRONTIER_TOP)]
            nf = int(round(n * self.frontier_frac))
            wt = 1.0 / np.sqrt(1.0 + np.asarray([self.n_sel[i] for i in top], np.float64))
            fpick = rng.choice(top, size=nf, replace=len(top) < nf, p=wt / wt.sum())
        else:
            fpick = np.zeros(0, np.int64)
        pick = rng.choice(ids, size=min(n - nf, len(ids)), replace=len(ids) < n - nf,
                          p=w / w.sum())
        pick = np.concatenate([fpick, pick])
        for i in pick:
            self.n_sel[int(i)] += 1
        return [int(i) for i in pick]

    def chain(self, nid):
        out = []
        while nid >= 0:
            out.append(nid)
            nid = self.parent[nid]
        return out[::-1]


class Flyer:
    """Flies batches of (node, choice) on the recorder's scratch core with the checkpoint's own
    executor wrapper, the way goalsearch.PrimMCTS._expand does, but one DIFFERENT node per slot
    group (every node is captured at an executor decision tick, so all slots share the phase)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.P = ctx.planner
        self.sc = ctx.scratch
        self.core = self.sc.core
        self.S = int(self.core.num_envs)
        self.C = int(getattr(self.P, "n_choice", 0))
        if self.C <= 0:
            raise SystemExit("edge_archive: the checkpoint's planner has no --plan-choices (v1 "
                             "flies the fixed choices)")
        pol = self.sc.make_policy(self.core, self.sc.line)
        self.K = max(1, int(getattr(pol, "_k", 1)))
        self.keys_hold = bool(getattr(pol, "keys_hold", False))
        from surfgym.goalprimplan import L_MAX
        self.L_MAX = L_MAX
        self.dur = int(getattr(self.P, "commit_ticks", 0) or self.P.budget_ticks)
        self.steps = 0
        self.live_ticks = 0         # ticks of REAL (not padding, still open) flights: the budget
        self.ramp_map = None        # --moves ramp: stop a flight at its first new surface contact

    def fresh_keys(self):
        """The held-keys state of a fresh episode (what the wrapper starts from)."""
        if not self.keys_hold:
            return None
        from surfgym.keyshold import KeysHold
        k = KeysHold(1)
        return (k.state[0].copy(), k.boot[0].copy())

    def fly(self, jobs, arch, finish, record_path=True, keep_clock=False):
        """``jobs``: [(node id, choice)] (<= S). -> per job dict(end, obs, keys, died, fin,
        ticks, path). ``keep_clock``: the restored states keep their episode clock and stall
        counter (tools/contingent_archive.py - the real deadline; a flight reaching the core's
        episode cap is a failure); off = the discovery archive's zeroed clock."""
        core, P, S, K = self.core, self.P, self.S, self.K
        n = len(jobs)
        assert 0 < n <= S
        if getattr(P, "shape", "") == "ramp" and not P.queue:
            # any caller (replay, edge fidelity, the analysis) flies the SAME ramp-command
            # family: coast-plan the job parents in this archive and queue them per slot
            P.plan(self, arch, [j[0] for j in jobs])
            P.queue = [(id(arch), jobs[min(i, n - 1)][0]) for i in range(S)]
        st_all = np.empty(S, dtype=arch.state[jobs[0][0]].dtype)
        for i in range(S):
            nid = jobs[min(i, n - 1)][0]
            st = arch.state[nid].copy()
            if not keep_clock:
                st["tick"] = 0
                st["stuck_ticks"] = 0
            st_all[i] = st
            core.set_state(i, st)
        o = st_all["origin"].astype(np.float64)
        v = st_all["velocity"].astype(np.float64)
        y = st_all["yaw"].astype(np.float64)
        lines = []
        for i in range(S):
            k = jobs[min(i, n - 1)][1]
            ln, _pts = P.line_and_curve_of(o[i], v[i], float(y[i]), P.choice_nums[int(k)],
                                           int(k))
            lines.append(ln[:self.L_MAX])
        self.sc.line.set_lines(np.arange(S), lines)
        pol = self.sc.make_policy(core, self.sc.line)
        if self.keys_hold:
            from surfgym.keyshold import KeysHold
            pol.keys = KeysHold(S)
            for i in range(S):
                ks = arch.keys_state[jobs[min(i, n - 1)][0]]
                if ks is not None:
                    pol.keys.state[i] = ks[0]
                    pol.keys.boot[i] = ks[1]
            # the previous decision was one period before the restored clock, so the wrapper's
            # episode-start detector does not collapse the held keys
            pol._keys_tick = (st_all["tick"].astype(np.int64)
                              - int(getattr(pol, "_period", K)))
        pol._tick = 0
        obs = np.ascontiguousarray(np.stack([arch.obs[jobs[min(i, n - 1)][0]]
                                             for i in range(S)]).astype(np.float32))
        open_ = np.zeros(S, bool)
        open_[:n] = True
        died = np.zeros(S, bool)
        fnd = np.zeros(S, bool)
        ticks = np.zeros(S, np.int64)
        end = [None] * n
        end_obs = [None] * n
        end_keys = [None] * n
        # --mid-states: the first decision-aligned state at or past half the plan, of every flight
        # still open there (the archive admits it as a child too: a finer grain for the search)
        mid, mid_obs, mid_keys, mid_ticks = [None] * n, [None] * n, [None] * n, [0] * n
        mid_path = [None] * n
        mid_done = not getattr(self, "mid_states", False)
        # --pre-death L: a ring of every open flight's decision-tick snapshots, so a flight that
        # DIES can hand back its state L ticks before the death (the last states before a failure
        # - the hard transitions the endpoints never contain; Codex 2026-09-27)
        pre_l = int(getattr(self, "pre_death", 0) or 0)
        ring = [[] for _ in range(n)] if pre_l > 0 else None
        pre = [None] * n
        paths = [[o[i].copy()] for i in range(n)] if record_path else None
        term = [None] * n
        dt = float(self.ctx.tick.ms) / 1000.0
        rmap = self.ramp_map
        if rmap is not None:
            # the ramp-command contract (Codex 17:35Z): contact with the SOURCE surface is ignored
            # until the flight departs it; the first new contact after that ends the command
            # the SOURCE: the surface the node is touching when the command starts (proximity to
            # the extracted contact planes, any category); the flight must leave it first
            src = rmap.contact(o[:n])
            hit = np.full(n, -1, np.int64)
            hit_tick = np.full(n, -1, np.int64)
            hit_set = [None] * n
            use_touch = hasattr(core, "get_touch")
            # with telemetry the SOURCE is the node's last contact (the set that ended the command
            # that created it) + the proximity source + every surface touched before the
            # departure; departure = DEPART_TICKS consecutive contact-free ticks (proximity alone
            # is the fallback without telemetry)
            last_c = getattr(arch, "last_contact", {})
            src_set = [set(last_c.get(jobs[i][0], ())) | ({int(src[i])} if src[i] >= 0 else set())
                       for i in range(n)]
            departed = (np.array([len(x) == 0 for x in src_set]) if use_touch else (src < 0))
            free_run = np.zeros(n, np.int64)
        for t in range(self.dur + K):
            self.live_ticks += int(open_[:n].sum())
            acts = pol.act(obs)
            view = getattr(pol, "view", None)
            # the pre-step position and velocity: the core autoresets an ended row inside the
            # step, so this is the last state of an episode that ends on this tick
            sv = core.states_view
            pre_o = sv["origin"].astype(np.float64)
            pre_v = sv["velocity"].astype(np.float64)
            obs, _r, done, trunc, _term = (core.step(acts) if view is None
                                           else core.step(acts, view=view))
            self.steps += S
            done = np.asarray(done, bool)
            ended = open_ & (done | np.asarray(trunc, bool))
            if ended.any() and ring is not None:
                for i in np.flatnonzero(ended[:n] & ~(done[:n] & np.asarray(core.goal_hits,
                                                                            bool)[:n])):
                    best = None
                    for (tk, sti, obi, kyi) in ring[i]:
                        if tk <= t + 1 - pre_l:
                            best = (tk, sti, obi, kyi)
                    if best is not None:
                        pre[i] = best
            if ended.any():
                hits = np.asarray(core.goal_hits, bool)
                fin_now = ended & done & hits
                fnd |= fin_now
                died |= ended & ~(done & hits)
                ticks[ended] = t + 1
                # terminal-complete path (Codex 2026-09-27): every ended flight gets its last
                # pre-step position; a FINISHING one also the tick's end, pre-step position +
                # velocity x tick - the swept segment that crossed the goal box, so a route built
                # from the path reaches into the box instead of stopping up to 10 ticks short
                for i in np.flatnonzero(ended[:n]):
                    if paths is not None:
                        paths[i].append(pre_o[i].copy())
                    if fin_now[i]:
                        term[i] = pre_o[i] + pre_v[i] * dt
                        if paths is not None:
                            paths[i].append(term[i].copy())
                open_ &= ~ended
            if rmap is not None and open_[:n].any():
                live = open_[:n] & (hit_tick < 0)
                if use_touch:
                    # COLLISION TRUTH (Codex 17:17Z): the planes the movement actually hit this
                    # tick; after the departure from the source, the first tick with any contact
                    # ends the command (walls included - strict) and its whole touched SET is kept
                    cnt, tn_, tp_ = core.get_touch()
                    sets = rmap.touch_sets(cnt[:n], tn_[:n], tp_[:n])
                    for i in np.flatnonzero(live):
                        ts_ = sets[i]
                        if not departed[i]:
                            if ts_:
                                src_set[i] |= ts_   # still in the source's contact phase
                                free_run[i] = 0
                                continue
                            free_run[i] += 1
                            if free_run[i] >= DEPART_TICKS:
                                departed[i] = True  # left the source for real
                            continue
                        if ts_:
                            hit_set[i] = sorted(ts_)
                            hit[i] = next(iter(ts_)) if len(ts_) == 1 else -5   # -5: a SET
                            hit_tick[i] = t + 1
                else:
                    c = rmap.contact(core.states_view["origin"][:n].astype(np.float64),
                                     targetable=True)
                    departed |= live & ~departed & (c != src)
                    h = live & departed & (c >= 0)
                    hit[h] = c[h]
                    hit_tick[h] = t + 1
                if int(pol._tick) % K == 0:
                    # the command ends at the first decision boundary at or after the contact
                    stop = open_[:n] & (hit_tick >= 0)
                    if stop.any():
                        cur_s = core.get_states()
                        for i in np.flatnonzero(stop):
                            end[i] = cur_s[i].copy()
                            end_obs[i] = np.array(obs[i], np.float32, copy=True)
                            end_keys[i] = ((pol.keys.state[i].copy(), pol.keys.boot[i].copy())
                                           if self.keys_hold and pol.keys is not None else None)
                            ticks[i] = t + 1
                            if paths is not None:
                                paths[i].append(cur_s[i]["origin"].astype(np.float64).copy())
                        open_[:n] &= ~stop
            if ring is not None and int(pol._tick) % K == 0 and open_[:n].any():
                cur_r = core.get_states()
                keep = pre_l // K + 3
                for i in np.flatnonzero(open_[:n]):
                    ring[i].append((t + 1, cur_r[i].copy(), np.array(obs[i], np.float32, copy=True),
                                    ((pol.keys.state[i].copy(), pol.keys.boot[i].copy())
                                     if self.keys_hold and pol.keys is not None else None)))
                    if len(ring[i]) > keep:
                        del ring[i][0]
            if paths is not None and t % 10 == 9:
                cur_o = core.states_view["origin"]
                for i in np.flatnonzero(open_[:n]):
                    paths[i].append(cur_o[i].astype(np.float64).copy())
            if (not mid_done and t + 1 >= self.dur // 2 and int(pol._tick) % K == 0
                    and t + 1 < self.dur):
                cur_m = core.get_states()
                for i in np.flatnonzero(open_[:n]):
                    mid[i] = cur_m[i].copy()
                    mid_obs[i] = np.array(obs[i], np.float32, copy=True)
                    mid_keys[i] = ((pol.keys.state[i].copy(), pol.keys.boot[i].copy())
                                   if self.keys_hold and pol.keys is not None else None)
                    mid_ticks[i] = t + 1
                    if paths is not None:
                        mid_path[i] = np.round(np.asarray(paths[i] + [cur_m[i]["origin"]
                                                                      .astype(np.float64)]),
                                               1).tolist()
                mid_done = True
            if t + 1 >= self.dur and int(pol._tick) % K == 0 and open_[:n].any():
                cur = core.get_states()
                for i in np.flatnonzero(open_[:n]):
                    end[i] = cur[i].copy()
                    end_obs[i] = np.array(obs[i], np.float32, copy=True)
                    end_keys[i] = ((pol.keys.state[i].copy(), pol.keys.boot[i].copy())
                                   if self.keys_hold and pol.keys is not None else None)
                    ticks[i] = t + 1
                open_[:n] = False
            if not open_[:n].any():
                break
        out = []
        for i in range(n):
            out.append({"pre": (None if pre[i] is None else pre[i][1]),
                        "pre_obs": (None if pre[i] is None else pre[i][2]),
                        "pre_keys": (None if pre[i] is None else pre[i][3]),
                        "pre_ticks": (0 if pre[i] is None else int(pre[i][0])),
                        "mid": mid[i], "mid_obs": mid_obs[i], "mid_keys": mid_keys[i],
                        "mid_ticks": int(mid_ticks[i]), "mid_path": mid_path[i],
                        "end": end[i], "obs": end_obs[i], "keys": end_keys[i],
                        "died": bool(died[i]), "fin": bool(fnd[i]), "ticks": int(ticks[i]),
                        "terminal": (None if term[i] is None else
                                     np.round(term[i], 1).tolist()),
                        "src": (int(src[i]) if rmap is not None else None),
                        "src_set": (sorted(src_set[i]) if rmap is not None else None),
                        "hit": (int(hit[i]) if rmap is not None else None),
                        "hit_set": (hit_set[i] if rmap is not None else None),
                        "hit_tick": (int(hit_tick[i]) if rmap is not None else None),
                        "path": (None if paths is None else
                                 np.round(np.asarray(paths[i]), 1).tolist())})
        return out


def _load_rampmap(path):
    from ramps import RampMap
    return RampMap(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=None, help="another map than the checkpoint's own")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--expansions", type=int, default=0, help="stop after this many (0 = none)")
    ap.add_argument("--parents", type=int, default=64, help="nodes expanded per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-secs", type=float, default=30.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop at the first finish (count finishing chains)")
    ap.add_argument("--analyze", type=int, default=0,
                    help="after the first chain: every edge re-flown 32 x from its exact parent - "
                         "the child-key hit rate, the distinct child keys, deaths, the position / "
                         "speed error to the stored child - and for the first N edges the share of "
                         "(up to 16) re-flown children from which a fresh search (400 expansions) "
                         "still finds a finishing chain (Codex's fidelity questions)")
    ap.add_argument("--rays", type=int, default=0,
                    help="3 = the planner-free move operator (RayOperator): the three level rays "
                         "(0 / +-45 deg), committed 2 s, flown by ANY executor checkpoint - a "
                         "primlearn one without --plan-choices, or step 1's --goal-planner prim "
                         "follower; the checkpoint's planner (if any) is not used")
    ap.add_argument("--select-cells", action="store_true",
                    help="cell-first parent selection: a 128 u position cell by 1 / sqrt(1 + its "
                         "selections), then a key in it, then an elite (Go-Explore's cells)")
    ap.add_argument("--select-keys", action="store_true",
                    help="key-first parent selection: a key by 1 / sqrt(1 + its selections), then "
                         "one of its (first / fastest) elites uniformly")
    ap.add_argument("--moves", choices=("rays", "rays4", "prim", "mix", "prim_surf", "widen",
                                         "prim_tangent", "ramp"),
                    default="rays",
                    help="with --rays 3 / for any executor: rays = the three level rays "
                         "(RayOperator); prim = --n-moves RANDOM step-1 primitives per expansion "
                         "(PrimOperator: the mover's own training distribution, climbs and dives "
                         "included)")
    ap.add_argument("--n-moves", type=int, default=3,
                    help="--moves prim: primitives drawn per expanded node")
    ap.add_argument("--ramps", default=None,
                    help="--moves ramp: the map's surfaces (tools/ramps.py v2 .npz)")
    ap.add_argument("--ramp-k", type=int, default=4,
                    help="--moves ramp: commands per expansion (the node's next K in its coast "
                         "order; progressive widening)")
    ap.add_argument("--max-live-ticks", type=int, default=0,
                    help="stop once this many ticks of real flights were simulated (0 = off): the "
                         "equal simulated budget across move operators")
    ap.add_argument("--exec-temp", type=float, default=None,
                    help="the mover samples at this temperature on every head (record_ckpt "
                         "--exec-temp: the trainer's TemperedTorchPolicy); default = native")
    ap.add_argument("--select-frontier", type=float, default=0.0,
                    help="F in [0, 1): F of every batch's parents come from the top 5%% of live "
                         "nodes by the map's goal-potential progress (the baked goal field; "
                         "Go-Explore's score-aware selection), the rest count-only")
    ap.add_argument("--pre-death", type=float, default=0.0,
                    help="SECONDS: a flight that dies also admits its decision-tick state this "
                         "long before the death (0 = off) - the states just before a failure, "
                         "which no surviving endpoint contains")
    ap.add_argument("--mid-states", action="store_true",
                    help="every flight's midpoint (the first decision tick at or past half the "
                         "plan) is admitted to the archive as a child as well: a 1 s grain for the "
                         "search at no extra flight")
    ap.add_argument("--speed-bins", choices=("log2", "sqrt2"), default="log2",
                    help="the key's horizontal-speed edges: log2 = 128..2048 doubling (v1), "
                         "sqrt2 = 128..2896 at a sqrt(2) ratio")
    ap.add_argument("--dump-states", action="store_true",
                    help="write <out>/archive_states.npy at the end: every live node's full "
                         "STATE_DTYPE row, clocks zeroed - the agent's own states, a "
                         "SELF_STATES spawn source for train_fast --spawn-states")
    ap.add_argument("--dump-weights", choices=("none", "goid"), default="none",
                    help="with --dump-states: goid = a node's row is repeated 1 + round(3 w / "
                         "max w) times, w = p (1 - p) with p its flights' death rate (>= 3 "
                         "flights; fewer: the median w) - practice concentrated where the "
                         "mover's outcome is uncertain (Florensa et al. 2018's goals of "
                         "intermediate difficulty), with the file still a plain state array")
    ap.add_argument("--dump-nodes", action="store_true",
                    help="write <out>/nodes.npz at the end: every live node's origin, velocity, "
                         "depth, ticks from the root and times selected (where the archive "
                         "reached and where it stalls - a measurement, never a reward)")
    ap.add_argument("--cold-policy", type=int, default=None,
                    help="SEED: the mover is the checkpoint's architecture at INITIALISATION "
                         "(record_ckpt --cold-policy: the trainer's step-0 draw, no weights "
                         "loaded) - discovery with no trained component; it ignores the rays, so "
                         "the three moves are three sampled 2 s rollouts")
    ap.add_argument("--greedy", action="store_true",
                    help="the executor acts GREEDILY (deterministic with the simulator): every "
                         "(node, move) has ONE outcome, so each node is expanded once "
                         "(breadth-first over the key graph) and a found chain replays exactly "
                         "from the true start")
    ap.add_argument("--closed-loop", type=int, default=0,
                    help="N > 0: the archive AS THE PLANNER - N episodes from the map start; at "
                         "every decision a fresh archive search from the TRUE current state, the "
                         "first move of a finishing chain (else of the chain to the deepest node) "
                         "committed and flown once for real, then search again")
    ap.add_argument("--decision-exp", type=int, default=6000,
                    help="--closed-loop: the expansion budget of one decision's search")
    ap.add_argument("--decision-secs", type=float, default=20.0,
                    help="--closed-loop: the wall-clock budget of one decision's search")
    ap.add_argument("--cap-secs", type=float, default=60.0,
                    help="--closed-loop: an episode's game-time cap")
    a = ap.parse_args(argv)
    global _EDGES
    _EDGES = SPEED_EDGES[a.speed_bins]
    import record_ckpt
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(a.seed))
    if a.moves == "prim" and int(a.rays) != 3:
        raise SystemExit("--moves prim is a planner-free operator: pass --rays 3 as well "
                         "(the checkpoint's planner is not used)")
    if a.moves == "ramp" and not a.ramps:
        raise SystemExit("--moves ramp needs --ramps <tools/ramps.py v2 .npz>")
    probe = (({"prim": int(a.n_moves), "rays4": 4, "mix": 3 + int(a.n_moves),
               "prim_surf": int(a.n_moves) + 1, "prim_tangent": int(a.n_moves) + 1,
               "widen": 3, "ramp": int(a.ramp_k)}.get(a.moves, 3))
             if int(a.rays) == 3 else _ckpt_choices(a.ckpt))
    S = int(a.parents) * probe
    rargv = [str(a.ckpt), "--episodes", "1", "--plan-scratch", str(S)]
    if not a.greedy:
        rargv.append("--stochastic")
    if a.map:
        rargv += ["--map", str(a.map)]
    if a.moves == "ramp":
        # a command lasts up to RAMP_TIMEOUT: the scratch core's episode cap must exceed it, or a
        # long command is TRUNCATED and the Flyer counts that as a death
        rargv += ["--ep-ticks", str(int(round(RAMP_TIMEOUT * 100)) + 400)]
    if a.cold_policy is not None:
        rargv += ["--cold-policy", str(int(a.cold_policy))]
    if a.exec_temp is not None:
        if a.greedy:
            raise SystemExit("--exec-temp with --greedy: the greedy mover has no temperature")
        rargv += ["--exec-temp", str(float(a.exec_temp))]
    ctx = record_ckpt.build(rargv, device=a.device)
    if getattr(ctx, "scratch", None) is None:
        raise SystemExit("edge_archive: the recorder built no scratch core")
    if int(a.rays) == 3:
        # the planner-free move operator: the checkpoint's planner, if it has one, is not used -
        # only its executor, following the rays the operator draws
        fb = getattr(ctx, "finish_box", None)
        if fb is None or fb[0] is None:
            raise SystemExit("edge_archive: the recorder returned no finish box")
        _fc = 0.5 * (np.asarray(fb[0], np.float64) + np.asarray(fb[1], np.float64))
        ctx.planner = (PrimOperator(float(ctx.tick.ms), _fc, int(a.n_moves), int(a.seed),
                                    cfg=getattr(ctx, "cfg", None))
                       if a.moves == "prim" else
                       SurfOperator(float(ctx.tick.ms), _fc, int(a.n_moves), int(a.seed),
                                    ctx.scratch.core, cfg=getattr(ctx, "cfg", None),
                                    kind=("tangent" if a.moves == "prim_tangent" else "surf"))
                       if a.moves in ("prim_surf", "prim_tangent") else
                       MixOperator(float(ctx.tick.ms), _fc, int(a.n_moves), int(a.seed),
                                   cfg=getattr(ctx, "cfg", None))
                       if a.moves == "mix" else
                       # --moves widen: the three rays + ONE primitive slot (index 3)
                       MixOperator(float(ctx.tick.ms), _fc, 1, int(a.seed),
                                   cfg=getattr(ctx, "cfg", None))
                       if a.moves == "widen" else
                       RampOperator(float(ctx.tick.ms), _fc, _load_rampmap(a.ramps), int(a.ramp_k))
                       if a.moves == "ramp" else
                       RayOperator(float(ctx.tick.ms), _fc, n=(4 if a.moves == "rays4" else 3)))
        print("edge_archive: --rays 3 - " + ctx.planner.describe(), flush=True)
    elif ctx.planner is None:
        raise SystemExit("edge_archive: the recorder built no primitive planner (pass --rays 3 "
                         "for the planner-free operator)")
    fl = Flyer(ctx)
    if a.moves == "ramp":
        fl.ramp_map = ctx.planner.rm
    ramp_stats = {"direct": 0, "wrong": 0, "none": 0, "died": 0}
    ramp_info = {}              # node -> the command that created it (target, src, hit, set, tick)
    ramp_edges = set()          # witnessed (source surface, first new contact) pairs
    fl.mid_states = bool(a.mid_states)
    fl.pre_death = int(round(float(a.pre_death) * 1000.0 / float(ctx.tick.ms)))
    P = ctx.planner
    fin = np.asarray(P.finish, np.float64)
    mins = np.asarray(ctx.core.map_bounds()[0], np.float64)
    # the root: the map start, reset like a recording's first episode
    core1 = ctx.core
    core1.set_spawn_pool(np.asarray(ctx.pool)[0:1])
    obs0 = core1.reset(int(a.seed))
    st0 = core1.get_states()[0].copy()
    Archive.greedy = bool(a.greedy)
    arch = Archive()
    arch.key_first = bool(a.select_keys)
    arch.cell_first = bool(a.select_cells)
    if float(a.select_frontier) > 0.0:
        # the map's own goal potential (the trainer's baked geodesic field) as the frontier score
        from surfgym.goalfield import load_goal_field
        _bsp = Path(ctx.map_path)
        _cands = sorted(_bsp.parent.glob(f"{_bsp.stem}.goal_*.npz"))
        _cands = [c for c in _cands if c.stem.split(".goal_")[-1].isdigit()]
        if not _cands:
            raise SystemExit(f"--select-frontier: no baked goal field next to {_bsp}")
        _gf = load_goal_field(str(_cands[0]))
        _d0 = float(_gf.sample(np.asarray([st0["origin"]], np.float64))[0])
        _pcache = {}

        def _prog(i, _a=arch):
            if i not in _pcache:
                d = float(_gf.sample(np.asarray([_a.state[i]["origin"]], np.float64))[0])
                _pcache[i] = (_d0 - d) if np.isfinite(d) else -1e9
            return _pcache[i]
        arch.frontier_frac = float(a.select_frontier)
        arch.progress_of = _prog
        print(f"edge_archive: --select-frontier {a.select_frontier:g} - the frontier score is "
              f"{_cands[0].name}'s progress (d0 {_d0:,.0f} u)", flush=True)
    root = arch.add(st0, fl.fresh_keys(), np.asarray(obs0)[0], keys_of(st0[None], mins)[0], -1,
                    -1, 0, 0, [np.round(st0["origin"].astype(np.float64), 1).tolist()])
    arch.index[arch.key[root]] = [root, root]
    arch.cells.add(arch.key[root][:3])
    d0 = float(np.linalg.norm(st0["origin"].astype(np.float64) - fin))
    print(f"edge_archive: {a.ckpt} on {Path(ctx.map_path).name}: root at "
          f"{np.round(st0['origin'], 0).tolist()}, {d0:,.0f} u from the finish; {fl.C} choices x "
          f"{a.parents} parents per batch = {S} envs; plan {fl.dur} ticks + decision alignment; "
          f"executor SAMPLES (native temperature); count-only selection 1/sqrt(1+n)", flush=True)
    if int(a.closed_loop) > 0:
        return closed_loop(a, ctx, fl, fin, mins, out, rng)
    prog_f = open(out / "progress.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    t_rep = t0
    expansions = 0
    yield_new = np.zeros(fl.C, np.int64)
    tried = np.zeros(fl.C, np.int64)
    deaths = fins = 0
    best_d = d0
    best_node = root
    # MEASUREMENT only (never read by selection): the route progress on the map's baked geodesic
    # goal field, if one sits next to the map - on a winding map the straight-line distance above
    # says nothing about how far along the route a node is
    from surfgym.goalfield import load_goal_field as _lgf
    _gcands = [c for c in sorted(Path(ctx.map_path).parent.glob(f"{Path(ctx.map_path).stem}.goal_*.npz"))
               if c.stem.split(".goal_")[-1].isdigit()]
    geo = {"gf": _lgf(str(_gcands[0])) if _gcands else None, "d": {}}
    if geo["gf"] is not None:
        geo["d0"] = float(geo["gf"].sample(np.asarray([st0["origin"]], np.float64))[0])
    finishers = []
    terminal = {}           # finish node -> the flight's terminal position (the goal crossing)
    move_nums = {}          # --moves prim: node -> the primitive numbers of the move that made it

    def report(final=False):
        ids = arch.live_ids()
        dmin = min(float(np.linalg.norm(arch.state[i]["origin"].astype(np.float64) - fin))
                   for i in ids)
        rec = {"secs": round(time.time() - t0, 1), "expansions": expansions,
               "nodes": len(arch), "keys": len(arch.index), "cells": len(arch.cells),
               "new": arch.admit_new, "dup": arch.admit_dup, "replaced": arch.replaced,
               "deaths": deaths, "finishes": fins, "best_dist": round(dmin, 1),
               "best_progress": round(1.0 - dmin / d0, 4),
               "max_depth": max(arch.depth[i] for i in ids),
               "yield": [round(float(yield_new[k]) / max(1, int(tried[k])), 4)
                         for k in range(fl.C)],
               "sim_steps_per_s": round(fl.steps / max(1e-9, time.time() - t0)),
               "live_ticks": int(fl.live_ticks),
               "final": bool(final)}
        if a.moves == "ramp":
            rec["ramp_outcomes"] = dict(ramp_stats)
            rec["ramp_edges"] = len(ramp_edges)
        if geo["gf"] is not None:
            new_ids = [i for i in ids if i not in geo["d"]]
            if new_ids:
                dd = geo["gf"].sample(np.asarray([arch.state[i]["origin"] for i in new_ids],
                                                 np.float64))
                geo["d"].update(zip(new_ids, (float(x) for x in dd)))
            dg = [(geo["d"][i], i) for i in ids if np.isfinite(geo["d"][i])]
            gmin, gnode = min(dg) if dg else (geo["d0"], root)
            rec["geo_best_progress"] = round((geo["d0"] - gmin) / geo["d0"], 4)
            rec["geo_best_node"] = int(gnode)
        prog_f.write(json.dumps(rec) + "\n")
        prog_f.flush()
        print(f"[{rec['secs']:7.1f}s] exp {expansions:,} nodes {rec['nodes']:,} keys "
              f"{rec['keys']:,} cells {rec['cells']:,} | new {arch.admit_new:,} dup "
              f"{arch.admit_dup:,} repl {arch.replaced:,} | deaths {deaths:,} fin {fins} | "
              f"best {rec['best_progress']:.1%} ({rec['best_dist']:,.0f} u) depth "
              f"{rec['max_depth']} | " + (f"yield F/L/R {rec['yield']}" if fl.C <= 8 else
                                           f"ramp outcomes {rec.get('ramp_outcomes')} edges "
                                           f"{rec.get('ramp_edges')}")
              + (f" | ROUTE {rec['geo_best_progress']:.2%}" if "geo_best_progress" in rec else "")
              + f" | live ticks {fl.live_ticks:,} | {rec['sim_steps_per_s']:,} steps/s",
              flush=True)
        return rec

    while True:
        parents = arch.select(int(a.parents), rng)
        if not parents:
            print("edge_archive: every node expanded - the (greedy) archive is exhausted",
                  flush=True)
            break
        if a.moves == "widen":
            # PROGRESSIVE WIDENING (Codex 2026-09-27): a node's FIRST selection flies the three
            # level rays; every later selection flies ONE fresh step-1 primitive (slot 3). The
            # batch is filled to the scratch core's width with further selections.
            jobs = []
            if not hasattr(arch, "n_exp"):
                arch.n_exp = {}
            while True:
                for p in parents:
                    # the node's first expansion (in the whole run, duplicates within a batch
                    # included) flies the rays, every later one a primitive
                    if arch.n_exp.get(p, 0) == 0:
                        jobs += [(p, 0), (p, 1), (p, 2)]
                    else:
                        jobs.append((p, 3))
                    arch.n_exp[p] = arch.n_exp.get(p, 0) + 1
                if len(jobs) >= fl.S:
                    break
                parents = arch.select(max(1, (fl.S - len(jobs)) // 3), rng)
                if not parents:
                    break
            jobs = jobs[:fl.S]
            parents = sorted({p for (p, _k) in jobs})
        elif a.moves == "ramp":
            # the ramp commands: each selected node's NEXT K commands in its coast order
            ctx.planner.plan(fl, arch, parents)
            jobs = [(p, t) for p in parents for t in ctx.planner.next_jobs(arch, p)][:fl.S]
            parents = sorted({p for (p, _t) in jobs})
            ctx.planner.queue = ([(id(arch), p) for (p, _t) in jobs]
                                 + [(id(arch), jobs[-1][0])] * (fl.S - len(jobs)))
        else:
            jobs = [(p, k) for p in parents for k in range(fl.C)]
        if isinstance(ctx.planner, (PrimOperator, MixOperator, SurfOperator)):
            ctx.planner.last.clear()
        res = fl.fly(jobs, arch, fin)
        drawn = None                 # --moves mix: a ray slot draws nothing, so no per-slot alignment
        if isinstance(ctx.planner, PrimOperator):
            drawn = list(ctx.planner.last)
        expansions += len(parents)
        for j, ((p, k), r) in enumerate(zip(jobs, res)):
            if drawn is not None and j < len(drawn):
                r["nums"] = drawn[j]
            if a.moves == "ramp":
                tgt = (ctx.planner.targets[k] if k < ctx.planner.FIN else "finish")
                if r["died"] and not r["fin"]:
                    ramp_stats["died"] += 1
                elif r.get("hit_tick") is not None and r["hit_tick"] >= 0:
                    # direct = the first contact is B ALONE; another surface, a simultaneous set
                    # (-5) or an unextracted piece (-4) is an outcome of command B, not a success
                    ramp_stats["direct" if r["hit"] == tgt else "wrong"] += 1
                    ramp_edges.add((int(r["src"]), tuple(r["hit_set"] or [int(r["hit"])])))
                else:
                    ramp_stats["none"] += 1
            tried[k] += 1
            arch.n_fly[p] += 1
            arch.n_die[p] += int(bool(r["died"]))
            if r["fin"]:
                fins += 1
                nid = arch.add(arch.state[p], arch.keys_state[p], arch.obs[p], ("FIN",), p, k,
                               arch.depth[p] + 1, arch.t[p] + r["ticks"], r["path"])
                terminal[nid] = r.get("terminal")
                if r.get("nums") is not None:
                    move_nums[nid] = r["nums"]
                finishers.append(nid)
                continue
            if r.get("mid") is not None:
                # --mid-states: the flight's midpoint, a child of the same parent by the same move
                mkey = keys_of(r["mid"][None], mins)[0]
                if arch.admit(r["mid"], r["mid_keys"], r["mid_obs"], mkey, p, k,
                              arch.depth[p] + 1, arch.t[p] + r["mid_ticks"],
                              r.get("mid_path")) == "new":
                    yield_new[k] += 1
            if r.get("pre") is not None:
                # --pre-death: the dead flight's state before the failure, a child of the parent
                pkey = keys_of(r["pre"][None], mins)[0]
                if arch.admit(r["pre"], r["pre_keys"], r["pre_obs"], pkey, p, k,
                              arch.depth[p] + 1, arch.t[p] + r["pre_ticks"], None) == "new":
                    yield_new[k] += 1
            if r["died"] or r["end"] is None:
                deaths += 1
                continue
            key = keys_of(r["end"][None], mins)[0]
            n_before = len(arch)
            what = arch.admit(r["end"], r["keys"], r["obs"], key, p, k, arch.depth[p] + 1,
                              arch.t[p] + r["ticks"], r["path"])
            if a.moves == "ramp":
                if not hasattr(arch, "last_contact"):
                    arch.last_contact = {}
                for _nid in range(n_before, len(arch)):
                    if r.get("hit_set"):
                        arch.last_contact[_nid] = set(r["hit_set"])
                    ramp_info[_nid] = {"target": (ctx.planner.targets[k] if k < ctx.planner.FIN
                                                  else "finish"),
                                       "src": r.get("src"), "hit": r.get("hit"),
                                       "hit_set": r.get("hit_set"), "hit_tick": r.get("hit_tick")}
            if r.get("nums") is not None:
                for _nid in range(n_before, len(arch)):
                    move_nums[_nid] = r["nums"]
            if what == "new":
                yield_new[k] += 1
                d = float(np.linalg.norm(r["end"]["origin"].astype(np.float64) - fin))
                if d < best_d:
                    best_d, best_node = d, len(arch) - 1
        now = time.time()
        if now - t_rep >= float(a.report_secs):
            report()
            t_rep = now
        if finishers and not a.keep_going:
            break
        if a.expansions and expansions >= int(a.expansions):
            break
        if a.max_live_ticks and fl.live_ticks >= int(a.max_live_ticks):
            break
        if now - t0 >= 60.0 * float(a.minutes):
            break
    rec = report(final=True)
    prog_f.close()
    if a.dump_states:
        _ids = arch.live_ids()
        _st = np.stack([arch.state[i] for i in _ids]).copy()
        _st["tick"] = 0
        _st["stuck_ticks"] = 0
        _rep = None
        if a.dump_weights == "goid":
            nf = np.asarray([arch.n_fly[i] for i in _ids], np.float64)
            nd = np.asarray([arch.n_die[i] for i in _ids], np.float64)
            pd = (nd + 0.5) / (nf + 1.0)
            w = pd * (1.0 - pd)
            seen = nf >= 3
            if seen.any():
                w[~seen] = float(np.median(w[seen]))
            _rep = 1 + np.rint(3.0 * w / max(float(w.max()), 1e-9)).astype(np.int64)
            _st = np.repeat(_st, _rep)
        np.save(out / "archive_states.npy", _st)
        print(f"edge_archive: {len(_ids):,} live node states -> {out / 'archive_states.npy'}"
              + (f" (goid-weighted: {len(_st):,} rows, repeats 1-{int(_rep.max())}, "
                 f"{int((_rep > 1).sum()):,} nodes repeated)" if _rep is not None else ""),
              flush=True)
    if a.dump_nodes:
        ids = arch.live_ids()
        _paths = [np.asarray(arch.path[i] or [arch.state[i]["origin"].tolist()],
                             np.float32).reshape(-1, 3) for i in range(len(arch))]
        _path_pts = (np.concatenate(_paths) if _paths else np.zeros((0, 3), np.float32))
        _path_off = np.concatenate([[0], np.cumsum([len(pp) for pp in _paths])]).astype(np.int64)
        np.savez_compressed(out / "nodes.npz",
                            origin=np.stack([arch.state[i]["origin"] for i in ids]).astype(np.float32),
                            velocity=np.stack([arch.state[i]["velocity"] for i in ids])
                            .astype(np.float32),
                            depth=np.asarray([arch.depth[i] for i in ids], np.int32),
                            node_id=np.asarray(ids, np.int64),
                            parent=np.asarray([arch.parent[i] for i in ids], np.int64),
                            # every node's origin by id (a retired elite can be a live node's
                            # ancestor), so a chain to ANY live node can be read back
                            all_origin=np.stack([arch.state[i]["origin"] for i in
                                                 range(len(arch))]).astype(np.float32),
                            all_parent=np.asarray(arch.parent, np.int64),
                            # every node's flown path from its parent (ragged: points + offsets), so
                            # a route to ANY node can be rebuilt afterwards (tools/archive_route.py)
                            all_path_pts=_path_pts, all_path_off=_path_off,
                            t=np.asarray([arch.t[i] for i in ids], np.int32),
                            n_sel=np.asarray([arch.n_sel[i] for i in ids], np.int32),
                            root=np.asarray(st0["origin"], np.float32), finish=fin.astype(np.float32))
        print(f"edge_archive: {len(ids):,} live nodes -> {out / 'nodes.npz'}", flush=True)
    summary = dict(rec, ckpt=str(a.ckpt), map=Path(ctx.map_path).name, d0=d0,
                   choices=fl.C, parents=int(a.parents), plan_ticks=fl.dur,
                   finishers=len(finishers), seed=int(a.seed),
                   operator=(a.moves if int(a.rays) == 3 else "planner"),
                   flights=int(sum(int(x) for x in tried)),
                   cold_policy=a.cold_policy, speed_bins=a.speed_bins,
                   mid_states=bool(a.mid_states), pre_death=float(a.pre_death),
                   terminal_complete=True, live_ticks=int(fl.live_ticks),
                   ramps=a.ramps, ramp_k=(int(a.ramp_k) if a.moves == "ramp" else None))
    if finishers:
        nid = min(finishers, key=lambda i: arch.t[i])
        ch = arch.chain(nid)
        moves = [arch.move[i] for i in ch[1:]]
        chain = {"ticks_from_root": arch.t[nid], "secs": arch.t[nid] * float(ctx.tick.ms) / 1000.0,
                 "moves": moves,
                 "nodes": [{"origin": np.round(arch.state[i]["origin"].astype(np.float64), 1)
                            .tolist(), "t": arch.t[i], "move": arch.move[i],
                            "nums": move_nums.get(i), "path": arch.path[i],
                            # --moves ramp: the command that made this node, and what it touched
                            "ramp": ramp_info.get(i)} for i in ch],
                 # the goal crossing (the last node's own state is its parent's: the core
                 # autoresets a finished row, so only the position survives)
                 "terminal": terminal.get(nid),
                 "operator": (ctx.planner.describe()
                              if isinstance(ctx.planner, (RayOperator, RampOperator))
                              else "the checkpoint's --plan-choices planner lines")}
        (out / "chain.json").write_text(json.dumps(chain), encoding="utf-8")
        print(f"edge_archive: FIRST FINISHING CHAIN after {expansions:,} expansions "
              f"({rec['secs']:.0f} s): {len(moves)} plans, {chain['secs']:.1f} s from the root, "
              f"moves {moves}", flush=True)
        if isinstance(ctx.planner, (PrimOperator, MixOperator, SurfOperator)):
            rep = None
            print("edge_archive: --moves prim - no open-loop replay (a slot index does not name "
                  "a move)", flush=True)
        else:
            rep = replay(fl, arch, root, moves, fin)
            print(f"edge_archive: the chain's plan sequence replayed from the true start, no "
                  f"restores, executor sampling: {rep['finished']}/{rep['n']} finish; median "
                  f"plans completed {rep['median_plans']}", flush=True)
        summary["replay"] = rep
        # the chain's exact states, root first, clocks zeroed: a SELF_STATES spine for the
        # trainer's --demo-file (the agent's own search states - no human input)
        sp = np.stack([arch.state[i] for i in ch]).copy()
        sp["tick"] = 0
        sp["stuck_ticks"] = 0
        np.save(out / "chain_states.npy", sp)
        fid = ([] if isinstance(ctx.planner, (PrimOperator, MixOperator, SurfOperator))
               else edge_fidelity(fl, arch, ch, fin))
        summary["edge_fidelity"] = fid
        chain["edge_fidelity"] = fid
        (out / "chain.json").write_text(json.dumps(chain), encoding="utf-8")
        if int(a.analyze) > 0:
            an = analyze_edges(fl, arch, ch, fin, mins, rng, int(a.analyze), int(a.parents))
            summary["analysis"] = an
            (out / "edge_analysis.json").write_text(json.dumps(an, indent=1), encoding="utf-8")
            for e in an:
                print(f"  edge {e['edge']} move {e['move']}: child-key hit {e['key_hit']}/{e['n']}, "
                      f"{e['distinct_keys']} distinct keys, {e['deaths']} deaths, pos err median "
                      f"{e['pos_err_med']} u, speed err median {e['spd_err_med']} u/s"
                      + (f", downstream-solvable {e['solvable']}/{e['probed']}"
                         if e.get('probed') else ""), flush=True)
        pr = (float(np.prod([max(e["survived"], 0) / e["n"] for e in fid])) if fid
              else float("nan"))
        summary["fidelity_product"] = pr
        print(f"edge_archive: product of the edges' survival rates {pr:.4f}", flush=True)
        print("edge_archive: each chain edge re-flown 32 x from its EXACT parent state "
              "(survived / finished): "
              + " ".join(f"{e['move']}:{e['survived']}/{e['finished']}" for e in fid), flush=True)
    else:
        print(f"edge_archive: no finish after {expansions:,} expansions; best "
              f"{rec['best_progress']:.1%} of the start distance (node depth "
              f"{arch.depth[best_node]})", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


ROOT_SAMPLES = 32      # --closed-loop: flights per root move, for its survival rate
VIAB_SAMPLES = (4, 2)  # --closed-loop: flights per move at each lookahead level below the root


def search(fl, st, ks, obs, fin, mins, rng, parents, max_exp, max_secs):
    """One decision's search from an exact state, ROBUST to the executor's sampling: each root move
    is flown ROOT_SAMPLES times (its survival rate), the survivors seed an archive whose nodes
    remember the root move they descend from, and the archive search runs until the budget or
    until the most reliable move has a finishing chain behind it. The committed move: the most
    reliable one with a finishing chain; else the most reliable one. -> dict(found, move, surv,
    found_by_move, expansions, secs, nodes)."""
    C = fl.C
    t0 = time.time()
    arch = Archive()
    root = arch.add(st, ks, obs, ("ROOT",), -1, -1, 0, 0, None)
    rmove = {root: -1}
    n_per = max(1, min(ROOT_SAMPLES, fl.S // C))
    jobs = [(root, k) for k in range(C) for _ in range(n_per)]
    res = fl.fly(jobs, arch, fin, record_path=False)
    surv = np.zeros(C)
    fin_by = np.zeros(C, bool)
    for (q, k), r in zip(jobs, res):
        if r["fin"]:
            surv[k] += 1
            fin_by[k] = True
        elif not r["died"] and r["end"] is not None:
            surv[k] += 1
            key = keys_of(r["end"][None], mins)[0] + (k,)     # the root move is part of the key
            n0 = len(arch)
            arch.admit(r["end"], r["keys"], r["obs"], key, q, k, 1, r["ticks"], None)
            for i in range(n0, len(arch)):
                rmove[i] = k
    surv /= n_per
    # k-step viability (VIAB_SAMPLES per level): a surviving end state can still be DOOMED (in
    # the air over the void, every next plan dies). Level 1 = the root moves' samples above; each
    # survivor at level L < depth is probed with every move VIAB_SAMPLES[L] times; a state at the
    # deepest level is viable when it survived; a state above is viable when SOME move keeps at
    # least half of its samples viable. value[k] = viable level-1 samples of move k / its flights
    surv_ids = {k: [] for k in range(C)}
    for (q, k), r in zip(jobs, res):
        if not r["fin"] and not r["died"] and r["end"] is not None:
            surv_ids[k].append(arch.add(r["end"], r["keys"], r["obs"], ("SV",), q, k, 1,
                                        r["ticks"], None))
    value = np.array([float(fin_by[k]) for k in range(C)])  # a finish in the first plan: 1
    kids = {}               # node -> {move: [child ids or 'FIN' or None (dead)]}
    level = [nid for k in range(C) for nid in surv_ids[k]]
    probe = []
    for L, ns in enumerate(VIAB_SAMPLES):
        jobs2 = [(nid, k2) for nid in level for k2 in range(C) for _ in range(ns)]
        probe += jobs2
        nxt = []
        for c0 in range(0, len(jobs2), fl.S):
            chunk = jobs2[c0:c0 + fl.S]
            for (nid, k2), r in zip(chunk, fl.fly(chunk, arch, fin, record_path=False)):
                if r["fin"]:
                    kids.setdefault(nid, {}).setdefault(k2, []).append("FIN")
                elif r["died"] or r["end"] is None:
                    kids.setdefault(nid, {}).setdefault(k2, []).append(None)
                else:
                    cid = arch.add(r["end"], r["keys"], r["obs"], ("SV",), nid, k2, 0, 0, None)
                    arch.n_sel[cid] = 10 ** 9
                    kids.setdefault(nid, {}).setdefault(k2, []).append(cid)
                    nxt.append(cid)
        level = nxt
    memo = {}

    def viable(nid):
        if nid == "FIN":
            return True
        if nid is None:
            return False
        if nid not in kids:
            return True                     # the deepest level: it survived
        if nid in memo:
            return memo[nid]
        v = any(np.mean([viable(c) for c in cs]) >= 0.5 for cs in kids[nid].values())
        memo[nid] = v
        return v
    for k in range(C):
        if fin_by[k]:
            continue
        value[k] = sum(1 for nid in surv_ids[k] if viable(nid)) / n_per
    exp_probe = len(probe)
    arch.n_sel[root] = 10 ** 9                               # never expanded again
    for k in range(C):
        for nid in surv_ids[k]:
            arch.n_sel[nid] = 10 ** 9                       # probes, not archive nodes
    # the probes are not archive nodes: the finishing-chain search starts from the root's own
    # survivors (keyed with their root move) admitted above
    exp = C * n_per + exp_probe // C
    best_k = int(np.argmax(value))
    while (exp < max_exp and time.time() - t0 < max_secs and not fin_by[best_k]
           and len(arch.live_ids()) > 1):
        par = [q for q in arch.select(int(parents), rng) if q != root]
        if not par:
            break
        jobs = [(q, k) for q in par for k in range(C)]
        res = fl.fly(jobs, arch, fin, record_path=False)
        exp += len(par)
        for (q, k), r in zip(jobs, res):
            m = rmove.get(q, -1)
            if r["fin"]:
                if m >= 0:
                    fin_by[m] = True
            elif not r["died"] and r["end"] is not None:
                key = keys_of(r["end"][None], mins)[0] + (m,)
                n0 = len(arch)
                arch.admit(r["end"], r["keys"], r["obs"], key, q, k, arch.depth[q] + 1,
                           arch.t[q] + r["ticks"], None)
                for i in range(n0, len(arch)):
                    rmove[i] = m
    if fin_by.any():
        cand = np.flatnonzero(fin_by)
        move = int(cand[np.argmax(value[cand])])
    else:
        move = best_k
    return {"found": bool(fin_by[move]), "move": move,
            "surv": [round(float(v), 3) for v in surv],
            "value": [round(float(v), 3) for v in value],
            "found_by_move": [bool(v) for v in fin_by], "expansions": int(exp),
            "secs": round(time.time() - t0, 2), "depth": 0, "nodes": len(arch)}


def closed_loop(a, ctx, fl, fin, mins, out, rng):
    """--closed-loop N: the archive as the planner, from the map start, re-searched at every
    decision from the true state; each committed move is flown ONCE (the executor sampling, as in
    the searches) and the next search starts where that flight really ended."""
    pool = np.asarray(ctx.pool)
    core1 = ctx.core
    tick_s = float(ctx.tick.ms) / 1000.0
    eps = []
    for e in range(int(a.closed_loop)):
        j = e % max(1, len(pool))
        core1.set_spawn_pool(pool[j:j + 1])
        obs0 = core1.reset(int(a.seed) * 1000 + e)
        st = core1.get_states()[0].copy()
        ks, ob = fl.fresh_keys(), np.asarray(obs0)[0]
        d0 = float(np.linalg.norm(st["origin"].astype(np.float64) - fin))
        t = 0
        dec = []
        path = [np.round(st["origin"].astype(np.float64), 1).tolist()]
        end = "cap"
        while t * tick_s < float(a.cap_secs):
            sr = search(fl, st, ks, ob, fin, mins, rng, a.parents, int(a.decision_exp),
                        float(a.decision_secs))
            real = Archive()
            nid = real.add(st, ks, ob, ("REAL",), -1, -1, 0, 0, None)
            r = fl.fly([(nid, sr["move"])], real, fin)[0]
            dec.append({"t": round(t * tick_s, 2),
                        "pos": np.round(st["origin"].astype(np.float64), 0).tolist(),
                        "found": sr["found"], "move": sr["move"], "surv": sr["surv"],
                        "value": sr["value"],
                        "found_by_move": sr["found_by_move"],
                        "expansions": sr["expansions"], "search_secs": sr["secs"]})
            if r["path"]:
                path += r["path"][1:]
            t += r["ticks"]
            if r["fin"]:
                end = "finish"
                break
            if r["died"] or r["end"] is None:
                end = "died"
                break
            st, ks, ob = r["end"], r["keys"], r["obs"]
        rec = {"episode": e, "end": end, "secs": round(t * tick_s, 2), "decisions": len(dec),
               "d0": round(d0, 1), "found_share": round(float(np.mean([d["found"] for d in dec]))
                                                        if dec else 0.0, 3),
               "decisions_log": dec, "path": path}
        eps.append(rec)
        print(f"closed-loop episode {e}: {end.upper()} at {rec['secs']:.1f} s after {len(dec)} "
              f"decisions (a finishing chain in view at {rec['found_share']:.0%} of them; "
              f"search {np.mean([d['search_secs'] for d in dec]) if dec else 0:.1f} s / "
              f"{np.mean([d['expansions'] for d in dec]) if dec else 0:,.0f} expansions per "
              f"decision)", flush=True)
    n_fin = sum(1 for r in eps if r["end"] == "finish")
    ts = [r["secs"] for r in eps if r["end"] == "finish"]
    summ = {"episodes": len(eps), "finished": n_fin,
            "finish_secs": ts, "median_finish_secs": (float(np.median(ts)) if ts else None),
            "ckpt": str(a.ckpt), "map": Path(ctx.map_path).name,
            "decision_exp": int(a.decision_exp), "decision_secs": float(a.decision_secs),
            "parents": int(a.parents), "episodes_log": eps}
    (out / "closed_loop.json").write_text(json.dumps(summ), encoding="utf-8")
    print(f"closed-loop: {n_fin}/{len(eps)} episodes finished from the map start"
          + (f" (median {np.median(ts):.1f} s)" if ts else ""), flush=True)
    return 0


def replay(fl, arch, root, moves, fin):
    """The chain's plan sequence from the true start, REPLAYS parallel copies, no restores between
    plans: each copy flies plan i from wherever plan i-1 left it."""
    n = min(REPLAYS, fl.S)
    rep = Archive()
    r0 = rep.add(arch.state[root], arch.keys_state[root], arch.obs[root], ("R",), -1, -1, 0, 0,
                 None)
    cur = [r0] * n
    alive = np.ones(n, bool)
    done_plans = np.zeros(n, np.int64)
    finished = np.zeros(n, bool)
    for m in moves:
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        res = fl.fly([(cur[i], m) for i in idx], rep, fin, record_path=False)
        for i, r in zip(idx, res):
            if r["fin"]:
                finished[i] = True
                alive[i] = False
                done_plans[i] += 1
            elif r["died"] or r["end"] is None:
                alive[i] = False
            else:
                cur[i] = rep.add(r["end"], r["keys"], r["obs"], ("R",), cur[i], m, 0, 0, None)
                done_plans[i] += 1
    return {"n": int(n), "finished": int(finished.sum()),
            "median_plans": float(np.median(done_plans)), "plans": int(len(moves))}


def edge_fidelity(fl, arch, chain_ids, fin):
    """Each edge of the chain re-flown REPLAYS times from its exact parent state (the node the
    archive stored): how often the executor survives that one plan, and finishes (the last edge).
    A chain that finishes once but whose edges survive rarely is a lucky path, not a route the
    executor can fly."""
    n = min(REPLAYS, fl.S)
    out = []
    for a_, b_ in zip(chain_ids[:-1], chain_ids[1:]):
        m = arch.move[b_]
        res = fl.fly([(a_, m)] * n, arch, fin, record_path=False)
        out.append({"from": np.round(arch.state[a_]["origin"].astype(np.float64), 0).tolist(),
                    "move": int(m), "survived": int(sum((not r["died"]) and
                                                        (r["end"] is not None or r["fin"])
                                                        for r in res)),
                    "finished": int(sum(r["fin"] for r in res)), "n": int(n)})
    return out


def analyze_edges(fl, arch, chain_ids, fin, mins, rng, n_probe, parents):
    """Per chain edge (parent -> stored child via move m): 32 re-flights from the exact parent ->
    the child-key hit rate, distinct child keys, deaths, position / speed error to the stored child;
    for the first ``n_probe`` edges, the share of up to 16 surviving re-flown children from which a
    fresh search (400 expansions, 8 s) still finds a finishing chain."""
    n = min(REPLAYS, fl.S)
    out = []
    for j, (a_, b_) in enumerate(zip(chain_ids[:-1], chain_ids[1:])):
        m = arch.move[b_]
        res = fl.fly([(a_, m)] * n, arch, fin, record_path=False)
        target_key = arch.key[b_]
        tgt = arch.state[b_]
        hits, keys, pe, se, deaths, kids = 0, set(), [], [], 0, []
        for r in res:
            if r["fin"]:
                keys.add(("FIN",))
                hits += int(target_key == ("FIN",))
                continue
            if r["died"] or r["end"] is None:
                deaths += 1
                continue
            k = keys_of(r["end"][None], mins)[0]
            keys.add(k)
            hits += int(k == target_key)
            pe.append(float(np.linalg.norm(r["end"]["origin"].astype(np.float64)
                                           - tgt["origin"].astype(np.float64))))
            se.append(abs(float(np.linalg.norm(r["end"]["velocity"].astype(np.float64)))
                          - float(np.linalg.norm(tgt["velocity"].astype(np.float64)))))
            kids.append(r)
        e = {"edge": j + 1, "move": int(m), "n": n, "key_hit": hits,
             "distinct_keys": len(keys), "deaths": deaths,
             "pos_err_med": round(float(np.median(pe)), 1) if pe else None,
             "spd_err_med": round(float(np.median(se)), 1) if se else None}
        if j < n_probe and kids and target_key != ("FIN",):
            solv = 0
            probe = kids[:16]
            for r in probe:
                sr = search(fl, r["end"], r["keys"], r["obs"], fin, mins, rng, parents, 400, 8.0)
                solv += int(any(sr["found_by_move"]))
            e["solvable"], e["probed"] = solv, len(probe)
        out.append(e)
    return out


def _ckpt_choices(path) -> int:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = int((ck.get("config") or {}).get("plan_choices") or 0)
    if c <= 0:
        raise SystemExit("edge_archive: v1 needs a --plan-choices checkpoint")
    return c


if __name__ == "__main__":
    sys.exit(main())
