#!/usr/bin/env python3
"""kino_search.py - a physics-exact, time-ordered search over (position, velocity) cells.

The user (2026-10-16): "imagine you have this grid of voxels ... with each voxel you also
associate the velocity ... to which nearby voxels can we transition and which velocity will we
have ... maybe value iteration ... before we even run the agent on this map". The geodesic
goal field is a BFS over POSITION only, so it believes the player can glide level across a
void or climb out of a pit with no speed. This searches over position AND velocity, and every
edge is the game's own physics (the C core), so an edge exists only if the engine flies it.

    python tools/kino_search.py --map maps_pool/surf_unitfarmer2.bsp --out runs/research/kino/uf2

**A node** (generic: the same constants on every map, nothing read off a map) is

    (x, y, z) cell of --cell units  x  horizontal-speed bin  x  heading bin (--headings)
    x  vertical-speed bin

and keeps the FIRST full engine state that reached it (Go-Explore's archive): every expansion
restarts from an exact simulated state, so the bins decide only which states count as the
same, never where the next one is.

**An edge** - the user's "to which neighbouring voxels can we transition, and with which
velocity" - holds one control from CONTROLS (a side key, the view at a yaw offset from the
velocity heading as in the trainer's --view-absolute velocity mode, optionally W / jump) from
the node's state until the state's node changes: it moved into a neighbouring cell or its
speed, heading or vertical-speed bin changed. --max-edge ticks without a change (standing
still, stuck) is no edge. The map's fail zones (--teleport-fail) end an edge as a death; the
finish box is the goal.

**The search** is Dijkstra by game time from the map's spawn (edges take different numbers of
ticks), batched by --bucket ticks of arrival time. The first arrival at a node is its
earliest within one bucket, and the first goal hit is the fastest route this control set and
these bins can express - found by search over the real physics, with no reward and no
reference line.

Outputs: <out>.npz (each node's key, state fields, arrival tick, parent, control, the goal
hits), <out>.states.npy (the full engine states), <out>.log.jsonl (one line per bucket).
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import SurfCore, default_config          # noqa: E402
from surfgym.core import STATE_DTYPE, SurfState       # noqa: E402
from surfgym.zones import load_zones                  # noqa: E402

# horizontal-speed bin edges (u/s): finer where surf speeds change behaviour
SPEED_EDGES = np.array([0, 60, 150, 300, 450, 600, 800, 1000, 1250, 1500, 1800, 2100, 2500,
                        3000, 3600, 1e9], np.float64)
# vertical-speed bin edges (u/s)
VZ_EDGES = np.array([-1e9, -600, -200, 200, 600, 1e9], np.float64)
# the control set: (forward key bin, side key bin, jump, view yaw offset deg from the velocity
# heading). Engine bins: fwd/side {-400, 0, +400}[i] -> 1 = none; side 0 = A, 2 = D. Under
# 100 u/s the core turns the view by the offset per tick (clamped), i.e. it spins in place.
CONTROLS = (
    (1, 1, 0, 0.0),      # coast
    (2, 1, 0, 0.0),      # W (ground acceleration; in the air, accelerate along the view)
    (2, 1, 1, 0.0),      # W + jump
    (1, 0, 0, 15.0),     # A, turning gently left
    (1, 2, 0, -15.0),    # D, turning gently right
    (1, 0, 0, 50.0),     # A, turning hard left
    (1, 2, 0, -50.0),    # D, turning hard right
    (1, 0, 0, 0.0),      # A with the view ALONG the velocity: surfing a ramp on the left
    (1, 2, 0, 0.0),      # D, the same into a ramp on the right (the record holds these two)
)


# the controls expanded from a node in the air above GROUND_CTL_SPEED (no W, no W + jump)
AIR_CTL = (0, 3, 4, 5, 6, 7, 8)
GROUND_CTL_SPEED = 300.0


def states_rw(core):
    """A WRITABLE numpy view of the core's per-env states (the buffer set_pitch writes)."""
    ptr = core._lib.surf_states_ptr(core._handle())
    buf = (SurfState * core.num_envs).from_address(ptr)
    return np.frombuffer(buf, dtype=STATE_DTYPE)


GRAVITY = 800.0          # sv_gravity, the engine default every map here runs


def energy_height(origin, velocity):
    """z + |v|^2 / 2g: the height the state could reach by turning all of its speed into climb."""
    v = np.asarray(velocity, np.float64)
    return np.asarray(origin, np.float64)[:, 2] + (v * v).sum(1) / (2.0 * GRAVITY)


class Keyer:
    def __init__(self, mins, maxs, cell, headings, vz_edges=None):
        self.vz_edges = VZ_EDGES if vz_edges is None else np.asarray(vz_edges, np.float64)
        self.mins = np.asarray(mins, np.float64)
        self.cell = float(cell)
        self.dims = np.ceil((np.asarray(maxs, np.float64) - self.mins) / self.cell).astype(np.int64) + 1
        self.nh = int(headings)
        self.ns = len(SPEED_EDGES) - 1
        self.nz = len(self.vz_edges) - 1

    def group(self, origin, velocity, yaw, contact):
        """(cell, heading, vertical-speed bin, contact) - the node key WITHOUT speed: within a
        group, states are told apart by (arrival time, energy) Pareto labels instead"""
        c = np.floor((np.asarray(origin, np.float64) - self.mins) / self.cell).astype(np.int64)
        c = np.clip(c, 0, self.dims - 1)
        v = np.asarray(velocity, np.float64)
        vh = np.hypot(v[:, 0], v[:, 1])
        head = np.where(vh > 30.0, np.degrees(np.arctan2(v[:, 1], v[:, 0])), yaw)
        hb = (np.floor((np.mod(head, 360.0) / 360.0) * self.nh).astype(np.int64)) % self.nh
        zb = np.clip(np.searchsorted(self.vz_edges, v[:, 2], side="right") - 1, 0, self.nz - 1)
        k = c[:, 0]
        k = k * self.dims[1] + c[:, 1]
        k = k * self.dims[2] + c[:, 2]
        k = k * self.nh + hb
        k = k * self.nz + zb
        k = k * 2 + np.asarray(contact, np.int64)
        return k

    def __call__(self, origin, velocity, yaw):
        c = np.floor((np.asarray(origin, np.float64) - self.mins) / self.cell).astype(np.int64)
        c = np.clip(c, 0, self.dims - 1)
        v = np.asarray(velocity, np.float64)
        vh = np.hypot(v[:, 0], v[:, 1])
        sb = np.clip(np.searchsorted(SPEED_EDGES, vh, side="right") - 1, 0, self.ns - 1)
        head = np.where(vh > 30.0, np.degrees(np.arctan2(v[:, 1], v[:, 0])), yaw)
        hb = (np.floor((np.mod(head, 360.0) / 360.0) * self.nh).astype(np.int64)) % self.nh
        zb = np.clip(np.searchsorted(self.vz_edges, v[:, 2], side="right") - 1, 0, self.nz - 1)
        k = c[:, 0]
        k = k * self.dims[1] + c[:, 1]
        k = k * self.dims[2] + c[:, 2]
        k = k * self.ns + sb
        k = k * self.nh + hb
        k = k * self.nz + zb
        return k


class Archive:
    """Nodes = engine states, grouped by Keyer.group; within a group a node is kept only if no
    earlier node of that group has at least as much energy (Pareto labels over arrival time and
    energy height, with --energy-eps of slack)."""

    def __init__(self, cap=1 << 20):
        self.cap, self.n = cap, 0
        self.state = np.zeros(cap, STATE_DTYPE)
        self.parent = np.full(cap, -1, np.int64)
        self.ctrl = np.full(cap, -1, np.int8)
        self.tick = np.zeros(cap, np.int64)          # arrival time, physics ticks
        self.group = np.zeros(cap, np.int64)
        self.energy = np.zeros(cap, np.float32)
        self.contact = np.zeros(cap, np.int8)
        self.labels: dict = {}                       # group -> [node, ...]

    def _grow(self, m):
        while self.n + m > self.cap:
            self.cap *= 2
            for f in ("state", "parent", "ctrl", "tick", "group", "energy", "contact"):
                setattr(self, f, np.resize(getattr(self, f), self.cap))

    def offer(self, state, parent, ctrl, tick, group, energy, contact, eps):
        """Add the state as a node unless a node of its group arrived no later with at least
        energy - eps. Offers must come in nondecreasing arrival time. -> node index or -1."""
        lab = self.labels.get(group)
        if lab is not None:
            for i in lab:
                if self.energy[i] >= energy - eps:
                    return -1
        if self.n >= self.cap:
            self._grow(1)
        i = self.n
        self.state[i], self.parent[i], self.ctrl[i] = state, parent, ctrl
        self.tick[i], self.group[i], self.energy[i], self.contact[i] = tick, group, energy, contact
        self.n += 1
        if lab is None:
            self.labels[group] = [i]
        else:
            lab.append(i)
        return i


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--cell", type=float, default=64.0)
    ap.add_argument("--headings", type=int, default=16)
    ap.add_argument("--energy-eps", type=float, default=32.0,
                    help="energy-height slack (units) under which two states of one group count "
                         "as equal: the later one is dropped")
    ap.add_argument("--max-edge", type=int, default=100, help="ticks before an edge is dropped")
    ap.add_argument("--bucket", type=int, default=4, help="arrival-time batch, ticks")
    ap.add_argument("--envs", type=int, default=65536)
    ap.add_argument("--max-seconds", type=float, default=60.0, help="game-time horizon")
    ap.add_argument("--max-nodes", type=int, default=60_000_000)
    ap.add_argument("--maxvel", type=float, default=4000.0)
    ap.add_argument("--goal-extra", type=float, default=2.0,
                    help="game seconds to keep searching after the first goal hit")
    ap.add_argument("--bench-demo", default=None,
                    help="ANALYSIS BENCH ONLY (CLAUDE.md section 0): start the search from a "
                         "parsed record's state (a *.frames.npz) at --bench-time instead of the "
                         "spawn, to measure what the search can fly from there. Never a route or "
                         "a field for training")
    ap.add_argument("--bench-time", type=float, default=0.0)
    ap.add_argument("--tick-ms", type=int, default=10,
                    help="the physics tick, whole milliseconds (10 = the trainer's default; 8 ~ "
                         "the 125-131 fps real players run, under which the air-strafe cap of "
                         "30 u/s per frame gives 25-30%% more acceleration per second)")
    ap.add_argument("--vz-step", type=float, default=0.0,
                    help="vertical-speed bins of this width (u/s) over -3000..3000 instead of the "
                         "5 default bins")
    a = ap.parse_args()

    bsp = Path(a.map)
    zones = load_zones(str(bsp))
    core = SurfCore(str(bsp), default_config(
        num_envs=a.envs, lidar_w=0, lidar_h=0, view_mode=1, lookahead_k=0,
        sv_maxvelocity=a.maxvel, max_episode_ticks=10 ** 8, spawn_mode=1,
        yaw_jitter_deg=0.0), tick_ms=float(a.tick_ms))
    core.set_teleport_fail(True)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    core.reset(0)
    rw = states_rw(core)
    sv = core.states_view
    TS = a.tick_ms / 1000.0                     # seconds per tick
    mins, maxs = core.map_bounds()
    vz_edges = None
    if a.vz_step > 0:
        vz_edges = np.concatenate([[-1e9], np.arange(-3000.0, 3000.0 + 1e-6, a.vz_step), [1e9]])
    key = Keyer(mins, maxs, a.cell, a.headings, vz_edges)
    nc = len(CONTROLS)
    ctl = np.asarray(CONTROLS, np.float64)
    print(f"map {bsp.stem}: groups = {a.cell:g} u cells x {key.nh} headings x {key.nz} vz bins x "
          f"contact; Pareto labels over (arrival time, energy height) with {a.energy_eps:g} u "
          f"slack; {nc} controls; an edge runs until the group changes (<= {a.max_edge} ticks); "
          f"Dijkstra by game time in {a.bucket}-tick buckets", flush=True)

    # the goal field, for REPORTING how far along the map the search is (it never steers it)
    gf = None
    try:
        from surfgym.goalfield import build_goal_field
        gf = build_goal_field(core, zones["end"], cell=32)
    except Exception as exc:                                # pragma: no cover
        print(f"(no goal field for reporting: {exc!r})")

    start = core.get_states()[0].copy()
    start["velocity"] = (0.0, 0.0, 0.0)
    start["tick"] = 0
    if a.bench_demo:
        w = np.load(a.bench_demo)
        i = int(np.searchsorted(w["t"] - w["t"][0], a.bench_time))
        start["origin"] = w["simorg"][i]
        start["velocity"] = w["simvel"][i]
        start["yaw"] = float(w["viewangles"][i, 1])
        start["onground"] = -1                      # the engine re-categorises on its first tick
        print(f"!! ANALYSIS BENCH: starting from {Path(a.bench_demo).name} at {a.bench_time:g} s "
              f"(origin {np.round(w['simorg'][i]).tolist()}, speed "
              f"{np.linalg.norm(w['simvel'][i]):.0f} u/s) - measurement only, never training",
              flush=True)
    d0 = float(gf.sample(np.asarray(start["origin"], np.float64)[None])[0]) if gf else float("nan")

    arc = Archive()
    so, svv = np.asarray(start["origin"])[None], np.asarray(start["velocity"])[None]
    g0 = int(key.group(so, svv, np.array([float(start["yaw"])]), np.array([1]))[0])
    arc.offer(start, -1, -1, 0, g0, float(energy_height(so, svv)[0]), 1, a.energy_eps)
    heap = [(0, 0)]                     # (arrival tick, node) - unexpanded nodes
    goals = []                          # (node, control, arrival tick)
    first_goal = None
    best_prog = 0.0
    log = open(f"{a.out}.log.jsonl", "w", encoding="utf-8")
    t_wall = time.perf_counter()
    n_edges = n_deaths = n_dominated = 0
    horizon = int(round(a.max_seconds / TS))
    last_print = -1

    while heap:
        t_now = heap[0][0]
        if t_now > horizon or (first_goal is not None and t_now > first_goal + a.goal_extra / TS):
            break
        if arc.n >= a.max_nodes:
            print(f"--max-nodes {a.max_nodes:,} reached - stopping")
            break
        batch = []
        while heap and heap[0][0] < t_now + a.bucket:
            batch.append(heapq.heappop(heap)[1])
        batch = np.asarray(batch, np.int64)
        # in the air above a walk, W (and its jump) adds nothing a coast does not: the air-accel
        # cap of 30 u/s along the wish direction is already exceeded along the velocity. Those
        # two controls are expanded only from nodes on the ground or under GROUND_CTL_SPEED.
        st_b = arc.state[batch]
        slow = (st_b["onground"] != -1) | (np.hypot(st_b["velocity"][:, 0],
                                                    st_b["velocity"][:, 1]) < GROUND_CTL_SPEED)
        jobs_node = np.concatenate([np.repeat(batch[slow], nc),
                                    np.repeat(batch[~slow], len(AIR_CTL))])
        jobs_ctl = np.concatenate([np.tile(np.arange(nc), int(slow.sum())),
                                   np.tile(np.asarray(AIR_CTL), int((~slow).sum()))])
        found_S, found_P, found_C, found_T, found_K = [], [], [], [], []
        for b0 in range(0, len(jobs_node), a.envs):
            jn = jobs_node[b0:b0 + a.envs]
            jk = jobs_ctl[b0:b0 + a.envs]
            m = len(jn)
            rw[:m] = arc.state[jn]
            if m < a.envs:
                rw[m:] = start
            rw["tick"][:] = 0
            rw["stuck_ticks"][:] = 0
            g_from = arc.group[jn]
            act = np.zeros((a.envs, 6), np.int32)
            act[:, 0], act[:, 1], act[:, 2], act[:, 3] = 7, 3, 1, 1
            view = np.zeros((a.envs, 2), np.float32)
            view[:, 1] = -10.0
            c = ctl[jk]
            act[:m, 2], act[:m, 3], act[:m, 4] = c[:, 0], c[:, 1], c[:, 2]
            view[:m, 0] = c[:, 3]
            alive = np.zeros(a.envs, bool)
            alive[:m] = True
            for tick in range(1, a.max_edge + 1):
                _o, _r, done, trunc, _t = core.step(act, view=view)
                ended = (done | trunc).astype(bool) & alive
                if ended.any():
                    hit = core.goal_hits.astype(bool) & ended
                    for j in np.flatnonzero(hit):
                        goals.append((int(jn[j]), int(jk[j]), int(arc.tick[jn[j]]) + tick))
                    n_deaths += int((ended & ~hit).sum())
                    alive &= ~ended
                if tick == 1:
                    act[:, 4] = 0                     # a jump is one press
                li = np.flatnonzero(alive)
                if len(li) == 0:
                    break
                cnt = core.get_touch()[0]
                G = key.group(sv["origin"][li], sv["velocity"][li], sv["yaw"][li],
                              (cnt[li] > 0).astype(np.int64))
                moved = G != g_from[li]
                if moved.any():
                    mv = li[moved]
                    found_S.append(sv[mv].copy())
                    found_P.append(jn[mv])
                    found_C.append(jk[mv])
                    found_T.append(arc.tick[jn[mv]] + tick)
                    found_K.append((cnt[mv] > 0).astype(np.int8))
                    alive[mv] = False
            n_edges += m
        if found_S:
            S = np.concatenate(found_S)
            P = np.concatenate(found_P)
            C = np.concatenate(found_C).astype(np.int8)
            T = np.concatenate(found_T)
            CT = np.concatenate(found_K)
            G = key.group(S["origin"], S["velocity"], S["yaw"], CT.astype(np.int64))
            E = energy_height(S["origin"], S["velocity"])
            S["tick"] = 0
            order = np.lexsort((-E, T))               # by arrival time, then the most energy
            for j in order.tolist():
                i = arc.offer(S[j], int(P[j]), int(C[j]), int(T[j]), int(G[j]), float(E[j]),
                              int(CT[j]), a.energy_eps)
                if i < 0:
                    n_dominated += 1
                else:
                    heapq.heappush(heap, (int(T[j]), int(i)))
        if goals and first_goal is None:
            first_goal = min(g[2] for g in goals)
            print(f"** GOAL reached at {first_goal * TS:.2f} s of game time "
                  f"({len(goals)} hit(s)) after {time.perf_counter() - t_wall:.0f} s", flush=True)
        if len(batch) and gf is not None:
            d = gf.sample(np.asarray(arc.state["origin"][batch], np.float64))
            d = d[d < gf.reach_max]
            if len(d):
                best_prog = max(best_prog, 1.0 - float(d.min()) / d0)
        sec = int(t_now * TS * 2)
        if sec != last_print:
            last_print = sec
            rec = {"game_s": t_now * TS, "nodes": int(arc.n), "open": len(heap),
                   "batch": int(len(batch)), "edges": int(n_edges), "deaths": int(n_deaths),
                   "dominated": int(n_dominated), "goals": len(goals),
                   "best_progress": best_prog, "wall_s": round(time.perf_counter() - t_wall, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"t {t_now * TS:6.2f} s: nodes {arc.n:>11,}  open {len(heap):>10,}  "
                  f"edges {n_edges:>12,}  deaths {n_deaths:>11,}  dominated {n_dominated:>11,}  "
                  f"best progress {best_prog:6.1%}  goals {len(goals)}  "
                  f"[{time.perf_counter() - t_wall:7.1f} s]", flush=True)

    log.close()
    n = arc.n
    np.savez_compressed(
        f"{a.out}.npz", group=arc.group[:n], origin=arc.state["origin"][:n],
        velocity=arc.state["velocity"][:n], yaw=arc.state["yaw"][:n],
        onground=arc.state["onground"][:n], energy=arc.energy[:n], contact=arc.contact[:n],
        tick=arc.tick[:n], parent=arc.parent[:n], ctrl=arc.ctrl[:n],
        goals=np.asarray(goals, np.int64).reshape(-1, 3),
        controls=np.asarray(CONTROLS, np.float64), cell=a.cell, headings=a.headings,
        vz_edges=key.vz_edges, energy_eps=a.energy_eps, mins=np.asarray(mins),
        maxs=np.asarray(maxs), d0=d0, map=str(bsp.stem), tick_ms=a.tick_ms,
        bench=(f"{a.bench_demo}@{a.bench_time:g}" if a.bench_demo else ""))
    np.save(f"{a.out}.states.npy", arc.state[:n])
    print(f"wrote {a.out}.npz: {n:,} nodes, {n_edges:,} edges, {len(goals)} goal hits"
          + (f", fastest {first_goal * TS:.2f} s" if first_goal is not None else ""))


if __name__ == "__main__":
    main()
