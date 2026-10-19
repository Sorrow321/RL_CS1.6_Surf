#!/usr/bin/env python3
"""go_cem.py - exploration by OPTIMAL CONTROL: a Go-Explore archive whose explorer is the
cross-entropy method in the real engine (the user, 2026-10-16: "maybe the optimal solution would
be to combine ... optimal control ... guided ... instead of brute force all states").

Why: precision manoeuvres (unitfarmer2's pit climb needs view offsets of ~0.5 deg updated every
tick, measured 2026-10-16) are out of reach of random actions and of exhaustive search over a
binned state, but CEM finds such a climb in seconds once it is ASKED for something the climb
achieves. A finite-horizon objective on the geodesic field cannot ask for it (the field punishes
the dive for ~5 s), so the objective here asks for NOVELTY in (position x energy) instead: a
fast state deep in the pit is a new cell, so the explorer is drawn to exactly the states the
field punishes. No reward, no reference line, no demo; the same constants on every map.

    python tools/go_cem.py --map maps_pool/surf_unitfarmer2.bsp --out runs/research/kino/gocem_uf2

**The archive**: cell = (x, y, z) // --cell  x  energy height (z + |v|^2/2g) // --ecell. Each cell
keeps the EARLIEST state that reached it (game time from the spawn) and how: the expansion that
produced it and the tick within it.

**An expansion**: pick a cell (weight 1 / sqrt(1 + times picked), Go-Explore's count rule),
run --iters rounds of CEM from its exact state over --horizon seconds (side key A / none / D,
forward key, view yaw offset from the live velocity, at knots every --knot ticks), scoring
each candidate by the number of archive-NEW cells it visits alive, then add every state of the
--keep best candidates to the archive (new cells, or earlier arrivals at known ones).

A finish-box crossing is the goal: the route is rebuilt from the expansions that led to it.

--bench-demo FILE --bench-time T starts the archive from a parsed record's state instead of the
spawn: an ANALYSIS BENCH (CLAUDE.md section 0) that splits a failure into "cannot reach that
state" and "cannot continue from it". The record's inputs are never read, nothing is trained, and
progress stays measured against the TRUE spawn's field value.

**v2 options (2026-10-16 morning; the defaults are v1, bit for bit).** v1 filled unitfarmer2's
start pit (13,554 cells) and never left it: it reaches the pit ~300 u of energy height below
the record, and a cell that keeps its EARLIEST arrival throws away a later, more energetic one,
so energy gains smaller than one --ecell bin per expansion can never chain. Generic fixes:
    --key pos-heading   cell = position x horizontal-heading sector (--headings), no energy bin;
                        --vbins K also splits it by the velocity's vertical angle (K equal bins
                        over -90..90 deg). Energy dominance is OPTIMISTIC about contact losses
                        still to come: a steep free fall past a ramp holds more energy than the
                        surfing state on it, so in one (position x heading) cell it can
                        overwrite the state that leads on. In unitfarmer2's dive the archive
                        kept the steep fall (velocity vertical angle -66 deg, 1,375 u) where
                        the record surfs (-47 deg, 1,290 u), and every plan from it died
                        (measured 2026-10-16)
    --rep energy        a cell keeps its MOST ENERGETIC state (ties: the earliest); energy can
                        only be raised by skill (strafing) and only lost to contacts, so at one
                        position and heading the more energetic state dominates
    --novelty energy    score = archive-new cells + per known cell visited alive, its energy
                        gain over the cell's state / --ecell, clipped to [0, 1]
    --select fresh      pick weight 1 / sqrt(1 + picks since the cell was created or raised by
                        more than --improve-eps)
    --select fresh-lift the fresh weight x (1 + the number of cells within --lift-radius whose
                        energy is more than --ecell BELOW this cell's): the expected yield of the
                        energy-gain score, so a fast state deep in a pit full of slow states is
                        expanded until its neighbours catch up (self-correcting; recomputed every
                        10 expansions, new cells count 0 until then)
    --energy-coef L     score += L x (energy at the last alive tick - the start's) / --ecell:
                        explore WITHOUT WASTING ENERGY (climbs are neutral, strafing gains,
                        contacts cost); 0 = off
    --jump              the jump key joins the action set (a per-knot Bernoulli, initial
                        probability --jump-p0, refit like the others). The agent's action space
                        has it; without it the explorer cannot leave the ground on a gentle
                        slope, and on the ground speed is capped at 250 u/s while an airborne
                        strafe is not capped (unitfarmer2's archives never exceeded 1,253 u of
                        energy height without it)
    --mpc-exec N        an expansion is a CLOSED-LOOP flight of --horizon seconds instead of one
                        open-loop plan: CEM over --mpc-plan seconds from the CURRENT state, the
                        first N ticks of the best plan executed, replanned from where it landed
                        (warm-started), --mpc-iters rounds per replan, plans that die within the
                        plan horizon charged 1,000. Measured: open-loop CEM keeps 869 u of energy
                        height through unitfarmer2's pit U-turn where the record keeps 1,118 and
                        leaves the archive in DOOMED states (every plan dies within 1.2 s); a
                        1.2 s-lookahead MPC keeps 1,173-1,250 (tools/mpc_bench.py). Expansions
                        then store per-tick controls (saved knot = 1, replayed by go_cem_route.py
                        unchanged)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym import SurfCore, default_config          # noqa: E402
from surfgym.core import STATE_DTYPE                  # noqa: E402
from surfgym.zones import load_zones                  # noqa: E402
from kino_search import states_rw                     # noqa: E402

G = 800.0


class Cells:
    def __init__(self, mins, maxs, cell, ecell, key="pos-energy", headings=4, vbins=0):
        self.mins = np.asarray(mins, np.float64)
        self.cell, self.ecell = float(cell), float(ecell)
        self.key, self.headings, self.vbins = key, int(headings), int(vbins)
        self.dims = np.ceil((np.asarray(maxs, np.float64) - self.mins) / self.cell).astype(np.int64) + 1
        self.e0 = float(self.mins[2])

    def __call__(self, origin, velocity):
        o = np.asarray(origin, np.float64)
        v = np.asarray(velocity, np.float64)
        c = np.clip(np.floor((o - self.mins) / self.cell).astype(np.int64), 0, self.dims - 1)
        if self.key == "pos-energy":
            e = o[..., 2] + (v * v).sum(-1) / (2 * G)
            sub = np.clip(np.floor((e - self.e0) / self.ecell).astype(np.int64), 0, 4095)
        else:                                       # pos-heading: sector of the horizontal velocity
            ang = np.arctan2(v[..., 1], v[..., 0])
            sub = np.floor((ang + np.pi) / (2 * np.pi) * self.headings).astype(np.int64)
            sub = np.clip(sub, 0, self.headings - 1)
            sub = np.where(np.hypot(v[..., 0], v[..., 1]) < 50.0, self.headings, sub)
            if self.vbins > 0:                  # x the vertical angle of the velocity
                pit = np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1]))
                pb = np.floor((pit + np.pi / 2) / np.pi * self.vbins).astype(np.int64)
                sub = sub * self.vbins + np.clip(pb, 0, self.vbins - 1)
        return ((c[..., 0] * self.dims[1] + c[..., 1]) * self.dims[2] + c[..., 2]) * 4096 + sub


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tick-ms", type=int, default=7)
    ap.add_argument("--cell", type=float, default=192.0)
    ap.add_argument("--ecell", type=float, default=96.0)
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--knot", type=int, default=4)
    ap.add_argument("--pop", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--keep", type=int, default=12)
    ap.add_argument("--max-offset", type=float, default=8.0)
    ap.add_argument("--expansions", type=int, default=5000)
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-demo", default=None, help="ANALYSIS BENCH: start from this record")
    ap.add_argument("--bench-time", type=float, default=0.0)
    ap.add_argument("--key", default="pos-energy", choices=("pos-energy", "pos-heading"))
    ap.add_argument("--headings", type=int, default=4)
    ap.add_argument("--vbins", type=int, default=0,
                    help="(pos-heading) also bin the velocity's vertical angle into this many bins")
    ap.add_argument("--rep", default="earliest", choices=("earliest", "energy"))
    ap.add_argument("--novelty", default="cells", choices=("cells", "energy"))
    ap.add_argument("--select", default="count", choices=("count", "fresh", "fresh-lift"))
    ap.add_argument("--lift-radius", type=float, default=0.0,
                    help="(fresh-lift) neighbour radius in units, default 2 x --cell")
    ap.add_argument("--improve-eps", type=float, default=24.0,
                    help="energy gain (u) that makes a raised cell fresh again (--select fresh)")
    ap.add_argument("--energy-coef", type=float, default=0.0,
                    help="weight of the expansion's own energy change in the CEM score")
    ap.add_argument("--jump", action="store_true", help="add the jump key to the action set")
    ap.add_argument("--jump-p0", type=float, default=0.1)
    ap.add_argument("--mpc-exec", type=int, default=0,
                    help="ticks executed per replan; > 0 makes every expansion a closed-loop flight")
    ap.add_argument("--mpc-plan", type=float, default=1.2, help="MPC planning horizon, seconds")
    ap.add_argument("--mpc-iters", type=int, default=3)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    bsp = Path(a.map)
    zones = load_zones(str(bsp))
    core = SurfCore(str(bsp), default_config(
        num_envs=a.pop, lidar_w=0, lidar_h=0, view_mode=1, lookahead_k=0,
        sv_maxvelocity=4000.0, max_episode_ticks=10 ** 8, spawn_mode=1, yaw_jitter_deg=0.0),
        tick_ms=float(a.tick_ms))
    core.set_teleport_fail(True)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    core.reset(0)
    rw = states_rw(core)
    sv = core.states_view
    mins, maxs = core.map_bounds()
    cells = Cells(mins, maxs, a.cell, a.ecell, a.key, a.headings, a.vbins)
    from surfgym.goalfield import build_goal_field
    gf = build_goal_field(core, zones["end"], cell=32)      # REPORTING only
    TS = a.tick_ms / 1000.0
    H = int(round(a.horizon / TS))
    K = H // a.knot + 1

    start = core.get_states()[0].copy()
    start["velocity"] = (0.0, 0.0, 0.0)
    start["tick"], start["stuck_ticks"] = 0, 0
    d0 = float(gf.sample(np.asarray(start["origin"], np.float64)[None])[0])
    if a.bench_demo:
        w = np.load(a.bench_demo)
        i = int(np.searchsorted(w["t"] - w["t"][0], a.bench_time))
        start["origin"], start["velocity"] = w["simorg"][i], w["simvel"][i]
        start["yaw"], start["onground"] = float(w["viewangles"][i, 1]), -1
        print(f"!! ANALYSIS BENCH: the archive starts from {Path(a.bench_demo).name} at "
              f"{a.bench_time:g} s (z {w['simorg'][i][2]:.0f}, speed "
              f"{np.linalg.norm(w['simvel'][i]):.0f}); the record's inputs are never read; "
              f"progress is against the true spawn", flush=True)

    # the archive
    A_key, A_state, A_t, A_exp, A_tick, A_picked = [], [], [], [], [], []
    A_E, A_fresh = [], []    # the representative's energy height; picks since created / raised
    index = {}
    expansions = []          # (start archive id, side (K,), fwd (K,), off (K,))

    def put(key, state, t, exp, tick, e):
        i = index.get(key)
        if i is None:
            index[key] = len(A_key)
            A_key.append(key); A_state.append(state); A_t.append(t)
            A_exp.append(exp); A_tick.append(tick); A_picked.append(0)
            A_E.append(e); A_fresh.append(0)
            return True
        if a.rep == "earliest":
            better = t < A_t[i]
        else:
            better = e > A_E[i] + 1.0 or (abs(e - A_E[i]) <= 1.0 and t < A_t[i])
        if better:
            if e > A_E[i] + a.improve_eps:
                A_fresh[i] = 0
            A_state[i], A_t[i], A_exp[i], A_tick[i], A_E[i] = state, t, exp, tick, e
        return False

    def energy_of(o, v):
        o = np.asarray(o, np.float64)
        v = np.asarray(v, np.float64)
        return o[..., 2] + (v * v).sum(-1) / (2 * G)

    k0 = int(cells(np.asarray(start["origin"])[None], np.asarray(start["velocity"])[None])[0])
    put(k0, start.copy(), 0, -1, 0, float(energy_of(start["origin"], start["velocity"])))
    goals = []               # (expansion id, candidate rank, tick, arrival game tick)
    log = open(f"{a.out}.log.jsonl", "w", encoding="utf-8")
    t_wall = time.perf_counter()

    def save():
        _save(a, A_key, A_t, A_exp, A_tick, A_state, expansions, goals, d0, bsp, A_E)
    last_save = [time.perf_counter()]
    best_prog = [0.0]

    def post_expansion(added, n_before, ex):
        """progress report (field = REPORTING only), periodic save, log line, goal notice"""
        if len(A_key) > n_before:
            o = np.asarray([s["origin"] for s in A_state[n_before:]], np.float64)
            d = gf.sample(o)
            d = d[d < gf.reach_max]
            if len(d):
                best_prog[0] = max(best_prog[0], 1.0 - float(d.min()) / d0)
        if time.perf_counter() - last_save[0] > 600:
            save()
            last_save[0] = time.perf_counter()
        if ex % 10 == 0 or goals:
            rec = {"expansion": ex, "cells": len(A_key), "added": added, "goals": len(goals),
                   "best_progress": best_prog[0], "max_energy": round(max(A_E), 1),
                   "wall_s": round(time.perf_counter() - t_wall, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"exp {ex:5d}: cells {len(A_key):>8,}  +{added:<5d}  best progress "
                  f"{best_prog[0]:6.1%}  max energy {max(A_E):7.0f}  goals {len(goals)}  "
                  f"[{time.perf_counter() - t_wall:7.1f} s]", flush=True)
        if goals:
            g = min(goals, key=lambda x: x[3])
            print(f"** GOAL: reached at {g[3] * TS:.2f} s of game time (expansion {g[0]})",
                  flush=True)

    act = np.zeros((a.pop, 6), np.int32)
    act[:, 0], act[:, 1] = 7, 3
    view = np.zeros((a.pop, 2), np.float32)
    view[:, 1] = -10.0

    def simulate(st0, side, fwd, off, record=False, jump=None, n=None):
        """all candidates from st0 over n ticks (default H); -> per-tick cell keys (pop, n), alive
        mask (pop, n), goal tick per candidate (-1 none), (record) the per-tick states of every
        candidate, and the per-tick energy heights"""
        n = H if n is None else n
        Kc = off.shape[1]
        rw[:] = st0
        keys = np.zeros((a.pop, n), np.int64)
        en = np.zeros((a.pop, n), np.float64)
        alive_t = np.zeros((a.pop, n), bool)
        goal_tick = np.full(a.pop, -1, np.int64)
        alive = np.ones(a.pop, bool)
        states = np.zeros((a.pop, n), STATE_DTYPE) if record else None
        for k in range(n):
            j, f = divmod(k, a.knot)
            j1 = min(j + 1, Kc - 1)
            view[:, 0] = off[:, j] + (off[:, j1] - off[:, j]) * (f / a.knot)
            act[:, 2] = fwd[:, j]
            act[:, 3] = side[:, j]
            if jump is not None:
                act[:, 4] = jump[:, j]
            _o, _r, done, trunc, _t = core.step(act, view=view)
            ended = (done | trunc).astype(bool) & alive
            if ended.any():
                hit = core.goal_hits.astype(bool) & ended
                goal_tick[hit] = k + 1
                alive &= ~ended
            keys[:, k] = cells(sv["origin"], sv["velocity"])
            en[:, k] = energy_of(sv["origin"], sv["velocity"])
            alive_t[:, k] = alive
            if record:
                states[:, k] = sv
        return keys, alive_t, goal_tick, states, en

    def novelty(keys, alive_t, known, en=None, known_E=None):
        """number of distinct archive-new cells each candidate visits while alive; with
        --novelty energy, plus per distinct KNOWN cell its best energy gain over the cell's
        state, in --ecell units clipped to [0, 1]. `known` is sorted, known_E aligned to it"""
        pos = np.minimum(np.searchsorted(known, keys), len(known) - 1)
        found = known[pos] == keys
        m = alive_t & ~found
        rows_all = np.repeat(np.arange(a.pop), keys.shape[1])
        rows = rows_all[m.ravel()]
        ks = keys.ravel()[m.ravel()]
        score = np.zeros(a.pop)
        if len(ks):
            pair = np.unique(np.stack([rows, ks], 1), axis=0)
            score += np.bincount(pair[:, 0], minlength=a.pop).astype(np.float64)
        if a.novelty == "energy":
            gain = np.where(alive_t & found, (en - known_E[pos]) / a.ecell, 0.0)
            g = np.clip(gain, 0.0, 1.0).ravel()
            sel = g > 0
            if sel.any():
                r, kk, gg = rows_all[sel], keys.ravel()[sel], g[sel]
                o = np.lexsort((kk, r))
                r, kk, gg = r[o], kk[o], gg[o]
                first = np.flatnonzero(np.r_[True, (r[1:] != r[:-1]) | (kk[1:] != kk[:-1])])
                mx = np.maximum.reduceat(gg, first)
                score += np.bincount(r[first], weights=mx, minlength=a.pop)
        return score

    def energy_term(en, alive_t, e0):
        """--energy-coef x (energy at each candidate's last alive tick - e0) / --ecell"""
        if a.energy_coef == 0.0:
            return 0.0
        last = alive_t.shape[1] - 1 - np.argmax(alive_t[:, ::-1], axis=1)
        e_end = en[np.arange(len(en)), last]
        return np.where(alive_t.any(1), a.energy_coef * (e_end - e0) / a.ecell, 0.0)

    def sample_plan(Kc, p_side, p_fwd, mu, sig, p_jump):
        u = rng.random((a.pop, Kc, 1))
        side = (u > np.cumsum(p_side, 1)[None, :, :]).sum(-1).clip(0, 2).astype(np.int32)
        fwd = np.where(rng.random((a.pop, Kc)) < p_fwd, 2, 1).astype(np.int32)
        off = np.clip(mu + sig * rng.standard_normal((a.pop, Kc)), -a.max_offset, a.max_offset)
        jump = (rng.random((a.pop, Kc)) < p_jump).astype(np.int32) if a.jump else None
        return side, fwd, off, jump

    def mpc_flight(st0, known, known_E):
        """closed-loop flight of up to H ticks from st0 (see --mpc-exec). -> per-tick controls
        (side, fwd, off float32, jump), the per-tick states / cell keys / energies of the flight
        while alive, and the goal tick (or -1)"""
        Hp = int(round(a.mpc_plan / TS))
        Kp = Hp // a.knot + 1
        sh = max(1, a.mpc_exec // a.knot)
        p_side = np.full((Kp, 3), 1.0 / 3.0)
        p_fwd = np.full(Kp, 0.3)
        mu = np.zeros(Kp)
        sig = np.full(Kp, 3.0)
        p_jump = np.full(Kp, a.jump_p0)
        n_el = max(8, a.pop // 25)
        best = None
        cur = st0
        flight = {}                          # cells this flight visited -> its best energy there
        c_side, c_fwd, c_off, c_jump, f_st, f_key, f_en = [], [], [], [], [], [], []
        goal = -1
        while len(c_side) < H:
            kk, EE = known, known_E
            if flight:                       # this flight's own cells are no longer novel
                ek = np.fromiter(flight.keys(), np.int64, len(flight))
                eE = np.fromiter(flight.values(), np.float64, len(flight))
                pos = np.minimum(np.searchsorted(known, ek), len(known) - 1)
                inside = known[pos] == ek
                EE = known_E.copy()
                EE[pos[inside]] = np.maximum(EE[pos[inside]], eE[inside])
                kk = np.concatenate([known, ek[~inside]])
                EE = np.concatenate([EE, eE[~inside]])
                o = np.argsort(kk)
                kk, EE = kk[o], EE[o]
            e_cur = float(energy_of(cur["origin"], cur["velocity"]))
            for it in range(a.mpc_iters):
                side, fwd, off, jump = sample_plan(Kp, p_side, p_fwd, mu, sig, p_jump)
                if best is not None:
                    side[0], fwd[0], off[0] = best[:3]
                    if a.jump:
                        jump[0] = best[3]
                keys, alive_t, goal_tick, _, en = simulate(cur, side, fwd, off, jump=jump, n=Hp)
                score = novelty(keys, alive_t, kk, en, EE) + energy_term(en, alive_t, e_cur)
                score[~alive_t[:, -1]] -= 1000.0
                score[goal_tick > 0] += 1e6
                el = np.argsort(-score)[:n_el]
                best = (side[el[0]].copy(), fwd[el[0]].copy(), off[el[0]].copy(),
                        jump[el[0]].copy() if a.jump else None)
                for c in range(3):
                    p_side[:, c] = 0.7 * (side[el] == c).mean(0) + 0.3 * p_side[:, c]
                p_side = np.clip(p_side, 0.02, 1.0)
                p_side /= p_side.sum(1, keepdims=True)
                p_fwd = np.clip(0.7 * (fwd[el] == 2).mean(0) + 0.3 * p_fwd, 0.02, 0.98)
                mu = 0.7 * off[el].mean(0) + 0.3 * mu
                sig = np.maximum(0.7 * off[el].std(0) + 0.3 * sig, 0.2)
                if a.jump:
                    p_jump = np.clip(0.7 * jump[el].mean(0) + 0.3 * p_jump, 0.02, 0.98)
            # execute the first --mpc-exec ticks of the best plan (every env identical)
            b_side, b_fwd, b_off, b_jump = best
            rw[:] = cur
            ended = False
            for k in range(a.mpc_exec):
                if len(c_side) >= H:
                    break
                j, f = divmod(k, a.knot)
                j1 = min(j + 1, Kp - 1)
                view[:, 0] = b_off[j] + (b_off[j1] - b_off[j]) * (f / a.knot)
                act[:, 2] = b_fwd[j]
                act[:, 3] = b_side[j]
                act[:, 4] = b_jump[j] if a.jump else 0
                c_side.append(int(b_side[j])); c_fwd.append(int(b_fwd[j]))
                c_off.append(view[0, 0]); c_jump.append(int(b_jump[j]) if a.jump else 0)
                _o, _r, done, trunc, _t = core.step(act, view=view)
                if done[0] or trunc[0]:
                    if core.goal_hits[0]:
                        goal = len(c_side)
                    ended = True
                    break
                st = sv[0].copy()
                key = int(cells(st["origin"][None], st["velocity"][None])[0])
                e = float(energy_of(st["origin"], st["velocity"]))
                f_st.append(st); f_key.append(key); f_en.append(e)
                flight[key] = max(flight.get(key, -np.inf), e)
            if ended:
                break
            cur = sv[0].copy()               # NOT tick-reset: a replay continues the same way
            # warm start: shift everything forward by the executed knots
            mu = np.r_[mu[sh:], np.zeros(sh)]
            sig = np.r_[np.maximum(sig[sh:], 1.0), np.full(sh, 3.0)]
            p_side = np.r_[p_side[sh:], np.full((sh, 3), 1.0 / 3.0)]
            p_fwd = np.r_[p_fwd[sh:], np.full(sh, 0.3)]
            p_jump = np.r_[p_jump[sh:], np.full(sh, a.jump_p0)]
            best = (np.r_[b_side[sh:], np.repeat(b_side[-1], sh)],
                    np.r_[b_fwd[sh:], np.repeat(b_fwd[-1], sh)],
                    np.r_[b_off[sh:], np.repeat(b_off[-1], sh)],
                    np.r_[b_jump[sh:], np.repeat(b_jump[-1], sh)] if a.jump else None)
        T = len(c_side)
        pad = H - T
        ctrl = (np.r_[np.asarray(c_side, np.int32), np.ones(pad, np.int32)],
                np.r_[np.asarray(c_fwd, np.int32), np.ones(pad, np.int32)],
                np.r_[np.asarray(c_off, np.float32), np.zeros(pad, np.float32)],
                np.r_[np.asarray(c_jump, np.int32), np.zeros(pad, np.int32)])
        return ctrl, f_st, f_key, f_en, goal

    lift = np.zeros(0)
    for ex in range(a.expansions):
        if time.perf_counter() - t_wall > a.minutes * 60:
            print("time budget reached")
            break
        n = len(A_key)
        if a.select == "count":
            w = 1.0 / np.sqrt(1.0 + np.asarray(A_picked, np.float64))
        else:
            w = 1.0 / np.sqrt(1.0 + np.asarray(A_fresh, np.float64))
        if a.select == "fresh-lift":
            if ex % 10 == 0:
                from scipy.spatial import cKDTree
                P_ = np.asarray([q["origin"] for q in A_state], np.float64)
                E_ = np.asarray(A_E, np.float64)
                nb = cKDTree(P_).query_ball_point(P_, a.lift_radius or 2.0 * a.cell)
                lift = np.array([int(np.count_nonzero(E_[idx] < E_[i] - a.ecell))
                                 for i, idx in enumerate(nb)], np.float64)
            w = w * (1.0 + np.r_[lift, np.zeros(n - len(lift))])
        src = int(rng.choice(n, p=w / w.sum()))
        A_picked[src] += 1
        A_fresh[src] += 1
        st0 = A_state[src]
        e_src = float(energy_of(st0["origin"], st0["velocity"]))
        known = np.asarray(A_key, np.int64)
        korder = np.argsort(known)
        known = known[korder]
        known_E = np.asarray(A_E, np.float64)[korder]
        if a.mpc_exec > 0:
            ctrl, f_st, f_key, f_en, goal = mpc_flight(st0, known, known_E)
            added = 0
            n_before = len(A_key)
            t_src = A_t[src]
            src_exp, src_tick = A_exp[src], A_tick[src]
            exp_id = len(expansions)
            expansions.append((src, ctrl[0], ctrl[1], ctrl[2], src_exp, src_tick, ctrl[3]))
            if goal > 0:
                goals.append((exp_id, 0, int(goal), int(t_src + goal)))
            for k, (st, key, e) in enumerate(zip(f_st, f_key, f_en)):
                st = st.copy()
                st["tick"], st["stuck_ticks"] = 0, 0
                added += put(key, st, int(t_src + k + 1), exp_id, k + 1, e)
            post_expansion(added, n_before, ex)
            if goals:
                break
            continue
        # CEM over (side 3-way, forward 2-way, view offset) per knot
        p_side = np.full((K, 3), 1.0 / 3.0)
        p_fwd = np.full(K, 0.3)
        mu = np.zeros(K)
        sig = np.full(K, 3.0)
        p_jump = np.full(K, a.jump_p0)
        jump = None
        best = None
        for it in range(a.iters):
            u = rng.random((a.pop, K, 1))
            side = (u > np.cumsum(p_side, 1)[None, :, :]).sum(-1).clip(0, 2).astype(np.int32)
            fwd = np.where(rng.random((a.pop, K)) < p_fwd, 2, 1).astype(np.int32)
            off = np.clip(mu + sig * rng.standard_normal((a.pop, K)), -a.max_offset, a.max_offset)
            if a.jump:
                jump = (rng.random((a.pop, K)) < p_jump).astype(np.int32)
            if best is not None:
                side[0], fwd[0], off[0] = best[:3]
                if a.jump:
                    jump[0] = best[3]
            keys, alive_t, goal_tick, _, en = simulate(st0, side, fwd, off, jump=jump)
            score = novelty(keys, alive_t, known, en, known_E) + energy_term(en, alive_t, e_src)
            score[goal_tick > 0] += 1e6
            el = np.argsort(-score)[:max(8, a.pop // 25)]
            best = (side[el[0]].copy(), fwd[el[0]].copy(), off[el[0]].copy(),
                    jump[el[0]].copy() if a.jump else None)
            if score[el[0]] <= 0:
                continue
            for c in range(3):
                p_side[:, c] = 0.7 * (side[el] == c).mean(0) + 0.3 * p_side[:, c]
            p_side = np.clip(p_side, 0.02, 1.0)
            p_side /= p_side.sum(1, keepdims=True)
            p_fwd = np.clip(0.7 * (fwd[el] == 2).mean(0) + 0.3 * p_fwd, 0.02, 0.98)
            mu = 0.7 * off[el].mean(0) + 0.3 * mu
            sig = np.maximum(0.7 * off[el].std(0) + 0.3 * sig, 0.2)
            if a.jump:
                p_jump = np.clip(0.7 * jump[el].mean(0) + 0.3 * p_jump, 0.02, 0.98)
        # the final population's best --keep candidates go into the archive
        keys, alive_t, goal_tick, states, en = simulate(st0, side, fwd, off, record=True,
                                                        jump=jump)
        score = novelty(keys, alive_t, known, en, known_E) + energy_term(en, alive_t, e_src)
        score[goal_tick > 0] += 1e6
        top = np.argsort(-score)[:a.keep]
        added = 0
        n_before = len(A_key)
        t_src = A_t[src]
        # the source's lineage AS EXPANDED: a later, earlier arrival may replace the cell's state,
        # and this expansion's controls only replay from the state they were optimised from
        src_exp, src_tick = A_exp[src], A_tick[src]
        for rank, c in enumerate(top.tolist()):
            exp_id = len(expansions)
            expansions.append((src, side[c].copy(), fwd[c].copy(), off[c].copy(), src_exp,
                               src_tick, jump[c].copy() if a.jump else np.zeros(K, np.int32)))
            if goal_tick[c] > 0:
                goals.append((exp_id, rank, int(goal_tick[c]), int(t_src + goal_tick[c])))
            ok = np.flatnonzero(alive_t[c])
            for k in ok.tolist():
                st = states[c, k].copy()
                st["tick"], st["stuck_ticks"] = 0, 0
                added += put(int(keys[c, k]), st, int(t_src + k + 1), exp_id, k + 1,
                             float(en[c, k]))
        post_expansion(added, n_before, ex)
        if goals:
            break

    log.close()
    save()
    print(f"wrote {a.out}.npz: {len(A_key):,} cells, {len(expansions):,} expansions, "
          f"{len(goals)} goal hit(s)")


def _save(a, A_key, A_t, A_exp, A_tick, A_state, expansions, goals, d0, bsp, A_E):
    np.savez_compressed(
        f"{a.out}.npz", key=np.asarray(A_key, np.int64), t=np.asarray(A_t, np.int64),
        exp=np.asarray(A_exp, np.int64), tick=np.asarray(A_tick, np.int64),
        origin=np.asarray([s["origin"] for s in A_state]),
        velocity=np.asarray([s["velocity"] for s in A_state]),
        exp_src=np.asarray([e[0] for e in expansions], np.int64),
        exp_side=np.asarray([e[1] for e in expansions], np.int8),
        exp_fwd=np.asarray([e[2] for e in expansions], np.int8),
        exp_off=np.asarray([e[3] for e in expansions], np.float32),
        exp_src_exp=np.asarray([e[4] for e in expansions], np.int64),
        exp_src_tick=np.asarray([e[5] for e in expansions], np.int64),
        exp_jump=np.asarray([e[6] for e in expansions], np.int8),
        goals=np.asarray(goals, np.int64).reshape(-1, 4),
        knot=1 if a.mpc_exec > 0 else a.knot, tick_ms=a.tick_ms,
        horizon=a.horizon, d0=d0, map=str(bsp.stem), energy=np.asarray(A_E, np.float64),
        args=json.dumps(vars(a)))
    np.save(f"{a.out}.states.npy", np.asarray(A_state, STATE_DTYPE))


if __name__ == "__main__":
    main()
