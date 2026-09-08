#!/usr/bin/env python3
"""line_to_abs.py - a DISCRETE planner line -> an ABSOLUTE-view line.

``--view-absolute velocity`` (docs/contyaw.md) reads the policy's view row
as a (yaw offset from the velocity heading, pitch target) pair that the core
re-derives from the LIVE state every tick.  A discrete line's yaw bins are
per-tick DELTAS, so ``beam_tas --prefix-line``, ``plan_to_bc`` and
``BCDataset`` all refuse to read one under an absolute mode - correctly: the
two rows mean different things and no static table converts between them.

They can, however, be converted through the PHYSICS, which is what this
does.  The discrete line is replayed once to get its per-tick states (the
reference trajectory).  Then, decision by decision, the absolute-mode core
is stepped from the transcript's OWN current state over a grid of candidate
yaw offsets and the offset whose K-tick outcome lands closest to the
reference state is committed.  That is a closed-loop transcription: drift is
corrected at every decision rather than accumulated, and the search is over
the one scalar the mode adds.

The measurement that says this can work at all (surf_src_cannonball, the
68.54 s line): the champion's own yaw sits on its velocity heading - the
offset is 0.02 deg at the median and -6.1 / +4.2 deg at p1 / p99 - i.e. the
line lives where ``off_warp`` has its FINEST resolution (3.5 deg per unit u
at zero), not on the steep part the delta warp puts the strafe optimum on.

    python tools/line_to_abs.py --plan beam_best.npz --ckpt <discrete.pt> \
        --map C:/RL_Surf/maps/surf_src_cannonball.bsp --out abs_best.npz

The output is a beam_best-shaped npz with ``view_continuous``,
``view_mode 1`` and a (D, 2) ``view``, which plan_to_bc / beam_tas
--prefix-line read as any searched absolute line.  It carries the SAME
non-view action columns as the source line; only the yaw/pitch command is
rewritten.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch

import beam_tas
import plan_to_bc as p2b
from surfgym.bc import make_eval_feeds, replay_line
from surfgym.core import STATE_DTYPE
from surfgym.tick import TickClock
from surfgym.view import wrap180


def say(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def heading_deg(vel):
    """The core's velocity-frame base: atan2(vy, vx) in deg above 100 u/s."""
    vel = np.asarray(vel, np.float64).reshape(-1, 3)
    return np.degrees(np.arctan2(vel[:, 1], vel[:, 0]))


def base_deg(state):
    """``base`` for ONE state row: the heading above 100 u/s, else the yaw."""
    v = np.asarray(state["velocity"], np.float64).reshape(3)
    vh = float(np.hypot(v[0], v[1]))
    return float(heading_deg(v)[0]) if vh >= 100.0 else float(state["yaw"])


def open_core(cfg, map_path, n_envs, ep_cap, tick):
    """beam_tas' armed core at n envs (plan_to_bc.open_planner_core is 1)."""
    from surfgym.zones import load_zones
    core = beam_tas.build_sim(cfg, map_path, n_envs, ep_cap, tick=tick)
    zones = load_zones(core.bsp_path)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)
    return core, zones


#: the offset grid around each anchor, degrees.  Fine where the champion
#: lives (its offset is 0.02 deg at the median) and coarse enough to reach
#: the +-40 deg bang-bang the line uses below 100 u/s.
FINE = np.array([0.0, 0.03, -0.03, 0.1, -0.1, 0.3, -0.3, 1.0, -1.0,
                 3.0, -3.0, 10.0, -10.0])
COARSE = np.array([30.0, -30.0, 60.0, -60.0, 120.0, -120.0, 180.0])


def beam_transcribe(ref_states, acts, K, core, spawn_state, phase0,
                    beam=45, w_vel=0.5, w_yaw=1.0, dedup_pos=0.5,
                    dedup_vel=1.0, max_children=4, verbose_every=250):
    """Beam version: keep ``beam`` candidate transcripts, expand each over
    the offset grid, score against the reference state at the end of the
    block, prune.  A greedy fit accumulates the ~0.6 u per block the mode
    cannot express (measured) until the line is lost; a beam keeps the
    alternatives that pay off two blocks later."""
    D = len(acts)
    T = len(ref_states)
    E = 3 * len(FINE) + len(COARSE)
    B = int(beam)
    N = B * E
    if N > core.num_envs:
        raise SystemExit(f"beam {B} x {E} candidates = {N} envs > core's "
                         f"{core.num_envs}; raise --envs")
    pat_n = len(core.tick_pattern)
    st_b = np.repeat(np.asarray(spawn_state, STATE_DTYPE).reshape(-1)[:1], B)
    hist = np.zeros((B, D), np.float32)
    pit = np.zeros(D, np.float32)
    alive = 1                      # only entry 0 is distinct at the spawn
    phase = int(phase0)
    t0 = time.time()
    for d in range(D):
        t_end = min((d + 1) * K, T - 1)
        ref = ref_states[t_end]
        pitch_t = float(ref["pitch"])
        pit[d] = pitch_t
        cands = np.zeros((B, E), np.float64)
        for b in range(B):
            c = float(wrap180(float(ref["yaw"]) - base_deg(st_b[b])))
            prev = float(hist[b, d - 1]) if d else 0.0
            cands[b] = np.concatenate([c + FINE, FINE, prev + FINE, c + COARSE])
        core.set_tick_phase(phase)
        for b in range(B):
            for e in range(E):
                core.set_state(b * E + e, st_b[b])
        v = np.zeros((core.num_envs, 2), np.float32)
        v[:N, 0] = cands.reshape(-1)
        v[:, 1] = pitch_t
        a = np.tile(np.asarray(acts[d], np.int32), (core.num_envs, 1))
        dead = np.zeros(core.num_envs, bool)
        hit = np.zeros(core.num_envs, bool)
        for _ in range(K):
            _o, _r, done, trunc, _t = core.step(a, view=v)
            hit |= np.asarray(core.goal_hits, bool)
            dead |= (np.asarray(done, bool) | np.asarray(trunc, bool))
        st = core.get_states()[:N].copy()
        dp = st["origin"].astype(np.float64) - np.asarray(ref["origin"], np.float64)
        dv = st["velocity"].astype(np.float64) - np.asarray(ref["velocity"], np.float64)
        dy = wrap180(st["yaw"].astype(np.float64) - float(ref["yaw"]))
        sc = (dp ** 2).sum(1) + w_vel * (dv ** 2).sum(1) + w_yaw * dy ** 2
        sc[dead[:N]] = np.inf
        sc[hit[:N]] = -1.0
        sc[(np.arange(N) // E) >= alive] = np.inf       # unfilled beam slots
        order = np.argsort(sc, kind="stable")
        keep, seen, per_parent = [], set(), {}
        for j in order:
            if not np.isfinite(sc[j]):
                break
            p = int(j) // E
            if per_parent.get(p, 0) >= max_children:
                continue
            key = (tuple(np.round(st[j]["origin"] / dedup_pos).astype(np.int64)),
                   tuple(np.round(st[j]["velocity"] / dedup_vel).astype(np.int64)))
            if key in seen:
                continue
            seen.add(key)
            per_parent[p] = per_parent.get(p, 0) + 1
            keep.append(int(j))
            if len(keep) >= B:
                break
        if not keep:
            raise SystemExit(f"the beam died at decision {d} of {D} "
                             f"(every candidate ended the episode)")
        nh = np.zeros_like(hist)
        for i, j in enumerate(keep):
            p = j // E
            nh[i, :d] = hist[p, :d]
            nh[i, d] = cands[p, j % E]
            st_b[i] = st[j]
        for i in range(len(keep), B):
            nh[i] = nh[0]
            st_b[i] = st_b[0]
        hist = nh
        alive = len(keep)
        phase = (phase + K) % pat_n
        if verbose_every and (d % verbose_every == 0 or d == D - 1):
            e0 = np.linalg.norm(np.asarray(st_b[0]["origin"], np.float64)
                                - np.asarray(ref["origin"], np.float64))
            say(f"  decision {d}/{D}: beam {alive}, best off "
                f"{hist[0, d]:+.3f} deg, |dpos| {e0:.3f} u, "
                f"score {sc[keep[0]]:.4g} ({time.time() - t0:.0f}s)")
        if float(sc[keep[0]]) < 0.0:
            say(f"  GOAL crossed at decision {d} of {D}")
            hist = hist[:, :d + 1]
            pit = pit[:d + 1]
            break
    view = np.stack([hist[0], pit[:hist.shape[1]]], 1).astype(np.float32)
    return view, hist.shape[1]


def transcribe(ref_states, acts, K, core, n_envs, spawn_state, phase0,
               coarse_span=30.0, coarse_n=None, fine_span=0.4, fine_n=None,
               w_vel=0.01, w_yaw=0.1, tail=1, verbose_every=250):
    """Greedy per-decision fit of the absolute yaw offset.

    Returns ``(view (D, 2) float32, err per decision, n_committed)``."""
    coarse_n = int(coarse_n or min(n_envs, 241))
    fine_n = int(fine_n or min(n_envs, 161))
    D = len(acts)
    T = len(ref_states)
    view = np.zeros((D, 2), np.float32)
    errs = np.full(D, np.nan)
    cur = np.asarray(spawn_state, STATE_DTYPE).reshape(-1)[0].copy()
    phase = int(phase0)
    pat_n = len(core.tick_pattern)
    stop = max(0, D - int(tail))

    def run(cands, pitch_t, a_row):
        n = len(cands)
        core.set_tick_phase(phase)
        for i in range(n_envs):
            core.set_state(i, cur)
        a = np.tile(np.asarray(a_row, np.int32), (n_envs, 1))
        v = np.zeros((n_envs, 2), np.float32)
        v[:n, 0] = cands
        v[n:, 0] = cands[0]
        v[:, 1] = pitch_t
        dead = np.zeros(n_envs, bool)
        for _ in range(K):
            _obs, _r, done, trunc, _t = core.step(a, view=v)
            dead |= (np.asarray(done, bool) | np.asarray(trunc, bool))
        return core.get_states()[:n].copy(), dead[:n]

    def score(st, ref):
        dp = st["origin"].astype(np.float64) - np.asarray(ref["origin"], np.float64)
        dv = st["velocity"].astype(np.float64) - np.asarray(ref["velocity"], np.float64)
        dy = wrap180(st["yaw"].astype(np.float64) - float(ref["yaw"]))
        return (dp ** 2).sum(1) + w_vel * (dv ** 2).sum(1) + w_yaw * dy ** 2

    t0 = time.time()
    for d in range(stop):
        t_end = min((d + 1) * K, T - 1)
        ref = ref_states[t_end]
        center = float(wrap180(float(ref["yaw"]) - base_deg(cur)))
        pitch_t = float(ref["pitch"])
        cands = center + np.linspace(-coarse_span, coarse_span, coarse_n)
        st, dead = run(np.asarray(cands, np.float32), pitch_t, acts[d])
        e = score(st, ref)
        e[dead] = np.inf
        j = int(np.argmin(e))
        c2 = cands[j] + np.linspace(-fine_span, fine_span, fine_n)
        st2, dead2 = run(np.asarray(c2, np.float32), pitch_t, acts[d])
        e2 = score(st2, ref)
        e2[dead2] = np.inf
        j2 = int(np.argmin(e2))
        if e2[j2] <= e[j]:
            off, st_b, err = float(c2[j2]), st2[j2], float(e2[j2])
        else:
            off, st_b, err = float(cands[j]), st[j], float(e[j])
        view[d] = (off, pitch_t)
        errs[d] = err
        cur = st_b.copy()
        phase = (phase + K) % pat_n
        if verbose_every and (d % verbose_every == 0 or d == stop - 1):
            dp = np.linalg.norm(np.asarray(st_b["origin"], np.float64)
                                - np.asarray(ref["origin"], np.float64))
            say(f"  decision {d}/{stop}: off {off:+.3f} deg, |dpos| {dp:.3f} u "
                f"({time.time() - t0:.0f}s)")
    # the tail decisions: the reference's own offset, no search (the block
    # that crosses the goal box ends the episode and cannot be scored)
    for d in range(stop, D):
        t_end = min((d + 1) * K, T - 1)
        ref = ref_states[t_end]
        view[d] = (float(wrap180(float(ref["yaw"])
                                 - base_deg(ref_states[min(d * K, T - 1)]))),
                   float(ref["pitch"]))
    return view, errs, stop


def main():
    ap = argparse.ArgumentParser(
        description="discrete planner line -> absolute-view line")
    ap.add_argument("--plan", required=True, help="beam_best.npz (discrete)")
    ap.add_argument("--ckpt", required=True,
                    help="the checkpoint it was searched with")
    ap.add_argument("--map", default=None)
    ap.add_argument("--out", required=True, help="output beam_best-shaped npz")
    ap.add_argument("--line", type=int, default=0,
                    help="which kept lineage (0 = fastest)")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--mode", default="velocity", choices=["velocity"])
    ap.add_argument("--coarse-span", type=float, default=30.0)
    ap.add_argument("--fine-span", type=float, default=0.4)
    ap.add_argument("--w-vel", type=float, default=0.01)
    ap.add_argument("--w-yaw", type=float, default=0.1)
    ap.add_argument("--tail", type=int, default=1)
    ap.add_argument("--max-children", type=int, default=4,
                    help="cap on a beam entry's descendants per decision")
    ap.add_argument("--beam", type=int, default=0,
                    help="beam width (0 = the greedy one-candidate fit)")
    a = ap.parse_args()

    plans = p2b.load_plans([a.plan])
    if plans["view_continuous"]:
        raise SystemExit("the plan is already continuous - nothing to transcribe")
    cfg = (torch.load(a.ckpt, map_location="cpu",
                      weights_only=False).get("config") or {})
    K = int(plans["K"])
    if K != int(cfg.get("act_every", 1)):
        raise SystemExit(f"plan act_every {K} != ckpt {cfg.get('act_every')}")
    tick = TickClock(float(plans["tick_ms_requested"]))
    if list(tick.pattern) != list(plans["tick_pattern_ms"]):
        raise SystemExit("tick pattern mismatch - different physics")
    map_path = beam_tas.resolve_map(a.map or plans["map"], cfg.get("map"))
    line = plans["lines"][int(a.line)]
    acts = np.asarray(line["acts"], np.int32)
    D = len(acts)
    ft = int(line["finish_tick"])
    say(f"plan {a.plan}: line {a.line}, {D} decisions, finish_tick {ft}, "
        f"tick {tick.describe()}")
    ep_cap = max(int(ft), D * K) + 4 * K

    # ---- 1. the reference: the discrete line replayed, per-tick states ----
    dcore, gf, d0, _z = p2b.open_planner_core(cfg, map_path, ep_cap, tick=tick)
    _slot, rf, lf = make_eval_feeds(cfg, gf, d0, K, tick_ms=tick.requested_ms)
    dcore.reset(int(plans["gate_seed"]))
    _rows, ref_states, finished, ticks = replay_line(
        dcore, plans["spawn_state"], plans["obs_start"], acts, K,
        reward_feed=rf, latch_fn=lf, keep_final=True)
    ref_states = np.asarray(ref_states)
    say(f"reference replay: finished {finished}, {ticks} ticks, "
        f"{len(ref_states)} states")
    if not finished:
        raise SystemExit("the source line does not finish in this build - stop")

    # ---- 2. the transcript ------------------------------------------------
    cfg_abs = dict(cfg)
    cfg_abs["view_absolute"] = a.mode
    cfg_abs["view_continuous"] = 1
    core, _zones = open_core(cfg_abs, map_path, int(a.envs), ep_cap, tick)
    say(f"absolute core: view_mode {int(core.config.view_mode)}, "
        f"{a.envs} envs, pattern {core.tick_pattern}")
    core.reset(int(plans["gate_seed"]))
    phase0 = int(getattr(core, "tick_phase", 0))
    if a.beam > 0:
        view, nd = beam_transcribe(ref_states, acts, K, core,
                                   plans["spawn_state"], phase0,
                                   beam=int(a.beam), w_vel=a.w_vel,
                                   w_yaw=a.w_yaw,
                                   max_children=int(a.max_children))
        acts = acts[:nd]
        D = nd
        say(f"transcript (beam {a.beam}): {D} decisions, |offset| p50 "
            f"{np.percentile(np.abs(view[:, 0]), 50):.3f} p99 "
            f"{np.percentile(np.abs(view[:, 0]), 99):.3f} max "
            f"{np.abs(view[:, 0]).max():.3f} deg")
    else:
        view, errs, stop = transcribe(ref_states, acts, K, core, int(a.envs),
                                      plans["spawn_state"], phase0,
                                      coarse_span=a.coarse_span,
                                      fine_span=a.fine_span, w_vel=a.w_vel,
                                      w_yaw=a.w_yaw, tail=a.tail)
        fin = np.isfinite(errs[:stop])
        say(f"transcript: |offset| p50 "
            f"{np.percentile(np.abs(view[:, 0]), 50):.3f} "
            f"p99 {np.percentile(np.abs(view[:, 0]), 99):.3f} max "
            f"{np.abs(view[:, 0]).max():.3f} deg; block error p50 "
            f"{np.percentile(errs[:stop][fin], 50):.3g} p99 "
            f"{np.percentile(errs[:stop][fin], 99):.3g}")

    # ---- 3. verify: open-loop replay of the transcript --------------------
    vcore, vg, vd0, _z2 = p2b.open_planner_core(cfg_abs, map_path, ep_cap,
                                                tick=tick)
    _s2, rf2, lf2 = make_eval_feeds(cfg_abs, vg, vd0, K,
                                    tick_ms=tick.requested_ms)
    vcore.reset(int(plans["gate_seed"]))
    _r2, vstates, vfin, vticks = replay_line(
        vcore, plans["spawn_state"], plans["obs_start"], acts, K,
        reward_feed=rf2, latch_fn=lf2, view=view, keep_final=True)
    say(f"VERIFY (open-loop, view_mode {int(vcore.config.view_mode)}): "
        f"finished {vfin}, {vticks} ticks vs the source's {ticks}")
    if vfin:
        n = min(len(vstates), len(ref_states))
        dp = np.linalg.norm(
            np.asarray(vstates)[:n]["origin"].astype(np.float64)
            - ref_states[:n]["origin"].astype(np.float64), axis=1)
        say(f"   |dpos| vs the source line: p50 {np.percentile(dp, 50):.3f} "
            f"p99 {np.percentile(dp, 99):.3f} max {dp.max():.3f} u")

    # ---- 4. save ----------------------------------------------------------
    out = {
        "acts": acts.astype(np.int8),
        "view": view.astype(np.float32),
        "act_every": np.int32(K),
        "finish_ticks": np.int32(vticks if vfin else 0),
        "spawn_state": np.asarray(plans["spawn_state"],
                                  STATE_DTYPE).reshape(-1)[:1],
        "obs_start": np.asarray(plans["obs_start"], np.float32),
        "gate_seed": np.int32(plans["gate_seed"]),
        "acts_all": acts.astype(np.int8)[None],
        "view_all": view.astype(np.float32)[None],
        "acts_len": np.asarray([D], np.int32),
        "finish_ticks_all": np.asarray([vticks if vfin else 0], np.int32),
        "end_tick_all": np.asarray([vticks], np.int32),
        "greedy_ticks": np.int32(0),
        "greedy_prefix": np.int32(0),
        "seed": np.int32(0), "torch_seed": np.int32(0),
        "eps": np.float32(0.0), "commit": np.int32(0),
        "ckpt": np.str_(str(a.ckpt)), "map": np.str_(str(map_path)),
        "objective": np.str_("finish"),
        "view_continuous": np.int32(1),
        "view_mode": np.int32(1 if a.mode == "velocity" else 2),
        "view_absolute": np.str_(a.mode),
        "transcribed_from": np.str_(str(a.plan)),
        "source_finish_ticks": np.int32(ticks),
    }
    out.update(beam_tas.tick_npz(tick, cfg.get("tick_ms")))
    np.savez(a.out, **out)
    say(f"wrote {a.out}")
    if not vfin:
        raise SystemExit("the transcript does NOT finish - report that, do "
                         "not seed a loop with it")


if __name__ == "__main__":
    main()
