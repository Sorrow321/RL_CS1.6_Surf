#!/usr/bin/env python3
"""line_to_bc_abs.py - a DISCRETE finisher line -> ABSOLUTE-view BC rows.

``tools/line_to_abs.py`` asked whether the absolute-velocity command can
REPLAY the champion line open-loop.  It cannot: one held offset per decision
reproduces the champion's next state to 0.598 u at the median but the
residual accumulates and the transcript dies at 50.7 s of 69.5 whatever the
beam width (600 and 2,000 stop at the same decision).

Behaviour cloning does not need a replayable line.  It needs (state, action)
pairs, and the per-decision fit IS that: at every champion state, the held
absolute offset whose K-tick outcome best reproduces the champion's next
state, fitted from the CHAMPION's state rather than from the transcript's
own drifted one (re-anchored).  A policy trained on those rows is
closed-loop and never has to survive 1,653 blocks of open loop.

    python tools/line_to_bc_abs.py --plan beam_best.npz --ckpt <discrete.pt> \
        --student <absolute.pt> --map C:/RL_Surf/maps/surf_src_cannonball.bsp \
        --out bc.npz --spine spine.npy

``--ckpt`` is the checkpoint the line was SEARCHED with (its act_every and
physics define the replay); ``--student`` is the absolute-view checkpoint
the rows will be cloned into (its config picks the side-channel columns -
``obs_reward``, ``n_latch`` - and its ``view_absolute`` mode).  The two must
agree on every physics key, which is checked.

Also emits, with ``--act-every-scan``, the same one-block fit at other
decision rates: the residual is a WITHIN-BLOCK lock (a held offset makes the
yaw rotate at the velocity heading's own rate after the first tick), so it
should fall as the block shortens.  That is a physics measurement about the
action space and needs no policy.
"""
from __future__ import annotations

import argparse
import json
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
from line_to_abs import base_deg, open_core, say
from surfgym.bc import (decision_gamma, make_eval_feeds, make_line_reward,
                        replay_line, returns_to_go, save_bc_dataset)
from surfgym.core import STATE_DTYPE
from surfgym.tick import TickClock, ticks_to_secs
from surfgym.view import view_mode_code, wrap180, z_from_view_abs

def phys_of(cfg):
    """Every core-physics value ``beam_tas.build_sim`` derives from a config,
    with build_sim's OWN defaults applied - a config that says nothing about
    ``yaw_blend`` builds the same core as one that says 1.0, and comparing
    the raw keys would reject that pair."""
    return {"maxvel": float(cfg.get("maxvel", 2000.0)),
            "yaw_adaptive": 1 if cfg.get("yaw_adaptive") else 0,
            "yaw_blend": float(cfg.get("yaw_blend") or 1.0),
            "side_hold": int(cfg.get("side_hold") or 0),
            "pitch_rate": (0.0 if cfg.get("fix_pitch") is not None
                           else float(cfg.get("pitch_rate", -1.0))),
            "act_every": int(cfg.get("act_every", 1))}


def fit_offsets(ref_states, acts, K, core, phase0, w_vel=0.5, w_yaw=1.0,
                coarse=None, fine_span=0.5, verbose_every=500):
    """The RE-ANCHORED per-decision fit.

    For every decision, teleport every env to the CHAMPION's state at the
    start of the block, sweep held yaw offsets, and keep the one whose
    K-tick outcome lands closest to the champion's state at the end of the
    block.  Returns ``(view (D, 2), err (D,), dead (D,))`` with err the
    residual position error in units."""
    N = int(core.num_envs)
    n = int(coarse or min(N, 511))
    D = len(acts)
    T = len(ref_states)
    pat_n = len(core.tick_pattern)
    view = np.zeros((D, 2), np.float32)
    err = np.full(D, np.inf)
    dead_out = np.zeros(D, bool)
    t0 = time.time()

    def run(cands, pitch_t, a_row, phase):
        m = len(cands)
        core.set_tick_phase(int(phase))
        for i in range(N):
            core.set_state(i, cur)
        v = np.zeros((N, 2), np.float32)
        v[:m, 0] = cands
        v[m:, 0] = cands[0]
        v[:, 1] = pitch_t
        a = np.tile(np.asarray(a_row, np.int32), (N, 1))
        dead = np.zeros(N, bool)
        for _ in range(K):
            _o, _r, dn, tr, _t = core.step(a, view=v)
            dead |= (np.asarray(dn, bool) | np.asarray(tr, bool))
        return core.get_states()[:m].copy(), dead[:m]

    for d in range(D):
        cur = ref_states[min(d * K, T - 1)]
        r = ref_states[min((d + 1) * K, T - 1)]
        phase = (int(phase0) + d * K) % pat_n
        center = float(wrap180(float(r["yaw"]) - base_deg(cur)))
        pitch_t = float(r["pitch"])
        cands = center + np.linspace(-180.0, 180.0, n)
        st, dead = run(np.asarray(cands, np.float32), pitch_t, acts[d], phase)
        e = _score(st, r, w_vel, w_yaw)
        e[dead] = np.inf
        j = int(np.argmin(e))
        c2 = cands[j] + np.linspace(-fine_span, fine_span, n)
        st2, dead2 = run(np.asarray(c2, np.float32), pitch_t, acts[d], phase)
        e2 = _score(st2, r, w_vel, w_yaw)
        e2[dead2] = np.inf
        if e2[int(np.argmin(e2))] <= e[j]:
            j2 = int(np.argmin(e2))
            off, best = float(c2[j2]), st2[j2]
        else:
            off, best = float(cands[j]), st[j]
        view[d] = (off, pitch_t)
        dead_out[d] = not np.isfinite(min(e.min(), e2.min()))
        err[d] = float(np.linalg.norm(
            np.asarray(best["origin"], np.float64)
            - np.asarray(r["origin"], np.float64)))
        if verbose_every and (d % verbose_every == 0 or d == D - 1):
            say(f"  fit {d}/{D}: off {off:+.3f} deg, residual {err[d]:.4f} u "
                f"({time.time() - t0:.0f}s)")
    return view, err, dead_out


def _score(st, r, w_vel, w_yaw):
    dp = st["origin"].astype(np.float64) - np.asarray(r["origin"], np.float64)
    dv = st["velocity"].astype(np.float64) - np.asarray(r["velocity"],
                                                        np.float64)
    dy = wrap180(st["yaw"].astype(np.float64) - float(r["yaw"]))
    return (dp ** 2).sum(1) + w_vel * (dv ** 2).sum(1) + w_yaw * dy ** 2


def describe(err, tag):
    q = np.percentile(err, [50, 90, 99])
    say(f"{tag}: residual p50 {q[0]:.4f} p90 {q[1]:.4f} p99 {q[2]:.4f} "
        f"max {err.max():.4f} u; over 1.0 u {100 * (err > 1.0).mean():.2f}% ; "
        f"over 0.1 u {100 * (err > 0.1).mean():.2f}%")
    return {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(err.max()), "over_1u": float((err > 1.0).mean()),
            "over_0p1u": float((err > 0.1).mean())}


def main():
    ap = argparse.ArgumentParser(
        description="discrete finisher line -> absolute-view BC rows")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--ckpt", required=True, help="the line's own checkpoint")
    ap.add_argument("--student", required=True,
                    help="the ABSOLUTE-view checkpoint the rows clone into")
    ap.add_argument("--map", default=None)
    ap.add_argument("--out", default=None, help="BC dataset .npz")
    ap.add_argument("--spine", default=None, help="demo spine .npy")
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--line", type=int, default=0)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--w-vel", type=float, default=0.5)
    ap.add_argument("--w-yaw", type=float, default=1.0)
    ap.add_argument("--no-value-target", action="store_true")
    ap.add_argument("--max-residual", type=float, default=10.0,
                    help="drop rows whose one-block fit residual exceeds "
                         "this (u); the block that crosses the goal box ends "
                         "the episode and cannot be fitted")
    ap.add_argument("--act-every-scan", default="",
                    help="comma-separated act_every values to re-fit at "
                         "(measurement only, writes no dataset)")
    a = ap.parse_args()

    plans = p2b.load_plans([a.plan])
    if plans["view_continuous"]:
        raise SystemExit("the plan is already continuous")
    cfg = (torch.load(a.ckpt, map_location="cpu",
                      weights_only=False).get("config") or {})
    stu = (torch.load(a.student, map_location="cpu",
                      weights_only=False).get("config") or {})
    mode = stu.get("view_absolute") or None
    if not stu.get("view_continuous") or mode is None:
        raise SystemExit(f"{a.student} is not a --view-absolute checkpoint")
    pt, ps = phys_of(cfg), phys_of(stu)
    bad = [k for k in pt if pt[k] != ps[k]]
    if bad:
        raise SystemExit("teacher and student build different cores: "
                         + ", ".join(f"{k}: {pt[k]!r} vs {ps[k]!r}"
                                     for k in bad))
    K = int(plans["K"])
    tick = TickClock(float(plans["tick_ms_requested"]))
    map_path = beam_tas.resolve_map(a.map or plans["map"], cfg.get("map"))
    line = plans["lines"][int(a.line)]
    acts = np.asarray(line["acts"], np.int32)
    D = len(acts)
    ft = int(line["finish_tick"])
    ep_cap = max(int(ft), D * K) + 4 * K
    say(f"line {a.line}: {D} decisions, finish_tick {ft} "
        f"({ticks_to_secs(ft, tick.ms, tick.pattern):.3f} s spawn clock), "
        f"tick {tick.describe()}")
    say(f"student {a.student}: view_absolute {mode}, obs_reward "
        f"{bool(stu.get('obs_reward'))}, race_latch {stu.get('race_latch')}")

    # ---- the reference, replayed under the STUDENT's side channels --------
    core1, gf, d0, _z = p2b.open_planner_core(stu, map_path, ep_cap, tick=tick)
    slot, rf, lf = make_eval_feeds(stu, gf, d0, K, tick_ms=tick.requested_ms)
    n_latch = 0 if lf is None else 1
    obs_reward = rf is not None
    rfn = rinfo = None
    rew_dec = []
    if not a.no_value_target:
        rfn, rline_info = make_line_reward(stu, gf, d0, K,
                                           tick_ms=tick.requested_ms)
        rinfo = {}
    core1.reset(int(plans["gate_seed"]))
    rows, ref_states, finished, ticks = replay_line(
        core1, plans["spawn_state"], plans["obs_start"], acts, K,
        reward_feed=rf, latch_fn=lf, max_ticks=ep_cap, keep_final=True,
        reward_fn=rfn, rewards_out=(rew_dec if rfn is not None else None),
        info_out=rinfo)
    ref_states = np.asarray(ref_states)
    say(f"reference replay under the student's feeds: finished {finished}, "
        f"{ticks} ticks, {len(rows)} decision rows")
    if not finished:
        raise SystemExit("the line does not finish under the student's core")

    # ---- the fit ----------------------------------------------------------
    cfg_abs = dict(stu)
    core, _zn = open_core(cfg_abs, map_path, int(a.envs), ep_cap, tick)
    core.reset(int(plans["gate_seed"]))
    phase0 = int(getattr(core, "tick_phase", 0))
    say(f"absolute core: view_mode {int(core.config.view_mode)}, "
        f"{a.envs} envs")
    view, err, dead = fit_offsets(ref_states, acts, K, core, phase0,
                                  w_vel=a.w_vel, w_yaw=a.w_yaw)
    keep = err <= float(a.max_residual)
    say(f"fit: {int((~keep).sum())} of {len(err)} decisions over the "
        f"{a.max_residual:g} u residual cap (the goal-crossing block cannot "
        f"be fitted) - dropped from the dataset")
    stats = describe(err[keep], f"act_every {K} fit")
    summary = {"act_every": {str(K): stats}, "line": str(a.plan),
               "student": str(a.student), "decisions": D,
               "finish_ticks": int(ticks)}

    # ---- the act_every scan (measurement only) ----------------------------
    for s in [int(x) for x in a.act_every_scan.split(",") if x.strip()]:
        if s == K:
            continue
        if K % s:
            say(f"skip act_every {s}: {K} is not a multiple of it")
            continue
        rep = K // s
        acts_s = np.repeat(acts, rep, axis=0)
        v_s, e_s, _d = fit_offsets(ref_states, acts_s, s, core, phase0,
                                   w_vel=a.w_vel, w_yaw=a.w_yaw,
                                   verbose_every=0)
        summary["act_every"][str(s)] = describe(
            e_s[e_s <= float(a.max_residual)], f"act_every {s} fit")

    if a.summary_out:
        Path(a.summary_out).write_text(json.dumps(summary, indent=1),
                                       encoding="utf-8")
    if not a.out:
        say("no --out: measurement only, no dataset written")
        return

    # ---- the BC rows ------------------------------------------------------
    n0 = len(rows)
    if n0 > D:
        raise SystemExit(f"{n0} decision rows for {D} decisions")
    sel = np.nonzero(keep[:n0])[0]
    n = len(sel)
    states = np.array([rows[i][0] for i in sel], dtype=STATE_DTYPE)
    scal = np.array([rows[i][1] for i in sel], np.float32)
    latch = np.array([rows[i][2] for i in sel], np.float32)
    act_arr = np.array([rows[i][3] for i in sel], np.int64)
    vw = view[sel]
    vmu = z_from_view_abs(vw, mode).astype(np.float32)
    vsd = np.zeros_like(vmu)
    gamma_dec = decision_gamma(float(stu.get("gamma") or 0.9995), K,
                               tick.requested_ms)
    zret = zmask = None
    if rfn is not None:
        z_line = (returns_to_go(rew_dec, gamma_dec) if len(rew_dec)
                  else np.zeros(0, np.float32))
        zret = np.array([float(z_line[i]) if i < len(z_line) else 0.0
                         for i in sel], np.float32)
        zmask = np.full(n, 1.0 if bool(rinfo.get("terminal")) else 0.0,
                        np.float32)
    meta = {"plan": plans["files"], "ckpt": str(a.student),
            "teacher_ckpt": str(a.ckpt), "map": map_path,
            "act_every": K, "obs_reward": bool(obs_reward),
            "n_latch": int(n_latch),
            "d_latch": (0.0 if lf is None else float(lf.d_latch)),
            "d0": d0, "objective": "finish", "lines": 1,
            "line_ticks": [int(ticks)], "line_rows": [n],
            "line_weights": [1.0], "line_finished": [True],
            "finishers": 1, "best_ticks": int(ticks),
            "best_s": float(ticks_to_secs(ticks, tick.ms, tick.pattern)),
            "tick_ms": float(tick.ms),
            "tick_pattern_ms": [int(v) for v in tick.pattern],
            "tick_ms_requested": float(tick.requested_ms),
            "gate_seed": int(plans["gate_seed"]),
            "view_continuous": 1, "view_absolute": mode,
            "view_mode": int(view_mode_code(mode)),
            "view_target": "point", "target_kind": "argmax",
            "value_target": bool(rfn is not None),
            "source": "line_to_bc_abs re-anchored per-decision fit",
            "fit": stats,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_bc_dataset(a.out, states, scal, latch, act_arr,
                    np.ones(n, np.float32), np.zeros(n, np.int32), meta,
                    probs=None, zret=zret, zmask=zmask,
                    view=vw, view_zmu=vmu, view_zsd=vsd)
    say(f"bc: {n:,} rows -> {a.out} (obs_reward {obs_reward}, n_latch "
        f"{n_latch}, value target {rfn is not None})")
    if a.spine:
        np.save(a.spine, np.ascontiguousarray(ref_states))
        say(f"spine: {len(ref_states):,} states -> {a.spine}")


if __name__ == "__main__":
    main()
