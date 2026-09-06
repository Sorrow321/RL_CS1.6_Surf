"""wr_scan.py - the world record's line through OUR policy's eyes and through
OUR action space (diagnostic only: the record is a comparison, never a
training input).

Three measurements from one pass over the record's frames
(runs/wr_demo/<stem>.frames.npz from tools/demo/parse_hldemo.py, which
carries the usercmd stream: view angles, forward/side move, buttons, and
the 8/8/7 ms cadence our core also runs):

1. ACTION-SPACE FEASIBILITY, per decision. From the record's own state at
   every 4-tick decision boundary, our core is stepped 4 ticks with the
   record's input converted into our action grid, and the state it reaches
   is compared with the record's state 4 ticks later. Three conversions,
   from coarse to exact:
     quant  the record's 4 ticks collapsed into ONE held decision (yaw bin
            nearest to the 4-tick turn, majority side/forward key) - what
            the policy could actually emit;
     tick   the nearest yaw / pitch bin and the exact keys per TICK - the
            grid without the 4-tick hold;
     phys   the record's view angles written straight into the state each
            tick, keys exact - no grid at all, only our physics.
   Where `phys` is small the simulator reproduces the record's motion;
   where `tick` is small the bins can express it; where `quant` is small a
   4-tick hold can express it. The record also DUCKS in 12 short segments
   (buttons decoded from the demo; duck is inert in our core), so the
   segments are listed and the errors inside them are flagged.

2. POLICY LIKELIHOOD along the record. The observation at each decision
   (depth render + scalars + the reward and latch side channels, assembled
   exactly as an eval assembles them) is fed to the policy and the
   log-probability of the record's next decision (its quantised action) is
   read off per head, with the policy's argmax, the rank of the record's
   yaw bin and the head entropies. Summed over decisions this is the
   log-likelihood of the record's line under the policy, teacher-forced on
   the record's states.

3. VALUE along the record versus along our own lines: V(s) at the record's
   states, at the states of our searched line (open-loop replay of a
   beam_best.npz) and along the policy's own greedy episode, matched by
   route arc, so "is the one-touch finish a region the critic thinks is
   bad, or merely one it never visits" can be read directly.

Outputs (--out): wr_scan.csv (per record decision), line_scan.csv,
greedy_scan.csv, summary.md, wr_scan.png.

Usage:
    python tools/demo/wr_scan.py --ckpt CKPT --wr runs/wr_demo/wr_cannonball.frames.npz \
        --line runs/research/tas_68.54/beam_best.npz --out runs/research/wr_scan
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
import torch.nn.functional as F

from beam_tas import build_sim, resolve_map
from surfgym.bc import make_eval_feeds
from surfgym.core import STATE_DTYPE, SURF_IN_JUMP, PITCH_BINS
from surfgym.goalfield import EuclidField, build_goal_field
from surfgym.mapfleet import map_tag
from surfgym.rewards import map_spawn_pool
from surfgym.route import ArcProgress
from surfgym.tick import TickClock
from surfgym.vision import GpuLidar, pick_cell
from surfgym.zones import load_zones
from train_fast import GreedyTorchPolicy, HeadPacker, Policy, NVEC

# src/env.c K_BINS: a yaw bin under --yaw-adaptive is this multiple of the
# analytic optimal-strafe angle atan(30/|v_h|) per tick, clamped to
# cfg.yaw_rate_max_deg
K_BINS = np.array([-20.0, -8.0, -3.0, -1.5, -1.0, -0.75, -0.5, 0.0,
                   0.5, 0.75, 1.0, 1.5, 3.0, 8.0, 20.0], np.float64)
YAW_BINS_FIXED = np.array([-10.0, -7.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0,
                           0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0], np.float64)
HEADS = ["yaw", "pitch", "fwd", "side", "jump", "duck"]
NEUTRAL = np.array([7, 3, 1, 1, 0, 0], np.int32)
WR_CURTAIN_S = 1.81      # playback seconds before the record's start curtain (compare_wr)
WR_RAMP1 = (61.30, 63.87)   # the record's ramp-1 phase, record clock (compare_wr)


# --------------------------------------------------------------------------
# setup, copied from beam_tas.main so the observation is the planner's
# --------------------------------------------------------------------------
def setup(ckpt_path, map_arg, n_envs, ep_cap=30000):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    if cfg.get("reward") != "race":
        raise SystemExit("needs a race checkpoint")
    for bad in ("route_file", "chunk", "frame_stack", "race_arc"):
        if cfg.get(bad):
            raise SystemExit(f"checkpoint uses {bad}, not supported here")
    TICK = TickClock(float(cfg.get("tick_ms") or 10.0))
    K = int(cfg.get("act_every", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    map_path = resolve_map(map_arg, cfg.get("map", "surf_ski_2"))
    core1 = build_sim(cfg, map_path, 1, ep_cap, tick=TICK)
    coreN = build_sim(cfg, map_path, n_envs, ep_cap, tick=TICK)
    cell = float((cfg.get("map_cells") or {}).get(
        map_tag(Path(map_path).stem), cfg.get("lidar_cell") or pick_cell(core1)))
    gcell = cell
    gcells, gc = cfg.get("goal_cells"), cfg.get("goal_cell")
    if isinstance(gcells, dict) and gcells:
        gcell = float(gcells.get(map_tag(Path(map_path).stem), gcell))
    elif isinstance(gc, str) and "," in gc:
        parts = [x.strip() for x in gc.split(",")]
        names = cfg.get("maps") or []
        tag = map_tag(Path(map_path).stem)
        i = next((j for j, m in enumerate(names) if map_tag(m) == tag), None)
        if i is not None and i < len(parts) and parts[i]:
            gcell = float(parts[i])
    elif gc:
        gcell = float(gc)
    zones = load_zones(core1.bsp_path)
    t0 = time.time()
    gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
          else build_goal_field(core1, zones["end"], cell=gcell))
    dt = time.time() - t0
    if dt > 30:
        print(f"** goal field took {dt:.0f}s: that smells like a RE-BAKE (map path / mtime?) **")
    raw = map_spawn_pool(core1)
    pool = map_spawn_pool(core1, yaw=gf.descent_yaw(raw["origin"]))
    pool["pitch"] = -10.0
    if cfg.get("fix_pitch") is not None:
        pool["pitch"] = float(cfg["fix_pitch"])
    d0 = float(np.mean(gf.sample(raw["origin"])))
    for core in (core1, coreN):
        core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        core.set_teleport_fail(True)
        core.set_spawn_pool(pool)
    lidar = GpuLidar(core1, lw, lh, range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"), cell=cell, device=device,
                     surf_mask=bool(cfg.get("surf_mask", 0)), pinhole=bool(cfg.get("pinhole", 0)))
    extra = (12,) if cfg.get("obs_reward") else ()
    n_latch = 1 if (float(cfg.get("race_latch") or 0.0) > 0.0
                    or float(cfg.get("race_latch_frac") or 0.0) > 0.0) else 0
    policy = Policy(core1.obs_dim + n_latch + lw * lh * lidar.channels, lw, lh,
                    emb=int(cfg.get("emb", 256)), hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)), trunk=str(cfg.get("trunk") or "plain"),
                    extra_feat=extra, in_ch=lidar.channels, n_codes=0, chunk=0,
                    route_dim=n_latch, route_critic_only=bool(cfg.get("route_critic_only"))
                    ).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    packer = HeadPacker(device)

    def feeds():
        slot, rf, lf = make_eval_feeds(cfg, gf, d0, K, tick_ms=TICK.requested_ms)
        return slot, rf, lf

    return dict(cfg=cfg, TICK=TICK, K=K, device=device, core1=core1, coreN=coreN,
                gf=gf, d0=d0, zones=zones, lidar=lidar, policy=policy, packer=packer,
                feeds=feeds, map_path=map_path, step=int(ck.get("global_step", 0)))


def wrapper(S, core):
    slot, rf, lf = S["feeds"]()
    w = GreedyTorchPolicy(S["policy"], S["packer"], S["device"], S["lidar"], core,
                          S["K"], 1, extra_slot=slot, extra_fn=rf, latch_fn=lf)
    return w, rf, lf


@torch.inference_mode()
def policy_eval(S, w, obs):
    """(padded logits (N,6,15), log-softmax, value (N,)) for the core's current states."""
    full = w._obs(obs)
    logits, value = S["policy"](full)
    padded = S["packer"].pad(logits.float())
    lsm = F.log_softmax(padded, dim=-1)
    return padded, lsm, value.detach().float().reshape(-1).cpu().numpy()


def head_stats(lsm, acts):
    """Per-head log-prob of `acts` (N,6), argmax, rank of the chosen bin
    (1 = the mode), per-head entropy. numpy out."""
    a = torch.as_tensor(np.asarray(acts, np.int64), device=lsm.device)
    lp = lsm.gather(-1, a.unsqueeze(-1)).squeeze(-1)             # (N,6)
    am = lsm.argmax(-1)                                           # (N,6)
    p = lsm.exp()
    chosen = p.gather(-1, a.unsqueeze(-1))                        # (N,6,1)
    rank = (p > chosen).sum(-1) + 1                               # (N,6)
    ent = -(p * lsm).sum(-1)                                      # (N,6)
    return (lp.cpu().numpy(), am.cpu().numpy(), rank.cpu().numpy(), ent.cpu().numpy())


# --------------------------------------------------------------------------
# the record -> our states and our actions
# --------------------------------------------------------------------------
def decode_buttons(raw):
    """parse_hldemo reads the usercmd ushort one byte early (lightlevel is a
    byte at 28, a pad byte follows, buttons sits at 30): the real low byte
    lands in bits 8..15. Verified: IN_FORWARD decoded this way agrees with
    forwardmove > 0 on 100% of frames."""
    return (np.asarray(raw, np.int64) >> 8) & 0xFF


def load_record(npz_path):
    d = np.load(npz_path)
    t = np.asarray(d["t"], np.float64)
    org = np.asarray(d["simorg"], np.float64)
    vel = np.asarray(d["simvel"], np.float64)
    ang = np.asarray(d["uc_viewangles"], np.float64)
    fsu = np.asarray(d["uc_fsu"], np.float64)
    btn = decode_buttons(d["uc_buttons"])
    og = np.asarray(d["onground"], np.int64)
    msec = np.asarray(d["uc_msec"], np.int64)
    yaw = np.unwrap(np.deg2rad(ang[:, 1]))
    yaw = np.rad2deg(yaw)
    pitch = -ang[:, 0]                       # simulator sign: negative = down
    return dict(t=t, org=org, vel=vel, yaw=yaw, pitch=pitch, fsu=fsu, btn=btn,
                og=og, msec=msec, n=len(t))


def record_states(R):
    n = R["n"]
    st = np.zeros(n, STATE_DTYPE)
    st["origin"] = R["org"].astype(np.float32)
    st["velocity"] = R["vel"].astype(np.float32)
    st["yaw"] = (R["yaw"] % 360.0).astype(np.float32)
    st["pitch"] = np.clip(R["pitch"], -70.0, 30.0).astype(np.float32)
    st["onground"] = np.where(R["og"] != 0, 0, -1).astype(np.int32)
    st["oldbuttons"] = np.where((R["btn"] & 2) > 0, SURF_IN_JUMP, 0).astype(np.int32)
    return st


def per_tick_actions(R, yaw_adaptive, yaw_max, pitch_max):
    """Our per-tick action (n,6) whose step from record frame k reproduces
    the record's cmd k: yaw delta = cmd yaw[k] - cmd yaw[k-1] (the state's
    yaw before step k is the previous cmd's), nearest bin under the state's
    own speed; keys from forward/side move; jump/duck from the buttons."""
    n = R["n"]
    dy = np.zeros(n); dy[1:] = R["yaw"][1:] - R["yaw"][:-1]
    dp = np.zeros(n); dp[1:] = R["pitch"][1:] - R["pitch"][:-1]
    vh = np.hypot(R["vel"][:, 0], R["vel"][:, 1])
    w = np.degrees(np.arctan(30.0 / np.maximum(vh, 1.0)))
    if yaw_adaptive:
        eff = np.clip(K_BINS[None, :] * w[:, None], -yaw_max, yaw_max)   # (n,15) deg/tick
    else:
        eff = np.repeat((YAW_BINS_FIXED * (yaw_max / 10.0))[None, :], n, 0)
    ybin = np.abs(eff - dy[:, None]).argmin(1)
    peff = PITCH_BINS.astype(np.float64) * (pitch_max / 10.0)
    pbin = np.abs(peff[None, :] - dp[:, None]).argmin(1)
    fwd = np.where(R["fsu"][:, 0] > 0, 2, np.where(R["fsu"][:, 0] < 0, 0, 1))
    side = np.where(R["fsu"][:, 1] > 0, 2, np.where(R["fsu"][:, 1] < 0, 0, 1))
    jump = ((R["btn"] & 2) > 0).astype(np.int64)
    duck = ((R["btn"] & 4) > 0).astype(np.int64)
    acts = np.stack([ybin, pbin, fwd, side, jump, duck], 1).astype(np.int32)
    return acts, eff, dy, dp, peff


def decision_actions(R, f, K, eff, dy, dp, peff, acts_tick):
    """ONE held action for the K ticks starting at frame f: yaw bin whose
    summed effective delta over the K ticks is nearest the record's summed
    turn, pitch likewise, majority keys, any jump / duck."""
    sl = slice(f, f + K)
    ybin = int(np.abs(eff[sl].sum(0) - dy[sl].sum()).argmin())
    pbin = int(np.abs(peff * K - dp[sl].sum()).argmin())
    def maj(col):
        v = acts_tick[sl, col]
        c = np.bincount(v, minlength=3)
        best = np.flatnonzero(c == c.max())
        return int(best[0] if len(best) == 1 else v[0])
    return np.array([ybin, pbin, maj(2), maj(3), int(acts_tick[sl, 4].max()),
                     int(acts_tick[sl, 5].max())], np.int32)


def phase_for(msec_seq, pattern):
    """The core phase whose next len(msec_seq) ticks best match the record's msec."""
    P = len(pattern)
    best, bp = None, 0
    for p in range(P):
        err = sum(abs(int(pattern[(p + j) % P]) - int(msec_seq[j])) for j in range(len(msec_seq)))
        if best is None or err < best:
            best, bp = err, p
    return bp, best


def step_ticks(core, acts_per_tick, n_ticks):
    """Step `n_ticks` with per-tick (T,N,6) actions; returns the last obs and
    a per-env flag of any episode end inside the window."""
    ended = None
    obs = None
    for j in range(n_ticks):
        obs, _r, done, trunc, _term = core.step(np.ascontiguousarray(acts_per_tick[j], dtype=np.int32))
        e = np.asarray(done, bool) | np.asarray(trunc, bool)
        ended = e if ended is None else (ended | e)
    return obs, ended


# --------------------------------------------------------------------------
# main passes
# --------------------------------------------------------------------------
def scan_record(S, R, args, pitch_frames=None):
    """pitch_frames: per record frame, a pitch written into the record's
    state instead of its own (the policy's camera at the matched point of
    its own line): the depth image is the policy's eye and the record
    looked elsewhere. With it the pitch head is held neutral."""
    K = S["K"]
    coreN = S["coreN"]
    N = int(coreN.config.num_envs)
    cfgc = coreN.config
    yaw_max = float(cfgc.yaw_rate_max_deg)
    pitch_max = float(cfgc.pitch_rate_max_deg)
    yaw_adaptive = bool(S["cfg"].get("yaw_adaptive"))
    pattern = tuple(getattr(coreN, "tick_pattern", (int(cfgc.phys.msec),)))
    st_all = record_states(R)
    acts_tick, eff, dy, dp, peff = per_tick_actions(R, yaw_adaptive, yaw_max, pitch_max)
    if pitch_frames is not None:
        st_all["pitch"] = np.clip(np.asarray(pitch_frames, np.float64), -70.0, 30.0).astype(np.float32)
        acts_tick[:, 1] = NEUTRAL[1]
    sp = np.linalg.norm(R["vel"], axis=1)
    f0 = int(np.argmax(sp > 50.0))                      # first moving frame
    t_rec = R["t"] - WR_CURTAIN_S
    f_end = int(np.argmax(t_rec > args.t_end)) if (t_rec > args.t_end).any() else R["n"] - 1
    frames = np.arange(f0, f_end - K, K)                  # decision starts
    D = len(frames)
    print(f"record: {R['n']} frames, first moving frame {f0} (t {R['t'][f0]:.3f}), "
          f"{D} decisions of {K} ticks to record clock {t_rec[frames[-1]]:.2f} s; "
          f"pattern {pattern}, yaw ceiling {yaw_max:g} deg/tick, adaptive={yaw_adaptive}")
    # the record's own geodesic distance + true sticky latch along the line
    d_geo = S["gf"].sample(R["org"]).astype(np.float64)
    d_latch = float(S["cfg"].get("race_latch") or 0.0)
    latch_true = np.maximum.accumulate((d_geo <= d_latch).astype(np.int8)).astype(bool) if d_latch > 0 else np.zeros(R["n"], bool)
    # route arc per frame (order-only, corridor-gated)
    arcp = ArcProgress.load(args.route)
    arc = np.zeros(R["n"])
    arcp.reset(R["org"][f0:f0 + 1])
    a_cur = float(arcp.arc[0])
    for f in range(f0, R["n"]):
        dl, _ins = arcp.advance(R["org"][f:f + 1])
        a_cur += float(dl[0])
        arc[f] = a_cur
    # per-decision held action (the policy's target) for every decision
    a_dec = np.stack([decision_actions(R, f, K, eff, dy, dp, peff, acts_tick) for f in frames], 0)
    if pitch_frames is not None:
        a_dec[:, 1] = NEUTRAL[1]

    out = {k: np.full(D, np.nan) for k in
           ["err_quant", "err_tick", "err_phys", "verr_quant", "verr_tick", "verr_phys",
            "V", "logp", "logp5", "ent_yaw", "ent_side"]}
    for h in HEADS:
        out["lp_" + h] = np.full(D, np.nan)
        out["am_" + h] = np.full(D, -1)
        out["rank_" + h] = np.full(D, -1)
    out["ended"] = np.zeros(D, bool)
    out["phase_err"] = np.zeros(D, np.int64)

    # group the decisions by the core phase that matches the record's msec
    phases = np.array([phase_for(R["msec"][f:f + K], pattern)[0] for f in frames])
    out["phase_err"] = np.array([phase_for(R["msec"][f:f + K], pattern)[1] for f in frames])
    w, rf, lf = wrapper(S, coreN)
    for p in sorted(set(phases.tolist())):
        idx = np.flatnonzero(phases == p)
        for c0 in range(0, len(idx), N):
            chunk = idx[c0:c0 + N]
            n = len(chunk)
            fr = frames[chunk]

            def load(variant):
                coreN.set_tick_phase(p)
                for e in range(N):
                    f = fr[e] if e < n else fr[0]
                    coreN.set_state(e, st_all[f])
                # per-tick action table (K, N, 6)
                A = np.zeros((K, N, 6), np.int32)
                for e in range(N):
                    f = fr[e] if e < n else fr[0]
                    if variant == "quant":
                        A[:, e] = a_dec[chunk[e] if e < n else chunk[0]]
                    elif variant == "tick":
                        A[:, e] = acts_tick[f:f + K]
                    else:   # phys: keys exact, view forced each tick
                        A[:, e] = acts_tick[f:f + K]
                        A[:, e, 0] = NEUTRAL[0]
                        A[:, e, 1] = NEUTRAL[1]
                return A

            def compare(tag):
                sv = coreN.get_states()
                tgt = st_all[fr + K]
                dpos = np.linalg.norm(sv["origin"][:n].astype(np.float64) - tgt["origin"].astype(np.float64), axis=1)
                dvel = np.linalg.norm(sv["velocity"][:n].astype(np.float64) - tgt["velocity"].astype(np.float64), axis=1)
                out["err_" + tag][chunk] = dpos
                out["verr_" + tag][chunk] = dvel

            # -- quant: one held decision
            A = load("quant")
            _obs, ended = step_ticks(coreN, A, K)
            compare("quant")
            # -- phys: the record's view written into the state before every tick
            A = load("phys")
            for j in range(K):
                sv = coreN.get_states()
                for e in range(N):
                    f = (fr[e] if e < n else fr[0]) + j
                    sv["yaw"][e] = np.float32(R["yaw"][f] % 360.0)
                    sv["pitch"][e] = np.float32(np.clip(R["pitch"][f], -70.0, 30.0))
                    coreN.set_state(e, sv[e])
                coreN.step(np.ascontiguousarray(A[j], dtype=np.int32))
            compare("phys")
            # -- tick: per-tick bins, and the observation + policy at the end
            A = load("tick")
            # feeds: previous-decision distance and the sticky latch as they
            # stood at the decision start
            if rf is not None:
                rf(coreN)
            if lf is not None:
                lf(coreN)
                lf.state["f"] = np.array([latch_true[fr[e] if e < n else fr[0]] for e in range(N)], bool)
            obs, ended = step_ticks(coreN, A, K)
            compare("tick")
            out["ended"][chunk] = ended[:n]
            _padded, lsm, V = policy_eval(S, w, obs)
            # the record's NEXT decision (from frame f+K) is what the policy is scored on
            nxt = np.array([a_dec[min(chunk[e] + 1, D - 1)] if e < n else a_dec[chunk[0]] for e in range(N)], np.int64)
            lp, am, rank, ent = head_stats(lsm, nxt)
            out["V"][chunk] = V[:n]
            for hi, h in enumerate(HEADS):
                out["lp_" + h][chunk] = lp[:n, hi]
                out["am_" + h][chunk] = am[:n, hi]
                out["rank_" + h][chunk] = rank[:n, hi]
            out["logp5"][chunk] = lp[:n, :5].sum(1)         # duck excluded: inert in this core
            out["logp"][chunk] = lp[:n, [0, 2, 3, 4]].sum(1)  # movement heads: yaw, fwd, side, jump
            out["ent_yaw"][chunk] = ent[:n, 0]
            out["ent_side"][chunk] = ent[:n, 3]
    # the decision index is scored on the NEXT decision's action, so shift the
    # bookkeeping: row k describes the observation at frame frames[k]+K
    rows = dict(k=np.arange(D), frame=frames + K, t_rec=t_rec[frames + K],
                x=R["org"][frames + K, 0], y=R["org"][frames + K, 1], z=R["org"][frames + K, 2],
                speed=sp[frames + K], d_geo=d_geo[frames + K], latch=latch_true[frames + K].astype(int),
                arc=arc[frames + K], duck=a_dec[:, 5], onground=(R["og"][frames + K] != 0).astype(int),
                a_yaw=np.array([a_dec[min(k + 1, D - 1), 0] for k in range(D)]),
                a_side=np.array([a_dec[min(k + 1, D - 1), 3] for k in range(D)]))
    rows.update(out)
    return rows, dict(f0=f0, D=D, frames=frames, a_dec=a_dec, acts_tick=acts_tick, arc_total=float(arc.max()))


def scan_line(S, args, spawn_state, obs_start, acts, seed, label, greedy=False, max_dec=4000):
    """Replay `acts` (D,6) from spawn on the 1-env core (or the policy's own
    argmax when greedy) and score each decision's action under the policy."""
    K = S["K"]
    core = S["core1"]
    core.reset(int(seed))
    core.set_state(0, spawn_state)
    core.set_tick_phase(0)
    obs = np.array(obs_start, np.float32).reshape(1, -1).copy()
    w, rf, lf = wrapper(S, core)
    arcp = ArcProgress.load(args.route)
    arcp.reset(np.asarray(spawn_state["origin"]).reshape(1, 3))
    a_cur = float(arcp.arc[0])
    rec = {k: [] for k in ["k", "t", "x", "y", "z", "speed", "d_geo", "arc", "V", "logp", "logp5", "ent_yaw", "ent_side",
                           "a_yaw", "a_side", "pitch", "yaw"] + ["lp_" + h for h in HEADS] + ["rank_" + h for h in HEADS]}
    n_dec = len(acts) if not greedy else max_dec
    finished, tick = False, 0
    for d in range(n_dec):
        padded, lsm, V = policy_eval(S, w, obs)
        if greedy:
            a = padded.argmax(-1).cpu().numpy().astype(np.int32)      # (1,6)
        else:
            a = np.asarray(acts[d], np.int32).reshape(1, 6)
        lp, am, rank, ent = head_stats(lsm, a)
        sv = core.get_states()[0]
        rec["k"].append(d); rec["t"].append(S["TICK"].ticks_to_secs(tick))
        rec["x"].append(float(sv["origin"][0])); rec["y"].append(float(sv["origin"][1])); rec["z"].append(float(sv["origin"][2]))
        rec["speed"].append(float(np.linalg.norm(sv["velocity"])))
        rec["d_geo"].append(float(S["gf"].sample(np.asarray(sv["origin"]).reshape(1, 3))[0]))
        rec["arc"].append(a_cur)
        rec["V"].append(float(V[0])); rec["logp"].append(float(lp[0, [0, 2, 3, 4]].sum())); rec["logp5"].append(float(lp[0, :5].sum()))
        rec["pitch"].append(float(sv["pitch"])); rec["yaw"].append(float(sv["yaw"]))
        rec["ent_yaw"].append(float(ent[0, 0])); rec["ent_side"].append(float(ent[0, 3]))
        rec["a_yaw"].append(int(a[0, 0])); rec["a_side"].append(int(a[0, 3]))
        for hi, h in enumerate(HEADS):
            rec["lp_" + h].append(float(lp[0, hi])); rec["rank_" + h].append(int(rank[0, hi]))
        ended = False
        for j in range(K):
            obs, _r, done, trunc, _term = core.step(np.ascontiguousarray(a, dtype=np.int32))
            tick += 1
            hit = bool(np.asarray(core.goal_hits)[0] > 0)
            dl, _ins = arcp.advance(np.asarray(core.get_states()[0]["origin"]).reshape(1, 3))
            a_cur += float(dl[0])
            if hit:
                finished = True
            if bool(done[0]) or bool(trunc[0]) or hit:
                ended = True
                break
        if ended:
            break
    print(f"{label}: {len(rec['k'])} decisions, {'FINISHED' if finished else 'ended'} at tick {tick} "
          f"({S['TICK'].ticks_to_secs(tick):.3f} s spawn clock), arc {a_cur:.0f}")
    return {k: np.asarray(v) for k, v in rec.items()}, finished, tick


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def write_csv(path, rows, order):
    keys = [k for k in order if k in rows] + [k for k in rows if k not in order]
    n = len(rows[keys[0]])
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(keys) + "\n")
        for i in range(n):
            f.write(",".join(_fmt(rows[k][i]) for k in keys) + "\n")


def _fmt(v):
    if isinstance(v, (np.bool_, bool)):
        return "1" if v else "0"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return "nan" if np.isnan(v) else f"{float(v):.5g}"
    return str(v)


def bin_mean(x, mask):
    m = mask & np.isfinite(x)
    return float(np.mean(x[m])) if m.any() else float("nan")


def summarize(S, W, meta, L, G, R, args, out_dir):
    D = meta["D"]
    t = W["t_rec"]
    K_ = K_BINS
    lines = []
    lines.append(f"# wr_scan: the record through the policy's eyes (ckpt step {S['step']:,})\n")
    lines.append(f"record: {D} decisions of {S['K']} ticks from the first moving frame to record clock {t[-1]:.2f} s. "
                 "log-prob = log pi(the record's next decision) summed over the MOVEMENT heads (yaw, forward, side, jump); "
                 "the pitch head only aims the depth camera and duck is inert in this core.\n")
    lines.append("## 0. Two cameras\n")
    lines.append("The depth image is the policy's eye, and the record's camera pitch is the human's, not the policy's "
                 "(record p50 -16 deg, the policy's own line p10..p90 = -50..+28). So every number is given twice: at the "
                 "record's states with the RECORD's pitch, and with the POLICY's pitch at the nearest point of its own line.\n")
    lines.append("| camera | log-likelihood (movement) | mean per decision | yaw = argmax | side = argmax | yaw rank | mean V | V 61.3-63.9 s (record ramp 1) | V 64-68 s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    m_r1 = (t >= WR_RAMP1[0]) & (t <= WR_RAMP1[1]); m_f = (t >= 64.0) & (t <= 68.0)
    allm = np.ones(D, bool)
    for cam, suf in (("record's pitch", ""), ("policy's pitch (matched)", "_op")):
        lines.append(f"| {cam} | {np.nansum(W['logp' + suf]):.0f} | {np.nanmean(W['logp' + suf]):.2f} | "
                     f"{100*bin_mean((W['rank_yaw' + suf] == 1).astype(float), allm):.0f}% | {100*bin_mean((W['rank_side' + suf] == 1).astype(float), allm):.0f}% | "
                     f"{bin_mean(W['rank_yaw' + suf].astype(float), allm):.2f} | {np.nanmean(W['V' + suf]):.2f} | {bin_mean(W['V' + suf], m_r1):.2f} | {bin_mean(W['V' + suf], m_f):.2f} |")
    lines.append(f"| our line (its own states, its own actions) | {np.nansum(L['logp']):.0f} | {np.nanmean(L['logp']):.3f} | "
                 f"{100*np.mean(L['rank_yaw'] == 1):.0f}% | {100*np.mean(L['rank_side'] == 1):.0f}% | {np.mean(L['rank_yaw']):.2f} | {np.nanmean(L['V']):.2f} | "
                 f"{bin_mean(L['V'], (L['t'] - 0.96 >= WR_RAMP1[0]) & (L['t'] - 0.96 <= WR_RAMP1[1])):.2f} | {bin_mean(L['V'], (L['t'] - 0.96 >= 64) & (L['t'] - 0.96 <= 68)):.2f} |")
    lines.append(f"\nrecord vs our line: nearest-point distance median {np.nanmedian(W['dist_line']):.0f} u, p90 {np.nanpercentile(W['dist_line'], 90):.0f} u; "
                 f"in the last 8 s median {np.nanmedian(W['dist_line'][t >= 60.8]):.0f} u.")
    # feasibility
    lines.append("\n## 1. Can our action space follow the record? (state error after one 4-tick decision, u)\n")
    lines.append("| conversion | median | p90 | p99 | max | decisions > 20 u | > 100 u |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag, name in (("phys", "phys: our physics, the record's view angles forced, keys exact"),
                      ("tick", "tick: nearest bins per tick"),
                      ("quant", "quant: one held decision (what the policy can emit)")):
        e = W["err_" + tag]
        ok = np.isfinite(e)
        lines.append(f"| {name} | {np.nanmedian(e):.2f} | {np.nanpercentile(e, 90):.2f} | {np.nanpercentile(e, 99):.2f} | "
                     f"{np.nanmax(e):.1f} | {int((e[ok] > 20).sum())} | {int((e[ok] > 100).sum())} |")
    lines.append("")
    lines.append("velocity error after one decision (u/s): " + ", ".join(
        f"{tag} median {np.nanmedian(W['verr_' + tag]):.1f} / p99 {np.nanpercentile(W['verr_' + tag], 99):.1f}"
        for tag in ("phys", "tick", "quant")))
    duck = W["duck"] > 0
    lines.append(f"\nthe record ducks on {int(duck.sum())} of {D} decisions (duck is inert in our core); "
                 f"median phys error inside duck decisions {np.nanmedian(W['err_phys'][duck]) if duck.any() else float('nan'):.2f} u "
                 f"vs {np.nanmedian(W['err_phys'][~duck]):.2f} u outside")
    big = np.flatnonzero(np.isfinite(W["err_tick"]) & (W["err_tick"] > 20))
    if len(big):
        segs = []
        s0 = big[0]; prev = big[0]
        for i in big[1:]:
            if i != prev + 1:
                segs.append((s0, prev)); s0 = i
            prev = i
        segs.append((s0, prev))
        lines.append("\nwhere the per-tick grid misses by > 20 u (record clock s, max error u, ducking, on ground):")
        for a_, b_ in segs[:40]:
            sl = slice(a_, b_ + 1)
            lines.append(f"- {t[a_]:.2f}-{t[b_]:.2f} s: max {np.nanmax(W['err_tick'][sl]):.0f} u (phys {np.nanmax(W['err_phys'][sl]):.0f} u), "
                         f"duck {int(duck[sl].any())}, onground {int((W['onground'][sl] > 0).any())}, speed {W['speed'][a_]:.0f}")
    # by decile
    lines.append("\n## 2. The policy's opinion of the record, by tenth of the record clock (both cameras)\n")
    lines.append("| record clock | dist to our line (u) | logp rec cam | logp pol cam | yaw=argmax rec/pol | side=argmax rec/pol | yaw rank rec/pol | V rec cam | V pol cam | V our line (matched point) | V our line (same clock) | logp our line |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    T_end = float(t[-1])
    edges = np.linspace(0, T_end, 11)
    for i in range(10):
        m = (t >= edges[i]) & (t < edges[i + 1] + (1e-6 if i == 9 else 0))
        if not m.any():
            continue
        ml = (L["t"] - 0.96 >= edges[i]) & (L["t"] - 0.96 < edges[i + 1])
        lines.append(f"| {edges[i]:.1f}-{edges[i+1]:.1f} s | {bin_mean(W['dist_line'], m):.0f} | {bin_mean(W['logp'], m):.1f} | {bin_mean(W['logp_op'], m):.1f} | "
                     f"{100*bin_mean((W['rank_yaw'] == 1).astype(float), m):.0f}/{100*bin_mean((W['rank_yaw_op'] == 1).astype(float), m):.0f}% | "
                     f"{100*bin_mean((W['rank_side'] == 1).astype(float), m):.0f}/{100*bin_mean((W['rank_side_op'] == 1).astype(float), m):.0f}% | "
                     f"{bin_mean(W['rank_yaw'].astype(float), m):.1f}/{bin_mean(W['rank_yaw_op'].astype(float), m):.1f} | "
                     f"{bin_mean(W['V'], m):.2f} | {bin_mean(W['V_op'], m):.2f} | {bin_mean(W['V_line_matched'], m):.2f} | {bin_mean(L['V'], ml):.2f} | {bin_mean(L['logp'], ml):.2f} |")
    lines.append("\n(our line is placed on the record clock by spawn clock - 0.96 s; 'matched point' = the nearest point of our line to the record's position)")
    # the finish room
    lines.append("\n## 3. The finish room, half-second bins (record clock)\n")
    lines.append("| record clock | dist to line | logp rec/pol cam | logp yaw | logp side | WR yaw K -> argmax (pol cam) | WR side -> argmax (pol cam) | yaw rank | V rec cam | V pol cam | V line matched | latch | speed | z | err tick | err quant |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a_ in np.arange(max(0.0, args.room_t0), T_end, 0.5):
        m = (t >= a_) & (t < a_ + 0.5)
        if not m.any():
            continue
        ky = W["a_yaw"][m]; ay = W["am_yaw_op"][m]; ks = W["a_side"][m]; as_ = W["am_side_op"][m]
        lines.append(f"| {a_:.1f} | {bin_mean(W['dist_line'], m):.0f} | {bin_mean(W['logp'], m):.1f}/{bin_mean(W['logp_op'], m):.1f} | {bin_mean(W['lp_yaw_op'], m):.1f} | {bin_mean(W['lp_side_op'], m):.1f} | "
                     f"{np.round(K_[ky].mean(), 2)} -> {np.round(K_[ay].mean(), 2)} | "
                     f"{np.round(ks.mean() - 1, 2)} -> {np.round(as_.mean() - 1, 2)} | {bin_mean(W['rank_yaw_op'].astype(float), m):.1f} | "
                     f"{bin_mean(W['V'], m):.2f} | {bin_mean(W['V_op'], m):.2f} | {bin_mean(W['V_line_matched'], m):.2f} | {int(W['latch'][m].max())} | {bin_mean(W['speed'], m):.0f} | {bin_mean(W['z'], m):.0f} | "
                     f"{bin_mean(W['err_tick'], m):.1f} | {bin_mean(W['err_quant'], m):.1f} |")
    lines.append("\n(yaw in units of the optimal-strafe angle atan(30/|v|) per tick, the K_BINS scale; side: -1 left, 0 none, +1 right)")
    # value by arc
    lines.append("\n## 4. Value by route arc (order-only arc, 20 bins)\n")
    lines.append("| arc (ku) | V record (rec cam) | V record (pol cam) | V our line | n rec | n line |")
    lines.append("|---|---|---|---|---|---|")
    amax = max(float(np.nanmax(W["arc"])), float(np.nanmax(L["arc"])))
    ae = np.linspace(0, amax, 21)
    for i in range(20):
        m = (W["arc"] >= ae[i]) & (W["arc"] < ae[i + 1])
        ml = (L["arc"] >= ae[i]) & (L["arc"] < ae[i + 1])
        lines.append(f"| {ae[i]/1000:.0f}-{ae[i+1]/1000:.0f} | {bin_mean(W['V'], m):.2f} | {bin_mean(W['V_op'], m):.2f} | "
                     f"{bin_mean(L['V'], ml):.2f} | {int(m.sum())} | {int(ml.sum())} |")
    # worst decisions
    lines.append("\n## 5. The record's 25 least likely decisions (policy's camera)\n")
    lines.append("| record clock | logp | worst head | WR action (yaw K, fwd, side, jump) | policy argmax | V | speed | duck | dist to line |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    order = np.argsort(np.nan_to_num(W["logp_op"], nan=0.0))[:25]
    for k in sorted(order.tolist()):
        lps = np.array([W["lp_" + h + "_op"][k] for h in ("yaw", "fwd", "side", "jump")])
        wh = ("yaw", "fwd", "side", "jump")[int(np.argmin(lps))]
        a_ = meta["a_dec"][min(k + 1, D - 1)]
        am = [int(W["am_yaw_op"][k]), int(W["am_side_op"][k])]
        lines.append(f"| {t[k]:.2f} | {W['logp_op'][k]:.1f} | {wh} ({lps.min():.1f}) | "
                     f"({K_[a_[0]]:g}, {a_[2]-1:+d}, {a_[3]-1:+d}, {a_[4]}) | (yaw {K_[am[0]]:g}, side {am[1]-1:+d}) | "
                     f"{W['V_op'][k]:.2f} | {W['speed'][k]:.0f} | {int(W['duck'][k])} | {W['dist_line'][k]:.0f} |")
    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(txt, encoding="utf-8")
    return txt


def plot(W, L, G, args, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:      # pragma: no cover
        print("no matplotlib:", e)
        return
    t = W["t_rec"]
    def smooth(x, n=16):
        x = np.asarray(x, np.float64)
        k = np.ones(n) / n
        m = np.isfinite(x)
        return np.convolve(np.where(m, x, 0.0), k, "same") / np.maximum(np.convolve(m.astype(float), k, "same"), 1e-9)
    fig, ax = plt.subplots(5, 1, figsize=(15, 16), sharex=True)
    ax[0].plot(t, smooth(W["logp"]), label="record, record's camera", lw=0.9, alpha=0.7)
    ax[0].plot(t, smooth(W["logp_op"]), label="record, policy's camera", lw=1.2)
    ax[0].plot(L["t"] - 0.96, smooth(L["logp"]), label="our searched line", lw=1.0)
    ax[0].set_ylabel("log pi(next action)\nmovement heads, 0.5 s mean"); ax[0].legend(loc="lower left"); ax[0].grid(alpha=.3)
    ax[1].plot(t, W["V"], label="V at the record's states, record's camera", lw=0.8, alpha=0.6)
    ax[1].plot(t, W["V_op"], label="V at the record's states, policy's camera", lw=1.1)
    ax[1].plot(t, W["V_line_matched"], label="V of our line at the nearest point", lw=1.0)
    ax[1].plot(L["t"] - 0.96, L["V"], label="V along our line (same clock)", lw=0.8, alpha=0.7)
    ax[1].set_ylabel("value"); ax[1].legend(loc="upper left"); ax[1].grid(alpha=.3)
    ax[2].semilogy(t, np.maximum(W["dist_line"], 1.0), lw=1.0, color="tab:purple")
    ax[2].set_ylabel("record to our line (u)"); ax[2].grid(alpha=.3)
    for tag, c in (("phys", "k"), ("tick", "tab:blue"), ("quant", "tab:red")):
        ax[3].semilogy(t, np.maximum(W["err_" + tag], 1e-2), ".", ms=2, color=c, label=tag)
    ax[3].set_ylabel("state error after one\n4-tick decision (u)"); ax[3].legend(loc="upper left"); ax[3].grid(alpha=.3)
    ax[4].plot(t, W["speed"], label="record speed", lw=1.0)
    ax[4].plot(L["t"] - 0.96, L["speed"], label="our line speed", lw=1.0)
    ax[4].set_ylabel("u/s"); ax[4].set_xlabel("record clock (s)"); ax[4].legend(loc="upper left"); ax[4].grid(alpha=.3)
    duck = W["duck"] > 0
    for a_ in ax:
        a_.axvspan(WR_RAMP1[0], WR_RAMP1[1], color="orange", alpha=0.15)
        for k in np.flatnonzero(duck):
            a_.axvline(t[k], color="green", alpha=0.06, lw=1)
    fig.suptitle("the record through the policy's eyes: orange = the record's ramp-1 phase, green = record ducking")
    fig.tight_layout()
    fig.savefig(out_dir / "wr_scan.png", dpi=110)
    print("wrote", out_dir / "wr_scan.png")


def match_line(R, L):
    """Per record frame: the nearest decision of our line (3-D), its distance,
    the policy's camera pitch there and the critic's value there."""
    P = np.stack([L["x"], L["y"], L["z"]], 1).astype(np.float64)
    n = R["n"]
    j = np.zeros(n, np.int64)
    dist = np.zeros(n)
    for a in range(0, n, 512):
        q = R["org"][a:a + 512]
        d2 = ((q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        jj = d2.argmin(1)
        j[a:a + 512] = jj
        dist[a:a + 512] = np.sqrt(d2[np.arange(len(q)), jj])
    return j, dist, L["pitch"][j], L["V"][j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wr", required=True, help="<stem>.frames.npz from parse_hldemo.py")
    ap.add_argument("--line", required=True, help="beam_best.npz of our searched line (open-loop replay; its states are the policy's own)")
    ap.add_argument("--map", default="C:/RL_Surf/maps/surf_src_cannonball.bsp")
    ap.add_argument("--route", default="C:/RL_Surf/maps/surf_src_cannonball.route.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--t-end", type=float, default=68.8, help="last record-clock second to scan")
    ap.add_argument("--room-t0", type=float, default=56.0)
    ap.add_argument("--greedy", action="store_true", help="also run the policy's closed-loop greedy episode from the line's spawn")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = setup(args.ckpt, args.map, args.envs)
    print(f"setup {time.time()-t0:.1f}s: K={S['K']} tick={S['TICK'].requested_ms} map={S['map_path']}")
    R = load_record(args.wr)
    z = np.load(args.line, allow_pickle=False)
    if int(z["act_every"]) != S["K"]:
        raise SystemExit("line act_every mismatch")
    L, _fin, _tk = scan_line(S, args, z["spawn_state"][0], z["obs_start"], np.asarray(z["acts"], np.int32),
                             int(z["gate_seed"]), "searched line")
    write_csv(out_dir / "line_scan.csv", L, ["k", "t", "x", "y", "z", "speed", "d_geo", "arc", "V", "logp", "pitch"])
    G = None
    if args.greedy:
        G, _fin, _tk = scan_line(S, args, z["spawn_state"][0], z["obs_start"], None, int(z["gate_seed"]),
                                 "greedy episode", greedy=True)
        write_csv(out_dir / "greedy_scan.csv", G, ["k", "t", "x", "y", "z", "speed", "d_geo", "arc", "V", "logp", "pitch"])
    jl, dist_line, pitch_line, V_line = match_line(R, L)
    W, meta = scan_record(S, R, args)
    print(f"record (its own camera) scanned in {time.time()-t0:.1f}s: logL(move) {np.nansum(W['logp']):.0f}, "
          f"median errors phys {np.nanmedian(W['err_phys']):.2f} / tick {np.nanmedian(W['err_tick']):.2f} / quant {np.nanmedian(W['err_quant']):.2f} u")
    W2, _m2 = scan_record(S, R, args, pitch_frames=pitch_line)
    print(f"record (the policy's camera pitch at the matched point of its own line): logL(move) {np.nansum(W2['logp']):.0f}")
    for k in ("V", "logp", "logp5", "ent_yaw", "ent_side", "lp_yaw", "lp_side", "lp_fwd", "lp_jump",
              "am_yaw", "am_side", "rank_yaw", "rank_side"):
        W[k + "_op"] = W2[k]
    fr = W["frame"]
    W["dist_line"] = dist_line[fr]
    W["pitch_rec"] = np.clip(R["pitch"][fr], -70, 30)
    W["pitch_line"] = pitch_line[fr]
    W["V_line_matched"] = V_line[fr]
    write_csv(out_dir / "wr_scan.csv", W, ["k", "frame", "t_rec", "x", "y", "z", "speed", "d_geo", "latch", "arc", "duck",
                                            "onground", "dist_line", "pitch_rec", "pitch_line", "err_phys", "err_tick", "err_quant",
                                            "verr_phys", "verr_tick", "verr_quant", "ended", "logp", "logp_op", "V", "V_op", "V_line_matched"])
    txt = summarize(S, W, meta, L, G, R, args, out_dir)
    print(txt[:9000])
    plot(W, L, G, args, out_dir)
    print(f"done in {time.time()-t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
