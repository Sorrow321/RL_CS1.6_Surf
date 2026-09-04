#!/usr/bin/env python3
"""expert_dagger.py - the DAgger relabelling phase of expert iteration.

Expert iteration (tools/expert_loop.py) distils the planner's ELITE line
into the policy and plateaued 1.5-2.4 s behind that line at 98% per-head
agreement (ledger round 30, exit10/exit30): the policy's own small
deviations compound into states the elite line never covers, where it has
no supervision. This phase supplies it: roll the CURRENT policy out, sample
its states, plan a SHORT window from each with the same beam machinery and
take the planner's first decision(s) there as the label (surfgym.dagger).

    round r:  eval -> plan elite line -> distil (bc.npz + spine.npy)
              -> [this] relabel from the policy's own states -> bc_dagger.npz
              -> train on elite + relabelled rows -> eval

Standalone (one round's files; the same thing the driver runs):

    python tools/expert_dagger.py runs/exit/round_5/train/ckpt_final.pt \
        --bc runs/exit/round_6/bc.npz --spine runs/exit/round_6/spine.npy \
        --out runs/exit/round_6/bc_dagger.npz --k 600 --window 3

Inside the loop: ``python tools/expert_loop.py ... --dagger-k 600`` (see
add_args; default off = the loop byte-for-byte).

Outputs next to --out: ``<out>`` (elite + relabelled rows, the file the
trainer's --bc-file takes), ``<stem>_rows.npz`` (the relabelled rows alone,
same format, for inspection), ``<stem>_samples.npz`` (the sampled states
with the feed internals) and ``<stem>_summary.json`` (--summary-out).

Checkpoint-faithful loading mirrors tools/beam_tas.py / record_ckpt.py
(config audit imported, unsupported per-env inference state refused). The
map is resolved in the MAIN checkout unless given (cache mtimes: CLAUDE.md).

PHYSICS TICK. Every core here is built at the CHECKPOINT'S ``tick_ms``
(``beam_tas.build_sim(..., tick=)``, which drives SurfCore's integer-ms
pattern and rescales the per-tick view rates - not the yaw ceiling under
--yaw-adaptive), the --obs-reward slot-12 mirror is built at the same tick
(``surfgym.bc.make_eval_feeds(tick_ms=)``), and every seconds flag
(--every / --window / --rollout-secs / --spine-secs) converts through that
clock, never through a 100 Hz literal. So a 7.63 ms checkpoint relabels at
7.63 ms (pattern [8, 8, 7] = 130.4 Hz) and a 10 ms one is the legacy
arithmetic bit for bit. Every core the phase opens is checked against the
clock (``surfgym.dagger.check_core_tick``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
PY = sys.executable

from surfgym.bc import N_SCALAR, REWARD_SLOT, make_eval_feeds, save_bc_dataset  # noqa: E402
from surfgym.dagger import (LABEL_TARGETS, SRC_GREEDY, SRC_NAMES,  # noqa: E402
                            SRC_SPINE, SRC_STOCH,
                            SampleBank, check_actions, check_core_tick,
                            collect_rollout_samples,
                            divergence_weights, even_subset,
                            merge_bc_datasets, nearest_distance,
                            relabel_windows, rows_from_results,
                            summarize_results)
from surfgym.tick import TickClock  # noqa: E402

_RowPolicy = None


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# --------------------------------------------------------------------------
# the policy wrapper that keeps the row it acted on
# --------------------------------------------------------------------------
def row_policy_class():
    """GreedyTorchPolicy that (a) samples envs outside ``greedy_mask`` from
    the logits at temperature ``temp`` with its own torch.Generator and (b)
    keeps ``last_row`` (the scalar half of the row it just built: the 15
    core scalars with the slot-12 mirror, then the latch column) and
    ``last_greedy`` (the argmax actions of every env). Built lazily so
    importing this module (expert_loop does) costs no torch import."""
    global _RowPolicy
    if _RowPolicy is not None:
        return _RowPolicy
    import torch
    from train_fast import GreedyTorchPolicy

    class RowPolicy(GreedyTorchPolicy):
        def __init__(self, *a, greedy_mask=None, temp: float = 1.0, gen=None,
                     n_cols: int = N_SCALAR, keep_logits: bool = False, **kw):
            super().__init__(*a, **kw)
            self._keep_logits = bool(keep_logits)
            self._temp = max(float(temp), 1e-6)
            self._gen = gen
            self._ncols = int(n_cols)
            self._mask_t = None
            if greedy_mask is not None and not bool(np.all(greedy_mask)):
                self._mask_t = torch.as_tensor(
                    np.ascontiguousarray(np.asarray(greedy_mask, bool)),
                    device=self.device)
            self.last_row = None
            self.last_greedy = None
            self.last_logits = None

        @torch.inference_mode()
        def _decide(self, obs):
            row = self._obs(obs)
            logits, _ = self._net(row)
            padded = self.packer.pad(logits.float())
            greedy = padded.argmax(-1)
            if self._mask_t is None:
                act = greedy
            else:
                u = torch.rand(padded.shape, generator=self._gen,
                               device=padded.device).clamp_min_(1e-20)
                samp = (padded / self._temp - torch.log(-torch.log(u))
                        ).argmax(-1)
                act = torch.where(self._mask_t[:, None], greedy, samp)
            self.last_row = row[:, :self._ncols].to("cpu").numpy().copy()
            self.last_greedy = greedy.to("cpu").numpy().astype(np.int32)
            # --label-target gumbel needs the PRIOR the improved policy is
            # built on; the counts-only path never reads it
            self.last_logits = (padded.to("cpu").numpy()
                                if self._keep_logits else None)
            return act.to("cpu").numpy().astype(np.int32)

    _RowPolicy = RowPolicy
    return RowPolicy


# --------------------------------------------------------------------------
# checkpoint bundle
# --------------------------------------------------------------------------
def load_bundle(ckpt_path, map_path, device, audit: bool = True) -> dict:
    """The checkpoint's policy, lidar, goal field, spawn pool and feeds,
    loaded exactly as beam_tas / record_ckpt load them."""
    import torch
    import beam_tas
    import record_ckpt as _rc
    from plan_to_bc import goal_cell_for
    from surfgym.goalfield import EuclidField, build_goal_field
    from surfgym.mapfleet import map_tag
    from surfgym.rewards import map_spawn_pool
    from surfgym.vision import GpuLidar, pick_cell
    from surfgym.zones import load_zones
    from train_fast import HeadPacker, Policy

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    step = int(ck.get("global_step", 0))
    if cfg.get("reward") != "race":
        raise SystemExit("expert_dagger needs a race checkpoint; this one "
                         f"has reward={cfg.get('reward')!r}")
    bad = [k for k in beam_tas.UNSUPPORTED if cfg.get(k)]
    bad += [k for k in ("act_hist", "obs_compass") if cfg.get(k)]
    if str(cfg.get("rnn") or "none") != "none":
        bad.append("rnn")
    if bad:
        raise SystemExit("expert_dagger cannot clone the per-env inference "
                         "state of: " + ", ".join(bad))
    if audit:
        _rc.audit_cfg(cfg, strict=True)
    K = int(cfg.get("act_every", 1))
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    # the checkpoint's physics tick: every seconds flag of this tool is
    # converted at it, and the episode cap's default is a DURATION (120 s =
    # 12,000 ticks at 10 ms, 15,652 at the 7.667 ms pattern), never a tick
    # count. 10 ms is the legacy arithmetic bit for bit.
    TICK = TickClock(float(cfg.get("tick_ms") or 10.0))
    ep_120 = TICK.secs_to_ticks(120.0)
    ep_cap = max(int(cfg.get("ep_ticks", ep_120)), ep_120)
    map_path = beam_tas.resolve_map(map_path, cfg.get("map", "surf_ski_2"))
    # ``tick=TICK`` is what makes a --tick-ms checkpoint relabel in ITS OWN
    # physics: build_sim drives the integer-ms pattern into SurfCore and
    # rescales the per-TICK view rates so the deg per SECOND is unchanged -
    # except the yaw ceiling under --yaw-adaptive, which is the per-FRAME
    # strafe optimum and must NOT scale (beam_tas.build_sim gates it;
    # CLAUDE.md / commit ecc0506). At 10 ms build_sim ignores the clock and
    # builds the core it always built, bit for bit.
    core1 = beam_tas.build_sim(cfg, map_path, 1, ep_cap, tick=TICK)
    # a POST-condition now, not a refusal-to-run: the tick IS wired, so a
    # mismatch here means build_sim stopped honouring tick= (or the DLL has
    # no surf_set_msec and silently kept 10 ms)
    check_core_tick(core1, TICK, "expert_dagger probe core",
                    "beam_tas.build_sim did not apply tick=, so every "
                    "relabelled action would be planned in the wrong "
                    "physics and --every/--window/--rollout-secs/"
                    "--spine-secs would convert at the wrong rate. Rebuild "
                    "the DLL (surf_set_msec) and check build_sim's tick "
                    "wiring (and that its yaw ceiling stays gated on "
                    "--yaw-adaptive, CLAUDE.md).")
    zones = load_zones(core1.bsp_path)
    tag = map_tag(Path(map_path).stem)
    cell = float((cfg.get("map_cells") or {}).get(
        tag, cfg.get("lidar_cell") or pick_cell(core1)))
    gcell = goal_cell_for(cfg, map_path, core1)
    t0 = time.time()
    gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
          else build_goal_field(core1, zones["end"], cell=gcell))
    dt = time.time() - t0
    say(f"goal field @ cell {gcell:g} in {dt:.1f}s"
        + ("  ** WARNING: that smells like a RE-BAKE - wrong map "
           "path/mtime? **" if dt > 30 else ""))
    raw = map_spawn_pool(core1)
    pool = map_spawn_pool(core1, yaw=gf.descent_yaw(raw["origin"]))
    pool["pitch"] = -10.0
    if cfg.get("fix_pitch") is not None:
        pool["pitch"] = float(cfg["fix_pitch"])
    d0 = float(np.mean(gf.sample(raw["origin"])))
    lidar = GpuLidar(core1, lw, lh,
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"), cell=cell,
                     device=device, surf_mask=bool(cfg.get("surf_mask", 0)),
                     pinhole=bool(cfg.get("pinhole", 0)),
                     normals=bool(cfg.get("normals", 0)))
    n_latch = 1 if (float(cfg.get("race_latch") or 0.0) > 0.0
                    or float(cfg.get("race_latch_frac") or 0.0) > 0.0) else 0
    extra = (REWARD_SLOT,) if cfg.get("obs_reward") else ()
    policy = Policy(core1.obs_dim + n_latch + lw * lh * lidar.channels,
                    lw, lh, emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    trunk=str(cfg.get("trunk") or "plain"),
                    tower_depth=int(cfg.get("tower_depth") or 2),
                    conv_mult=int(cfg.get("conv_mult") or 1),
                    extra_feat=extra, in_ch=lidar.channels,
                    n_codes=0, chunk=0, route_dim=n_latch,
                    route_critic_only=bool(cfg.get("route_critic_only"))
                    ).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    # the --obs-reward slot-12 mirror is a per-TICK reward (time_pen) and a
    # per-tick gamma: handing the 10 ms-referenced values to a 7.667 ms core
    # puts a constant offset into the one column the policy reads its own
    # reward from (record_ckpt / plan_to_bc pass the same tick_ms)
    slot, _rf, _lf = make_eval_feeds(cfg, gf, d0, K,
                                     tick_ms=TICK.requested_ms)
    return {"cfg": cfg, "step": step, "K": K, "ep_cap": ep_cap, "tick": TICK,
            "map_path": map_path, "zones": zones, "gf": gf, "d0": d0,
            "pool": pool, "lidar": lidar, "policy": policy,
            "packer": HeadPacker(device), "device": device,
            "n_latch": n_latch, "slot": slot,
            "obs_reward": bool(cfg.get("obs_reward")),
            "d_floor": float(cfg.get("race_dfloor") or 0.0),
            "pitch_fixed": cfg.get("pitch_fixed"), "core1": core1}


def open_core(B: dict, n: int):
    """An armed n-env core (goal box, teleport fail, the map's spawn pool)
    exactly as beam_tas arms its search core, at the CHECKPOINT'S tick.

    The pattern PHASE is whatever ``reset()`` leaves (0), as it is in
    beam_tas's own population search: a relabel chunk holds several groups
    whose sampled states sit at different ticks of different rollouts, and
    the core steps every env in lockstep, so no single phase can be right
    for more than one group. Under [8,8,7] that is at most a 1 ms shift of
    the first ticks of a window, and none at the reference tick."""
    import beam_tas
    core = beam_tas.build_sim(B["cfg"], B["map_path"], int(n), B["ep_cap"],
                              tick=B["tick"])
    check_core_tick(core, B["tick"], f"expert_dagger {n}-env core")
    z = B["zones"]
    core.set_goal_box(z["end"]["mins"], z["end"]["maxs"])
    if B["cfg"].get("teleport_fail") or B["cfg"].get("reward") == "race":
        core.set_teleport_fail(True)
    core.set_spawn_pool(B["pool"])
    return core


def make_feeds(B: dict):
    """Fresh (reward_feed, latch_fn) for one core run (surfgym.bc), at the
    checkpoint's tick so the slot-12 mirror is the trainer's own."""
    _slot, rf, lf = make_eval_feeds(B["cfg"], B["gf"], B["d0"], B["K"],
                                    tick_ms=B["tick"].requested_ms)
    return rf, lf


def make_decider(B: dict, core, feeds, greedy_mask, temp: float, gen,
                 keep_logits: bool = False):
    cls = row_policy_class()
    rf, lf = feeds
    return cls(B["policy"], B["packer"], B["device"], B["lidar"], core,
               B["K"], 1, extra_slot=B["slot"], extra_fn=rf, latch_fn=lf,
               pitch_fixed=B["pitch_fixed"], greedy_mask=greedy_mask,
               temp=temp, gen=gen, n_cols=N_SCALAR + B["n_latch"],
               keep_logits=keep_logits)


def make_value_fn(holder: dict):
    """V(s) on the current obs through the chunk's own decider (its _obs is
    the row the policy sees), the feeds' per-env internals restored around
    the call because _obs advances them (beam_tas's value_fn)."""
    import torch

    def value_fn(obs):
        dec, (rf, lf) = holder["decider"], holder["feeds"]
        saved = None if rf is None else rf.state.get("d")
        lsaved = None if lf is None else (lf.state.get("f"),
                                          lf.state.get("tick"))
        with torch.inference_mode():
            _, v = dec.policy(dec._obs(obs))
        if rf is not None:
            rf.state["d"] = saved
        if lsaved is not None:
            lf.state["f"], lf.state["tick"] = lsaved
        return v.detach().float().reshape(-1).cpu().numpy()
    return value_fn


def make_grouped_scorer(gf, mode: str, value_fn, v_switch: float,
                        copies: int, d_floor: float = 0.0):
    """score(states, obs) -> (higher is better, geodesic d), ranked WITHIN
    each group of ``copies`` envs: 'd' = -geodesic; 'v' = the critic; 'dv'
    = -d until a GROUP's frontier is within v_switch of the goal, then the
    critic for that group (beam_tas --score dv, per group)."""
    copies = int(copies)

    def score(states, obs):
        d = np.asarray(gf.sample(states["origin"]), np.float64)
        if d_floor > 0.0:
            d = np.maximum(d, d_floor)
        if mode == "d":
            return -d, d
        v = np.asarray(value_fn(obs), np.float64)
        if mode == "v":
            return v, d
        n = len(d)
        G = n // copies
        sw = np.zeros(n, bool)
        if G > 0:
            dm = d[:G * copies].reshape(G, copies).min(axis=1)
            sw[:G * copies] = np.repeat(dm <= float(v_switch), copies)
        return np.where(sw, v, -d), d
    return score


# --------------------------------------------------------------------------
# the phase
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="DAgger relabelling: plan short windows from the "
                    "policy's own states and add the labels to the BC rows")
    ap.add_argument("ckpt", help="the policy to roll out and plan with")
    ap.add_argument("--bc", required=True,
                    help="the round's elite BC rows (tools/plan_to_bc.py)")
    ap.add_argument("--spine", required=True,
                    help="the round's spine.npy (the elite line's states)")
    ap.add_argument("--out", required=True,
                    help="merged BC file (elite + relabelled rows)")
    ap.add_argument("--map", default=None,
                    help="absolute .bsp (default: the ckpt's map in the "
                         "MAIN checkout)")
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--k", type=int, default=600,
                    help="states to relabel (spread evenly over the "
                         "candidates)")
    ap.add_argument("--every", type=float, default=0.5,
                    help="seconds between candidate states along a rollout")
    ap.add_argument("--episodes", type=int, default=9,
                    help="greedy map-start rollouts")
    ap.add_argument("--stoch-episodes", type=int, default=9,
                    help="tempered map-start rollouts")
    ap.add_argument("--temp", type=float, default=0.5,
                    help="sampling temperature of the stochastic rollouts "
                         "(1 = the policy's own distribution, ->0 greedy)")
    ap.add_argument("--rollout-secs", type=float, default=0.0,
                    help="cap on a map-start rollout (0 = the episode cap)")
    ap.add_argument("--spine-spawns", type=int, default=18,
                    help="rollouts spawned along the elite line (half "
                         "greedy, half tempered)")
    ap.add_argument("--spine-secs", type=float, default=20.0)
    ap.add_argument("--window", type=float, default=3.0,
                    help="planner window per state, seconds")
    ap.add_argument("--envs", type=int, default=2048,
                    help="envs of the search core (envs // copies states "
                         "per chunk)")
    ap.add_argument("--copies", type=int, default=256,
                    help="clones per state")
    ap.add_argument("--greedy-envs", type=int, default=16,
                    help="greedy envs per group (the policy's own "
                         "continuation is never cloned over)")
    ap.add_argument("--resample", type=int, default=25,
                    help="decisions between clonings (0 = none)")
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--plan-temp", type=float, default=1.0,
                    help="sampling temperature of the search's sampled envs")
    ap.add_argument("--score", choices=["d", "v", "dv"], default="dv")
    ap.add_argument("--v-switch", type=float, default=20000.0)
    ap.add_argument("--label-decisions", type=int, default=1,
                    help="decisions of the winning lineage taken as labels "
                         "(1 = the DAgger label at the sampled state only)")
    ap.add_argument("--label-target", choices=list(LABEL_TARGETS),
                    default="count",
                    help="what the stored per-head DISTRIBUTION is (the "
                         "winner's action index is stored either way, so "
                         "--bc-target argmax is unaffected): 'count' = the "
                         "surviving copies' own first decisions (ExIt's "
                         "TPT); 'gumbel' = Danihelka 2022's "
                         "softmax(logits + sigma(completedQ)) over the same "
                         "population, per head")
    ap.add_argument("--c-visit", type=float, default=50.0,
                    help="--label-target gumbel: sigma's c_visit (the "
                         "paper's 50)")
    ap.add_argument("--c-scale", type=float, default=0.1,
                    help="--label-target gumbel: sigma's c_scale (0.1, the "
                         "shipped mctx default; Go/chess use 1.0, and our Q "
                         "is a min-max normalised window score, not theirs)")
    ap.add_argument("--share", type=float, default=0.25,
                    help="share of the BC loss mass the relabelled rows "
                         "carry together (0 = raw divergence factors)")
    ap.add_argument("--div-scale", type=float, default=256.0,
                    help="map units of divergence from the elite line "
                         "that double a row's weight")
    ap.add_argument("--div-cap", type=float, default=3.0,
                    help="cap on the divergence factor (weight <= 1 + cap)")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="wall-clock cap on the window phase, seconds "
                         "(0 = none); unprocessed states are reported")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-config-audit", action="store_true")
    return ap


def run_relabel(args) -> dict:
    import torch
    t_all = time.time()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) * 1000003 + 17)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = out.with_name(out.stem + "_rows.npz")
    samples_out = out.with_name(out.stem + "_samples.npz")
    summ_out = Path(args.summary_out) if args.summary_out \
        else out.with_name(out.stem + "_summary.json")

    B = load_bundle(args.ckpt, args.map, device,
                    audit=not args.no_config_audit)
    K, gf = B["K"], B["gf"]
    TICK = B["tick"]
    say(f"ckpt {args.ckpt} step {B['step']:,} act_every {K} n_latch "
        f"{B['n_latch']} obs_reward {B['obs_reward']} device {device} map "
        f"{B['map_path']}")
    say("physics " + TICK.describe())
    # every seconds flag converts at the CORE's tick (surfgym.dagger.
    # core_clock), not at a 100 Hz literal
    every_ticks = max(1, TICK.secs_to_ticks(args.every, "round"))
    bank = SampleBank()
    phase = {}

    # ---- phase A: the policy's own states ---------------------------
    t0 = time.time()
    n_ms = int(args.episodes) + int(args.stoch_episodes)
    if n_ms > 0:
        core = open_core(B, n_ms)
        obs = core.reset(int(args.seed))
        feeds = make_feeds(B)
        mask = np.arange(n_ms) < int(args.episodes)
        dec = make_decider(B, core, feeds, mask, args.temp, gen)
        n_ticks = (B["ep_cap"] if args.rollout_secs <= 0
                   else TICK.secs_to_ticks(args.rollout_secs, "round"))
        src = np.where(mask, SRC_GREEDY, SRC_STOCH)
        # decision 0 of a fresh-feed rollout reads slot 12 = 0, which no
        # primed restart reproduces (and the spawn is the elite line's own
        # start): sample from the second grid decision on
        info = collect_rollout_samples(core, dec, feeds, obs, K, n_ticks,
                                       every_ticks, src, np.arange(n_ms),
                                       bank, skip_first=feeds[0] is not None)
        phase["map_start"] = {
            "envs": n_ms, "greedy": int(args.episodes),
            "stoch": int(args.stoch_episodes), "temp": float(args.temp),
            "finished": int(info["finished"].sum()),
            "end_ticks": [int(x) for x in info["end_tick"]],
            "recorded": info["recorded"]}
        say(f"map-start rollouts: {n_ms} envs ({int(args.episodes)} greedy "
            f"+ {int(args.stoch_episodes)} at temp {args.temp:g}), "
            f"{int(info['finished'].sum())} finished, end ticks "
            f"{[int(x) for x in info['end_tick']]}, {info['recorded']} "
            f"states ({time.time() - t0:.0f}s)")
        del core, dec
    n_sp = int(args.spine_spawns)
    ez = np.load(args.bc, allow_pickle=False)
    e_lid = np.asarray(ez["line_id"])
    e_sel = np.nonzero(e_lid == 0)[0]
    e_st = np.asarray(ez["states"])[e_sel]
    e_sc = np.asarray(ez["scal"], np.float32)[e_sel]
    e_la = np.asarray(ez["latch"], np.float32)[e_sel]
    e_w_sum = float(np.asarray(ez["weights"], np.float32).sum())
    if len(e_st) > 1 and not bool(np.all(np.diff(e_st["tick"]) > 0)):
        raise SystemExit(f"{args.bc}: line 0's rows are not tick-ordered")
    if n_sp > 0 and len(e_st) >= 4:
        t1 = time.time()
        n_el = len(e_st)
        lo = max(1, int(0.05 * n_el))
        hi = max(lo + 1, int(0.92 * n_el))
        picks = lo + even_subset(hi - lo, n_sp)
        n_sp = int(len(picks))
        core = open_core(B, n_sp)
        obs = np.array(core.reset(int(args.seed) + 1))
        for i, p in enumerate(picks):
            core.set_state(i, e_st[p])
            obs[i, :N_SCALAR] = e_sc[p]
        feeds = make_feeds(B)
        rf, lf = feeds
        prev = picks - 1
        if rf is not None:
            dp = np.asarray(gf.sample(e_st[prev]["origin"]), np.float64)
            if B["d_floor"] > 0.0:
                dp = np.maximum(dp, B["d_floor"])
            rf.state["d"] = dp
        if lf is not None:
            lf.state["f"] = e_la[prev] > 0.5
            lf.state["tick"] = np.asarray(e_st[prev]["tick"], np.int64).copy()
        mask = np.arange(n_sp) < (n_sp + 1) // 2
        dec = make_decider(B, core, feeds, mask, args.temp, gen)
        info = collect_rollout_samples(
            core, dec, feeds, obs, K,
            TICK.secs_to_ticks(args.spine_secs, "round"),
            every_ticks, SRC_SPINE, picks, bank)
        phase["spine"] = {
            "envs": n_sp, "picks": [int(p) for p in picks],
            "pick_ticks": [int(x) for x in e_st[picks]["tick"]],
            "secs": float(args.spine_secs),
            "finished": int(info["finished"].sum()),
            "end_ticks": [int(x) for x in info["end_tick"]],
            "recorded": info["recorded"]}
        say(f"spine rollouts: {n_sp} spawns at decisions {picks[0]}..{picks[-1]}"
            f" of {n_el}, {args.spine_secs:g}s each, {info['recorded']} states "
            f"({time.time() - t1:.0f}s)")
        del core, dec
    n_cand = len(bank)
    if n_cand == 0:
        raise SystemExit("no candidate states (every rollout ended before "
                         "its first grid decision?)")
    A = bank.arrays()
    cand_by_src = {SRC_NAMES[s]: int((A["src"] == s).sum()) for s in range(3)}
    # candidates in (source, rollout, time) order, then one random pick per
    # stratum: every rollout contributes in proportion to its length and no
    # stride can alias with the rollouts' interleaving
    order = np.lexsort((A["tick"], A["ep"], A["src"]))
    idx = order[even_subset(n_cand, int(args.k),
                            np.random.default_rng(int(args.seed) + 11))]
    samples = bank.select(np.sort(idx))
    samples.save(samples_out)
    SA = samples.arrays()
    chosen_by_src = {SRC_NAMES[s]: int((SA["src"] == s).sum())
                     for s in range(3)}
    phase["sample_wall_s"] = round(time.time() - t0, 1)
    say(f"candidates {n_cand} {cand_by_src} -> {len(samples)} sampled "
        f"{chosen_by_src} every {args.every:g}s ({phase['sample_wall_s']}s)")

    # ---- phase B: short planner windows ---------------------------------
    t0 = time.time()
    n_envs = int(args.envs)
    copies = int(min(args.copies, n_envs))
    if n_envs % copies:
        say(f"WARNING: --envs {n_envs} is not a multiple of --copies "
            f"{copies}: {n_envs % copies} envs idle")
    H = max(1, int(round(args.window * TICK.hz / K)))
    core = open_core(B, n_envs)
    holder = {"decider": None, "feeds": None}
    value_fn = make_value_fn(holder) if args.score in ("v", "dv") else None
    scorer = make_grouped_scorer(gf, args.score, value_fn, args.v_switch,
                                 copies, B["d_floor"])

    def _mk_decider(feeds, greedy_mask):
        dec = make_decider(B, core, feeds, greedy_mask, args.plan_temp, gen,
                           keep_logits=(args.label_target == "gumbel"))
        holder["decider"], holder["feeds"] = dec, feeds
        return dec

    say(f"windows: {len(samples)} states x {copies} copies ({n_envs // copies}"
        f" per chunk of {n_envs} envs), {H} decisions ({H * K} ticks = "
        f"{TICK.ticks_to_secs(H * K):.2f}s), resample every "
        f"{args.resample}, "
        f"elite {max(1, int(round(copies * args.elite_frac)))}, greedy "
        f"{args.greedy_envs}, score {args.score}, plan temp {args.plan_temp:g}"
        f", label target {args.label_target}"
        + (f", budget {args.budget:.0f}s" if args.budget > 0 else ""))
    results = relabel_windows(core, _mk_decider, lambda: make_feeds(B), gf,
                              scorer, samples, K, copies, H, args.resample,
                              args.elite_frac, args.greedy_envs,
                              label_decisions=args.label_decisions,
                              seed=int(args.seed) + 7, budget_s=args.budget,
                              d_floor=B["d_floor"],
                              label_target=args.label_target,
                              c_visit=args.c_visit, c_scale=args.c_scale,
                              log=say)
    phase["window_wall_s"] = round(time.time() - t0, 1)
    rs = summarize_results(results)
    say(f"windows done: {rs['labelled']}/{rs['samples']} labelled, "
        f"{rs['finished']} finished, {rs['extinct']} extinct, "
        f"{rs['unprocessed']} unprocessed; disagree "
        f"{rs['disagree']} ({(rs['disagree_rate'] or 0) * 100:.1f}%), "
        f"planner gain in d mean {rs['gain_d_mean']} median "
        f"{rs['gain_d_median']} (positive {rs['gain_positive']}); target "
        f"top1 {rs['target_top1_mean']} entropy {rs['target_entropy_mean']} "
        f"over {rs['target_copies_mean']} copies "
        f"({phase['window_wall_s']}s)")

    # ---- phase C: weights, rows, the merged file -----------------------
    t0 = time.time()
    spine = np.load(args.spine, allow_pickle=False)
    dist = nearest_distance(SA["states"]["origin"], spine["origin"])
    w = divergence_weights(dist, e_w_sum, args.share, args.div_scale,
                           args.div_cap)
    rows = rows_from_results(results, w, samples)
    if rows is not None:
        check_actions(rows["actions"])
    dagger_meta = {
        "ckpt": str(args.ckpt), "ckpt_md5": md5(args.ckpt),
        "ckpt_step": B["step"], "map": B["map_path"], "act_every": K,
        "k": int(args.k), "every_s": float(args.every),
        "window_s": float(args.window), "window_decisions": H,
        # the time base every *_s field above was converted at, so a reader
        # never has to assume 100 Hz (beam_tas.tick_stamp's keys)
        "tick_ms": float(TICK.ms), "every_ticks": int(every_ticks),
        "tick_pattern_ms": [int(v) for v in TICK.pattern],
        "envs": n_envs, "copies": copies, "greedy_envs": int(args.greedy_envs),
        "resample": int(args.resample), "elite_frac": float(args.elite_frac),
        "score": args.score, "v_switch": float(args.v_switch),
        "plan_temp": float(args.plan_temp), "temp": float(args.temp),
        "label_decisions": int(args.label_decisions),
        "label_target": str(args.label_target),
        "c_visit": float(args.c_visit), "c_scale": float(args.c_scale),
        "share": float(args.share), "div_scale": float(args.div_scale),
        "div_cap": float(args.div_cap), "seed": int(args.seed),
        "device": str(device), "candidates": n_cand,
        "candidates_by_src": cand_by_src, "sampled": int(len(samples)),
        "sampled_by_src": chosen_by_src,
        "divergence_u": {"min": float(dist.min()), "median": float(np.median(dist)),
                         "mean": float(dist.mean()), "max": float(dist.max()),
                         "p90": float(np.percentile(dist, 90))},
        "weights": {"min": float(w.min()), "mean": float(w.mean()),
                    "max": float(w.max()), "sum": float(w.sum()),
                    "elite_sum": e_w_sum},
        "results": rs,
        "row0_mismatch": (0 if rows is None else rows["row0_mismatch"]),
        "phase": phase, "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if rows is not None:
        save_bc_dataset(rows_out, rows["states"], rows["scal"], rows["latch"],
                        rows["actions"], rows["weights"],
                        np.full(len(rows["states"]), -1, np.int32),
                        {"obs_reward": B["obs_reward"], "n_latch": B["n_latch"],
                         "act_every": K, "kind": "dagger_rows", "lines": 0,
                         "dagger": dagger_meta},
                        probs=rows["probs"], zret=rows["zret"],
                        zmask=rows["zmask"])
    meta = merge_bc_datasets(args.bc, rows, out, dagger_meta, B["n_latch"],
                             B["obs_reward"])
    phase["merge_wall_s"] = round(time.time() - t0, 1)
    summary = {
        "out": str(out), "rows_out": str(rows_out),
        "samples_out": str(samples_out), "elite": str(args.bc),
        "spine": str(args.spine), "rows_elite": meta["rows_elite"],
        "rows_dagger": meta["rows_dagger"], "rows_total": meta["rows"],
        "weight_elite": meta["weight_elite"],
        "weight_dagger": meta["weight_dagger"],
        "labelled": rs["labelled"], "k": int(len(samples)),
        "finished": rs["finished"], "extinct": rs["extinct"],
        "unprocessed": rs["unprocessed"], "disagree": rs["disagree"],
        "disagree_rate": rs["disagree_rate"],
        "gain_d_mean": rs["gain_d_mean"], "gain_d_median": rs["gain_d_median"],
        "row0_mismatch": dagger_meta["row0_mismatch"],
        "divergence_u": dagger_meta["divergence_u"],
        "weights": dagger_meta["weights"],
        "candidates": n_cand, "candidates_by_src": cand_by_src,
        "sampled_by_src": chosen_by_src, "phase": phase,
        "config": {k: dagger_meta[k] for k in (
            "window_s", "window_decisions", "envs", "copies", "greedy_envs",
            "resample", "elite_frac", "score", "v_switch", "plan_temp",
            "temp", "label_decisions", "label_target", "c_visit", "c_scale",
            "share", "div_scale", "div_cap",
            "every_s", "seed", "device")},
        "wall_s": round(time.time() - t_all, 1)}
    summ_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    say(f"dagger: {meta['rows_dagger']} relabelled rows (weight "
        f"{meta['weight_dagger']:.0f} vs elite {meta['weight_elite']:.0f}) "
        f"+ {meta['rows_elite']} elite rows -> {out} ({summary['wall_s']}s "
        f"total; row0 mismatches {summary['row0_mismatch']})")
    return summary


# --------------------------------------------------------------------------
# the expert_loop hook (default off = the loop byte-for-byte)
# --------------------------------------------------------------------------
def add_args(ap: argparse.ArgumentParser) -> None:
    """The driver's DAgger flags. --dagger-k 0 (default) = no relabel phase
    and no other change to the round."""
    g = ap.add_argument_group("dagger", "DAgger relabelling from the "
                              "policy's own states (default off)")
    g.add_argument("--dagger-k", type=int, default=0,
                   help="states to relabel per round (0 = off)")
    g.add_argument("--dagger-window", type=float, default=3.0,
                   help="planner window per state, seconds")
    g.add_argument("--dagger-budget", type=float, default=600.0,
                   help="wall-clock cap on the window phase per round, s")
    g.add_argument("--dagger-envs", type=int, default=2048,
                   help="envs of the relabel search core")
    g.add_argument("--dagger-copies", type=int, default=256,
                   help="clones per state")
    g.add_argument("--dagger-share", type=float, default=0.25,
                   help="share of the BC loss mass the relabelled rows carry")
    g.add_argument("--dagger-label-target", choices=list(LABEL_TARGETS),
                   default="count",
                   help="expert_dagger --label-target (P2: what the stored "
                        "per-head distribution is; the argmax label is "
                        "stored either way)")
    g.add_argument("--dagger-extra", default="",
                   help="extra flags for tools/expert_dagger.py, one string")


def maybe_relabel(args, policy, rdir, map_path, bc, spine, bmeta, fh, run,
                  log):
    """The relabel phase of a round, between distil and train. Returns
    ``(bc, bmeta)`` unchanged when --dagger-k is 0; otherwise runs
    tools/expert_dagger.py as a subprocess (dagger.log) and returns the
    merged file plus bmeta with ``rows`` = total rows and ``dagger`` = the
    phase summary (round.json carries it through bmeta)."""
    k = int(getattr(args, "dagger_k", 0) or 0)
    if k <= 0:
        return bc, bmeta
    rdir = Path(rdir)
    out = rdir / "bc_dagger.npz"
    summ = rdir / "dagger_summary.json"
    try:
        rseed = int(str(rdir.name).split("_")[-1])
    except ValueError:
        rseed = 0
    cmd = [PY, "-u", ROOT / "tools" / "expert_dagger.py", policy,
           "--bc", bc, "--spine", spine, "--out", out, "--map", map_path,
           "--k", k, "--window", args.dagger_window,
           "--budget", args.dagger_budget, "--envs", args.dagger_envs,
           "--copies", args.dagger_copies, "--share", args.dagger_share,
           "--label-target", getattr(args, "dagger_label_target", "count"),
           "--seed", rseed, "--summary-out", summ]
    cmd += shlex.split(getattr(args, "dagger_extra", "") or "")
    t0 = time.time()
    rc = run(cmd, rdir / "dagger.log", max(1800, int(args.dagger_budget * 3)))
    if rc != 0 or not out.exists() or not summ.exists():
        raise SystemExit(f"expert_dagger failed (rc={rc}, see dagger.log)")
    s = json.loads(summ.read_text(encoding="utf-8"))
    s["wall_s"] = round(time.time() - t0, 1)
    bmeta = dict(bmeta)
    bmeta["rows_elite"] = bmeta.get("rows")
    bmeta["rows"] = int(s["rows_total"])
    bmeta["dagger"] = s
    log(fh, f"dagger: {s['rows_dagger']} relabelled rows from "
            f"{s['labelled']}/{s['k']} states ({s['finished']} finished, "
            f"{s['extinct']} extinct, disagree "
            f"{(s.get('disagree_rate') or 0) * 100:.1f}%, gain d median "
            f"{s.get('gain_d_median')}), weight {s['weight_dagger']:.0f} vs "
            f"elite {s['weight_elite']:.0f}, total {s['rows_total']:,} rows "
            f"({s['wall_s']}s)")
    return out, bmeta


def main() -> int:
    args = build_parser().parse_args()
    run_relabel(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
