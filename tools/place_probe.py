"""place_probe.py - placement + entry-velocity screens for a stuck policy.

Two screens, one process, N parallel envs.

  SCREEN 0 (placement).  Put the policy at states it has ACTUALLY reached -
  its own respawn reservoir - inside a depth band past its greedy frontier,
  and count finishes.  If it cannot finish from past the wall, the wall is a
  downstream-competence problem and an entry-velocity probe there answers
  nothing (round 19 ran one at the 63% wall and got 0/48 with an empty
  reservoir behind it: an uninterpretable null).  At the 20% wall the same
  screen finished 12/12, which is what licensed the velocity probe that
  found the ~1,550 u/s gate.

  SCREEN 1 (entry velocity).  Same states, velocity scaled by k, direction
  and position untouched - `RespawnBuffer.build_pool`'s own perturbation, so
  a rung IS what `--respawn-speed k k` would place.  A threshold in finishes
  across k is a speed gate; a flat line is not.

Both screens run the eval semantics the trainer uses: greedy argmax,
`--obs-reward` slot-12 mirror, teleport-fail, goal box armed, and NO
stall-kill (core.force_fail has one call site and it is in the training
rollout - an eval episode runs to the tick cap).

    python tools/place_probe.py CKPT --band 68.8 100 --episodes 96
    python tools/place_probe.py CKPT --band 60 68.8 --vel-scale 1 1.5 2 2.5

Reports, per band/rung: episodes, finishes, and the frontier as % of d0
(d0 = mean geodesic over the map's own spawn entities, the same number
race/eval_progress is normalized by).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import torch

from surfgym import STATE_DTYPE, SurfCore, default_config
from surfgym.rewards import map_spawn_pool
from train_fast import GreedyTorchPolicy, HeadPacker, Policy


def build(ckpt_path, map_path=None, envs=256, ep_ticks=None, device=None):
    """Load a checkpoint into a running N-env sim + greedy eval policy.

    Mirrors tools/record_ckpt.py's setup for the flags that change what an
    observation MEANS (--obs-reward, --yaw-adaptive, --maxvel, lidar
    geometry).  Refuses the checkpoint outright for the flags this file does
    not mirror, rather than silently probing a different policy.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    for k in ("route_file", "chunk", "frame_stack", "race_latch"):
        if cfg.get(k):
            raise SystemExit(
                f"this checkpoint sets {k}={cfg[k]!r}; place_probe does not "
                "mirror that observation and will not guess")
    if cfg.get("reward") != "race" or cfg.get("race_dist") != "geodesic":
        raise SystemExit("place_probe needs a geodesic race checkpoint")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg_map = cfg.get("map", "surf_ski_2")
    map_path = str(map_path or (ROOT / "maps" / f"{cfg_map}.bsp"))
    lw, lh = int(cfg.get("lidar_w", 64)), int(cfg.get("lidar_h", 32))
    fix_pitch = cfg.get("fix_pitch")
    pitch_rate = 0.0 if fix_pitch is not None else float(cfg.get("pitch_rate", -1.0))
    ep_ticks = int(ep_ticks or cfg.get("ep_ticks", 12000))

    core = SurfCore(map_path, default_config(
        num_envs=int(envs), spawn_mode=2, max_episode_ticks=ep_ticks,
        water_fail=1, sv_maxvelocity=float(cfg.get("maxvel", 2000.0)),
        yaw_adaptive=1 if cfg.get("yaw_adaptive") else 0,
        lidar_w=0, lidar_h=0, pitch_rate_max_deg=pitch_rate))

    from surfgym.goalfield import build_goal_field
    from surfgym.vision import GpuLidar, pick_cell
    from surfgym.zones import load_zones

    cell = float(cfg.get("lidar_cell") or pick_cell(core))
    zones = load_zones(core.bsp_path)
    field = build_goal_field(core, zones["end"], cell=cell)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    core.set_teleport_fail(True)

    raw = map_spawn_pool(core)
    start_pool = map_spawn_pool(core, yaw=field.descent_yaw(raw["origin"]))
    start_pool["pitch"] = -10.0
    if fix_pitch is not None:
        start_pool["pitch"] = float(fix_pitch)
    d0 = float(np.mean(field.sample(raw["origin"])))

    lidar = GpuLidar(core, lw, lh,
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"), cell=cell, device=dev,
                     surf_mask=bool(cfg.get("surf_mask", 0)),
                     pinhole=bool(cfg.get("pinhole", 0)))

    extra = (12,) if cfg.get("obs_reward") else ()
    policy = Policy(core.obs_dim + lw * lh * lidar.channels, lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)), extra_feat=extra,
                    in_ch=lidar.channels).to(dev)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    act_every = int(cfg.get("act_every", 1))
    extra_slot, extra_fn = -1, None
    if cfg.get("obs_reward"):
        # the trainer's own eval mirror (_make_eval_reward_feed): slot 12
        # carries tanh((geodesic delta * 100/d0 * shaping - time_pen*k)/0.1)
        scale = 100.0 / max(d0, 1.0) * float(cfg.get("race_shaping") or 1.0)
        tp = float(cfg.get("time_pen") or 0.005)
        d_floor = float(cfg.get("race_dfloor") or 0.0)
        st = {"d": None}

        def _feed(c, _f=field, _s=scale, _tp=tp, _k=act_every, _fl=d_floor):
            d = _f.sample(c.states_view["origin"]).astype(np.float64)
            if _fl > 0.0:
                d = np.maximum(d, _fl)
            prev, st["d"] = st["d"], d
            if prev is None or len(prev) != len(d):
                return np.zeros(len(d), np.float32)
            delta = np.clip(prev - d, -100.0 * _k, 100.0 * _k)
            return np.tanh((delta * _s - _tp * _k) / 0.1).astype(np.float32)

        extra_slot, extra_fn = 12, _feed

    greedy = GreedyTorchPolicy(policy, HeadPacker(dev), dev, lidar, core,
                              act_every, 1, extra_slot=extra_slot,
                              extra_fn=extra_fn)
    return dict(ck=ck, cfg=cfg, core=core, field=field, policy=greedy,
                d0=d0, start_pool=start_pool, ep_ticks=ep_ticks,
                step=int(ck.get("global_step", 0)))


def reservoir(ck, field, d0):
    """The checkpoint's own respawn reservoir + each state's % of d0."""
    r = ck.get("respawn") or {}
    S = r.get("states")
    if S is None or len(S) == 0:
        raise SystemExit("this checkpoint carries no respawn reservoir")
    S = np.asarray(S)
    d = field.sample(np.asarray(S["origin"], np.float64))
    return S, (d0 - d) / d0 * 100.0


def run(env, pool, episodes, seed=0, label=""):
    """Roll greedy episodes from `pool` until `episodes` have ENDED.

    Returns one record per ended episode: where it spawned, how far it got
    (both as % of d0), whether it crossed the goal box, and its length.
    Same-step autoreset means every env is always inside an episode, so the
    tail past `episodes` is simply discarded.
    """
    core, field, policy, d0 = env["core"], env["field"], env["policy"], env["d0"]
    n = core.num_envs
    core.set_spawn_pool(pool)
    obs = core.reset(seed)
    policy._held = None
    policy._tick = 0

    spawn_pc = np.full(n, np.nan)
    spawn_v = np.zeros(n)
    best_pc = np.full(n, -1e9)
    best_v = np.zeros(n)                  # speed AT the frontier point
    best_xyz = np.zeros((n, 3))
    ticks = np.zeros(n, np.int64)
    out = []
    # a fresh episode can be no longer than the cap; +2 covers the first
    # partial one every env is in when the batch starts
    budget = (episodes // max(n, 1) + 2) * env["ep_ticks"]
    for _ in range(budget):
        sv = core.states_view          # zero-copy, pre-step (s_t)
        org = np.asarray(sv["origin"], np.float64)
        pc = (d0 - field.sample(org)) / d0 * 100.0
        spd = np.linalg.norm(np.asarray(sv["velocity"], np.float64), axis=1)
        fresh = np.isnan(spawn_pc)
        spawn_pc = np.where(fresh, pc, spawn_pc)
        spawn_v = np.where(fresh, spd, spawn_v)
        adv = pc > best_pc
        best_v = np.where(adv, spd, best_v)
        best_xyz[adv] = org[adv]
        best_pc = np.maximum(best_pc, pc)
        obs, _rew, done, trunc, _term = core.step(policy.act(obs))
        hit = np.asarray(core.goal_hits, np.uint8).copy()
        ended = np.asarray(done, bool) | np.asarray(trunc, bool)
        ticks += 1
        if ended.any():
            for i in np.flatnonzero(ended):
                out.append(dict(spawn=float(spawn_pc[i]), best=float(best_pc[i]),
                                fin=bool(hit[i]), ticks=int(ticks[i]),
                                v_spawn=float(spawn_v[i]),
                                v_best=float(best_v[i]),
                                xyz=[round(float(v), 1) for v in best_xyz[i]],
                                label=label))
            spawn_pc[ended] = np.nan
            best_pc[ended] = -1e9
            ticks[ended] = 0
            if len(out) >= episodes:
                break
    return out[:episodes]


def capture(env, pc_at, want, seed=0, max_ticks=60000):
    """The policy's OWN on-ramp states: roll greedy from the map start and
    snapshot each episode the first tick it reaches `pc_at` % of d0.

    This is what the 20% probe scaled - not a reservoir sample. A reservoir
    state at the same DEPTH was put there by a training episode that may
    have been respawned upstream with a boosted velocity, so it is not the
    state this policy actually arrives in. Resuming a captured state at
    scale 1.0 continues the greedy trajectory that produced it, so the
    k = 1.0 rung is the failure itself, not an approximation of it.
    """
    core, field, policy, d0 = env["core"], env["field"], env["policy"], env["d0"]
    n = core.num_envs
    core.set_spawn_pool(env["start_pool"])
    obs = core.reset(seed)
    policy._held = None
    policy._tick = 0
    done_cap = np.zeros(n, bool)
    got = []
    for _ in range(max_ticks):
        sv = core.states_view
        pc = (d0 - field.sample(np.asarray(sv["origin"], np.float64))) / d0 * 100.0
        hit = (pc >= pc_at) & ~done_cap
        if hit.any():
            snap = core.get_states()
            for i in np.flatnonzero(hit):
                got.append(snap[i].copy())
            done_cap |= hit
            if len(got) >= want:
                break
        obs, _r, done, trunc, _t = core.step(policy.act(obs))
        ended = np.asarray(done, bool) | np.asarray(trunc, bool)
        done_cap &= ~ended          # a new episode may capture again
    if not got:
        raise SystemExit(f"greedy never reached {pc_at:g}% of d0 - nothing to "
                         "scale")
    return np.asarray(got, dtype=STATE_DTYPE)


def profile(env, marks, seed=0, max_ticks=60000):
    """One greedy roll from the start line, snapshotting every env the first
    tick it passes each mark (% of d0).  Returns {mark: STATE array}.

    The speed the policy CARRIES at each mark, against what its own
    reservoir holds at the same depth, is where a speed gate is lost - the
    gate itself is downstream of that.
    """
    core, field, policy, d0 = env["core"], env["field"], env["policy"], env["d0"]
    n = core.num_envs
    core.set_spawn_pool(env["start_pool"])
    obs = core.reset(seed)
    policy._held = None
    policy._tick = 0
    marks = sorted(float(m) for m in marks)
    got = {m: [] for m in marks}
    seen = {m: np.zeros(n, bool) for m in marks}
    for _ in range(max_ticks):
        sv = core.states_view
        pc = (d0 - field.sample(np.asarray(sv["origin"], np.float64))) / d0 * 100.0
        snap = None
        for m in marks:
            hit = (pc >= m) & ~seen[m]
            if hit.any():
                if snap is None:
                    snap = core.get_states()
                for i in np.flatnonzero(hit):
                    got[m].append(snap[i].copy())
                seen[m] |= hit
        obs, _r, done, trunc, _t = core.step(policy.act(obs))
        ended = np.asarray(done, bool) | np.asarray(trunc, bool)
        for m in marks:
            seen[m] &= ~ended
        if all(len(got[m]) >= n for m in marks):
            break
    return {m: (np.asarray(v, dtype=STATE_DTYPE) if v else None)
            for m, v in got.items()}


def summarise(recs, tag):
    if not recs:
        print(f"  {tag:<22} no episodes")
        return
    fin = sum(r["fin"] for r in recs)
    best = np.array([r["best"] for r in recs])
    spawn = np.array([r["spawn"] for r in recs])
    tk = np.array([r["ticks"] for r in recs])
    ft = tk[[r["fin"] for r in recs]]
    note = f"  finish {ft.mean() / 100.0:6.2f}s" if len(ft) else ""
    vb = np.array([r["v_best"] for r in recs])
    vs = np.array([r["v_spawn"] for r in recs])
    print(f"  {tag:<22} n {len(recs):>4}  fin {fin:>3}/{len(recs):<4} "
          f"({100.0 * fin / len(recs):5.1f}%)  spawn {spawn.mean():5.1f}%  "
          f"MAX {best.max():5.1f}%  mean {best.mean():5.1f}%  "
          f"ep {tk.mean() / 100.0:5.1f}s{note}")
    xyz = np.array([r["xyz"] for r in recs])
    print(f"  {'':<22} v@spawn {vs.mean():6.0f}  v@frontier {vb.mean():6.0f} "
          f"u/s   frontier xyz "
          f"({xyz[:, 0].mean():.0f}+-{xyz[:, 0].std():.0f}, "
          f"{xyz[:, 1].mean():.0f}+-{xyz[:, 1].std():.0f}, "
          f"{xyz[:, 2].mean():.0f}+-{xyz[:, 2].std():.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--map", default=None,
                    help="ABSOLUTE path into the main checkout when running "
                         "from a worktree, or the goal field re-bakes")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=96,
                    help="episodes PER band/rung")
    ap.add_argument("--band", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="reservoir depth band, %% of d0 (screen 1)")
    ap.add_argument("--sweep", type=float, nargs="+", default=None,
                    help="screen 0: band EDGES, %% of d0, e.g. 60 70 80 90 100")
    ap.add_argument("--vel-scale", type=float, nargs="+", default=[1.0],
                    help="screen 1: entry-velocity multipliers")
    ap.add_argument("--start", action="store_true",
                    help="also run greedy from the map's own start line")
    ap.add_argument("--profile", type=float, nargs="+", default=None,
                    help="speed the greedy policy CARRIES at these %% of d0, "
                         "against its own reservoir at the same depth")
    ap.add_argument("--capture", type=float, default=None, metavar="PC",
                    help="screen 1 on the policy's OWN on-ramp state: roll "
                         "greedy from the start line and snapshot each "
                         "episode the first tick it passes PC%% of d0, then "
                         "scale THOSE velocities (the 20%%-wall protocol)")
    ap.add_argument("--ep-ticks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="per-episode records")
    args = ap.parse_args()

    env = build(args.ckpt, args.map, args.envs, args.ep_ticks)
    print(f"ckpt {Path(args.ckpt).name}  step {env['step']:,}  "
          f"d0 {env['d0']:,.0f}u  envs {env['core'].num_envs}  "
          f"ep_ticks {env['ep_ticks']}")
    S, pc = reservoir(env["ck"], env["field"], env["d0"])
    sp = np.linalg.norm(np.asarray(S["velocity"], np.float64), axis=1)
    print(f"reservoir {len(S):,} states  depth: max {pc.max():.1f}%  "
          f"p99 {np.percentile(pc, 99):.1f}%  median {np.median(pc):.1f}%")

    all_recs = []
    if args.start:
        recs = run(env, env["start_pool"], args.episodes, args.seed, "start")
        all_recs += recs
        print("\nGREEDY FROM THE START LINE")
        summarise(recs, "start")

    if args.sweep:
        print("\nSCREEN 0 - PLACEMENT (reservoir states, own velocity)")
        edges = list(args.sweep)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pc >= lo) & (pc < hi)
            if m.sum() < 8:
                print(f"  {lo:.1f}-{hi:.1f}%".ljust(24)
                      + f"only {int(m.sum())} states - skipped")
                continue
            band = S[m].copy()
            print(f"  [{lo:.1f}-{hi:.1f}%] {int(m.sum())} states, "
                  f"speed {sp[m].mean():.0f} u/s")
            recs = run(env, band, args.episodes, args.seed + int(lo), f"{lo}-{hi}")
            all_recs += recs
            summarise(recs, f"{lo:.1f}-{hi:.1f}%")

    if args.profile:
        print("")
        print("SPEED CARRIED vs SPEED HARVESTED (greedy from the start line)")
        print("   %d0   reached   v greedy   v reservoir   ratio   "
              "greedy xyz")
        got = profile(env, args.profile, args.seed)
        for m in sorted(float(x) for x in args.profile):
            arr = got[m]
            near = np.abs(pc - m) < 1.5
            vr = sp[near].mean() if near.sum() >= 5 else float("nan")
            if arr is None:
                print(f"  {m:5.1f}   never       -            {vr:6.0f}")
                continue
            vg = np.linalg.norm(np.asarray(arr["velocity"], np.float64),
                                axis=1)
            o = np.asarray(arr["origin"], np.float64)
            print(f"  {m:5.1f}   {len(arr):4d}/{env['core'].num_envs:<4d} "
                  f"{vg.mean():8.0f}      {vr:6.0f}     "
                  f"{vg.mean() / vr if vr == vr else float('nan'):5.2f}   "
                  f"({o[:, 0].mean():6.0f},{o[:, 1].mean():6.0f},"
                  f"{o[:, 2].mean():6.0f})")

    if args.capture is not None:
        cap = capture(env, args.capture, max(args.envs, 96), args.seed)
        cs = np.linalg.norm(np.asarray(cap["velocity"], np.float64), axis=1)
        cd = env["field"].sample(np.asarray(cap["origin"], np.float64))
        cpc = (env["d0"] - cd) / env["d0"] * 100.0
        near = np.abs(pc - args.capture) < 2.0
        print("")
        print(f"SCREEN 1 - ENTRY VELOCITY on the policy's OWN state at "
              f"{args.capture:g}% of d0")
        print(f"  captured {len(cap)} states at {cpc.mean():.1f}% "
              f"({np.asarray(cap['origin'])[:, 0].mean():.0f}, "
              f"{np.asarray(cap['origin'])[:, 1].mean():.0f}, "
              f"{np.asarray(cap['origin'])[:, 2].mean():.0f}) carrying "
              f"{cs.mean():.0f} u/s;  the reservoir at that depth carries "
              f"{sp[near].mean() if near.any() else float('nan'):.0f} u/s")
        for k in args.vel_scale:
            band = cap.copy()
            band["velocity"] = band["velocity"] * np.float32(k)
            recs = run(env, band, args.episodes, args.seed + int(k * 100),
                       f"cap x{k:g}")
            all_recs += recs
            summarise(recs, f"x{k:g} ({cs.mean() * k:.0f} u/s)")

    if args.band:
        lo, hi = args.band
        m = (pc >= lo) & (pc < hi)
        print(f"\nSCREEN 1 - ENTRY VELOCITY at [{lo:.1f}-{hi:.1f}%] "
              f"({int(m.sum())} states, own speed {sp[m].mean():.0f} u/s)")
        if m.sum() < 8:
            raise SystemExit("band holds too few reservoir states to probe")
        for k in args.vel_scale:
            band = S[m].copy()
            band["velocity"] = band["velocity"] * np.float32(k)
            recs = run(env, band, args.episodes, args.seed + int(k * 100),
                       f"x{k:g}")
            all_recs += recs
            summarise(recs, f"x{k:g} ({sp[m].mean() * k:.0f} u/s)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            "\n".join(json.dumps(r) for r in all_recs), encoding="utf-8")
        print(f"\n{len(all_recs)} episodes -> {args.json}")


if __name__ == "__main__":
    main()
