"""Record a greedy trajectory from a train_fast checkpoint, on demand.

Runs in its own process with its own env — safe to point at ckpt_latest.pt
of a training run that is still going. The output lands in the run's
directory named traj_<global_step>.jsonl, so the dashboard picks it up like
the trainer's own recordings.

    python tools\record_ckpt.py runs\marathon_10B\ckpt_latest.pt
    python tools\record_ckpt.py runs\marathon_10B\ckpt_latest.pt --episodes 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import torch

from surfgym import SurfCore, default_config
from surfgym.record import record_rollout
from surfgym.rewards import (drop_spawn_pool, map_spawn_pool,
                             platform_spawn_pool, ramp_spawn_pool)
from train_fast import (GreedyTorchPolicy, HeadPacker, Policy,
                        SampledTorchPolicy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--map", default=None, help="defaults to the ckpt's map")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default=None,
                    help="defaults to <ckpt dir>/traj_<global_step>.jsonl")
    ap.add_argument("--seed", type=int, default=None,
                    help="spawn seed (default: derived from the ckpt step, "
                         "so successive snapshots sample different spawns)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of argmax — what "
                         "rollout/ep_rew_mean actually measures")
    ap.add_argument("--spawn", choices=["platform", "ramp", "mixed",
                                        "reservoir"],
                    default=None,
                    help="spawn pool (default: the ckpt's training pool, "
                         "i.e. what rollout/ep_rew_mean averages over; "
                         "reservoir = the ckpt's respawn buffer — states "
                         "agents ACTUALLY reached, i.e. the live frontier)")
    ap.add_argument("--ep-ticks", type=int, default=None,
                    help="episode length for the recording (default: the "
                         "ckpt's training length; the policy has no episode "
                         "clock, so longer rollouts are fine)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    step = int(ck.get("global_step", 0))
    map_path = args.map or str(ROOT / "maps" / f"{cfg.get('map', 'surf_ski_2')}.bsp")
    ep_ticks = int(args.ep_ticks or cfg.get("ep_ticks", 700))
    if cfg.get("reward") == "race" and ep_ticks < int(cfg.get("ep_ticks", 0)):
        # a race episode runs until the finish (or the cap) — a shorter
        # recording cap (dashboard hand-records pass 3000) would cut runs off
        ep_ticks = int(cfg["ep_ticks"])
        print(f"race ckpt: episode cap restored to {ep_ticks} ticks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    fix_pitch = cfg.get("fix_pitch")
    pitch_rate = 0.0 if fix_pitch is not None else float(cfg.get("pitch_rate", -1.0))
    core = SurfCore(map_path, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=ep_ticks, water_fail=1,
        sv_maxvelocity=float(cfg.get("maxvel", 2000.0)),  # physics parity
        lidar_w=0, lidar_h=0,           # eyeless core; vision is GPU-side
        pitch_rate_max_deg=pitch_rate))
    spawn = args.spawn or cfg.get("spawn", "platform")
    drop_rng = (float(cfg.get("drop_min", 400.0)),
                float(cfg.get("drop_max", 800.0)))
    punch = (float(cfg.get("punch_min", 100.0)),
             float(cfg.get("punch_max", 400.0)))
    from surfgym.vision import GpuLidar, pick_cell
    cell = float(cfg.get("lidar_cell") or pick_cell(core))
    gf = None
    if cfg.get("reward") == "race":
        # finish zone is armed for ANY race recording, whatever the spawns
        from surfgym.goalfield import EuclidField, build_goal_field
        from surfgym.zones import load_zones
        zones = load_zones(core.bsp_path)
        gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
              else build_goal_field(core, zones["end"], cell=cell))
        core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])

    def race_start_pool():
        raw = map_spawn_pool(core)
        p = map_spawn_pool(core, yaw=gf.descent_yaw(raw["origin"]))
        p["pitch"] = -10.0
        print(f"race: start geodesic "
              f"{float(np.mean(gf.sample(raw['origin']))):.0f}u")
        return p

    if gf is not None and args.spawn is None:
        # race default: the run is judged from the map's real start line
        spawn = "start"
        pool = race_start_pool()
    elif spawn == "reservoir":
        rs = (ck.get("respawn") or {}).get("states")
        if rs is None or len(rs) == 0:
            raise SystemExit("this ckpt has no respawn reservoir "
                             "(run trained without --respawn-frac?)")
        pool = np.asarray(rs)
        print(f"reservoir pool: {len(pool)} frontier states from the ckpt")
    elif spawn in ("ramp", "mixed"):
        dp = drop_spawn_pool(core, h_range=drop_rng, speed_range=punch)
        if gf is not None:
            keep = gf.reachable(dp["origin"]) & (gf.sample(dp["origin"]) > 400.0)
            dp = dp[keep]              # training parity: on-track drops only
        if spawn == "mixed":
            base = race_start_pool() if gf is not None \
                else platform_spawn_pool(core)
            # the env resets by UNIFORM pool draw, so entry counts are the
            # probabilities: a few start entries beside thousands of drops
            # is not a mix, it is drops. Replicate to a real 50/50.
            reps = max(1, int(round(len(dp) / max(len(base), 1))))
            pool = np.concatenate([np.concatenate([base] * reps), dp])
        else:
            pool = dp
    else:
        # a race ckpt trains from the map's start entities, not the
        # walk-off-the-edge audition pool: asking for 'platform' on one
        # used to raise 'no edge-facing-ramp spawn found'
        pool = race_start_pool() if gf is not None else platform_spawn_pool(core)
    if fix_pitch is not None:
        pool["pitch"] = float(fix_pitch)
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)     # eval parity with training semantics
    core.set_spawn_pool(pool)
    print(f"spawn pool: {spawn} ({len(pool)} points)"
          + (f", pitch fixed {fix_pitch:g}" if fix_pitch is not None else ""))

    lidar = GpuLidar(core, lw, lh,
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"),
                     cell=cell,
                     device=device,
                     surf_mask=bool(cfg.get("surf_mask", 0)),
                     pinhole=bool(cfg.get("pinhole", 0)))
    # --frame-stack: the recording policy keeps its own ring (see
    # _TorchPolicyBase._push_frame), so a stacked ckpt records honestly
    stack = max(1, int(cfg.get("frame_stack") or 1))
    policy = Policy(core.obs_dim + lw * lh * lidar.channels * stack, lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    in_ch=lidar.channels * stack).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    suffix = f"_{args.spawn}" if args.spawn else ""
    suffix += "_stoch" if args.stochastic else ""
    out = Path(args.out) if args.out else \
        Path(args.ckpt).parent / f"traj_{step:010d}{suffix}.jsonl"
    seed = args.seed if args.seed is not None else step & 0x7FFFFFFF
    cls = SampledTorchPolicy if args.stochastic else GreedyTorchPolicy
    act_every = int(cfg.get("act_every", 1))
    record_rollout(core, cls(policy, HeadPacker(device), device, lidar, core,
                             act_every, stack),
                   out, episodes=args.episodes, max_ticks=args.episodes * ep_ticks,
                   seed=seed)
    kind = "stochastic" if args.stochastic else "greedy"
    print(f"recorded {args.episodes} {kind} episode(s) at step {step:,} -> {out}")


if __name__ == "__main__":
    main()
