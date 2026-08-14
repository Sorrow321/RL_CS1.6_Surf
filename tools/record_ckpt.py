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
from surfgym.rewards import (drop_spawn_pool, platform_spawn_pool,
                             ramp_spawn_pool)
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
    ap.add_argument("--spawn", choices=["platform", "ramp", "mixed"],
                    default=None,
                    help="spawn pool (default: the ckpt's training pool, "
                         "i.e. what rollout/ep_rew_mean averages over)")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    fix_pitch = cfg.get("fix_pitch")
    pitch_rate = 0.0 if fix_pitch is not None else float(cfg.get("pitch_rate", -1.0))
    core = SurfCore(map_path, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=ep_ticks, water_fail=1,
        lidar_w=0, lidar_h=0,           # eyeless core; vision is GPU-side
        pitch_rate_max_deg=pitch_rate))
    spawn = args.spawn or cfg.get("spawn", "platform")
    drop_rng = (float(cfg.get("drop_min", 400.0)),
                float(cfg.get("drop_max", 800.0)))
    punch = (float(cfg.get("punch_min", 100.0)),
             float(cfg.get("punch_max", 400.0)))
    if spawn == "ramp":
        pool = drop_spawn_pool(core, h_range=drop_rng, speed_range=punch)
    elif spawn == "mixed":
        pool = np.concatenate([platform_spawn_pool(core),
                               drop_spawn_pool(core, h_range=drop_rng,
                                               speed_range=punch)])
    else:
        pool = platform_spawn_pool(core)
    if fix_pitch is not None:
        pool["pitch"] = float(fix_pitch)
    if cfg.get("teleport_fail"):
        core.set_teleport_fail(True)     # eval parity with training semantics
    core.set_spawn_pool(pool)
    print(f"spawn pool: {spawn} ({len(pool)} points)"
          + (f", pitch fixed {fix_pitch:g}" if fix_pitch is not None else ""))

    from surfgym.vision import GpuLidar
    lidar = GpuLidar(core, lw, lh,
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"),
                     device=device)
    policy = Policy(core.obs_dim + lw * lh, lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True))).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    suffix = "_stoch" if args.stochastic else ""
    out = Path(args.out) if args.out else \
        Path(args.ckpt).parent / f"traj_{step:010d}{suffix}.jsonl"
    seed = args.seed if args.seed is not None else step & 0x7FFFFFFF
    cls = SampledTorchPolicy if args.stochastic else GreedyTorchPolicy
    act_every = int(cfg.get("act_every", 1))
    record_rollout(core, cls(policy, HeadPacker(device), device, lidar, core,
                             act_every),
                   out, episodes=args.episodes, max_ticks=args.episodes * ep_ticks,
                   seed=seed)
    kind = "stochastic" if args.stochastic else "greedy"
    print(f"recorded {args.episodes} {kind} episode(s) at step {step:,} -> {out}")


if __name__ == "__main__":
    main()
