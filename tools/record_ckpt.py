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

import torch

from surfgym import SurfCore, default_config
from surfgym.record import record_rollout
from surfgym.rewards import platform_spawn_pool, ramp_spawn_pool
from train_fast import GreedyTorchPolicy, HeadPacker, Policy, sample_padded


class SampledTorchPolicy:
    """Acts by sampling the policy distribution — the policy training actually
    optimizes and measures. Under a high entropy coefficient the argmax mode
    can be much weaker than the sampled policy (it drifts unoptimized while
    the stochastic policy learns to rely on its own action noise)."""

    def __init__(self, policy, packer, device):
        self.policy, self.packer, self.device = policy, packer, device

    @torch.inference_mode()
    def act(self, obs):
        import numpy as np
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, _ = self.policy(t)
        act, _ = sample_padded(self.packer.pad(logits))
        return act.to("cpu").numpy().astype(np.int32)


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
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    step = int(ck.get("global_step", 0))
    map_path = args.map or str(ROOT / "maps" / f"{cfg.get('map', 'surf_ski_2')}.bsp")
    ep_ticks = int(cfg.get("ep_ticks", 700))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = SurfCore(map_path, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=ep_ticks, water_fail=1))
    pool = (ramp_spawn_pool(core) if cfg.get("spawn") == "ramp"
            else platform_spawn_pool(core))
    core.set_spawn_pool(pool)

    policy = Policy(core.obs_dim).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    out = Path(args.out) if args.out else \
        Path(args.ckpt).parent / f"traj_{step:010d}.jsonl"
    seed = args.seed if args.seed is not None else step & 0x7FFFFFFF
    cls = SampledTorchPolicy if args.stochastic else GreedyTorchPolicy
    record_rollout(core, cls(policy, HeadPacker(device), device),
                   out, episodes=args.episodes, max_ticks=args.episodes * ep_ticks,
                   seed=seed)
    kind = "stochastic" if args.stochastic else "greedy"
    print(f"recorded {args.episodes} {kind} episode(s) at step {step:,} -> {out}")


if __name__ == "__main__":
    main()
