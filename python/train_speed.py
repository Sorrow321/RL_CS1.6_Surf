"""train_speed.py — first training task: maximize speed on a ramp (docs/09).

Spawns from a scanned pool of ramp faces (30u above, facing down-slope, zero
velocity), 5-second episodes, reward = per-tick horizontal speed delta
(telescopes to final speed). PPO via stable-baselines3.

    python python\\train_speed.py                    # defaults: 512 envs, 20M steps
    python python\\train_speed.py --steps 2e6 --run smoke
    tensorboard --logdir runs\\tb

Artifacts land in runs/<run>/: checkpoints, final model, and periodic
greedy-policy trajectories (drag into the viewer to watch it learn).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "python"))

from surfgym import SurfCore, default_config
from surfgym.rewards import SpeedReward, ramp_spawn_pool
from surfgym.record import record_rollout
from surfgym.vec_env import SurfVecEnv

EP_TICKS = 500  # 5 s at 100 Hz


def make_core(map_path: str, num_envs: int) -> SurfCore:
    cfg = default_config(
        num_envs=num_envs,
        spawn_mode=2,                 # spawn pool
        max_episode_ticks=EP_TICKS,
        water_fail=1,
        yaw_jitter_deg=8.0,
    )
    return SurfCore(map_path, cfg)


class GreedyPolicy:
    """Deterministic policy adapter for record_rollout."""

    def __init__(self, model):
        self.model = model

    def act(self, obs):
        actions, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(actions, dtype=np.int32).reshape(obs.shape[0], -1)


def final_speeds(traj_path: Path) -> list[float]:
    import json
    speeds, last = [], None
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, list):
                last = (row[4] ** 2 + row[5] ** 2) ** 0.5
            elif isinstance(row, dict) and "end" in row and last is not None:
                speeds.append(last)
                last = None
    return speeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_ski_2.bsp"))
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=float, default=20e6)
    ap.add_argument("--run", default=time.strftime("speed_%m%d_%H%M"))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--record-every", type=float, default=2e6,
                    help="env steps between greedy trajectory recordings")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt", default=None, help="resume from a saved model")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import VecMonitor

    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)

    core = make_core(args.map, args.envs)
    pool = ramp_spawn_pool(core)
    core.set_spawn_pool(pool)
    print(f"spawn pool: {len(pool)} ramp faces")

    venv = VecMonitor(SurfVecEnv(core, reward_fn=SpeedReward(scale=0.01)))

    # separate 1-env core for greedy eval recordings (same pool)
    eval_core = make_core(args.map, 1)
    eval_core.set_spawn_pool(pool)

    class RecordCallback(BaseCallback):
        def __init__(self, every: int):
            super().__init__()
            self.every = every
            self.next_at = 0

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_at:
                self.next_at = self.num_timesteps + self.every
                path = out / f"traj_{self.num_timesteps:010d}.jsonl"
                record_rollout(eval_core, GreedyPolicy(self.model), path,
                               episodes=3, max_ticks=3 * EP_TICKS, seed=1234)
                fs = final_speeds(path)
                mean_fs = float(np.mean(fs)) if fs else 0.0
                self.logger.record("eval/final_speed", mean_fs)
                print(f"[{self.num_timesteps:>12,d}] greedy final speed: "
                      f"{mean_fs:7.1f} u/s  ({len(fs)} eps) -> {path.name}")
            return True

    if args.ckpt:
        model = PPO.load(args.ckpt, env=venv, device=args.device)
        print(f"resumed from {args.ckpt}")
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=128, batch_size=16384, n_epochs=4,
            learning_rate=args.lr, gamma=0.995, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(ROOT / "runs" / "tb"),
            device=args.device, verbose=1,
        )

    callbacks = [
        RecordCallback(int(args.record_every)),
        CheckpointCallback(save_freq=max(1, int(2e6) // args.envs),
                           save_path=str(out), name_prefix="ppo_speed"),
    ]
    t0 = time.perf_counter()
    model.learn(total_timesteps=int(args.steps), callback=callbacks,
                tb_log_name=args.run, reset_num_timesteps=args.ckpt is None)
    dt = time.perf_counter() - t0
    model.save(out / "final")
    print(f"done: {int(args.steps):,} steps in {dt/60:.1f} min "
          f"({int(args.steps)/dt:,.0f} steps/s) -> {out / 'final.zip'}")


if __name__ == "__main__":
    main()
