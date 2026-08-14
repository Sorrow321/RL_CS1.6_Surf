"""train_fast.py — lean GPU-resident PPO for surfgym (the 10x trainer).

Same task/reward/spawns as train_speed.py, none of the per-step SB3 overhead:
one policy forward per env step (sampling + value in the same call, tensors
stay on GPU), rollout storage on GPU, fused GAE, few-epoch updates.
Measured bottlenecks it removes: SB3 predict machinery (712k -> multi-M
rollout ceiling) and the 4-epoch minibatch Python loop.

    python python\\train_fast.py --steps 100e6 --run fast1
    python python\\train_fast.py --ckpt runs\\fast1\\ckpt_latest.pt --steps 200e6
    python python\\train_fast.py --sb3 runs\\forward_100M\\final.zip --run fast_cont

Checkpoints: runs/<run>/ckpt_latest.pt (+ periodic ckpt_<steps>.pt), resumable
with optimizer state. --sb3 imports weights from a train_speed.py model zip
(same architecture) to continue its training here.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "python"))

import torch
import torch.nn as nn

from surfgym import SurfCore, default_config
from surfgym.record import record_rollout
from surfgym.rewards import (ForwardProgressReward, PathLengthReward,
                             platform_spawn_pool, ramp_spawn_pool)

NVEC = (15, 3, 3, 2, 2)
NACT = len(NVEC)


# ---------------------------------------------------------------------------
# policy (mirrors SB3 MlpPolicy net_arch=[256,256] so --sb3 import works)
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        def mlp():
            return nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(),
                                 nn.Linear(256, 256), nn.Tanh())
        self.pi = mlp()
        self.vf = mlp()
        self.action_head = nn.Linear(256, sum(NVEC))
        self.value_head = nn.Linear(256, 1)
        for m in list(self.pi) + list(self.vf):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.action_head.weight, 0.01)
        nn.init.zeros_(self.action_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs):
        logits = self.action_head(self.pi(obs))
        value = self.value_head(self.vf(obs)).squeeze(-1)
        return logits, value

    @staticmethod
    def split(logits):
        return torch.split(logits, list(NVEC), dim=-1)

    def dist_sample(self, logits):
        """Sample actions + total logprob + entropy from split categoricals."""
        acts, logps, ents = [], 0.0, 0.0
        for lg in self.split(logits):
            d = torch.distributions.Categorical(logits=lg)
            a = d.sample()
            acts.append(a)
            logps = logps + d.log_prob(a)
            ents = ents + d.entropy()
        return torch.stack(acts, dim=-1), logps, ents

    def logprob_entropy(self, logits, actions):
        logps, ents = 0.0, 0.0
        for i, lg in enumerate(self.split(logits)):
            d = torch.distributions.Categorical(logits=lg)
            logps = logps + d.log_prob(actions[..., i])
            ents = ents + d.entropy()
        return logps, ents


def import_sb3(policy: Policy, zip_path: str) -> None:
    """Load weights from a train_speed.py (SB3 MlpPolicy [256,256]) model."""
    import io
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu",
                        weights_only=True)
    mapping = {
        "mlp_extractor.policy_net.0": policy.pi[0],
        "mlp_extractor.policy_net.2": policy.pi[2],
        "mlp_extractor.value_net.0": policy.vf[0],
        "mlp_extractor.value_net.2": policy.vf[2],
        "action_net": policy.action_head,
        "value_net": policy.value_head,
    }
    for k, mod in mapping.items():
        mod.weight.data.copy_(sd[k + ".weight"])
        mod.bias.data.copy_(sd[k + ".bias"])
    print(f"imported SB3 weights from {zip_path}")


class GreedyTorchPolicy:
    def __init__(self, policy: Policy, device):
        self.policy = policy
        self.device = device

    @torch.inference_mode()
    def act(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, _ = self.policy(t)
        acts = [lg.argmax(-1) for lg in Policy.split(logits)]
        return torch.stack(acts, -1).to("cpu").numpy().astype(np.int32)


def episode_stats(traj_path: Path):
    out, rows = [], []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict) and "map" in row:
                rows = []
            elif isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict) and "end" in row and rows:
                a = np.asarray(rows, dtype=np.float64)
                yaw0 = np.radians(a[0, 7])
                d = ((a[:, 1] - a[0, 1]) * np.cos(yaw0) +
                     (a[:, 2] - a[0, 2]) * np.sin(yaw0))
                out.append({"fwd_max": float(d.max()),
                            "speed_max": float(np.hypot(a[:, 4], a[:, 5]).max())})
                rows = []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_ski_2.bsp"))
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=float, default=100e6)
    ap.add_argument("--run", default=time.strftime("fast_%m%d_%H%M"))
    ap.add_argument("--spawn", choices=["platform", "ramp"], default="platform")
    ap.add_argument("--ep-ticks", type=int, default=700)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.005,
                    help="entropy coef at run start")
    ap.add_argument("--ent-final", type=float, default=None,
                    help="linearly anneal entropy coef to this by --steps "
                         "(default: constant --ent)")
    ap.add_argument("--yaw-jitter", type=float, default=8.0,
                    help="spawn yaw jitter deg (start-state diversity)")
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--record-every", type=float, default=10e6)
    ap.add_argument("--ckpt-every", type=float, default=10e6)
    ap.add_argument("--ckpt", default=None, help="resume from a ckpt_*.pt")
    ap.add_argument("--sb3", default=None, help="import weights from an SB3 .zip")
    ap.add_argument("--reset-steps", action="store_true",
                    help="with --ckpt: load weights but restart the step count "
                         "(warm-starting a NEW experiment)")
    ap.add_argument("--reward", choices=["path", "forward"], default="path",
                    help="path = total horizontal distance traveled (teleports "
                         "filtered); forward = max displacement along spawn yaw")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, T = args.envs, args.n_steps
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)

    cfg = default_config(num_envs=N, spawn_mode=2, max_episode_ticks=args.ep_ticks,
                         water_fail=1, yaw_jitter_deg=args.yaw_jitter)
    core = SurfCore(args.map, cfg)
    pool = platform_spawn_pool(core) if args.spawn == "platform" else ramp_spawn_pool(core)
    core.set_spawn_pool(pool)
    print(f"spawn pool ({args.spawn}): {len(pool)} | envs {N} | device {device}")

    eval_core = SurfCore(args.map, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=args.ep_ticks, water_fail=1))
    eval_core.set_spawn_pool(pool)

    obs_dim = core.obs_dim
    policy = Policy(obs_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    reward_fn = (PathLengthReward(0.01) if args.reward == "path"
                 else ForwardProgressReward(0.01))

    global_step = 0
    if args.sb3:
        import_sb3(policy, args.sb3)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["optimizer"])
        global_step = 0 if args.reset_steps else int(ck.get("global_step", 0))
        print(f"resumed {args.ckpt} at step {global_step:,}"
              + (" (steps reset: warm start)" if args.reset_steps else ""))

    meta = {"label": args.run, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished": None,
            "config": {"trainer": "fast", "map": Path(args.map).stem, "envs": N,
                       "steps": int(args.steps), "spawn": args.spawn,
                       "lr": args.lr, "ep_ticks": args.ep_ticks,
                       "epochs": args.epochs, "reward": args.reward}}
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    csv_f = open(out / "progress.csv", "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        csv_w.writerow(["time/total_timesteps", "rollout/ep_rew_mean",
                        "rollout/ep_len_mean", "time/fps", "train/loss",
                        "train/value_loss", "train/entropy_loss",
                        "train/approx_kl", "eval/fwd_max", "eval/speed_max"])

    # rollout storage (GPU)
    b_obs = torch.zeros((T, N, obs_dim), device=device)
    b_act = torch.zeros((T, N, NACT), dtype=torch.long, device=device)
    b_logp = torch.zeros((T, N), device=device)
    b_val = torch.zeros((T, N), device=device)
    b_rew = torch.zeros((T, N), device=device)
    b_done = torch.zeros((T, N), device=device)
    obs_pin = torch.zeros((N, obs_dim), pin_memory=(device.type == "cuda"))

    def upload(o: np.ndarray) -> torch.Tensor:
        obs_pin.copy_(torch.from_numpy(o))
        return obs_pin.to(device, non_blocking=True)

    obs_np = core.reset(0).copy()
    reward_fn.on_reset(core)
    prev_obs = obs_np.copy()
    obs_t = upload(obs_np)
    ep_ret = np.zeros(N, np.float64)
    ep_len = np.zeros(N, np.int64)
    ret_hist, len_hist = [], []

    next_record = global_step
    next_ckpt = global_step + int(args.ckpt_every)
    eval_fwd = eval_speed = float("nan")
    t_start, step_start = time.perf_counter(), global_step

    def save_ckpt(tag):
        torch.save({"policy": policy.state_dict(), "optimizer": opt.state_dict(),
                    "global_step": global_step, "config": meta["config"]},
                   out / f"ckpt_{tag}.pt")

    while global_step < int(args.steps):
        # ---------------- rollout ----------------
        # no_grad (not inference_mode): adv/ret feed the update's autograd later
        with torch.no_grad():
            for t in range(T):
                logits, value = policy(obs_t)
                act_t, logp_t, _ = policy.dist_sample(logits)
                b_obs[t] = obs_t
                b_act[t] = act_t
                b_logp[t] = logp_t
                b_val[t] = value
                actions = act_t.to("cpu").numpy().astype(np.int32)
                o2, base_r, done, trunc, term_obs = core.step(
                    np.ascontiguousarray(actions))
                r = reward_fn(prev_obs, o2, term_obs, base_r, done, trunc, core)
                prev_obs = o2.copy()
                ended = (done | trunc).astype(bool)
                # bootstrap truncated episodes with V(terminal_obs)
                if trunc.any():
                    ti = np.flatnonzero(trunc.astype(bool) & ~done.astype(bool))
                    if len(ti):
                        tv = policy(torch.as_tensor(
                            term_obs[ti], dtype=torch.float32, device=device))[1]
                        r[ti] += args.gamma * tv.to("cpu").numpy()
                ep_ret += r
                ep_len += 1
                if ended.any():
                    for i in np.flatnonzero(ended):
                        ret_hist.append(ep_ret[i]); len_hist.append(ep_len[i])
                    ep_ret[ended] = 0; ep_len[ended] = 0
                b_rew[t] = torch.as_tensor(r, device=device)
                b_done[t] = torch.as_tensor(ended.astype(np.float32), device=device)
                obs_t = upload(o2)
                global_step += N
            # GAE
            _, last_val = policy(obs_t)
            adv = torch.zeros_like(b_rew)
            lastgae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nextval = last_val if t == T - 1 else b_val[t + 1]
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + args.gamma * nextval * nonterm - b_val[t]
                lastgae = delta + args.gamma * args.gae * nonterm * lastgae
                adv[t] = lastgae
            ret = adv + b_val

        # ---------------- update ----------------
        f_obs = b_obs.reshape(T * N, obs_dim)
        f_act = b_act.reshape(T * N, NACT)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = ret.reshape(-1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        mb = T * N // args.minibatches
        kl = loss_v = loss_pi = loss_ent = 0.0
        if args.ent_final is not None:
            frac = min(1.0, global_step / max(1.0, float(args.steps)))
            ent_coef = args.ent + (args.ent_final - args.ent) * frac
        else:
            ent_coef = args.ent
        for _ in range(args.epochs):
            perm = torch.randperm(T * N, device=device)
            for s in range(0, T * N, mb):
                idx = perm[s:s + mb]
                logits, value = policy(f_obs[idx])
                logp, ent = policy.logprob_entropy(logits, f_act[idx])
                ratio = torch.exp(logp - f_logp[idx])
                a = f_adv[idx]
                pg = torch.max(-a * ratio,
                               -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                vl = 0.5 * (value - f_ret[idx]).pow(2).mean()
                el = -ent.mean()
                loss = pg + args.vf * vl + ent_coef * el
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl = float((f_logp[idx] - logp).mean())
                loss_v, loss_pi, loss_ent = float(vl), float(pg), float(el)

        # ---------------- logging / artifacts ----------------
        fps = (global_step - step_start) / (time.perf_counter() - t_start)
        rmean = float(np.mean(ret_hist[-200:])) if ret_hist else 0.0
        lmean = float(np.mean(len_hist[-200:])) if len_hist else 0.0
        if global_step >= next_record:
            next_record = global_step + int(args.record_every)
            path = out / f"traj_{global_step:010d}.jsonl"
            record_rollout(eval_core, GreedyTorchPolicy(policy, device), path,
                           episodes=3, max_ticks=3 * args.ep_ticks, seed=1234)
            st = episode_stats(path)
            eval_fwd = float(np.mean([e["fwd_max"] for e in st])) if st else 0.0
            eval_speed = float(np.mean([e["speed_max"] for e in st])) if st else 0.0
            print(f"[{global_step:>12,d}] greedy: fwd {eval_fwd:7.0f}u  peak "
                  f"{eval_speed:6.0f} u/s -> {path.name}")
        if global_step >= next_ckpt:
            next_ckpt = global_step + int(args.ckpt_every)
            save_ckpt(f"{global_step:010d}")
        save_ckpt("latest")
        csv_w.writerow([global_step, round(rmean, 4), round(lmean, 1), round(fps),
                        round(loss_pi + args.vf * loss_v + args.ent * loss_ent, 5),
                        round(loss_v, 5), round(loss_ent, 5), round(kl, 6),
                        round(eval_fwd, 1), round(eval_speed, 1)])
        csv_f.flush()
        print(f"step {global_step:>12,d}  rew {rmean:8.2f}  len {lmean:6.0f}  "
              f"fps {fps:,.0f}  kl {kl:.4f}")

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["duration_s"] = round(time.perf_counter() - t_start, 1)
    meta["total_steps"] = global_step
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_ckpt("final")
    csv_f.close()
    print(f"done: {global_step:,} steps, avg {(global_step - step_start) / (time.perf_counter() - t_start):,.0f} steps/s")


if __name__ == "__main__":
    main()
