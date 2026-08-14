"""train_fast.py — lean GPU-resident PPO for surfgym.

v2 optimizations (on top of the GPU rollout/update loop):
  * fused action sampling: the 6 categorical heads are packed into one padded
    (N, 6, 15) tensor — one gumbel-argmax + one log_softmax for all heads,
    instead of six torch.distributions objects (~30 kernels -> ~8).
  * CUDA Graphs: the whole per-step forward+sample is captured once and
    replayed, eliminating kernel-launch overhead (--no-graphs to disable;
    falls back to eager automatically if capture fails).
  * zero-copy rewards: reward hooks read the env state through the DLL's
    surf_states_ptr view (no per-tick copy).
  * bf16 autocast updates + fused Adam.

    python python\\train_fast.py --steps 1e9 --run marathon
    python python\\train_fast.py --ckpt runs\\marathon\\ckpt_latest.pt      # resume
    python python\\train_fast.py --ckpt ... --reset-steps                  # warm start
    python python\\train_fast.py --sb3 runs\\forward_100M\\final.zip       # import SB3
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "python"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from surfgym import SurfCore, default_config
from surfgym.record import record_rollout
from surfgym.rewards import (BlendedReward, ForwardProgressReward,
                             PathLengthReward, platform_spawn_pool,
                             ramp_spawn_pool)
from surfgym.vision import GpuLidar

NVEC = (15, 7, 3, 3, 2, 2)            # yaw, pitch, fwd, side, jump, duck
NACT = len(NVEC)
NPAD = max(NVEC)                      # heads padded to (NACT, NPAD)
NEG = -1e30                           # finite -inf (keeps p*logp == 0, no NaN)
N_SCALAR = 15                         # surfcore.h fixed scalars (core runs eyeless)
LIDAR_W, LIDAR_H = 128, 64            # GPU lidar (surfgym.vision): ~6ms per
                                      # 2048-env batch on the SDF triton kernel
# generalizing feature set: drop absolute heading (7,8) and position (12..14)
# — both enable memorize-the-map policies; the rest is honest proprioception
SCALAR_NOGPS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11)


class Policy(nn.Module):
    """Scalars + lidar depth image: conv trunk (shared by pi/vf) embeds the
    depth image to `emb` features, concatenated with the selected scalars
    into [hidden, hidden] tanh towers. gps=False (default) hides absolute
    heading + position from the network — honest-perception mode."""

    def __init__(self, obs_dim: int, lidar_w: int = LIDAR_W, lidar_h: int = LIDAR_H,
                 emb: int = 512, hidden: int = 448, gps: bool = False):
        super().__init__()
        assert obs_dim == N_SCALAR + lidar_w * lidar_h, \
            f"obs_dim {obs_dim} != {N_SCALAR}+{lidar_w}x{lidar_h}"
        self.lidar_w, self.lidar_h = lidar_w, lidar_h
        idx = tuple(range(N_SCALAR)) if gps else SCALAR_NOGPS
        self.register_buffer("feat_idx", torch.tensor(idx, dtype=torch.long),
                             persistent=False)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
            nn.Linear(64 * 4 * 8, emb), nn.ReLU(),
        )
        feat = len(idx) + emb
        def mlp():
            return nn.Sequential(nn.Linear(feat, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = mlp()
        self.vf = mlp()
        self.action_head = nn.Linear(hidden, sum(NVEC))
        self.value_head = nn.Linear(hidden, 1)
        for m in list(self.conv) + list(self.pi) + list(self.vf):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.action_head.weight, 0.01)
        nn.init.zeros_(self.action_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs):
        img = obs[:, N_SCALAR:].reshape(-1, 1, self.lidar_h, self.lidar_w)
        f = torch.cat([obs[:, self.feat_idx], self.conv(img)], dim=1)
        return self.action_head(self.pi(f)), self.value_head(self.vf(f)).squeeze(-1)


class HeadPacker:
    """Flat (B, sum(NVEC)) logits <-> padded (B, 6, 15) with NEG in unused slots."""

    def __init__(self, device):
        idx = []
        for h, n in enumerate(NVEC):
            idx.extend(h * NPAD + j for j in range(n))
        self.scatter = torch.tensor(idx, dtype=torch.long, device=device)

    def pad(self, logits):
        B = logits.shape[0]
        out = logits.new_full((B, NACT * NPAD), NEG)
        out[:, self.scatter] = logits
        return out.view(B, NACT, NPAD)


def sample_padded(padded):
    """Gumbel-argmax sample + total logprob from padded logits (no grad)."""
    u = torch.rand_like(padded).clamp_min_(1e-20)
    act = (padded - torch.log(-torch.log(u))).argmax(-1)
    lsm = F.log_softmax(padded, dim=-1)
    logp = lsm.gather(-1, act.unsqueeze(-1)).squeeze(-1).sum(-1)
    return act, logp


def logprob_entropy_padded(padded, actions):
    lsm = F.log_softmax(padded, dim=-1)
    logp = lsm.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    ent = -(lsm.exp() * lsm).sum(-1).sum(-1)
    return logp, ent


def import_sb3(policy: Policy, zip_path: str) -> None:
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


class _TorchPolicyBase:
    """Shared obs assembly: the core emits the 15 scalars; when a GpuLidar +
    its core are attached, the depth image is rendered from the core's live
    states and concatenated — the same fusion the trainer does."""

    def __init__(self, policy: Policy, packer: HeadPacker, device,
                 lidar=None, core=None):
        self.policy, self.packer, self.device = policy, packer, device
        self.lidar, self.core = lidar, core

    def _obs(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self.lidar is not None:
            sv = self.core.states_view
            o = torch.as_tensor(np.ascontiguousarray(sv["origin"]),
                                dtype=torch.float32, device=self.device)
            yw = torch.as_tensor(sv["yaw"].copy(), dtype=torch.float32,
                                 device=self.device)
            pt = torch.as_tensor(sv["pitch"].copy(), dtype=torch.float32,
                                 device=self.device)
            dk = torch.as_tensor(sv["ducked"].copy().astype(np.int32),
                                 device=self.device)
            depth = self.lidar.render(o, yw, pt, dk)
            t = torch.cat([t, depth.reshape(t.shape[0], -1)], dim=1)
        return t


class GreedyTorchPolicy(_TorchPolicyBase):
    @torch.inference_mode()
    def act(self, obs):
        logits, _ = self.policy(self._obs(obs))
        act = self.packer.pad(logits).argmax(-1)
        return act.to("cpu").numpy().astype(np.int32)


class SampledTorchPolicy(_TorchPolicyBase):
    """Acts by sampling the distribution — the policy training actually
    optimizes and logs. Under a high entropy coefficient the argmax mode can
    be much weaker (it drifts unoptimized while the stochastic policy learns
    to rely on its own action noise)."""

    @torch.inference_mode()
    def act(self, obs):
        logits, _ = self.policy(self._obs(obs))
        act, _ = sample_padded(self.packer.pad(logits))
        return act.to("cpu").numpy().astype(np.int32)


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
                dx = np.diff(a[:, 1]); dy = np.diff(a[:, 2])
                d = np.hypot(dx, dy)
                tel = d > 50.0                              # teleport filter
                d[tel] = 0.0
                fstep = dx * np.cos(yaw0) + dy * np.sin(yaw0)
                fstep[tel] = 0.0                            # jumps aren't progress
                fwd = np.concatenate(([0.0], np.cumsum(fstep)))
                out.append({"fwd_max": float(fwd.max()), "path": float(d.sum()),
                            "speed_max": float(np.hypot(a[:, 4], a[:, 5]).max())})
                rows = []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_ski_2.bsp"))
    # 2048 envs, not more: at fixed update density, doubling rollout width
    # halves PPO iterations per sample (rew-20 at 52M steps here vs 98M at
    # 8192 envs) and the extra raw throughput doesn't pay for it
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=float, default=100e6)
    ap.add_argument("--run", default=time.strftime("fast_%m%d_%H%M"))
    # mixed = exploring starts: platform spawns + mid-air spawns over every
    # surfable ramp face map-wide. Entropy only dithers actions locally; a
    # policy collapsed to one groove never *visits* other states, so its
    # value estimates there stay garbage and it can never rationally detour.
    # Diverse starts break that data loop. Eval stays on the platform pool.
    ap.add_argument("--spawn", choices=["platform", "ramp", "mixed"],
                    default="platform")
    ap.add_argument("--ep-ticks", type=int, default=None)   # 700; ckpt overrides
    # update density matters as much as throughput: these defaults match SB3's
    # 1-gradient-update-per-4k-samples (64 -> 300M-step sample-efficiency
    # regression when this was 2 epochs x 8 minibatches over 1M-sample rollouts)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--ent-final", type=float, default=None)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--yaw-jitter", type=float, default=8.0)
    # 128 rays cost ~17ns each and dominate env time (13M steps/s eyeless ->
    # 0.44M at 16x8); drop to 12x6 for ~1.7x env throughput at coarser vision
    ap.add_argument("--lidar-w", type=int, default=None)   # 128; ckpt overrides
    ap.add_argument("--lidar-h", type=int, default=None)   # 64
    # fixed-gaze experiment: freeze view pitch at this angle (deg, + = up);
    # the pitch action head stays in the action space but is physically inert
    ap.add_argument("--fix-pitch", type=float, default=None)
    ap.add_argument("--emb", type=int, default=None)      # 512; ckpt overrides
    ap.add_argument("--hidden", type=int, default=None)   # 448; ckpt overrides
    ap.add_argument("--gps", action="store_true",
                    help="re-include absolute heading+position scalars "
                         "(default hides them: they enable pure memorization)")
    # new runs default ON: falling into a jail teleport must end the episode
    # (circling the cell farms path reward forever); resumes preserve the
    # ckpt's setting so old runs keep their semantics
    ap.add_argument("--keep-teleports", action="store_true",
                    help="disable the teleport-ends-episode rule")
    ap.add_argument("--teleport-fail", action="store_true",
                    help="force the rule ON even when warm-starting an old "
                         "ckpt whose config predates it")
    ap.add_argument("--lidar-range", type=float, default=None)  # 2000; ckpt overrides
    ap.add_argument("--lidar-near", type=float, default=None)   # = range (legacy code)
    ap.add_argument("--record-every", type=float, default=10e6)
    ap.add_argument("--ckpt-every", type=float, default=10e6)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--sb3", default=None)
    ap.add_argument("--reset-steps", action="store_true")
    ap.add_argument("--reward", choices=["forward", "path", "blend"],
                    default=None,
                    help="forward = max displacement along spawn yaw (default; "
                         "path-length turned out to reward circling in place); "
                         "blend = curriculum: forward until --blend-start, then "
                         "anneal linearly to pure path-length by --blend-end")
    ap.add_argument("--blend-start", type=float, default=None)   # 100e6
    ap.add_argument("--blend-end", type=float, default=None)     # 200e6
    ap.add_argument("--no-graphs", action="store_true")
    # with 128x64 vision the conv update dominates (94ms/minibatch fp32 vs
    # ~25ms bf16) — bf16 is now the default; --fp32 restores exact math at
    # ~3x the wall-clock (the old ~20% sample-efficiency tax measured on the
    # MLP is the price)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    # a bare `--ckpt` resume must not silently change the training objective:
    # settings that define the run (reward mode, blend window, episode length)
    # come from the checkpoint's saved config unless explicitly overridden
    ck = None
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_cfg = ck.get("config") or {}
        restored = []
        if args.reward is None and ck_cfg.get("reward"):
            args.reward = ck_cfg["reward"]
            restored.append(f"reward={args.reward}")
        if ck_cfg.get("blend"):
            if args.blend_start is None:
                args.blend_start = float(ck_cfg["blend"][0])
                restored.append(f"blend_start={args.blend_start:g}")
            if args.blend_end is None:
                args.blend_end = float(ck_cfg["blend"][1])
                restored.append(f"blend_end={args.blend_end:g}")
        if args.ep_ticks is None and ck_cfg.get("ep_ticks"):
            args.ep_ticks = int(ck_cfg["ep_ticks"])
            restored.append(f"ep_ticks={args.ep_ticks}")
        if args.lidar_w is None and ck_cfg.get("lidar_w"):
            args.lidar_w = int(ck_cfg["lidar_w"])
            restored.append(f"lidar_w={args.lidar_w}")
        if args.lidar_h is None and ck_cfg.get("lidar_h"):
            args.lidar_h = int(ck_cfg["lidar_h"])
            restored.append(f"lidar_h={args.lidar_h}")
        if not args.fp32 and ck_cfg.get("bf16") is False:
            args.fp32 = True
            restored.append("fp32")
        if args.fix_pitch is None and ck_cfg.get("fix_pitch") is not None:
            args.fix_pitch = float(ck_cfg["fix_pitch"])
            restored.append(f"fix_pitch={args.fix_pitch:g}")
        if args.emb is None and ck_cfg.get("emb"):
            args.emb = int(ck_cfg["emb"])
            restored.append(f"emb={args.emb}")
        if args.hidden is None and ck_cfg.get("hidden"):
            args.hidden = int(ck_cfg["hidden"])
            restored.append(f"hidden={args.hidden}")
        if not args.gps and ck_cfg.get("gps"):
            args.gps = True
            restored.append("gps")
        if (not args.teleport_fail and not args.keep_teleports
                and not ck_cfg.get("teleport_fail", False)):
            args.keep_teleports = True     # preserve old-run semantics
            restored.append("keep_teleports")
        if args.lidar_range is None and ck_cfg.get("lidar_range"):
            args.lidar_range = float(ck_cfg["lidar_range"])
            restored.append(f"lidar_range={args.lidar_range:g}")
        if args.lidar_near is None and ck_cfg.get("lidar_near"):
            args.lidar_near = float(ck_cfg["lidar_near"])
            restored.append(f"lidar_near={args.lidar_near:g}")
        if restored:
            print("restored from checkpoint config: " + ", ".join(restored))
    if args.reward is None:
        args.reward = "forward"
    if args.blend_start is None:
        args.blend_start = 100e6
    if args.blend_end is None:
        args.blend_end = 200e6
    if args.ep_ticks is None:
        args.ep_ticks = 700
    if args.lidar_w is None:
        args.lidar_w = LIDAR_W
    if args.lidar_h is None:
        args.lidar_h = LIDAR_H
    if args.lidar_w < 1 or args.lidar_h < 1:
        raise SystemExit("this trainer's policy needs the lidar block; "
                         "--lidar-w/--lidar-h must be >= 1")
    if args.emb is None:
        args.emb = 512
    if args.hidden is None:
        args.hidden = 448
    if args.lidar_range is None:
        args.lidar_range = 2000.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_graphs = device.type == "cuda" and not args.no_graphs
    use_bf16 = device.type == "cuda" and not args.fp32
    N, T = args.envs, args.n_steps
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)

    # cores run EYELESS (13M raw steps/s); vision is rendered on the GPU from
    # the map SDF and fused into the obs here in the trainer
    pitch_rate = 0.0 if args.fix_pitch is not None else -1.0   # -1 = core default
    cfg = default_config(num_envs=N, spawn_mode=2, max_episode_ticks=args.ep_ticks,
                         water_fail=1, yaw_jitter_deg=args.yaw_jitter,
                         lidar_w=0, lidar_h=0, pitch_rate_max_deg=pitch_rate)
    core = SurfCore(args.map, cfg)
    plat_pool = platform_spawn_pool(core)
    if args.spawn == "platform":
        pool = plat_pool
    elif args.spawn == "ramp":
        pool = ramp_spawn_pool(core)
    else:
        pool = np.concatenate([plat_pool, ramp_spawn_pool(core)])
    if args.fix_pitch is not None:
        pool["pitch"] = args.fix_pitch
        plat_pool["pitch"] = args.fix_pitch
    if not args.keep_teleports:
        core.set_teleport_fail(True)
    core.set_spawn_pool(pool)
    print(f"pool({args.spawn}) {len(pool)} | envs {N} | {device} | "
          f"graphs={use_graphs} bf16={use_bf16}"
          + (f" | pitch fixed {args.fix_pitch:g}" if args.fix_pitch is not None else ""))

    # eval on the game-authentic platform start regardless of the training
    # pool, so eval/* metrics and recordings stay comparable across runs
    eval_core = SurfCore(args.map, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=args.ep_ticks, water_fail=1,
        lidar_w=0, lidar_h=0, pitch_rate_max_deg=pitch_rate))
    if not args.keep_teleports:
        eval_core.set_teleport_fail(True)
    eval_core.set_spawn_pool(plat_pool)

    lidar = GpuLidar(core, args.lidar_w, args.lidar_h,
                     range_units=args.lidar_range, near_range=args.lidar_near,
                     device=device)
    mn_b, mx_b = core.map_bounds()
    map_center = ((mn_b + mx_b) / 2.0).astype(np.float32)
    obs_dim = core.obs_dim + args.lidar_w * args.lidar_h
    policy = Policy(obs_dim, args.lidar_w, args.lidar_h,
                    emb=args.emb, hidden=args.hidden, gps=args.gps).to(device)
    packer = HeadPacker(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5,
                           fused=(device.type == "cuda"))
    if args.reward == "path":
        reward_fn = PathLengthReward(0.01)
    elif args.reward == "blend":
        reward_fn = BlendedReward(ForwardProgressReward(0.01),
                                  PathLengthReward(0.01),
                                  args.blend_start, args.blend_end)
    else:
        reward_fn = ForwardProgressReward(0.01)

    global_step = 0
    if args.sb3:
        raise SystemExit("--sb3 import predates the lidar architecture; "
                         "the SB3 MlpPolicy weights don't map onto the conv trunk")
    if ck is not None:
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["optimizer"])
        global_step = 0 if args.reset_steps else int(ck.get("global_step", 0))
        print(f"resumed {args.ckpt} at step {global_step:,}"
              + (" (steps reset)" if args.reset_steps else ""))

    meta = {"label": args.run, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished": None,
            "config": {"trainer": "fast2", "map": Path(args.map).stem, "envs": N,
                       "steps": int(args.steps), "spawn": args.spawn,
                       "reward": args.reward, "lr": args.lr,
                       "blend": ([args.blend_start, args.blend_end]
                                 if args.reward == "blend" else None),
                       "lidar_w": args.lidar_w, "lidar_h": args.lidar_h,
                       "fix_pitch": args.fix_pitch,
                       "emb": args.emb, "hidden": args.hidden, "gps": args.gps,
                       "teleport_fail": not args.keep_teleports,
                       "lidar_range": args.lidar_range,
                       "lidar_near": args.lidar_near or args.lidar_range,
                       "ep_ticks": args.ep_ticks, "epochs": args.epochs,
                       "graphs": use_graphs, "bf16": use_bf16}}
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    csv_f = open(out / "progress.csv", "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        csv_w.writerow(["time/total_timesteps", "rollout/ep_rew_mean",
                        "rollout/ep_len_mean", "time/fps", "train/loss",
                        "train/value_loss", "train/entropy_loss",
                        "train/approx_kl", "eval/fwd_max", "eval/path",
                        "eval/speed_max", "train/blend_w"])

    # ---- static rollout buffers (graph-capturable) --------------------------
    b_obs = torch.zeros((T, N, obs_dim), device=device)
    b_act = torch.zeros((T, N, NACT), dtype=torch.long, device=device)
    b_logp = torch.zeros((T, N), device=device)
    b_val = torch.zeros((T, N), device=device)
    b_rew = torch.zeros((T, N), device=device)
    b_done = torch.zeros((T, N), device=device)

    static_obs = torch.zeros((N, obs_dim), device=device)
    static_act = torch.zeros((N, NACT), dtype=torch.long, device=device)
    static_logp = torch.zeros(N, device=device)
    static_val = torch.zeros(N, device=device)

    obs_pin = torch.zeros((N, N_SCALAR), pin_memory=(device.type == "cuda"))
    act_pin = torch.zeros((N, NACT), dtype=torch.long,
                          pin_memory=(device.type == "cuda"))
    act_np32 = np.zeros((N, NACT), dtype=np.int32)

    # GPU vision fusion: pose upload (one pinned (N, 6) copy) -> SDF ray-march
    # -> lidar slice of the obs. States are read post-step/post-autoreset, so
    # the depth image always matches the scalar obs row.
    sv_view = core.states_view
    vis_pin = torch.zeros((N, 6), pin_memory=(device.type == "cuda"))
    vis_np = vis_pin.numpy()
    vis_gpu = torch.zeros((N, 6), device=device)

    def fill_vision(dst):
        vis_np[:, 0:3] = sv_view["origin"]
        vis_np[:, 3] = sv_view["yaw"]
        vis_np[:, 4] = sv_view["pitch"]
        vis_np[:, 5] = sv_view["ducked"]
        vis_gpu.copy_(vis_pin, non_blocking=True)
        depth = lidar.render(vis_gpu[:, 0:3], vis_gpu[:, 3],
                             vis_gpu[:, 4], vis_gpu[:, 5])
        dst[:, N_SCALAR:].copy_(depth.reshape(N, -1))

    def step_compute():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_bf16):
            logits, value = policy(static_obs)
        act, logp = sample_padded(packer.pad(logits.float()))
        static_act.copy_(act)
        static_logp.copy_(logp)
        static_val.copy_(value.float())

    graph = None
    if use_graphs:
        try:
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    step_compute()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                step_compute()
            print("CUDA graph captured for the rollout step")
        except Exception as exc:  # pragma: no cover
            print(f"CUDA graph capture failed ({exc!r}) — eager fallback")
            graph = None

    def policy_step():
        if graph is not None:
            graph.replay()
        else:
            with torch.no_grad():
                step_compute()

    obs_np = core.reset(0).copy()
    reward_fn.on_reset(core)
    prev_obs = obs_np.copy()
    obs_pin.copy_(torch.from_numpy(obs_np))
    static_obs[:, :N_SCALAR].copy_(obs_pin, non_blocking=True)
    fill_vision(static_obs)
    ep_ret = np.zeros(N, np.float64)
    ep_len = np.zeros(N, np.int64)
    ret_hist = deque(maxlen=200)     # bounded: a 10B run finishes ~10M episodes
    len_hist = deque(maxlen=200)

    next_record = global_step
    next_ckpt = global_step + int(args.ckpt_every)
    eval_fwd = eval_path = eval_speed = float("nan")
    t_start, step_start = time.perf_counter(), global_step

    def save_ckpt(tag):
        torch.save({"policy": policy.state_dict(), "optimizer": opt.state_dict(),
                    "global_step": global_step, "config": meta["config"]},
                   out / f"ckpt_{tag}.pt")

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                         enabled=use_bf16)

    while global_step < int(args.steps):
        if hasattr(reward_fn, "set_step"):
            reward_fn.set_step(global_step)   # authoritative (survives resume)
        # ---------------- rollout ----------------
        with torch.no_grad():
            for t in range(T):
                policy_step()
                b_obs[t].copy_(static_obs)
                b_act[t].copy_(static_act)
                b_logp[t].copy_(static_logp)
                b_val[t].copy_(static_val)
                act_pin.copy_(static_act, non_blocking=True)
                torch.cuda.synchronize() if device.type == "cuda" else None
                np.copyto(act_np32, act_pin.numpy(), casting="unsafe")
                o2, base_r, done, trunc, term_obs = core.step(act_np32)
                r = reward_fn(prev_obs, o2, term_obs, base_r, done, trunc, core)
                prev_obs = o2.copy()
                ended = (done | trunc).astype(bool)
                ep_ret += r          # pure collected reward only: the trunc
                ep_len += 1          # bootstrap below is a GAE construct and
                                     # must not inflate the logged return
                if trunc.any():
                    ti = np.flatnonzero(trunc.astype(bool) & ~done.astype(bool))
                    if len(ti):
                        # states are already the NEW episode's (autoreset), so
                        # the terminal pose is reconstructed from the terminal
                        # scalar obs to render its lidar for V(s_T)
                        to = term_obs[ti]
                        ts = torch.as_tensor(to, dtype=torch.float32, device=device)
                        pos = torch.as_tensor(to[:, 12:15] * 2000.0 + map_center,
                                              dtype=torch.float32, device=device)
                        yawd = torch.rad2deg(torch.atan2(ts[:, 7], ts[:, 8]))
                        depth = lidar.render(pos, yawd, ts[:, 9] * 90.0, ts[:, 5])
                        full = torch.cat([ts, depth.reshape(len(ti), -1)], dim=1)
                        tv = policy(full)[1]
                        r[ti] += args.gamma * tv.to("cpu").numpy()
                if ended.any():
                    for i in np.flatnonzero(ended):
                        ret_hist.append(ep_ret[i]); len_hist.append(ep_len[i])
                    ep_ret[ended] = 0; ep_len[ended] = 0
                b_rew[t].copy_(torch.from_numpy(r).to(device, non_blocking=True))
                b_done[t].copy_(torch.from_numpy(
                    ended.astype(np.float32)).to(device, non_blocking=True))
                obs_pin.copy_(torch.from_numpy(o2))
                static_obs[:, :N_SCALAR].copy_(obs_pin, non_blocking=True)
                fill_vision(static_obs)
                global_step += N
            _, last_val = policy(static_obs)
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
        mb = T * N // args.minibatches
        if args.ent_final is not None:
            frac = min(1.0, global_step / max(1.0, float(args.steps)))
            ent_coef = args.ent + (args.ent_final - args.ent) * frac
        else:
            ent_coef = args.ent
        kl = loss_v = loss_pi = loss_ent = 0.0
        for _ in range(args.epochs):
            perm = torch.randperm(T * N, device=device)
            for s0 in range(0, T * N, mb):
                idx = perm[s0:s0 + mb]
                with amp:
                    logits, value = policy(f_obs[idx])
                    logp, ent = logprob_entropy_padded(
                        packer.pad(logits.float()), f_act[idx])
                    value = value.float()
                ratio = torch.exp(logp - f_logp[idx])
                a = f_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)   # per-minibatch, like SB3
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
        rmean = float(np.mean(ret_hist)) if ret_hist else 0.0
        lmean = float(np.mean(len_hist)) if len_hist else 0.0
        if global_step >= next_record:
            next_record = global_step + int(args.record_every)
            path = out / f"traj_{global_step:010d}.jsonl"
            # per-recording seed: a fixed seed replays the same few spawns
            # forever, and with a wide per-spawn spread (56..100 at 2.6B) a
            # single weak-tail spawn makes every eval look bad
            record_rollout(eval_core,
                           GreedyTorchPolicy(policy, packer, device,
                                             lidar, eval_core),
                           path, episodes=5, max_ticks=5 * args.ep_ticks,
                           seed=global_step & 0x7FFFFFFF)
            st = episode_stats(path)
            eval_fwd = float(np.mean([e["fwd_max"] for e in st])) if st else 0.0
            eval_path = float(np.mean([e["path"] for e in st])) if st else 0.0
            eval_speed = float(np.mean([e["speed_max"] for e in st])) if st else 0.0
            print(f"[{global_step:>13,d}] greedy: fwd {eval_fwd:7.0f}u  path "
                  f"{eval_path:7.0f}u  peak {eval_speed:6.0f} u/s -> {path.name}")
            spath = out / f"traj_{global_step:010d}_stoch.jsonl"
            record_rollout(eval_core,
                           SampledTorchPolicy(policy, packer, device,
                                              lidar, eval_core),
                           spath, episodes=5, max_ticks=5 * args.ep_ticks,
                           seed=global_step & 0x7FFFFFFF)
            sst = episode_stats(spath)
            if sst:
                print(f"[{global_step:>13,d}] stoch : path "
                      f"{np.mean([e['path'] for e in sst]):7.0f}u -> {spath.name}")
        if global_step >= next_ckpt:
            next_ckpt = global_step + int(args.ckpt_every)
            save_ckpt(f"{global_step:010d}")
        save_ckpt("latest")
        csv_w.writerow([global_step, round(rmean, 4), round(lmean, 1), round(fps),
                        round(loss_pi + args.vf * loss_v + ent_coef * loss_ent, 5),
                        round(loss_v, 5), round(loss_ent, 5), round(kl, 6),
                        round(eval_fwd, 1), round(eval_path, 1),
                        round(eval_speed, 1),
                        round(getattr(reward_fn, "weight", 0.0), 4)])
        csv_f.flush()
        print(f"step {global_step:>13,d}  rew {rmean:8.2f}  len {lmean:6.0f}  "
              f"fps {fps:,.0f}  kl {kl:.4f}  ent {ent_coef:.4f}")

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["duration_s"] = round(time.perf_counter() - t_start, 1)
    meta["total_steps"] = global_step
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_ckpt("final")
    csv_f.close()
    print(f"done: {global_step:,} steps, avg "
          f"{(global_step - step_start) / (time.perf_counter() - t_start):,.0f} steps/s")


if __name__ == "__main__":
    main()
