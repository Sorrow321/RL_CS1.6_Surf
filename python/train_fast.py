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

import os


def _default_omp_threads() -> str:
    """Pick an OpenMP team size — this MUST run before numpy/torch/surfcore
    initialise their runtimes, which they do at import/dlopen.

    ``surf_step`` is an ``omp parallel for`` over the envs (src/env.c) and the
    rollout forks that team 384 times per iteration; numpy's reward math
    shares the same runtime. Letting the team default to the core count is
    catastrophic, because a team sized to *every* core spins against the
    master thread — measured with tools/bench_env.py:

        16-core box:   16 threads 5.99 ms/step   vs   8 threads 0.88
        192-core box: 192 threads 6.99 ms/step   vs  32 threads 0.23

    Half the cores, capped at 32, was at or near the in-situ optimum on every
    box measured. Export OMP_NUM_THREADS to override.
    """
    try:
        n = len(os.sched_getaffinity(0))     # respects cgroup/taskset limits
    except AttributeError:                    # pragma: no cover — Windows
        n = os.cpu_count() or 8
    return str(max(4, min(32, n // 2)))


os.environ.setdefault("OMP_NUM_THREADS", _default_omp_threads())

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
from surfgym.goalfield import build_goal_field
from surfgym.record import record_rollout
from surfgym.respawn import RespawnBuffer
from surfgym.rewards import (AcroCoverageReward, BlendedReward,
                             CoverageSpeedReward, ForwardProgressReward,
                             MaxSpeedReward, PathLengthReward, RaceReward,
                             drop_spawn_pool, map_spawn_pool,
                             platform_spawn_pool, ramp_spawn_pool)
from surfgym.vision import GpuLidar, pick_cell
from surfgym.zones import load_zones

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
        # cuDNN's tensor-core convolutions are NHWC; fed NCHW they transpose
        # in and back out around every conv, forward AND backward. On the
        # 5090 that was 34% of ALL update GPU time (three nchwToNhwc /
        # nhwcToNchw kernels, 44.8 of 131.2 ms — tools/bench_update.py
        # --profile). Holding the trunk channels_last deletes those kernels;
        # the arithmetic and the weights are unchanged, so checkpoints stay
        # interchangeable in both directions.
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, obs):
        """One fused (B, 15 + H*W) fp32 row — the rollout and every eval."""
        return self.forward_split(obs[:, :N_SCALAR], obs[:, N_SCALAR:])

    def forward_split(self, scal, img):
        """Scalars and depth as separate tensors, so the PPO update can keep
        its depth buffer in bf16 (S3) without materialising a fused fp32 row.
        `scal` is indexed with feat_idx, whose entries are all < N_SCALAR, so
        this selects exactly what the fused path selects."""
        # depth is one channel, so (B,1,H,W) NCHW and its NHWC twin hold the
        # same bytes in the same order: view+permute is a free RESTRIDE that
        # declares NHWC. The one layout copy conv still makes is the copy the
        # NCHW path already made — and autocast does it in bf16, which is why
        # this is left to conv instead of an explicit .contiguous() in fp32.
        im = img.reshape(-1, self.lidar_h, self.lidar_w, 1).permute(0, 3, 1, 2)
        f = torch.cat([scal[:, self.feat_idx], self.conv(im)], dim=1)
        return self.action_head(self.pi(f)), self.value_head(self.vf(f)).squeeze(-1)


class PhaseTimer:
    """Per-iteration phase accounting for --timing (one TIMING line/iter).

    CPU phases are ``perf_counter`` brackets; GPU phases are CUDA event pairs
    read ONCE per iteration — a per-phase ``synchronize`` would serialize the
    very pipeline being measured. Event pairs are pooled and reused, so a
    128-decision rollout allocates its ~256 events once and then costs only
    the record() launches (~1 us each, ~0.5 ms/iter against a ~4 s iteration).

    Disabled (the default) every method is one attribute test: the hot loop
    calls this ~1.5k times per iteration, which must stay free.

    GPU phase names must stay DISJOINT from CPU phase names — both land in
    the same accumulator, so a shared name silently reports cpu+gpu summed
    (which is how the first version reported an update phase longer than the
    whole iteration). Hence the ``_gpu`` suffix on the paired ones.
    """

    # printed in this order; anything else is appended alphabetically
    FIELDS = ("pool", "rollout_fwd", "sync_copy", "env", "reward_py", "boot",
              "book", "respawn", "vis_cpu", "lidar", "rollout_wall",
              "gae", "gae_gpu", "update", "update_gpu", "mb_gpu",
              "ckpt", "record", "misc", "total")
    # phases that are disjoint slices of the iteration wall (for `misc`)
    _WALL = ("pool", "rollout_wall", "gae", "update", "ckpt", "record")

    def __init__(self, enabled: bool, cuda: bool) -> None:
        self.on = bool(enabled)
        self.cuda = bool(cuda) and self.on
        self.acc: dict[str, float] = {}
        self._pool: dict[str, list] = {}      # name -> [(start_ev, end_ev), ..]
        self._used: dict[str, int] = {}       # name -> pairs recorded this iter
        self._t0 = 0.0

    # -- CPU brackets --------------------------------------------------------
    def now(self) -> float:
        return time.perf_counter() if self.on else 0.0

    def add(self, name: str, t0: float) -> None:
        if self.on:
            self.acc[name] = self.acc.get(name, 0.0) \
                + (time.perf_counter() - t0) * 1e3

    # -- GPU brackets --------------------------------------------------------
    def gpu_start(self, name: str):
        if not self.cuda:
            return None
        pool = self._pool.setdefault(name, [])
        k = self._used.get(name, 0)
        if k == len(pool):
            pool.append((torch.cuda.Event(enable_timing=True),
                         torch.cuda.Event(enable_timing=True)))
        self._used[name] = k + 1
        pool[k][0].record()
        return pool[k]

    @staticmethod
    def gpu_end(pair) -> None:
        if pair is not None:
            pair[1].record()

    # -- per-iteration -------------------------------------------------------
    def start_iter(self) -> None:
        if self.on:
            self.acc = {}
            # also drop event pairs recorded OUTSIDE an iteration — the setup
            # fill_vision() records one before the loop starts, and it carries
            # the Triton JIT, so iteration 1 would otherwise report 129 lidar
            # pairs including a compile
            self._used.clear()
            self._t0 = time.perf_counter()

    def flush(self, iter_no: int) -> None:
        """Sync once, fold the event pairs in, print the TIMING line."""
        if not self.on:
            return
        if self.cuda:
            torch.cuda.synchronize()      # events are only readable once done
            for name, pool in self._pool.items():
                ms = 0.0
                for k in range(self._used.get(name, 0)):
                    ms += pool[k][0].elapsed_time(pool[k][1])
                self._used[name] = 0
                if ms:
                    self.acc[name] = self.acc.get(name, 0.0) + ms
        self.acc["total"] = total = (time.perf_counter() - self._t0) * 1e3
        self.acc["misc"] = total - sum(self.acc.get(k, 0.0) for k in self._WALL)
        keys = [k for k in self.FIELDS if k in self.acc]
        keys += sorted(k for k in self.acc if k not in self.FIELDS)
        print(f"TIMING iter={iter_no} "
              + " ".join(f"{k}={self.acc[k]:.1f}" for k in keys), flush=True)


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


def contiguous_optimizer_state(sd: dict) -> dict:
    """Normalise an optimizer state_dict's tensors to contiguous BEFORE saving.

    Adam's moments are allocated with `zeros_like(param)`, so with the
    channels_last trunk they come out NHWC-strided, and `torch.save`/`load`
    preserve strides while `Optimizer.load_state_dict` only ever casts dtype
    and device — never layout. A checkpoint written here would therefore fail
    to resume under any code whose trunk is NCHW (including simply checking
    out the pre-channels_last commit) with "params, grads, exp_avgs, and
    exp_avg_sqs must have same dtype, device, and layout".

    Writing contiguous and restoring the layout on load
    (:func:`relayout_optimizer_state`) is what actually makes checkpoints
    portable ACROSS a layout change, in both directions. Three conv weights;
    the copy is free.

    ``.contiguous()`` cannot do this job: PyTorch's contiguity check skips
    size-1 dimensions, so conv1's ``(16, 1, 5, 5)`` weight reports contiguous
    under BOTH layouts while its strides still differ — ``(25, 1, 5, 1)`` vs
    ``(25, 25, 5, 1)`` — which is exactly what the fused optimizer compares.
    So compare against canonical row-major strides explicitly.

    Returns a NEW mapping and never mutates the argument: ``state_dict()``
    hands back the optimizer's own per-parameter dicts BY REFERENCE, so
    rewriting an entry in place would silently re-layout the LIVE optimizer
    and the next ``opt.step()`` would die on the mismatch it just created.
    """
    def row_major(shape):
        strides, acc = [1] * len(shape), 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = acc
            acc *= shape[i]
        return tuple(strides)

    state = {}
    for pid, st in (sd.get("state") or {}).items():
        fixed = {}
        for k, v in st.items():
            if torch.is_tensor(v) and v.stride() != row_major(v.shape):
                out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
                fixed[k] = out.copy_(v)       # moves VALUES, not bytes
            else:
                fixed[k] = v
        state[pid] = fixed
    return {**sd, "state": state}


def relayout_optimizer_state(opt) -> int:
    """Make each Adam state tensor share its parameter's memory layout.

    Fused Adam requires params, grads and moments to agree on dtype, device
    AND layout. Every checkpoint written before the channels_last trunk
    carries NCHW moments, so a bare `--ckpt` resume would otherwise die with
    "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device,
    and layout" on the first opt.step().

    Keep this even if the trunk's layout is ever changed back: it is what
    makes checkpoints portable ACROSS a layout change, in both directions.
    Returns the number of tensors restrided (0 on a matching checkpoint).
    """
    n = 0
    for p, st in opt.state.items():
        for k, v in list(st.items()):
            if torch.is_tensor(v) and v.shape == p.shape and v.stride() != p.stride():
                st[k] = torch.empty_like(p).copy_(v)   # empty_like keeps p's format
                n += 1
    return n


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
    states and concatenated — the same fusion the trainer does. act_every
    repeats each decision for K ticks, matching frame-skip training."""

    def __init__(self, policy: Policy, packer: HeadPacker, device,
                 lidar=None, core=None, act_every: int = 1):
        self.policy, self.packer, self.device = policy, packer, device
        self.lidar, self.core = lidar, core
        self._k = max(1, int(act_every))
        self._tick = 0
        self._held = None

    def act(self, obs):
        if self._held is None or self._tick % self._k == 0:
            self._held = self._decide(obs)
        self._tick += 1
        return self._held

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
    def _decide(self, obs):
        logits, _ = self.policy(self._obs(obs))
        act = self.packer.pad(logits).argmax(-1)
        return act.to("cpu").numpy().astype(np.int32)


class SampledTorchPolicy(_TorchPolicyBase):
    """Acts by sampling the distribution — the policy training actually
    optimizes and logs. Under a high entropy coefficient the argmax mode can
    be much weaker (it drifts unoptimized while the stochastic policy learns
    to rely on its own action noise)."""

    @torch.inference_mode()
    def _decide(self, obs):
        logits, _ = self.policy(self._obs(obs))
        act, _ = sample_padded(self.packer.pad(logits))
        return act.to("cpu").numpy().astype(np.int32)


def race_progress(traj_path: Path, field) -> float:
    """Mean geodesic progress (d at spawn - min d reached, map units) over a
    recording's episodes — the race-relevant eval number. The freestyle
    eval/path metric is undirected horizontal wander: a spawn dive that
    races 0u of track still 'walks' thousands of units."""
    rows, per_ep = [], []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict) and "map" in row:
                rows = []
            elif isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict) and "end" in row and rows:
                a = np.asarray(rows, dtype=np.float64)
                d = field.sample(a[:, 1:4])
                per_ep.append(float(d[0] - d.min()))
                rows = []
    return float(np.mean(per_ep)) if per_ep else float("nan")


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
                # teleports jump thousands of units per tick; the fastest
                # legit move at sv_maxvelocity 4000 is ~69u (3D per-axis diag)
                tel = d > 110.0                             # teleport filter
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
    ap.add_argument("--maxvel", type=float, default=None,
                    help="sv_maxvelocity (default 2000, the GoldSrc stock "
                         "value all pre-race runs trained on; real surf "
                         "servers run 3500 — the per-axis clamp at 2000 "
                         "bleeds speed on fast lines and makes momentum maps "
                         "like cannonball physically uncompletable; ckpt "
                         "restores)")
    # 128 rays cost ~17ns each and dominate env time (13M steps/s eyeless ->
    # 0.44M at 16x8); drop to 12x6 for ~1.7x env throughput at coarser vision
    ap.add_argument("--lidar-w", type=int, default=None)   # 128; ckpt overrides
    ap.add_argument("--lidar-h", type=int, default=None)   # 64
    # fixed-gaze experiment: freeze view pitch at this angle (deg, + = up);
    # the pitch action head stays in the action space but is physically inert
    ap.add_argument("--fix-pitch", type=float, default=None)
    ap.add_argument("--free-pitch", action="store_true",
                    help="re-enable the pitch head when warm-starting a "
                         "fixed-gaze ckpt (view clamp [-70,+30] still applies)")
    # decisions every K physics ticks (100Hz physics / K): calmer camera,
    # human-scale reaction granularity, and ~K x cheaper policy+update per
    # game-second. Held yaw/pitch deltas keep applying, so turn RATE is
    # unchanged — only decision frequency drops.
    ap.add_argument("--act-every", type=int, default=None)   # 3; ckpt restores
    ap.add_argument("--pitch-rate", type=float, default=None,
                    help="max view-pitch delta per tick, deg (core default 10; "
                         "4 makes gaze deliberate instead of whippy)")
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
    # mixed spawns drop the agent U(drop-min, drop-max) above ramp faces with
    # randomized entry velocity/yaw/pitch — every scattered start is a live,
    # unfamiliar surf-catch situation (fall speed sqrt(2*g*h))
    ap.add_argument("--drop-min", type=float, default=400.0)
    ap.add_argument("--drop-max", type=float, default=800.0)
    # initial horizontal velocity range for drop spawns ("punch")
    ap.add_argument("--punch-min", type=float, default=100.0)
    ap.add_argument("--punch-max", type=float, default=400.0)
    ap.add_argument("--revisit-pen", type=float, default=None,
                    help="coverage reward: cost of entering an already-"
                         "visited voxel (default 0.25; ckpt restores)")
    ap.add_argument("--record-every", type=float, default=25e6)
    ap.add_argument("--ckpt-every", type=float, default=10e6)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--sb3", default=None)
    ap.add_argument("--reset-steps", action="store_true")
    ap.add_argument("--reward", choices=["forward", "path", "blend", "maxspeed",
                                         "coverage", "acro", "race"],
                    default=None,
                    help="forward = max displacement along spawn yaw (default; "
                         "path-length turned out to reward circling in place); "
                         "blend = curriculum: forward until --blend-start, then "
                         "anneal linearly to pure path-length by --blend-end")
    ap.add_argument("--blend-start", type=float, default=None)   # 100e6
    ap.add_argument("--blend-end", type=float, default=None)     # 200e6
    # ---- race objective (--reward race; needs maps/<map>.zones.json) ------
    ap.add_argument("--lidar-cell", type=float, default=None,
                    help="vision SDF voxel size, units (default: auto by map "
                         "volume — 16 for ski_2, 32 for cannonball-sized)")
    ap.add_argument("--race-dist", choices=["geodesic", "euclid"], default=None,
                    help="shaping distance: geodesic = through-the-track "
                         "field (minutes of one-time GPU bake per map); "
                         "euclid = straight-line A*-heuristic proxy (zero "
                         "precompute, scales to many maps, but shaping goes "
                         "negative around hairpins — more exploration needed)")
    ap.add_argument("--time-pen", type=float, default=None,      # 0.005/tick
                    help="race: per-tick time cost — with potential shaping "
                         "fixed, minimizing time IS the objective")
    ap.add_argument("--success-bonus", type=float, default=None,  # 50
                    help="race: paid on crossing the finish zone")
    ap.add_argument("--fail-pen", type=float, default=None,       # 0 = off
                    help="race: terminal penalty on death (falls, nets, "
                         "stall-kills; truncation exempt) — at an unlearned "
                         "frontier, shaping alone cannot prefer surviving a "
                         "catch over dying at max progress")
    ap.add_argument("--speed-coef", type=float, default=None,     # 0 = off
                    help="race: per-tick bonus speed_coef*h_speed/1000 — "
                         "tilts line choice toward carrying speed (0.005 => "
                         "0.01/tick at 2000 u/s, ~40%% of shaping income)")
    ap.add_argument("--stall-secs", type=float, default=None,     # 15
                    help="race: kill an episode whose distance-to-finish "
                         "best hasn't improved for this long (0 = off)")
    ap.add_argument("--respawn-frac", type=float, default=None,   # 0 = off
                    help="race: fraction of episodes respawned from recent "
                         "mid-run snapshots (Go-Explore style reset-to-state; "
                         "same position+velocity, view/speed perturbed, "
                         "harvested >=10s before their episode ended); the "
                         "rest start from the configured spawn pool")
    ap.add_argument("--respawn-margin", type=float, default=None,  # 10 s
                    help="race: only snapshots at least this many seconds "
                         "before the episode's end are respawnable (closer "
                         "states are usually already doomed)")
    ap.add_argument("--respawn-reservoir", type=int, default=None,  # 100k
                    help="race: FIFO reservoir of respawnable states")
    ap.add_argument("--respawn-speed", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),          # (0.9, 1.1)
                    help="race: spawn speed multiplier range for respawned "
                         "states; e.g. 1.0 1.5 practices speed-gated jumps "
                         "at up to +50%% entry speed")
    ap.add_argument("--int-coef", type=float, default=None,       # 0 = off
                    help="race: count-based intrinsic novelty — "
                         "int_coef/sqrt(visits) on entering a 256u map cell, "
                         "counts global across envs+episodes (self-anneals; "
                         "pushes the frontier past fail-walls). Calibration: "
                         "racing income is ~0.025/tick (100 shaping over a "
                         "run) and a full-speed dive crosses a fresh cell "
                         "every ~8 ticks, so 0.1-0.5 keeps novelty a frontier "
                         "premium rather than the main income; the total "
                         "intrinsic budget grows with map size (cells on the "
                         "line ~ d0/256) while shaping stays fixed at 100 — "
                         "re-tune when switching maps")
    ap.add_argument("--timing", action="store_true",
                    help="print one parse-friendly TIMING line per iteration "
                         "(per-phase ms; GPU phases via CUDA events read once "
                         "per iteration) — the perf harness, see "
                         "docs/perf-implementation-plan.md")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile on the PPO minibatch step "
                         "(default: compile it — 1.067x on the update, "
                         "measured; falls back to eager by itself if the "
                         "inductor toolchain is unavailable)")
    # with 128x64 vision the conv update dominates (94ms/minibatch fp32 vs
    # ~25ms bf16) — bf16 is now the default; --fp32 restores exact math at
    # ~3x the wall-clock (the old ~20% sample-efficiency tax measured on the
    # MLP is the price)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    # a bare `--ckpt` resume must not silently change the training objective:
    # settings that define the run (reward mode, blend window, episode length)
    # come from the checkpoint's saved config unless explicitly overridden
    def flag_given(name):
        # argparse also accepts --flag=value; the bare token test missed it
        # and let the ckpt silently override an explicitly passed flag
        return any(a == name or a.startswith(name + "=") for a in sys.argv[1:])

    ck = None
    obj_changed = False
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_cfg = ck.get("config") or {}
        restored = []
        # warm start onto a DIFFERENT objective: the old run's episode length
        # must not leak in (a race warm-started from a 700-tick forward ckpt
        # can never reach a 94ku finish — success would be structurally 0)
        obj_changed = (args.reward is not None
                       and ck_cfg.get("reward") not in (None, args.reward))
        if args.reward is None and ck_cfg.get("reward"):
            args.reward = ck_cfg["reward"]
            restored.append(f"reward={args.reward}")
        if not flag_given("--map") and ck_cfg.get("map"):
            args.map = str(ROOT / "maps" / f"{ck_cfg['map']}.bsp")
            restored.append(f"map={ck_cfg['map']}")
        if not flag_given("--spawn") and ck_cfg.get("spawn") and not obj_changed:
            args.spawn = ck_cfg["spawn"]     # bare resume kept falling back
            restored.append(f"spawn={args.spawn}")   # to platform before this
        if args.lidar_cell is None and ck_cfg.get("lidar_cell"):
            args.lidar_cell = float(ck_cfg["lidar_cell"])
            restored.append(f"lidar_cell={args.lidar_cell:g}")
        if args.time_pen is None and ck_cfg.get("time_pen") is not None:
            args.time_pen = float(ck_cfg["time_pen"])
            restored.append(f"time_pen={args.time_pen:g}")
        if args.success_bonus is None and ck_cfg.get("success_bonus") is not None:
            args.success_bonus = float(ck_cfg["success_bonus"])
            restored.append(f"success_bonus={args.success_bonus:g}")
        if args.fail_pen is None and ck_cfg.get("fail_pen") is not None:
            args.fail_pen = float(ck_cfg["fail_pen"])
            restored.append(f"fail_pen={args.fail_pen:g}")
        if args.speed_coef is None and ck_cfg.get("speed_coef") is not None:
            args.speed_coef = float(ck_cfg["speed_coef"])
            restored.append(f"speed_coef={args.speed_coef:g}")
        if args.stall_secs is None and ck_cfg.get("stall_secs") is not None:
            args.stall_secs = float(ck_cfg["stall_secs"])
            restored.append(f"stall_secs={args.stall_secs:g}")
        if args.race_dist is None and ck_cfg.get("race_dist"):
            args.race_dist = ck_cfg["race_dist"]
            restored.append(f"race_dist={args.race_dist}")
        if args.int_coef is None and ck_cfg.get("int_coef") is not None:
            args.int_coef = float(ck_cfg["int_coef"])
            restored.append(f"int_coef={args.int_coef:g}")
        if args.maxvel is None and ck_cfg.get("maxvel") is not None:
            args.maxvel = float(ck_cfg["maxvel"])
            restored.append(f"maxvel={args.maxvel:g}")
        if args.respawn_frac is None and ck_cfg.get("respawn_frac") is not None:
            args.respawn_frac = float(ck_cfg["respawn_frac"])
            restored.append(f"respawn_frac={args.respawn_frac:g}")
        if args.respawn_margin is None and ck_cfg.get("respawn_margin") is not None:
            args.respawn_margin = float(ck_cfg["respawn_margin"])
            restored.append(f"respawn_margin={args.respawn_margin:g}")
        if (args.respawn_reservoir is None
                and ck_cfg.get("respawn_reservoir") is not None):
            args.respawn_reservoir = int(ck_cfg["respawn_reservoir"])
        if args.respawn_speed is None and ck_cfg.get("respawn_speed"):
            args.respawn_speed = [float(v) for v in ck_cfg["respawn_speed"]]
            restored.append(f"respawn_speed={args.respawn_speed[0]:g}-"
                            f"{args.respawn_speed[1]:g}")
        if ck_cfg.get("blend"):
            if args.blend_start is None:
                args.blend_start = float(ck_cfg["blend"][0])
                restored.append(f"blend_start={args.blend_start:g}")
            if args.blend_end is None:
                args.blend_end = float(ck_cfg["blend"][1])
                restored.append(f"blend_end={args.blend_end:g}")
        if args.ep_ticks is None and ck_cfg.get("ep_ticks") and not obj_changed:
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
        if (args.fix_pitch is None and not args.free_pitch
                and ck_cfg.get("fix_pitch") is not None):
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
        if args.act_every is None:
            args.act_every = int(ck_cfg.get("act_every", 1))
            restored.append(f"act_every={args.act_every}")
        if args.pitch_rate is None and ck_cfg.get("pitch_rate") is not None:
            args.pitch_rate = float(ck_cfg["pitch_rate"])
            restored.append(f"pitch_rate={args.pitch_rate:g}")
        if args.revisit_pen is None and ck_cfg.get("revisit_pen") is not None:
            args.revisit_pen = float(ck_cfg["revisit_pen"])
            restored.append(f"revisit_pen={args.revisit_pen:g}")
        if not flag_given("--punch-min") and ck_cfg.get("punch_min") is not None:
            args.punch_min = float(ck_cfg["punch_min"])
            args.punch_max = float(ck_cfg.get("punch_max", args.punch_max))
            restored.append(f"punch={args.punch_min:g}-{args.punch_max:g}")
        if restored:
            print("restored from checkpoint config: " + ", ".join(restored))
    if args.reward is None:
        args.reward = "forward"
    if args.blend_start is None:
        args.blend_start = 100e6
    if args.blend_end is None:
        args.blend_end = 200e6
    if args.ep_ticks is None:
        # race: "play until you finish" — the stagnation kill does the real
        # episode control, the 2-minute cap is just a backstop
        args.ep_ticks = 12000 if args.reward == "race" else 700
    if args.race_dist is None:
        args.race_dist = "geodesic"
    if args.time_pen is None:
        args.time_pen = 0.005
    if args.success_bonus is None:
        args.success_bonus = 50.0
    if args.fail_pen is None:
        args.fail_pen = 0.0
    if args.speed_coef is None:
        args.speed_coef = 0.0
    if args.stall_secs is None:
        # euclid shaping legitimately runs negative on away-from-goal legs
        # (hairpins) — a tight no-improvement window would execute progress
        args.stall_secs = 30.0 if args.race_dist == "euclid" else 15.0
    if args.int_coef is None:
        args.int_coef = 0.0
    if args.maxvel is None:
        args.maxvel = 2000.0     # every pre-race ckpt trained under this
    if args.respawn_frac is None:
        args.respawn_frac = 0.0
    if args.respawn_margin is None:
        args.respawn_margin = 10.0
    if args.respawn_reservoir is None:
        args.respawn_reservoir = 100_000
    if args.respawn_speed is None:
        args.respawn_speed = [0.9, 1.1]
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
    if args.act_every is None:
        args.act_every = 3
    K = max(1, int(args.act_every))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    tm = PhaseTimer(args.timing, device.type == "cuda")
    use_graphs = device.type == "cuda" and not args.no_graphs
    use_compile = device.type == "cuda" and not args.no_compile
    use_bf16 = device.type == "cuda" and not args.fp32
    N, T = args.envs, args.n_steps
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)

    # cores run EYELESS (13M raw steps/s); vision is rendered on the GPU from
    # the map SDF and fused into the obs here in the trainer
    if args.fix_pitch is not None:
        pitch_rate = 0.0
    else:
        pitch_rate = args.pitch_rate if args.pitch_rate is not None else -1.0
    cfg = default_config(num_envs=N, spawn_mode=2, max_episode_ticks=args.ep_ticks,
                         water_fail=1, yaw_jitter_deg=args.yaw_jitter,
                         sv_maxvelocity=args.maxvel,
                         lidar_w=0, lidar_h=0, pitch_rate_max_deg=pitch_rate)
    core = SurfCore(args.map, cfg)

    # race objective: labeled finish zone + geodesic distance-to-finish field
    goal_field = None
    goal_box = None
    race_d0 = None
    if args.reward == "race":
        zones = load_zones(args.map)
        if not zones.get("end"):
            raise SystemExit(
                f"--reward race needs an end zone for {Path(args.map).stem}: "
                f"auto-extraction found none — hand-label "
                f"maps/{Path(args.map).stem}.zones.json (see surfgym/zones.py)")
        goal_box = zones["end"]
        if args.race_dist == "euclid":
            from surfgym.goalfield import EuclidField
            goal_field = EuclidField(goal_box)
        else:
            cell = args.lidar_cell or pick_cell(core)
            goal_field = build_goal_field(core, goal_box, cell=cell)
        core.set_goal_box(goal_box["mins"], goal_box["maxs"])
        if args.keep_teleports:
            print("race mode forces the teleport-ends-episode rule "
                  "(--keep-teleports ignored: fallers respawn at start)")
            args.keep_teleports = False

    if args.reward == "race":
        # game-authentic race starts: the map's own spawn points, facing
        # along the track (entity yaw on ported maps is unreliable)
        raw = map_spawn_pool(core)
        plat_pool = map_spawn_pool(core, yaw=goal_field.descent_yaw(raw["origin"]))
        race_d0 = float(np.mean(goal_field.sample(raw["origin"])))
        print(f"race: start geodesic {race_d0:.0f}u, "
              f"finish box {goal_box['mins']} .. {goal_box['maxs']}")
    else:
        plat_pool = platform_spawn_pool(core)
    # platform starts gaze slightly down regardless of pitch mode
    plat_pool["pitch"] = args.fix_pitch if args.fix_pitch is not None else -10.0
    if args.spawn == "platform":
        pool = plat_pool
    else:
        dp = drop_spawn_pool(core, h_range=(args.drop_min, args.drop_max),
                             speed_range=(args.punch_min, args.punch_max))
        if goal_field is not None:
            # exploring starts must lie ON the track: reachable (not in a
            # disconnected bonus area) and short of the finish
            d_dp = goal_field.sample(dp["origin"])
            keep = goal_field.reachable(dp["origin"]) & (d_dp > 400.0)
            print(f"race: drop pool {len(dp)} -> {int(keep.sum())} on-track")
            dp = dp[keep]
        pool = dp if args.spawn == "ramp" else np.concatenate([plat_pool, dp])
    if not args.keep_teleports:
        core.set_teleport_fail(True)
    core.set_spawn_pool(pool)
    print(f"pool({args.spawn}) {len(pool)} | envs {N} | {device} | "
          f"omp={os.environ['OMP_NUM_THREADS']} | "
          f"graphs={use_graphs} bf16={use_bf16}"
          + (f" | pitch fixed {args.fix_pitch:g}" if args.fix_pitch is not None else ""))
    respawn = None
    if args.respawn_frac > 0.0:
        respawn = RespawnBuffer(N, reservoir=args.respawn_reservoir,
                                margin_ticks=int(args.respawn_margin * 100.0),
                                map_id=Path(args.map).stem)
        print(f"respawn: {args.respawn_frac:.0%} of episodes from mid-run "
              f"snapshots, harvested >= {args.respawn_margin:g}s before "
              f"episode end")

    # eval on the game-authentic platform start regardless of the training
    # pool, so eval/* metrics and recordings stay comparable across runs
    eval_core = SurfCore(args.map, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=args.ep_ticks, water_fail=1,
        sv_maxvelocity=args.maxvel,
        lidar_w=0, lidar_h=0, pitch_rate_max_deg=pitch_rate))
    if not args.keep_teleports:
        eval_core.set_teleport_fail(True)
    if goal_box is not None:
        eval_core.set_goal_box(goal_box["mins"], goal_box["maxs"])
    eval_core.set_spawn_pool(plat_pool)

    lidar = GpuLidar(core, args.lidar_w, args.lidar_h,
                     range_units=args.lidar_range, near_range=args.lidar_near,
                     cell=(args.lidar_cell or pick_cell(core)),
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
    elif args.reward == "maxspeed":
        reward_fn = MaxSpeedReward(0.05)     # return = 0.05 * episode top h-speed
    elif args.reward == "coverage":
        reward_fn = CoverageSpeedReward(
            0.001, 512.0, revisit_pen=(args.revisit_pen
                                       if args.revisit_pen is not None else 0.25))
    elif args.reward == "acro":
        reward_fn = AcroCoverageReward(
            0.001, 512.0, revisit_pen=(args.revisit_pen
                                       if args.revisit_pen is not None else 1.0))
    elif args.reward == "race":
        # 100 total shaping over a full start->finish run regardless of map
        # size (generalist-comparable across maps); bonus + time cost on top
        reward_fn = RaceReward(goal_field, scale=100.0 / race_d0,
                               time_pen=args.time_pen,
                               success_bonus=args.success_bonus,
                               stall_ticks=int(args.stall_secs * 100.0),
                               int_coef=args.int_coef,
                               fail_pen=args.fail_pen)
        reward_fn.speed_coef = args.speed_coef
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
        n_re = relayout_optimizer_state(opt)
        if n_re:
            print(f"optimizer state restrided to the params' layout ({n_re} tensors)")
        global_step = 0 if args.reset_steps else int(ck.get("global_step", 0))
        if (isinstance(reward_fn, RaceReward)
                and ck.get("int_counts") is not None):
            reward_fn.restore_counts(ck["int_counts"])
            n_visits = int(np.asarray(ck["int_counts"]).sum(dtype=np.int64))
            print(f"restored novelty counts ({n_visits:,} visits)")
        if respawn is not None and ck.get("respawn") is not None:
            respawn.load_state_dict(ck["respawn"])
            print(f"restored respawn reservoir ({respawn.size:,} states)")
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
                       "drop_min": args.drop_min, "drop_max": args.drop_max,
                       "punch_min": args.punch_min, "punch_max": args.punch_max,
                       "revisit_pen": args.revisit_pen,
                       "act_every": K, "pitch_rate": pitch_rate,
                       "lidar_cell": args.lidar_cell or pick_cell(core),
                       "time_pen": (args.time_pen if args.reward == "race"
                                    else None),
                       "success_bonus": (args.success_bonus
                                         if args.reward == "race" else None),
                       "stall_secs": (args.stall_secs
                                      if args.reward == "race" else None),
                       "fail_pen": (args.fail_pen
                                    if args.reward == "race" else None),
                       "speed_coef": (args.speed_coef
                                      if args.reward == "race" else None),
                       "race_dist": (args.race_dist
                                     if args.reward == "race" else None),
                       "int_coef": (args.int_coef
                                    if args.reward == "race" else None),
                       "maxvel": args.maxvel,
                       "respawn_frac": args.respawn_frac,
                       "respawn_margin": args.respawn_margin,
                       "respawn_reservoir": args.respawn_reservoir,
                       "respawn_speed": args.respawn_speed,
                       "ep_ticks": args.ep_ticks, "epochs": args.epochs,
                       "graphs": use_graphs, "compile": use_compile,
                       "bf16": use_bf16}}
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    CSV_COLS = ["time/total_timesteps", "rollout/ep_rew_mean",
                "rollout/ep_len_mean", "time/fps", "train/loss",
                "train/value_loss", "train/entropy_loss",
                "train/approx_kl", "eval/fwd_max", "eval/path",
                "eval/speed_max", "train/blend_w",
                "race/success_rate", "race/finish_s",
                "race/eval_progress"]
    csv_path = out / "progress.csv"
    if csv_path.exists() and csv_path.stat().st_size:
        # schema migration: rows always carry len(CSV_COLS) fields, so a
        # resumed pre-extension file needs its header padded or the new
        # columns are silently dropped by every csv reader
        text = csv_path.read_text(encoding="utf-8").splitlines(True)
        head = text[0].rstrip("\r\n").split(",")
        if head != CSV_COLS and head == CSV_COLS[:len(head)]:
            nl = text[0][len(text[0].rstrip("\r\n")):] or "\r\n"
            text[0] = ",".join(CSV_COLS) + nl
            csv_path.write_text("".join(text), encoding="utf-8")
            print(f"progress.csv header extended to {len(CSV_COLS)} columns")
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        csv_w.writerow(CSV_COLS)

    # ---- static rollout buffers (graph-capturable) --------------------------
    # S3: the rollout buffer is split, and its depth half is bf16. This is
    # numerically EXACT, not an approximation: autocast already rounds the
    # depth image to bf16 on the way into the conv, so storing the same
    # rounded value hands the update precisely the tensor it saw before.
    # It halves the buffer (8.6 GB -> 4.3 GB) and the per-minibatch gather.
    b_scal = torch.zeros((T, N, N_SCALAR), device=device)
    b_img = torch.zeros((T, N, obs_dim - N_SCALAR), device=device,
                        dtype=torch.bfloat16 if use_bf16 else torch.float32)
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
        t0 = tm.now()
        vis_np[:, 0:3] = sv_view["origin"]
        vis_np[:, 3] = sv_view["yaw"]
        vis_np[:, 4] = sv_view["pitch"]
        vis_np[:, 5] = sv_view["ducked"]
        vis_gpu.copy_(vis_pin, non_blocking=True)
        ev = tm.gpu_start("lidar")
        depth = lidar.render(vis_gpu[:, 0:3], vis_gpu[:, 3],
                             vis_gpu[:, 4], vis_gpu[:, 5])
        dst[:, N_SCALAR:].copy_(depth.reshape(N, -1))
        tm.gpu_end(ev)
        tm.add("vis_cpu", t0)

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
        ev = tm.gpu_start("rollout_fwd")
        if graph is not None:
            graph.replay()
        else:
            with torch.no_grad():
                step_compute()
        tm.gpu_end(ev)

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
    last_latest_save = 0.0                   # force one write on iteration 1
    eval_fwd = eval_path = eval_speed = eval_prog = float("nan")
    t_start, step_start = time.perf_counter(), global_step

    def save_ckpt(tag):
        state = {"policy": policy.state_dict(),
                 "optimizer": contiguous_optimizer_state(opt.state_dict()),
                 "global_step": global_step, "config": meta["config"]}
        if isinstance(reward_fn, RaceReward):
            # novelty counts are cross-episode reward state: without them a
            # resume re-pays "first visit" for the whole beaten path
            state["int_counts"] = reward_fn.counts_state()
        if respawn is not None:
            state["respawn"] = respawn.state_dict()   # keep the frontier
        torch.save(state, out / f"ckpt_{tag}.pt")

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                         enabled=use_bf16)

    # ---- the PPO minibatch step, as one compilable function -----------------
    # Everything here is static-shaped (mb is constant), so inductor sees one
    # graph and never re-traces. The gathers stay INSIDE: fusing them with the
    # bf16 cast is part of what the compile buys.
    ent_t = torch.zeros((), device=device)

    def mb_step(f_scal, f_img, f_act, f_logp, f_adv, f_ret, idx, ent_coef):
        with amp:
            logits, value = policy.forward_split(f_scal[idx], f_img[idx])
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
        return pg + args.vf * vl + ent_coef * el, pg, vl, el, logp

    MB = T * N // args.minibatches            # constant: the compiled shape
    if use_compile:
        # max-autotune-no-cudagraphs, not reduce-overhead: measured 1.067x vs
        # 1.011x on the isolated step (tools/bench_update.py). The
        # no-cudagraphs part matters — the rollout already owns a CUDA graph,
        # and inductor's own graph capture has nothing to win here anyway
        # (the update is 99.8% GPU-busy, not launch-bound).
        #
        # Warm it HERE rather than on iteration 1: autotune takes minutes, and
        # a toolchain failure has to land somewhere it can fall back to eager
        # instead of killing an overnight run. The warm-up runs a real
        # forward+backward on the (zeroed) rollout buffers and then drops the
        # gradients — no opt.step(), so no weights move.
        eager_mb_step = mb_step
        try:
            t_c = time.perf_counter()
            mb_step = torch.compile(eager_mb_step,
                                    mode="max-autotune-no-cudagraphs")
            mb_step(b_scal.reshape(T * N, N_SCALAR),
                    b_img.reshape(T * N, obs_dim - N_SCALAR),
                    b_act.reshape(T * N, NACT),
                    b_logp.reshape(-1), b_val.reshape(-1), b_rew.reshape(-1),
                    torch.arange(MB, device=device), ent_t)[0].backward()
            opt.zero_grad(set_to_none=True)
            print(f"torch.compile: minibatch step compiled in "
                  f"{time.perf_counter() - t_c:.0f}s "
                  f"(max-autotune-no-cudagraphs)")
        except Exception as exc:            # pragma: no cover
            print(f"torch.compile failed ({exc!r}) — eager update")
            mb_step = eager_mb_step
            opt.zero_grad(set_to_none=True)
            use_compile = False
            meta["config"]["compile"] = False      # keep run.json honest
            (out / "run.json").write_text(json.dumps(meta, indent=2),
                                          encoding="utf-8")

    it_no = 0
    while global_step < int(args.steps):
        it_no += 1
        tm.start_iter()
        if hasattr(reward_fn, "set_step"):
            reward_fn.set_step(global_step)   # authoritative (survives resume)
        t_pool = tm.now()
        if respawn is not None and respawn.size >= 2000:
            # refresh the spawn pool: fresh starts + perturbed mid-run
            # states. The 2000-state floor keeps the first lucky episode's
            # snapshots from seeding 90% of the fleet (degenerate,
            # self-reinforcing rollout correlation).
            core.set_spawn_pool(respawn.build_pool(
                pool, fresh_frac=1.0 - args.respawn_frac,
                vel_scale=tuple(args.respawn_speed),
                pitch_jitter=0.0 if args.fix_pitch is not None else 5.0))
        tm.add("pool", t_pool)
        # ---------------- rollout ----------------
        t_roll = tm.now()
        with torch.no_grad():
            for t in range(T):
                policy_step()
                t_sync = tm.now()
                b_scal[t].copy_(static_obs[:, :N_SCALAR])
                b_img[t].copy_(static_obs[:, N_SCALAR:])
                b_act[t].copy_(static_act)
                b_logp[t].copy_(static_logp)
                b_val[t].copy_(static_val)
                act_pin.copy_(static_act, non_blocking=True)
                torch.cuda.synchronize() if device.type == "cuda" else None
                np.copyto(act_np32, act_pin.numpy(), casting="unsafe")
                tm.add("sync_copy", t_sync)
                # action repeat: hold the decision for K physics ticks (100Hz
                # physics, 100/K Hz decisions). Rewards sum over the repeat;
                # GAE runs at decision granularity with gamma^K. Sub-tick
                # episode ends mark the decision boundary done; the couple of
                # post-reset sub-ticks inherit the held action (standard
                # frame-skip semantics, negligible contamination).
                if isinstance(reward_fn, RaceReward):
                    sm = reward_fn.pop_stall_mask()
                    if sm is not None:
                        core.force_fail(sm)     # stagnation kill, next tick
                r_acc = np.zeros(N, np.float32)
                ended_acc = np.zeros(N, bool)
                for _j in range(K):
                    t_env = tm.now()
                    o2, base_r, done, trunc, term_obs = core.step(act_np32)
                    tm.add("env", t_env)
                    t_rew = tm.now()
                    r = reward_fn(prev_obs, o2, term_obs, base_r, done, trunc, core)
                    prev_obs = o2.copy()
                    ended = (done | trunc).astype(bool)
                    tm.add("reward_py", t_rew)
                    t_book = tm.now()
                    ep_ret += r          # pure collected reward only: the trunc
                    ep_len += 1          # bootstrap below is a GAE construct and
                                         # must not inflate the logged return
                    tm.add("book", t_book)
                    t_boot = tm.now()
                    if trunc.any():
                        ti = np.flatnonzero(trunc.astype(bool) & ~done.astype(bool))
                        if len(ti):
                            # states are already the NEW episode's (autoreset),
                            # so the terminal pose is reconstructed from the
                            # terminal scalar obs to render its lidar for V(s_T)
                            to = term_obs[ti]
                            ts = torch.as_tensor(to, dtype=torch.float32,
                                                 device=device)
                            pos = torch.as_tensor(to[:, 12:15] * 2000.0 + map_center,
                                                  dtype=torch.float32, device=device)
                            yawd = torch.rad2deg(torch.atan2(ts[:, 7], ts[:, 8]))
                            depth = lidar.render(pos, yawd, ts[:, 9] * 90.0,
                                                 ts[:, 5])
                            full = torch.cat([ts, depth.reshape(len(ti), -1)],
                                             dim=1)
                            tv = policy(full)[1]
                            r[ti] += args.gamma * tv.to("cpu").numpy()
                    tm.add("boot", t_boot)
                    t_book = tm.now()
                    if ended.any():
                        for i in np.flatnonzero(ended):
                            ret_hist.append(ep_ret[i]); len_hist.append(ep_len[i])
                        ep_ret[ended] = 0; ep_len[ended] = 0
                    tm.add("book", t_book)
                    t_resp = tm.now()
                    if respawn is not None:
                        # never snapshot stagnating states: the pre-END
                        # margin can't see stall onsets (kills fire 15s in)
                        stag = (reward_fn.stagnant_mask()
                                if isinstance(reward_fn, RaceReward) else None)
                        respawn.observe(sv_view, ended, stagnant=stag)
                    tm.add("respawn", t_resp)
                    r_acc += r
                    ended_acc |= ended
                    global_step += N
                t_sync = tm.now()
                b_rew[t].copy_(torch.from_numpy(r_acc).to(device, non_blocking=True))
                b_done[t].copy_(torch.from_numpy(
                    ended_acc.astype(np.float32)).to(device, non_blocking=True))
                obs_pin.copy_(torch.from_numpy(o2))
                static_obs[:, :N_SCALAR].copy_(obs_pin, non_blocking=True)
                tm.add("sync_copy", t_sync)
                fill_vision(static_obs)
            tm.add("rollout_wall", t_roll)
            t_gae = tm.now()
            ev_gae = tm.gpu_start("gae_gpu")
            _, last_val = policy(static_obs)
            adv = torch.zeros_like(b_rew)
            lastgae = torch.zeros(N, device=device)
            g_eff = args.gamma ** K          # decision-granularity discount
            for t in reversed(range(T)):
                nextval = last_val if t == T - 1 else b_val[t + 1]
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + g_eff * nextval * nonterm - b_val[t]
                lastgae = delta + g_eff * args.gae * nonterm * lastgae
                adv[t] = lastgae
            ret = adv + b_val
            tm.gpu_end(ev_gae)
            tm.add("gae", t_gae)

        # ---------------- update ----------------
        t_upd = tm.now()
        ev_upd = tm.gpu_start("update_gpu")
        f_scal = b_scal.reshape(T * N, N_SCALAR)
        f_img = b_img.reshape(T * N, obs_dim - N_SCALAR)
        f_act = b_act.reshape(T * N, NACT)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = ret.reshape(-1)
        mb = MB
        if args.ent_final is not None:
            frac = min(1.0, global_step / max(1.0, float(args.steps)))
            ent_coef = args.ent + (args.ent_final - args.ent) * frac
        else:
            ent_coef = args.ent
        ent_t.fill_(ent_coef)     # a tensor, not a float: a Python scalar in a
        # compiled signature is baked in as a constant, so an --ent-final
        # schedule would recompile the whole region every iteration
        kl = loss_v = loss_pi = loss_ent = 0.0
        for _ in range(args.epochs):
            perm = torch.randperm(T * N, device=device)
            for s0 in range(0, T * N, mb):
                idx = perm[s0:s0 + mb]
                ev_mb = tm.gpu_start("mb_gpu")
                loss, pg, vl, el, logp = mb_step(
                    f_scal, f_img, f_act, f_logp, f_adv, f_ret, idx, ent_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                tm.gpu_end(ev_mb)     # before the float() syncs: mb_gpu vs
                # update measures how much of the update is GPU vs host gaps
                with torch.no_grad():
                    kl = float((f_logp[idx] - logp).mean())
                loss_v, loss_pi, loss_ent = float(vl), float(pg), float(el)
        tm.gpu_end(ev_upd)
        tm.add("update", t_upd)

        # ---------------- logging / artifacts ----------------
        fps = (global_step - step_start) / (time.perf_counter() - t_start)
        rmean = float(np.mean(ret_hist)) if ret_hist else 0.0
        lmean = float(np.mean(len_hist)) if len_hist else 0.0
        race_sr = race_fin = race_int = float("nan")
        if isinstance(reward_fn, RaceReward):
            rs = reward_fn.pop_stats()
            race_sr, race_fin = rs["success_rate"], rs["finish_s"]
            race_int = rs["int_per_ep"]
        t_rec = tm.now()
        if global_step >= next_record:
            next_record = global_step + int(args.record_every)
            path = out / f"traj_{global_step:010d}.jsonl"
            # per-recording seed: a fixed seed replays the same few spawns
            # forever, and with a wide per-spawn spread (56..100 at 2.6B) a
            # single weak-tail spawn makes every eval look bad
            n_rec = 3 if args.reward == "race" else 5   # race eps run minutes
            record_rollout(eval_core,
                           GreedyTorchPolicy(policy, packer, device,
                                             lidar, eval_core, K),
                           path, episodes=n_rec, max_ticks=n_rec * args.ep_ticks,
                           seed=global_step & 0x7FFFFFFF)
            st = episode_stats(path)
            eval_fwd = float(np.mean([e["fwd_max"] for e in st])) if st else 0.0
            eval_path = float(np.mean([e["path"] for e in st])) if st else 0.0
            eval_speed = float(np.mean([e["speed_max"] for e in st])) if st else 0.0
            prog_note = ""
            if goal_field is not None:
                eval_prog = race_progress(path, goal_field)
                prog_note = (f"  track {eval_prog:7.0f}u"
                             f"/{race_d0:.0f}u" if eval_prog == eval_prog else "")
            print(f"[{global_step:>13,d}] greedy: fwd {eval_fwd:7.0f}u  path "
                  f"{eval_path:7.0f}u  peak {eval_speed:6.0f} u/s{prog_note}"
                  f" -> {path.name}")
            spath = out / f"traj_{global_step:010d}_stoch.jsonl"
            record_rollout(eval_core,
                           SampledTorchPolicy(policy, packer, device,
                                              lidar, eval_core, K),
                           spath, episodes=n_rec, max_ticks=n_rec * args.ep_ticks,
                           seed=global_step & 0x7FFFFFFF)
            sst = episode_stats(spath)
            if sst:
                print(f"[{global_step:>13,d}] stoch : path "
                      f"{np.mean([e['path'] for e in sst]):7.0f}u -> {spath.name}")
        tm.add("record", t_rec)
        t_ck = tm.now()
        if global_step >= next_ckpt:
            next_ckpt = global_step + int(args.ckpt_every)
            save_ckpt(f"{global_step:010d}")
        # ckpt_latest is for crash recovery + dashboard record buttons: a
        # ~1-min cadence loses nothing and stops paying a 24-35MB torch.save
        # every iteration
        if time.perf_counter() - last_latest_save >= 60.0:
            save_ckpt("latest")
            last_latest_save = time.perf_counter()
        tm.add("ckpt", t_ck)
        csv_w.writerow([global_step, round(rmean, 4), round(lmean, 1), round(fps),
                        round(loss_pi + args.vf * loss_v + ent_coef * loss_ent, 5),
                        round(loss_v, 5), round(loss_ent, 5), round(kl, 6),
                        round(eval_fwd, 1), round(eval_path, 1),
                        round(eval_speed, 1),
                        round(getattr(reward_fn, "weight", 0.0), 4),
                        round(race_sr, 4) if race_sr == race_sr else "",
                        round(race_fin, 2) if race_fin == race_fin else "",
                        round(eval_prog, 1) if eval_prog == eval_prog else ""])
        csv_f.flush()
        race_note = ""
        if isinstance(reward_fn, RaceReward) and race_sr == race_sr:
            race_note = f"  win {race_sr:6.2%}"
            if race_fin == race_fin:
                race_note += f" @{race_fin:5.1f}s"
            if reward_fn.int_coef > 0.0 and race_int == race_int:
                race_note += f"  int {race_int:5.2f}/ep"
            if respawn is not None:
                race_note += f"  res {respawn.size:,}"
        print(f"step {global_step:>13,d}  rew {rmean:8.2f}  len {lmean:6.0f}  "
              f"fps {fps:,.0f}  kl {kl:.4f}  ent {ent_coef:.4f}{race_note}")
        tm.flush(it_no)

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
