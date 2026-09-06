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
        n = len(os.sched_getaffinity(0))     # respects cpuset/taskset limits
    except AttributeError:                    # pragma: no cover - Windows
        n = os.cpu_count() or 8
    # ... but NOT the CFS quota, which is how a container is given a
    # FRACTION of a host. A vast box at gpu_frac 0.25 hands out a 7.68-CPU
    # quota while affinity still reports all 32 cores, so this asked for 16
    # threads on 7.68 CPUs and cost 21.7% (280,673 -> 341,697 steps/s at
    # OMP 8). Measured 2026-08-23.
    quota = _cgroup_cpu_quota()
    n = min(n, quota or n)
    # torchrun (DDP): the ranks split one machine, so each team takes a
    # 1/world_size share of the half-the-cores rule - UNLESS the launcher
    # already narrowed this rank's affinity (numactl/taskset), in which
    # case n is per-rank already and halving is all that is left. The
    # affinity is "narrowed" when the per-rank share times the rank count
    # fits in the machine. The quota bounds BOTH sides of that test: it is
    # a container-wide budget every rank shares, so comparing a
    # quota-limited n against the raw host core count would read as
    # "already narrowed" and hand every rank the whole budget.
    lws = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    total = os.cpu_count() or n
    if quota:
        total = min(total, quota)
    div = 2 if n * lws <= total else 2 * lws
    return str(max(4, min(32, n // div)))


def _cgroup_cpu_quota():
    """CPUs the CFS quota actually allows, or None when unlimited."""
    try:                                      # cgroup v2
        raw = open("/sys/fs/cgroup/cpu.max").read().split()
        if raw[0] != "max":
            return max(1, int(float(raw[0]) / float(raw[1])))
        return None
    except (OSError, ValueError, IndexError):
        pass
    try:                                      # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        pr = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        return max(1, q // pr) if q > 0 and pr > 0 else None
    except (OSError, ValueError):
        return None


os.environ.setdefault("OMP_NUM_THREADS", _default_omp_threads())

import argparse
import csv
import hashlib
import json
import math
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
from surfgym import distributed
from surfgym.core import STATE_DTYPE
from surfgym.goalfield import build_goal_field
from surfgym.mapfleet import HeldoutSlot, MapFleet, MapSlot
from surfgym.obsaux import ACT_FEAT, CMP_FEAT, ObsAux
from surfgym.privfeat import PRIV_DIM, PRIV_FEATURES, PrivFeat, velocity_from_obs
from surfgym.record import record_rollout
from surfgym.bc import BCDataset
from surfgym.respawn import DemoCurriculum, RespawnBuffer
from surfgym.rewards import (AcroCoverageReward, BlendedReward,
                             CoverageSpeedReward, ForwardProgressReward,
                             MaxSpeedReward, PathLengthReward, RaceReward,
                             drop_spawn_pool, map_spawn_pool,
                             platform_spawn_pool, ramp_spawn_pool)
from surfgym.route import ArcProgress, RouteLine
from surfgym.tailrl import tail_weights
from surfgym.tick import episode_seconds
from surfgym.view import (K_MAX, LOG_STD_INIT, OFF_ALPHA, OFF_MAX,
                          PITCH_ABS_HALF, PITCH_ABS_MID, WARP_ALPHA, n_z,
                          view_mode_code)
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

# ---- --chunk: temporally-abstract actions (docs/action-chunks-design.md) ----
# One policy decision picks ONE index out of K behavior codes; a LEARNABLE
# decoder — an (K, H, sum(NVEC)) logit table inside Policy — expands that code
# into H consecutive action distributions, one per decision. The action stream
# reaching the engine stays at 100/act_every Hz (only the DELIBERATION rate
# drops H-fold), so the sharp 33 Hz control optimum the act-every ladder found
# is untouched while trunk forwards and lidar renders drop H x.
#
# The decoder reads NO observation, so expanding a code is a gather, not H
# forward passes; PPO trains it end-to-end through the joint log-prob
# log pi(code|s) + sum_h log p(a_h | decoder[code, h]), so what a code MEANS is
# discovered by the run rather than imported from a fitted codebook.
KCODES = 64                           # default K (--codes)
# "hold everything": yaw bin 7 = 0 deg/tick, pitch bin 3 = 0 deg/tick,
# fwd/side centred, no jump, no duck. What the tail of a chunk is masked to
# once its episode ended mid-chunk (design doc 4.3) — static shapes, so the
# CUDA-graphed region never sees a ragged loop.
NEUTRAL_ACT = (7, 3, 1, 1, 0, 0)

# ---- --view-continuous: heads 0 (yaw) and 1 (pitch) as squashed Gaussians
# (docs/contyaw.md, surfgym/view.py). The policy's flat output grows two
# MEAN columns after the sum(NVEC) categorical logits; a state-independent
# log sigma per head is a parameter. The pre-tanh z is THE action PPO
# scores (Gaussian log-density, no Jacobian: the tanh, the warp and the
# core's clamp are the environment's deterministic response, exactly as the
# bin table was). The four other heads stay categorical.
N_VIEW = 2                            # the two view heads
LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0  # clamp on log sigma (sigma 0.0067..2.7)
LOG2PI = float(np.log(2.0 * np.pi))
# --bc-file under the flag: the view heads' cloning loss is the Gaussian
# NLL of the target z at THIS fixed sigma - an MSE on mu with a fixed scale
# - not at the policy's live sigma. Measured 2026-09-06 on the transplant
# (sigma 0.05, the teacher's own spread): the live-sigma NLL's gradient on
# mu scales as 1/sigma^2 (400x), dragged mu by a sigma per iteration and
# PPO's approx_kl read 1.46 / 0.62 on the first two iterations; the same
# NLL also fits sigma to the residual (0.03 z) and collapses it further.
# The policy's sigma is PPO's business (its entropy term and its ratio);
# the BC term only says where mu should be, at the resolution of the init.
BC_VIEW_SIGMA = 0.3
# ---- --view-absolute {velocity,world}: ABSOLUTE targets on top of the flag
# (docs/contyaw.md "Absolute targets", core view_mode 1 / 2). The core's
# (N, 2) row becomes (yaw target deg, pitch target deg) and the core turns
# toward it by at most the per-tick ceiling, EVERY tick, from the live
# velocity and yaw. velocity: z = (z_yaw, z_pitch), the yaw target is an
# offset from the heading of v_h through off_warp (u = +-0.5 -> +-10 deg,
# u = +-1 -> +-180). world: z = (z_c, z_s, z_pitch), the yaw target is
# atan2(tanh z_s, tanh z_c) - the cos/sin reading of the +-180 seam; the
# vector's norm is ignored. Pitch target = -20 + 50 tanh z in both. z is
# still THE action PPO scores; only the environment's deterministic map
# of it changes (view_from_z_t), so sample_view / logprob_entropy_view /
# the BC-free update are the delta path's, with n_z = 3 in world mode.
# Absent (None) = the delta command above, byte-identical.


# --- --trunk resnet: a residual conv trunk sized for the 64x32 depth image -
# An ImageNet stem (7x7/2 + maxpool/2) is wrong here: 32x64 collapses to 1x2
# by layer4, so the last two stages see one pixel of spatial context. This
# keeps stride 1 at the stem and takes exactly three /2 stages, landing on
# 4x8 - the same grid AdaptiveAvgPool2d((4,8)) hands the plain trunk, so the
# final Linear and everything downstream of it are unchanged.
#
# NORM = GroupNorm, not BatchNorm, and the reason is mechanical rather than
# stylistic (see docstring of _resnet_trunk).
_GN_GROUPS = 8


class _BasicBlock(nn.Module):
    """conv3x3-GN-ReLU-conv3x3-GN (+ 1x1 projection when the shape changes),
    then ReLU. Bias-free convs: the norm right after them has its own shift,
    so a conv bias is a redundant parameter with no gradient direction of its
    own."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1,
                               bias=False)
        self.norm1 = nn.GroupNorm(_GN_GROUPS, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_GN_GROUPS, cout)
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.GroupNorm(_GN_GROUPS, cout))
        else:
            self.short = None

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.norm2(self.conv2(y))
        return F.relu(y + (x if self.short is None else self.short(x)),
                      inplace=True)


def _resnet_trunk(in_ch: int, emb: int) -> nn.Sequential:
    """2.79M-parameter residual trunk with the plain trunk's output contract.

    Shape: stem conv3x3(in_ch->32, s1) + GN + ReLU, then three two-block
    stages 32->32, 32->64, 64->128, each stage downsampling /2 in its first
    block. On a 32x64 input that is 32x64 -> 16x32 -> 8x16 -> 4x8, so the
    AdaptiveAvgPool2d((4,8)) is an identity that only pins the contract if
    the image size ever changes.

    GroupNorm rather than BatchNorm, for three reasons that are all about
    THIS trainer:

    * the rollout runs inside a captured CUDA graph. BatchNorm in train mode
      mutates running_mean/running_var/num_batches_tracked, and the graph
      REPLAYS those mutations, so the statistics a replay writes are a
      function of the buffers frozen at capture time rather than of the
      steps actually taken;
    * PPO's update sees minibatches of a different size and composition than
      the rollout batch, and the eval/record path sees a handful of episodes;
      BatchNorm makes the policy a different function in each of those, which
      is the train/eval mismatch this project has already been burned by
      twice (--obs-reward, --yaw-adaptive);
    * GroupNorm is batch-independent, has no buffers, and is identical in
      train() and eval(), so a checkpoint means exactly one function.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 3, stride=1, padding=1, bias=False),
        nn.GroupNorm(_GN_GROUPS, 32), nn.ReLU(),
        _BasicBlock(32, 32, stride=2), _BasicBlock(32, 32),
        _BasicBlock(32, 64, stride=2), _BasicBlock(64, 64),
        _BasicBlock(64, 128, stride=2), _BasicBlock(128, 128),
        nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
        nn.Linear(128 * 4 * 8, emb), nn.ReLU(),
    )


class Policy(nn.Module):
    """Scalars + lidar depth image: conv trunk (shared by pi/vf) embeds the
    depth image to `emb` features, concatenated with the selected scalars
    into [hidden, hidden] tanh towers. gps=False (default) hides absolute
    heading + position from the network — honest-perception mode.

    in_ch=2 (--surf-mask) feeds the depth image and the per-pixel
    surfability mask as two channels of one image; everything downstream of
    conv[0] is unchanged, so at in_ch=1 the state_dict is bit-identical to
    the pre-mask model's."""

    def __init__(self, obs_dim: int, lidar_w: int = LIDAR_W, lidar_h: int = LIDAR_H,
                 emb: int = 512, hidden: int = 448, gps: bool = False,
                 in_ch: int = 1, extra_feat: tuple = (), n_codes: int = 0,
                 chunk: int = 0, route_dim: int = 0,
                 route_critic_only: bool = False, trunk: str = "plain",
                 rnn: str = "none", rnn_size: int = 256,
                 tower_depth: int = 2, conv_mult: int = 1,
                 fp32_heads: bool = False,
                 priv_dim: int = 0, priv_hidden: int = 128,
                 yaw_cond: bool = False, view_continuous: bool = False,
                 view_absolute=None):
        super().__init__()
        # --priv-critic (asymmetric actor-critic, Pinto et al. 2017): the
        # CRITIC additionally reads a privileged state block the simulator
        # has and the actor does not (surfgym/privfeat.py). It enters
        # RIGHT BEFORE value_head - value = value_head(cat(vf_tower(f),
        # priv_mlp(priv))) - so the trunk stays shared (VRAM, and a
        # checkpoint that is still recognisably the same architecture) while
        # the two action heads are provably untouched: nothing on the pi
        # path can reach `priv`. priv_dim 0 builds no module, draws no RNG
        # and adds no state_dict key, and `hidden + 0` below is the same
        # Linear the pre-flag model had - byte-identical
        # (tests/python/test_priv_critic.py).
        self.priv_dim = int(priv_dim)
        self.priv_hidden = int(priv_hidden) if self.priv_dim > 0 else 0
        if self.priv_dim > 0 and self.priv_hidden < 1:
            raise SystemExit(f"--priv-hidden {priv_hidden} < 1")
        # --fp32-heads: run the two output Linears (action/code logits and
        # the value) with autocast DISABLED, so what the rollout stores and
        # what the update differences is the fp32 result rather than a
        # bf16-rounded one (7 explicit mantissa bits). The towers and the
        # conv trunk stay in bf16 - this is about the QUANTIZATION of the
        # stored logit and value, not about the arithmetic upstream. Not a
        # module and not a buffer, so the state_dict is unchanged and a
        # checkpoint moves between the two settings freely; False is the
        # same expression the pre-flag heads() was.
        self.fp32_heads = bool(fp32_heads)
        # --tower-depth / --conv-mult: the two capacity knobs that are NOT
        # emb/hidden. tower_depth is how many Linear+Tanh layers each of the
        # pi/vf towers has (2 = the historical stack); conv_mult scales the
        # plain trunk's three conv widths 16/32/64 by M, and the Linear after
        # the AdaptiveAvgPool follows it. At (2, 1) every module below is
        # constructed by the same calls in the same order as before the
        # flags, so the RNG draw and the state_dict are bit-identical
        # (tests/python/test_flags_round30.py).
        self.tower_depth = int(tower_depth)
        self.conv_mult = int(conv_mult)
        if self.tower_depth < 1:
            raise SystemExit(f"--tower-depth {self.tower_depth} < 1: a tower "
                             "with no layers is not a tower")
        if self.conv_mult < 1:
            raise SystemExit(f"--conv-mult {self.conv_mult} < 1")
        # --rnn gru: ONE GRU layer between the fused trunk output (core
        # scalars + conv embedding, `f` in forward_split) and the towers.
        # The towers read [f | route | h_t]: the memoryless features stay and
        # the recurrent state is APPENDED as trailing columns, so (a) a
        # feed-forward checkpoint warm-starts by zero-padding exactly those
        # columns (widen_for_rnn - function-identical at step 0, the trick
        # widen_for_route uses) and (b) the reactive path is never squeezed
        # through the GRU. Actor and critic share h. "none" builds no module,
        # draws no RNG and adds no state_dict key: byte-identical to the
        # pre-flag policy (tests/python/test_rnn_policy.py).
        self.rnn = str(rnn or "none")
        if self.rnn not in ("none", "gru"):
            raise SystemExit(f"unknown --rnn {self.rnn!r}")
        self.rnn_size = int(rnn_size) if self.rnn == "gru" else 0
        # --route widens the SCALAR half of the row: [15 core | R route | img].
        # The route block sits between them rather than at the end so the
        # image stays one contiguous trailing slice (Policy.forward_split
        # restrides it into the channels_last trunk with no copy).
        self.route_dim = int(route_dim)
        self.route_critic_only = bool(route_critic_only) and self.route_dim > 0
        self.scal_dim = N_SCALAR + self.route_dim
        assert obs_dim == self.scal_dim + lidar_w * lidar_h * in_ch, \
            (f"obs_dim {obs_dim} != {N_SCALAR}+{self.route_dim}+"
             f"{lidar_w}x{lidar_h}x{in_ch}")
        self.lidar_w, self.lidar_h, self.in_ch = lidar_w, lidar_h, in_ch
        # extra_feat re-enables scalar slots the no-GPS mask normally hides,
        # used to carry side-channel signals (see --obs-reward) without
        # widening obs_dim and disturbing the image slice
        idx = tuple(range(N_SCALAR)) if gps else SCALAR_NOGPS
        idx = tuple(sorted(set(idx) | set(extra_feat)))
        self.register_buffer("feat_idx", torch.tensor(idx, dtype=torch.long),
                             persistent=False)
        # --trunk selects the image encoder ONLY; both trunks emit `emb`
        # features, so the towers, the heads and forward_split's restride are
        # the same code in both. "plain" is the historical stack, constructed
        # by the same calls in the same order as before the flag existed, so
        # every checkpoint ever trained still loads key-for-key.
        self.trunk = str(trunk or "plain")
        if self.trunk == "resnet":
            # --conv-mult scales the PLAIN stack's three widths; the resnet
            # trunk's shape is fixed by its stage table and has no such knob
            if self.conv_mult != 1:
                raise SystemExit("--conv-mult scales the plain trunk's three "
                                 "conv widths; --trunk resnet has a fixed "
                                 "stage table - pick one")
            self.conv = _resnet_trunk(in_ch, emb)
        elif self.trunk == "plain":
            _m = self.conv_mult
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 16 * _m, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16 * _m, 32 * _m, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32 * _m, 64 * _m, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
                nn.Linear(64 * _m * 4 * 8, emb), nn.ReLU(),
            )
        else:
            raise SystemExit(f"unknown --trunk {self.trunk!r}")
        # The route block is concatenated LAST, after the conv embedding, so
        # growing it onto an existing checkpoint is a zero-pad of the first
        # Linear's TRAILING columns (widen_for_route). A resumed arm then
        # computes exactly the function the baseline computed, and starts on
        # the baseline curve instead of near it.
        feat = len(idx) + emb
        self.feat_dim = feat
        def mlp(extra=0):
            # + rnn_size: the GRU block is the LAST input block of both
            # towers (0 wide without --rnn, so the Linear is the old one).
            # --tower-depth: the FIRST Linear is the only one whose input
            # width depends on the features, so widening the observation
            # (--route, --rnn) still zero-pads exactly `<tower>.0.weight`
            # and widen_for_route keeps working at any depth.
            layers = [nn.Linear(feat + extra + self.rnn_size, hidden),
                      nn.Tanh()]
            for _ in range(self.tower_depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            return nn.Sequential(*layers)
        self.pi = mlp(0 if self.route_critic_only else self.route_dim)
        self.vf = mlp(self.route_dim)
        self.action_head = nn.Linear(hidden, sum(NVEC))
        # + priv_hidden: the privileged block is the LAST input block of the
        # value head (0 wide without --priv-critic, so this is the old
        # Linear). Trailing columns, like every other widening in this file,
        # so widen_for_priv can zero-pad a pre-flag checkpoint onto it and
        # the resumed critic computes exactly its old function at step 0.
        self.value_head = nn.Linear(hidden + self.priv_hidden, 1)
        # .modules() rather than iterating the Sequential: the resnet trunk
        # nests its convs inside blocks. For "plain" the Sequential has no
        # nested modules, so this yields exactly the same Conv2d/Linear list
        # in exactly the same order and consumes the RNG identically - the
        # bit-identity the flag promises (tests/python/test_trunk.py).
        for m in list(self.conv.modules()) + list(self.pi) + list(self.vf):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:     # resnet convs are bias-free: the
                    nn.init.zeros_(m.bias)  # norm after them carries the shift
        nn.init.orthogonal_(self.action_head.weight, 0.01)
        nn.init.zeros_(self.action_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)
        # ---- --chunk: code head + learnable decoder ------------------------
        # `action_head` is KEPT (initialised, unused, receiving no gradient)
        # so the two modes stay structurally comparable and a chunked ckpt is
        # still recognisably the same architecture. n_codes/chunk both 0 =
        # flat mode, and then NOTHING below runs: the state_dict, the forward
        # and the sampling are bit-for-bit the pre-chunk model's.
        self.n_codes, self.chunk = int(n_codes), int(chunk)
        if self.n_codes > 0 and self.chunk > 0:
            self.code_head = nn.Linear(hidden, self.n_codes)
            nn.init.orthogonal_(self.code_head.weight, 0.01)
            nn.init.zeros_(self.code_head.bias)
            # The decoder: per-code, per-decision FLAT logits in the same
            # sum(NVEC) layout action_head emits, so HeadPacker.pad_seq turns
            # it into the padded (.., 6, 15) tensor sample_padded already
            # consumes. std 0.5 breaks the inter-code symmetry PPO would
            # otherwise have to break from a dead-flat table (identical codes
            # = zero gradient signal to distinguish them) while leaving every
            # head still near-uniform at init — no code starts deterministic,
            # which is what makes the first iterations explorable.
            self.decoder = nn.Parameter(
                torch.randn(self.n_codes, self.chunk, sum(NVEC)) * 0.5)
        else:
            self.code_head = None
            self.decoder = None
        # ---- --rnn gru ------------------------------------------------------
        # Registered LAST so every pre-existing parameter keeps its index in
        # policy.parameters() - Adam's state is keyed by that index, which is
        # what lets widen_for_rnn append fresh slots to a feed-forward
        # checkpoint's optimizer state instead of rebuilding it. nn.GRU (not
        # GRUCell): the same module runs one step at a time in the rollout
        # (a seq_len-1 call, which captures into the rollout's CUDA graph
        # bit-identically to eager) and over whole episode segments in the
        # update (one rectangular cuDNN call per minibatch, gru_sequence).
        # Orthogonal/zero init like the towers; no dropout, so train() and
        # eval() compute the same function.
        if self.rnn == "gru":
            self.gru = nn.GRU(feat, self.rnn_size)
            for name, p in self.gru.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(p, 1.0)
                else:
                    nn.init.zeros_(p)
        else:
            self.gru = None
        # ---- --priv-critic --------------------------------------------------
        # Registered LAST, after the GRU, for the reason the GRU block gives:
        # every pre-existing parameter keeps its index in policy.parameters(),
        # which is the key Adam's state is stored under - that is what lets
        # widen_for_priv append fresh slots to a checkpoint's optimizer state
        # instead of rebuilding it. Two 128-wide LayerNorm+Tanh layers: small
        # next to the 512-wide conv embedding, and bounded on the output so
        # the block joins the vf tower's tanh output on the same scale.
        if self.priv_dim > 0:
            ph = self.priv_hidden
            self.priv_mlp = nn.Sequential(
                nn.Linear(self.priv_dim, ph), nn.LayerNorm(ph), nn.Tanh(),
                nn.Linear(ph, ph), nn.LayerNorm(ph), nn.Tanh())
            for m in self.priv_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.zeros_(m.bias)
        else:
            self.priv_mlp = None
        # ---- --yaw-cond -----------------------------------------------------
        # Registered LAST, after the privileged block, for the reason that
        # block gives: every pre-existing parameter keeps its index in
        # policy.parameters(), which is the key Adam's state is stored
        # under, and the new table is APPENDED with no moments - what a
        # fresh parameter has (widen_for_yawcond). yaw_cond False builds no
        # module, adds no state_dict key and draws no RNG, so the model is
        # byte-identical to the pre-flag one
        # (tests/python/test_yaw_cond.py).
        self.yaw_cond = bool(yaw_cond)
        if self.yaw_cond:
            if self.n_codes > 0 and self.chunk > 0:
                raise SystemExit(
                    "--yaw-cond is not implemented for --chunk: a chunk's H "
                    "decisions all come out of ONE code drawn at the chunk "
                    "start, so there is no per-decision yaw to condition the "
                    "side key on. Run the arm flat.")
            self.yaw_side = _YawSideCond(NVEC[H_YAW], NVEC[H_SIDE])
        else:
            self.yaw_side = None
        # ---- --view-continuous ----------------------------------------------
        # Registered LAST, after the conditioning table, for the reason every
        # block above gives: every pre-existing parameter keeps its index in
        # policy.parameters(), which is what Adam's state is keyed by, so the
        # two new tensors are APPENDED with no moments (what a fresh
        # parameter has; tools/transplant_view.py). The mean head reads the
        # pi tower like action_head does; the log sigma is state-independent.
        # action_head is KEPT at its full sum(NVEC) width - its yaw/pitch
        # logits are dead outputs under the flag, never scored and never
        # sampled - so a discrete checkpoint's every tensor still loads key
        # for key and the transplant only has to FIT the two new ones. False
        # builds no module, draws no RNG and adds no state_dict key.
        self.view_continuous = bool(view_continuous)
        # --view-absolute: None (the delta command), "velocity" or "world".
        # It changes the number of Gaussian heads (3 in world mode: cos,
        # sin, pitch) and what the env does with z, nothing else here.
        self.view_absolute = str(view_absolute) if view_absolute else None
        if self.view_absolute is not None and not self.view_continuous:
            raise SystemExit("--view-absolute needs --view-continuous (it is "
                             "a reading of the continuous view row)")
        if self.view_absolute not in (None, "velocity", "world"):
            raise SystemExit(f"--view-absolute must be velocity or world, "
                             f"got {self.view_absolute!r}")
        self.n_z = n_z(self.view_absolute) if self.view_continuous else 0
        if self.view_continuous:
            if self.n_codes > 0 and self.chunk > 0:
                raise SystemExit("--view-continuous is not implemented for "
                                 "--chunk (a code decodes into BIN "
                                 "distributions)")
            if self.yaw_cond:
                raise SystemExit("--view-continuous excludes --yaw-cond: the "
                                 "side key would condition on a yaw BIN the "
                                 "continuous head no longer emits")
            self.view_head = nn.Linear(hidden, self.n_z)
            nn.init.orthogonal_(self.view_head.weight, 0.01)
            nn.init.zeros_(self.view_head.bias)
            # a MODULE, not a bare Parameter on Policy: a module's own
            # parameters come BEFORE its submodules' in parameters(), so a
            # bare one would take index 0 and shift every Adam slot of the
            # checkpoint being transplanted; registered last, it is last
            self.view_std = _ViewLogStd(self.n_z, float(LOG_STD_INIT))
        else:
            self.view_head = None
            self.view_std = None
        # cuDNN's tensor-core convolutions are NHWC; fed NCHW they transpose
        # in and back out around every conv, forward AND backward. On the
        # 5090 that was 34% of ALL update GPU time (three nchwToNhwc /
        # nhwcToNchw kernels, 44.8 of 131.2 ms — tools/bench_update.py
        # --profile). Holding the trunk channels_last deletes those kernels;
        # the arithmetic and the weights are unchanged, so checkpoints stay
        # interchangeable in both directions.
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, obs, h=None, priv=None):
        """One fused (B, 15 + R + H*W) fp32 row — the rollout and every eval.
        --rnn: `h` (B, rnn_size) is the state entering this decision and a
        third output, the state leaving it, is returned.
        --priv-critic: `priv` (B, priv_dim) is the CRITIC's extra input; the
        logits are the same function of `obs` with it or without it."""
        return self.forward_split(obs[:, :self.scal_dim],
                                  obs[:, self.scal_dim:], h, priv)

    def forward_split(self, scal, img, h=None, priv=None):
        """Scalars and depth as separate tensors, so the PPO update can keep
        its depth buffer in bf16 (S3) without materialising a fused fp32 row.
        `scal` is indexed with feat_idx, whose entries are all < N_SCALAR, so
        this selects exactly what the fused path selects.

        Without --rnn this is features() then heads(), which is the pre-flag
        forward op for op. With it the GRU step sits between the two and the
        call returns (logits, value, h_next)."""
        f = self.features(scal, img)
        if self.gru is None:
            return self.heads(f, scal, priv=priv)
        if h is None:
            raise ValueError("--rnn policy called without its hidden state")
        h1 = self.gru_step(f, h)
        # --rnn + --priv-critic: the concat happens AFTER the GRU, i.e. on
        # the vf tower's output, so the recurrent state is memory of what
        # the ACTOR could see and the privileged block never enters it.
        logits, value = self.heads(f, scal, h1, priv=priv)
        return logits, value, h1

    def features(self, scal, img):
        """The fused trunk output `f`: selected scalars ++ conv embedding."""
        # The renderer emits the image channel-FASTEST (NHWC), so view+permute
        # is a free RESTRIDE that declares NHWC — at in_ch=1 the two layouts
        # are even the same bytes. The one layout copy conv still makes is the
        # copy the NCHW path already made — and autocast does it in bf16,
        # which is why this is left to conv instead of an explicit
        # .contiguous() in fp32. At in_ch=2 the restride is still free and the
        # channels_last trunk consumes it natively; two SEPARATE planes would
        # have made this a real transpose per forward (perf-results.md S9).
        im = img.reshape(-1, self.lidar_h, self.lidar_w,
                         self.in_ch).permute(0, 3, 1, 2)
        return torch.cat([scal[:, self.feat_idx], self.conv(im)], dim=1)

    def heads(self, f, scal, g=None, priv=None):
        """Towers + heads on the fused features. `g` (--rnn) is the GRU
        output for these rows, appended LAST to both tower inputs; `priv`
        (--priv-critic) reaches the VALUE head only."""
        # --route: the lookahead fan feeds the critic always and the actor
        # unless --route-critic-only, which is the asymmetric-critic
        # arrangement (Vasco RLC 2024: "providing the critic with global
        # features during training is fundamental", and a SYMMETRIC critic
        # measurably degraded their vision agent). route_dim 0 = neither
        # branch exists and this is the pre-route model, byte for byte.
        f_pi = f_vf = f
        if self.route_dim:
            f_vf = torch.cat([f, scal[:, N_SCALAR:]], dim=1)
            f_pi = f if self.route_critic_only else f_vf
        if g is not None:
            f_pi = torch.cat([f_pi, g], dim=1)
            f_vf = torch.cat([f_vf, g], dim=1)
        # --chunk swaps WHICH head the trunk feeds — code logits (B, n_codes)
        # instead of flat action logits — so every existing call site
        # (`logits, value = policy(obs)`) keeps working unchanged. self.code_head
        # is None in flat mode, so this resolves to action_head at trace time.
        head = self.action_head if self.code_head is None else self.code_head
        if self.fp32_heads:
            # the tower output arrives bf16 under autocast; .float() before
            # the disabled-autocast Linear is what makes the matmul and its
            # result fp32. Captured into the rollout CUDA graph and traced by
            # inductor exactly like gru_step's own autocast(enabled=False).
            # --priv-critic follows the same rule: _value casts the priv MLP
            # to the tower's dtype, so inside this block the whole value
            # path (priv MLP included) is fp32.
            with torch.autocast(device_type="cuda", enabled=False):
                return (self._pi_out(head, self.pi(f_pi).float()),
                        self._value(self.vf(f_vf).float(), priv).squeeze(-1))
        return (self._pi_out(head, self.pi(f_pi)),
                self._value(self.vf(f_vf), priv).squeeze(-1))

    def _pi_out(self, head, t):
        """`head` over the pi tower output `t`; under --view-continuous the
        two view MEANS are appended after the categorical logits (split_view
        takes them apart). view_head None is `head(t)` and nothing else -
        the pre-flag expression, op for op."""
        o = head(t)
        if self.view_head is None:
            return o
        return torch.cat([o, self.view_head(t)], dim=-1)

    def log_std(self):
        """The view heads' log sigma, clamped to [LOG_STD_MIN, LOG_STD_MAX]
        (a bound that never binds at init: log 0.3 = -1.2)."""
        return self.view_std.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def _value(self, t, priv):
        """value_head over the vf tower output `t`, plus --priv-critic.

        priv_dim 0 is `self.value_head(t)` and nothing else - the pre-flag
        expression, op for op.

        With the flag and NO privileged row, the value comes back as NaN
        rather than as value_head over a zero block. A caller that only
        wants the ACTOR (tools/record_ckpt.py, which never reads this
        output) is unaffected; a caller that actually uses V gets a number
        that cannot be mistaken for the trained critic's. The critic is a
        TRAINING-TIME object: at deployment there is no privileged state to
        give it and a plausible wrong value is the worse failure.
        """
        if self.priv_dim == 0:
            return self.value_head(t)
        if priv is None:
            return torch.full(t.shape[:-1] + (1,), float("nan"),
                              dtype=t.dtype, device=t.device)
        return self.value_head(
            torch.cat([t, self.priv_mlp(priv).to(t.dtype)], dim=1))

    def gru_step(self, f, h):
        """One decision: (B, feat), (B, R) -> (B, R). fp32 and TF32-free
        regardless of the caller's autocast: autocast would hand the cuDNN
        RNN fp16 (measured - it ignores the requested bf16), and the same
        arithmetic here and in gru_sequence is what makes the update's
        re-run reproduce the rollout's states."""
        with torch.autocast(device_type="cuda", enabled=False), _no_tf32():
            return self.gru(f.float().unsqueeze(0), h.float().unsqueeze(0))[1][0]

    def gru_sequence(self, f, h0, seg):
        """The update's re-run of the rollout: (T*B, feat) time-major rows,
        (B, R) states entering t=0, a SeqPlan from gru_segments -> (T*B, R)
        GRU outputs in the same row order. Each env's T rows are cut at
        its episode ends into segments; the first segment starts from h0,
        every later one from ZERO (a respawn is a new episode), and cuDNN
        runs all S segments of the minibatch as ONE rectangular (L, S)
        batch. A segment shorter than L is padded with rows that are
        computed and never read: the state only flows forward in time
        within a column, so the valid positions are exact, and in the
        backward the padding carries zero upstream gradient, so the
        weight and input gradients are exact too. Measured against
        pack_padded_sequence on the 5090 (T=128, B=64): 3.8 vs 4.9 ms
        fwd+bwd uncut, 6.0 vs 8.4 ms at one cut per three envs, 24 vs
        30 ms at 297 segments - the same 5e-5 of the stepwise rollout."""
        with torch.autocast(device_type="cuda", enabled=False), _no_tf32():
            f = f.float()
            xpad = f[seg.idx]                       # (L, S, feat); pads unread
            hseg = torch.where(seg.first.unsqueeze(1), h0.float()[seg.env],
                               torch.zeros_like(h0[:1].float()))
            out, _ = self.gru(xpad, hseg.unsqueeze(0))      # (L, S, R)
            return out.reshape(-1, out.shape[-1])[seg.inv]


class _ViewLogStd(nn.Module):
    """--view-continuous: the state-independent log sigma of the two view
    heads, as a module so its parameter sits LAST in Policy.parameters()
    (see Policy.__init__). state_dict key: view_std.log_std."""

    def __init__(self, n: int, init: float):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((int(n),), float(init)))


class _no_tf32:
    """cuDNN runs its RNN GEMMs in TF32 by default on Ampere+ (measured
    7e-4 between the fused cell and cuDNN on one step). Off for the GRU
    only - the flag is read at call time (and at CUDA-graph capture time),
    and nothing else in the trainer depends on it under bf16 autocast."""

    def __enter__(self):
        self._was = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cudnn.allow_tf32 = self._was


class SeqPlan:
    """Index tensors that turn a minibatch's (T, B) time-major rows into
    the (L, S) segment-column layout gru_sequence hands cuDNN, and back."""

    __slots__ = ("idx", "inv", "lens", "first", "env", "n_seg")

    def __init__(self, idx, inv, lens, first, env):
        self.idx, self.inv, self.lens = idx, inv, lens
        self.first, self.env, self.n_seg = first, env, int(lens.numel())


def gru_segments(done, device):
    """Cut a minibatch's rollout into episode segments for gru_sequence.

    `done` is (T, B) bool: done[t, e] means env e's episode ended during
    decision t, so the state entering decision t+1 is ZERO. Segments are
    [start, end) runs of one env between such cuts; every (t, e) row lands
    in exactly one. Returns a SeqPlan with idx (L, S) row gathers into the
    time-major (T*B) flat layout (pad entries point at row 0: computed by
    the rectangular GRU call and never read back), inv (T*B) the inverse
    gather out of the (L*S) column layout, lens (S) on the CPU, first (S)
    bool = segment starts at t=0 and takes the stored h_0, env (S) its
    env index within the minibatch."""
    done = np.asarray(done, bool)
    T, B = done.shape
    starts = np.zeros((T, B), bool)
    starts[0] = True
    starts[1:] = done[:-1]
    env, st = np.nonzero(starts.T)          # env-major: (e, t) pairs, sorted
    nxt_env = np.r_[env[1:], -1]
    nxt_st = np.r_[st[1:], T]
    end = np.where(nxt_env == env, nxt_st, T)
    lens = end - st
    S, L = len(lens), int(lens.max())
    ar = np.arange(L)[:, None]
    valid = ar < lens[None, :]                              # (L, S)
    rows = (st[None, :] + ar) * B + env[None, :]            # (L, S)
    idx = np.where(valid, rows, 0)
    inv = np.empty(T * B, np.int64)
    inv[rows[valid]] = np.arange(L * S).reshape(L, S)[valid]
    return SeqPlan(torch.as_tensor(idx, device=device),
                   torch.as_tensor(inv, device=device),
                   torch.as_tensor(lens, dtype=torch.int64),
                   torch.as_tensor(st == 0, device=device),
                   torch.as_tensor(env, device=device))


def widen_for_route(ck, policy, flag="--route"):
    """Zero-pad a pre-route checkpoint onto a route-widened Policy.

    Adding lookahead features grows the FIRST Linear of the pi and vf towers
    by ``route_dim`` trailing columns. Zeroing exactly those columns makes the
    resumed policy compute its old function on its first forward, so the arm
    starts ON the baseline curve and every later divergence is the feature
    rather than a re-initialisation shock. (This is why the route block is
    concatenated last in forward_split — a middle insert would have permuted
    the existing columns and silently scrambled the checkpoint.)

    Adam's exp_avg/exp_avg_sq are padded the same way: the new columns get
    zero history, which is exactly what they have.

    Returns the number of tensors padded; 0 means the checkpoint already
    matched and nothing was touched.
    """
    sd = ck.get("policy") or {}
    n = 0

    def _pad(t, want, what):
        if t.dim() != 2 or t.shape[0] != want[0] or t.shape[1] > want[1]:
            raise SystemExit(
                f"{flag} cannot warm-start this checkpoint: {what} is "
                f"{tuple(t.shape)} and the model wants {tuple(want)}")
        return torch.cat([t, torch.zeros(want[0], want[1] - t.shape[1],
                                         dtype=t.dtype)], dim=1)

    for name, p in policy.state_dict().items():
        t = sd.get(name)
        if t is None or tuple(t.shape) == tuple(p.shape):
            continue
        sd[name] = _pad(t, p.shape, name)
        n += 1
    # optimizer moments are keyed by parameter INDEX in policy.parameters()
    # order — the same order Adam was constructed with
    ost = ((ck.get("optimizer") or {}).get("state")) or {}
    params = list(policy.parameters())
    for i, st in ost.items():
        try:
            want = params[int(i)].shape
        except (ValueError, IndexError, TypeError):
            continue
        for k in ("exp_avg", "exp_avg_sq"):
            t = st.get(k)
            if t is None or tuple(t.shape) == tuple(want):
                continue
            st[k] = _pad(t, want, f"optimizer {k}[{i}]")
            n += 1
    return n


#: Config keys that fix a checkpoint's TENSOR SHAPES, with the flag that
#: sets each. Every one is restored from the checkpoint when its flag is
#: absent (the arg defaults to None), so a mismatch only ever happens when
#: someone PASSED a different value - and then the network built here is a
#: different network and no warm start exists.
ARCH_KEYS = (("emb", "--emb"), ("hidden", "--hidden"), ("trunk", "--trunk"),
             ("tower_depth", "--tower-depth"), ("conv_mult", "--conv-mult"),
             ("lidar_w", "--lidar-w"), ("lidar_h", "--lidar-h"),
             ("normals", "--normals"), ("surf_mask", "--surf-mask"))


def check_arch_matches(ck_cfg, args, policy) -> None:
    """Refuse a resume whose ARCHITECTURE flags disagree with the checkpoint.

    This exists because the failure is otherwise reported by whichever
    observation-block widener runs first, in ITS flag's language. A real
    case, and it cost a debugging session: resuming a 512-emb checkpoint
    with ``--emb 64 --hidden 64`` (tools/expert_loop.py's --dry-run
    overrides) printed

        --route cannot warm-start this checkpoint: pi.0.weight is
        (448, 524), i.e. 449 route-side columns over a 75-wide trunk,
        and this run wants 1

    - a route-block message for a run with no route file, about a
    checkpoint with none either. 524 is 11 scalars + 512 conv + 1 latch and
    75 is 11 + 64: the whole discrepancy is the conv embedding, and
    ``ck_obs_block`` charged it to the block it happened to be measuring.
    A widener can only ever see ``ck_tensor_width - policy.feat_dim``; it
    cannot know which of the two terms is wrong. This can, because it reads
    the checkpoint's own config, so it says which FLAG to drop.

    Silent is not an option either: with --emb alone the towers would
    happen to line up at some sizes and the arm would resume a scrambled
    trunk. Keys the checkpoint does not carry are skipped (an old file
    predates them), and the tensor-level backstop below catches what the
    config cannot.
    """
    bad = []
    for key, flag in ARCH_KEYS:
        want = ck_cfg.get(key)
        got = getattr(args, key, None)
        if want is None or got is None:
            continue
        if isinstance(want, str) or isinstance(got, str):
            same = str(got) == str(want)
        else:
            same = int(got) == int(want)
        if not same:
            bad.append(f"{flag} {got!r} (the checkpoint trained at {want!r})")
    if bad:
        raise SystemExit(
            "this checkpoint cannot be warm-started with a DIFFERENT "
            "architecture: " + "; ".join(bad) + ". Every one of these is "
            "restored from the checkpoint when its flag is absent, so drop "
            "the flag - there is no meaning-preserving way to reshape a "
            "trained trunk. (For a small SMOKE run shrink --envs / "
            "--n-steps / --minibatches / --ep-ticks instead: none of those "
            "changes a tensor.)")


def ck_trunk_mismatch(ck, policy):
    """``(name, tower_width, remainder)`` when a tower's first Linear is
    NARROWER than this run's trunk alone, else None.

    The tower reads ``feat_dim + observation block + rnn`` and neither of
    the last two can be negative, so a negative remainder is proof that the
    two trunks differ - the one thing every widener's message assumes away.
    """
    sd = ck.get("policy") or {}
    hh = sd.get("gru.weight_hh_l0")
    rnn = int(hh.shape[1]) if hh is not None and hh.dim() == 2 else 0
    for name in ("pi.0.weight", "vf.0.weight"):
        t = sd.get(name)
        if t is None or t.dim() != 2:
            continue
        rem = int(t.shape[1]) - int(policy.feat_dim) - rnn
        if rem < 0:
            return name, int(t.shape[1]), rem
    return None


def ck_obs_block(ck, policy, name="vf.0.weight"):
    """How many route-side scalar columns the CHECKPOINT's tower reads.

    Read off the TENSOR, not off the config: the tower's first Linear is
    ``feat + route_dim + rnn_size`` wide (``Policy.mlp``), ``feat`` is fixed
    by --emb, --conv-mult and the scalar mask, and the checkpoint says for
    itself whether it is recurrent - the GRU block is appended LAST, after
    this one, so its width comes out of ``gru.weight_hh_l0`` (3R, R).
    Deriving it from the config instead would mean re-deriving the route
    fan's feature count, the latch column and both aux blocks from six flags
    and getting all six right. Returns None when the checkpoint has no such
    tensor.
    """
    sd = ck.get("policy") or {}
    t = sd.get(name)
    if t is None or t.dim() != 2:
        return None
    hh = sd.get("gru.weight_hh_l0")
    rnn = int(hh.shape[1]) if hh is not None and hh.dim() == 2 else 0
    return int(t.shape[1]) - int(policy.feat_dim) - rnn


def widen_for_obs(ck, policy, route_dim, flag="--act-hist"):
    """Grow the route-side scalar block of a checkpoint IN PLACE.

    That block - ``[fan | latch | 6*K act-hist | 5 compass]`` - enters both
    towers as ``scal[:, N_SCALAR:]``, concatenated BETWEEN the fused trunk
    output ``f`` and the GRU state (``Policy.heads``). Growing it is
    therefore an insert at column ``feat_dim + old_width``, which coincides
    with the tensor's tail only when there is no GRU. ``widen_for_route``'s
    unconditional TRAILING pad is right in that case and wrong under --rnn,
    where it pushes the new columns past the recurrent block and hands the
    checkpoint's own weights permuted inputs (measured: 1.4e-2 on the
    logits, i.e. a different policy dressed as a warm start).

    With the new columns at zero the resumed policy computes exactly the
    checkpoint's function on its first forward - the new inputs multiply
    zero weights - so an arm that ADDS ``--act-hist``/``--obs-compass`` to a
    finisher starts ON its own control curve and everything after that is
    the feature learning to use the history. "Exactly" is up to the wider
    GEMM's summation order, the same ~1 ulp of fp32 ``widen_for_priv``
    documents (measured 1.9e-9 on the logits, 7.5e-8 on the value).

    Adam's exp_avg/exp_avg_sq get the same insert: the new columns carry
    zero history, which is what they have.

    ``route_dim`` is the width the MODEL was built for (N_ROUTE). Only the
    two tower Linears are touched; anything else that changed shape is left
    to widen_for_route/-rnn/-priv, which run after this. Returns the number
    of tensors touched; 0 means the block already matched.
    """
    sd = ck.get("policy") or {}
    want = policy.state_dict()
    feat = int(policy.feat_dim)
    hh = sd.get("gru.weight_hh_l0")
    ck_rnn = int(hh.shape[1]) if hh is not None and hh.dim() == 2 else 0
    idx_of = {n: i for i, (n, _) in enumerate(policy.named_parameters())}
    ost = ((ck.get("optimizer") or {}).get("state")) or {}
    n = 0

    def _ins(t, at, k):
        return torch.cat([t[:, :at], torch.zeros(t.shape[0], k, dtype=t.dtype),
                          t[:, at:]], dim=1)

    # --route-critic-only routes the whole block to the value tower alone,
    # so pi.0 never carries it (and the flag refuses --act-hist for exactly
    # that reason - a history the ACTOR cannot read is not the feature)
    for name in ("pi.0.weight", "vf.0.weight"):
        if name == "pi.0.weight" and policy.route_critic_only:
            continue
        t, p = sd.get(name), want.get(name)
        if t is None or p is None or t.dim() != 2 or p.dim() != 2:
            continue
        old = ck_obs_block(ck, policy, name)
        if old is None:
            continue
        k = int(route_dim) - int(old)
        if k == 0:
            continue
        if k < 0 or old < 0 or t.shape[0] != p.shape[0]:
            raise SystemExit(
                f"{flag} cannot warm-start this checkpoint: {name} is "
                f"{tuple(t.shape)}, i.e. {old} route-side columns over a "
                f"{feat}-wide trunk, and this run wants {route_dim}. The "
                "block only ever GROWS - a narrower run would have to drop "
                "trained columns, and there is no meaning-preserving way to "
                "choose which")
        at = feat + int(old)
        sd[name] = _ins(t, at, k)
        n += 1
        i = idx_of.get(name)
        st = ost.get(i) if i in ost else ost.get(str(i))
        for key in ("exp_avg", "exp_avg_sq"):
            m = (st or {}).get(key)
            if m is None or m.dim() != 2 or int(m.shape[1]) != int(t.shape[1]):
                continue
            st[key] = _ins(m, at, k)
            n += 1
    return n


def widen_for_rnn(ck, policy):
    """Warm-start a feed-forward checkpoint onto a --rnn Policy.

    The GRU block is the LAST input block of both towers, so the towers'
    first Linear grows by rnn_size TRAILING columns and widen_for_route's
    zero-pad applies unchanged: with those columns at zero the resumed
    policy computes exactly its old function on its first forward, however
    the (fresh, untrained) GRU state happens to look. The GRU's own tensors
    are taken from the freshly initialised model, and because the module
    is registered last its parameters are appended to Adam's group with no
    moments - exactly what fresh parameters have. Returns the number of
    tensors touched; 0 means the checkpoint is already recurrent.
    """
    sd = ck.get("policy") or {}
    if policy.gru is None or any(k.startswith("gru.") for k in sd):
        return 0
    n = widen_for_route(ck, policy, flag="--rnn")
    fresh = policy.state_dict()
    for k in fresh:
        if k.startswith("gru.") and k not in sd:
            sd[k] = fresh[k].detach().cpu().clone()
            n += 1
    n_params = len(list(policy.parameters()))
    for g in (ck.get("optimizer") or {}).get("param_groups", []):
        have = [int(i) for i in g.get("params", [])]
        g["params"] = have + [i for i in range(n_params) if i not in set(have)]
    return n


def widen_for_priv(ck, policy):
    """Warm-start a checkpoint with no privileged critic onto a
    --priv-critic Policy.

    The privileged block is the LAST input block of ``value_head``, so that
    Linear grows by ``priv_hidden`` TRAILING columns and widen_for_route's
    zero-pad applies unchanged: with those columns at zero the resumed
    critic emits exactly its old V(s) on its first forward, whatever the
    freshly initialised priv MLP happens to compute. The MLP's own tensors
    are taken from the new model, and because the module is registered last
    its parameters are appended to Adam's group with no moments - which is
    what fresh parameters have.

    The ACTOR is untouched at the BIT level, by construction: no tensor on
    the pi path changes shape at all, so the resumed policy's logits are
    bit-identical to the checkpoint's and the arm starts ON the control
    curve. For V, exactly as for --route and --rnn, "its old function" is up
    to the summation order of a wider GEMM - the zero columns contribute 0
    but move where the accumulation happens, worth ~1 ulp of fp32 (measured
    6e-8 absolute).

    Returns the number of tensors touched; 0 means the checkpoint already
    carries a privileged critic.
    """
    sd = ck.get("policy") or {}
    if policy.priv_mlp is None or any(k.startswith("priv_mlp.") for k in sd):
        return 0
    n = widen_for_route(ck, policy, flag="--priv-critic")
    fresh = policy.state_dict()
    for k in fresh:
        if k.startswith("priv_mlp.") and k not in sd:
            sd[k] = fresh[k].detach().cpu().clone()
            n += 1
    n_params = len(list(policy.parameters()))
    for g in (ck.get("optimizer") or {}).get("param_groups", []):
        have = [int(i) for i in g.get("params", [])]
        g["params"] = have + [i for i in range(n_params) if i not in set(have)]
    return n


def widen_for_yawcond(ck, policy):
    """Warm-start a checkpoint with no yaw->side conditioning onto a
    --yaw-cond Policy.

    Nothing changes SHAPE here - unlike --route / --rnn / --priv-critic, no
    existing Linear grows. The conditioning is a NEW (n_yaw, n_side) table
    that enters as an additive term on the side head's logits and is
    initialised to ZERO, so the resumed policy's logits are bit-identical
    to the checkpoint's on its first forward: the arm starts ON the control
    curve and the conditioning grows from zero. The module is registered
    LAST, so its parameter is appended to Adam's group with no moments,
    which is what a fresh parameter has.

    Returns the number of tensors added; 0 means the checkpoint already
    carries the table.
    """
    sd = ck.get("policy") or {}
    if policy.yaw_side is None or any(k.startswith("yaw_side.") for k in sd):
        return 0
    n = 0
    fresh = policy.state_dict()
    for k in fresh:
        if k.startswith("yaw_side.") and k not in sd:
            sd[k] = fresh[k].detach().cpu().clone()
            n += 1
    n_params = len(list(policy.parameters()))
    for g in (ck.get("optimizer") or {}).get("param_groups", []):
        have = [int(i) for i in g.get("params", [])]
        g["params"] = have + [i for i in range(n_params) if i not in set(have)]
    return n


# ---- strided frame stacking (--frame-stack) ---------------------------------
# Depth is a still photograph: it says where the geometry is, never how fast
# it is coming. The scalars carry the agent's OWN velocity, so what a single
# frame cannot express is relative motion — closing speed on a ramp, whether
# a moving brush is heading toward you. Stacking past DECISIONS (not ticks:
# at act_every=3 a decision is 30 ms) restores that, and strides make the
# window long without making it dense — [1, 2, 4, 8] reaches 240 ms back
# with 5 frames where a dense stack would need 9.
STACK_STRIDES = (1, 2, 4, 8)


def frame_offsets(k: int):
    """Decision offsets of a K-frame stack, NEWEST FIRST. k <= 1 = off."""
    if k <= 1:
        return (0,)
    if k - 1 > len(STACK_STRIDES):
        raise ValueError(f"--frame-stack {k}: only {len(STACK_STRIDES) + 1} "
                         f"frames are defined (strides {STACK_STRIDES})")
    return (0,) + STACK_STRIDES[:k - 1]


def interleave_frames(frames):
    """K (B, P) frames, newest first -> one (B, P*K) row, CHANNEL-FASTEST.

    The layout is the whole point and it is shared by both producers (the
    rollout's ring and the update's gather) so they cannot drift: pixel-major
    with the frame index innermost is what makes Policy.forward_split's
    reshape+permute a free restride into the channels_last trunk. Frames as
    separate PLANES would turn that into a real transpose per forward — the
    same argument vision._march_kernel_nz makes for interleaving its channels
    (docs/perf-results.md S9)."""
    return torch.stack(frames, dim=-1).reshape(frames[0].shape[0], -1)


def stack_from_ring(ring, head, age, k):
    """Rollout side: (R, N, P) ring of past renders -> (N, P*K).

    ``head`` is the slot holding the newest frame and ``age`` the number of
    PREVIOUS decisions this env has taken in its current episode. Clamping
    the reach-back to ``age`` is what collapses history at a spawn: with
    age 0 every offset resolves to the spawn frame, which is exactly what a
    policy can know on its first decision of an episode."""
    R, N = ring.shape[0], ring.shape[1]
    cols = torch.arange(N, device=ring.device)
    return interleave_frames([ring[(head - torch.clamp(age, max=s)) % R, cols]
                              for s in frame_offsets(k)])


class FrameRing:
    """The rollout's per-env history of past decision renders.

    Deliberately a plain object with no torch.nn or graph entanglement: it
    lives OUTSIDE the CUDA-graphed region, which captures step_compute() over
    static_obs alone. All this has to do is write the composed stack into
    static_obs' image slice before the replay, and the graph never learns the
    feature exists.

    ``age`` counts PREVIOUS decisions of the current episode, capped at the
    ring depth; push(ended=...) is where an episode boundary collapses it.
    """

    def __init__(self, k: int, n_env: int, frame: int, device, dtype=None):
        self.k = int(k)
        self.pro = max(frame_offsets(self.k))       # deepest reach-back
        self.buf = torch.zeros((self.pro + 1, n_env, frame), device=device,
                               dtype=dtype or torch.float32)
        self.age = torch.zeros(n_env, dtype=torch.long, device=device)
        self.head = 0

    def push(self, frame, ended=None) -> None:
        """Newest render in. ``ended=None`` is the run's first decision (no
        history at all); ``ended[i]`` marks env i as having just respawned."""
        self.head = (self.head + 1) % self.buf.shape[0]
        self.buf[self.head].copy_(frame)
        if ended is None:
            self.age.zero_()
        else:
            self.age.add_(1).clamp_(max=self.pro).masked_fill_(ended, 0)

    def compose(self):
        """(N, FRAME*K) — what the policy sees this decision."""
        return stack_from_ring(self.buf, self.head, self.age, self.k)

    def tail(self, back: int):
        """(N, FRAME) whole-batch frame ``back`` decisions behind the newest."""
        return self.buf[(self.head - back) % self.buf.shape[0]]

    def rows_back(self, back, rows):
        """Per-env reach-back: frame ``back[i]`` decisions old, for env
        ``rows[i]``. Used for the truncation bootstrap, whose terminal state
        sits one decision AHEAD of the ring's newest frame."""
        return self.buf[(self.head - back) % self.buf.shape[0], rows]

    # -- the update's view of the same history --------------------------------
    # Both of these live here, not at the call site, because they encode the
    # buffer LAYOUT that stack_from_buffer decodes. Split across two files
    # they drift; together they are one testable object.

    def fill_prologue(self, b_img) -> None:
        """Seed a rollout buffer's leading ``pro`` rows with this ring's
        history, oldest first — the previous iteration's tail, so a t=0
        sample reaches back into real frames instead of clamping."""
        for p in range(self.pro):
            b_img[p].copy_(self.tail(self.pro - p))

    def record(self, b_img, b_age, t: int) -> None:
        """Store decision ``t``: the single newest frame (never the stack)
        and the age, exactly where stack_from_buffer will look for them."""
        b_img[self.pro + t].copy_(self.tail(0))
        b_age[t].copy_(self.age)


def stack_from_buffer(f_img, idx, age, k, n_env, pro):
    """Update side: the same stack out of the flat rollout buffer.

    ``f_img`` is ((pro + T) * N, P) — single-frame per timestep, because a
    stack is a different GATHER, not a bigger buffer (tools/bench_capacity.py
    measures exactly this). ``idx`` indexes the SAMPLE space (T*N); the
    leading ``pro`` rows are the previous iteration's tail, so a t=0 sample
    reaches back into real history instead of running off the buffer.
    ``age`` is the per-sample decision age recorded during the rollout, so
    this clamps identically to stack_from_ring."""
    base = idx + pro * n_env
    return interleave_frames([f_img[base - torch.clamp(age, max=s) * n_env]
                              for s in frame_offsets(k)])


def check_vision_exclusive(surf_mask, pinhole, frame_stack, normals=0) -> None:
    """One vision experiment at a time.

    --surf-mask widens the image to 2 channels, --normals to 4, --frame-stack
    to K, and --pinhole changes what every pixel means. Each is a screen of
    its own; combining them before either has won confounds the read and
    needs kernels/gathers nobody has written. Refuse loudly rather than
    train a week on an arm whose result cannot be attributed."""
    on = [n for n, v in (("--surf-mask", surf_mask), ("--pinhole", pinhole),
                         ("--frame-stack", (frame_stack or 0) > 1),
                         ("--normals", normals)) if v]
    if len(on) > 1:
        raise SystemExit(" and ".join(on) + " are separate experiments; run "
                         "them on separate screens (no combined path exists)")


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
              "gae", "gae_gpu", "tail", "update", "update_gpu", "mb_gpu",
              "allreduce", "skew", "share",
              "ckpt", "record", "misc", "total")
    # phases that are disjoint slices of the iteration wall (for `misc`).
    # "tail" is --tail-weight's own bracket and is absent (hence 0.0 through
    # .get) on every run without the flag.
    _WALL = ("pool", "rollout_wall", "gae", "tail", "skew", "share",
             "update", "ckpt", "record")

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

    def pad_seq(self, logits):
        """(..., sum(NVEC)) -> (..., 6, 15). pad() over arbitrary leading dims.

        --chunk scores H decisions at once, so the padded tensor grows a
        decision axis. sample_padded / logprob_entropy_padded already reduce
        over the LAST two axes only, so both work on the result unchanged and
        return one value per (row, decision)."""
        lead = logits.shape[:-1]
        return self.pad(logits.reshape(-1, logits.shape[-1])).view(
            *lead, NACT, NPAD)


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


# --------------------------------------------------------------------------
# --view-continuous: the mixed distribution (surfgym/view.py has the numpy
# half and the rationale; tests/python/test_view_continuous.py pins the
# two halves against each other and against a hand computation)
# --------------------------------------------------------------------------
def split_view(logits):
    """The --view-continuous policy's flat output is [sum(NVEC) categorical
    logits | n_z view means] -> (cat_logits, mu); n_z is 2, or 3 under
    --view-absolute world (every column past the logits is a mean). A
    discrete policy's output is sum(NVEC) wide and never reaches this."""
    n = sum(NVEC)
    return logits[..., :n], logits[..., n:]


def warp_t(u):
    """surfgym.view.warp in torch: u in [-1, 1] -> K in [-20, 20], odd,
    monotone, warp(+-0.5) = +-1 (the analytic optimal-strafe multiple),
    warp(+-1) = +-20 (the old outermost bin)."""
    a = WARP_ALPHA
    return (K_MAX * torch.sign(u) * torch.expm1(a * u.abs())
            / float(np.expm1(a)))


def off_warp_t(u):
    """surfgym.view.off_warp in torch: u in [-1, 1] -> a yaw OFFSET in
    degrees, [-180, 180], odd, monotone, off_warp(+-0.5) = +-10."""
    b = OFF_ALPHA
    return (OFF_MAX * torch.sign(u) * torch.expm1(b * u.abs())
            / float(np.expm1(b)))


def view_from_z_t(z, pitch_max: float, absolute=None):
    """(B, n_z) pre-tanh z -> (B, N_VIEW) float32 view row for the core.
    absolute None (the delta command): K = warp(tanh(z_yaw)), pitch =
    tanh(z_pitch) * pitch_max deg/tick (pitch_max = the core's
    pitch_rate_max_deg; 0 under a frozen gaze). "velocity": (off_warp(tanh
    z_yaw) deg, -20 + 50 tanh z_pitch deg); "world": (atan2(tanh z_s, tanh
    z_c) deg, the same pitch) - surfgym.view.view_from_z_abs, which the
    tests pin this against."""
    u = torch.tanh(z.float())
    if absolute is None:
        return torch.stack([warp_t(u[:, 0]), u[:, 1] * float(pitch_max)], dim=1)
    if absolute == "velocity":
        yaw = off_warp_t(u[:, 0])
        pitch = PITCH_ABS_MID + PITCH_ABS_HALF * u[:, 1]
    elif absolute == "world":
        yaw = torch.atan2(u[:, 1], u[:, 0]) * (180.0 / math.pi)
        pitch = PITCH_ABS_MID + PITCH_ABS_HALF * u[:, 2]
    else:
        raise ValueError(f"absolute view mode {absolute!r}")
    return torch.stack([yaw, pitch], dim=1)


def gauss_logp(z, mu, log_std):
    """Sum over the last axis of log N(z; mu, exp(log_std))."""
    return (-0.5 * ((z - mu) / log_std.exp()).pow(2) - log_std
            - 0.5 * LOG2PI).sum(-1)


def gauss_entropy(log_std):
    """Differential entropy of the factored Gaussian, summed over heads."""
    return (0.5 + 0.5 * LOG2PI + log_std).sum()


def sample_view(padded, mu, log_std):
    """The --view-continuous rollout draw (no grad): heads N_VIEW..5 by
    Gumbel-argmax on the padded slice (sample_padded, unchanged), the view
    by z = mu + sigma * eps. Static shapes and one randn_like next to the
    one rand_like, so it captures into the rollout CUDA graph like the
    discrete draw. -> (act (B, NACT) long with NEUTRAL_ACT in the two view
    columns, z (B, N_VIEW), the joint log-prob (B,))."""
    cat, logp_c = sample_padded(padded[:, N_VIEW:])
    z = mu + log_std.exp() * torch.randn_like(mu)
    B = padded.shape[0]
    neutral = torch.full((B, N_VIEW), 0, dtype=cat.dtype, device=cat.device)
    neutral[:, 0] = NEUTRAL_ACT[0]
    neutral[:, 1] = NEUTRAL_ACT[1]
    return torch.cat([neutral, cat], dim=1), z, logp_c + gauss_logp(z, mu, log_std)


def logprob_entropy_view(padded, actions, mu, log_std, z):
    """The update's recomputation under --view-continuous: the categorical
    log-prob / entropy over heads N_VIEW..5 (logprob_entropy_padded on the
    slice) plus the Gaussian log-density of the STORED z and the Gaussian
    entropy. PPO's ratio is exp(logp_new - logp_old) of this joint."""
    logp_c, ent_c = logprob_entropy_padded(padded[:, N_VIEW:],
                                           actions[:, N_VIEW:])
    return (logp_c + gauss_logp(z, mu, log_std),
            ent_c + gauss_entropy(log_std))


# --------------------------------------------------------------------------
# opt-in action masks: the "air keys" (--mask-forward-air, --jump-cooldown,
# --duck-air-mask). All three default OFF and the off path touches nothing.
# --------------------------------------------------------------------------
# Head indices into NVEC = (yaw, pitch, fwd, side, jump, duck).
H_YAW, H_PITCH, H_FWD, H_SIDE, H_JUMP, H_DUCK = 0, 1, 2, 3, 4, 5
A_FWD_NONE = 1          # a[2] = {-400, 0, +400}[i] (surfcore.h): 1 is "none"
OBS_ONGROUND = 4        # obs slot 4 (surfcore.h): 1.0 on ground, 0.0 airborne
# The two heads --yaw-cond couples, and their NEUTRAL bin. YAW_BINS is
# ascending with 0 deg in the middle (surfgym/core.py), so index 7 of 15 is
# "hold the view"; the side key is {-400, 0, +400}[i] like forward/back, so
# index 1 is "no strafe key". BELOW its neutral the yaw head turns the view
# RIGHT (clockwise, negative delta) and the side head presses A (-400, whose
# wishdir is LEFT); ABOVE, the yaw head turns left and the side head presses
# D (+400, wishdir right). The pairing air-strafing wants is therefore
# OPPOSITE sides of the two neutrals - see act/yaw_side_agree.
NEUTRAL_YAW = NVEC[H_YAW] // 2      # 7
NEUTRAL_SIDE = A_FWD_NONE           # 1 - the side head has forward's layout


class ActionMasks:
    """-inf masks on three of the six factored heads. Every one is opt-in.

    Why (runs/research/wr_demo/wr_vs_ours.md, sections 5 and 6): GoldSrc air
    movement accelerates along wishdir, the NORMALISED sum of the forward and
    side move vectors, and PM_AirAccelerate caps the gain at 30 u/s per frame
    on the projection onto wishdir. A strafe key alone puts wishdir 90 deg
    off velocity, which is the whole gain; adding W swings it to 45 deg and
    buys nothing at flight speed. The human world record holds W/S on 0% of
    its airborne frames and ours on 11.4%, presses jump 0 times against our
    283 and duck 0 against our 176, and flips strafe direction 0.42 times a
    second against our 1.50.

    Each mask is applied as an ADDITIVE -inf on the offending logits, in the
    four places a factored-categorical policy has to agree with itself:

      1. the rollout's sample  (train_fast step_compute)
      2. the update's log-prob recomputation  (mb_step / seq_loss)
      3. the greedy / stochastic eval  (Greedy|SampledTorchPolicy._decide)
      4. tools/record_ckpt.py, which rebuilds the policy from the ckpt config

    Masking the LOGITS rather than overwriting the sampled action is what
    keeps PPO honest: the ratio is exp(logp_new - logp_old) and both terms
    must be log-probabilities of the SAME distribution. Overwriting a sample
    leaves pi_old crediting an action the behaviour policy never emitted, and
    the recomputed pi_new would score it under an unmasked head - a ratio of
    two different measures, which is a silent, permanent bias. See
    tests/python/test_air_masks.py.

    NEG is finite (-1e30), so a masked slot's probability is exactly 0, its
    log-softmax term is exactly -1e30 and never selected, and p*logp is 0
    rather than NaN - the same trick HeadPacker already uses for the padding
    slots of the short heads.
    """

    __slots__ = ("fwd_air", "jump_cd", "duck_air")

    def __init__(self, fwd_air=False, jump_cd=0, duck_air=False):
        self.fwd_air = bool(fwd_air)
        self.jump_cd = int(jump_cd or 0)
        self.duck_air = bool(duck_air)
        if self.jump_cd < 0:
            raise ValueError(f"--jump-cooldown must be >= 0, got {jump_cd}")

    @property
    def on(self) -> bool:
        return self.fwd_air or self.jump_cd > 0 or self.duck_air

    @property
    def needs_air(self) -> bool:
        return self.fwd_air or self.duck_air

    def config(self) -> dict:
        """The run.json / ckpt-config keys, written ONLY when set - a run
        with no mask dumps byte-for-byte the config it dumped before this
        existed."""
        d = {}
        if self.fwd_air:
            d["mask_forward_air"] = 1
        if self.jump_cd > 0:
            d["jump_cooldown"] = self.jump_cd
        if self.duck_air:
            d["duck_air_mask"] = 1
        return d

    @classmethod
    def from_config(cls, cfg: dict) -> "ActionMasks":
        cfg = cfg or {}
        return cls(fwd_air=bool(cfg.get("mask_forward_air")),
                   jump_cd=int(cfg.get("jump_cooldown") or 0),
                   duck_air=bool(cfg.get("duck_air_mask")))

    def describe(self) -> str:
        if not self.on:
            return "action masks: none"
        bits = []
        if self.fwd_air:
            bits.append("forward/back forced to none while airborne")
        if self.jump_cd > 0:
            bits.append(f"jump masked for {self.jump_cd} decision(s) "
                        "after a press")
        if self.duck_air:
            bits.append("duck masked while airborne")
        return "action masks: " + "; ".join(bits)

    def add_mask(self, padded, air=None, jblk=None):
        """padded (..., NACT, NPAD) -> padded + an additive -inf mask.

        `air` (...,) 1 where the player is AIRBORNE at the decision tick;
        `jblk` (...,) 1 where jump is inside its cooldown. Out of place and
        shape-preserving, so all four call sites hand it the same arguments
        and get the same distribution back. The mask tensor carries no grad,
        so the gradient reaching a masked logit is exactly zero.
        """
        if not self.on:
            return padded
        m = torch.zeros_like(padded)
        if self.needs_air:
            a = air.to(padded.dtype) * NEG          # 0.0 or NEG
            if self.fwd_air:
                m[..., H_FWD, 0] = a                # -400 (S)
                m[..., H_FWD, 2] = a                # +400 (W)
            if self.duck_air:
                m[..., H_DUCK, 1] = a               # IN_DUCK held
        if self.jump_cd > 0:
            m[..., H_JUMP, 1] = jblk.to(padded.dtype) * NEG   # IN_JUMP held
        return padded + m

    def legalize_(self, act, air=None, jblk=None):
        """Force an EXTERNALLY drawn action onto the mask's support, in place.

        Only ez-greedy / --spawn-burst rows need this: those actions are
        drawn uniformly, not from the logits, so the mask has no say over
        them. They are excluded from the PPO pool by b_ez, so this changes
        behaviour only, never a ratio."""
        if not self.on:
            return act
        if self.needs_air:
            a = air.to(torch.bool)
            if self.fwd_air:
                act[:, H_FWD] = torch.where(
                    a, torch.full_like(act[:, H_FWD], A_FWD_NONE),
                    act[:, H_FWD])
            if self.duck_air:
                act[:, H_DUCK] = torch.where(
                    a, torch.zeros_like(act[:, H_DUCK]), act[:, H_DUCK])
        if self.jump_cd > 0:
            j = jblk.to(torch.bool)
            act[:, H_JUMP] = torch.where(
                j, torch.zeros_like(act[:, H_JUMP]), act[:, H_JUMP])
        return act

    @staticmethod
    def step_cooldown(cd, pressed, reload_):
        """The cooldown recurrence, one decision: a press re-arms to N, no
        press counts down to 0. Shared by the trainer (tensors) and the eval
        wrapper (numpy, _mask_note) so there is ONE definition of it."""
        return torch.where(pressed, reload_, (cd - 1.0).clamp_min(0.0))


# --------------------------------------------------------------------------
# opt-in AUTOREGRESSIVE side key: --yaw-cond
# --------------------------------------------------------------------------
# docs/research-litsurvey-temporal.md section 1.4, proposal #3. The six heads
# are conditionally independent given the trunk features, and air-strafing
# needs the yaw delta and the side key to turn the view TOWARD the held key's
# wish direction - hold A (side 0, -400) and turn left (yaw bin > 7), hold D
# (side 2, +400) and turn right (yaw bin < 7). A factored distribution that
# wants "either (left, left) or (right, right), never a mixed pair" cannot
# express it, and at the symmetric point each head's gradient toward
# committing is proportional to (2 p_other - 1), which is exactly 0 when the
# other head is undecided: the left/right symmetry is a SADDLE. Measured on
# the finisher xQR32, the two disagree on 12.7% of fast airborne decisions
# against the human world record's 2.7%.
#
# The fix is one factorisation step, the AlphaStar / Metz et al. (arXiv
# 1705.05035) / VPT arrangement:
#
#     log p(a) = log p(yaw) + log p(side | yaw) + sum_{other heads} log p(.)
#
# implemented as an ADDITIVE term on the side head's logits, gathered out of
# an (n_yaw, n_side) = (15, 3) table by the yaw bin THIS decision uses. The
# log-prob is therefore exact - PPO's ratio is exp(logp_new - logp_old) and
# both terms are log-probabilities of the same measure, unchanged in FORM -
# and the table is initialised to ZERO, so at step 0 the model is
# function-identical to the unconditioned one and a warm resume starts on
# the checkpoint's own curve (widen_for_yawcond).
#
# The conditioning enters in four places, like the action masks:
#   1. the rollout's sample      (step_compute, inside the CUDA graph)
#   2. the update's log-prob     (mb_step / seq_loss, on the STORED yaw)
#   3. the greedy / stochastic eval (Greedy|SampledTorchPolicy._decide)
#   4. tools/record_ckpt.py, which rebuilds the policy from the ckpt config
#
# ORDER against the action masks: the conditioning is added FIRST and the
# mask LAST, at every site. Both are additive on the padded logits and today
# they touch disjoint heads (the masks write H_FWD / H_JUMP / H_DUCK, the
# conditioning writes H_SIDE), so the two orders are bit-identical right
# now - but NEG is finite (-1e30) and a finite conditioning term added to a
# masked slot would leave it merely very negative instead of exactly zero
# probability. Mask last is the only order in which the mask has the last
# word, and it is what tests/python/test_yaw_cond.py pins.
#
# ENTROPY. logprob_entropy_padded is handed the CONDITIONED logits, so the
# side head contributes H(side | yaw = a_yaw) - the conditional entropy at
# the yaw bin this row actually drew - and the yaw head contributes its own
# exact H(yaw). Summed over the minibatch that is a ONE-SAMPLE estimator of
# H(yaw) + E_{yaw ~ pi}[H(side | yaw)], the joint entropy, and the sample is
# drawn from pi_old rather than pi_new. Two consequences, both deliberate:
# the estimate is unbiased only at the start of an epoch (where pi_new =
# pi_old), and the gradient it carries is the exact one through the side
# head and the table at that yaw but omits the d/d(yaw logits) term of
# E_yaw[H(side|yaw)]. This is a bias in the ENTROPY BONUS - a regulariser
# whose coefficient is 0.005 - never in the log-prob, which is exact and is
# what PPO's ratio and its clip are built on.


class _YawSideCond(nn.Module):
    """The (n_yaw, n_side) conditioning table, as a MODULE.

    A module rather than a bare nn.Parameter on Policy because
    ``Module.parameters()`` yields a module's OWN parameters before its
    children's: a direct Parameter would land at index 0 and shift every
    existing parameter's index, and Adam's state is keyed by that index -
    a resume would silently pair each tensor with the previous one's
    moments. Registered LAST among the submodules, its parameter is
    APPENDED instead, which is what lets widen_for_yawcond hand a pre-flag
    checkpoint a fresh slot rather than rebuild the optimizer state.

    Zeros, not a draw: it consumes no RNG (so a scratch run with the flag
    initialises every other tensor exactly as its control does) and it
    makes the conditioned logits identical to the unconditioned ones at
    step 0.
    """

    def __init__(self, n_yaw: int, n_side: int):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(n_yaw, n_side))


def add_yaw_cond(padded, table, yaw_act):
    """padded (B, NACT, NPAD) + table[yaw_act] on the SIDE head's live slots.

    `yaw_act` (B,) long is the yaw bin the decision uses - sampled in the
    rollout, argmax in a greedy eval, the STORED action in the update. Out
    of place and shape-preserving, like ActionMasks.add_mask, so every call
    site hands it the same arguments and gets the same distribution back.
    The gather is an index_select on a (15, 3) table: static shape, no
    data-dependent control flow, captures into the rollout's CUDA graph.
    Only the side head's NVEC[H_SIDE] live slots move - the padding slots
    of the short heads stay at NEG, so they stay unselectable.
    """
    n_side = NVEC[H_SIDE]
    row = table.index_select(0, yaw_act.reshape(-1)).to(padded.dtype)
    add = torch.zeros_like(padded)
    add[..., H_SIDE, :n_side] = row.view(*yaw_act.shape, n_side)
    return padded + add


def sample_padded_yawcond(padded, table, mask_fn=None):
    """--yaw-cond sampling: the yaw bin, then the side key GIVEN it.

    Two Gumbel-argmax stages sharing ONE noise tensor. The yaw head reads
    `gum[..., H_YAW, :]` and the side head `gum[..., H_SIDE, :]`; those are
    independent slices of iid noise, so the pair is drawn exactly from
    p(yaw) p(side | yaw) even though the noise is drawn once. Same
    rand_like, same static shapes and the same reductions as sample_padded,
    so this captures into the rollout CUDA graph the same way.

    `mask_fn` applies the action masks and is called AFTER the conditioning
    (see the ORDER note above). It is applied before the yaw draw as well,
    so the yaw bin comes from the distribution the yaw head actually has -
    a no-op today, since no mask touches H_YAW, and correct if one ever
    does.
    """
    u = torch.rand_like(padded).clamp_min_(1e-20)
    gum = -torch.log(-torch.log(u))
    p0 = padded if mask_fn is None else mask_fn(padded)
    yaw = (p0[..., H_YAW, :] + gum[..., H_YAW, :]).argmax(-1)
    p1 = add_yaw_cond(padded, table, yaw)
    if mask_fn is not None:
        p1 = mask_fn(p1)
    act = (p1 + gum).argmax(-1)
    lsm = F.log_softmax(p1, dim=-1)
    logp = lsm.gather(-1, act.unsqueeze(-1)).squeeze(-1).sum(-1)
    return act, logp


def greedy_padded_yawcond(padded, table, mask_fn=None):
    """--yaw-cond greedy: argmax yaw, then argmax side given it.

    The eval half of sample_padded_yawcond, and the mode of the same joint:
    argmax over p(yaw) p(side | yaw) factorises head by head because the
    other four heads stay independent of both.
    """
    p0 = padded if mask_fn is None else mask_fn(padded)
    yaw = p0[..., H_YAW, :].argmax(-1)
    p1 = add_yaw_cond(padded, table, yaw)
    if mask_fn is not None:
        p1 = mask_fn(p1)
    return p1.argmax(-1)


def sample_code(logits):
    """Gumbel-argmax sample + logprob from ONE categorical over K codes.

    The --chunk counterpart of sample_padded, and strictly simpler: no
    scatter, no per-head sum. Same rand_like gumbel trick, so it is equally
    CUDA-graph capturable (the rollout's step_compute is captured)."""
    u = torch.rand_like(logits).clamp_min_(1e-20)
    code = (logits - torch.log(-torch.log(u))).argmax(-1)
    lsm = F.log_softmax(logits, dim=-1)
    return code, lsm.gather(-1, code.unsqueeze(-1)).squeeze(-1)


def logprob_entropy_code(logits, codes):
    """logp/entropy of the single code categorical (the --chunk update).

    Entropy here is over BEHAVIORS, which is the whole point: the flat
    six-head entropy sums six independent per-head entropies, so "high
    entropy" means 33 Hz white noise on every channel at once. Over codes it
    means the policy is undecided between coherent H-decision behaviors."""
    lsm = F.log_softmax(logits, dim=-1)
    logp = lsm.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
    return logp, -(lsm.exp() * lsm).sum(-1)
def h64(b) -> int:
    """Deterministic cross-process 64-bit digest. Builtin ``hash()`` is
    salted per process (PYTHONHASHSEED), so it can never be compared across
    DDP ranks or across runs. Accepts bytes or an ndarray — arrays hash
    through the buffer protocol with NO byte copy (the goal-field grid is
    ~GBs; four ranks each cloning it via tobytes() is a real memory spike)."""
    if isinstance(b, np.ndarray):
        b = memoryview(np.ascontiguousarray(b))
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(),
                          "little", signed=True)


def adv_moments64(st_m, n_g: float):
    """Fleet advantage moments from the SUMMED (2, M) f64 [sum, sumsq]
    stack: mean and ddof=1 std per minibatch, exactly torch.std's estimator
    (docs/ddp-plan.md §2). Module-level so the tier-A test exercises the
    shipped formula, not a re-derivation."""
    m64 = st_m[0] / n_g
    v64 = (st_m[1] - st_m[0] * m64) / (n_g - 1.0)
    return m64.float(), v64.clamp_min(0).sqrt().float()


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
                 lidar=None, core=None, act_every: int = 1, stack: int = 1,
                 extra_slot: int = -1, extra_fn=None, route=None,
                 latch_fn=None, pitch_fixed=None, aux=None, masks=None,
                 priv_fn=None):
        self.policy, self.packer, self.device = policy, packer, device
        self.lidar, self.core = lidar, core
        # --pitch-fixed: the trainer pins the states' pitch column before
        # every render, so an eval that did not would aim these weights
        # somewhere they were never trained to look. None = off.
        self.pitch_fixed = (None if pitch_fixed is None
                            else float(pitch_fixed))
        # --route: an eval that skipped the lookahead fan would feed the
        # policy a row of the right WIDTH only by accident and of the wrong
        # CONTENT always — the same class of bug the extra_slot note below
        # describes, and it lands on race/eval_progress, which is the number
        # every arm is judged by
        self.route = route
        # --race-latch: the flag is a real observation column the policy
        # was trained on, and it is the ONLY thing separating the two
        # reward regimes. An eval that fed it a constant would be
        # evaluating a different network input than training wrote - the
        # same class of bug as a skipped route fan, and it lands on the
        # number every arm is judged by.
        self.latch_fn = latch_fn
        # --act-hist / --obs-compass: a surfgym.obsaux.ObsAux, the SAME class
        # the rollout drives. Two implementations of one feature drift, and
        # the drift only shows up as eval recordings that disagree with
        # training (tests/python/test_framestack.py makes the same argument
        # about the frame ring). The eval has no episode-end signal, so
        # ObsAux.eval_features reads the core's per-env tick counter, exactly
        # like _push_frame below.
        self.aux = aux
        # --obs-reward writes a side-channel value into a scalar slot during
        # TRAINING. The core does not produce it, so without this hook an
        # eval feeds whatever the core has in that slot (slot 12 is absolute
        # position / 2000, magnitude up to ~10) to a policy trained on
        # tanh(reward) in [-1, 1] - a badly out-of-distribution feature that
        # wrecks the eval while training looks fine.
        self.extra_slot, self.extra_fn = int(extra_slot), extra_fn
        # --priv-critic: the CRITIC's privileged block (surfgym/privfeat.py).
        # An eval reads only the logits, and the privileged block cannot
        # reach them - so this changes no recorded action. It is wired
        # anyway because leaving it None makes V(s) NaN here (Policy._value),
        # and a value column silently full of NaN in an eval is the kind of
        # thing that is discovered three rounds later. None = no privileged
        # critic in this checkpoint.
        self.priv_fn = priv_fn
        self._k = max(1, int(act_every))
        self._tick = 0
        self._held = None
        self._stack = max(1, int(stack))
        self._ring = self._prev_tick = None
        # --rnn: the recurrent state, carried across decisions and zeroed
        # at every episode start (see _net); _period is the tick distance
        # between two decisions, K here and K*H for the chunk classes
        self._h = self._h_tick = None
        self._period = self._k
        # --mask-forward-air / --jump-cooldown / --duck-air-mask. An eval or
        # a recording that skipped them would be running a DIFFERENT policy
        # than training optimised - the support is part of what the weights
        # mean - so the recorder reads them out of the ckpt config and hands
        # them here. `None` = no mask, which is every pre-flag checkpoint.
        self.masks = masks if masks is not None else ActionMasks()
        self._jcd = self._jcd_tick = None      # jump cooldown, per env
        # --yaw-cond: the conditioning table lives IN the policy, so its
        # presence IS the flag - a recorder or an in-trainer eval needs no
        # extra argument and cannot forget it. None on every pre-flag
        # checkpoint. An eval that skipped it would draw the side key from
        # an unconditioned head, which is a different policy than the one
        # PPO optimised (the same class of mismatch as a skipped mask).
        self.yaw_table = (None if getattr(policy, "yaw_side", None) is None
                          else policy.yaw_side.table)
        # --view-continuous: the float view command of the decision being
        # held, (N, 2) float32, read by record_rollout / beam_tas's
        # run_episode next to the int action row; None on a discrete policy
        # so every existing caller keeps calling core.step(acts) alone. The
        # pitch ceiling comes off the CORE the wrapper drives, which is the
        # one the trainer built with the same pitch_rate_core.
        self.view = None
        self.view_continuous = bool(getattr(policy, "view_continuous", False))
        # --view-absolute rides in the policy like the flag does: the
        # wrapper hands the mode to view_from_z_t and the core it drives
        # was built with the matching view_mode (train_fast / record_ckpt)
        self.view_absolute = getattr(policy, "view_absolute", None)
        self._pitch_max = (float(core.config.pitch_rate_max_deg)
                           if core is not None else 10.0)

    def _finish_view(self, cat_act, z):
        """--view-continuous: assemble the int action row (NEUTRAL in the two
        view columns) and publish the view command for this decision."""
        n = cat_act.shape[0]
        act = np.empty((n, NACT), np.int32)
        act[:, 0] = NEUTRAL_ACT[0]
        act[:, 1] = NEUTRAL_ACT[1]
        act[:, N_VIEW:] = cat_act.to("cpu").numpy().astype(np.int32)
        self.view = np.ascontiguousarray(
            view_from_z_t(z, self._pitch_max, self.view_absolute)
            .to("cpu").numpy(), np.float32)
        self._mask_note(act)
        return act

    def act(self, obs):
        if self._held is None or self._tick % self._k == 0:
            self._held = self._decide(obs)
            if self.aux is not None:
                # AFTER _decide, so the row the policy just read carried the
                # history as of the PREVIOUS decisions - the rollout's order
                # (push at the decision boundary, observe on the next one)
                self.aux.push(self._held)
        self._tick += 1
        return self._held

    def _net(self, x):
        """policy(x) -> (logits, value), with the --rnn state carried.

        Episode starts come off the core's per-env tick counter, which
        reset_env zeroes (src/env.c): between two decisions exactly _period
        ticks elapse, so a counter that advanced by LESS has been reset in
        between and that env's state is zeroed before the step - the same
        rule the trainer applies through b_done, and stricter than the
        frame ring's `tick <= prev` (which misses a reset inside an
        episode younger than one decision). With no core attached nothing
        can be detected and h is simply carried."""
        pv = None
        if self.priv_fn is not None:
            pv = torch.as_tensor(np.ascontiguousarray(self.priv_fn(self.core)),
                                 dtype=torch.float32, device=x.device)
        if self.policy.gru is None:
            return self.policy(x, priv=pv)
        n = x.shape[0]
        if self._h is None or self._h.shape[0] != n:
            self._h = torch.zeros(n, self.policy.rnn_size, device=x.device)
            self._h_tick = None
        if self.core is not None:
            tick = np.asarray(self.core.states_view["tick"], np.int64)
            if self._h_tick is not None:
                started = tick < self._h_tick + self._period
                if started.any():
                    self._h = self._h * torch.as_tensor(
                        np.ascontiguousarray(~started).astype(np.float32),
                        device=x.device).unsqueeze(1)
            self._h_tick = tick.copy()
        logits, value, self._h = self.policy(x, self._h, priv=pv)
        return logits, value

    def _obs(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self.extra_fn is not None and self.extra_slot >= 0:
            t[:, self.extra_slot] = torch.as_tensor(
                self.extra_fn(self.core), dtype=torch.float32,
                device=self.device)
        if self.route is not None:
            sv = self.core.states_view
            o = torch.as_tensor(np.ascontiguousarray(sv["origin"]),
                                dtype=torch.float32, device=self.device)
            yw = torch.as_tensor(sv["yaw"].copy(), dtype=torch.float32,
                                 device=self.device)
            t = torch.cat([t, self.route.features(o, yw, t[:, 3] * 1000.0)],
                          dim=1)
        if self.latch_fn is not None:
            # LAST column of the scalar half, exactly where the trainer
            # writes it and where widen_for_route padded the checkpoint
            t = torch.cat([t, torch.as_tensor(
                self.latch_fn(self.core), dtype=torch.float32,
                device=self.device).reshape(-1, 1)], dim=1)
        if self.aux is not None:
            # the TRAILING block of the scalar half, after the fan and the
            # latch - exactly the order fill_vision writes it in
            sv = self.core.states_view
            t = torch.cat([t, torch.as_tensor(
                np.ascontiguousarray(self.aux.eval_features(
                    sv["origin"], sv["yaw"], sv["tick"])),
                dtype=torch.float32, device=self.device)], dim=1)
        if self.lidar is not None:
            if self.pitch_fixed is not None:
                self.core.set_pitch(self.pitch_fixed)
            sv = self.core.states_view
            o = torch.as_tensor(np.ascontiguousarray(sv["origin"]),
                                dtype=torch.float32, device=self.device)
            yw = torch.as_tensor(sv["yaw"].copy(), dtype=torch.float32,
                                 device=self.device)
            pt = torch.as_tensor(sv["pitch"].copy(), dtype=torch.float32,
                                 device=self.device)
            dk = torch.as_tensor(sv["ducked"].copy().astype(np.int32),
                                 device=self.device)
            depth = self.lidar.render(o, yw, pt, dk).reshape(t.shape[0], -1)
            if self._stack > 1:
                depth = self._push_frame(depth, sv["tick"])
            t = torch.cat([t, depth], dim=1)
        return t

    def _mask_air(self, obs):
        """The airborne flag the mask keys on: obs slot 4, the same column
        the trainer reads out of static_obs. It is written by write_obs
        (src/env.c) from `st->onground != -1` of the state this decision is
        being made in, so it is the state AT the decision tick - not one
        tick stale - and it is held for the whole act_every repeat, exactly
        as in training."""
        return torch.as_tensor(
            np.ascontiguousarray(obs[:, OBS_ONGROUND] < 0.5),
            device=self.device)

    def _mask_jblk(self, n):
        """The jump-cooldown flag, per env, as of THIS decision.

        The counter is cleared at every episode start, read off the core's
        per-env tick counter the same way the --rnn state and the frame ring
        are: between two decisions exactly _period ticks elapse, so a counter
        that advanced by less has been reset in between."""
        if self.masks.jump_cd <= 0:
            return None
        if self._jcd is None or self._jcd.shape[0] != n:
            self._jcd = np.zeros(n, np.int64)
            self._jcd_tick = None
        if self.core is not None:
            tick = np.asarray(self.core.states_view["tick"], np.int64)
            if self._jcd_tick is not None:
                self._jcd[tick < self._jcd_tick + self._period] = 0
            self._jcd_tick = tick.copy()
        return torch.as_tensor(self._jcd > 0, device=self.device)

    def _mask_note(self, act):
        """Advance the cooldown with the action just taken (ActionMasks.
        step_cooldown in numpy: a press re-arms to N, otherwise count down)."""
        if self.masks.jump_cd > 0 and self._jcd is not None:
            self._jcd = np.where(act[:, H_JUMP] > 0, self.masks.jump_cd,
                                 np.maximum(self._jcd - 1, 0))

    def _mask_padded(self, padded, obs):
        """packer.pad(...) -> the masked logits the trainer would have used."""
        if not self.masks.on:
            return padded
        return self.masks.add_mask(
            padded, self._mask_air(obs) if self.masks.needs_air else None,
            self._mask_jblk(obs.shape[0]))

    def _push_frame(self, frame, tick):
        """The rollout's ring, one decision at a time (--frame-stack).

        Episode starts are read off the core's per-env tick counter, which
        reset_env zeroes (src/env.c): record_rollout never tells a policy an
        episode ended, and inference has to collapse its history at a spawn
        exactly like training does or the recorded policy is not the trained
        one. Called once per DECISION — _obs only runs inside _decide."""
        n = frame.shape[0]
        if self._ring is None or self._ring.buf.shape[1] != n:
            self._ring = FrameRing(self._stack, n, frame.shape[1],
                                   frame.device, frame.dtype)
            self._prev_tick = None
        tick = np.asarray(tick, np.int64)
        started = None if self._prev_tick is None else torch.as_tensor(
            np.ascontiguousarray(tick <= self._prev_tick), device=frame.device)
        self._prev_tick = tick.copy()
        self._ring.push(frame, started)
        return self._ring.compose()


class GreedyTorchPolicy(_TorchPolicyBase):
    @torch.inference_mode()
    def _decide(self, obs):
        logits, _ = self._net(self._obs(obs))
        if self.view_continuous:
            # --view-continuous, greedy: z = mu for the view, argmax for
            # the four categorical heads (masked like the discrete branch)
            cat, mu = split_view(logits.float())
            padded = self._mask_padded(self.packer.pad(cat), obs)
            return self._finish_view(padded[:, N_VIEW:].argmax(-1), mu)
        if self.yaw_table is not None:
            # PLACE 3 of 4 for --yaw-cond: argmax the yaw bin, then argmax
            # the side key from logits carrying that bin's conditioning
            # row. The masks go through the same _mask_padded the plain
            # branch uses, applied INSIDE and after the conditioning.
            act = greedy_padded_yawcond(
                self.packer.pad(logits), self.yaw_table,
                lambda p: self._mask_padded(p, obs))
        else:
            act = self._mask_padded(self.packer.pad(logits), obs).argmax(-1)
        act = act.to("cpu").numpy().astype(np.int32)
        self._mask_note(act)
        return act


class SampledTorchPolicy(_TorchPolicyBase):
    """Acts by sampling the distribution — the policy training actually
    optimizes and logs. Under a high entropy coefficient the argmax mode can
    be much weaker (it drifts unoptimized while the stochastic policy learns
    to rely on its own action noise)."""

    @torch.inference_mode()
    def _decide(self, obs):
        logits, _ = self._net(self._obs(obs))
        if self.view_continuous:
            # --view-continuous, stochastic: the rollout's own draw
            cat, mu = split_view(logits.float())
            padded = self._mask_padded(self.packer.pad(cat), obs)
            act, z, _ = sample_view(padded, mu, self.policy.log_std())
            return self._finish_view(act[:, N_VIEW:], z)
        if self.yaw_table is not None:
            # PLACE 3 of 4 for --yaw-cond, stochastic half (same argument
            # as the greedy one above)
            act, _ = sample_padded_yawcond(
                self.packer.pad(logits), self.yaw_table,
                lambda p: self._mask_padded(p, obs))
        else:
            act, _ = sample_padded(
                self._mask_padded(self.packer.pad(logits), obs))
        act = act.to("cpu").numpy().astype(np.int32)
        self._mask_note(act)
        return act


class _ChunkPolicyBase(_TorchPolicyBase):
    """--chunk eval: ONE trunk forward per chunk of H decisions.

    Mirrors _TorchPolicyBase.act's act_every hold, one level up. The policy is
    asked once every act_every*H ticks and answers with a whole PLAN — the
    (H, 6) action sequence its chosen code decodes to — whose rows are then
    held act_every ticks each. The decoder is a Parameter of the policy, so a
    checkpoint carries it and an eval needs no side file.

    Unlike the base class this re-deliberates at an episode boundary. In the
    rollout a chunk is a fixed row of a rectangular buffer, so a mid-chunk
    episode end masks the tail to NEUTRAL_ACT (design doc 4.3); an eval has no
    buffer to keep rectangular, and letting a stale plan run on into a fresh
    spawn would inject up to act_every*H ticks of unrelated behavior at the
    start line — which is exactly what the eval metric measures.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if getattr(self.policy, "decoder", None) is None:
            raise ValueError("this policy has no chunk decoder — the "
                             "checkpoint was not trained with --chunk")
        if self.view_continuous:
            raise ValueError("--chunk and --view-continuous are exclusive")
        self._H = int(self.policy.decoder.shape[1])
        self._plan = None
        self._chunk_tick = None
        self._period = self._k * self._H      # one GRU step per CHUNK

    def act(self, obs):
        if self.core is not None:
            # src/env.c zeroes the per-env tick counter on reset; record.py
            # never tells a policy an episode ended (the same signal
            # _push_frame reads for the frame ring)
            tk = np.asarray(self.core.states_view["tick"], np.int64)
            if (self._chunk_tick is not None
                    and bool((tk <= self._chunk_tick).any())):
                self._plan, self._tick = None, 0
            self._chunk_tick = tk.copy()
        if self._plan is None or self._tick % (self._k * self._H) == 0:
            self._plan = self._decide_chunk(obs)
            self._tick = 0
        h = (self._tick // self._k) % self._H
        row = np.ascontiguousarray(self._plan[:, h])
        if self.aux is not None and self._tick % self._k == 0:
            # --act-hist under --chunk: the engine still sees H separate
            # DECISIONS per chunk, and the rollout pushes each of them
            # (train_fast's `_j % K == 0`), so this pushes each of them too
            self.aux.push(row)
        self._tick += 1
        return row


class GreedyChunkPolicy(_ChunkPolicyBase):
    @torch.inference_mode()
    def _decide_chunk(self, obs):
        logits, _ = self._net(self._obs(obs))
        dec = self.policy.decoder[logits.argmax(-1)]        # (B, H, sum(NVEC))
        plan = self.packer.pad_seq(dec.float()).argmax(-1)  # (B, H, 6)
        return plan.to("cpu").numpy().astype(np.int32)


class SampledChunkPolicy(_ChunkPolicyBase):
    @torch.inference_mode()
    def _decide_chunk(self, obs):
        logits, _ = self._net(self._obs(obs))
        code, _ = sample_code(logits.float())
        plan, _ = sample_padded(
            self.packer.pad_seq(self.policy.decoder[code].float()))
        return plan.to("cpu").numpy().astype(np.int32)


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


def _seg_hits_box(pts, box):
    """True where ANY consecutive segment of ``pts`` (T, 3) pierces the AABB.

    The same swept slab test ``src/env.c:seg_hits_box`` uses for the finish
    curtain, and for the same reason: a 1 u thin trigger at 3,500 u/s is
    35 u of travel per tick, and a point-in-box check tunnels straight
    through it. The recording is per PHYSICS TICK, so the segments here are
    literally the segments the simulator tested - this reproduces the env's
    own goal_hit for the recorded env rather than approximating it.
    """
    if len(pts) < 2:
        return False
    p0 = np.asarray(pts[:-1], np.float64)
    p1 = np.asarray(pts[1:], np.float64)
    bmin = np.asarray(box["mins"], np.float64)
    bmax = np.asarray(box["maxs"], np.float64)
    t0 = np.zeros(len(p0))
    t1 = np.ones(len(p0))
    ok = np.ones(len(p0), bool)
    for k in range(3):
        d = p1[:, k] - p0[:, k]
        par = np.abs(d) < 1e-9
        ok &= ~par | ((p0[:, k] >= bmin[k]) & (p0[:, k] <= bmax[k]))
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (bmin[k] - p0[:, k]) / d
            b = (bmax[k] - p0[:, k]) / d
        lo = np.where(par, -np.inf, np.minimum(a, b))
        hi = np.where(par, np.inf, np.maximum(a, b))
        t0 = np.maximum(t0, lo)
        t1 = np.minimum(t1, hi)
    return bool((ok & (t0 <= t1)).any())


# ---- PPO hygiene: explained variance, return normalization, eval stalls ---
# All three are additive. With --ret-norm 0 and --eval-stall 0 (the defaults)
# nothing below touches a number the trainer produced before; the two logged
# fractions and the explained variance are read-outs of tensors and masks the
# rollout already computes.


def explained_var_from_sums(n, sum_y, sum_yy, sum_e, sum_ee) -> float:
    """``1 - Var(returns - values) / Var(returns)`` from SUFFICIENT
    STATISTICS, so the DDP path can reduce it exactly with one all-reduce.

    ``y`` is the return, ``e = y - V(s)`` the residual (which, because
    ``ret = adv + b_val`` by construction, IS the GAE advantage - the
    residual never has to be materialised).

    Population variances, matching SB3's ``explained_variance``. The result
    is unbounded below by definition (a critic worse than the mean scores
    negative), and it is NOT clipped here - a large negative number is the
    diagnostic. What IS guarded is the degenerate case: a rollout whose
    returns have zero (or non-finite) variance has no variance to explain,
    and reports NaN rather than -inf or a ZeroDivisionError.
    """
    n = float(n)
    if n <= 0.0:
        return float("nan")
    var_y = sum_yy / n - (sum_y / n) ** 2
    var_e = sum_ee / n - (sum_e / n) ** 2
    if not (var_y > 0.0) or not math.isfinite(var_y) or not math.isfinite(var_e):
        return float("nan")
    ev = 1.0 - var_e / var_y
    return ev if math.isfinite(ev) else float("nan")


# 300 u/s. Below this an episode did not fly: it slid, it walked, or it
# spent the whole clock stuck on a wall. The threshold is deliberately far
# under anything a surfing policy does (the champion line holds 2,800-3,700
# u/s through the hard part) - it separates "did not move" from "died fast",
# which is the pair ep_len_mean alone cannot tell apart.
CRAWL_SPEED = 300.0


def episode_hygiene(ended, is_trunc, spd_sum, ep_len,
                    crawl_ku=CRAWL_SPEED / 1000.0):
    """``(n_ended, n_truncated, n_crawling)`` for the episodes ending NOW.

    Module-level so the shipped arithmetic is what the test exercises (same
    reason ``eval_aggregate`` and ``adv_moments64`` live out here).

    ``spd_sum`` is the running sum of obs slot 3 over the live episode, i.e.
    ``|v_xy| / 1000`` per physics tick, and ``ep_len`` its tick count - so
    ``spd_sum / ep_len`` is the episode's mean horizontal speed in ku/s and
    ``crawl_ku`` is the threshold in the same units. All three counts share
    one denominator by construction; reporting a truncation rate against a
    different episode count than the crawl rate is how a read-out starts
    disagreeing with itself.
    """
    ei = np.flatnonzero(ended)
    if not len(ei):
        return 0, 0, 0
    n_tr = int(np.count_nonzero(np.asarray(is_trunc, bool)[ei]))
    mean_spd = spd_sum[ei] / np.maximum(ep_len[ei], 1)
    return len(ei), n_tr, int(np.count_nonzero(mean_spd < crawl_ku))


class ReturnNorm:
    """PopArt-lite running normalizer for the VALUE TARGET (``--ret-norm``).

    The critic is asked to predict ``(G - mu) / sigma`` instead of ``G``.
    Every read of ``V`` that lives in reward units - GAE's bootstrap, the
    truncation bootstrap, the rollout buffer the advantages come out of - is
    de-normalized by the same pair, so the algorithm outside the value loss
    is unchanged and the advantages stay in reward units (advantage
    normalization is untouched).

    "lite" is the honest label: real PopArt (1602.07714) ALSO rescales the
    value head's last layer whenever ``(mu, sigma)`` move, so the function
    the network represents is preserved exactly across a statistics update.
    This does not; it relies on the EMA moving slowly (``beta = 0.99`` per
    ITERATION, i.e. a ~100-iteration time constant) so the frame the critic
    was fitted in is nearly the frame its output is read in.

    The statistics are DEBIASED (Adam-style, by ``1 - beta^k``): without it
    the first iterations would divide by a std that is 1% of the truth and
    the value loss would explode exactly where the run is least able to
    absorb it. After ONE update the pair is exactly that iteration's batch
    mean/std, which is the correct answer with one iteration of evidence.

    ``sigma`` is clipped to ``[1e-4, 1e6]`` as PopArt's own implementation
    does: a rollout in which every return is identical (a policy that dies at
    the same tick every episode - which is precisely the regime this project
    keeps landing in) has zero variance, and dividing by ``sqrt(eps)`` would
    hand the optimizer a 1e4x target.
    """

    SIGMA_MIN = 1e-4
    SIGMA_MAX = 1e6

    def __init__(self, beta: float = 0.99, eps: float = 1e-8) -> None:
        self.beta = float(beta)
        self.eps = float(eps)
        self._m = 0.0        # EMA of E[G], undebiased
        self._s = 0.0        # EMA of E[G^2], undebiased
        self._w = 0.0        # EMA of 1 -> 1 - beta^k, the debias weight
        self.mean = 0.0
        self.std = 1.0
        self.count = 0

    def update(self, mean_g: float, sq_g: float) -> tuple[float, float]:
        """Fold one iteration's return moments in and return (mean, std)."""
        b = self.beta
        self._m = b * self._m + (1.0 - b) * float(mean_g)
        self._s = b * self._s + (1.0 - b) * float(sq_g)
        self._w = b * self._w + (1.0 - b)
        d = max(self._w, 1e-12)
        mean = self._m / d
        var = max(self._s / d - mean * mean, 0.0)
        self.mean = mean
        self.std = min(max(math.sqrt(var + self.eps), self.SIGMA_MIN),
                       self.SIGMA_MAX)
        self.count += 1
        return self.mean, self.std

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"beta": self.beta, "eps": self.eps, "m": self._m,
                "s": self._s, "w": self._w, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.beta = float(d.get("beta", self.beta))
        self.eps = float(d.get("eps", self.eps))
        self._m = float(d.get("m", 0.0))
        self._s = float(d.get("s", 0.0))
        self._w = float(d.get("w", 0.0))
        self.count = int(d.get("count", 0))
        if self._w > 0.0:
            mean = self._m / self._w
            var = max(self._s / self._w - mean * mean, 0.0)
            self.mean = mean
            self.std = min(max(math.sqrt(var + self.eps), self.SIGMA_MIN),
                           self.SIGMA_MAX)


def make_eval_stall_hook(core, field, stall_ticks, stall_eps, every, env=0):
    """``record_rollout`` on_tick that applies TRAINING's stall rule.

    CLAUDE.md: "Evals do NOT stall-kill. Training does." ``core.force_fail``
    is only reached from the training rollout, so an eval episode of a
    crawling policy runs the full ``--ep-ticks`` budget and the platform eval
    measures a regime training never allowed. This is the mirror, and it is
    a MIRROR on purpose - every constant is read off the training reward:

      * the same distance field (``RaceReward.field``, RAW ``d`` - the
        ``--race-dfloor`` clamp is deliberately not applied, exactly as
        training's stall detector keeps the raw value);
      * the same 32u improvement threshold (``stall_eps``) against a running
        MINIMUM, not a rate and not a window budget;
      * the same window in physics ticks (``--stall-secs * 100``);
      * the same PER-CALL cadence: the rule is evaluated once every ``every``
        ticks, which is ``act_every * chunk`` under --reward-per-decision and
        1 otherwise. This matters - the threshold is per call, so evaluating
        it every tick at act_every 4 would quarter the effective step and
        kill legitimate flight (CLAUDE.md, "--stall-eps is now a flag").

    The kill is ``core.force_fail``, i.e. the episode ends as a FAIL on the
    next tick, which is what training does. Returns the hook; ``hook.state``
    carries ``n`` (kills issued) for the caller to report.
    """
    st = {"best": None, "since": 0, "phase": 0, "n": 0}

    def hook(t, states, rewards, done, trunc):
        if bool(done[env]) or bool(trunc[env]):
            st["best"] = None
            st["since"] = 0
            st["phase"] = 0
            return
        if stall_ticks <= 0:
            return
        st["phase"] += 1
        if st["phase"] < every:
            return
        st["phase"] = 0
        d = float(field.sample(
            core.states_view["origin"][env:env + 1].astype(np.float64))[0])
        best = st["best"]
        if best is None:                 # first call of the episode: arm it
            st["best"] = d
            return
        if d < best - stall_eps:
            st["best"] = d
            st["since"] = 0
            return
        st["best"] = min(best, d)
        st["since"] += every
        if st["since"] >= stall_ticks:
            st["since"] = 0              # re-arm, exactly like pop_stall_mask
            st["n"] += 1
            m = np.zeros(core.num_envs, np.uint8)
            m[env] = 1
            core.force_fail(m)

    hook.state = st
    return hook


def chain_ticks(*hooks):
    """Compose record_rollout on_tick callbacks (None entries dropped)."""
    live = [h for h in hooks if h is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def chained(t, states, rewards, done, trunc):
        for h in live:
            h(t, states, rewards, done, trunc)

    return chained


# ---- multi-map eval aggregation ------------------------------------------
# The per-map eval result row. All SUMS and COUNTS, deliberately: the DDP
# path reduces this table with a plain all-reduce SUM (each row is written
# by exactly the one rank that owns that map's eval), so no mean and no NaN
# may ever enter the collective.
EVAL_ROW = ("prog_u_sum", "finish_s_sum", "n_finish_geo", "pct_sum",
            "n_eps", "n_finish_box", "fwd_sum", "path_sum", "speed_sum",
            "evaluated")
EVAL_K = len(EVAL_ROW)


def eval_aggregate(ev, finish_kinds, n_rec: int) -> dict:
    """Turn the reduced (n_maps, EVAL_K) eval table into the reported numbers.

    Module-level so ``tests/python/test_multimap_ddp_metrics.py`` exercises
    the shipped formula rather than a re-derivation of it - the same reason
    ``adv_moments64`` lives at module level.

    The two headline numbers are deliberately NOT ``race/eval_progress``:

    * ``map_pct`` is the mean over maps of the mean over that map's eval
      episodes of the share of its OWN route covered. eval_progress is in
      map units, and this pool spans a 5x range of route length, so a units
      mean is a weighted vote in which the long maps decide the answer.
      It also saturates: the geodesic field's minimum along a route can sit
      mid-route (cannonball's is at 88%), and round 18 measured
      eval_progress moving ANTI-correlated with the true frontier.
    * ``maps_finished`` is the fraction of maps with at least one eval
      episode whose recorded path crosses the finish AABB - the env's own
      win test. Not the geodesic <= 150 u proxy, which a death-dive into
      goal-adjacent airspace passes without finishing anything.

    Both are also reported restricted to ``finish_kind == "trigger"``.
    43 maps of the pool have a real trigger curtain; the rest are +use
    button boxes ~8x smaller in face area which the simulator cannot press
    at all, so a null on one of those is much weaker evidence and pooling
    the two into one headline without the split beside it is a claim the
    data does not support (CLAUDE.md 4b).
    """
    ev = np.asarray(ev, np.float64)
    if ev.ndim != 2 or ev.shape[1] != EVAL_K:
        raise ValueError(f"eval table must be (n_maps, {EVAL_K}), got "
                         f"{ev.shape}")
    kinds = list(finish_kinds)
    if len(kinds) != len(ev):
        raise ValueError("finish_kinds must have one entry per map")
    n_ep = ev[:, 4]
    has = n_ep > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(has, ev[:, 3] / np.maximum(n_ep, 1.0), np.nan)
    nbox = ev[:, 5]
    trig = np.array([k == "trigger" for k in kinds], bool)
    tsel = trig & has

    def _mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    n_fin_geo = ev[:, 2].sum()
    return {
        "pct": pct,
        "n_eps": n_ep,
        "n_box": nbox,
        "evaluated": ev[:, 9] > 0,
        "map_pct": _mean(pct[has]),
        "maps_finished": _mean((nbox[has] > 0).astype(np.float64)),
        "map_pct_trigger": _mean(pct[tsel]),
        "maps_finished_trigger": _mean((nbox[tsel] > 0).astype(np.float64)),
        "n_maps_scored": int(has.sum()),
        "n_maps_finished": int((nbox[has] > 0).sum()),
        "n_trigger": int(tsel.sum()),
        "eval_prog": (float(ev[:, 0].sum() / (len(ev) * n_rec))
                      if n_rec else float("nan")),
        "eval_fin": (float(ev[:, 1].sum() / n_fin_geo)
                     if n_fin_geo else float("nan")),
        "eval_fwd": float(ev[:, 6].mean()),
        "eval_path": float(ev[:, 7].mean()),
        "eval_speed": float(ev[:, 8].mean()),
    }


# ---- --heldout-maps: the held-out block of the CSV ------------------------
# Per held-out map, in this order. "field" is in the column name on purpose:
# these are the geodesic FIELD metric (progress in that map's units and the
# share of the field's start distance covered), and CLAUDE.md records
# race/eval_progress being anti-correlated with the truth on cannonball. On
# a pool map with no route line it is all there is; where maps/<stem>.route
# .npz exists the honest order-only corridor MAX is logged beside it.
HELD_COLS = ("race/heldout_progress_field", "race/heldout_finish_s",
             "race/heldout_finishes", "race/heldout_pct_field")
HELD_CORR_COL = "race/heldout_corridor_max"


def heldout_columns(heldout) -> list:
    """CSV column names for the held-out slots: HELD_COLS per map, plus the
    corridor column for maps that have a route file. Appended AFTER every
    other column, so an existing progress.csv header stays a strict prefix
    of the new one (the header-migration rule)."""
    cols = []
    for h in heldout:
        cols += [f"{c}.{h.tag}" for c in HELD_COLS]
        if getattr(h, "route", None) is not None:
            cols.append(f"{HELD_CORR_COL}.{h.tag}")
    return cols


def heldout_csv_values(heldout, eval_per_map, held_corr) -> list:
    """The cells matching :func:`heldout_columns`, NaN -> ''."""
    def _r(v, nd):
        if v != v:
            return ""
        return int(v) if nd is None else round(float(v), nd)
    out = []
    for h in heldout:
        e = eval_per_map[h.tag]
        out += [_r(e[0], 1), _r(e[1], 2), _r(e[2], None), _r(e[3], 3)]
        if getattr(h, "route", None) is not None:
            out.append(_r(held_corr.get(h.tag, float("nan")), 1))
    return out


def corridor_max(traj_path: Path, route_path, corridor: float = 1500.0,
                 window: int = 16) -> float:
    """MAX over a recording's episodes of the ORDER-ONLY corridor progress
    along ``route_path`` - the same number ``tools/eval_honesty.py
    --order-only 16`` prints as the per-episode "order-only" figure, taken
    at its maximum (CLAUDE.md: "Corridor MAX and finishes are the
    frontier"). Pure geometry over the recorded positions; NaN when the
    recording holds no episode."""
    from surfgym.route import ArcProgress
    z = np.load(route_path)
    pts = np.asarray(z["route"], np.float64)
    spacing = float(z["spacing"]) if "spacing" in z.files else 128.0
    best_all = float("nan")
    rows = []

    def _close(rs):
        nonlocal best_all
        pp = np.asarray(rs, np.float64)[:, 1:4]
        ap_ = ArcProgress(pts, spacing, corridor=corridor, window=window)
        ap_.reset(pp[:1])
        best = float(ap_.arc[0])
        for k in range(1, len(pp)):
            ap_.advance(pp[k:k + 1])
            best = max(best, float(ap_.arc[0]))
        best_all = best if best_all != best_all else max(best_all, best)

    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict) and "map" in row:
                rows = []
            elif isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict) and "end" in row and rows:
                _close(rows)
                rows = []
    return best_all


def race_coverage(traj_path: Path, field, goal_box=None):
    """(pct_sum, n_episodes, n_box_finishes) for one recording.

    ``pct`` is per episode ``100 * (d_at_spawn - d_min) / d_at_spawn`` - the
    share of THAT episode's own remaining route the policy covered, clipped
    to [0, 100]. It exists because the aggregate over a fleet of maps has to
    be a PERCENTAGE and ``race/eval_progress`` is map units: cannonball's
    d0 is 198,380 and petrus_lite's is 35,637, so a units mean is 85%
    cannonball and a fleet that learns only the long map reads as a fleet
    that generalises. Per episode, not against the pool mean d0, because the
    spawn pool is a distribution and the honest denominator is the distance
    this episode actually had to cover.

    ``n_box_finishes`` counts episodes whose recorded path crosses the
    finish AABB - the env's own win test, not the geodesic <= 150 u proxy
    ``eval_finish_times`` uses. CLAUDE.md's death-dive warning is exactly
    the gap between the two: an agent that falls PAST the finish into
    goal-adjacent airspace scores ~178k of 198,380 on the geodesic and has
    not finished anything.
    """
    rows, pct, n, fin = [], 0.0, 0, 0

    def _close(rs):
        nonlocal pct, n, fin
        a = np.asarray(rs, dtype=np.float64)
        d = field.sample(a[:, 1:4])
        d0 = float(d[0])
        if d0 > 1.0:
            pct += float(min(100.0, max(0.0, 100.0 * (d0 - d.min()) / d0)))
            n += 1
        if goal_box is not None and _seg_hits_box(a[:, 1:4], goal_box):
            fin += 1

    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict) and "map" in row:
                rows = []
            elif isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict) and "end" in row and rows:
                _close(rows)
                rows = []
    return pct, n, fin


def eval_finish_times(traj_path: Path, field):
    """(n_finished, mean_s, best_s) over a recording's episodes. Finished =
    the episode's LAST frame sits at the goal (geodesic distance <= 150u);
    the trailer's end label is cosmetic (record.py infers it from base
    rewards, which lack the race bonus). This is the scoreboard clock:
    start-line greedy eval seconds — training's finish_s is from-SPAWN time
    and mostly measures respawn-curriculum episodes."""
    fins, rows, hdr = [], [], None
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict) and "map" in row:
                rows, hdr = [], row
            elif isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict) and "end" in row and rows:
                d = float(field.sample(
                    np.asarray(rows[-1][1:4], np.float64)[None])[0])
                if d <= 150.0:
                    # seconds from the recording's OWN tick (header
                    # tick_ms / pattern), never a hard-coded 100 Hz
                    fins.append(episode_seconds(
                        hdr, int(row.get("ticks", len(rows))), str(traj_path)))
                rows = []
    if not fins:
        return 0, float("nan"), float("nan")
    return len(fins), float(np.mean(fins)), float(min(fins))


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
    ap.add_argument("--maps", default=None,
                    help="train ONE shared policy on several maps at once: "
                         "a comma-separated list of .bsp paths (or bare map "
                         "names resolved under maps/). --envs is split "
                         "evenly, one core per map, and everything "
                         "map-shaped is per map: goal field, spawn pool, "
                         "respawn reservoir, novelty counts, d0 and the "
                         "reward scale 100/d0 — so finishing ANY map is "
                         "worth 100 whatever its length. The PPO update is "
                         "untouched; it sees one batch whose rows come from "
                         "different maps. With one entry this is exactly "
                         "--map. NOTE --ep-ticks is still one number for the "
                         "whole fleet (12,000 = 120 s, tuned for "
                         "cannonball), so on a map five times shorter most "
                         "of an episode is wasted - a per-slot episode cap "
                         "is the follow-up. ckpt restores")
    ap.add_argument("--heldout-maps", default=None,
                    help="EVAL-ONLY maps: a comma-separated list of .bsp "
                         "paths (or bare names under maps/) the policy is "
                         "evaluated on at every eval and NEVER trained on - "
                         "no env rows, no reservoir, no reward, no gradient "
                         "(surfgym/mapfleet.py HeldoutSlot). The "
                         "generalisation probe: a policy that learned "
                         "surfing makes progress on a map it has never "
                         "seen; a map-memoriser does not. Each held-out map "
                         "gets its own zones/field/caches, its own greedy "
                         "eval (traj_<step>_<tag>.jsonl) and its own "
                         "race/heldout_*.<tag> CSV columns - the FIELD "
                         "metric (geodesic progress in map units, finish "
                         "time, box finishes, %% of the field's start "
                         "distance) plus the honest order-only corridor MAX "
                         "when maps/<stem>.route.npz exists. Never part of "
                         "the aggregates. Needs --reward race; excludes "
                         "--goals/--route/--demo-file/--race-arc. ckpt "
                         "restores")
    ap.add_argument("--heldout-goal-cell", default=None,
                    help="goal-field voxel size for --heldout-maps: one "
                         "value for all, or one per map in --heldout-maps "
                         "order (the --goal-cell rule). Pass the cell each "
                         "map's field was BAKED at (pool_args.py reads it "
                         "off the .goal_<cell>.npz) or it rebakes at "
                         "startup. Default: each map's lidar cell")
    # 2048 envs, not more: at fixed update density, doubling rollout width
    # halves PPO iterations per sample (rew-20 at 52M steps here vs 98M at
    # 8192 envs) and the extra raw throughput doesn't pay for it
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=float, default=100e6)
    # --run-name is an exact alias: torchrun's argparse prefix-matches its
    # own --run-path against a --run token even inside script args, so DDP
    # launches must use the alias (tools/ddp_launch.sh does)
    ap.add_argument("--run", "--run-name",
                    default=time.strftime("fast_%m%d_%H%M"))
    # mixed = exploring starts: platform spawns + mid-air spawns over every
    # surfable ramp face map-wide. Entropy only dithers actions locally; a
    # policy collapsed to one groove never *visits* other states, so its
    # value estimates there stay garbage and it can never rationally detour.
    # Diverse starts break that data loop. Eval stays on the platform pool.
    ap.add_argument("--spawn", choices=["platform", "ramp", "mixed"],
                    default="platform")
    ap.add_argument("--ep-ticks", type=int, default=None)   # 700; ckpt overrides
    ap.add_argument("--ep-secs", type=float, default=None,
                    help="episode cap in SECONDS (converted to --ep-ticks at "
                         "the run's tick); wins over the checkpoint value")
    # ---- --tick-ms: the physics tick length --------------------------------
    # GoldSrc's air-accelerate impulse saturates at 30 u/s PER FRAME
    # (pm.c PM_AirAccelerate), so a strafer's acceleration is proportional to
    # the frame rate: the cannonball WR demo runs 7/8 ms frames (7.63 ms =
    # 131 fps), 31% more air-accelerate steps per second than our 10 ms tick.
    # A non-integer tick is realised as the shortest repeating integer-ms
    # pattern within 0.05 ms (7.63 -> 8,8,7 = 7.667 ms, surfgym.tick), driven
    # into the core per batch step. EVERY per-tick constant keeps its meaning
    # in SECONDS: gamma -> gamma**(tick/10), --time-pen / --speed-coef /
    # --stall-eps / --pitch-rate (and the 10 deg/tick yaw ceiling) scale by
    # tick/10, --stall-secs / --respawn-margin / --goal-kmin/kmax /
    # snap_every / --finish-tref convert seconds -> ticks at the real tick.
    # --act-every is NOT changed (K ticks = K*tick ms per decision; say so).
    # 10.0 = today, byte-identical. ckpt restores; a checkpoint trained at
    # 10 ms resumed with another value is ALLOWED with a loud notice (the
    # warm transfer is the first experiment) and both values land in
    # run.json (tick_ms / tick_ms_ckpt).
    ap.add_argument("--tick-ms", type=float, default=None)   # 10.0; ckpt restores
    # ---- --tick-ms-schedule: the tick as a RAMP ----------------------------
    # A policy has MEMORISED its line against the tick it trained at: the
    # frozen finisher xQR32 finishes 9/9 at 10 ms and 0/9 at 7.63 ms, so a
    # warm resume at the target tick starts from a non-finisher and has
    # nothing left to improve. FROM:TO:STEPS moves the tick LINEARLY IN MS
    # from FROM to TO over STEPS environment steps counted from the step the
    # run launches at, then HOLDS TO. The realised tick stays an integer-ms
    # pattern and is re-derived only when the request has moved more than
    # 0.05 ms (39 times over a 10 -> 7.63 ramp), each change logged once.
    # Every per-second constant the block below converts follows the ramp;
    # the three that CANNOT (they are baked into the core at surf_create and
    # have no C setter) are --ep-ticks and the yaw / pitch deg-per-tick
    # ceilings, which stay frozen at the LAUNCH tick - so the run starts
    # byte-identical to an unscheduled one and the action space keeps its
    # meaning in deg PER TICK, which is what a warm-resumed policy reads.
    ap.add_argument("--tick-ms-schedule", default=None, metavar="FROM:TO:STEPS",
                    help="ramp the physics tick linearly in ms, e.g. "
                         "10:7.63:600e6 (10 ms -> 7.63 ms over the first "
                         "600M env steps, then held). Wins over --tick-ms; "
                         "a checkpoint carries the ramp (origin included) so "
                         "a bare resume CONTINUES it and passing the flag "
                         "again REPLACES it from the resumed step")
    # update density matters as much as throughput: these defaults match SB3's
    # 1-gradient-update-per-4k-samples (64 -> 300M-step sample-efficiency
    # regression when this was 2 epochs x 8 minibatches over 1M-sample rollouts)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=16,
                    help="minibatches per epoch: T*N/M shuffled rows each; "
                         "under --rnn, M groups of N/M whole env SEQUENCES "
                         "(all T decisions, re-run through the GRU) - the "
                         "count keeps its meaning, the shuffled unit is the "
                         "env")
    ap.add_argument("--train-stride", type=int, default=None,  # 1; ckpt restores
                    help="optimize on every S-th decision timestep only "
                         "(GAE still runs on the full chain; offset rotates "
                         "per iteration). Adjacent 30ms samples are near-"
                         "duplicates: stride 3 cuts the update phase "
                         "(~50%% of the iteration) to ~1/3 at equal game-time")
    ap.add_argument("--lr", type=float, default=None)      # 3e-4; ckpt restores
    ap.add_argument("--gamma", type=float, default=None)   # 0.995; ckpt restores
    ap.add_argument("--gae", type=float, default=None)     # 0.95; ckpt restores
    ap.add_argument("--clip", type=float, default=None)    # 0.2; ckpt restores
    ap.add_argument("--ent", type=float, default=None)     # 0.005; ckpt restores
    ap.add_argument("--ent-final", type=float, default=None)
    ap.add_argument("--vf", type=float, default=None)      # 0.5; ckpt restores
    ap.add_argument("--yaw-jitter", type=float, default=8.0)
    ap.add_argument("--drop-frac", type=float, default=0.5,
                    help="--spawn mixed: fraction of the spawn pool that is "
                         "random ramp drops rather than the map's start "
                         "entries. The env resets by uniform pool draw, so "
                         "without this the ~18 start entries are swamped by "
                         "thousands of drops and the start line is never "
                         "trained")
    ap.add_argument("--obs-reward", action="store_true",
                    help="feed the previous decision's reward back as an "
                         "observation. The agent has no absolute position, so "
                         "it cannot compute its own progress; the shaping "
                         "reward IS geodesic progress, so this hands it a "
                         "direct 'am I going the right way' signal it "
                         "otherwise has to infer from depth alone. Carried in "
                         "scalar slot 12 (an absolute-position channel the "
                         "no-GPS policy already ignores), squashed with tanh, "
                         "so obs_dim is unchanged. Changes what the network "
                         "reads: scratch runs only")
    ap.add_argument("--ez-eps", type=float, default=None,      # 0 = off
                    help="ez-greedy temporally-extended exploration: with "
                         "this probability per decision an env commits to ONE "
                         "random action for n ~ zeta(mu) decisions (capped by "
                         "--ez-max). iid per-step noise cancels out over long "
                         "trajectories - a committed burst actually goes "
                         "somewhere. Burst transitions are EXCLUDED from the "
                         "policy loss (they are off-policy), so PPO stays "
                         "sound; their effect reaches the policy through the "
                         "states they discover and the returns of the "
                         "on-policy steps around them. ckpt restores")
    ap.add_argument("--ez-max", type=int, default=None,        # 60 decisions
                    help="cap on ez-greedy burst length, in decisions")
    ap.add_argument("--ez-mu", type=float, default=None,       # 2.0
                    help="zeta exponent for ez-greedy burst length")
    ap.add_argument("--respawn-mode",
                    choices=["uniform", "goex", "florensa", "backward"],
                    default=None,
                    help="reservoir start-state selection over the distance "
                         "bins (any non-uniform mode implies the binned "
                         "machinery): goex = Go-Explore 1/sqrt(chosen+1) "
                         "cell weights (1901.10995); florensa = success band "
                         "(0.1,0.9) with 1/3 of draws reserved for mastered "
                         "bins (1707.05300); backward = Salimans-Chen "
                         "moving window, advances away from the goal at 0.2 "
                         "window success (1812.03381)")
    ap.add_argument("--respawn-bins", type=int, default=None,   # 16
                    help="distance bins for binned/mode respawn sampling")
    ap.add_argument("--spawn-burst", type=int, default=None,    # 0 = off
                    help="Go-Explore post-return exploration: for this many "
                         "decisions after every respawn the env takes random "
                         "actions (held, resampled with prob 1-spawn-burst-p "
                         "per decision), excluded from the PPO update "
                         "exactly like ez-greedy bursts")
    ap.add_argument("--spawn-burst-p", type=float, default=None,  # 0.95
                    help="action repeat probability inside the spawn burst")
    ap.add_argument("--race-shaping", type=float, default=None,  # 1.0
                    help="multiplier on the geodesic potential shaping term "
                         "(and the speed-equiv potential, which shares its "
                         "scale). 0 = sparse reward: only the success bonus, "
                         "fail/time penalties and the intrinsic novelty "
                         "bonus remain. The obs-reward eval feed follows it. "
                         "ckpt restores")
    ap.add_argument("--race-dfloor", type=float, default=None,   # 0 = off
                    help="race: FLOOR the shaping potential at this geodesic "
                         "distance - d_eff = max(d, dfloor), Phi = -d_eff. "
                         "Every state closer than dfloor shares one "
                         "potential, so inside that shell the shaping pays "
                         "zero for approaching AND charges zero for leaving; "
                         "outside it nothing changes. Still potential-based "
                         "and a function of state alone (NOT a running-min "
                         "ratchet). For a distance field whose low-d shell "
                         "reaches into space the player cannot survive - "
                         "cannonball's geodesic believes an 8,700u level "
                         "glide across open air from route vertex 1600 and "
                         "pays ~+2 for the fatal fall that follows - this "
                         "deletes that income and nothing else. The stall "
                         "detector and the respawn stagnant mask keep the "
                         "RAW d. ckpt restores")
    ap.add_argument("--race-latch", type=float, default=None,   # 0 = off
                    help="race: LATCH the shaping term off once an "
                         "episode first reaches this geodesic distance - "
                         "from that tick to the end of that episode the "
                         "potential term contributes exactly 0, in both "
                         "directions. --race-dfloor flattens the "
                         "potential INSIDE the shell but still charges "
                         "the climb back OUT of it, which on cannonball "
                         "is the whole valley (route vertices 1600 -> "
                         "1680 raise d 6,632 -> 14,976, charged -4.02). "
                         "This deletes that charge with no reference "
                         "line. The switch is episode history, so the "
                         "flag is fed to the network as one extra input "
                         "feature (the route block, 1 wide, last "
                         "column); a warm resume zero-pads it and stays "
                         "function-identical at step 0. The stall "
                         "detector and the respawn stagnant mask keep "
                         "the RAW d. ckpt restores")
    ap.add_argument("--race-latch-frac", type=float, default=None,  # 0 = off
                    help="--race-latch expressed as a FRACTION of each "
                         "map's own start geodesic d0, which is what a "
                         "multi-map run needs: 6,996 u is 3.53%% of "
                         "cannonball's d0 (198,380 u) and 19.6%% of "
                         "petrus_lite's (35,637 u), so an absolute latch "
                         "means two completely different things on two "
                         "maps. Each slot gets d_latch = frac * its own "
                         "rf_d0. Mutually exclusive with --race-latch, "
                         "which stays absolute and is single-map only. "
                         "ckpt restores")
    ap.add_argument("--race-ng", type=int, default=0, choices=(0, 1, 2, 3),
                    help="race: Ng-conformant shaping (question 4). The "
                         "stock potential difference does not telescope "
                         "under gamma<1 (per-decision leak ~(1-gamma^k)*"
                         "banked, ~9x time_pen deep in a map) and death "
                         "keeps all collected shaping income for free. "
                         "1 = strict: per-call tax (1-gamma^k)*Phi plus a "
                         "terminal charge -Phi on death AND finish, so "
                         "shaping nets zero over every episode and only "
                         "success_bonus/time_pen set the objective. "
                         "2 = bond: death forfeits the bank, finishing "
                         "keeps it. 3 = tax only, terminals stock. "
                         "Truncation exempt (bootstrapped). ckpt restores. "
                         "MEASURED: 1 from scratch collapses to fast "
                         "suicide (round 27, xNGS)")
    ap.add_argument("--death-charge", type=float, default=None,
                    help="race: charge kappa*Phi(last pre-death state) at "
                         "death, ON TOP of the stock scheme (no per-step "
                         "tax, no goal charge). Doomed depth still nets "
                         "(1-kappa)*Phi so the curriculum survives; a "
                         "deliberate deep dive abandons kappa*Phi. 0 = "
                         "off, byte-identical. Mutually exclusive with "
                         "--race-ng terminal modes. ckpt restores")
    ap.add_argument("--demo-file", default=None,
                    help="Salimans-Chen backward curriculum (1812.03381): "
                         "path to a time-ordered STATE_DTYPE .npy demo spine "
                         "(last row = just before the goal). Replaces the "
                         "reservoir share of the spawn pool with draws from "
                         "a window of demo states nearest the goal; the "
                         "window slides earlier at 20 percent finish rate")
    ap.add_argument("--demo-window", type=int, default=None,   # 10 states
                    help="demo window width D, in demo states")
    ap.add_argument("--demo-rate", type=float, default=None,   # 0.2
                    help="window finish rate that advances the curriculum")
    ap.add_argument("--demo-min-ep", type=float, default=None,  # 50
                    help="episodes required in-window before moving")
    # ---- expert iteration: distil the planner's line (surfgym/bc.py) ----
    ap.add_argument("--bc-file", default=None,
                    help="behaviour-cloning rows from tools/plan_to_bc.py "
                         "(planner states + the six head indices it "
                         "committed). Adds a weighted cross-entropy of the "
                         "factored action heads against those indices to "
                         "every PPO minibatch step; the rows' depth images "
                         "are rendered per minibatch from the stored states "
                         "with the trainer's own lidar. Never restored from "
                         "a checkpoint: one round, one file")
    ap.add_argument("--bc-coef", type=float, default=0.5,
                    help="--bc-file: aux loss weight at the start of the run")
    ap.add_argument("--bc-coef-final", type=float, default=0.0,
                    help="--bc-file: weight it decays linearly to")
    ap.add_argument("--bc-steps", type=float, default=None,
                    help="--bc-file: env steps (from the resume point) over "
                         "which the weight decays; default = the run's "
                         "remaining budget")
    ap.add_argument("--bc-batch", type=int, default=2048,
                    help="--bc-file: planner rows scored per PPO minibatch "
                         "step (a constant: the BC step is compiled)")
    ap.add_argument("--bc-target", choices=("argmax", "dist"),
                    default="argmax",
                    help="--bc-file: WHAT the cloning loss regresses onto. "
                         "'argmax' (default, byte-identical) is the NLL of "
                         "the planner's own action index - Expert "
                         "Iteration's CAT. 'dist' is the per-head cross-"
                         "entropy against the search's stored distribution "
                         "over first decisions (`probs`, a version-2 BC "
                         "file) - ExIt's TPT, which that paper measured 50 "
                         "+/- 13 Elo stronger at INDISTINGUISHABLE top-1 "
                         "accuracy. On a version-1 file (or any row where "
                         "only the winner survived the search) the "
                         "distribution IS the one-hot and 'dist' is 'argmax' "
                         "exactly")
    ap.add_argument("--bc-value-coef", type=float, default=0.0,
                    help="--bc-file: weight of a value term on the planner "
                         "line's own realised return-to-go (AlphaZero's "
                         "`(z - v)^2`, stored as `zret`/`zmask` by "
                         "tools/plan_to_bc.py). The BC step's loss becomes "
                         "`nll + C * 0.5*(V(s) - z)^2`, weight-averaged over "
                         "the rows whose return is COMPLETE, and the sum "
                         "then rides --bc-coef like the policy half (MuZero "
                         "weights value 0.25 against 1.0 for policy). 0 = "
                         "off and byte-identical; the value is read with the "
                         "privileged block under --priv-critic")
    ap.add_argument("--respawn-killsafe", type=int, default=None,  # 0 = off
                    help="bin reservoir states on the KILL-MASKED goal field "
                         "(goalk cache): states inside fail/teleport volumes "
                         "read invalid there and are excluded from mode-based "
                         "start selection. The goal field is blind to kill "
                         "volumes, so without this a frontier-seeking mode "
                         "concentrates spawns onto fail floors whose voxels "
                         "carry small d. Shaping is unchanged")
    ap.add_argument("--yaw-adaptive", action="store_true",
                    help="interpret the yaw action as a MULTIPLE of the "
                         "analytic optimal-strafe turn rate atan(30/|v_h|) "
                         "instead of a fixed deg/tick ladder, so 'strafe "
                         "optimally' is the constant action k=+-1 at every "
                         "speed. The gain window is +-arcsin(30/|v|) (~0.5 "
                         "deg at 3000 u/s) and the fixed bins cannot resolve "
                         "it. Changes action semantics: scratch runs only")
    ap.add_argument("--yaw-blend", type=float, default=1.0,
                    help="exponential blending of the applied per-tick yaw delta: "
                         "yd = b*cmd + (1-b)*previous applied (env.c). 1.0 = off, "
                         "bit-identical; 0.5 halves the turn-rate dither. ckpt restores")
    ap.add_argument("--side-hold", type=int, default=0,
                    help="minimum hold of the side key (A/D) in physics ticks, enforced in "
                         "env.c (a change within N ticks of the last is ignored). 0 = off, "
                         "bit-identical; 40 = 0.3 s at 131 Hz. ckpt restores")
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
    # --pitch-fixed is the STRONGER sibling of --fix-pitch. --fix-pitch only
    # sets pitch_rate_max_deg = 0, which freezes the gaze at whatever value
    # the STATE happens to carry - the platform pool's -10, but also whatever
    # a reservoir respawn restored, so one "fixed gaze" run can hold several
    # different angles at once. This one PINS the value: the pitch column of
    # the states is written immediately before every lidar render (training
    # rollout, truncation bootstrap, in-trainer eval, record_ckpt.py), so
    # every rendered frame is aimed at exactly this angle whatever the state
    # said. pitch_rate is forced to 0 as well, because the C side applies the
    # pitch head's delta on EVERY tick of a decision (src/env.c:543-558) and
    # would otherwise walk the pinned value away between two renders - and
    # obs slots 9/11 (pitch/90, last delta/rate) are written by the C step,
    # so they agree with the pinned render only when nothing moves them.
    # The pitch head stays in the action space and is simply inert.
    ap.add_argument("--pitch-fixed", type=float, default=None,
                    help="hold the view pitch at this angle (deg, + = up) "
                         "every tick: the lidar is always aimed here and the "
                         "pitch action is ignored. Pitch aims the LIDAR "
                         "only, never movement (env.c passes 0.0 to pm_tick), "
                         "so this changes what the policy SEES and nothing "
                         "about the physics. Default off = today")
    # decisions every K physics ticks (100Hz physics / K): calmer camera,
    # human-scale reaction granularity, and ~K x cheaper policy+update per
    # game-second. Held yaw/pitch deltas keep applying, so turn RATE is
    # unchanged — only decision frequency drops.
    ap.add_argument("--act-every", type=int, default=None)   # 3; ckpt restores
    ap.add_argument("--pitch-rate", type=float, default=None,
                    help="max view-pitch delta per tick, deg (core default 10; "
                         "4 makes gaze deliberate instead of whippy)")
    # ---- --chunk: behavior codes (docs/action-chunks-design.md) ------------
    ap.add_argument("--chunk", type=int, default=None,   # 0 = off; ckpt restores
                    help="temporally-abstract actions: ONE policy decision "
                         "picks one of --codes behavior codes, and a LEARNABLE "
                         "decoder inside the policy expands it into H "
                         "consecutive per-decision action distributions (H = "
                         "this value). The engine still gets a fresh action "
                         "every --act-every ticks, so the CONTROL rate is "
                         "unchanged and only the DELIBERATION rate drops H x "
                         "(H x fewer trunk forwards and lidar renders). "
                         "--n-steps then counts CHUNKS, and the GAE discount "
                         "becomes gamma**(act_every*H). Changes what one "
                         "decision means: scratch runs only. ckpt restores")
    ap.add_argument("--codes", type=int, default=None,   # KCODES; ckpt restores
                    help="--chunk: number of behavior codes K. VQ-BeT's total "
                         "mode count is 64-256 and BeT's kitchen k-means uses "
                         "64; below that codes stop covering the repertoire, "
                         "above it they go dead (a dead code is a logit that "
                         "can never learn anything)")
    ap.add_argument("--dec-ent", type=float, default=None,   # 5e-4; ckpt restores
                    help="--chunk: entropy coefficient for the DECODER's "
                         "per-decision categoricals, separate from --ent "
                         "(which now weights the code-level entropy). Two "
                         "knobs, opposite jobs: --ent keeps CODE CHOICE "
                         "exploratory, --dec-ent lets a code crystallize into "
                         "a distinct behavior. Keep it small — a large value "
                         "makes every code decode to the same near-uniform "
                         "mush, which is code collapse by another route")
    ap.add_argument("--codebook", default=None,
                    help="--chunk: OPTIONAL warm INIT for the decoder from an "
                         "npz built by tools/build_action_codebook.py "
                         "(codebook (K,H,6) int8 of action indices). Adds "
                         "--codebook-bias to the logit of each stored index, "
                         "so code k starts out biased toward the fitted "
                         "behavior instead of uniform noise. The decoder is "
                         "trainable either way and lives in the state_dict, "
                         "so this file is NOT needed to resume or record; "
                         "off by default (random init)")
    ap.add_argument("--codebook-bias", type=float, default=None,   # 3.0
                    help="--codebook: logit added to each fitted action index "
                         "(3.0 => ~20x the odds of its head's other bins; the "
                         "decoder can still move off it in one update)")
    ap.add_argument("--trunk", choices=("plain", "resnet"), default=None,
                    help="image encoder: plain (the historical 3-conv "
                         "stack, default) or resnet (residual, 2.79M "
                         "params). ckpt overrides")
    ap.add_argument("--rnn", choices=("none", "gru"), default=None,
                    help="recurrent policy: one GRU layer between the fused "
                         "trunk output and the pi/vf towers, its state "
                         "carried across decisions and zeroed at every "
                         "episode start (autoreset, reservoir respawn, "
                         "eval). none (default) is the feed-forward policy, "
                         "byte-identical. ckpt overrides")
    ap.add_argument("--rnn-size", type=int, default=None,   # 256
                    help="--rnn: GRU hidden width; ckpt overrides")
    ap.add_argument("--emb", type=int, default=None)      # 512; ckpt overrides
    ap.add_argument("--hidden", type=int, default=None)   # 448; ckpt overrides
    # the two capacity knobs that are not emb/hidden. Both change WHICH
    # tensors exist, so a checkpoint carries exactly one value of each and a
    # mismatch is refused (same treatment as --trunk) rather than left to
    # load_state_dict three screens later.
    ap.add_argument("--tower-depth", type=int, default=None,   # 2
                    help="Linear+Tanh layers in EACH of the pi/vf towers "
                         "(2 = today). The first layer is the only one whose "
                         "input width follows the observation, so --route / "
                         "--rnn zero-pad warm starts work at any depth. "
                         "ckpt overrides")
    ap.add_argument("--conv-mult", type=int, default=None,     # 1
                    help="channel multiplier for the plain trunk's three "
                         "convs: 16/32/64 -> 16M/32M/64M, and the Linear "
                         "after the AdaptiveAvgPool follows (1 = today). "
                         "Not available with --trunk resnet. ckpt overrides")
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
    # depth alone cannot tell floor from ramp from wall — they are the same
    # pixel. The mask is the hit surface's |n_z| (1 flat, 0 vertical, the
    # rideable band ~0.3-0.7; physics walks at >= 0.7, src/pm.c), baked per
    # voxel from the map mesh (surfgym.surfmask). Doubles the image slice
    # and the conv's input channels, so a ckpt cannot switch it mid-run.
    ap.add_argument("--goal-cell", type=str, default=None,
                    help="ONE value, or a comma-separated list aligned with "
                         "--maps (a per-map gate verdict, e.g. "
                         "48,48,48,32,48 - 21 of the 110-map pool tunnel at "
                         "48 and must keep 32). "
                         "voxel size for the SHAPING field only, decoupled "
                         "from --lidar-cell (which stays perception). "
                         "Voxels scale as cell^-3 so this is the cheapest "
                         "lever on bake time and RAM. Measured on "
                         "cannonball: 48 is faithful (d0 +0.3%%, monotone "
                         "95.5%% vs 95.6%%), 64 TUNNELS through thin floors "
                         "and halves d0 - so raising it needs the per-map "
                         "d0/monotonicity gate, not just a reachability "
                         "check, which cell 64 passes while being nonsense")
    ap.add_argument("--surf-mask", type=int, default=None,
                    help="vision channels. 0 = depth only "
                    "(default). 1 = depth + the hit surface's |n_z| "
                    "as a second channel. 2 = the |n_z| MASK ALONE, "
                    "no depth: in_ch stays 1 and the march is "
                    "unchanged (it computes depth to find the hit "
                    "either way), so 2 costs the same as 0 and asks "
                    "whether surfability alone is a sufficient "
                    "percept")
    # the shipped camera is equiangular (write_lidar's convention): a fixed
    # angle per pixel, so straight world edges bow across the image and the
    # conv sees a ramp's slope change with where it sits in frame. --pinhole
    # is the rectilinear alternative — same fov, same centre ray, uniform
    # spacing on the tangent plane. Pixel values change, so this is a fresh
    # run, not a warm start (the ckpt's setting is restored on resume).
    ap.add_argument("--pinhole", type=int, default=None,
                    choices=(0, 1),                # 0; ckpt restores
                    help="rectilinear camera instead of the equiangular one")
    # --normals: the hit surface's full unit normal as three more channels
    # (depth, nx, ny, nz), in the player's ego frame - rotated by the view
    # yaw only, x forward / y left / z up, flipped to face the ray, 0 where
    # nothing was hit (surfgym/vision.py). Baked per voxel from the map
    # mesh like --surf-mask (surfgym.surfmask.build_surfnormal; |n_z| is
    # the fourth channel's magnitude, so the two are exclusive). in_ch
    # becomes 4, so a ckpt cannot switch it mid-run.
    ap.add_argument("--normals", type=int, default=None,
                    choices=(0, 1),                # 0; ckpt restores
                    help="1 = depth + the hit surface's ego-frame unit "
                         "normal (nx forward, ny left, nz up) as channels "
                         "1..3; 0 where the ray hit nothing")
    # the camera's field of view, degrees. 120 x 90 is write_lidar's
    # convention and what every checkpoint so far was trained on; the pixel
    # grid (yoff/poff) follows, and so do the goal-ball wrapper and the POV
    # renderer, which read them out of run.json. Pixel VALUES change, not
    # shapes, so like --pinhole this is a fresh run, not a warm start.
    ap.add_argument("--lidar-hfov", type=float, default=None,   # 120; ckpt restores
                    help="lidar horizontal fov, degrees (default 120)")
    ap.add_argument("--lidar-vfov", type=float, default=None,   # 90; ckpt restores
                    help="lidar vertical fov, degrees (default 90)")
    ap.add_argument("--frame-stack", type=int, default=None,
                    choices=(0, 1, 2, 3, 4, 5),    # 0 = off; ckpt restores
                    help="feed the conv K depth frames as K channels: the "
                         "current render plus past DECISIONS at strides "
                         f"{list(STACK_STRIDES)}[:K-1] (at --act-every 3 a "
                         "decision is 30ms, so K=4 spans 120ms). Depth alone "
                         "cannot show relative motion")
    # --- lookahead route geometry (surfgym/route.py) ------------------------
    # The observation every superhuman racer has and this project did not:
    # Sophy's 60 course points spanning ~6s at current velocity (ablation
    # +2.64s on a 114s lap, the largest in that paper), Fuchs' 10 curvature
    # samples, Linesight's 40 ego-frame virtual checkpoints, Swift's gate
    # corners. Build the file with tools/build_route.py.
    ap.add_argument("--route", default=None,
                    help="reference route .npy/.npz: adds an ego-frame "
                         "lookahead fan to the observation (ckpt restores)")
    ap.add_argument("--route-span", type=float, default=None,   # 6.0 s
                    help="furthest lookahead horizon in SECONDS; the fan is "
                         "scaled to it (default 6, Sophy's span)")
    ap.add_argument("--route-points", type=int, default=None,   # 8
                    help="how many lookahead horizons (default 8)")
    ap.add_argument("--route-critic-only", type=int, default=None,  # 0 = off
                    help="1 = feed the fan to the VALUE tower only "
                         "(asymmetric critic, Vasco RLC 2024); the actor "
                         "stays honest-perception")
    # --- --priv-critic: the full asymmetric actor-critic --------------------
    # Pinto et al. 2017 ("Asymmetric Actor Critic for Image-Based Robot
    # Learning"), the arrangement OpenAI's Dactyl and the Isaac Gym
    # locomotion stack both use: the critic reads the SIMULATOR state at
    # training time, the actor reads only what a deployment would have.
    # --route-critic-only above is the same idea applied to one existing
    # observation block; this is the state itself.
    ap.add_argument("--priv-critic", type=int, default=None,   # 0 = off
                    help="1 = give the CRITIC a privileged state block the "
                         "actor never sees (asymmetric actor-critic, Pinto "
                         "et al. 2017). TEN columns per env, all of them "
                         "state the rollout already has in hand: "
                         "(0-2) position, (origin - map_centre)/map_scale "
                         "with map_scale the largest half-extent of the BSP "
                         "bounds, so the three share one scale; "
                         "(3-5) the world velocity vector / 4000; "
                         "(6) d/d0, the SAME geodesic distance RaceReward "
                         "shapes on over the map's start geodesic; "
                         "(7) arc/route_length under --race-arc, else a "
                         "constant 0; (8) tick/ep_ticks; (9) the "
                         "--race-latch flag as 0/1. Nothing is clipped. "
                         "They enter through a 2x128 LayerNorm+Tanh MLP "
                         "concatenated to the value tower's output right "
                         "before value_head; the two action heads cannot "
                         "reach them, so the deployed policy is identical "
                         "in form to the control's. Watch "
                         "train/explained_var - that is the direct effect. "
                         "0 = today, byte for byte")
    ap.add_argument("--priv-hidden", type=int, default=None,   # 128
                    help="width of the --priv-critic MLP (default 128). "
                         "Changes the policy's SHAPE, so it is recorded in "
                         "the checkpoint config and mirrored by "
                         "tools/record_ckpt.py")
    # --- handcrafted scalar-side blocks (surfgym/obsaux.py) -----------------
    # Both ride the SAME trailing block the route fan and --race-latch use,
    # which is where widen_for_route zero-pads a checkpoint that has never
    # seen them. Default off, and off is byte-identical to no flag at all.
    ap.add_argument("--act-hist", type=int, default=None,   # 0 = off; ckpt restores
                    help="append the agent's last K DECISIONS to the scalar "
                         "observation, most recent first: 6 numbers each "
                         "(yaw delta, pitch delta, forward, side, jump, duck) "
                         "all scale-free in [-1,1]. A memoryless policy "
                         "cannot see the phase it is in, and air-strafing "
                         "needs the strafe key and the yaw change "
                         "phase-locked. Zeroed at every episode start")
    ap.add_argument("--obs-compass", type=int, default=None, choices=(0, 1),
                    help="append 5 columns off the shaping distance field: "
                         "its DOWNHILL direction as an ego-frame unit vector "
                         "(fwd, left, up), d/d0, and |grad| clipped to [0,1]. "
                         "Zeros on the field's unreachable sentinel. Follows "
                         "whichever field RaceReward shapes on, so a "
                         "--goal-reward euclid/geo run points at the GOAL")
    # --- goal conditioning (surfgym/goalsys.py, research-plan-goalcond.md)
    ap.add_argument("--goals", type=int, default=None, choices=(0, 1),
                    help="1 = per-env sphere goals from the agent's own "
                         "reached states (reservoir goal harvest) + the "
                         "lookahead fan on a per-env line; entering the "
                         "sphere ends the episode with the success bonus "
                         "(ckpt restores)")
    ap.add_argument("--goal-radius", type=float, default=192.0)
    ap.add_argument("--goal-kmin", type=float, default=1.0,
                    help="goal horizon band, seconds ahead of the start")
    ap.add_argument("--goal-kmax", type=float, default=5.0)
    ap.add_argument("--goal-kcap", type=float, default=60.0)
    ap.add_argument("--goal-air-frac", type=float, default=0.0,
                    help="share of starts given a random reachable-AIR "
                         "goal (chord line) instead of a reached state")
    ap.add_argument("--goal-holdout", default=None,
                    help="lo,hi geodesic FRACTIONS of d0 excluded from "
                         "training goals (the G3 held-out sector)")
    ap.add_argument("--goal-curriculum", type=int, default=0,
                    help="1 = widen kmax by the 10-90%% success-band rule")
    ap.add_argument("--goal-obs", default=None,
                    choices=("fan", "ball", "both"),
                    help="how the goal is SHOWN: fan = lookahead fan on "
                         "the per-env line (27 scalars); ball = the goal "
                         "sphere as a second depth channel with an "
                         "off-screen border marker (surfgym/goalball.py); "
                         "both. ckpt restores; default fan")
    ap.add_argument("--goal-route", default=None,
                    help="--goals: a map line .npz (e.g. the goal-completed "
                         "self-line); ROUTE-DEPTH goals are placed on it "
                         "delta ahead of the start (delta from the "
                         "curriculum), the finish being delta -> end")
    ap.add_argument("--goal-route-frac", type=float, default=0.7,
                    help="share of starts given a route-depth goal when "
                         "--goal-route is set (the rest: reached-state / air)")
    ap.add_argument("--goal-frontier", type=int, default=0, choices=(0, 1),
                    help="1 = ONE map frontier F: route goals in the band "
                         "behind F, F += step when frontier success >= "
                         "rate, F = 1 -> the finish itself (needs "
                         "--goal-route)")
    ap.add_argument("--goal-fixed", type=int, default=0, choices=(0, 1),
                    help="1 = a FIXED goal set generated once: route points "
                         "every --goal-fixed-spacing u of arc + the finish, "
                         "and --goal-fixed-air reachable air points near the "
                         "route; sampled per rollout (route goals ahead of "
                         "the start), no reached-state goals")
    ap.add_argument("--goal-fixed-spacing", type=float, default=2000.0)
    ap.add_argument("--goal-fixed-air", type=int, default=100)
    ap.add_argument("--goal-fixed-decay", type=float, default=0.0,
                    help="fixed set: training weight exp(-i/N) on the i-th "
                         "goal ahead of the start (route by arc order, air by "
                         "distance rank); 0 = uniform. Eval stays uniform.")
    ap.add_argument("--goal-route-uniform", type=int, default=0, choices=(0, 1),
                    help="1 = route goals at EVERY distance ahead of each "
                         "start, uniformly, the finish included: a "
                         "stationary task distribution (no frontier)")
    ap.add_argument("--goal-front-start", type=float, default=0.05)
    ap.add_argument("--goal-front-band", type=float, default=0.05)
    ap.add_argument("--goal-front-step", type=float, default=0.10)
    ap.add_argument("--goal-front-rate", type=float, default=0.30)
    ap.add_argument("--goal-front-min-ep", type=int, default=300)
    ap.add_argument("--goal-euclid-scale", type=float, default=0.005,
                    help="--goal-reward euclid: reward per unit of Euclidean "
                         "distance-to-goal reduced (0.005 -> a 10k u approach "
                         "pays 50, the size of the bonus)")
    ap.add_argument("--goal-reward", default=None,
                    choices=("sparse", "arc", "euclid", "geo"),
                    help="--goals: sparse = success bonus + time penalty "
                         "(run with --race-shaping 0); arc = signed arc "
                         "progress along each env's OWN goal line "
                         "(surfgym/goalarc.py, corridor-frozen like "
                         "--race-arc; needs --race-shaping > 0) + the "
                         "bonus. ckpt restores; default sparse")
    ap.add_argument("--goal-views", type=int, default=4, choices=(1, 4),
                    help="--goal-obs ball: 4 = front/back/left/right ball "
                         "views as 4 channels (the goal is never out of "
                         "sight for a memoryless policy); 1 = front view "
                         "with an off-screen border marker")
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
    ap.add_argument("--eval-eps", type=int, default=None,
                    help="episodes per eval recording (default 3 for race, "
                         "5 otherwise; ckpt restores). race/eval_progress "
                         "noise scales as 1/sqrt of this")
    ap.add_argument("--eval-greedy-only", action="store_true",
                    help="skip the stochastic eval recording (A/B arms: no "
                         "verdict reads it, and it taxes the surviving, "
                         "long-episode policies the most)")
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
    ap.add_argument("--finish-k", type=float, default=None,       # 0 = off
                    help="race: time-scaled finish bonus — "
                         "+= k * max(0, tref - T_episode_seconds) on the "
                         "finish tick. Per-second time pressure paid ONLY "
                         "to finishers: no suicide channel, unlike raising "
                         "--time-pen. ckpt restores")
    ap.add_argument("--finish-tref", type=float, default=None,    # 120 s
                    help="race: reference time for --finish-k; keep above "
                         "typical episode finish times so the >=0 clamp "
                         "never bites the gradient. ckpt restores")
    ap.add_argument("--reward-per-decision", action="store_true",
                    help="race: evaluate the Python reward once per decision "
                         "(K ticks) instead of every tick — the potential "
                         "shaping telescopes so the sums are identical; cuts "
                         "the reward_py phase ~K x. Stall/finish bookkeeping "
                         "coarsens to one decision (30ms at K=3)")
    ap.add_argument("--reset-int-counts", action="store_true",
                    help="race: discard the checkpointed novelty count "
                         "table on resume — re-arms count-based curiosity "
                         "on a converged policy (the whole beaten path "
                         "re-pays first-visit novelty; that IS the point)")
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
    ap.add_argument("--stall-eps", type=float, default=None,      # 32
                    help="race: how much a SINGLE decision must improve the "
                         "episode's best distance to re-arm the stall timer. "
                         "_best is a running MINIMUM updated every call, so "
                         "this is per-CALL and not a budget spread over the "
                         "window - which means it scales with --act-every. "
                         "Measured on a real petrus flight: 23.8u mean "
                         "improvement and 13.7%% of calls clearing 32u at "
                         "act_every 3, but a 25.0u PEAK at act_every 1, "
                         "where the default would never re-arm and the "
                         "detector would kill legitimate flight. Raise it "
                         "with the decision rate; lower it if real flight "
                         "gets killed. 32 = today")
    ap.add_argument("--max-step", type=float, default=None,       # 100
                    help="race: per-TICK teleport clip on the shaping delta, "
                         "map units (100 = today). A legal tick moves <= "
                         "~35u at sv_maxvelocity 4000, so anything larger is "
                         "a relocation and must not cash shaping; the clip "
                         "the reward applies is this times the call cadence "
                         "(--reward-per-decision widens it to act_every*chunk "
                         "ticks). It also sets --race-arc's re-anchor "
                         "threshold, which is the same 'one decision of legal "
                         "motion' quantity")
    # ---- action masks (the air keys) ------------------------------------
    # runs/research/wr_demo/wr_vs_ours.md sections 5-6. All three are OFF by
    # default and the off path is byte-identical to the trainer without them:
    # no config key is written, no buffer is allocated, no kernel runs.
    ap.add_argument("--mask-forward-air", action="store_true",
                    help="airborne decisions cannot press W or S: the "
                         "forward/back head is masked to -inf on both "
                         "non-zero moves, leaving 'none' as its only legal "
                         "value while off the ground. In GoldSrc air "
                         "movement wishdir is the NORMALISED forward+side "
                         "sum and PM_AirAccelerate caps the gain at 30 u/s "
                         "per frame along it, so W with a strafe key swings "
                         "wishdir to 45 deg off perpendicular and gives up "
                         "most of the strafe. The WR holds W/S on 0%% of its "
                         "airborne frames, ours on 11.4%%. On the GROUND the "
                         "head is free (walking off the platform needs it)")
    ap.add_argument("--jump-cooldown", type=int, default=0,
                    help="after a jump PRESS, mask the jump head for this "
                         "many DECISIONS (0 = off, today). A per-env counter "
                         "re-arms to N on a press, counts down otherwise and "
                         "is cleared at every episode start. In the air jump "
                         "is a no-op except for bhop timing at landing, and "
                         "our agent presses it 283 times in a run the WR "
                         "finishes with 0")
    ap.add_argument("--duck-air-mask", action="store_true",
                    help="mask duck (IN_DUCK) while airborne. Duck shifts "
                         "the hull 18u and changes the view height at every "
                         "ramp contact; the WR presses it 0 times in 68 s "
                         "and ours 176. NOTE the caveat: this is a BLANKET "
                         "air mask, so it also forbids the ducked-hull "
                         "clearances a human would use on maps that need "
                         "them - on cannonball the WR needs none")
    # ---- --yaw-cond: the autoregressive side key --------------------------
    # docs/research-litsurvey-temporal.md proposal #3. OFF by default and the
    # off path is byte-identical to the trainer without it: no module, no
    # state_dict key, no config key, no RNG draw, no kernel.
    ap.add_argument("--yaw-cond", action="store_true",
                    help="make the side-key (A/D) head AUTOREGRESSIVE on the "
                         "yaw bin: log p(a) = log p(yaw) + log p(side | yaw) "
                         "+ the other four heads, with p(side | yaw) an "
                         "additive (15 x 3) table on the side logits, "
                         "gathered by the yaw bin this decision uses. Air "
                         "strafing needs the two to agree and a FACTORED "
                         "policy cannot express that: at the symmetric point "
                         "each head's gradient toward committing is "
                         "proportional to (2 p_other - 1), a saddle. The "
                         "table starts at ZERO, so a warm resume is "
                         "function-identical to the checkpoint at step 0 "
                         "(widen_for_yawcond) and the log-prob stays exact, "
                         "so PPO's ratio is unchanged in form. Not "
                         "implemented for --chunk")
    ap.add_argument("--view-continuous", action="store_true",
                    help="CONTINUOUS yaw and pitch: heads 0 and 1 become "
                         "squashed Gaussians (pre-tanh z ~ N(mu(s), sigma), "
                         "sigma a state-independent parameter per head, "
                         "z scored by PPO as the action) and the core "
                         "receives a float view command per env in place "
                         "of the bins: yaw K = 20*sign(u)*(exp(a|u|)-1)/"
                         "(exp(a)-1) with u = tanh(z) and a = 2 ln 19 "
                         "(u=+-0.5 -> K=+-1, the analytic strafe optimum; "
                         "u=+-1 -> K=+-20, the old outermost bin), pitch = "
                         "tanh(z) * pitch rate. Bins are separate classes "
                         "to a softmax; the record's turn rate falls "
                         "between two of ours 42%% of the time. Restored "
                         "from a checkpoint that carries it; a DISCRETE "
                         "checkpoint needs tools/transplant_view.py first. "
                         "Excludes --yaw-cond, --chunk, --act-hist, "
                         "--frame-stack, --rnn, --ez-eps, --spawn-burst")
    ap.add_argument("--view-absolute", choices=["velocity", "world"],
                    default=None,
                    help="with --view-continuous: the view row is an "
                         "ABSOLUTE TARGET (yaw target deg, pitch target "
                         "deg) that the core approaches at the per-tick "
                         "rate ceilings, EVERY tick, instead of a per-tick "
                         "delta. velocity: one Gaussian per head, the yaw "
                         "target is an offset from the heading of the "
                         "horizontal velocity (off = 180*sign(u)*(exp(b|u|)"
                         "-1)/(exp(b)-1), u = tanh z, b = 2 ln 17: u=+-0.5 "
                         "-> +-10 deg, u=+-1 -> +-180; below 100 u/s the "
                         "base is the current yaw), so 'look along v' - "
                         "the strafe optimum - is the zero action and the "
                         "view tracks a turning velocity between "
                         "decisions. world: two Gaussians (c, s) = tanh z, "
                         "yaw target = atan2(s, c) - the cos/sin reading "
                         "of the +-180 seam; the norm is ignored. Pitch "
                         "target = -20 + 50 tanh z in both (the core's "
                         "[-70, 30]). Saved in the checkpoint config "
                         "(view_absolute) and restored on resume. Not "
                         "implemented with --bc-file, the planner, "
                         "plan_to_bc or expert_dagger.")
    ap.add_argument("--eval-stall", type=int, default=0,
                    help="1 = apply the TRAINING stall rule to eval episodes "
                         "too (same --stall-secs window, same 32u threshold, "
                         "same distance field, same per-call cadence) and "
                         "end the episode as a fail. 0 (default) is today: "
                         "nothing stops an eval episode, so a crawling policy "
                         "burns the whole --ep-ticks budget and the platform "
                         "eval does not reflect training conditions "
                         "(CLAUDE.md: 'Evals do NOT stall-kill. Training "
                         "does.')")
    ap.add_argument("--ret-norm", type=int, default=None,   # 0 = today
                    help="1 = PopArt-lite return normalization: the critic "
                         "regresses NORMALIZED returns (running debiased EMA "
                         "mean/std of the discounted returns) and every V(s) "
                         "read outside the value loss - GAE, the truncation "
                         "bootstrap - is de-normalized back into reward "
                         "units. Advantage normalization is unchanged. "
                         "ckpt restores")
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
    ap.add_argument("--int-view", type=int, default=None,
                    help="yaw sectors in the novelty count key (0 = off; "
                         "8 = 45-degree sectors). Position-only counts are "
                         "blind to gaze: same voxel looking left vs right "
                         "is a different observation. ckpt restores")
    ap.add_argument("--speed-equiv", type=float, default=None,  # 0 = off
                    help="fold speed into the shaping potential: d_eff = "
                         "d - beta*h_speed. Potential-based (loops net 0, "
                         "optimum unchanged), unlike --speed-coef which "
                         "pays the speed level per tick. Includes the "
                         "on-death credit refund. ckpt restores")
    ap.add_argument("--int-speed", type=int, default=None,      # 0 = off
                    help="speed buckets in the novelty count key (walls "
                         "are speed-gated: a known place at a new speed "
                         "is a new state). ckpt restores")
    ap.add_argument("--rnd-coef", type=float, default=None,   # 0 = off
                    help="Random Network Distillation bonus per decision, "
                         "on the scalar obs (continuous novelty over "
                         "position x velocity x gaze; RMS-normalized, "
                         "non-episodic; ckpt restores)")
    ap.add_argument("--respawn-min-speed", type=float, default=None,
                    help="never snapshot a state slower than this (u/s) "
                         "into the respawn reservoir: the deep bins fill "
                         "with stalled arrivals otherwise. 0 = off; ckpt "
                         "restores")
    ap.add_argument("--respawn-binned", type=int, default=None,
                    choices=(0, 1),                # S1; 0; ckpt restores
                    help="sample respawns uniformly over goal-distance bins "
                         "instead of uniformly over states (uniform-over-"
                         "states mirrors visitation: the mastered early "
                         "track is over-trained, the frontier starved)")
    # --- TailRL (arXiv 2609.02987): tail-likelihood advantage reweighting ----
    # J = integral_0^1 log p(x, tau) dtau instead of the mean return, which
    # decomposes into a harmonically weighted mixture of best-of-k
    # objectives (their Theorem 1). The algorithm is one line: reweight each
    # rollout's advantage by w = integral_0^r dtau / p(tau), with p the
    # empirical survival function over the N rollouts sharing an input.
    # python/surfgym/tailrl.py carries the derivation and every place this
    # transplant departs from the paper (GAE advantages instead of their
    # critic-free score, mean-ONE instead of mean-centred, a per-GROUP
    # [0, 1] frame instead of a global one).
    ap.add_argument("--tail-weight", type=float, default=None,
                    help="TailRL advantage reweighting strength. 0 (default) "
                         "is off and BIT-IDENTICAL; 1 is the paper's weight; "
                         "in between blends w -> 1 + f*(w-1), which is a "
                         "safety knob because a group's top weight can reach "
                         "N. Groups are the GOAL-DISTANCE BIN each episode "
                         "spawned in (--tail-bins), which is the closest "
                         "thing here to the paper's 'N rollouts for the "
                         "same prompt'; it is read off the race field, not "
                         "the reservoir, so the flag never moves the start "
                         "distribution")
    ap.add_argument("--tail-outcome", default=None,
                    choices=("return", "time"),
                    help="the scalar outcome the tail is taken over: "
                         "`return` the episode's undiscounted collected "
                         "reward (what rollout/ep_rew_mean sums), or `time` "
                         "negative finish time for finishers with "
                         "non-finishers ranked below every finisher by "
                         "their return. Only read when --tail-weight > 0")
    ap.add_argument("--tail-bins", type=int, default=None,
                    help="goal-distance bins the TailRL GROUPS are cut on "
                         "(default 16). Read off the fleet's own race field, "
                         "independently of --respawn-binned, so turning the "
                         "flag on never moves the start distribution. More "
                         "bins = a smaller start-state confound inside a "
                         "group and fewer rollouts in it; on the round-30 "
                         "finisher (~300 episodes ended per rollout) 16 "
                         "bins is ~19 per group at 2.0 s of confound, 32 is "
                         "~9 at 1.0 s, 64 is ~5 at 0.5 s. tail/n_med and "
                         "tail/groups are the read-out, and the arithmetic "
                         "is in surfgym/tailrl.py. Only read when "
                         "--tail-weight > 0")
    ap.add_argument("--tail-min-n", type=int, default=None,
                    help="groups with fewer than this many ended episodes "
                         "in a rollout keep weight 1 - there is no tail to "
                         "estimate from a handful of samples (the paper "
                         "draws N = 8-64 per prompt). Only read when "
                         "--tail-weight > 0")
    # --- Linesight's progress reward (survey section 3) ---------------------
    # "0.01/m advanced along the centerline", from a reference line that
    # "does not need to be fast... usually the centerline", later re-extracted
    # from the AI's own best runs. Pays progress ALONG a line and never
    # distance TO it, which is the distinction Song & Scaramuzza (RSS 2023)
    # draw: line TRACKING scored 44%/0%, gate PROGRESS 100%.
    ap.add_argument("--race-arc", default=None,
                    help="reference route .npz: shape on ARC LENGTH along it "
                         "instead of the geodesic distance-to-goal field. "
                         "Arc length is monotone along the route by "
                         "construction, so it cannot have the interior local "
                         "minimum the voxel geodesic has at route vertex "
                         "1601 (the graph believes an 8,700u glide across "
                         "open air). Scale = 100/route_length, the same "
                         "total collectible shaping as 100/d0. Off-corridor "
                         "pays ZERO, never a penalty. ckpt restores")
    ap.add_argument("--race-arc-corridor", type=float, default=None,  # 1500
                    help="how far off the line still earns arc progress; "
                         "matches tools/eval_honesty.py's default corridor")
    ap.add_argument("--race-arc-window", type=int, default=None,      # 16
                    help="how many route vertices the arc anchor may move "
                         "per tick (anti-farming: an off-route flight cannot "
                         "walk the coordinate down the track)")
    ap.add_argument("--race-kill-aware", type=int, default=None,
                    choices=(0, 1),                # S2; 0; ckpt restores
                    help="mask fail-teleport/fatal-hurt volumes as walls in "
                         "the goal graph so the shaping gradient routes "
                         "around kill zones instead of through them (eval "
                         "progress still measured on the standard field)")
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
    ap.add_argument("--seed", type=int, default=0,
                    help="master seed for env streams, policy init, action "
                         "noise and the minibatch permutation. Under DDP the "
                         "per-rank streams derive from it so the fleet's env "
                         "stream SET is bit-identical to a single-GPU run "
                         "with the same seed (docs/ddp-plan.md step 5)")
    ap.add_argument("--warm-caches", action="store_true",
                    help="build every map artifact (zones, occupancy, vision "
                         "SDF, geodesic goal field) single-process and exit. "
                         "Run this BEFORE torchrun: four ranks baking the "
                         "9-11 GB goal field concurrently is an OOM, and a "
                         "torn cache npz read by another rank is silently "
                         "wrong vision for the whole run")
    ap.add_argument("--dump-invariants", action="store_true",
                    help="print one JSON line of startup invariants (spawn "
                         "origins hash over the whole fleet, pool hash, "
                         "race_d0, param checksum) after reset and exit — "
                         "the C1 exactness gate of docs/ddp-plan.md §5")
    ap.add_argument("--int-sync-every", type=int, default=0,
                    help="DDP: decisions between novelty-count syncs "
                         "(0 = once per iteration). Tighten only if an A/B "
                         "on int/ep shows the frontier over-payment matters")
    ap.add_argument("--ddp-overlap", type=int, default=1, choices=(0, 1),
                    help="DDP: overlap the gradient all-reduce with backward "
                         "via per-bucket post-accumulate-grad hooks (plan "
                         "step 15). On P2P-less boxes the exposed collective "
                         "is ~35%% of the iteration; the bucket split hides "
                         "most of it behind convolution_backward. 0 = the "
                         "fully exposed single flat all-reduce")
    ap.add_argument("--ddp-assert-every", type=int, default=10,
                    help="DDP: iterations between cross-rank state checks "
                         "(respawn ring / counts table / spawn pool hashes; "
                         "the param checksum runs every 100 regardless). "
                         "1 = every iteration, for validation runs")
    ap.add_argument("--timing", action="store_true",
                    help="print one parse-friendly TIMING line per iteration "
                         "(per-phase ms; GPU phases via CUDA events read once "
                         "per iteration) — the perf harness, see "
                         "docs/perf-implementation-plan.md")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--no-eval-at-start", action="store_true",
                    help="skip the eval on the FIRST iteration. next_record "
                         "starts at global_step, so an eval always fires "
                         "before any training - and the eval loop is ONE map "
                         "at a time on ONE env, so at 107 maps that is ~50 "
                         "minutes before the first gradient step, and it "
                         "makes early throughput readings meaningless")
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
    ap.add_argument("--fp32-heads", type=int, default=None,   # 0 = off
                    help="run the action/code head and the value head with "
                         "autocast disabled, so the stored logits and values "
                         "are the fp32 result rather than a bf16-rounded one "
                         "(bf16 keeps 7 explicit mantissa bits, and every "
                         "value the rollout stores and every ratio the update "
                         "differences goes through these two Linears). The "
                         "trunk and towers stay bf16. 0 = today")
    args = ap.parse_args()

    # DDP facade: reads the torchrun env, pins this rank's CUDA device
    # BEFORE any cuda-touching line. At world_size==1 (all of Windows dev,
    # every single-GPU launch) it is a literal no-op object.
    D = distributed.init()
    if not D.is_main and not os.environ.get("DDP_DEBUG_STDOUT"):
        # rank 0 owns every artifact and every log line; interleaved output
        # from four ranks corrupts the TIMING protocol and every parse.
        # stderr stays live so rank tracebacks are never swallowed.
        # DDP_DEBUG_STDOUT=1 keeps every rank talking (deadlock hunts with
        # torchrun --redirects, where per-rank logs are the whole point).
        sys.stdout = open(os.devnull, "w")

    if (args.lidar_w is None) != (args.lidar_h is None):
        raise SystemExit("pass BOTH --lidar-w and --lidar-h: a lone flag "
                         "silently pairs with the checkpoint's other dim "
                         "and renders a distorted FOV with no error")

    # a bare `--ckpt` resume must not silently change the training objective:
    # settings that define the run (reward mode, blend window, episode length)
    # come from the checkpoint's saved config unless explicitly overridden
    def flag_given(name):
        # argparse also accepts --flag=value; the bare token test missed it
        # and let the ckpt silently override an explicitly passed flag
        return any(a == name or a.startswith(name + "=") for a in sys.argv[1:])

    # --tick-ms-schedule: the CLI spec is authoritative over both --tick-ms
    # and anything the checkpoint carries. Parsed BEFORE the resume block so
    # the TICK TRANSFER notice below compares the ramp's STARTING tick
    # against the checkpoint's, which is the comparison that matters.
    tick_sched = None
    tick_sched_resumed = False
    if args.tick_ms_schedule:
        from surfgym.tick import TickSchedule
        tick_sched = TickSchedule.parse(args.tick_ms_schedule)
        if flag_given("--tick-ms"):
            raise SystemExit(
                f"--tick-ms {args.tick_ms:g} and --tick-ms-schedule "
                f"{tick_sched.spec()} both set the physics tick. The "
                "schedule owns it (it starts at FROM); drop --tick-ms.")
        args.tick_ms = tick_sched.from_ms

    ck = None
    obj_changed = False
    tick_ms_ckpt = None          # set when a resume changes the tick
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
        # the MAP LIST is part of what the run is, exactly like the reward
        # mode: a bare resume of a two-map checkpoint that silently trained
        # on one map would be a different experiment wearing the same name
        if (not flag_given("--maps") and not flag_given("--map")
                and ck_cfg.get("maps")):
            args.maps = ",".join(str(m) for m in ck_cfg["maps"])
            restored.append(f"maps={args.maps}")
        elif not flag_given("--map") and ck_cfg.get("map"):
            args.map = str(ROOT / "maps" / f"{ck_cfg['map']}.bsp")
            restored.append(f"map={ck_cfg['map']}")
        # the held-out list is part of the run for the same reason: a bare
        # resume that silently dropped it would stop producing the one
        # metric the run exists for, with nothing in the log saying so
        if not flag_given("--heldout-maps") and ck_cfg.get("heldout_maps"):
            args.heldout_maps = ",".join(str(m) for m in ck_cfg["heldout_maps"])
            restored.append(f"heldout_maps={args.heldout_maps}")
            if (not flag_given("--heldout-goal-cell")
                    and ck_cfg.get("heldout_goal_cell") not in (None, "")):
                args.heldout_goal_cell = str(ck_cfg["heldout_goal_cell"])
                restored.append(
                    f"heldout_goal_cell={args.heldout_goal_cell}")
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
        if args.finish_k is None and ck_cfg.get("finish_k") is not None:
            args.finish_k = float(ck_cfg["finish_k"])
            restored.append(f"finish_k={args.finish_k:g}")
        if args.finish_tref is None and ck_cfg.get("finish_tref") is not None:
            args.finish_tref = float(ck_cfg["finish_tref"])
            restored.append(f"finish_tref={args.finish_tref:g}")
        if not flag_given("--obs-reward") and ck_cfg.get("obs_reward"):
            args.obs_reward = True
            restored.append("obs_reward")
        if not flag_given("--race-ng") and ck_cfg.get("race_ng"):
            args.race_ng = int(ck_cfg["race_ng"])
            restored.append(f"race_ng={args.race_ng}")
        if args.death_charge is None and ck_cfg.get("death_charge"):
            args.death_charge = float(ck_cfg["death_charge"])
            restored.append(f"death_charge={args.death_charge:g}")
        if args.ez_eps is None and ck_cfg.get("ez_eps") is not None:
            args.ez_eps = float(ck_cfg["ez_eps"])
            restored.append(f"ez_eps={args.ez_eps:g}")
        if args.ez_max is None and ck_cfg.get("ez_max") is not None:
            args.ez_max = int(ck_cfg["ez_max"])
        if args.ez_mu is None and ck_cfg.get("ez_mu") is not None:
            args.ez_mu = float(ck_cfg["ez_mu"])
        if args.train_stride is None and ck_cfg.get("train_stride") is not None:
            args.train_stride = int(ck_cfg["train_stride"])
            restored.append(f"train_stride={args.train_stride}")
        if not flag_given("--yaw-adaptive") and ck_cfg.get("yaw_adaptive"):
            args.yaw_adaptive = True
            restored.append("yaw_adaptive")
        if not flag_given("--side-hold") and ck_cfg.get("side_hold") is not None \
                and int(ck_cfg["side_hold"]) != int(args.side_hold):
            args.side_hold = int(ck_cfg["side_hold"])
            restored.append(f"side_hold={args.side_hold}")
        if not flag_given("--yaw-blend") and ck_cfg.get("yaw_blend") is not None \
                and float(ck_cfg["yaw_blend"]) != float(args.yaw_blend):
            args.yaw_blend = float(ck_cfg["yaw_blend"])
            restored.append(f"yaw_blend={args.yaw_blend}")
        if (not flag_given("--reward-per-decision")
                and ck_cfg.get("reward_per_decision")):
            args.reward_per_decision = True
            restored.append("reward_per_decision")
        if args.fail_pen is None and ck_cfg.get("fail_pen") is not None:
            args.fail_pen = float(ck_cfg["fail_pen"])
            restored.append(f"fail_pen={args.fail_pen:g}")
        if args.speed_coef is None and ck_cfg.get("speed_coef") is not None:
            args.speed_coef = float(ck_cfg["speed_coef"])
            restored.append(f"speed_coef={args.speed_coef:g}")
        if args.stall_secs is None and ck_cfg.get("stall_secs") is not None:
            args.stall_secs = float(ck_cfg["stall_secs"])
            restored.append(f"stall_secs={args.stall_secs:g}")
        if args.stall_eps is None and ck_cfg.get("stall_eps") is not None:
            args.stall_eps = float(ck_cfg["stall_eps"])
            restored.append(f"stall_eps={args.stall_eps:g}")
        if args.max_step is None and ck_cfg.get("max_step") is not None:
            args.max_step = float(ck_cfg["max_step"])
            restored.append(f"max_step={args.max_step:g}")
        if (not flag_given("--eval-stall")
                and ck_cfg.get("eval_stall") is not None):
            # eval CONDITIONS, carried across a resume for the same reason
            # every other arm setting is: a run that resumes and quietly
            # stops stall-killing its evals has changed what its own metric
            # measures, half way through, with nothing in the log saying so.
            args.eval_stall = int(ck_cfg["eval_stall"])
            restored.append(f"eval_stall={args.eval_stall:d}")
        if args.ret_norm is None and ck_cfg.get("ret_norm") is not None:
            # --ret-norm changes what the VALUE HEAD MEANS (normalized
            # returns, not returns). Resuming without it would read the
            # checkpoint's critic in the wrong frame - silently, and
            # everywhere V is used. It restores like every other reward-side
            # setting that a checkpoint cannot be reinterpreted without.
            args.ret_norm = int(ck_cfg["ret_norm"])
            restored.append(f"ret_norm={args.ret_norm:d}")
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
        if args.respawn_min_speed is None and ck_cfg.get("respawn_min_speed") is not None:
            args.respawn_min_speed = float(ck_cfg["respawn_min_speed"])
            restored.append(f"respawn_min_speed={args.respawn_min_speed:g}")
        if args.respawn_binned is None and ck_cfg.get("respawn_binned") is not None:
            args.respawn_binned = int(ck_cfg["respawn_binned"])
            restored.append(f"respawn_binned={args.respawn_binned}")
        # --tail-weight and its two riders: same contract as the explore
        # flags below - a resume that silently dropped the reweighting would
        # revert the arm to its own control while run.json honestly claims
        # the treatment was off. An explicit flag still wins.
        if args.tail_weight is None and ck_cfg.get("tail_weight") is not None:
            args.tail_weight = float(ck_cfg["tail_weight"])
            restored.append(f"tail_weight={args.tail_weight:g}")
        if args.tail_outcome is None and ck_cfg.get("tail_outcome") is not None:
            args.tail_outcome = str(ck_cfg["tail_outcome"])
            restored.append(f"tail_outcome={args.tail_outcome}")
        if args.tail_min_n is None and ck_cfg.get("tail_min_n") is not None:
            args.tail_min_n = int(ck_cfg["tail_min_n"])
            restored.append(f"tail_min_n={args.tail_min_n}")
        if args.tail_bins is None and ck_cfg.get("tail_bins") is not None:
            args.tail_bins = int(ck_cfg["tail_bins"])
            restored.append(f"tail_bins={args.tail_bins}")
        # explore-arm flags: without these a resumed arm silently reverts to
        # the control while its new run.json honestly claims uniform/off
        if args.race_shaping is None and ck_cfg.get("race_shaping") is not None:
            args.race_shaping = float(ck_cfg["race_shaping"])
            restored.append(f"race_shaping={args.race_shaping:g}")
        # --race-dfloor changes what the reward IS, and under --obs-reward it
        # also changes scalar slot 12; a resume that silently dropped it would
        # hand the policy a different objective than its weights were fitted
        # to. Same restore contract as --race-shaping.
        if args.race_dfloor is None and ck_cfg.get("race_dfloor") is not None:
            args.race_dfloor = float(ck_cfg["race_dfloor"])
            restored.append(f"race_dfloor={args.race_dfloor:g}")
        # --race-latch is the same contract, and stricter: dropping it on
        # a resume would also drop an OBSERVATION column, so the widened
        # checkpoint would not even load
        # the two latch forms are one setting with two units, so an explicit
        # flag in either form must stop the OTHER one being restored - a
        # resume that carried both would raise on the mutual-exclusion check
        # with the user's own flag as the apparent cause
        _latch_given = (flag_given("--race-latch")
                        or flag_given("--race-latch-frac"))
        if (args.race_latch is None and not _latch_given
                and ck_cfg.get("race_latch") is not None):
            args.race_latch = float(ck_cfg["race_latch"])
            restored.append(f"race_latch={args.race_latch:g}")
        if (args.race_latch_frac is None and not _latch_given
                and ck_cfg.get("race_latch_frac") is not None):
            args.race_latch_frac = float(ck_cfg["race_latch_frac"])
            restored.append(f"race_latch_frac={args.race_latch_frac:g}")
        # --race-arc changes what the reward IS, and under --obs-reward it
        # also changes scalar slot 12; a resume that silently dropped it
        # would hand the policy a different objective than its weights were
        # fitted to. Same restore contract as --route.
        if args.race_arc is None and ck_cfg.get("race_arc"):
            args.race_arc = str(ck_cfg["race_arc"])
            restored.append(f"race_arc={Path(args.race_arc).name}")
        if (args.race_arc_corridor is None
                and ck_cfg.get("race_arc_corridor") is not None):
            args.race_arc_corridor = float(ck_cfg["race_arc_corridor"])
            restored.append(f"race_arc_corridor={args.race_arc_corridor:g}")
        if (args.race_arc_window is None
                and ck_cfg.get("race_arc_window") is not None):
            args.race_arc_window = int(ck_cfg["race_arc_window"])
            restored.append(f"race_arc_window={args.race_arc_window}")
        # --goals widens the observation like --route; a resume that
        # dropped it could not even load the widened checkpoint
        if args.goals is None and ck_cfg.get("goals") is not None:
            args.goals = int(ck_cfg["goals"])
            restored.append(f"goals={args.goals}")
            for _k in ("goal_radius", "goal_kmin", "goal_kmax", "goal_kcap",
                       "goal_air_frac", "goal_holdout", "goal_curriculum",
                       "goal_obs", "goal_views", "goal_route",
                       "goal_route_frac", "goal_reward", "goal_frontier",
                       "goal_route_uniform", "goal_fixed", "goal_fixed_spacing",
                       "goal_fixed_air", "goal_euclid_scale", "goal_fixed_decay",
                       "goal_front_start", "goal_front_band",
                       "goal_front_step", "goal_front_rate",
                       "goal_front_min_ep"):
                if ck_cfg.get(_k) is not None:
                    setattr(args, _k, ck_cfg[_k])
        if args.respawn_mode is None and ck_cfg.get("respawn_mode"):
            args.respawn_mode = str(ck_cfg["respawn_mode"])
            restored.append(f"respawn_mode={args.respawn_mode}")
        if args.respawn_bins is None and ck_cfg.get("respawn_bins") is not None:
            args.respawn_bins = int(ck_cfg["respawn_bins"])
            restored.append(f"respawn_bins={args.respawn_bins}")
        # --chunk redefines what ONE decision is; a bare resume that silently
        # dropped it would decode 64 code logits as flat action logits. The
        # decoder itself rides in the state_dict, so --codebook (init only) is
        # deliberately NOT restored: on a resume it has nothing left to do.
        if args.chunk is None and ck_cfg.get("chunk") is not None:
            args.chunk = int(ck_cfg["chunk"])
            restored.append(f"chunk={args.chunk}")
        if args.codes is None and ck_cfg.get("n_codes") is not None:
            args.codes = int(ck_cfg["n_codes"])
            restored.append(f"codes={args.codes}")
        if args.dec_ent is None and ck_cfg.get("dec_ent") is not None:
            args.dec_ent = float(ck_cfg["dec_ent"])
            restored.append(f"dec_ent={args.dec_ent:g}")
        if (args.respawn_killsafe is None
                and ck_cfg.get("respawn_killsafe") is not None):
            args.respawn_killsafe = int(ck_cfg["respawn_killsafe"])
            restored.append(f"respawn_killsafe={args.respawn_killsafe}")
        if args.spawn_burst is None and ck_cfg.get("spawn_burst") is not None:
            args.spawn_burst = int(ck_cfg["spawn_burst"])
            restored.append(f"spawn_burst={args.spawn_burst}")
        if (args.spawn_burst_p is None
                and ck_cfg.get("spawn_burst_p") is not None):
            args.spawn_burst_p = float(ck_cfg["spawn_burst_p"])
        if args.demo_file is None and ck_cfg.get("demo_file"):
            args.demo_file = str(ck_cfg["demo_file"])
            restored.append(f"demo_file={args.demo_file}")
        if args.demo_window is None and ck_cfg.get("demo_window") is not None:
            args.demo_window = int(ck_cfg["demo_window"])
        if args.demo_rate is None and ck_cfg.get("demo_rate") is not None:
            args.demo_rate = float(ck_cfg["demo_rate"])
        if args.demo_min_ep is None and ck_cfg.get("demo_min_ep") is not None:
            args.demo_min_ep = float(ck_cfg["demo_min_ep"])
        if args.int_view is None and ck_cfg.get("int_view") is not None:
            args.int_view = int(ck_cfg["int_view"])
            restored.append(f"int_view={args.int_view}")
        if args.rnd_coef is None and ck_cfg.get("rnd_coef") is not None:
            args.rnd_coef = float(ck_cfg["rnd_coef"])
            restored.append(f"rnd_coef={args.rnd_coef:g}")
        if args.speed_equiv is None and ck_cfg.get("speed_equiv") is not None:
            args.speed_equiv = float(ck_cfg["speed_equiv"])
            restored.append(f"speed_equiv={args.speed_equiv:g}")
        if args.int_speed is None and ck_cfg.get("int_speed") is not None:
            args.int_speed = int(ck_cfg["int_speed"])
            restored.append(f"int_speed={args.int_speed}")
        if args.race_kill_aware is None and ck_cfg.get("race_kill_aware") is not None:
            args.race_kill_aware = int(ck_cfg["race_kill_aware"])
            restored.append(f"race_kill_aware={args.race_kill_aware}")
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
        if args.surf_mask is None and ck_cfg.get("surf_mask") is not None:
            args.surf_mask = int(ck_cfg["surf_mask"])
            restored.append(f"surf_mask={args.surf_mask}")
        elif args.surf_mask is not None \
                and int(args.surf_mask) != int(ck_cfg.get("surf_mask") or 0):
            # conv1 is (16, in_ch, 5, 5): load_state_dict would die on the
            # shape anyway, three screens later and without the reason
            raise SystemExit(
                "--surf-mask changes the conv trunk's input channels, and a "
                "checkpoint's first layer cannot be widened or narrowed — "
                "start a fresh run, or drop the flag to keep the ckpt's "
                f"setting ({int(ck_cfg.get('surf_mask') or 0)})")
        # no shape guard for --pinhole: the tensors are the same size, only
        # the pixel VALUES change. Warm-starting across cameras is a
        # legitimate (if lossy) thing to ask for, same as --lidar-range
        if args.pinhole is None and ck_cfg.get("pinhole") is not None:
            args.pinhole = int(ck_cfg["pinhole"])
            restored.append(f"pinhole={args.pinhole}")
        if args.normals is None and ck_cfg.get("normals") is not None:
            args.normals = int(ck_cfg["normals"])
            restored.append(f"normals={args.normals}")
        elif args.normals is not None \
                and int(args.normals) != int(ck_cfg.get("normals") or 0):
            # conv1 is (16, in_ch, 5, 5) - same wall --surf-mask hits
            raise SystemExit(
                "--normals changes the conv trunk's input channels, and a "
                "checkpoint's first layer cannot be widened or narrowed - "
                "start a fresh run, or drop the flag to keep the ckpt's "
                f"setting ({int(ck_cfg.get('normals') or 0)})")
        # the fov, like --pinhole: same tensor shapes, different pixel
        # values, so a warm start across cameras is allowed and lossy
        if args.lidar_hfov is None and ck_cfg.get("lidar_hfov"):
            args.lidar_hfov = float(ck_cfg["lidar_hfov"])
            restored.append(f"lidar_hfov={args.lidar_hfov:g}")
        if args.lidar_vfov is None and ck_cfg.get("lidar_vfov"):
            args.lidar_vfov = float(ck_cfg["lidar_vfov"])
            restored.append(f"lidar_vfov={args.lidar_vfov:g}")
        if args.frame_stack is None and ck_cfg.get("frame_stack") is not None:
            args.frame_stack = int(ck_cfg["frame_stack"])
            restored.append(f"frame_stack={args.frame_stack}")
        elif (args.frame_stack is not None
              and max(1, int(args.frame_stack))
              != max(1, int(ck_cfg.get("frame_stack") or 1))):
            # conv1 is (16, K, 5, 5) — same wall --surf-mask hits
            raise SystemExit(
                "--frame-stack changes the conv trunk's input channels, and a "
                "checkpoint's first layer cannot be widened or narrowed — "
                "start a fresh run, or drop the flag to keep the ckpt's "
                f"setting ({int(ck_cfg.get('frame_stack') or 0)})")
        # --route restores like any other obs-shaping flag: a resumed arm that
        # silently dropped its lookahead fan would have a narrower scalar row
        # than its own weights expect. Growing a route ONTO a pre-route ckpt
        # is the supported direction (widen_for_route); shrinking is not.
        if args.route is None and ck_cfg.get("route_file"):
            args.route = str(ck_cfg["route_file"])
            restored.append(f"route={Path(args.route).name}")
        if args.route_points is None and ck_cfg.get("route_points"):
            args.route_points = int(ck_cfg["route_points"])
            restored.append(f"route_points={args.route_points}")
        if args.route_span is None and ck_cfg.get("route_span"):
            args.route_span = float(ck_cfg["route_span"])
            restored.append(f"route_span={args.route_span:g}")
        if (args.route_critic_only is None
                and ck_cfg.get("route_critic_only") is not None):
            args.route_critic_only = int(ck_cfg["route_critic_only"])
            restored.append(f"route_critic_only={args.route_critic_only}")
        # --act-hist / --obs-compass are observation columns like the rest:
        # a resumed arm that silently dropped them would have a narrower
        # scalar row than its own weights expect. ADDING them to a
        # checkpoint is the supported direction - widen_for_obs inserts the
        # new columns at the block's own position with ZERO weights, so the
        # resumed policy computes its old function at step 0 and the arm
        # starts on its own control curve. Not restored the way route_*
        # above is when the flag was SPELLED OUT: that is the whole point of
        # the arm.
        #
        # What is refused, and why. The block is ordered
        # [6*K history | 5 compass], so the only growth that leaves every
        # trained column where it was is (a) history 0 -> K with no compass
        # yet, or (b) compass 0 -> 1 at an unchanged K. Widening the history
        # IN FRONT of an existing compass would shift its five columns, and
        # shrinking either would mean dropping trained columns with no
        # meaning-preserving way to choose which.
        _ck_hist = int(ck_cfg.get("act_hist") or 0)
        _ck_cmp = int(ck_cfg.get("obs_compass") or 0)
        if args.act_hist is None and ck_cfg.get("act_hist") is not None:
            args.act_hist = _ck_hist
            restored.append(f"act_hist={args.act_hist}")
        if args.obs_compass is None and ck_cfg.get("obs_compass") is not None:
            args.obs_compass = _ck_cmp
            restored.append(f"obs_compass={args.obs_compass}")
        _new_hist = int(args.act_hist or 0)
        _new_cmp = int(args.obs_compass or 0)
        if _new_hist < _ck_hist or _new_cmp < _ck_cmp:
            raise SystemExit(
                f"this checkpoint carries act_hist={_ck_hist} "
                f"obs_compass={_ck_cmp} and you asked for "
                f"act_hist={_new_hist} obs_compass={_new_cmp}: the "
                "scalar-side block only ever GROWS. Narrowing it would drop "
                "trained columns (and re-index the ones left), which is a "
                "partially re-initialised policy dressed as a warm start. "
                "Start a fresh run, or drop the flags to keep the "
                "checkpoint's own setting")
        if _new_hist > _ck_hist and _ck_cmp:
            raise SystemExit(
                f"--act-hist {_new_hist}: this checkpoint was trained with "
                f"act_hist={_ck_hist} AND --obs-compass, and the block is "
                "ordered [history | compass] - growing the history in front "
                "of the compass would shift its five columns and feed the "
                "checkpoint's weights permuted inputs. Resume at "
                f"--act-hist {_ck_hist}, or start a fresh run")
        # --priv-critic changes the SHAPE of value_head and adds the priv
        # MLP, so a resume has to agree with the checkpoint on it. Restoring
        # it silently (like route_critic_only above) is right for a bare
        # resume; the two mismatches are handled separately below, because
        # they are not symmetric - one is a supported warm start and the
        # other cannot work at all.
        _ck_priv = int(ck_cfg.get("priv_critic") or 0)
        _ck_ph = int(ck_cfg.get("priv_hidden") or 128)
        if args.priv_hidden is None and ck_cfg.get("priv_hidden") is not None:
            args.priv_hidden = _ck_ph
            restored.append(f"priv_hidden={args.priv_hidden}")
        # NOT restored from the config the way route_critic_only above is:
        # --priv-critic decides what the CRITIC is fitted on, and silently
        # inheriting it either way is how a "control" run turns out to have
        # been the treatment. A privileged checkpoint has to be resumed with
        # the flag SPELLED OUT; adding it to a plain checkpoint is the
        # supported warm start (widen_for_priv) and is announced.
        if _ck_priv and not int(args.priv_critic or 0):
            raise SystemExit(
                "this checkpoint was trained with --priv-critic: its "
                f"value_head is {_ck_ph} columns wider than a plain one and "
                "it carries a priv_mlp. Resuming it without the flag would "
                "mean throwing those tensors away - a partially "
                "re-initialised critic dressed as a warm start. Pass "
                "--priv-critic 1 (the ACTOR is the same weights either "
                "way), or start the control from a non-priv checkpoint")
        if _ck_priv and int(args.priv_hidden) != _ck_ph:
            raise SystemExit(
                f"--priv-hidden {args.priv_hidden} but this checkpoint's "
                f"privileged critic is {_ck_ph} wide: its value_head and "
                "priv_mlp are that shape and there is no meaning-preserving "
                "way to re-widen them")
        if not args.fp32 and ck_cfg.get("bf16") is False:
            args.fp32 = True
            restored.append("fp32")
        # --fp32-heads changes no tensor, only where the rounding happens, so
        # unlike the capacity flags a mismatch is legal - but a bare resume
        # keeps the run's own numerics rather than silently switching them
        if args.fp32_heads is None and ck_cfg.get("fp32_heads") is not None:
            args.fp32_heads = int(ck_cfg["fp32_heads"])
            restored.append(f"fp32_heads={args.fp32_heads}")
        if (args.fix_pitch is None and not args.free_pitch
                and ck_cfg.get("fix_pitch") is not None):
            args.fix_pitch = float(ck_cfg["fix_pitch"])
            restored.append(f"fix_pitch={args.fix_pitch:g}")
        # --free-pitch releases this one too: it is the same "the gaze is
        # frozen" property, only pinned rather than inherited
        if (args.pitch_fixed is None and not args.free_pitch
                and ck_cfg.get("pitch_fixed") is not None):
            args.pitch_fixed = float(ck_cfg["pitch_fixed"])
            restored.append(f"pitch_fixed={args.pitch_fixed:g}")
        if args.emb is None and ck_cfg.get("emb"):
            args.emb = int(ck_cfg["emb"])
            restored.append(f"emb={args.emb}")
        if args.hidden is None and ck_cfg.get("hidden"):
            args.hidden = int(ck_cfg["hidden"])
            restored.append(f"hidden={args.hidden}")
        # --tower-depth / --conv-mult change WHICH tensors exist and how wide
        # they are. There is no zero-pad warm start across them (a deeper
        # tower has extra Linears, a wider trunk has different conv shapes),
        # so a disagreement is refused here with the two numbers rather than
        # left to a load_state_dict size error naming one tensor.
        for _k, _dflt in (("tower_depth", 2), ("conv_mult", 1)):
            _ckv = int(ck_cfg.get(_k) or _dflt)
            _cur = getattr(args, _k)
            if _cur is None:
                if _ckv != _dflt:
                    setattr(args, _k, _ckv)
                    restored.append(f"{_k}={_ckv}")
            elif int(_cur) != _ckv:
                raise SystemExit(
                    f"--{_k.replace('_', '-')} {int(_cur)} != the "
                    f"checkpoint's {_ckv}: it decides which tensors the "
                    "policy has, and there is no warm start across it - "
                    "start a fresh run, or drop the flag to keep the ckpt's")
        # --trunk changes WHICH modules exist, so a mismatch is not a warm
        # start, it is a different network wearing the checkpoint's name.
        # Same treatment as --surf-mask: say so here rather than let
        # load_state_dict fail three screens later on a missing key.
        if args.trunk is None and ck_cfg.get("trunk"):
            args.trunk = str(ck_cfg["trunk"])
            restored.append(f"trunk={args.trunk}")
        elif (args.trunk is not None
                and args.trunk != str(ck_cfg.get("trunk") or "plain")):
            raise SystemExit(
                "--trunk selects the image encoder and a checkpoint carries "
                "exactly one - start a fresh run, or drop the flag to keep "
                f"the ckpt's setting ({str(ck_cfg.get('trunk') or 'plain')})")
        # --rnn: a recurrent ckpt must resume recurrent, at its own width
        # (the GRU is a state_dict module and the towers are wider). The
        # other direction - --rnn gru onto a feed-forward ckpt - is the
        # supported warm start (widen_for_rnn), so it is NOT a mismatch.
        _ck_rnn = str(ck_cfg.get("rnn") or "none")
        if args.rnn is None:
            if _ck_rnn != "none":
                args.rnn = _ck_rnn
                restored.append(f"rnn={args.rnn}")
        elif _ck_rnn != "none" and args.rnn != _ck_rnn:
            raise SystemExit(
                f"this checkpoint is recurrent (rnn={_ck_rnn}) and cannot "
                "be resumed feed-forward - drop --rnn or start a fresh run")
        if _ck_rnn != "none" and ck_cfg.get("rnn_size"):
            if args.rnn_size is None:
                args.rnn_size = int(ck_cfg["rnn_size"])
                restored.append(f"rnn_size={args.rnn_size}")
            elif int(args.rnn_size) != int(ck_cfg["rnn_size"]):
                raise SystemExit(
                    f"--rnn-size {args.rnn_size} != the checkpoint's "
                    f"{int(ck_cfg['rnn_size'])}: the GRU and the towers "
                    "are sized by it - drop the flag to keep the ckpt's")
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
        # --mask-forward-air / --jump-cooldown / --duck-air-mask change the
        # policy's SUPPORT, so a resume that silently dropped one would
        # optimise a different action space than the weights were trained
        # in. Restore what the checkpoint carries when the flag was not
        # given; say so loudly when it was and it disagrees (turning a mask
        # on or off IS the arm, so it is allowed, never silent).
        for _fl, _key in (("--mask-forward-air", "mask_forward_air"),
                          ("--jump-cooldown", "jump_cooldown"),
                          ("--duck-air-mask", "duck_air_mask")):
            _at = _fl[2:].replace("-", "_")
            _ck_v = int(ck_cfg.get(_key) or 0)
            _cli = int(getattr(args, _at) or 0)
            if not flag_given(_fl):
                if _ck_v:
                    setattr(args, _at, _ck_v if _key == "jump_cooldown"
                            else True)
                    restored.append(f"{_at}={_ck_v}")
            elif _cli != _ck_v:
                print(f"!! ACTION MASK CHANGE: {_key} {_ck_v} -> {_cli}. "
                      "The policy's support changes with it, so the first "
                      "eval after this resume is NOT comparable to the "
                      "checkpoint's last one - compare arm vs matched "
                      "control from the same resume, not against history.")
        # --yaw-cond changes the policy's SHAPE (a yaw_side.table tensor)
        # and its factorisation, so a resume that silently dropped it would
        # fail a strict load three screens later without the reason.
        # Restored when the flag was not given; adding it to a plain
        # checkpoint is the supported warm start (widen_for_yawcond) and is
        # announced below.
        if int(ck_cfg.get("yaw_cond") or 0) and not flag_given("--yaw-cond"):
            args.yaw_cond = True
            restored.append("yaw_cond=1")
        # --view-continuous changes the policy's SHAPE (view_head,
        # view_log_std) and what an action IS; same restore contract
        if (int(ck_cfg.get("view_continuous") or 0)
                and not flag_given("--view-continuous")):
            args.view_continuous = True
            restored.append("view_continuous=1")
        # --view-absolute: the same contract (it changes what the view row
        # MEANS and, in world mode, the policy's shape)
        if ck_cfg.get("view_absolute") and not flag_given("--view-absolute"):
            args.view_absolute = str(ck_cfg["view_absolute"])
            restored.append(f"view_absolute={args.view_absolute}")
        if args.pitch_rate is None and ck_cfg.get("pitch_rate") is not None:
            args.pitch_rate = float(ck_cfg["pitch_rate"])
            restored.append(f"pitch_rate={args.pitch_rate:g}")
        # --tick-ms-schedule: the ramp is RUN STATE, not a hyperparameter -
        # a checkpoint saved half way down one carries FROM/TO/STEPS and the
        # step the ramp was measured from, so a bare resume continues it
        # exactly rather than restarting at FROM. An explicit flag on the
        # resume replaces it (and re-origins at the resumed step, above).
        if tick_sched is None and ck_cfg.get("tick_schedule"):
            from surfgym.tick import TickSchedule
            tick_sched = TickSchedule.from_dict(ck_cfg["tick_schedule"])
            tick_sched_resumed = True
            restored.append(f"tick_schedule={tick_sched.spec()}"
                            f"@{tick_sched.origin}")
        # --tick-ms: part of the physics the weights were trained under.
        # Older checkpoints have no key and were all trained at 10 ms.
        ck_tick = float(ck_cfg.get("tick_ms") or 10.0)
        if args.tick_ms is None:
            args.tick_ms = ck_tick
            restored.append(f"tick_ms={args.tick_ms:g}")
        elif abs(float(args.tick_ms) - ck_tick) > 1e-9:
            from surfgym.tick import TickClock as _TC
            _new, _old = _TC(args.tick_ms), _TC(ck_tick)
            tick_ms_ckpt = ck_tick
            print("!! TICK TRANSFER: this checkpoint was trained at "
                  f"{ck_tick:g} ms ticks ({_old.hz:.1f} Hz) and is being "
                  f"resumed at --tick-ms {args.tick_ms:g} "
                  f"({_new.describe()}).")
            print("!! The physics under the weights changes: "
                  f"{_new.hz / _old.hz:.3f}x the air-accelerate impulses per "
                  "second, the same turn rate / time penalty / discount "
                  "horizon / stall window in SECONDS (every per-tick "
                  "constant is rescaled), and a decision every "
                  f"{int(args.act_every or ck_cfg.get('act_every', 1))} ticks "
                  "is now "
                  f"{int(args.act_every or ck_cfg.get('act_every', 1)) * _new.ms:.1f} ms "
                  f"instead of "
                  f"{int(args.act_every or ck_cfg.get('act_every', 1)) * _old.ms:.1f} ms. "
                  "Allowed on purpose (the warm transfer is the experiment); "
                  "run.json records tick_ms AND tick_ms_ckpt.")
            if (not flag_given("--ep-ticks") and args.ep_secs is None
                    and ck_cfg.get("ep_ticks") and not obj_changed):
                # keep the episode cap in SECONDS across the transfer
                _secs = _old.ticks_to_secs(int(ck_cfg["ep_ticks"]))
                args.ep_ticks = _new.secs_to_ticks(_secs, "round")
                print(f"!! ep_ticks {int(ck_cfg['ep_ticks'])} at {ck_tick:g} ms "
                      f"= {_secs:g} s -> {args.ep_ticks} ticks at "
                      f"{_new.ms:.4f} ms (pass --ep-ticks/--ep-secs to "
                      "override)")
        if args.revisit_pen is None and ck_cfg.get("revisit_pen") is not None:
            args.revisit_pen = float(ck_cfg["revisit_pen"])
            restored.append(f"revisit_pen={args.revisit_pen:g}")
        if not flag_given("--punch-min") and ck_cfg.get("punch_min") is not None:
            args.punch_min = float(ck_cfg["punch_min"])
            args.punch_max = float(ck_cfg.get("punch_max", args.punch_max))
            restored.append(f"punch={args.punch_min:g}-{args.punch_max:g}")
        # PPO knobs: without these a resumed arm silently reverts to the
        # argparse defaults while run.json claims otherwise
        for k in ("lr", "gamma", "gae", "clip", "ent", "vf", "ent_final"):
            if getattr(args, k) is None and ck_cfg.get(k) is not None:
                setattr(args, k, float(ck_cfg[k]))
                restored.append(f"{k}={getattr(args, k):g}")
        # The BATCH SHAPE is the same category and was the hole in it: these
        # three have real argparse defaults rather than None, so the loop
        # above cannot carry them and a bare resume of a T=32 arm silently
        # went back to T=128 while its run.json still said 32. Round 21
        # measured T as a real variable (2.2x on corridor MAX between 32 and
        # 128) and --minibatches is a COUNT, so changing T also changes
        # update density and minibatch size - three variables moving at once,
        # unlogged, half way through a run. An explicit flag still wins:
        # flag_given is the test, exactly like --maps and --punch-min.
        for k, _flag in (("n_steps", "--n-steps"), ("epochs", "--epochs"),
                         ("minibatches", "--minibatches")):
            if not flag_given(_flag) and ck_cfg.get(k) is not None:
                setattr(args, k, int(ck_cfg[k]))
                restored.append(f"{k}={getattr(args, k)}")
        if args.eval_eps is None and ck_cfg.get("eval_eps"):
            args.eval_eps = int(ck_cfg["eval_eps"])
            restored.append(f"eval_eps={args.eval_eps}")
        if not args.eval_greedy_only and ck_cfg.get("eval_greedy_only"):
            args.eval_greedy_only = True
            restored.append("eval_greedy_only")
        if restored:
            print("restored from checkpoint config: " + ", ".join(restored))
    if args.reward is None:
        args.reward = "forward"
    if args.blend_start is None:
        args.blend_start = 100e6
    if args.blend_end is None:
        args.blend_end = 200e6
    if args.tick_ms is None:
        args.tick_ms = 10.0          # the reference tick: today, byte-identical
    from surfgym.tick import TickClock as _TC
    if tick_sched is not None:
        # the ramp is measured from the step this run LAUNCHES at, so a
        # fresh ramp starts at FROM wherever it is resumed from, and a ramp
        # restored from a checkpoint keeps its own origin and picks up at
        # the tick that checkpoint was saved under. --reset-steps folds the
        # step axis back to 0, and the ramp follows it (both are "this run
        # starts here"). global_step itself is read off the checkpoint far
        # below; the tick has to be final before the cores are built, and
        # the --ep-ticks default below is a DURATION at that tick.
        _sched_step0 = (0 if (ck is None or args.reset_steps)
                        else int(ck.get("global_step", 0)))
        # A CONTINUED ramp keeps its origin AND starts at the checkpoint's
        # OWN tick (already restored into args.tick_ms above) rather than at
        # the ramp's value for this step: the two are less than one
        # re-derivation apart by construction, and taking the schedule's
        # value here would fire a spurious TICK TRANSFER on every bare
        # resume. The first iteration moves it on, exactly as the ramp says.
        if not tick_sched_resumed or args.reset_steps:
            tick_sched.origin = _sched_step0
            args.tick_ms = tick_sched.ms_at(_sched_step0)
    if args.ep_ticks is None:
        # race: "play until you finish" — the stagnation kill does the real
        # episode control, the 2-minute cap is just a backstop. The DEFAULT
        # is a DURATION, so it converts at the run's tick: a literal 12000
        # would be 92 s at --tick-ms 7.63 and cannonball's own finishers
        # take 77-81 s. 120 s / 7 s are exactly 12000 / 700 at 10 ms. An
        # explicit --ep-ticks is the caller naming a tick count and stands.
        args.ep_ticks = _TC(args.tick_ms).secs_to_ticks(
            120.0 if args.reward == "race" else 7.0, "round")
    if args.ep_secs is not None:
        args.ep_ticks = _TC(args.tick_ms).secs_to_ticks(args.ep_secs, "round")
    if tick_sched is not None and not flag_given("--ep-ticks"):
        # --tick-ms-schedule: max_episode_ticks is baked into the core at
        # surf_create and has no C setter, so a cap sized at the LAUNCH tick
        # SHRINKS as a duration exactly the way the literal 12000 default
        # did before ecc0506 made it a duration - 12000 ticks is 92.0 s at
        # 7.667 ms and cannonball's own finishers take 77-81 s, which is
        # that review's defect 3 re-appearing one ramp later. Size the one
        # frozen number at the ramp's SHORTEST tick instead: the cap is then
        # never less than the duration it stands for, and is exactly that
        # duration at the end of a downward ramp. Only an explicit
        # --ep-ticks names a tick count and stands; --ep-secs and a
        # checkpoint's restored cap are durations and convert.
        _cap_s = _TC(args.tick_ms).ticks_to_secs(args.ep_ticks)
        _MIN = _TC(min(tick_sched.from_ms, tick_sched.to_ms))
        _cap_n = _MIN.secs_to_ticks(_cap_s, "round")
        if _cap_n != args.ep_ticks:
            print(f"tick schedule: episode cap {args.ep_ticks} ticks = "
                  f"{_cap_s:g} s at the launch tick -> {_cap_n} ticks, so it "
                  f"is still {_MIN.ticks_to_secs(_cap_n):.1f} s at the "
                  f"ramp's shortest tick ({_MIN.ms:.4f} ms) instead of "
                  f"{_MIN.ticks_to_secs(args.ep_ticks):.1f} s; it reads "
                  f"{_TC(args.tick_ms).ticks_to_secs(_cap_n):.1f} s at the "
                  f"launch tick. Pass --ep-ticks to name a tick count.")
            args.ep_ticks = _cap_n
    if args.race_dist is None:
        args.race_dist = "geodesic"
    if args.lr is None:
        args.lr = 3e-4
    if args.gamma is None:
        args.gamma = 0.995
    if args.gae is None:
        args.gae = 0.95
    if args.clip is None:
        args.clip = 0.2
    if args.ent is None:
        args.ent = 0.005
    if args.vf is None:
        args.vf = 0.5
    if args.time_pen is None:
        args.time_pen = 0.005
    if args.success_bonus is None:
        args.success_bonus = 50.0
    if args.finish_k is None:
        args.finish_k = 0.0
    if args.finish_tref is None:
        args.finish_tref = 120.0
    if args.train_stride is None:
        args.train_stride = 1
    if args.ez_eps is None:
        args.ez_eps = 0.0
    if args.ez_max is None:
        args.ez_max = 60
    if args.ez_mu is None:
        args.ez_mu = 2.0
    if args.fail_pen is None:
        args.fail_pen = 0.0
    if args.speed_coef is None:
        args.speed_coef = 0.0
    if args.stall_secs is None:
        # euclid shaping legitimately runs negative on away-from-goal legs
        # (hairpins) — a tight no-improvement window would execute progress
        args.stall_secs = 30.0 if args.race_dist == "euclid" else 15.0
    if args.stall_eps is None:
        args.stall_eps = 32.0                 # RaceReward's own default
    if args.max_step is None:
        args.max_step = 100.0                 # RaceReward's own default
    if args.ret_norm is None:
        args.ret_norm = 0
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
    if args.respawn_binned is None:
        args.respawn_binned = 0
    if args.tail_weight is None:
        args.tail_weight = 0.0
    if args.tail_outcome is None:
        args.tail_outcome = "return"
    if args.tail_min_n is None:
        args.tail_min_n = 4
    if args.tail_bins is None:
        args.tail_bins = 16
    if args.respawn_min_speed is None:
        args.respawn_min_speed = 0.0
    if args.respawn_mode is None:
        args.respawn_mode = "uniform"
    if args.respawn_bins is None:
        args.respawn_bins = 16
    if args.spawn_burst is None:
        args.spawn_burst = 0
    if args.spawn_burst_p is None:
        args.spawn_burst_p = 0.95
    if args.respawn_killsafe is None:
        args.respawn_killsafe = 0
    if args.race_shaping is None:
        args.race_shaping = 1.0
    if args.race_dfloor is None:
        args.race_dfloor = 0.0
    if args.race_latch is None:
        args.race_latch = 0.0
    if args.race_latch_frac is None:
        args.race_latch_frac = 0.0
    if args.race_latch > 0.0 and args.race_latch_frac > 0.0:
        raise SystemExit("--race-latch and --race-latch-frac are the same "
                         "setting in two units (absolute u vs a fraction of "
                         "the map's d0): pass one")
    if args.demo_window is None:
        args.demo_window = 10
    if args.demo_rate is None:
        args.demo_rate = 0.2
    if args.demo_min_ep is None:
        args.demo_min_ep = 50.0
    if args.int_view is None:
        args.int_view = 0
    if args.rnd_coef is None:
        args.rnd_coef = 0.0
    if args.speed_equiv is None:
        args.speed_equiv = 0.0
    if args.int_speed is None:
        args.int_speed = 0
    if args.race_kill_aware is None:
        args.race_kill_aware = 0
    if args.respawn_speed is None:
        args.respawn_speed = [0.9, 1.1]
    if args.lidar_w is None:
        args.lidar_w = LIDAR_W
    if args.lidar_h is None:
        args.lidar_h = LIDAR_H
    if args.lidar_w < 1 or args.lidar_h < 1:
        raise SystemExit("this trainer's policy needs the lidar block; "
                         "--lidar-w/--lidar-h must be >= 1")
    if args.surf_mask is None:
        args.surf_mask = 0
    if args.pinhole is None:
        args.pinhole = 0
    if args.normals is None:
        args.normals = 0
    if args.lidar_hfov is None:
        args.lidar_hfov = 120.0
    if args.lidar_vfov is None:
        args.lidar_vfov = 90.0
    if not (0.0 < args.lidar_hfov <= 360.0 and 0.0 < args.lidar_vfov <= 180.0):
        raise SystemExit("--lidar-hfov must be in (0, 360] and --lidar-vfov "
                         "in (0, 180] degrees")
    if args.frame_stack is None:
        args.frame_stack = 0
    check_vision_exclusive(args.surf_mask, args.pinhole, args.frame_stack,
                           args.normals)
    if args.act_hist is None:
        args.act_hist = 0
    if args.obs_compass is None:
        args.obs_compass = 0
    args.act_hist = max(0, int(args.act_hist))
    args.obs_compass = int(bool(args.obs_compass))
    if args.act_hist > 16:
        raise SystemExit(f"--act-hist {args.act_hist}: 16 decisions is "
                         "already 96 scalar columns; the cap is there to "
                         "catch a ticks-vs-decisions mix-up")
    if args.obs_compass and args.reward != "race":
        # the compass reads the field the race shaping walks down; no race
        # reward, no field, and a silently-zero block is worse than a refusal
        raise SystemExit("--obs-compass reads the shaping distance field, "
                         f"which only --reward race builds (got "
                         f"--reward {args.reward})")
    if args.route_critic_only and (args.act_hist or args.obs_compass):
        # both blocks live in the route-side block, and --route-critic-only
        # routes that whole block to the VALUE tower alone. An action history
        # the ACTOR cannot read is not the feature this flag advertises.
        raise SystemExit("--route-critic-only sends the whole scalar-side "
                         "block to the critic, so --act-hist/--obs-compass "
                         "would never reach the actor - which is the only "
                         "place they can change behaviour")
    if args.priv_critic:
        # Six of the ten columns are RaceReward's own state (the geodesic d
        # it shapes on, the arc anchor, the latch flag) and the seventh is
        # normalised by that map's start geodesic. Without --reward race
        # none of it exists, and a block that is silently zero in seven of
        # ten columns is the exact "fed constants" failure --obs-compass
        # refuses above.
        if args.reward != "race":
            raise SystemExit("--priv-critic reads the geodesic distance the "
                             "race shaping is built from, which only "
                             f"--reward race has (got --reward {args.reward})")
        if args.goals:
            # under --goals the reward's field is a PER-ENV GoalDistField
            # re-centred on each episode's assigned goal, so "d over the
            # map's start geodesic" is two different quantities in one
            # column and the truncation bootstrap cannot resample it for a
            # subset of rows. Refused rather than approximated.
            raise SystemExit("--priv-critic and --goals: the goal arms shape "
                             "on a per-env distance field, so column 6 (d/d0) "
                             "would mix a per-episode distance with the map's "
                             "start geodesic. Not supported")
    if args.goals and args.goal_obs is None:
        args.goal_obs = "fan"
    if args.goals and args.goal_reward is None:
        args.goal_reward = "sparse"
    # --normals is allowed under the ball: GoalBallLidar appends its views
    # after ALL the lidar's channels, so the image is (depth, nx, ny, nz,
    # ball views) and in_ch follows
    if args.goals and args.goal_obs in ("ball", "both") and (
            args.surf_mask or args.pinhole or int(args.frame_stack or 1) > 1):
        raise SystemExit("--goal-obs ball rides on the plain equiangular "
                         "depth image (optionally with --normals): exclusive "
                         "with --surf-mask, --pinhole and --frame-stack")
    if args.trunk is None:
        args.trunk = "plain"
    if args.rnn is None:
        args.rnn = "none"
    if args.rnn_size is None:
        args.rnn_size = 256
    RNN = args.rnn != "none"
    if RNN:
        # The update trains on whole per-env SEQUENCES (truncated BPTT over
        # the rollout), so every row of the rollout takes part and a row
        # subset cannot be dropped: --train-stride and the burst masks
        # (ez-greedy / --spawn-burst) both work by dropping rows. --chunk
        # is one GRU step per CHUNK in principle, but neither its eval
        # re-deliberation nor its neutral-tail mask has been checked
        # against a carried state. Each is a separate arm, not a drive-by.
        if int(args.chunk or 0) > 0:
            raise SystemExit("--rnn is not implemented with --chunk")
        if int(args.train_stride) > 1:
            raise SystemExit("--rnn trains on whole sequences: "
                             "--train-stride must be 1")
        if float(args.ez_eps) > 0.0 or int(args.spawn_burst or 0) > 0:
            raise SystemExit("--rnn trains on whole sequences and cannot "
                             "drop burst rows: --ez-eps 0, no --spawn-burst")
        if int(args.rnn_size) <= 0:
            raise SystemExit("--rnn-size must be positive")
    if args.emb is None:
        args.emb = 512
    if args.hidden is None:
        args.hidden = 448
    if args.tower_depth is None:
        args.tower_depth = 2          # the historical two-layer tower
    if args.conv_mult is None:
        args.conv_mult = 1            # the historical 16/32/64 stack
    if args.fp32_heads is None:
        args.fp32_heads = 0           # heads inside autocast, as they were
    if args.priv_critic is None:
        args.priv_critic = 0          # symmetric critic, as it was
    if args.priv_hidden is None:
        args.priv_hidden = 128
    if args.lidar_range is None:
        args.lidar_range = 2000.0
    if args.act_every is None:
        args.act_every = 3
    K = max(1, int(args.act_every))
    # ---- --chunk: H decisions per policy decision -------------------------
    # H = 0 is the flat six-head action space and EVERY chunk branch below is
    # dead: KH == K, b_act keeps its (T, N, 6) shape, Policy builds no code
    # head and no decoder, and the sampling/update/eval paths are the ones
    # that shipped. Gate on `H > 0`, never on a truthy config value.
    H = max(0, int(args.chunk or 0))
    KH = K * max(1, H)                # physics ticks per POLICY decision
    # ---- action masks (the air keys) -------------------------------------
    # MASKS.on is False on every run without a flag, and every branch keyed
    # on it below is dead: no buffer, no kernel, no config key.
    MASKS = ActionMasks(fwd_air=args.mask_forward_air,
                        jump_cd=args.jump_cooldown,
                        duck_air=args.duck_air_mask)
    # ---- --yaw-cond -------------------------------------------------------
    # A Python constant, like MASKS.on / H / STACK: every branch keyed on it
    # is decided at trace time, so a control run compiles and captures the
    # graph that shipped.
    YCOND = bool(args.yaw_cond)
    # ---- --view-continuous ------------------------------------------------
    # A Python constant like YCOND: every branch keyed on it is decided at
    # trace time, so a control run compiles and captures the graph that
    # shipped. The combinations refused here need real design (docs/
    # contyaw.md): each of them either reads the yaw BIN (yaw-cond, act-hist)
    # or draws whole action rows outside the policy (bursts, chunk codes),
    # and --frame-stack/--rnn carry per-env inference state the planner and
    # the transplant do not clone.
    VIEWC = bool(args.view_continuous)
    if VIEWC:
        _bad = []
        if YCOND:
            _bad.append("--yaw-cond")
        if H > 0 or args.codebook:
            _bad.append("--chunk/--codebook")
        if int(args.act_hist or 0) > 0:
            _bad.append("--act-hist")
        if int(args.frame_stack or 0) > 1:
            _bad.append("--frame-stack")
        if RNN:
            _bad.append("--rnn")
        if float(args.ez_eps or 0.0) > 0.0:
            _bad.append("--ez-eps")
        if int(args.spawn_burst or 0) > 0:
            _bad.append("--spawn-burst")
        if _bad:
            raise SystemExit("--view-continuous is not implemented with "
                             + ", ".join(_bad) + " (docs/contyaw.md: these "
                             "read the yaw bin or draw action rows outside "
                             "the policy). Drop them for this arm.")
    # --view-absolute: absolute targets (docs/contyaw.md "Absolute
    # targets"). The core gets view_mode 1 / 2 in every slot it drives -
    # rollout, eval and the record cores below - through _view_env; the
    # control run passes no such key and builds the config it always did.
    VIEW_ABS = str(args.view_absolute) if args.view_absolute else None
    if VIEW_ABS and not VIEWC:
        raise SystemExit("--view-absolute needs --view-continuous: it is a "
                         "reading of the continuous view row, not a flag "
                         "of its own")
    if VIEW_ABS and args.bc_file:
        raise SystemExit("--bc-file is not implemented under --view-absolute: "
                         "the BC targets (surfgym.view.z_from_view, the "
                         "file's view_zmu / view_zsd) are delta-space z. "
                         "Run the arm without it.")
    _view_env = {"view_mode": view_mode_code(VIEW_ABS)} if VIEW_ABS else {}
    if YCOND and H > 0:
        raise SystemExit(
            "--yaw-cond is not implemented for --chunk: a chunk emits H "
            "decisions from ONE code drawn at the chunk start, so there is "
            "no per-decision yaw bin to condition the side key on and the "
            "conditioning could not be reproduced in the update. Run the "
            "arm flat.")
    if MASKS.on and H > 0:
        raise SystemExit(
            "--mask-forward-air / --jump-cooldown / --duck-air-mask are not "
            "implemented for --chunk: a chunk emits H decisions from ONE "
            "code drawn at the chunk start, so the on-ground flag of "
            "decisions 1..H-1 is not known when the plan is sampled and the "
            "mask could not be reproduced in the update. Run the arm flat.")
    if MASKS.on and args.bc_file:
        raise SystemExit(
            "--mask-* with --bc-file: the cloning loss would fit the "
            "planner's unmasked actions through masked logits (an infinite "
            "NLL on any row the mask forbids). Mask the planner instead, or "
            "run the arm without --bc-file.")
    # ---- --tick-ms: every per-tick constant, converted ONCE here ---------
    # TICK.ms is the REALISED mean tick (7.667 for a 7.63 request); the
    # per-tick flags are defined at the 10 ms reference and rescaled so
    # their per-SECOND meaning is unchanged. At 10 ms every one of these is
    # the flag value itself, bit for bit (surfgym.tick.TickClock).
    from surfgym.tick import PATTERN_TOL_MS, TickClock
    TICK = TickClock(args.tick_ms)
    GAMMA_T = TICK.gamma(args.gamma)          # same horizon in seconds
    TIME_PEN_T = TICK.per_tick(args.time_pen)     # same reward per second
    SPEED_COEF_T = TICK.per_tick(args.speed_coef)
    # --stall-eps is a per-CALL distance threshold (CLAUDE.md: it scales with
    # the decision rate); a shorter tick makes the same K a shorter decision,
    # so the threshold scales with it to keep "u per second of decision".
    STALL_EPS_T = TICK.per_tick(args.stall_eps)
    print(f"tick: {TICK.describe()}")
    if not TICK.is_reference:
        print(f"tick: gamma {args.gamma:g}/10ms -> {GAMMA_T:.8f}/tick "
              f"(horizon {1.0 / (1.0 - GAMMA_T) * TICK.ms / 1000.0:.1f} s "
              f"unchanged); time_pen {args.time_pen:g} -> {TIME_PEN_T:.6g}/tick; "
              f"stall_eps {args.stall_eps:g} -> {STALL_EPS_T:.4g} u/call; "
              f"stall {args.stall_secs:g} s = "
              f"{TICK.secs_to_ticks(args.stall_secs)} ticks; respawn margin "
              f"{args.respawn_margin:g} s = "
              f"{TICK.secs_to_ticks(args.respawn_margin)} ticks")
    print(f"tick: decisions every {KH} tick(s) = {KH * TICK.ms:.1f} ms "
          f"({1000.0 / (KH * TICK.ms):.1f} Hz; --act-every is NOT rescaled "
          f"with the tick), episodes capped at {args.ep_ticks} ticks = "
          f"{TICK.ticks_to_secs(args.ep_ticks):.1f} s")
    if tick_sched is not None:
        _END = TickClock(tick_sched.to_ms)
        print(f"tick schedule: {tick_sched.describe()}")
        print(f"tick schedule: {len(tick_sched.pattern_changes())} pattern "
              f"re-derivations over the ramp (a new integer-ms pattern "
              f"whenever the request has moved more than "
              f"{PATTERN_TOL_MS:g} ms); each is logged once")
        print(f"tick schedule: AT THE END - gamma "
              f"{_END.gamma(args.gamma):.8f}/tick, time_pen "
              f"{_END.per_tick(args.time_pen):.6g}/tick, stall_eps "
              f"{_END.per_tick(args.stall_eps):.4g} u/call, stall "
              f"{args.stall_secs:g} s = {_END.secs_to_ticks(args.stall_secs)}"
              f" ticks, respawn margin {args.respawn_margin:g} s = "
              f"{_END.secs_to_ticks(args.respawn_margin)} ticks, decisions "
              f"every {KH} tick(s) = {KH * _END.ms:.1f} ms "
              f"({1000.0 / (KH * _END.ms):.1f} Hz)")
        # the three quantities that CANNOT follow the ramp: they live in the
        # SurfEnvConfig the core copies at surf_create and the C API exposes
        # only surf_set_msec. Frozen at the LAUNCH tick, which is what keeps
        # the first step of the run identical to an unscheduled one.
        print(f"tick schedule: FROZEN, no C setter (surf_create COPIES the "
              f"SurfEnvConfig and only surf_set_msec mutates it) - "
              f"episode cap {args.ep_ticks} ticks "
              f"({TICK.ticks_to_secs(args.ep_ticks):.1f} s now, "
              f"{_END.ticks_to_secs(args.ep_ticks):.1f} s at the end of the "
              f"ramp - the cap is sized at the ramp's SHORTEST tick unless "
              f"--ep-ticks names a count), and the "
              f"yaw / pitch ceilings stay in deg PER TICK, so deg/SECOND "
              f"rises {tick_sched.from_ms / _END.ms:.3f}x over the ramp - "
              f"the yaw bin "
              f"keeps the meaning a warm-resumed policy learned, which is "
              f"the point of ramping rather than switching")
    NCODES = 0
    if args.codes is None:
        args.codes = KCODES
    if args.dec_ent is None:
        args.dec_ent = 5e-4
    if args.codebook_bias is None:
        args.codebook_bias = 3.0
    if H > 0:
        NCODES = max(2, int(args.codes))
        if args.ez_eps > 0.0 or args.spawn_burst > 0:
            # both draw a uniform random SIX-TUPLE and freeze it; under
            # --chunk the policy's sample is a code, and a frozen 6-tuple is
            # neither on-policy nor even representable in b_act. A random
            # CODE is already 300 ms of committed coherent behavior sampled
            # from pi, which is what ez-greedy was manufacturing (design doc
            # 6.4) — so run the chunked arm with --ez-eps 0.
            raise SystemExit("--chunk is incompatible with --ez-eps / "
                             "--spawn-burst: those hold a frozen random "
                             "6-tuple, which the code head cannot express "
                             "and PPO could not score. A random CODE already "
                             "IS a temporally-extended on-policy sample")
        if args.yaw_adaptive:
            # --yaw-adaptive makes bin b mean k*atan(30/|v|) instead of a
            # fixed deg/tick. That is a different action space, so a decoder
            # (or a --codebook init fitted on the fixed ladder) means
            # something else in it. Separate arm, separate screen.
            raise SystemExit("--chunk with --yaw-adaptive is a second, "
                             "confounded experiment (design doc 3.5): the "
                             "yaw bin's meaning changes underneath the "
                             "decoder. Run them on separate screens")
        print(f"chunk {H}: {NCODES} codes x {H} decisions, decoder learned "
              f"end-to-end; {KH} ticks per policy decision, "
              f"gamma_eff = gamma**{KH}, ent(code) {args.ent:g}, "
              f"dec-ent {args.dec_ent:g}")
        for _i, _n in enumerate(NVEC):
            assert 0 <= NEUTRAL_ACT[_i] < _n, "NEUTRAL_ACT out of range"
    elif args.chunk is not None and args.chunk != 0:
        raise SystemExit(f"--chunk {args.chunk}: H must be >= 1 (0 = off)")

    device = D.device
    torch.backends.cudnn.benchmark = True
    # rank 0 owns the one TIMING line per iteration tools/perf_report.py
    # parses; interleaved lines from four ranks would silently corrupt
    # every measurement the frozen benchmark protocol depends on
    tm = PhaseTimer(args.timing and D.is_main, device.type == "cuda")
    use_graphs = device.type == "cuda" and not args.no_graphs
    use_compile = device.type == "cuda" and not args.no_compile
    use_bf16 = device.type == "cuda" and not args.fp32
    # --envs is the GLOBAL fleet; each rank simulates its 1/world_size
    # share. "Each GPU gets 2048" would be 8192 globally - the repo's own
    # ablation measures that as rew-20 at 98M steps vs 52M, a sample-
    # efficiency regression that presents as a throughput win (plan §6.4).
    N_GLOBAL, T = args.envs, args.n_steps
    if N_GLOBAL % D.world_size:
        raise SystemExit(f"--envs {N_GLOBAL} (GLOBAL fleet) is not "
                         f"divisible by world_size {D.world_size}")
    N = N_GLOBAL // D.world_size
    if (T * N) % args.minibatches:
        raise SystemExit(f"per-rank rollout T*N = {T}*{N} = {T * N} is not "
                         f"divisible by --minibatches {args.minibatches}")
    if RNN and N % args.minibatches:
        # --rnn: a minibatch is N/M whole ENV sequences of T decisions,
        # not T*N/M shuffled rows (see the update); the count keeps its
        # meaning, the unit of shuffling changes
        raise SystemExit(f"--rnn splits the {N} per-rank envs into "
                         f"--minibatches {args.minibatches} groups of whole "
                         "sequences: --envs must divide by it")
    if D.enabled:
        if not (flag_given("--run") or flag_given("--run-name")):
            # the default is time.strftime evaluated PER PROCESS - ranks
            # launched across a minute boundary derive different run dirs
            raise SystemExit("DDP needs an explicit --run-name "
                             "(the timestamp default is per-process)")
        if args.rnd_coef > 0.0:
            raise SystemExit("--rnd-coef under DDP is not implemented: the "
                             "RND predictor trains per-rank and the ranks' "
                             "intrinsic bonuses diverge")
        if args.warm_caches:
            raise SystemExit("--warm-caches is a SINGLE-process pre-pass - "
                             "run it before torchrun (tools/ddp_launch.sh "
                             "does)")

    # ---- map slots (--maps): one core per map, envs split evenly ----------
    # Everything map-shaped lives in a slot: core, lidar, goal field, reward
    # (whose scale is 100/d0 of THAT map), spawn pool, respawn reservoir,
    # novelty counts, d0, goal box - plus the contiguous env range it owns.
    # With ONE slot every path below is the pre---maps trainer expression for
    # expression; surfgym/mapfleet.py short-circuits the aggregation and
    # tests/python/test_multimap.py pins the result bit for bit.
    #
    # HOW --maps AND DDP COMPOSE (the integration's central decision).
    # They are NESTED, not competing: --envs is the GLOBAL fleet, DDP cuts it
    # into world_size rank-shares, and --maps cuts each rank's share into
    # NMAPS slots. Every rank holds EVERY map, with
    #
    #     PER = envs / (world_size * maps)
    #
    # envs in each of its slots. Maps are REPLICATED across ranks, not
    # sharded, and that is a deliberate reversal of the plan's gate E:
    #  * RAM stopped forcing it. Measured 2026-08-24, the whole 110-map field
    #    set at the gated cells is 3.69 GB, i.e. 0.46 GB/rank over 8 even
    #    unsharded, against ~63 GB of headroom per rank (plan, "BAKED").
    #  * Replication is what makes the aggregate metric cheap and correct.
    #    Every rank holds every slot, so every cross-rank sync - novelty
    #    counts, the respawn ring, the race stat counters, the per-map eval
    #    table - is a FIXED-SHAPE all-reduce over NMAPS. Sharded maps would
    #    make each of those a variable-length gather over a per-rank map
    #    subset, and a slot that exists on one rank only cannot use a
    #    world-wide collective at all without a subgroup per map.
    #  * It keeps the gradient batch map-BALANCED. Under sharding a rank's
    #    whole minibatch comes from one map, so the per-minibatch advantage
    #    normalisation (which IS all-reduced, plan §2) would be centring a
    #    fleet whose composition differs rank to rank.
    # The cost is NMAPS cores + NMAPS lidar SDFs resident per rank instead of
    # NMAPS/world_size. Sharding stays available as an optimisation if a
    # future pool makes that bite; nothing below assumes replication except
    # the fixed shapes, which degrade to a gather.
    if args.maps:
        _names = [b.strip() for b in str(args.maps).split(",") if b.strip()]
        if not _names:
            raise SystemExit("--maps was passed but lists no maps")
    else:
        _names = [args.map]

    def _resolve_bsp(name):
        p = Path(name)
        if p.suffix.lower() != ".bsp":
            p = p.with_name(p.name + ".bsp")
        if not p.exists() and not p.is_absolute():
            q = ROOT / "maps" / p.name
            if q.exists():
                p = q
        if not p.is_file():
            raise SystemExit(f"--maps: no such BSP for {name!r} (tried {p})")
        return p

    BSPS = [_resolve_bsp(b) for b in _names]
    STEMS = [p.stem for p in BSPS]
    if len(set(STEMS)) != len(STEMS):
        raise SystemExit(f"--maps lists the same map twice: {STEMS}")
    NMAPS = len(BSPS)
    MULTI = NMAPS > 1
    if MULTI:
        # Every one of these is a single-map artifact whose meaning does not
        # survive a second map. Refusing loudly beats training an arm whose
        # shaping silently means something different on each half of the
        # fleet.
        if args.reward != "race":
            raise SystemExit(
                f"--maps needs --reward race: the per-map reward scale is "
                f"100/d0 and d0 comes from the goal field, which only the "
                f"race objective builds (got --reward {args.reward})")
        if args.route:
            raise SystemExit("--route is ONE map's reference line; a "
                             "multi-map run would feed the other maps a fan "
                             "sampled from geometry they do not have")
        if args.demo_file:
            raise SystemExit("--demo-file is ONE map's demo spine (raw map "
                             "coordinates); it cannot seed a second map")
        if args.race_dfloor > 0.0:
            raise SystemExit(
                "--race-dfloor is an ABSOLUTE distance and maps differ in "
                "size by 5x here, so one number is two different treatments "
                "(6,996u is 3.5% of cannonball's d0 and 20% of "
                "petrus_lite's). Single-map only until it grows a "
                "fractional form, like --race-latch-frac")
        if args.race_latch > 0.0:
            raise SystemExit(
                "--race-latch is an ABSOLUTE distance: on a multi-map run "
                "use --race-latch-frac, which is a fraction of each map's "
                "own d0")
        if N % NMAPS:
            raise SystemExit(
                f"--envs {N_GLOBAL} over {D.world_size} rank(s) is {N} env(s)"
                f" per rank, which does not divide evenly over {NMAPS} maps "
                f"({N / NMAPS:.2f} each). Pick a multiple of "
                f"{D.world_size * NMAPS} - silently truncating would leave "
                "one map with fewer envs than its logs claim")
    PER = N // NMAPS
    # --heldout-maps: parsed and guarded HERE, before any core, field or SDF
    # is loaded, so a bad combination fails in a second rather than after a
    # multi-minute cache pass. The slots themselves are built after the
    # training slots (search "EVAL-ONLY slots").
    HBSPS, HSTEMS, HELD_CELLS = [], [], []
    if args.heldout_maps:
        if args.reward != "race":
            raise SystemExit("--heldout-maps needs --reward race (the "
                             "held-out metric is geodesic progress on that "
                             "map's own field)")
        for _flag, _val in (("--goals", args.goals), ("--route", args.route),
                            ("--demo-file", args.demo_file),
                            ("--race-arc", args.race_arc)):
            if _val:
                raise SystemExit(
                    f"--heldout-maps cannot combine with {_flag}: that is "
                    "ONE map's object (goal lines, a reference line, a demo "
                    "spine) and a held-out map has none of it")
        _hnames = [b.strip() for b in str(args.heldout_maps).split(",")
                   if b.strip()]
        if not _hnames:
            raise SystemExit("--heldout-maps was passed but lists no maps")
        HBSPS = [_resolve_bsp(b) for b in _hnames]
        HSTEMS = [q.stem for q in HBSPS]
        if len(set(HSTEMS)) != len(HSTEMS):
            raise SystemExit(f"--heldout-maps lists the same map twice: "
                             f"{HSTEMS}")
        _dup = sorted(set(HSTEMS) & set(STEMS))
        if _dup:
            raise SystemExit(
                f"--heldout-maps overlaps the TRAINING maps "
                f"({', '.join(_dup)}): a held-out map is one the policy "
                "never trains on")
        if args.heldout_goal_cell in (None, ""):
            HELD_CELLS = [None] * len(HBSPS)
        else:
            _hc = [q.strip() for q in str(args.heldout_goal_cell).split(",")
                   if q.strip()]
            try:
                _hc = [float(v) for v in _hc]
            except ValueError:
                raise SystemExit(f"--heldout-goal-cell "
                                 f"{args.heldout_goal_cell!r} is not a "
                                 "number or a comma-separated list of numbers")
            if len(_hc) == 1:
                HELD_CELLS = _hc * len(HBSPS)
            elif len(_hc) == len(HBSPS):
                HELD_CELLS = _hc
            else:
                raise SystemExit(
                    f"--heldout-goal-cell lists {len(_hc)} cells for "
                    f"{len(HBSPS)} held-out maps; pass one value or exactly "
                    "one per map, in --heldout-maps order")
    # --goal-cell: one value for the whole fleet, or one PER MAP in --maps
    # order. Per-map is not a nicety - the coarsening gate is per map and
    # 21 of the 110 usable maps TUNNEL at cell 48 (the wavefront flows
    # through thin floors, d0 collapses, and the field stops being a
    # progress coordinate at all: surf_texture reads a d0 ratio of 0.134).
    # A single global value therefore either wastes 3.3x of bake and RAM on
    # the 89 that are fine, or silently ships a nonsense field for the 21
    # that are not.
    if args.goal_cell in (None, ""):
        GOAL_CELLS = [None] * NMAPS
    else:
        _gc = [p.strip() for p in str(args.goal_cell).split(",") if p.strip()]
        try:
            _gc = [float(v) for v in _gc]
        except ValueError:
            raise SystemExit(f"--goal-cell {args.goal_cell!r} is not a "
                             "number or a comma-separated list of numbers")
        if len(_gc) == 1:
            GOAL_CELLS = _gc * NMAPS
        elif len(_gc) == NMAPS:
            GOAL_CELLS = _gc
        else:
            raise SystemExit(
                f"--goal-cell lists {len(_gc)} cells for {NMAPS} maps; pass "
                "one value for the whole fleet or exactly one per map, in "
                "--maps order")
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    if args.ckpt is None:
        # A FRESH launch into a directory that already holds a run appends
        # to its progress.csv (the step folds back to 0: two lines on every
        # dashboard plot), overwrites its ckpt_latest.pt and mixes its
        # trajectories. Reused names did exactly this three times
        # (2026-09-02). Refuse; a resume (--ckpt) is the only legal way in.
        _held = [p.name for p in (out / "progress.csv", out / "ckpt_latest.pt")
                 if p.exists()]
        _held += sorted(q.name for q in out.glob("traj_*.jsonl"))[:1]
        if _held:
            raise SystemExit(
                f"runs/{args.run} already holds a run ({', '.join(_held)}): a "
                f"fresh launch would append to its progress.csv and draw a "
                f"second line on every plot. Pick a fresh --run, or resume "
                f"it with --ckpt runs/{args.run}/ckpt_latest.pt.")

    # cores run EYELESS (13M raw steps/s); vision is rendered on the GPU from
    # the map SDF and fused into the obs here in the trainer
    if args.fix_pitch is not None and args.pitch_fixed is not None:
        raise SystemExit("--fix-pitch and --pitch-fixed both freeze the gaze "
                         "and disagree about at WHAT (spawn value vs a pinned "
                         "angle) - pass exactly one")
    if args.fix_pitch is not None or args.pitch_fixed is not None:
        # both make the pitch head inert; --pitch-fixed additionally pins the
        # column before every render (see the flag's note)
        pitch_rate = 0.0
    else:
        pitch_rate = args.pitch_rate if args.pitch_rate is not None else -1.0
    # --tick-ms: the view rates are deg PER TICK in the core (env.c scales
    # its bins by yaw_rate_max_deg / 10 and pitch_rate_max_deg / 10), so at
    # another tick both are rescaled to keep the same deg per SECOND. 0 stays
    # 0 (frozen gaze); -1 (core default 10) becomes the explicit 10 * scale.
    # At the reference tick the core receives exactly what it always did.
    # --tick-ms-schedule: whatever of these the tick does reach is baked
    # into the core at surf_create and cannot follow the ramp, so it is
    # anchored to the ramp's START rather than to this launch's tick - a
    # crash-resume half way down a ramp then keeps exactly the deg-per-tick
    # ladder the run has had all along, which is the continuity the ramp
    # exists to preserve.
    VIEW_TICK = TICK if tick_sched is None else TickClock(tick_sched.from_ms)
    if VIEW_TICK.is_reference:
        pitch_rate_core = pitch_rate
        _tick_env = {}
    else:
        pitch_rate_core = (0.0 if pitch_rate == 0.0
                           else VIEW_TICK.per_tick(10.0 if pitch_rate < 0 else pitch_rate))
        # --yaw-adaptive redefines a yaw bin as K_BINS * atan(30/|v|) - the
        # optimal-strafe angle per FRAME, which does NOT depend on the tick.
        # yaw_rate_max_deg is then only (a) a per-tick clamp and (b) the
        # divisor of obs column 10 (env.c: last_yd / yaw_rate_max_deg), so
        # scaling it buys no constant deg/s (MEASURED: the yaw delta is
        # bit-identical at 800 and 2000 u/s, because the clamp does not bind
        # above ~223 u/s) and DOES multiply the action-echo observation by
        # 10/tick for the same action at the same speed. Fixed bins scale;
        # adaptive bins keep the reference ceiling.
        _tick_env = ({} if args.yaw_adaptive
                     else {"yaw_rate_max_deg": VIEW_TICK.per_tick(10.0)})
        _yaw_note = ("yaw bins are adaptive (K_BINS * atan(30/|v|), tick-free);"
                     " ceiling stays 10 deg/tick"
                     if args.yaw_adaptive else
                     f"yaw {_tick_env['yaw_rate_max_deg']:.4g} deg (1000 deg/s)")
        print(f"tick: view rates per tick -> {_yaw_note}, "
              f"pitch {pitch_rate_core:.4g} deg "
              f"({pitch_rate_core * 1000.0 / VIEW_TICK.ms:.4g} deg/s)"
              + ("" if tick_sched is None else
                 f" - anchored to the ramp's START tick "
                 f"{tick_sched.from_ms:g} ms, not to this launch's "
                 f"{TICK.requested_ms:g} ms"))

    slots = []
    for _i, _bsp in enumerate(BSPS):
        cfg = default_config(num_envs=PER, spawn_mode=2,
                             max_episode_ticks=args.ep_ticks,
                             water_fail=1, yaw_jitter_deg=args.yaw_jitter,
                             yaw_adaptive=1 if args.yaw_adaptive else 0, yaw_blend=float(args.yaw_blend), side_hold_ticks=int(args.side_hold),
                             sv_maxvelocity=args.maxvel,
                             lidar_w=0, lidar_h=0,
                             pitch_rate_max_deg=pitch_rate_core, **_tick_env,
                             **_view_env)
        core = SurfCore(str(_bsp), cfg, tick_ms=args.tick_ms)
        slot = MapSlot(_bsp.stem, str(_bsp), core, _i * PER, (_i + 1) * PER)
        # the vision voxel size is a property of the MAP (pick_cell reads its
        # bounds), so a 5x smaller map gets its own, finer grid unless
        # --lidar-cell pins one globally
        slot.cell = args.lidar_cell or pick_cell(core)
        # --goal-cell decouples the SHAPING field's voxel size from the
        # LIDAR's. They were one variable, but they do unrelated jobs: the
        # lidar cell is perception fidelity (depth error ~ voxel size),
        # while the goal cell is only the reward's spatial resolution, read
        # trilinearly out of a smooth distance function. Coarsening the
        # field is cubically cheap (cell 48 is 3.3x fewer voxels than 32),
        # which is what makes a 47-map fleet fit in RAM - and it is an open
        # question whether a coarser field TRAINS better, since the 88%
        # wall was an interior minimum that a smoother field may not have.
        # Not free, though: measured on cannonball, cell 64 lets the
        # wavefront tunnel through thin floors and d0 halves.
        slot.goal_cell = GOAL_CELLS[_i] or slot.cell

        # race objective: labeled finish zone + geodesic distance-to-finish
        # field. goal_field = the STANDARD field (eval progress, comparable
        # across arms); reward_field = what the shaping actually uses (==
        # goal_field unless --race-kill-aware masks fail volumes in).
        goal_field = None
        reward_field = None
        goal_box = None
        race_d0 = None
        rf_d0 = None
        if args.reward == "race":
            with D.rank0_first():   # zones.json may be auto-extracted+written
                zones = load_zones(str(_bsp))
            if not zones.get("end"):
                raise SystemExit(
                    f"--reward race needs an end zone for {_bsp.stem}: "
                    f"auto-extraction found none — hand-label "
                    f"maps/{_bsp.stem}.zones.json (see surfgym/zones.py)")
            goal_box = zones["end"]
            # TRIGGER or BUTTON, and the aggregate metric reports the two
            # split as well as pooled. A type-1 map's finish is an invisible
            # trigger_multiple curtain (median face 808,960 u^2); types 2/3
            # are a +use button whose box is padded by 64 u and roughly 8x
            # smaller in face area, and THE SIMULATOR CANNOT PRESS A BUTTON
            # at all - `func_button` is only in the solid list, so arriving
            # inside the box is substituted for the press (CLAUDE.md 4b).
            # Both the gateway service and the in-BSP func_button fallback
            # emit `true_aabb` (the unpadded ground truth); a real trigger
            # brush has no such key.
            slot.finish_kind = ("button"
                                if (goal_box.get("true_aabb") is not None
                                    or goal_box.get("from") == "func_button"
                                    or zones.get("source") == "gateway")
                                else "trigger")
            if args.race_dist == "euclid":
                from surfgym.goalfield import EuclidField
                goal_field = EuclidField(goal_box)
                if args.race_kill_aware and _i == 0:
                    print("--race-kill-aware needs the geodesic field; "
                          "ignored under --race-dist euclid")
            else:
                # THE FIELD IS SEEDED FROM THE BOX THAT IS ARMED. Do not
                # "fix" this to seed from `true_aabb` - that was tried and
                # reverted, and here is why it is wrong.
                #
                # Arrival is `seg_hits_box(prev_org, origin, goal box)`
                # (env.c:595) against the box `set_goal_box` arms below -
                # the PADDED one. So every free voxel inside the padded box
                # already ends the episode. Seeding the wavefront anywhere
                # smaller makes the field claim you are still up to 192 u
                # out from a place where you have in fact already won.
                #
                # The counter-argument was that a 192 u pad reaches through
                # a thin wall: of 130 button-finish failures, 42 have a real
                # seal, median 48 u, and 29 of those are <= 128 u. True, but
                # it is not a false positive - the trigger reaches through
                # that wall too, so the map really is finishable by touching
                # the near side. That is the user's own "inflate the button
                # substantially and make it on-touch" decision behaving as
                # specified. If a particular map should not finish through
                # its wall, shrink the ARMED box for that map; never split
                # the two, or the field stops describing the finish.
                #
                # A seed voxel inside solid cannot leak either way:
                # _bfs_geodesic does `seed & ~solid` (goalfield.py:342).
                #
                # Splitting them also re-keys every cache - the seed box is
                # part of the signature - which stranded all 65 button-map
                # fields in the shipped pool and cost a re-bake.
                #
                # rank 0 bakes while the others wait on the cache: NMAPS
                # concurrent 9-11 GB Bellman-Ford bakes on one card is an
                # OOM, and racing npz writes tear the file (plan 6.13).
                # tools/ddp_launch.sh runs --warm-caches once out of band
                # so this is the fallback, not the plan.
                with D.rank0_first():
                    goal_field = build_goal_field(core, goal_box,
                                                  cell=slot.goal_cell,
                                                  device=device)
                if args.race_kill_aware:
                    with D.rank0_first():
                        reward_field = build_goal_field(core, goal_box,
                                                        cell=slot.goal_cell,
                                                        device=device,
                                                        mask_kill=True)
            if reward_field is None:
                reward_field = goal_field
            core.set_goal_box(goal_box["mins"], goal_box["maxs"])
            if args.keep_teleports:
                print("race mode forces the teleport-ends-episode rule "
                      "(--keep-teleports ignored: fallers respawn at start)")
                args.keep_teleports = False

        if args.reward == "race":
            # game-authentic race starts: the map's own spawn points, facing
            # along the track (entity yaw on ported maps is unreliable)
            raw = map_spawn_pool(core)
            # spawn yaw deliberately from the STANDARD field even under S2,
            # so spawn conditions stay identical across arms
            plat_pool = map_spawn_pool(core,
                                       yaw=goal_field.descent_yaw(raw["origin"]))
            race_d0 = float(np.mean(goal_field.sample(raw["origin"])))
            rf_d0 = race_d0
            if reward_field is not goal_field:
                # masking the fail nets must not disconnect the route: if the
                # start can no longer reach the finish, the voxelized route
                # model is wrong (a sealed gap?) and the arm must not run
                if not reward_field.reachable(raw["origin"]).all():
                    raise SystemExit(
                        "kill-aware goal graph disconnects the start from the "
                        "finish — investigate the route model before running "
                        "--race-kill-aware")
                rf_d0 = float(np.mean(reward_field.sample(raw["origin"])))
                print(f"race: kill-aware start geodesic {rf_d0:.0f}u "
                      f"(standard {race_d0:.0f}u)")
            print(f"race{f'[{_bsp.stem}]' if MULTI else ''}: start geodesic "
                  f"{race_d0:.0f}u, finish box {goal_box['mins']} .. "
                  f"{goal_box['maxs']}")
        else:
            plat_pool = platform_spawn_pool(core)
        # platform starts gaze slightly down regardless of pitch mode
        plat_pool["pitch"] = (args.fix_pitch if args.fix_pitch is not None
                              else args.pitch_fixed
                              if args.pitch_fixed is not None else -10.0)
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
                if reward_field is not None and reward_field is not goal_field:
                    # under --race-kill-aware a spawn inside a masked volume
                    # earns no shaping until it leaves; filter on BOTH fields
                    keep &= reward_field.reachable(dp["origin"])
                print(f"race: drop pool {len(dp)} -> {int(keep.sum())} on-track")
                dp = dp[keep]
            if args.spawn == "ramp":
                pool = dp
            else:
                # --spawn mixed concatenates the two pools, and the env resets
                # by UNIFORM pool draw, so entry counts are the probabilities:
                # a ~18-entry map-spawn pool next to a multi-thousand drop pool
                # means the agent essentially never starts at the start line.
                # Replicate the start entries to hit the requested drop
                # fraction.
                f = float(np.clip(args.drop_frac, 0.01, 0.99))
                reps = max(1, int(round(len(dp) * (1.0 - f)
                                        / (f * max(len(plat_pool), 1)))))
                pool = np.concatenate([np.concatenate([plat_pool] * reps), dp])
                print(f"race: start entries x{reps} -> drop fraction "
                      f"{len(dp) / len(pool):.2f} (requested {f:.2f})")
        if not args.keep_teleports:
            core.set_teleport_fail(True)
        core.set_spawn_pool(pool)
        slot.goal_field = goal_field
        slot.reward_field = reward_field
        slot.goal_box = goal_box
        slot.d0 = race_d0
        slot.rf_d0 = rf_d0
        slot.pool = pool
        slot.plat_pool = plat_pool
        if MULTI:
            print(f"  slot {_i}: {_bsp.stem} envs [{slot.lo}, {slot.hi}) "
                  f"pool({args.spawn}) {len(pool)} cell {slot.cell:g}")
        slots.append(slot)

    # a few aliases onto slot 0 for the code that is genuinely run-wide
    # (obs_dim, the "is this a race run" guards, the startup banner)
    goal_field = slots[0].goal_field
    core = slots[0].core
    pool = slots[0].pool
    print(f"pool({args.spawn}) {len(pool)} | envs {N} | {device} | "
          f"omp={os.environ['OMP_NUM_THREADS']} | "
          f"graphs={use_graphs} bf16={use_bf16}"
          + (f" | maps {NMAPS} x {PER} envs" if MULTI else "")
          + (f" | pitch fixed {args.fix_pitch:g}" if args.fix_pitch is not None else "")
          + (f" | pitch PINNED {args.pitch_fixed:g} deg every render"
             if args.pitch_fixed is not None else ""))
    if args.respawn_frac > 0.0:
        # a reservoir per map: its states are RAW MAP COORDINATES, so a state
        # harvested on one map spawns inside solid geometry (or the void) on
        # any other. Same reason the checkpointed reservoir carries a map_id.
        for _i, slot in enumerate(slots):
            _c = slot.core
            if args.respawn_mode != "uniform" and slot.reward_field is None:
                raise SystemExit(f"--respawn-mode {args.respawn_mode} needs the "
                                 "race goal field (--reward race)")
            binned = ((bool(args.respawn_binned) or args.respawn_mode != "uniform")
                      and slot.reward_field is not None)
            bin_field, bin_d0 = slot.reward_field, slot.rf_d0
            if binned and args.respawn_mode != "uniform" and args.respawn_killsafe:
                bin_field = build_goal_field(_c, slot.goal_box,
                                             cell=slot.cell, mask_kill=True)
                raw = map_spawn_pool(_c)
                if not bin_field.reachable(raw["origin"]).all():
                    raise SystemExit(
                        "kill-masked binning field disconnects the start from "
                        "the finish — bad route model, refusing to run")
                bin_d0 = float(np.mean(bin_field.sample(raw["origin"])))
                print(f"respawn killsafe: binning on the kill-masked field "
                      f"(start geodesic {bin_d0:.0f}u vs standard "
                      f"{slot.d0:.0f}u); fail-floor states unsampleable")
            slot.respawn = RespawnBuffer(
                slot.n, reservoir=args.respawn_reservoir,
                margin_ticks=TICK.secs_to_ticks(args.respawn_margin),
                map_id=slot.name,
                dist_fn=bin_field.sample if binned else None,
                dist_max=bin_d0 if binned else None,
                dist_valid_max=(getattr(bin_field, "_valid_max", None)
                                if binned else None),
                bins=args.respawn_bins,
                mode=args.respawn_mode,
                goal_k=((TICK.secs_to_ticks(args.goal_kmin, "round"),
                         TICK.secs_to_ticks(args.goal_kmax, "round"))
                        if args.goals else None),
                goal_min_dist=(2.5 * float(args.goal_radius)
                               if args.goals else 0.0),
                # goal arms: successful episodes are ~1.5 s, so a 1 s
                # snapshot cadence never harvests the goal-entry state
                # (the deepest one); 0.25 s does
                snap_every=TICK.secs_to_ticks(0.25 if args.goals else 1.0,
                                              "round"),
                min_speed=float(args.respawn_min_speed or 0.0),
                seed=23 + 101 * _i)
        respawn = slots[0].respawn
        print(f"respawn: {args.respawn_frac:.0%} of episodes from mid-run "
              f"snapshots, harvested >= {args.respawn_margin:g}s before "
              f"episode end"
              + (f", {args.respawn_mode} over {respawn.bins} distance bins"
                 if respawn.dist_fn is not None else ""))
    else:
        respawn = None
    demo = None
    if args.demo_file:
        demo = DemoCurriculum(np.load(args.demo_file),
                              window=args.demo_window, rate=args.demo_rate,
                              min_ep=args.demo_min_ep)
        print(f"demo curriculum: {demo.n} states from {args.demo_file}, "
              f"window {args.demo_window}, advance/backoff at "
              f"{args.demo_rate:.0%} window finish rate "
              f"(demo replaces the reservoir share of the pool)")

    # eval on the game-authentic platform start regardless of the training
    # pool, so eval/* metrics and recordings stay comparable across runs.
    # One eval core PER MAP: race/eval_progress is a per-map number and a
    # shared core could only ever measure one of them.
    #
    # Under DDP the eval is SHARDED OVER MAPS, round-robin by rank, and the
    # per-map results are all-reduced back so every rank ends the eval
    # holding the whole table (the aggregate the user reads must be right on
    # every rank, not rank 0's slice). Rank r therefore builds eval cores
    # only for the maps it owns - a second SurfCore per map on every rank is
    # the one place replication would actually cost real memory.
    for _i, slot in enumerate(slots):
        slot.eval_rank = _i % D.world_size
        if not D.enabled:
            slot.eval_rank = 0
        # --warm-caches builds artifacts and exits; an eval core loads the
        # BSP a second time and evaluates nothing. One extra load is noise
        # at one map and 107 of them is not.
        if slot.eval_rank != D.rank or args.warm_caches:
            slot.eval_core = None
            continue
        ec = SurfCore(slot.bsp, default_config(
            num_envs=1, spawn_mode=2, max_episode_ticks=args.ep_ticks,
            water_fail=1,
            yaw_adaptive=1 if args.yaw_adaptive else 0, yaw_blend=float(args.yaw_blend), side_hold_ticks=int(args.side_hold),
            sv_maxvelocity=args.maxvel,
            lidar_w=0, lidar_h=0, pitch_rate_max_deg=pitch_rate_core,
            **_tick_env, **_view_env), tick_ms=args.tick_ms)
        if not args.keep_teleports:
            ec.set_teleport_fail(True)
        if slot.goal_box is not None:
            ec.set_goal_box(slot.goal_box["mins"], slot.goal_box["maxs"])
        ec.set_spawn_pool(slot.plat_pool)
        slot.eval_core = ec
    _raw_lidar = {}
    for slot in slots:
        with D.rank0_first():        # vision SDF npz build/write
            slot.lidar = GpuLidar(slot.core, args.lidar_w, args.lidar_h,
                                  hfov_deg=float(args.lidar_hfov),
                                  vfov_deg=float(args.lidar_vfov),
                                  range_units=args.lidar_range,
                                  near_range=args.lidar_near,
                                  cell=slot.cell,
                                  device=device,
                                  surf_mask=bool(args.surf_mask),
                                  mask_only=(int(args.surf_mask or 0) == 2),
                                  pinhole=bool(args.pinhole),
                                  normals=bool(args.normals))
        _raw_lidar[slot.name] = slot.lidar
        if args.normals:
            print(f"normals: {slot.name} ego-frame unit normal (x fwd, y left, "
                  f"z up) as channels 1..3 of the {args.lidar_w}x{args.lidar_h} "
                  f"image (hfov {args.lidar_hfov:g}, vfov {args.lidar_vfov:g}), "
                  f"grid {slot.lidar.snrm_flat.numel() / 1e9:.2f} GB on "
                  f"{device} -> in_ch {slot.lidar.channels}")
        if args.goals and args.goal_obs in ("ball", "both"):
            # --goal-obs ball: the goal sphere rendered as depth
            # channel 2 in the same camera (surfgym/goalball.py);
            # lidar.channels becomes 2 and in_ch follows, exactly as
            # --surf-mask does
            from surfgym.goalball import GoalBallLidar
            slot.lidar = GoalBallLidar(slot.lidar, slot.n,
                                       radius=float(args.goal_radius),
                                       views=int(args.goal_views))
        mn_b, mx_b = slot.core.map_bounds()
        # map_center is per map: the truncation bootstrap reconstructs a
        # terminal pose from obs slots 12..14 = (pos - centre)/2000, and a
        # shared centre would put it thousands of units off the map it
        # belongs to and render the wrong depth image for V(s_T)
        slot.map_center = ((mn_b + mx_b) / 2.0).astype(np.float32)

    # ---- --heldout-maps: EVAL-ONLY slots ----------------------------------
    # A held-out map is evaluated at every eval and never trained on. It is
    # built exactly like a training slot's EVAL half - its own zones, its own
    # goal field at its own cell, its own platform start pool, its own lidar
    # SDF - and owns NO env rows: the fleet keeps it in a separate list
    # (MapFleet.heldout), so the rollout, the reward, the reservoir, the
    # novelty counts, the render and the truncation bootstrap cannot reach
    # it. The measurement: a policy that learned "a ramp can be surfed"
    # makes progress on a map it has never seen; a map-memoriser does not.
    # --dump-invariants prints heldout_envs (must be 0) and train_envs.
    heldout = []
    if HBSPS:
        for _j, _bsp in enumerate(HBSPS):
            # the 1-env eval core IS the slot's core: there is no training
            # core, which is the whole point
            ec = SurfCore(str(_bsp), default_config(
                num_envs=1, spawn_mode=2, max_episode_ticks=args.ep_ticks,
                water_fail=1, yaw_adaptive=1 if args.yaw_adaptive else 0, yaw_blend=float(args.yaw_blend), side_hold_ticks=int(args.side_hold),
                sv_maxvelocity=args.maxvel, lidar_w=0, lidar_h=0,
                pitch_rate_max_deg=pitch_rate_core, **_tick_env, **_view_env),
                tick_ms=args.tick_ms)
            hs = HeldoutSlot(_bsp.stem, str(_bsp), ec, N)
            hs.cell = args.lidar_cell or pick_cell(ec)
            hs.goal_cell = HELD_CELLS[_j] or hs.cell
            with D.rank0_first():   # zones.json may be auto-extracted+written
                zones = load_zones(str(_bsp))
            if not zones.get("end"):
                raise SystemExit(
                    f"--heldout-maps needs an end zone for {_bsp.stem}: "
                    f"auto-extraction found none - hand-label "
                    f"maps/{_bsp.stem}.zones.json (see surfgym/zones.py)")
            hs.goal_box = zones["end"]
            hs.finish_kind = ("button"
                              if (hs.goal_box.get("true_aabb") is not None
                                  or hs.goal_box.get("from") == "func_button"
                                  or zones.get("source") == "gateway")
                              else "trigger")
            if args.race_dist == "euclid":
                from surfgym.goalfield import EuclidField
                gf = EuclidField(hs.goal_box)
            else:
                # seeded from the ARMED box, like every training slot (see
                # the long note above); rank 0 bakes, the rest read the cache
                with D.rank0_first():
                    gf = build_goal_field(ec, hs.goal_box, cell=hs.goal_cell,
                                          device=device)
            # the STANDARD field for both roles: nothing ever shapes on a
            # held-out map, so --race-kill-aware has nothing to mask here
            hs.goal_field = hs.reward_field = gf
            ec.set_goal_box(hs.goal_box["mins"], hs.goal_box["maxs"])
            if not args.keep_teleports:
                ec.set_teleport_fail(True)
            raw = map_spawn_pool(ec)
            hs.plat_pool = map_spawn_pool(ec,
                                          yaw=gf.descent_yaw(raw["origin"]))
            hs.plat_pool["pitch"] = (args.fix_pitch
                                     if args.fix_pitch is not None
                                     else args.pitch_fixed
                                     if args.pitch_fixed is not None
                                     else -10.0)
            hs.pool = hs.plat_pool
            hs.d0 = hs.rf_d0 = float(np.mean(gf.sample(raw["origin"])))
            ec.set_spawn_pool(hs.plat_pool)
            # eval sharding continues the training maps' round-robin
            hs.eval_rank = ((NMAPS + _j) % D.world_size) if D.enabled else 0
            hs.eval_core = (ec if (hs.eval_rank == D.rank
                                   and not args.warm_caches) else None)
            with D.rank0_first():        # vision SDF npz build/write
                hs.lidar = GpuLidar(ec, args.lidar_w, args.lidar_h,
                                    hfov_deg=float(args.lidar_hfov),
                                    vfov_deg=float(args.lidar_vfov),
                                    range_units=args.lidar_range,
                                    near_range=args.lidar_near,
                                    cell=hs.cell, device=device,
                                    surf_mask=bool(args.surf_mask),
                                    mask_only=(int(args.surf_mask or 0) == 2),
                                    pinhole=bool(args.pinhole),
                                    normals=bool(args.normals))
            mn_b, mx_b = ec.map_bounds()
            hs.map_center = ((mn_b + mx_b) / 2.0).astype(np.float32)
            _rp = _bsp.with_name(f"{_bsp.stem}.route.npz")
            hs.route = _rp if _rp.exists() else None
            print(f"heldout[{_j}]: {_bsp.stem} EVAL-ONLY (never trained)  "
                  f"d0 {hs.d0:.0f}u  cell {hs.cell:g}  goal cell "
                  f"{hs.goal_cell:g}  finish {hs.finish_kind}  "
                  + (f"route {_rp.name} -> corridor MAX logged"
                     if hs.route is not None
                     else "no route file -> field metric only"))
            heldout.append(hs)
    NHELD = len(heldout)
    if args.warm_caches:
        # every map artifact of every slot is now on disk (zones, occupancy,
        # SDF, goal field); the torchrun ranks that follow only ever read
        # caches. Multi-map makes this MORE important, not less: NMAPS
        # concurrent Bellman-Ford bakes on one card is NMAPS x the OOM.
        print(f"warm-caches: map artifacts built for {NMAPS} map(s)"
              + (f" + {NHELD} held-out" if NHELD else "") + " - exiting")
        return
    fleet = MapFleet(slots, heldout=heldout)
    fleet.retag()
    lidar = slots[0].lidar     # FRAME/channels only: every slot renders the
                               # same shape, and the per-slot renderers are
                               # reached through the fleet
    # every buffer below sizes itself off obs_dim, so the image slice widens
    # with the channel count on its own. FRAME is ONE render — the rollout
    # buffer stores that, never the stack (a stack is a gather).
    STACK = max(1, int(args.frame_stack))
    PRO = max(frame_offsets(STACK))     # prologue rows; 0 when stacking is off
    FRAME = args.lidar_w * args.lidar_h * lidar.channels
    img_ch = lidar.channels * STACK
    # --route: the lookahead fan is a SCALAR-side block, [15 core | R | image]
    route = None
    if args.route:
        rp = Path(args.route)
        if not rp.exists():
            rp = ROOT / args.route
        span = 6.0 if args.route_span is None else float(args.route_span)
        npts = 8 if args.route_points is None else int(args.route_points)
        if npts < 1:
            raise SystemExit("--route-points must be >= 1")
        # geometric spread from 1/24 of the span out to the span itself: dense
        # where the next few decisions land, sparse where only the coarse
        # direction of the route is actionable
        offs = tuple(float(span * (24.0 ** (-(npts - 1 - i) / max(1, npts - 1))))
                     for i in range(npts))
        route = RouteLine.load(rp, offsets=offs, device=device)
        print(route.describe())
    if args.goals and args.goal_obs in ("fan", "both"):
        # --goals: the fan rides a PER-ENV line (surfgym.goals.MultiLine),
        # same 27 columns, same math per env - the policy sees "where
        # this episode's goal is" through the observation every racing
        # agent has. Exclusive with --route: one fan slot.
        if route is not None:
            raise SystemExit("--goals and --route are exclusive (one fan)")
        from surfgym.goals import MultiLine
        _lmax = 768
        if args.goal_route:
            _lmax = int(len(np.load(args.goal_route)["route"])) + 8
        route = MultiLine(N, l_max=_lmax, device=device)
        print(route.describe())
    # --race-latch rides the SAME scalar-side block as the route fan: one
    # extra column, concatenated LAST, which is exactly where
    # widen_for_route zero-pads a checkpoint that has never seen it. A
    # core scalar slot would NOT do - Policy.feat_idx is sorted, so a new
    # scalar lands in the middle of the row and the zero-pad silently
    # permutes every existing column.
    if ((args.race_latch > 0.0 or args.race_latch_frac > 0.0)
            and args.reward != "race"):
        raise SystemExit("--race-latch is a term of the race shaping "
                         "reward and does nothing under --reward "
                         f"{args.reward}")
    # --race-latch-frac resolves to a DIFFERENT absolute distance per map
    # (frac * that map's own rf_d0), but it is one observation column either
    # way, so the row width is the same on every slot
    for _s in slots + heldout:
        _s.d_latch = (args.race_latch_frac * _s.rf_d0
                      if args.race_latch_frac > 0.0 else args.race_latch)
    N_LATCH = 1 if (args.race_latch > 0.0
                    or args.race_latch_frac > 0.0) else 0
    N_FAN = route.n_features if route is not None else 0
    # --act-hist / --obs-compass (surfgym/obsaux.py) ride the same scalar-side
    # block, AFTER the latch, so the whole block reads
    # [fan | latch | 6*K history | 5 compass] and each new piece is a
    # TRAILING widen of the one before it - the only growth direction
    # widen_for_route's zero-pad is function-identical for.
    N_HIST = ACT_FEAT * int(args.act_hist or 0)
    N_CMP = CMP_FEAT if args.obs_compass else 0
    N_AUX = N_HIST + N_CMP
    N_ROUTE = N_FAN + N_LATCH + N_AUX
    # column of the --race-latch flag, and the first column of the aux block.
    # With no aux block LATCH_COL is N_SCALAR + N_ROUTE - 1 exactly as before.
    LATCH_COL = N_SCALAR + N_FAN + N_LATCH - 1
    AUX0 = N_SCALAR + N_FAN + N_LATCH
    # --race-arc: a route used by the REWARD, not by the observation. It is a
    # separate object from --route on purpose - the lookahead fan widens the
    # policy's input row and --race-arc must not, or the arm would be moving
    # two things at once.
    arc_line = None
    arc_scale = 0.0
    if args.race_arc:
        if args.reward != "race":
            raise SystemExit("--race-arc needs --reward race")
        if len(slots) > 1:
            raise SystemExit("--race-arc is single-map for now: one line, "
                             "one arc coordinate; the multi-map fleet needs "
                             "per-slot lines (research-plan-goallines.md E3)")
        ap_ = Path(args.race_arc)
        if not ap_.exists():
            ap_ = ROOT / args.race_arc
        arc_line = ArcProgress.load(
            ap_,
            corridor=(1500.0 if args.race_arc_corridor is None
                      else float(args.race_arc_corridor)),
            window=(16 if args.race_arc_window is None
                    else int(args.race_arc_window)))
        # the SAME budget the geodesic term has: 100 total collectible
        # shaping over one start->finish run. 100/d0 -> 100/route_length.
        # Not tuned; derived, so the arm changes the SHAPE of the potential
        # and nothing about its size.
        arc_scale = 100.0 / arc_line.length * args.race_shaping
        print(arc_line.describe()
              + f" -> shaping scale {arc_scale:.6g}/u "
                f"(vs geodesic {100.0 / max(rf_d0 or 1.0, 1.0) * args.race_shaping:.6g}/u)")
    goal_dist_field = None
    if args.goals and args.goal_reward in ("euclid", "geo"):
        # --goal-reward euclid: the shaping potential is the Euclidean
        # distance to THIS env's goal (surfgym.goals.GoalDistField), set on
        # every assignment; the geodesic field keeps its other jobs
        # (respawn bins, eval progress). scale is per unit, not 100/d0.
        if len(slots) > 1:
            raise SystemExit("--goal-reward euclid/geo is single-map for now")
        from surfgym.goals import GoalDistField
        # geo: the per-goal potential is composed from the map's ONE baked
        # field, max(|dF(x) - dF(goal)|, |x - goal|) - no rebake
        goal_dist_field = GoalDistField(
            N, geo=(slots[0].reward_field if args.goal_reward == "geo"
                    else None))
        print(f"goal reward: {goal_dist_field.describe()}, "
              f"{args.goal_euclid_scale:g}/u, + the arrival bonus")
    elif args.goals and args.goal_reward == "arc":
        # --goal-reward arc: arc progress along each env's OWN goal line
        # (the route slice / reached-state segment / chord that the fan
        # shows), corridor-frozen, replacing the geodesic term exactly
        # like --race-arc. One scale for every goal: 100 per L_ref of
        # arc, L_ref = the map line's length (or speed_est * k_cap).
        if args.reward != "race":
            raise SystemExit("--goal-reward arc needs --reward race")
        if len(slots) > 1:
            raise SystemExit("--goal-reward arc is single-map for now")
        if not args.race_shaping:
            raise SystemExit("--goal-reward arc pays through the shaping "
                             "scale: run with --race-shaping 1 (the "
                             "geodesic term is replaced, not added)")
        from surfgym.goalarc import MultiArcProgress
        from surfgym.route import DEFAULT_SPACING
        arc_line = MultiArcProgress(
            N, l_max=(int(len(np.load(args.goal_route)["route"])) + 8
                      if args.goal_route else 768),
            spacing=DEFAULT_SPACING,
            corridor=(1500.0 if args.race_arc_corridor is None
                      else float(args.race_arc_corridor)),
            window=(16 if args.race_arc_window is None
                    else int(args.race_arc_window)))
        if args.goal_route:
            _rl = np.asarray(np.load(args.goal_route)["route"], np.float64)
            _lref = float(np.linalg.norm(np.diff(_rl, axis=0), axis=1).sum())
        else:
            _lref = 1500.0 * float(args.goal_kcap)
        arc_line.length = _lref
        arc_line.source = "goals:" + (Path(args.goal_route).name
                                      if args.goal_route else "segments")
        arc_line.corridor = float(getattr(arc_line, "corridor",
                                          args.race_arc_corridor or 1500.0))
        arc_line.window = int(getattr(arc_line, "window",
                                      args.race_arc_window or 16))
        arc_scale = 100.0 / _lref * args.race_shaping
        print(arc_line.describe() + f" -> goal arc shaping scale "
              f"{arc_scale:.6g}/u (100 per {_lref:,.0f}u)")
    # --act-hist / --obs-compass. Built HERE, after goal_dist_field, because
    # the compass has to follow whichever field RaceReward shapes on: the
    # per-env GoalDistField under --goal-reward euclid/geo (so the compass
    # points at the GOAL), otherwise each map slot's own reward_field. The
    # width N_AUX above needs only args, which is why the two are separated.
    obs_aux = None
    if N_AUX:
        _cmp_field = None
        if args.obs_compass:
            _cmp_field = ([(slice(0, N), goal_dist_field)]
                          if goal_dist_field is not None
                          else [(_s.sl, _s.reward_field) for _s in slots])
            for _sl, _f in _cmp_field:
                if _f is None:
                    raise SystemExit(
                        "--obs-compass: this run has no shaping distance "
                        "field to read (no goal field was built)")
        obs_aux = ObsAux(N, k=int(args.act_hist or 0), field=_cmp_field,
                         yaw_adaptive=bool(args.yaw_adaptive))
        if obs_aux.n_features != N_AUX:          # layout bug, not a user error
            raise SystemExit(f"obs aux block is {obs_aux.n_features} columns, "
                             f"the row was sized for {N_AUX}")
        print(obs_aux.describe() + f" -> obs columns "
              f"{AUX0}..{AUX0 + N_AUX - 1}")
    SCAL = N_SCALAR + N_ROUTE                 # the whole scalar half of a row
    obs_dim = core.obs_dim + N_ROUTE + FRAME * STACK
    REWARD_SLOT = 12          # an absolute-position channel, hidden at gps=False

    def _make_eval_reward_feed(field, scale, time_pen, k, d_floor=0.0,
                               latch_feed=None, ng=0, ng_g=1.0, ng_d0=0.0,
                               max_step=100.0):
        """Mirror the training --obs-reward signal for evaluation rollouts.

        The eval core produces no reward, so this recomputes the same
        potential-shaping term from the goal field: the per-decision
        geodesic progress minus the time cost, squashed identically. Keeps
        its own previous-distance state and re-anchors on a jump (episode
        reset or teleport) so a relocation is not cashed as progress.

        ``time_pen`` and ``ng_g`` may be CALLABLES: under
        --tick-ms-schedule both are per-tick quantities that move with the
        ramp, and a value captured at startup would feed an eval the
        shaping of a tick the training had already left. A float behaves
        exactly as before, and the callable resolves to the same number, so
        the fixed-tick path is bit-identical.
        """
        st = {"d": None}

        def feed(core):
            time_pen_now = time_pen() if callable(time_pen) else time_pen
            ng_g_now = ng_g() if callable(ng_g) else ng_g
            d = field.sample(core.states_view["origin"]).astype(np.float64)
            if d_floor > 0.0:
                # --race-dfloor: slot 12 is the policy's own shaping, so the
                # eval mirror has to be clamped too or an eval feeds a
                # clamp-trained policy the unclamped signal
                d = np.maximum(d, d_floor)
            prev = st["d"]
            st["d"] = d
            if prev is None or len(prev) != len(d):
                return np.zeros(len(d), np.float32)
            delta = np.clip(prev - d, -max_step * k, max_step * k)
            if latch_feed is not None:
                # the flag as of the PREVIOUS decision - the one that
                # governed the reward this slot reports. latch_feed is
                # advanced later in the same _obs call, so reading its
                # state here is reading t-1, which is what training did.
                was = latch_feed.state["f"]
                if was is not None and len(was) == len(delta):
                    delta = np.where(was, 0.0, delta)
            r = delta * scale - time_pen_now * k
            if ng:
                # --race-ng: mirror the conformant tax or an ng-trained
                # policy reads a slot the training never produced. The
                # terminal charge has no mirror (the episode ends there);
                # training zeroes the slot on reset rows to match.
                r = r - (1.0 - ng_g_now) * (ng_d0 - d) * scale
            return np.tanh(r / 0.1).astype(np.float32)

        return feed

    def _make_eval_latch_feed(field, d_latch):
        """Mirror the training --race-latch flag for evaluation rollouts.

        The eval core computes no reward, so nothing there owns the flag;
        without this the policy would be fed a constant 0 in a column it
        was trained to read, in exactly the states where the column is
        the only thing distinguishing two regimes. Episode starts are
        read off the core's per-env tick counter, which reset_env zeroes
        (src/env.c) - the same idiom the frame ring uses."""
        st = {"f": None, "tick": None}

        def feed(core):
            sv = core.states_view
            d = field.sample(sv["origin"]).astype(np.float64)
            tick = np.asarray(sv["tick"], np.int64).copy()
            f, pt = st["f"], st["tick"]
            if f is None or len(f) != len(d) or pt is None:
                f = np.zeros(len(d), bool)
            else:
                f = f & ~(tick <= pt)          # a fresh episode clears it
            f |= d <= d_latch
            st["f"], st["tick"] = f, tick
            return f.astype(np.float32)

        feed.state = st      # the obs-reward mirror reads t-1's flag here
        return feed

    def _make_eval_arc_feed(line, scale, time_pen, k, corridor, window,
                            max_step=100.0):
        """The --race-arc twin of the feed above.

        Under --obs-reward the policy READS its own shaping in scalar slot
        12, so an eval that fed the geodesic term to an arc-trained policy
        would be the exact train/eval mismatch that made sOBSR's evals
        meaningless. This keeps its own ArcProgress (never the trainer's -
        different env count, different episodes) and re-anchors whenever the
        player relocates further than one decision of legal motion, which is
        the eval-side stand-in for the `ended` mask the trainer has.

        ``time_pen`` may be a CALLABLE for the same reason as the feed
        above: --tick-ms-schedule moves the per-tick cost while the run
        runs, and a captured float would freeze the mirror at the tick the
        eval feed was built at.
        """
        arc = ArcProgress(line.pts, line.spacing, corridor=corridor,
                          window=window)
        st = {"p": None}

        def feed(core):
            time_pen_now = time_pen() if callable(time_pen) else time_pen
            p = core.states_view["origin"].astype(np.float64)
            prev = st["p"]
            st["p"] = p.copy()
            if prev is None or len(prev) != len(p):
                arc.reset(p)
                return np.zeros(len(p), np.float32)
            jump = np.linalg.norm(p - prev, axis=1) > max_step * k
            if jump.any():
                arc.reset(p, mask=jump)
            delta, _inside = arc.advance(p)
            delta = np.where(jump, 0.0, delta)
            r = np.clip(delta, -max_step * k,
                        max_step * k) * scale - time_pen_now * k
            return np.tanh(r / 0.1).astype(np.float32)

        return feed

    def _make_eval_priv_feed(priv, field, line=None, latch_feed=None,
                             corridor=1500.0, window=16, max_step=100.0,
                             k=1):
        """The --priv-critic block on an eval core (surfgym/privfeat.py).

        The eval reads only the LOGITS, and no privileged column can reach
        them - so nothing here can change a recorded action. It exists so
        V(s) on the eval path is the critic's actual output instead of the
        NaN Policy._value returns without a privileged row, and so there is
        exactly ONE implementation of the block: the same PrivFeat instance
        the rollout fills, handed the eval core's own numbers.

        The eval core produces no reward, so d is resampled from the field
        and the arc anchor is kept here (its own ArcProgress, like
        _make_eval_arc_feed - a different env count and different episodes)
        with the same re-anchor-on-relocation rule. The latch column reads
        the state _make_eval_latch_feed just wrote for this decision.
        """
        arc = (None if line is None else
               ArcProgress(line.pts, line.spacing, corridor=corridor,
                           window=window))
        st = {"p": None, "out": None}

        def feed(core):
            sv = core.states_view
            p = np.asarray(sv["origin"], np.float64)
            n = len(p)
            if st["out"] is None or len(st["out"]) != n:
                st["out"] = np.zeros((n, priv.dim), np.float32)
            a = None
            if arc is not None:
                prev = st["p"]
                if prev is None or len(prev) != n:
                    arc.reset(p)
                else:
                    jump = np.linalg.norm(p - prev, axis=1) > max_step * k
                    if jump.any():
                        arc.reset(p, mask=jump)
                    arc.advance(p)
                a = arc.arc
            st["p"] = p.copy()
            lt = None
            if latch_feed is not None:
                # the flag as of THIS decision: _obs has already advanced
                # the latch feed by the time _net asks for the priv block,
                # which is the order fill_vision writes the two in
                lt = latch_feed.state["f"]
            priv.fill(st["out"], p, sv["velocity"],
                      field.sample(p).astype(np.float64), sv["tick"],
                      arc=a, latch=lt)
            return st["out"]

        return feed
    # three seed streams, three rank-affinities (docs/ddp-plan.md step 5):
    # (b) policy init rank-COMMON - identical weights everywhere (the
    #     explicit broadcast below is belt-and-braces);
    torch.manual_seed(args.seed)
    policy = Policy(obs_dim, args.lidar_w, args.lidar_h,
                    extra_feat=(REWARD_SLOT,) if args.obs_reward else (),
                    emb=args.emb, hidden=args.hidden, gps=args.gps,
                    in_ch=img_ch, n_codes=NCODES, chunk=H,
                    route_dim=N_ROUTE, trunk=args.trunk,
                    route_critic_only=bool(args.route_critic_only),
                    rnn=args.rnn, rnn_size=args.rnn_size,
                    tower_depth=args.tower_depth,
                    conv_mult=args.conv_mult,
                    fp32_heads=bool(args.fp32_heads),
                    priv_dim=(PRIV_DIM if args.priv_critic else 0),
                    priv_hidden=int(args.priv_hidden),
                    yaw_cond=YCOND, view_continuous=VIEWC,
                    view_absolute=VIEW_ABS).to(device)
    R = policy.rnn_size                    # 0 without --rnn
    # (c) action noise rank-DISTINCT, and set BEFORE the graph capture: the
    #     Gumbel rand_like runs inside the captured graph, whose philox seed
    #     is frozen at capture time - re-seeding after has no effect.
    #     Identical action noise + identical weights + a common env seed
    #     would make every rank simulate bit-identical trajectories with
    #     not one logged number changing (plan §6.1).
    if device.type == "cuda":
        torch.cuda.manual_seed(
            (args.seed * 1000003 + 1 + D.rank) & 0x7FFFFFFFFFFFFFFF)
    # (d) minibatch permutation rank-DISTINCT and deterministic, from its
    #     own generator. NOT rank-common: local index j = t*N+e, so a
    #     common permutation gives global minibatch k the same timestep
    #     multiset from every rank, roughly doubling the sd of its
    #     timestep composition (plan §1 correction 1).
    perm_gen = torch.Generator(device=device)
    perm_gen.manual_seed(
        (args.seed * 2654435761 + 7919 + D.rank) & 0x7FFFFFFFFFFFFFFF)
    packer = HeadPacker(device)
    # --chunk: the in-trainer eval unrolls the same code -> (H, 6) plan the
    # rollout does, one trunk forward per chunk. This path is EAGER (no graph,
    # no compile), so the unroll is a plain held-plan loop. Eval quality gates
    # every verdict, so it mirrors training rather than being disabled.
    EVAL_GREEDY = GreedyChunkPolicy if H > 0 else GreedyTorchPolicy
    EVAL_SAMPLE = SampledChunkPolicy if H > 0 else SampledTorchPolicy
    cb_file = None
    if H > 0 and args.codebook:
        # OPTIONAL warm init only. The decoder is a trainable Parameter that
        # ships in the state_dict, so this file is never needed again — not
        # to resume, not to record. It just moves iteration 0 off uniform
        # noise and onto a repertoire fitted from real trajectories.
        cb_path = Path(args.codebook)
        if not cb_path.exists():
            cb_path = ROOT / args.codebook
        z = np.load(cb_path)
        cb = np.asarray(z["codebook"])
        if cb.ndim != 3 or cb.shape[1] != H or cb.shape[2] != NACT:
            raise SystemExit(f"--codebook {cb_path}: codebook is {cb.shape}, "
                             f"expected (K, {H}, {NACT}) for --chunk {H}")
        if int(cb.shape[0]) != NCODES:
            raise SystemExit(f"--codebook {cb_path} has {cb.shape[0]} codes "
                             f"but --codes is {NCODES}")
        if "nvec" in z.files and tuple(int(v) for v in z["nvec"]) != NVEC:
            raise SystemExit(f"--codebook {cb_path} was fitted for nvec "
                             f"{tuple(int(v) for v in z['nvec'])}, not {NVEC}")
        if (cb < 0).any() or (cb >= np.asarray(NVEC)).any():
            raise SystemExit(f"--codebook {cb_path} holds out-of-range action "
                             "indices for NVEC")
        cb_file = str(cb_path.resolve())
        if ck is not None:
            print(f"--codebook ignored on resume: the checkpoint's own "
                  f"decoder wins ({cb_path.name})")
        else:
            # flat index of head i's bin b inside the sum(NVEC) row
            off = np.concatenate([[0], np.cumsum(NVEC)[:-1]]).astype(np.int64)
            idx = torch.as_tensor(cb.astype(np.int64) + off, device=device)
            with torch.no_grad():
                policy.decoder.scatter_add_(
                    2, idx, torch.full(idx.shape, float(args.codebook_bias),
                                       device=device))
            print(f"decoder initialised from {cb_path.name} "
                  f"(+{args.codebook_bias:g} on {NCODES}x{H}x{NACT} fitted "
                  "indices; still fully trainable)")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5,
                           fused=(device.type == "cuda"))
    rnd = None
    if args.rnd_coef > 0.0:
        from surfgym.rnd import RND
        rnd = RND(core.obs_dim, device=device)
        print(f"RND novelty on {core.obs_dim} scalars, coef {args.rnd_coef:g}")
    # ---- one reward function PER SLOT -------------------------------------
    # `scale = 100 / d0` is computed from THAT map's own start geodesic, so a
    # full start->finish run is worth 100 on every map whatever its length.
    # A shared scale would make the short map invisible: cannonball's d0 is
    # 198,380 u and petrus_lite's 35,637 u, a 5.6x difference in what one
    # unit of progress pays. The novelty count table is per slot for the same
    # reason - its keys are 256u cells of a specific map.
    for _s in slots:
        if args.reward == "path":
            _s.reward_fn = PathLengthReward(0.01)
        elif args.reward == "maxspeed":
            _s.reward_fn = MaxSpeedReward(0.05)   # return = 0.05 * top h-speed
        elif args.reward == "coverage":
            _s.reward_fn = CoverageSpeedReward(
                0.001, 512.0, revisit_pen=(args.revisit_pen
                                           if args.revisit_pen is not None
                                           else 0.25))
        elif args.reward == "acro":
            _s.reward_fn = AcroCoverageReward(
                0.001, 512.0, revisit_pen=(args.revisit_pen
                                           if args.revisit_pen is not None
                                           else 1.0))
        elif args.reward == "race":
            # 100 total shaping over a full start->finish run regardless of
            # map size (generalist-comparable across maps); bonus + time cost
            # on top. --race-shaping scales the whole potential (0 = sparse:
            # bonus + penalties + intrinsic only); folded into scale so the
            # obs-reward eval feed and every downstream term inherit it
            # consistently
            _s.reward_fn = RaceReward(
                (goal_dist_field if goal_dist_field is not None
                 else _s.reward_field),
                scale=(float(args.goal_euclid_scale)
                       if goal_dist_field is not None
                       else 100.0 / _s.rf_d0 * args.race_shaping),
                time_pen=TIME_PEN_T,
                success_bonus=args.success_bonus,
                stall_ticks=TICK.secs_to_ticks(args.stall_secs),
                stall_eps=STALL_EPS_T,
                max_step=args.max_step,
                int_coef=args.int_coef,
                int_view=args.int_view,
                int_speed=args.int_speed,
                speed_equiv=args.speed_equiv,
                fail_pen=args.fail_pen,
                finish_k=args.finish_k,
                finish_tref=args.finish_tref,
                # --chunk: one POLICY decision is K*H ticks, so the
                # per-decision cadence widens with it. The potential shaping
                # telescopes across the whole window, so the sum is still
                # exact; time_pen and the tick counters scale by `every`.
                every=(KH if args.reward_per_decision else 1),
                d_floor=args.race_dfloor,
                d_latch=_s.d_latch,
                ng=args.race_ng, ng_gamma=GAMMA_T, ng_d0=_s.rf_d0,
                death_charge=(args.death_charge or 0.0),
                # --race-arc: single-map by the guard above, so handing the
                # one line to the (single) slot is exact
                arc=arc_line, arc_scale=arc_scale,
                d0_per_env=(goal_dist_field is not None),
                tick_ms=TICK.ms)
            _s.reward_fn.speed_coef = SPEED_COEF_T
            if args.race_ng:
                _g = GAMMA_T ** (KH if args.reward_per_decision else 1)
                _term = {1: "terminal charge on death and finish",
                         2: "terminal charge on death only",
                         3: "terminals stock (tax only)"}[args.race_ng]
                print(f"race: NG-CONFORMANT shaping v{args.race_ng} - "
                      f"per-call tax (1-{_g:.6f})*Phi, {_term}; "
                      f"Phi(spawn-mean)=0, full bank = "
                      f"{100.0 * args.race_shaping:g}")
            if args.death_charge:
                print(f"race: DEATH CHARGE kappa={args.death_charge:g} - "
                      f"death abandons kappa*Phi of the bank; per-step "
                      f"shaping and finish stock")
            if args.race_dfloor > 0.0:
                print(f"race: potential FLOORED at d = "
                      f"{args.race_dfloor:,.0f}u "
                      f"({100.0 * args.race_dfloor / max(_s.rf_d0, 1.0):.2f}%"
                      f" of the start distance) - shaping pays 0 and charges "
                      f"0 inside that shell; stall/stagnant keep the raw d")
            if _s.d_latch > 0.0:
                print(f"race{f'[{_s.name}]' if MULTI else ''}: shaping "
                      f"LATCHED OFF once an episode reaches "
                      f"d = {_s.d_latch:,.0f}u "
                      f"({100.0 * _s.d_latch / max(_s.rf_d0, 1.0):.2f}% of "
                      f"the start distance) - zero in BOTH directions for the "
                      f"rest of that episode; the flag is obs column "
                      f"{LATCH_COL}; stall/stagnant keep the raw d")
        elif args.reward == "blend":
            _s.reward_fn = BlendedReward(ForwardProgressReward(0.01),
                                         PathLengthReward(0.01),
                                         args.blend_start, args.blend_end)
        else:
            _s.reward_fn = ForwardProgressReward(0.01)
    reward_fn = slots[0].reward_fn
    for _s in heldout:
        # EVAL-ONLY, never called on a rollout (the slot owns no env rows and
        # is not in fleet.slots). It exists so --eval-stall and the
        # --obs-reward / --race-latch eval mirrors below read this map's
        # field, its 100/d0 scale and the same stall constants off the same
        # attribute they read on a training slot. int_coef 0: no novelty
        # table, and on_reset is never called on it either.
        _s.reward_fn = RaceReward(
            _s.goal_field, scale=100.0 / _s.rf_d0 * args.race_shaping,
            time_pen=TIME_PEN_T, success_bonus=args.success_bonus,
            stall_ticks=TICK.secs_to_ticks(args.stall_secs),
            stall_eps=STALL_EPS_T, max_step=args.max_step,
            every=(KH if args.reward_per_decision else 1),
            d_floor=args.race_dfloor, d_latch=_s.d_latch,
            ng=args.race_ng, ng_gamma=GAMMA_T, ng_d0=_s.rf_d0,
            tick_ms=TICK.ms)

    # per-decision reward path: only RaceReward knows how to telescope
    rpd = bool(args.reward_per_decision) and isinstance(reward_fn, RaceReward)

    # --ret-norm / --eval-stall. Both are args-derived, hence RANK-SYMMETRIC,
    # which is what makes the single collective --ret-norm adds legal inside
    # the update (docs/ddp-plan.md: every collective must be unconditional on
    # every rank).
    RETN = bool(args.ret_norm)
    EVAL_STALL = bool(args.eval_stall)
    retn = ReturnNorm()
    if RETN:
        print(f"--ret-norm: the critic predicts NORMALIZED returns "
              f"(debiased EMA, beta {retn.beta:g}, eps {retn.eps:g}); every "
              f"V(s) read outside the value loss - GAE, the truncation "
              f"bootstrap - is de-normalized back into reward units")
    if EVAL_STALL:
        if not isinstance(reward_fn, RaceReward):
            raise SystemExit("--eval-stall mirrors the RACE stall rule and "
                             "needs --reward race (nothing else owns a "
                             "distance field to stall against)")
        print(f"--eval-stall: eval episodes are killed by TRAINING's rule - "
              f"{TICK.ticks_to_secs(reward_fn.stall_ticks):g}s without a "
              f"{reward_fn.stall_eps:g}u improvement of the running best, "
              f"evaluated every {reward_fn.every} tick(s), ending the "
              f"episode as a fail")

    # the eval feeds mirror TRAINING's side channels on a core that produces
    # no reward, so they are per map too: each closes over its own field,
    # its own scale (100/d0) and its own latch threshold. Held-out slots get
    # the same mirrors: their eval must feed the policy what training fed it.
    for _s in slots + heldout:
        if N_LATCH:
            _s.eval_latch_feed = _make_eval_latch_feed(
                _s.reward_field if _s.reward_field is not None
                else _s.goal_field, _s.reward_fn.d_latch)
        # --priv-critic: one PrivFeat per map, holding THAT map's centre,
        # scale and start geodesic. The rollout fills it from the live core
        # (MapFleet.fill_priv), the truncation bootstrap from a
        # reconstructed terminal state (MapFleet.terminal_priv) and the eval
        # from its own core - three callers, one implementation.
        _s.priv = None
        _s.eval_priv_feed = None
        if args.priv_critic and _s.core is not None:
            _mn, _mx = _s.core.map_bounds()
            _s.priv = PrivFeat(_s.map_center,
                               float(np.max((_mx - _mn) / 2.0)),
                               _s.rf_d0 if _s.rf_d0 else _s.d0,
                               args.ep_ticks,
                               arc_len=(arc_line.length
                                        if arc_line is not None else 0.0))
            if not MULTI or _s is slots[0]:
                print(_s.priv.describe())
            if _s.eval_core is not None:
                _s.eval_priv_feed = _make_eval_priv_feed(
                    _s.priv,
                    _s.reward_field if _s.reward_field is not None
                    else _s.goal_field,
                    line=arc_line,
                    latch_feed=(_s.eval_latch_feed if N_LATCH else None),
                    corridor=(arc_line.corridor if arc_line is not None
                              else 1500.0),
                    window=(arc_line.window if arc_line is not None else 16),
                    max_step=_s.reward_fn.max_step, k=K)
        if args.obs_reward:
            if not (isinstance(_s.reward_fn, RaceReward)
                    and _s.goal_field is not None):
                raise SystemExit("--obs-reward currently needs --reward race "
                                 "(the eval feed mirrors the geodesic shaping)")
            if arc_line is not None:
                # an arc-trained policy reads its own ARC shaping in slot 12;
                # feeding it the geodesic mirror would be the train/eval
                # mismatch _make_eval_arc_feed exists to prevent
                _s.eval_reward_feed = _make_eval_arc_feed(
                    arc_line, arc_scale,
                    (lambda _rf=_s.reward_fn: _rf.time_pen), K,
                    arc_line.corridor, arc_line.window,
                    max_step=_s.reward_fn.max_step)
            else:
                _s.eval_reward_feed = _make_eval_reward_feed(
                    _s.reward_field if _s.reward_field is not None
                    else _s.goal_field,
                    _s.reward_fn.scale,
                    (lambda _rf=_s.reward_fn: _rf.time_pen), K,
                    d_floor=_s.reward_fn.d_floor,
                    latch_feed=_s.eval_latch_feed,
                    ng=args.race_ng, ng_g=(lambda: GAMMA_T ** K),
                    ng_d0=_s.rf_d0, max_step=_s.reward_fn.max_step)
        # --act-hist / --obs-compass: the eval core is ONE env, so it gets
        # its own ObsAux with its own history ring and its own d0 anchor -
        # the SAME class the rollout drives, never a second implementation.
        # Under --goal-reward euclid/geo the compass has to follow the EVAL
        # goal, which is a different object from the training fleet's
        # per-env field, so a 1-wide twin is built and handed to GoalSystem
        # to re-centre on every eval episode.
        _s.eval_aux = None
        if N_AUX and _s.eval_core is not None:
            _ef = None
            if args.obs_compass:
                if goal_dist_field is not None:
                    from surfgym.goals import GoalDistField
                    _ef = GoalDistField(
                        1, geo=(_s.reward_field if args.goal_reward == "geo"
                                else None))
                else:
                    _ef = _s.reward_field
            _s.eval_aux = ObsAux(1, k=int(args.act_hist or 0), field=_ef,
                                 yaw_adaptive=bool(args.yaw_adaptive))

    global_step = 0
    if args.sb3:
        raise SystemExit("--sb3 import predates the lidar architecture; "
                         "the SB3 MlpPolicy weights don't map onto the conv trunk")
    if ck is not None:
        # --chunk changes the ARCHITECTURE (code head + decoder) and the
        # optimizer's parameter list with it, so neither direction is a warm
        # start — load_state_dict would die on missing/unexpected keys and
        # opt.load_state_dict on the group size, three screens later and
        # without the reason. Same wall --surf-mask hits.
        has_dec = "decoder" in (ck.get("policy") or {})
        if H > 0 and not has_dec:
            raise SystemExit(
                "--chunk cannot warm-start from a FLAT checkpoint: it has no "
                "code head and no decoder, and the optimizer state has no "
                "slots for them. The chunked action space is a scratch run "
                "(design doc 8); drop --ckpt, or resume a chunked ckpt")
        if H == 0 and has_dec:
            raise SystemExit(
                "this checkpoint was trained with --chunk (it carries a "
                "decoder): resuming it without --chunk would read its code "
                "logits as flat action logits. Pass --chunk with the ckpt's H")
        # BEFORE every widener: each of them can only see
        # `checkpoint tensor width - policy.feat_dim` and has to assume the
        # two trunks agree, so a trunk mismatch comes out in whichever
        # flag's language happens to run first. Both checks below name the
        # actual cause instead (check_arch_matches' docstring has the case
        # that motivated them).
        check_arch_matches(ck_cfg, args, policy)
        _tm = ck_trunk_mismatch(ck, policy)
        if _tm is not None:
            _nm, _w, _rem = _tm
            raise SystemExit(
                f"this checkpoint's trunk is a different width: {_nm} reads "
                f"{_w} columns but this run's trunk alone is "
                f"{policy.feat_dim} wide ({-_rem} short before any "
                "observation block). The trunk is set by --emb / "
                "--conv-mult / --trunk / --lidar-w / --lidar-h and the "
                "scalar mask, all of which are restored from the checkpoint "
                "when their flag is absent - drop the flag that shrank it")
        if N_ROUTE:
            # FIRST, before the GRU and the trailing pads: the route-side
            # scalar block sits BETWEEN the trunk output and the GRU state,
            # so growing it is an insert at its own position and only a
            # trailing pad when there is no recurrence. Doing it here means
            # widen_for_rnn's trailing pad below still lands where the GRU
            # block goes, and both directions compose in one resume.
            _gh = int((ck_cfg.get("act_hist") or 0))
            _gc = int((ck_cfg.get("obs_compass") or 0))
            _grew = []
            if int(args.act_hist or 0) > _gh:
                _grew.append(f"--act-hist {int(args.act_hist)}")
            if int(args.obs_compass or 0) > _gc:
                _grew.append("--obs-compass 1")
            _gflag = " + ".join(_grew) if _grew else "--route"
            _old_blk = ck_obs_block(ck, policy)
            n_w = widen_for_obs(ck, policy, N_ROUTE, flag=_gflag)
            if n_w and _old_blk is not None:
                print(f"{_gflag}: this checkpoint's towers read {_old_blk} "
                      f"of this run's {N_ROUTE} scalar-side columns; widened "
                      f"{n_w} tensors (pi.0/vf.0 and their Adam moments gain "
                      f"{N_ROUTE - _old_blk} ZERO columns at scalar-row "
                      f"{N_SCALAR + _old_blk}..{N_SCALAR + N_ROUTE - 1}) - "
                      "the new inputs multiply zero weights, so the resumed "
                      "policy computes the checkpoint's own function at "
                      "step 0 (to ~1 ulp of fp32) and the new block grows "
                      "from zero")
        if RNN:
            # after the block above and before the route widening: both pad
            # the SAME two Linears to the model's width, so whichever runs
            # first does all of it and the other finds nothing to do - this
            # one names the GRU
            n_w = widen_for_rnn(ck, policy)
            if n_w:
                print(f"--rnn: feed-forward checkpoint widened ({n_w} "
                      f"tensors: {R} trailing zero columns per tower and "
                      "their Adam moments, plus a fresh GRU) - the resumed "
                      "policy is function-identical to the baseline at "
                      "step 0 and the GRU path grows from zero")
        if N_ROUTE:
            # anything the two passes above did not reach (a tower under a
            # flag combination they skip): the historical trailing pad
            n_w = widen_for_route(ck, policy)
            if n_w:
                print(f"--route: widened {n_w} checkpoint tensors by "
                      f"{N_ROUTE} zero columns — the resumed policy is "
                      "function-identical to the baseline at step 0")
        if args.priv_critic:
            n_w = widen_for_priv(ck, policy)
            if n_w:
                print(f"--priv-critic: this checkpoint has no privileged "
                      f"critic; widened {n_w} tensors (value_head and its "
                      f"Adam moments grow {policy.priv_hidden} TRAILING "
                      "zero columns, plus a fresh priv MLP with no moments) "
                      "- the ACTOR is bit-identical to the checkpoint's and "
                      "V(s) is its old function to ~1 ulp, so the arm starts "
                      "ON the control curve and the privileged path grows "
                      "from zero")
        if YCOND:
            n_w = widen_for_yawcond(ck, policy)
            if n_w:
                print(f"--yaw-cond: this checkpoint has no yaw->side "
                      f"conditioning; added {n_w} fresh tensor(s) "
                      f"({NVEC[H_YAW]}x{NVEC[H_SIDE]} zeros, no Adam "
                      "moments). No existing tensor changes shape, so the "
                      "resumed policy's LOGITS are bit-identical to the "
                      "checkpoint's - the arm starts ON the control curve "
                      "and the conditioning grows from zero")
        elif "yaw_side.table" in (ck.get("policy") or {}):
            raise SystemExit(
                "this checkpoint was trained with --yaw-cond: it carries a "
                "yaw_side.table and its side key is drawn from p(side|yaw). "
                "Resuming it without the flag would throw that tensor away "
                "and read the side head as unconditioned - a different "
                "policy dressed as a warm start. Pass --yaw-cond.")
        _has_view = "view_head.weight" in (ck.get("policy") or {})
        if VIEWC and not _has_view:
            raise SystemExit(
                "--view-continuous cannot warm-start a DISCRETE checkpoint "
                "directly: its view heads are categorical and the two new "
                "tensors (view_head, view_std) would start from noise. "
                "Transplant it first - python tools/transplant_view.py "
                "<ckpt> <out.pt> fits the continuous means to the "
                "categorical heads - then resume the transplanted file.")
        if not VIEWC and _has_view:
            raise SystemExit(
                "this checkpoint was trained with --view-continuous (it "
                "carries view_head / view_std): resuming it without the "
                "flag would read its dead yaw/pitch logits as the view. "
                "Pass --view-continuous.")
        _ck_abs = (ck.get("config") or {}).get("view_absolute") or None
        if VIEWC and _ck_abs != VIEW_ABS:
            raise SystemExit(
                f"this checkpoint was trained with view_absolute="
                f"{_ck_abs!r} and this run asks for {VIEW_ABS!r}: the view "
                "row would be READ differently from how these weights "
                "learned to write it (and world mode has a third head). "
                "Drop --view-absolute to restore the checkpoint's own mode.")
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["optimizer"])
        n_re = relayout_optimizer_state(opt)
        if n_re:
            print(f"optimizer state restrided to the params' layout ({n_re} tensors)")
        # load_state_dict replaces param_groups wholesale, lr included — an
        # explicit --lr was silently ignored on resume before this
        ck_lr = float(opt.param_groups[0]["lr"])
        if ck_lr != args.lr:
            print(f"optimizer lr: {ck_lr:g} (ckpt state) -> {args.lr:g}")
        for g in opt.param_groups:
            g["lr"] = args.lr
        global_step = 0 if args.reset_steps else int(ck.get("global_step", 0))
        if (isinstance(reward_fn, RaceReward)
                and ck.get("int_counts") is not None):
            if args.reset_int_counts:
                print("novelty counts DISCARDED (--reset-int-counts): "
                      "curiosity re-armed from an empty table")
            else:
                # single-map ckpts hold ONE table (the historical payload);
                # multi-map ones hold a dict keyed by map stem. Count keys
                # are cells of a specific map, so a table only ever goes
                # back to the slot it came from - RaceReward.on_reset
                # re-zeroes any table whose length disagrees anyway.
                ic = ck["int_counts"]
                by_map = ic if isinstance(ic, dict) else {STEMS[0]: ic}
                n_visits = 0
                for _s in slots:
                    arr = by_map.get(_s.name)
                    if arr is None:
                        continue
                    _s.reward_fn.restore_counts(arr)
                    n_visits += int(np.asarray(arr).sum(dtype=np.int64))
                print(f"restored novelty counts ({n_visits:,} visits)")
        if RETN and ck.get("ret_norm") is not None:
            # the critic's outputs are only meaningful against the (mu,
            # sigma) they were fitted under: resuming with a fresh EMA would
            # read every V(s) in the wrong frame for the ~100 iterations the
            # EMA takes to converge back, which is a silent value-function
            # reset dressed as a warm start.
            retn.load_state_dict(ck["ret_norm"])
            print(f"restored return normalizer (mean {retn.mean:,.4f}  "
                  f"std {retn.std:,.4f}  {retn.count} updates)")
        if rnd is not None and ck.get("rnd") is not None:
            rnd.load_state_dict_all(ck["rnd"])
            print("restored RND state (target/predictor/normalizers)")
        if respawn is not None and ck.get("respawn") is not None:
            # same shape rule as the counts, and RespawnBuffer already
            # refuses a payload whose map_id does not match its own
            rs = ck["respawn"]
            by_map = rs if isinstance(rs, dict) and "states" not in rs                 else {STEMS[0]: rs}
            for _s in slots:
                if _s.respawn is not None and by_map.get(_s.name) is not None:
                    _s.respawn.load_state_dict(by_map[_s.name])
            print(f"restored respawn reservoir "
                  f"({fleet.reservoir_size():,} states)")
        print(f"resumed {args.ckpt} at step {global_step:,}"
              + (" (steps reset)" if args.reset_steps else ""))

    # DDP: no wrapper constructor syncs weights here — broadcast explicitly,
    # then hold every rank to an EXACTLY equal param checksum (startup and
    # every 100 iterations, on by default). The chain: identical init ->
    # identical all-reduced grads (NCCL is rank-symmetric) -> identical clip
    # scale -> identical fused-Adam step -> identical params. cudnn
    # benchmark=True may pick different conv algorithms per rank, but that
    # only perturbs the LOCAL grad before the all-reduce; after it all
    # ranks hold identical bytes, so the check is exact, not a tolerance.
    if D.enabled:
        with torch.no_grad():
            for p in policy.parameters():
                D.broadcast_(p.data)

    def param_checksum():
        with torch.no_grad():
            return torch.stack([sum(p.double().sum()
                                    for p in policy.parameters())])

    D.assert_equal("policy_params@startup", param_checksum())

    meta = {"label": args.run, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished": None,
            # "map" stays the FIRST map so every existing consumer (the
            # dashboard, record_ckpt, the honesty tools) keeps working; a
            # single-map run therefore writes exactly the config it always
            # did, "maps" included as None
            "config": {"trainer": "fast2", "map": STEMS[0],
                       "maps": (list(STEMS) if MULTI else None),
                       "map_cells": ({s.tag: s.cell for s in slots}
                                     if MULTI else None),
                       # --heldout-maps: evaluated, never trained. None on
                       # every run without the flag, exactly like "maps"
                       "heldout_maps": (list(HSTEMS) if NHELD else None),
                       "heldout_goal_cell": (args.heldout_goal_cell
                                             if NHELD else None),
                       "heldout_goal_cells": ({s.tag: s.goal_cell
                                               for s in heldout}
                                              if NHELD else None),
                       "heldout_map_cells": ({s.tag: s.cell for s in heldout}
                                             if NHELD else None),
                       "envs": N_GLOBAL,          # the GLOBAL fleet
                       "envs_per_rank": N, "world_size": D.world_size,
                       "envs_per_slot": PER, "n_maps": NMAPS,
                       "ddp": D.enabled, "seed": args.seed,
                       "steps": int(args.steps), "spawn": args.spawn,
                       "reward": args.reward, "lr": args.lr,
                       "blend": ([args.blend_start, args.blend_end]
                                 if args.reward == "blend" else None),
                       "lidar_w": args.lidar_w, "lidar_h": args.lidar_h,
                       "surf_mask": args.surf_mask,
                       "pinhole": args.pinhole,
                       # --normals / the fov are what the policy SEES:
                       # record_ckpt.py and render_pov.py mirror all three
                       "normals": args.normals,
                       "lidar_hfov": args.lidar_hfov,
                       "lidar_vfov": args.lidar_vfov,
                       "frame_stack": args.frame_stack,
                       # --act-hist/--obs-compass are OBSERVATION
                       # columns: they set the scalar width the
                       # weights were trained at, so a resume and
                       # tools/record_ckpt.py both need them
                       "act_hist": int(args.act_hist or 0),
                       "obs_compass": int(args.obs_compass or 0),
                       # the route FILE is part of the observation spec: a
                       # resume against a different line would feed the same
                       # weights a differently-shaped world
                       "route_file": (getattr(route, "source", None) if route is not None
                                      else None),
                       "route_points": (len(route.offsets)
                                        if route is not None else None),
                       "route_span": (route.offsets[-1]
                                      if route is not None else None),
                       "priv_critic": int(bool(args.priv_critic)),
                       # the exact block the critic was trained on, written
                       # out so a ledger entry and any later reader never
                       # have to infer it from the flag alone
                       "priv_features": (list(PRIV_FEATURES)
                                         if args.priv_critic else None),
                       "priv_hidden": (int(args.priv_hidden)
                                       if args.priv_critic else None),
                       "route_critic_only": (int(bool(args.route_critic_only))
                                             if route is not None else None),
                       "fix_pitch": args.fix_pitch,
                       # --pitch-fixed aims the lidar, so it is part of what
                       # the policy SEES: record_ckpt.py mirrors it rather
                       # than listing it in TRAIN_ONLY
                       "pitch_fixed": args.pitch_fixed,
                       "emb": args.emb, "hidden": args.hidden, "gps": args.gps,
                       # --trunk changes what the policy IS, so record_ckpt.py
                       # mirrors it rather than listing it in TRAIN_ONLY
                       "trunk": args.trunk,
                       # --rnn likewise: a recurrent ckpt needs its state
                       # carried by whoever runs it, and the towers are
                       # rnn_size wider
                       "rnn": args.rnn,
                       "rnn_size": (args.rnn_size if RNN else None),
                       # --tower-depth/--conv-mult change what the policy IS,
                       # so record_ckpt.py mirrors them like --trunk
                       "tower_depth": args.tower_depth,
                       "conv_mult": args.conv_mult,
                       # --fp32-heads is TRAIN_ONLY in record_ckpt.py: the
                       # recorder never enters autocast, so its heads are
                       # already fp32 whatever this says
                       "fp32_heads": args.fp32_heads,
                       "teleport_fail": not args.keep_teleports,
                       "lidar_range": args.lidar_range,
                       "lidar_near": args.lidar_near or args.lidar_range,
                       "drop_min": args.drop_min, "drop_max": args.drop_max,
                       "punch_min": args.punch_min, "punch_max": args.punch_max,
                       "revisit_pen": args.revisit_pen,
                       "act_every": K, "pitch_rate": pitch_rate,
                       # --tick-ms: the physics tick (requested / realised
                       # mean / integer pattern), the tick the checkpoint
                       # was trained at when a resume changed it, and the
                       # per-tick constants AS APPLIED. gamma / time_pen /
                       # stall_eps / pitch_rate above stay the 10 ms-
                       # referenced flag values so arms compare directly.
                       "tick_ms": args.tick_ms,
                       "tick_ms_eff": TICK.ms,
                       "tick_pattern_ms": list(TICK.pattern),
                       "tick_ms_ckpt": tick_ms_ckpt,
                       # --tick-ms-schedule: the ramp, ORIGIN INCLUDED, so a
                       # bare resume of this checkpoint continues it. Present
                       # only on a scheduled run - a control's config dump is
                       # byte-identical to before the flag existed. The tick
                       # keys above are kept LIVE as the ramp moves (see
                       # _tick_retune), so a checkpoint always states the tick
                       # its weights were actually trained at.
                       **({"tick_schedule": tick_sched.to_dict()}
                          if tick_sched is not None else {}),
                       "gamma_tick": GAMMA_T,
                       "time_pen_tick": (TIME_PEN_T if args.reward == "race"
                                         else None),
                       "stall_eps_tick": (STALL_EPS_T if args.reward == "race"
                                          else None),
                       "ep_secs": TICK.ticks_to_secs(args.ep_ticks),
                       # --chunk/n_codes change what ONE decision means, so
                       # record_ckpt.py must mirror them (see its TRAIN_ONLY
                       # note). dec_ent/codebook/codebook_bias are training-
                       # side only: the trained decoder ships in the
                       # state_dict, so a recorder reads it from there.
                       "chunk": (H or None),
                       "n_codes": (NCODES or None),
                       "dec_ent": (args.dec_ent if H > 0 else None),
                       "codebook": cb_file,
                       "codebook_bias": (args.codebook_bias
                                         if cb_file else None),
                       # one global cell under --lidar-cell; otherwise the
                       # per-map pick, which record_ckpt re-derives from the
                       # map it is recording ("map_cells" is the record)
                       "lidar_cell": (args.lidar_cell if MULTI
                                      else (args.lidar_cell or slots[0].cell)),
                       "time_pen": (args.time_pen if args.reward == "race"
                                    else None),
                       "success_bonus": (args.success_bonus
                                         if args.reward == "race" else None),
                       "finish_k": (args.finish_k
                                    if args.reward == "race" else None),
                       "finish_tref": (args.finish_tref
                                       if args.reward == "race" else None),
                       "train_stride": args.train_stride,
                       "obs_reward": args.obs_reward,
                       "ez_eps": args.ez_eps, "ez_max": args.ez_max,
                       "ez_mu": args.ez_mu,
                       "reward_per_decision": args.reward_per_decision,
                       "stall_secs": (args.stall_secs
                                      if args.reward == "race" else None),
                       # --stall-eps decides when an episode ENDS, in
                       # training AND (via --eval-stall) in a recording, so
                       # record_ckpt.py mirrors it into its own stall hook
                       "stall_eps": (args.stall_eps
                                     if args.reward == "race" else None),
                       # --max-step is a reward TERM, but it is one of the
                       # terms the --obs-reward eval feed reproduces, so
                       # record_ckpt.py mirrors it into its own feed
                       "max_step": (args.max_step
                                    if args.reward == "race" else None),
                       # --eval-stall changes what an EVAL measures, not what
                       # training does; --ret-norm changes what the value
                       # head MEANS, so a resume has to see it (restored
                       # above). Both belong in run.json either way: an arm
                       # whose eval conditions differ from the control's is
                       # not comparable to it, and nothing else records that.
                       "eval_stall": int(args.eval_stall or 0),
                       "ret_norm": int(args.ret_norm or 0),
                       "fail_pen": (args.fail_pen
                                    if args.reward == "race" else None),
                       "race_ng": (args.race_ng
                                   if args.reward == "race" else None),
                       "death_charge": (args.death_charge
                                        if args.reward == "race" else None),
                       "speed_coef": (args.speed_coef
                                      if args.reward == "race" else None),
                       "race_dist": (args.race_dist
                                     if args.reward == "race" else None),
                       "int_coef": (args.int_coef
                                    if args.reward == "race" else None),
                       "int_view": (args.int_view
                                    if args.reward == "race" else None),
                       "rnd_coef": (args.rnd_coef
                                    if args.reward == "race" else None),
                       "speed_equiv": (args.speed_equiv
                                       if args.reward == "race" else None),
                       "int_speed": (args.int_speed
                                     if args.reward == "race" else None),
                       "maxvel": args.maxvel,
                       "yaw_adaptive": args.yaw_adaptive,
                       "yaw_blend": args.yaw_blend,
                       "side_hold": args.side_hold,
                       "respawn_frac": args.respawn_frac,
                       "respawn_margin": args.respawn_margin,
                       "respawn_binned": args.respawn_binned,
                       "respawn_min_speed": args.respawn_min_speed,
                       "respawn_mode": args.respawn_mode,
                       "respawn_bins": args.respawn_bins,
                       "respawn_killsafe": args.respawn_killsafe,
                       "race_shaping": args.race_shaping,
                       "race_dfloor": args.race_dfloor,
                       "race_latch": args.race_latch,
                       "race_latch_frac": (args.race_latch_frac or None),
                       # the arc route is part of the REWARD spec (and, under
                       # --obs-reward, of the observation): a resume must not
                       # be able to lose it silently
                       "race_arc": (arc_line.source if arc_line is not None
                                    else None),
                       "race_arc_corridor": (arc_line.corridor
                                             if arc_line is not None else None),
                       "race_arc_window": (arc_line.window
                                           if arc_line is not None else None),
                       "goals": int(args.goals or 0),
                       "goal_radius": args.goal_radius,
                       "goal_kmin": args.goal_kmin,
                       "goal_kmax": args.goal_kmax,
                       "goal_kcap": args.goal_kcap,
                       "goal_air_frac": args.goal_air_frac,
                       "goal_holdout": args.goal_holdout,
                       "goal_curriculum": int(args.goal_curriculum or 0),
                       "goal_obs": args.goal_obs,
                       "goal_views": int(args.goal_views),
                       "goal_route": args.goal_route,
                       "goal_route_frac": args.goal_route_frac,
                       "goal_reward": args.goal_reward,
                       "goal_frontier": int(args.goal_frontier or 0),
                       "goal_route_uniform": int(args.goal_route_uniform or 0),
                       "goal_fixed": int(args.goal_fixed or 0),
                       "goal_fixed_spacing": args.goal_fixed_spacing,
                       "goal_fixed_air": args.goal_fixed_air,
                       "goal_euclid_scale": args.goal_euclid_scale,
                       "goal_fixed_decay": args.goal_fixed_decay,
                       "goal_front_start": args.goal_front_start,
                       "goal_front_band": args.goal_front_band,
                       "goal_front_step": args.goal_front_step,
                       "goal_front_rate": args.goal_front_rate,
                       "goal_front_min_ep": args.goal_front_min_ep,
                       "spawn_burst": args.spawn_burst,
                       "spawn_burst_p": args.spawn_burst_p,
                       "demo_file": args.demo_file,
                       "demo_window": (args.demo_window
                                       if args.demo_file else None),
                       "demo_rate": (args.demo_rate
                                     if args.demo_file else None),
                       "demo_min_ep": (args.demo_min_ep
                                       if args.demo_file else None),
                       "bc_file": args.bc_file,
                       "bc_coef": args.bc_coef if args.bc_file else None,
                       "bc_coef_final": (args.bc_coef_final
                                         if args.bc_file else None),
                       "bc_steps": args.bc_steps if args.bc_file else None,
                       "bc_batch": args.bc_batch if args.bc_file else None,
                       "bc_target": args.bc_target if args.bc_file else None,
                       "bc_value_coef": (args.bc_value_coef if args.bc_file
                                         else None),
                       "race_kill_aware": args.race_kill_aware,
                       "respawn_reservoir": args.respawn_reservoir,
                       "respawn_speed": args.respawn_speed,
                       "ep_ticks": args.ep_ticks, "epochs": args.epochs,
                       # n_steps and minibatches were NOT recorded, and an
                       # ablation whose arms differ only in one of them then
                       # has no record of what it ran - found while checking a
                       # control was really the control (2026-08-23)
                       "goal_cell": args.goal_cell,
                       "goal_cells": ({s.tag: s.goal_cell for s in slots}
                                      if MULTI else None),
                       "n_steps": args.n_steps,
                       "minibatches": args.minibatches,
                       "gamma": args.gamma, "gae": args.gae,
                       "clip": args.clip, "vf": args.vf, "ent": args.ent,
                       "ent_final": args.ent_final,
                       "eval_eps": args.eval_eps,
                       "eval_greedy_only": args.eval_greedy_only,
                       "graphs": use_graphs, "compile": use_compile,
                       "bf16": use_bf16}}
    # --mask-*: keys appear ONLY when the mask is on, so a control run's
    # run.json and every checkpoint config it writes stay byte-identical to
    # the pre-flag trainer's (record_ckpt.py mirrors them; they change what
    # an action MEANS, so they are not TRAIN_ONLY).
    meta["config"].update(MASKS.config())
    if MASKS.on:
        print(MASKS.describe())
    # --yaw-cond: written ONLY when set, for the reason the masks give - a
    # control run's run.json and every checkpoint config it writes stay
    # byte-identical to the pre-flag trainer's. It changes what an action
    # MEANS (the side key is drawn from a different distribution), so
    # record_ckpt.py mirrors it rather than listing it TRAIN_ONLY.
    # --tail-weight: written ONLY when set, for the reason the masks give
    # above - a control run's run.json and every checkpoint config it writes
    # stay byte-identical to the pre-flag trainer's. It is TRAIN_ONLY in
    # record_ckpt.py: it reweights a gradient and changes neither what an
    # action means nor what the policy sees.
    if args.tail_weight > 0.0:
        meta["config"]["tail_weight"] = args.tail_weight
        meta["config"]["tail_outcome"] = args.tail_outcome
        meta["config"]["tail_min_n"] = args.tail_min_n
        meta["config"]["tail_bins"] = args.tail_bins
    if YCOND:
        meta["config"]["yaw_cond"] = 1
        print(f"--yaw-cond: side-key head conditioned on the sampled yaw "
              f"bin, {NVEC[H_YAW]}x{NVEC[H_SIDE]} additive table "
              "(zero-initialised)")
    # --view-continuous: written ONLY when set, for the reason the masks
    # give - a control run's config dump stays byte-identical. It changes
    # what an action IS, so record_ckpt.py mirrors it.
    if VIEWC:
        meta["config"]["view_continuous"] = 1
        print("--view-continuous: yaw and pitch are squashed Gaussians "
              "(z ~ N(mu(s), sigma), K = warp(tanh z), pitch = tanh z * "
              f"{float(core.config.pitch_rate_max_deg):g} deg/tick); "
              f"log sigma {float(policy.log_std()[0]):.3f} / "
              f"{float(policy.log_std()[1]):.3f}")
        if VIEW_ABS:
            meta["config"]["view_absolute"] = VIEW_ABS
            _how = ("yaw target = heading(v_h) + off_warp(tanh z) deg "
                    "(u=+-0.5 -> +-10, u=+-1 -> +-180; base = current yaw "
                    "below 100 u/s)" if VIEW_ABS == "velocity" else
                    "yaw target = atan2(tanh z_s, tanh z_c) deg, 3 heads")
            print(f"--view-absolute {VIEW_ABS}: the view row is an ABSOLUTE "
                  f"target the core approaches every tick at <= "
                  f"{float(core.config.yaw_rate_max_deg):g} / "
                  f"{float(core.config.pitch_rate_max_deg):g} deg/tick; "
                  f"{_how}; pitch target = -20 + 50 tanh z (core view_mode "
                  f"{int(core.config.view_mode)})")
    if D.is_main:
        (out / "run.json").write_text(json.dumps(meta, indent=2),
                                      encoding="utf-8")
    CSV_COLS = ["time/total_timesteps", "rollout/ep_rew_mean",
                "rollout/ep_len_mean", "time/fps", "train/loss",
                "train/value_loss", "train/entropy_loss",
                "train/approx_kl", "eval/fwd_max", "eval/path",
                "eval/speed_max", "train/blend_w",
                "race/success_rate", "race/finish_s",
                "race/eval_progress", "race/eval_finish_s"]
    # --maps: the aggregate columns above stay where they are (mean over
    # maps), and each map appends its own suffixed quad AFTER them. Appending
    # is what makes the header migration below work on a resumed run: an old
    # file's header is a strict PREFIX of the new one, so it is padded rather
    # than mismatched. Single-map runs add nothing and write today's 16.
    #
    # The two AGGREGATE columns the multi-map run is actually judged on go
    # in front of the per-map block, so they exist on single-map runs too
    # (where they are just that map's own numbers):
    #   race/map_pct       mean over maps of the mean over that map's eval
    #                      episodes of 100*(d_spawn - d_min)/d_spawn - a
    #                      PERCENTAGE of each map's own route, so a 5x
    #                      longer map does not out-vote a short one the way
    #                      raw eval_progress units do.
    #   race/maps_finished fraction of maps with >= 1 greedy eval episode
    #                      inside the finish box.
    CSV_COLS += ["race/map_pct", "race/maps_finished",
                 "race/map_pct_trigger", "race/maps_finished_trigger"]
    EVAL_COLS = ("race/eval_progress", "race/eval_finish_s",
                 "race/eval_finishes", "race/map_pct")
    if MULTI:
        for _s in slots:
            CSV_COLS += [f"{c}.{_s.tag}" for c in EVAL_COLS]
    # PPO hygiene, appended LAST - after the per-map block, not before it -
    # because the header migration above only pads a header that is a strict
    # PREFIX of the new one, and inserting ahead of the suffixed quads would
    # break that for every resumed multi-map run.
    #
    #   train/explained_var  1 - Var(G - V)/Var(G) over the rollout buffer.
    #                        The critic's share of the return's variance:
    #                        ~1 is a fitted critic, ~0 a critic no better
    #                        than the mean, negative a critic that is worse.
    #                        Advantages are as good as this number.
    #   race/trunc_frac      share of the episodes that ENDED this iteration
    #                        that ended by TRUNCATION (the --ep-ticks limit)
    #                        rather than by a terminal.
    #   race/stall_frac      stall kills issued this iteration / episodes
    #                        ended. A stall kill lands as an ordinary fail,
    #                        so this is the only place it is visible.
    #   race/crawl_frac      share of ended episodes whose MEAN horizontal
    #                        speed over the episode was under 300 u/s.
    #                        Distinguishes "died fast" from "never moved" -
    #                        the two failures ep_len_mean confuses.
    #   train/ret_mean/std   the --ret-norm running statistics (constant
    #                        0 / 1 when the flag is off).
    CSV_COLS += ["train/explained_var", "race/trunc_frac", "race/stall_frac",
                 "race/crawl_frac", "train/ret_mean", "train/ret_std"]
    # --heldout-maps, appended after EVERYTHING else for the same prefix
    # rule: race/heldout_progress_field / _finish_s / _finishes /
    # _pct_field per held-out map, plus race/heldout_corridor_max where the
    # map has a route file (heldout_columns above). Never in an aggregate.
    CSV_COLS += heldout_columns(heldout)
    #   tick/tick_ms  the REALISED mean physics tick this iteration ran at
    #                 (constant without --tick-ms-schedule; the ramp
    #                 otherwise). LAST, like every other addition, so an
    #                 older progress.csv header stays a strict prefix and the
    #                 migration above pads it instead of refusing.
    CSV_COLS += ["tick/tick_ms"]
    # Key-use diagnostics, appended LAST for the same strict-prefix reason
    # the hygiene block gives above. They need no flag - every arm writes
    # them - so an existing run can be read for the air-key behaviour the
    # masks target (runs/research/wr_demo/wr_vs_ours.md section 5):
    #   act/fwd_air     share of AIRBORNE decisions whose forward/back head
    #                   was not "none" - W or S held, exactly what
    #                   --mask-forward-air forbids. WR 0.000, ours ~0.114.
    #   act/strafe_flip share of consecutive decision pairs whose side (A/D)
    #                   head changed value, over pairs that do not straddle
    #                   an episode end. A rate per DECISION, not per second:
    #                   multiply by 100/act_every to compare with the WR's
    #                   0.42 flips/s against our 1.50.
    #   act/jump_air    share of airborne decisions with jump held.
    #   act/duck_air    share of airborne decisions with duck held.
    #                   WR: 0 presses in 68 s; ours 283 jump / 176 duck.
    # All four are TRAINING rollout rates (the policy PPO is optimising),
    # not eval rates, and blank under --chunk.
    CSV_COLS += ["act/fwd_air", "act/strafe_flip", "act/jump_air",
                 "act/duck_air"]
    #   act/yaw_side_agree  the direct read-out of what --yaw-cond treats,
    #                 and logged with or WITHOUT the flag so a control run
    #                 states the starting point. Over decisions where BOTH
    #                 the yaw bin and the side key are non-neutral (yaw bin
    #                 != 7, side != 1), the share whose yaw delta turns the
    #                 view TOWARD the held key's wish direction. GoldSrc:
    #                 right = (sin yaw, -cos yaw) (src/pm.c angle_vectors at
    #                 roll 0), so +side (index 2, D) accelerates along
    #                 `right` and needs the view to rotate CLOCKWISE, i.e. a
    #                 NEGATIVE yaw delta (bin < 7); -side (index 0, A) needs
    #                 bin > 7. In raw indices "agree" is therefore OPPOSITE
    #                 signs about their neutral - that is the physics, not a
    #                 typo. 1.0 = every strafing decision coordinated.
    #                 Appended LAST, so the old header stays a strict prefix
    #                 and a resumed progress.csv migrates instead of
    #                 breaking. Over ALL decisions, not just airborne ones:
    #                 the litsurvey's 12.7% / 2.7% disagreement figures are
    #                 gated on fast airborne ticks, so read this column
    #                 against a matched control, not against those levels.
    CSV_COLS += ["act/yaw_side_agree"]
    #   bc/*  --bc-file's own diagnostics, blank on every run without it,
    #         appended LAST for the strict-prefix rule like everything above.
    #         They are here and not only in bc_log.csv because the
    #         distillation quality IS the expert-iteration loop's binding
    #         constraint (docs/research-litsurvey-zero.md) and the dashboard
    #         reads progress.csv.
    #   bc/ce_dist   cross-entropy of the six heads against the SEARCH's
    #                stored distribution over first decisions. Equals the
    #                planner-action NLL wherever the target is a one-hot, so
    #                a version-1 file logs the same number in both places.
    #   bc/head_acc  per-HEAD argmax agreement - the metric Expert Iteration
    #                measured 50 +/- 13 Elo ACROSS (47.0% vs 47.7% top-1
    #                error). Never report a distillation change on it.
    #   bc/joint_acc all six heads agree: the honest per-DECISION agreement.
    #                At 98% per head this is 0.886, i.e. ~285 of ~2,500
    #                decisions differ from the planner every run - which is
    #                the number the plateau is actually about.
    #   bc/value_mse (V(s) - z)^2 over the rows carrying a COMPLETE planner
    #                return (--bc-value-coef; blank when no row does).
    CSV_COLS += ["bc/ce_dist", "bc/head_acc", "bc/joint_acc", "bc/value_mse"]
    #   tail/*  --tail-weight's diagnostics (TailRL, arXiv 2609.02987).
    #           Blank on every run without the flag, appended LAST for the
    #           strict-prefix header-migration rule everything above uses.
    #           All of them are per ITERATION, over the episodes that ENDED
    #           in that rollout.
    #   tail/w_max   the largest weight applied. A group of N can produce at
    #                most N (all mass at the bottom, one rollout at the top),
    #                so this against tail/n_med says how close the batch got
    #                to betting everything on one episode.
    #   tail/w_p90   the 90th percentile weight - the level the top decile
    #                actually trained at, which w_max alone cannot say.
    #   tail/groups  spawn-depth-bin groups large enough to weight
    #                (>= --tail-min-n episodes).
    #   tail/n_med   median episodes per weighted group. This is the paper's
    #                N; their prompts get 8-64.
    #   tail/ess     normalised effective sample size of the weights,
    #                (sum w)^2 / (E * sum w^2): 1.0 if every episode weighs
    #                the same, 1/E if one episode carries the batch. The
    #                direct read-out of the variance this trades for tail
    #                pressure.
    #   tail/cov     share of the rollout buffer belonging to an episode
    #                that ENDED here, i.e. that carries a real weight; the
    #                rest is the in-progress buffer edge, held at 1.
    #   tail/p50/75/90  mean over groups of the empirical tail probability
    #                p(tau) at tau = 0.5 / 0.75 / 0.9 of each group's OWN
    #                normalised outcome range - the p the weights are 1/p of.
    CSV_COLS += ["tail/w_max", "tail/w_p90", "tail/groups", "tail/n_med",
                 "tail/ess", "tail/cov", "tail/p50", "tail/p75", "tail/p90"]
    csv_f = csv_w = None
    if D.is_main:                    # four append handles corrupt the file
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
                print(f"progress.csv header extended to "
                      f"{len(CSV_COLS)} columns")
        if csv_path.exists() and csv_path.stat().st_size and args.ckpt is not None:
            # a resume re-runs the steps after its checkpoint, so the rows
            # past global_step are the abandoned tail of the previous life
            # and would fold the x-axis back. Drop them; the file stays
            # monotone in time/total_timesteps.
            _text = csv_path.read_text(encoding="utf-8").splitlines(True)
            _head = _text[0].rstrip("\r\n").split(",")
            _xi = (_head.index("time/total_timesteps")
                   if "time/total_timesteps" in _head else 0)
            _keep, _dropped = [_text[0]], 0
            for _ln in _text[1:]:
                try:
                    _st = float(_ln.rstrip("\r\n").split(",")[_xi])
                except (ValueError, IndexError):
                    _st = None
                if _st is None or _st <= global_step:
                    _keep.append(_ln)
                else:
                    _dropped += 1
            if _dropped:
                csv_path.write_text("".join(_keep), encoding="utf-8")
                print(f"progress.csv: dropped {_dropped} rows past the "
                      f"checkpoint step {global_step:,} (resume)")
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
    b_scal = torch.zeros((T, N, SCAL), device=device)
    # --priv-critic: the critic's privileged block for every stored decision.
    # (T, N, 10) fp32 = 5 MB at T=32/N=2048 next to b_img's gigabytes; the
    # update gathers it exactly like b_scal. PRIV is 0 without the flag and
    # torch.zeros((T, N, 0)) allocates nothing, so the control run's memory
    # and its gathers are untouched.
    PRIV = PRIV_DIM if args.priv_critic else 0
    b_priv = torch.zeros((T, N, PRIV), device=device)
    # single-frame per timestep even under --frame-stack; the PRO leading rows
    # hold the PREVIOUS iteration's tail so a t=0 sample reaches back into
    # real history instead of clamping (which would fake an episode start
    # every T decisions, and the value head would learn the period)
    b_img = torch.zeros((PRO + T, N, FRAME), device=device,
                        dtype=torch.bfloat16 if use_bf16 else torch.float32)
    b_age = (torch.zeros((T, N), dtype=torch.long, device=device)
             if STACK > 1 else None)
    # --chunk: `t` indexes a CHUNK, so the acted 6-tuples grow a decision
    # axis. Per-decision actions are sampled from the decoder (they are NOT a
    # deterministic function of the code), so PPO has to score them, which is
    # why they are stored rather than re-derived. At T=16/N=2048/H=10 that is
    # 16 MB — b_img shrank by T anyway (one render per CHUNK, not per
    # decision), so the rollout's total footprint drops.
    b_act = torch.zeros((T, N, NACT) if H == 0 else (T, N, H, NACT),
                        dtype=torch.long, device=device)
    ACT_FLAT = (T * N, NACT) if H == 0 else (T * N, H, NACT)
    # --view-continuous: the pre-tanh z of the two view heads per decision,
    # the action PPO scores (the int row above keeps NEUTRAL there)
    NZ = int(getattr(policy, "n_z", N_VIEW)) if VIEWC else 0   # 3 in world mode
    b_z = torch.zeros((T, N, NZ), device=device) if VIEWC else None
    # the acted code, and the per-decision mask of decisions that ACTUALLY ran
    # from the decoder (0 where a mid-chunk episode end forced NEUTRAL_ACT).
    # Masked decisions are excluded from the recomputed joint log-prob and
    # from the decoder entropy — the ratio must only cover what pi emitted.
    b_code = (torch.zeros((T, N), dtype=torch.long, device=device)
              if H > 0 else None)
    b_dmask = torch.zeros((T, N, H), device=device) if H > 0 else None
    # ez-greedy state: how many more decisions each env is committed to its
    # burst action, and what that action is. b_ez marks the transitions that
    # were NOT drawn from the policy, so the update can drop them.
    ez_left = torch.zeros(N, dtype=torch.long, device=device)
    ez_act = torch.zeros((N, NACT), dtype=torch.long, device=device)
    b_ez = torch.zeros((T, N), dtype=torch.bool, device=device)
    NVEC_T = torch.tensor(NVEC, device=device)
    # Go-Explore post-return exploration state (--spawn-burst): decisions
    # left in each env's post-respawn random burst and the current held
    # action. Shares the b_ez off-policy mask/exclusion with ez-greedy.
    USE_BURST = args.ez_eps > 0.0 or args.spawn_burst > 0
    sb_left = torch.zeros(N, dtype=torch.long, device=device)
    sb_act = torch.zeros((N, NACT), dtype=torch.long, device=device)
    # per-env distance-bin of the CURRENT episode's start (-1 = unknown /
    # invalid), for attributing episode outcomes to start bins
    start_bin = np.full(N, -1, np.int64)
    track_bins = respawn is not None and respawn.mode != "uniform"
    # --tail-weight (TailRL, arXiv 2609.02987). TAILW == 0.0 keeps every
    # branch below unentered, so the control run is the run that shipped.
    #   tail_eps   one tuple per episode that ENDED inside this rollout:
    #              (decision index t, env, return, ticks, finished, group)
    #   tail_seg   first decision index of each env's in-buffer segment.
    #              An episode that carried over the buffer edge is weighted
    #              from t = 0 (its outcome is known once it ends), and one
    #              still running at the edge keeps weight 1 - stated,
    #              because that is the one place the reweighting is
    #              incomplete.
    TAILW = float(args.tail_weight or 0.0)
    tail_eps: list[tuple] = []
    tail_seg = np.zeros(N, np.int64)
    #   tail_bin   the goal-distance bin each env's CURRENT episode spawned
    #              in - its TailRL group. Written off the fleet's own race
    #              field, never off the reservoir, so the flag neither needs
    #              --respawn-binned nor changes the start distribution.
    #   tail_bin0  the same for the episode that most recently ENDED. Under
    #              --reward-per-decision the return is only final at the
    #              decision boundary, by which point the per-tick stash has
    #              already moved tail_bin to the FRESH episode's bin, so the
    #              ending episode's group is snapshotted while still true.
    #              The per-tick path records off tail_bin directly, because
    #              there the episode is booked BEFORE the stash runs.
    tail_bin = np.full(N, -1, np.int64)
    tail_bin0 = np.full(N, -1, np.int64)
    TAIL_BINS = max(1, int(args.tail_bins))
    tail_stats = None
    if TAILW > 0.0:
        if slots[0].reward_field is None or not slots[0].rf_d0:
            print("!! --tail-weight with no race goal field: every episode "
                  "falls into ONE group, which is the map-start reading of "
                  "'N rollouts for the same input' and a much weaker one")
        print(f"--tail-weight {TAILW:g}: TailRL advantage reweighting, "
              f"outcome={args.tail_outcome}, groups = spawn depth over "
              f"{TAIL_BINS} bins (min {args.tail_min_n} episodes), weights "
              f"mean 1 per group. Watch tail/n_med: if it falls under "
              f"{args.tail_min_n} the groups are too fine to estimate a tail")
    demo_idx = np.full(N, -1, np.int64)   # demo index of each env's start
    b_logp = torch.zeros((T, N), device=device)
    b_val = torch.zeros((T, N), device=device)
    b_rew = torch.zeros((T, N), device=device)
    b_done = torch.zeros((T, N), device=device)
    # --mask-*: the two flags the rollout MASKED WITH, recorded per decision
    # so the update's log-prob recomputation replays exactly the same
    # distribution. b_air could be re-derived from b_scal slot 4 (it is the
    # same column), but PPO's ratio is only sound if pi_old and pi_new are
    # the same measure, and storing the flag makes that structural instead
    # of an argument about which column means what. 2x (T, N) fp32.
    b_air = torch.zeros((T, N), device=device) if MASKS.on else None
    b_jblk = (torch.zeros((T, N), device=device)
              if MASKS.jump_cd > 0 else None)

    static_obs = torch.zeros((N, obs_dim), device=device)
    static_act = torch.zeros((N, NACT), dtype=torch.long, device=device)
    # --jump-cooldown: decisions of jump lockout left, per env. A STATIC
    # buffer, read inside the captured graph and written in place outside it
    # (exactly like static_obs), so the graph replays the live counter.
    static_jcd = (torch.zeros(N, device=device) if MASKS.jump_cd > 0
                  else None)
    jcd_reload = (torch.full((N,), float(MASKS.jump_cd), device=device)
                  if MASKS.jump_cd > 0 else None)
    static_logp = torch.zeros(N, device=device)
    static_val = torch.zeros(N, device=device)
    # --view-continuous: the drawn z and the view command the core gets,
    # STATIC buffers written inside the captured graph like static_act
    static_z = torch.zeros((N, NZ), device=device) if VIEWC else None
    static_view = torch.zeros((N, N_VIEW), device=device) if VIEWC else None
    VIEW_PITCH_MAX = float(core.config.pitch_rate_max_deg) if VIEWC else 0.0
    # --rnn: the per-env recurrent state. static_h is the state ENTERING the
    # next decision - the graphed step reads it and writes the state leaving
    # the decision back into it; the rollout loop then zeroes the rows whose
    # episode ended (a respawn from the reservoir is a new episode: the
    # reservoir holds physics states, not memories, and an eval placed at
    # that state would start from zero too - carrying a stale h across the
    # respawn would train the policy on a memory no eval can reproduce).
    # b_h0 is the state each env entered THIS rollout with, from which the
    # update's sequence re-run starts (truncated BPTT over T; the state
    # carried across the rollout boundary was made by the previous weights
    # and is treated as a constant). b_done_np mirrors b_done on the host
    # for cutting the sequences (gru_segments) without a device sync.
    static_h = torch.zeros((N, max(R, 1)), device=device)
    b_h0 = torch.zeros((N, max(R, 1)), device=device)
    b_done_np = np.zeros((T, N), bool)

    obs_pin = torch.zeros((N, N_SCALAR), pin_memory=(device.type == "cuda"))
    act_pin = torch.zeros((N, NACT), dtype=torch.long,
                          pin_memory=(device.type == "cuda"))
    act_np32 = np.zeros((N, NACT), dtype=np.int32)
    view_pin = (torch.zeros((N, N_VIEW), pin_memory=(device.type == "cuda"))
                if VIEWC else None)
    view_np32 = np.zeros((N, N_VIEW), dtype=np.float32) if VIEWC else None
    # --chunk staging. step_compute samples the code AND all H decisions in
    # one graph replay (the decoder reads no observation, so the whole plan is
    # available at the chunk start) — so the chunk costs ONE D2H sync, not H,
    # and the tick loop below never touches the GPU.
    if H > 0:
        static_code = torch.zeros(N, dtype=torch.long, device=device)
        static_plan = torch.zeros((N, H, NACT), dtype=torch.long, device=device)
        static_dlogp = torch.zeros((N, H), device=device)
        plan_pin = torch.zeros((N, H, NACT), dtype=torch.long,
                               pin_memory=(device.type == "cuda"))
        dmask_gpu = torch.zeros((N, H), device=device)
        dmask_np = np.ones((N, H), np.float32)
        NEUTRAL_NP = np.array(NEUTRAL_ACT, np.int32)

    # GPU vision fusion: pose upload (one pinned (N, 6) copy) -> SDF ray-march
    # -> lidar slice of the obs. States are read post-step/post-autoreset, so
    # the depth image always matches the scalar obs row.
    sv_view = core.states_view
    # --pitch-fixed: None = off, and off touches no array the control did not
    PITCH_PIN = args.pitch_fixed
    # --maps: one staging tensor the per-slot renders are written into. Never
    # allocated on the single-map path, where render() returns the
    # renderer's own tensor exactly as it always did.
    img_stage = (torch.zeros((N, FRAME), device=device) if MULTI else None)
    vis_pin = torch.zeros((N, 6), pin_memory=(device.type == "cuda"))
    vis_np = vis_pin.numpy()
    vis_gpu = torch.zeros((N, 6), device=device)
    # --race-latch: a pinned staging row for the one flag column
    latch_pin = torch.zeros((N, 1), pin_memory=(device.type == "cuda"))
    latch_np = latch_pin.numpy()[:, 0]
    # --priv-critic: one pinned (N, 10) staging block, filled in place off
    # the live core states and the reward object, then uploaded with the
    # rest of the per-decision traffic. static_priv is a STATIC buffer like
    # static_obs, so the captured CUDA graph reads whatever the host wrote
    # into it before the replay - the same contract latch_pin has.
    priv_pin = (torch.zeros((N, PRIV), pin_memory=(device.type == "cuda"))
                if PRIV else None)
    priv_np = priv_pin.numpy() if PRIV else None
    static_priv = torch.zeros((N, PRIV), device=device) if PRIV else None
    # --act-hist / --obs-compass: one pinned staging block, filled in place by
    # ObsAux so the per-decision path allocates nothing
    aux_pin = (torch.zeros((N, N_AUX), pin_memory=(device.type == "cuda"))
               if N_AUX else None)
    aux_np = aux_pin.numpy() if N_AUX else None

    # --frame-stack: a per-env ring of past renders, held OUTSIDE the CUDA
    # graph. The graph captures step_compute() over static_obs alone, so as
    # long as the composed stack lands in static_obs' image slice before the
    # replay, the graphed region never learns this feature exists.
    ring = FrameRing(STACK, N, FRAME, device) if STACK > 1 else None
    # with no stack to compose, the frame b_img records IS static_obs' image
    # slice (a view, not a copy); with one, the ring holds it
    cur = static_obs[:, SCAL:]

    def fill_vision(dst, ended=None):
        """Render this decision's frame into `dst`. With --frame-stack the
        render goes through the ring and `dst` receives the composed stack;
        `ended` (bool, N) collapses an env's history to its spawn frame."""
        t0 = tm.now()
        if PITCH_PIN is not None:
            # --pitch-fixed: aim every ray at the pinned angle. This runs
            # AFTER the reward call and after the respawn, so a reservoir
            # state's inherited pitch never reaches a render; with
            # pitch_rate 0 the C step then carries the same value into obs
            # slot 9, so the scalars and the image agree.
            for _sl in fleet.slots:
                _sl.core.set_pitch(PITCH_PIN)
        fleet.fill_pose(vis_np)
        vis_gpu.copy_(vis_pin, non_blocking=True)
        # --route: the lookahead fan rides the SAME pose upload the renderer
        # needs, so it costs one argmin and no extra host->device traffic.
        # Speed comes from the scalar row (slot 3 = |v_xy|/1000), which the
        # caller has already refreshed — so the fan and the depth image and
        # the scalars all describe one instant.
        if route is not None:
            dst[:, N_SCALAR:N_SCALAR + N_FAN] = route.features(
                vis_gpu[:, 0:3], vis_gpu[:, 3], dst[:, 3] * 1000.0)
        if N_LATCH:
            # fill_vision runs AFTER the reward call, so this is the flag
            # as of the state the policy is about to act on - i.e. the
            # one that decides whether the NEXT reward pays shaping.
            # 8 KB of host->device per decision, off the graph.
            latch_np[:] = fleet.latch_flags()
            dst[:, LATCH_COL:LATCH_COL + 1].copy_(latch_pin,
                                                  non_blocking=True)
        if N_AUX:
            # the history as of the decision about to be made (its rows were
            # zeroed for every env that just ended) and the compass at the
            # pose this same call is about to render from - so the history,
            # the compass, the fan, the depth image and the scalars all
            # describe ONE instant.
            obs_aux.features(vis_np[:, 0:3], vis_np[:, 3], out=aux_np)
            dst[:, AUX0:SCAL].copy_(aux_pin, non_blocking=True)
        if PRIV:
            # the CRITIC's block, at the same instant as everything above:
            # this call runs AFTER the reward and after the autoreset, so
            # RaceReward's d, arc anchor and latch flag all describe the
            # state the policy is about to act on. ~82 KB of host->device
            # per decision at N=2048, off the graph, next to the pose
            # upload's 49 KB.
            fleet.fill_priv(priv_np)
            static_priv.copy_(priv_pin, non_blocking=True)
        ev = tm.gpu_start("lidar")
        # (N,H,W) or (N,H,W,2) under --surf-mask; flattening keeps the
        # channel fastest, which is what Policy.forward_split restrides.
        # --maps: each slot marches ITS map's SDF into its own env block.
        img = fleet.render(vis_gpu, img_stage)
        if ring is None:
            dst[:, SCAL:].copy_(img)
        else:
            ring.push(img, ended)
            dst[:, SCAL:].copy_(ring.compose())
        tm.gpu_end(ev)
        tm.add("vis_cpu", t0)

    def roll_mask(padded):
        """PLACE 1 of 4. static_obs is the row this decision is being made
        on, so slot 4 is the on-ground flag AT the decision tick; static_jcd
        is the cooldown as of this decision. Both are static buffers, so
        this captures into the CUDA graph like everything else in
        step_compute. Hoisted out of step_compute so --yaw-cond can apply
        it AFTER the conditioning without a second textual call site."""
        return MASKS.add_mask(
            padded,
            static_obs[:, OBS_ONGROUND] < 0.5 if MASKS.needs_air else None,
            static_jcd > 0 if MASKS.jump_cd > 0 else None)

    # MASKS.on is a Python constant: with no flag ROLL_MASK is None, the
    # branch is decided at trace time and the graph is the one that shipped.
    ROLL_MASK = roll_mask if MASKS.on else None

    def step_compute():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_bf16):
            if RNN:
                # the GRU step is INSIDE the captured graph: static_h is a
                # static buffer like static_obs, read as the entering state
                # and overwritten with the leaving one (probed: a seq_len-1
                # cuDNN GRU captures and replays bit-identically to eager;
                # gru_step disables autocast itself, which would otherwise
                # hand cuDNN fp16)
                logits, value, h1 = policy(static_obs, static_h,
                                           priv=static_priv)
                static_h.copy_(h1)
            else:
                logits, value = policy(static_obs, priv=static_priv)
        if H > 0:
            # one categorical over codes, then the code's whole (H, 6) plan
            # out of the decoder in one gather+gumbel. All shapes constant —
            # this captures into the CUDA graph exactly like the flat path.
            code, logp = sample_code(logits.float())
            plan, dlogp = sample_padded(
                packer.pad_seq(policy.decoder[code].float()))
            static_code.copy_(code)
            static_plan.copy_(plan)
            static_dlogp.copy_(dlogp)   # per-decision; the JOINT logp is
            # assembled after the chunk, when the neutral mask is known
        elif VIEWC:
            # --view-continuous: the four categorical heads by the same
            # Gumbel draw on the padded slice, the view by z = mu + sigma *
            # eps; the core command is derived here so the host copies one
            # more pinned row and never touches the GPU in the tick loop
            cat, mu = split_view(logits.float())
            padded = packer.pad(cat)
            if ROLL_MASK is not None:
                padded = ROLL_MASK(padded)
            act, z, logp = sample_view(padded, mu, policy.log_std())
            static_act.copy_(act)
            static_z.copy_(z)
            static_view.copy_(view_from_z_t(z, VIEW_PITCH_MAX, VIEW_ABS))
        else:
            padded = packer.pad(logits.float())
            if YCOND:
                # PLACE 1 of 4 for --yaw-cond: the yaw bin is drawn first
                # and the side key from logits carrying that bin's row of
                # the conditioning table. One rand_like and static shapes,
                # exactly like sample_padded, so this captures into the
                # CUDA graph the same way; ROLL_MASK is applied INSIDE and
                # AFTER the conditioning, so a masked slot stays at NEG.
                act, logp = sample_padded_yawcond(
                    padded, policy.yaw_side.table, ROLL_MASK)
            else:
                if ROLL_MASK is not None:
                    padded = ROLL_MASK(padded)
                act, logp = sample_padded(padded)
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
            # thread_local: a live NCCL communicator runs a watchdog thread
            # doing cudaEventQuery, which the default "global" mode counts
            # as a capture-invalidating call — and the except below would
            # turn that into a silent graph=None on a race-dependent subset
            # of ranks (numerics identical, throughput and skew not)
            with torch.no_grad(), torch.cuda.graph(
                    graph, capture_error_mode="thread_local"):
                step_compute()
            print("CUDA graph captured for the rollout step")
        except Exception as exc:  # pragma: no cover
            print(f"CUDA graph capture failed ({exc!r}) — eager fallback")
            graph = None
    if D.enabled:
        # log which ranks captured — a partial-capture fleet is legal but
        # its skew profile is not the one the perf numbers assume
        n_cap = torch.tensor([0.0 if graph is None else 1.0], device=device)
        D.all_reduce_sum_(n_cap)
        if int(n_cap) != D.world_size:
            print(f"WARNING: CUDA graph captured on {int(n_cap)}/"
                  f"{D.world_size} ranks (eager elsewhere)")

    def policy_step():
        ev = tm.gpu_start("rollout_fwd")
        if graph is not None:
            graph.replay()
        else:
            with torch.no_grad():
                step_compute()
        tm.gpu_end(ev)

    # DDP counts sharing needs touched-cell tracking armed BEFORE on_reset:
    # on_reset only allocates the delta base when tracking is on (a ~256 MB
    # never-read duplicate otherwise), and arming it later would leave the
    # first sync without a base. Per SLOT - the novelty table is keyed by
    # that map's cells and slot 0's is not the fleet's.
    if D.enabled:
        for _s in slots:
            if (isinstance(_s.reward_fn, RaceReward)
                    and _s.reward_fn.int_coef > 0.0):
                _s.reward_fn.track_touched = True
    # (a) env streams rank-DISTINCT and exactly a partition: env.c derives
    # stream i from seed+i, so rank r's envs draw streams seed + r*N + i -
    # the union over ranks is bit-for-bit the SET a single-GPU N_GLOBAL-env
    # run with the same seed draws. MapFleet.reset offsets slot i by 1013*i
    # on top, so the rank shares of one map stay a partition too.
    goalsys = None
    if args.goals:
        from surfgym.goalsys import GoalSystem
        if respawn is None or MULTI:
            raise SystemExit("--goals needs the respawn reservoir and a "
                             "single map (per-slot goals: plan G5)")
        _ball = _eval_ball = None
        if args.goal_obs in ("ball", "both"):
            from surfgym.goalball import GoalBallLidar
            _ball = slots[0].lidar
            _eval_ball = GoalBallLidar(_raw_lidar[slots[0].name], 1,
                                       radius=float(args.goal_radius),
                                       views=int(args.goal_views))
            print(_ball.describe())
        goalsys = GoalSystem(core, N, route, slots[0].goal_field,
                             slots[0].d0, args, device, out,
                             seed=args.seed + 777, ball=_ball,
                             eval_ball=_eval_ball,
                             arc=(arc_line if args.goal_reward == "arc"
                                  else None),
                             reward_fn=slots[0].reward_fn,
                             dist_field=goal_dist_field,
                             # reached-state goals arrive as a COUNT of
                             # reservoir snapshots; the curriculum's k is
                             # in seconds, so the assigner needs the cadence
                             snap_every=respawn.snap_every,
                             tick_ms=TICK.ms)
        if slots[0].goal_box is not None:
            goalsys.set_finish(slots[0].goal_box["mins"],
                               slots[0].goal_box["maxs"])
        if (goal_dist_field is not None
                and getattr(slots[0], "eval_aux", None) is not None
                and slots[0].eval_aux.blocks):
            # --obs-compass + --goal-reward euclid/geo: hand GoalSystem the
            # eval's 1-wide distance field so every eval goal re-centres it
            goalsys.eval_dist_field = slots[0].eval_aux.blocks[0][1]
        print(goalsys.describe())
        if goalsys.fixed:
            print(goalsys.describe_fixed())
    obs_np = fleet.reset(args.seed + D.rank * N).copy()
    fleet.on_reset()
    if goalsys is not None:
        goalsys.assign(np.arange(N))
    prev_obs = obs_np.copy()
    obs_pin.copy_(torch.from_numpy(obs_np))
    static_obs[:, :N_SCALAR].copy_(obs_pin, non_blocking=True)
    fill_vision(static_obs)
    if args.obs_reward:
        static_obs[:, REWARD_SLOT] = 0.0     # no previous reward at reset
    if RNN:
        # every env just started an episode; also undoes the GRU steps the
        # graph warm-up/capture ran on the zeroed static_obs above
        static_h.zero_()

    # ---- startup invariants (docs/ddp-plan.md step 7) ----------------------
    # Turn silent divergence into a startup failure: everything the shared
    # gradient depends on must be rank-identical, and the env streams must
    # NOT be. All collectives here are unconditional on every rank.
    # every slot, in slot order: slot 0's spawns alone would still be
    # rank-distinct if a later map's cores had collapsed, and the point of
    # the check is that nothing in the fleet is a duplicate of another rank
    origin_bytes = b"".join(
        np.ascontiguousarray(s.core.states_view["origin"]).tobytes()
        for s in slots)
    if D.enabled or args.dump_invariants:
        cfg_json = json.dumps(meta["config"], sort_keys=True, default=str)
        inv_i64 = torch.tensor(
            [h64(b"".join(np.ascontiguousarray(s.pool).tobytes()
                          for s in slots)),
             h64(b"".join(
                 np.ascontiguousarray(
                     getattr(s.goal_field, "grid", np.zeros(1))).tobytes()
                 for s in slots)),
             h64(cfg_json.encode()),
             obs_dim, N, D.world_size, args.minibatches, args.epochs, T,
             NMAPS, PER, h64("|".join(STEMS).encode()),
             NHELD, h64("|".join(HSTEMS).encode())],
            dtype=torch.int64, device=device)
        # EVERY map's d0, not slot 0's: a rank that silently resolved a
        # different BSP for slot 3 (a stale cache, a half-synced staging
        # directory) would otherwise pass every check here and then train
        # one map against another map's reward scale
        inv_f64 = torch.cat([param_checksum(),
                             torch.tensor([(s.d0 if s.d0 is not None else 0.0)
                                           for s in slots + heldout],
                                          dtype=torch.float64,
                                          device=device)])
        D.assert_equal("startup_invariants_i64", inv_i64)
        D.assert_equal("startup_invariants_f64", inv_f64)
        # the single most valuable line in the plan: catches a reverted
        # reset seed, a stray global manual_seed, and any future
        # "reproducibility fix" that collapses the fleet into R copies
        D.assert_distinct("spawn_origins", h64(origin_bytes))
    if args.dump_invariants:
        fleet_origins = b"".join(D.all_gather_var_bytes(origin_bytes))
        print(json.dumps({
            "fleet_origins_hash": h64(fleet_origins),
            "pool_hash": h64(pool),
            "race_d0": race_d0,
            "maps": list(STEMS), "n_maps": NMAPS, "envs_per_slot": PER,
            "map_d0": {s.tag: s.d0 for s in slots},
            # --heldout-maps: listed apart from "maps", and the env rows they
            # own are asserted ZERO here - the C1-style gate that a held-out
            # map never contributes a rollout row (train_envs is the fleet's
            # whole env count, i.e. the training slots alone)
            "heldout": list(HSTEMS), "n_heldout": NHELD,
            "heldout_d0": {s.tag: s.d0 for s in heldout},
            "heldout_envs": int(sum(s.n for s in heldout)),
            "train_envs": int(fleet.n_envs),
            "heldout_in_fleet_slots": int(sum(
                1 for s in fleet.slots if getattr(s, "heldout", False))),
            "param_checksum": float(param_checksum()[0]),
            "obs_dim": obs_dim, "envs_per_rank": N,
            "world_size": D.world_size, "mb": T * N // args.minibatches,
            "grad_steps_per_iter": args.epochs * args.minibatches,
            "seed": args.seed}, sort_keys=True), flush=True)
        D.finalize()
        return
    ep_ret = np.zeros(N, np.float64)
    ep_len = np.zeros(N, np.int64)
    # race/crawl_frac: running sum of |v_xy| over the live episode, read off
    # obs slot 3 (which src/env.c writes as |v_xy|/1000), so the whole
    # measurement is one numpy add per physics tick on an array the rollout
    # already has in hand - no extra states_view read, no extra copy.
    spd_sum = np.zeros(N, np.float64)
    CRAWL_KU = CRAWL_SPEED / 1000.0    # 300 u/s, in slot 3's ku/s units
    ret_hist = deque(maxlen=200)     # bounded: a 10B run finishes ~10M episodes
    len_hist = deque(maxlen=200)

    next_record = (global_step + int(args.record_every)
                   if args.no_eval_at_start else global_step)
    next_ckpt = global_step + int(args.ckpt_every)
    last_latest_save = 0.0                   # force one write on iteration 1
    eval_fwd = eval_path = eval_speed = eval_prog = eval_fin = float("nan")
    # the two aggregates the multi-map run is judged on (see the eval block):
    # mean over maps of the % of that map's own route covered, and the
    # fraction of maps with at least one box finish; then the same two
    # restricted to REAL trigger finishes, because a button box is ~8x
    # smaller in face area and a null on one is much weaker evidence.
    agg_pct = agg_fin_frac = float("nan")
    agg_pct_trig = agg_fin_trig = float("nan")
    # --maps: (eval_progress, eval_finish_s, box finishes, % covered) per map
    # tag, carried between evals exactly like the aggregates above
    eval_per_map = {s.tag: (float("nan"),) * 4 for s in slots + heldout}
    # --heldout-maps: the order-only corridor MAX per held-out map that has
    # a route file, carried between evals like the rest
    held_corr = {s.tag: float("nan") for s in heldout if s.route is not None}
    t_start, step_start = time.perf_counter(), global_step

    def save_ckpt(tag):
        # rank 0 only, and the branch is collective-free BY CONSTRUCTION:
        # the shared tables (counts, respawn) are replicated on every rank,
        # so rank 0's copy IS the fleet's and no gather is needed. Never
        # add a collective anywhere under this function (plan §6.7).
        if not D.is_main:
            return
        state = {"policy": policy.state_dict(),
                 "optimizer": contiguous_optimizer_state(opt.state_dict()),
                 "global_step": global_step, "config": meta["config"]}
        if isinstance(reward_fn, RaceReward):
            # novelty counts are cross-episode reward state: without them a
            # resume re-pays "first visit" for the whole beaten path. A
            # single-map run writes the bare array it always wrote, so old
            # tooling still reads its checkpoints; multi-map writes a dict
            # keyed by map stem, because a count table is one map's cells.
            state["int_counts"] = (
                {s.name: s.reward_fn.counts_state() for s in slots} if MULTI
                else reward_fn.counts_state())
        if RETN:
            # the running (mu, sigma) IS part of the value function under
            # --ret-norm: without it the restored critic's outputs have no
            # scale. Written only when the flag is on, so a control run's
            # checkpoint keeps exactly the keys it has today.
            state["ret_norm"] = retn.state_dict()
        if rnd is not None:
            state["rnd"] = rnd.state_dict_all()   # target net INCLUDED: a
            # re-rolled target makes every fitted state novel again
        if respawn is not None:
            state["respawn"] = (      # keep the frontier
                {s.name: s.respawn.state_dict() for s in slots} if MULTI
                else respawn.state_dict())
        torch.save(state, out / f"ckpt_{tag}.pt")

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                         enabled=use_bf16)

    # ---- the PPO minibatch step, as one compilable function -----------------
    # Everything here is static-shaped (mb is constant), so inductor sees one
    # graph and never re-traces. The gathers stay INSIDE: fusing them with the
    # bf16 cast is part of what the compile buys.
    ent_t = torch.zeros((), device=device)

    def mb_step(f_scal, f_img, f_act, f_logp, f_adv, f_ret, idx, ent_coef,
                f_age=None, f_code=None, f_dmask=None,
                adv_mean=None, adv_std=None, f_air=None, f_jblk=None,
                f_priv=None, f_z=None):
        with amp:
            # STACK/N/PRO are Python constants, so the branch is decided at
            # trace time and inductor still sees one static-shaped graph
            img = (f_img[idx] if STACK == 1 else
                   stack_from_buffer(f_img, idx, f_age[idx], STACK, N, PRO))
            # --priv-critic: the same gather the scalars take. PRIV is a
            # Python constant, so this branch is decided at trace time too
            # and the control run compiles the graph it always compiled.
            logits, value = policy.forward_split(
                f_scal[idx], img,
                priv=(None if f_priv is None else f_priv[idx]))
            if H > 0:
                # the JOINT log-prob PPO's ratio is taken over:
                #   log pi(code | s_chunk)  +  sum_h log p(a_h | dec[code, h])
                # The trunk gives the first term (one forward per CHUNK, from
                # the chunk-start obs in f_scal/f_img); the second needs only
                # the stored actions and the LIVE decoder, which is where the
                # decoder's gradient comes from.
                logp, ent = logprob_entropy_code(logits.float(), f_code[idx])
                dl = packer.pad_seq(policy.decoder[f_code[idx]].float())
                dlp, dent = logprob_entropy_padded(dl, f_act[idx])
                m = f_dmask[idx]                # (mb, H): decisions that ran
                logp = logp + (dlp * m).sum(-1)
                # summed over the chunk's live decisions, so the coefficient
                # sees the entropy of the whole H-decision behavior, not a
                # per-decision average that would shrink with H
                dl_ent = (dent * m).sum(-1)
            else:
                if VIEWC:
                    # --view-continuous: the categorical slice pads as
                    # before; the two means go to the Gaussian term below
                    cat, mu = split_view(logits.float())
                    padded = packer.pad(cat)
                else:
                    padded = packer.pad(logits.float())
                if YCOND:
                    # PLACE 2 of 4 for --yaw-cond. The STORED yaw action,
                    # gathered by the same idx, so pi_new conditions on
                    # exactly what pi_old conditioned on and the ratio is a
                    # true importance ratio. BEFORE the masks (ORDER note
                    # by _YawSideCond). The side head's entropy term below
                    # is then H(side | yaw = this row's yaw) - see the
                    # ENTROPY note there.
                    padded = add_yaw_cond(padded, policy.yaw_side.table,
                                          f_act[idx][:, H_YAW])
                if MASKS.on:
                    # PLACE 2 of 4. The SAME flags the rollout sampled under,
                    # gathered by the same idx, so pi_new is the identical
                    # measure pi_old was and exp(logp_new - logp_old) is a
                    # true importance ratio. Re-deriving the flags from
                    # f_scal here instead would work today and break the
                    # moment an obs column moves; f_air/f_jblk is what was
                    # actually used. f_air is None on every unmasked run, so
                    # inductor traces the shipped graph.
                    padded = MASKS.add_mask(
                        padded, None if f_air is None else f_air[idx],
                        None if f_jblk is None else f_jblk[idx])
                if VIEWC:
                    # the joint over the four categorical heads and the
                    # Gaussian density of the STORED z (the action taken),
                    # gathered by the same idx; the entropy adds the
                    # Gaussian's, which is what keeps sigma from collapsing
                    logp, ent = logprob_entropy_view(
                        padded, f_act[idx], mu, policy.log_std(), f_z[idx])
                else:
                    logp, ent = logprob_entropy_padded(padded, f_act[idx])
            value = value.float()
        ratio = torch.exp(logp - f_logp[idx])
        a = f_adv[idx]
        if adv_mean is None:
            # world_size==1 keeps the LITERAL estimator so the single-GPU
            # path stays bit-identical — do not "unify" the two branches
            a = (a - a.mean()) / (a.std() + 1e-8)   # per-minibatch, like SB3
        else:
            # DDP: the moments are fleet-wide (all-reduced per epoch, plan
            # §2) so the estimator runs over the same T*N_GLOBAL/M rows a
            # single-GPU minibatch normalises over. A per-shard split would
            # rescale each rank's loss by its own sample std (~1.1% sd at
            # 4096 rows) — systematic, permanent, and absent from every log
            a = (a - adv_mean) / (adv_std + 1e-8)
        pg = torch.max(-a * ratio,
                       -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
        vl = 0.5 * (value - f_ret[idx]).pow(2).mean()
        el = -ent.mean()
        loss = pg + args.vf * vl + ent_coef * el
        if H > 0:
            # two entropy terms, opposite jobs: ent_coef keeps CODE CHOICE
            # exploratory, dec_ent lets a code crystallize into a distinct
            # behavior. Both are needed — code entropy alone is satisfied by
            # 64 identical codes (that IS code collapse), decoder entropy
            # alone by one code the policy always picks.
            loss = loss - args.dec_ent * dl_ent.mean()
        return loss, pg, vl, el, logp

    # ---- --rnn: the SEQUENCE minibatch step ---------------------------------
    # A minibatch is B = N/minibatches whole envs x all T decisions, in
    # time-major row order (idx = t*N + env), so every flat buffer is gathered
    # exactly as mb_step gathers it and f_adv/f_logp/f_ret need no reshaping.
    # The recurrence is re-run from b_h0 with the rollout's episode cuts
    # (gru_segments), truncated BPTT over the T decisions. Three pieces:
    # the trunk and the loss are static-shaped and compiled like mb_step; the
    # cuDNN recurrence between them is eager (inductor does not lower it,
    # and the segment layout is data-dependent anyway) - one rectangular
    # call per minibatch, ~6 ms fwd+bwd at T=128, B=64 on the 5090 against
    # ~8 ms packed and far more for a Python loop over T (gru_sequence).
    def seq_trunk(f_scal, f_img, idx, f_age=None):
        with amp:
            img = (f_img[idx] if STACK == 1 else
                   stack_from_buffer(f_img, idx, f_age[idx], STACK, N, PRO))
            return policy.features(f_scal[idx], img)   # fp32 (cat promotes)

    def seq_loss(feat, g, f_scal, f_act, f_logp, f_adv, f_ret, idx, ent_coef,
                 adv_mean=None, adv_std=None, f_air=None, f_jblk=None,
                 f_priv=None):
        with amp:
            logits, value = policy.heads(
                feat, f_scal[idx], g,
                priv=(None if f_priv is None else f_priv[idx]))
            padded = packer.pad(logits.float())
            if YCOND:
                # PLACE 2 of 4 for --yaw-cond, the --rnn half (same
                # argument as mb_step: the STORED yaw, before the masks)
                padded = add_yaw_cond(padded, policy.yaw_side.table,
                                      f_act[idx][:, H_YAW])
            if MASKS.on:
                # PLACE 2 of 4, the --rnn half (same argument as mb_step)
                padded = MASKS.add_mask(
                    padded, None if f_air is None else f_air[idx],
                    None if f_jblk is None else f_jblk[idx])
            logp, ent = logprob_entropy_padded(padded, f_act[idx])
            value = value.float()
        ratio = torch.exp(logp - f_logp[idx])
        a = f_adv[idx]
        if adv_mean is None:
            a = (a - a.mean()) / (a.std() + 1e-8)   # per-minibatch, like SB3
        else:
            a = (a - adv_mean) / (adv_std + 1e-8)   # DDP: fleet-wide moments
        pg = torch.max(-a * ratio,
                       -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
        vl = 0.5 * (value - f_ret[idx]).pow(2).mean()
        el = -ent.mean()
        loss = pg + args.vf * vl + ent_coef * el
        return loss, pg, vl, el, logp

    def mb_step_seq(f_scal, f_img, f_act, f_logp, f_adv, f_ret, idx, ent_coef,
                    f_age=None, adv_mean=None, adv_std=None, f_air=None,
                    f_jblk=None, f_priv=None, *, envs, seg):
        # `envs` (B,) the minibatch's env ids in idx order, `seg` the
        # SeqPlan cut from b_done_np[:, envs]. seq_trunk/seq_loss are looked
        # up at call time, so the compile block below can rebind them. The
        # recurrence sits between them in fp32 on `feat` (fp32: autocast's
        # cat promotes) and its output joins the towers under autocast.
        feat = seq_trunk(f_scal, f_img, idx, f_age)
        g = policy.gru_sequence(feat, b_h0[envs], seg)
        return seq_loss(feat, g, f_scal, f_act, f_logp, f_adv, f_ret, idx,
                        ent_coef, adv_mean, adv_std, f_air, f_jblk, f_priv)

    MB = T * N // args.minibatches            # constant: the compiled shape
    if args.train_stride > 1 and (T // args.train_stride) * N < MB:
        raise SystemExit(f"--train-stride {args.train_stride} leaves fewer "
                         f"than one {MB}-sample minibatch of the {T}x{N} "
                         "rollout — lower the stride or --minibatches")
    if use_compile and RNN:
        # the two static-shaped halves are compiled the way mb_step is; the
        # warm-up runs the whole step on the zeroed buffers (one all-zero
        # segment plan: no episode cut) and drops the gradients. The
        # recurrence's cuDNN call is not compiled, so a later plan with a
        # different (L, S) never recompiles anything.
        eager_seq_trunk, eager_seq_loss = seq_trunk, seq_loss
        try:
            t_c = time.perf_counter()
            seq_trunk = torch.compile(eager_seq_trunk,
                                      mode="max-autotune-no-cudagraphs")
            seq_loss = torch.compile(eager_seq_loss,
                                     mode="max-autotune-no-cudagraphs")
            _B = N // args.minibatches
            _idx0 = (torch.arange(T, device=device)[:, None] * N
                     + torch.arange(_B, device=device)[None, :]).reshape(-1)
            mb_step_seq(b_scal.reshape(T * N, SCAL),
                        b_img.reshape((PRO + T) * N, FRAME),
                        b_act.reshape(ACT_FLAT),
                        b_logp.reshape(-1), b_val.reshape(-1),
                        b_rew.reshape(-1), _idx0, ent_t,
                        None if b_age is None else b_age.reshape(-1),
                        torch.zeros((), device=device) if D.enabled else None,
                        torch.ones((), device=device) if D.enabled else None,
                        None if b_air is None else b_air.reshape(-1),
                        None if b_jblk is None else b_jblk.reshape(-1),
                        b_priv.reshape(T * N, PRIV) if PRIV else None,
                        envs=torch.arange(_B, device=device),
                        seg=gru_segments(np.zeros((T, _B), bool), device),
                        )[0].backward()
            opt.zero_grad(set_to_none=True)
            print(f"torch.compile: --rnn sequence step (trunk + loss) "
                  f"compiled in {time.perf_counter() - t_c:.0f}s "
                  "(max-autotune-no-cudagraphs; the cuDNN recurrence stays "
                  "eager)")
        except Exception as exc:            # pragma: no cover
            print(f"torch.compile failed ({exc!r}) — eager update")
            seq_trunk, seq_loss = eager_seq_trunk, eager_seq_loss
            opt.zero_grad(set_to_none=True)
            use_compile = False
            meta["config"]["compile"] = False      # keep run.json honest
            if D.is_main:
                (out / "run.json").write_text(json.dumps(meta, indent=2),
                                              encoding="utf-8")
        if D.enabled:
            any_failed = D.all_reduce_max_scalar(0.0 if use_compile else 1.0)
            if any_failed > 0.0 and use_compile:
                print("torch.compile dropped fleet-wide (a peer rank's "
                      "inductor failed)")
                seq_trunk, seq_loss = eager_seq_trunk, eager_seq_loss
                opt.zero_grad(set_to_none=True)
                use_compile = False
                meta["config"]["compile"] = False    # keep run.json honest
                if D.is_main:
                    (out / "run.json").write_text(
                        json.dumps(meta, indent=2), encoding="utf-8")
    elif use_compile:
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
            # the warm-up traces the SAME adv-normalisation branch the
            # steady state will run (None-ness is a trace-time guard); the
            # gradients are dropped - no opt.step(), and NEVER sync_grads()
            # here (there is no matching step and the grads are discarded)
            mb_step(b_scal.reshape(T * N, SCAL),
                    b_img.reshape((PRO + T) * N, FRAME),
                    b_act.reshape(ACT_FLAT),
                    b_logp.reshape(-1), b_val.reshape(-1), b_rew.reshape(-1),
                    torch.arange(MB, device=device), ent_t,
                    None if b_age is None else b_age.reshape(-1),
                    None if b_code is None else b_code.reshape(-1),
                    None if b_dmask is None else b_dmask.reshape(T * N, H),
                    torch.zeros((), device=device) if D.enabled else None,
                    torch.ones((), device=device) if D.enabled else None,
                    None if b_air is None else b_air.reshape(-1),
                    None if b_jblk is None else b_jblk.reshape(-1),
                    b_priv.reshape(T * N, PRIV) if PRIV else None,
                    f_z=(b_z.reshape(T * N, NZ) if VIEWC else None),
                    )[0].backward()
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
            if D.is_main:
                (out / "run.json").write_text(json.dumps(meta, indent=2),
                                              encoding="utf-8")
        if D.enabled:
            # a per-rank compile fallback must be COLLECTIVE: if any rank's
            # inductor failed, every rank drops to eager so the compiled
            # region stays uniform across the fleet (plan step 14)
            any_failed = D.all_reduce_max_scalar(0.0 if use_compile else 1.0)
            if any_failed > 0.0 and use_compile:
                print("torch.compile dropped fleet-wide (a peer rank's "
                      "inductor failed)")
                mb_step = eager_mb_step
                opt.zero_grad(set_to_none=True)
                use_compile = False
                meta["config"]["compile"] = False    # keep run.json honest
                if D.is_main:
                    (out / "run.json").write_text(
                        json.dumps(meta, indent=2), encoding="utf-8")

    # ---- --bc-file: expert-iteration distillation (surfgym/bc.py) ---------
    # The planner's (state, action) rows enter the SAME optimizer step as
    # the PPO minibatch: mb_step above is untouched, a second compiled
    # function scores one planner batch, and the two losses are summed
    # before the single backward / clip / step. Without --bc-file nothing
    # in this block or in the loop runs, and the update is byte-for-byte
    # the trainer without it. Its images are rendered per minibatch from
    # the stored STATES with this slot's own lidar into b_img's dtype - the
    # exact path fill_vision + the bf16 rollout buffer take - so the row
    # the policy is asked to imitate on is [15 core | latch | depth], the
    # rollout's own layout (tests/python/test_expert_iteration.py).
    bc = None
    bc_coef_t = torch.zeros((), device=device)
    bc_coef_now = 0.0
    # --bc-value-coef: the target `z` is stored in RAW reward units, and
    # under --ret-norm the critic predicts (G - mu)/sigma, so the same pair
    # has to be applied to z before the MSE. Tensors, not Python floats, for
    # the reason ent_t is one: a scalar baked into a compiled signature is a
    # constant and the per-iteration update would retrace the region.
    bc_zmu = torch.zeros((), device=device)
    bc_zsig = torch.ones((), device=device)
    BCV = float(args.bc_value_coef or 0.0) if args.bc_file else 0.0
    BCDIST = bool(args.bc_file) and args.bc_target == "dist"
    bc_stats = None
    if args.bc_file:
        if MULTI or route is not None or STACK > 1 or H > 0:
            raise SystemExit("--bc-file: single-map, flat (no --chunk), "
                             "unstacked, fan-less race policies only - the "
                             "planner cannot clone the other per-env states")
        if BCV > 0.0 and args.race_arc:
            raise SystemExit("--bc-value-coef with --race-arc: the stored z "
                             "is the GEODESIC objective's return (plan_to_bc "
                             "refuses to build an arc BC file at all) and "
                             "the priv block's arc column would be 0 on "
                             "every BC row and non-zero on every rollout row")
        # --priv-critic: the critic reads a privileged block, so the BC
        # rows need theirs or V(s) is read through zeros on ten columns the
        # rollout never has at zero. Built ONCE, from the SAME PrivFeat
        # instance the rollout fills (privfeat.py: one implementation, three
        # callers - a second copy here is exactly the drift it warns about).
        # d is resampled on the reward field, tick comes off the stored
        # state, latch is the file's own column, arc is 0 (the guard above).
        bc_priv_fn = None
        if args.priv_critic:
            _pv = slots[0].priv
            _pf = (slots[0].reward_field if slots[0].reward_field is not None
                   else slots[0].goal_field)

            def bc_priv_fn(states, latch, _pv=_pv, _pf=_pf):
                n = len(states)
                out_pv = np.zeros((n, PRIV_DIM), np.float32)
                _pv.fill(out_pv, states["origin"], states["velocity"],
                         np.asarray(_pf.sample(states["origin"]), np.float64),
                         states["tick"], arc=None,
                         latch=(np.asarray(latch, np.float32) > 0.5))
                return out_pv

        bc = BCDataset(args.bc_file, device, n_latch=N_LATCH,
                       obs_reward=bool(args.obs_reward),
                       seed=args.seed + 31 * D.rank, priv_fn=bc_priv_fn,
                       # --view-continuous: the rows' view targets (z), read
                       # from the file's own view columns or derived from
                       # the bins the way the core would apply them
                       view_continuous=VIEWC,
                       yaw_adaptive=bool(args.yaw_adaptive),
                       pitch_rate_max_deg=float(
                           core.config.pitch_rate_max_deg))
        print(bc.describe())
        bc_lidar, bc_dtype = slots[0].lidar, b_img.dtype
        bc_steps = (float(args.bc_steps) if args.bc_steps
                    else max(1.0, float(args.steps) - float(global_step)))
        print(f"bc: coef {args.bc_coef:g} -> {args.bc_coef_final:g} over "
              f"{bc_steps:,.0f} steps, {args.bc_batch} rows per minibatch "
              f"step, summed into the PPO step; target {args.bc_target}"
              + (f", value coef {BCV:g} on {bc.value_rows:,} rows"
                 if BCV > 0.0 else ", no value term"))
        if BCDIST and not bc.has_probs:
            print("bc: --bc-target dist on a version-1 file - every target "
                  "is the one-hot of the stored action, i.e. the argmax "
                  "loss exactly. Rebuild the file with tools/plan_to_bc.py "
                  "to get a real distribution.")
        if BCV > 0.0 and not bc.has_value:
            raise SystemExit(f"--bc-value-coef {BCV:g} but {args.bc_file} is "
                             "a version-1 BC file and carries no return at "
                             "all. Rebuild it with tools/plan_to_bc.py (the "
                             "value target is on by default), or drop the "
                             "flag")
        if BCV > 0.0 and bc.value_rows == 0:
            # not fatal: an unattended expert_loop round must not die
            # because THIS round's planner window ended with every lineage
            # still alive (no terminal -> no complete return -> nothing to
            # regress onto). The term is inert and every read-out says so:
            # this line, the file's own meta, and a BLANK bc/value_mse.
            print(f"WARNING: --bc-value-coef {BCV:g} but no row of "
                  f"{args.bc_file} carries a COMPLETE return (zmask is 0 "
                  "everywhere - every kept lineage was still alive when the "
                  "search window ended). The value term is a NO-OP this "
                  "round; bc/value_mse will be blank.")

        def bc_loss_fn(scal, img, act, w, probs, z, zmask, priv,
                       vz=None, vmu=None, vsd=None):
            with amp:
                logits, value = policy.forward_split(scal, img, priv=priv)
                if VIEWC:
                    cat, mu = split_view(logits.float())
                    padded = packer.pad(cat)
                else:
                    padded = packer.pad(logits.float())
                if YCOND:
                    # the cloning loss is the NLL of the planner's action
                    # under the policy's own factorisation, so the side
                    # term has to be log p(side | the planner's yaw). Fed
                    # the unconditioned head it would fit a distribution
                    # the rollout never samples from.
                    padded = add_yaw_cond(padded, policy.yaw_side.table,
                                          act[:, H_YAW])
                if VIEWC:
                    # --view-continuous: the four categorical heads as
                    # before, on their slice; the view heads by the
                    # Gaussian NLL of the executed z (`vz`, from the row's
                    # physical view) at the FIXED reference sigma
                    # BC_VIEW_SIGMA (see its note: the live sigma made the
                    # term a 1/sigma^2 drag on mu) and, for --bc-target
                    # dist, the Gaussian CROSS-ENTROPY to the elite copies'
                    # moment-matched N(vmu, vsd) at the same sigma:
                    # E_{z~N(vmu,vsd)}[-log N(z; mu, s)] = log s + log
                    # sqrt(2 pi) + (vsd^2 + (vmu - mu)^2) / (2 s^2). A row
                    # a single line survived at has vsd = 0 and the two
                    # coincide, as the one-hot does with the argmax loss.
                    logp_c, _ent = logprob_entropy_padded(
                        padded[:, N_VIEW:], act[:, N_VIEW:])
                    _ls = float(np.log(BC_VIEW_SIGMA))
                    _var2 = 2.0 * BC_VIEW_SIGMA * BC_VIEW_SIGMA
                    nll_pt = (_ls + 0.5 * LOG2PI
                              + (vz - mu).pow(2) / _var2).sum(-1)
                    nll_mm = (_ls + 0.5 * LOG2PI
                              + (vsd.pow(2) + (vmu - mu).pow(2))
                              / _var2).sum(-1)
                    logp = logp_c - nll_pt
                else:
                    logp, _ent = logprob_entropy_padded(padded, act)
            # weighted per-row negative log-likelihood of the six factored
            # heads = the cross-entropy of each categorical against the
            # planner's index, summed over heads
            nll = -(logp * w).sum() / w.sum().clamp_min(1e-6)
            # the cross-entropy of the same six categoricals against the
            # SEARCH's distribution. The padded slots hold NEG = -1e30 and
            # the target is 0 there, and 0 * -1e30 is -0.0, so those terms
            # drop out exactly - which is why a one-hot target reproduces
            # `nll` bit for bit rather than approximately.
            lsm = F.log_softmax(padded, dim=-1)
            if VIEWC:
                ce_row = (-(probs[:, N_VIEW:] * lsm[:, N_VIEW:]).sum(-1).sum(-1)
                          + nll_mm)
            else:
                ce_row = -(probs * lsm).sum(-1).sum(-1)
            ce = (ce_row * w).sum() / w.sum().clamp_min(1e-6)
            loss = ce if BCDIST else nll
            vm = w * zmask
            vmass = vm.sum()
            v = value.float().reshape(-1)
            v_err = (v - (z - bc_zmu) / bc_zsig).pow(2)
            v_mse = (v_err * vm).sum() / vmass.clamp_min(1e-6)
            if BCV > 0.0:
                # the trainer's own value-loss FORM (0.5 * squared error, no
                # clipping - mb_step does not clip either), so the two terms
                # are in the same units and C is a plain ratio
                loss = loss + BCV * 0.5 * v_mse
            with torch.no_grad():
                if VIEWC:
                    # agreement over the four categorical heads; the view
                    # heads report the mean squared z error of mu instead
                    agree = padded[:, N_VIEW:].argmax(-1) == act[:, N_VIEW:]
                    _vm = (vz - mu).pow(2).mean(-1)
                    view_mse = (_vm * w).sum() / w.sum().clamp_min(1e-6)
                else:
                    agree = padded.argmax(-1) == act
                    view_mse = torch.zeros((), device=w.device)
                stats = torch.stack([
                    nll.detach(), agree.float().mean(),
                    agree.all(-1).float().mean(), ce.detach(),
                    v_mse.detach(), vmass.detach(), view_mse.detach()])
            return loss, stats

        bc_step = bc_loss_fn
        if use_compile:
            try:
                t_c = time.perf_counter()
                bc_step = torch.compile(bc_loss_fn,
                                        mode="max-autotune-no-cudagraphs")
                _s, _p, _a, _w, _pr, _z, _zm, _pv, _vz, _vmu, _vsd = \
                    bc.sample_all(args.bc_batch, view=True)
                bc_step(_s, bc.render(bc_lidar, _p, bc_dtype), _a, _w,
                        _pr, _z, _zm, _pv, _vz, _vmu, _vsd)[0].backward()
                opt.zero_grad(set_to_none=True)
                print(f"torch.compile: bc step compiled in "
                      f"{time.perf_counter() - t_c:.0f}s")
            except Exception as exc:            # pragma: no cover
                print(f"torch.compile (bc step) failed ({exc!r}) - eager")
                bc_step = bc_loss_fn
                opt.zero_grad(set_to_none=True)
        bc_log = None
        if D.is_main:
            bc_log = open(out / "bc_log.csv", "a", newline="",
                          encoding="utf-8")
            if bc_log.tell() == 0:
                # the last four are APPENDED, so an existing bc_log.csv from
                # a pre-flag round stays a strict prefix of this header
                bc_log.write("time/total_timesteps,bc/coef,bc/loss,bc/acc,"
                             "bc/ce_dist,bc/joint_acc,bc/value_mse,"
                             "bc/value_rows"
                             # --view-continuous APPENDS its column, so a
                             # discrete round's file is byte-identical
                             + (",bc/view_mse" if VIEWC else "") + "\n")

    # ---- DDP gradient path (docs/ddp-plan.md steps 9 + 15) ------------------
    # Bucketed flat all-reduces; the views borrow each parameter's own
    # strides so the copies are pure memcpys (correctness never depends on
    # strides — copy_ is a value copy). With --ddp-overlap each bucket's
    # collective launches from a post-accumulate-grad hook the moment its
    # last gradient lands, overlapping the wire time with the rest of
    # backward; sync_grads() then pays only the exposed remainder. The
    # hooks are registered AFTER the compile warm-up on purpose: the
    # warm-up backward must not issue collectives (a rank whose inductor
    # failed never reaches its backward, and the fleet would deadlock).
    if D.enabled:
        _inv_ws = 1.0 / D.world_size
        if args.ddp_overlap:
            # readiness-ordered buckets (plan step 15): heads+towers finish
            # backward first, the conv Linear next, the three convs last —
            # buckets 0 and 1 then overlap convolution_backward
            _lin_ids = {id(q) for m in policy.conv
                        if isinstance(m, nn.Linear) for q in m.parameters()}
            _buckets = [[], [], []]
            for pname, p in policy.named_parameters():
                if pname.startswith(("action_head", "value_head",
                                     "pi.", "vf.")):
                    _buckets[0].append(p)
                elif id(p) in _lin_ids:
                    _buckets[1].append(p)
                else:
                    _buckets[2].append(p)
        else:
            _buckets = [list(policy.parameters())]
        _b_flat, _b_views = [], []
        for b in _buckets:
            fl = torch.zeros(sum(p.numel() for p in b), device=device)
            vs, off = [], 0
            for p in b:
                vs.append(torch.as_strided(fl, p.shape, p.stride(), off))
                off += p.numel()
            _b_flat.append(fl)
            _b_views.append(vs)
        _arrived = [0] * len(_buckets)
        _handles: list = [None] * len(_buckets)

        def _launch(bi):
            # RE-READ p.grad at launch: zero_grad(set_to_none=True) rebinds
            # every grad each minibatch — a cached list would sync stale
            # storage while opt.step() uses fresh tensors: divergent nets,
            # no error, caught only by the checksum
            torch._foreach_copy_(_b_views[bi],
                                 [p.grad for p in _buckets[bi]])
            _handles[bi] = D.all_reduce_async(_b_flat[bi])

        if args.ddp_overlap:
            def _mk_hook(bi):
                def _hook(_p):
                    _arrived[bi] += 1
                    if _arrived[bi] == len(_buckets[bi]):
                        _launch(bi)
                return _hook
            for bi, b in enumerate(_buckets):
                for p in b:
                    p.register_post_accumulate_grad_hook(_mk_hook(bi))

        def sync_grads():
            ev = tm.gpu_start("allreduce")
            for bi in range(len(_buckets)):
                if _handles[bi] is None:      # no hooks, or a starved one
                    _launch(bi)
                _handles[bi].wait()           # a stream dependency, not a
                # host block: the scale and copy-back queue behind the wire
                _b_flat[bi].mul_(_inv_ws)
                torch._foreach_copy_([p.grad for p in _buckets[bi]],
                                     _b_views[bi])
                _handles[bi] = None
                _arrived[bi] = 0
            tm.gpu_end(ev)
    else:
        def sync_grads():
            return None

    # ---- cross-rank state sharing (docs/ddp-plan.md §3, steps 10-12) --------
    # Gathered rows carry (tick_in_iteration, GLOBAL env id) and are sorted
    # by that key before use, so the merged order — and therefore the
    # respawn ring bytes and the 200-episode deque — is exactly what a
    # single-GPU N_GLOBAL-env run would produce. Appending in rank order
    # would pin both to the last rank's env block (plan §1 correction 2).
    EP_DT = np.dtype([("tick", np.int32), ("env", np.int32),
                      ("ret", np.float64), ("len", np.int64)])
    HARVEST_DT = np.dtype([("tick", np.int32), ("env", np.int32),
                           ("state", STATE_DTYPE)])
    ep_out: list = []                 # (tick_in_iter, local env, ret, len)
    # the map slot rides along so ONE gather serves every map's table
    CNT_DT = np.dtype([("slot", np.int32), ("cell", np.int64),
                       ("inc", np.int32)])

    def _global_env(e):
        """Fleet-local env index on THIS rank -> the env id the identical run
        on ONE process with --envs N_GLOBAL would have given the same env.

        Only ever used as a SORT KEY, but it has to be the right key. Under
        --maps the fleet-local index is (slot, slot-local), and the two
        splits nest the other way round globally: a single-process run lays
        slot i out as one contiguous block of PER*world_size envs, of which
        this rank holds the r-th sub-block. Sorting on the naive
        `local + rank*N` would interleave MAPS instead of ranks and put the
        merged order somewhere no single-GPU run has ever been."""
        e = np.asarray(e, np.int64)
        if not MULTI:
            return (e + D.rank * N).astype(np.int32)
        i, j = np.divmod(e, PER)
        return (i * (PER * D.world_size) + D.rank * PER + j).astype(np.int32)

    def sync_counts():
        """Exchange novelty-count DELTAS as sparse (cell, inc) pairs —
        O(cells entered this window), never O(table). The champion's keyed
        table (--int-view 8 --int-speed 3) is ~32M cells, so the plan's
        dense all-reduce would move 128 MB per iteration and the dense
        CPU delta/apply another ~500 MB of memory traffic; the cells
        actually entered are a few tens of thousands. Deltas, never
        absolutes (checkpoint round-trip safety, plan §3a).

        The novelty table is PER MAP (its keys are that map's cells), so
        this is a FLEET operation and not slot 0's - which is exactly the
        class of bug this integration exists to avoid, since slot 0 syncing
        alone would look perfectly healthy in every logged number while the
        other maps' ranks silently explored past each other. Batched into
        one gather so the collective count does not grow with the maps."""
        if not D.enabled:
            return
        loc = fleet.counts_delta_sparse(CNT_DT)
        parts = D.all_gather_var_bytes(loc.tobytes())
        merged = np.frombuffer(b"".join(parts), dtype=CNT_DT)
        fleet.apply_counts_delta_sparse(merged)

    def gather_sorted(local: np.ndarray, dt) -> np.ndarray:
        """All-gather structured rows, stable-sorted by (tick, env). The
        stable sort preserves per-env snapshot order inside one tick."""
        parts = D.all_gather_var_bytes(local.tobytes())
        merged = np.frombuffer(b"".join(parts), dtype=dt)
        key = merged["tick"].astype(np.int64) * (N_GLOBAL + 1) \
            + merged["env"]
        return merged[np.argsort(key, kind="stable")]

    # ---- --tick-ms-schedule: the LIVE tick ---------------------------------
    # The ramp moves the physics tick while the run runs, so every constant
    # the TickClock block converted at startup has to be re-derived and
    # PUSHED into the object that consumes it. TICK is mutated in place
    # (TickClock.retune) rather than rebound, because printers and closures
    # built before this point hold the same object; the four per-tick
    # scalars below are locals of main() and are rebound with `nonlocal`,
    # which is what makes the GAE discount and the truncation bootstrap
    # follow the ramp without touching the hot loop.
    #
    # The re-derivation is gated on the request having moved more than
    # PATTERN_TOL_MS: the realised tick must be an INTEGER-ms pattern, and
    # tick_pattern already lands within that tolerance, so asking for a new
    # pattern more often would only re-push the same one (39 changes over a
    # 10 -> 7.63 ramp, one every ~15M steps).
    #
    # What CANNOT follow the ramp, and why: max_episode_ticks and the yaw /
    # pitch deg-per-tick ceilings live in the SurfEnvConfig that surf_create
    # COPIES into the sim, and the C API exposes exactly one mutator
    # (surf_set_msec). They stay at the launch tick - announced at startup -
    # which is also what keeps a scheduled run's first step identical to an
    # unscheduled one and keeps the action space's meaning (deg per tick)
    # fixed under the warm-resumed weights. --max-step (the per-tick
    # teleport clip) and the novelty/revisit terms are per-tick quantities
    # that --tick-ms itself does not rescale either; they are deliberately
    # left alone so a ramped arm stays comparable to the fixed-tick arms.
    tick_log = []                 # (step, requested_ms, pattern, hz)

    def _tick_retune(step):
        """Move the physics tick to the schedule's value for `step`."""
        nonlocal GAMMA_T, TIME_PEN_T, SPEED_COEF_T, STALL_EPS_T
        want = tick_sched.ms_at(step)
        if abs(want - TICK.requested_ms) <= PATTERN_TOL_MS:
            return
        TICK.retune(want)
        GAMMA_T = TICK.gamma(args.gamma)
        TIME_PEN_T = TICK.per_tick(args.time_pen)
        SPEED_COEF_T = TICK.per_tick(args.speed_coef)
        STALL_EPS_T = TICK.per_tick(args.stall_eps)
        _pat = list(TICK.pattern)
        _stall_t = TICK.secs_to_ticks(args.stall_secs)
        _margin_t = TICK.secs_to_ticks(args.respawn_margin)
        _snap_t = TICK.secs_to_ticks(0.25 if args.goals else 1.0, "round")
        _seen = set()
        for _s in slots + heldout:
            # one map can present the same core twice (a held-out slot's
            # core IS its eval core); pushing a pattern twice would only
            # re-zero the phase, but the id() set keeps the log honest
            for _c in (_s.core, getattr(_s, "eval_core", None)):
                if _c is not None and id(_c) not in _seen:
                    _seen.add(id(_c))
                    _c.set_tick_pattern(_pat)
            _rf = _s.reward_fn
            if isinstance(_rf, RaceReward):
                # tick_ms drives the finish clock (--finish-tref is in
                # SECONDS), stagnant_mask's 3 s window and pop_stats' finish
                # seconds; the rest are the startup conversions re-run
                _rf.tick_ms = TICK.ms
                _rf.time_pen = TIME_PEN_T
                _rf.stall_ticks = _stall_t
                _rf.stall_eps = STALL_EPS_T
                if args.race_ng:
                    # RaceReward folds gamma**every into _ng_g at build time
                    _rf._ng_g = float(GAMMA_T) ** float(_rf.every)
                if not getattr(_s, "heldout", False):
                    # startup sets speed_coef on TRAINING slots only
                    _rf.speed_coef = SPEED_COEF_T
            if _s.respawn is not None:
                _s.respawn.margin = _margin_t
                _s.respawn.snap_every = _snap_t
                if _s.respawn.goal_k is not None:
                    _s.respawn.goal_k = (
                        TICK.secs_to_ticks(args.goal_kmin, "round"),
                        TICK.secs_to_ticks(args.goal_kmax, "round"))
        if goalsys is not None:
            # k is in SECONDS everywhere in goalsys; it re-derives
            # respawn.goal_k and its own snap_secs from this on every
            # iterate(), so one setter carries the goal curriculum
            goalsys.set_tick_ms(TICK.ms)
        # the config dump follows the ramp, so a checkpoint saved here
        # states the tick its weights were trained at (record_ckpt.py and
        # any resume read tick_ms) and run.json ends on the last realised
        # tick rather than on FROM
        _cfg = meta["config"]
        _cfg["tick_ms"] = TICK.requested_ms
        _cfg["tick_ms_eff"] = TICK.ms
        _cfg["tick_pattern_ms"] = _pat
        _cfg["gamma_tick"] = GAMMA_T
        _cfg["ep_secs"] = TICK.ticks_to_secs(args.ep_ticks)
        if args.reward == "race":
            _cfg["time_pen_tick"] = TIME_PEN_T
            _cfg["stall_eps_tick"] = STALL_EPS_T
        _cfg["tick_schedule"] = tick_sched.to_dict()
        tick_log.append([int(step), round(TICK.requested_ms, 6), _pat,
                         round(TICK.hz, 3)])
        print(f"tick schedule @ {step:,}: requested "
              f"{TICK.requested_ms:.4f} ms -> pattern "
              f"[{','.join(str(v) for v in _pat)}] ms = {TICK.ms:.4f} ms "
              f"({TICK.hz:.2f} Hz); gamma {GAMMA_T:.8f}/tick, time_pen "
              f"{TIME_PEN_T:.6g}/tick, stall_eps {STALL_EPS_T:.4g} u/call, "
              f"stall {_stall_t} ticks, respawn margin {_margin_t} ticks, "
              f"decision {KH * TICK.ms:.1f} ms, episode cap "
              f"{TICK.ticks_to_secs(args.ep_ticks):.1f} s")

    int_sync = args.int_sync_every if D.enabled else 0
    it_no = 0
    while global_step < int(args.steps):
        it_no += 1
        tm.start_iter()
        if tick_sched is not None:
            # rank-identical by construction (global_step advances by
            # N_GLOBAL on every rank) and collective-free
            _tick_retune(global_step)
        fleet.set_step(global_step)   # authoritative (survives resume)
        t_pool = tm.now()
        for _s in slots:
            if demo is not None:
                # Salimans-Chen: the reservoir share of the pool is replaced
                # by exact demo-window states (velocities unscaled — the
                # paper resets to the demonstration state itself)
                _dpool = demo.build_pool(_s.pool,
                                         fresh_frac=1.0 - args.respawn_frac)
                _s.core.set_spawn_pool(_dpool)
                if goalsys is not None:
                    # demo rows carry no harvested goal (NaN -> the
                    # assigner draws route / air goals), but they ARE
                    # visited states along the route: the air-goal
                    # anchors
                    _nd = len(_dpool)
                    goalsys.set_pool(_dpool, np.full((_nd, 3), np.nan,
                                                     np.float32),
                                     np.zeros((_nd, 64, 3), np.float32),
                                     np.zeros(_nd, np.int32))
            elif _s.respawn is not None and _s.respawn.size >= 2000:
                # refresh the spawn pool: fresh starts + perturbed mid-run
                # states. The 2000-state floor keeps the first lucky
                # episode's snapshots from seeding 90% of the fleet
                # (degenerate, self-reinforcing rollout correlation).
                if goalsys is not None:
                    _pool, _pg, _ps, _psl = _s.respawn.build_pool(
                        _s.pool, fresh_frac=1.0 - args.respawn_frac,
                        vel_scale=tuple(args.respawn_speed),
                        pitch_jitter=(0.0 if args.fix_pitch is not None
                                      else 5.0), with_goals=True)
                    _s.core.set_spawn_pool(_pool)
                    goalsys.set_pool(_pool, _pg, _ps, _psl)
                else:
                    _s.core.set_spawn_pool(_s.respawn.build_pool(
                        _s.pool, fresh_frac=1.0 - args.respawn_frac,
                        vel_scale=tuple(args.respawn_speed),
                        pitch_jitter=(0.0 if args.fix_pitch is not None
                                      else 5.0)))
        if goalsys is not None:
            goalsys.iterate(respawn, step=global_step)
        if respawn is not None and goal_field is not None and it_no % 100 == 1:
            # reservoir depth vs the frontier: if min(d) trails eval progress
            # by a lot, the harvest margin (not the sampling) is what keeps
            # the agent from ever respawning near the wall
            for _s in slots:
                res = _s.respawn
                if res is None or not res.size:
                    continue
                fld = (_s.reward_field if _s.reward_field is not None
                       else _s.goal_field)
                rd = fld.sample(res._store[:res.size]["origin"])
                print(f"reservoir{f'[{_s.tag}]' if MULTI else ''} d: min "
                      f"{rd.min():,.0f}  p10 "
                      f"{np.percentile(rd, 10):,.0f}  median "
                      f"{np.median(rd):,.0f}  ({res.size:,} states)")
                if res.last_info:
                    ep = res.bin_ep
                    wins = res.bin_win.sum()
                    print(f"  {res.last_info}  |  outcome-tracked eps "
                          f"{ep.sum():,.0f}  wins {wins:,.1f}")
        if demo is not None and it_no % 100 == 1 and demo.last_info:
            print(f"  {demo.last_info}  |  demo-tracked eps "
                  f"{demo.ep.sum():,.0f}  wins {demo.win.sum():,.1f}")
        tm.add("pool", t_pool)
        # PPO hygiene counters: per-ITERATION deltas, zeroed here, not
        # cumulative totals. They are counted off the very masks the rollout
        # already builds, so trunc_frac and crawl_frac share one denominator
        # with each other and with the episodes the return deque saw.
        hyg_end = hyg_trunc = hyg_crawl = hyg_stall = 0
        if TAILW > 0.0:
            # --tail-weight bookkeeping is per ROLLOUT: the groups are the
            # episodes this buffer saw end, and nothing carries over
            tail_eps.clear()
            tail_seg.fill(0)
        # ---------------- rollout ----------------
        t_roll = tm.now()
        with torch.no_grad():
            if ring is not None:
                # carry the previous iteration's last PRO renders into the
                # buffer's prologue, oldest first, so the update's gather sees
                # the same history the rollout's ring did
                ring.fill_prologue(b_img)
            if RNN:
                # the state every env enters this rollout with: the update's
                # sequence re-run starts from it (truncated BPTT boundary)
                b_h0.copy_(static_h)
            for t in range(T):
                policy_step()
                t_sync = tm.now()
                b_scal[t].copy_(static_obs[:, :SCAL])
                if MASKS.on:
                    # the flags this decision was SAMPLED under, recorded
                    # BEFORE the counter moves - the update replays these
                    b_air[t].copy_(static_obs[:, OBS_ONGROUND] < 0.5)
                    if b_jblk is not None:
                        b_jblk[t].copy_(static_jcd > 0)
                if PRIV:
                    b_priv[t].copy_(static_priv)
                if ring is None:
                    b_img[t].copy_(cur)
                else:
                    # ONE frame plus its age; the stack is a gather, and the
                    # ring owns where both of them go
                    ring.record(b_img, b_age, t)
                if USE_BURST:
                    if args.ez_eps > 0.0:
                        # start bursts where none is running. Duration ~
                        # zeta(mu) via inverse transform, capped: heavy-
                        # tailed so most bursts are short but a few commit
                        # for seconds, which is the point (a Levy flight,
                        # not a jitter).
                        fresh = (ez_left == 0) & (torch.rand(N, device=device)
                                                  < args.ez_eps)
                        nf = int(fresh.sum())
                        if nf:
                            u = torch.rand(nf, device=device).clamp_(1e-6, 1.0)
                            dur = u.pow(-1.0 / (args.ez_mu - 1.0)).long()
                            ez_left[fresh] = dur.clamp_(1, args.ez_max)
                            r = torch.rand(nf, NACT, device=device)
                            ez_act[fresh] = (r * NVEC_T).long().clamp_(
                                torch.zeros_like(NVEC_T), NVEC_T - 1)
                    ez_only = ez_left > 0
                    live = ez_only
                    if args.spawn_burst > 0:
                        # Go-Explore post-return exploration: uniform random
                        # action, held with prob spawn-burst-p per decision
                        # (1901.10995 sec 2.1.4 / Nature 2004.12919)
                        sb_live = sb_left > 0
                        redraw = sb_live & (torch.rand(N, device=device)
                                            >= args.spawn_burst_p)
                        r = torch.rand(N, NACT, device=device)
                        rnd_act = (r * NVEC_T).long().clamp_(
                            torch.zeros_like(NVEC_T), NVEC_T - 1)
                        sb_act = torch.where(redraw.unsqueeze(1),
                                             rnd_act, sb_act)
                        # a spawn burst wins over a stale ez burst
                        static_act[sb_live] = sb_act[sb_live]
                        sb_left[sb_live] -= 1
                        ez_only = ez_only & ~sb_live
                        live = live | sb_live
                    if bool(ez_only.any()):
                        static_act[ez_only] = ez_act[ez_only]
                        ez_left[ez_only] -= 1
                    b_ez[t].copy_(live)
                    if MASKS.on:
                        # a burst action is drawn uniformly and never passed
                        # through the logits, so the mask has no say over it
                        # - force it onto the support so the ENGINE never
                        # sees a forbidden key. These rows are dropped from
                        # the PPO pool by b_ez, so no ratio is touched.
                        MASKS.legalize_(
                            static_act,
                            static_obs[:, OBS_ONGROUND] < 0.5
                            if MASKS.needs_air else None,
                            static_jcd > 0 if MASKS.jump_cd > 0 else None)
                if MASKS.jump_cd > 0:
                    # the counter the NEXT decision reads, off the action
                    # that is actually going to the engine (post-burst)
                    static_jcd.copy_(ActionMasks.step_cooldown(
                        static_jcd, static_act[:, H_JUMP] > 0, jcd_reload))
                b_act[t].copy_(static_act if H == 0 else static_plan)
                b_logp[t].copy_(static_logp)
                b_val[t].copy_(static_val)
                act_pin.copy_(static_act, non_blocking=True)
                if VIEWC:
                    b_z[t].copy_(static_z)
                    view_pin.copy_(static_view, non_blocking=True)
                if H > 0:
                    b_code[t].copy_(static_code)
                    plan_pin.copy_(static_plan, non_blocking=True)
                torch.cuda.synchronize() if device.type == "cuda" else None
                np.copyto(act_np32, act_pin.numpy(), casting="unsafe")
                if VIEWC:
                    np.copyto(view_np32, view_pin.numpy())
                if H > 0:
                    # the chunk's whole plan, host-side: the tick loop reads
                    # row _j//K out of it and needs no further GPU traffic
                    plan_np = plan_pin.numpy()
                    dmask_np[:] = 1.0
                if rnd is not None:
                    # novelty of the state this decision was made in; paid
                    # once per decision, after the K substeps, ended-masked
                    rnd_np = rnd.bonus(b_scal[t]).cpu().numpy()
                tm.add("sync_copy", t_sync)
                # action repeat: hold the decision for K physics ticks (100Hz
                # physics, 100/K Hz decisions). Rewards sum over the repeat;
                # GAE runs at decision granularity with gamma^K. Sub-tick
                # episode ends mark the decision boundary done; the couple of
                # post-reset sub-ticks inherit the held action (standard
                # frame-skip semantics, negligible contamination).
                hyg_stall += fleet.apply_stall_kills()   # stagnation kill,
                # next tick; the count is race/stall_frac's numerator
                r_acc = np.zeros(N, np.float32)
                ended_acc = np.zeros(N, bool)
                if rpd:
                    done_acc = np.zeros(N, bool)
                    goal_acc = np.zeros(N, bool)
                for _j in range(KH):
                    if H > 0 and _j % K == 0:
                        # --chunk: decision _j//K of this chunk. An episode
                        # end ABORTS the rest of the chunk (design doc 4.3):
                        # the core autoresets in place, so without the mask up
                        # to K*H ticks of an unrelated behavior land on a
                        # fresh spawn. Shapes are constant — only the VALUES
                        # change — and dmask records which decisions really
                        # ran, so the update's ratio covers only those.
                        t_dec = tm.now()
                        _h = _j // K
                        np.copyto(act_np32, plan_np[:, _h], casting="unsafe")
                        if ended_acc.any():
                            act_np32[ended_acc] = NEUTRAL_NP
                            dmask_np[ended_acc, _h] = 0.0
                        tm.add("sync_copy", t_dec)
                    if obs_aux is not None and _j % K == 0:
                        # --act-hist: the decision the engine is about to
                        # receive, recorded so the NEXT observation shows
                        # what this one just did. At the boundary of every
                        # DECISION, which under --chunk is once per K ticks
                        # inside the chunk (the engine sees H of them) and
                        # otherwise once per row (KH == K, so _j == 0 only).
                        obs_aux.push(act_np32)
                    t_env = tm.now()
                    if VIEWC:
                        o2, base_r, done, trunc, term_obs = fleet.step(
                            act_np32, view_np32)
                    else:
                        o2, base_r, done, trunc, term_obs = fleet.step(act_np32)
                    tm.add("env", t_env)
                    t_rew = tm.now()
                    gmask = None
                    if goalsys is not None:
                        gmask = goalsys.on_step(done, trunc, ep_len)
                    if rpd:
                        # per-decision reward: the potential shaping
                        # telescopes across the K ticks, so one evaluation at
                        # the decision boundary is exact — only the masks
                        # need per-tick accumulation (goal_hits mutates every
                        # step; done rows autoreset mid-decision)
                        r = None
                        done_acc |= done.astype(bool)
                        goal_acc |= fleet.goal_hits().astype(bool)
                        if gmask is not None:
                            goal_acc |= gmask
                    elif gmask is not None:
                        r = fleet.reward(prev_obs, o2, term_obs, base_r, done,
                                         trunc, goal=(fleet.goal_hits()
                                                      .astype(bool) | gmask))
                    else:
                        r = fleet.reward(prev_obs, o2, term_obs, base_r, done,
                                         trunc)
                    prev_obs = o2.copy()
                    ended = (done | trunc).astype(bool)
                    tm.add("reward_py", t_rew)
                    t_book = tm.now()
                    if r is not None:
                        ep_ret += r      # pure collected reward only: the trunc
                                         # bootstrap below is a GAE construct
                                         # and must not inflate the logged
                                         # return
                    ep_len += 1
                    spd_sum += o2[:, 3]      # |v_xy|/1000 of the post-step
                    # obs; on the tick an episode ENDS this is the fresh
                    # spawn's speed rather than the terminal one (autoreset),
                    # which is one sample in ~1,500 and cannot move a
                    # 300 u/s threshold
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
                            # obs slots 12..14 are (pos - map_center)/2000 and
                            # map_center is PER MAP, so the reconstruction
                            # has to use each row's own centre - a shared one
                            # would put s_T thousands of units off its map
                            pos_np = to[:, 12:15] * 2000.0 + fleet.map_centers(ti)
                            pos = torch.as_tensor(pos_np, dtype=torch.float32,
                                                  device=device)
                            yawd = torch.rad2deg(torch.atan2(ts[:, 7], ts[:, 8]))
                            # --pitch-fixed: s_T is rendered from a
                            # RECONSTRUCTED pose, and slot 9 of the terminal
                            # obs is the C side's pitch. Pin it here too so
                            # V(s_T) is evaluated on the same gaze every
                            # other frame of this run was taken at.
                            _pt = (ts[:, 9] * 90.0 if PITCH_PIN is None else
                                   torch.full_like(ts[:, 9], PITCH_PIN))
                            vis = fleet.render_rows(ti, pos, yawd,
                                                    _pt, ts[:, 5])
                            if ring is not None:
                                # s_T is where decision t+1 WOULD have looked,
                                # so its history is the ring as it stands: the
                                # terminal frame, then head, head-1, ...
                                tt = torch.as_tensor(ti, device=device)
                                a1 = ring.age[tt] + 1
                                fr = [vis]
                                for s in frame_offsets(STACK)[1:]:
                                    fr.append(ring.rows_back(
                                        torch.clamp(a1, max=s) - 1, tt))
                                vis = interleave_frames(fr)
                            # the truncation bootstrap forwards a RECONSTRUCTED
                            # terminal row, so it has to rebuild every block of
                            # it — a missing route fan here would feed V(s_T)
                            # zeros the policy never sees anywhere else
                            blocks = [ts]
                            if route is not None:
                                if getattr(route, "n_envs", None) is not None:
                                    # per-env lines (MultiLine, --goals): its
                                    # anchor needs the FULL fleet, one line per
                                    # env, so feed a full-width pose with the
                                    # truncated rows overwritten and keep [ti]
                                    # (the other rows are dummies and discarded).
                                    # Round 30 review: a subset call raised on
                                    # the first 120 s episode of any goal arm.
                                    _ti = torch.as_tensor(np.asarray(ti, np.int64),
                                                          device=device)
                                    _o = torch.zeros((route.n_envs, 3), dtype=pos.dtype,
                                                     device=device)
                                    _y = torch.zeros(route.n_envs, dtype=yawd.dtype,
                                                     device=device)
                                    _v = torch.zeros(route.n_envs, dtype=ts.dtype,
                                                     device=device)
                                    _o[_ti] = pos
                                    _y[_ti] = yawd
                                    _v[_ti] = ts[:, 3] * 1000.0
                                    blocks.append(route.features(_o, _y, _v)[_ti])
                                else:
                                    blocks.append(route.features(
                                        pos, yawd, ts[:, 3] * 1000.0))
                            if N_LATCH:
                                # the flag at s_T: what it was one reward
                                # call ago, plus the terminal state's own
                                # d. latch_flags() is no use here - the
                                # autoreset has already moved these rows
                                # on to the next episode's spawn.
                                lt = fleet.terminal_latch(ti, pos_np)
                                blocks.append(torch.as_tensor(
                                    lt.astype(np.float32), device=device
                                ).reshape(-1, 1))
                            if N_AUX:
                                # the history the TERMINAL state had: obs_aux
                                # is reset only after this loop, so what it
                                # holds now is exactly "the decisions taken
                                # up to s_T, most recent first" - the same
                                # argument the frame ring makes above. The
                                # compass is recomputed at the reconstructed
                                # terminal pose, at FULL fleet width because
                                # GoalDistField.sample is row-aligned to
                                # envs; only rows ti are kept, and latch=False
                                # keeps a terminal row from re-anchoring the
                                # live episodes' d0. Every other row is a
                                # placeholder, sampled only to hold that
                                # alignment - so the previous decision's pose
                                # serves, read from vis_np because that is
                                # the whole FLEET (sv_view is slot 0's envs
                                # alone, and --maps would slice short here).
                                p_t = np.array(vis_np[:, 0:3], np.float64)
                                y_t = np.array(vis_np[:, 3], np.float64)
                                p_t[ti] = pos_np
                                y_t[ti] = np.degrees(np.arctan2(to[:, 7],
                                                                to[:, 8]))
                                aux_t = obs_aux.features(p_t, y_t, latch=False)
                                blocks.append(torch.as_tensor(
                                    np.ascontiguousarray(aux_t[ti]),
                                    device=device))
                            blocks.append(vis)
                            full = torch.cat(blocks, dim=1)
                            pv = None
                            if PRIV:
                                # --priv-critic: V(s_T) needs the CRITIC's
                                # block at the terminal state too, and the
                                # autoreset has already moved the live core
                                # off it - so it is rebuilt from the same
                                # terminal row the depth image was rendered
                                # from. Position is pos_np (already
                                # reconstructed above); the world velocity
                                # comes back exactly out of the ego-frame
                                # columns 0..2 and the heading columns 7, 8
                                # (privfeat.velocity_from_obs); d is
                                # resampled on the row's own field and the
                                # tick is ep_ticks by definition
                                # (src/env.c truncates on tick >= that).
                                # Skipping this would feed the truncation
                                # bootstrap a NaN and poison the whole GAE
                                # backward pass.
                                pv_np = np.zeros((len(ti), PRIV), np.float32)
                                fleet.terminal_priv(
                                    pv_np, ti, pos_np,
                                    velocity_from_obs(to), args.ep_ticks)
                                pv = torch.as_tensor(pv_np, device=device)
                            if RNN:
                                # V(s_T) with the state that would have
                                # entered decision t+1: static_h holds the
                                # state LEAVING decision t for every env
                                # until the end-of-decision zeroing below,
                                # and that is exactly h at s_T
                                tv = policy(full, static_h[torch.as_tensor(
                                    ti, device=device)], priv=pv)[1]
                            else:
                                tv = policy(full, priv=pv)[1]
                            if RETN:
                                # V(s_T) is spliced into the REWARD stream,
                                # so it has to come back into reward units
                                # first - the one place a missed
                                # de-normalization would be invisible
                                tv = retn.denormalize(tv)
                            bv = GAMMA_T * tv.to("cpu").numpy()
                            if rpd:
                                r_acc[ti] += bv
                            else:
                                r[ti] += bv
                    tm.add("boot", t_boot)
                    t_book = tm.now()
                    if not rpd and ended.any():
                        # buffered, not appended to the deque yet: the
                        # (tick, env) key lets the end-of-iteration merge
                        # reproduce single-GPU append order fleet-wide
                        tick_i = t * K + _j
                        for i in np.flatnonzero(ended):
                            ep_out.append((tick_i, i, ep_ret[i], ep_len[i]))
                        if TAILW > 0.0:
                            # BEFORE ep_ret/ep_len are zeroed below. The
                            # DECISION index t is what the weight matrix
                            # indexes; goal_hits is this tick's finish
                            # flag, and tail_bin still holds the bin this
                            # episode SPAWNED in - the stash after the
                            # respawn block below moves it to the fresh
                            # spawn's.
                            _gh = fleet.goal_hits().astype(bool)
                            for i in np.flatnonzero(ended):
                                tail_eps.append((t, int(i), float(ep_ret[i]),
                                                 int(ep_len[i]),
                                                 bool(_gh[i]),
                                                 int(tail_bin[i])))
                        _e, _t, _c = episode_hygiene(
                            ended, trunc.astype(bool) & ~done.astype(bool),
                            spd_sum, ep_len, CRAWL_KU)
                        hyg_end += _e; hyg_trunc += _t; hyg_crawl += _c
                        ep_ret[ended] = 0; ep_len[ended] = 0
                        spd_sum[ended] = 0.0
                    tm.add("book", t_book)
                    t_resp = tm.now()
                    if respawn is not None:
                        # never snapshot stagnating states: the pre-END
                        # margin can't see stall onsets (kills fire 15s in)
                        stag = fleet.stagnant_mask()
                        fleet.observe_respawn(ended, stagnant=stag,
                                              success=gmask)
                        if goalsys is not None and ended.any():
                            goalsys.assign(np.flatnonzero(ended))
                        if track_bins and ended.any():
                            # attribute the ended episodes' outcomes to the
                            # distance bin they STARTED in, then stash the
                            # new episodes' start bins (ended rows of
                            # states_view are already the fresh spawns)
                            fleet.track_start_bins(ended, fleet.goal_hits(),
                                                   start_bin)
                        if demo is not None and ended.any():
                            # same stash-and-attribute, in demo-index space
                            ei = np.flatnonzero(ended)
                            goal_now = fleet.goal_hits().astype(bool)[ei]
                            known = demo_idx[ei] >= 0
                            if known.any():
                                demo.note_outcomes(demo_idx[ei][known],
                                                   goal_now[known])
                            demo_idx[ei] = demo.match(sv_view[ei]["origin"])
                    tm.add("respawn", t_resp)
                    if TAILW > 0.0 and ended.any():
                        # the ended episodes' groups, snapshotted BEFORE the
                        # fresh spawns overwrite them, then the new spawns'
                        # bins. Deliberately outside the respawn block and
                        # off the fleet's own race field: --tail-weight must
                        # not need (or perturb) a binned reservoir, so the
                        # A/B differs in the objective alone.
                        tail_bin0[ended] = tail_bin[ended]
                        fleet.stash_depth_bins(ended, tail_bin, TAIL_BINS)
                    if r is not None:
                        r_acc += r
                    ended_acc |= ended
                    # the FLEET's consumption, not this rank's: every
                    # step-gated branch (loop bound, record, ckpt, anneal)
                    # reads this and must stay rank-identical (plan step 8)
                    global_step += N_GLOBAL
                if rpd:
                    t_rew = tm.now()
                    # ended rows' shaping is zeroed inside the reward (their
                    # post-autoreset states belong to the NEW episode); the
                    # goal bonus rides on the accumulated goal mask
                    r_dec = reward_fn(prev_obs, o2, term_obs, base_r,
                                      done_acc, ended_acc & ~done_acc, core,
                                      goal=goal_acc)
                    r_acc += r_dec
                    tm.add("reward_py", t_rew)
                    t_book = tm.now()
                    ep_ret += r_dec
                    if ended_acc.any():
                        tick_i = t * K + K - 1     # decision-boundary tick
                        for i in np.flatnonzero(ended_acc):
                            ep_out.append((tick_i, i, ep_ret[i], ep_len[i]))
                        if TAILW > 0.0:
                            # goal_acc is the decision's accumulated finish
                            # mask (goal_hits mutates every sub-tick)
                            for i in np.flatnonzero(ended_acc):
                                tail_eps.append((t, int(i), float(ep_ret[i]),
                                                 int(ep_len[i]),
                                                 bool(goal_acc[i]),
                                                 int(tail_bin0[i])))
                        _e, _t, _c = episode_hygiene(
                            ended_acc, ended_acc & ~done_acc,
                            spd_sum, ep_len, CRAWL_KU)
                        hyg_end += _e; hyg_trunc += _t; hyg_crawl += _c
                        ep_ret[ended_acc] = 0; ep_len[ended_acc] = 0
                        spd_sum[ended_acc] = 0.0
                    tm.add("book", t_book)
                if rnd is not None:
                    live = ~ended_acc
                    r_acc[live] += args.rnd_coef * rnd_np[live]
                t_sync = tm.now()
                b_rew[t].copy_(torch.from_numpy(r_acc).to(device, non_blocking=True))
                b_done[t].copy_(torch.from_numpy(
                    ended_acc.astype(np.float32)).to(device, non_blocking=True))
                if RNN:
                    # decision t+1 is the first of a new episode for the
                    # ended rows: zero their state (stream-ordered after the
                    # b_done[t] copy it reads). The host mirror feeds
                    # gru_segments in the update.
                    b_done_np[t] = ended_acc
                    static_h.mul_((1.0 - b_done[t]).unsqueeze(1))
                if MASKS.jump_cd > 0:
                    # an episode end (terminal, truncation, stall kill or a
                    # respawn out of the reservoir) clears the lockout:
                    # decision t+1 is the first of a NEW episode, the same
                    # rule the --rnn state and the ez burst below apply
                    static_jcd.mul_(1.0 - b_done[t])
                if H > 0:
                    # the ACTED joint log-prob, finished now that the neutral
                    # mask is known:  log pi(code | s_chunk)
                    #               + sum_h [ran_h] * sum_heads log p(a_h | dec)
                    # b_logp[t] already holds the code term (written at the
                    # chunk start); the decisions a mid-chunk episode end
                    # replaced with NEUTRAL_ACT contribute nothing, because pi
                    # did not emit them and PPO's ratio must not cover them.
                    dmask_gpu.copy_(torch.from_numpy(dmask_np),
                                    non_blocking=True)
                    b_dmask[t].copy_(dmask_gpu)
                    b_logp[t] += (static_dlogp * dmask_gpu).sum(-1)
                if USE_BURST:
                    # episode end aborts any burst in flight (Go-Explore:
                    # "exploration is also aborted at the episode's end";
                    # review: ez bursts must not leak a frozen random action
                    # into the next episode's spawn either). b_done[t] is
                    # ended_acc already on the device; decision t+1 is the
                    # first decision of the freshly reset episodes.
                    just_reset = b_done[t] > 0
                    ez_left[just_reset] = 0
                if args.spawn_burst > 0:
                    sb_left[just_reset] = args.spawn_burst
                    # the burst's first action is drawn uniformly NOW; later
                    # decisions keep it with prob spawn-burst-p
                    r0 = torch.rand(N, NACT, device=device)
                    sb_act = torch.where(
                        just_reset.unsqueeze(1),
                        (r0 * NVEC_T).long().clamp_(
                            torch.zeros_like(NVEC_T), NVEC_T - 1), sb_act)
                obs_pin.copy_(torch.from_numpy(o2))
                static_obs[:, :N_SCALAR].copy_(obs_pin, non_blocking=True)
                if args.obs_reward:
                    # the reward just earned becomes part of the NEXT
                    # decision's observation. tanh(r/0.1) is sensitive across
                    # the ordinary shaping range (~0.05-0.2 per decision) and
                    # saturates on the finish bonus rather than swamping the
                    # feature. Written after the env obs copy so it survives
                    # into the slot the policy reads.
                    static_obs[:, REWARD_SLOT] = torch.tanh(
                        torch.from_numpy(r_acc).to(device,
                                                   non_blocking=True) / 0.1)
                    if args.race_ng or args.death_charge:
                        # ended rows carry the OLD episode's terminal charge
                        # (up to -Phi ~ -100); the row now holds the NEW
                        # episode's first obs, whose eval mirror starts at
                        # 0 - zero it here so train and eval agree
                        static_obs[:, REWARD_SLOT] *= torch.from_numpy(
                            (~ended_acc).astype(np.float32)).to(
                                device, non_blocking=True)
                tm.add("sync_copy", t_sync)
                if obs_aux is not None:
                    # every episode start - autoreset inside the rollout, a
                    # respawn out of the reservoir, a stall kill - collapses
                    # the action history to zero and re-arms the d0 anchor.
                    # AFTER the truncation bootstrap (which needs the
                    # terminal history) and BEFORE fill_vision, which is the
                    # observation the fresh episode's first decision reads.
                    obs_aux.reset(ended_acc)
                # b_done[t] is ended_acc already on the device — reuse it
                # rather than paying a second host->device copy
                fill_vision(static_obs, b_done[t] > 0 if ring is not None else None)
                # optional tighter novelty-count window (--int-sync-every);
                # t is rank-identical so the collective stays symmetric
                if int_sync > 0 and (t + 1) % int_sync == 0 and t + 1 < T:
                    sync_counts()
            tm.add("rollout_wall", t_roll)
            t_gae = tm.now()
            ev_gae = tm.gpu_start("gae_gpu")
            if RNN:
                # the state entering decision T (already zeroed where the
                # episode ended at T-1, whose bootstrap nonterm masks anyway)
                _, last_val, _ = policy(static_obs, static_h,
                                        priv=static_priv)
            else:
                # --priv-critic: static_priv is the block the last
                # fill_vision wrote, i.e. the state decision T would act on
                # - the same row static_obs holds
                _, last_val = policy(static_obs, priv=static_priv)
            if RETN:
                # the head emits NORMALIZED returns; GAE, the advantages and
                # every logged value live in reward units. De-normalize ONCE
                # here, on the whole buffer, OUTSIDE the captured CUDA graph
                # - static_val is written inside the capture and a scale/
                # shift applied there would be frozen at capture time.
                b_val.mul_(retn.std).add_(retn.mean)
                last_val = retn.denormalize(last_val)
            adv = torch.zeros_like(b_rew)
            lastgae = torch.zeros(N, device=device)
            # decision-granularity discount. Under --chunk one row is K*H
            # ticks and gamma**(K*H) is EXACT, not an SMDP approximation:
            # both occurrences below are multiplied by nonterm = 1 - b_done,
            # so a chunk cut short by an episode end never has g_eff applied
            # to it, and a chunk that runs to completion is always exactly
            # K*H ticks long (the neutral tail still burns wall-clock).
            g_eff = GAMMA_T ** KH
            for t in reversed(range(T)):
                nextval = last_val if t == T - 1 else b_val[t + 1]
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + g_eff * nextval * nonterm - b_val[t]
                lastgae = delta + g_eff * args.gae * nonterm * lastgae
                adv[t] = lastgae
            ret = adv + b_val
            # explained variance, as SUFFICIENT STATISTICS (n, sum y,
            # sum y^2, sum e, sum e^2) so the DDP reduction is EXACT - a mean
            # of per-rank ratios is not the fleet's explained variance. The
            # residual y - V is `adv` by construction (ret = adv + b_val), so
            # it costs no extra tensor. Both operands are the fp32 rollout
            # buffers the value loss reads; the SUMS are accumulated in fp64
            # because 65k squares of a ~1e2 return exhaust fp32's 7 digits
            # long before the mean does.
            _evy = ret.reshape(-1).double()
            _eve = adv.reshape(-1).double()
            ev_stat = torch.stack([
                torch.tensor(float(_evy.numel()), dtype=torch.float64,
                             device=device),
                _evy.sum(), (_evy * _evy).sum(),
                _eve.sum(), (_eve * _eve).sum()])
            # ---- act/* key-use diagnostics, no flag ----------------------
            # Six SUMS off the tensors this rollout already holds - b_scal
            # slot 4 is the on-ground flag AT the decision and b_act is the
            # action taken there - so it costs six reductions over (T, N)
            # and no core call, no host sync and no extra buffer. They ride
            # the hygiene collective below, so DDP pays nothing extra.
            #   [0] airborne decisions            (the denominator)
            #   [1] ... with the fwd/back head OFF "none" (W or S held)
            #   [2] ... with jump held
            #   [3] ... with duck held
            #   [4] consecutive decision PAIRS whose side head changed
            #   [5] such pairs that did not straddle an episode end
            #   [6] decisions with BOTH the yaw bin and the side key
            #       non-neutral whose turn direction matches the held key's
            #       wish direction (act/yaw_side_agree's numerator)
            #   [7] ... decisions with both non-neutral (its denominator)
            # Under --chunk one b_scal row covers H decisions with no
            # per-decision ground flag, so the block stays zero and the five
            # columns come out blank rather than wrong.
            if H == 0:
                _airm = b_scal[:, :, OBS_ONGROUND] < 0.5
                _f64 = dict(dtype=torch.float64)
                if T > 1:
                    _pair = b_done[:-1] < 0.5
                    _flip = ((b_act[1:, :, H_SIDE] != b_act[:-1, :, H_SIDE])
                             & _pair)
                    _fl_n, _pr_n = _flip.sum(**_f64), _pair.sum(**_f64)
                else:
                    _fl_n = _pr_n = torch.zeros((), dtype=torch.float64,
                                                device=device)
                # act/yaw_side_agree: the yaw bins are ASCENDING with index
                # NEUTRAL_YAW = 0 deg (surfgym/core.py YAW_BINS) and the
                # side key is {-400, 0, +400}[i], so a strafing decision is
                # one with both off their neutral index, and it AGREES when
                # the two sit on OPPOSITE sides of it - +side (D) is
                # `right`, which needs a clockwise, i.e. negative, yaw
                # delta. Two more f64 sums off b_act, riding the same
                # collective as the four above.
                _sb = b_act[:, :, H_SIDE]
                if VIEWC and VIEW_ABS == "world":
                    # world mode: z is a target ANGLE (cos, sin), not a
                    # turn; the direction of the applied delta is not in
                    # the buffers, so no decision counts as a strafe here
                    # and the column reads 0 (the ratio's numerator and
                    # denominator are both 0)
                    _ys = torch.zeros_like(_sb, dtype=torch.bool)
                    _yag = _ys
                elif VIEWC:
                    # the yaw is the drawn z: its sign is the turn direction
                    # (z > 0 -> K > 0 -> a positive, i.e. left, delta,
                    # exactly a bin above NEUTRAL_YAW) and |z| > 0.25
                    # (|K| > 0.2, inside the smallest bin) counts as a turn.
                    # Under --view-absolute velocity z > 0 is a view LEADING
                    # the velocity to the left (off_warp > 0) and 0.25 is
                    # ~2 deg of offset; the same rule, read as "leads left
                    # while pressing A".
                    _zy = b_z[:, :, 0]
                    _ys = (_zy.abs() > 0.25) & (_sb != NEUTRAL_SIDE)
                    _yag = ((_zy > 0.0) == (_sb < NEUTRAL_SIDE)) & _ys
                else:
                    _yb = b_act[:, :, H_YAW]
                    _ys = (_yb != NEUTRAL_YAW) & (_sb != NEUTRAL_SIDE)
                    _yag = ((_yb > NEUTRAL_YAW) == (_sb < NEUTRAL_SIDE)) & _ys
                act_stat = torch.stack([
                    _airm.sum(**_f64),
                    ((b_act[:, :, H_FWD] != A_FWD_NONE) & _airm).sum(**_f64),
                    ((b_act[:, :, H_JUMP] > 0) & _airm).sum(**_f64),
                    ((b_act[:, :, H_DUCK] > 0) & _airm).sum(**_f64),
                    _fl_n, _pr_n,
                    _yag.sum(**_f64), _ys.sum(**_f64)])
            else:
                act_stat = torch.zeros(8, dtype=torch.float64, device=device)
            tm.gpu_end(ev_gae)
            tm.add("gae", t_gae)
            if TAILW > 0.0:
                # ---- TailRL advantage reweighting (arXiv 2609.02987) -----
                # Deliberately AFTER `ret = adv + b_val` and after ev_stat:
                # the critic's TARGET and its explained-variance diagnostic
                # stay the untouched GAE ones, and only the POLICY gradient
                # is reweighted. Fitting the critic to a reweighted return
                # would move the very baseline the advantages are taken
                # against, which is a different (and unstated) algorithm.
                #
                # One weight per EPISODE, broadcast over every decision of
                # that episode that lives in this buffer - including the
                # part that carried over the buffer edge, whose outcome is
                # known now that the episode has ended. Decisions of an
                # episode STILL RUNNING at the edge keep weight 1: their
                # outcome does not exist yet, and inventing one from the
                # partial return would rank a truncated episode against
                # complete ones.
                t_tail = tm.now()
                tail_stats = None
                W = np.ones((T, N), np.float32)
                if tail_eps:
                    _tr = np.array([e[2] for e in tail_eps], np.float64)
                    _tl = np.array([e[3] for e in tail_eps], np.float64)
                    _tf = np.array([e[4] for e in tail_eps], bool)
                    _tg = np.array([e[5] for e in tail_eps], np.int64)
                    w_ep, tail_stats = tail_weights(
                        _tr, _tg, mode=args.tail_outcome, finished=_tf,
                        secs=_tl * (TICK.ms / 1000.0),
                        min_n=int(args.tail_min_n), blend=TAILW)
                    _ncov = 0
                    for _e, _wi in zip(tail_eps, w_ep):
                        _te, _ti = _e[0], _e[1]
                        W[tail_seg[_ti]:_te + 1, _ti] = _wi
                        _ncov += max(0, _te + 1 - int(tail_seg[_ti]))
                        tail_seg[_ti] = _te + 1
                    # share of the buffer that belongs to an episode that
                    # ENDED here, i.e. that carries a real TailRL weight;
                    # the rest is the in-progress edge, held at 1
                    tail_stats["cov"] = _ncov / float(T * N)
                _wt = torch.from_numpy(W).to(device, non_blocking=True)
                # mean-1 over the WHOLE buffer, fleet-wide under DDP, so the
                # effective learning rate is the control's. The collective
                # is unconditional given the flag, hence rank-symmetric.
                _wm = torch.stack([_wt.double().sum(),
                                   torch.tensor(float(T * N),
                                                dtype=torch.float64,
                                                device=device)])
                D.all_reduce_sum_(_wm)
                _wmean = float(_wm[0] / _wm[1])
                if _wmean > 0.0:
                    _wt = _wt / _wmean
                adv.mul_(_wt)
                tm.add("tail", t_tail)

        # rank skew, measured BEFORE the first end-of-iteration collective
        # (inside the share block it would be absorbed into sync_counts'
        # gather and read as share time). --timing only — a permanent
        # barrier would stop a fast rank overlapping a peer's rollout tail.
        if D.enabled and args.timing:
            t_skew = tm.now()
            D.barrier()
            tm.add("skew", t_skew)

        # ---------------- cross-rank state sharing ----------------
        # Every collective here is unconditional on every rank (the guards
        # are args-derived, hence rank-symmetric); an empty local batch
        # still participates in the size gather.
        t_share = tm.now()
        sync_counts()
        # ONE reservoir PER MAP: its states are raw map coordinates, so a
        # cannonball state respawns inside solid geometry on petrus. Each
        # slot's ring is therefore synced on its own, and the sort key is
        # SLOT-LOCAL (rank r holds sub-block r of that map's envs), which is
        # what makes the merged ring byte-identical to the single-process
        # run's. The loop is rank-symmetric because every rank holds every
        # slot - the collective count is NMAPS on every rank or none.
        for _s in slots:
            _res = _s.respawn
            if _res is None:
                continue
            h_rows, h_ticks, h_envs = _res.drain_harvest()
            if D.enabled:
                loc = np.empty(len(h_rows), HARVEST_DT)
                loc["tick"] = h_ticks
                loc["env"] = h_envs.astype(np.int32) + D.rank * _s.n
                loc["state"] = h_rows
                merged = gather_sorted(loc, HARVEST_DT)
                _res.push_many(merged["state"])
            else:
                # local order is already (tick, env, snap-order) - the
                # exact order the old per-tick _push produced
                _lg = getattr(_res, "_last_goals", None)
                if _lg is not None:
                    # --goals: the harvested goal + segment columns
                    # ride along (flush_harvest does the same)
                    _res.push_many(h_rows, goals=_lg[0], segs=_lg[1],
                                   seglen=_lg[2])
                else:
                    _res.push_many(h_rows)
        if ep_out:
            ep_loc = np.array(ep_out, dtype=EP_DT)
        else:
            ep_loc = np.empty(0, dtype=EP_DT)
        ep_out.clear()
        if D.enabled:
            ep_loc["env"] = _global_env(ep_loc["env"])   # single-GPU order
            ep_loc = gather_sorted(ep_loc, EP_DT)
        for row in ep_loc:
            ret_hist.append(float(row["ret"]))
            len_hist.append(int(row["len"]))
        tm.add("share", t_share)

        # ---------------- update ----------------
        t_upd = tm.now()
        ev_upd = tm.gpu_start("update_gpu")
        f_scal = b_scal.reshape(T * N, SCAL)
        f_priv = b_priv.reshape(T * N, PRIV) if PRIV else None
        f_img = b_img.reshape((PRO + T) * N, FRAME)
        f_age = None if b_age is None else b_age.reshape(-1)
        f_act = b_act.reshape(ACT_FLAT)
        f_code = None if b_code is None else b_code.reshape(-1)
        f_dmask = None if b_dmask is None else b_dmask.reshape(T * N, H)
        # --mask-*: the rollout's own flags, flattened like every other
        # buffer so `idx` gathers the row that produced f_logp[idx]
        f_air = None if b_air is None else b_air.reshape(-1)
        f_jblk = None if b_jblk is None else b_jblk.reshape(-1)
        f_z = b_z.reshape(T * N, NZ) if VIEWC else None
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        if RETN:
            # PopArt-lite, UPDATE-THEN-USE: fold this rollout's return
            # moments into the EMA and normalize the target with the result,
            # so the critic is fitted in exactly the frame the next rollout
            # will de-normalize it with. One 3-vector all-reduce keeps the
            # running statistic bit-identical across ranks - a per-rank sigma
            # would give every rank its own value scale, silently and for the
            # rest of the run. `ev_stat[:3]` is already (n, sum G, sum G^2).
            _rs = ev_stat[:3].clone()
            D.all_reduce_sum_(_rs)
            _rn, _rsum, _rsq = _rs.tolist()
            retn.update(_rsum / _rn, _rsq / _rn)
            f_ret = ((ret - retn.mean) / retn.std).reshape(-1)
        else:
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
        bc_last = None
        bc_stats = None
        if bc is not None:
            # --bc-coef -> --bc-coef-final, linear in steps since the resume
            _bf = min(1.0, max(0.0, (global_step - step_start) / bc_steps))
            bc_coef_now = (args.bc_coef
                           + (args.bc_coef_final - args.bc_coef) * _bf)
            bc_coef_t.fill_(bc_coef_now)
            if BCV > 0.0 and RETN:
                # the frame the critic is being FITTED in this iteration is
                # the one f_ret was normalised with below; z has to enter
                # the same one or the value term pulls against the PPO one
                bc_zmu.fill_(retn.mean)
                bc_zsig.fill_(retn.std)
        # --train-stride S: optimize on every S-th decision timestep only.
        # Adjacent 30ms samples are near-duplicates; dropping them cuts the
        # update (measured ~50% of the iteration) by ~1/S at equal game-time.
        # GAE above still runs on the full chain (advantages need every
        # step); the offset rotates per iteration so no phase of the track
        # is systematically unseen. Pool is trimmed to a multiple of mb —
        # a ragged last minibatch would recompile the compiled step.
        if args.train_stride > 1:
            t_sel = torch.arange((it_no - 1) % args.train_stride, T,
                                 args.train_stride, device=device)
            sub_pool = (t_sel[:, None] * N
                        + torch.arange(N, device=device)).reshape(-1)
        else:
            sub_pool = None
        if USE_BURST:
            # burst actions were not drawn from pi, so their log-probs are
            # wrong and PPO's ratio would be meaningless. Drop them from the
            # sample pool entirely; they still shape learning through the
            # states they reach and through the returns of the on-policy
            # steps whose GAE runs back across them.
            on_policy = (~b_ez.reshape(-1)).nonzero(as_tuple=False).squeeze(-1)
            sub_pool = (on_policy if sub_pool is None
                        else sub_pool[torch.isin(sub_pool, on_policy)])
        last_diag = None
        env_np = None
        for _ in range(args.epochs):
            if RNN:
                # --rnn: shuffle ENVS, not rows. Minibatch k is B whole
                # sequences, laid out time-major so perm[k*mb:(k+1)*mb] is
                # exactly what mb_step_seq expects and the DDP moment code
                # below reads the same slices it always did. --minibatches
                # keeps its count semantics (M groups of N/M envs).
                _B = N // args.minibatches
                env_perm = torch.randperm(N, device=device,
                                          generator=perm_gen)
                env_np = env_perm.cpu().numpy()        # one sync per epoch
                perm = (torch.arange(T, device=device)[None, :, None] * N
                        + env_perm.view(args.minibatches, 1, _B)).reshape(-1)
                n_mb = args.minibatches
                # every minibatch's segment plan up front: the plan's
                # host->device copies are synchronous, and issued here they
                # cost one stall per epoch instead of one per minibatch
                plans = [gru_segments(b_done_np[:, env_np[k * _B:(k + 1) * _B]],
                                      device) for k in range(n_mb)]
            elif sub_pool is None:
                perm = torch.randperm(T * N, device=device,
                                      generator=perm_gen)
                n_mb = (T * N) // mb
            else:
                perm = sub_pool[torch.randperm(sub_pool.numel(),
                                               device=device,
                                               generator=perm_gen)]
                n_mb = sub_pool.numel() // mb
                if D.enabled:
                    # ez-greedy leaves different on-policy sample counts
                    # per rank; every rank must run the SAME number of
                    # gradient steps or the collectives desync and hang
                    n_mb = D.all_reduce_min_scalar(n_mb)
            a_mean = a_std = None
            if D.enabled and n_mb > 0:   # n_mb is fleet-min, rank-symmetric
                # fleet-wide per-minibatch advantage moments, batched per
                # EPOCH: one (2, n_mb) f64 all-reduce, not 64 3-scalar
                # collectives each stuck behind the previous minibatch's
                # gradient all-reduce (plan §2). Never .item() these.
                ap = f_adv[perm[:n_mb * mb]].view(n_mb, mb).double()
                st_m = torch.stack([ap.sum(1), ap.pow(2).sum(1)])
                D.all_reduce_sum_(st_m)
                a_mean, a_std = adv_moments64(st_m, float(mb * D.world_size))
            for k_mb in range(n_mb):
                idx = perm[k_mb * mb:(k_mb + 1) * mb]
                ev_mb = tm.gpu_start("mb_gpu")
                if RNN:
                    _e = slice(k_mb * _B, (k_mb + 1) * _B)
                    loss, pg, vl, el, logp = mb_step_seq(
                        f_scal, f_img, f_act, f_logp, f_adv, f_ret, idx,
                        ent_t, f_age,
                        None if a_mean is None else a_mean[k_mb],
                        None if a_std is None else a_std[k_mb],
                        f_air, f_jblk, f_priv,
                        envs=env_perm[_e], seg=plans[k_mb])
                else:
                    loss, pg, vl, el, logp = mb_step(
                        f_scal, f_img, f_act, f_logp, f_adv, f_ret, idx,
                        ent_t, f_age, f_code, f_dmask,
                        None if a_mean is None else a_mean[k_mb],
                        None if a_std is None else a_std[k_mb],
                        f_air, f_jblk, f_priv, f_z=f_z)
                if bc is not None and bc_coef_now > 0.0:
                    # --bc-file: one planner batch per PPO minibatch, its
                    # loss summed in before the one backward (a zero
                    # coefficient skips the whole term)
                    _s, _p, _a, _w, _pr, _z, _zm, _pv, _vz, _vmu, _vsd = \
                        bc.sample_all(args.bc_batch, view=True)
                    _lb, _st = bc_step(_s, bc.render(bc_lidar, _p, bc_dtype),
                                       _a, _w, _pr, _z, _zm, _pv,
                                       _vz, _vmu, _vsd)
                    loss = loss + bc_coef_t * _lb
                    bc_last = _st
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sync_grads()          # MUST sit before the clip: clipping
                # local grads then averaging is a different algorithm
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                tm.gpu_end(ev_mb)     # before the float() syncs: mb_gpu vs
                # update measures how much of the update is GPU vs host gaps
                if rnd is not None:
                    rnd.train_step(f_scal[idx])   # tiny MLP, outside compile
                last_diag = (idx, logp, vl, pg, el)
        # diagnostics hoisted out of the inner loop (plan step 12c): the
        # per-minibatch float() syncs fired 256x/iteration and only the
        # last survived; one fleet-mean read reports the same numbers
        if last_diag is not None:
            idx, logp, vl, pg, el = last_diag
            with torch.no_grad():
                diag = torch.stack([(f_logp[idx] - logp).mean(),
                                    vl.detach(), pg.detach(), el.detach()])
                D.all_reduce_mean_(diag)
                kl, loss_v, loss_pi, loss_ent = diag.tolist()
        if bc is not None and bc_last is not None:
            # the last minibatch's BC diagnostics, ONE sync per iteration
            # (a stacked 6-vector, like the PPO block above).
            #   nll        the planner-action NLL (the argmax loss)
            #   head_acc   per-HEAD argmax agreement - the metric ExIt
            #              measured a 50-Elo gap ACROSS, never a verdict
            #   joint_acc  all six heads agree: the honest per-DECISION
            #              agreement, and the one 0.98^6 = 0.886 arithmetic
            #              in docs/research-litsurvey-zero.md is about
            #   ce_dist    cross-entropy to the stored search distribution
            #              (== nll wherever the target is the one-hot)
            #   value_mse  (V(s) - z)^2 over the rows with a complete return
            #   value_rows the weight mass those rows carry (0 = no term)
            bc_stats = [float(x) for x in bc_last]
            _bl, _ba, _bj, _bce, _bv, _bvm, _bvw = bc_stats
            print(f"bc: coef {bc_coef_now:.3f}  nll {_bl:.4f}  "
                  f"head-acc {_ba:.3f}  joint-acc {_bj:.3f}  "
                  f"ce {_bce:.4f}"
                  + (f"  v-mse {_bv:.4f}" if _bvm > 0.0 else "")
                  + (f"  view-mse {_bvw:.4f}" if VIEWC else ""))
            if bc_log is not None:
                bc_log.write(f"{global_step},{bc_coef_now:.5f},{_bl:.5f},"
                             f"{_ba:.5f},{_bce:.5f},{_bj:.5f},{_bv:.6f},"
                             f"{_bvm:.3f}"
                             + (f",{_bvw:.6f}" if VIEWC else "") + "\n")
                bc_log.flush()
        tm.gpu_end(ev_upd)
        tm.add("update", t_upd)
        if H > 0 and it_no % 10 == 1:
            # code collapse is THE failure mode of a discrete latent (VQ-BeT
            # names it), and it is invisible in reward: a policy that always
            # picks code 3 still trains, it just has a one-word vocabulary.
            # One bincount over the rollout's acted codes makes it a number.
            cnt = torch.bincount(b_code.reshape(-1),
                                 minlength=NCODES).float()
            p = (cnt / cnt.sum()).clamp_min(1e-12)
            bits = float(-(p * p.log2() * (cnt > 0)).sum())
            top = torch.topk(cnt, min(6, NCODES))
            print(f"codes: {int((cnt > 0).sum())}/{NCODES} used  "
                  f"entropy {bits:.2f}/{np.log2(NCODES):.2f} bits  "
                  f"perplexity {2.0 ** bits:.0f}  top "
                  + " ".join(f"{int(i)}:{c / float(cnt.sum()):.1%}"
                             for c, i in zip(top.values.tolist(),
                                             top.indices.tolist())))

        # ---------------- logging / artifacts ----------------
        fps = (global_step - step_start) / (time.perf_counter() - t_start)
        rmean = float(np.mean(ret_hist)) if ret_hist else 0.0
        lmean = float(np.mean(len_hist)) if len_hist else 0.0
        # ---- PPO hygiene read-outs ---------------------------------------
        # ONE f64 vector, so DDP pays a single collective and all three
        # fractions share one fleet-wide denominator. The .tolist() sync is
        # free here: the update's own diag read has already drained the GPU.
        hyg = torch.cat([torch.tensor(
            [float(hyg_end), float(hyg_trunc), float(hyg_stall),
             float(hyg_crawl)], dtype=torch.float64, device=device),
            ev_stat, act_stat])
        if D.enabled:
            D.all_reduce_sum_(hyg)
        _h = hyg.tolist()
        _nend = _h[0]
        trunc_frac = (_h[1] / _nend) if _nend else float("nan")
        stall_frac = (_h[2] / _nend) if _nend else float("nan")
        crawl_frac = (_h[3] / _nend) if _nend else float("nan")
        expl_var = explained_var_from_sums(_h[4], _h[5], _h[6], _h[7], _h[8])
        # act/*: fleet-wide rates, one denominator each (see act_stat above)
        _nair, _npair, _nys = _h[9], _h[14], _h[16]
        fwd_air = (_h[10] / _nair) if _nair else float("nan")
        jump_air = (_h[11] / _nair) if _nair else float("nan")
        duck_air = (_h[12] / _nair) if _nair else float("nan")
        strafe_flip = (_h[13] / _npair) if _npair else float("nan")
        yaw_side_agree = (_h[15] / _nys) if _nys else float("nan")
        race_sr = race_fin = race_int = float("nan")
        if isinstance(reward_fn, RaceReward):
            if D.enabled:
                # fleet totals over MAPS then over RANKS, then rates - a
                # per-rank success_rate has the same expectation and R x the
                # variance of the 1-GPU number it gets plotted against (plan
                # step 12a). Unconditional on every rank: this is a
                # collective.
                sv = torch.from_numpy(fleet.stats_vector()).to(device)
                D.all_reduce_sum_(sv)
                fleet.clear_stats()
                rs = RaceReward.stats_from_vector(sv.cpu().numpy(),
                                                  tick_ms=TICK.ms)
            else:
                rs = fleet.pop_stats()
            race_sr, race_fin = rs["success_rate"], rs["finish_s"]
            race_int = rs["int_per_ep"]
        t_rec = tm.now()
        # ---- evaluation, SHARDED OVER MAPS -------------------------------
        # Rank r evaluates maps r, r+W, r+2W, ... on its own eval cores and
        # writes their trajectories; the per-map result rows are then
        # all-reduced so that EVERY rank leaves this block holding the whole
        # table. That last property is the requirement: the aggregate the
        # run is judged on ("average % of the map completed", "% of maps
        # finished") has to be right on every rank, not rank 0's slice.
        #
        # This is a COLLECTIVE inside a branch the single-map DDP design
        # kept deliberately collective-free (plan §6.7: rank 0 records while
        # the fleet free-runs into its next rollout). The trade reverses
        # with the map count: rank 0 evaluating NMAPS maps serially blocks
        # every other rank at the next collective for NMAPS eval-lengths
        # anyway, so the stall is paid either way and sharding pays it
        # NMAPS-times shorter. The branch is rank-SYMMETRIC because
        # global_step and next_record are both derived from N_GLOBAL*T -
        # asserted, not assumed, because getting it wrong is a hang.
        if global_step >= next_record:
            D.assert_equal("eval_trigger", torch.tensor(
                [int(global_step), int(next_record)], dtype=torch.int64,
                device=device))
            next_record = global_step + int(args.record_every)
            # per-recording seed: a fixed seed replays the same few spawns
            # forever, and with a wide per-spawn spread (56..100 at 2.6B) a
            # single weak-tail spawn makes every eval look bad
            n_rec = args.eval_eps or (3 if args.reward == "race" else 5)
            # ONE eval per map, each on its own eval core, its own goal
            # field and its own feeds. The CSV keeps the aggregate columns
            # (mean over maps) and adds a suffixed quad per map, because a
            # mean over two maps of very different length hides exactly the
            # thing --maps exists to measure.
            # Row layout, all SUMS and COUNTS so the reduction is a plain
            # add and no NaN ever enters a collective:
            #   0 prog_u  1 finish_s  2 n_finish(geodesic)  3 pct
            #   4 n_eps   5 n_finish(BOX)  6 fwd  7 path  8 speed
            #   9 evaluated(1/0)
            ev_tab = torch.zeros((NMAPS + NHELD, EVAL_K), dtype=torch.float64,
                                 device=device)
            # --heldout-maps: rows NMAPS.. of ev_tab are the held-out maps
            # (never in an aggregate), and hc_tab carries their order-only
            # corridor MAX where a route file exists. Both are written by
            # the owning rank alone and SUM-reduced, like the training rows.
            hc_tab = torch.zeros((max(NHELD, 1),), dtype=torch.float64,
                                 device=device)
            # training slots first, then the held-out ones, so the training
            # maps' evals run in exactly the order they did without the flag
            for _i, _s in enumerate(slots + heldout):
                if _s.eval_core is None:
                    continue                  # another rank owns this map
                sfx = f"_{_s.tag}" if (MULTI or _s.heldout) else ""
                path = out / f"traj_{global_step:010d}{sfx}.jsonl"
                _ev_meta = _ev_tick = None
                if goalsys is not None:
                    _ev_meta, _ev_tick = goalsys.eval_hooks(
                        _s.eval_core, seed=global_step)
                # --eval-stall: TRAINING's stall rule, on the eval core. A
                # FRESH hook per recording (its running best is per episode
                # and per rollout), chained AFTER the goal hook so a goal
                # reached on the same tick still wins the kill. The header
                # records the conditions: an eval that stall-kills is not
                # comparable to one that does not, and the trajectory is the
                # only place a downstream honesty tool can read that.
                _ev_stall = None
                _hdr = {"eval_stall": 0}
                if EVAL_STALL and isinstance(_s.reward_fn, RaceReward) \
                        and _s.reward_fn.field is not None:
                    _hdr = {"eval_stall": 1,
                            "stall_ticks": int(_s.reward_fn.stall_ticks),
                            "stall_eps": float(_s.reward_fn.stall_eps),
                            "stall_every": int(_s.reward_fn.every)}
                    _ev_stall = make_eval_stall_hook(
                        _s.eval_core, _s.reward_fn.field,
                        _s.reward_fn.stall_ticks, _s.reward_fn.stall_eps,
                        _s.reward_fn.every)
                    _ev_tick = chain_ticks(_ev_tick, _ev_stall)
                record_rollout(_s.eval_core,
                               EVAL_GREEDY(policy, packer, device,
                                           (goalsys.eval_ball
                                            if (goalsys is not None
                                                and goalsys.eval_ball
                                                is not None)
                                            else _s.lidar),
                                           _s.eval_core, K, STACK,
                                           extra_slot=(REWARD_SLOT
                                                       if args.obs_reward
                                                       else -1),
                                           extra_fn=_s.eval_reward_feed,
                                           route=(goalsys.eval_line
                                                  if goalsys is not None
                                                  else route),
                                           latch_fn=_s.eval_latch_feed,
                                           pitch_fixed=args.pitch_fixed,
                                           aux=_s.eval_aux, masks=MASKS,
                                           priv_fn=_s.eval_priv_feed),
                               path, episodes=n_rec,
                               max_ticks=n_rec * args.ep_ticks,
                               seed=global_step & 0x7FFFFFFF,
                               on_tick=_ev_tick, episode_meta=_ev_meta,
                               header_extra=_hdr)
                st = episode_stats(path)
                _f = float(np.mean([e["fwd_max"] for e in st])) if st else 0.0
                _p = float(np.mean([e["path"] for e in st])) if st else 0.0
                _v = float(np.mean([e["speed_max"] for e in st])) if st else 0.0
                ev_tab[_i, 6] = _f
                ev_tab[_i, 7] = _p
                ev_tab[_i, 8] = _v
                ev_tab[_i, 9] = 1.0
                prog_note = ""
                p_prog = p_fin = float("nan")
                p_nfin = 0
                if _s.goal_field is not None:
                    p_prog = race_progress(path, _s.goal_field)
                    p_nfin, p_fin, fin_best = eval_finish_times(path,
                                                                _s.goal_field)
                    pct_sum, n_ep, n_box = race_coverage(
                        path, _s.goal_field, _s.goal_box)
                    ev_tab[_i, 0] = 0.0 if p_prog != p_prog else p_prog * n_rec
                    ev_tab[_i, 1] = 0.0 if not p_nfin else p_fin * p_nfin
                    ev_tab[_i, 2] = float(p_nfin)
                    ev_tab[_i, 3] = pct_sum
                    ev_tab[_i, 4] = float(n_ep)
                    ev_tab[_i, 5] = float(n_box)
                    if p_nfin:
                        prog_note = (f"  fin {p_nfin}/{n_rec} mean "
                                     f"{p_fin:.2f}s best {fin_best:.2f}s")
                    prog_note += (f"  track {p_prog:7.0f}u"
                                  f"/{_s.d0:.0f}u" if p_prog == p_prog else "")
                    if n_ep:
                        prog_note += (f"  cover {pct_sum / n_ep:5.1f}%"
                                      f"  box {n_box}/{n_ep}")
                if _s.heldout and _s.route is not None:
                    # the honest frontier on a held-out map that has a
                    # reference line (tools/eval_honesty.py --order-only 16)
                    _cm = corridor_max(path, _s.route)
                    hc_tab[_i - NMAPS] = 0.0 if _cm != _cm else _cm
                    prog_note += f"  corridor MAX {_cm:8.0f}u"
                if _ev_stall is not None and _ev_stall.state["n"]:
                    prog_note += f"  stallkill {_ev_stall.state['n']}"
                if goalsys is not None:
                    prog_note += goalsys.eval_note()
                print(f"[{global_step:>13,d}] greedy"
                      f"{f'[HELDOUT {_s.tag}]' if _s.heldout else f'[{_s.tag}]' if MULTI else ''}: fwd {_f:7.0f}u"
                      f"  path {_p:7.0f}u  peak {_v:6.0f} u/s"
                      f"{prog_note} -> {path.name}")
                if not args.eval_greedy_only:
                    spath = out / f"traj_{global_step:010d}{sfx}_stoch.jsonl"
                    _sv_stall = (None if _ev_stall is None
                                 else make_eval_stall_hook(
                                     _s.eval_core, _s.reward_fn.field,
                                     _s.reward_fn.stall_ticks,
                                     _s.reward_fn.stall_eps,
                                     _s.reward_fn.every))
                    record_rollout(_s.eval_core,
                                   EVAL_SAMPLE(policy, packer, device,
                                               _s.lidar, _s.eval_core, K,
                                               STACK, route=route,
                                               latch_fn=_s.eval_latch_feed,
                                               pitch_fixed=args.pitch_fixed,
                                               aux=_s.eval_aux, masks=MASKS,
                                               priv_fn=_s.eval_priv_feed),
                                   spath, episodes=n_rec,
                                   max_ticks=n_rec * args.ep_ticks,
                                   seed=global_step & 0x7FFFFFFF,
                                   on_tick=_sv_stall, header_extra=_hdr)
                    sst = episode_stats(spath)
                    if sst:
                        print(f"[{global_step:>13,d}] stoch : path "
                              f"{np.mean([e['path'] for e in sst]):7.0f}u"
                              f" -> {spath.name}")
            # ONE collective, fixed (NMAPS, 10) shape - every row was
            # written by exactly one rank, so a SUM is the gather. After
            # this line the table is identical on every rank, which is the
            # gate: rig one map solved and one at zero and the aggregate
            # must read the same everywhere.
            D.all_reduce_sum_(ev_tab)
            if NHELD:
                D.all_reduce_sum_(hc_tab)
            ev_all = ev_tab.cpu().numpy()
            # the TRAINING table: rows 0..NMAPS-1. Held-out rows never enter
            # an aggregate - they are the other side of the comparison.
            ev = ev_all[:NMAPS]
            AGG = eval_aggregate(ev, [s.finish_kind for s in slots], n_rec)
            if not AGG["evaluated"].all():
                raise RuntimeError(
                    f"[eval] {int((~AGG['evaluated']).sum())} of {NMAPS} "
                    "maps produced no eval row - a rank skipped its shard, "
                    "and every aggregate below would silently be a mean "
                    "over the rest")
            pct, n_ep, nbox = AGG["pct"], AGG["n_eps"], AGG["n_box"]
            for _i, _s in enumerate(slots):
                eval_per_map[_s.tag] = (
                    ev[_i, 0] / n_rec if n_rec else float("nan"),
                    ev[_i, 1] / ev[_i, 2] if ev[_i, 2] else float("nan"),
                    float(nbox[_i]),
                    float(pct[_i]))
            eval_fwd = AGG["eval_fwd"]
            eval_path = AGG["eval_path"]
            eval_speed = AGG["eval_speed"]
            eval_prog = AGG["eval_prog"]
            eval_fin = AGG["eval_fin"]
            agg_pct = AGG["map_pct"]
            agg_fin_frac = AGG["maps_finished"]
            agg_pct_trig = AGG["map_pct_trigger"]
            agg_fin_trig = AGG["maps_finished_trigger"]
            if NHELD:
                hev = ev_all[NMAPS:]
                if not (hev[:, 9] > 0).all():
                    raise RuntimeError(
                        f"[eval] {int((hev[:, 9] <= 0).sum())} of {NHELD} "
                        "held-out maps produced no eval row - a rank skipped "
                        "its shard")
                hc = hc_tab.cpu().numpy()
                for _j, _s in enumerate(heldout):
                    _r = hev[_j]
                    eval_per_map[_s.tag] = (
                        _r[0] / n_rec if n_rec else float("nan"),
                        _r[1] / _r[2] if _r[2] else float("nan"),
                        float(_r[5]),
                        float(_r[3] / _r[4]) if _r[4] else float("nan"))
                    if _s.route is not None:
                        held_corr[_s.tag] = float(hc[_j])
                if D.is_main or os.environ.get("DDP_DEBUG_STDOUT"):
                    print(f"[{global_step:>13,d}] HELD-OUT  (never trained: "
                          f"field cover % of own start distance | box "
                          f"finishes / eps | field progress"
                          + (" | corridor MAX where a route exists"
                             if held_corr else "") + ")")
                    for _j, _s in enumerate(heldout):
                        _e = eval_per_map[_s.tag]
                        _ln = (f"    {_s.finish_kind[:4]:<4} {_s.tag:<28} "
                               + (f"{_e[3]:6.2f}%" if _e[3] == _e[3]
                                  else "   n/a")
                               + f"  fin {int(_e[2]) if _e[2] == _e[2] else 0:>2}"
                               f"/{int(hev[_j, 4]):<2}  prog "
                               f"{_e[0] if _e[0] == _e[0] else 0.0:>9,.0f}u"
                               f"/{_s.d0:,.0f}u")
                        if _s.route is not None:
                            _ln += (f"  corridor MAX "
                                    f"{held_corr[_s.tag]:9,.0f}u")
                        print(_ln)
            # printed on EVERY rank under DDP_DEBUG_STDOUT: two ranks
            # printing the same table off the same reduced tensor is the
            # on-box half of the "correct from every rank" gate
            if MULTI and (D.is_main or os.environ.get("DDP_DEBUG_STDOUT")):
                # the full per-map table, split by finish kind. 43 of the
                # pool have a real trigger curtain; the rest are +use button
                # boxes ~8x smaller in face area that the simulator cannot
                # press at all (CLAUDE.md 4b), so a null on one of those is
                # much weaker evidence and the two must never be pooled into
                # a single headline without the split beside it.
                print(f"[{global_step:>13,d}] MAP TABLE  "
                      f"(cover % of own route | box finishes / eps)")
                order = np.argsort(-np.nan_to_num(pct, nan=-1.0))
                for _i in order:
                    _s = slots[int(_i)]
                    print(f"    {_s.finish_kind[:4]:<4} {_s.tag:<28} "
                          f"{pct[_i]:6.2f}%  fin {int(nbox[_i]):>2}/"
                          f"{int(n_ep[_i]):<2}  prog {ev[_i, 0] / max(n_rec, 1):>9,.0f}u"
                          f"/{_s.d0:,.0f}u")
                print(f"[{global_step:>13,d}] AGGREGATE[r{D.rank}]  "
                      f"cover {agg_pct:6.2f}%  maps finished "
                      f"{agg_fin_frac:6.2%} "
                      f"({AGG['n_maps_finished']}/{AGG['n_maps_scored']})"
                      f"  ||  trigger-only cover {agg_pct_trig:6.2f}% "
                      f"finished {agg_fin_trig:6.2%} "
                      f"({AGG['n_trigger']} maps)")
        tm.add("record", t_rec)
        t_ck = tm.now()
        if global_step >= next_ckpt:
            next_ckpt = global_step + int(args.ckpt_every)
            save_ckpt(f"{global_step:010d}")
        # ckpt_latest is for crash recovery + dashboard record buttons: a
        # ~1-min cadence loses nothing and stops paying a 24-35MB torch.save
        # every iteration. This branch is rank-DIVERGENT by construction
        # (each process's own clock) and no collective may ever be added
        # inside it — replication of the shared tables is what makes the
        # rank-0 save fleet-complete without a gather.
        if D.is_main and time.perf_counter() - last_latest_save >= 60.0:
            save_ckpt("latest")
            last_latest_save = time.perf_counter()
        tm.add("ckpt", t_ck)
        if D.is_main:
            csv_w.writerow([global_step, round(rmean, 4), round(lmean, 1),
                            round(fps),
                            round(loss_pi + args.vf * loss_v
                                  + ent_coef * loss_ent, 5),
                            round(loss_v, 5), round(loss_ent, 5), round(kl, 6),
                            round(eval_fwd, 1), round(eval_path, 1),
                            round(eval_speed, 1),
                            round(getattr(reward_fn, "weight", 0.0), 4),
                            round(race_sr, 4) if race_sr == race_sr else "",
                            round(race_fin, 2) if race_fin == race_fin else "",
                            round(eval_prog, 1) if eval_prog == eval_prog else "",
                            round(eval_fin, 2) if eval_fin == eval_fin else "",
                            round(agg_pct, 3) if agg_pct == agg_pct else "",
                            round(agg_fin_frac, 4)
                            if agg_fin_frac == agg_fin_frac else "",
                            round(agg_pct_trig, 3)
                            if agg_pct_trig == agg_pct_trig else "",
                            round(agg_fin_trig, 4)
                            if agg_fin_trig == agg_fin_trig else ""]
                           + ([v for s in slots
                               for v in (round(eval_per_map[s.tag][0], 1)
                                         if eval_per_map[s.tag][0]
                                         == eval_per_map[s.tag][0] else "",
                                         round(eval_per_map[s.tag][1], 2)
                                         if eval_per_map[s.tag][1]
                                         == eval_per_map[s.tag][1] else "",
                                         int(eval_per_map[s.tag][2])
                                         if eval_per_map[s.tag][2]
                                         == eval_per_map[s.tag][2] else "",
                                         round(eval_per_map[s.tag][3], 3)
                                         if eval_per_map[s.tag][3]
                                         == eval_per_map[s.tag][3] else "")]
                              if MULTI else [])
                           # PPO hygiene, LAST (the header migration needs
                           # the old header to stay a strict prefix)
                           + [round(expl_var, 5)
                              if expl_var == expl_var else "",
                              round(trunc_frac, 4)
                              if trunc_frac == trunc_frac else "",
                              round(stall_frac, 4)
                              if stall_frac == stall_frac else "",
                              round(crawl_frac, 4)
                              if crawl_frac == crawl_frac else "",
                              round(retn.mean, 5), round(retn.std, 6)]
                           # --heldout-maps, LAST (heldout_columns order)
                           + heldout_csv_values(heldout, eval_per_map,
                                                held_corr)
                           # --tick-ms-schedule: the realised tick
                           + [round(TICK.ms, 6)]
                           # act/* key use, after everything else
                           + [round(v, 5) if v == v else ""
                              for v in (fwd_air, strafe_flip, jump_air,
                                        duck_air, yaw_side_agree)]
                           # bc/*, LAST: blank without --bc-file, and
                           # bc/value_mse blank when no row of the file
                           # carries a complete planner return
                           + ([round(bc_stats[3], 5), round(bc_stats[1], 5),
                               round(bc_stats[2], 5),
                               (round(bc_stats[4], 6) if bc_stats[5] > 0.0
                                else "")]
                              if bc_stats is not None else ["", "", "", ""])
                           # tail/*, LAST: blank without --tail-weight
                           + (["", "", "", "", "", "", "", "", ""]
                              if tail_stats is None else
                              [round(tail_stats["w_max"], 4),
                               round(tail_stats["w_p90"], 4),
                               int(tail_stats["groups"]),
                               round(tail_stats["n_med"], 1),
                               round(tail_stats["ess"], 5),
                               round(tail_stats["cov"], 4)]
                              + [round(tail_stats["p"][_t], 4)
                                 for _t in (0.5, 0.75, 0.9)]))
            csv_f.flush()
        race_note = ""
        if isinstance(reward_fn, RaceReward) and race_sr == race_sr:
            race_note = f"  win {race_sr:6.2%}"
            if race_fin == race_fin:
                race_note += f" @{race_fin:5.1f}s"
            if reward_fn.int_coef > 0.0 and race_int == race_int:
                race_note += f"  int {race_int:5.2f}/ep"
            if respawn is not None:
                # CLAUDE.md: race/win_rate is the THIRD deceptive metric and
                # it has fired - round 19 read 18.46% off a reservoir that
                # had drifted to 1,485 u from the goal, i.e. the agent was
                # respawned next to the finish and walked in. A win rate
                # that rises while min-depth falls is measuring the harvest.
                # The two are printed together, always, on the same line.
                _rmd = fleet.reservoir_min_depth()
                race_note += (f"  res {fleet.reservoir_size():,}"
                              + (f" mind {_rmd:.3%}" if _rmd == _rmd else ""))
        if isinstance(reward_fn, RaceReward) and reward_fn.arc is not None:
            # the anti-farming read-out: arc GAINED per episode (what the
            # shaping actually paid), the deepest arc any episode REACHED,
            # and the share of ticks spent outside the corridor earning
            # nothing. A farming policy shows gain >> reach.
            g, rch = rs.get("arc_gain"), rs.get("arc_reach")
            p90, offf = rs.get("arc_p90"), rs.get("arc_off")
            if g is not None and g == g:
                race_note += (f"  arc gain {g:>8,.0f}u  reach {rch:>8,.0f}u"
                              f"  p90 {p90:>8,.0f}u  off {offf:5.1%}")
        if goalsys is not None:
            race_note += goalsys.note(global_step)
        hyg_note = (f"  ev {expl_var:+.3f}" if expl_var == expl_var
                    else "  ev   n/a")
        if tail_stats is not None:
            # the two numbers that say whether the reweighting is doing
            # anything and what it cost: how concentrated the weights are
            # (ess) and how big the biggest bet was, over how many groups
            hyg_note += (f"  tail ess {tail_stats['ess']:.3f}"
                         f" wmax {tail_stats['w_max']:.1f}"
                         f" g {tail_stats['groups']}"
                         f"x{tail_stats['n_med']:.0f}")
        if _nend:
            # trunc / stall / crawl, always together: each alone is
            # ambiguous. len 1,502 with tr 0% st 100% is every episode killed
            # for stalling; the same len with tr 100% st 0% is every episode
            # running out the clock, and CLAUDE.md records two agents reading
            # exactly this wrong off ep_len_mean alone.
            hyg_note += (f"  tr/st/cr {trunc_frac:.0%}/{stall_frac:.0%}"
                         f"/{crawl_frac:.0%}")
        if RETN:
            hyg_note += f"  ret {retn.mean:+.2f}+-{retn.std:.2f}"
        if VIEWC:
            # the two learned sigmas, in z: the one number that says whether
            # the continuous heads are sharpening or blowing up
            _ls = policy.log_std().exp().tolist()
            hyg_note += "  sig " + "/".join(f"{_v:.3f}" for _v in _ls)
        print(f"step {global_step:>13,d}  rew {rmean:8.2f}  len {lmean:6.0f}  "
              f"fps {fps:,.0f}  kl {kl:.4f}  ent {ent_coef:.4f}"
              f"{hyg_note}{race_note}")
        tm.flush(it_no)
        if D.enabled:
            # C2 production asserts (docs/ddp-plan.md §5): cheap, exact,
            # and they catch the silent bugs. Cadences are rank-identical.
            if it_no % 100 == 0:
                D.assert_equal(f"policy_params@it{it_no}", param_checksum())
            if (args.ddp_assert_every > 0
                    and it_no % args.ddp_assert_every == 0):
                chk = [global_step, len(ret_hist)]
                if isinstance(reward_fn, RaceReward) \
                        and reward_fn._counts is not None:
                    # cheap probe every assert (sum + strided sample); the
                    # full 256MB-table hash only on the slow cadence
                    chk.extend(reward_fn.counts_check())
                    if it_no % 100 == 0:
                        chk.append(h64(reward_fn._counts))
                if respawn is not None:
                    chk += [respawn.size, respawn._head,
                            h64(respawn._store[:respawn.size]),
                            h64(json.dumps(respawn.rng.bit_generator.state,
                                           sort_keys=True,
                                           default=str).encode())]
                D.assert_equal(f"ddp_state@it{it_no}", torch.tensor(
                    chk, dtype=torch.int64, device=device))

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["duration_s"] = round(time.perf_counter() - t_start, 1)
    meta["total_steps"] = global_step
    if tick_sched is not None:
        # the tick the run ENDED on (meta["config"]["tick_ms*"] carries the
        # same values - they are kept live so every checkpoint states its
        # own tick - and tick_changes is the whole ramp as it happened)
        meta["tick_ms_final"] = TICK.requested_ms
        meta["tick_ms_eff_final"] = TICK.ms
        meta["tick_pattern_ms_final"] = list(TICK.pattern)
        meta["tick_hz_final"] = round(TICK.hz, 3)
        meta["tick_schedule"] = tick_sched.to_dict()
        meta["tick_changes"] = tick_log
    if D.is_main:
        (out / "run.json").write_text(json.dumps(meta, indent=2),
                                      encoding="utf-8")
    save_ckpt("final")
    if csv_f is not None:
        csv_f.close()
    if bc is not None and bc_log is not None:
        bc_log.close()
    print(f"done: {global_step:,} steps, avg "
          f"{(global_step - step_start) / (time.perf_counter() - t_start):,.0f} steps/s")
    D.finalize()


if __name__ == "__main__":
    main()
