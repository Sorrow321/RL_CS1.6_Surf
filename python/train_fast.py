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
from surfgym.respawn import DemoCurriculum, RespawnBuffer
from surfgym.rewards import (AcroCoverageReward, BlendedReward,
                             CoverageSpeedReward, ForwardProgressReward,
                             MaxSpeedReward, PathLengthReward, RaceReward,
                             drop_spawn_pool, map_spawn_pool,
                             platform_spawn_pool, ramp_spawn_pool)
from surfgym.route import RouteLine
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
                 route_critic_only: bool = False, n_quant: int = 0):
        super().__init__()
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
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
            nn.Linear(64 * 4 * 8, emb), nn.ReLU(),
        )
        # The route block is concatenated LAST, after the conv embedding, so
        # growing it onto an existing checkpoint is a zero-pad of the first
        # Linear's TRAILING columns (widen_for_route). A resumed arm then
        # computes exactly the function the baseline computed, and starts on
        # the baseline curve instead of near it.
        feat = len(idx) + emb
        def mlp(extra=0):
            return nn.Sequential(nn.Linear(feat + extra, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = mlp(0 if self.route_critic_only else self.route_dim)
        self.vf = mlp(self.route_dim)
        self.action_head = nn.Linear(hidden, sum(NVEC))
        # ---- --quantiles: the DISTRIBUTIONAL critic ------------------------
        # Both superhuman racers use one (Sophy QR-SAC 32 quantiles, Linesight
        # IQN) and Sony's ablation says it is load-bearing: without the QR head
        # Sophy is +0.69 s on a 114 s lap, i.e. not faster than the best human.
        # The head emits N quantiles of the return distribution instead of its
        # mean; N = 0 is the scalar critic, and then this Linear is (1, hidden)
        # and every tensor here is the pre-quantile model's, byte for byte.
        self.n_quant = max(0, int(n_quant))
        self.value_head = nn.Linear(hidden, max(1, self.n_quant))
        for m in list(self.conv) + list(self.pi) + list(self.vf):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
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
        # cuDNN's tensor-core convolutions are NHWC; fed NCHW they transpose
        # in and back out around every conv, forward AND backward. On the
        # 5090 that was 34% of ALL update GPU time (three nchwToNhwc /
        # nhwcToNchw kernels, 44.8 of 131.2 ms — tools/bench_update.py
        # --profile). Holding the trunk channels_last deletes those kernels;
        # the arithmetic and the weights are unchanged, so checkpoints stay
        # interchangeable in both directions.
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, obs):
        """One fused (B, 15 + R + H*W) fp32 row — the rollout and every eval."""
        return self.forward_split(obs[:, :self.scal_dim],
                                  obs[:, self.scal_dim:])

    def forward_split(self, scal, img, quantiles: bool = False):
        """Scalars and depth as separate tensors, so the PPO update can keep
        its depth buffer in bf16 (S3) without materialising a fused fp32 row.
        `scal` is indexed with feat_idx, whose entries are all < N_SCALAR, so
        this selects exactly what the fused path selects."""
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
        f = torch.cat([scal[:, self.feat_idx], self.conv(im)], dim=1)
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
        # --chunk swaps WHICH head the trunk feeds — code logits (B, n_codes)
        # instead of flat action logits — so every existing call site
        # (`logits, value = policy(obs)`) keeps working unchanged. self.code_head
        # is None in flat mode, so this resolves to action_head at trace time.
        head = self.action_head if self.code_head is None else self.code_head
        v = self.value_head(self.vf(f_vf))
        # --quantiles: the critic's OUTPUT is a distribution, but everything
        # downstream of it - GAE, the returns, the truncation bootstrap, the
        # logged value - keeps consuming ONE scalar, and that scalar is the
        # MEAN of the quantiles (the QR-SAC Q-estimate the actor is updated
        # against). Only `quantiles=True` (the critic's own loss) ever sees
        # the distribution, so the arm changes the critic's loss and its
        # representation and nothing else. n_quant == 0 keeps the original
        # squeeze, which is the same expression the scalar model shipped.
        v = (v if quantiles else v.mean(-1)) if self.n_quant else v.squeeze(-1)
        return head(self.pi(f_pi)), v


QUANTILE_KAPPA = 1.0                 # QR-DQN's qr-dqn-1, and IQN's default


def quantile_midpoints(n: int, device=None, dtype=torch.float32):
    """QR-DQN (Dabney et al. 2017, arXiv 1710.10044) section 4, verbatim:

        tau_i     = i / N                       for i = 0 .. N
        tau_hat_i = (tau_{i-1} + tau_i) / 2     for 1 <= i <= N

    The MIDPOINTS, not the tau_i: Lemma 2 of that paper proves the midpoints
    are the projection minimising the 1-Wasserstein distance, and Eq. 10
    regresses quantile i onto tau_hat_i. (2i-1)/2N is the same numbers.
    """
    i = torch.arange(1, n + 1, device=device, dtype=dtype)
    return (2.0 * i - 1.0) / (2.0 * n)


def quantile_huber_loss(quantiles, target, taus, kappa: float = QUANTILE_KAPPA):
    """The quantile Huber loss, verbatim from the papers.

    QR-DQN (1710.10044) Eq. 9-10 and IQN (1806.06923) Eq. 3:

        L_kappa(u)       = 0.5 u^2                  if |u| <= kappa
                         = kappa (|u| - 0.5 kappa)  otherwise
        rho^kappa_tau(u) = |tau - 1{u < 0}| * L_kappa(u) / kappa

    (QR-DQN prints rho without the 1/kappa; IQN's Eq. 3 divides. Both ran
    kappa = 1, where the two are the same expression. We divide, so kappa
    stays a pure Huber knob and the loss scale does not move with it.)

    Aggregation, also verbatim (QR-DQN Alg. 1, IQN Eq. 4):

        sum_{i=1}^{N} E_j [ rho^kappa_{tau_hat_i}( T theta_j - theta_i ) ]

    SUM over the N current quantiles i, MEAN over the target atoms j. Our
    target is PPO's single GAE return per sample, so N' = 1 and the mean over
    j is a mean over one atom.

    quantiles: (B, N) current quantile estimates theta_i.
    target:    (B,) or (B, M) target atoms (no gradient flows through them).
    taus:      (N,) the tau_hat_i of quantile_midpoints.
    Returns (B,) - one loss per sample, NOT yet reduced over the batch.
    """
    tgt = (target.unsqueeze(-1) if target.dim() == 1 else target).detach()
    # u_ij = (target atom j) - (quantile i): (B, N, M)
    u = tgt.unsqueeze(1) - quantiles.unsqueeze(2)
    au = u.abs()
    huber = torch.where(au <= kappa, 0.5 * u * u, kappa * (au - 0.5 * kappa))
    # the asymmetry that makes row i converge to the tau_i-quantile:
    # over-estimates (u < 0) are charged (1 - tau), under-estimates tau
    w = (taus.view(1, -1, 1) - (u < 0).to(u.dtype)).abs()
    return (w * huber / kappa).mean(dim=2).sum(dim=1)


def quantile_value_loss(quantiles, target, taus, kappa: float = QUANTILE_KAPPA):
    """quantile_huber_loss, batch-meaned and put on the SCALAR critic's scale.

    The paper's aggregation sums over i, so its magnitude grows with N: with
    a single target atom and |u| <= kappa it is exactly (N/2) * L_kappa(u),
    i.e. N times the scalar critic's 0.5 u^2 (sum_i |tau_hat_i - 1{u<0}| is
    N/2 either way round). Left alone that multiplies the critic's gradient -
    into a conv trunk the ACTOR SHARES - by 32, and the arm would be
    measuring a 32x value-loss coefficient rather than a distributional
    critic. Multiplying by 2/N is exactly the reparameterised --vf that
    undoes it: at N = 1, kappa >= |u| this reduces to the baseline's
    0.5 (v - ret)^2 identically, so the arm starts on the baseline curve and
    the only surviving differences are the loss SHAPE and the head's width.
    """
    n = quantiles.shape[-1]
    return quantile_huber_loss(quantiles, target, taus, kappa).mean() * (2.0 / n)


def quantilize_value_head(ck, policy):
    """Replicate a scalar checkpoint's value row across N quantile rows.

    The same job widen_for_route does in the COLUMN direction, done in the
    ROW direction. The base checkpoint's value head is (1, hidden): one row,
    the trained scalar value. Copying that row into all N quantile rows makes
    the MEAN of the quantiles exactly the old scalar value for every input,
    and the mean is what GAE, the returns and the bootstrap consume - so the
    resumed policy computes the same values, advantages and returns on its
    first forward. The arm starts ON the baseline curve (docs: xCTL 24,307 at
    step 0) and every later divergence is the treatment. The rows separate
    immediately anyway: their tau_hat_i differ, so the quantile loss pulls
    each one somewhere else from the very first update.

    Adam's exp_avg/exp_avg_sq are replicated the same way. widen_for_route
    ZEROES the moments it adds because those columns are genuinely new and
    have no history; these rows are copies of a row that HAS one, and zero
    moments would hand every row a full lr*sign(g) step out of the gate.

    Returns the number of tensors expanded; 0 = nothing to do.
    """
    n_q = getattr(policy, "n_quant", 0)
    if not n_q:
        return 0
    sd = ck.get("policy") or {}
    names = ("value_head.weight", "value_head.bias")
    if any(n not in sd for n in names):
        return 0
    if sd["value_head.weight"].shape[0] == n_q:
        return 0                      # already a quantile head (a QR resume)
    if sd["value_head.weight"].shape[0] != 1:
        raise SystemExit(
            f"--quantiles cannot warm-start this checkpoint: its value head "
            f"is {tuple(sd['value_head.weight'].shape)}, neither the scalar "
            f"(1, hidden) head nor this run's {n_q} rows")
    n = 0
    for name in names:
        t = sd[name]
        sd[name] = t.repeat(n_q, *([1] * (t.dim() - 1))).contiguous()
        n += 1
    # optimizer moments are keyed by parameter INDEX in policy.parameters()
    # order (the order Adam was constructed with), and --quantiles adds no
    # parameters, so those indices are still the checkpoint's own
    params = list(policy.parameters())
    want = {id(policy.value_head.weight), id(policy.value_head.bias)}
    idx = {i for i, q in enumerate(params) if id(q) in want}
    ost = ((ck.get("optimizer") or {}).get("state")) or {}
    for i, st in ost.items():
        try:
            if int(i) not in idx:
                continue
        except (ValueError, TypeError):
            continue
        for k in ("exp_avg", "exp_avg_sq"):
            t = st.get(k)
            if t is None or t.shape[0] != 1:
                continue
            st[k] = t.repeat(n_q, *([1] * (t.dim() - 1))).contiguous()
            n += 1
    return n


def widen_for_route(ck, policy):
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
                f"--route cannot warm-start this checkpoint: {what} is "
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


def check_vision_exclusive(surf_mask, pinhole, frame_stack) -> None:
    """One vision experiment at a time.

    --surf-mask widens the image to 2 channels, --frame-stack to K, and
    --pinhole changes what every pixel means. Each is a screen of its own;
    combining them before either has won confounds the read and needs
    kernels/gathers nobody has written. Refuse loudly rather than train a
    week on an arm whose result cannot be attributed."""
    on = [n for n, v in (("--surf-mask", surf_mask), ("--pinhole", pinhole),
                         ("--frame-stack", (frame_stack or 0) > 1)) if v]
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
                 extra_slot: int = -1, extra_fn=None, route=None):
        self.policy, self.packer, self.device = policy, packer, device
        self.lidar, self.core = lidar, core
        # --route: an eval that skipped the lookahead fan would feed the
        # policy a row of the right WIDTH only by accident and of the wrong
        # CONTENT always — the same class of bug the extra_slot note below
        # describes, and it lands on race/eval_progress, which is the number
        # every arm is judged by
        self.route = route
        # --obs-reward writes a side-channel value into a scalar slot during
        # TRAINING. The core does not produce it, so without this hook an
        # eval feeds whatever the core has in that slot (slot 12 is absolute
        # position / 2000, magnitude up to ~10) to a policy trained on
        # tanh(reward) in [-1, 1] - a badly out-of-distribution feature that
        # wrecks the eval while training looks fine.
        self.extra_slot, self.extra_fn = int(extra_slot), extra_fn
        self._k = max(1, int(act_every))
        self._tick = 0
        self._held = None
        self._stack = max(1, int(stack))
        self._ring = self._prev_tick = None

    def act(self, obs):
        if self._held is None or self._tick % self._k == 0:
            self._held = self._decide(obs)
        self._tick += 1
        return self._held

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
            depth = self.lidar.render(o, yw, pt, dk).reshape(t.shape[0], -1)
            if self._stack > 1:
                depth = self._push_frame(depth, sv["tick"])
            t = torch.cat([t, depth], dim=1)
        return t

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
        self._H = int(self.policy.decoder.shape[1])
        self._plan = None
        self._chunk_tick = None

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
        self._tick += 1
        return np.ascontiguousarray(self._plan[:, h])


class GreedyChunkPolicy(_ChunkPolicyBase):
    @torch.inference_mode()
    def _decide_chunk(self, obs):
        logits, _ = self.policy(self._obs(obs))
        dec = self.policy.decoder[logits.argmax(-1)]        # (B, H, sum(NVEC))
        plan = self.packer.pad_seq(dec.float()).argmax(-1)  # (B, H, 6)
        return plan.to("cpu").numpy().astype(np.int32)


class SampledChunkPolicy(_ChunkPolicyBase):
    @torch.inference_mode()
    def _decide_chunk(self, obs):
        logits, _ = self.policy(self._obs(obs))
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


def eval_finish_times(traj_path: Path, field):
    """(n_finished, mean_s, best_s) over a recording's episodes. Finished =
    the episode's LAST frame sits at the goal (geodesic distance <= 150u);
    the trailer's end label is cosmetic (record.py infers it from base
    rewards, which lack the race bonus). This is the scoreboard clock:
    start-line greedy eval seconds — training's finish_s is from-SPAWN time
    and mostly measures respawn-curriculum episodes."""
    fins, rows = [], []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict) and "map" in row:
                rows = []
            elif isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict) and "end" in row and rows:
                d = float(field.sample(
                    np.asarray(rows[-1][1:4], np.float64)[None])[0])
                if d <= 150.0:
                    fins.append(int(row.get("ticks", len(rows))) / 100.0)
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
    ap.add_argument("--train-stride", type=int, default=None,  # 1; ckpt restores
                    help="optimize on every S-th decision timestep only "
                         "(GAE still runs on the full chain; offset rotates "
                         "per iteration). Adjacent 30ms samples are near-"
                         "duplicates: stride 3 cuts the update phase "
                         "(~50% of the iteration) to ~1/3 at equal game-time")
    ap.add_argument("--lr", type=float, default=None)      # 3e-4; ckpt restores
    ap.add_argument("--gamma", type=float, default=None)   # 0.995; ckpt restores
    ap.add_argument("--gae", type=float, default=None)     # 0.95; ckpt restores
    ap.add_argument("--clip", type=float, default=None)    # 0.2; ckpt restores
    ap.add_argument("--ent", type=float, default=None)     # 0.005; ckpt restores
    ap.add_argument("--ent-final", type=float, default=None)
    ap.add_argument("--vf", type=float, default=None)      # 0.5; ckpt restores
    ap.add_argument("--quantiles", type=int, default=None,  # 0 = off; ckpt restores
                    help="DISTRIBUTIONAL critic: the value head emits N "
                         "quantiles of the return distribution instead of its "
                         "mean, trained with QR-DQN's quantile Huber loss "
                         "(arXiv 1710.10044 Eq. 9-10; kappa via "
                         "--quantile-kappa). GAE, the returns and the "
                         "truncation bootstrap keep consuming ONE scalar: the "
                         "MEAN of the quantiles, which is what QR-SAC's actor "
                         "is updated against. Sophy uses 32, Vasco's "
                         "vision-GT agent 200, Linesight IQN. A scalar-critic "
                         "checkpoint warm-starts function-identically (the "
                         "trained row is replicated into all N rows, so the "
                         "mean is the old value exactly). 0 = the scalar "
                         "critic, byte for byte. ckpt restores")
    ap.add_argument("--quantile-kappa", type=float, default=None,  # 1.0
                    help="--quantiles: the Huber threshold kappa. QR-DQN's "
                         "qr-dqn-1 and IQN both use 1. ckpt restores")
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
    # depth alone cannot tell floor from ramp from wall — they are the same
    # pixel. The mask is the hit surface's |n_z| (1 flat, 0 vertical, the
    # rideable band ~0.3-0.7; physics walks at >= 0.7, src/pm.c), baked per
    # voxel from the map mesh (surfgym.surfmask). Doubles the image slice
    # and the conv's input channels, so a ckpt cannot switch it mid-run.
    ap.add_argument("--surf-mask", type=int, default=None,
                    choices=(0, 1),                # 0; ckpt restores
                    help="add the surfable-surface mask as a second lidar "
                         "channel (needs viewer/assets/<map>.mesh.json)")
    # the shipped camera is equiangular (write_lidar's convention): a fixed
    # angle per pixel, so straight world edges bow across the image and the
    # conv sees a ramp's slope change with where it sits in frame. --pinhole
    # is the rectilinear alternative — same fov, same centre ray, uniform
    # spacing on the tangent plane. Pixel values change, so this is a fresh
    # run, not a warm start (the ckpt's setting is restored on resume).
    ap.add_argument("--pinhole", type=int, default=None,
                    choices=(0, 1),                # 0; ckpt restores
                    help="rectilinear camera instead of the equiangular one")
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
    ap.add_argument("--respawn-binned", type=int, default=None,
                    choices=(0, 1),                # S1; 0; ckpt restores
                    help="sample respawns uniformly over goal-distance bins "
                         "instead of uniformly over states (uniform-over-"
                         "states mirrors visitation: the mastered early "
                         "track is over-trained, the frontier starved)")
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
        if args.finish_k is None and ck_cfg.get("finish_k") is not None:
            args.finish_k = float(ck_cfg["finish_k"])
            restored.append(f"finish_k={args.finish_k:g}")
        if args.finish_tref is None and ck_cfg.get("finish_tref") is not None:
            args.finish_tref = float(ck_cfg["finish_tref"])
            restored.append(f"finish_tref={args.finish_tref:g}")
        if not flag_given("--obs-reward") and ck_cfg.get("obs_reward"):
            args.obs_reward = True
            restored.append("obs_reward")
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
        if args.respawn_binned is None and ck_cfg.get("respawn_binned") is not None:
            args.respawn_binned = int(ck_cfg["respawn_binned"])
            restored.append(f"respawn_binned={args.respawn_binned}")
        # explore-arm flags: without these a resumed arm silently reverts to
        # the control while its new run.json honestly claims uniform/off
        if args.race_shaping is None and ck_cfg.get("race_shaping") is not None:
            args.race_shaping = float(ck_cfg["race_shaping"])
            restored.append(f"race_shaping={args.race_shaping:g}")
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
        # --quantiles changes the value head's SHAPE: a bare resume of a
        # quantile checkpoint that silently dropped it would try to load N
        # rows into a 1-row head and die three screens later
        if args.quantiles is None and ck_cfg.get("quantiles") is not None:
            args.quantiles = int(ck_cfg["quantiles"])
            restored.append(f"quantiles={args.quantiles}")
        if (args.quantile_kappa is None
                and ck_cfg.get("quantile_kappa") is not None):
            args.quantile_kappa = float(ck_cfg["quantile_kappa"])
            restored.append(f"quantile_kappa={args.quantile_kappa:g}")
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
        # PPO knobs: without these a resumed arm silently reverts to the
        # argparse defaults while run.json claims otherwise
        for k in ("lr", "gamma", "gae", "clip", "ent", "vf", "ent_final"):
            if getattr(args, k) is None and ck_cfg.get(k) is not None:
                setattr(args, k, float(ck_cfg[k]))
                restored.append(f"{k}={getattr(args, k):g}")
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
    if args.ep_ticks is None:
        # race: "play until you finish" — the stagnation kill does the real
        # episode control, the 2-minute cap is just a backstop
        args.ep_ticks = 12000 if args.reward == "race" else 700
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
    if args.frame_stack is None:
        args.frame_stack = 0
    check_vision_exclusive(args.surf_mask, args.pinhole, args.frame_stack)
    if args.emb is None:
        args.emb = 512
    if args.hidden is None:
        args.hidden = 448
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

    # ---- --quantiles: distributional critic -------------------------------
    # NQ = 0 is the scalar critic and EVERY quantile branch below is dead.
    if args.quantile_kappa is None:
        args.quantile_kappa = QUANTILE_KAPPA
    if args.quantiles is not None and args.quantiles < 0:
        raise SystemExit(f"--quantiles {args.quantiles}: N must be >= 0 "
                         "(0 = off, Sophy uses 32)")
    if args.quantile_kappa <= 0.0:
        raise SystemExit("--quantile-kappa must be > 0")
    NQ = max(0, int(args.quantiles or 0))
    if NQ:
        print(f"distributional critic: {NQ} quantiles, quantile Huber loss "
              f"(QR-DQN 1710.10044) kappa {args.quantile_kappa:g}; GAE and "
              "the bootstrap consume the quantile MEAN")

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
                         yaw_adaptive=1 if args.yaw_adaptive else 0,
                         sv_maxvelocity=args.maxvel,
                         lidar_w=0, lidar_h=0, pitch_rate_max_deg=pitch_rate)
    core = SurfCore(args.map, cfg)

    # race objective: labeled finish zone + geodesic distance-to-finish field.
    # goal_field = the STANDARD field (eval progress, comparable across arms);
    # reward_field = what the shaping actually uses (== goal_field unless
    # --race-kill-aware masks fail volumes into the goal graph)
    goal_field = None
    reward_field = None
    goal_box = None
    race_d0 = None
    rf_d0 = None
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
            if args.race_kill_aware:
                print("--race-kill-aware needs the geodesic field; "
                      "ignored under --race-dist euclid")
        else:
            cell = args.lidar_cell or pick_cell(core)
            goal_field = build_goal_field(core, goal_box, cell=cell)
            if args.race_kill_aware:
                reward_field = build_goal_field(core, goal_box, cell=cell,
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
        # spawn yaw deliberately from the STANDARD field even under S2, so
        # spawn conditions stay identical across arms
        plat_pool = map_spawn_pool(core, yaw=goal_field.descent_yaw(raw["origin"]))
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
            if reward_field is not None and reward_field is not goal_field:
                # under --race-kill-aware a spawn inside a masked volume
                # earns no shaping until it leaves; filter on BOTH fields
                keep &= reward_field.reachable(dp["origin"])
            print(f"race: drop pool {len(dp)} -> {int(keep.sum())} on-track")
            dp = dp[keep]
        if args.spawn == "ramp":
            pool = dp
        else:
            # --spawn mixed concatenates the two pools, and the env resets by
            # UNIFORM pool draw, so entry counts are the probabilities: a
            # ~18-entry map-spawn pool next to a multi-thousand drop pool means
            # the agent essentially never starts at the start line. Replicate
            # the start entries to hit the requested drop fraction.
            f = float(np.clip(args.drop_frac, 0.01, 0.99))
            reps = max(1, int(round(len(dp) * (1.0 - f) / (f * max(len(plat_pool), 1)))))
            pool = np.concatenate([np.concatenate([plat_pool] * reps), dp])
            print(f"race: start entries x{reps} -> drop fraction "
                  f"{len(dp) / len(pool):.2f} (requested {f:.2f})")
    if not args.keep_teleports:
        core.set_teleport_fail(True)
    core.set_spawn_pool(pool)
    print(f"pool({args.spawn}) {len(pool)} | envs {N} | {device} | "
          f"omp={os.environ['OMP_NUM_THREADS']} | "
          f"graphs={use_graphs} bf16={use_bf16}"
          + (f" | pitch fixed {args.fix_pitch:g}" if args.fix_pitch is not None else ""))
    respawn = None
    if args.respawn_frac > 0.0:
        if args.respawn_mode != "uniform" and reward_field is None:
            raise SystemExit(f"--respawn-mode {args.respawn_mode} needs the "
                             "race goal field (--reward race)")
        binned = ((bool(args.respawn_binned) or args.respawn_mode != "uniform")
                  and reward_field is not None)
        bin_field, bin_d0 = reward_field, rf_d0
        if binned and args.respawn_mode != "uniform" and args.respawn_killsafe:
            bin_field = build_goal_field(core, goal_box, cell=cell,
                                         mask_kill=True)
            if not bin_field.reachable(raw["origin"]).all():
                raise SystemExit(
                    "kill-masked binning field disconnects the start from "
                    "the finish — bad route model, refusing to run")
            bin_d0 = float(np.mean(bin_field.sample(raw["origin"])))
            print(f"respawn killsafe: binning on the kill-masked field "
                  f"(start geodesic {bin_d0:.0f}u vs standard "
                  f"{race_d0:.0f}u); fail-floor states unsampleable")
        respawn = RespawnBuffer(N, reservoir=args.respawn_reservoir,
                                margin_ticks=int(args.respawn_margin * 100.0),
                                map_id=Path(args.map).stem,
                                dist_fn=bin_field.sample if binned else None,
                                dist_max=bin_d0 if binned else None,
                                dist_valid_max=(getattr(bin_field,
                                                        "_valid_max", None)
                                                if binned else None),
                                bins=args.respawn_bins,
                                mode=args.respawn_mode)
        print(f"respawn: {args.respawn_frac:.0%} of episodes from mid-run "
              f"snapshots, harvested >= {args.respawn_margin:g}s before "
              f"episode end"
              + (f", {args.respawn_mode} over {respawn.bins} distance bins"
                 if binned else ""))
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
    # pool, so eval/* metrics and recordings stay comparable across runs
    eval_core = SurfCore(args.map, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=args.ep_ticks, water_fail=1,
        yaw_adaptive=1 if args.yaw_adaptive else 0,
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
                     device=device, surf_mask=bool(args.surf_mask),
                     pinhole=bool(args.pinhole))
    mn_b, mx_b = core.map_bounds()
    map_center = ((mn_b + mx_b) / 2.0).astype(np.float32)
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
    N_ROUTE = route.n_features if route is not None else 0
    SCAL = N_SCALAR + N_ROUTE                 # the whole scalar half of a row
    obs_dim = core.obs_dim + N_ROUTE + FRAME * STACK
    REWARD_SLOT = 12          # an absolute-position channel, hidden at gps=False

    def _make_eval_reward_feed(field, scale, time_pen, k):
        """Mirror the training --obs-reward signal for evaluation rollouts.

        The eval core produces no reward, so this recomputes the same
        potential-shaping term from the goal field: the per-decision
        geodesic progress minus the time cost, squashed identically. Keeps
        its own previous-distance state and re-anchors on a jump (episode
        reset or teleport) so a relocation is not cashed as progress.
        """
        st = {"d": None}

        def feed(core):
            d = field.sample(core.states_view["origin"]).astype(np.float64)
            prev = st["d"]
            st["d"] = d
            if prev is None or len(prev) != len(d):
                return np.zeros(len(d), np.float32)
            delta = np.clip(prev - d, -100.0 * k, 100.0 * k)
            r = delta * scale - time_pen * k
            return np.tanh(r / 0.1).astype(np.float32)

        return feed
    policy = Policy(obs_dim, args.lidar_w, args.lidar_h,
                    extra_feat=(REWARD_SLOT,) if args.obs_reward else (),
                    emb=args.emb, hidden=args.hidden, gps=args.gps,
                    in_ch=img_ch, n_codes=NCODES, chunk=H,
                    route_dim=N_ROUTE,
                    route_critic_only=bool(args.route_critic_only),
                    n_quant=NQ).to(device)
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
        # --race-shaping scales the whole potential (0 = sparse: bonus +
        # penalties + intrinsic only); folded into scale so the obs-reward
        # eval feed and every downstream term inherit it consistently
        reward_fn = RaceReward(reward_field,
                               scale=100.0 / rf_d0 * args.race_shaping,
                               time_pen=args.time_pen,
                               success_bonus=args.success_bonus,
                               stall_ticks=int(args.stall_secs * 100.0),
                               int_coef=args.int_coef,
                               int_view=args.int_view,
                               int_speed=args.int_speed,
                               speed_equiv=args.speed_equiv,
                               fail_pen=args.fail_pen,
                               finish_k=args.finish_k,
                               finish_tref=args.finish_tref,
                               # --chunk: one POLICY decision is K*H ticks, so
                               # the per-decision cadence widens with it. The
                               # potential shaping telescopes across the whole
                               # window, so the sum is still exact; time_pen
                               # and the tick counters scale by `every`.
                               every=(KH if args.reward_per_decision else 1))
        reward_fn.speed_coef = args.speed_coef
    elif args.reward == "blend":
        reward_fn = BlendedReward(ForwardProgressReward(0.01),
                                  PathLengthReward(0.01),
                                  args.blend_start, args.blend_end)
    else:
        reward_fn = ForwardProgressReward(0.01)

    # per-decision reward path: only RaceReward knows how to telescope
    rpd = bool(args.reward_per_decision) and isinstance(reward_fn, RaceReward)

    eval_reward_feed = None
    if args.obs_reward:
        if isinstance(reward_fn, RaceReward) and goal_field is not None:
            eval_reward_feed = _make_eval_reward_feed(
                reward_field if reward_field is not None else goal_field,
                reward_fn.scale, reward_fn.time_pen, K)
        else:
            raise SystemExit("--obs-reward currently needs --reward race "
                             "(the eval feed mirrors the geodesic shaping)")

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
        n_q = quantilize_value_head(ck, policy)
        if n_q:
            print(f"--quantiles: replicated the scalar value head into {NQ} "
                  f"quantile rows ({n_q} tensors incl. Adam moments) - the "
                  "quantile MEAN is the checkpoint's value exactly, so the "
                  "resumed policy is function-identical at step 0")
        if N_ROUTE:
            n_w = widen_for_route(ck, policy)
            if n_w:
                print(f"--route: widened {n_w} checkpoint tensors by "
                      f"{N_ROUTE} zero columns — the resumed policy is "
                      "function-identical to the baseline at step 0")
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
                reward_fn.restore_counts(ck["int_counts"])
                n_visits = int(np.asarray(ck["int_counts"]).sum(dtype=np.int64))
                print(f"restored novelty counts ({n_visits:,} visits)")
        if rnd is not None and ck.get("rnd") is not None:
            rnd.load_state_dict_all(ck["rnd"])
            print("restored RND state (target/predictor/normalizers)")
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
                       "surf_mask": args.surf_mask,
                       "pinhole": args.pinhole,
                       "frame_stack": args.frame_stack,
                       # the route FILE is part of the observation spec: a
                       # resume against a different line would feed the same
                       # weights a differently-shaped world
                       "route_file": (route.source if route is not None
                                      else None),
                       "route_points": (len(route.offsets)
                                        if route is not None else None),
                       "route_span": (route.offsets[-1]
                                      if route is not None else None),
                       "route_critic_only": (int(bool(args.route_critic_only))
                                             if route is not None else None),
                       "fix_pitch": args.fix_pitch,
                       "emb": args.emb, "hidden": args.hidden, "gps": args.gps,
                       "teleport_fail": not args.keep_teleports,
                       "lidar_range": args.lidar_range,
                       "lidar_near": args.lidar_near or args.lidar_range,
                       "drop_min": args.drop_min, "drop_max": args.drop_max,
                       "punch_min": args.punch_min, "punch_max": args.punch_max,
                       "revisit_pen": args.revisit_pen,
                       "act_every": K, "pitch_rate": pitch_rate,
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
                       "lidar_cell": args.lidar_cell or pick_cell(core),
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
                       "fail_pen": (args.fail_pen
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
                       "respawn_frac": args.respawn_frac,
                       "respawn_margin": args.respawn_margin,
                       "respawn_binned": args.respawn_binned,
                       "respawn_mode": args.respawn_mode,
                       "respawn_bins": args.respawn_bins,
                       "respawn_killsafe": args.respawn_killsafe,
                       "race_shaping": args.race_shaping,
                       "spawn_burst": args.spawn_burst,
                       "spawn_burst_p": args.spawn_burst_p,
                       "demo_file": args.demo_file,
                       "demo_window": (args.demo_window
                                       if args.demo_file else None),
                       "demo_rate": (args.demo_rate
                                     if args.demo_file else None),
                       "demo_min_ep": (args.demo_min_ep
                                       if args.demo_file else None),
                       "race_kill_aware": args.race_kill_aware,
                       "respawn_reservoir": args.respawn_reservoir,
                       "respawn_speed": args.respawn_speed,
                       "ep_ticks": args.ep_ticks, "epochs": args.epochs,
                       "gamma": args.gamma, "gae": args.gae,
                       "clip": args.clip, "vf": args.vf, "ent": args.ent,
                       "quantiles": NQ,
                       "quantile_kappa": args.quantile_kappa,
                       "ent_final": args.ent_final,
                       "eval_eps": args.eval_eps,
                       "eval_greedy_only": args.eval_greedy_only,
                       "graphs": use_graphs, "compile": use_compile,
                       "bf16": use_bf16}}
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    CSV_COLS = ["time/total_timesteps", "rollout/ep_rew_mean",
                "rollout/ep_len_mean", "time/fps", "train/loss",
                "train/value_loss", "train/entropy_loss",
                "train/approx_kl", "eval/fwd_max", "eval/path",
                "eval/speed_max", "train/blend_w",
                "race/success_rate", "race/finish_s",
                "race/eval_progress", "race/eval_finish_s"]
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
    b_scal = torch.zeros((T, N, SCAL), device=device)
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
    demo_idx = np.full(N, -1, np.int64)   # demo index of each env's start
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
    vis_pin = torch.zeros((N, 6), pin_memory=(device.type == "cuda"))
    vis_np = vis_pin.numpy()
    vis_gpu = torch.zeros((N, 6), device=device)

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
        vis_np[:, 0:3] = sv_view["origin"]
        vis_np[:, 3] = sv_view["yaw"]
        vis_np[:, 4] = sv_view["pitch"]
        vis_np[:, 5] = sv_view["ducked"]
        vis_gpu.copy_(vis_pin, non_blocking=True)
        # --route: the lookahead fan rides the SAME pose upload the renderer
        # needs, so it costs one argmin and no extra host->device traffic.
        # Speed comes from the scalar row (slot 3 = |v_xy|/1000), which the
        # caller has already refreshed — so the fan and the depth image and
        # the scalars all describe one instant.
        if route is not None:
            dst[:, N_SCALAR:SCAL] = route.features(
                vis_gpu[:, 0:3], vis_gpu[:, 3], dst[:, 3] * 1000.0)
        ev = tm.gpu_start("lidar")
        # (N,H,W) or (N,H,W,2) under --surf-mask; flattening keeps the
        # channel fastest, which is what Policy.forward_split restrides
        img = lidar.render(vis_gpu[:, 0:3], vis_gpu[:, 3],
                           vis_gpu[:, 4], vis_gpu[:, 5])
        if ring is None:
            dst[:, SCAL:].copy_(img.reshape(N, -1))
        else:
            ring.push(img.reshape(N, FRAME), ended)
            dst[:, SCAL:].copy_(ring.compose())
        tm.gpu_end(ev)
        tm.add("vis_cpu", t0)

    def step_compute():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_bf16):
            logits, value = policy(static_obs)
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
        else:
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
    if args.obs_reward:
        static_obs[:, REWARD_SLOT] = 0.0     # no previous reward at reset
    ep_ret = np.zeros(N, np.float64)
    ep_len = np.zeros(N, np.int64)
    ret_hist = deque(maxlen=200)     # bounded: a 10B run finishes ~10M episodes
    len_hist = deque(maxlen=200)

    next_record = global_step
    next_ckpt = global_step + int(args.ckpt_every)
    last_latest_save = 0.0                   # force one write on iteration 1
    eval_fwd = eval_path = eval_speed = eval_prog = eval_fin = float("nan")
    t_start, step_start = time.perf_counter(), global_step

    def save_ckpt(tag):
        state = {"policy": policy.state_dict(),
                 "optimizer": contiguous_optimizer_state(opt.state_dict()),
                 "global_step": global_step, "config": meta["config"]}
        if isinstance(reward_fn, RaceReward):
            # novelty counts are cross-episode reward state: without them a
            # resume re-pays "first visit" for the whole beaten path
            state["int_counts"] = reward_fn.counts_state()
        if rnd is not None:
            state["rnd"] = rnd.state_dict_all()   # target net INCLUDED: a
            # re-rolled target makes every fitted state novel again
        if respawn is not None:
            state["respawn"] = respawn.state_dict()   # keep the frontier
        torch.save(state, out / f"ckpt_{tag}.pt")

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                         enabled=use_bf16)

    # the tau_hat_i the N quantile rows regress onto - built once, on device
    TAUS = quantile_midpoints(NQ, device=device) if NQ else None

    # ---- the PPO minibatch step, as one compilable function -----------------
    # Everything here is static-shaped (mb is constant), so inductor sees one
    # graph and never re-traces. The gathers stay INSIDE: fusing them with the
    # bf16 cast is part of what the compile buys.
    ent_t = torch.zeros((), device=device)

    def mb_step(f_scal, f_img, f_act, f_logp, f_adv, f_ret, idx, ent_coef,
                f_age=None, f_code=None, f_dmask=None):
        with amp:
            # STACK/N/PRO are Python constants, so the branch is decided at
            # trace time and inductor still sees one static-shaped graph
            img = (f_img[idx] if STACK == 1 else
                   stack_from_buffer(f_img, idx, f_age[idx], STACK, N, PRO))
            logits, value = policy.forward_split(f_scal[idx], img,
                                                 quantiles=NQ > 0)
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
                logp, ent = logprob_entropy_padded(
                    packer.pad(logits.float()), f_act[idx])
            value = value.float()
        ratio = torch.exp(logp - f_logp[idx])
        a = f_adv[idx]
        a = (a - a.mean()) / (a.std() + 1e-8)   # per-minibatch, like SB3
        pg = torch.max(-a * ratio,
                       -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
        # NQ is a Python constant, so the branch is decided at trace time
        # and inductor still sees one static-shaped graph
        if NQ:
            vl = quantile_value_loss(value, f_ret[idx], TAUS,
                                     args.quantile_kappa)
        else:
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

    MB = T * N // args.minibatches            # constant: the compiled shape
    if args.train_stride > 1 and (T // args.train_stride) * N < MB:
        raise SystemExit(f"--train-stride {args.train_stride} leaves fewer "
                         f"than one {MB}-sample minibatch of the {T}x{N} "
                         "rollout — lower the stride or --minibatches")
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
            mb_step(b_scal.reshape(T * N, SCAL),
                    b_img.reshape((PRO + T) * N, FRAME),
                    b_act.reshape(ACT_FLAT),
                    b_logp.reshape(-1), b_val.reshape(-1), b_rew.reshape(-1),
                    torch.arange(MB, device=device), ent_t,
                    None if b_age is None else b_age.reshape(-1),
                    None if b_code is None else b_code.reshape(-1),
                    None if b_dmask is None else b_dmask.reshape(T * N, H)
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
            (out / "run.json").write_text(json.dumps(meta, indent=2),
                                          encoding="utf-8")

    it_no = 0
    while global_step < int(args.steps):
        it_no += 1
        tm.start_iter()
        if hasattr(reward_fn, "set_step"):
            reward_fn.set_step(global_step)   # authoritative (survives resume)
        t_pool = tm.now()
        if demo is not None:
            # Salimans-Chen: the reservoir share of the pool is replaced by
            # exact demo-window states (velocities unscaled — the paper
            # resets to the demonstration state itself)
            core.set_spawn_pool(demo.build_pool(
                pool, fresh_frac=1.0 - args.respawn_frac))
        elif respawn is not None and respawn.size >= 2000:
            # refresh the spawn pool: fresh starts + perturbed mid-run
            # states. The 2000-state floor keeps the first lucky episode's
            # snapshots from seeding 90% of the fleet (degenerate,
            # self-reinforcing rollout correlation).
            core.set_spawn_pool(respawn.build_pool(
                pool, fresh_frac=1.0 - args.respawn_frac,
                vel_scale=tuple(args.respawn_speed),
                pitch_jitter=0.0 if args.fix_pitch is not None else 5.0))
        if (respawn is not None and goal_field is not None and respawn.size
                and it_no % 100 == 1):
            # reservoir depth vs the frontier: if min(d) trails eval progress
            # by a lot, the harvest margin (not the sampling) is what keeps
            # the agent from ever respawning near the wall
            fld = reward_field if reward_field is not None else goal_field
            rd = fld.sample(respawn._store[:respawn.size]["origin"])
            print(f"reservoir d: min {rd.min():,.0f}  p10 "
                  f"{np.percentile(rd, 10):,.0f}  median {np.median(rd):,.0f}"
                  f"  ({respawn.size:,} states)")
            if respawn.last_info:
                ep = respawn.bin_ep
                wins = respawn.bin_win.sum()
                print(f"  {respawn.last_info}  |  outcome-tracked eps "
                      f"{ep.sum():,.0f}  wins {wins:,.1f}")
        if demo is not None and it_no % 100 == 1 and demo.last_info:
            print(f"  {demo.last_info}  |  demo-tracked eps "
                  f"{demo.ep.sum():,.0f}  wins {demo.win.sum():,.1f}")
        tm.add("pool", t_pool)
        # ---------------- rollout ----------------
        t_roll = tm.now()
        with torch.no_grad():
            if ring is not None:
                # carry the previous iteration's last PRO renders into the
                # buffer's prologue, oldest first, so the update's gather sees
                # the same history the rollout's ring did
                ring.fill_prologue(b_img)
            for t in range(T):
                policy_step()
                t_sync = tm.now()
                b_scal[t].copy_(static_obs[:, :SCAL])
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
                b_act[t].copy_(static_act if H == 0 else static_plan)
                b_logp[t].copy_(static_logp)
                b_val[t].copy_(static_val)
                act_pin.copy_(static_act, non_blocking=True)
                if H > 0:
                    b_code[t].copy_(static_code)
                    plan_pin.copy_(static_plan, non_blocking=True)
                torch.cuda.synchronize() if device.type == "cuda" else None
                np.copyto(act_np32, act_pin.numpy(), casting="unsafe")
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
                if isinstance(reward_fn, RaceReward):
                    sm = reward_fn.pop_stall_mask()
                    if sm is not None:
                        core.force_fail(sm)     # stagnation kill, next tick
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
                    t_env = tm.now()
                    o2, base_r, done, trunc, term_obs = core.step(act_np32)
                    tm.add("env", t_env)
                    t_rew = tm.now()
                    if rpd:
                        # per-decision reward: the potential shaping
                        # telescopes across the K ticks, so one evaluation at
                        # the decision boundary is exact — only the masks
                        # need per-tick accumulation (goal_hits mutates every
                        # step; done rows autoreset mid-decision)
                        r = None
                        done_acc |= done.astype(bool)
                        goal_acc |= core.goal_hits.astype(bool)
                    else:
                        r = reward_fn(prev_obs, o2, term_obs, base_r, done,
                                      trunc, core)
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
                            vis = lidar.render(pos, yawd, ts[:, 9] * 90.0,
                                               ts[:, 5]).reshape(len(ti), -1)
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
                            if route is not None:
                                rt = route.features(pos, yawd, ts[:, 3] * 1000.0)
                                full = torch.cat([ts, rt, vis], dim=1)
                            else:
                                full = torch.cat([ts, vis], dim=1)
                            tv = policy(full)[1]
                            bv = args.gamma * tv.to("cpu").numpy()
                            if rpd:
                                r_acc[ti] += bv
                            else:
                                r[ti] += bv
                    tm.add("boot", t_boot)
                    t_book = tm.now()
                    if not rpd and ended.any():
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
                        if track_bins and ended.any():
                            # attribute the ended episodes' outcomes to the
                            # distance bin they STARTED in, then stash the
                            # new episodes' start bins (ended rows of
                            # sv_view are already the fresh spawns)
                            ei = np.flatnonzero(ended)
                            goal_now = core.goal_hits.astype(bool)[ei]
                            known = start_bin[ei] >= 0
                            if known.any():
                                respawn.note_outcomes(start_bin[ei][known],
                                                      goal_now[known])
                            nb = respawn.bin_of(sv_view[ei]["origin"])
                            start_bin[ei] = nb
                            respawn.note_spawns(nb, ei)
                        if demo is not None and ended.any():
                            # same stash-and-attribute, in demo-index space
                            ei = np.flatnonzero(ended)
                            goal_now = core.goal_hits.astype(bool)[ei]
                            known = demo_idx[ei] >= 0
                            if known.any():
                                demo.note_outcomes(demo_idx[ei][known],
                                                   goal_now[known])
                            demo_idx[ei] = demo.match(sv_view[ei]["origin"])
                    tm.add("respawn", t_resp)
                    if r is not None:
                        r_acc += r
                    ended_acc |= ended
                    global_step += N
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
                        for i in np.flatnonzero(ended_acc):
                            ret_hist.append(ep_ret[i]); len_hist.append(ep_len[i])
                        ep_ret[ended_acc] = 0; ep_len[ended_acc] = 0
                    tm.add("book", t_book)
                if rnd is not None:
                    live = ~ended_acc
                    r_acc[live] += args.rnd_coef * rnd_np[live]
                t_sync = tm.now()
                b_rew[t].copy_(torch.from_numpy(r_acc).to(device, non_blocking=True))
                b_done[t].copy_(torch.from_numpy(
                    ended_acc.astype(np.float32)).to(device, non_blocking=True))
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
                tm.add("sync_copy", t_sync)
                # b_done[t] is ended_acc already on the device — reuse it
                # rather than paying a second host->device copy
                fill_vision(static_obs, b_done[t] > 0 if ring is not None else None)
            tm.add("rollout_wall", t_roll)
            t_gae = tm.now()
            ev_gae = tm.gpu_start("gae_gpu")
            _, last_val = policy(static_obs)
            adv = torch.zeros_like(b_rew)
            lastgae = torch.zeros(N, device=device)
            # decision-granularity discount. Under --chunk one row is K*H
            # ticks and gamma**(K*H) is EXACT, not an SMDP approximation:
            # both occurrences below are multiplied by nonterm = 1 - b_done,
            # so a chunk cut short by an episode end never has g_eff applied
            # to it, and a chunk that runs to completion is always exactly
            # K*H ticks long (the neutral tail still burns wall-clock).
            g_eff = args.gamma ** KH
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
        f_scal = b_scal.reshape(T * N, SCAL)
        f_img = b_img.reshape((PRO + T) * N, FRAME)
        f_age = None if b_age is None else b_age.reshape(-1)
        f_act = b_act.reshape(ACT_FLAT)
        f_code = None if b_code is None else b_code.reshape(-1)
        f_dmask = None if b_dmask is None else b_dmask.reshape(T * N, H)
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
        for _ in range(args.epochs):
            if sub_pool is None:
                perm = torch.randperm(T * N, device=device)
                n_train = T * N
            else:
                perm = sub_pool[torch.randperm(sub_pool.numel(),
                                               device=device)]
                n_train = sub_pool.numel() - sub_pool.numel() % mb
            for s0 in range(0, n_train, mb):
                idx = perm[s0:s0 + mb]
                ev_mb = tm.gpu_start("mb_gpu")
                loss, pg, vl, el, logp = mb_step(
                    f_scal, f_img, f_act, f_logp, f_adv, f_ret, idx, ent_t,
                    f_age, f_code, f_dmask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                tm.gpu_end(ev_mb)     # before the float() syncs: mb_gpu vs
                # update measures how much of the update is GPU vs host gaps
                if rnd is not None:
                    rnd.train_step(f_scal[idx])   # tiny MLP, outside compile
                with torch.no_grad():
                    kl = float((f_logp[idx] - logp).mean())
                loss_v, loss_pi, loss_ent = float(vl), float(pg), float(el)
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
            n_rec = args.eval_eps or (3 if args.reward == "race" else 5)
            record_rollout(eval_core,
                           EVAL_GREEDY(policy, packer, device,
                                       lidar, eval_core, K, STACK,
                                       extra_slot=(REWARD_SLOT
                                                   if args.obs_reward
                                                   else -1),
                                       extra_fn=eval_reward_feed,
                                       route=route),
                           path, episodes=n_rec, max_ticks=n_rec * args.ep_ticks,
                           seed=global_step & 0x7FFFFFFF)
            st = episode_stats(path)
            eval_fwd = float(np.mean([e["fwd_max"] for e in st])) if st else 0.0
            eval_path = float(np.mean([e["path"] for e in st])) if st else 0.0
            eval_speed = float(np.mean([e["speed_max"] for e in st])) if st else 0.0
            prog_note = ""
            if goal_field is not None:
                eval_prog = race_progress(path, goal_field)
                n_fin, eval_fin, fin_best = eval_finish_times(path, goal_field)
                if n_fin:
                    prog_note = (f"  fin {n_fin}/{n_rec} mean {eval_fin:.2f}s"
                                 f" best {fin_best:.2f}s")
                prog_note += (f"  track {eval_prog:7.0f}u"
                             f"/{race_d0:.0f}u" if eval_prog == eval_prog else "")
            print(f"[{global_step:>13,d}] greedy: fwd {eval_fwd:7.0f}u  path "
                  f"{eval_path:7.0f}u  peak {eval_speed:6.0f} u/s{prog_note}"
                  f" -> {path.name}")
            if not args.eval_greedy_only:
                spath = out / f"traj_{global_step:010d}_stoch.jsonl"
                record_rollout(eval_core,
                               EVAL_SAMPLE(policy, packer, device,
                                           lidar, eval_core, K, STACK,
                                           route=route),
                               spath, episodes=n_rec,
                               max_ticks=n_rec * args.ep_ticks,
                               seed=global_step & 0x7FFFFFFF)
                sst = episode_stats(spath)
                if sst:
                    print(f"[{global_step:>13,d}] stoch : path "
                          f"{np.mean([e['path'] for e in sst]):7.0f}u"
                          f" -> {spath.name}")
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
                        round(eval_prog, 1) if eval_prog == eval_prog else "",
                        round(eval_fin, 2) if eval_fin == eval_fin else ""])
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
