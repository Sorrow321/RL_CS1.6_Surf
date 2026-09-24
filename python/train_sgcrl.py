"""train_sgcrl.py - Single-Goal Contrastive RL (SGCRL) on a surf / walk map.

What this is
------------
A standalone off-policy actor-critic trainer implementing Single-Goal
Contrastive RL: Liu, Tang, Eysenbach, "A Single Goal is All You Need"
(ICLR 2025, arXiv 2408.05804), built on Contrastive RL (Eysenbach et al.
2022, arXiv 2206.07568). There is NO reward anywhere in training: not a
sparse one, not a shaped one, not a novelty bonus. The only task
information is one goal g* (the centre of the map's finish box) that every
training episode is collected with. The mechanism paper (Bastankhah et al.
arXiv 2510.14129) reads the result as an implicit reward
phi(s,a)^T psi(g*) whose value the contrastive critic lowers along visited
trajectories that never reach the goal, which pushes the policy to new
places.

The reference behaviour reproduced here is the authors' released code
(github.com/graliuce/sgcrl, contrastive/learning.py + builder.py) with
JaxGCRL (arXiv 2408.11052) as the GPU-parallel reference:

critic   C(s,a,g) = phi(s,a)^T psi(g); phi, psi = MLP 2x256 ReLU -> R^64,
         unnormalised inner product (SGCRL: the inner-product critic is
         essential; a monolithic critic does not explore).
loss     forward InfoNCE over the batch (row i = anchor (s_i,a_i); column j =
         goal g_j; positives on the diagonal, negatives = the other anchors'
         positives) + 0.01 * logsumexp_j(logits_ij)^2 (learning.py, use_cpc).
positive g_i = the achieved goal (position) of a STRICTLY future state of the
         same episode, offset k ~ Geometric(1 - gamma) truncated to the
         steps the episode actually has and renormalised (the categorical
         over gamma^(j-i), j > i, of builder.flatten_fn / JaxGCRL
         flatten_batch). ``--future clip`` puts the tail mass on the last
         state instead.
actor    pi(a|s,g) tanh-Gaussian, MLP 2x256 ReLU, loc = 10 tanh(raw/10),
         std = softplus(raw) + 1e-6 (networks.py / NormalTanhDistribution).
         loss = E[alpha log pi(a|s,g) - phi(s,a)^T psi(g)] with a ~ pi
         (reparameterised). Goals: the batch's future goals, and with
         random_goals = 0.5 (the code's default) the batch is doubled with
         the goals rolled by one, so the actor trains on many goals
         (SGCRL Eq. 4, Fig. 10: training the actor on g* only is worse).
entropy  ``--alpha auto`` = SAC adaptive temperature with target entropy
         ``--target-entropy`` (default -A as the task spec asked).
         NOTE THE THREE-WAY DISCREPANCY: the released SGCRL code sets
         entropy_coefficient = 0.0, i.e. alpha is FIXED at 0 and there is no
         entropy term at all (``--alpha 0`` reproduces it); the paper's
         Table 2 lists "actor target entropy 0"; the paper's text calls alpha
         adaptive. See the report / run.json.
data     every training episode is collected with pi(.|s, g*), a ~ pi
         (stochastic). The first ``--random-steps`` transitions are uniform
         random actions (InitiallyRandomActor, 10,000 transitions).
update   all three losses are computed from the pre-update parameters and
         then all optimisers step (the order of learning.update_step).
         Adam lr 3e-4 (eps 1e-7 for actor/critic as in builder.py), batch 256.
ratio    ``--spi`` samples per inserted transition. SGCRL/acme used 256 with a
         4-actor CPU sampler; JaxGCRL, the GPU-parallel reference, trains
         num_envs*(episode_length-1) samples per num_envs*62 inserted steps
         = 16.1. Default 16 (JaxGCRL): the simulator is ~free next to the
         learner here, so this is the ratio designed for this regime.

Environment (state-based, like the paper's point maze)
------------------------------------------------------
SurfCore, spawn_mode 2 from the map's own spawn pool, no rendering
(lidar 0x0), water_fail 1, yaw jitter 0, teleport = fail, finish box armed
(an episode ends on touching it), episodes of ``--ep-secs`` (30 s).
A decision every ``--act-every`` = 4 physics ticks (40 ms, 25 Hz); the
action is held for the 4 ticks. If an episode ends inside a decision, the
remaining ticks of that decision run the NEUTRAL action in the freshly
reset episode (standing still at the spawn), never the old action.

state s (11): position (isotropically normalised, below), velocity/1000,
    sin/cos yaw, pitch/90, onground, ducked.
goal space (3): position normalised the same way. g* = finish-box centre.
normalisation: (p - c) / h with c the centre of core.map_bounds() and h
    its LARGEST half-extent (the same scale on every axis), so the biggest
    axis spans [-1, 1] and goal-space distances stay proportional to world
    distances. Per-axis scaling would magnify a flat map's height axis 5-9x
    (labyrinth z extent 288 u vs 2,592 u in y) and put the finish-box centre
    - 48 u above standing height - far off the visited manifold.

Action mapping (A = 4, or 5 with --duck 1), a in [-1, 1]^A:
    yaw bin  = floor((a0 + 1)/2 * 15) clipped to 0..14. The 15 bins are the
               core's per-TICK yaw rates -10 .. +10 deg (ascending, bin 7 =
               0; + = turn left / counter-clockwise), held 4 ticks, so up to
               40 deg per decision.
    pitch    = bin 3 (level) always: pitch only aims the lidar, which is off.
    forward  = 0 (back) if a1 < -1/3, 2 (forward) if a1 > 1/3, else 1.
    side     = 0 (A, left) if a2 < -1/3, 2 (D, right) if a2 > 1/3, else 1.
    jump     = a3 > 0.
    duck     = a4 > 0 with --duck 1, else never. Off by default: in this
               trainer's action set duck only slows ground movement (duck
               walk is 1/3 speed) and a fifth dimension is a fifth thing to
               explore; it is ONE constant for every map, not a per-map
               choice.
The critic sees the continuous a (the relaxation the actor's gradient needs).

``--yaw world`` (opt-in, A = 5): a0, a1 are a direction and the core's
absolute world-frame view mode (view_mode 2) turns toward the heading
atan2(a1, a0) at up to 10 deg per tick; keys from a2... Holding an action
then holds a heading instead of a turn rate. Measured on labyrinth_left100
(14-min GPU sanity run with the rate bins): the final policy was saturated at
yaw bin 14 (+10 deg/tick), forward and strafe-left - circling in the spawn
room - with yaw std 0.003. A heading target cannot express that spin, which
is why the option exists; the default stays the task spec.

What is measured and never trained on (CLAUDE.md 0b)
---------------------------------------------------
race/map_pct uses the geodesic goal field (surfgym.goalfield) purely as an
evaluation metric: loaded read-only from the cache next to the map when its
signature matches, else baked into <run>/field_cache/ (never next to the
map). Training reads nothing from it. The same holds for the coverage grid
(sgcrl/cells_visited) and sgcrl/train_map_pct.

``--goal-point x,y,z`` is a TEST-ONLY override (a learner sanity check with an
easy goal): g* becomes that point and the armed box becomes a cube of
+-``--goal-radius`` around it. It is recorded in run.json as such and is
never a recipe setting.

Outputs (runs/<run>/)
---------------------
progress.csv, run.json (config, git hash), ckpt_latest.pt (actor, critic,
log_alpha, optimisers, config, counters), traj_<step>.jsonl (greedy evals
from the true start in the repo's docs/04 JSONL format: header / per-tick
rows [t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward, pitch,
fwd, side] / trailer), evals.jsonl (per-episode eval summaries),
coverage.npz (every 64-u cell a training decision state has visited).

Usage
-----
    python python/train_sgcrl.py --map maps/labyrinth_left100.bsp \\
        --run sgLAB100 --minutes 90

Speed: the update is one CUDA graph (sample -> actor/alpha/critic losses ->
three Adam steps) replayed K times per decision step, enqueued before the
simulator ticks so the GPU learns while the CPU simulates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch                                                  # noqa: E402
import torch.nn as nn                                         # noqa: E402
import torch.nn.functional as F                               # noqa: E402

from surfgym.core import ACTION_NVEC                          # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

F_DIM = 11                      # state features
G_DIM = 3                       # goal space: position
YAW_N = int(ACTION_NVEC[0])     # 15 yaw-rate bins
PITCH_LEVEL = 3                 # PITCH_BINS[3] == 0 deg
NEUTRAL = (7, PITCH_LEVEL, 1, 1, 0, 0)   # no turn, level, no keys
SURF_IN_JUMP, SURF_IN_DUCK = 2, 4
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
LOG2 = math.log(2.0)
TICK_MS = 10                    # the core's default tick; this trainer never changes it

STAT_NAMES = ("critic_loss", "critic_acc", "logits_pos", "logits_neg",
              "logsumexp", "actor_loss", "alpha", "entropy", "q_pi", "std")


def act_dim(duck: bool, yaw: str = "rate") -> int:
    """rate: (yaw, fwd, side, jump[, duck]); world: (cos, sin, fwd, side, jump[, duck])."""
    return (4 if yaw == "rate" else 5) + (1 if duck else 0)


# ---------------------------------------------------------------------------
# action mapping (pure numpy)
# ---------------------------------------------------------------------------

def _keys(k: np.ndarray, duck: bool, out: np.ndarray) -> None:
    """Key columns (fwd, side, jump[, duck]) in [-1, 1] -> out[:, 2:6]."""
    third = 1.0 / 3.0
    out[:, 2] = np.where(k[:, 0] > third, 2, np.where(k[:, 0] < -third, 0, 1))
    out[:, 3] = np.where(k[:, 1] > third, 2, np.where(k[:, 1] < -third, 0, 1))
    out[:, 4] = (k[:, 2] > 0.0)
    out[:, 5] = (k[:, 3] > 0.0) if duck else 0


def to_discrete(a, duck: bool = False, out=None) -> np.ndarray:
    """``--yaw rate`` (the default): continuous actions (N, A) in [-1, 1] ->
    the core's int32 (N, 6) MultiDiscrete [yaw, pitch, forward, side, jump,
    duck] (module docstring lists the mapping). Deterministic; every discrete
    combination with the level pitch bin is reachable."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != act_dim(duck):
        raise ValueError(f"actions must be (N, {act_dim(duck)}), got {a.shape}")
    n = a.shape[0]
    if out is None:
        out = np.empty((n, 6), np.int32)
    yb = np.floor((a[:, 0].astype(np.float64) + 1.0) * 0.5 * YAW_N).astype(np.int32)
    np.clip(yb, 0, YAW_N - 1, out=yb)
    out[:, 0] = yb
    out[:, 1] = PITCH_LEVEL
    _keys(a[:, 1:], duck, out)
    return out


class ActionMap:
    """Continuous a in [-1, 1]^A -> what the core steps with: ``(acts int32
    (N, 6), view float32 (N, 2) or None)``.

    ``rate`` (default, the task spec): a0 -> one of the 15 per-tick yaw-RATE
    bins (``to_discrete``); no view array.
    ``world`` (opt-in): the core's absolute world-frame view mode (view_mode
    2, surf_step_view): the yaw TARGET is atan2(a1, a0) in degrees (a
    direction, so no +-180 seam) and the core turns toward it by at most
    yaw_rate_max_deg (10) per tick; pitch target 0 (level). Holding an action
    holds a heading - the rate bins' held action is a constant turn, and a
    saturated one is an in-place spin. The repo's PPO default is the same
    idea (absolute targets, surfgym/view.py). Keys come from a2.. as in rate.
    Ticks after an in-decision reset get NaN view commands (the core applies
    no turn) together with the neutral keys."""

    def __init__(self, yaw: str = "rate", duck: bool = False) -> None:
        if yaw not in ("rate", "world"):
            raise ValueError(f"unknown --yaw {yaw!r}")
        self.yaw = yaw
        self.duck = bool(duck)
        self.A = act_dim(self.duck, yaw)
        self.view_mode = 0 if yaw == "rate" else 2

    def __call__(self, a):
        a = np.asarray(a, dtype=np.float32)
        if self.yaw == "rate":
            return to_discrete(a, self.duck), None
        if a.ndim != 2 or a.shape[1] != self.A:
            raise ValueError(f"actions must be (N, {self.A}), got {a.shape}")
        n = a.shape[0]
        acts = np.empty((n, 6), np.int32)
        acts[:, 0] = NEUTRAL[0]                # ignored by the core in view mode
        acts[:, 1] = PITCH_LEVEL
        _keys(a[:, 2:], self.duck, acts)
        view = np.empty((n, 2), np.float32)
        view[:, 0] = np.degrees(np.arctan2(a[:, 1].astype(np.float64), a[:, 0].astype(np.float64)))
        view[:, 1] = 0.0
        return acts, view

    def describe(self) -> str:
        keys = "fwd/side +-1/3, jump >0" + (", duck >0" if self.duck else ", no duck")
        if self.yaw == "rate":
            return f"yaw RATE bin floor((a0+1)/2*15) (+-10 deg/tick); pitch bin 3; {keys} (a1..)"
        return f"yaw TARGET atan2(a1, a0) deg, world frame (core view_mode 2); pitch target 0; {keys} (a2..)"


# ---------------------------------------------------------------------------
# features (pure numpy)
# ---------------------------------------------------------------------------

class Normaliser:
    """Isotropic position normalisation from the map bounds (see module doc)."""

    def __init__(self, mins, maxs) -> None:
        mins = np.asarray(mins, np.float64)
        maxs = np.asarray(maxs, np.float64)
        self.center = 0.5 * (mins + maxs)
        self.scale = float(max(np.max(0.5 * (maxs - mins)), 1.0))

    def pos(self, p) -> np.ndarray:
        return ((np.asarray(p, np.float64) - self.center) / self.scale).astype(np.float32)

    def unpos(self, q) -> np.ndarray:
        return np.asarray(q, np.float64) * self.scale + self.center


def state_features(states, norm: Normaliser, out=None) -> np.ndarray:
    """STATE_DTYPE rows (N,) -> float32 (N, F_DIM): position (normalised),
    velocity/1000, sin/cos yaw, pitch/90, onground, ducked."""
    n = len(states)
    if out is None:
        out = np.empty((n, F_DIM), np.float32)
    out[:, 0:3] = norm.pos(states["origin"])
    out[:, 3:6] = np.asarray(states["velocity"], np.float32) / 1000.0
    yr = np.radians(np.asarray(states["yaw"], np.float64))
    out[:, 6] = np.sin(yr)
    out[:, 7] = np.cos(yr)
    out[:, 8] = np.asarray(states["pitch"], np.float32) / 90.0
    out[:, 9] = np.asarray(states["onground"]) >= 0
    out[:, 10] = np.asarray(states["ducked"]) != 0
    return out


def box_distance(p, mins, maxs) -> np.ndarray:
    """Euclidean distance from points (N, 3) to an AABB (0 inside)."""
    p = np.atleast_2d(np.asarray(p, np.float64))
    q = np.clip(p, np.asarray(mins, np.float64), np.asarray(maxs, np.float64))
    return np.linalg.norm(p - q, axis=1)


# ---------------------------------------------------------------------------
# future-offset sampler (torch; unit-tested)
# ---------------------------------------------------------------------------

def future_offsets(L: torch.Tensor, u: torch.Tensor, gamma: float,
                   mode: str = "renorm") -> torch.Tensor:
    """Offsets k in 1..L, L >= 1 = the future steps the episode has.

    ``renorm``: P(k) proportional to gamma^(k-1) on 1..L (Geometric(1-gamma)
    truncated to the episode and renormalised - the reference implementations'
    categorical over gamma^(j-i), j > i). Inverse CDF:
    P(k <= m) = (1 - gamma^m) / (1 - gamma^L).
    ``clip``: k ~ Geometric(1-gamma) on 1, 2, ..., then min(k, L) - the tail
    mass lands on the last state.
    ``u`` is U[0, 1) of L's shape (drawn by the caller: CUDA-graph friendly).
    Computed in float64: in float32, 1 - gamma^L loses the digits that matter
    as gamma -> 1 (a percent-level skew at gamma = 1 - 1e-6), and there are
    only batch-many of these numbers.
    """
    Lf = L.to(torch.float64)
    u = u.to(torch.float64)
    lg = math.log(gamma)
    if mode == "renorm":
        c = 1.0 - torch.pow(gamma, Lf)
        k = torch.ceil(torch.log1p(-u * c) / lg)
    elif mode == "clip":
        k = torch.ceil(torch.log1p(-u) / lg)
    else:
        raise ValueError(f"unknown future mode {mode!r}")
    k = torch.maximum(k, torch.ones_like(k))
    k = torch.minimum(k, Lf)
    return k.long()


# ---------------------------------------------------------------------------
# replay buffer (torch, static shapes)
# ---------------------------------------------------------------------------

class Replay:
    """[T, N] time-major ring: one row per env per decision.

    Row (t, e) holds env e's decision state features obs (its first 3
    columns ARE the achieved goal), the continuous action, and ``ep_end`` =
    absolute row index of the LAST decision row of that row's episode (-1
    while the episode is still running). An episode's terminal position is
    stored on its last row (``term``) and counts as one more future goal.

    Every env writes exactly one row per decision step, so the write head is
    one scalar shared by all envs; the valid rows of every env are the
    absolute indices [max(0, head - T), head - 1]. Anchors are drawn from
    [lo, head - 2]: the newest row is excluded, and every other row has at
    least one future goal (the next row of a running episode, or the next
    row / the terminal of a finished one). No rejection, no data-dependent
    shapes - the sampler runs inside a CUDA graph. T must exceed the longest
    episode (checked by the caller), so an episode never wraps onto itself.
    """

    def __init__(self, n_envs: int, t_cap: int, a_dim: int, device) -> None:
        self.N = int(n_envs)
        self.T = int(t_cap)
        dev = torch.device(device)
        self.device = dev
        self.obs = torch.zeros(self.T, self.N, F_DIM, device=dev)
        self.act = torch.zeros(self.T, self.N, a_dim, device=dev)
        self.term = torch.zeros(self.T, self.N, G_DIM, device=dev)
        self.ep_end = torch.full((self.T, self.N), -1, dtype=torch.int64, device=dev)
        self.head_t = torch.zeros((), dtype=torch.int64, device=dev)
        self.head = 0                                   # host mirror
        self.ep_start = np.zeros(self.N, np.int64)      # host: current episode's first row

    @property
    def rows(self) -> int:
        return min(self.head, self.T) * self.N

    def write_step(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> None:
        s = self.head % self.T
        self.obs[s].copy_(obs_t)
        self.act[s].copy_(act_t)
        self.ep_end[s].fill_(-1)
        self.head += 1
        self.head_t.fill_(self.head)

    def close_episodes(self, envs: np.ndarray, term_goal: np.ndarray) -> None:
        """``envs`` finished during the decision whose row is head - 1:
        attach their terminal goals to that row and stamp ep_end on every
        row of the finished episodes."""
        envs = np.asarray(envs, np.int64)
        if envs.size == 0:
            return
        last = self.head - 1
        s = last % self.T
        e_t = torch.as_tensor(envs, device=self.device)
        self.term[s, e_t] = torch.as_tensor(np.asarray(term_goal, np.float32),
                                            device=self.device)
        starts = self.ep_start[envs]
        lens = last - starts + 1
        if (lens < 1).any() or (lens > self.T).any():
            raise RuntimeError(f"episode bookkeeping broken: lengths {lens}")
        tot = int(lens.sum())
        rep_env = np.repeat(envs, lens)
        offs = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
        rows = np.repeat(starts, lens) + offs
        self.ep_end[torch.as_tensor(rows % self.T, device=self.device),
                    torch.as_tensor(rep_env, device=self.device)] = last
        self.ep_start[envs] = last + 1

    def sample(self, B: int, gamma: float, mode: str = "renorm", debug: bool = False):
        """(obs (B,F), act (B,A), future goal (B,3)) for B uniform anchors.
        ``debug`` also returns the anchor/future absolute rows and env ids."""
        dev = self.device
        head = self.head_t
        lo = torch.clamp(head - self.T, min=0)
        count = torch.clamp(head - 1 - lo, min=1)           # rows lo .. head-2
        env = torch.randint(0, self.N, (B,), device=dev)
        a_abs = lo + torch.minimum((torch.rand(B, device=dev) * count).long(), count - 1)
        slot = a_abs % self.T
        end = self.ep_end[slot, env]
        fin = end >= 0
        last = torch.where(fin, end, head - 1)
        L = last - a_abs + fin.long()
        k = future_offsets(L, torch.rand(B, device=dev), gamma, mode)
        f_abs = a_abs + k
        is_term = fin & (f_abs > end)
        f_slot = torch.where(is_term, end, f_abs) % self.T
        goal = torch.where(is_term[:, None], self.term[f_slot, env],
                           self.obs[f_slot, env, 0:G_DIM])
        out = (self.obs[slot, env], self.act[slot, env], goal)
        if debug:
            return out + ({"env": env, "a_abs": a_abs, "f_abs": f_abs,
                           "is_term": is_term, "L": L, "k": k},)
        return out

    def recent_obs(self, steps: int) -> torch.Tensor:
        """Features of the last ``steps`` written decision steps, (n, F)."""
        m = min(int(steps), self.head, self.T)
        if m <= 0:
            return self.obs[:0].reshape(0, F_DIM)
        idx = torch.as_tensor([(self.head - 1 - i) % self.T for i in range(m)],
                              device=self.device)
        return self.obs[idx].reshape(-1, F_DIM)


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------

def _mlp(inp: int, hidden: int, out: int, init: str) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, out))
    for m in net:
        if isinstance(m, nn.Linear):
            if init == "glorot":        # VarianceScaling(1, fan_avg, uniform)
                nn.init.xavier_uniform_(m.weight)
            else:                       # VarianceScaling(1, fan_in, uniform)
                lim = math.sqrt(3.0 / m.in_features)
                nn.init.uniform_(m.weight, -lim, lim)
            nn.init.zeros_(m.bias)
    return net


class Critic(nn.Module):
    """phi(s, a) and psi(g), both MLP 2 x hidden ReLU -> R^repr (networks.py
    sa_encoder / g_encoder, Glorot-uniform init, zero biases)."""

    def __init__(self, a_dim: int, hidden: int = 256, repr_dim: int = 64) -> None:
        super().__init__()
        self.phi = _mlp(F_DIM + a_dim, hidden, repr_dim, "glorot")
        self.psi = _mlp(G_DIM, hidden, repr_dim, "glorot")


class Actor(nn.Module):
    """pi(a | s, g): MLP 2 x hidden ReLU -> tanh-Gaussian (NormalTanhDistribution:
    loc = 10 tanh(raw/10), std = softplus(raw) + min_std)."""

    def __init__(self, a_dim: int, hidden: int = 256, min_std: float = 1e-6) -> None:
        super().__init__()
        self.body = _mlp(F_DIM + G_DIM, hidden, 2 * a_dim, "lecun")
        self.a_dim = a_dim
        self.min_std = float(min_std)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        loc = 10.0 * torch.tanh(h[:, :self.a_dim] / 10.0)
        std = F.softplus(h[:, self.a_dim:]) + self.min_std
        return loc, std


def tanh_gauss_logp(eps: torch.Tensor, u: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """log pi(tanh(u)) for u = loc + std * eps: Gaussian log-density of u minus
    the tanh Jacobian, log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))."""
    lp = -0.5 * eps.pow(2) - torch.log(std) - HALF_LOG_2PI
    lp = lp - 2.0 * (LOG2 - u - F.softplus(-2.0 * u))
    return lp.sum(-1)


def critic_loss_fn(critic: Critic, obs, act, goal, lse_coef: float):
    """Forward InfoNCE + lse_coef * logsumexp^2 (SGCRL learning.py, use_cpc).
    Returns (loss, logits)."""
    sa = critic.phi(torch.cat([obs, act], 1))
    gg = critic.psi(goal)
    logits = sa @ gg.t()
    lse = torch.logsumexp(logits, 1)
    loss = (lse - logits.diagonal()).mean() + lse_coef * lse.pow(2).mean()
    return loss, logits, lse


# ---------------------------------------------------------------------------
# learner
# ---------------------------------------------------------------------------

class Learner:
    """Actor, critic, temperature, their optimisers and one update step that
    can be captured into a CUDA graph (sampling included)."""

    def __init__(self, rb: Replay, a_dim: int, args, device) -> None:
        self.rb = rb
        self.dev = torch.device(device)
        self.B = int(args.batch)
        self.gamma = float(args.gamma)
        self.future = str(args.future)
        self.rg = float(args.random_goals)
        self.lse_coef = float(args.lse_coef)
        self.a_dim = int(a_dim)
        self.actor = Actor(a_dim, args.hidden, args.min_std).to(self.dev)
        self.critic = Critic(a_dim, args.hidden, args.repr_dim).to(self.dev)
        cuda = self.dev.type == "cuda"
        self.adaptive = str(args.alpha).lower() == "auto"
        self.target_entropy = float(args.target_entropy if args.target_entropy is not None
                                    else -float(a_dim))
        # CUDA: fused + capturable Adam (one kernel per optimiser step, legal
        # inside a CUDA graph); CPU: the plain implementation
        kw = dict(capturable=True, fused=True) if cuda else {}
        if cuda:
            try:                    # an older torch may refuse fused + capturable
                torch.optim.Adam([torch.zeros(1, device=self.dev, requires_grad=True)], **kw)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"learner: fused capturable Adam unavailable ({exc}); using foreach")
                kw = dict(capturable=True)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=args.lr, eps=1e-7, **kw)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=args.lr, eps=1e-7, **kw)
        if self.adaptive:
            self.log_alpha = torch.zeros((), device=self.dev, requires_grad=True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr, **kw)
            self.alpha_fixed = None
        else:
            self.log_alpha = None
            self.opt_alpha = None
            self.alpha_fixed = torch.tensor(float(args.alpha), device=self.dev)
        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())
        self.stats = torch.zeros(len(STAT_NAMES), device=self.dev)
        self.nstat = torch.zeros((), device=self.dev)
        self._arange = torch.arange(self.B, device=self.dev)
        self.use_graph = bool(args.cuda_graph) and cuda
        self.graph = None
        self.n_updates = 0

    # -- temperature --------------------------------------------------------
    def alpha(self) -> torch.Tensor:
        if self.adaptive:
            return self.log_alpha.detach().exp()
        return self.alpha_fixed

    # -- one update (graph body) -------------------------------------------
    def _body(self) -> None:
        obs, act, goal = self.rb.sample(self.B, self.gamma, self.future)
        # ---- actor: goals relabelled from the batch (random_goals) ----
        if self.rg == 0.5:
            s2 = torch.cat([obs, obs], 0)
            g2 = torch.cat([goal, torch.roll(goal, 1, 0)], 0)
        elif self.rg == 1.0:
            s2, g2 = obs, torch.roll(goal, 1, 0)
        else:
            s2, g2 = obs, goal
        loc, std = self.actor(torch.cat([s2, g2], 1))
        eps = torch.randn_like(loc)
        u = loc + std * eps
        a_pi = torch.tanh(u)
        logp = tanh_gauss_logp(eps, u, std)
        alpha = self.alpha()
        q_pi = (self.critic.phi(torch.cat([s2, a_pi], 1)) * self.critic.psi(g2)).sum(1)
        actor_loss = (alpha * logp - q_pi).mean()
        actor_loss.backward(inputs=self.actor_params)
        if self.adaptive:
            alpha_loss = (self.log_alpha.exp() * (-logp.detach() - self.target_entropy)).mean()
            alpha_loss.backward(inputs=[self.log_alpha])
        # ---- critic ----
        critic_loss, logits, lse = critic_loss_fn(self.critic, obs, act, goal, self.lse_coef)
        critic_loss.backward(inputs=self.critic_params)
        # ---- steps (all losses above used the pre-update parameters) ----
        self.opt_c.step()
        self.opt_a.step()
        if self.adaptive:
            self.opt_alpha.step()
        with torch.no_grad():
            diag = logits.diagonal()
            B = logits.shape[0]
            off = (logits.sum() - diag.sum()) / (B * (B - 1))
            acc = (logits.argmax(1) == self._arange).float().mean()
            self.stats.add_(torch.stack([
                critic_loss.detach(), acc, diag.mean(), off, lse.mean(),
                actor_loss.detach(), alpha, -logp.mean(), q_pi.mean(), std.mean()]))
            self.nstat.add_(1.0)

    def _zero(self) -> None:
        self.opt_a.zero_grad(set_to_none=True)
        self.opt_c.zero_grad(set_to_none=True)
        if self.adaptive:
            self.opt_alpha.zero_grad(set_to_none=True)

    def update(self) -> None:
        if self.use_graph and self.graph is None:
            self._capture()
        if self.graph is not None:
            self.graph.replay()
        else:
            self._zero()
            self._body()
        self.n_updates += 1

    def _capture(self) -> None:
        """Whole-update CUDA graph (PyTorch 'whole network capture'): warm up
        on a side stream, grads set to None so backward allocates them from
        the graph pool, then capture. Falls back to eager on any failure."""
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):              # real updates: counted
                    self._zero()
                    self._body()
                    self.n_updates += 1
            torch.cuda.current_stream().wait_stream(s)
            self._zero()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._body()
            self.graph = g
            print("learner: update captured as one CUDA graph")
        except Exception as exc:                          # noqa: BLE001
            self.graph = None
            self.use_graph = False
            self._zero()
            print(f"learner: CUDA graph capture failed ({exc!r}); running eager")

    # -- acting ---------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs_t: torch.Tensor, goal_t: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        x = torch.cat([obs_t, goal_t.expand(obs_t.shape[0], -1)], 1)
        loc, std = self.actor(x)
        if greedy:
            return torch.tanh(loc)
        return torch.tanh(loc + std * torch.randn_like(loc))

    @torch.no_grad()
    def read_stats(self) -> dict:
        n = float(self.nstat.item())
        vals = (self.stats / max(n, 1.0)).tolist()
        self.stats.zero_()
        self.nstat.zero_()
        if n <= 0:
            return {}
        return dict(zip(STAT_NAMES, vals))

    @torch.no_grad()
    def goal_similarity(self, obs: torch.Tensor, goal_t: torch.Tensor) -> dict:
        """psi(pos)^T psi(g*) (sim_visited: the mechanism paper's
        psi-similarity) and phi(s, pi(s,g*))^T psi(g*) (the implicit reward
        the actor climbs) on the given states."""
        if obs.shape[0] == 0:
            return {}
        pg = self.critic.psi(goal_t.reshape(1, -1))[0]
        sim = (self.critic.psi(obs[:, :G_DIM]) @ pg).mean()
        a = self.act(obs, goal_t.reshape(1, -1), greedy=True)
        q = (self.critic.phi(torch.cat([obs, a], 1)) @ pg).mean()
        return {"sim_visited": float(sim.item()), "q_goal": float(q.item()),
                "psi_goal_norm": float(pg.norm().item())}

    # -- persistence ------------------------------------------------------------
    def state_dict(self) -> dict:
        d = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
             "opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict()}
        if self.adaptive:
            d["log_alpha"] = self.log_alpha.detach().cpu()
            d["opt_alpha"] = self.opt_alpha.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.opt_a.load_state_dict(d["opt_a"])
        self.opt_c.load_state_dict(d["opt_c"])
        if self.adaptive and "log_alpha" in d:
            with torch.no_grad():
                self.log_alpha.copy_(d["log_alpha"].to(self.dev))
            self.opt_alpha.load_state_dict(d["opt_alpha"])


# ---------------------------------------------------------------------------
# measurement-only geodesic field
# ---------------------------------------------------------------------------

def load_measure_field(core, zone, cell: float, run_dir: Path, device: str):
    """Geodesic distance-to-finish (surfgym.goalfield) for race/map_pct ONLY.
    The cache next to the map is read when its signature matches; a missing
    or stale one is baked into <run>/field_cache/, never next to the map."""
    from surfgym import goalfield as gfm
    from surfgym.vision import _map_sig
    bsp = Path(core.bsp_path)
    box = np.round(np.asarray(list(zone["mins"]) + list(zone["maxs"]), np.float64), 1)
    sig = (f"g{gfm._GOAL_BUILDER_VERSION}_{_map_sig(bsp)}_"
           + "_".join(f"{v:g}" for v in box))
    cache = bsp.parent / f"{bsp.stem}.goal_{cell:g}.npz"
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if "sig" in z and str(z["sig"]) == sig:
                gf = gfm.GoalField(z["grid"].astype(np.float32) * float(z["quant"]),
                                   z["mins"], float(z["cell"]), float(z["reach_max"]))
                return gf, str(cache)
        except Exception as exc:                          # noqa: BLE001
            print(f"field cache {cache.name} unreadable ({exc}); baking a private copy")
    fdir = run_dir / "field_cache"
    fdir.mkdir(parents=True, exist_ok=True)
    print(f"measurement field: no valid cache next to the map; baking into {fdir}")
    gf = gfm.build_goal_field(core, zone, cell, cache_dir=str(fdir), device=device)
    return gf, f"baked:{fdir}"


# ---------------------------------------------------------------------------
# environment helpers
# ---------------------------------------------------------------------------

def make_core(bsp: str, n: int, ep_ticks: int, view_mode: int = 0):
    from surfgym.core import SurfCore, default_config
    cfg = default_config(num_envs=int(n), spawn_mode=2, lidar_w=0, lidar_h=0,
                         water_fail=1, yaw_jitter_deg=0.0,
                         max_episode_ticks=int(ep_ticks), view_mode=int(view_mode))
    return SurfCore(str(bsp), cfg)


def arm_core(core, spawn_pool, box) -> None:
    core.set_teleport_fail(True)
    core.set_goal_box(box["mins"], box["maxs"])
    core.set_spawn_pool(spawn_pool)


def live_spawns(bsp: str, pool, box, ep_ticks: int, view_mode: int, ticks: int = 8) -> np.ndarray:
    """bool per spawn point: False where a player standing still at it ends
    the episode as a FAIL within ``ticks`` (2 decisions) - stuck in geometry
    (the core's 5-tick stuck rule) or inside a kill volume. One rule for every
    map, no constant read off any map. Measured: surf_edgeflow_blue050 loses
    its whole y = -1248 row (4 of 16, dead at tick 5); the labyrinths 0 of 16.
    Since 2026-09-24 rewards.map_spawn_pool lifts such spawns clear (4 u up there),
    so on the pool it builds this finds none; it stays as the check."""
    n = len(pool)
    core = make_core(bsp, n, ep_ticks, view_mode)
    arm_core(core, pool, box)
    core.reset(0)
    for i in range(n):
        core.set_state(i, pool[i:i + 1].copy())
    acts = np.tile(np.asarray(NEUTRAL, np.int32), (n, 1))
    view = np.full((n, 2), np.nan, np.float32) if view_mode else None
    dead = np.zeros(n, bool)
    for _ in range(int(ticks)):
        _o, _r, done, _tr, _t = core.step(acts, view=view)
        dead |= (done != 0) & ~np.asarray(core.goal_hits, bool)
    core.close()
    return ~dead


class Collector:
    """Drives the training core one decision at a time and keeps the
    host-side episode bookkeeping. Terminal goals are STASHED and applied to
    the buffer at the start of the next decision, so nothing here touches the
    GPU while the learner's updates are in flight."""

    def __init__(self, core, norm: Normaliser, act_every: int, duck: bool,
                 field, box, cell_stat: float, map_mins, map_maxs,
                 box_is_finish: bool = True) -> None:
        self.core = core
        self.N = core.num_envs
        self.norm = norm
        self.K = int(act_every)
        self.duck = bool(duck)
        self.field = field
        self.box = box
        # a hit counts as 100% of the route only when the armed box IS the
        # map's finish (not under the TEST-ONLY --goal-point)
        self.box_is_finish = bool(box_is_finish)
        self.sv = core.states_view
        self.feat = np.empty((self.N, F_DIM), np.float32)
        self.pending = None
        self.ticks = 0
        self.decisions = 0
        self.ep_rows = np.zeros(self.N, np.int64)
        self.d0 = np.full(self.N, np.nan)
        self.dmin = np.full(self.N, np.inf)
        self.neutral = np.asarray(NEUTRAL, np.int32)
        # coverage grid (measurement)
        self.cmins = np.asarray(map_mins, np.float64)
        self.cell = float(cell_stat)
        self.cdims = np.maximum(np.ceil((np.asarray(map_maxs, np.float64) - self.cmins)
                                        / self.cell).astype(np.int64) + 1, 1)
        self.cover = np.zeros(int(np.prod(self.cdims)), bool)
        self.best_pct = 0.0
        self.min_box = np.inf
        self.episodes = 0
        self.goals = 0
        self._reset_interval()

    def _reset_interval(self) -> None:
        self.iv = {"end": 0, "goal": 0, "fail": 0, "trunc": 0, "empty": 0,
                   "len_s": 0.0, "pct": 0.0, "pct_n": 0}

    def interval(self) -> dict:
        iv = self.iv
        self._reset_interval()
        n = max(iv["end"], 1)
        return {"train_success": iv["goal"] / n if iv["end"] else float("nan"),
                "train_fail_frac": iv["fail"] / n if iv["end"] else float("nan"),
                "train_ep_s": iv["len_s"] / n if iv["end"] else float("nan"),
                "train_map_pct": (iv["pct"] / iv["pct_n"]) if iv["pct_n"] else float("nan"),
                "train_episodes": iv["end"], "empty_episodes": iv["empty"]}

    def stagger(self, rng, ep_ticks: int) -> None:
        """Give every env's FIRST episode a random start clock, so episodes
        (and their synchronized resets) do not run in lock-step."""
        st = self.core.get_states()
        steps = max(ep_ticks // self.K, 1)
        off = rng.integers(0, steps, self.N) * self.K
        for e in range(self.N):
            row = st[e:e + 1].copy()
            row["tick"] = int(off[e])
            self.core.set_state(e, row)

    def observe(self) -> np.ndarray:
        """Decision-time features (a copy-free view into self.feat)."""
        st = self.sv
        state_features(st, self.norm, out=self.feat)
        org = np.asarray(st["origin"], np.float64)
        ix = np.clip(((org - self.cmins) // self.cell).astype(np.int64), 0, self.cdims - 1)
        self.cover[ix[:, 0] + self.cdims[0] * (ix[:, 1] + self.cdims[1] * ix[:, 2])] = True
        self.min_box = min(self.min_box, float(box_distance(org, self.box["mins"],
                                                            self.box["maxs"]).min()))
        if self.field is not None:
            d = self.field.sample(org).astype(np.float64)
            new = self.ep_rows == 0
            self.d0[new] = d[new]
            np.minimum(self.dmin, d, out=self.dmin)
        self.ep_rows += 1
        self.decisions += self.N
        return self.feat

    def step(self, a_int: np.ndarray, view=None) -> None:
        """Hold ``a_int`` (and the ``view`` targets under --yaw world) for K
        ticks; an env whose episode ends mid-decision runs NEUTRAL (NaN view
        = no turn) for the rest. Terminal goals: from terminal_obs for a goal
        hit / truncation, from the pre-tick state for a fail (a teleport fail
        can move the origin before the terminal obs is written)."""
        core = self.core
        acts = np.ascontiguousarray(a_int, dtype=np.int32).copy()
        if view is not None:
            view = np.ascontiguousarray(view, dtype=np.float32).copy()
        ended_any = np.zeros(self.N, bool)
        firsts, terms = [], []
        for j in range(self.K):
            pre = np.asarray(self.sv["origin"], np.float64).copy()
            _obs, _rew, done, trunc, term = core.step(acts, view=view)
            ended = (done != 0) | (trunc != 0)
            self.ticks += self.N
            if not ended.any():
                continue
            gh = np.asarray(core.goal_hits, bool)
            first = ended & ~ended_any
            second = ended & ended_any
            if second.any():
                self.iv["empty"] += int(second.sum())
            idx = np.flatnonzero(first)
            if idx.size:
                mc = self._map_center()
                tpos = term[idx, 12:15].astype(np.float64) * 2000.0 + mc
                fail = (done[idx] != 0) & ~gh[idx]
                tpos[fail] = pre[idx][fail]
                firsts.append(idx)
                terms.append(self.norm.pos(tpos))
                self._account(idx, gh[idx], fail, trunc[idx] != 0, j)
            ended_any |= ended
            acts[ended] = self.neutral
            if view is not None:
                view[ended] = np.nan
        if firsts:
            self.pending = (np.concatenate(firsts), np.concatenate(terms))
        else:
            self.pending = None

    def _map_center(self) -> np.ndarray:
        # the core writes obs positions relative to the centre of its world
        # bounds (src/env.c map_center); cached on first use
        if getattr(self, "_mc", None) is None:
            mn, mx = self.core.map_bounds()
            self._mc = 0.5 * (np.asarray(mn, np.float64) + np.asarray(mx, np.float64))
        return self._mc

    def _account(self, idx, goal, fail, trunc, j) -> None:
        iv = self.iv
        n = idx.size
        iv["end"] += n
        iv["goal"] += int(goal.sum())
        iv["fail"] += int(fail.sum())
        iv["trunc"] += int((trunc & ~goal & ~fail).sum())
        iv["len_s"] += float((((self.ep_rows[idx] - 1) * self.K + j + 1) * TICK_MS / 1000.0).sum())
        self.episodes += n
        self.goals += int(goal.sum())
        if self.field is not None:
            d0 = self.d0[idx]
            ok = np.isfinite(d0) & (d0 > 1.0)
            pct = np.clip(100.0 * (d0 - self.dmin[idx]) / np.where(ok, d0, 1.0), 0.0, 100.0)
            if self.box_is_finish:
                pct = np.where(goal, 100.0, pct)
            if ok.any():
                iv["pct"] += float(pct[ok].sum())
                iv["pct_n"] += int(ok.sum())
                self.best_pct = max(self.best_pct, float(pct[ok].max()))
            self.d0[idx] = np.nan
            self.dmin[idx] = np.inf
        self.ep_rows[idx] = 0

    def flush(self, rb: Replay) -> None:
        if self.pending is not None:
            rb.close_episodes(*self.pending)
            self.pending = None

    @property
    def cells_visited(self) -> int:
        return int(self.cover.sum())

    def save_coverage(self, path: Path) -> None:
        """Every cell any training decision state has been in (measurement
        only): grid [z, y, x] of ``cell``-unit voxels from ``mins``."""
        dx, dy, dz = (int(v) for v in self.cdims)
        np.savez_compressed(path, cover=self.cover.reshape(dz, dy, dx), mins=self.cmins,
                            cell=np.float32(self.cell))


# ---------------------------------------------------------------------------
# evaluation (greedy, true start, g*)
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, bsp: str, n_eps: int, ep_ticks: int, spawn_pool, box, norm,
                 act_every: int, amap: ActionMap, field, seed: int, map_name: str,
                 goal_world, box_is_finish: bool = True) -> None:
        from surfgym.core import phys_to_dict
        self.core = make_core(bsp, n_eps, ep_ticks, amap.view_mode)
        arm_core(self.core, spawn_pool, box)
        self.n = int(n_eps)
        self.ep_ticks = int(ep_ticks)
        self.norm = norm
        self.K = int(act_every)
        self.amap = amap
        self.field = field
        self.box = box
        self.seed = int(seed)
        phys = phys_to_dict(self.core.config.phys)
        phys["msec"] = int(self.core.nominal_msec)
        self.header = {"map": map_name, "tick_ms": TICK_MS, "phys": phys}
        self.goal_world = [round(float(v), 2) for v in goal_world]
        self.box_is_finish = bool(box_is_finish)

    def run(self, learner: Learner, goal_t: torch.Tensor, step: int, path: Path) -> dict:
        core = self.core
        core.reset(self.seed)
        sv = core.states_view
        n, K = self.n, self.K
        alive = np.ones(n, bool)
        rows = [[] for _ in range(n)]
        end = ["trunc"] * n
        fin_t = [None] * n
        acts = np.tile(np.asarray(NEUTRAL, np.int32), (n, 1))
        view = None
        feat = np.empty((n, F_DIM), np.float32)
        spawn = np.asarray(sv["origin"], np.float64).copy()
        dev = goal_t.device
        for t in range(self.ep_ticks + K):
            if t % K == 0:
                state_features(sv, self.norm, out=feat)
                a = learner.act(torch.as_tensor(feat, device=dev), goal_t.reshape(1, -1),
                                greedy=True).cpu().numpy()
                acts, view = self.amap(a)
                acts[~alive] = np.asarray(NEUTRAL, np.int32)
                if view is not None:
                    view[~alive] = np.nan
            st = sv.copy()
            _o, _r, done, trunc, _term = core.step(acts, view=view)
            gh = np.asarray(core.goal_hits, bool)
            for e in np.flatnonzero(alive):
                s = st[e]
                o, v = s["origin"], s["velocity"]
                buttons = (SURF_IN_JUMP if acts[e, 4] else 0) | (SURF_IN_DUCK if acts[e, 5] else 0)
                rows[e].append([len(rows[e]), round(float(o[0]), 2), round(float(o[1]), 2),
                                round(float(o[2]), 2), round(float(v[0]), 2), round(float(v[1]), 2),
                                round(float(v[2]), 2), round(float(s["yaw"]), 2), int(buttons),
                                int(int(s["onground"]) >= 0), 0.0, 0.0, round(float(s["pitch"]), 2),
                                int(acts[e, 2]), int(acts[e, 3])])
                if done[e] or trunc[e]:
                    alive[e] = False
                    if gh[e]:
                        end[e] = "done"
                        fin_t[e] = len(rows[e])
                    else:
                        end[e] = "fail" if done[e] else "trunc"
            if not alive.any():
                break
        eps = []
        for e in range(n):
            p = np.asarray([r[1:4] for r in rows[e]], np.float64)
            fin = fin_t[e] is not None
            dbox = 0.0 if fin else float(box_distance(p, self.box["mins"], self.box["maxs"]).min())
            pct = float("nan")
            if self.field is not None and len(p):
                d = self.field.sample(p).astype(np.float64)
                d0 = float(d[0])
                if d0 > 1.0:
                    pct = float(np.clip(100.0 * (d0 - d.min()) / d0, 0.0, 100.0))
                    if fin and self.box_is_finish:
                        pct = 100.0
            eps.append({"episode": e, "spawn": [round(float(v), 1) for v in spawn[e]],
                        "end": end[e], "ticks": len(rows[e]),
                        "finish_s": (fin_t[e] * TICK_MS / 1000.0) if fin else None,
                        "min_box_dist": round(dbox, 1), "map_pct": pct,
                        "xyz_min": [round(float(v), 1) for v in p.min(0)] if len(p) else None,
                        "xyz_max": [round(float(v), 1) for v in p.max(0)] if len(p) else None})
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for e in range(n):
                hdr = dict(self.header)
                hdr.update({"episode": e, "tool": "train_sgcrl", "step": int(step),
                            "goal": {"center": self.goal_world, "radius": 64.0}})
                f.write(json.dumps(hdr, separators=(",", ":")) + "\n")
                for r in rows[e]:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
                ep = eps[e]
                f.write(json.dumps({"end": ep["end"], "ticks": ep["ticks"], "best_progress": 0.0,
                                    "finish_s": ep["finish_s"], "min_box_dist": ep["min_box_dist"],
                                    "map_pct": None if not np.isfinite(ep["map_pct"]) else round(ep["map_pct"], 2)},
                                   separators=(",", ":")) + "\n")
        fins = [ep["finish_s"] for ep in eps if ep["finish_s"] is not None]
        pcts = [ep["map_pct"] for ep in eps if np.isfinite(ep["map_pct"])]
        return {"episodes": eps,
                "finishes": len(fins),
                "finish_s": min(fins) if fins else None,
                "map_pct": float(np.mean(pcts)) if pcts else float("nan"),
                "min_box_dist": float(np.mean([ep["min_box_dist"] for ep in eps])),
                "best_box_dist": float(min(ep["min_box_dist"] for ep in eps))}


# ---------------------------------------------------------------------------
# logging helpers
# ---------------------------------------------------------------------------

CSV_COLS = [
    "time/total_timesteps", "time/elapsed_s", "time/fps",
    "race/maps_finished", "race/eval_finish_s", "race/map_pct", "race/eval_finishes",
    "eval/box_hits", "eval/box_hit_s", "eval/min_box_dist", "eval/best_box_dist",
    "sgcrl/critic_loss", "sgcrl/critic_acc", "sgcrl/logits_pos", "sgcrl/logits_neg",
    "sgcrl/logsumexp", "sgcrl/actor_loss", "sgcrl/alpha", "sgcrl/entropy", "sgcrl/q_pi",
    "sgcrl/std", "sgcrl/train_success", "sgcrl/train_fail_frac", "sgcrl/train_ep_s",
    "sgcrl/train_map_pct", "sgcrl/train_best_pct", "sgcrl/train_min_box_dist",
    "sgcrl/sim_visited", "sgcrl/q_goal", "sgcrl/psi_goal_norm", "sgcrl/cells_visited",
    "sgcrl/decisions", "sgcrl/grad_steps", "sgcrl/updates_per_s", "sgcrl/decisions_per_s",
    "sgcrl/episodes", "sgcrl/goal_hits", "sgcrl/buffer_rows",
]


class CsvLog:
    def __init__(self, path: Path, resume: bool) -> None:
        self.path = path
        new = not (resume and path.exists())
        self.f = open(path, "a" if not new else "w", encoding="utf-8", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            self.w.writeheader()
        self.f.flush()

    def row(self, d: dict) -> None:
        out = {}
        for k in CSV_COLS:
            v = d.get(k)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                out[k] = ""
            elif isinstance(v, float):
                out[k] = f"{v:.6g}"
            else:
                out[k] = v
        self.w.writerow(out)
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def git_info() -> dict:
    def _run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:                                  # noqa: BLE001
            return ""
    return {"git": _run("rev-parse", "HEAD"),
            "git_branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_run("status", "--porcelain", "--untracked-files=no"))}


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def parse_point(s: str) -> np.ndarray:
    v = [float(x) for x in str(s).replace(" ", "").split(",")]
    if len(v) != 3:
        raise SystemExit(f"--goal-point needs x,y,z, got {s!r}")
    return np.asarray(v, np.float64)


def train(args) -> dict:
    from surfgym.rewards import map_spawn_pool
    from surfgym.zones import load_zones

    if args.threads and args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(int(args.threads))   # before the DLL loads
    if not (args.minutes > 0 or args.steps > 0):
        raise SystemExit("give a budget: --minutes and/or --steps")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    bsp = Path(args.map).resolve()
    if not bsp.is_file():
        raise SystemExit(f"map not found: {bsp}")
    map_name = bsp.stem
    out = Path(args.run)
    if not out.is_absolute() and len(out.parts) == 1:
        out = Path(args.runs_dir) / out
    out = out.resolve()
    if (out / "progress.csv").exists() and not args.resume:
        raise SystemExit(f"{out} already has a progress.csv; pick a new --run or --resume")
    out.mkdir(parents=True, exist_ok=True)

    K = int(args.act_every)
    ep_ticks = int(round(args.ep_secs * 1000.0 / TICK_MS))
    max_rows = -(-ep_ticks // K)                     # decision rows per episode
    duck = bool(args.duck)
    amap = ActionMap(args.yaw, duck)
    A = amap.A
    N = int(args.envs)

    zones = load_zones(str(bsp), create=False)
    finish = zones.get("end")
    if not finish:
        raise SystemExit(f"no finish ('end') zone for {map_name}")
    finish = {"mins": [float(v) for v in finish["mins"]], "maxs": [float(v) for v in finish["maxs"]]}
    test_goal = args.goal_point is not None
    if test_goal:
        gp = parse_point(args.goal_point)
        r = float(args.goal_radius)
        box = {"mins": (gp - r).tolist(), "maxs": (gp + r).tolist()}
        goal_world = gp
        print("!! --goal-point: TEST-ONLY easy goal (a learner sanity check, never a recipe) "
              f"at {goal_world.tolist()}, box +-{r:g} u; the real finish is NOT armed")
    else:
        box = finish
        goal_world = 0.5 * (np.asarray(box["mins"]) + np.asarray(box["maxs"]))

    core = make_core(str(bsp), N, ep_ticks, amap.view_mode)
    pool = map_spawn_pool(core)
    dropped = []
    if args.drop_dead_spawns:
        live = live_spawns(str(bsp), pool, box, ep_ticks, amap.view_mode)
        if not live.any():
            raise SystemExit(f"every one of the {len(pool)} spawn points of {map_name} fails within "
                             "2 decisions standing still - no start to train from")
        dropped = [[round(float(v), 1) for v in o] for o in pool["origin"][~live]]
        if dropped:
            print(f"spawn pool: dropped {len(dropped)} of {len(pool)} spawn points that fail within "
                  f"2 decisions standing still (stuck / kill volume): {dropped}")
        pool = pool[live]
    arm_core(core, pool, box)
    mins, maxs = core.map_bounds()
    norm = Normaliser(mins, maxs)
    goal_norm = norm.pos(goal_world[None, :])[0]

    field, field_src = None, None
    if not args.no_field:
        fdev = "cuda" if device.startswith("cuda") else "cpu"
        field, field_src = load_measure_field(core, finish, 32.0, out, fdev)
        d_sp = field.sample(pool["origin"].astype(np.float64))
        print(f"measurement field ({field_src}): spawn geodesic distance "
              f"median {float(np.median(d_sp)):.0f} u (never used in training)")

    t_cap = max(int(math.ceil(float(args.buffer) / N)), max_rows + 2)
    rb = Replay(N, t_cap, A, device)
    learner = Learner(rb, A, args, device)
    goal_t = torch.as_tensor(goal_norm, device=learner.dev)
    upd_per_step = float(args.spi) * N / float(args.batch)
    start_steps = max(2, int(math.ceil(float(args.random_steps) / N)))

    col = Collector(core, norm, K, duck, field, box, args.cell_stat, mins, maxs,
                    box_is_finish=not test_goal)
    ev = Evaluator(str(bsp), args.eval_eps, ep_ticks, pool, box, norm, K, amap, field,
                   args.seed + 7919, map_name, goal_world, box_is_finish=not test_goal)

    counters = {"ticks": 0, "decisions": 0, "grad_steps": 0, "episodes": 0, "goals": 0}
    if args.resume:
        ck = torch.load(args.resume, map_location=learner.dev, weights_only=False)
        learner.load_state_dict(ck["learner"])
        counters.update({k: int(v) for k, v in ck.get("counters", {}).items() if k in counters})
        col.ticks = counters["ticks"]
        col.decisions = counters["decisions"]
        col.episodes = counters["episodes"]
        col.goals = counters["goals"]
        learner.n_updates = counters["grad_steps"]
        print(f"resumed {args.resume} at {col.ticks:,} env steps (replay starts empty)")

    config = {k: v for k, v in vars(args).items()}
    config.update({
        "map": map_name, "map_path": str(bsp), "device": device, "act_dim": A, "obs_dim": F_DIM,
        "goal_dim": G_DIM, "ep_ticks": ep_ticks, "rows_per_episode_max": max_rows,
        "replay_T": t_cap, "replay_rows": t_cap * N, "updates_per_decision_step": upd_per_step,
        "prefill_decision_steps": start_steps, "tick_ms": TICK_MS,
        "target_entropy_eff": learner.target_entropy if learner.adaptive else None,
        "goal_world": [float(v) for v in goal_world], "goal_norm": [float(v) for v in goal_norm],
        "goal_box": box, "finish_box": finish, "goal_override_TEST_ONLY": bool(test_goal),
        "map_bounds": [[float(v) for v in mins], [float(v) for v in maxs]],
        "norm_center": norm.center.tolist(), "norm_scale": norm.scale,
        "spawns": int(len(pool)), "spawns_dropped_dead": dropped, "measure_field": field_src,
        "action_map": amap.describe(), "view_mode": amap.view_mode,
    })
    meta = {"tool": "train_sgcrl", "label": f"{out.name} (SGCRL {map_name})",
            "started": datetime.now().isoformat(timespec="seconds"), "finished": None,
            "config": config, **git_info(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None}
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"SGCRL | map {map_name} | envs {N} | A {A} ({args.yaw} yaw) | batch {args.batch} | gamma {args.gamma} | "
          f"spi {args.spi:g} -> {upd_per_step:.2f} updates per decision step | replay "
          f"{t_cap} x {N} = {t_cap * N:,} rows | alpha {args.alpha}"
          f"{f' (target entropy {learner.target_entropy:g})' if learner.adaptive else ''} | "
          f"device {device}{' + CUDA graph' if learner.use_graph else ''}")
    print(f"g* = {np.round(goal_world, 1).tolist()} (norm {np.round(goal_norm, 3).tolist()}) | "
          f"episode {args.ep_secs:g} s = {max_rows} decisions of {K} ticks | prefill "
          f"{start_steps} decision steps ({start_steps * N:,} transitions)"
          f"{' uniform random' if not args.resume else ' with the resumed policy'}")

    log = CsvLog(out / "progress.csv", bool(args.resume))
    t0 = time.time()
    last_log_t = t0
    last_log_ticks = col.ticks
    last_log_upd = learner.n_updates
    last_log_dec = col.decisions
    next_eval = col.ticks                     # an eval right away (the step-0 baseline)
    last_eval = None
    credit = 0.0
    step_i = 0
    core.reset(args.seed)
    if args.stagger:
        col.stagger(rng, ep_ticks)
    obs_host = np.empty((N, F_DIM), np.float32)

    last_eval_step = [-1]

    def do_eval() -> dict:
        nonlocal last_eval
        te = time.time()
        res = ev.run(learner, goal_t, col.ticks, out / f"traj_{col.ticks}.jsonl")
        last_eval = res
        last_eval_step[0] = col.ticks
        with open(out / "evals.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": col.ticks, "elapsed_s": round(time.time() - t0, 1),
                                "finishes": res["finishes"], "finish_s": res["finish_s"],
                                "map_pct": res["map_pct"], "min_box_dist": res["min_box_dist"],
                                "episodes": res["episodes"]}) + "\n")
        xs = [ep["xyz_min"][0] for ep in res["episodes"] if ep["xyz_min"]]
        print(f"[eval @ {col.ticks:,}] {'TEST-box hits' if test_goal else 'finishes'} "
              f"{res['finishes']}/{args.eval_eps} | best "
              f"{res['finish_s'] if res['finish_s'] is not None else '-'} s | map_pct "
              f"{res['map_pct']:.1f} | mean min box dist {res['min_box_dist']:.0f} u | "
              f"x range of eval paths {min(xs) if xs else float('nan'):.0f}.. | "
              f"{time.time() - te:.1f} s")
        save_ckpt()
        col.save_coverage(out / "coverage.npz")
        return res

    def save_ckpt() -> None:
        counters.update({"ticks": col.ticks, "decisions": col.decisions,
                         "grad_steps": learner.n_updates, "episodes": col.episodes,
                         "goals": col.goals})
        atomic_save({"learner": learner.state_dict(), "config": config, "counters": counters,
                     "arch": {"F": F_DIM, "G": G_DIM, "A": A, "hidden": args.hidden,
                              "repr_dim": args.repr_dim}},
                    out / "ckpt_latest.pt")

    def log_row(eval_res=None) -> None:
        nonlocal last_log_t, last_log_ticks, last_log_upd, last_log_dec
        now = time.time()
        dt = max(now - last_log_t, 1e-9)
        st = learner.read_stats()
        ci = col.interval()
        sim = learner.goal_similarity(rb.recent_obs(16)[:8192], goal_t) if rb.head > 0 else {}
        row = {"time/total_timesteps": col.ticks, "time/elapsed_s": round(now - t0, 1),
               "time/fps": (col.ticks - last_log_ticks) / dt,
               "sgcrl/decisions": col.decisions, "sgcrl/grad_steps": learner.n_updates,
               "sgcrl/updates_per_s": (learner.n_updates - last_log_upd) / dt,
               "sgcrl/decisions_per_s": (col.decisions - last_log_dec) / dt,
               "sgcrl/episodes": col.episodes, "sgcrl/goal_hits": col.goals,
               "sgcrl/buffer_rows": rb.rows, "sgcrl/cells_visited": col.cells_visited,
               "sgcrl/train_best_pct": col.best_pct if field is not None else None,
               "sgcrl/train_min_box_dist": col.min_box}
        for k, v in st.items():
            row[f"sgcrl/{k}"] = v
        for k in ("train_success", "train_fail_frac", "train_ep_s", "train_map_pct"):
            row[f"sgcrl/{k}"] = ci[k]
        for k, v in sim.items():
            row[f"sgcrl/{k}"] = v
        if eval_res is not None:
            # eval/box_*: the ARMED box; race/*: the map's real finish only, so
            # a TEST-ONLY --goal-point hit never reads as a finished map
            real = not test_goal
            row.update({"race/maps_finished": (1 if eval_res["finishes"] > 0 else 0) if real else 0,
                        "race/eval_finish_s": eval_res["finish_s"] if real else None,
                        "race/map_pct": eval_res["map_pct"],
                        "race/eval_finishes": eval_res["finishes"] if real else 0,
                        "eval/box_hits": eval_res["finishes"],
                        "eval/box_hit_s": eval_res["finish_s"],
                        "eval/min_box_dist": eval_res["min_box_dist"],
                        "eval/best_box_dist": eval_res["best_box_dist"]})
        log.row(row)
        print(f"{col.ticks:>13,} steps | {row['time/fps']:>9,.0f} fps | "
              f"{row['sgcrl/updates_per_s']:>6,.0f} upd/s | crit {st.get('critic_loss', float('nan')):.3f} "
              f"acc {st.get('critic_acc', float('nan')):.3f} | alpha {st.get('alpha', float('nan')):.3g} "
              f"ent {st.get('entropy', float('nan')):.2f} | train succ {ci['train_success']:.3f} "
              f"({ci['train_episodes']} eps) | cells {col.cells_visited} | sim {sim.get('sim_visited', float('nan')):.2f} "
              f"| best train pct {col.best_pct:.1f}")
        last_log_t, last_log_ticks, last_log_upd, last_log_dec = now, col.ticks, learner.n_updates, col.decisions

    try:
        while True:
            if col.ticks >= next_eval:
                res = do_eval()
                log_row(res)
                next_eval = col.ticks + float(args.eval_every)
            elapsed = time.time() - t0
            if (args.minutes > 0 and elapsed >= args.minutes * 60.0) or \
                    (args.steps > 0 and col.ticks >= args.steps):
                break
            # (1) observe on the CPU while the GPU is still running the last
            # decision's updates; (2) then the finished episodes of the last
            # decision go into the buffer - the first GPU touch, which waits
            # for those updates (one host sync per decision step)
            feats = col.observe()
            col.flush(rb)
            np.copyto(obs_host, feats)
            obs_t = torch.as_tensor(obs_host).to(learner.dev, non_blocking=False)
            learning = step_i >= start_steps
            if learning or args.resume:
                a_t = learner.act(obs_t, goal_t.reshape(1, -1))
            else:
                a_t = torch.as_tensor(rng.uniform(-1.0, 1.0, (N, A)).astype(np.float32),
                                      device=learner.dev)
            a_np = a_t.cpu().numpy()
            # (3) write the decision rows, (4) enqueue the updates (async on GPU)
            rb.write_step(obs_t, a_t)
            if learning:
                credit += upd_per_step
                k = int(credit)
                credit -= k
                for _ in range(k):
                    learner.update()
            # (5) simulate the decision while the GPU learns
            acts_np, view_np = amap(a_np)
            col.step(acts_np, view_np)
            step_i += 1
            if time.time() - last_log_t >= args.log_secs:
                log_row()
    except KeyboardInterrupt:
        print("interrupted - final eval and checkpoint")
    if last_eval is not None and last_eval_step[0] == col.ticks:
        res = last_eval                      # the loop ended right after an eval
        save_ckpt()
    else:
        res = do_eval()
        log_row(res)
    log.close()
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    meta["duration_s"] = round(time.time() - t0, 1)
    meta["total_steps"] = int(col.ticks)
    meta["result"] = {"eval_finishes": 0 if test_goal else res["finishes"],
                      "eval_finish_s": None if test_goal else res["finish_s"],
                      "eval_box_hits": res["finishes"], "eval_box_hit_s": res["finish_s"],
                      "eval_map_pct": res["map_pct"], "train_box_hits": col.goals,
                      "train_episodes": col.episodes, "grad_steps": learner.n_updates,
                      "cells_visited": col.cells_visited,
                      "train_best_pct": col.best_pct if field is not None else None}
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ev.core.close()
    core.close()
    print(f"done: {col.ticks:,} env steps, {learner.n_updates:,} gradient steps, "
          f"{col.episodes:,} training episodes ({col.goals} goal hits), {meta['duration_s']:.0f} s -> {out}")
    return {"out": str(out), "final_eval": res, "ticks": col.ticks,
            "grad_steps": learner.n_updates, "episodes": col.episodes, "goals": col.goals}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Single-Goal Contrastive RL (Liu, Tang, Eysenbach 2025) on a surf map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--map", required=True, help="path to the .bsp (its zones.json next to it)")
    ap.add_argument("--run", required=True, help="run name (under --runs-dir) or a path")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--envs", type=int, default=256, help="parallel training envs")
    ap.add_argument("--minutes", type=float, default=0.0, help="wall-clock budget (0 = none)")
    ap.add_argument("--steps", type=float, default=0.0,
                    help="env-step budget, physics ticks summed over envs (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.99, help="per DECISION (40 ms)")
    ap.add_argument("--repr-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--buffer", type=float, default=1e6, help="replay size in transitions")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", default="auto",
                    help="'auto' = adaptive temperature; a number = fixed alpha "
                         "(0 = the released SGCRL code: no entropy term)")
    ap.add_argument("--target-entropy", type=float, default=None,
                    help="adaptive-alpha target (default -A, the task spec; SGCRL Table 2 says 0)")
    ap.add_argument("--lse-coef", type=float, default=0.01, help="logsumexp^2 regulariser")
    ap.add_argument("--random-goals", type=float, default=0.5, choices=(0.0, 0.5, 1.0),
                    help="actor-goal relabelling (CRL/SGCRL code default 0.5)")
    ap.add_argument("--future", choices=("renorm", "clip"), default="renorm",
                    help="truncated geometric: renormalised (reference code) or tail on the last state")
    ap.add_argument("--spi", type=float, default=16.0,
                    help="samples per inserted transition (JaxGCRL 16; SGCRL/acme 256)")
    ap.add_argument("--random-steps", type=int, default=10000,
                    help="uniform-random transitions before learning (SGCRL 10,000)")
    ap.add_argument("--min-std", type=float, default=1e-6)
    ap.add_argument("--ep-secs", type=float, default=30.0)
    ap.add_argument("--act-every", type=int, default=4, help="physics ticks per decision")
    ap.add_argument("--duck", type=int, default=0, choices=(0, 1))
    ap.add_argument("--yaw", choices=("rate", "world"), default="rate",
                    help="rate = the 15 per-tick yaw-rate bins (task spec); world = an absolute "
                         "world-frame heading target atan2(a1, a0) (core view_mode 2)")
    ap.add_argument("--drop-dead-spawns", type=int, default=1, choices=(0, 1),
                    help="drop map spawn points where standing still fails within 2 decisions "
                         "(stuck in geometry / kill volume); applied to training and evals")
    ap.add_argument("--stagger", type=int, default=1, choices=(0, 1),
                    help="random start clock for each env's first episode")
    ap.add_argument("--eval-every", type=float, default=2e7, help="env steps between evals")
    ap.add_argument("--eval-eps", type=int, default=9)
    ap.add_argument("--log-secs", type=float, default=15.0,
                    help="progress.csv heartbeat (the dashboard calls a run dead after 30 s)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--cuda-graph", type=int, default=1, choices=(0, 1))
    ap.add_argument("--threads", type=int, default=0, help="OMP_NUM_THREADS for the core (0 = as is)")
    ap.add_argument("--cell-stat", type=float, default=64.0,
                    help="coverage-grid cell (u) for sgcrl/cells_visited - measurement only")
    ap.add_argument("--no-field", action="store_true",
                    help="skip the measurement-only geodesic field (race/map_pct blank)")
    ap.add_argument("--goal-point", default=None,
                    help="TEST-ONLY: x,y,z replaces g* and the armed goal box (learner sanity check)")
    ap.add_argument("--goal-radius", type=float, default=64.0,
                    help="TEST-ONLY: half-size of the --goal-point box")
    ap.add_argument("--resume", default=None, help="ckpt_latest.pt to continue from (weights, "
                                                   "optimisers, counters; the replay starts empty)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
