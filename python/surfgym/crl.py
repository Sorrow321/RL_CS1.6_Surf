"""crl.py - single-goal contrastive RL inside the PPO trainer (--crl).

The method
----------
CPPO (Osman et al. 2026, arXiv 2605.13554) is PPO whose advantage comes from
a CONTRASTIVE critic instead of a reward and a learned value function:

    A(s, a) = Q(s, a, g*) - V(s, g*),    V(s, g*) = E_{a ~ pi(.|s)} Q(s, a, g*)

Q(s, a, g) = f(phi(s, a), psi(g)) is trained with InfoNCE on hindsight-
relabelled futures (CRL, Eysenbach et al. 2022, arXiv 2206.07568): the
POSITIVE goal of an anchor (s_t, a_t) is the position the SAME episode
reached k ~ Geometric(1 - gamma) decisions later, the NEGATIVES are the other
rows' positives. At the optimum f = log p^pi(g | s, a) - log p(g) + const,
i.e. the log discounted occupancy of g from (s, a). The goal is always the
ONE hard goal g*, the centre of the finish box (SGCRL, Liu, Tang, Eysenbach
2024, arXiv 2408.05804): no reward enters learning, no curriculum, no
subgoal, and the policy is never shown g* (it is a constant).

Why it should explore (Bastankhah et al. 2025, arXiv 2510.14129): the policy
maximises an implicit reward psi(s)^T psi(g*), and the contrastive update
LOWERS that similarity on the states the agent keeps visiting without
reaching g*, so a dead end loses its pull; once g* is reached the
similarity rises along the successful path. `sim_visited` below is that
quantity over the states the fleet just visited: it should FALL before the
goal is first found.

What is where
-------------
* ``MapFrame`` / ``state_features`` / ``action_features``: the critic's input.
  phi reads a PRIVILEGED low-dimensional state straight off the core
  (position normalised by the map bounds, velocity / 1000, sin/cos yaw,
  pitch / 90, on-ground and ducked flags) and the action the POLICY chose
  (one-hot per categorical head, plus tanh(z) of the continuous view heads).
  psi reads a position normalised the same way. The trainer appends the 7
  --keys-hold held-key columns to the state, because under that flag a
  "keep" bin means whatever key is held.
* ``FutureRing``: a per-env GPU history of the last L decisions (all envs
  step in lockstep, so one head serves the fleet) holding phi's input, the
  position, the episode-end flag, the TERMINAL position of an episode that
  ended during a decision, and a per-env episode id. ``sample`` draws
  anchors whose future is resolvable, positives at t + k, k ~ Geom(1 - gamma)
  clipped at the episode end (the terminal position - a death, a finish, a
  stall kill or a truncation - is the last legitimate future: the absorbing
  convention) and never across an episode boundary.
* ``ContrastiveCritic``: phi and psi, 2 x ``hidden`` ReLU MLPs into a
  ``repr_dim`` space (CRL's defaults 256 / 64, Glorot-uniform like its
  VarianceScaling(1, fan_avg, uniform)); ``dot`` = phi^T psi (SGCRL: the
  inner product is essential, a monolithic critic fails) or ``l2`` =
  -||phi - psi||_2 (CPPO's energy).
* ``infonce``: forward InfoNCE (softmax over the goals of each anchor row,
  positives on the diagonal) + ``lse`` x mean(logsumexp(logits, 1)^2), the
  LogSumExp regulariser of SGCRL Eq. 3 in the public CRL code's form.
* ``ContrastiveRL``: the critic, its own Adam, the ring, a dedicated RNG,
  ``train`` (N updates x batch per iteration), ``advantage`` (Q of the acted
  row minus the Monte-Carlo V over K actions drawn from the behaviour
  distribution the rollout sampled from) and ``pop_stats``.

Nothing here runs unless the trainer is launched with --crl.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RAW_DIM", "STATE_DIM", "STATE_FEATURES", "MapFrame",
           "raw_from_states", "state_features", "action_features",
           "geometric_offsets", "sample_actions", "ContrastiveCritic",
           "infonce", "FutureRing", "ContrastiveRL"]

#: the raw per-decision fields the rollout copies off ``core.states_view``:
#: origin (3), velocity (3), yaw deg, pitch deg, onground, ducked
RAW_DIM = 10
#: phi's state block, in order (the trainer may append the held-key columns)
STATE_FEATURES = ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
                  "sin_yaw", "cos_yaw", "pitch", "onground", "ducked")
STATE_DIM = len(STATE_FEATURES)
VEL_SCALE = 1000.0
PITCH_SCALE = 90.0
#: "this episode has not ended inside the ring" in next_end
_OPEN = 1 << 40


def raw_from_states(states, out: np.ndarray) -> None:
    """Copy the fields the critic reads off a STATE_DTYPE array (the core's
    zero-copy ``states_view``) into ``out`` (n, RAW_DIM) float32."""
    out[:, 0:3] = states["origin"]
    out[:, 3:6] = states["velocity"]
    out[:, 6] = states["yaw"]
    out[:, 7] = states["pitch"]
    out[:, 8] = states["onground"]
    out[:, 9] = states["ducked"]


class MapFrame:
    """The position normalisation phi's state and psi's goal share:
    (p - centre) / scale, centre = the BSP bounds' midpoint and scale = their
    LARGEST half-extent, so the three axes share one scale and a position
    inside the map lands in [-1, 1] (surfgym/privfeat.py's convention)."""

    def __init__(self, mins, maxs):
        mins = np.asarray(mins, np.float64).reshape(3)
        maxs = np.asarray(maxs, np.float64).reshape(3)
        self.center = (mins + maxs) / 2.0
        self.scale = float(max(float(np.max((maxs - mins) / 2.0)), 1.0))

    def norm(self, p: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(self.center, dtype=p.dtype, device=p.device)
        return (p - c) / self.scale

    def norm_np(self, p) -> np.ndarray:
        return ((np.asarray(p, np.float64) - self.center)
                / self.scale).astype(np.float32)


def state_features(raw: torch.Tensor, frame: MapFrame) -> torch.Tensor:
    """(..., RAW_DIM) raw state -> (..., STATE_DIM) phi state block."""
    pos = frame.norm(raw[..., 0:3])
    vel = raw[..., 3:6] / VEL_SCALE
    yaw = raw[..., 6] * (math.pi / 180.0)
    pitch = raw[..., 7] / PITCH_SCALE
    onground = (raw[..., 8] >= 0.0).to(raw.dtype)   # -1 airborne, else a solid
    ducked = (raw[..., 9] != 0.0).to(raw.dtype)
    return torch.cat([pos, vel, torch.sin(yaw)[..., None],
                      torch.cos(yaw)[..., None], pitch[..., None],
                      onground[..., None], ducked[..., None]], dim=-1)


def _offsets(sizes: Sequence[int], device) -> torch.Tensor:
    off = np.concatenate([[0], np.cumsum(np.asarray(sizes, np.int64))[:-1]])
    return torch.as_tensor(off, dtype=torch.long, device=device)


def action_features(cat_idx: torch.Tensor, sizes: Sequence[int],
                    z: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(..., H) categorical bins (policy space) and (..., n_z) pre-tanh view
    z -> (..., sum(sizes) [+ n_z]) float: one one-hot block per head, then
    tanh(z) (bounded, and monotone in the view command the core receives)."""
    d = int(sum(int(s) for s in sizes))
    flat = cat_idx.long() + _offsets(sizes, cat_idx.device)
    oh = torch.zeros(cat_idx.shape[:-1] + (d,), device=cat_idx.device)
    oh.scatter_(-1, flat, 1.0)
    if z is None:
        return oh
    return torch.cat([oh, torch.tanh(z.float())], dim=-1)


def geometric_offsets(n: int, gamma: float, gen: torch.Generator,
                      device) -> torch.Tensor:
    """n draws of k ~ Geometric(1 - gamma) on {1, 2, ...}: P(k) =
    gamma^(k-1) (1 - gamma), mean 1 / (1 - gamma). Inverse CDF off one
    uniform per draw, so it runs on the dedicated generator."""
    u = torch.rand(n, generator=gen, device=device,
                   dtype=torch.float64).clamp_(1e-300, 1.0 - 1e-16)
    k = 1.0 + torch.floor(torch.log(u) / math.log(float(gamma)))
    return k.clamp_(1.0, 1e12).long()


def sample_actions(padded_cat: torch.Tensor, gen: torch.Generator, k: int,
                   cat_temp=None, mu: Optional[torch.Tensor] = None,
                   log_std: Optional[torch.Tensor] = None):
    """k draws from the behaviour distribution the rollout sampled from:
    Gumbel-argmax on the padded categorical logits divided by ``cat_temp``
    (the trainer's _temper_logits) and z = mu + exp(log_std) * eps with the
    already-tempered log sigma (its _temper_log_std). -> (idx (k, R, H)
    long, z (k, R, n_z) or None). The same law as sample_padded /
    sample_view, drawn off ``gen`` instead of the global stream."""
    p = padded_cat.float()
    if cat_temp is not None:
        p = p / cat_temp
    u = torch.rand((k,) + tuple(p.shape), generator=gen,
                   device=p.device).clamp_min_(1e-20)
    idx = (p.unsqueeze(0) - torch.log(-torch.log(u))).argmax(-1)
    if mu is None:
        return idx, None
    eps = torch.randn((k,) + tuple(mu.shape), generator=gen, device=mu.device)
    z = mu.float().unsqueeze(0) + log_std.float().exp() * eps
    return idx, z


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    m = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                      nn.Linear(hidden, hidden), nn.ReLU(),
                      nn.Linear(hidden, d_out))
    for lin in m:
        if isinstance(lin, nn.Linear):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
    return m


class ContrastiveCritic(nn.Module):
    """phi(s, a) and psi(g) into one ``repr_dim`` space; the score is their
    inner product (``dot``, SGCRL) or minus their L2 distance (``l2``,
    CPPO). Representations are NOT normalised (SGCRL)."""

    def __init__(self, sa_dim: int, g_dim: int = 3, repr_dim: int = 64,
                 hidden: int = 256, kind: str = "dot"):
        super().__init__()
        if kind not in ("dot", "l2"):
            raise ValueError(f"critic kind must be dot or l2, got {kind!r}")
        self.kind = kind
        self.sa_dim, self.g_dim = int(sa_dim), int(g_dim)
        self.phi = _mlp(self.sa_dim, int(hidden), int(repr_dim))
        self.psi = _mlp(self.g_dim, int(hidden), int(repr_dim))

    def pair_logits(self, sa: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """(B, d), (B', d) -> (B, B') scores of every anchor with every goal."""
        if self.kind == "dot":
            return sa @ g.t()
        d2 = ((sa * sa).sum(1, keepdim=True) + (g * g).sum(1)[None, :]
              - 2.0 * (sa @ g.t()))
        return -torch.sqrt(d2.clamp_min(0.0) + 1e-8)

    def score(self, sa: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Row-wise (broadcasting) score f(sa, g) -> (...,)."""
        if self.kind == "dot":
            return (sa * g).sum(-1)
        return -torch.sqrt(((sa - g) ** 2).sum(-1) + 1e-8)


def infonce(logits: torch.Tensor, lse_coef: float):
    """Forward InfoNCE over a (B, B) score matrix whose diagonal holds the
    positives, plus lse_coef x mean(logsumexp(logits, 1)^2) (SGCRL Eq. 3, the
    public CRL code's logsumexp_penalty). -> (loss, ce, top-1 accuracy,
    mean logsumexp), the last three detached."""
    b = logits.shape[0]
    lab = torch.arange(b, device=logits.device)
    ce = F.cross_entropy(logits, lab)
    lse = torch.logsumexp(logits, dim=1)
    loss = ce + float(lse_coef) * (lse * lse).mean()
    acc = (logits.argmax(1) == lab).float().mean()
    return loss, ce.detach(), acc.detach(), lse.mean().detach()


class FutureRing:
    """The last ``L`` decisions of every env, in lockstep, on the device.

    Row (slot, env) holds phi's input for the decision (``xs`` the state
    block, ``xa`` the action block), the position at the decision, whether
    the episode ENDED during it, the terminal position when it did, and the
    per-env episode id. Decision number ``g`` (counted from 0 over the whole
    run) lives at slot ``g % L``; ``order`` maps a TIME position p (0 = the
    oldest decision still held) to its slot, and ``next_end[p, env]`` is the
    time position of the last decision of the episode decision p belongs to
    (or _OPEN while that episode is still running at the newest decision).
    """

    def __init__(self, L: int, n_env: int, ds: int, da: int, device):
        self.L, self.N = int(L), int(n_env)
        if self.L < 2 or self.N < 1:
            raise ValueError("FutureRing needs L >= 2 and at least one env")
        self.device = torch.device(device)
        dev = self.device
        self.xs = torch.zeros((self.L, self.N, int(ds)), device=dev)
        self.xa = torch.zeros((self.L, self.N, int(da)), device=dev)
        self.pos = torch.zeros((self.L, self.N, 3), device=dev)
        self.tpos = torch.zeros((self.L, self.N, 3), device=dev)
        self.end = torch.zeros((self.L, self.N), dtype=torch.bool, device=dev)
        self.ep = torch.zeros((self.L, self.N), dtype=torch.long, device=dev)
        self.ep_ctr = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.n = 0                       # decisions written over the run
        self.valid = 0
        self.order = torch.zeros(0, dtype=torch.long, device=dev)
        self.next_end = torch.zeros((0, self.N), dtype=torch.long, device=dev)

    def push(self, xs, xa, pos, end, tpos) -> None:
        """Append T decisions: (T, N, ds), (T, N, da), (T, N, 3), (T, N) bool,
        (T, N, 3). Row t's episode id is the env's counter plus the ends
        BEFORE it in this block, so an episode that ends at t owns row t."""
        T = int(xs.shape[0])
        if T > self.L:
            raise ValueError(f"cannot push {T} decisions into a ring of "
                             f"{self.L} (--crl-history must be >= --n-steps)")
        slots = (self.n + torch.arange(T, device=self.device)) % self.L
        endb = end.to(device=self.device, dtype=torch.bool)
        self.xs[slots] = xs.to(self.device, torch.float32)
        self.xa[slots] = xa.to(self.device, torch.float32)
        self.pos[slots] = pos.to(self.device, torch.float32)
        self.tpos[slots] = tpos.to(self.device, torch.float32)
        self.end[slots] = endb
        el = endb.long()
        cs = el.cumsum(0)
        self.ep[slots] = self.ep_ctr.unsqueeze(0) + cs - el
        self.ep_ctr += cs[-1]
        self.n += T
        self._reindex()

    def _reindex(self) -> None:
        valid = min(self.n, self.L)
        g0 = self.n - valid
        dev = self.device
        self.order = (g0 + torch.arange(valid, device=dev)) % self.L
        end_t = self.end[self.order]                          # (valid, N)
        tix = torch.arange(valid, device=dev).unsqueeze(1).expand(valid,
                                                                  self.N)
        cand = torch.where(end_t, tix, torch.full_like(tix, _OPEN))
        self.next_end = torch.cummin(cand.flip(0), dim=0).values.flip(0)
        self.valid = valid

    def sample_index(self, m: int, gamma: float, gen: torch.Generator):
        """m candidate (anchor, positive) pairs over the whole ring, with the
        resolvability test applied but nothing dropped yet. Returns a dict of
        (m,) tensors: p (anchor time position), e (env), k (offset), q = p+k,
        ne (the anchor episode's last time position, or _OPEN), in_ep (the
        positive is decision q of the same episode), term (q is past the
        episode's last decision, so the positive is its TERMINAL position)
        and ok = in_ep | term."""
        if self.valid < 1:
            raise RuntimeError("FutureRing.sample_index on an empty ring")
        dev = self.device
        p = torch.randint(0, self.valid, (m,), generator=gen, device=dev)
        e = torch.randint(0, self.N, (m,), generator=gen, device=dev)
        k = geometric_offsets(m, gamma, gen, dev)
        q = p + k
        ne = self.next_end[p, e]
        last = self.valid - 1
        in_ep = q <= torch.clamp(ne, max=last)
        term = (q > ne) & (ne <= last)
        return {"p": p, "e": e, "k": k, "q": q, "ne": ne, "in_ep": in_ep,
                "term": term, "ok": in_ep | term}

    def gather(self, ix: dict):
        """Anchors (m, ds + da), positives (m, 3) and the two episode ids
        (m,) of the pairs in ``ix`` (the rows that are not ok carry junk)."""
        last = self.valid - 1
        sp = self.order[ix["p"]]
        sq = self.order[torch.clamp(ix["q"], max=last)]
        sn = self.order[torch.clamp(ix["ne"], max=last)]
        e = ix["e"]
        x = torch.cat([self.xs[sp, e], self.xa[sp, e]], dim=-1)
        g = torch.where(ix["in_ep"].unsqueeze(1), self.pos[sq, e],
                        self.tpos[sn, e])
        ep_a = self.ep[sp, e]
        ep_g = torch.where(ix["in_ep"], self.ep[sq, e], self.ep[sn, e])
        return x, g, ep_a, ep_g

    def sample(self, batch: int, gamma: float, gen: torch.Generator,
               oversample: int = 4):
        """Up to ``batch`` resolvable (anchor, positive) pairs, drawn
        uniformly over the ring's (decision, env) rows. -> (x (B, ds+da),
        g (B, 3), n_candidates, n_resolvable)."""
        m = int(batch) * max(1, int(oversample))
        ix = self.sample_index(m, gamma, gen)
        x, g, _, _ = self.gather(ix)
        sel = torch.nonzero(ix["ok"], as_tuple=False).squeeze(1)[:int(batch)]
        return x[sel], g[sel], m, int(ix["ok"].sum())


class ContrastiveRL:
    """Everything --crl owns: the critic, its own Adam, the future ring, a
    dedicated RNG (anchors, offsets and the V samples never touch the global
    streams the rollout draws from) and the per-iteration statistics."""

    def __init__(self, sa_dim: int, ds: int, n_env: int, frame: MapFrame,
                 goal_xyz, device, *, history: int = 1024,
                 gamma: float = 0.99, repr_dim: int = 64, hidden: int = 256,
                 kind: str = "dot", lr: float = 3e-4, lse: float = 0.01,
                 v_samples: int = 8, seed: int = 0):
        if not 0.0 < float(gamma) < 1.0:
            raise ValueError("--crl-gamma must be in (0, 1)")
        self.device = torch.device(device)
        self.frame = frame
        self.gamma = float(gamma)
        self.lse = float(lse)
        self.k = int(v_samples)
        self.ds, self.sa_dim = int(ds), int(sa_dim)
        # the critic is built on the CPU under its own seed, inside a forked
        # CPU stream: the flag does not move the global RNG anything else
        # (the policy init, the reservoir) reads afterwards
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.critic = ContrastiveCritic(sa_dim, 3, repr_dim, hidden, kind)
        self.critic.to(self.device)
        self.opt = torch.optim.Adam(self.critic.parameters(), lr=float(lr))
        self.ring = FutureRing(history, n_env, ds, sa_dim - ds, self.device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed) + 1)
        self.goal_raw = np.asarray(goal_xyz, np.float64).reshape(3)
        self.goal = torch.as_tensor(frame.norm_np(self.goal_raw),
                                    device=self.device).reshape(1, 3)
        self.n_updates = 0
        self._reset_stats()

    # -- statistics ------------------------------------------------------
    def _reset_stats(self) -> None:
        z = torch.zeros((), device=self.device)
        self._st = {"loss": z.clone(), "acc": z.clone(), "lse": z.clone(),
                    "n": 0, "cand": 0, "ok": 0}
        self._adv = None

    def pop_stats(self) -> dict:
        """The iteration's diagnostics as floats (ONE device sync), then
        reset. Blank-able NaN where a quantity was not produced."""
        st = self._st
        n = st["n"]
        vals = torch.stack([st["loss"], st["acc"], st["lse"]]).tolist()
        out = {"loss": vals[0] / n if n else float("nan"),
               "acc": vals[1] / n if n else float("nan"),
               "lse": vals[2] / n if n else float("nan"),
               "valid": (st["ok"] / st["cand"]) if st["cand"] else float("nan"),
               "updates": n}
        a = self._adv
        if a is not None:
            q, s, sd = torch.stack(a).tolist()
            out.update(q_goal=q, sim_visited=s, adv_std=sd)
        else:
            out.update(q_goal=float("nan"), sim_visited=float("nan"),
                       adv_std=float("nan"))
        self._reset_stats()
        return out

    # -- the history -----------------------------------------------------
    def push(self, xs, xa, pos, end, tpos) -> None:
        self.ring.push(xs, xa, pos, end, tpos)

    # -- the critic update ------------------------------------------------
    def train(self, n_updates: int, batch: int) -> None:
        """``n_updates`` InfoNCE steps of up to ``batch`` resolvable pairs
        each, off the whole ring. A step with fewer than 2 resolvable pairs
        (the very first rollout can be that thin) is skipped."""
        for _ in range(int(n_updates)):
            x, g, m, ok = self.ring.sample(batch, self.gamma, self.gen)
            self._st["cand"] += m
            self._st["ok"] += ok
            if x.shape[0] < 2:
                continue
            with torch.enable_grad():
                sa = self.critic.phi(x)
                gr = self.critic.psi(g)
                loss, ce, acc, lse = infonce(self.critic.pair_logits(sa, gr),
                                             self.lse)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
            self._st["loss"] += ce
            self._st["acc"] += acc
            self._st["lse"] += lse
            self._st["n"] += 1
            self.n_updates += 1

    # -- the CPPO advantage -----------------------------------------------
    @torch.no_grad()
    def advantage(self, xs: torch.Tensor, xa: torch.Tensor,
                  padded_cat: torch.Tensor, sizes: Sequence[int],
                  cat_temp=None, mu: Optional[torch.Tensor] = None,
                  log_std: Optional[torch.Tensor] = None,
                  pos: Optional[torch.Tensor] = None, chunk: int = 16384):
        """A = Q - V over R rollout rows (CPPO Eqs. 6-9, the Monte-Carlo V of
        Eq. 8 because the view heads are continuous).

        xs (R, ds) and xa (R, da) are the ACTED rows (the ring's own input);
        padded_cat (R, H, P) the rollout's padded categorical logits, which
        ``sample_actions`` tempers with ``cat_temp`` exactly as the rollout
        did; mu (R, n_z) and log_std (n_z,) the view Gaussians (log_std
        already tempered), both None on the bins. ``pos`` (R, 3), the
        normalised positions, feeds the sim_visited diagnostic.
        -> (A, Q, V), each (R,)."""
        c = self.critic
        g = c.psi(self.goal)                                     # (1, d)
        R = int(xs.shape[0])
        q = torch.empty(R, device=self.device)
        v = torch.empty(R, device=self.device)
        sim = torch.zeros((), device=self.device)
        k = self.k
        for a0 in range(0, R, int(chunk)):
            a1 = min(R, a0 + int(chunk))
            xs_c = xs[a0:a1].float()
            q[a0:a1] = c.score(c.phi(torch.cat([xs_c, xa[a0:a1].float()],
                                               -1)), g)
            idx, z = sample_actions(
                padded_cat[a0:a1], self.gen, k, cat_temp,
                None if mu is None else mu[a0:a1], log_std)
            xa_k = action_features(idx, sizes, z)               # (k, c, da)
            x_k = torch.cat([xs_c.unsqueeze(0).expand(k, -1, -1), xa_k], -1)
            sk = c.score(c.phi(x_k.reshape(k * (a1 - a0), -1)), g)
            v[a0:a1] = sk.reshape(k, a1 - a0).mean(0)
            if pos is not None:
                sim = sim + c.score(c.psi(pos[a0:a1].float()), g).sum()
        a = q - v
        self._adv = (q.mean(), (sim / max(R, 1)) if pos is not None
                     else torch.full((), float("nan"), device=self.device),
                     a.std() if R > 1 else torch.zeros((), device=self.device))
        return a, q, v

    # -- persistence ------------------------------------------------------
    def state_dict_all(self) -> dict:
        """The critic, its Adam and the RNG. The ring is NOT saved (L x N
        rows are hundreds of MB): a resume refills it in L / T iterations and
        trains the restored critic on it meanwhile."""
        return {"critic": self.critic.state_dict(),
                "opt": self.opt.state_dict(),
                "gen": self.gen.get_state(), "gen_device": self.device.type,
                "n_updates": int(self.n_updates),
                "goal": self.goal_raw.tolist()}

    def load_state_dict_all(self, d: dict) -> None:
        self.critic.load_state_dict(d["critic"])
        self.opt.load_state_dict(d["opt"])
        if d.get("gen") is not None and d.get("gen_device") == self.device.type:
            self.gen.set_state(d["gen"])
        self.n_updates = int(d.get("n_updates", 0))
