"""goalprimplan.py - the LEARNED primitive planner (``--goal-planner primlearn``, step 2).

Whenever an env's primitive closes - completed (arc >= COMPLETE_FRAC of its length), out of time
(BUDGET_MULT x its duration) or its episode ended - the planner picks the next one at the next
executor decision: a sample from a MIXTURE of Gaussians over the primitive's numbers
(surfgym/goalprim.py: the sideways and the vertical turn rate at K knots), tanh-squashed into
goalprim's ranges. The executor follows it through the fan exactly as in step 1 and keeps training.
No search: one primitive, then plan again.

What the planner sees - egocentric, no map, no top-down view:
  * N_RAYS point traces from the agent (the simulator's own trace) at N_AZ azimuths around its
    direction of motion and len(ELEVS) elevations, RAY_U long, log-scaled into [0, 1] - computed
    from the simulator state, so the trainer, the eval and a recording on a new map get exactly
    the same input;
  * the finish as a unit vector in the frame of the motion (forward, left, up) + log distance;
  * the velocity (speed, horizontal speed, vertical speed).
A share ``plan_uniform`` of episodes starts with step 1's uniform draw instead of a planner
decision: the executor keeps practising the whole primitive space, not only what the planner
currently likes.

Reward per primitive (the planner's step): progress toward the finish (Euclidean, per 1000 u)
x plan_progress, + plan_r_ok / plan_r_fail for completed / not,
+ plan_finish_bonus when the episode finishes, + plan_novelty / sqrt(n) over 128 u cells of the
primitive's END (global counts over the map's box; none on a death). PPO over each env's chain of
primitives (goallearn.plan_gae: a semi-MDP step per primitive).

The progress term is EXACT potential-based shaping (Ng 1999) with every terminal at potential 0
(Grzes 2017): the potential is the progress banked in the episode, Phi = (d_spawn - d) / 1000,
a primitive that closes alive pays gamma * Phi(s') - Phi(s) with the planner's own gamma (0.95 per
primitive, goallearn.PLAN_GAMMA), and ANY episode end - death, time cap or the finish - pays
-Phi(s). The discounted shaping of every trajectory telescopes to exactly 0, so the planner's
objective is the finish bonus reached as early as possible (plus the exploration terms), while
progress still gives dense credit; holding banked progress costs (1 - gamma) * Phi per primitive,
so waiting for the clock is never better than trying. The bank is the planner's 8th scalar, so the
reward stays a function of what it sees. Measured on the way (2026-09-25):
  * death merely paying no progress (prim2d_b025): the planner aimed at the finish (+120-160 u
    planned per primitive) and 40% of its primitives ended in a death - over the void is the
    straight line;
  * the charge on a death only (prim2e_b025): a time-out keeps its bank, so "reach platform 2
    and wait for the clock" beats "try the ramp and sometimes fall" - a local optimum the true
    objective does not have;
  * the charge on every failed end but undiscounted shaping (prim2f_b025, p2_b050): a refund
    that comes LATER is discounted more, so bouncing safely until the cap beats trying the move
    that might fail - blue050's greedy agent surfed the bottom row back and forth for 25 s
    instead of turning up the column. Exact shaping removes that bias.

plan_r_ok and plan_r_fail are both 0: the planner is paid for the TASK only (progress, the finish,
new places), and an infeasible primitive costs what it costs - time and the progress it did not
make. Both per-primitive terms were measured to break it (2026-09-25):
  * a completion that PAYS is a farm: with +0.5 (prim2_b025) the planner collapsed in 40 updates
    (entropy 5.2 -> 0.4) onto primitives the executor completes 83% of the time and that go
    nowhere - 91% of the episodes ran out their 20 s, none finished;
  * a failure that COSTS with a free death is a suicide incentive: with -0.5 (prim2b_b025) a
    stuck agent's future is a stream of -0.5s and one dive ends it, so the greedy planner's
    second primitive in every recorded episode was a steep turning dive off the platform.

A primitive the executor cannot fly from here (a climb at walking speed) fails, earns r_fail and
no progress, so the planner learns FEASIBILITY from its own reward - the capability term of the
factorised distribution is learned, never hand-written.
"""
from __future__ import annotations

import ctypes
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .goallearn import (BUDGET_MULT, COMPLETE_FRAC, NOVELTY_CELL_U, PLAN_CLIP, PLAN_GAMMA,
                        PLAN_MB,
                        PLAN_VF, plan_gae)

N_AZ = 8
ELEVS = (-30.0, 0.0, 20.0)            # deg above the horizontal
N_RAYS = N_AZ * len(ELEVS)
RAY_U = 4000.0
RAY_LOG_U = 64.0                      # log1p(d / 64) / log1p(4000 / 64): 0 at contact .. 1
N_SCAL = 8
N_OBS = N_RAYS + N_SCAL
MIX = 3
HIDDEN = 256
L_MAX = 128                           # line vertices: 2 s at 8,000 u/s over 128 u spacing
PRIMLEARN_DEFAULTS = {"plan_lr": 3e-4, "plan_ent": 0.01, "plan_batch": 2048, "plan_epochs": 4,
                      "plan_novelty": 0.5, "plan_progress": 1.0, "plan_finish_bonus": 10.0,
                      "plan_r_ok": 0.0, "plan_r_fail": 0.0, "plan_uniform": 0.5,
                      "plan_obey": 0, "plan_cover": 0.0, "plan_shaping": "pbrs",
                      "plan_mu_bound": 0.0}
COVER_MAX_BITS = 400_000_000   # --plan-cover's per-env visited bitmap (n_envs x cells) budget
PRIMLEARN_SEED_OFFSET = 5519
PRIMLEARN_COLS = [
    # does the EXECUTOR do what the planner asks? (the planner's own primitives)
    "exec/complete", "exec/arc_frac", "exec/complete_unif",
    # does the PLANNER advance us toward the finish?
    "plan/adv_plan", "plan/adv_real", "plan/plan_fwd", "plan/death", "plan/ep_prog",
    "plan/ep_prog_start", "plan/finish", "plan/finish_start", "plan/eval_finish",
    # the planner's own learning
    "plan/credit_frac", "plan/cover_ep",
    "plan/chosen", "plan/uniform", "plan/closed", "plan/reward", "plan/novelty", "plan/entropy",
    "plan/loss_pi", "plan/loss_v", "plan/kl", "plan/updates", "plan/cover"]
PRIMLEARN_COLS += ["exec/track_strict", "exec/track_lenient"]
# how well the executor flew the primitive, per tick, averaged over the primitive: exp(-e / sigma)
#   strict  - e = distance to where the curve wants the agent AT THIS TICK (time-aligned)
#   lenient - e = distance to the nearest point of the curve (the path, timing ignored)
# 1 = on the curve; the sigma is the strictness knob (u)
TRACK_SIGMA_STRICT = 64.0
TRACK_SIGMA_LENIENT = 256.0
TRACK_PATH_STRIDE = 10          # the lenient path is sampled every 10 ticks of the curve
MAX_GRAD = 0.5


class FinishRef:
    """What the goal system holds as its ``planner`` under --goal-planner primlearn: no graph,
    no targets - only the finish of ITS map (a held-out map carries its own)."""

    primitive = False
    fin = None

    def __init__(self, box):
        self.finish_center = 0.5 * (np.asarray(box["mins"], np.float64)
                                     + np.asarray(box["maxs"], np.float64))
        self.finish_half = 0.5 * float(np.max(np.asarray(box["maxs"], np.float64)
                                              - np.asarray(box["mins"], np.float64)))

    def describe(self) -> str:
        c = self.finish_center
        return f"finish at ({c[0]:,.0f}, {c[1]:,.0f}, {c[2]:,.0f}) (no graph: primitives)"


class RayCaster:
    """Batched point traces through the core's own trace (one ctypes call per ray, the
    argument buffers built once)."""

    def __init__(self, core):
        from .core import SurfTrace
        self.core = core
        self._fn = core._lib.surf_trace
        self._s = (ctypes.c_float * 3)()
        self._e = (ctypes.c_float * 3)()
        self._o = SurfTrace()
        self._ref = ctypes.byref(self._o)
        self._h = ctypes.c_int32(2)

    def fractions(self, starts, ends) -> np.ndarray:
        sim = self.core._handle()
        S = np.asarray(starts, np.float64).tolist()
        E = np.asarray(ends, np.float64).tolist()
        out = np.empty(len(S), np.float32)
        s, e, o, fn, ref, h = self._s, self._e, self._o, self._fn, self._ref, self._h
        for k in range(len(S)):
            s[:] = S[k]
            e[:] = E[k]
            fn(sim, s, e, h, ref)
            out[k] = o.fraction
        return out


def motion_frame(vel, yaw_deg):
    """-> (forward, left) horizontal unit vectors (n, 3): along the horizontal velocity, or the
    view yaw below 50 u/s (goalprim.SPEED_DIR_MIN)."""
    from .goalprim import SPEED_DIR_MIN
    v = np.atleast_2d(np.asarray(vel, np.float64))
    vh = np.hypot(v[:, 0], v[:, 1])
    yaw = np.where(vh >= SPEED_DIR_MIN, np.arctan2(v[:, 1], v[:, 0]),
                   np.radians(np.broadcast_to(np.asarray(yaw_deg, np.float64), (len(v),))))
    z = np.zeros_like(yaw)
    return (np.stack([np.cos(yaw), np.sin(yaw), z], 1),
            np.stack([-np.sin(yaw), np.cos(yaw), z], 1))


_AZ = np.radians(np.arange(N_AZ) * 360.0 / N_AZ)
_EL = np.radians(np.asarray(ELEVS, np.float64))


def observe(caster: RayCaster, pos, vel, yaw_deg, finish, bank=None) -> np.ndarray:
    """-> (n, N_OBS) float32: the depth rays, then the finish / velocity scalars and the
    progress banked in this episode (per 1000 u; what a death would charge back)."""
    p = np.atleast_2d(np.asarray(pos, np.float64))
    v = np.atleast_2d(np.asarray(vel, np.float64))
    n = len(p)
    fwd, left = motion_frame(v, yaw_deg)
    ca, sa = np.cos(_AZ), np.sin(_AZ)
    ce, se = np.cos(_EL), np.sin(_EL)
    # (n, E, A, 3): horizontal part (fwd cos a + left sin a) cos e, vertical sin e
    hor = fwd[:, None, :] * ca[None, :, None] + left[:, None, :] * sa[None, :, None]
    d = hor[:, None, :, :] * ce[None, :, None, None]
    d[..., 2] = se[None, :, None]
    d = d.reshape(n, N_RAYS, 3)
    starts = np.repeat(p, N_RAYS, axis=0)
    ends = starts + RAY_U * d.reshape(-1, 3)
    fr = caster.fractions(starts, ends).reshape(n, N_RAYS).astype(np.float64)
    out = np.zeros((n, N_OBS), np.float32)
    out[:, :N_RAYS] = np.log1p(fr * (RAY_U / RAY_LOG_U)) / math.log1p(RAY_U / RAY_LOG_U)
    g = np.asarray(finish, np.float64).reshape(1, 3) - p
    dist = np.linalg.norm(g, axis=1)
    gu = g / np.maximum(dist, 1e-6)[:, None]
    out[:, N_RAYS + 0] = (gu * fwd).sum(1)
    out[:, N_RAYS + 1] = (gu * left).sum(1)
    out[:, N_RAYS + 2] = gu[:, 2]
    out[:, N_RAYS + 3] = np.log1p(dist / 1000.0)
    out[:, N_RAYS + 4] = np.linalg.norm(v, axis=1) / 1000.0
    out[:, N_RAYS + 5] = np.hypot(v[:, 0], v[:, 1]) / 1000.0
    out[:, N_RAYS + 6] = v[:, 2] / 1000.0
    if bank is not None:
        out[:, N_RAYS + 7] = np.clip(np.asarray(bank, np.float64), -5.0, 5.0)
    return out


class PrimPlannerNet(nn.Module):
    """observation -> mixture (logits (n, M), means (n, M, D), log stds (n, M, D)) + value (n,)."""

    def __init__(self, d_in: int, d_act: int, hidden: int = HIDDEN, mix: int = MIX,
                 mu_bound: float = 0.0):
        super().__init__()
        self.d_act, self.mix = int(d_act), int(mix)
        # --plan-mu-bound B: the pre-squash means are softly bounded, B * tanh(raw / B). Unbounded,
        # they drift past the action bounds once the advantage pushes a knot outward (blue200's
        # planner pinned at +-180 deg/s sideways and +90 up), and then even the widest sample
        # (sigma <= e^0.5) lands at the bound: the centre of the action space is never tried
        # again. 0 = unbounded (the default, the network that shipped)
        self.mu_bound = float(mu_bound)
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.logits = nn.Linear(hidden, self.mix)
        self.mu = nn.Linear(hidden, self.mix * self.d_act)
        self.log_std = nn.Linear(hidden, self.mix * self.d_act)
        self.v = nn.Linear(hidden, 1)
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        for m in (self.logits, self.mu, self.log_std):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.zeros_(m.bias)
        # the components start apart - sideways rates left / straight / right (the first knot
        # block is the sideways function) - so a fork need not wait for symmetry breaking
        with torch.no_grad():
            k = self.d_act // 2
            b = self.mu.bias.view(self.mix, self.d_act)
            for j, s in enumerate(np.linspace(0.6, -0.6, self.mix)):
                b[j, :k] = float(s)
            self.log_std.bias.fill_(math.log(0.5))
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, x):
        h = self.body(x)
        n = x.shape[0]
        mu = self.mu(h).view(n, self.mix, self.d_act)
        if self.mu_bound > 0.0:
            mu = self.mu_bound * torch.tanh(mu / self.mu_bound)
        return (self.logits(h), mu,
                self.log_std(h).view(n, self.mix, self.d_act).clamp(-3.0, 0.5),
                self.v(h).squeeze(-1))


def mix_logp(logits, mu, log_std, u):
    """log density of pre-squash samples ``u`` (n, D) under the mixture. The tanh Jacobian is the
    same under the old and the new policy, so it cancels in PPO's ratio and is left out."""
    lw = F.log_softmax(logits.float(), dim=-1)
    ls = log_std.float()
    z = (u.float().unsqueeze(1) - mu.float()) / ls.exp()
    comp = (-0.5 * z * z - ls - 0.5 * math.log(2.0 * math.pi)).sum(-1)
    return torch.logsumexp(lw + comp, dim=-1)


def mix_entropy(logits, log_std):
    """An entropy proxy: the component choice's entropy + the weighted Gaussians' entropies."""
    lw = F.log_softmax(logits.float(), dim=-1)
    w = lw.exp()
    hg = (log_std.float() + 0.5 * math.log(2.0 * math.pi * math.e)).sum(-1)
    return -(w * lw).sum(-1) + (w * hg).sum(-1)


def mix_sample(logits, mu, log_std, gen, greedy: bool = False):
    """-> pre-squash samples (n, D); greedy = the heaviest component's mean."""
    rows = torch.arange(mu.shape[0], device=mu.device)
    if greedy:
        return mu[rows, logits.argmax(-1)].float()
    j = torch.multinomial(F.softmax(logits.float(), dim=-1), 1, generator=gen).squeeze(1)
    m = mu[rows, j].float()
    s = log_std[rows, j].float().exp()
    return m + s * torch.randn(m.shape, generator=gen, device=m.device)


def squash(prim, u) -> np.ndarray:
    """pre-squash (n, D) -> the primitive's numbers in deg/s (goalprim's ranges: sideways
    +-side, vertical -down .. +up)."""
    a = np.tanh(np.asarray(u, np.float64))
    k = prim.knots
    out = np.empty_like(a)
    out[:, :k] = a[:, :k] * prim.side
    vv = a[:, k:2 * k]
    out[:, k:2 * k] = np.where(vv >= 0.0, vv * prim.up, vv * prim.down)
    return out


class PrimLearnedPlanner:
    """The network, its optimizer and PPO, the fleet's open primitives, the global end-cell
    counts and the diagnostics. ``prim`` is step 1's PrimitivePlanner (the ranges, the curve, the
    uniform draw); ``core`` the training core (the rays); ``finish`` the finish centre."""

    jump = False
    primlearn = True
    eval_label = "prim planner greedy"

    def __init__(self, prim, core, n_envs: int, device, *, finish, bounds, start_pts=None,
                 tick_ms: float = 10.0, act_every: int = 1, corridor: float = 192.0,
                 cfg: Optional[dict] = None, seed: int = 0):
        from .goalarc import MultiArcProgress
        self.prim, self.core = prim, core
        self.caster = RayCaster(core)
        self.device = torch.device(device)
        self.cfg = dict(PRIMLEARN_DEFAULTS)
        for k, v in (cfg or {}).items():
            if v is not None:
                self.cfg[k] = v
        self.n = int(n_envs)
        self.d_act = int(prim.n_numbers)
        self.act_every = max(1, int(act_every))
        self.net = PrimPlannerNet(N_OBS, self.d_act,
                                  mu_bound=float(self.cfg.get("plan_mu_bound") or 0.0)
                                  ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg["plan_lr"]),
                                    eps=1e-5)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))
        self.rng = np.random.default_rng(int(seed))
        self.finish = np.asarray(finish, np.float64).reshape(3)
        self.start_pts = (None if start_pts is None else
                          np.atleast_2d(np.asarray(start_pts, np.float64)))
        self.corridor = float(corridor)
        self.track = MultiArcProgress(self.n, l_max=L_MAX, spacing=prim.spacing,
                                      corridor=self.corridor, window=16)
        self.active = np.zeros(self.n, bool)
        self.need = np.ones(self.n, bool)
        self.fresh = np.ones(self.n, bool)
        self.decided = np.zeros(self.n, bool)       # the open primitive was the planner's
        self.from_start = np.zeros(self.n, bool)
        self.elapsed = np.zeros(self.n, np.int64)
        self.set_tick_ms(tick_ms)
        self.o_x = np.zeros((self.n, N_OBS), np.float32)
        self.o_u = np.zeros((self.n, self.d_act), np.float32)
        self.o_logp = np.zeros(self.n, np.float32)
        self.o_val = np.zeros(self.n, np.float32)
        self.o_d0 = np.zeros(self.n, np.float64)
        self.o_dplan = np.zeros(self.n, np.float64)     # progress the primitive PROMISES, u
        # the open primitive's raw curve (one point per tick) and its path samples, and the
        # running sums of the two tracking scores
        self.ncurve = int(round(prim.secs / 0.01)) + 1
        self.o_curve = np.zeros((self.n, self.ncurve, 3), np.float32)
        self.npath = (self.ncurve - 1) // TRACK_PATH_STRIDE + 1
        self.o_path = np.zeros((self.n, self.npath, 3), np.float32)
        self.tr_s = np.zeros(self.n, np.float64)
        self.tr_l = np.zeros(self.n, np.float64)
        self.tr_n = np.zeros(self.n, np.int64)
        # per-episode progress toward the finish: distance at the spawn, the least reached alive
        self.d_spawn = np.ones(self.n, np.float64)
        self.d_min = np.ones(self.n, np.float64)
        # progress paid to the planner in this episode (per 1000 u) - a death charges it back
        self.bank = np.zeros(self.n, np.float64)
        self.buf = [[] for _ in range(self.n)]
        mins, maxs = (np.asarray(b, np.float64).reshape(3) for b in bounds)
        self.nov_mins = mins
        self.nov_shape = tuple(int(v) for v in
                               np.maximum(1, np.ceil((maxs - mins) / NOVELTY_CELL_U)))
        self.nov_count = np.zeros(self.nov_shape, np.int32)
        self.cover = np.zeros(self.nov_shape, bool)
        # --plan-cover C (episodic x count-based coverage): the 128 u cells each env has visited
        # ALIVE in its current episode; a planner primitive that ends alive earns, for each cell
        # it added to its episode, C / sqrt(1 + N) where N = how many episodes covered that cell
        # before. The count makes it FRONTIER-seeking: with the episodic part alone (cov_b050,
        # 2026-09-25) the planner farmed the big safe start platform - 7,000 u of path, 200 u of
        # progress, 30 s alive - because every episode re-covers it; a cell thousands of
        # episodes covered pays ~0, one only a few reached pays ~C
        self.cov_c = float(self.cfg.get("plan_cover") or 0.0)
        self.ep_seen = None
        if self.cov_c > 0.0:
            ncell = int(np.prod(self.nov_shape))
            if self.n * ncell > COVER_MAX_BITS:
                raise ValueError(f"--plan-cover: {self.n} envs x {ncell:,} cells of 128 u is "
                                 f"over the {COVER_MAX_BITS:,}-cell budget of the visited map")
            self.ep_seen = np.zeros((self.n, ncell), bool)
        self.ep_cov = np.zeros(self.n, np.int64)           # cells covered this episode
        self.ep_covr = np.zeros(self.n, np.float64)        # their count-weighted value
        self.o_covr = np.zeros(self.n, np.float64)         # ... at the primitive's start
        self.cov_n = (np.zeros(int(np.prod(self.nov_shape)), np.int64)
                      if self.ep_seen is not None else None)   # episodes that covered a cell
        self.updates = 0
        self.last_upd = None
        self.last_eval = None
        self._reset_window()

    # ------------------------------------------------------------------ helpers
    def set_tick_ms(self, tick_ms: float) -> None:
        self.tick_ms = float(tick_ms)
        self.budget_ticks = int(math.ceil(BUDGET_MULT * self.prim.secs * 1000.0 / self.tick_ms))

    def _cells(self, pos):
        k = np.floor((np.atleast_2d(pos) - self.nov_mins[None, :])
                     / NOVELTY_CELL_U).astype(np.int64)
        for a in range(3):
            np.clip(k[:, a], 0, self.nov_shape[a] - 1, out=k[:, a])
        return k[:, 0], k[:, 1], k[:, 2]

    def _set_curve(self, i: int, pts) -> None:
        """Store env i's primitive curve (one point per tick) and its path samples, padded with
        the end point (a primitive shorter than self.ncurve ticks stays at its end)."""
        pts = np.asarray(pts, np.float32)
        k = min(len(pts), self.ncurve)
        self.o_curve[i, :k] = pts[:k]
        self.o_curve[i, k:] = pts[k - 1]
        self.o_path[i] = self.o_curve[i, ::TRACK_PATH_STRIDE][:self.npath]

    def _reset_window(self):
        self.w = {"chosen": 0, "unif": 0, "closed": 0, "complete": 0, "closed_u": 0,
                  "complete_u": 0, "rew": 0.0, "nov": 0.0, "nov_n": 0, "ent": 0.0, "ep": 0,
                  "fin": 0, "ep_start": 0, "fin_start": 0, "arc": 0.0, "adv_plan": 0.0,
                  "adv_real": 0.0, "plan_fwd": 0, "death": 0, "prog": 0.0, "prog_start": 0.0,
                  "cred": 0.0, "cred_raw": 0.0, "covr": 0.0, "trs": 0.0, "trl": 0.0}

    def describe(self) -> str:
        c = self.cfg
        return (f"planner LEARNED PRIMITIVES (--goal-planner primlearn): a {MIX}-component "
                f"mixture over {self.d_act} numbers (tanh-squashed into the primitive ranges) from "
                f"{N_RAYS} point traces ({N_AZ} azimuths x {len(ELEVS)} elevations around the "
                f"motion, {RAY_U:g} u) + {N_SCAL} scalars (finish in the motion frame, log dist, "
                f"velocity, banked progress); a death charges the bank back; {float(c['plan_uniform']):.0%} of episodes open with a uniform "
                f"primitive; a primitive closes on arc >= {COMPLETE_FRAC:g} (corridor "
                f"{self.corridor:g} u), after {self.budget_ticks} ticks or with its episode; "
                f"reward progress {c['plan_progress']:g} per 1000 u to the finish, "
                f"{c['plan_r_ok']:+g} / {c['plan_r_fail']:+g} completed / not, "
                f"+{c['plan_finish_bonus']:g} finish, novelty {c['plan_novelty']:g}/sqrt(n) over "
                f"{NOVELTY_CELL_U:g} u end cells; PPO lr {c['plan_lr']:g} ent {c['plan_ent']:g} "
                f"batch >= {int(c['plan_batch'])} epochs {int(c['plan_epochs'])}; "
                f"{sum(p.numel() for p in self.net.parameters()):,} params")

    # ------------------------------------------------------------------ the fleet
    def request(self, idx, origins=None) -> None:
        """Envs ``idx`` start a new episode: they get a primitive at the next decision."""
        idx = np.asarray(idx, np.int64).reshape(-1)
        if not len(idx):
            return
        self.active[idx] = False
        self.need[idx] = True
        self.fresh[idx] = True
        self.bank[idx] = 0.0
        self.ep_cov[idx] = 0
        self.ep_covr[idx] = 0.0
        self.o_covr[idx] = 0.0
        if self.ep_seen is not None:
            self.ep_seen[idx] = False
        if origins is not None:
            o = np.atleast_2d(np.asarray(origins, np.float64))
            ds = np.linalg.norm(o - self.finish[None, :], axis=1)
            self.d_spawn[idx] = np.maximum(ds, 1.0)
            self.d_min[idx] = ds
            if self.start_pts is not None:
                d = np.linalg.norm(o[:, None, :] - self.start_pts[None, :, :], axis=2).min(axis=1)
                self.from_start[idx] = d < 1.0

    def on_tick(self, pos, ended, finished, died, term_pos=None) -> None:
        """Per physics tick, after the step: advance every open primitive, close the completed /
        timed-out / ended ones and score the planner's."""
        pos = np.asarray(pos, np.float64)
        ended = np.asarray(ended, bool)
        finished = np.asarray(finished, bool)
        died = np.asarray(died, bool)
        waiting = self.need & ~self.active
        self.track.advance(pos.astype(np.float32))
        act = self.active
        self.elapsed[act] += 1
        ai = np.flatnonzero(act & ~ended)
        if len(ai):
            k = np.minimum(self.elapsed[ai], self.ncurve - 1)
            pa = pos[ai].astype(np.float32)
            e_t = np.linalg.norm(pa - self.o_curve[ai, k], axis=1)
            e_p = np.min(np.linalg.norm(pa[:, None, :] - self.o_path[ai], axis=2), axis=1)
            self.tr_s[ai] += np.exp(-e_t / TRACK_SIGMA_STRICT)
            self.tr_l[ai] += np.exp(-e_p / TRACK_SIGMA_LENIENT)
            self.tr_n[ai] += 1
        comp = act & ~ended & (self.track.arc >= COMPLETE_FRAC * self.track.total_arc())
        tout = act & ~ended & ~comp & (self.elapsed >= self.budget_ticks)
        closed = act & (ended | comp | tout)
        live = ~ended
        if live.any():
            cx, cy, cz = self._cells(pos[live])
            self.cover[cx, cy, cz] = True
            if self.ep_seen is not None:
                li = np.flatnonzero(live)
                kk = (cx * self.nov_shape[1] + cy) * self.nov_shape[2] + cz
                new = ~self.ep_seen[li, kk]
                if new.any():
                    ln, kn = li[new], kk[new]
                    self.ep_seen[ln, kn] = True
                    self.ep_cov[ln] += 1
                    self.ep_covr[ln] += self.cov_c / np.sqrt(1.0 + self.cov_n[kn])
                    np.add.at(self.cov_n, kn, 1)
        # the episode's closest approach to the finish, ALIVE (a death's dive does not count;
        # a finish is distance 0; a time-out's last position counts)
        dn = np.linalg.norm(pos - self.finish[None, :], axis=1)
        np.minimum(self.d_min, np.where(live, dn, np.inf), out=self.d_min)
        if ended.any() and term_pos is not None:
            tout_ep = ended & ~died & ~finished
            if tout_ep.any():
                dt = np.linalg.norm(np.asarray(term_pos, np.float64)[tout_ep]
                                    - self.finish[None, :], axis=1)
                self.d_min[tout_ep] = np.minimum(self.d_min[tout_ep], dt)
        self.d_min[finished] = 0.0
        fb = float(self.cfg["plan_finish_bonus"])
        w = self.w
        if closed.any():
            ci = np.flatnonzero(closed)
            endp = pos[ci].copy()
            e = ended[ci]
            if e.any() and term_pos is not None:
                endp[e] = np.asarray(term_pos, np.float64)[ci[e]]
            dec = self.decided[ci]
            cm = comp[ci]
            w["closed_u"] += int((~dec).sum())
            w["complete_u"] += int((cm & ~dec).sum())
            # how much of the primitive the executor covered (inside the corridor), capped at 1
            af = np.minimum(1.0, self.track.arc[ci]
                            / np.maximum(self.track.total_arc()[ci], 1e-6))
            if dec.any():
                r = np.where(cm, float(self.cfg["plan_r_ok"]), float(self.cfg["plan_r_fail"]))
                r = r + fb * finished[ci]
                d1 = np.linalg.norm(endp - self.finish[None, :], axis=1)
                prog = (self.o_d0[ci] - d1) / 1000.0
                # --plan-obey (the user's "min", Dayan & Hinton's managers that learn only when
                # obeyed): forward progress is credited in proportion to the share of the
                # primitive the executor actually flew, f = min(1, arc covered / 0.9); backward
                # progress counts in full. credit = min(p, f * p): never more than plain
                # progress, so it cannot be farmed, and no reward cycle opens
                if int(self.cfg.get("plan_obey") or 0):
                    fo = np.minimum(1.0, af / COMPLETE_FRAC)
                    cred = np.where(prog > 0.0, fo * prog, prog)
                else:
                    cred = prog
                if str(self.cfg.get("plan_shaping") or "pbrs") == "refund":
                    # --plan-shaping refund (2026-09-25 06:50-07:10 default): progress paid as
                    # it comes, the bank refunded at a FAILED end (death, cap) and kept at the
                    # finish - undiscounted, so it keeps a mild forward pull (a later refund is
                    # discounted more) at the price of a procrastination bias
                    dd = e & ~finished[ci] & dec
                    pay = np.where(dd, -np.maximum(self.bank[ci], 0.0), cred)
                else:
                    dd = e & dec
                    # exact potential-based shaping, Phi = the banked progress, Phi(terminal) =
                    # 0: alive gamma * Phi(s') - Phi(s); ANY episode end (death, cap, finish)
                    # -Phi(s)
                    pay = np.where(dd, -self.bank[ci],
                                   PLAN_GAMMA * (self.bank[ci] + cred) - self.bank[ci])
                r = r + float(self.cfg["plan_progress"]) * pay
                self.bank[ci] = np.where(dd, 0.0, np.where(dec, self.bank[ci] + cred,
                                                           self.bank[ci]))
                nov = np.zeros(len(ci), np.float64)
                alive = dec & ~died[ci]
                if alive.any():
                    # novelty of the END cell, counted over the whole fleet; several envs ending
                    # in one cell on this tick see the counts one after the other
                    ax, ay, az = self._cells(endp[alive])
                    keys = (ax * self.nov_shape[1] + ay) * self.nov_shape[2] + az
                    flat = self.nov_count.reshape(-1)
                    order = np.argsort(keys, kind="stable")
                    ks = keys[order]
                    first = np.r_[True, ks[1:] != ks[:-1]]
                    grp = np.cumsum(first) - 1
                    rank = np.arange(len(ks)) - np.flatnonzero(first)[grp]
                    nv = np.empty(len(ks), np.float64)
                    nv[order] = float(self.cfg["plan_novelty"]) / np.sqrt(flat[ks] + rank + 1.0)
                    np.add.at(flat, ks, 1)
                    nov[alive] = nv
                r = r + nov
                if self.ep_seen is not None:
                    # episodic coverage: the cells this primitive added, if it ended alive (a
                    # fall through the void covers cells too, and pays nothing)
                    cv = np.where(dec & ~died[ci],
                                  self.ep_covr[ci] - self.o_covr[ci], 0.0)
                    r = r + cv
                    w["covr"] += float(cv[dec].sum())
                for j in np.flatnonzero(dec):
                    i = ci[j]
                    self.buf[i].append((self.o_x[i].copy(), self.o_u[i].copy(),
                                        float(self.o_logp[i]), float(self.o_val[i]),
                                        float(r[j]), bool(e[j])))
                w["closed"] += int(dec.sum())
                w["complete"] += int((cm & dec).sum())
                w["arc"] += float(af[dec].sum())
                nn_ = np.maximum(self.tr_n[ci], 1)
                w["trs"] += float((self.tr_s[ci] / nn_)[dec].sum())
                w["trl"] += float((self.tr_l[ci] / nn_)[dec].sum())
                w["adv_plan"] += float(self.o_dplan[ci][dec].sum())
                w["adv_real"] += float(1000.0 * np.where(died[ci], np.minimum(prog, 0.0),
                                                         prog)[dec].sum())
                w["plan_fwd"] += int((self.o_dplan[ci][dec] > 0.0).sum())
                pos = dec & ~dd & (prog > 0.0)
                w["cred"] += float(cred[pos].sum())
                w["cred_raw"] += float(prog[pos].sum())
                w["death"] += int((died[ci] & dec).sum())
                w["nov"] += float(nov[alive].sum())
                w["nov_n"] += int(alive.sum())
                w["rew"] += float(r[dec].sum())
            self.active[ci] = False
            self.need[ci] = True
        self.need[ended] = True
        late = waiting & ended
        if late.any():
            # the episode ended while the env waited for its next decision: its last primitive's
            # transition is terminal (and carries the finish)
            for i in np.flatnonzero(late):
                b = self.buf[i]
                if b and not b[-1][5]:
                    t = b[-1]
                    # the transition's next state turned out TERMINAL (Phi 0): take back the
                    # gamma * Phi(s') it was paid
                    if str(self.cfg.get("plan_shaping") or "pbrs") == "refund":
                        ch = (-float(self.cfg["plan_progress"]) * max(self.bank[i], 0.0)
                              if not finished[i] else 0.0)
                    else:
                        ch = -float(self.cfg["plan_progress"]) * PLAN_GAMMA * self.bank[i]
                    b[-1] = t[:4] + (t[4] + fb * float(finished[i]) + ch, True)
            self.bank[late] = 0.0
        if ended.any():
            ei = np.flatnonzero(ended)
            w["ep"] += len(ei)
            w["fin"] += int(finished[ei].sum())
            fs = self.from_start[ei]
            w["ep_start"] += int(fs.sum())
            w["fin_start"] += int((finished[ei] & fs).sum())
            pr = np.clip((self.d_spawn[ei] - self.d_min[ei]) / self.d_spawn[ei], 0.0, 1.0)
            w["prog"] += float(pr.sum())
            w["prog_start"] += float(pr[fs].sum())

    def choose(self, caster, pos, vel, yaw_deg, finish=None, greedy: bool = False, bank=None):
        """-> (x, u, logp, value, entropy) for these states (numpy)."""
        x = observe(caster, pos, vel, yaw_deg, self.finish if finish is None else finish, bank)
        with torch.no_grad():
            lg, mu, ls, v = self.net(torch.as_tensor(x, device=self.device))
            u = mix_sample(lg, mu, ls, self.gen, greedy=greedy)
            lp = mix_logp(lg, mu, ls, u)
            ent = mix_entropy(lg, ls)
        return (x, u.cpu().numpy().astype(np.float32), lp.cpu().numpy(),
                v.float().cpu().numpy(), ent.cpu().numpy())

    def plan(self, pos, vel, yaw_deg):
        """At an executor decision boundary: a primitive for every env that needs one ->
        (idx, lines, fresh)."""
        idx = np.flatnonzero(self.need)
        if not len(idx):
            return idx, [], np.zeros(0, bool)
        p = np.asarray(pos, np.float64)[idx]
        v = np.asarray(vel, np.float64)[idx]
        y = np.asarray(yaw_deg, np.float64)[idx]
        fresh = self.fresh[idx].copy()
        unif = fresh & (self.rng.random(len(idx)) < float(self.cfg["plan_uniform"]))
        dec = ~unif
        nums = np.zeros((len(idx), self.d_act), np.float64)
        for j in np.flatnonzero(unif):
            nums[j] = self.prim.sample(self.rng)
        if dec.any():
            di = np.flatnonzero(dec)
            x, u, lp, val, ent = self.choose(self.caster, p[di], v[di], y[di],
                                             bank=self.bank[idx[di]])
            nums[di] = squash(self.prim, u)
            rows = idx[di]
            self.o_x[rows], self.o_u[rows] = x, u
            self.o_logp[rows], self.o_val[rows] = lp, val
            self.w["chosen"] += len(di)
            self.w["ent"] += float(ent.sum())
        self.w["unif"] += int(unif.sum())
        self.o_d0[idx] = np.linalg.norm(p - self.finish[None, :], axis=1)
        self.o_covr[idx] = self.ep_covr[idx]
        lines = []
        for j in range(len(idx)):
            ln, pts = self.prim.line_and_curve(p[j], v[j], float(y[j]), nums[j])
            lines.append(ln[:L_MAX])
            self._set_curve(int(idx[j]), pts)
        self.tr_s[idx] = 0.0
        self.tr_l[idx] = 0.0
        self.tr_n[idx] = 0
        ends = np.asarray([ln[-1] for ln in lines], np.float64)
        self.o_dplan[idx] = self.o_d0[idx] - np.linalg.norm(ends - self.finish[None, :], axis=1)
        self.track.set_lines(idx, lines)
        self.active[idx] = True
        self.need[idx] = False
        self.elapsed[idx] = 0
        self.decided[idx] = dec
        self.fresh[idx] = False
        return idx, lines, fresh

    # ------------------------------------------------------------------ PPO
    def n_ready(self) -> int:
        return sum(len(b) for b in self.buf)

    def update(self, force: bool = False):
        """PPO over every closed planner primitive once ``plan_batch`` are in (a clipped
        surrogate on the mixture's log density, a value loss, the entropy bonus)."""
        n = self.n_ready()
        if n == 0 or (n < int(self.cfg["plan_batch"]) and not force):
            return None
        xs, us, lps, advs, rets = [], [], [], [], []
        for i, b in enumerate(self.buf):
            if not b:
                continue
            vb = float(self.o_val[i]) if (self.active[i] and self.decided[i]) else 0.0
            adv, ret = plan_gae([t[4] for t in b], [t[3] for t in b], [t[5] for t in b], vb)
            for t in b:
                xs.append(t[0])
                us.append(t[1])
                lps.append(t[2])
            advs.append(adv)
            rets.append(ret)
            b.clear()
        dev = self.device
        x = torch.as_tensor(np.stack(xs), device=dev)
        u = torch.as_tensor(np.stack(us), device=dev)
        old = torch.as_tensor(np.asarray(lps, np.float32), device=dev)
        adv = np.concatenate(advs)
        ret = torch.as_tensor(np.concatenate(rets).astype(np.float32), device=dev)
        adv_n = torch.as_tensor(((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32),
                                device=dev)
        M = len(xs)
        mb = min(max(PLAN_MB, M // 8), M)
        ent_c = float(self.cfg["plan_ent"])
        st = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "n_mb": 0}
        epochs = max(1, int(self.cfg["plan_epochs"]))
        for ep in range(epochs):
            perm = self.rng.permutation(M)
            for s in range(0, M, mb):
                j = torch.as_tensor(perm[s:s + mb], device=dev)
                lg, mu, ls, v = self.net(x[j])
                lp = mix_logp(lg, mu, ls, u[j])
                ratio = torch.exp((lp - old[j]).clamp(-20.0, 20.0))
                a = adv_n[j]
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 1.0 - PLAN_CLIP, 1.0 + PLAN_CLIP) * a).mean()
                vl = ((v.float() - ret[j]) ** 2).mean()
                ent = mix_entropy(lg, ls).mean()
                loss = pg + PLAN_VF * vl - ent_c * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD)
                self.opt.step()
                if ep == epochs - 1:
                    st["pi"] += float(pg.detach())
                    st["v"] += float(vl.detach())
                    st["ent"] += float(ent.detach())
                    st["kl"] += float((old[j] - lp).mean().detach())
                    st["n_mb"] += 1
        k = max(1, st["n_mb"])
        self.updates += 1
        self.last_upd = {"loss_pi": st["pi"] / k, "loss_v": st["v"] / k,
                         "entropy": st["ent"] / k, "kl": st["kl"] / k, "n": M,
                         "updates": self.updates, "ret_mean": float(ret.mean())}
        return self.last_upd

    def return_weights(self, origins) -> np.ndarray:
        """--plan-return: a reservoir state's respawn weight, 1 / sqrt(1 + N) of the 128 u cell it
        stands in (N = the episodes that covered the cell, --plan-cover's counts): spawns go where
        few episodes have been - Go-Explore's return over the policy's own states."""
        if self.cov_n is None:
            return np.ones(len(origins), np.float64)
        cx, cy, cz = self._cells(np.asarray(origins, np.float64))
        k = (cx * self.nov_shape[1] + cy) * self.nov_shape[2] + cz
        return 1.0 / np.sqrt(1.0 + self.cov_n[k].astype(np.float64))

    # ------------------------------------------------------------------ logging
    def note_and_row(self):
        """-> (log text, progress.csv row for PRIMLEARN_COLS); resets the window."""
        w = self.w
        u, ev = self.last_upd, self.last_eval
        self.last_upd = None
        self.last_eval = None

        def rate(a, b):
            return (a / b) if b else float("nan")

        def f(v, nd):
            return round(float(v), nd) if v == v else ""
        cmpl = rate(w["complete"], w["closed"])
        cmpl_u = rate(w["complete_u"], w["closed_u"])
        arc = rate(w["arc"], w["closed"])
        trs = rate(w["trs"], w["closed"])
        trl = rate(w["trl"], w["closed"])
        a_plan = rate(w["adv_plan"], w["closed"])
        a_real = rate(w["adv_real"], w["closed"])
        fwd = rate(w["plan_fwd"], w["closed"])
        death = rate(w["death"], w["closed"])
        credf = rate(w["cred"], w["cred_raw"])
        covr = rate(w["covr"], w["closed"])
        prog = rate(w["prog"], w["ep"])
        prog_s = rate(w["prog_start"], w["ep_start"])
        rew = rate(w["rew"], w["closed"])
        nov = rate(w["nov"], w["closed"])
        fin = rate(w["fin"], w["ep"])
        fin_s = rate(w["fin_start"], w["ep_start"])
        ent = rate(w["ent"], w["chosen"])
        cov = int(self.cover.sum())
        evf = (ev[0] / ev[1]) if (ev and ev[1]) else float("nan")
        row_tail = [f(trs, 4), f(trl, 4)]
        row = [f(cmpl, 4), f(arc, 4), f(cmpl_u, 4),
               f(a_plan, 1), f(a_real, 1), f(fwd, 4), f(death, 4), f(prog, 4), f(prog_s, 4),
               f(fin, 4), f(fin_s, 4), f(evf, 4), f(credf, 4), f(covr, 4),
               w["chosen"], w["unif"], w["closed"], f(rew, 4), f(nov, 4), f(ent, 4),
               (f(u["loss_pi"], 5) if u else ""), (f(u["loss_v"], 5) if u else ""),
               (f(u["kl"], 6) if u else ""), self.updates, cov]

        def pc(v):
            return f"{v:.1%}" if v == v else "-"

        def un(v):
            return f"{v:+,.0f}u" if v == v else "-"
        txt = (f"  EXEC track {trs:.2f}/{trl:.2f} (strict/lenient) cmpl {pc(cmpl)} arc {pc(arc)}"
               if trs == trs else f"  EXEC cmpl {pc(cmpl)} arc {pc(arc)}")
        txt = (txt + f" (uniform {pc(cmpl_u)}/{w['closed_u']})"
               + f"  PLAN adv {un(a_plan)} planned / {un(a_real)} real, fwd {pc(fwd)}, "
               f"death {pc(death)}, ep prog {pc(prog)} (start {pc(prog_s)}/{w['ep_start']}), "
               f"fin {pc(fin)}/{w['ep']} (start {pc(fin_s)})"
               + f"; {w['chosen']} chosen H " + (f"{ent:.2f}" if ent == ent else "-")
               + (f" r {rew:+.3f} nov {nov:.3f}" if w["closed"] else "")
               + f" cover {cov}"
               + (f" | upd {u['updates']} n {u['n']} pi {u['loss_pi']:+.4f} "
                  f"v {u['loss_v']:.4f} kl {u['kl']:.4f} ret {u['ret_mean']:+.2f}"
                  if u else ""))
        self._reset_window()
        return txt, row + row_tail

    # ------------------------------------------------------------------ checkpoint
    def state_dict_all(self) -> dict:
        return {"primlearn": True, "cfg": dict(self.cfg), "d_act": self.d_act,
                "obs": {"n_az": N_AZ, "elevs": list(ELEVS), "ray_u": RAY_U, "n_scal": N_SCAL,
                        "mix": MIX, "hidden": HIDDEN, "mu_bound": self.net.mu_bound},
                "net": self.net.state_dict(), "opt": self.opt.state_dict(),
                "counts": self.nov_count.copy(), "cover": self.cover.copy(),
                "cov_n": (None if self.cov_n is None else self.cov_n.copy()),
                "updates": self.updates}

    def load_state_dict_all(self, sd: dict) -> None:
        if not sd.get("primlearn"):
            raise ValueError("not a --goal-planner primlearn planner state")
        if int(sd.get("d_act", -1)) != self.d_act:
            raise ValueError(f"the stored planner draws {sd.get('d_act')} numbers, this run "
                             f"{self.d_act} (--prim-knots)")
        self.net.load_state_dict(sd["net"])
        if sd.get("opt") is not None:
            try:
                self.opt.load_state_dict(sd["opt"])
            except (ValueError, KeyError):
                pass
        if tuple(np.shape(sd.get("counts"))) == self.nov_shape:
            self.nov_count[...] = sd["counts"]
            self.cover[...] = sd["cover"]
            if self.cov_n is not None and sd.get("cov_n") is not None \
                    and len(sd["cov_n"]) == len(self.cov_n):
                self.cov_n[...] = sd["cov_n"]
        self.updates = int(sd.get("updates", 0))

    def eval_hooks(self, core, ev: dict, *, line=None, graph=None, finish_radius=None):
        """The greedy eval on ``core`` (graph = the map's FinishRef: a held-out map's own finish)."""
        fin = getattr(graph, "finish_center", None)
        return make_primlearn_hooks(self, core, ev, line=line,
                                    finish=(self.finish if fin is None else fin),
                                    finish_radius=finish_radius)


def make_primlearn_hooks(planner: PrimLearnedPlanner, core, ev: dict, *, line=None, finish=None,
                         finish_radius=None, search=None, override=None):
    """(episode_meta, on_tick) for record_rollout on a core whose env 0 is recorded - the learned
    primitive planner's GREEDY eval (the heaviest component's mean), shared by the trainer and
    tools/record_ckpt.py. From wherever the core spawned env 0 the goal is the finish box (the
    core's own test); a primitive closes like in training and the next one is chosen at the next
    executor decision (t + 1) % act_every == 0. ``search``: a dict whose "s" holds a
    goalsearch.PrimSearch (filled in once the executor wrapper exists) - each choice is then the
    best of its simulated candidates instead of the heaviest mean. ``override`` (an ABLATION of
    how much the executor needs the planner): "straight" = every primitive all-zero numbers, a
    straight line along the motion; "random" = step 1's uniform draw; "frozen" = the first
    primitive is kept for the whole episode (no re-plan)."""
    from .goalarc import MultiArcProgress
    P = planner
    fin = np.asarray(P.finish if finish is None else finish, np.float64).reshape(3)
    caster = RayCaster(core)
    K = P.act_every
    ev.update({"n": 0, "succ": 0, "pending": False, "center": None, "ticks": [], "dists": [],
               "t0": 0, "box": True, "plans": 0, "closed": 0, "complete": 0, "plan_log": [],
               "tick": 0})
    trk = MultiArcProgress(1, l_max=L_MAX, spacing=P.prim.spacing, corridor=P.corridor,
                           window=16)
    st = {"elapsed": 0, "active": False, "need": False, "ep": 0, "bank": 0.0, "d0": 0.0,
          "curve": None, "path": None, "trs": 0.0, "trl": 0.0, "trn": 0}
    ev["track"] = []

    def _close_track():
        """The open primitive's two tracking scores -> its plan_log entry and ev["track"]."""
        if st["curve"] is None or not ev["plan_log"]:
            return
        n_ = max(1, st["trn"])
        s_, l_ = st["trs"] / n_, st["trl"] / n_
        ev["plan_log"][-1]["track"] = [round(s_, 3), round(l_, 3)]
        ev["track"].append((s_, l_))
        st["curve"] = None

    def _issue():
        sv = core.states_view
        o = sv["origin"][0:1].astype(np.float64)
        v = sv["velocity"][0:1].astype(np.float64)
        y = np.array([float(sv["yaw"][0])])
        _x, u, _lp, _v, _e = P.choose(caster, o, v, y, finish=fin, greedy=True,
                                      bank=np.array([st["bank"]]))
        if override is not None:
            # the ablation: the planner's choice replaced (pre-squash u, so the squash below
            # gives the intended numbers; all-zero u = all-zero numbers = straight)
            if override == "straight":
                u = np.zeros_like(u)
            elif override == "random":
                nr = P.prim.sample(ov_rng)
                u = np.arctanh(np.clip(np.concatenate([nr[:P.prim.knots] / P.prim.side,
                                                       np.where(nr[P.prim.knots:] >= 0,
                                                                nr[P.prim.knots:] / max(P.prim.up, 1e-6),
                                                                nr[P.prim.knots:] / max(P.prim.down, 1e-6))]),
                                       -0.999, 0.999))[None, :].astype(np.float32)
        if st.get("pred_end_kind"):
            # --plan-mcts: the previous searched primitive has just closed alive, at the same
            # decision tick the simulation took its end state - how far apart are they?
            ev.setdefault("pred_kind", []).append((st["pred_end_kind"], "alive"))
            if st.get("pred") is not None:
                ev.setdefault("pred_err", []).append(float(np.linalg.norm(o[0] - st["pred"])))
            st["pred_end_kind"] = None
        S = search.get("s") if search is not None else None
        if S is not None:
            # the real env's current core observation: the simulated executors read it first
            # (no neutral tick spent), with the real wrapper's held action / keys / phase
            _ob = getattr(core, "_obs", None)
            ub, sc, info = S.choose(core.get_states()[0:1], fin, np.array([st["bank"]]), _x,
                                    P.gen, obs=None if _ob is None else _ob[0:1].copy())
            u = ub.astype(np.float32)
            ev.setdefault("search", []).append({"best": int(info.get("best",
                                                                     np.argmax(sc[0]))),
                                                "scores": [round(float(z), 3) for z in sc[0]],
                                                "died": int(info["died"][0].sum()),
                                                "finished": int(info["finished"][0].sum()),
                                                "expansions": int(info.get("expansions", 1)),
                                                "depth": int(info.get("depth", 1))})
            # --plan-mcts: where the simulation says the committed primitive closes - compared
            # with where it really closes (the model's fidelity: surf is chaotic, and a tree
            # is only as deep as its simulation stays true)
            st["pred"] = info.get("pred_end")
            st["pred_end_kind"] = ("fin" if info.get("pred_fin") else
                                   "died" if info.get("pred_died") else
                                   "alive" if "pred_end" in info else None)
        st["d0"] = float(np.linalg.norm(o[0] - fin))
        nums = squash(P.prim, u)[0]
        _close_track()
        ln, pts = P.prim.line_and_curve(o[0], v[0], float(y[0]), nums)
        ln = ln[:L_MAX]
        st["curve"] = np.asarray(pts, np.float32)
        st["path"] = st["curve"][::TRACK_PATH_STRIDE]
        st.update(trs=0.0, trl=0.0, trn=0)
        trk.set_lines(np.array([0]), [ln])
        if line is not None:
            line.set_lines(np.array([0]), [ln])
        st.update(elapsed=0, active=True, need=False)
        ev["plans"] += 1
        ev["plan_log"].append({"ep": int(st["ep"]), "tick": int(ev["tick"]),
                               # the episode tick it starts on = the row index in the
                               # trajectory the viewer plays (row 0 is the spawn)
                               "t": int(ev["tick"]) - int(ev["t0"]),
                               "numbers": [round(float(z), 1) for z in nums],
                               "anchor": [float(z) for z in o[0]],
                               "line": [[float(z) for z in q] for q in ln]})
        return ln, nums

    ov_rng = np.random.default_rng(12345)

    def episode_meta(ep):
        st["ep"] = int(ev["n"])
        st["bank"] = 0.0
        ln, nums = _issue()
        ev["n"] += 1
        ev["pending"] = False
        o = core.states_view["origin"][0].astype(np.float64)
        ev["dists"].append(float(np.linalg.norm(fin - o)))
        rad = float(finish_radius) if finish_radius is not None else 192.0
        return {"goal": {"center": [float(z) for z in fin], "radius": rad},
                "line": [[float(z) for z in q] for q in ln],
                "plan": {"planner": "primlearn", "numbers": [round(float(z), 1) for z in nums]}}

    def on_tick(t, states, rewards, done, trunc):
        ev["tick"] = int(t) + 1
        if bool(done[0]) or bool(trunc[0]):
            real_fin = bool(done[0]) and bool(np.asarray(core.goal_hits)[0])
            if real_fin:
                ev["succ"] += 1
                ev["ticks"].append(t - ev["t0"])
            if st.get("pred_end_kind"):
                # the episode ended before the searched primitive's next decision: did the
                # simulation say so?
                ev.setdefault("pred_kind", []).append((st["pred_end_kind"],
                                                       "fin" if real_fin else "died"))
            st["pred_end_kind"] = None
            _close_track()
            if st["active"]:
                ev["closed"] += 1
            st.update(active=False, need=False)      # episode_meta plans the next episode
            ev["t0"] = t + 1
            return
        if st["active"]:
            trk.advance(core.states_view["origin"][0:1].astype(np.float32))
            st["elapsed"] += 1
            if st["curve"] is not None:
                p0 = core.states_view["origin"][0].astype(np.float32)
                c = st["curve"][min(st["elapsed"], len(st["curve"]) - 1)]
                st["trs"] += float(np.exp(-np.linalg.norm(p0 - c) / TRACK_SIGMA_STRICT))
                st["trl"] += float(np.exp(-np.min(np.linalg.norm(st["path"] - p0[None, :], axis=1))
                                          / TRACK_SIGMA_LENIENT))
                st["trn"] += 1
            comp = bool(trk.arc[0] >= COMPLETE_FRAC * trk.total_arc()[0])
            if comp or st["elapsed"] >= P.budget_ticks:
                ev["closed"] += 1
                ev["complete"] += int(comp)
                o1 = core.states_view["origin"][0].astype(np.float64)
                st["bank"] += (st["d0"] - float(np.linalg.norm(o1 - fin))) / 1000.0
                st.update(active=False, need=True)
        if st["need"] and (t + 1) % K == 0:
            if override == "frozen":
                # the ablation: no re-plan - the executor keeps the episode's first primitive
                st["need"] = False
                return
            _issue()

    def episode_end(ep):
        """Every primitive the planner chose in episode ``ep``, with the episode tick it starts
        on - merged into the trajectory's trailer (surfgym.record.record_rollout), so the viewer
        can show the ACTIVE primitive at each moment; the header's line is only the first."""
        return {"plans": [{"t": int(p["t"]), "numbers": p["numbers"],
                           "track": p.get("track"),
                           "line": [[round(z, 1) for z in q] for q in p["line"]]}
                          for p in ev["plan_log"] if p["ep"] == int(ep)]}

    episode_meta.episode_end = episode_end
    return episode_meta, on_tick
