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

Any episode end that is NOT the finish - a death or the time cap - charges back the progress the
planner banked in the episode (the race reward's death charge, Grzes 2017's correction of
potential-based shaping under termination, kappa 1, applied to every failed end): the shaped
return of a failed episode is 0 and a finished one keeps all of its progress plus the bonus, so
the planner's optimum is the task's (reach the finish), while the per-primitive progress still
gives dense credit on the way. The bank is the planner's 8th scalar, so the reward stays a
function of what it sees. Measured on blue025 (2026-09-25):
  * death merely paying no progress (prim2d_b025): the planner aimed at the finish (+120-160 u
    planned per primitive) and 40% of its primitives ended in a death - over the void is the
    straight line;
  * the charge on a death only (prim2e_b025): a time-out keeps its bank, so "reach platform 2
    and wait for the clock" beats "try the ramp and sometimes fall" - a local optimum the true
    objective does not have.

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

from .goallearn import (BUDGET_MULT, COMPLETE_FRAC, NOVELTY_CELL_U, PLAN_CLIP, PLAN_MB,
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
                      "plan_r_ok": 0.0, "plan_r_fail": 0.0, "plan_uniform": 0.5}
PRIMLEARN_SEED_OFFSET = 5519
PRIMLEARN_COLS = [
    # does the EXECUTOR do what the planner asks? (the planner's own primitives)
    "exec/complete", "exec/arc_frac", "exec/complete_unif",
    # does the PLANNER advance us toward the finish?
    "plan/adv_plan", "plan/adv_real", "plan/plan_fwd", "plan/death", "plan/ep_prog",
    "plan/ep_prog_start", "plan/finish", "plan/finish_start", "plan/eval_finish",
    # the planner's own learning
    "plan/chosen", "plan/uniform", "plan/closed", "plan/reward", "plan/novelty", "plan/entropy",
    "plan/loss_pi", "plan/loss_v", "plan/kl", "plan/updates", "plan/cover"]
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

    def __init__(self, d_in: int, d_act: int, hidden: int = HIDDEN, mix: int = MIX):
        super().__init__()
        self.d_act, self.mix = int(d_act), int(mix)
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
        return (self.logits(h), self.mu(h).view(n, self.mix, self.d_act),
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
        self.net = PrimPlannerNet(N_OBS, self.d_act).to(self.device)
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

    def _reset_window(self):
        self.w = {"chosen": 0, "unif": 0, "closed": 0, "complete": 0, "closed_u": 0,
                  "complete_u": 0, "rew": 0.0, "nov": 0.0, "nov_n": 0, "ent": 0.0, "ep": 0,
                  "fin": 0, "ep_start": 0, "fin_start": 0, "arc": 0.0, "adv_plan": 0.0,
                  "adv_real": 0.0, "plan_fwd": 0, "death": 0, "prog": 0.0, "prog_start": 0.0}

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
        comp = act & ~ended & (self.track.arc >= COMPLETE_FRAC * self.track.total_arc())
        tout = act & ~ended & ~comp & (self.elapsed >= self.budget_ticks)
        closed = act & (ended | comp | tout)
        live = ~ended
        if live.any():
            cx, cy, cz = self._cells(pos[live])
            self.cover[cx, cy, cz] = True
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
                dd = e & ~finished[ci] & dec
                # a FAILED end (death or time cap) pays no progress and charges back what this
                # episode banked (kappa 1): progress counts only if it leads to the finish, and a
                # dive toward a finish that lies below cannot out-earn flying there
                pay = np.where(dd, -np.maximum(self.bank[ci], 0.0), prog)
                r = r + float(self.cfg["plan_progress"]) * pay
                self.bank[ci] = np.where(dd, 0.0, np.where(dec, self.bank[ci] + prog,
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
                for j in np.flatnonzero(dec):
                    i = ci[j]
                    self.buf[i].append((self.o_x[i].copy(), self.o_u[i].copy(),
                                        float(self.o_logp[i]), float(self.o_val[i]),
                                        float(r[j]), bool(e[j])))
                w["closed"] += int(dec.sum())
                w["complete"] += int((cm & dec).sum())
                w["arc"] += float(af[dec].sum())
                w["adv_plan"] += float(self.o_dplan[ci][dec].sum())
                w["adv_real"] += float(1000.0 * np.where(died[ci], np.minimum(prog, 0.0),
                                                         prog)[dec].sum())
                w["plan_fwd"] += int((self.o_dplan[ci][dec] > 0.0).sum())
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
                    ch = (-float(self.cfg["plan_progress"]) * max(self.bank[i], 0.0)
                          if not finished[i] else 0.0)
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
        lines = [self.prim.line_of(p[j], v[j], float(y[j]), nums[j])[0][:L_MAX]
                 for j in range(len(idx))]
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
        a_plan = rate(w["adv_plan"], w["closed"])
        a_real = rate(w["adv_real"], w["closed"])
        fwd = rate(w["plan_fwd"], w["closed"])
        death = rate(w["death"], w["closed"])
        prog = rate(w["prog"], w["ep"])
        prog_s = rate(w["prog_start"], w["ep_start"])
        rew = rate(w["rew"], w["closed"])
        nov = rate(w["nov"], w["closed"])
        fin = rate(w["fin"], w["ep"])
        fin_s = rate(w["fin_start"], w["ep_start"])
        ent = rate(w["ent"], w["chosen"])
        cov = int(self.cover.sum())
        evf = (ev[0] / ev[1]) if (ev and ev[1]) else float("nan")
        row = [f(cmpl, 4), f(arc, 4), f(cmpl_u, 4),
               f(a_plan, 1), f(a_real, 1), f(fwd, 4), f(death, 4), f(prog, 4), f(prog_s, 4),
               f(fin, 4), f(fin_s, 4), f(evf, 4),
               w["chosen"], w["unif"], w["closed"], f(rew, 4), f(nov, 4), f(ent, 4),
               (f(u["loss_pi"], 5) if u else ""), (f(u["loss_v"], 5) if u else ""),
               (f(u["kl"], 6) if u else ""), self.updates, cov]

        def pc(v):
            return f"{v:.1%}" if v == v else "-"

        def un(v):
            return f"{v:+,.0f}u" if v == v else "-"
        txt = (f"  EXEC cmpl {pc(cmpl)} arc {pc(arc)} (uniform {pc(cmpl_u)}/{w['closed_u']})"
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
        return txt, row

    # ------------------------------------------------------------------ checkpoint
    def state_dict_all(self) -> dict:
        return {"primlearn": True, "cfg": dict(self.cfg), "d_act": self.d_act,
                "obs": {"n_az": N_AZ, "elevs": list(ELEVS), "ray_u": RAY_U, "n_scal": N_SCAL,
                        "mix": MIX, "hidden": HIDDEN},
                "net": self.net.state_dict(), "opt": self.opt.state_dict(),
                "counts": self.nov_count.copy(), "cover": self.cover.copy(),
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
        self.updates = int(sd.get("updates", 0))

    def eval_hooks(self, core, ev: dict, *, line=None, graph=None, finish_radius=None):
        """The greedy eval on ``core`` (graph = the map's FinishRef: a held-out map's own finish)."""
        fin = getattr(graph, "finish_center", None)
        return make_primlearn_hooks(self, core, ev, line=line,
                                    finish=(self.finish if fin is None else fin),
                                    finish_radius=finish_radius)


def make_primlearn_hooks(planner: PrimLearnedPlanner, core, ev: dict, *, line=None, finish=None,
                         finish_radius=None):
    """(episode_meta, on_tick) for record_rollout on a core whose env 0 is recorded - the learned
    primitive planner's GREEDY eval (the heaviest component's mean), shared by the trainer and
    tools/record_ckpt.py. From wherever the core spawned env 0 the goal is the finish box (the
    core's own test); a primitive closes like in training and the next one is chosen at the next
    executor decision (t + 1) % act_every == 0."""
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
    st = {"elapsed": 0, "active": False, "need": False, "ep": 0, "bank": 0.0, "d0": 0.0}

    def _issue():
        sv = core.states_view
        o = sv["origin"][0:1].astype(np.float64)
        v = sv["velocity"][0:1].astype(np.float64)
        y = np.array([float(sv["yaw"][0])])
        _x, u, _lp, _v, _e = P.choose(caster, o, v, y, finish=fin, greedy=True,
                                      bank=np.array([st["bank"]]))
        st["d0"] = float(np.linalg.norm(o[0] - fin))
        nums = squash(P.prim, u)[0]
        ln = P.prim.line_of(o[0], v[0], float(y[0]), nums)[0][:L_MAX]
        trk.set_lines(np.array([0]), [ln])
        if line is not None:
            line.set_lines(np.array([0]), [ln])
        st.update(elapsed=0, active=True, need=False)
        ev["plans"] += 1
        ev["plan_log"].append({"ep": int(st["ep"]), "tick": int(ev["tick"]),
                               "numbers": [round(float(z), 1) for z in nums],
                               "anchor": [float(z) for z in o[0]],
                               "line": [[float(z) for z in q] for q in ln]})
        return ln, nums

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
            if bool(done[0]) and bool(np.asarray(core.goal_hits)[0]):
                ev["succ"] += 1
                ev["ticks"].append(t - ev["t0"])
            if st["active"]:
                ev["closed"] += 1
            st.update(active=False, need=False)      # episode_meta plans the next episode
            ev["t0"] = t + 1
            return
        if st["active"]:
            trk.advance(core.states_view["origin"][0:1].astype(np.float32))
            st["elapsed"] += 1
            comp = bool(trk.arc[0] >= COMPLETE_FRAC * trk.total_arc()[0])
            if comp or st["elapsed"] >= P.budget_ticks:
                ev["closed"] += 1
                ev["complete"] += int(comp)
                o1 = core.states_view["origin"][0].astype(np.float64)
                st["bank"] += (st["d0"] - float(np.linalg.norm(o1 - fin))) / 1000.0
                st.update(active=False, need=True)
        if st["need"] and (t + 1) % K == 0:
            _issue()

    return episode_meta, on_tick
