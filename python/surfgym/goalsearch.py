"""goalsearch.py - SEARCH for the learned primitive planner: simulate candidate primitives with the
real executor on a scratch core, then commit the best one (``--plan-search M``).

The planner proposes M primitives (its mixture's heaviest mean + M-1 samples). Each is simulated
from the env's exact state: the scratch core's slot is teleported to that state, the candidate's
curve goes onto the slot's fan, and the executor's own greedy policy flies it for the primitive's
budget (1.5 x its duration), exactly as it would for real. A candidate is scored on what actually
happened, with the planner's own reward:
  * the finish: + the finish bonus + the progress it made (per 1000 u);
  * a death: minus the progress the episode banked (the failed-end charge);
  * alive at the budget: the progress it made + gamma x the planner's value of the end state
    + the end cell's novelty bonus.
So a candidate is judged by the executor's RESPONSE to it, not by whether it could be followed
literally (the learned planner signals rather than plans, ledger 2026-09-25 06:30), and a
primitive that kills is discarded before anyone dies. Nothing about the map is written here: the
simulator is the model.

Used two ways: at decision time in the greedy eval (``make_primlearn_hooks(..., search=)``) and by
a few SCOUT envs in training, whose committed choices are left out of the planner's PPO (a
selected sample is off-policy) while their reached states feed the respawn reservoir.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

from .goalprimplan import L_MAX, mix_sample, observe, squash, RayCaster

SEARCH_GAMMA = 0.95          # the planner's per-primitive discount (goallearn.PLAN_GAMMA)
NEUTRAL_ACT = (7, 3, 1, 1, 0, 0)     # train_fast.NEUTRAL_ACT: centre view bins, no keys


class PrimSearch:
    """``core``: a scratch SurfCore of ``slots`` envs, built like the eval core (same map, physics,
    teleport-fail, finish box). ``line``: a goals.MultiLine of ``slots`` envs with the training fan
    offsets. ``make_policy(core, line)``: a fresh greedy executor wrapper on that core (the
    trainer's EVAL_GREEDY with ``route=line``). ``planner``: the PrimLearnedPlanner."""

    def __init__(self, core, line, make_policy: Callable, planner, m: int = 8,
                 horizon_ticks: Optional[int] = None):
        self.core, self.line, self.make_policy = core, line, make_policy
        self.P = planner
        self.m = max(2, int(m))
        self.slots = int(core.num_envs)
        if self.slots < self.m:
            raise ValueError(f"--plan-search {self.m}: the scratch core has {self.slots} envs")
        self.horizon = int(horizon_ticks) if horizon_ticks else int(planner.budget_ticks)
        self.caster = RayCaster(core)
        self.calls = 0
        self.sims = 0
        self.deaths_avoided = 0          # candidates that died in simulation
        self.finishes_seen = 0

    def describe(self) -> str:
        return (f"search: {self.m} candidate primitives per decision (the mixture's heaviest "
                f"mean + {self.m - 1} samples), each flown by the greedy executor for "
                f"{self.horizon} ticks from the env's exact state on a {self.slots}-env scratch "
                f"core; scored by the planner's own reward (finish, failed-end charge, progress, "
                f"gamma x V_planner of the end state, end novelty)")

    # ------------------------------------------------------------------ candidates
    def candidates(self, x, gen):
        """(B, N_OBS) planner observations -> (B, M, D) pre-squash candidates: the greedy mean
        first, then M-1 samples of the mixture."""
        P = self.P
        with torch.no_grad():
            xt = torch.as_tensor(x, device=P.device)
            lg, mu, ls, _v = P.net(xt)
            out = [mix_sample(lg, mu, ls, gen, greedy=True)]
            for _ in range(self.m - 1):
                out.append(mix_sample(lg, mu, ls, gen))
        return torch.stack(out, 1).float().cpu().numpy()

    # ------------------------------------------------------------------ simulation
    def evaluate(self, states, finish, bank, cand_u):
        """``states``: (B,) STATE_DTYPE of the real envs; ``cand_u``: (B, M, D) pre-squash
        candidates; ``bank``: (B,) the episodes' banked progress. -> (scores (B, M), info)."""
        P, core = self.P, self.core
        B, M = cand_u.shape[:2]
        fin = np.asarray(finish, np.float64).reshape(3)
        scores = np.zeros((B, M), np.float64)
        died_all = np.zeros((B, M), bool)
        fin_all = np.zeros((B, M), bool)
        per = self.slots // M                  # real envs simulated per round
        for b0 in range(0, B, per):
            bb = list(range(b0, min(B, b0 + per)))
            n = len(bb) * M
            st = np.empty(self.slots, dtype=states.dtype)
            st[:] = states[bb[0]]
            for k, b in enumerate(bb):
                st[k * M:(k + 1) * M] = states[b]
            for i in range(self.slots):
                core.set_state(i, st[i])
            o = st["origin"].astype(np.float64)
            v = st["velocity"].astype(np.float64)
            y = st["yaw"].astype(np.float64)
            u = np.zeros((self.slots, P.d_act), np.float64)
            for k, b in enumerate(bb):
                u[k * M:(k + 1) * M] = cand_u[b]
            nums = squash(P.prim, u)
            lines = [P.prim.line_of(o[i], v[i], float(y[i]), nums[i])[0][:L_MAX]
                     for i in range(self.slots)]
            self.line.set_lines(np.arange(self.slots), lines)
            d0 = np.linalg.norm(o - fin[None, :], axis=1)
            pol = self.make_policy(core, self.line)
            # one neutral tick fills the scratch core's observation buffer after the teleport
            obs = self._neutral_step()
            alive = np.ones(self.slots, bool)
            died = np.zeros(self.slots, bool)
            fnd = np.zeros(self.slots, bool)
            endp = o.copy()
            for _t in range(self.horizon):
                acts = pol.act(obs)
                view = getattr(pol, "view", None)
                pre = core.states_view["origin"].astype(np.float64)
                obs, _r, done, trunc, _term = (core.step(acts) if view is None
                                               else core.step(acts, view=view))
                ended = alive & (np.asarray(done, bool) | np.asarray(trunc, bool))
                if ended.any():
                    hits = np.asarray(core.goal_hits, bool)
                    fnd |= ended & np.asarray(done, bool) & hits
                    died |= ended & np.asarray(done, bool) & ~hits
                    endp[ended] = pre[ended]      # the last live position
                    alive &= ~ended
                if not alive[:n].any():
                    break
            endp[alive] = core.states_view["origin"][alive].astype(np.float64)
            self.sims += n
            # the planner's value of each surviving candidate's end state (its own input,
            # the bank grown by the progress the candidate made)
            prog = (d0 - np.linalg.norm(endp - fin[None, :], axis=1)) / 1000.0
            bnk = np.repeat(np.asarray(bank, np.float64)[bb], M)
            bnk = np.concatenate([bnk, np.zeros(self.slots - n)])
            live_i = np.flatnonzero(alive[:n])
            val = np.zeros(self.slots, np.float64)
            nov = np.zeros(self.slots, np.float64)
            if len(live_i):
                sv = core.states_view
                x = observe(self.caster, endp[live_i], sv["velocity"][live_i].astype(np.float64),
                            sv["yaw"][live_i].astype(np.float64), fin,
                            bnk[live_i] + prog[live_i])
                with torch.no_grad():
                    _lg, _mu, _ls, vv = P.net(torch.as_tensor(x, device=P.device))
                val[live_i] = vv.float().cpu().numpy()
                cx, cy, cz = P._cells(endp[live_i])
                cnt = P.nov_count[cx, cy, cz].astype(np.float64)
                nov[live_i] = float(P.cfg["plan_novelty"]) / np.sqrt(cnt + 1.0)
            fb = float(P.cfg["plan_finish_bonus"])
            pp = float(P.cfg["plan_progress"])
            sc = np.where(fnd, fb + pp * prog,
                          np.where(died, -pp * np.maximum(bnk, 0.0),
                                   pp * prog + SEARCH_GAMMA * val + nov))
            for k, b in enumerate(bb):
                scores[b] = sc[k * M:(k + 1) * M]
                died_all[b] = died[k * M:(k + 1) * M]
                fin_all[b] = fnd[k * M:(k + 1) * M]
        self.calls += 1
        self.deaths_avoided += int(died_all.sum())
        self.finishes_seen += int(fin_all.sum())
        return scores, {"died": died_all, "finished": fin_all}

    def _neutral_step(self):
        acts = np.tile(np.asarray(NEUTRAL_ACT, np.int32), (self.slots, 1))
        obs = self.core.step(acts)[0]
        return obs

    def choose(self, states, finish, bank, x, gen):
        """The committed choice for B real envs: (u (B, D) pre-squash, scores (B, M), info)."""
        cand = self.candidates(x, gen)
        scores, info = self.evaluate(states, finish, bank, cand)
        best = scores.argmax(1)
        return cand[np.arange(len(cand)), best], scores, info
