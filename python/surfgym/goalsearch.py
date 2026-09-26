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

Used at decision time in the greedy eval only (``make_primlearn_hooks(..., search=)``, driven by
``tools/record_ckpt.py --plan-search M`` / ``--plan-mcts N``); the trainer does not search.
``PrimMCTS`` (below) extends the one level of candidates to a tree.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

from .goalprimplan import L_MAX, mix_sample, observe, squash, RayCaster

SEARCH_GAMMA = 0.95          # the planner's per-primitive discount (goallearn.PLAN_GAMMA)
NEUTRAL_ACT = (7, 3, 1, 1, 0, 0)     # train_fast.NEUTRAL_ACT: centre view bins, no keys
REUSE_TOL_U = 16.0     # MCTS reuses the committed subtree when the real env lands this close
SHAPINGS = ("plain", "refund", "refund_i", "pbrs")


def edge_reward(shaping: str, fb: float, pp: float, prog, bank, fnd, died, g):
    """The planner's own reward for simulated primitives under --plan-shaping, as the trainer pays
    it (PrimLearnedPlanner.on_tick) -> (r, the bank after each primitive).

    ``prog``: each primitive's progress in the planner's unit; ``bank``: the episode's bank before
    it; ``fnd`` / ``died``: how it ended (neither = closed alive); ``g``: its discount (refund_i's
    interest, pbrs's gamma). plain: every primitive is paid its progress, a death included.
    refund: a death refunds a POSITIVE bank only (-max(bank, 0)). refund_i: a death refunds the
    bank whatever its sign, and a primitive that closes alive grows the bank by 1 / g (interest),
    so every failed episode nets 0. pbrs: alive g (bank + prog) - bank, any end -bank. A finish
    keeps its progress under plain / refund / refund_i and pays -bank under pbrs; the finish bonus
    on top. Under the refund rules the dying primitive's own progress is not paid."""
    prog = np.asarray(prog, np.float64)
    bank = np.broadcast_to(np.asarray(bank, np.float64), prog.shape)
    g = np.broadcast_to(np.asarray(g, np.float64), prog.shape)
    fnd = np.asarray(fnd, bool)
    died = np.asarray(died, bool)
    if shaping not in SHAPINGS:
        raise ValueError(f"plan_shaping {shaping!r} is not one of {SHAPINGS}")
    grown = bank + prog
    if shaping == "plain":
        r_dead = pp * prog
    elif shaping == "refund":
        r_dead = -pp * np.maximum(bank, 0.0)
    else:
        r_dead = -pp * bank
    if shaping == "pbrs":
        r_fin = fb - pp * bank
        r_alive = pp * (g * grown - bank)
    else:
        r_fin = fb + pp * prog
        r_alive = pp * prog
    r = np.where(fnd, r_fin, np.where(died, r_dead, r_alive))
    return r, (grown / g if shaping == "refund_i" else grown)


def unsquash(prim, nums) -> np.ndarray:
    """goalprimplan.squash inverted: (n, D) numbers in deg/s -> pre-squash u (clipped inside
    tanh's range)."""
    nums = np.asarray(nums, np.float64)
    k = prim.knots
    a = np.empty_like(nums)
    a[:, :k] = nums[:, :k] / max(prim.side, 1e-6)
    vv = nums[:, k:2 * k]
    a[:, k:2 * k] = np.where(vv >= 0.0, vv / max(prim.up, 1e-6), vv / max(prim.down, 1e-6))
    u = np.arctanh(np.clip(a, -0.999, 0.999))
    if getattr(prim, "frame", "velocity") == "map":
        # --prim-frame map: the heading dims are linear and wrapped (goalprimplan.squash)
        u[:, :k] = (np.mod(nums[:, :k] + 180.0, 360.0) - 180.0) / 180.0
    return u


class PrimSearch:
    """``core``: a scratch SurfCore of ``slots`` envs, built like the eval core (same map, physics,
    teleport-fail, finish box). ``line``: a goals.MultiLine of ``slots`` envs with the training fan
    offsets. ``make_policy(core, line)``: a fresh greedy executor wrapper on that core (the
    trainer's EVAL_GREEDY with ``route=line``). ``planner``: the PrimLearnedPlanner."""

    def __init__(self, core, line, make_policy: Callable, planner, m: int = 8,
                 horizon_ticks: Optional[int] = None, real_policy=None, explore: bool = False):
        if getattr(planner, "use_prev", False):
            raise ValueError("--plan-prev: the search does not carry the previous primitive yet")
        self.core, self.line, self.make_policy = core, line, make_policy
        self.P = planner
        self.m = max(2, int(m))
        self.slots = int(core.num_envs)
        if self.slots < self.m:
            raise ValueError(f"--plan-search {self.m}: the scratch core has {self.slots} envs")
        self.horizon = int(horizon_ticks) if horizon_ticks else int(planner.budget_ticks)
        self.caster = RayCaster(core)
        if getattr(planner, "flat", False):
            # --prim-flat: horizontal plans - the simulated executors' fan ignores height,
            # exactly like the real one's
            self.line.set_flat(True)
        self.calls = 0
        self.sims = 0
        self.deaths_avoided = 0          # candidates that died in simulation
        self.finishes_seen = 0
        # the wrapper flying the REAL env (env 0): its held keys (--keys-hold) are copied into
        # every simulated slot, or the simulation starts from released keys and diverges from
        # what the real executor will do (the first blue050 test: 0/9 with search, 2/9 without)
        self.real_policy = real_policy
        # novelty in the score is for exploration; an eval wants the best candidate only
        self.explore = bool(explore)

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
    def evaluate(self, states, finish, bank, cand_u, obs=None):
        """``states``: (B,) STATE_DTYPE of the real envs; ``cand_u``: (B, M, D) pre-squash
        candidates; ``bank``: (B,) the episodes' banked progress; ``obs``: (B, obs_dim) their
        current core observations (None: a neutral tick is spent instead). The executor's
        wrapper state is ``real_policy``'s env 0 (the eval records one env). -> (scores (B, M),
        info)."""
        P, core = self.P, self.core
        obs_in = obs
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
            pol = self._policy(self._row_of(self.real_policy, 0))
            obs = self._start_obs(None if obs_in is None else np.asarray(obs_in)[bb[0]])
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
            prog = (d0 - np.linalg.norm(endp - fin[None, :], axis=1)) / getattr(
                P, "eval_unit", 1000.0)
            bnk = np.repeat(np.asarray(bank, np.float64)[bb], M)
            bnk = np.concatenate([bnk, np.zeros(self.slots - n)])
            fb = float(P.cfg["plan_finish_bonus"])
            pp = float(P.cfg["plan_progress"])
            # the planner's own reward rule (a refund_i bank grows by its interest)
            r_e, nbk = edge_reward(str(P.cfg.get("plan_shaping") or "refund"), fb, pp, prog, bnk,
                                   fnd, died, SEARCH_GAMMA)
            live_i = np.flatnonzero(alive[:n])
            val = np.zeros(self.slots, np.float64)
            nov = np.zeros(self.slots, np.float64)
            if len(live_i):
                sv = core.states_view
                x = observe(self.caster, endp[live_i], sv["velocity"][live_i].astype(np.float64),
                            sv["yaw"][live_i].astype(np.float64), fin, nbk[live_i],
                            frame=getattr(P.prim, "frame", "velocity"))
                with torch.no_grad():
                    _lg, _mu, _ls, vv = P.net(torch.as_tensor(x, device=P.device))
                val[live_i] = vv.float().cpu().numpy()
                cx, cy, cz = P._cells(endp[live_i])
                cnt = P.nov_count[cx, cy, cz].astype(np.float64)
                nov[live_i] = float(P.cfg["plan_novelty"]) / np.sqrt(cnt + 1.0)
            sc = np.where(fnd | died, r_e,
                          r_e + SEARCH_GAMMA * val + (nov if self.explore else 0.0))
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

    def choose(self, states, finish, bank, x, gen, obs=None):
        """The committed choice for B real envs: (u (B, D) pre-squash, scores (B, M), info)."""
        cand = self.candidates(x, gen)
        scores, info = self.evaluate(states, finish, bank, cand, obs=obs)
        best = scores.argmax(1)
        return cand[np.arange(len(cand)), best], scores, info

    # ------------------------------------------------------------------ the executor's state
    @staticmethod
    def _row_of(pol, i: int):
        """Everything the executor wrapper carries for env ``i`` besides the core state and the
        observation - what its next decisions depend on: the action and view command it is
        holding between decisions, the held keys (--keys-hold) and their episode-start
        detector, and the decision phase. None without a wrapper."""
        if pol is None:
            return None
        k = getattr(pol, "keys", None)
        held = getattr(pol, "_held", None)
        view = getattr(pol, "view", None)
        kt = getattr(pol, "_keys_tick", None)
        return {"held": None if held is None else np.array(held[i], copy=True),
                "view": None if view is None else np.array(view[i], copy=True),
                "keys": None if k is None else (k.state[i].copy(), k.boot[i].copy()),
                "keys_tick": None if kt is None else int(np.asarray(kt)[i]),
                "tick": int(getattr(pol, "_tick", 0))}

    def _policy(self, row):
        """A fresh greedy executor on the scratch core, every slot continuing ``row`` (the
        wrapper state of the env being simulated): it holds the same action and view until its
        next decision, which falls on the same tick, with the same held keys. Without this a
        simulation starts on released keys and a fresh decision - it diverged from the real
        flight within one primitive (a first blue050 test: 0/9 with search, 2/9 without)."""
        pol = self.make_policy(self.core, self.line)
        if row is None:
            return pol
        S = self.slots
        if row["keys"] is not None and getattr(pol, "keys_hold", False):
            from .keyshold import KeysHold
            pol.keys = KeysHold(S)
            pol.keys.state[:] = row["keys"][0]
            pol.keys.boot[:] = row["keys"][1]
            if row["keys_tick"] is not None:
                pol._keys_tick = np.full(S, row["keys_tick"], np.int64)
        if row["held"] is not None and hasattr(pol, "_held"):
            pol._held = np.ascontiguousarray(np.repeat(row["held"][None, :], S, 0))
            pol._tick = int(row["tick"])
            if row["view"] is not None:
                pol.view = np.ascontiguousarray(np.repeat(row["view"][None, :], S, 0))
        return pol

    def _start_obs(self, obs_row):
        """The observation the simulated executors read first: the real env's own row when
        given (no tick is spent), else one neutral tick to fill the scratch core's buffer."""
        if obs_row is None:
            return self._neutral_step()
        return np.ascontiguousarray(np.repeat(np.asarray(obs_row, np.float32)[None, :],
                                              self.slots, 0))


# ==========================================================================================
# MCTS over primitives (--plan-mcts, eval-time): a search TREE whose nodes are exact states
# ==========================================================================================
class _Edge:
    """One primitive flown from its parent node by the greedy executor: its reward, how it
    ended (closed alive, died, finished), the exact state it closed in (+ the executor's held
    keys and the episode's bank there), the planner's value of that state, and the edges below
    it once that state has been expanded."""

    __slots__ = ("u", "r", "v", "died", "fin", "state", "row", "obs", "bank", "ticks", "child",
                 "n", "disc", "line", "path")

    def __init__(self, u, r, v, died, fin, state, row, obs, bank, ticks):
        self.u, self.r, self.v = u, float(r), float(v)
        self.died, self.fin = bool(died), bool(fin)
        self.state, self.row, self.obs = state, row, obs
        self.bank, self.ticks = float(bank), int(ticks)
        self.child = None
        self.n = 0
        self.disc = None          # the discount on what follows (None: the gamma passed to q)
        self.line = None          # --plan-mcts-dump: the planned curve and the simulated flight
        self.path = None

    @property
    def term(self) -> bool:
        return self.died or self.fin

    def q(self, gamma: float) -> float:
        """The edge's value by a MAX backup: r + gamma x the best continuation found below it
        (the value head where nothing is expanded yet). The simulator and the greedy executor
        are deterministic, so an edge is worth its best continuation, not the mean of the
        ones tried."""
        g = gamma if self.disc is None else self.disc
        if self.term:
            return self.r
        if self.child is None:
            return self.r + g * self.v
        return self.r + g * max(c.q(gamma) for c in self.child)


class _Node(list):
    """A node's children (a list of _Edge) plus what progressive widening needs: the node's own
    exact state, the executor wrapper's row, the observation and the bank there, and how many
    batches of candidates it holds so far."""

    def __init__(self, edges, src):
        super().__init__(edges)
        self.src = src
        self.batches = 1


MCTS_MAX_BATCHES = 8     # progressive widening: a node holds at most 8 batches of candidates


class PrimMCTS(PrimSearch):
    """``--plan-mcts N``: a search TREE over primitives instead of one level of candidates.

    A node is an exact simulator state (+ the executor's held keys, + the episode's banked
    progress). EXPANDING it draws ``m`` primitives from the planner's mixture AT THAT STATE (its
    heaviest mean first, then m-1 samples) and flies all of them with the real greedy executor
    in one batch on the scratch core, each until it closes the way a real one does (arc >=
    COMPLETE_FRAC inside the corridor, or the primitive's budget), dies or finishes. An edge's
    reward is the planner's own (progress per 1000 u; + the finish bonus; a failed end - death
    or the episode clock - refunds the bank), a leaf is valued by the planner's value head,
    edges back up by MAX (a deterministic model), and selection is PUCT over min-max-normalised
    values with a uniform prior over the sampled children (they already are draws from the
    planner's policy). ``sims`` expansions per decision, the tree at most ``depth`` primitives
    deep; the root edge with the most visits is committed (ties: the higher value) and the tree
    is rebuilt at the next decision. Nothing about the map is written here: the simulator is
    the model."""

    def __init__(self, core, line, make_policy, planner, m: int = 6, sims: int = 16,
                 depth: int = 0, c_puct: float = 1.25, time_disc: bool = False,
                 gamma: float = SEARCH_GAMMA, uniform: float = 0.0, reuse: bool = True,
                 nov_coef: float = 0.0, leaf: str = "value", dump: bool = False,
                 no_planner: bool = False, reward: str = "", commit: str = "visits", **kw):
        super().__init__(core, line, make_policy, planner, m=m, **kw)
        # --plan-mcts-leaf: what an unexpanded leaf is worth beyond its edge's own reward -
        # value = the planner's value head (r + gamma V(end)); zero = nothing (the user,
        # 2026-09-26: the reward is not sparse here, so let the tree score paths by the reward
        # they actually earned in simulation, not by a value head trained on another
        # distribution)
        if leaf not in ("value", "zero"):
            raise ValueError("--plan-mcts-leaf value | zero")
        self.leaf = str(leaf)
        # the edge reward is the planner's own: under --plan-shaping plain a death is paid the
        # progress it made (nothing taken back); under the refund rules it refunds the bank
        # --plan-mcts-reward plain | refund overrides it (a search-only question, e.g. a refund
        # checkpoint's executor scored with the plain reward)
        if reward not in ("",) + SHAPINGS:
            raise ValueError("--plan-mcts-reward " + " | ".join(SHAPINGS))
        self.shaping = (str(reward) if reward
                        else str(planner.cfg.get("plan_shaping") or "refund"))
        self.plain = self.shaping == "plain"
        # --plan-mcts-no-planner: EVERY candidate is a uniform draw over the primitive ranges -
        # the planner proposes nothing (not even its greedy mean), so what the tree finds is what
        # the executor can fly, not what the planner already knows
        self.no_planner = bool(no_planner)
        # --plan-mcts-commit: which root primitive is flown for real - the most visited (visits,
        # AlphaZero's rule, the default) or the best-valued (value: under a max backup in a
        # deterministic simulator the visits measure where the search LOOKED, the value what it
        # FOUND - a finishing branch seen late keeps few visits)
        if commit not in ("visits", "value"):
            raise ValueError("--plan-mcts-commit visits | value")
        self.commit = str(commit)
        # --plan-mcts-dump: keep every simulated primitive's planned curve and flown path, and
        # export each decision's tree (info["tree"]) for tools / visualisation
        self.dump = bool(dump)
        # --plan-mcts-explore: a count-based novelty bonus on every edge that ends alive,
        # nov_coef / sqrt(1 + N) with N = the planner's own global end-cell count (the counts
        # its training built, restored from the checkpoint): the tree expands toward cells the
        # training rarely reached instead of settling where the Euclidean distance is smallest
        self.nov_coef = float(nov_coef)
        from .goallearn import COMPLETE_FRAC
        self.n_exp = max(1, int(sims))
        # --plan-mcts-depth 0 (the default): NO depth limit - the tree grows wherever the
        # selection sends it, within the expansion budget
        self.depth = math.inf if int(depth) <= 0 else int(depth)
        # the committed edge's subtree is the next decision's root (--plan-mcts-noreuse: off)
        self.reuse = bool(reuse)
        self._keep = None
        self.reused = 0
        self.verbose = False            # --plan-mcts-verbose: one line per decision
        self.widened = 0                # progressive-widening batches added
        self.c_puct = float(c_puct)
        # --plan-replan: the search closes a primitive where training does
        self.complete_frac = float(getattr(planner, "close_frac", COMPLETE_FRAC))
        # --plan-mcts-time: discount per SECOND of flight instead of per primitive - gamma per
        # nominal primitive duration, so a primitive that takes longer to close costs more and
        # the search prefers the faster of two equal-progress lines (per primitive, a slow and
        # a fast primitive are discounted alike)
        self.time_disc = bool(time_disc)
        # --plan-mcts-gamma: the tree's discount (default the planner's own, 0.95);
        # --plan-mcts-uniform F: that share of each expansion's children drawn UNIFORMLY from
        # the primitive ranges (step 1's draw) instead of from the planner's mixture - a search
        # wider than the planner's own habits
        self.gamma = float(gamma)
        self.n_uniform = int(round(float(uniform) * (self.m - 1)))
        self.urng = np.random.default_rng(4321)
        self.nominal_ticks = float(planner.prim.secs) * 1000.0 / float(planner.tick_ms)
        self.expansions = 0
        self.depth_hist = {}            # the tree's depth (primitives) at each decision -> count
        self.tree_fin = 0                                     # finishes seen anywhere in trees

    def describe(self) -> str:
        return (f"MCTS: {self.n_exp} expansions per decision; an expansion flies {self.m} "
                + (f"UNIFORM primitives (--plan-mcts-no-planner: the planner proposes nothing)"
                   if self.no_planner else
                   f"primitives (the planner's heaviest mean + {self.m - 1} samples, drawn at that "
                   f"node's state)")
                + f" with the greedy executor from the node's EXACT state until each "
                f"closes (arc >= {self.complete_frac:g} or {self.horizon} ticks), dies or "
                f"finishes; tree depth "
                + ("unlimited" if self.depth == math.inf else f"<= {self.depth} primitives")
                + ("; the committed subtree is reused" if self.reuse else "")
                + (f"; novelty {self.nov_coef:g}/sqrt(1 + N) on every alive edge (N = the "
                   f"planner's end-cell counts)" if self.nov_coef > 0.0 else "")
                + f"; progressive widening (a node gets another {self.m} candidates as "
                  f"its visits pass K^2 x batches^2, or at once when all it holds die; <= "
                  f"{MCTS_MAX_BATCHES} batches)"
                + f"; PUCT c {self.c_puct:g} over "
                f"min-max-normalised max-backup values; edge reward = the planner's (progress, "
                f"finish, "
                + ("a death keeps the progress it made" if self.plain
                   else {"refund": "failed-end refund of a positive bank",
                         "refund_i": "failed-end refund of the bank with interest (refund_i)",
                         "pbrs": "exact potential-based shaping (pbrs)"}[self.shaping])
                + "), leaf = "
                + ("its value head; " if self.leaf == "value"
                   else "NOTHING (--plan-mcts-leaf zero: paths scored by the reward earned); ")
                + (f"{self.n_uniform} of the {self.m} children drawn uniformly; "
                   if self.n_uniform else "")
                + f"gamma {self.gamma:g} "
                + (f"per {self.nominal_ticks:.0f} ticks of flight (time-discounted); "
                   if self.time_disc else "per primitive; ")
                + ("commit the best-valued root primitive" if self.commit == "value"
                   else "commit the most-visited root primitive"))

    # ------------------------------------------------------------------ one expansion
    def _expand(self, state, row, obs_row, bank: float, fin, gen):
        """Fly ``m`` candidates from one exact state (+ the executor wrapper's ``row`` and the
        core observation ``obs_row`` there) -> their ``m`` edges. A primitive closes like in the
        real eval: once complete or out of budget, at the executor's next decision tick (that
        is where the real planner issues the next one)."""
        from .goalarc import MultiArcProgress
        P, core, M, S = self.P, self.core, self.m, self.slots
        for i in range(S):
            core.set_state(i, state)
        o = np.repeat(np.asarray(state["origin"], np.float64)[None, :], S, 0)
        v = np.repeat(np.asarray(state["velocity"], np.float64)[None, :], S, 0)
        yaw = float(state["yaw"])
        x0 = observe(self.caster, o[:1], v[:1], np.array([yaw]), fin, np.array([bank]),
                     frame=getattr(P.prim, "frame", "velocity"))
        cand = self.candidates(x0, gen)[0]                          # (M, D) pre-squash
        if self.no_planner:
            for j in range(M):
                cand[j] = unsquash(P.prim, P.prim.sample(self.urng)[None, :])[0]
        elif self.n_uniform:
            # the last n_uniform children: uniform draws over the primitive ranges
            for j in range(M - self.n_uniform, M):
                cand[j] = unsquash(P.prim, P.prim.sample(self.urng)[None, :])[0]
        u = np.zeros((S, P.d_act), np.float64)
        u[:M] = cand
        nums = squash(P.prim, u)
        lines = [P.prim.line_of(o[i], v[i], yaw, nums[i])[0][:L_MAX] for i in range(S)]
        self.line.set_lines(np.arange(S), lines)
        trk = MultiArcProgress(S, l_max=L_MAX, spacing=P.prim.spacing, corridor=P.corridor,
                               window=16)
        if getattr(P, "flat", False):
            trk.set_flat(True)
        trk.set_lines(np.arange(S), lines)
        pol = self._policy(row)
        obs = self._start_obs(obs_row)
        K = max(1, int(getattr(pol, "_k", 1)))
        d0 = float(np.linalg.norm(o[0] - fin))
        open_ = np.zeros(S, bool)
        open_[:M] = True                  # still flying its primitive
        closing = np.zeros(S, bool)       # complete / out of budget, waiting for the decision
        died = np.zeros(S, bool)
        fnd = np.zeros(S, bool)
        ticks = np.zeros(S, np.int64)
        endp = o.copy()
        end_arr = None
        end_obs = [None] * M
        end_rows = [None] * M
        paths = [[o[i].copy()] for i in range(M)] if self.dump else None
        for t in range(self.horizon + K):
            acts = pol.act(obs)
            view = getattr(pol, "view", None)
            pre = core.states_view["origin"].astype(np.float64)
            obs, _r, done, trunc, _term = (core.step(acts) if view is None
                                           else core.step(acts, view=view))
            done = np.asarray(done, bool)
            ended = open_ & (done | np.asarray(trunc, bool))
            if ended.any():
                hits = np.asarray(core.goal_hits, bool)
                fnd |= ended & done & hits
                died |= ended & ~(done & hits)        # a death, or the episode clock ran out
                endp[ended] = pre[ended]              # the last live position
                ticks[ended] = t + 1
                open_ &= ~ended
            trk.advance(core.states_view["origin"].astype(np.float32))
            if paths is not None and t % 5 == 4:
                cur_o = core.states_view["origin"]
                for i in np.flatnonzero(open_[:M]):
                    paths[i].append(cur_o[i].astype(np.float64).copy())
            closing |= open_ & (trk.arc >= self.complete_frac * trk.total_arc())
            if t + 1 >= self.horizon:
                closing |= open_                      # the budget: it closes where it is
            ready = closing & open_
            if int(getattr(pol, "_tick", 0)) % K != 0:
                ready[:] = False                      # the next act() is not a decision yet
            if ready.any():
                cur = core.get_states()
                if end_arr is None:
                    end_arr = np.empty(M, dtype=cur.dtype)
                for i in np.flatnonzero(ready[:M]):
                    end_arr[i] = cur[i]
                    endp[i] = cur[i]["origin"]
                    end_obs[i] = np.array(obs[i], copy=True)
                    end_rows[i] = self._row_of(pol, i)
                    ticks[i] = t + 1
                open_ &= ~ready
            if not open_[:M].any():
                break
        self.sims += M
        self.expansions += 1
        prog = (d0 - np.linalg.norm(endp[:M] - fin[None, :], axis=1)) / getattr(
            P, "eval_unit", 1000.0)
        term = died[:M] | fnd[:M]
        fb = float(P.cfg["plan_finish_bonus"])
        pp = float(P.cfg["plan_progress"])
        # each edge's discount: per primitive, or per nominal duration (--plan-mcts-time)
        gdisc = (self.gamma ** (np.asarray(ticks[:M], np.float64) / self.nominal_ticks)
                 if self.time_disc else np.full(M, self.gamma))
        # the planner's own reward rule, and the bank each child carries (refund_i: + interest)
        r_e, nbk = edge_reward(self.shaping, fb, pp, prog, bank, fnd[:M], died[:M], gdisc)
        val = np.zeros(M, np.float64)
        li = np.flatnonzero(~term)
        if len(li) and self.leaf == "value":
            x = observe(self.caster, endp[li], end_arr["velocity"][li].astype(np.float64),
                        end_arr["yaw"][li].astype(np.float64), fin, nbk[li],
                        frame=getattr(P.prim, "frame", "velocity"))
            with torch.no_grad():
                val[li] = P.net(torch.as_tensor(x, device=P.device))[3].float().cpu().numpy()
        nov = np.zeros(M, np.float64)
        if self.nov_coef > 0.0 and len(li):
            cx, cy, cz = P._cells(endp[li])
            nov[li] = self.nov_coef / np.sqrt(P.nov_count[cx, cy, cz].astype(np.float64) + 1.0)
        edges = []
        for i in range(M):
            r = float(r_e[i]) + (0.0 if term[i] else float(nov[i]))
            edges.append(_Edge(cand[i], r, val[i], died[i], fnd[i],
                               None if term[i] else end_arr[i], end_rows[i], end_obs[i],
                               float(nbk[i]), ticks[i]))
            if paths is not None:
                edges[-1].line = np.asarray(lines[i], np.float64)
                edges[-1].path = np.vstack(paths[i] + [endp[i]])
            if self.time_disc:
                edges[-1].disc = self.gamma ** (float(ticks[i]) / self.nominal_ticks)
        self.deaths_avoided += int(died[:M].sum())
        self.finishes_seen += int(fnd[:M].sum())
        node = _Node(edges, (state, row, obs_row, bank))
        node.x0 = x0          # the planner's observation the candidates were drawn at
        return node

    # ------------------------------------------------------------------ the search
    def _want_widen(self, node) -> bool:
        """Progressive widening (Coulom 2007; Couetoux et al. 2011) for a continuous action
        space: a node that holds b batches of K candidates gets another batch once its visits
        reach (K b)^2 - i.e. its width grows like sqrt(visits) - and at once when every
        candidate it holds died (a finite sample of a continuous action space is not the node's
        last word)."""
        if node.src is None:
            return False
        if all(c.died for c in node):
            return True
        return sum(c.n for c in node) >= (self.m * node.batches) ** 2

    def _tree_depth(self, edges) -> int:
        """Primitives below ``edges`` along the deepest expanded path (1 = the root's own)."""
        best = 1
        for e in edges:
            if e.child is not None:
                best = max(best, 1 + self._tree_depth(e.child))
        return best

    def choose(self, states, finish, bank, x, gen, obs=None):
        """(u (1, D) pre-squash, scores (1, M) = the root edges' values, info) for env 0 of
        ``states`` (``obs``: its current core observation); info["best"] is the committed root
        edge (the most visited), info["pred_end"] where the simulation says it will close (None
        if it ends the episode). With reuse, the subtree below the edge committed at the last
        decision is the new root when the real env arrived where the simulation said it would
        (within REUSE_TOL_U) - the tree keeps growing over the episode."""
        fin = np.asarray(finish, np.float64).reshape(3)
        g = self.gamma
        b0 = float(np.asarray(bank, np.float64).reshape(-1)[0])
        root = None
        was_reused = False
        pos = np.asarray(states[0]["origin"], np.float64)
        if self.reuse and self._keep is not None:
            kept, at = self._keep
            if kept is not None and float(np.linalg.norm(pos - at)) <= REUSE_TOL_U:
                root = kept
                was_reused = True
                self.reused += 1
        self._keep = None
        n_exp = 0
        if root is None:
            root = self._expand(states[0], self._row_of(self.real_policy, 0),
                                None if obs is None else np.asarray(obs)[0], b0, fin, gen)
            n_exp = 1
        lo, hi = math.inf, -math.inf
        for _it in range(8 * self.n_exp):
            if n_exp >= self.n_exp:
                break
            edges, d, path, grown = root, 0, [], False
            while True:
                if edges.batches < MCTS_MAX_BATCHES and self._want_widen(edges):
                    # PROGRESSIVE WIDENING: more candidates from this node's state - as its
                    # visits grow (batches ~ sqrt(visits) / K), and at once when every
                    # candidate it holds dies (a dead end the fixed set cannot leave)
                    extra = self._expand(*edges.src, fin, gen)
                    edges.extend(extra)
                    edges.batches += 1
                    n_exp += 1
                    self.widened += 1
                    self.tree_fin += sum(c.fin for c in extra)
                    grown = True
                    break
                qs = np.array([e.q(g) for e in edges])
                lo, hi = min(lo, float(qs.min())), max(hi, float(qs.max()))
                qn = (qs - lo) / (hi - lo) if hi > lo else np.zeros_like(qs)
                nv = np.array([e.n for e in edges], np.float64)
                ucb = qn + self.c_puct * math.sqrt(nv.sum() + 1.0) / (1.0 + nv) / len(edges)
                e = edges[int(np.argmax(ucb))]
                path.append(e)
                d += 1
                if e.term or e.child is None or d >= self.depth:
                    break
                edges = e.child
            if not grown and not e.term and e.child is None and d < self.depth:
                e.child = self._expand(e.state, e.row, e.obs, e.bank, fin, gen)
                n_exp += 1
                self.tree_fin += sum(c.fin for c in e.child)
            for ed in path:
                ed.n += 1
        self.calls += 1
        deepest = self._tree_depth(root)
        self.depth_hist[deepest] = self.depth_hist.get(deepest, 0) + 1
        nv = np.array([e.n for e in root])
        qs = np.array([e.q(g) for e in root])
        if self.commit == "value":
            best = int(np.lexsort((nv, qs))[-1])                 # best value, then visits
        else:
            best = int(np.lexsort((qs, nv))[-1])                 # most visited, then value
        eb = root[best]
        if self.verbose:
            size = [0, 0, 0]                                    # edges, died, finished

            def _count(es):
                for c in es:
                    size[0] += 1
                    size[1] += int(c.died)
                    size[2] += int(c.fin)
                    if c.child is not None:
                        _count(c.child)
            _count(root)
            print(f"  mcts decision {self.calls}: pos {pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f} "
                  f"bank {b0:+.2f} | {n_exp} new expansions, tree {size[0]} primitives "
                  f"({size[1]} died, {size[2]} finished), depth {deepest} | root visits "
                  f"{nv.tolist()} values {[round(float(z), 2) for z in qs]} -> #{best}"
                  + (" (reused subtree)" if was_reused else ""), flush=True)
        if self.reuse and not eb.term:
            self._keep = (eb.child, np.asarray(eb.state["origin"], np.float64))
        tree = None
        if self.dump:
            # --plan-mcts-dump: the whole tree, parents before children (ids in DFS order)
            tree = []

            def _walk(es, parent, depth):
                for k_, e_ in enumerate(es):
                    nid = len(tree)
                    tree.append({"id": nid, "parent": parent, "depth": depth,
                                 "root_choice": bool(parent == -1 and k_ == best),
                                 "r": round(float(e_.r), 4), "q": round(float(e_.q(g)), 4),
                                 "v": round(float(e_.v), 4), "n": int(e_.n),
                                 "died": bool(e_.died), "fin": bool(e_.fin),
                                 "ticks": int(e_.ticks),
                                 "line": (None if e_.line is None else
                                          np.round(e_.line, 0).astype(int).tolist()),
                                 "path": (None if e_.path is None else
                                          np.round(e_.path, 0).astype(int).tolist())})
                    if e_.child is not None:
                        _walk(e_.child, nid, depth + 1)
            _walk(root, -1, 1)
        info = {"died": np.array([[e.died for e in root]]),
                "finished": np.array([[e.fin for e in root]]),
                "tree": tree,
                "best": best, "visits": nv.tolist(), "expansions": n_exp, "depth": deepest,
                "pred_end": (None if eb.term else np.asarray(eb.state["origin"], np.float64)),
                "pred_ticks": eb.ticks, "pred_fin": eb.fin, "pred_died": eb.died,
                # the search's policy target at the root (tools/az_worker.py): every root
                # candidate's pre-squash numbers (visits / values above, in the same order) and
                # the planner observation they were drawn at
                "root_u": np.stack([np.asarray(e.u, np.float32) for e in root]),
                "root_x": getattr(root, "x0", None)}
        return eb.u[None, :], qs[None, :], info
