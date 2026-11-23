"""az_mcts.py - AlphaZero-style test-time search, ONE DECISION PER EDGE (the user, 2026-11-21).

The user's idea: search at inference time as the generalisation tool on a map the agent never saw -
a deterministic policy has one attempt and no memory, a tree search over the simulator with the
trained policy as the proposal distribution keeps memory (failed branches are not retried). Their
spec: "the most straightforward alpha zero type approach, like MCTS that is guided by network, ...
every node is one action, at 30 Hz"; "one node is one state, and we sample the action from policy
using some ... temperature"; "it doesn't have the exact amount of children ... exploring the
branches that give more value ... the tree turns out very asymmetrical".

Driven by ``tools/beam_tas.py --az`` (the checkpoint-faithful core, lidar, wrappers and feeds, as
--mcts; nothing here re-implements the observation pipeline).

THE TREE. A node is one EXACT simulator state at a decision boundary plus the wrapper's per-env
inference state (mcts_search.SideState / RowStore). On creation a node gets, from ONE forward of
the network: its value V(s) (the critic) and K candidate decisions - candidate 0 the policy's mode,
the others draws at --az-keys-temp / --az-view-temp (Sampled MuZero, Hubert et al. 2021: the prior
over a sampled set is uniform). A child exists only once PUCT has picked its edge (lazy expansion):
one simulation = select down the tree, step ONE decision (act_every ticks) from the leaf's parent,
evaluate the new node, back up - so the tree grows deep where the value is and stays asymmetric.

VALUE. --az-reward none: pure AlphaZero - the edge reward is 0 except the terminals (a finish pays
--az-finish, a death is a terminal of value 0), the leaf is V(s'). banked: MuZero's backup with the
map's own reward - an edge pays scale x the NEW banked progress (the episode's record d at a tick
where the map pushed back: a fall pays nothing, tools/mcts_search.contact_mask), scale = 100 / d0
as the trainer's race reward, minus the trainer's time penalty; then gamma x V(s'). Q is min-max
normalised over the tree (MuZero) before PUCT:

    U(s, a) = Qn(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a)),  P = 1 / K

an unvisited edge reads its parent's own normalised value (first-play urgency). A batch of B
simulations runs per wave with virtual loss (Chaslot et al. 2008).

MEMORY (--az-novelty BETA, 0 = off): the tree's Go-Explore archive - every expansion that reaches
a cell (position / --mcts-cell u, energy height / --mcts-cell-energy u) the whole search has
visited n times before earns BETA / sqrt(1 + n) on its edge (count-based novelty, Bellemare 2016 /
Go-Explore's cell weight), so the tree remembers where it has been and pushes into new parts of
the map; the critic still scores safety. Only a decision during which the map pushed back counts
(contact_mask: a surface stood or surfed on), so a fall through new air does not pay.

WIDENING (--az-widen ALPHA, 0 = off: all K candidates open at once): progressive widening
(Coulom 2007, Couetoux et al. 2011) - a node visited N times offers only its first
ceil((N + 1) ** ALPHA) candidates (the mode first, then the draws), so a node the search has
barely visited extends the policy's own choice and the tree runs deep along it, widening where
visits pile up. Every commit logs how deep below the root its simulations reached.

PRIOR (--az-prior): uniform = 1 / K per candidate (Sampled MuZero's rule for candidates drawn
from the policy itself); policy = each candidate weighted by its own probability under the policy
(the key heads + the yaw), so PUCT runs deep along the policy's choice (AlphaZero's prior).
ROLLOUTS (--az-rollout H, 0 = off): a new leaf is judged by the policy's mode played on for H
decisions from it (finish = --az-finish, death = 0, survivors bootstrap on the critic), classic
MCTS's rollout, so a fall that ends a second later is seen at the leaf.

MACRO EDGES (--az-hold M, 1 = one decision per edge): an edge is M decisions (~0.5 s at M = 16,
K = 4, 7.667 ms): the candidate is the FIRST decision, then the policy acts closed-loop for the
other M - 1 (child 0 on its mode, the others on its own sampling), so the tree is seconds deep at
the same simulation budget - the training loop's macro-action tree (mcts_search.py) finished
surf_left_turn_curve in 70 s with ~0.4 s moves where one decision per edge reached ~0.25 s deep
(the user, 2026-11-22: "let's try these ideas"). gamma per edge = gamma per decision ** M.

DEATH WINDOW (--az-death-window W s, 0 = off; --az-reward banked): a simulation that ends in a death
- a tree terminal or a death inside its leaf rollout - gives back the progress it was paid in its
last W seconds, edge by edge on the way up (the trainer's --death-charge-window, Grzes 2017's
terminal rule over a window): a dive that banks depth and dies is worth nothing, progress the line
survived for W seconds is kept. A tree with this rule tries a pit once per branch.

THE CRITIC'S PRIVILEGED INPUT: a --priv-critic 2 checkpoint's critic reads d / --race-pay-d0 (the
shaping field's distance to the finish in plain units); the search feeds it from the same field it
banks on, so V(s) is the trained critic's and not the NaN a missing block returns.

ACTING (receding horizon, AlphaZero's play): --az-sims simulations from the root, then the root's
most visited edge is COMMITTED - the real episode advances one decision and keeps that subtree -
until the root finishes, dies, reaches --max-ticks or the --az-minutes budget ends. The committed
line is the result (beam_tas --save-frontier replays it open-loop and checks it bit-exact).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from mcts_search import CellArchive, RowStore, SideState, contact_mask, field_d  # noqa: E402


def make_az_policy():
    """The AzPolicy class (deferred: train_fast imports torch and the CUDA stack)."""
    import torch
    from mcts_policy import MctsPolicy
    from train_fast import N_VIEW, sample_view, split_view

    class AzPolicy(MctsPolicy):
        """MctsPolicy + FORCED decisions (the envs of force_idx take the given policy-space row
        and view z instead of a draw) and propose(): the value and K candidate decisions per env
        from ONE forward (candidate 0 = the mode)."""

        force_idx = None
        force_act = None
        force_z = None
        # --critic-pay-d0: this map's k (value_k); None for a one-head critic
        pay_k = None

        def _v1(self, v):
            """V(s) per env in this map's reward units: a --critic-pay-d0 critic's [V_R, V_P]
            row as V_R + k V_P (k = pay_k), a one-head critic's value as is."""
            v = v.float()
            if v.dim() == 2 and v.shape[1] == 2:
                if self.pay_k is None:
                    raise RuntimeError("a two-column critic and no pay_k: set it from "
                                       "az_mcts.value_k(cfg, d0)")
                return v[:, 0] + float(self.pay_k) * v[:, 1]
            return v.reshape(-1)

        @torch.inference_mode()
        def _decide(self, obs):
            logits, _ = self._net(self._obs(obs))
            cat, mu = split_view(logits.float())
            padded = self._mask_padded(self.packer.pad(cat), obs)
            act, z, _ = sample_view(padded, mu, self.policy.log_std())
            g = self.greedy
            if g is not None and np.any(g):
                # --az-rollout: the rolled-out envs take the policy's mode (MctsPolicy's mask)
                gi = torch.as_tensor(np.flatnonzero(g), device=act.device)
                act[gi, N_VIEW:] = padded[gi, N_VIEW:].argmax(-1)
                z[gi] = mu[gi]
            if self.force_idx is not None and len(self.force_idx):
                fi = torch.as_tensor(self.force_idx, device=act.device)
                act[fi, N_VIEW:] = torch.as_tensor(self.force_act, device=act.device,
                                                   dtype=act.dtype)
                z[fi] = torch.as_tensor(self.force_z, device=z.device, dtype=z.dtype)
            return self._finish_view(act[:, N_VIEW:], z)

        @torch.inference_mode()
        def propose(self, obs, k: int, keys_temp: float, view_temp: float):
            """-> (acts (n, k, heads - N_VIEW) int, z (n, k, n_z) float, value (n,) float,
            logp (n, k) float: each candidate's log-probability under the policy at temperature 1,
            the key heads plus the yaw (the pitch moves nothing))."""
            logits, v = self._net(self._obs(obs))
            cat, mu = split_view(logits.float())
            padded = self._mask_padded(self.packer.pad(cat), obs)
            n = padded.shape[0]
            g_act = padded[:, N_VIEW:].argmax(-1)
            acts = [g_act.unsqueeze(1)]
            zs = [mu.unsqueeze(1)]
            if k > 1:
                P = padded.repeat_interleave(k - 1, dim=0)
                MU = mu.repeat_interleave(k - 1, dim=0)
                n_z = mu.shape[1]
                tv = torch.ones(n_z, device=mu.device)
                tv[0] = float(view_temp)
                a_s, z_s, _ = sample_view(P, MU, self.policy.log_std(),
                                          temp=float(keys_temp), temp_view=tv)
                acts.append(a_s[:, N_VIEW:].reshape(n, k - 1, -1))
                zs.append(z_s.reshape(n, k - 1, -1))
            val = self._v1(v).reshape(n)
            A = torch.cat(acts, 1)
            Z = torch.cat(zs, 1)
            lsm = torch.log_softmax(padded[:, N_VIEW:], dim=-1)
            lk = torch.gather(lsm.unsqueeze(1).expand(-1, A.shape[1], -1, -1), 3,
                              A.long().unsqueeze(-1)).squeeze(-1).sum(-1)
            ls = self.policy.log_std().float()
            ls0 = ls[..., 0].reshape(-1, 1) if ls.dim() > 1 else ls[0]
            lv = -0.5 * ((Z[..., 0] - mu[:, None, 0]) / ls0.exp()) ** 2 - ls0
            return (A.cpu().numpy(), Z.cpu().numpy(), val.cpu().numpy(),
                    (lk + lv).double().cpu().numpy())

    return AzPolicy


def priv_feed(cfg: dict, gf):
    """A --priv-critic 2 checkpoint's critic reads d / --race-pay-d0 off the shaping field (the
    trainer's RaceReward samples the same field): -> the wrapper's priv_fn(core), or None for a
    plain critic. Without it V(s) would be the NaN a missing block returns."""
    pmode = int(cfg.get("priv_critic") or 0)
    if pmode == 0:
        return None
    if pmode != 2:
        raise SystemExit("--az: a --priv-critic 1 checkpoint (the ten-column block) is not fed "
                         "here, only --priv-critic 2")
    from surfgym.privfeat import PRIV_DIM, PrivFeat
    pd = float(cfg.get("race_pay_d0") or 0.0)
    if not pd > 0.0:
        raise SystemExit("a --priv-critic 2 checkpoint without race_pay_d0 in its config")
    pf = PrivFeat([0.0, 0.0, 0.0], 1.0, pd, 1, mode="dist")

    def priv_fn(c, _g=gf, _pf=pf):
        o = np.asarray(c.states_view["origin"], np.float64).reshape(-1, 3)
        out = np.zeros((len(o), PRIV_DIM), np.float32)
        _pf.fill(out, o, np.zeros_like(o), np.asarray(_g.sample(o), np.float64), np.zeros(len(o)))
        return out
    return priv_fn


def pay_scale(cfg: dict, d0: float) -> float:
    """What one unit of banked progress pays: the checkpoint's own rule - 100 / --race-pay-d0 when
    it was trained in plain units, else 100 / this map's start distance (the trainer's 100/d0)."""
    if float(cfg.get("race_pay_d0") or 0.0) > 0.0:
        return 100.0 / float(cfg["race_pay_d0"])
    return 100.0 / d0 if np.isfinite(d0) and d0 > 0 else 0.0


def value_k(cfg: dict, d0: float):
    """--critic-pay-d0 D: the critic's two columns [V_R, V_P] give V(s) = V_R + k V_P on this map,
    k = this map's progress pay (pay_scale) over the critic's one rate 100 / D (D / d0 under the
    default pay). None for a one-head critic. A map with no start distance has no k: refused."""
    D = float(cfg.get("critic_pay_d0") or 0.0)
    if not D > 0.0:
        return None
    s = pay_scale(cfg, d0)
    if not s > 0.0:
        raise SystemExit("a --critic-pay-d0 checkpoint on a map with no start distance: "
                         "its progress head has no exchange rate here")
    return s * D / 100.0


def ranks(x) -> np.ndarray:
    """Average ranks (ties share their mean rank), 0-based."""
    x = np.asarray(x, np.float64)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[o] = np.arange(len(x), dtype=np.float64)
    for v in np.unique(x):
        t = x == v
        if t.sum() > 1:
            r[t] = r[t].mean()
    return r


def spearman(a, b) -> float:
    """Spearman's rho with average ranks; NaN when either side is constant."""
    ra, rb = ranks(a), ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


class AzTree:
    """Nodes (a state slot, V, the edge reward into it, terminal kind, banked record d, tick) and
    per-node edge tables (K candidates: child id or -1, visits, value sums)."""

    def __init__(self, K: int, heads: int, n_z: int, cap: int = 1 << 16, hold: int = 1):
        self.K, self.heads, self.n_z = int(K), int(heads), int(n_z)
        self.hold = int(hold)
        self.n = 0
        self._alloc(cap)

    def _alloc(self, cap: int):
        K = self.K
        self.parent = np.full(cap, -1, np.int64)
        self.pa = np.full(cap, -1, np.int64)
        self.depth = np.zeros(cap, np.int64)
        self.slot = np.full(cap, -1, np.int64)
        self.V = np.zeros(cap, np.float64)
        self.r = np.zeros(cap, np.float64)
        self.prog = np.zeros(cap, np.float64)         # the progress part of r (--az-death-window)
        self.ndec = np.zeros(cap, np.int64)           # decisions the edge executed (<= hold)
        self.term = np.zeros(cap, np.int8)            # 0 alive, 1 death, 2 finish, 3 cap
        self.rec = np.full(cap, np.inf)               # the episode's banked record d
        self.d = np.full(cap, np.inf)
        self.tick = np.zeros(cap, np.int64)
        self.child = np.full((cap, K), -1, np.int64)
        self.Ne = np.zeros((cap, K), np.float64)
        self.We = np.zeros((cap, K), np.float64)
        self.cand_act = np.zeros((cap, K, self.heads), np.int64)
        self.cand_z = np.zeros((cap, K, self.n_z), np.float32)
        self.prior = np.full((cap, K), 1.0 / K, np.float64)
        self.e_act = None                              # the engine rows / views each edge executed
        self.e_view = np.zeros((cap, self.hold, 2), np.float32)

    def _grow(self, need: int):
        cap = len(self.parent)
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for name in ("parent", "pa", "depth", "slot", "V", "r", "prog", "ndec", "term", "rec",
                     "d", "tick", "child", "Ne", "We", "cand_act", "cand_z", "e_view", "e_act",
                     "prior"):
            a = getattr(self, name)
            if a is None:
                continue
            b = np.zeros((new,) + a.shape[1:], a.dtype)
            if name in ("parent", "pa", "slot", "child"):
                b[:] = -1
            if name in ("rec", "d"):
                b[:] = np.inf
            if name == "prior":
                b[:] = 1.0 / self.K
            b[:cap] = a
            setattr(self, name, b)

    def add(self, m: int) -> np.ndarray:
        self._grow(self.n + m)
        ids = np.arange(self.n, self.n + m)
        self.n += m
        return ids


def run_az(ctx: dict, a) -> dict:
    """The search. ``ctx`` as mcts_search.run_mcts's (built by beam_tas.main); ``a`` = beam_tas's
    args (the --az-* flags). -> {frontier, summary} for beam_tas's --save-frontier."""
    import torch
    from surfgym.core import STATE_DTYPE

    AzPolicy = make_az_policy()
    t_all = time.time()
    K_ACT = int(ctx["K"])                      # ticks per decision
    KC = int(a.az_k)                           # candidates per node
    B = int(ctx["N"])                          # simulations per wave (one env each)
    SIMS = max(B, int(a.az_sims))
    c_puct = float(a.az_cpuct)
    mode = str(a.az_reward)
    if not ctx["VIEWC"]:
        raise SystemExit("--az needs a --view-continuous checkpoint")
    gf = ctx["gf"]
    max_ticks = int(ctx["max_ticks"])
    budget_s = float(a.az_minutes) * 60.0
    cfg = ctx["cfg"]
    # the trainer's own constants: the per-tick discount and time penalty at the checkpoint's tick
    tps = 1000.0 / float(ctx["TICK"].ms)
    gamma_tick = float(cfg.get("gamma", 0.9995)) ** (float(ctx["TICK"].ms) / 10.0)
    gamma_dec = gamma_tick ** K_ACT
    time_pen = float(cfg.get("time_pen") or 0.0) * (float(ctx["TICK"].ms) / 10.0)
    d0 = float(field_d(gf, np.asarray(ctx["row0"]["origin"]).reshape(-1, 3))[0])
    # --race-pay-d0: a checkpoint paid in plain units banks at the same per-unit pay as its
    # critic's training, whatever this map's own start distance
    scale = pay_scale(cfg, d0)
    fin_pay = float(a.az_finish)
    widen = float(a.az_widen)
    prior_mode = str(a.az_prior)
    prior_temp = float(getattr(a, "az_prior_temp", 1.0) or 1.0)
    H_roll = int(a.az_rollout)
    # --az-hold M: decisions per edge (the candidate first, then the policy closed-loop)
    M_HOLD = max(1, int(getattr(a, "az_hold", 1) or 1))
    gamma_edge = gamma_dec ** M_HOLD
    # --az-death-window W: a dying simulation gives back the progress of its last W seconds
    dwin = float(getattr(a, "az_death_window", 0.0) or 0.0)
    if dwin > 0.0 and mode != "banked":
        raise SystemExit("--az-death-window gives back BANKED progress: --az-reward banked")
    W_TK = int(round(dwin * tps)) if dwin > 0.0 else 0
    _pmode = int(cfg.get("priv_critic") or 0)
    priv_fn = priv_feed(cfg, gf)

    def priors(logp):
        # --az-prior policy: each candidate weighted by its probability under the policy
        # (softmax of the log-probs over the node's K candidates); uniform: 1 / K (Sampled
        # MuZero's rule for candidates drawn from the policy itself)
        if prior_mode != "policy":
            return np.full(logp.shape, 1.0 / logp.shape[1])
        # --az-prior-temp T: P ~ pi ** (1 / T); T < 1 sharpens it towards the policy's mode
        x = (logp - logp.max(axis=1, keepdims=True)) / prior_temp
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)
    beta = float(a.az_novelty)
    arch = CellArchive(float(a.mcts_cell), float(a.mcts_cell_energy)) if beta > 0 else None

    def mk_wrapper(core):
        es, ef, fstate, lf = ctx["mk_feed"]()
        xkw, xfe = ctx["XF"].make(core)
        pol = AzPolicy(ctx["policy"], ctx["packer"], ctx["device"], ctx["lidar"], core, K_ACT,
                       ctx["stack"], extra_slot=es, extra_fn=ef, latch_fn=lf,
                       cc_fn=(ctx["cc_fn1"] if core.num_envs == 1 else ctx["cc_fnN"]),
                       keys_hold=ctx["KEYS"], **xkw)
        pol.spawn_duck = ctx["SPAWN_DUCK"]
        pol.greedy = np.zeros(core.num_envs, bool)
        pol.pay_k = value_k(cfg, d0)
        if priv_fn is not None:
            pol.priv_fn = priv_fn
        return pol, SideState(pol, xfe, fstate, lf)

    core = ctx["build_sim"](cfg, ctx["map_path"], B, ctx["ep_cap"], tick=ctx["TICK"])
    ctx["arm"](core)
    obs = np.array(core.reset(int(ctx["gate_seed"])))
    for i in range(B):
        core.set_state(i, ctx["row0"])
    obs[:] = np.asarray(ctx["obs_start"], np.float32)[None, :]
    pol, side = mk_wrapper(core)
    if getattr(a, "az_blank_pot", False):
        # --az-blank-pot: the goal field's image channel (--obs-potential, the second channel of
        # every (H, W, 2) render) set to 0 - under norm the frame's own mean, a field that points
        # nowhere - for the policy AND the critic: what the agent does without its field
        _render = pol.lidar.render

        def _render_blank(*args, **kw):
            out = _render(*args, **kw)
            if out.dim() == 4 and out.shape[-1] >= 2:
                out = out.clone()
                out[..., 1] = 0.0
            return out
        pol.lidar.render = _render_blank
    with torch.inference_mode():
        pol._obs(obs)                       # allocate every lazily-built side array (tick 0)
    root_line = getattr(a, "az_root_line", None)
    if root_line:
        # --az-root-line PATH --az-root-index I: the search starts from a state of a line (a
        # bench of where the agent can go on from), not the spawn. A restored state has no
        # observation of its own, so every env takes ONE tick on the policy's decision from the
        # spawn's scalars and the root is the state after it (one tick of drift); the wrapper
        # then decides afresh. No --save-frontier (that replays from the spawn): the committed
        # states go to az_line_states.npy
        Lz = np.load(root_line, allow_pickle=False)
        Lz = Lz["states"] if str(root_line).endswith(".npz") else Lz
        st = np.array(Lz[int(a.az_root_index)], copy=True).reshape(1).astype(STATE_DTYPE)
        st["tick"] = 0
        st["stuck_ticks"] = 0
        for i in range(B):
            core.set_state(i, st)
        pol._tick, pol._held = 0, None
        act_r = np.ascontiguousarray(pol.act(obs), np.int32)
        obs, _r, dn, tr, _i = core.step(act_r, view=np.ascontiguousarray(pol.view, np.float32))
        obs = np.array(obs)
        if bool(np.asarray(dn)[0]) or bool(np.asarray(tr)[0]):
            raise SystemExit(f"--az-root-line: index {a.az_root_index} ended within one tick")
        pol._tick, pol._held = 0, None
        print(f"az: root = {root_line} index {a.az_root_index} at "
              f"{np.round(core.get_states()[0]['origin'], 1).tolist()} (+1 tick)", flush=True)
    g_step = float(ctx["g_step"])
    _pat = getattr(core, "tick_pattern", None)
    PL = len(_pat) if _pat else 1
    vmax = float(core.config.phys.sv_maxvelocity)
    tol = 1.0
    heads = None
    store = RowStore(STATE_DTYPE, obs.shape[1], cap=1 << 15)
    # ---- the root: the spawn --------------------------------------------------------------
    side0 = side.gather([0])
    slot0 = store.add(core.get_states()[0:1].copy(), obs[0:1].copy(), side0)
    # the wrapper renders every env of the core at once: always the full batch, then the rows
    side_all = side.gather(np.arange(B))
    acts0, z0, v0, lp0 = pol.propose(obs, KC, a.az_keys_temp, a.az_view_temp)
    side.scatter(np.arange(B), side_all)
    heads = acts0.shape[2]
    tree = AzTree(KC, heads, z0.shape[2], hold=M_HOLD)
    root = int(tree.add(1)[0])
    tree.slot[root], tree.V[root] = int(slot0[0]), float(v0[0])
    tree.prior[root] = priors(lp0[0:1])[0]
    tree.cand_act[root], tree.cand_z[root] = acts0[0], z0[0]
    tree.d[root] = tree.rec[root] = d0
    if root_line:
        tree.d[root] = tree.rec[root] = float(field_d(
            gf, np.asarray(core.get_states()[0:1]["origin"], np.float64).reshape(-1, 3))[0])
    line_states = [store.S[int(slot0[0])].copy()]
    q_lo, q_hi = float(v0[0]), float(v0[0]) + 1e-6

    def qnorm(q):
        return (q - q_lo) / max(q_hi - q_lo, 1e-6)

    line_act, line_view = [], []
    dep_means, dep_maxes = [], []
    commits, n_sims, waves = 0, 0, 0
    why = "cap"
    t_wave = 0.0
    n_roll = 0
    print(f"az: {M_HOLD} decision(s) per edge ({K_ACT * M_HOLD} ticks"
          + (": the candidate, then the policy closed-loop" if M_HOLD > 1 else "")
          + f"), {KC} candidates per node (mode + "
          f"{KC - 1} draws at keys temp {a.az_keys_temp:g} / view temp {a.az_view_temp:g}), "
          f"{SIMS} simulations per committed decision in waves of {B}, c_puct {c_puct:g}, "
          f"reward {mode} (scale {scale:.5f}/u, d0 {d0:,.0f}), gamma/decision {gamma_dec:.6f}, "
          f"value = the checkpoint's critic"
          + (f", progressive widening (N + 1) ** {widen:g}" if widen > 0 else "")
          + (", the field's image channel BLANKED" if getattr(a, "az_blank_pot", False) else "")
          + f", prior {prior_mode}" + (f" at temperature {prior_temp:g}" if prior_temp != 1.0 else "")
          + (f", leaf rollouts of {H_roll} decisions (the policy's mode)" if H_roll > 0 else "")
          + (f", a death gives back the progress of its last {dwin:g} s" if W_TK > 0 else "")
          + (", the critic fed d / race_pay_d0" if priv_fn is not None else "")
          + (f", novelty {beta:g} / sqrt(1 + n) per cell ({float(a.mcts_cell):g} u x "
             f"{float(a.mcts_cell_energy):g} u of energy)" if beta > 0 else ""), flush=True)
    while True:
        if tree.term[root]:
            why = {1: "death", 2: "finish", 3: "cap"}[int(tree.term[root])]
            break
        if int(tree.tick[root]) >= max_ticks:
            why = "max-ticks"
            break
        if time.time() - t_all > budget_s:
            why = "budget"
            break
        done_sims = 0
        dep_sum, dep_max = 0, 0
        while done_sims < SIMS:
            # ---- select B leaf edges (virtual loss), back up terminal hits at once ------------
            leaves, paths = [], []
            pend = {}
            for _b in range(B):
                s, path = root, []
                leaf = None
                while True:
                    ne = tree.Ne[s]
                    ns = ne.sum()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        q = np.where(ne > 0, tree.We[s] / np.maximum(ne, 1e-9), tree.V[s])
                    u = qnorm(q) + c_puct * tree.prior[s] * np.sqrt(ns + 1.0) / (1.0 + ne)
                    if widen > 0:
                        u[min(KC, int(np.ceil((ns + 1.0) ** widen))):] = -np.inf
                    for k_ in pend.get(s, ()):
                        u[k_] = -np.inf
                    if not np.isfinite(u).any():
                        break
                    k = int(np.argmax(u))
                    path.append((s, k))
                    tree.Ne[s, k] += 1.0              # virtual visit (undone at backup)
                    c = int(tree.child[s, k])
                    if c < 0:
                        leaf = (s, k)
                        pend.setdefault(s, set()).add(k)
                        break
                    if tree.term[c]:
                        leaf = ("term", c)
                        break
                    s = c
                if leaf is None:
                    for (ps, pk) in path:
                        tree.Ne[ps, pk] -= 1.0
                    continue
                leaves.append(leaf)
                paths.append(path)
                dep_sum += len(path)
                dep_max = max(dep_max, len(path))
            if not leaves:
                break
            exp = [i for i, l in enumerate(leaves) if l[0] != "term"]
            vals = np.zeros(len(leaves), np.float64)
            die_abs = np.full(len(leaves), np.inf)      # each simulation's death tick, inf = none
            new_nodes = {}
            if exp:
                t0 = time.time()
                # one group per PHASE of the tick pattern ([8, 8, 7] ms at 7.63): the core runs
                # one ms sequence for the whole batch, so a node captured T ticks into the
                # episode steps its next ticks with pattern[(T + t) % L] - what an open-loop
                # replay from tick 0 runs - only if its group's batch starts at phase T % L
                groups = {}
                for i in exp:
                    groups.setdefault(int(tree.tick[leaves[i][0]] % PL), []).append(i)
                for ph, gexp in sorted(groups.items()):
                    m = len(gexp)
                    idx = np.arange(m)
                    par = np.asarray([leaves[i][0] for i in gexp], np.int64)
                    kk = np.asarray([leaves[i][1] for i in gexp], np.int64)
                    S, O, SD = store.rows(tree.slot[par])
                    for j in range(m):
                        core.set_state(j, S[j:j + 1])
                    obs[:m] = O
                    side.scatter(idx, SD)
                    pol._tick, pol._held = 0, None
                    core.set_tick_phase(ph)
                    pol.force_idx = idx
                    pol.force_act = tree.cand_act[par, kk]
                    pol.force_z = tree.cand_z[par, kk]
                    alive = np.ones(m, bool)
                    term = np.zeros(m, np.int8)
                    rec = tree.rec[par].copy()
                    d_end = tree.d[par].copy()
                    touch = np.zeros(m, bool)      # the map pushed back during this edge
                    t_end = np.full(m, K_ACT * M_HOLD, np.int64)    # ticks each edge ran
                    e_view = np.zeros((m, M_HOLD, 2), np.float32)
                    e_act = None
                    for t in range(K_ACT * M_HOLD):
                        dec, sub = divmod(t, K_ACT)
                        if sub == 0 and dec == 1:
                            # --az-hold: the rest of the macro is the policy's, closed-loop -
                            # child 0 on its mode, the others on its own sampling
                            pol.force_idx = None
                            gm = np.zeros(B, bool)
                            gm[:m] = kk == 0
                            pol.greedy = gm
                        act = pol.act(obs)             # decides every K_ACT ticks (the first forced)
                        if sub == 0:
                            act_t = np.ascontiguousarray(act, np.int32)
                            view_t = np.ascontiguousarray(pol.view, np.float32)
                            if e_act is None:
                                e_act = np.zeros((m, M_HOLD, act_t.shape[1]), act_t.dtype)
                            e_act[:, dec] = act_t[:m]
                            e_view[:, dec] = view_t[:m]
                        act = act_t
                        vz0 = np.asarray(core.states_view["velocity"][:m, 2], np.float64).copy()
                        obs, _r, dn, tr, _i = core.step(act, view=view_t)
                        obs = np.array(obs)
                        dn = np.asarray(dn)[:m].astype(bool)
                        tr = np.asarray(tr)[:m].astype(bool)
                        st = core.states_view
                        fin = np.asarray(core.goal_hits[:m]).astype(bool)
                        ended = alive & (dn | tr)
                        t_end[ended] = t + 1
                        term[ended & dn & fin] = 2
                        term[ended & dn & ~fin] = 1
                        term[ended & tr & ~dn] = 3
                        live = alive & ~ended
                        if live.any():
                            dd = field_d(gf, np.asarray(st["origin"][:m], np.float64)[live])
                            d_end[live] = dd
                            cm = contact_mask(vz0[live], np.asarray(st["velocity"][:m, 2],
                                                                    np.float64)[live],
                                              g_step, tol, vmax)
                            rr = rec[live]
                            rec[live] = np.where(cm & (dd < rr), dd, rr)
                            touch[live] |= cm
                        alive = live
                        if not alive.any():
                            break
                    pol.force_idx = None
                    pol.greedy = np.zeros(B, bool)
                    states = core.get_states()[:m].copy()
                    side_rows = side.gather(idx)
                    slots = store.add(states, obs[:m].copy(), side_rows)
                    acts_c, z_c, v_c, lp_c = pol.propose(obs, KC, a.az_keys_temp, a.az_view_temp)
                    acts_c, z_c, v_c, lp_c = acts_c[:m], z_c[:m], v_c[:m], lp_c[:m]
                    side.scatter(idx, side_rows)
                    ids = tree.add(m)
                    if tree.e_act is None:
                        tree.e_act = np.zeros((len(tree.parent),) + e_act.shape[1:], e_act.dtype)
                        tree._grow(tree.n)
                    tree.parent[ids], tree.pa[ids] = par, kk
                    tree.depth[ids] = tree.depth[par] + 1
                    tree.slot[ids] = slots
                    tree.tick[ids] = tree.tick[par] + t_end
                    tree.ndec[ids] = (t_end + K_ACT - 1) // K_ACT
                    tree.term[ids] = term
                    tree.rec[ids], tree.d[ids] = rec, d_end
                    tree.cand_act[ids], tree.cand_z[ids] = acts_c, z_c
                    tree.prior[ids] = priors(lp_c)
                    tree.e_act[ids] = e_act
                    tree.e_view[ids] = e_view
                    tree.child[par, kk] = ids
                    gain = np.where(np.isfinite(tree.rec[par]) & np.isfinite(rec),
                                    np.maximum(0.0, tree.rec[par] - rec), 0.0)
                    r = np.zeros(m, np.float64)
                    prog = np.zeros(m, np.float64)
                    if mode == "banked":
                        prog = scale * gain
                        r = prog - time_pen * t_end
                    tree.prog[ids] = prog
                    r = r + np.where(term == 2, fin_pay, 0.0)
                    if arch is not None and ((term == 0) & touch).any():
                        # the tree's memory: a cell seen n times before pays beta / sqrt(1 + n),
                        # counted only where the map pushed back (a surface stood or surfed on):
                        # a fall through new air would otherwise pay its way into the void
                        lv = np.flatnonzero((term == 0) & touch)
                        cid, _nn = arch.visit(arch.keys(states["origin"][lv],
                                                        states["velocity"][lv]))
                        r[lv] += beta / np.sqrt(arch.seen[cid])
                    tree.r[ids] = r
                    v = np.where(term == 0, v_c.astype(np.float64), 0.0)
                    tree.V[ids] = v
                    vleaf = v
                    roll_die = np.full(m, -1, np.int64)   # rollout tick of a death, -1 = none
                    if H_roll > 0 and (term == 0).any():
                        # --az-rollout H: the leaf judged by the policy's own mode played on for H
                        # decisions (classic MCTS's rollout, which AlphaGo mixed with its critic):
                        # a finish pays fin_pay, a death 0, the survivors bootstrap on V
                        live_r = np.zeros(B, bool)
                        live_r[:m] = term == 0
                        g_roll = np.zeros(m, np.float64)
                        disc = np.ones(m, np.float64)
                        rec_r = rec.copy()
                        # --az-death-window: each tick's discounted progress, given back when
                        # the rollout dies within W of it
                        ph = np.zeros((m, H_roll * K_ACT), np.float64) if W_TK > 0 else None
                        pol.greedy = np.ones(B, bool)
                        pol._tick, pol._held = 0, None
                        ob_r = obs
                        for t in range(H_roll * K_ACT):
                            act_r = np.ascontiguousarray(pol.act(ob_r), np.int32)
                            vz0 = np.asarray(core.states_view["velocity"][:m, 2],
                                             np.float64).copy()
                            ob_r, _r, dn, tr, _i = core.step(
                                act_r, view=np.ascontiguousarray(pol.view, np.float32))
                            ob_r = np.array(ob_r)
                            dn = np.asarray(dn)[:m].astype(bool)
                            tr = np.asarray(tr)[:m].astype(bool)
                            fin = np.asarray(core.goal_hits[:m]).astype(bool)
                            lr = live_r[:m]
                            ended = lr & (dn | tr)
                            hit = ended & dn & fin
                            g_roll[hit] += disc[hit] * fin_pay
                            if W_TK > 0:
                                roll_die[ended & dn & ~fin] = t + 1
                            still = lr & ~ended
                            if mode == "banked" and still.any():
                                st = core.states_view
                                dd = field_d(gf, np.asarray(st["origin"][:m], np.float64)[still])
                                cm = contact_mask(vz0[still], np.asarray(
                                    st["velocity"][:m, 2], np.float64)[still], g_step, tol, vmax)
                                rr = rec_r[still]
                                nr = np.where(cm & (dd < rr), dd, rr)
                                gr = scale * np.maximum(0.0, rr - nr)
                                g_roll[still] += disc[still] * (gr - time_pen)
                                if ph is not None:
                                    ph[still, t] = disc[still] * gr
                                rec_r[still] = nr
                            live_r[:m] = still
                            if (t + 1) % K_ACT == 0:
                                disc[still] *= gamma_dec
                            if not live_r.any():
                                break
                        pol.greedy = np.zeros(B, bool)
                        if live_r.any():
                            _a, _z, v_end, _l = pol.propose(ob_r, 1, 1.0, 1.0)
                            lr = live_r[:m]
                            g_roll[lr] += disc[lr] * v_end[:m][lr].astype(np.float64)
                        if ph is not None:
                            for j in np.flatnonzero(roll_die >= 0):
                                g_roll[j] -= ph[j, max(0, int(roll_die[j]) - W_TK):int(roll_die[j])].sum()
                        vleaf = np.where(term == 0, g_roll, 0.0)
                        n_roll += int((term == 0).sum())
                    die_new = np.full(m, np.inf)
                    if W_TK > 0:
                        dt = term == 1
                        die_new[dt] = tree.tick[ids][dt]
                        rd = (roll_die >= 0) & (term == 0)
                        die_new[rd] = tree.tick[ids][rd] + roll_die[rd]
                    for j, i in enumerate(gexp):
                        rj = r[j]
                        if tree.tick[ids[j]] > die_new[j] - W_TK:
                            rj -= prog[j]          # this edge ended within W of the death
                        vals[i] = rj + gamma_edge * vleaf[j]
                        die_abs[i] = die_new[j]
                        new_nodes[i] = int(ids[j])
                t_wave += time.time() - t0
            for i, l in enumerate(leaves):
                if l[0] == "term":
                    c = int(l[1])
                    vals[i] = tree.r[c]
                    if W_TK > 0 and tree.term[c] == 1:
                        vals[i] -= tree.prog[c]
                        die_abs[i] = tree.tick[c]
            # ---- back up (MuZero: G = r + gamma G along the path, the leaf edge first) --------
            for i, path in enumerate(paths):
                g = vals[i]
                td = die_abs[i]
                for depth_i in range(len(path) - 1, -1, -1):
                    ps, pk = path[depth_i]
                    if depth_i < len(path) - 1:
                        c = int(tree.child[ps, pk])
                        rc = tree.r[c]
                        if tree.tick[c] > td - W_TK:
                            rc -= tree.prog[c]     # --az-death-window: paid within W of the death
                        g = rc + gamma_edge * g
                    tree.We[ps, pk] += g
                    q = tree.We[ps, pk] / max(tree.Ne[ps, pk], 1.0)
                    q_lo, q_hi = min(q_lo, q), max(q_hi, q)
            done_sims += len(leaves)
            n_sims += len(leaves)
            waves += 1
        # ---- commit the most visited edge ---------------------------------------------------
        ne = tree.Ne[root]
        if ne.max() <= 0:
            why = "no expansion"
            break
        k = int(np.argmax(ne + 1e-6 * (tree.child[root] >= 0)))
        c = int(tree.child[root, k])
        if c < 0:
            why = "unexpanded best edge"
            break
        dep_mean = dep_sum / max(done_sims, 1)
        dep_means.append(dep_mean)
        dep_maxes.append(dep_max)
        nd = int(tree.ndec[c]) or M_HOLD
        line_act.extend(tree.e_act[c][:nd].copy())
        line_states.append(store.S[int(tree.slot[c])].copy())
        line_view.extend(tree.e_view[c][:nd].copy())
        root = c
        commits += 1
        if commits % int(a.az_log_every) == 0 or tree.term[root]:
            best_rec = float(tree.rec[root])
            print(f"az commit {commits:4d}: tick {int(tree.tick[root]):5d} "
                  f"({tree.tick[root] / tps:6.2f}s), d {float(tree.d[root]):7,.0f}, banked record "
                  f"{best_rec:7,.0f} ({100 * (d0 - best_rec) / d0:5.1f}%), V {tree.V[root]:7.2f}, "
                  f"root visits {int(ne.sum())}, depth mean {dep_mean:4.1f} max {dep_max:3d}, "
                  f"tree {tree.n:,} nodes, sims {n_sims:,}, "
                  + (f"cells {arch.n:,}, " if arch is not None else "")
                  + f"{time.time() - t_all:5.0f}s", flush=True)
    fin = bool(tree.term[root] == 2)
    print(f"az: {why} after {commits} committed decisions ({tree.tick[root] / tps:.2f}s), "
          f"{n_sims:,} simulations in {waves:,} waves, {tree.n:,} nodes, wall "
          f"{time.time() - t_all:.0f}s (expansion {t_wave:.0f}s)", flush=True)
    best = float(tree.rec[root])
    frontier = {"acts": np.asarray(line_act, np.int8) if line_act else np.zeros((0, heads), np.int8),
                "view": np.asarray(line_view, np.float32) if line_view else np.zeros((0, 2), np.float32),
                "tick": int(tree.tick[root]), "d": 0.0 if fin else best, "finished": fin}
    summary = {"az": True, "why": why, "commits": commits, "simulations": n_sims,
               "waves": waves, "nodes": int(tree.n), "finished": fin,
               "end_tick": int(tree.tick[root]), "end_s": float(tree.tick[root] / tps),
               "best_d": best, "d0": d0, "best_pct": 100.0 * (d0 - best) / d0,
               "greedy_pct": float("nan"), "reward": mode, "k": KC, "sims": SIMS,
               "cpuct": c_puct, "keys_temp": float(a.az_keys_temp),
               "view_temp": float(a.az_view_temp), "novelty": beta, "widen": widen,
               "prior": prior_mode, "prior_temp": prior_temp, "rollout": H_roll,
               "hold": M_HOLD, "death_window_s": dwin, "priv_critic": _pmode,
               "pay_scale_per_u": scale, "critic_pay_k": value_k(cfg, d0),
               "blank_pot": bool(getattr(a, "az_blank_pot", False)),
               "rollouts": n_roll,
               "depth_mean": float(np.mean(dep_means)) if dep_means else None,
               "depth_max": int(max(dep_maxes)) if dep_maxes else None,
               "cells": int(arch.n) if arch is not None else None,
               "wall_s": time.time() - t_all}
    od = getattr(a, "out_dir", None)
    if od:
        Path(od).mkdir(parents=True, exist_ok=True)
        np.save(Path(od) / "az_line_states.npy", np.asarray(line_states, STATE_DTYPE))
    if root_line:
        summary.update(root_line=str(root_line), root_index=int(a.az_root_index))
    return {"frontier": frontier, "summary": summary}


def run_critic_probe(ctx: dict, a) -> dict:
    """--az-critic-probe N: the critic measured the way the search uses it, on the map it is given
    (a held-out map: nothing is trained). The user, 2026-11-22: a search on a new map bootstraps on
    the critic at its leaves, so what matters is whether V ranks the branches of a node the way
    their real outcomes rank.

    (1) FLIGHTS: --envs episodes from the spawn (a quarter on the policy's mode, the rest on its own
    sampling) for up to --max-ticks; every --az-probe-every decisions each live episode leaves a
    root (its exact state and the wrapper's per-env state, as a search node). (2) N roots drawn
    from those. (3) From each root --az-k SIBLINGS, the search's own children: one --az-hold macro
    each (child 0 on the mode, the others on the policy's own sampling), then the policy's mode for
    --az-probe-secs seconds. Per child: R = the macro's banked reward (the search's edge reward:
    contact-banked progress x the checkpoint's pay, the time penalty, a finish), V = the critic at
    the macro's end, G = the ACTUAL discounted return after it (the same reward, a death giving
    back its last --az-death-window seconds of progress) at a 2 s and the full horizon, and the
    trainer's own ratchet (every tick's record, no give-back) at the full horizon.

    -> per map: does R + gamma V rank the siblings as R + gamma G ranks them (Spearman per root,
    the top-1 hit rate), against R alone (no critic); and V against G over all children (Pearson,
    Spearman, the slope of G on V: 1 = the critic is in the units of the return on this map)."""
    import torch
    from surfgym.core import STATE_DTYPE
    from train_fast import N_VIEW, sample_view, split_view

    AzPolicy = make_az_policy()

    class ProbePolicy(AzPolicy):
        """AzPolicy whose decisions keep the value of the forward they were made from."""
        last_v = None

        @torch.inference_mode()
        def _decide(self, obs):
            logits, v = self._net(self._obs(obs))
            self.last_v = self._v1(v).cpu().numpy()
            cat, mu = split_view(logits.float())
            padded = self._mask_padded(self.packer.pad(cat), obs)
            act, z, _ = sample_view(padded, mu, self.policy.log_std())
            g = self.greedy
            if g is not None and np.any(g):
                gi = torch.as_tensor(np.flatnonzero(g), device=act.device)
                act[gi, N_VIEW:] = padded[gi, N_VIEW:].argmax(-1)
                z[gi] = mu[gi]
            return self._finish_view(act[:, N_VIEW:], z)

    t_all = time.time()
    K_ACT = int(ctx["K"])
    KC = max(2, int(a.az_k))
    B = int(ctx["N"])
    cfg = ctx["cfg"]
    gf = ctx["gf"]
    tps = 1000.0 / float(ctx["TICK"].ms)
    gamma_tick = float(cfg.get("gamma", 0.9995)) ** (float(ctx["TICK"].ms) / 10.0)
    gamma_dec = gamma_tick ** K_ACT
    time_pen = float(cfg.get("time_pen") or 0.0) * (float(ctx["TICK"].ms) / 10.0)
    d0 = float(field_d(gf, np.asarray(ctx["row0"]["origin"]).reshape(-1, 3))[0])
    scale = pay_scale(cfg, d0)
    fin_pay = float(a.az_finish)
    M_HOLD = max(1, int(getattr(a, "az_hold", 1) or 1))
    gamma_edge = gamma_dec ** M_HOLD
    dwin = float(getattr(a, "az_death_window", 0.0) or 0.0)
    W_TK = int(round(dwin * tps)) if dwin > 0.0 else 0
    H_DEC = max(1, int(round(float(a.az_probe_secs) * tps / K_ACT)))
    SHORT_DEC = max(1, int(round(2.0 * tps / K_ACT)))
    every = max(1, int(a.az_probe_every))
    n_want = int(a.az_critic_probe)
    nb = max(1, B // KC)                       # roots per batch
    priv_fn = priv_feed(cfg, gf)
    rng = np.random.default_rng(int(getattr(a, "seed", 0) or 0))

    core = ctx["build_sim"](cfg, ctx["map_path"], B, ctx["ep_cap"], tick=ctx["TICK"])
    ctx["arm"](core)
    obs = np.array(core.reset(int(ctx["gate_seed"])))
    for i in range(B):
        core.set_state(i, ctx["row0"])
    obs[:] = np.asarray(ctx["obs_start"], np.float32)[None, :]
    es, ef, fstate, lf = ctx["mk_feed"]()
    xkw, xfe = ctx["XF"].make(core)
    pol = ProbePolicy(ctx["policy"], ctx["packer"], ctx["device"], ctx["lidar"], core, K_ACT,
                      ctx["stack"], extra_slot=es, extra_fn=ef, latch_fn=lf,
                      cc_fn=(ctx["cc_fn1"] if core.num_envs == 1 else ctx["cc_fnN"]),
                      keys_hold=ctx["KEYS"], **xkw)
    pol.spawn_duck = ctx["SPAWN_DUCK"]
    pol.pay_k = value_k(cfg, d0)
    if priv_fn is not None:
        pol.priv_fn = priv_fn
    side = SideState(pol, xfe, fstate, lf)
    with torch.inference_mode():
        pol._obs(obs)
    g_step = float(ctx["g_step"])
    _pat = getattr(core, "tick_pattern", None)
    PL = len(_pat) if _pat else 1
    vmax = float(core.config.phys.sv_maxvelocity)
    store = RowStore(STATE_DTYPE, obs.shape[1], cap=4096)

    # ---- (1) flights: the policy's own episodes, a root every `every` decisions ------------------
    n_greedy = max(1, B // 4)
    pol.greedy = np.arange(B) < n_greedy
    pol._tick, pol._held = 0, None
    alive = np.ones(B, bool)
    roots = []                                  # (slot, tick, V, env, d)
    fl_end = np.zeros(B, np.int64)
    fl_fin = np.zeros(B, bool)
    for t in range(int(ctx["max_ticks"])):
        dec, sub = divmod(t, K_ACT)
        take = None
        if sub == 0 and dec > 0 and dec % every == 0:
            take = np.flatnonzero(alive)
            if len(take):
                sl = store.add(core.get_states()[take].copy(), obs[take].copy(),
                               side.gather(take))
        act = np.ascontiguousarray(pol.act(obs), np.int32)
        if take is not None and len(take):
            dd = field_d(gf, np.asarray(core.states_view["origin"], np.float64)[take])
            for j, i in enumerate(take):
                roots.append((int(sl[j]), t, float(pol.last_v[i]), int(i), float(dd[j])))
        obs, _r, dn, tr, _i = core.step(act, view=np.ascontiguousarray(pol.view, np.float32))
        obs = np.array(obs)
        dn = np.asarray(dn).astype(bool)
        tr = np.asarray(tr).astype(bool)
        ended = alive & (dn | tr)
        fl_end[ended] = t + 1
        fl_fin |= ended & dn & np.asarray(core.goal_hits).astype(bool)
        alive &= ~ended
        if not alive.any():
            break
    fl_end[alive] = int(ctx["max_ticks"])
    print(f"critic probe: {B} flights ({n_greedy} on the mode), median "
          f"{np.median(fl_end) / tps:.1f} s, {int(fl_fin.sum())} finished; {len(roots)} roots "
          f"recorded every {every} decisions", flush=True)
    if not roots:
        return {"summary": {"critic_probe": True, "roots": 0, "why": "no root recorded"}}
    pick = (rng.choice(len(roots), size=n_want, replace=False) if len(roots) > n_want
            else np.arange(len(roots)))
    chosen = [roots[i] for i in sorted(pick)]

    # ---- (2) siblings from every root: a macro each, then the mode for H_DEC decisions ----------
    TM = M_HOLD * K_ACT                         # ticks of the macro
    TT = TM + H_DEC * K_ACT                     # ticks of macro + continuation
    out = {k: [] for k in ("root", "k", "R", "R_eff", "R_eff2", "V", "G", "G2", "Gr", "died",
                           "fin", "mac_dead")}
    groups: dict = {}
    for ri, rt in enumerate(chosen):
        groups.setdefault(int(rt[1] % PL), []).append(ri)
    for ph, grp in sorted(groups.items()):
        for c0 in range(0, len(grp), nb):
            chunk = grp[c0:c0 + nb]
            n = len(chunk) * KC
            idx = np.arange(n)
            rid = np.repeat(np.asarray(chunk), KC)
            kk = np.tile(np.arange(KC), len(chunk))
            S, O, SD = store.rows(np.asarray([chosen[r][0] for r in rid], np.int64))
            for j in range(n):
                core.set_state(j, S[j:j + 1])
            obs[:n] = O
            side.scatter(idx, SD)
            core.set_tick_phase(ph)
            pol._tick, pol._held = 0, None
            gm = np.zeros(B, bool)
            gm[:n] = kk == 0
            pol.greedy = gm
            d_root = np.asarray([chosen[r][4] for r in rid], np.float64)
            rec_b = d_root.copy()               # the search's banked record, from the root
            rec_r = d_root.copy()               # the trainer's ratchet record, from the root
            prog = np.zeros((n, TT), np.float64)    # discounted banked progress per tick
            pen = np.zeros((n, TT), np.float64)     # discounted time penalty per tick
            ratg = np.zeros((n, TT), np.float64)    # discounted ratchet progress per tick
            disc_t = np.zeros(TT, np.float64)       # the discount of each tick's reward
            died = np.full(n, -1, np.int64)
            fin = np.full(n, -1, np.int64)
            V = np.zeros(n, np.float64)
            alv = np.ones(n, bool)
            disc = 1.0
            for t in range(TT):
                dec, sub = divmod(t, K_ACT)
                if sub == 0 and dec == M_HOLD:
                    pol.greedy = np.ones(B, bool)
                if sub == 0 and dec > M_HOLD:
                    disc *= gamma_dec
                act = np.ascontiguousarray(pol.act(obs), np.int32)
                if sub == 0 and dec == M_HOLD:
                    V[alv] = pol.last_v[:n][alv]
                disc_t[t] = disc
                vz0 = np.asarray(core.states_view["velocity"][:n, 2], np.float64).copy()
                obs, _r, dn, tr, _i = core.step(act, view=np.ascontiguousarray(pol.view, np.float32))
                obs = np.array(obs)
                dn = np.asarray(dn)[:n].astype(bool)
                tr = np.asarray(tr)[:n].astype(bool)
                gh = np.asarray(core.goal_hits[:n]).astype(bool)
                ended = alv & (dn | tr)
                fin[ended & dn & gh] = t
                died[ended & dn & ~gh] = t
                live = alv & ~ended
                pen[alv, t] = disc * time_pen
                if live.any():
                    st = core.states_view
                    dd = field_d(gf, np.asarray(st["origin"][:n], np.float64)[live])
                    cm = contact_mask(vz0[live], np.asarray(st["velocity"][:n, 2],
                                                            np.float64)[live], g_step, 1.0, vmax)
                    rb = rec_b[live]
                    nb_ = np.where(cm & (dd < rb), dd, rb)
                    prog[live, t] = disc * scale * np.where(np.isfinite(rb),
                                                            np.maximum(0.0, rb - nb_), 0.0)
                    rec_b[live] = nb_
                    rr = rec_r[live]
                    nr = np.where(np.isfinite(dd), np.minimum(rr, dd), rr)
                    ratg[live, t] = disc * scale * np.where(np.isfinite(rr),
                                                            np.maximum(0.0, rr - nr), 0.0)
                    rec_r[live] = nr
                alv = live
                if not alv.any():
                    break
            pol.greedy = np.zeros(B, bool)

            def eff(upto_death):
                """the banked progress with each death's last W given back - for deaths before
                tick upto_death (a truncated horizon does not know a later death)"""
                pe = prog.copy()
                if W_TK > 0:
                    for j in np.flatnonzero((died >= 0) & (died < upto_death)):
                        pe[j, max(0, int(died[j]) - W_TK):int(died[j]) + 1] = 0.0
                return pe

            def seg(pe, a0, a1):
                """reward over ticks [a0, a1): progress - time penalty + a finish in the span"""
                g = pe[:, a0:a1].sum(1) - pen[:, a0:a1].sum(1)
                f = (fin >= a0) & (fin < a1)
                g[f] += fin_pay * disc_t[fin[f]]
                return g
            T2 = TM + SHORT_DEC * K_ACT
            pe_full, pe_mac, pe_2 = eff(TT), eff(TM), eff(T2)
            # R: what the tree knows when the edge is made (a death inside the macro gives back);
            # R_eff / R_eff2: the same edge once the later death (full / 2 s horizon) is known;
            # G, G2: the continuation's own return from the child state, its discount starting
            # at 1 on the macro's end - what V predicts; Gr: the trainer's all-tick ratchet
            for key, val in (("root", rid), ("k", kk), ("R", seg(pe_mac, 0, TM)),
                             ("R_eff", seg(pe_full, 0, TM)), ("R_eff2", seg(pe_2, 0, TM)),
                             ("V", V), ("G", seg(pe_full, TM, TT)), ("G2", seg(pe_2, TM, T2)),
                             ("Gr", seg(ratg, TM, TT)), ("died", died), ("fin", fin),
                             ("mac_dead", (died >= 0) & (died < TM))):
                out[key].append(np.asarray(val))
    res = {k: np.concatenate(v) for k, v in out.items()}
    root_V = np.asarray([r[2] for r in chosen])

    # ---- (3) the numbers ----------------------------------------------------------------------
    def sib_stats(pred, act_):
        rho, top, used = [], [], 0
        for r in range(len(chosen)):
            m_ = res["root"] == r
            if m_.sum() < 2:
                continue
            a_ = act_[m_]
            if np.ptp(a_) <= 1e-9:
                continue
            used += 1
            rho.append(spearman(pred[m_], a_))
            best = np.flatnonzero(a_ >= a_.max() - 1e-9)
            top.append(float(int(np.argmax(pred[m_])) in set(best.tolist())))
        rho = [x for x in rho if np.isfinite(x)]
        return {"roots_used": used, "spearman_mean": float(np.mean(rho)) if rho else None,
                "top1": float(np.mean(top)) if top else None}

    tot_act = res["R_eff"] + gamma_edge * res["G"]
    tot_act2 = res["R_eff2"] + gamma_edge * res["G2"]
    pred_c = res["R"] + gamma_edge * res["V"]
    pred_n = res["R"]
    alive_m = ~res["mac_dead"]
    v, g = res["V"][alive_m], res["G"][alive_m]

    def fit(x, y):
        if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
            return {"n": int(len(x))}
        b, c = np.polyfit(x, y, 1)
        return {"n": int(len(x)), "pearson": float(np.corrcoef(x, y)[0, 1]),
                "spearman": spearman(x, y), "slope": float(b), "intercept": float(c),
                "v_mean": float(np.mean(x)), "g_mean": float(np.mean(y))}
    # V at a root predicts the whole greedy return from it: child 0's macro + continuation
    g0 = tot_act[res["k"] == 0]
    rv0 = root_V[res["root"][res["k"] == 0]]
    summary = {
        "critic_probe": True, "roots": len(chosen), "siblings": KC, "hold": M_HOLD,
        "probe_secs": float(a.az_probe_secs), "death_window_s": dwin, "pay_scale_per_u": scale,
        "d0": d0, "flights": B, "flight_median_s": float(np.median(fl_end) / tps),
        "flights_finished": int(fl_fin.sum()), "roots_recorded": len(roots),
        "children_dead_in_macro": int(res["mac_dead"].sum()),
        "children_died": int((res["died"] >= 0).sum()),
        "children_finished": int((res["fin"] >= 0).sum()),
        "siblings_critic": sib_stats(pred_c, tot_act),
        "siblings_no_critic": sib_stats(pred_n, tot_act),
        "siblings_critic_2s": sib_stats(pred_c, tot_act2),
        "siblings_no_critic_2s": sib_stats(pred_n, tot_act2),
        "v_vs_g": fit(v, g),
        "v_vs_ratchet": fit(v, res["Gr"][alive_m]),
        "root_v_vs_greedy_g": fit(rv0, g0),
        "priv_critic": int(cfg.get("priv_critic") or 0),
        "critic_pay_k": pol.pay_k,
        "wall_s": time.time() - t_all}
    od = getattr(a, "out_dir", None)
    if od:
        Path(od).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(od) / "critic_probe.npz", root_V=root_V,
                            root_tick=np.asarray([r[1] for r in chosen]),
                            root_d=np.asarray([r[4] for r in chosen]), **res)
    return {"summary": summary}
