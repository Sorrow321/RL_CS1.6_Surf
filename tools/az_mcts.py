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
            val = v.float().reshape(n)
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


class AzTree:
    """Nodes (a state slot, V, the edge reward into it, terminal kind, banked record d, tick) and
    per-node edge tables (K candidates: child id or -1, visits, value sums)."""

    def __init__(self, K: int, heads: int, n_z: int, cap: int = 1 << 16):
        self.K, self.heads, self.n_z = int(K), int(heads), int(n_z)
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
        self.e_act = None                              # the engine row / view each edge executed
        self.e_view = np.zeros((cap, 2), np.float32)

    def _grow(self, need: int):
        cap = len(self.parent)
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for name in ("parent", "pa", "depth", "slot", "V", "r", "term", "rec", "d", "tick",
                     "child", "Ne", "We", "cand_act", "cand_z", "e_view", "e_act", "prior"):
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
    scale = 100.0 / d0 if np.isfinite(d0) and d0 > 0 else 0.0
    fin_pay = float(a.az_finish)
    widen = float(a.az_widen)
    prior_mode = str(a.az_prior)
    prior_temp = float(getattr(a, "az_prior_temp", 1.0) or 1.0)
    H_roll = int(a.az_rollout)

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
    tree = AzTree(KC, heads, z0.shape[2])
    root = int(tree.add(1)[0])
    tree.slot[root], tree.V[root] = int(slot0[0]), float(v0[0])
    tree.prior[root] = priors(lp0[0:1])[0]
    tree.cand_act[root], tree.cand_z[root] = acts0[0], z0[0]
    tree.d[root] = tree.rec[root] = d0
    q_lo, q_hi = float(v0[0]), float(v0[0]) + 1e-6

    def qnorm(q):
        return (q - q_lo) / max(q_hi - q_lo, 1e-6)

    line_act, line_view = [], []
    dep_means, dep_maxes = [], []
    commits, n_sims, waves = 0, 0, 0
    why = "cap"
    t_wave = 0.0
    n_roll = 0
    print(f"az: one decision per edge ({K_ACT} ticks), {KC} candidates per node (mode + "
          f"{KC - 1} draws at keys temp {a.az_keys_temp:g} / view temp {a.az_view_temp:g}), "
          f"{SIMS} simulations per committed decision in waves of {B}, c_puct {c_puct:g}, "
          f"reward {mode} (scale {scale:.5f}/u, d0 {d0:,.0f}), gamma/decision {gamma_dec:.6f}, "
          f"value = the checkpoint's critic"
          + (f", progressive widening (N + 1) ** {widen:g}" if widen > 0 else "")
          + (", the field's image channel BLANKED" if getattr(a, "az_blank_pot", False) else "")
          + f", prior {prior_mode}" + (f" at temperature {prior_temp:g}" if prior_temp != 1.0 else "")
          + (f", leaf rollouts of {H_roll} decisions (the policy's mode)" if H_roll > 0 else "")
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
                    touch = np.zeros(m, bool)      # the map pushed back during this decision
                    e_view = None
                    e_act = None
                    for t in range(K_ACT):
                        act = pol.act(obs)             # decides at t = 0 (forced), holds after
                        if t == 0:
                            act_t = np.ascontiguousarray(act, np.int32)
                            e_act = act_t[:m].copy()
                            e_view = np.asarray(pol.view[:m], np.float32).copy()
                            view_t = np.ascontiguousarray(pol.view, np.float32)
                        act = act_t
                        vz0 = np.asarray(core.states_view["velocity"][:m, 2], np.float64).copy()
                        obs, _r, dn, tr, _i = core.step(act, view=view_t)
                        obs = np.array(obs)
                        dn = np.asarray(dn)[:m].astype(bool)
                        tr = np.asarray(tr)[:m].astype(bool)
                        st = core.states_view
                        fin = np.asarray(core.goal_hits[:m]).astype(bool)
                        ended = alive & (dn | tr)
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
                    pol.force_idx = None
                    states = core.get_states()[:m].copy()
                    side_rows = side.gather(idx)
                    slots = store.add(states, obs[:m].copy(), side_rows)
                    acts_c, z_c, v_c, lp_c = pol.propose(obs, KC, a.az_keys_temp, a.az_view_temp)
                    acts_c, z_c, v_c, lp_c = acts_c[:m], z_c[:m], v_c[:m], lp_c[:m]
                    side.scatter(idx, side_rows)
                    ids = tree.add(m)
                    if tree.e_act is None:
                        tree.e_act = np.zeros((len(tree.parent), e_act.shape[1]), e_act.dtype)
                        tree._grow(tree.n)
                    tree.parent[ids], tree.pa[ids] = par, kk
                    tree.depth[ids] = tree.depth[par] + 1
                    tree.slot[ids] = slots
                    tree.tick[ids] = tree.tick[par] + K_ACT
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
                    if mode == "banked":
                        r = scale * gain - time_pen * K_ACT
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
                    if H_roll > 0 and (term == 0).any():
                        # --az-rollout H: the leaf judged by the policy's own mode played on for H
                        # decisions (classic MCTS's rollout, which AlphaGo mixed with its critic):
                        # a finish pays fin_pay, a death 0, the survivors bootstrap on V
                        live_r = np.zeros(B, bool)
                        live_r[:m] = term == 0
                        g_roll = np.zeros(m, np.float64)
                        disc = np.ones(m, np.float64)
                        rec_r = rec.copy()
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
                            still = lr & ~ended
                            if mode == "banked" and still.any():
                                st = core.states_view
                                dd = field_d(gf, np.asarray(st["origin"][:m], np.float64)[still])
                                cm = contact_mask(vz0[still], np.asarray(
                                    st["velocity"][:m, 2], np.float64)[still], g_step, tol, vmax)
                                rr = rec_r[still]
                                nr = np.where(cm & (dd < rr), dd, rr)
                                g_roll[still] += disc[still] * (scale * np.maximum(0.0, rr - nr)
                                                                - time_pen)
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
                        vleaf = np.where(term == 0, g_roll, 0.0)
                        n_roll += int((term == 0).sum())
                    for j, i in enumerate(gexp):
                        vals[i] = r[j] + gamma_dec * vleaf[j]
                        new_nodes[i] = int(ids[j])
                t_wave += time.time() - t0
            for i, l in enumerate(leaves):
                if l[0] == "term":
                    c = int(l[1])
                    vals[i] = tree.r[c]
            # ---- back up (MuZero: G = r + gamma G along the path, the leaf edge first) --------
            for i, path in enumerate(paths):
                g = vals[i]
                for depth_i in range(len(path) - 1, -1, -1):
                    ps, pk = path[depth_i]
                    if depth_i < len(path) - 1:
                        c = int(tree.child[ps, pk])
                        g = tree.r[c] + gamma_dec * g
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
        line_act.append(tree.e_act[c].copy())
        line_view.append(tree.e_view[c].copy())
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
               "blank_pot": bool(getattr(a, "az_blank_pot", False)),
               "rollouts": n_roll,
               "depth_mean": float(np.mean(dep_means)) if dep_means else None,
               "depth_max": int(max(dep_maxes)) if dep_maxes else None,
               "cells": int(arch.n) if arch is not None else None,
               "wall_s": time.time() - t_all}
    return {"frontier": frontier, "summary": summary}
