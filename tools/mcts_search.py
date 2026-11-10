"""mcts_search.py - policy-prior MCTS over macro-actions as Go-Explore's explore step (unr1b3).

The unified-recipe program's slot 3 (autoresearch/PROTOCOL.md): AlphaZero's PUCT as the explore
step, with the policy as the proposal/prior and a short policy rollout as the value. Driven by
``tools/beam_tas.py --mcts`` (which builds the checkpoint-faithful core, lidar, wrappers and side
feeds; nothing here re-implements the observation pipeline).

THE TREE. A node is an EXACT simulator state plus the policy's per-env inference state (the held
keys and their episode tick, the compass block's episode-start d, the ratchet record, the surf
tracker, the obs-reward feed, the latch: everything a wrapper reads off past decisions), stored
row-wise and restored into any env slot of the batched core (SideState). An edge is a MACRO-ACTION
of M decisions (per-decision branching at 25 Hz is hopeless at this horizon): the policy acts
closed-loop for M decisions under a per-child proposal - child 0 of every expansion takes the
policy's MODE (greedy), the others SAMPLE it at keys temperature T_k / view temperature T_v and
hold a yaw offset delta ~ N(0, sigma) degrees (the view is a target in the velocity frame, so a held
offset is a held steer) for the whole macro. The knob from policy-guided to random is (T_k, T_v,
sigma): T = 1, sigma = 0 is the policy's own sampling; large values approach uniform held actions.
Under --keys-hold only the VIEW is edited after the policy decides, so the held keys the policy
reads are always the keys the engine holds.

THE ROOTS. The policy's own greedy line from the true start (a 1-env pass, deterministic), cut where
it dies or stops making progress; the roots are its states every --mcts-root-every decisions over
its last --mcts-roots-secs seconds, children of one virtual root. A node's line = the greedy prefix
up to its root + the macros on its path: replayed open-loop by beam_tas's --save-frontier
(bit-exact physics, checked).

SELECTION. PUCT (Silver et al. 2017; batched with virtual visits, Chaslot et al. 2008):

    U(c) = Qn(c) + c_puct * P(c) * W(c) * sqrt(N(p)) / (1 + N(c))

* P(c) = 1 / (children of p): the children are SAMPLES of the policy, and for actions sampled from
  the policy the prior over the sampled set is uniform (Sampled MuZero, Hubert et al. 2021) - the
  policy is the prior through what it proposes;
* Q(c) = the best BANKED progress anywhere in c's subtree (max backup: a deterministic simulator,
  so the best line found is a line that exists), min-max normalised over the tree (MuZero);
* W(c) = 1 / sqrt(1 + C_seen(cell of c's end state)) - Go-Explore's count weight (Ecoffet et al.
  2021, eq. 4) on the archive cell the child ended in, so a branch that ends where the search has
  been many times is explored less (--mcts-cell-w 0 = off).

THE VALUE is a rollout: the macro itself is a short policy rollout, and a child's score is the
deepest goal-field distance its macro BANKED - the best d at a tick where the map pushed back
(|dvz - one gravity step| above a tolerance, as tools/pick_selfline.py's contact cut; a fall at
the speed cap does not count as contact), so a fatal ballistic fall never scores the depth it falls
to. A finish beats everything.

THE ARCHIVE. Every decision-boundary state of every macro is a Go-Explore cell (position / --cell
u, energy height h = z + |v|^2/1600 / --cell-energy u): C_seen counts visits (the selection
weight), and the summary reports how many cells the search found - the coverage number to compare
explorers with. Nothing in this file names a map, a place, a speed or a progress value; every
constant is a search budget, a cell size or a tolerance and ships unchanged to a new map.

THE CHILD MIX (azr1b2, 2026-11-09). Each expansion's SAMPLED children (ranks 1..K-1; child 0 is
always the policy's mode) are of three kinds: P = a plain policy sample (temperature 1), E = a
perturbed policy sample (--mcts-keys-temp + a held steer), R = random-held (the first decision
drawn at --mcts-random-temp on every head, held for the macro). Runs 4/5 fixed the shares
(--mcts-perturb-frac / --mcts-random-frac). --mcts-mix-rule sib sets them by ONE rule from the
search's own results (MixRule): after every wave, in each expansion the child that did better
than all its siblings (child 0 included) earns its kind one point - "better" on the tree's own
value (the backed-up banked score) when some child improved on its parent's value, else (nobody
went deeper) on the new archive cells a LIVE child opened; ties split the point; a win of child 0
is credited to no kind. Each kind's share of the next wave's sampled children is probability
matching on its points per child (the memory counted in wins: the last ~--mcts-mix-horizon wins
decide), with a floor (--mcts-mix-floor) so that no kind dies out. Off (the default) draws the
kinds exactly as before.
"""
from __future__ import annotations

import math
import time

import numpy as np

G_ENERGY = 800.0           # sv_gravity: h = z + |v|^2 / (2 g)

# the kinds of child: 0 = child 0 (the policy's mode), 1 = P (policy sample), 2 = E (perturbed),
# 3 = R (random-held); MODE_CHARS[k] names them in logs and in the best line's composition
MODE_GREEDY, MODE_POLICY, MODE_PERTURB, MODE_RANDOM = 0, 1, 2, 3
MODE_CHARS = "GPER"


# --------------------------------------------------------------------------------------------
# the child mix: the fixed knob of runs 4/5, and the rule (--mcts-mix-rule)
# --------------------------------------------------------------------------------------------
def legacy_child_kinds(rng, rank, perturb_frac: float, random_frac: float):
    """The fixed knob of runs 4/5, draw for draw (two uniform draws over the whole batch, the
    perturb draw first): -> (explore, rnd) masks. Kept verbatim so that --mcts-mix-rule off
    consumes the same random stream as before (tests/python/test_mcts_mix.py)."""
    n = len(rank)
    explore = (rank > 0) & (rng.random(n) < float(perturb_frac))
    rnd = (rank > 0) & (rng.random(n) < float(random_frac))
    explore &= ~rnd
    return explore, rnd


def kinds_from_masks(rank, explore, rnd) -> np.ndarray:
    """-> per-env kind (MODE_*) from the legacy masks."""
    k = np.where(np.asarray(rnd, bool), MODE_RANDOM,
                 np.where(np.asarray(explore, bool), MODE_PERTURB, MODE_POLICY))
    return np.where(np.asarray(rank) > 0, k, MODE_GREEDY).astype(np.int8)


def mix_shares(wins, children, floor: float, prev=None) -> np.ndarray:
    """Probability matching with a floor: the (P, E, R) shares from each kind's points per child.
    share_k = floor + (1 - 3 floor) * y_k / sum(y), y_k = wins_k / children_k (0 for a kind with
    no children). No evidence at all (sum y = 0) keeps ``prev`` (equal shares if None)."""
    w = np.asarray(wins, np.float64)
    c = np.asarray(children, np.float64)
    y = np.where(c > 0, w / np.maximum(c, 1e-12), 0.0)
    s = float(y.sum())
    if s <= 0.0 or not np.isfinite(s):
        return (np.full(3, 1.0 / 3.0) if prev is None else np.asarray(prev, np.float64).copy())
    fl = float(floor)
    return fl + (1.0 - 3.0 * fl) * y / s


def sibling_credit(own, sc, par_sc, alive, n_new, tol: float = 1e-9):
    """One wave's tournament. ``own`` (N,) the leaf each env expanded, ``sc`` (N,) each child's
    score (the tree's value), ``par_sc`` (N,) its leaf's score, ``alive`` (N,) alive at the macro's
    end, ``n_new`` (N,) new archive cells its macro opened. Per leaf: if some child's score beats
    the leaf's own by more than ``tol``, the best-scoring children win (a dead child that banked
    deeper counts: its line exists up to the banked tick); else the LIVE children that opened the
    most new cells win (none if no live child opened one). Ties split the point.
    -> (credit (N,) float, how (N,) int8: 1 = won on value, 2 = won on new cells, 0 = no win)."""
    own = np.asarray(own, np.int64)
    sc = np.asarray(sc, np.float64)
    par_sc = np.asarray(par_sc, np.float64)
    alive = np.asarray(alive, bool)
    n_new = np.asarray(n_new, np.float64)
    N = len(own)
    credit = np.zeros(N, np.float64)
    how = np.zeros(N, np.int8)
    if N == 0:
        return credit, how
    order = np.argsort(own, kind="stable")
    own_s = own[order]
    starts = np.flatnonzero(np.r_[True, own_s[1:] != own_s[:-1]])
    ends = np.r_[starts[1:], N]
    for s, e in zip(starts.tolist(), ends.tolist()):
        ch = order[s:e]
        v = sc[ch]
        best = float(np.max(v))
        if np.isfinite(best) and best > float(par_sc[ch[0]]) + tol:
            win = ch[v >= best - tol]
            credit[win] = 1.0 / len(win)
            how[win] = 1
            continue
        nn = np.where(alive[ch], n_new[ch], 0.0)
        m = float(nn.max())
        if m > 0.0:
            win = ch[nn >= m]
            credit[win] = 1.0 / len(win)
            how[win] = 2
    return credit, how


class MixRule:
    """--mcts-mix-rule: the (P, E, R) shares of the sampled children.

    ``mode`` 'sib' = the rule (draw from the shares, update them every wave); 'log' = the legacy
    fixed knob (``legacy_child_kinds``, the same random stream as 'off') with the rule's
    bookkeeping beside it, so a run at the fixed knob reports which kinds won (a diagnostic).
    ``floor``: the least share of each kind; ``horizon``: the memory, counted in WINS (expansions
    that had a winner, child 0's included): each win decays the old evidence (points AND
    children) by exp(-1 / horizon), so the shares follow the last ~horizon wins - fast while the
    tree is finding things (tens of wins per wave), steady through the long tail where wins are
    rare (a memory counted in waves let a handful of tail wins swing the shares: bench cp6,
    2026-11-09 21:10, P 0.2 -> 0.88 -> 0.2 within 100 waves; run 5's wall-opening lines were
    found in that tail, search 15 at wave ~392, search 20 at ~832). A wave without a win changes
    nothing: children drawn in proportion to the shares leave the ratios where they are.
    ``warm``: waves at equal shares before the rule speaks. Every constant is a search budget or
    a sample size: nothing here reads a map, a place or a progress value."""

    def __init__(self, mode: str = "sib", floor: float = 0.05, horizon: float = 256.0,
                 warm: int = 2, perturb_frac: float = 0.5, random_frac: float = 0.5):
        if mode not in ("sib", "log"):
            raise ValueError(f"MixRule mode {mode!r}")
        if not 0.0 <= float(floor) < 1.0 / 3.0:
            raise ValueError("--mcts-mix-floor must be in [0, 1/3)")
        self.mode = mode
        self.floor = float(floor)
        self.horizon = max(1.0, float(horizon))
        self.warm = int(warm)
        self.pf, self.rf = float(perturb_frac), float(random_frac)
        self.share = np.full(3, 1.0 / 3.0)            # (P, E, R)
        if mode == "log":
            # the fixed knob's own shares (the legacy draw: R first, then E of the rest)
            self.share = np.array([(1.0 - self.rf) * (1.0 - self.pf),
                                   (1.0 - self.rf) * self.pf, self.rf])
        self.w = np.zeros(3)                          # decayed points
        self.c = np.zeros(3)                          # decayed children
        self.waves = 0
        self.tot_children = np.zeros(3, np.int64)
        self.tot_wins = np.zeros(4)                   # (G, P, E, R)
        self.tot_wins_value = np.zeros(4)
        self.tot_wins_cells = np.zeros(4)
        self.tot_new_cells = np.zeros(4)
        self.tot_alive = np.zeros(4)
        self.tot_n = np.zeros(4)
        self.hist = []                                # (wave, P, E, R) after each update
        self.share_sum = np.zeros(3)

    def draw(self, rng, rank) -> np.ndarray:
        """-> per-env kind (MODE_*); rank 0 is always child 0 (the policy's mode)."""
        rank = np.asarray(rank)
        if self.mode == "log":
            ex, rn = legacy_child_kinds(rng, rank, self.pf, self.rf)
            return kinds_from_masks(rank, ex, rn)
        u = rng.random(len(rank))
        cum = np.cumsum(self.share)
        k = np.minimum(np.searchsorted(cum, u * cum[-1], side="right"), 2)
        return np.where(rank > 0, 1 + k, MODE_GREEDY).astype(np.int8)

    def update(self, kinds, credit, how, alive, n_new) -> None:
        """One wave's evidence (``sibling_credit``'s output for its children)."""
        kinds = np.asarray(kinds, np.int64)
        credit = np.asarray(credit, np.float64)
        how = np.asarray(how)
        alive = np.asarray(alive, bool)
        n_new = np.asarray(n_new, np.float64)
        wins = np.zeros(4)
        for k in range(4):
            m = kinds == k
            wins[k] = float(credit[m].sum())
            self.tot_wins_value[k] += float(credit[m & (how == 1)].sum())
            self.tot_wins_cells[k] += float(credit[m & (how == 2)].sum())
            self.tot_new_cells[k] += float(n_new[m].sum())
            self.tot_alive[k] += float(alive[m].sum())
            self.tot_n[k] += float(m.sum())
        self.tot_wins += wins
        ch = np.array([(kinds == k).sum() for k in (1, 2, 3)], np.float64)
        self.tot_children += ch.astype(np.int64)
        # the memory is counted in WINS: this wave's wins (every kind's, child 0's included)
        # decay the old points and children alike, then this wave's evidence enters in full
        f = math.exp(-float(wins.sum()) / self.horizon)
        self.w = self.w * f + wins[1:]
        self.c = self.c * f + ch
        self.waves += 1
        if self.mode == "sib" and self.waves >= self.warm:
            self.share = mix_shares(self.w, self.c, self.floor, prev=self.share)
        self.share_sum += self.share
        self.hist.append((self.waves - 1, *[float(x) for x in self.share]))

    def text(self) -> str:
        p, e, r = self.share
        return f"mix P/E/R {p:.2f}/{e:.2f}/{r:.2f}"

    def summary(self, path_kinds: str = "") -> dict:
        """The summary.json block (only written when the flag is on)."""
        n = max(1, len(self.hist))
        pick = (np.unique(np.linspace(0, len(self.hist) - 1, min(len(self.hist), 60))
                          .round().astype(int)) if self.hist else [])
        names = "GPER"

        def per(a, ks=(0, 1, 2, 3), nd=3):
            return {names[k]: round(float(a[k]), nd) for k in ks}

        return {"mix_rule": self.mode, "mix_floor": self.floor, "mix_horizon": self.horizon,
                "mix_warm": self.warm,
                "mix_share_mean": {k: round(float(v) / n, 4) for k, v in
                                   zip("PER", self.share_sum)},
                "mix_share_final": {k: round(float(v), 4) for k, v in zip("PER", self.share)},
                "mix_share_by_wave": [[int(self.hist[i][0])] + [round(x, 3)
                                                                for x in self.hist[i][1:]]
                                      for i in pick],
                "mix_children": {k: int(v) for k, v in zip("PER", self.tot_children)},
                "mix_wins": per(self.tot_wins, nd=1),
                "mix_wins_value": per(self.tot_wins_value, nd=1),
                "mix_wins_cells": per(self.tot_wins_cells, nd=1),
                "mix_wins_per_child": {names[k]: round(float(self.tot_wins[k])
                                                       / max(1.0, float(self.tot_n[k])), 4)
                                       for k in range(4)},
                "mix_new_cells_per_child": {names[k]: round(float(self.tot_new_cells[k])
                                                            / max(1.0, float(self.tot_n[k])), 4)
                                            for k in range(4)},
                "mix_alive_end": {names[k]: round(float(self.tot_alive[k])
                                                  / max(1.0, float(self.tot_n[k])), 4)
                                  for k in range(4)},
                "best_path_kinds": path_kinds}


# --------------------------------------------------------------------------------------------
# the per-env inference state of a wrapper, row-wise
# --------------------------------------------------------------------------------------------
class SideState:
    """Gather / scatter EVERY per-env array a policy wrapper and its feeds read off past
    decisions, by name, so a node can be stored and restored into any env slot.

    ``pol``: the wrapper (train_fast._TorchPolicyBase); ``xfe``: its beam_tas.XBundle;
    ``feed_state``: the --obs-reward feed's state dict (or None); ``latch``: the --race-latch fn
    (or None). Arrays are looked up at access time - several feeds REPLACE their arrays on every
    call (the ratchet record, _keys_tick), so a cached reference would go stale."""

    def __init__(self, pol, xfe, feed_state=None, latch=None):
        self.pol, self.xfe, self.feed_state, self.latch = pol, xfe, feed_state, latch

    def _slots(self):
        """-> [(name, getter, setter)]; getter() returns the live array (np or torch) or None."""
        out = []
        p = self.pol
        if getattr(p, "keys", None) is not None:
            out.append(("keys.state", lambda: p.keys.state, None))
            out.append(("keys.boot", lambda: p.keys.boot, None))
        if getattr(p, "_keys_tick", None) is not None:
            def _set_kt(v, p=p):
                p._keys_tick = v
            out.append(("keys.tick", lambda: p._keys_tick, _set_kt))
        if self.xfe is not None:
            for i, (o, k, is_dict) in enumerate(self.xfe._slots()):
                if is_dict:
                    out.append((f"x{i}.{k}", (lambda o=o, k=k: o[k]), None))
                else:
                    out.append((f"x{i}.{k}", (lambda o=o, k=k: getattr(o, k)), None))
        fs = self.feed_state
        if fs is not None:
            out.append(("feed.d", lambda: fs.get("d"), None))
        if self.latch is not None and getattr(self.latch, "state", None) is not None:
            st = self.latch.state
            out.append(("latch.f", lambda: st.get("f"), None))
            out.append(("latch.tick", lambda: st.get("tick"), None))
        return out

    def names(self):
        return [n for n, g, _s in self._slots() if g() is not None]

    def gather(self, idx):
        """-> {name: rows} of the env rows ``idx`` (copies; torch tensors come back as numpy)."""
        idx = np.asarray(idx, np.int64)
        out = {}
        for name, get, _set in self._slots():
            a = get()
            if a is None:
                continue
            if hasattr(a, "detach"):
                out[name] = a.detach().cpu().numpy()[idx].copy()
            else:
                out[name] = np.asarray(a)[idx].copy()
        return out

    def scatter(self, idx, rows: dict):
        """Write ``rows`` (gather's output, one row per entry of ``idx``) into the env rows
        ``idx`` in place. A name the live wrapper does not hold yet is an error: run one
        decision first (the arrays are allocated lazily)."""
        idx = np.asarray(idx, np.int64)
        for name, get, _set in self._slots():
            if name not in rows:
                continue
            a = get()
            if a is None:
                raise RuntimeError(f"SideState.scatter: {name} is not allocated on this wrapper")
            r = rows[name]
            if hasattr(a, "detach"):
                import torch
                with torch.inference_mode():
                    a[torch.as_tensor(idx, device=a.device)] = torch.as_tensor(
                        r, device=a.device, dtype=a.dtype)
            else:
                a[idx] = r


class RowStore:
    """Growable row storage for node states: the core's STATE_DTYPE row, the scalar obs row and
    every SideState array, indexed by a slot id."""

    def __init__(self, state_dtype, obs_dim: int, cap: int = 4096):
        self.state_dtype, self.obs_dim = state_dtype, int(obs_dim)
        self.n = 0
        self.S = np.zeros(cap, state_dtype)
        self.O = np.zeros((cap, self.obs_dim), np.float32)
        self.side: dict = {}

    def _grow(self, need: int):
        cap = len(self.S)
        if need <= cap:
            return
        new = max(need, 2 * cap)
        S = np.zeros(new, self.state_dtype)
        S[:cap] = self.S
        self.S = S
        O = np.zeros((new, self.obs_dim), np.float32)
        O[:cap] = self.O
        self.O = O
        for k, a in self.side.items():
            b = np.zeros((new,) + a.shape[1:], a.dtype)
            b[:cap] = a
            self.side[k] = b

    def add(self, states, obs, side: dict) -> np.ndarray:
        m = len(states)
        self._grow(self.n + m)
        sl = np.arange(self.n, self.n + m)
        self.S[sl] = states
        self.O[sl] = obs
        for k, r in side.items():
            if k not in self.side:
                self.side[k] = np.zeros((len(self.S),) + r.shape[1:], r.dtype)
            self.side[k][sl] = r
        self.n += m
        return sl

    def rows(self, slots) -> tuple:
        slots = np.asarray(slots, np.int64)
        return self.S[slots], self.O[slots], {k: a[slots] for k, a in self.side.items()}


# --------------------------------------------------------------------------------------------
# the cell archive
# --------------------------------------------------------------------------------------------
class CellArchive:
    """Go-Explore cells over generic quantities: position / cell u and energy height / ecell u.
    Dense cell ids (in discovery order), visit counts, and the first (node, decision) that
    reached each cell."""

    def __init__(self, cell: float = 128.0, ecell: float = 256.0):
        self.cell, self.ecell = float(cell), float(ecell)
        self.index: dict = {}
        self.seen = np.zeros(1024, np.float64)
        self.first_node = np.full(1024, -1, np.int64)
        self.n = 0

    def keys(self, origin, velocity) -> np.ndarray:
        o = np.asarray(origin, np.float64).reshape(-1, 3)
        v = np.asarray(velocity, np.float64).reshape(-1, 3)
        h = o[:, 2] + (v * v).sum(axis=1) / (2.0 * G_ENERGY)
        q = np.floor(o / self.cell).astype(np.int64) + (1 << 15)
        e = np.floor(h / self.ecell).astype(np.int64) + (1 << 15)
        return (q[:, 0] << 48) | (q[:, 1] << 32) | (q[:, 2] << 16) | (e & 0xFFFF)

    def visit(self, keys, nodes=None, is_new=None) -> tuple[np.ndarray, int]:
        """Count one visit per key; -> (dense ids, number of NEW cells). ``is_new``: an optional
        (len(keys),) bool array set True where the key opened a new cell."""
        keys = np.asarray(keys, np.int64)
        ids = np.empty(len(keys), np.int64)
        new = 0
        for j, k in enumerate(keys.tolist()):
            c = self.index.get(k)
            if c is None:
                c = self.n
                self.index[k] = c
                self.n += 1
                if c >= len(self.seen):
                    self.seen = np.concatenate([self.seen, np.zeros_like(self.seen)])
                    self.first_node = np.concatenate([self.first_node,
                                                      np.full_like(self.first_node, -1)])
                if nodes is not None:
                    self.first_node[c] = int(nodes[j])
                if is_new is not None:
                    is_new[j] = True
                new += 1
            ids[j] = c
        np.add.at(self.seen, ids, 1.0)
        return ids, new

    def weight(self, ids) -> np.ndarray:
        ids = np.asarray(ids, np.int64)
        w = np.ones(len(ids), np.float64)
        ok = ids >= 0
        w[ok] = 1.0 / np.sqrt(1.0 + self.seen[ids[ok]])
        return w


# --------------------------------------------------------------------------------------------
# the tree (pure numpy: tests/python/test_mcts_search.py)
# --------------------------------------------------------------------------------------------
class Tree:
    """PUCT tree with max backup and batched selection under virtual visits.

    Node 0 is the virtual root. Children of a node are added in one contiguous block. A node is
    EXPANDABLE when it holds a stored state (alive), has no children yet and is not pending in
    the current wave; DEAD / FINISHED nodes are terminal; EXHAUSTED marks a subtree with no
    expandable node left (never descended into again).

    ``use_qv`` (--mcts-leaf-value, azr1b4): selection reads ``qv`` instead of ``q``. A new child's
    ``qv`` is the caller's (the network's value at its end state, never below its own score); an
    expanded node's ``qv`` is the max of its children's, recomputed up the path on every expansion
    (so an optimistic guess is replaced by what its children show, and can go DOWN - unlike ``q``,
    the real banked max, which only rises and which every label and the best line read). Off
    (False, the default): ``qv`` is filled and never read."""

    def __init__(self, cap: int = 4096):
        self.n = 1
        self._cap = 0
        self._alloc(max(16, int(cap)))
        self.q[0] = -np.inf
        self.score[0] = -np.inf
        self.qlo, self.qhi = np.inf, -np.inf
        self.use_qv = False

    _F = (("parent", np.int64, -1), ("cstart", np.int64, -1), ("ncld", np.int64, 0),
          ("depth", np.int64, 0), ("visits", np.float64, 0.0), ("vloss", np.float64, 0.0),
          ("q", np.float64, -np.inf), ("score", np.float64, -np.inf),
          ("alive", bool, False), ("pending", bool, False), ("exhausted", bool, False),
          ("blocked", bool, False),
          ("finished", bool, False), ("slot", np.int64, -1), ("cell", np.int64, -1),
          ("tick", np.int64, 0), ("bank_d", np.float64, np.inf), ("end_d", np.float64, np.inf),
          ("h", np.float64, 0.0), ("bank_t", np.int64, -1), ("ndec", np.int64, 0),
          ("root_dec", np.int64, -1), ("wave", np.int64, -1), ("delta", np.float64, 0.0),
          ("nov", np.float64, 0.0), ("ox", np.float32, 0.0), ("oy", np.float32, 0.0),
          ("oz", np.float32, 0.0), ("raw_d", np.float64, np.inf),
          ("h_bank", np.float64, np.nan), ("kind", np.int8, -1), ("qv", np.float64, -np.inf))

    def _alloc(self, cap: int):
        for name, dt, fill in self._F:
            a = np.full(cap, fill, dt)
            if self._cap:
                a[:self._cap] = getattr(self, name)
            setattr(self, name, a)
        self._cap = cap

    def _grow(self, need: int):
        if need > self._cap:
            self._alloc(max(need, 2 * self._cap))

    def expandable(self, i: int) -> bool:
        return bool(self.alive[i] and self.ncld[i] == 0 and not self.pending[i])

    def add_children(self, parent: int, score, alive, finished=None, **fields) -> np.ndarray:
        """Append one contiguous block of children under ``parent`` -> their ids. ``fields``:
        any per-node array of _F (slot, cell, tick, bank_d, end_d, h, bank_t, ndec, root_dec,
        wave). Backs their scores up the path (max) and counts one visit per child."""
        score = np.asarray(score, np.float64).reshape(-1)
        m = len(score)
        if m == 0:
            return np.zeros(0, np.int64)
        if self.ncld[parent] != 0:
            raise ValueError(f"node {parent} already has children")
        self._grow(self.n + m)
        ids = np.arange(self.n, self.n + m)
        self.n += m
        self.parent[ids] = parent
        self.depth[ids] = self.depth[parent] + 1
        self.score[ids] = score
        self.q[ids] = score
        self.visits[ids] = 1.0
        al = np.asarray(alive, bool).reshape(-1)
        fin = (np.zeros(m, bool) if finished is None
               else np.asarray(finished, bool).reshape(-1))
        self.alive[ids] = al & ~fin
        self.finished[ids] = fin
        self.exhausted[ids] = ~(al & ~fin)
        self.qv[ids] = score
        for k, v in fields.items():
            getattr(self, k)[ids] = v
        self.cstart[parent] = ids[0]
        self.ncld[parent] = m
        self.pending[parent] = False
        fin_s = score[np.isfinite(score)]
        if len(fin_s):
            self.qlo = min(self.qlo, float(fin_s.min()))
            self.qhi = max(self.qhi, float(fin_s.max()))
        # backup: one visit per child, max of the block's scores, the SUM of its novelty
        # rewards (a subtree's novelty value is that sum over its visits: the rate at which
        # expanding it still opens new archive cells)
        best = float(score.max()) if len(score) else -np.inf
        nsum = float(self.nov[ids].sum())
        p = int(parent)
        while p >= 0:
            self.visits[p] += m
            self.nov[p] += nsum
            if best > self.q[p]:
                self.q[p] = best
            p = int(self.parent[p])
        if self.use_qv:
            self.qv[ids] = np.maximum(self.qv[ids], score)
            fin_v = self.qv[ids][np.isfinite(self.qv[ids])]
            if len(fin_v):
                self.qlo = min(self.qlo, float(fin_v.min()))
                self.qhi = max(self.qhi, float(fin_v.max()))
            # the expanded node's guess is replaced by its children's values, and every ancestor
            # takes the max of its children again (stop where nothing changed)
            p = int(parent)
            while p >= 0:
                cs, nc = int(self.cstart[p]), int(self.ncld[p])
                new = float(self.qv[cs:cs + nc].max()) if nc else float(self.qv[p])
                if new == float(self.qv[p]) and p != int(parent):
                    break
                self.qv[p] = new
                p = int(self.parent[p])
        if not self.alive[ids].any():
            self._exhaust_up(int(parent))
        return ids

    def _exhaust_up(self, i: int):
        """Mark ``i`` exhausted when nothing below it can be expanded, and walk up."""
        while i >= 0:
            if self.alive[i] and self.ncld[i] == 0:
                return                       # expandable (or pending) itself
            if self.ncld[i] == 0 and self.pending[i]:
                return
            cs, nc = int(self.cstart[i]), int(self.ncld[i])
            if nc and not self.exhausted[cs:cs + nc].all():
                return
            self.exhausted[i] = True
            i = int(self.parent[i])

    def select(self, n_leaves: int, c_puct: float = 1.0, cell_w=None, max_fail: int = None,
               novelty: float = 0.0):
        """Up to ``n_leaves`` distinct expandable leaves by PUCT descents from the root, each
        descent adding one virtual visit along its path. ``cell_w``: f(cell ids) -> weights
        (CellArchive.weight) or None. Leaves come back marked pending."""
        leaves = []
        fails = 0
        max_fail = int(max_fail if max_fail is not None else max(8, n_leaves))
        span = (self.qhi - self.qlo) if np.isfinite(self.qhi - self.qlo) else 0.0
        lo = self.qlo if np.isfinite(self.qlo) else 0.0
        inv = 1.0 / span if span > 1e-12 else 0.0
        while (len(leaves) < n_leaves and fails < max_fail and not self.exhausted[0]
               and not self.blocked[0]):
            node, path = 0, [0]
            ok = False
            while True:
                if self.expandable(node):
                    ok = True
                    break
                nc = int(self.ncld[node])
                if nc == 0:
                    break                     # pending leaf / terminal
                cs = int(self.cstart[node])
                ch = np.arange(cs, cs + nc)
                selm = (~self.exhausted[ch] & ~self.blocked[ch]
                        & ~(self.pending[ch] & (self.ncld[ch] == 0)))
                if not selm.any():
                    if (self.pending[ch] & (self.ncld[ch] == 0)).any() or self.blocked[ch].any():
                        # everything below is pending this wave: block the node for the rest of
                        # the wave, so the next descent spreads to its siblings instead of
                        # failing here again (batched selection, one wave = one batch)
                        self.blocked[node] = True
                    else:
                        self._exhaust_up(node)
                    break
                ch = ch[selm]
                _qs = self.qv[ch] if self.use_qv else self.q[ch]
                qn = (np.where(np.isfinite(_qs), _qs, lo) - lo) * inv
                npar = self.visits[node] + self.vloss[node]
                nch = self.visits[ch] + self.vloss[ch]
                if cell_w is not None:
                    # Go-Explore's 1/sqrt(1 + visits), RELATIVE among the siblings: the least
                    # visited end cell keeps the full exploration term (an absolute weight
                    # vanishes once every cell has thousands of visits)
                    w = cell_w(self.cell[ch])
                    w = w / max(float(w.max()), 1e-12)
                else:
                    w = 1.0
                u = qn + c_puct * (1.0 / nc) * w * math.sqrt(max(npar, 1.0)) / (1.0 + nch)
                if novelty > 0.0:
                    # the novelty value: new cells per visit in the child's subtree
                    u = u + novelty * self.nov[ch] / np.maximum(self.visits[ch], 1.0)
                node = int(ch[int(np.argmax(u))])
                path.append(node)
            pth = np.asarray(path, np.int64)
            self.vloss[pth] += 1.0
            if ok:
                self.pending[node] = True
                leaves.append(node)
            else:
                fails += 1
        return leaves

    def end_wave(self):
        self.vloss[:self.n] = 0.0
        self.pending[:self.n] = False
        self.blocked[:self.n] = False

    def path_to(self, i: int) -> list:
        out = []
        while i > 0:
            out.append(int(i))
            i = int(self.parent[i])
        return out[::-1]

    def best_node(self) -> int:
        """The finished node reached earliest (tick), else the node with the best banked score
        (ties: the earliest bank tick)."""
        n = self.n
        fin = np.flatnonzero(self.finished[1:n]) + 1
        if len(fin):
            return int(fin[np.argmin(self.tick[fin])])
        bd = self.bank_d[1:n]
        if not np.isfinite(bd).any():
            return -1
        # the DEEPEST banked node (the score may carry an energy term for the search's guidance;
        # the line handed on is judged on depth, as the trainer compares it)
        top = np.flatnonzero(bd <= np.nanmin(bd[np.isfinite(bd)]) + 1e-9) + 1
        return int(top[np.argmin(np.where(self.bank_t[top] >= 0, self.bank_t[top],
                                          self.tick[top]))])


# --------------------------------------------------------------------------------------------
# --mcts-labels: the tree's own training targets (azr1b4, the tree-only loop)
# --------------------------------------------------------------------------------------------
def node_targets(tree, nodes, c_pact, c_pz, nvec, npad: int, n_view: int = 2,
                 tol: float = 1e-9, prefer_first: bool = True, first_floor=None) -> dict:
    """AlphaZero-style targets at EXPANDED nodes, read off the finished tree (pure numpy).

    The policy target of a node is the set of its children that BEAT THEIR SIBLINGS: the ones
    whose backed-up value Q (the best score anywhere in the child's subtree) is the best among
    the node's children. A node carries a target only when that best Q is above the node's own
    score, i.e. something below it banked more than the node had banked on arrival - a node
    whose children all tie at the parent's own value says nothing about which move was better.
    ``prefer_first``: when the node's FIRST child (child 0 of an expansion: the policy's own
    greedy macro) is among the best, the target is that child alone - a tie with random-held
    children is no reason to move the policy toward randomness (slot 3 measured on run 5's
    4.53B checkpoint: the greedy child is the best at ~70% of the nodes whose children differ).
    Every number here is a rank or a value normalised inside the tree (the tree's own score:
    banked progress / the start distance), never a constant of a map.
    ``first_floor`` ({node id: value}, --mcts-root-chain): a lower bound on the FIRST child's Q of
    those nodes - for a root, the best Q of the later roots on its own greedy line, which the
    policy's own move reaches by flying on (roots are siblings in the tree, so nothing else backs
    that value up to the earlier root).

    ``nodes``: node ids (one label row each); ``c_pact`` (n_nodes_total, n_heads): each child's
    FIRST decision in policy space; ``c_pz`` (n_nodes_total, n_z): its view z; ``nvec``: the
    policy's head widths (categorical heads at index >= n_view). -> dict with, per input node:
    ``keep`` bool, ``probs`` (n, n_heads, npad) f32 (the best children's first-move histogram
    per categorical head, rows sum to 1; view heads 0), ``vmu`` / ``vsd`` (n, n_z) the best
    children's view z moments, ``value`` (n,) the node's backed-up Q, ``n_best`` (n,), ``adv`` (n,)
    the target's ADVANTAGE: the target children's Q minus the mean Q of all the node's children,
    over the tree's own Q range (tree.qhi - tree.qlo) - how much the move the label names beat
    the moves it was compared with, in the tree's normalisation (an AWR-style weight)."""
    nodes = np.asarray(nodes, np.int64).reshape(-1)
    n = len(nodes)
    nh = len(nvec)
    n_z = int(c_pz.shape[1]) if c_pz.ndim == 2 else 0
    out = {"keep": np.zeros(n, bool), "probs": np.zeros((n, nh, int(npad)), np.float32),
           "vmu": np.zeros((n, n_z), np.float32), "vsd": np.zeros((n, n_z), np.float32),
           "value": np.zeros(n, np.float32), "n_best": np.zeros(n, np.int64),
           "adv": np.zeros(n, np.float32), "best_child": np.full(n, -1, np.int64)}
    if n == 0:
        return out
    nc = np.asarray(tree.ncld[nodes], np.int64)
    has = nc > 0
    sel = np.flatnonzero(has)
    if len(sel) == 0:
        return out
    nd = nodes[sel]
    cnt = nc[sel]
    cs = np.asarray(tree.cstart[nd], np.int64)
    offs = np.cumsum(cnt) - cnt
    idx = np.repeat(cs - offs, cnt) + np.arange(int(cnt.sum()))     # child ids, block by block
    q = np.asarray(tree.q[idx], np.float64).copy()
    if first_floor:
        fl = np.asarray([first_floor.get(int(x), -np.inf) for x in nd], np.float64)
        q[offs] = np.maximum(q[offs], fl)
    qmax = np.maximum.reduceat(q, offs)
    best = q >= np.repeat(qmax, cnt) - 1e-12
    if prefer_first:
        first = np.zeros(len(idx), bool)
        first[offs] = True
        f_best = np.repeat(best[offs], cnt)      # the node's first child is among the best
        best = np.where(f_best, first, best)
    nb = np.add.reduceat(best.astype(np.int64), offs)
    own = np.asarray(tree.score[nd], np.float64)
    keep = np.isfinite(qmax) & (qmax > own + tol)
    owner = np.repeat(np.arange(len(nd)), cnt)
    probs = np.zeros((len(nd), nh, int(npad)), np.float64)
    bi = np.flatnonzero(best)
    for h in range(n_view, nh):
        b = np.asarray(c_pact[idx[bi], h], np.int64)
        if (b < 0).any() or (b >= int(nvec[h])).any():
            raise ValueError(f"head {h}: a child's action is outside 0..{int(nvec[h]) - 1}")
        np.add.at(probs[:, h, :], (owner[bi], b), 1.0)
    probs[:, n_view:, :] /= np.maximum(nb, 1)[:, None, None]
    if n_z:
        z = np.asarray(c_pz[idx], np.float64)
        zb = np.where(best[:, None], z, 0.0)
        s1 = np.add.reduceat(zb, offs, axis=0)
        s2 = np.add.reduceat(zb * zb, offs, axis=0)
        mu = s1 / np.maximum(nb, 1)[:, None]
        var = np.maximum(s2 / np.maximum(nb, 1)[:, None] - mu * mu, 0.0)
        out["vmu"][sel] = mu.astype(np.float32)
        out["vsd"][sel] = np.sqrt(var).astype(np.float32)
    out["keep"][sel] = keep
    out["probs"][sel] = probs.astype(np.float32)
    out["value"][sel] = np.asarray(tree.q[nd], np.float32)
    out["n_best"][sel] = nb
    _pos = np.where(best, np.arange(len(idx)), len(idx))
    _fp = np.minimum.reduceat(_pos, offs)
    out["best_child"][sel] = np.where(_fp < len(idx), idx[np.minimum(_fp, len(idx) - 1)], -1)
    span = float(tree.qhi - tree.qlo) if np.isfinite(tree.qhi - tree.qlo) else 0.0
    q_tgt = np.add.reduceat(np.where(best, q, 0.0), offs) / np.maximum(nb, 1)
    q_mean = np.add.reduceat(q, offs) / cnt
    out["adv"][sel] = (np.maximum(q_tgt - q_mean, 0.0) / span if span > 1e-12
                       else np.zeros(len(sel))).astype(np.float32)
    return out


def exec_view_z(pz, view, mask, mode, pitch_max: float) -> np.ndarray:
    """(N, n_z) the policy's drawn z, (N, 2) the EXECUTED view row, (N,) bool -> (N, n_z) z with
    the masked rows replaced by the z of the view they flew (surfgym.view.z_from_view_any: exact
    in the velocity frame). A perturbed child's yaw is its draw plus a held steer offset, so its
    label must be the flown view, not the draw; every other child flies its own draw."""
    pz = np.asarray(pz, np.float32)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return pz
    from surfgym.view import z_from_view_any
    out = pz.copy()
    zi = z_from_view_any(np.asarray(view, np.float32)[mask], mode, float(pitch_max))
    out[mask] = np.asarray(zi, np.float32)[:, :out.shape[1]]
    return out


def line_decisions(path, ndec, kinds, root: int, rdec: int, r_first: int, n_dec: int):
    """The best line's decisions -> (node, kind, m) per decision: the greedy prefix belongs to the
    ``root`` (kind -1 before ``r_first``, -2 after it), then each path node's ``ndec`` decisions
    with that node's child kind and the decision's index m inside its macro."""
    node_d = [int(root)] * int(rdec)
    kind_d = [-1 if d < int(r_first) else -2 for d in range(int(rdec))]
    m_d = list(range(int(rdec)))
    for nid in path:
        nd = int(ndec[nid])
        node_d += [int(nid)] * nd
        kind_d += [int(kinds[nid])] * nd
        m_d += list(range(nd))
    return node_d[:n_dec], kind_d[:n_dec], m_d[:n_dec]


def onehot_rows(pact, nvec, npad: int, n_view: int = 2) -> np.ndarray:
    """(n, n_heads) policy-space rows -> (n, n_heads, npad) one-hot targets on the categorical
    heads (view heads 0): a line row's single action as the same layout node_targets writes."""
    pact = np.asarray(pact, np.int64).reshape(len(pact), -1)
    out = np.zeros((len(pact), len(nvec), int(npad)), np.float32)
    r = np.arange(len(pact))
    for h in range(n_view, len(nvec)):
        out[r, h, pact[:, h]] = 1.0
    return out


# --------------------------------------------------------------------------------------------
# helpers on states
# --------------------------------------------------------------------------------------------
def energy_h(states) -> np.ndarray:
    o = np.asarray(states["origin"], np.float64).reshape(-1, 3)
    v = np.asarray(states["velocity"], np.float64).reshape(-1, 3)
    return o[:, 2] + (v * v).sum(axis=1) / (2.0 * G_ENERGY)


def field_d(field, origin) -> np.ndarray:
    d = np.asarray(field.sample(np.asarray(origin, np.float64).reshape(-1, 3)), np.float64)
    return np.where(d < field.reach_max - 0.5 * field.cell, d, np.inf)


def contact_mask(vz_pre, vz_post, g_step: float, tol: float, vmax: float) -> np.ndarray:
    """The map pushed back on this tick: vz did not change by exactly one gravity step - unless
    the body is falling at the speed cap (vz pinned at -vmax: dvz = 0 there in free fall)."""
    dvz = np.asarray(vz_post, np.float64) - np.asarray(vz_pre, np.float64)
    capped = (np.asarray(vz_post, np.float64) <= -float(vmax) + 1.0) & (np.abs(dvz) <= tol)
    return (np.abs(dvz - float(g_step)) > float(tol)) & ~capped


# --------------------------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------------------------
def run_mcts(ctx: dict, a) -> dict:
    """The search. ``ctx`` (built by beam_tas.main): policy, packer, device, lidar, K, stack, KEYS,
    SPAWN_DUCK, VIEWC, VIEW_ABS, mk_feed, XF, cc_fn1, cc_fnN, build_sim, cfg, map_path, ep_cap,
    TICK, arm, row0, obs_start, gate_seed, gf (the d field), N (envs), max_ticks, secs.
    ``a``: beam_tas's args (the --mcts-* flags). -> a result dict with ``frontier`` (acts / view
    / tick / d / finished) for beam_tas's --save-frontier and a ``summary``."""
    import torch
    from mcts_policy import MctsPolicy
    from surfgym.core import STATE_DTYPE

    t_all = time.time()
    K = int(ctx["K"])
    M = int(a.mcts_macro)
    N = int(ctx["N"])
    VIEW_ABS = ctx["VIEW_ABS"]
    if not ctx["VIEWC"]:
        raise SystemExit("--mcts needs a --view-continuous checkpoint (the held steer is a view "
                         "offset)")
    gf = ctx["gf"]
    tps = 1000.0 / float(ctx["TICK"].ms)
    rng = np.random.default_rng(int(a.torch_seed) + 6007)
    max_ticks = int(ctx["max_ticks"])
    budget_s = float(a.mcts_minutes) * 60.0

    def mk_wrapper(core, greedy_all: bool):
        es, ef, fstate, lf = ctx["mk_feed"]()
        xkw, xfe = ctx["XF"].make(core)
        pol = MctsPolicy(ctx["policy"], ctx["packer"], ctx["device"], ctx["lidar"], core, K,
                         ctx["stack"], extra_slot=es, extra_fn=ef, latch_fn=lf,
                         cc_fn=(ctx["cc_fn1"] if core.num_envs == 1 else ctx["cc_fnN"]),
                         keys_hold=ctx["KEYS"], keys_temp=float(a.mcts_keys_temp),
                         view_temp=float(a.mcts_view_temp), **xkw)
        pol.spawn_duck = ctx["SPAWN_DUCK"]
        if greedy_all:
            pol.greedy = np.ones(core.num_envs, bool)
        return pol, SideState(pol, xfe, fstate, lf)

    # ---- the greedy pass(es): the roots ---------------------------------------------------------
    # --mcts-spawns K (azr1b4): K greedy passes from K spawns of the map start - the caller's, and
    # K-1 more drawn by the core's own reset with derived seeds (the start's yaw jitter, which every
    # eval episode has) - each contributing max_roots // K of the roots. K = 1 is the single pass.
    g_step = float(ctx["g_step"])
    tol = float(a.mcts_contact_tol)
    every = max(1, int(a.mcts_root_every))
    stall_ticks = int(round(float(a.mcts_stall_secs) * tps))
    K_sp = max(1, int(getattr(a, "mcts_spawns", 1) or 1))
    spawns = [(int(ctx["gate_seed"]), ctx["row0"], np.asarray(ctx["obs_start"], np.float32))]
    if K_sp > 1:
        _cs = ctx["build_sim"](ctx["cfg"], ctx["map_path"], 1, ctx["ep_cap"], tick=ctx["TICK"])
        ctx["arm"](_cs)
        for j in range(1, K_sp):
            sj = int(ctx["gate_seed"]) + 104729 * j
            oj = np.array(_cs.reset(sj))
            spawns.append((sj, _cs.get_states()[0:1].copy(), oj[0].copy()))
    passes = []
    vmax = None
    for k_sp, (seed_k, row_k, obs_k) in enumerate(spawns):
        core1 = ctx["build_sim"](ctx["cfg"], ctx["map_path"], 1, ctx["ep_cap"], tick=ctx["TICK"])
        ctx["arm"](core1)
        obs1 = np.array(core1.reset(int(seed_k)))
        core1.set_state(0, row_k)
        obs1[0] = np.asarray(obs_k, np.float32)
        vmax = float(core1.config.phys.sv_maxvelocity)
        pol1, side1 = mk_wrapper(core1, True)
        with torch.inference_mode():
            # allocate the lazily-built side arrays at tick 0, so the spawn itself can be a root
            # (every feed reads tick 0 as an episode start, so the decision's own call sees the
            # same state)
            pol1._obs(obs1)
        G_acts, G_view = [], []
        snaps = []                  # (decision index, tick, state row, obs row, side rows)
        gbest, gbest_t, bank_best, g_bank_t = np.inf, 0, np.inf, 0
        g_mark = np.inf             # the best d when the stall clock last re-armed
        g_end, g_why = 0, "cap"
        t0 = time.time()
        dec = 0
        tick = 0
        while tick < max_ticks:
            if dec % every == 0:
                s_ = core1.get_states()[0:1].copy()
                snaps.append((dec, tick, s_, obs1[0:1].copy(), side1.gather([0]),
                              bank_best, g_bank_t))
            act = pol1.act(obs1)
            view = pol1.view
            G_acts.append(act[0].copy())
            G_view.append(view[0].copy())
            dead = False
            for _k in range(K):
                if _k:
                    act = pol1.act(obs1)
                vz0 = float(core1.states_view["velocity"][0, 2])
                obs1, _r, dn, tr, _t = core1.step(act, view=np.ascontiguousarray(view, np.float32))
                obs1 = np.array(obs1)
                tick += 1
                if bool(np.asarray(dn)[0]) or bool(np.asarray(tr)[0]):
                    dead = True
                    g_why = "finish" if bool(core1.goal_hits[0]) else ("trunc" if bool(
                        np.asarray(tr)[0]) else "death")
                    break
                st = core1.states_view
                d = float(field_d(gf, st["origin"][0:1])[0])
                if d < g_mark - 32.0:
                    g_mark, gbest_t = d, tick
                if d < gbest:
                    gbest = d
                if contact_mask([vz0], [float(st["velocity"][0, 2])], g_step, tol, vmax)[0] \
                        and d < bank_best:
                    bank_best, g_bank_t = d, tick
            dec += 1
            if dead:
                break
            if tick - gbest_t > stall_ticks:
                g_why = "stall"
                break
        g_end = tick
        G_acts = np.asarray(G_acts, np.int8)
        G_view = np.asarray(G_view, np.float32)
        print(f"mcts: greedy line {g_why} at tick {g_end} ({ctx['secs'](g_end):.2f}s), best d "
              f"{gbest:,.0f} (banked {bank_best:,.0f} at tick {g_bank_t}), {len(snaps)} snapshots "
              f"in {time.time() - t0:.0f}s"
              + (f" [spawn {k_sp}: seed {seed_k}, yaw {float(row_k['yaw'][0]):.2f}]"
                 if K_sp > 1 else ""), flush=True)
        # roots end where the line ended: a STALL at its last banked contact, a DEATH at the death
        # itself (--mcts-roots-flight 1: a fatal flight after the last contact can still be
        # steered - air control - so its states are roots too; 0 = stop at the last contact)
        if g_why == "stall" or (g_why == "death" and not int(a.mcts_roots_flight)):
            end_ref = g_bank_t
        else:
            end_ref = g_end
        lo_t = max(0, end_ref - int(round(float(a.mcts_roots_secs) * tps)))
        roots_k = [s_ for s_ in snaps if lo_t <= s_[1] <= end_ref and s_[4] is not None]
        if not roots_k:
            roots_k = [s_ for s_ in snaps if s_[4] is not None][-1:]
        if not roots_k:
            raise SystemExit("--mcts: the greedy line ended before its first root")
        mr = max(1, int(a.mcts_max_roots) // K_sp)
        if len(roots_k) > mr:
            pick = np.unique(np.linspace(0, len(roots_k) - 1, mr).round().astype(int))
            roots_k = [roots_k[i] for i in pick]
        passes.append({"G_acts": G_acts, "G_view": G_view, "g_end": g_end, "g_why": g_why,
                       "gbest": gbest, "bank_best": bank_best, "g_bank_t": g_bank_t,
                       "roots": [r + (k_sp,) for r in roots_k], "spawn": spawns[k_sp]})
    roots = [r for P in passes for r in P["roots"]]
    # the first pass's numbers carry the single-pass names (the summary, the prints)
    P0 = passes[0]
    G_acts, G_view = P0["G_acts"], P0["G_view"]
    g_end, g_why, gbest = P0["g_end"], P0["g_why"], P0["gbest"]
    bank_best, g_bank_t = P0["bank_best"], P0["g_bank_t"]
    g_end_all = int(sum(P["g_end"] for P in passes))
    d_ref = float(field_d(gf, np.asarray(ctx["row0"]["origin"]).reshape(-1, 3))[0])
    if not np.isfinite(d_ref) or d_ref <= 0:
        d_ref = float(gbest) if np.isfinite(gbest) else 1.0

    def prog(d):
        return (d_ref - np.asarray(d, np.float64)) / d_ref

    # --mcts-energy W: the search's value also holds the ENERGY HEIGHT h kept at the banked point,
    # in the trainer's own units - its race reward pays 100 per d0 of progress and --energy-pot K
    # pays K per 1000 u of h, so 1 u of h is worth K x d0 / 1e5 u of progress; W scales that
    # (1 = the recipe's own exchange rate, 0 = off). A line that banks progress AND keeps its speed
    # outranks one that banks a little more and stops dead.
    _K = float(ctx["cfg"].get("energy_pot") or 0.0)
    lam = float(a.mcts_energy) * _K * d_ref / 1e5
    h_ref = float(energy_h(np.asarray(ctx["row0"]).reshape(-1))[0])

    def e_score(h):
        if lam <= 0.0:
            return 0.0
        h = np.asarray(h, np.float64)
        return np.where(np.isfinite(h), lam * (h - h_ref) / d_ref, 0.0)

    # ---- the population core and the stores ---------------------------------------------------
    coreN = ctx["build_sim"](ctx["cfg"], ctx["map_path"], N, ctx["ep_cap"], tick=ctx["TICK"])
    ctx["arm"](coreN)
    obsN = np.array(coreN.reset(int(ctx["gate_seed"])))
    polN, sideN = mk_wrapper(coreN, False)
    polN.greedy = np.zeros(N, bool)
    # --mcts-labels (azr1b4): the tree's own training labels - the row every EXPANDED node's
    # decision read (its scalar half + the pose of its image) and every child's first decision
    # in POLICY space (keep bins unresolved) with its view z; built into targets after the search
    LAB = bool(int(getattr(a, "mcts_labels", 0) or 0))
    if LAB:
        polN.capture = True
        polN.record_policy_act = True
    lab_leaf, lab_scal, lab_pose = [], [], []
    for i in range(N):
        coreN.set_state(i, ctx["row0"])
    obsN[:] = np.asarray(ctx["obs_start"], np.float32)[None, :]
    with torch.inference_mode():
        polN._obs(obsN)                  # allocate every lazily-built side array
    names1, namesN = sorted(roots[0][4].keys()), sorted(sideN.names())
    if names1 != namesN:
        raise SystemExit(f"--mcts: the 1-env and N-env wrappers hold different side state "
                         f"({names1} vs {namesN})")
    store = RowStore(STATE_DTYPE, obsN.shape[1], cap=max(4096, 8 * N))
    tree = Tree(cap=max(4096, 8 * N))
    # --mcts-leaf-value (azr1b4): the network's value at every live child's end state guides
    # SELECTION (Tree.qv = max(its banked score, V), V capped at 1 = the goal); the labels, the
    # value targets and the best line keep reading the real banked q
    LEAFV = bool(int(getattr(a, "mcts_leaf_value", 0) or 0))
    tree.use_qv = LEAFV
    lv_n, lv_up, lv_gap = 0, 0, 0.0
    arch = CellArchive(float(a.mcts_cell), float(a.mcts_cell_energy))
    r_states = np.concatenate([r[2] for r in roots])
    r_obs = np.concatenate([r[3] for r in roots])
    r_side = {k: np.concatenate([r[4][k] for r in roots]) for k in names1}
    r_slots = store.add(r_states, r_obs, r_side)
    r_cells, _ = arch.visit(arch.keys(r_states["origin"], r_states["velocity"]))
    r_bank = np.asarray([r[5] for r in roots], np.float64)
    r_bank = np.where(np.isfinite(r_bank), r_bank, d_ref)
    r_h = energy_h(r_states)
    r_ids = tree.add_children(0, prog(r_bank) + e_score(r_h), np.ones(len(roots), bool), slot=r_slots,
                      cell=r_cells, h_bank=r_h,
                      tick=np.asarray([r[1] for r in roots], np.int64),
                      bank_d=r_bank, end_d=field_d(gf, r_states["origin"]),
                      h=energy_h(r_states), root_dec=np.asarray([r[0] for r in roots]),
                      bank_t=np.asarray([r[6] for r in roots], np.int64), ndec=0, wave=-1,
                      ox=r_states["origin"][:, 0], oy=r_states["origin"][:, 1],
                      oz=r_states["origin"][:, 2])
    root_sp = {int(i): int(r[7]) for i, r in zip(r_ids, roots)}     # root node -> its spawn
    print(f"mcts: {len(roots)} roots over ticks {roots[0][1]}..{roots[-1][1]} "
          f"({ctx['secs'](roots[0][1]):.1f}-{ctx['secs'](roots[-1][1]):.1f}s), d_ref {d_ref:,.0f}; "
          f"side state {namesN}", flush=True)

    # edge storage: per node M decisions of engine rows + views
    E_acts = np.zeros((tree._cap, M, 6), np.int8)
    E_view = np.zeros((tree._cap, M, 2), np.float32)
    # --mcts-labels: per node, its macro's FIRST decision in policy space and its view z, and
    # how the child was drawn (0 the policy's mode, 1 random-held, 2 a sampled / perturbed one)
    n_z = int(getattr(ctx["policy"], "n_z", 2))
    C_pact = np.zeros((tree._cap if LAB else 0, 6), np.int16)
    C_pz = np.zeros((tree._cap if LAB else 0, n_z), np.float32)
    C_kind = np.zeros(tree._cap if LAB else 0, np.int8)
    # ... and EVERY decision's policy-space draw and flown view z (a sampled child's later
    # decisions are fresh draws: the best line's replay labels them from here)
    E_pact = np.zeros((tree._cap if LAB else 0, M, 6), np.int16)
    E_pz = np.zeros((tree._cap if LAB else 0, M, n_z), np.float32)

    Kc = max(2, int(a.mcts_k))
    sigma = float(a.mcts_yaw_sigma)
    cpuct = float(a.mcts_cpuct)
    cell_w = arch.weight if float(a.mcts_cell_w) > 0 else None
    # --mcts-mix-rule (azr1b2): off = the fixed knob, exactly as before; sib = the rule sets the
    # (P, E, R) shares; log = the fixed knob with the rule's bookkeeping beside it (diagnostic)
    _mr = str(getattr(a, "mcts_mix_rule", "off") or "off")
    mix = None
    if _mr != "off":
        mix = MixRule(_mr, floor=float(a.mcts_mix_floor), horizon=float(a.mcts_mix_horizon),
                      warm=int(a.mcts_mix_warm), perturb_frac=float(a.mcts_perturb_frac),
                      random_frac=float(a.mcts_random_frac))
        print(f"mcts: child mix rule {_mr} (floor {mix.floor:g}, horizon {mix.horizon:g} wins, "
              f"warm {mix.warm}); " + ("--mcts-perturb-frac / --mcts-random-frac are the rule's "
                                       "to set" if _mr == "sib" else "the fixed knob draws"),
              flush=True)
    wave, sim_ticks = 0, 0
    t_loop = time.time()
    log_every = max(1, int(a.mcts_log_every))
    best_hist = []
    idxN = np.arange(N)
    while time.time() - t_all < budget_s and wave < int(a.mcts_max_waves):
        leaves = tree.select(max(1, N // Kc), cpuct, cell_w=cell_w,
                             novelty=float(a.mcts_novelty))
        if not leaves:
            print(f"mcts wave {wave}: nothing left to expand - the tree is exhausted", flush=True)
            break
        L = len(leaves)
        own = np.asarray(leaves, np.int64)[idxN % L]          # env -> leaf (round robin)
        rank = idxN // L                                       # child rank within its leaf
        # restore every env's leaf
        S, O, SD = store.rows(tree.slot[own])
        for i in range(N):
            coreN.set_state(i, S[i:i + 1])
        obsN = np.array(obsN)
        obsN[:] = O
        sideN.scatter(idxN, SD)
        polN.greedy = rank == 0
        # the knob: of the SAMPLED children, --mcts-perturb-frac explore (the held steer and the
        # keys / view temperatures), the rest are the policy's own samples at temperature 1 (the
        # Nature paper's policy-based explorer commits 50/50 to random or policy per step);
        # random children (--mcts-random-frac of the sampled ones, before the explore draw): the
        # first decision drawn at --mcts-random-temp on every head and HELD for the macro
        if mix is None:
            explore, rnd = legacy_child_kinds(rng, rank, float(a.mcts_perturb_frac),
                                              float(a.mcts_random_frac))
            kinds = None
        else:
            kinds = mix.draw(rng, rank)
            explore = kinds == MODE_PERTURB
            rnd = kinds == MODE_RANDOM
        polN.hold = rnd if rnd.any() else None
        if sigma > 0:
            # coherent steering: an explore child keeps its parent's held offset with
            # probability --mcts-hold-p (Go-Explore's action repeat, one level up), else draws
            # a fresh one; the greedy child and the policy samples steer with none
            if float(a.mcts_yaw_sigma_min or 0.0) > 0.0:
                # a SCALE MIXTURE: each child's steer scale log-uniform in [min, sigma] - fine
                # offsets thread a precision gate, coarse ones explore (one knob, both ends)
                _lo = math.log(float(a.mcts_yaw_sigma_min))
                _sc = np.exp(rng.uniform(_lo, math.log(sigma), N))
                fresh = rng.normal(0.0, 1.0, N) * _sc
            else:
                fresh = rng.normal(0.0, sigma, N)
            keep = rng.random(N) < float(a.mcts_hold_p)
            delta = np.where(explore, np.where(keep, tree.delta[own], fresh), 0.0)
        else:
            delta = np.zeros(N)
        _rt = float(a.mcts_random_temp)
        polN.set_temps(np.where(rnd, _rt, np.where(explore, float(a.mcts_keys_temp), 1.0)),
                       np.where(rnd, _rt, np.where(explore, float(a.mcts_view_temp), 1.0)))
        alive = np.ones(N, bool)
        fin = np.zeros(N, bool)
        fin_t = np.full(N, -1, np.int64)
        bank_d = np.full(N, np.inf)
        bank_t = np.full(N, -1, np.int64)
        raw_d = np.full(N, np.inf)          # diagnostics: the deepest d at ANY live tick
        h_bk = np.full(N, np.nan)           # the energy height at the banked tick
        acts_w = np.zeros((M, N, 6), np.int8)
        view_w = np.zeros((M, N, 2), np.float32)
        cell_keys = []
        n_new = np.zeros(N, np.float64)
        tt = 0
        if LAB:
            wpa_m = np.zeros((M, N, 6), np.int16)
            wpz_m = np.zeros((M, N, n_z), np.float32)
        for m in range(M):
            polN.hold_store = m == 0
            if LAB:
                polN.capture = m == 0      # the node's own row only (one device read per wave)
            act = polN.act(obsN)
            if LAB and m == 0:
                # every env of a leaf restored the leaf's state and side state: its rank-0 env's
                # row IS the node's row; every env's policy-space draw is its child's first move
                e0 = np.flatnonzero(rank == 0)
                lab_leaf.append(own[e0].copy())
                lab_scal.append(np.asarray(polN.last_scal, np.float32)[e0].copy())
                lab_pose.append(np.asarray(polN.last_pose, np.float32)[e0].copy())
            view = np.array(polN.view, np.float32)
            if sigma > 0:
                if VIEW_ABS:
                    view[:, 0] = ((view[:, 0] + delta + 180.0) % 360.0) - 180.0
                else:
                    view[:, 0] = view[:, 0] + delta
            if LAB:
                # every env's policy-space draw and the z of the view it FLIES (a perturbed
                # child's held steer included); m == 0 is its child's first move
                wpa_m[m] = np.asarray(polN.policy_act, np.int16)
                wpz_m[m] = exec_view_z(polN.view_z, view, explore & (delta != 0.0), VIEW_ABS,
                                       polN._pitch_max)
                if m == 0:
                    w_pact = wpa_m[0].copy()
                    w_pz = wpz_m[0].copy()
            act = np.ascontiguousarray(act, np.int32)
            view = np.ascontiguousarray(view, np.float32)
            acts_w[m] = act
            view_w[m] = view
            for _k in range(K):
                if _k:
                    polN.act(obsN)       # advances the wrapper's tick (held decision)
                vz0 = coreN.states_view["velocity"][:, 2].copy()
                obsN, _r, dn, tr, _t = coreN.step(act, view=view)
                obsN = np.array(obsN)
                tt += 1
                gh = np.asarray(coreN.goal_hits, bool)
                ended = (np.asarray(dn, bool) | np.asarray(tr, bool)) & alive
                if ended.any():
                    fh = ended & gh
                    fin |= fh
                    fin_t[fh] = tt
                    alive &= ~ended
                st = coreN.states_view
                d = field_d(gf, st["origin"])
                c = contact_mask(vz0, st["velocity"][:, 2], g_step, tol, vmax) & alive
                imp = c & (d < bank_d)
                bank_d[imp] = d[imp]
                bank_t[imp] = tt
                if imp.any():
                    _vv = st["velocity"][imp].astype(np.float64)
                    h_bk[imp] = (st["origin"][imp, 2].astype(np.float64)
                                 + (_vv * _vv).sum(axis=1) / (2.0 * G_ENERGY))
                raw_d = np.where(alive, np.minimum(raw_d, d), raw_d)
            st = coreN.states_view
            if alive.any():
                ai = np.flatnonzero(alive)
                cell_keys.append((ai, arch.keys(st["origin"][ai], st["velocity"][ai])))
        sim_ticks += tt
        if wave == 0:
            # restore check: a root's GREEDY child must repeat the greedy pass's own decisions
            # (the same state and inference state -> the same mode), so a mismatch means some
            # per-env state was not carried by SideState / RowStore
            n_cmp, n_key, v_err = 0, 0, 0.0
            for e in np.flatnonzero(rank == 0):
                lf = int(own[e])
                if tree.parent[lf] != 0:
                    continue
                d0 = int(tree.root_dec[lf])
                _GA = passes[root_sp[lf]]["G_acts"]
                _GV = passes[root_sp[lf]]["G_view"]
                mm = min(M, len(_GA) - d0)
                if mm <= 0:
                    continue
                n_cmp += mm
                n_key += int((acts_w[:mm, e] == _GA[d0:d0 + mm]).all(axis=1).sum())
                dv = np.abs(view_w[:mm, e, 0] - _GV[d0:d0 + mm, 0])
                v_err = max(v_err, float(np.minimum(dv, 360.0 - dv).max()))
            print(f"mcts restore check: greedy children repeat the greedy line's engine rows on "
                  f"{n_key}/{n_cmp} decisions, max yaw-target diff {v_err:.4f} deg", flush=True)
        # the children: every env's end state
        stN = coreN.get_states()
        gath = sideN.gather(idxN)
        al = alive.copy()
        slots = np.full(N, -1, np.int64)
        if al.any():
            ai = np.flatnonzero(al)
            slots[ai] = store.add(stN[ai], obsN[ai], {k: v[ai] for k, v in gath.items()})
        end_d = np.where(al, field_d(gf, stN["origin"]), np.inf)
        cells = np.full(N, -1, np.int64)
        new_cells = 0
        for ai, ks in cell_keys:
            isn = np.zeros(len(ks), bool)
            ids, nn = arch.visit(ks, is_new=isn)
            new_cells += nn
            n_new[ai] += isn
        if al.any():
            ai = np.flatnonzero(al)
            cells[ai], _ = arch.visit(arch.keys(stN["origin"][ai], stN["velocity"][ai]))
        hN = energy_h(stN)
        # a child's banked point: its own macro's deepest contact if deeper than its parent's,
        # else the parent's (the score of a line is the deepest point it BANKED); a finish beats all
        own_better = bank_d < tree.bank_d[own]
        c_bank_d = np.where(own_better, bank_d, tree.bank_d[own])
        c_bank_t = np.where(own_better, tree.tick[own] + bank_t, tree.bank_t[own])
        c_h_bank = np.where(own_better, h_bk, tree.h_bank[own])
        sc = np.where(fin, 2.0, prog(c_bank_d) + e_score(c_h_bank))
        _kv = {}
        if LEAFV:
            # one more read of the network, at every child's end state: AFTER the store took the
            # children's side state (the read advances the feeds; the next wave scatters every
            # env's side state back from the store)
            with torch.inference_mode():
                _vv = polN._net(polN._obs(obsN))[1]
            vh = np.minimum(np.asarray(_vv.float().reshape(-1).cpu().numpy(), np.float64), 1.0)
            _lv = al & ~fin
            _kv = {"qv": np.where(_lv, np.maximum(sc, vh), sc)}
            lv_n += int(_lv.sum())
            lv_up += int((_lv & (vh > sc)).sum())
            lv_gap += float(np.where(_lv, _kv["qv"] - sc, 0.0).sum())
        ntick = tree.tick[own] + np.where(fin, fin_t, M * K)
        n_before = tree.n
        if mix is not None:
            # the rule's evidence: which kind's child beat its siblings in each expansion (on the
            # tree's value, else on new cells opened by a live child); before add_children, so
            # the leaf's score is its own (a backup only ever raises q, never score)
            _cr, _how = sibling_credit(own, sc, tree.score[own], al, n_new)
            mix.update(kinds, _cr, _how, al, n_new)
        _kx = {} if mix is None else {"kind": kinds}
        for l, leaf in enumerate(leaves):
            ch = np.flatnonzero(own == leaf)
            ch = ch[np.argsort(rank[ch])]
            ids = tree.add_children(int(leaf), sc[ch], al[ch], finished=fin[ch],
                                    slot=slots[ch], cell=cells[ch], tick=ntick[ch],
                                    bank_d=c_bank_d[ch], end_d=end_d[ch], h=hN[ch],
                                    bank_t=c_bank_t[ch],
                                    ndec=np.where(fin[ch], (fin_t[ch] + K - 1) // K, M),
                                    wave=wave, delta=delta[ch], nov=n_new[ch] / float(M),
                                    ox=stN["origin"][ch, 0], oy=stN["origin"][ch, 1],
                                    oz=stN["origin"][ch, 2], raw_d=raw_d[ch],
                                    h_bank=c_h_bank[ch],
                                    **{k: v[ch] for k, v in {**_kx, **_kv}.items()})
            if tree._cap > len(E_acts):
                E_acts = np.concatenate([E_acts, np.zeros((tree._cap - len(E_acts), M, 6),
                                                          np.int8)])
                E_view = np.concatenate([E_view, np.zeros((tree._cap - len(E_view), M, 2),
                                                          np.float32)])
            E_acts[ids] = np.transpose(acts_w[:, ch], (1, 0, 2))
            E_view[ids] = np.transpose(view_w[:, ch], (1, 0, 2))
            if LAB:
                if tree._cap > len(C_pact):
                    _g = tree._cap - len(C_pact)
                    C_pact = np.concatenate([C_pact, np.zeros((_g, 6), np.int16)])
                    C_pz = np.concatenate([C_pz, np.zeros((_g, n_z), np.float32)])
                    C_kind = np.concatenate([C_kind, np.zeros(_g, np.int8)])
                if tree._cap > len(E_pact):
                    _g = tree._cap - len(E_pact)
                    E_pact = np.concatenate([E_pact, np.zeros((_g, M, 6), np.int16)])
                    E_pz = np.concatenate([E_pz, np.zeros((_g, M, n_z), np.float32)])
                C_pact[ids] = w_pact[ch]
                C_pz[ids] = w_pz[ch]
                E_pact[ids] = np.transpose(wpa_m[:, ch], (1, 0, 2))
                E_pz[ids] = np.transpose(wpz_m[:, ch], (1, 0, 2))
                C_kind[ids] = np.where(rank[ch] == 0, 0, np.where(rnd[ch], 1, 2))
        tree.end_wave()
        b = tree.best_node()
        best_hist.append(float(prog(tree.bank_d[b])) if b > 0 else float("nan"))
        if wave % log_every == 0 or fin.any():
            _bp = 100 * float(prog(tree.bank_d[b])) if b > 0 else float("nan")
            dt = time.time() - t_loop
            print(f"mcts wave {wave:4d}: {L:4d} leaves, nodes {tree.n:,} (+{tree.n - n_before}), "
                  f"alive {int(al.sum())}/{N}, fin {int(fin.sum())}, cells {arch.n:,} "
                  f"(+{new_cells}), best {_bp:.2f}% "
                  f"(d {tree.bank_d[b]:,.0f}, depth {int(tree.depth[b])}), "
                  f"{sim_ticks * N / max(dt, 1e-9):,.0f} env-steps/s"
                  + ("" if mix is None else ", " + mix.text()), flush=True)
        wave += 1
        if fin.any() and bool(a.mcts_stop_on_finish):
            break

    # ---- the best line -------------------------------------------------------------------------
    b = tree.best_node()
    path = tree.path_to(b)
    root = path[0]
    rdec = int(tree.root_dec[root])
    k_root = root_sp.get(int(root), 0)            # the spawn whose greedy pass the line starts on
    acts = [passes[k_root]["G_acts"][:rdec]]
    views = [passes[k_root]["G_view"][:rdec]]
    for j, nid in enumerate(path[1:]):
        nd = int(tree.ndec[nid])
        acts.append(E_acts[nid, :nd])
        views.append(E_view[nid, :nd])
    acts = np.concatenate(acts).astype(np.int32)
    views = np.concatenate(views).astype(np.float32)
    finished = bool(tree.finished[b])
    # the line ends at the finish, or at the tick its deepest contact was banked (an absolute
    # episode tick: inside the best node's own macro or inherited from an ancestor's)
    f_tick = int(tree.tick[b]) if finished else int(tree.bank_t[b])
    if f_tick <= 0:
        f_tick = int(tree.tick[b])
    if f_tick > len(acts) * K:
        f_tick = len(acts) * K
    n_dec = (f_tick + K - 1) // K
    frontier = {"acts": acts[:n_dec], "view": views[:n_dec], "tick": int(f_tick),
                "d": float(tree.bank_d[b]) if not finished else 0.0, "finished": finished,
                "spawn": int(k_root)}
    dt = time.time() - t_loop
    depths = tree.depth[1:tree.n]
    summary = {"mode": "mcts", "waves": wave, "nodes": int(tree.n - 1),
               "alive_nodes": int(tree.alive[1:tree.n].sum()), "cells": int(arch.n),
               "roots": len(roots), "macro_decisions": M, "children": Kc,
               "yaw_sigma": sigma, "yaw_sigma_min": float(a.mcts_yaw_sigma_min or 0.0),
               "roots_flight": int(a.mcts_roots_flight), "hold_p": float(a.mcts_hold_p),
               "novelty": float(a.mcts_novelty), "perturb_frac": float(a.mcts_perturb_frac),
               "energy": float(a.mcts_energy), "energy_lambda": lam,
               "random_frac": float(a.mcts_random_frac), "random_temp": float(a.mcts_random_temp),
               "keys_temp": float(a.mcts_keys_temp),
               "view_temp": float(a.mcts_view_temp), "c_puct": cpuct,
               "cell_w": float(a.mcts_cell_w), "cell": float(a.mcts_cell),
               "cell_energy": float(a.mcts_cell_energy),
               "greedy_end_tick": int(g_end), "greedy_end": g_why,
               "greedy_best_d": float(gbest), "greedy_bank_d": float(bank_best),
               "best_node": int(b), "best_d": frontier["d"], "best_tick": int(f_tick),
               "best_pct": round(100.0 * float(prog(tree.bank_d[b])), 3) if not finished else 100.0,
               "greedy_pct": round(100.0 * float(prog(bank_best)), 3),
               "raw_best_pct": round(100.0 * float(prog(np.min(tree.raw_d[1:tree.n]))), 3)
               if tree.n > 1 else None,
               "best_depth": int(tree.depth[b]), "max_depth": int(depths.max()) if len(depths)
               else 0, "finished": finished, "finishes": int(tree.finished[1:tree.n].sum()),
               "search_wall_s": round(dt, 1),
               "env_steps_per_s": round(sim_ticks * N / max(dt, 1e-9)),
               "best_by_wave": [round(100 * x, 3) for x in best_hist[::max(1, len(best_hist)
                                                                          // 50)]]}
    if mix is not None:
        # the kinds of child along the best line's macros (G child 0, P sample, E perturbed,
        # R random-held), and the rule's trajectory and totals
        _pk = "".join(MODE_CHARS[int(tree.kind[i])] if int(tree.kind[i]) >= 0 else "?"
                      for i in path[1:])
        summary.update(mix.summary(_pk))
        print(f"mcts: child mix ({mix.mode}) mean P/E/R "
              f"{summary['mix_share_mean']['P']:.2f}/{summary['mix_share_mean']['E']:.2f}/"
              f"{summary['mix_share_mean']['R']:.2f}, final {mix.text()}; wins G/P/E/R "
              + "/".join(f"{summary['mix_wins'][k]:.0f}" for k in "GPER")
              + f"; best line's macros {_pk or '-'}", flush=True)
    if LEAFV:
        # how often the network's value raised a live child above its banked score, and by how
        # much on average (in the tree's units: progress over the start distance)
        summary["leaf_value"] = {"children": lv_n,
                                 "share_raised": round(lv_up / max(1, lv_n), 4),
                                 "mean_raise": round(lv_gap / max(1, lv_n), 5)}
        print(f"mcts: leaf value raised {summary['leaf_value']['share_raised']:.1%} of {lv_n:,} "
              f"live children, by {summary['leaf_value']['mean_raise']:.4f} on average",
              flush=True)
    print(f"mcts: deepest RAW d any live tick reached {summary['raw_best_pct']}% (greedy raw "
          f"{100.0 * float(prog(gbest)):.2f}%)", flush=True)
    print(f"mcts: {wave} waves, {tree.n - 1:,} nodes, {arch.n:,} cells, best "
          f"{summary['best_pct']:.2f}% (greedy {summary['greedy_pct']:.2f}%) at tick {f_tick} "
          f"({ctx['secs'](f_tick):.2f}s), depth {int(tree.depth[b])}"
          + (" FINISHED" if finished else "") + f", {dt:.0f}s", flush=True)
    # EVERY simulator tick this call spent: the greedy pass (1 env) + every wave's copies
    summary["ticks_greedy"] = int(g_end_all)
    summary["ticks_waves"] = int(sim_ticks) * int(N)
    summary["ticks_total"] = int(g_end_all) + int(sim_ticks) * int(N)
    if K_sp > 1:
        summary["spawns"] = K_sp
        summary["spawn_greedy_pct"] = [round(100.0 * float(prog(P["bank_best"])), 3)
                                       for P in passes]
        summary["best_spawn"] = int(k_root)
    labels = None
    if LAB:
        labels = _tree_labels(ctx, a, mk_wrapper, tree, b, path, root, rdec, acts, views, n_dec,
                              passes[k_root]["spawn"], root_sp,
                              dict(store=store, coreN=coreN, polN=polN, sideN=sideN,
                                   E_acts=E_acts, E_view=E_view, E_pact=E_pact, E_pz=E_pz),
                              lab_leaf, lab_scal, lab_pose, C_pact, C_pz, C_kind, M, K)
        summary["ticks_line"] = int(labels.pop("ticks"))
        summary["ticks_total"] += summary["ticks_line"]
        summary["labels_node"] = int((labels["kind"] == 0).sum())
        summary["labels_line"] = int((labels["kind"] == 1).sum())
        summary["labels_prefix"] = int((labels["kind"] == 2).sum())
        summary["labels_nodes_expanded"] = int(labels.pop("n_expanded"))
        summary["line_replay_d"] = labels.pop("replay_d")
    return {"frontier": frontier, "summary": summary, "tree": tree, "archive": arch,
            "labels": labels}



def _dense_rows(tree, nodes, best_child, adv, pop, C_kind, C_pact, C_pz, M, K, nvec, npad,
                n_view):
    """--mcts-labels-dense: every LATER decision of each labelled node's chosen macro (decisions
    1..ndec-1; decision 0 is the node row itself), replayed open loop from the node's stored state
    in the population core, its engine rows and views the tree's own (E_acts / E_view), the held
    keys set to each executed row - the line replay's rule, batched. A random-held child's label is
    its held move at every decision; the policy's own child's is the network's greedy move at the
    replayed state. Rows are node rows (kind 0) with their node's advantage and the chosen
    child's Q. -> (rows dict, ticks)."""
    import torch
    from surfgym.keyshold import HELD_HEADS
    store, coreN, polN, sideN = pop["store"], pop["coreN"], pop["polN"], pop["sideN"]
    E_acts, E_view = pop["E_acts"], pop["E_view"]
    N = coreN.num_envs
    out = {k_: [] for k_ in ("scal", "pose", "probs", "vmu", "vsd", "value", "kind", "adv")}
    ticks = 0
    nodes = np.asarray(nodes, np.int64)
    best_child = np.asarray(best_child, np.int64)
    ok = (best_child >= 0) & (tree.slot[nodes] >= 0)
    nodes, best_child, adv = nodes[ok], best_child[ok], np.asarray(adv)[ok]
    idxN = np.arange(N)
    polN.hold = None
    polN.capture = True
    for s0 in range(0, len(nodes), N):
        nb = nodes[s0:s0 + N]
        cb = best_child[s0:s0 + N]
        ab = adv[s0:s0 + N]
        n = len(nb)
        fill = np.concatenate([np.arange(n), np.zeros(N - n, np.int64)])  # spare envs copy row 0
        S, O, SD = store.rows(tree.slot[nb[fill]])
        for i in range(N):
            coreN.set_state(i, S[i:i + 1])
        obsN = np.array(O, np.float32)
        sideN.scatter(idxN, SD)
        kind_b = np.asarray(C_kind[cb], np.int64)
        polN.greedy = np.concatenate([kind_b == 0, np.zeros(N - n, bool)])
        nd = np.minimum(np.asarray(tree.ndec[cb], np.int64), M)
        alive = np.ones(n, bool)
        for m in range(M):
            polN.act(obsN)
            if m >= 1:
                take = np.flatnonzero(alive & (m < nd))
                if len(take):
                    sc = np.asarray(polN.last_scal, np.float32)[take]
                    po = np.asarray(polN.last_pose, np.float32)[take]
                    held = kind_b[take] == 1
                    pa = np.where(held[:, None], np.asarray(C_pact[cb[take]], np.int64),
                                  np.asarray(polN.policy_act, np.int64)[take])
                    pz = np.where(held[:, None], np.asarray(C_pz[cb[take]], np.float32),
                                  np.asarray(polN.view_z, np.float32)[take])
                    out["scal"].append(sc)
                    out["pose"].append(po)
                    out["probs"].append(onehot_rows(pa, nvec, npad, n_view))
                    out["vmu"].append(pz.astype(np.float32))
                    out["vsd"].append(np.zeros_like(pz, dtype=np.float32))
                    out["value"].append(np.asarray(tree.q[cb[take]], np.float32))
                    out["kind"].append(np.zeros(len(take), np.int8))
                    out["adv"].append(np.asarray(ab[take], np.float32))
            rowsE = np.zeros((N, 6), np.int32)
            viewE = np.zeros((N, 2), np.float32)
            rowsE[:n] = E_acts[cb, m]
            viewE[:n] = E_view[cb, m]
            rowsE[n:] = rowsE[0]
            viewE[n:] = viewE[0]
            if getattr(polN, "keys", None) is not None:
                polN.keys.state[:] = rowsE[:, list(HELD_HEADS)]
                polN.keys.boot[:] = polN.keys.state
            for _k in range(K):
                if _k:
                    polN.act(obsN)
                obsN, _r, dn, tr, _t = coreN.step(np.ascontiguousarray(rowsE),
                                                  view=np.ascontiguousarray(viewE))
                obsN = np.array(obsN)
                ticks += N
                ended = (np.asarray(dn, bool) | np.asarray(tr, bool))[:n]
                alive &= ~ended
    polN.capture = False
    polN.greedy = np.zeros(N, bool)
    rows = {k_: (np.concatenate(v) if v else None) for k_, v in out.items()}
    if rows["kind"] is None:
        return {k_: np.zeros(0, np.float32) for k_ in out}, ticks
    return rows, ticks

def _tree_labels(ctx, a, mk_wrapper, tree, b, path, root, rdec, acts, views, n_dec, spawn,
                 root_sp, pop,
                 lab_leaf, lab_scal, lab_pose, C_pact, C_pz, C_kind, M, K) -> dict:
    """--mcts-labels: the finished tree as training rows (azr1b4, the tree-only loop).

    NODE rows - every expanded node with a target (node_targets): the row its decision read
    (scalar half + image pose, captured when its children were drawn), the best children's
    first moves, the node's Q (kind 0). LINE rows - every decision of the best line from the true
    start, replayed open loop on a 1-env core through a greedy wrapper that only READS (the executed
    engine row and view are the line's; the held keys are set to the executed row after each
    decision, beam_tas.prefix_sync's rule): where the line took a random-held child the label is
    that held move, elsewhere the network's own greedy move at the replayed state; the greedy
    prefix is labelled only BEFORE the tree's earliest root. A line row's value is the Q of the
    path node its decision belongs to (the root's for the greedy prefix).
    -> dict of arrays (scal, pose, probs, vmu, vsd, value, kind: 0 node / 1 the line's searched
    part / 2 its greedy prefix, w) + ticks."""
    import torch
    import train_fast as _tf
    from surfgym.keyshold import HELD_HEADS

    nvec, npad = tuple(_tf.NVEC), int(_tf.NPAD)
    n_view = int(_tf.N_VIEW)
    # ---- node rows -------------------------------------------------------------------------
    if lab_leaf:
        L = np.concatenate(lab_leaf)
        LS = np.concatenate(lab_scal)
        LP = np.concatenate(lab_pose)
    else:
        L = np.zeros(0, np.int64)
        LS = np.zeros((0, 0), np.float32)
        LP = np.zeros((0, 6), np.float32)
    floor = None
    if int(getattr(a, "mcts_root_chain", 0) or 0):
        # --mcts-root-chain: a root's own (greedy) move reaches every LATER root of its greedy
        # line, so its Q is at least theirs; per spawn, the suffix max over later roots
        floor = {}
        by_sp = {}
        for rid, k in root_sp.items():
            by_sp.setdefault(int(k), []).append(int(rid))
        for k, rids in by_sp.items():
            rids.sort(key=lambda r_: int(tree.root_dec[r_]))
            best_after = -np.inf
            for r_ in reversed(rids):
                if np.isfinite(best_after):
                    floor[r_] = best_after
                best_after = max(best_after, float(tree.q[r_]))
    nt = node_targets(tree, L, C_pact, C_pz, nvec, npad, n_view, first_floor=floor)
    k = nt["keep"]
    rows = {"scal": [LS[k]], "pose": [LP[k]], "probs": [nt["probs"][k]], "vmu": [nt["vmu"][k]],
            "vsd": [nt["vsd"][k]], "value": [nt["value"][k]],
            "kind": [np.zeros(int(k.sum()), np.int8)], "adv": [nt["adv"][k]]}
    dense_ticks, n_dense = 0, 0
    if int(getattr(a, "mcts_labels_dense", 0) or 0) and k.any():
        dr, dense_ticks = _dense_rows(tree, L[k], nt["best_child"][k], nt["adv"][k], pop,
                                      C_kind, C_pact, C_pz, M, K, nvec, npad, n_view)
        n_dense = len(dr["kind"])
        for key in rows:
            if n_dense:
                rows[key].append(dr[key])
    # ---- line rows ---------------------------------------------------------------------------
    # the greedy prefix up to the line's root: ANCHOR rows (-1) only before the tree's EARLIEST
    # root - outside the search window, the policy's own moves; inside it (-2) the roots' own
    # node rows say whether the policy's move was the best one, and a self-label there would
    # contradict them (an idling untrained policy: "keep idling" at the very states where the
    # tree found the way off the platform)
    _c0, _n0 = int(tree.cstart[0]), int(tree.ncld[0])
    r_first = int(np.min(tree.root_dec[_c0:_c0 + _n0])) if _n0 else int(rdec)
    if int(getattr(a, "mcts_labels_prefix_all", 0) or 0):
        # --mcts-labels-prefix-all: the WHOLE greedy prefix up to the line's root is labelled with
        # the policy's own moves - the path that reaches the root the tree's line starts from (with
        # --mcts-root-chain the roots' own node rows agree with it)
        r_first = int(rdec)
    node_d, kind_d, m_d = line_decisions(path[1:], tree.ndec, C_kind, root, rdec, r_first,
                                         n_dec)
    E_pact, E_pz = pop.get("E_pact"), pop.get("E_pz")
    cfg = ctx["cfg"]
    core1 = ctx["build_sim"](cfg, ctx["map_path"], 1, ctx["ep_cap"], tick=ctx["TICK"])
    ctx["arm"](core1)
    # the line starts on its root's spawn (--mcts-spawns: one of several; else the caller's)
    o = np.array(core1.reset(int(spawn[0])))
    core1.set_state(0, spawn[1])
    o[0] = np.asarray(spawn[2], np.float32)
    pol, _side = mk_wrapper(core1, True)
    pol.capture = True
    pol.record_policy_act = True
    with torch.inference_mode():
        pol._obs(o)               # the greedy pass's own tick-0 call (its feeds' first read)
    keys = bool(ctx["KEYS"])
    l_scal, l_pose, l_pact, l_pz, l_val, l_rk = [], [], [], [], [], []
    l_bank = []                  # the replay's banked d when each labelled decision was taken
    ticks = 0
    n_held = 0
    g_step = float(ctx["g_step"])
    vmax = float(core1.config.phys.sv_maxvelocity)
    tol = float(a.mcts_contact_tol)
    bank_now = np.inf
    for d in range(int(n_dec)):
        pol.act(o)
        # a decision of a SAMPLED / perturbed child (kind 2: none at --mcts-random-frac 1) is
        # labelled with the draw it executed and the view it flew, stored per decision during the
        # search (E_pact / E_pz); the prefix inside the search window (-2) is stepped unlabelled
        if kind_d[d] != -2:
            l_scal.append(np.asarray(pol.last_scal, np.float32)[0].copy())
            l_pose.append(np.asarray(pol.last_pose, np.float32)[0].copy())
            if kind_d[d] == 1:
                l_pact.append(np.asarray(C_pact[node_d[d]], np.int64).copy())
                l_pz.append(np.asarray(C_pz[node_d[d]], np.float32).copy())
                n_held += 1
            elif kind_d[d] == 2:
                l_pact.append(np.asarray(E_pact[node_d[d], m_d[d]], np.int64).copy())
                l_pz.append(np.asarray(E_pz[node_d[d], m_d[d]], np.float32).copy())
                n_held += 1
            else:
                l_pact.append(np.asarray(pol.policy_act, np.int64)[0].copy())
                l_pz.append(np.asarray(pol.view_z, np.float32)[0].copy())
            l_val.append(float(tree.q[node_d[d]]))
            # row kind: 1 = the line's SEARCHED part (the tree's choices), 2 = its greedy prefix
            l_rk.append(2 if kind_d[d] == -1 else 1)
            l_bank.append(bank_now)
        row = np.ascontiguousarray(np.asarray(acts[d], np.int32).reshape(1, -1))
        vw = np.ascontiguousarray(np.asarray(views[d], np.float32).reshape(1, -1))
        if keys and getattr(pol, "keys", None) is not None:
            # after the decision the held keys ARE the executed engine row's fwd / side / duck
            pol.keys.state[0] = row[0, list(HELD_HEADS)]
            pol.keys.boot[0] = pol.keys.state[0]
        ended = False
        for _k in range(K):
            if _k:
                pol.act(o)
            vz0 = float(core1.states_view["velocity"][0, 2])
            o, _r, dn, tr, _t = core1.step(row, view=vw)
            o = np.array(o)
            ticks += 1
            if bool(np.asarray(dn)[0]) or bool(np.asarray(tr)[0]):
                ended = True
                break
            _sv = core1.states_view
            if contact_mask([vz0], [float(_sv["velocity"][0, 2])], g_step, tol, vmax)[0]:
                bank_now = min(bank_now, float(field_d(ctx["gf"], _sv["origin"][0:1])[0]))
        if ended:
            break
    # --mcts-labels-cut U: the searched rows stop at the first decision whose banked d is within U
    # of the line's final banked d - the rest of such a line is time spent for a last few units
    # (a ledge the line wiggles on for a minute to bank a deeper contact) and would flood the
    # line rows with it. Rows taken before the cut decision are kept; the prefix is untouched.
    cut_u = float(getattr(a, "mcts_labels_cut", 0.0) or 0.0)
    n_cut = 0
    if cut_u > 0.0 and l_scal and np.isfinite(bank_now):
        lb = np.asarray(l_bank, np.float64)
        rk = np.asarray(l_rk)
        reach = np.flatnonzero((lb <= bank_now + cut_u) & (rk == 1))
        if len(reach):
            keep_n = int(reach[0]) + 1
            drop = np.zeros(len(l_scal), bool)
            drop[keep_n:] = rk[keep_n:] == 1
            n_cut = int(drop.sum())
            if n_cut:
                sel = np.flatnonzero(~drop)
                l_scal = [l_scal[i] for i in sel]
                l_pose = [l_pose[i] for i in sel]
                l_pact = [l_pact[i] for i in sel]
                l_pz = [l_pz[i] for i in sel]
                l_val = [l_val[i] for i in sel]
                l_rk = [l_rk[i] for i in sel]
    nl = len(l_scal)
    d_end = float(field_d(ctx["gf"], core1.states_view["origin"][0:1])[0])
    if nl:
        rows["scal"].append(np.stack(l_scal))
        rows["pose"].append(np.stack(l_pose))
        rows["probs"].append(onehot_rows(np.stack(l_pact), nvec, npad, n_view))
        rows["vmu"].append(np.stack(l_pz))
        rows["vsd"].append(np.zeros((nl, len(l_pz[0])), np.float32))
        rows["value"].append(np.asarray(l_val, np.float32))
        rows["kind"].append(np.asarray(l_rk, np.int8))
        rows["adv"].append(np.ones(nl, np.float32))    # a line row is not weighed by advantage
    out = {}
    for key, parts in rows.items():
        parts = [p for p in parts if len(p)]
        out[key] = (np.concatenate(parts) if parts else
                    np.zeros((0,), np.float32))
    n_all = len(out["kind"]) if out["kind"].ndim else 0
    out["w"] = np.ones(n_all, np.float32)
    out["ticks"] = ticks + dense_ticks
    out["n_expanded"] = len(L)
    out["n_dense"] = n_dense
    out["replay_d"] = d_end
    if n_dense:
        print(f"mcts labels: {n_dense:,} dense rows (the chosen macros' later decisions), "
              f"{dense_ticks:,} replay ticks", flush=True)
    print(f"mcts labels: {int(k.sum()):,} node rows (of {len(L):,} expanded nodes), {nl:,} line "
          f"rows ({n_held} on random-held macros, {n_cut} cut past the last {cut_u:g} u); "
          f"line replay "
          f"{ticks} ticks, ends at d {d_end:,.0f} (search said {float(tree.bank_d[b]):,.0f})",
          flush=True)
    return out
