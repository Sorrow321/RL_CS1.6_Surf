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
"""
from __future__ import annotations

import math
import time

import numpy as np

G_ENERGY = 800.0           # sv_gravity: h = z + |v|^2 / (2 g)


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
    expandable node left (never descended into again)."""

    def __init__(self, cap: int = 4096):
        self.n = 1
        self._cap = 0
        self._alloc(max(16, int(cap)))
        self.q[0] = -np.inf
        self.score[0] = -np.inf
        self.qlo, self.qhi = np.inf, -np.inf

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
          ("h_bank", np.float64, np.nan))

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
                qn = (np.where(np.isfinite(self.q[ch]), self.q[ch], lo) - lo) * inv
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
                 tol: float = 1e-9, prefer_first: bool = True) -> dict:
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

    ``nodes``: node ids (one label row each); ``c_pact`` (n_nodes_total, n_heads): each child's
    FIRST decision in policy space; ``c_pz`` (n_nodes_total, n_z): its view z; ``nvec``: the
    policy's head widths (categorical heads at index >= n_view). -> dict with, per input node:
    ``keep`` bool, ``probs`` (n, n_heads, npad) f32 (the best children's first-move histogram
    per categorical head, rows sum to 1; view heads 0), ``vmu`` / ``vsd`` (n, n_z) the best
    children's view z moments, ``value`` (n,) the node's backed-up Q, ``n_best`` (n,)."""
    nodes = np.asarray(nodes, np.int64).reshape(-1)
    n = len(nodes)
    nh = len(nvec)
    n_z = int(c_pz.shape[1]) if c_pz.ndim == 2 else 0
    out = {"keep": np.zeros(n, bool), "probs": np.zeros((n, nh, int(npad)), np.float32),
           "vmu": np.zeros((n, n_z), np.float32), "vsd": np.zeros((n, n_z), np.float32),
           "value": np.zeros(n, np.float32), "n_best": np.zeros(n, np.int64)}
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
    q = np.asarray(tree.q[idx], np.float64)
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
    return out


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

    # ---- the greedy pass: the roots ----------------------------------------------------------
    core1 = ctx["build_sim"](ctx["cfg"], ctx["map_path"], 1, ctx["ep_cap"], tick=ctx["TICK"])
    ctx["arm"](core1)
    obs1 = np.array(core1.reset(int(ctx["gate_seed"])))
    core1.set_state(0, ctx["row0"])
    obs1[0] = np.asarray(ctx["obs_start"], np.float32)
    g_step = float(ctx["g_step"])
    vmax = float(core1.config.phys.sv_maxvelocity)
    tol = float(a.mcts_contact_tol)
    pol1, side1 = mk_wrapper(core1, True)
    with torch.inference_mode():
        # allocate the lazily-built side arrays at tick 0, so the spawn itself can be a root (every
        # feed reads tick 0 as an episode start, so the decision's own call sees the same state)
        pol1._obs(obs1)
    every = max(1, int(a.mcts_root_every))
    stall_ticks = int(round(float(a.mcts_stall_secs) * tps))
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
            s = core1.get_states()[0:1].copy()
            snaps.append((dec, tick, s, obs1[0:1].copy(), side1.gather([0]),
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
          f"in {time.time() - t0:.0f}s", flush=True)
    # roots end where the line ended: a STALL at its last banked contact, a DEATH at the death
    # itself (--mcts-roots-flight 1: a fatal flight after the last contact can still be steered -
    # air control - so its states are roots too; 0 = stop at the last contact, as before)
    if g_why == "stall" or (g_why == "death" and not int(a.mcts_roots_flight)):
        end_ref = g_bank_t
    else:
        end_ref = g_end
    lo_t = max(0, end_ref - int(round(float(a.mcts_roots_secs) * tps)))
    roots = [s for s in snaps if lo_t <= s[1] <= end_ref and s[4] is not None]
    if not roots:
        roots = [s for s in snaps if s[4] is not None][-1:]
    if not roots:
        raise SystemExit("--mcts: the greedy line ended before its first root")
    if len(roots) > int(a.mcts_max_roots):
        pick = np.unique(np.linspace(0, len(roots) - 1, int(a.mcts_max_roots)).round().astype(int))
        roots = [roots[i] for i in pick]
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
    arch = CellArchive(float(a.mcts_cell), float(a.mcts_cell_energy))
    r_states = np.concatenate([r[2] for r in roots])
    r_obs = np.concatenate([r[3] for r in roots])
    r_side = {k: np.concatenate([r[4][k] for r in roots]) for k in names1}
    r_slots = store.add(r_states, r_obs, r_side)
    r_cells, _ = arch.visit(arch.keys(r_states["origin"], r_states["velocity"]))
    r_bank = np.asarray([r[5] for r in roots], np.float64)
    r_bank = np.where(np.isfinite(r_bank), r_bank, d_ref)
    r_h = energy_h(r_states)
    tree.add_children(0, prog(r_bank) + e_score(r_h), np.ones(len(roots), bool), slot=r_slots,
                      cell=r_cells, h_bank=r_h,
                      tick=np.asarray([r[1] for r in roots], np.int64),
                      bank_d=r_bank, end_d=field_d(gf, r_states["origin"]),
                      h=energy_h(r_states), root_dec=np.asarray([r[0] for r in roots]),
                      bank_t=np.asarray([r[6] for r in roots], np.int64), ndec=0, wave=-1,
                      ox=r_states["origin"][:, 0], oy=r_states["origin"][:, 1],
                      oz=r_states["origin"][:, 2])
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

    Kc = max(2, int(a.mcts_k))
    sigma = float(a.mcts_yaw_sigma)
    cpuct = float(a.mcts_cpuct)
    cell_w = arch.weight if float(a.mcts_cell_w) > 0 else None
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
        # Nature paper's policy-based explorer commits 50/50 to random or policy per step)
        explore = (rank > 0) & (rng.random(N) < float(a.mcts_perturb_frac))
        # random children (--mcts-random-frac of the sampled ones, before the explore draw): the
        # first decision drawn at --mcts-random-temp on every head and HELD for the macro
        rnd = (rank > 0) & (rng.random(N) < float(a.mcts_random_frac))
        explore &= ~rnd
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
                w_pact = np.asarray(polN.policy_act, np.int16).copy()
                w_pz = np.asarray(polN.view_z, np.float32).copy()
            view = np.array(polN.view, np.float32)
            if sigma > 0:
                if VIEW_ABS:
                    view[:, 0] = ((view[:, 0] + delta + 180.0) % 360.0) - 180.0
                else:
                    view[:, 0] = view[:, 0] + delta
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
                mm = min(M, len(G_acts) - d0)
                if mm <= 0:
                    continue
                n_cmp += mm
                n_key += int((acts_w[:mm, e] == G_acts[d0:d0 + mm]).all(axis=1).sum())
                dv = np.abs(view_w[:mm, e, 0] - G_view[d0:d0 + mm, 0])
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
        ntick = tree.tick[own] + np.where(fin, fin_t, M * K)
        n_before = tree.n
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
                                    h_bank=c_h_bank[ch])
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
                C_pact[ids] = w_pact[ch]
                C_pz[ids] = w_pz[ch]
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
                  f"{sim_ticks * N / max(dt, 1e-9):,.0f} env-steps/s", flush=True)
        wave += 1
        if fin.any() and bool(a.mcts_stop_on_finish):
            break

    # ---- the best line -------------------------------------------------------------------------
    b = tree.best_node()
    path = tree.path_to(b)
    root = path[0]
    rdec = int(tree.root_dec[root])
    acts = [G_acts[:rdec]]
    views = [G_view[:rdec]]
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
                "d": float(tree.bank_d[b]) if not finished else 0.0, "finished": finished}
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
    print(f"mcts: deepest RAW d any live tick reached {summary['raw_best_pct']}% (greedy raw "
          f"{100.0 * float(prog(gbest)):.2f}%)", flush=True)
    print(f"mcts: {wave} waves, {tree.n - 1:,} nodes, {arch.n:,} cells, best "
          f"{summary['best_pct']:.2f}% (greedy {summary['greedy_pct']:.2f}%) at tick {f_tick} "
          f"({ctx['secs'](f_tick):.2f}s), depth {int(tree.depth[b])}"
          + (" FINISHED" if finished else "") + f", {dt:.0f}s", flush=True)
    # EVERY simulator tick this call spent: the greedy pass (1 env) + every wave's copies
    summary["ticks_greedy"] = int(g_end)
    summary["ticks_waves"] = int(sim_ticks) * int(N)
    summary["ticks_total"] = int(g_end) + int(sim_ticks) * int(N)
    labels = None
    if LAB:
        labels = _tree_labels(ctx, a, mk_wrapper, tree, b, path, root, rdec, acts, views, n_dec,
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


def _tree_labels(ctx, a, mk_wrapper, tree, b, path, root, rdec, acts, views, n_dec,
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
    nt = node_targets(tree, L, C_pact, C_pz, nvec, npad, n_view)
    k = nt["keep"]
    rows = {"scal": [LS[k]], "pose": [LP[k]], "probs": [nt["probs"][k]], "vmu": [nt["vmu"][k]],
            "vsd": [nt["vsd"][k]], "value": [nt["value"][k]],
            "kind": [np.zeros(int(k.sum()), np.int8)]}
    # ---- line rows ---------------------------------------------------------------------------
    # the greedy prefix up to the line's root: ANCHOR rows (-1) only before the tree's EARLIEST
    # root - outside the search window, the policy's own moves; inside it (-2) the roots' own
    # node rows say whether the policy's move was the best one, and a self-label there would
    # contradict them (an idling untrained policy: "keep idling" at the very states where the
    # tree found the way off the platform)
    _c0, _n0 = int(tree.cstart[0]), int(tree.ncld[0])
    r_first = int(np.min(tree.root_dec[_c0:_c0 + _n0])) if _n0 else int(rdec)
    node_d = [int(root)] * int(rdec)
    kind_d = [-1 if d < r_first else -2 for d in range(int(rdec))]
    for nid in path[1:]:
        nd = int(tree.ndec[nid])
        node_d += [int(nid)] * nd
        kind_d += [int(C_kind[nid])] * nd
    node_d, kind_d = node_d[:n_dec], kind_d[:n_dec]
    cfg = ctx["cfg"]
    core1 = ctx["build_sim"](cfg, ctx["map_path"], 1, ctx["ep_cap"], tick=ctx["TICK"])
    ctx["arm"](core1)
    o = np.array(core1.reset(int(ctx["gate_seed"])))
    core1.set_state(0, ctx["row0"])
    o[0] = np.asarray(ctx["obs_start"], np.float32)
    pol, _side = mk_wrapper(core1, True)
    pol.capture = True
    pol.record_policy_act = True
    with torch.inference_mode():
        pol._obs(o)               # the greedy pass's own tick-0 call (its feeds' first read)
    keys = bool(ctx["KEYS"])
    l_scal, l_pose, l_pact, l_pz, l_val, l_rk = [], [], [], [], [], []
    ticks = 0
    n_held = 0
    for d in range(int(n_dec)):
        pol.act(o)
        # a decision of a SAMPLED / perturbed child (kind 2: none at --mcts-random-frac 1) has no
        # label here - its executed move was a draw this replay cannot rebuild - and nor has the
        # prefix inside the search window (-2); both are stepped
        if kind_d[d] not in (2, -2):
            l_scal.append(np.asarray(pol.last_scal, np.float32)[0].copy())
            l_pose.append(np.asarray(pol.last_pose, np.float32)[0].copy())
            if kind_d[d] == 1:
                l_pact.append(np.asarray(C_pact[node_d[d]], np.int64).copy())
                l_pz.append(np.asarray(C_pz[node_d[d]], np.float32).copy())
                n_held += 1
            else:
                l_pact.append(np.asarray(pol.policy_act, np.int64)[0].copy())
                l_pz.append(np.asarray(pol.view_z, np.float32)[0].copy())
            l_val.append(float(tree.q[node_d[d]]))
            # row kind: 1 = the line's SEARCHED part (the tree's choices), 2 = its greedy prefix
            l_rk.append(2 if kind_d[d] == -1 else 1)
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
            o, _r, dn, tr, _t = core1.step(row, view=vw)
            o = np.array(o)
            ticks += 1
            if bool(np.asarray(dn)[0]) or bool(np.asarray(tr)[0]):
                ended = True
                break
        if ended:
            break
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
    out = {}
    for key, parts in rows.items():
        parts = [p for p in parts if len(p)]
        out[key] = (np.concatenate(parts) if parts else
                    np.zeros((0,), np.float32))
    n_all = len(out["kind"]) if out["kind"].ndim else 0
    out["w"] = np.ones(n_all, np.float32)
    out["ticks"] = ticks
    out["n_expanded"] = len(L)
    out["replay_d"] = d_end
    print(f"mcts labels: {int(k.sum()):,} node rows (of {len(L):,} expanded nodes), {nl:,} line "
          f"rows ({n_held} on random-held macros); line replay "
          f"{ticks} ticks, ends at d {d_end:,.0f} (search said {float(tree.bank_d[b]):,.0f})",
          flush=True)
    return out
