"""treelabels.py - labels from EVERY search tree, for the trainer (the one-search series, slot 3).

THE IDEA (autoresearch/PROTOCOL.md, slot 3: "the tree teaches, beside PPO"). Run 5's MCTS
(tools/mcts_search.py) builds a tree of macro-actions rooted on the policy's own greedy line and
hands on ONE line, which reaches the policy only as spawn states. Here every EXPANDED node of every
tree also becomes a training label, and the trainer fits those labels as an extra loss next to
PPO's (train_fast.py --tree-labels). The search writes the labels itself (beam_tas.py
--mcts-labels -> <out-dir>/labels.npz); nobody passes a label file by hand.

* POLICY target. At an expanded node, the children that did better than the POLICY'S OWN
  child. Child 0 of every expansion is the policy's greedy macro (mcts_search: the mode, closed
  loop); the others are the search's samples (in run 5 all random-held). Each child is ranked
  among its siblings by its backed-up score Q (the tree's own value: the best BANKED progress
  anywhere in its subtree, max backup; a finish beats everything); at equal Q a child that
  survived its macro ranks above one that died; ties share their average rank, scaled to
  r in [0, 1]. A child's weight is w = max(0, r - r_0), r_0 the policy's own child's rank,
  normalised to sum 1 over the node's children: the label moves the policy only where the tree
  found something better than what the policy does, and more toward the bigger improvement. A
  node where nothing beats the policy's child teaches the policy nothing (measured on run 5's
  4.53B checkpoint: the greedy child is the best at ~70% of the nodes whose children differ, and
  a "better half of the siblings" target there is mostly random-held actions - a strong pull
  toward randomness, not a teacher). The label is the child's FIRST decision (the state it was
  taken in is the node's): its policy-space key / jump / duck bins and its EXECUTED yaw target
  as u = off_warp_inv(offset) in [-1, 1] (the velocity-frame offset the core steered to; under
  --view-absolute velocity that is the whole steer). The pitch only aims the camera and is not
  labelled.
* VALUE target. The node's own backed-up score Q, as its percentile rank among the tree's
  expanded nodes (0 = the tree's worst, 1 = its best), on EVERY expanded node. It is not PPO's discounted return and it
  never enters PPO's critic: the trainer fits it with its own head on the ACTOR tower (the
  auxiliary value head of Phasic Policy Gradient, Cobbe et al. 2021), so PPO's baseline is
  untouched.

Both targets are ranks inside one tree: no number in them comes from the map (no progress
value, no place, no speed), so the rule ships unchanged to a map nobody has opened.

The input row of a label is the policy's own: the scalar half of the row the search's policy
wrapper built at that node (train_fast._TorchPolicyBase._obs, everything before the image) and
the pose the trainer re-renders the image from (origin, yaw, pitch, ducked - the image is a pure
function of the pose, exactly as surfgym/bc.py's BCDataset renders its rows).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["LABELS_VERSION", "sibling_weights", "improve_weights", "value_ranks",
           "build_labels", "save_labels", "load_labels", "TreeLabelBuffer"]

LABELS_VERSION = 2


def _avg_rank_in_groups(group, keys):
    """Average rank (0-based, ties share their mean position) of every element inside its
    group, by ascending ``keys`` (a tuple of 1-d arrays, the LAST one the primary key, as
    np.lexsort reads them). -> (rank, group size), both float64, in the input order."""
    group = np.asarray(group, np.int64)
    n = len(group)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    order = np.lexsort(tuple(np.asarray(k) for k in keys) + (group,))
    g = group[order]
    ks = [np.asarray(k)[order] for k in keys]
    new_group = np.ones(n, bool)
    new_group[1:] = g[1:] != g[:-1]
    gstart = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))
    pos = (np.arange(n) - gstart).astype(np.float64)
    new_tie = new_group.copy()
    for k in ks:
        d = np.ones(n, bool)
        d[1:] = k[1:] != k[:-1]
        new_tie |= d
    tie = np.cumsum(new_tie) - 1
    avg = np.bincount(tie, weights=pos) / np.bincount(tie)
    gid = np.cumsum(new_group) - 1
    gsize = np.bincount(gid).astype(np.float64)
    rank = np.empty(n)
    size = np.empty(n)
    rank[order] = avg[tie]
    size[order] = gsize[gid]
    return rank, size


def sibling_weights(group, q, survived):
    """The policy target's weights. ``group``: (m,) the parent of each child; ``q``: (m,) the
    child's backed-up score; ``survived``: (m,) bool, the child lived through its macro (or
    finished). -> (m,) float64 weights: max(0, r - 1/2) with r the child's average rank among its
    siblings scaled to [0, 1], normalised to sum 1 per parent (0 for a parent whose children all
    tie, and for an only child)."""
    q = np.asarray(q, np.float64)
    q = np.where(np.isnan(q), -np.inf, q)
    rank, size = _avg_rank_in_groups(group, (np.asarray(survived, np.int8), q))
    r = np.where(size > 1, rank / np.maximum(size - 1.0, 1.0), 0.5)
    w = np.maximum(0.0, r - 0.5)
    gid = np.unique(np.asarray(group, np.int64), return_inverse=True)[1]
    s = np.bincount(gid, weights=w)
    return np.where(s[gid] > 0, w / np.where(s[gid] > 0, s[gid], 1.0), 0.0)


def improve_weights(group, q, survived, own):
    """The policy target's weights: toward the children that beat the POLICY'S OWN child.
    ``group``: (m,) the parent of each child; ``q``: (m,) backed-up score; ``survived``: (m,)
    bool; ``own``: (m,) bool, True on each parent's own (greedy) child - exactly one per parent.
    -> (m,) float64 weights max(0, r - r_own) (average sibling ranks scaled to [0, 1]),
    normalised to sum 1 per parent; 0 everywhere under a parent where nothing beats its own
    child."""
    q = np.asarray(q, np.float64)
    q = np.where(np.isnan(q), -np.inf, q)
    group = np.asarray(group, np.int64)
    own = np.asarray(own, bool)
    rank, size = _avg_rank_in_groups(group, (np.asarray(survived, np.int8), q))
    r = np.where(size > 1, rank / np.maximum(size - 1.0, 1.0), 0.5)
    gid = np.unique(group, return_inverse=True)[1]
    r_own = np.full(gid.max() + 1 if len(gid) else 0, np.nan)
    r_own[gid[own]] = r[own]
    if np.isnan(r_own).any():
        raise ValueError("improve_weights: every parent needs exactly one own child")
    w = np.maximum(0.0, r - r_own[gid])
    s = np.bincount(gid, weights=w)
    return np.where(s[gid] > 0, w / np.where(s[gid] > 0, s[gid], 1.0), 0.0)


def value_ranks(q):
    """Percentile rank of each value among all of them, in [0, 1] (ties share their mean;
    a single value -> 0.5)."""
    q = np.asarray(q, np.float64)
    q = np.where(np.isnan(q), -np.inf, q)
    if len(q) <= 1:
        return np.full(len(q), 0.5)
    rank, _ = _avg_rank_in_groups(np.zeros(len(q), np.int64), (q,))
    return rank / (len(q) - 1.0)


def build_labels(parent, ncld, cstart, q, alive, finished, lab_node, lab_scal, lab_pose,
                 c_act, c_u):
    """One tree -> its labels.

    ``parent, ncld, cstart, q, alive, finished``: the tree's per-node arrays (mcts_search.Tree,
    node 0 the virtual root; the first child of every block is the policy's own greedy child);
    ``lab_node`` (m,): the expanded nodes whose input row was captured when they were expanded,
    ``lab_scal`` (m, D) / ``lab_pose`` (m, 6) those rows; ``c_act`` (n, 6) int: every node's
    FIRST decision as a child of its parent (policy space), ``c_u`` (n,) its executed yaw target
    in u. -> a dict of arrays - per expanded node ``scal``, ``pose``, ``value`` and ``first``
    (the index of its first policy row, -1 when nothing beat its own child); per policy row
    (w > 0) ``node``, ``act``, ``u``, ``w`` - plus ``stats``; None when no node was captured."""
    lab_node = np.asarray(lab_node, np.int64)
    keep = (lab_node > 0) & (np.asarray(ncld)[lab_node] > 0)
    P = lab_node[keep]
    S = np.asarray(lab_scal, np.float32)[keep]
    Po = np.asarray(lab_pose, np.float32)[keep]
    if len(P) == 0:
        return None
    ncld = np.asarray(ncld, np.int64)
    cstart = np.asarray(cstart, np.int64)
    cnt = ncld[P]
    tot = int(cnt.sum())
    pidx = np.repeat(np.arange(len(P)), cnt)
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ch = np.repeat(cstart[P], cnt) + off
    q = np.asarray(q, np.float64)
    qc = np.where(np.isnan(q[ch]), -np.inf, q[ch])
    surv = np.asarray(alive, bool)[ch] | np.asarray(finished, bool)[ch]
    own = off == 0
    w = improve_weights(pidx, qc, surv, own)
    val = value_ranks(q[P])
    pos = w > 0
    taught = np.zeros(len(P), bool)
    taught[pidx[pos]] = True
    rows_node = pidx[pos]
    first = np.full(len(P), -1, np.int64)
    if pos.any():
        # rows come out grouped by node (ch is node-major): a node's first row
        _r = np.arange(int(pos.sum()))
        _new = np.ones(len(_r), bool)
        _new[1:] = rows_node[1:] != rows_node[:-1]
        first[rows_node[_new]] = _r[_new]
    # how the tree compares with the policy: how often its own (greedy) child is a best child
    # of its siblings, how often some child beats it, how often the children all tie
    best = np.full(len(P), -np.inf)
    np.maximum.at(best, pidx, qc)
    rank, _size = _avg_rank_in_groups(pidx, (surv.astype(np.int8), qc))
    rmax = np.full(len(P), -np.inf)
    rmin = np.full(len(P), np.inf)
    np.maximum.at(rmax, pidx, rank)
    np.minimum.at(rmin, pidx, rank)
    tie_all = rmax == rmin
    stats = {"expanded": int(len(P)), "taught": int(taught.sum()), "rows": int(pos.sum()),
             "children": int(tot), "tied_nodes": int(tie_all.sum()),
             "own_best": round(float((qc[own] >= best).mean()), 4),
             "own_beaten": round(float(taught.mean()), 4)}
    return {"scal": S, "pose": Po, "value": val.astype(np.float32), "first": first,
            "node": rows_node.astype(np.int32),
            "act": np.asarray(c_act)[ch[pos]].astype(np.int8),
            "u": np.asarray(c_u, np.float32)[ch[pos]], "w": w[pos].astype(np.float32),
            "stats": stats}


def save_labels(path, lab: dict, meta: dict) -> None:
    """Write one tree's labels (build_labels' dict) and its meta (json) to ``path`` (.npz)."""
    m = dict(meta)
    m["version"] = LABELS_VERSION
    m["stats"] = lab.get("stats", {})
    np.savez(path, version=np.int64(LABELS_VERSION), meta=json.dumps(m),
             scal=lab["scal"], pose=lab["pose"], value=lab["value"], first=lab["first"],
             node=lab["node"], act=lab["act"], u=lab["u"], w=lab["w"])


def load_labels(path) -> dict:
    """-> {meta, scal, pose, value, first, node, act, u, w} as numpy arrays (validated)."""
    z = np.load(path, allow_pickle=False)
    ver = int(z["version"])
    if ver != LABELS_VERSION:
        raise ValueError(f"{path}: tree-label version {ver}, this reader knows {LABELS_VERSION}")
    out = {"meta": json.loads(str(z["meta"]))}
    for k in ("scal", "pose", "value", "first", "node", "act", "u", "w"):
        out[k] = np.asarray(z[k])
    nn, nr = len(out["scal"]), len(out["node"])
    if out["pose"].shape != (nn, 6) or out["value"].shape != (nn,) or out["first"].shape != (nn,):
        raise ValueError(f"{path}: node arrays disagree ({out['pose'].shape}, "
                         f"{out['value'].shape} for {nn} rows)")
    if out["act"].shape[0] != nr or out["u"].shape != (nr,) or out["w"].shape != (nr,):
        raise ValueError(f"{path}: row arrays disagree")
    if nr and (out["node"].min() < 0 or out["node"].max() >= nn):
        raise ValueError(f"{path}: a row points outside the node table")
    f = out["first"]
    if (f >= nr).any() or (f < -1).any() or (nr and not np.array_equal(
            np.flatnonzero(f >= 0), np.unique(out["node"]))):
        raise ValueError(f"{path}: the nodes' first rows disagree with the rows")
    if not (np.isfinite(out["scal"]).all() and np.isfinite(out["pose"]).all()
            and np.isfinite(out["u"]).all() and np.isfinite(out["w"]).all()
            and np.isfinite(out["value"]).all()):
        raise ValueError(f"{path}: non-finite labels")
    return out


class TreeLabelBuffer:
    """The trainer's labels: the last ``keep`` trees, on the device, sampled with a PRIVATE
    torch generator (the PPO permutation stream and the action noise are untouched).

    ``sample(n)`` draws n expanded nodes uniformly over the kept trees and, for each node where
    some child beat the policy's own, one of its policy rows in proportion to the rows' weights.
    -> (scal, pose, act, u, value, mask): the node's scalar row and pose, the drawn child's
    first-decision bins and yaw u (zeros where the node has no row), the node's value rank, and
    mask 1.0 where the node carries a policy row (the value term reads every node)."""

    def __init__(self, device, scal_dim: int, keep: int, seed: int = 0):
        import torch
        self.device = device
        self.scal_dim = int(scal_dim)
        self.keep = max(1, int(keep))
        self.trees = []
        self.n_nodes = self.n_rows = self.n_taught = 0
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed((int(seed) * 6700417 + 977) & 0x7FFFFFFFFFFFFFFF)
        self._cat = None

    def __len__(self):
        return self.n_nodes

    def add(self, path, tag: str = "") -> dict:
        """Load one tree's labels; the oldest tree drops out past ``keep``. -> the file's stats
        (raises ValueError on a malformed or mismatched file: the caller reports it)."""
        import torch
        lab = load_labels(path)
        if lab["scal"].shape[1] != self.scal_dim:
            raise ValueError(f"{path}: label rows are {lab['scal'].shape[1]} scalars wide, the "
                             f"trainer's policy reads {self.scal_dim}")
        if len(lab["scal"]) == 0:
            return lab["meta"].get("stats", {})
        d = self.device
        nr = len(lab["node"])
        cnt = np.bincount(lab["node"], minlength=len(lab["scal"])) if nr else \
            np.zeros(len(lab["scal"]), np.int64)
        t = {"tag": str(tag), "meta": lab["meta"],
             "scal": torch.as_tensor(lab["scal"], dtype=torch.float32, device=d),
             "pose": torch.as_tensor(lab["pose"], dtype=torch.float32, device=d),
             "value": torch.as_tensor(lab["value"], dtype=torch.float32, device=d),
             "first": torch.as_tensor(lab["first"], dtype=torch.long, device=d),
             "count": torch.as_tensor(cnt, dtype=torch.long, device=d),
             "act": torch.as_tensor(lab["act"].reshape(nr, 6), dtype=torch.long, device=d),
             "u": torch.as_tensor(lab["u"], dtype=torch.float32, device=d),
             "w": torch.as_tensor(lab["w"], dtype=torch.float64, device=d)}
        self.trees.append(t)
        while len(self.trees) > self.keep:
            self.trees.pop(0)
        self._rebuild()
        return lab["meta"].get("stats", {})

    def _rebuild(self):
        import torch
        roff, firsts = 0, []
        for t in self.trees:
            firsts.append(torch.where(t["first"] >= 0, t["first"] + roff, t["first"]))
            roff += len(t["u"])
        cat = {k: torch.cat([t[k] for t in self.trees]) for k in ("scal", "pose", "value",
                                                                  "count", "act", "u", "w")}
        cat["first"] = torch.cat(firsts)
        cdf = torch.cumsum(cat["w"], 0)
        # a node's rows span (cdf[first] - w[first], cdf[last]]: lo and its mass
        f = cat["first"].clamp(min=0)
        has = cat["first"] >= 0
        if len(cdf):
            lo = cdf[f] - cat["w"][f]
            hi = cdf[(f + cat["count"] - 1).clamp(min=0)]
            cat["lo"] = torch.where(has, lo, torch.zeros_like(lo))
            cat["mass"] = torch.where(has, hi - lo, torch.zeros_like(lo))
        else:
            cat["lo"] = cat["mass"] = torch.zeros(len(f), dtype=torch.float64,
                                                  device=self.device)
        cat["cdf"] = cdf
        cat["has"] = has
        self.n_nodes = int(len(cat["scal"]))
        self.n_rows = int(len(cat["u"]))
        self.n_taught = int(has.sum().item())
        self._cat = cat

    def sample(self, n: int):
        import torch
        c = self._cat
        n = int(n)
        k = torch.randint(0, self.n_nodes, (n,), generator=self.gen, device=self.device)
        r = torch.rand(n, generator=self.gen, device=self.device, dtype=torch.float64)
        has = c["has"][k]
        if self.n_rows:
            x = c["lo"][k] + r * c["mass"][k]
            i = torch.searchsorted(c["cdf"], x, right=True)
            f = c["first"][k].clamp(min=0)
            i = torch.minimum(torch.maximum(i, f), f + (c["count"][k] - 1).clamp(min=0))
            i = torch.where(has, i, torch.zeros_like(i)).clamp_(max=self.n_rows - 1)
            act = torch.where(has.unsqueeze(1), c["act"][i], torch.zeros_like(c["act"][i]))
            u = torch.where(has, c["u"][i], torch.zeros_like(c["u"][i]))
        else:
            act = torch.zeros((n, 6), dtype=torch.long, device=self.device)
            u = torch.zeros(n, dtype=torch.float32, device=self.device)
        return (c["scal"][k], c["pose"][k], act, u, c["value"][k],
                has.to(torch.float32))

    @staticmethod
    def render(lidar, pose, dtype):
        """(n, FRAME) image in the rollout buffer's dtype from a (n, 6) pose block - the same
        call surfgym.bc.BCDataset.render makes (the trainer's own lidar, every channel)."""
        img = lidar.render(pose[:, 0:3], pose[:, 3], pose[:, 4], pose[:, 5])
        return img.reshape(pose.shape[0], -1).to(dtype)

    def describe(self) -> str:
        if not self.trees:
            return "tree labels: none yet"
        return (f"tree labels: {self.n_nodes:,} nodes ({self.n_taught:,} with a policy row, "
                f"{self.n_rows:,} rows) from {len(self.trees)} tree(s) "
                f"[{', '.join(t['tag'] for t in self.trees)}]")
