"""treelabels.py - labels from EVERY search tree, for the trainer (the one-search series, slot 3).

THE IDEA (autoresearch/PROTOCOL.md, slot 3: "the tree teaches, beside PPO"). Run 5's MCTS
(tools/mcts_search.py) builds a tree of macro-actions rooted on the policy's own greedy line and
hands on ONE line, which reaches the policy only as spawn states. Here every EXPANDED node of every
tree also becomes a training label, and the trainer fits those labels as an extra loss next to
PPO's (train_fast.py --tree-labels). The search writes the labels itself (beam_tas.py
--mcts-labels -> <out-dir>/labels.npz); nobody passes a label file by hand.

* POLICY target. At an expanded node, the children that did better than their siblings. Each
  child is ranked among its siblings by its backed-up score Q (the tree's own value: the best
  BANKED progress anywhere in its subtree, max backup; a finish beats everything); at equal Q a
  child that survived its macro ranks above one that died. Its average rank (ties share one) is
  scaled to r in [0, 1] (0 = the worst sibling, 1 = the best) and its weight is
  w = max(0, r - 1/2), normalised to sum 1 over the node's children: the better half pulls, the
  best child hardest, the worse half not at all. A node whose children all tie teaches nothing.
  The label is the child's FIRST decision (the state it was taken in is the node's): its
  policy-space key / jump / duck bins and its EXECUTED yaw target as u = off_warp_inv(offset)
  in [-1, 1] (the velocity-frame offset the core steered to; under --view-absolute velocity
  that is the whole steer). The pitch only aims the camera and is not labelled.
* VALUE target. The node's own backed-up score Q, as its percentile rank among the tree's
  expanded nodes (0 = the tree's worst, 1 = its best). It is not PPO's discounted return and it
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

__all__ = ["LABELS_VERSION", "sibling_weights", "value_ranks", "build_labels", "save_labels",
           "load_labels", "TreeLabelBuffer"]

LABELS_VERSION = 1


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
    node 0 the virtual root); ``lab_node`` (m,): the expanded nodes whose input row was captured
    when they were expanded, ``lab_scal`` (m, D) / ``lab_pose`` (m, 6) those rows; ``c_act``
    (n, 6) int: every node's FIRST decision as a child of its parent (policy space), ``c_u``
    (n,) its executed yaw target in u. -> a dict of arrays (``scal``, ``pose``, ``value`` per
    labelled node; ``node``, ``act``, ``u``, ``w`` per (node, child) row with w > 0) plus
    ``stats``, or None when no node teaches anything."""
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
    surv = np.asarray(alive, bool)[ch] | np.asarray(finished, bool)[ch]
    w = sibling_weights(pidx, q[ch], surv)
    val = value_ranks(q[P])
    pos = w > 0
    taught = np.zeros(len(P), bool)
    taught[pidx[pos]] = True
    if not taught.any():
        return None
    remap = np.cumsum(taught) - 1
    child0 = off == 0
    # how much the tree disagrees with the policy: the weight its own greedy child (child 0 of
    # every expansion) gets, and how often that child is the best of its siblings
    w0 = w[child0 & taught[pidx]]
    best = np.full(len(P), -np.inf)
    np.maximum.at(best, pidx, np.where(np.isnan(q[ch]), -np.inf, q[ch]))
    c0_best = (q[ch][child0] >= best[pidx[child0]]) & taught
    stats = {"expanded": int(len(P)), "nodes": int(taught.sum()), "rows": int(pos.sum()),
             "children": int(tot), "tied_nodes": int((~taught).sum()),
             "child0_w": round(float(w0.mean()) if len(w0) else 0.0, 4),
             "child0_best": round(float(c0_best.sum() / max(1, taught.sum())), 4)}
    return {"scal": S[taught], "pose": Po[taught], "value": val[taught].astype(np.float32),
            "node": remap[pidx[pos]].astype(np.int32),
            "act": np.asarray(c_act)[ch[pos]].astype(np.int8),
            "u": np.asarray(c_u, np.float32)[ch[pos]], "w": w[pos].astype(np.float32),
            "stats": stats}


def save_labels(path, lab: dict, meta: dict) -> None:
    """Write one tree's labels (build_labels' dict) and its meta (json) to ``path`` (.npz)."""
    m = dict(meta)
    m["version"] = LABELS_VERSION
    m["stats"] = lab.get("stats", {})
    np.savez(path, version=np.int64(LABELS_VERSION), meta=json.dumps(m),
             scal=lab["scal"], pose=lab["pose"], value=lab["value"], node=lab["node"],
             act=lab["act"], u=lab["u"], w=lab["w"])


def load_labels(path) -> dict:
    """-> {meta, scal, pose, value, node, act, u, w} as numpy arrays (validated)."""
    z = np.load(path, allow_pickle=False)
    ver = int(z["version"])
    if ver != LABELS_VERSION:
        raise ValueError(f"{path}: tree-label version {ver}, this reader knows {LABELS_VERSION}")
    out = {"meta": json.loads(str(z["meta"]))}
    for k in ("scal", "pose", "value", "node", "act", "u", "w"):
        out[k] = np.asarray(z[k])
    nn, nr = len(out["scal"]), len(out["node"])
    if out["pose"].shape != (nn, 6) or out["value"].shape != (nn,):
        raise ValueError(f"{path}: node arrays disagree ({out['pose'].shape}, "
                         f"{out['value'].shape} for {nn} rows)")
    if out["act"].shape[0] != nr or out["u"].shape != (nr,) or out["w"].shape != (nr,):
        raise ValueError(f"{path}: row arrays disagree")
    if nr and (out["node"].min() < 0 or out["node"].max() >= nn):
        raise ValueError(f"{path}: a row points outside the node table")
    if not (np.isfinite(out["scal"]).all() and np.isfinite(out["pose"]).all()
            and np.isfinite(out["u"]).all() and np.isfinite(out["w"]).all()
            and np.isfinite(out["value"]).all()):
        raise ValueError(f"{path}: non-finite labels")
    return out


class TreeLabelBuffer:
    """The trainer's labels: the last ``keep`` trees, on the device, sampled with a PRIVATE
    torch generator (the PPO permutation stream and the action noise are untouched).

    ``sample(n)`` draws n (node, child) rows with probability proportional to their weight -
    every labelled node carries total weight 1, so the nodes of the kept trees are drawn
    uniformly and a node's children by their weight - and returns (scal, pose, act, u, value):
    the node's scalar row and pose, the child's first-decision bins and yaw u, and the node's
    value rank."""

    def __init__(self, device, scal_dim: int, keep: int, seed: int = 0):
        import torch
        self.device = device
        self.scal_dim = int(scal_dim)
        self.keep = max(1, int(keep))
        self.trees = []
        self.n_nodes = self.n_rows = 0
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed((int(seed) * 6700417 + 977) & 0x7FFFFFFFFFFFFFFF)
        self._cat = None

    def __len__(self):
        return self.n_rows

    def add(self, path, tag: str = "") -> dict:
        """Load one tree's labels; the oldest tree drops out past ``keep``. -> the file's stats
        (raises ValueError on a malformed or mismatched file: the caller reports it)."""
        import torch
        lab = load_labels(path)
        if lab["scal"].shape[1] != self.scal_dim:
            raise ValueError(f"{path}: label rows are {lab['scal'].shape[1]} scalars wide, the "
                             f"trainer's policy reads {self.scal_dim}")
        if len(lab["node"]) == 0:
            return lab["meta"].get("stats", {})
        d = self.device
        t = {"tag": str(tag), "meta": lab["meta"],
             "scal": torch.as_tensor(lab["scal"], dtype=torch.float32, device=d),
             "pose": torch.as_tensor(lab["pose"], dtype=torch.float32, device=d),
             "value": torch.as_tensor(lab["value"], dtype=torch.float32, device=d),
             "node": torch.as_tensor(lab["node"], dtype=torch.long, device=d),
             "act": torch.as_tensor(lab["act"], dtype=torch.long, device=d),
             "u": torch.as_tensor(lab["u"], dtype=torch.float32, device=d),
             "w": torch.as_tensor(lab["w"], dtype=torch.float64, device=d)}
        self.trees.append(t)
        while len(self.trees) > self.keep:
            self.trees.pop(0)
        self._rebuild()
        return lab["meta"].get("stats", {})

    def _rebuild(self):
        import torch
        off, nodes = 0, []
        for t in self.trees:
            nodes.append(t["node"] + off)
            off += len(t["scal"])
        self.n_nodes = off
        cat = {k: torch.cat([t[k] for t in self.trees]) for k in ("scal", "pose", "value",
                                                                  "act", "u", "w")}
        cat["node"] = torch.cat(nodes)
        cdf = torch.cumsum(cat.pop("w"), 0)
        cat["total"] = float(cdf[-1].item())
        cat["cdf"] = cdf
        self.n_rows = int(len(cat["node"]))
        self._cat = cat

    def sample(self, n: int):
        import torch
        c = self._cat
        r = torch.rand(int(n), generator=self.gen, device=self.device, dtype=torch.float64)
        i = torch.searchsorted(c["cdf"], r * c["total"], right=True).clamp_(max=self.n_rows - 1)
        nd = c["node"][i]
        return c["scal"][nd], c["pose"][nd], c["act"][i], c["u"][i], c["value"][nd]

    @staticmethod
    def render(lidar, pose, dtype):
        """(n, FRAME) image in the rollout buffer's dtype from a (n, 6) pose block - the same
        call surfgym.bc.BCDataset.render makes (the trainer's own lidar, every channel)."""
        img = lidar.render(pose[:, 0:3], pose[:, 3], pose[:, 4], pose[:, 5])
        return img.reshape(pose.shape[0], -1).to(dtype)

    def describe(self) -> str:
        if not self.trees:
            return "tree labels: none yet"
        return (f"tree labels: {self.n_nodes:,} nodes / {self.n_rows:,} rows from "
                f"{len(self.trees)} tree(s) [{', '.join(t['tag'] for t in self.trees)}]")
