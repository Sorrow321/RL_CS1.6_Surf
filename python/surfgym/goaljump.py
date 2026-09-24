"""goaljump.py - the JUMP-POINT planner (docs/planner-design.md section 7, step 1). No network.

At every plan decision an env's OPTIONS are the distinct places it can walk to in one jump
(``--jump-len`` u, ~3 s at walking speed):

1. Dijkstra on stage 1's walkable graph (goalplan.BFSPlanner) from the agent's node, stopped at
   the jump length - every floor cell reachable in one jump, with its fastest walk;
2. 8 compass probes: each takes the reachable cell FURTHEST ALONG its direction (the largest
   component of its offset along it; a direction that cannot advance MIN_ADV_FRAC of a jump is
   dropped); a finish-box cell in reach is always an option;
3. probe ends within MERGE_U of each other ALONG THE FLOOR are one option.

A SEARCH expands options of options ``--jump-depth`` jumps deep. A node is worth U(node) (or
FIN_BONUS for the finish box) minus LEN_COST per 1,000 u walked to it; each first option is worth
the best node of its subtree. The plan is a draw from softmax(value / ``--jump-T``) (the eval
takes the argmax) and the executor's line is the chosen option's fastest walk - a polyline like a
BFS plan's, read through the same fan.

U (``--jump-u``): ``euclid`` = -(straight-line distance to the finish) / 1,000 u; ``novelty`` =
1 / sqrt(1 + n), n the fleet's count of executor visits in the node's 128 u cell over the whole
run (the planner's memory of where it has been, kept in the checkpoint); ``episodic`` = the same
over THIS episode's own cell entries (reset every episode - the loop breaker: "not where I have
already been"); ``euclid+episodic`` = euclid + EPI_W x episodic.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .goallearn import PlanState, BUDGET_MULT, BUDGET_SPEED_U, COMPLETE_FRAC

__all__ = ["JumpGraph", "JumpPlanner", "make_jump_hooks", "JUMP_COLS",
           "JUMP_SEED_OFFSET", "JUMP_DEFAULTS"]

JUMP_LEN_U = 750.0          # one jump: ~3 s at player_maxspeed 250 u/s
MIN_ADV_FRAC = 0.3          # a probe must advance this share of a jump along its direction
MERGE_U = 200.0             # probe ends this close along the floor are one option
N_DIRS = 8
NOVELTY_CELL_U = 128.0
LEN_COST = 0.1              # value per 1,000 u walked (prefers the shorter of equal plans)
FIN_BONUS = 10000.0         # a jump that reaches the finish box: always above any U - the
                            # first version (100) went NEGATIVE on routes > 10,000 u (depth 16 on
                            # labyrinth_hard01), so a seen finish lost to plain straight-line options
FIN_LEN_COST = 10.0         # ... minus this per 1,000 u walked: the SHORTEST route to it wins
NOV_REFRESH_TICKS = 250     # novelty values are re-read from the counts this often
EPI_W = 1.0                 # euclid+episodic: weight of the episodic novelty term
U_KINDS = ("euclid", "novelty", "episodic", "euclid+episodic")
L_MAX = 24                  # line points (a jump <= 750 u at the 128 u fan spacing, + finish)
JUMP_SEED_OFFSET = 4421     # the planner's RNG: --seed + this
JUMP_DEFAULTS = {"jump_depth": 3, "jump_u": "euclid", "jump_t": 0.05,
                 "jump_len": JUMP_LEN_U}

JUMP_COLS = ["plan/chosen", "plan/options", "plan/value_best", "plan/closed",
             "plan/complete", "plan/fin_seen", "plan/cover", "plan/finish",
             "plan/finish_start", "plan/eval_finish"]


class JumpGraph:
    """The options of every walkable node, computed on first use and cached (they depend only
    on the map), and the depth-limited search over them."""

    def __init__(self, graph, jump_len: float = JUMP_LEN_U):
        import scipy.sparse as sp
        if getattr(graph, "fin", None) is None:
            raise ValueError("the jump planner needs the map's finish box")
        self.g = graph
        self.xyz = np.asarray(graph.xyz, np.float64)
        self.xy = self.xyz[:, :2]
        n = len(self.xyz)
        nbr = np.asarray(graph.nbr)
        rows = np.repeat(np.arange(n), nbr.shape[1])
        cols = nbr.reshape(-1)
        # --plan-graph tight: its per-edge costs (a jump's length is then route COST)
        w = (np.asarray(graph.wedge, np.float64).reshape(-1)
             if getattr(graph, "wedge", None) is not None
             else np.tile(np.asarray(graph.wk, np.float64), n))
        m = cols >= 0
        self.A = sp.csr_matrix((w[m], (rows[m], cols[m])), shape=(n, n))
        self.n = n
        self.is_fin = np.zeros(n, bool)
        self.is_fin[np.asarray(graph.finish_nodes, np.int64)] = True
        self.jump_len = float(jump_len)
        self.min_adv = MIN_ADV_FRAC * self.jump_len
        a = np.radians(np.arange(N_DIRS) * 360.0 / N_DIRS)
        self.dirs = np.stack([np.cos(a), np.sin(a)], axis=1)
        self._opts = {}

    def _dij(self, s, limit):
        from scipy.sparse.csgraph import dijkstra
        return dijkstra(self.A, directed=True, indices=int(s), limit=float(limit),
                        return_predecessors=True)

    @staticmethod
    def _walk(pred, s, t):
        out = [int(t)]
        while out[-1] != s and out[-1] >= 0:
            out.append(int(pred[out[-1]]))
        return np.asarray(out[::-1], np.int64)

    def options(self, s: int):
        """-> list of (end node, path nodes, path length, reaches the finish)."""
        s = int(s)
        hit = self._opts.get(s)
        if hit is not None:
            return hit
        d, pred = self._dij(s, self.jump_len)
        idx = np.flatnonzero(np.isfinite(d))
        ends = []
        fin_in = idx[self.is_fin[idx]]
        if len(fin_in):
            ends.append(int(fin_in[np.argmin(d[fin_in])]))
        off = self.xy[idx] - self.xy[s]
        for u in self.dirs:
            pr = off @ u
            j = int(np.argmax(pr - 0.02 * d[idx]))
            if pr[j] >= self.min_adv:
                ends.append(int(idx[j]))
        if not ends and len(idx) > 1:
            # a pocket where nothing advances far: the reachable cell furthest along the floor
            ends.append(int(idx[np.argmax(d[idx])]))
        opts = []
        cand = [t for t in dict.fromkeys(ends) if t != s]
        if cand:
            # one multi-source Dijkstra for every probe end (same merges as one call per pair)
            from scipy.sparse.csgraph import dijkstra
            D = np.atleast_2d(dijkstra(self.A, directed=True, indices=np.asarray(cand, np.int64),
                                       limit=MERGE_U))
            row = {t: i for i, t in enumerate(cand)}
            for t in ends:
                if t == s:
                    continue
                if any(np.isfinite(D[row[o[0]], t]) for o in opts):
                    continue
                opts.append((t, self._walk(pred, s, t), float(d[t]), bool(self.is_fin[t])))
        self._opts[s] = opts
        return opts

    def decide(self, root: int, depth: int, uval, memo: dict):
        """-> (options, value per option). ``uval(node)`` is U; ``memo`` caches, per (node,
        jumps left), (the best U - LEN_COST x walk over the subtree, the shortest walk to the
        finish box within the subtree - inf when it is out of reach). An option whose subtree
        reaches the finish is worth FIN_BONUS - FIN_LEN_COST x the whole walk there (the
        shortest route to the finish wins); any other, its best U."""
        def best(v: int, r: int):
            if self.is_fin[v]:
                return uval(v), 0.0
            key = (v, r)
            b = memo.get(key)
            if b is not None:
                return b
            ub, fd = uval(v), math.inf
            if r > 0:
                for t, _p, L, _f in self.options(v):
                    ub_t, fd_t = best(t, r - 1)
                    ub = max(ub, ub_t - LEN_COST * L / 1000.0)
                    fd = min(fd, L + fd_t)
            memo[key] = (ub, fd)
            return ub, fd

        opts = self.options(root)
        vals = []
        for t, _p, L, _f in opts:
            ub, fd = best(t, max(0, depth - 1))
            vals.append(FIN_BONUS - FIN_LEN_COST * (L + fd) / 1000.0 if fd < math.inf
                        else ub - LEN_COST * L / 1000.0)
        return opts, np.asarray(vals, np.float64)

    def line(self, o, path, graph):
        """The executor's line for a jump from the agent at ``o`` along ``path`` (graph nodes),
        built like goalplan.BFSPlanner.plan's: the start's height above its floor kept all
        along, Douglas-Peucker at one cell, resampled at the fan spacing; a jump that reaches
        the finish box ends at its centre."""
        o = np.asarray(o, np.float64).reshape(3)
        path = np.asarray(path, np.int64)
        h = float(o[2] - graph.floor[path[0]])
        pts = self.xyz[path].copy()
        pts[:, 2] = graph.floor[path] + h
        raw = np.vstack([o[None, :], pts[1:]]) if len(pts) > 1 else np.vstack([o, pts[0]])
        g = raw[-1].copy()
        if self.is_fin[path[-1]]:
            fc = np.asarray(graph.finish_center, np.float64).copy()
            fc[2] = float(graph.floor[path[-1]] + h)
            raw = np.vstack([raw, fc[None, :]])
            g = fc
        ln = np.asarray(graph.line_from(raw, o, g), np.float32)
        return ln[:L_MAX]


class _Novelty:
    """The fleet's visit counts on 128 u cells over the graph's box (never reset)."""

    def __init__(self, graph):
        xyz = np.asarray(graph.xyz, np.float64)
        self.mins = xyz.min(axis=0) - NOVELTY_CELL_U
        span = xyz.max(axis=0) + NOVELTY_CELL_U - self.mins
        self.shape = tuple(int(v) for v in np.ceil(span / NOVELTY_CELL_U).astype(int) + 1)
        self.count = np.zeros(self.shape, np.int64)
        self.node_cell = self.cells(xyz)

    def cells(self, p):
        k = np.floor((np.atleast_2d(p) - self.mins[None, :]) / NOVELTY_CELL_U).astype(np.int64)
        for a in range(3):
            np.clip(k[:, a], 0, self.shape[a] - 1, out=k[:, a])
        return (k[:, 0] * self.shape[1] + k[:, 1]) * self.shape[2] + k[:, 2]

    def add(self, p) -> None:
        np.add.at(self.count.reshape(-1), self.cells(p), 1)

    def u(self, node: int) -> float:
        return 1.0 / math.sqrt(1.0 + float(self.count.reshape(-1)[self.node_cell[node]]))

    def covered(self) -> int:
        return int((self.count > 0).sum())


class _Episodic:
    """Per-env cell ENTRIES of the current episode on the _Novelty lattice (reset per env)."""

    def __init__(self, n_envs: int, nov: _Novelty):
        self.nov = nov
        self.n = int(n_envs)
        self.count = np.zeros((self.n, int(np.prod(nov.shape))), np.int32)
        self.prev = np.full(self.n, -1, np.int64)

    def reset(self, idx) -> None:
        idx = np.asarray(idx, np.int64).reshape(-1)
        self.count[idx] = 0
        self.prev[idx] = -1

    def add(self, pos) -> None:
        c = self.nov.cells(pos)
        new = np.flatnonzero(c != self.prev)
        if len(new):
            self.count[new, c[new]] += 1
        self.prev[:] = c

    def u(self, env: int, node: int) -> float:
        return 1.0 / math.sqrt(1.0 + float(self.count[env, self.nov.node_cell[node]]))


def _euclid_table(jg: JumpGraph):
    fin = np.asarray(jg.g.finish_center, np.float64)[:2]
    return np.linalg.norm(jg.xy - fin[None, :], axis=1) / 1000.0


def _uval(kind: str, jg: JumpGraph, nov: Optional[_Novelty], epi: Optional[_Episodic] = None,
          env: int = 0):
    if kind == "euclid":
        d = _euclid_table(jg)
        return lambda v: -float(d[v])
    if kind == "novelty":
        return nov.u
    if kind == "episodic":
        return lambda v: epi.u(env, v)
    if kind == "euclid+episodic":
        d = _euclid_table(jg)
        return lambda v: -float(d[v]) + EPI_W * epi.u(env, v)
    raise ValueError(f"--jump-u {kind!r}: one of {', '.join(U_KINDS)}")


def _softmax(v, t):
    z = np.asarray(v, np.float64) / max(float(t), 1e-9)
    z = np.exp(z - z.max())
    return z / z.sum()


class JumpPlanner:
    """The fleet's jump planner behind the learned planner's interface (goalsys and the trainer
    drive both the same way: request, on_tick, plan, note_and_row, eval_hooks, set_tick_ms;
    update is a no-op - nothing here learns)."""

    jump = True
    eval_label = "jump planner greedy"
    cols = JUMP_COLS

    def __init__(self, graph, n_envs: int, *, depth: int = 3, u: str = "euclid",
                 temp: float = 0.05, jump_len: float = JUMP_LEN_U, start_pts=None,
                 tick_ms: float = 10.0, act_every: int = 1, corridor: float = 64.0,
                 seed: int = 0):
        self.graph = graph
        self.jg = JumpGraph(graph, jump_len)
        self._jgs = {id(graph): self.jg}
        self.n = int(n_envs)
        self.depth = max(1, int(depth))
        self.ukind = str(u)
        self.temp = float(temp)
        self.act_every = max(1, int(act_every))
        self.corridor = float(corridor)
        self.tick_ms = float(tick_ms)
        self.nov = _Novelty(graph)
        self.episodic = self.ukind in ("episodic", "euclid+episodic")
        self.epi = _Episodic(self.n, self.nov) if self.episodic else None
        self.uval = (None if self.episodic else _uval(self.ukind, self.jg, self.nov))
        self.memo = {}
        self._ticks = 0
        stub = SimpleNamespace(n_line=L_MAX, spacing=float(graph.spacing),
                               length=self.jg.jump_len)
        self.st = PlanState(self.n, None, stub, tick_ms, corridor, visits=False,
                            l_max=L_MAX, strict=True)
        self.rng = np.random.default_rng(int(seed))
        self.start_pts = (None if start_pts is None else
                          np.atleast_2d(np.asarray(start_pts, np.float64)))
        self.fresh = np.ones(self.n, bool)
        self.from_start = np.zeros(self.n, bool)
        self.updates = 0
        self.last_eval = None
        self.last_eval_cmpl = None
        self.cover = self.nov.count          # the trainer prints cover.sum() on a restore
        self._reset_window()

    # ------------------------------------------------------------ helpers
    def describe(self) -> str:
        return (f"JUMP planner (--goal-planner jump, no network): options = the distinct places "
                f"one jump ({self.jg.jump_len:.0f} u of walking) away on the walkable graph "
                f"({N_DIRS} compass probes, each the reachable cell furthest along its "
                f"direction, dropped under {self.jg.min_adv:.0f} u; ends within {MERGE_U:.0f} u "
                f"along the floor merge; a finish cell in reach is always an option); search "
                f"{self.depth} jump(s) deep, a node worth U = {self.ukind} (finish box "
                f"{FIN_BONUS:g}) - {LEN_COST:g} per 1,000 u walked ({FIN_LEN_COST:g} to the "
                f"finish), an option worth the best node of its subtree; plan = a draw from softmax(value / {self.temp:g}), the "
                f"eval takes the argmax; the line is the option's fastest walk; plans close on "
                f"arc >= {COMPLETE_FRAC:g} (strict judge, corridor {self.corridor:g} u), on "
                f"{BUDGET_MULT:g} x their walking time or at the episode's end")

    def set_tick_ms(self, tick_ms: float) -> None:
        self.tick_ms = float(tick_ms)
        self.st.set_tick_ms(tick_ms)

    def _jg_for(self, graph) -> JumpGraph:
        jg = self._jgs.get(id(graph))
        if jg is None:
            jg = JumpGraph(graph, self.jg.jump_len)
            self._jgs[id(graph)] = jg
        return jg

    def _reset_window(self) -> None:
        self.w = {"chosen": 0, "opts": 0, "vbest": 0.0, "closed": 0, "complete": 0,
                  "fin_seen": 0, "ep": 0, "fin": 0, "ep_start": 0, "fin_start": 0}

    def _budget(self, L: float) -> int:
        tps = 1000.0 / self.tick_ms
        return max(int(tps), int(math.ceil(BUDGET_MULT * L / BUDGET_SPEED_U * tps)))

    # --------------------------------------------------------- the fleet
    def request(self, idx, origins=None) -> None:
        idx = np.asarray(idx, np.int64).reshape(-1)
        if not len(idx):
            return
        self.st.active[idx] = False
        self.st.need[idx] = True
        self.fresh[idx] = True
        if self.epi is not None:
            self.epi.reset(idx)
        if origins is not None and self.start_pts is not None:
            o = np.atleast_2d(np.asarray(origins, np.float64))
            dd = np.linalg.norm(o[:, None, :] - self.start_pts[None, :, :], axis=2).min(axis=1)
            self.from_start[idx] = dd < 1.0

    def on_tick(self, pos, ended, finished, died, term_pos=None) -> None:
        pos = np.asarray(pos, np.float64)
        ended = np.asarray(ended, bool)
        finished = np.asarray(finished, bool)
        closed, comp = self.st.tick(pos, ended)
        self.nov.add(pos)
        if self.epi is not None:
            self.epi.add(pos)
        self._ticks += 1
        w = self.w
        if closed.any():
            ci = np.flatnonzero(closed)
            w["closed"] += len(ci)
            w["complete"] += int(comp[ci].sum())
        if ended.any():
            ei = np.flatnonzero(ended)
            w["ep"] += len(ei)
            w["fin"] += int(finished[ei].sum())
            fs = self.from_start[ei]
            w["ep_start"] += int(fs.sum())
            w["fin_start"] += int((finished[ei] & fs).sum())

    def _choose(self, p, greedy: bool, jg: Optional[JumpGraph] = None, memo=None,
                uval=None):
        jg = jg or self.jg
        root = int(jg.g.snap(np.asarray(p, np.float64)[None, :])[0])
        opts, vals = jg.decide(root, self.depth, uval or self.uval,
                               self.memo if memo is None else memo)
        if not opts:
            return None, None, 0, float("nan"), False
        if greedy:
            k = int(np.argmax(vals))
        else:
            k = int(self.rng.choice(len(opts), p=_softmax(vals, self.temp)))
        t, path, L, _f = opts[k]
        return (jg.line(p, path, jg.g), L, len(opts), float(vals.max()),
                bool(vals.max() >= FIN_BONUS * 0.5))

    def plan(self, pos, vel, yaw_deg):
        """Plans for every env waiting for one -> (idx, lines, fresh)."""
        idx = np.flatnonzero(self.st.need)
        if not len(idx):
            return idx, [], np.zeros(0, bool)
        if self.ukind == "novelty" and self._ticks >= NOV_REFRESH_TICKS:
            self.memo = {}                   # the counts moved: re-read them
            self._ticks = 0
        p = np.asarray(pos, np.float64)[idx]
        lines, budgets, keep = [], [], []
        w = self.w
        for j, i in enumerate(idx):
            if self.episodic:
                # this env's own episode memory: its own values, a fresh memo
                ln, L, nopt, vbest, fin_seen = self._choose(
                    p[j], greedy=False, memo={},
                    uval=_uval(self.ukind, self.jg, self.nov, self.epi, int(i)))
            else:
                ln, L, nopt, vbest, fin_seen = self._choose(p[j], greedy=False)
            if ln is None or len(ln) < 2:
                continue                     # nowhere to go: asked again next boundary
            keep.append(j)
            lines.append(ln)
            budgets.append(self._budget(L))
            w["chosen"] += 1
            w["opts"] += nopt
            w["vbest"] += vbest if vbest == vbest else 0.0
            w["fin_seen"] += int(fin_seen)
        if not keep:
            return np.zeros(0, np.int64), [], np.zeros(0, bool)
        idx = idx[np.asarray(keep, np.int64)]
        fresh = self.fresh[idx].copy()
        self.st.begin(idx, np.full(len(idx), -1, np.int64), p[np.asarray(keep)],
                      lines=lines, budgets=np.asarray(budgets, np.int64))
        self.fresh[idx] = False
        return idx, lines, fresh

    def update(self, force: bool = False):
        return None                          # nothing here learns

    def n_ready(self) -> int:
        return 0

    # ------------------------------------------------------------ logging
    def pop_window(self) -> dict:
        w = self.w
        rate = (lambda a, b: (a / b) if b else float("nan"))
        out = {"chosen": w["chosen"], "options": rate(w["opts"], w["chosen"]),
               "value_best": rate(w["vbest"], w["chosen"]), "closed": w["closed"],
               "complete": rate(w["complete"], w["closed"]),
               "fin_seen": rate(w["fin_seen"], w["chosen"]), "cover": self.nov.covered(),
               "ep": w["ep"], "finish": rate(w["fin"], w["ep"]),
               "ep_start": w["ep_start"], "finish_start": rate(w["fin_start"], w["ep_start"])}
        self._reset_window()
        return out

    def note_and_row(self):
        """-> (step-line text, progress.csv values in JUMP_COLS order)."""
        w = self.pop_window()
        ev = self.last_eval
        self.last_eval = None

        def f(v, nd):
            return round(float(v), nd) if v == v else ""
        row = [w["chosen"], f(w["options"], 3), f(w["value_best"], 4), w["closed"],
               f(w["complete"], 4), f(w["fin_seen"], 4), w["cover"], f(w["finish"], 4),
               f(w["finish_start"], 4), (f(ev[0] / ev[1], 4) if ev and ev[1] else "")]
        pc = (lambda v: f"{v:.1%}" if v == v else "-")
        txt = (f"  JUMP chosen {w['chosen']} (options {w['options']:.2f}, finish in sight "
               f"{pc(w['fin_seen'])}) closed {w['closed']} (cmpl {pc(w['complete'])}) cover "
               f"{w['cover']} cells"
               + (f" fin {pc(w['finish'])}/{w['ep']} from-start {pc(w['finish_start'])}/"
                  f"{w['ep_start']}" if w["ep"] else ""))
        return txt, row

    def state_dict_all(self) -> dict:
        return {"jump": {"depth": self.depth, "u": self.ukind, "temp": self.temp,
                         "jump_len": self.jg.jump_len},
                "counts": self.nov.count.copy()}

    def load_state_dict_all(self, sd: dict) -> None:
        c = (sd or {}).get("counts")
        if c is not None and tuple(np.shape(c)) == self.nov.shape:
            self.nov.count[...] = c

    # --------------------------------------------------------------- eval
    def eval_hooks(self, core, ev: dict, *, line=None, graph=None,
                   finish_radius: Optional[float] = None, **_kw):
        g = graph if graph is not None else self.graph
        same = g is self.graph
        return make_jump_hooks(self._jg_for(g), core, ev, depth=self.depth, u=self.ukind,
                               nov=(self.nov if same else None), line=line,
                               act_every=self.act_every, tick_ms=self.tick_ms,
                               corridor=self.corridor, finish_radius=finish_radius)


def make_jump_hooks(jg: JumpGraph, core, ev: dict, *, depth: int, u: str,
                    nov: Optional[_Novelty] = None, line=None, act_every: int = 1,
                    tick_ms: float = 10.0, corridor: float = 64.0,
                    finish_radius: Optional[float] = None):
    """(episode_meta, on_tick) for record_rollout on a core whose env 0 is recorded - the jump
    planner's eval, shared by the trainer and tools/record_ckpt.py: from wherever the core
    spawned env 0, the GREEDY choice (argmax), re-planned like training (completion, the strict
    judge, the budget, at the next decision boundary). Novelty reads ``nov`` (the training
    fleet's counts, or a checkpoint's) and adds this episode's own visits to a private copy."""
    graph = jg.g
    nov_e = _Novelty(graph)
    if nov is not None:
        nov_e.count[...] = nov.count
    epi_e = _Episodic(1, nov_e) if u in ("episodic", "euclid+episodic") else None
    uval = _uval(u, jg, nov_e, epi_e, 0)
    per_call = u in ("novelty", "episodic", "euclid+episodic")
    stub = SimpleNamespace(n_line=L_MAX, spacing=float(graph.spacing), length=jg.jump_len)
    st = PlanState(1, None, stub, tick_ms, corridor, visits=False, l_max=L_MAX, strict=True)
    finish = np.asarray(graph.finish_center, np.float64)
    K = max(1, int(act_every))
    ev.update({"n": 0, "succ": 0, "pending": False, "center": None, "ticks": [], "dists": [],
               "t0": 0, "box": True, "plans": 0, "closed": 0, "complete": 0, "wall": 0,
               "shapes": [], "opts": [], "plan_log": [], "tick": 0})
    zero = np.zeros(1, bool)
    tps = 1000.0 / float(tick_ms)

    def _choose():
        p = core.states_view["origin"][0:1].astype(np.float64)[0]
        root = int(graph.snap(p[None, :])[0])
        opts, vals = jg.decide(root, depth, uval, {} if per_call else _memo)
        if not opts:
            return None
        k = int(np.argmax(vals))
        t, path, L, _f = opts[k]
        ln = jg.line(p, path, graph)
        st.begin(np.zeros(1, np.int64), [-1], p[None, :], lines=[ln],
                 budgets=np.asarray([max(int(tps), int(math.ceil(
                     BUDGET_MULT * L / BUDGET_SPEED_U * tps)))], np.int64))
        if line is not None:
            line.set_lines(np.array([0]), [ln])
        ev["plans"] += 1
        ev["shapes"].append(int(t))
        ev["opts"].append(len(opts))
        ev["plan_log"].append({"ep": int(ev.get("ep_cur", 0)), "tick": int(ev["tick"]),
                               "shape": int(t), "anchor": [float(v) for v in p],
                               "line": [[float(v) for v in q] for q in np.asarray(ln)]})
        return ln

    _memo = {}

    def episode_meta(ep):
        ev["ep_cur"] = int(ev["n"])
        if epi_e is not None:
            epi_e.reset([0])
            epi_e.add(core.states_view["origin"][0:1].astype(np.float64))
        ln = _choose()
        ev["n"] += 1
        p = core.states_view["origin"][0:1].astype(np.float64)
        s = int(graph.snap(p)[0])
        dfin = float(graph.dist[graph.fin, s]) if graph.fin is not None else float("nan")
        ev["dists"].append(dfin)
        if ln is None:
            ln = np.vstack([p[0], p[0]]).astype(np.float32)
        thin = ln[:: max(1, len(ln) // 64)]
        rad = float(finish_radius) if finish_radius is not None else 192.0
        return {"goal": {"center": [float(v) for v in finish], "radius": rad},
                "line": [[float(v) for v in q] for q in thin],
                "plan": {"planner": "jump", "depth": int(depth), "u": str(u),
                         "graph_dist": (round(dfin, 1) if np.isfinite(dfin) else None)}}

    def on_tick(t, states, rewards, done, trunc):
        ev["tick"] = int(t) + 1
        if bool(done[0]) or bool(trunc[0]):
            won = bool(done[0]) and bool(np.asarray(core.goal_hits)[0])
            if won:
                ev["succ"] += 1
                ev["ticks"].append(t - ev["t0"])
            if st.active[0]:
                ev["closed"] += 1
            st.active[0] = False
            st.need[0] = False
            ev["t0"] = t + 1
            return
        p = core.states_view["origin"][0:1].astype(np.float64)
        nov_e.add(p)
        if epi_e is not None:
            epi_e.add(p)
        closed, comp = st.tick(p, zero)
        if closed[0]:
            ev["closed"] += 1
            ev["complete"] += int(comp[0])
        if st.need[0] and (t + 1) % K == 0:
            _choose()

    return episode_meta, on_tick
