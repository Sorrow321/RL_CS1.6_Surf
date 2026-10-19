#!/usr/bin/env python3
"""ramp_search.py - the ramp-level search: WHICH ramp next, found without any human sequence.

Nodes are EXACT engine states at the first contact with a target piece (a ramp or a floor of the
map's ramp vocabulary), each reached from the true spawn by a chain of witnessed edges. An edge is
one tools/ramp_mpc.py flight: MPC "touch piece B alive with the most energy" from the node's exact
state - seconds of CPU, no training. Every success replays exactly, so it is a witness that the
transition exists from that state.

  * candidates of a node: every other target piece, ordered by the closest approach of the node's
    ballistic arc (--arc s) to the piece's standing contact origins (the trainer's ramp-window
    prior);
  * --alloc round (the first version): every pair once in arc order, then again (--pair-tries
    rounds); which node next: the node whose piece has been reached least, then the one expanded
    least, then the more energetic. Measured on unitfarmer2 (ledger 2026-10-17 13:21): 74% of the
    flights and 82% of the compute went to flights that stalled (no 50 u of progress toward B
    for 3 s), and the pit's exit pair got one flight;
  * --alloc pw (progressive widening, Couetoux et al. 2011): a node with N flights has opened at
    most 2 + sqrt(N) pieces; the rest of its flights retry its open pairs, fewest tries first and
    then the one whose flights came closest to B (progress = 1 - closest / starting distance, 1
    for a touch). A pair whose flight never closed 50 u on B is closed, and so is one whose touch
    landed in a cell a more energetic node already holds (a re-flight would only re-land there).
    The candidate order puts first the pieces that a flight from the SAME piece has touched
    anywhere in the tree; everything else keeps its arc rank. Which node next: the largest
    1 / sqrt(touches into its cell) / sqrt(1 + its flights) x (0.5 + 0.5 x its energy rank among
    the open nodes). The first two factors are Go-Explore's count weights; the cell is the
    dominance key below (piece x heading), so a node that enters a piece a new way is not
    charged for the other ways in. The last factor is there because energy height is what a
    surf state can still reach, and a rank has no map scale. Ties go to the more energetic node.
    A node whose first --dead-after flights all die within 0.25 s is dropped;
  * --alloc bayes: the same node choice; within the node, the pair with the highest touch
    estimate (m x prior + touches + part touches) / (m + flights), where the prior is the touch
    rate of first flights at the pair's arc-rank bucket learned over the whole tree, a failed
    flight counts progress ** --near-pow as a part touch, a pair that has touched is weighted
    --refine-w, and a pair below --p-min or at --max-tries flights gets none. It replaces the
    widening schedule, the 50 u rule and --pair-tries: run 3 closed the pit exit's pair after
    three near misses on a 15%-per-flight state, run 4 closed petrus's first ramp after one
    flight on a 33%-per-flight state (ledger 2026-10-17 18:47);
  * dominance at CONTACT: a new node on a piece where another node moving the same way (one of
    --headings horizontal sectors x rising / falling) already holds at least as much energy is
    not kept - both have paid their landing, so the comparison is fair (go_cem's fall-vs-surf
    optimism cannot occur); with --dom-radius only a node within that many u dominates;
  * the goal field is used for REPORTING only; --ruler names a sequence (e.g. a human record's)
    whose targets are counted as a measuring stick, never used by the search.

    python tools/ramp_search.py --map maps_pool/surf_unitfarmer2.bsp \\
        --vocab runs/research/ramps_mesh_surf_unitfarmer2.npz --minutes 120 \\
        --out runs/research/rampmpc/search_uf2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE                  # noqa: E402
from ramp_mpc import Engine, Target, fly_edge, replay, energy, G   # noqa: E402


def heading_key(v, headings):
    """horizontal heading sector x rising / falling; below 100 u/s horizontal (where the core's
    view base switches from the velocity heading to the yaw) the heading is noise: one extra
    sector for every slow state"""
    v = np.asarray(v, np.float64)
    if np.hypot(v[0], v[1]) < 100.0:
        return headings * 2 + int(v[2] > 0)
    ang = np.arctan2(v[1], v[0])
    sec = int(np.floor((ang + np.pi) / (2 * np.pi) * headings)) % headings
    return sec * 2 + int(v[2] > 0)


_W = {}                                           # a worker process's engine and targets


def _worker_init(args):
    a = argparse.Namespace(**args)
    _W["eng"], _W["a"], _W["tg"] = Engine(a), a, {}


def _worker_fly(sid, start, seed, iters):
    eng, a = _W["eng"], _W["a"]
    if sid not in _W["tg"]:
        _W["tg"][sid] = Target(eng.voc, sid)
    return fly_edge(eng, _W["tg"][sid], start, np.random.default_rng(seed),
                    argparse.Namespace(**{**vars(a), "iters": iters}))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--tick-ms", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--headings", type=int, default=8)
    ap.add_argument("--alloc", default="round", choices=("round", "pw", "bayes"),
                    help="which (node, piece) pair gets the next flight (see above)")
    ap.add_argument("--prior-m", type=float, default=2.0,
                    help="--alloc bayes: pseudo-flights the arc-rank prior is worth in a pair's "
                         "touch estimate")
    ap.add_argument("--near-pow", type=float, default=8.0,
                    help="--alloc bayes: a failed flight counts progress ** this as a part touch "
                         "(0.95 -> 0.66, 0.8 -> 0.17, 0.5 -> 0.004)")
    ap.add_argument("--p-min", type=float, default=0.02,
                    help="--alloc bayes: a pair whose touch estimate is below this gets no flight")
    ap.add_argument("--max-tries", type=int, default=12,
                    help="--alloc bayes: flights per (node, piece) pair at most")
    ap.add_argument("--refine-w", type=float, default=0.5,
                    help="--alloc bayes: weight of a re-flight of a pair that has touched (it "
                         "can only buy energy or a new heading)")
    ap.add_argument("--pair-tries", type=int, default=2,
                    help="flights per (node, piece) pair at most (the solver is stochastic)")
    ap.add_argument("--dead-after", type=int, default=3,
                    help="--alloc pw: drop a node whose first N flights all die within 0.25 s")
    ap.add_argument("--arc", type=float, default=3.0, help="s of ballistic arc for the ordering")
    ap.add_argument("--dom-eps", type=float, default=10.0, help="u of energy a new node must add")
    ap.add_argument("--dom-radius", type=float, default=0.0,
                    help="u: a node dominates only new nodes this close (0 = anywhere on the piece)")
    ap.add_argument("--ruler", default=None, help="a sequence counted for REPORTING only")
    # the edge solver (tools/ramp_mpc.py's flags and defaults)
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--elite", type=float, default=0.05)
    ap.add_argument("--plan", type=float, default=1.2)
    ap.add_argument("--passive", type=float, default=2.0)
    ap.add_argument("--viable", type=float, default=1.0)
    ap.add_argument("--touch-score", default="touch", choices=("touch", "post"))
    ap.add_argument("--ground-homing", type=int, default=0, choices=(0, 1))
    ap.add_argument("--score", default="lex", choices=("lex", "soft"))
    ap.add_argument("--touch-bonus", type=float, default=300.0)
    ap.add_argument("--dist-w", type=float, default=0.2)
    ap.add_argument("--exec", type=int, default=8)
    ap.add_argument("--knot", type=int, default=4)
    ap.add_argument("--max-offset", type=float, default=8.0)
    ap.add_argument("--off-units", default="deg", choices=("deg", "strafe"))
    ap.add_argument("--max-offset-u", type=float, default=4.0)
    ap.add_argument("--fwd-p0", type=float, default=0.3)
    ap.add_argument("--jump-p0", type=float, default=0.1)
    ap.add_argument("--energy-w", type=float, default=1.0)
    ap.add_argument("--passive-death", type=float, default=1000.0)
    ap.add_argument("--dist-every", type=int, default=4)
    ap.add_argument("--edge-timeout", type=float, default=8.0)
    ap.add_argument("--stall-abort", type=float, default=3.0,
                    help="give a flight up when its distance to B has not improved for this many s")
    ap.add_argument("--hold-p", type=float, default=0.8)
    ap.add_argument("--off-rho", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refine-iters", type=int, default=0,
                    help="CEM iterations for a re-flight of a pair that has already touched B (0: "
                         "--iters): the first flight asks whether B is reachable, a re-flight "
                         "only buys energy, so it gets the bigger search")
    ap.add_argument("--save-every", type=float, default=20.0,
                    help="minutes between tree saves (<out>.json, <out>_nodes.npz) while it runs; "
                         "touch <out>.stop to end a run early with a full save")
    ap.add_argument("--resume", default=None,
                    help="continue an earlier run's tree: its --out (reads <out>.json and "
                         "<out>_nodes.npz). Every node is re-flown from its parent's exact state and "
                         "must land on its saved state; the earlier flights seed the pair statistics")
    ap.add_argument("--workers", type=int, default=0,
                    help="flights at once, each in its own process, all feeding one tree (0: one "
                         "flight at a time in this process). The order results arrive in then "
                         "varies run to run; every route still replays exactly. On a 32-thread "
                         "box 4 workers at OMP_NUM_THREADS=8 fly 3.4x the flights per hour")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Path(f"{a.out}.stop").unlink(missing_ok=True)   # a stop file from an earlier run
    eng = Engine(a)
    voc = eng.voc
    print(voc.describe(), flush=True)
    # one representative target surface per piece (the largest face): the edge's B
    pieces = {}
    for s in voc.targets:
        q = int(voc.piece[s])
        if q not in pieces or voc.area[s] > voc.area[pieces[q]]:
            pieces[q] = int(s)
    tg = {q: Target(voc, s) for q, s in pieces.items()}
    print(f"{len(pieces)} target pieces", flush=True)
    ruler = [int(s) for s in a.ruler.split(",")] if a.ruler else []
    ruler_pieces = [int(voc.piece[s]) for s in ruler]
    gf = None
    try:
        from surfgym.goalfield import build_goal_field
        gf = build_goal_field(eng.core, eng.zones["end"], cell=32)       # REPORTING only
    except Exception as ex:                                              # noqa: BLE001
        print(f"(no goal field for reporting: {ex})", flush=True)

    def d_of(p):
        if gf is None:
            return float("nan")
        return float(gf.sample(np.asarray(p, np.float64)[None])[0])

    spawn = eng.spawn.copy()
    d0 = d_of(spawn["origin"])
    sp_sid = int(voc.contact_of(np.asarray(spawn["origin"], np.float64)[None],
                                np.zeros(1, np.int64))[0])
    sp_piece = int(voc.piece[sp_sid]) if sp_sid >= 0 else -1
    nodes = [dict(state=spawn, piece=sp_piece, E=float(energy(spawn["origin"], spawn["velocity"])),
                  t=0.0, parent=-1, edge=None, depth=0, tried={}, picks=0, route=[], stats={},
                  early=0, pending=set(), cell=("root",))]
    best_at = {}                                  # (piece, heading key) -> node ids
    reached = {sp_piece: 1}                       # piece -> nodes created on it (the root: 1)
    seen = {("root",): 1}                         # cell -> touches into it, kept or not
    pair_best = {}                                # (piece, piece) -> best progress, any node
    rank_stats = {}                               # arc-rank bucket -> [first flights, touches]
    edges = []                                    # every attempt
    if a.resume:
        rj = json.loads(Path(f"{a.resume}.json").read_text())
        rz = np.load(f"{a.resume}_nodes.npz")
        for k, jn in enumerate(rj["nodes"][1:], start=1):
            par = nodes[jn["parent"]]
            lo, hi = int(rz["edge_lo"][k]), int(rz["edge_hi"][k])
            side, fwd, off, jump = (rz[f][lo:hi] for f in ("side", "fwd", "off", "jump"))
            states = replay(eng, par["state"], side, fwd, off, jump)
            st = rz["state"][k].copy()
            err = float(np.abs(np.asarray(states["origin"][-1], np.float64)
                               - np.asarray(st["origin"], np.float64)).max())
            if err > 1e-3:
                raise SystemExit(f"--resume: node {k} re-flies {err:.3f} u off its saved state")
            q = int(jn["piece"])
            key = (q, heading_key(st["velocity"], a.headings))
            nodes.append(dict(state=st, piece=q, E=float(jn["E"]), t=float(jn["t"]),
                              parent=int(jn["parent"]),
                              edge=dict(side=side, fwd=fwd, off=off, jump=jump, states=states),
                              depth=int(jn["depth"]), tried={}, picks=0, route=par["route"] + [q],
                              stats={}, early=0, pending=set(), cell=key,
                              dominated=bool(jn["dominated"]), dead=bool(jn.get("dead"))))
            best_at.setdefault(key, []).append(k)
            reached[q] = reached.get(q, 0) + 1
            seen[key] = seen.get(key, 0) + 1
        for e in rj["attempts"]:                  # the pair statistics, as integrate() keeps them
            n, q = nodes[e["src"]], int(e["piece"])
            n["picks"] += 1
            n["tried"][q] = n["tried"].get(q, 0) + 1
            ps = n["stats"].setdefault(q, dict(tries=0, best=0.0, closed=False, touches=0,
                                               credit=0.0))
            ps["tries"] += 1
            prog = float(e.get("progress", 1.0 if e["result"] in ("TOUCH", "FINISH") else 0.0))
            ps["best"] = max(ps["best"], prog)
            hit = e["result"] in ("TOUCH", "FINISH")
            ps["touches"] += hit
            if not hit:
                ps["credit"] += prog ** a.near_pow
            if (a.alloc != "bayes" and not hit and "d_min" in e
                    and e["d_start"] - e["d_min"] < 50.0):
                ps["closed"] = True
            if e.get("dominated"):
                ps["closed"] = True
            pair_best[(n["piece"], q)] = max(pair_best.get((n["piece"], q), 0.0), prog)
            if e["result"] == "DIED" and e["game_s"] < 0.25:
                n["early"] += 1
        edges.extend(rj["attempts"])
        print(f"resumed {len(nodes) - 1} nodes and {len(rj['attempts'])} flights from {a.resume}; "
              f"every node re-flies onto its saved state", flush=True)
    t_start = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    finish = None
    print(f"root: spawn z {spawn['origin'][2]:.0f} on piece {sp_piece}; field d0 {d0:.0f}",
          flush=True)

    def arc_order(n):
        if "cands" not in n:                      # the ordering, computed once per node
            st = n["state"]
            p = np.asarray(st["origin"], np.float64)
            v = np.asarray(st["velocity"], np.float64)
            ts = np.arange(0.0, a.arc, 0.05)
            path = (p[None] + v[None] * ts[:, None]
                    + 0.5 * np.array([0.0, 0.0, -G])[None] * ts[:, None] ** 2)
            out = []
            for q, t in tg.items():
                if q != n["piece"]:
                    d, _ = t.tree.query(path, k=1)
                    out.append((float(d.min()), q))
            n["cands"] = [q for _d, q in sorted(out)]
        return n["cands"]

    # --alloc bayes: one touch estimate per (node, piece) pair, from the pair's own flights and a
    # prior for its arc rank that the tree learns as it goes (first flights per rank bucket); no
    # map scale in it. Measured on unitfarmer2 (runs 2-3, ledger 2026-10-17 18:47): first flights
    # at arc rank 0 touched 31/61, ranks 8+ 0/722; on petrus ranks 8-15 still touched 41/69 - so
    # the prior is learned per run, not fixed. A failed flight that came close counts as part of a
    # touch (progress ** --near-pow): retries that had come within 10% touched 1/5, those that
    # had not 23/1,692; the uf2 pit exit's state touches S19 on 3/20 flights (run 3) to 13/20
    # (run 2), so three flights are not enough to give it up.
    RANK_EDGES = (0, 1, 2, 3, 4, 8, 16, 32)

    BUCKET = [max(j for j, lo in enumerate(RANK_EDGES) if r >= lo) for r in range(4096)]

    def bucket_of(r):
        return BUCKET[min(r, len(BUCKET) - 1)]

    def rank_prior(r):
        b = bucket_of(r)
        f, t = rank_stats.get(b, (0, 0))
        p0 = 1.0 / (2.0 + RANK_EDGES[b])          # before any flight: 1/2, 1/3, ... 1/34
        return (t + 2.0 * p0) / (f + 2.0)

    def p_hat(n, q, r):
        s = n["stats"].get(q)
        pi = rank_prior(r)
        if s is None:
            return pi
        if s["closed"] or s["tries"] >= a.max_tries:
            return 0.0
        p = (a.prior_m * pi + s["touches"] + s["credit"]) / (a.prior_m + s["tries"])
        return p * (a.refine_w if s["touches"] else 1.0)

    def note_first_flight(n, q, hit):
        b = bucket_of(arc_order(n).index(q))
        f = rank_stats.setdefault(b, [0, 0])
        f[0] += 1
        f[1] += int(hit)

    if a.resume:                                  # the rank prior, from the resumed flights
        firsts = set()
        for e in edges:
            if (e["src"], e["piece"]) not in firsts:
                firsts.add((e["src"], e["piece"]))
                note_first_flight(nodes[e["src"]], int(e["piece"]), e["result"] in ("TOUCH", "FINISH"))

    def candidates(n):
        if a.alloc == "round":                    # every piece once in arc order, then again
            for r in range(a.pair_tries):
                left = [q for q in arc_order(n)
                        if n["tried"].get(q, 0) == r and q not in n["pending"]]
                if left:
                    return left
            return []
        if n.get("dead"):
            return []
        if a.alloc == "bayes":                    # highest touch estimate first
            out = [(-p, r, q) for r, q in enumerate(arc_order(n)) if q not in n["pending"]
                   for p in (p_hat(n, q, r),) if p >= a.p_min]
            return [q for _p, _r, q in sorted(out)]
        st = n["stats"]
        old = [q for q, s in st.items() if not s["closed"] and q not in n["pending"]
               and n["tried"][q] < a.pair_tries]
        old.sort(key=lambda q: (st[q]["tries"], -st[q]["best"]))
        rank = {q: r for r, q in enumerate(arc_order(n))}
        # a piece that a flight from the same piece has TOUCHED anywhere in the tree moves up;
        # everything else keeps its arc rank. A failure from another state never moves a piece
        # down (one bad entry on S18 buried S18 -> S19 for every S18 state: run 2a), and a near
        # miss elsewhere does not move it up (the S19 exit node of run 2 spent 32 flights on
        # near-missed floors before trying S20)
        new = sorted((q for q in rank if q not in st and q not in n["pending"]),
                     key=lambda q: (-(pair_best.get((n["piece"], q), 0.0) >= 1.0), rank[q]))
        opened = len(set(st) | n["pending"])
        if new and (opened < 2 + int(np.sqrt(n["picks"])) or not old):
            return new + old                      # widen: open the next piece
        return old + new

    def node_key(j, e_rank):
        nn = nodes[j]
        if a.alloc == "round":                    # min: rarest piece, least expanded, energy
            return (reached.get(nn["piece"], 0), nn["picks"], -nn["E"], -nn["depth"])
        w = 1.0 / np.sqrt(seen.get(nn["cell"], 1)) / np.sqrt(1.0 + nn["picks"])
        return (-w * (0.5 + 0.5 * e_rank[j]), -nn["E"], -nn["depth"])

    def progress_of(res, q, start):
        """1 - closest approach to B / starting distance (1 for a touch), and the two distances"""
        d_s = float(tg[q].tree.query(np.asarray(start["origin"], np.float64)[None], k=1)[0][0])
        if res["result"] in ("TOUCH", "FINISH"):
            return 1.0, d_s, 0.0
        o = np.asarray(res["states"]["origin"], np.float64)
        if not len(o):
            return 0.0, d_s, d_s
        d_m = min(d_s, float(tg[q].tree.query(o[::2], k=1)[0].min()))
        return max(0.0, 1.0 - d_m / max(d_s, 1.0)), d_s, d_m

    def ruler_progress(route):
        """the longest prefix of the ruler matched IN ORDER by a route's pieces (reporting only)"""
        k = 0
        for q in route:
            if k < len(ruler_pieces) and q == ruler_pieces[k]:
                k += 1
        return k

    def route_of(i):
        chain = []
        while i > 0:
            chain.append(i)
            i = nodes[i]["parent"]
        return chain[::-1]

    def choose():
        """the next (node, piece) pair, counted as assigned at once, or None"""
        open_ids = [i for i, n in enumerate(nodes)
                    if not n.get("dominated") and len(candidates(n)) > 0]
        if not open_ids:
            return None
        order = sorted(open_ids, key=lambda j: nodes[j]["E"])
        e_rank = {j: r / max(1, len(order) - 1) for r, j in enumerate(order)}
        i = min(open_ids, key=lambda j: node_key(j, e_rank))
        n = nodes[i]
        q = candidates(n)[0]
        n["picks"] += 1
        n["tried"][q] = n["tried"].get(q, 0) + 1
        n["pending"].add(q)
        return i, q

    def integrate(i, q, res, iters):
        nonlocal finish
        n = nodes[i]
        n["pending"].discard(q)
        prog, d_s, d_m = progress_of(res, q, n["state"])
        hit = res["result"] in ("TOUCH", "FINISH")
        if q not in n["stats"]:
            note_first_flight(n, q, hit)
        s = n["stats"].setdefault(q, dict(tries=0, best=0.0, closed=False, touches=0, credit=0.0))
        s["tries"] += 1
        s["best"] = max(s["best"], prog)
        s["touches"] += hit
        if not hit:
            s["credit"] += prog ** a.near_pow
        if a.alloc != "bayes" and not hit and d_s - d_m < 50.0:
            s["closed"] = True                     # never closed 50 u on B: not from here
        pair_best[(n["piece"], q)] = max(pair_best.get((n["piece"], q), 0.0), prog)
        if res["result"] == "DIED" and res["game_s"] < 0.25:
            n["early"] += 1
            if (n["early"] >= a.dead_after
                    and n["early"] == sum(x["tries"] for x in n["stats"].values())):
                n["dead"] = True                   # every flight dies at once: a doomed state
        rec = dict(src=i, piece=q, sid=pieces[q], result=res["result"], game_s=res["game_s"],
                   E=res["touch_E"], compute_s=res["compute_s"], progress=prog, d_start=d_s,
                   d_min=d_m, iters=iters)
        edges.append(rec)
        tag = ""
        if res["result"] in ("TOUCH", "FINISH"):
            st = res["state"]
            E = float(energy(st["origin"], st["velocity"]))
            key = (q, heading_key(st["velocity"], a.headings))
            seen[key] = seen.get(key, 0) + 1
            pos = np.asarray(st["origin"], np.float64)
            near = [j for j in best_at.get(key, []) if not nodes[j].get("dominated")
                    and (a.dom_radius <= 0 or np.linalg.norm(
                        np.asarray(nodes[j]["state"]["origin"], np.float64) - pos) < a.dom_radius)]
            dom = [j for j in near if E <= nodes[j]["E"] + a.dom_eps]
            if res["result"] == "FINISH" or not dom:
                for j in near:
                    nodes[j]["dominated"] = True
                route = n["route"] + [q]
                nodes.append(dict(state=st, piece=q, E=E, t=n["t"] + res["game_s"], parent=i,
                                  edge=res, depth=n["depth"] + 1, tried={}, picks=0, route=route,
                                  stats={}, early=0, pending=set(), cell=key))
                best_at.setdefault(key, []).append(len(nodes) - 1)
                reached[q] = reached.get(q, 0) + 1
                tag = (f"NEW node {len(nodes) - 1}: " + " ".join(f"S{pieces[x]}" for x in route)
                       + f" (ruler {ruler_progress(route)}/{len(ruler)})")
                if res["result"] == "FINISH":
                    finish = len(nodes) - 1
            else:
                tag = f"dominated by node {dom[0]} (E {nodes[dom[0]]['E']:.0f})"
                rec["dominated"] = True
                if a.alloc in ("pw", "bayes"):
                    s["closed"] = True             # this pair only re-lands where a better
                                                   # state already is: no more flights
        elif n.get("dead"):
            tag = "node dropped: its flights all die at once"
        el = time.perf_counter() - t_start
        e_txt = (f"E {res['touch_E']:5.0f}" if res["touch_E"] is not None
                 else f"p {prog:4.2f}")
        print(f"[{el / 60:6.1f} min] node {i:3d} (S{pieces.get(n['piece'], -1) if n['piece'] >= 0 else -1}, "
              f"d{n['depth']}) -> S{pieces[q]:<3d} {res['result']:<17s} {res['game_s']:5.2f} s "
              f"{e_txt} {res['compute_s']:5.1f}s  {tag}", flush=True)
        if res["result"] == "FINISH" and finish == len(nodes) - 1:
            print(f"FINISH reached by node {finish} at game time {nodes[finish]['t']:.2f} s",
                  flush=True)
        if len(edges) % 20 == 0:
            live = [k for k, nn in enumerate(nodes) if not nn.get("dominated")]
            dd = [d_of(nodes[k]["state"]["origin"]) for k in live]
            best_k = live[int(np.nanargmin(dd))] if dd and not np.all(np.isnan(dd)) else 0
            rp = max(ruler_progress(nn["route"]) for nn in nodes)
            cs = sum(e["compute_s"] for e in edges)
            cst = sum(e["compute_s"] for e in edges if e["result"] == "stalled")
            print(f"  -- {len(edges)} attempts, {len(nodes)} nodes, {len(reached)} pieces reached; "
                  f"best field progress {(1 - min(dd) / d0) * 100 if dd else 0:.1f}% (node {best_k}, "
                  f"depth {nodes[best_k]['depth']}); ruler (reporting only): {rp}/{len(ruler)} "
                  f"in order; stalled flights {cst / max(cs, 1e-9) * 100:.0f}% of compute",
                  flush=True)
            if a.alloc == "bayes":
                print("     rank prior: " + " ".join(
                    f"r{RANK_EDGES[b]}+ {rank_prior(RANK_EDGES[b]):.2f} ({rank_stats[b][1]}/"
                    f"{rank_stats[b][0]})" for b in sorted(rank_stats)), flush=True)

    def save_tree():
        """every node's exact state and the controls of the edge into it (any route replays from
        the spawn by concatenating the edges along its parents) + the json; -> the best node"""
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        live = [k for k, nn in enumerate(nodes) if not nn.get("dominated")]
        dd = np.array([d_of(nodes[k]["state"]["origin"]) for k in live])
        best_k = finish if finish is not None else (live[int(np.nanargmin(dd))] if len(live) else 0)
        node_state = np.zeros(len(nodes), STATE_DTYPE)
        lo, hi, ctl = np.zeros(len(nodes), np.int64), np.zeros(len(nodes), np.int64), []
        for k, nn in enumerate(nodes):
            node_state[k] = nn["state"]
            lo[k] = hi[k - 1] if k else 0
            hi[k] = lo[k] + (len(nn["edge"]["side"]) if nn["edge"] is not None else 0)
            if nn["edge"] is not None:
                ctl.append(nn["edge"])
        cat = (lambda f, dt: np.concatenate([e[f] for e in ctl]).astype(dt) if ctl
               else np.zeros(0, dt))
        np.savez_compressed(f"{out}_nodes.npz", state=node_state,
                            parent=np.asarray([nn["parent"] for nn in nodes], np.int64),
                            edge_lo=lo, edge_hi=hi, side=cat("side", np.int8),
                            fwd=cat("fwd", np.int8), off=cat("off", np.float32),
                            jump=cat("jump", np.int8), spawn=np.asarray([spawn], STATE_DTYPE),
                            tick_ms=a.tick_ms)
        summary = dict(args=vars(a), n_nodes=len(nodes), attempts=edges,
                       nodes=[dict(id=k, piece=nn["piece"], sid=pieces.get(nn["piece"], -1),
                                   E=nn["E"], t=nn["t"], parent=nn["parent"], depth=nn["depth"],
                                   dominated=bool(nn.get("dominated")), dead=bool(nn.get("dead")),
                                   picks=nn["picks"], d=d_of(nn["state"]["origin"]),
                                   pos=[float(x) for x in nn["state"]["origin"]],
                                   vel=[float(x) for x in nn["state"]["velocity"]],
                                   route=[pieces[x] for x in nn["route"]])
                              for k, nn in enumerate(nodes)],
                       finish=finish, best=best_k)
        Path(f"{out}.json").write_text(json.dumps(summary, indent=1, default=float))
        return best_k

    last_save = [time.perf_counter()]

    def maybe_save():
        if a.save_every > 0 and time.perf_counter() - last_save[0] > a.save_every * 60:
            save_tree()
            last_save[0] = time.perf_counter()

    def iters_for(i, q):
        s = nodes[i]["stats"].get(q)
        return (a.refine_iters if a.refine_iters > 0 and s is not None and s["best"] >= 1.0
                else a.iters)

    def time_left():
        if Path(f"{a.out}.stop").exists():       # a graceful stop: finish the flights, save
            return False
        return time.perf_counter() - t_start < a.minutes * 60 and finish is None

    if a.workers <= 0:                            # one flight at a time, in this process
        while time_left():
            c = choose()
            if c is None:
                print("every node is fully expanded", flush=True)
                break
            i, q = c
            it = iters_for(i, q)
            integrate(i, q, fly_edge(eng, tg[q], nodes[i]["state"],
                                     np.random.default_rng(int(rng.integers(1 << 31))),
                                     argparse.Namespace(**{**vars(a), "iters": it})), it)
            maybe_save()
    else:                                         # --workers flights at once, one tree
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        inflight = {}
        with ProcessPoolExecutor(a.workers, initializer=_worker_init,
                                 initargs=(vars(a),)) as pool:
            while True:
                while time_left() and len(inflight) < a.workers:
                    c = choose()
                    if c is None:
                        break
                    i, q = c
                    it = iters_for(i, q)
                    inflight[pool.submit(_worker_fly, pieces[q], nodes[i]["state"],
                                         int(rng.integers(1 << 31)), it)] = (i, q, it)
                if not inflight:
                    if time_left():
                        print("every node is fully expanded", flush=True)
                    break
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for f in done:
                    i, q, it = inflight.pop(f)
                    integrate(i, q, f.result(), it)
                maybe_save()

    # save: every node's route as controls from the spawn (exactly replayable)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_k = save_tree()
    rk = (max(range(len(nodes)), key=lambda k: (ruler_progress(nodes[k]["route"]), -nodes[k]["t"]))
          if ruler else 0)
    for name, k in (("best", best_k), ("ruler-deepest (reporting only)", rk)):
        chain = route_of(k)
        if not chain or (name != "best" and k == best_k):
            continue
        side = np.concatenate([nodes[c]["edge"]["side"] for c in chain])
        fwd = np.concatenate([nodes[c]["edge"]["fwd"] for c in chain])
        off = np.concatenate([nodes[c]["edge"]["off"] for c in chain])
        jump = np.concatenate([nodes[c]["edge"]["jump"] for c in chain])
        states = np.concatenate([nodes[c]["edge"]["states"] for c in chain])
        rep = replay(eng, spawn, side, fwd, off, jump)
        err = float(np.abs(np.asarray(rep["origin"], np.float64)
                           - np.asarray(states["origin"], np.float64)).max())
        seq = [pieces[nodes[c]["piece"]] for c in chain]
        print(f"{name} route: node {k}, {len(chain)} edges, game {nodes[k]['t']:.2f} s: "
              + " -> ".join(f"S{s}" for s in seq) + f"; replay error {err:.2f} u", flush=True)
        if name == "best":
            np.savez_compressed(f"{out}.npz", side=side, fwd=fwd, off=off, jump=jump,
                                states=states, spawn=np.asarray([spawn], STATE_DTYPE),
                                sequence=np.asarray(seq), tick_ms=a.tick_ms,
                                sequence_source="search")


if __name__ == "__main__":
    main()
