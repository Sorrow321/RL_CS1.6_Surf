#!/usr/bin/env python3
"""pb_frontier_step.py - one step of the frontier loop (pbr2b4, round 2), run ON THE BOX:

  1. record the checkpoint's greedy line from the true start (record_ckpt --dump-states);
  2. roots = the best episode's last --window-s seconds before its best-progress tick;
  3. --ge N reward-free exact-chain Go-Explore searches rooted there (explore_phase1
     --exact-chains, seeds differ), --minutes each, side by side;
  4. the deepest archive cell on the grader field over all searches -> tools/ge_replay.py;
  5. spine = the greedy episode's own states up to the chain's root + the replayed chain.

Generic: no map coordinate, box or demo; every state is the policy's own or its own search's.

  python3 autoresearch/analysis/pb_frontier_step.py --ckpt runs/X/ckpt_latest.pt \\
      --out runs/pbr2b4_geK [--minutes 12 --ge 2 --window-s 4]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "autoresearch" / "analysis"))

import pb_line_tools as T                          # noqa: E402
from surfgym.core import STATE_DTYPE              # noqa: E402

MAP = "maps/surf_princessburglar_b26.bsp"
FIELD = "maps/surf_princessburglar_b26.goalc_32.npz"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--ge", type=int, default=2)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--ep-ticks", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--tol", type=float, default=32.0,
                    help="chain end = the shortest chain among cells within this many u of the "
                         "deepest (half a 64 u cell)")
    a = ap.parse_args(argv)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    f = T.field()
    t0 = time.time()
    dump = out / "greedy.npz"
    r = sh([sys.executable, "tools/record_ckpt.py", a.ckpt, "--episodes", str(a.episodes),
            "--ep-ticks", str(a.ep_ticks), "--out", str(out / "greedy.jsonl"),
            "--dump-states", str(dump)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("record_ckpt failed")
    eps = T.eps_of(dump)
    best = [(float(np.nanmax(T.prog(f, e))), k) for k, e in enumerate(eps)]
    for p, k in best:
        print(f"greedy ep {k}: best {p:.3f}%  ({len(eps[k])} ticks)")
    pbest, kbest = max(best)
    e = eps[kbest]
    pe = T.prog(f, e)
    ib = int(np.nanargmax(pe))
    lo = max(0, ib - int(round(a.window_s * 100)))
    roots_idx = np.arange(lo, ib + 1, 2)
    roots = np.zeros(len(roots_idx), STATE_DTYPE)
    for n in STATE_DTYPE.names:
        roots[n] = e[roots_idx][n]
    roots["tick"] = 0
    roots["stuck_ticks"] = 0
    np.save(out / "roots.npy", roots)
    print(f"roots: ep {kbest} ticks {lo}..{ib} stride 2 = {len(roots)} (best {pbest:.3f}%)", flush=True)
    procs = []
    for g in range(a.ge):
        d = out / f"ge{g}"
        log = open(out / f"ge{g}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, "-u", "tools/explore_phase1.py", "--map", MAP, "--roots-spine",
             str(out / "roots.npy"), "--exact-chains", "--cell", "64", "--cell-energy", "32",
             "--envs", "4096", "--act-every", "4", "--decisions", "100", "--maxvel", "3500",
             "--max-minutes", str(a.minutes), "--goals", "1", "--threads", str(a.threads),
             "--seed", str(a.seed + g), "--out", str(d)],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    cand, pool = [], []
    for g in range(a.ge):
        z = np.load(out / f"ge{g}" / "archive.npz", allow_pickle=False)
        st, dep = z["state"], z["depth"]
        d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
        d[d >= f.reach_max - 0.5 * f.cell] = np.inf
        i = int(np.argmin(d))
        cand.append((float(d[i]), g, i, len(st)))
        pool += [(float(d[j]), int(dep[j]), g, int(j)) for j in np.flatnonzero(d <= d[i] + a.tol)]
        print(f"ge{g}: {len(st):,} cells, deepest {100 * (57149 - d[i]) / 57149:.3f}% (cell {i}, "
              f"depth {int(dep[i])} ticks)")
    # the chain end: among the cells within --tol u of the deepest (any search), the SHORTEST
    # chain - fewer states for the curriculum to learn for (almost) the same depth
    dlo = min(c[0] for c in pool)
    pool = [c for c in pool if c[0] <= dlo + a.tol]
    dmin, dep_best, g, i = min(pool, key=lambda c: (c[1], c[0]))
    print(f"chain end: ge{g} cell {i}, {100 * (57149 - dmin) / 57149:.3f}%, depth {dep_best} ticks "
          f"(shortest of {len(pool)} cells within {a.tol:g} u of the deepest)")
    chain = out / "chain.npy"
    r = sh([sys.executable, "tools/ge_replay.py", "--map", MAP, "--archive",
            str(out / f"ge{g}" / "archive.npz"), "--field", FIELD, "--cell", str(i),
            "--out", str(chain)], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-1000:])
    line = np.load(chain, allow_pickle=False)
    o0 = line[0]["origin"].astype(np.float64)
    dist = np.linalg.norm(e["origin"].astype(np.float64) - o0, axis=1)
    j = int(np.argmin(dist))
    if dist[j] > 0.01:
        raise SystemExit(f"the chain root is not a state of greedy ep {kbest} ({dist[j]:.3f} u)")
    pre = np.zeros(j, STATE_DTYPE)
    for n in STATE_DTYPE.names:
        pre[n] = e[:j][n]
    spine = np.concatenate([pre, line.astype(STATE_DTYPE)])
    spine["tick"] = 0
    spine["stuck_ticks"] = 0
    np.save(out / "spine.npy", spine)
    ps = T.prog(f, spine)
    info = {"ckpt": a.ckpt, "greedy_best_pct": pbest, "greedy_ep": kbest, "root_tick": j,
            "chain_len": int(len(line)), "spine_len": int(len(spine)),
            "spine_end_pct": float(ps[-1]), "spine_best_pct": float(np.nanmax(ps)),
            "ge": [{"g": c[1], "cells": c[3], "deepest_pct": 100 * (57149 - c[0]) / 57149}
                   for c in cand], "minutes": (time.time() - t0) / 60}
    (out / "spine.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
