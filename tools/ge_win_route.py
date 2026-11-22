#!/usr/bin/env python3
"""ge_win_route.py - the full route of a ge_iter run whose last stage hit the finish.

ge_iter.py stops at the stage whose explorer hit the goal (win_1.*: the archive chain to the
burst's origin cell + the burst itself). This replays that chain exactly from the stage's archive
(tools/ge_replay.py --cell <the chain's last cell>), finds its root on the line the stage was
rooted on (an exact copy, so the match is 0 u), and writes the route: the line up to the root,
the replayed chain, then the burst's recorded states. It prints the route's length in ticks and
seconds at the explorer's tick, the root match, the largest one-tick move (a stitch that did not
match would show as a jump) and where it ends.

    python tools/ge_win_route.py --run runs/nm_night/geit3 --map maps/M.bsp \\
        --field maps/M.goalc_32.npz [--tick-ms 8]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the ge_iter --out directory")
    ap.add_argument("--map", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--tick-ms", type=float, default=8.0, help="the explorer's tick")
    a = ap.parse_args(argv)
    run = Path(a.run)
    it = json.loads((run / "ge_iter.json").read_text())
    st = [s for s in it["stages"] if s.get("goal")]
    if not st:
        raise SystemExit(f"{run}: no stage hit the goal")
    k = int(st[-1]["stage"])
    sd = run / f"st{k:02d}"
    chain = json.loads((sd / "win_1.chain.json").read_text())
    cell = int(chain["chain"][-1]["idx"])
    out = sd / "winchain.npy"
    if not out.exists():
        subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", a.map,
                        "--archive", str(sd / "archive.npz"), "--field", a.field,
                        "--out", str(out), "--cell", str(cell)], check=True)
    # the line the winning stage was rooted on: the last line saved before it
    prev = sorted(p for p in run.glob("line_*.npy") if p.stem != "line_best"
                  and int(p.stem.split("_")[1]) < k)
    if prev:
        base = np.load(prev[-1], allow_pickle=False)
    else:
        raise SystemExit(f"{run}: no line before stage {k}")
    ch = np.load(out, allow_pickle=False)
    po = np.asarray(base["origin"], np.float64)
    dist = np.linalg.norm(po - np.asarray(ch["origin"][0], np.float64), axis=1)
    j = int(np.argmin(dist))
    rows = np.array([json.loads(ln)[1:4] for ln in open(sd / "win_1.traj.jsonl")
                     if ln.startswith("[")], np.float64)
    states = np.concatenate([base[:j + 1], ch[1:]])
    full = np.concatenate([np.asarray(states["origin"], np.float64), rows[1:]])
    np.save(run / "route_states.npy", states)
    np.save(run / "route_origin.npy", full)
    steps = np.linalg.norm(np.diff(full, axis=0), axis=1)
    n = len(full) - 1
    rec = {"run": str(run), "stage": k, "cell": cell, "root_tick": j, "root_match_u": float(dist[j]),
           "chain_states": int(len(ch)), "burst_rows": int(len(rows)), "ticks": int(n),
           "seconds": round(n * a.tick_ms / 1000.0, 3), "max_step_u": round(float(steps.max()), 1),
           "end": np.round(full[-1], 1).tolist()}
    (run / "route.json").write_text(json.dumps(rec, indent=1))
    print("GE_WIN_ROUTE " + json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
