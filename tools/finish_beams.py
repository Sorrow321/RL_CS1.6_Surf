#!/usr/bin/env python3
"""finish_beams.py - policy-guided finish searches from the policy's OWN frontier states.

A frontier pool (tools/frontier_states.py: per greedy episode, the states from shortly before
its last contact-record improvement to shortly before its death, concatenated) is split into its
episodes (a jump of more than --jump u between consecutive states starts a new one). From the
first state of each episode (and, with --offsets, from states S seconds later in it) this runs
tools/beam_tas.py with the checkpoint's own stochastic policy as the proposal, ranked by the goal
field (--score d: lineages nearer the finish survive each resample), until a lineage enters the
goal or --max-ticks runs out. A crossing is replayed bit-exactly by beam_tas itself and its line
(the per-tick states up to the pre-finish state) is saved as <out>/<tag>/beam_frontier.npz by
--save-frontier; tools/finish_pool.py turns those lines into spawn states.

Generic: nothing here reads the map beyond the goal field every search already ranks by; the
roots are the policy's own states, the proposals its own action distribution. Reporting and
search only - nothing here trains.

    python tools/finish_beams.py CKPT POOL.npy --out runs/fb/a [--envs 2048] [--max-ticks 7000]
        [--offsets 0,10] [--stop-after 2] -- [extra beam_tas flags]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def episode_starts(pool: np.ndarray, jump: float) -> list[int]:
    """Indices where a new episode starts: 0 and every state more than ``jump`` u from the one
    before it (a frontier pool concatenates episodes; within one, consecutive ticks move at most
    |v| x tick, ~25 u at 3,500 u/s and 7 ms)."""
    o = np.asarray(pool["origin"], np.float64)
    if len(o) == 0:
        return []
    dp = np.linalg.norm(np.diff(o, axis=0), axis=1)
    return [0] + [int(i) + 1 for i in np.nonzero(dp > jump)[0]]


def plan_roots(pool: np.ndarray, jump: float, offsets_s: list[float], tick_s: float) -> list[int]:
    """Root indices: each episode's first state, then the state ``s`` seconds later for every
    offset (skipped when the episode is shorter), in episode order then offset order."""
    starts = episode_starts(pool, jump)
    ends = starts[1:] + [len(pool)]
    roots = []
    for off in offsets_s:
        k = int(round(off / tick_s))
        for a, b in zip(starts, ends):
            if a + k < b:
                roots.append(a + k)
    return roots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("pool", help="a 1-D STATE_DTYPE .npy of the policy's own states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--max-ticks", type=int, default=7000)
    ap.add_argument("--resample-every", type=int, default=25)
    ap.add_argument("--score", default="d")
    ap.add_argument("--jump", type=float, default=200.0)
    ap.add_argument("--offsets", default="0", help="seconds into each episode, comma-separated")
    ap.add_argument("--tick-ms", type=float, default=7.0,
                    help="the pool's tick, to turn --offsets into states")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop after this many crossing searches (0 = run every root)")
    ap.add_argument("--max-roots", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    # everything after a bare "--" goes to beam_tas.py unchanged
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    a = ap.parse_args(argv)
    pool = np.load(a.pool, allow_pickle=False)
    offs = [float(x) for x in str(a.offsets).split(",") if x.strip()]
    roots = plan_roots(pool, a.jump, offs, a.tick_ms / 1000.0)
    if a.max_roots > 0:
        roots = roots[:a.max_roots]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"finish_beams: {len(roots)} roots from {a.pool} ({len(pool):,} states, "
          f"{len(episode_starts(pool, a.jump))} episodes): {roots}", flush=True)
    results, n_cross = [], 0
    for r in roots:
        tag = f"r{r:06d}"
        od = out / tag
        od.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "tools" / "beam_tas.py"), a.ckpt,
               "--start-state", f"{a.pool}:{r}", "--skip-gate", "--envs", str(a.envs),
               "--max-ticks", str(a.max_ticks), "--resample-every", str(a.resample_every),
               "--score", a.score, "--save-frontier", "--out-dir", str(od), *extra]
        o = pool["origin"][r]
        v = pool["velocity"][r]
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: root ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f}) "
              f"|v| {float(np.linalg.norm(v)):.0f}: {' '.join(cmd[2:])}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        with open(od / "log.txt", "w", encoding="utf-8") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        summ = {}
        if (od / "summary.json").exists():
            try:
                summ = json.loads((od / "summary.json").read_text(encoding="utf-8"))
            except ValueError:
                summ = {}
        crossed = bool(summ.get("crossed"))
        n_cross += int(crossed)
        tail = (od / "log.txt").read_text(encoding="utf-8", errors="replace").strip().splitlines()
        res = {"root": r, "rc": rc, "crossed": crossed, "best_s": summ.get("best_s"),
               "wall_s": round(time.time() - t0, 1), "last": tail[-1] if tail else ""}
        results.append(res)
        print(f"   -> rc {rc} crossed {crossed} best_s {summ.get('best_s')} "
              f"wall {res['wall_s']}s | {res['last'][:160]}", flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        if a.stop_after > 0 and n_cross >= a.stop_after:
            print(f"finish_beams: {n_cross} crossing search(es): stopping", flush=True)
            break
    print(f"finish_beams: done, {n_cross} of {len(results)} searches crossed", flush=True)


if __name__ == "__main__":
    main()
