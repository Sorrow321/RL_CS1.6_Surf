#!/usr/bin/env python3
"""corridor_bench.py - ANALYSIS ONLY: what does a checkpoint do when it starts in the
corridor under sidistic's kill plate *45, moving east (its own frontier states, moved
there the way --respawn-novel moves them)? Records N stochastic + M greedy episodes from
/root/corridor_east.npy and summarises each: end, seconds, field progress of the best
point, how far east it got, whether it reached the ramp area (x > 5,300, z < 11,600), and
whether it came back west past x 3,000 above the pit floor (z > 11,332) afterwards.

    python3 corridor_bench.py <ckpt> <tag> [n_stoch] [n_greedy]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/RL_Surf")
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, "/root/RL_Surf")
import sid_cp2_nudge_sweep as ns  # noqa: E402  (episodes(), GF); copy both to /root/RL_Surf on the box


def summarise(path):
    rows = []
    for hdr, a, tr in ns.episodes(path):
        p = a[:, 1:4]
        d = ns.GF.sample(p).astype(np.float64)
        ok = d < ns.GF.reach_max - 0.5 * ns.GF.cell
        dmin = float(d[ok].min()) if ok.any() else float(d[0])
        ramp = np.flatnonzero((p[:, 0] > 5300) & (p[:, 2] < 11600))
        back = False
        xret = None
        if len(ramp):
            after = p[ramp[0]:]
            back = bool(np.any((after[:, 0] < 3000) & (after[:, 2] > 11332)))
            xret = int(after[:, 0].min())        # how far west it came back after the ramp
        rows.append(dict(end=tr.get("end"), secs=round(len(a) / 100.0, 2),
                         prog=round(100 * (97768.0 - dmin) / 97768.0, 2),
                         xmax=int(p[:, 0].max()), zmin=int(p[:, 2].min()),
                         ramp=bool(len(ramp)), back=back, xret=xret,
                         last=[int(x) for x in p[-1]]))
    return rows


def main():
    ck, tag = sys.argv[1], sys.argv[2]
    ns_, ng = int(sys.argv[3]) if len(sys.argv) > 3 else 16, int(sys.argv[4]) if len(sys.argv) > 4 else 4
    states = sys.argv[5] if len(sys.argv) > 5 else "/root/corridor_east.npy"
    out = Path("/root/bench_" + tag)
    out.mkdir(exist_ok=True)
    allrows = []
    for mode, n in (("stoch", ns_), ("greedy", ng)):
        if n <= 0:
            continue
        jl = out / f"{mode}.jsonl"
        cmd = [sys.executable, str(ROOT / "tools/record_ckpt.py"), ck,
               "--map", str(ROOT / "maps/surf_src_sidistic.bsp"), "--episodes", str(n),
               "--seed", "0", "--out", str(jl), "--spawn-states", states,
               "--ep-ticks", "1500"]
        if mode == "stoch":
            cmd.append("--stochastic")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode != 0:
            print(mode, "FAILED", r.stdout[-800:], r.stderr[-800:])
            continue
        rows = summarise(jl)
        for x in rows:
            x["mode"] = mode
            print(json.dumps(x))
        allrows += rows
    n = len(allrows)
    if n:
        xr = [r["xret"] for r in allrows if r["xret"] is not None]
        print(f"SUMMARY {tag}: n={n} ramp={sum(r['ramp'] for r in allrows)} "
              f"min_xret={min(xr) if xr else None} "
              f"back={sum(r['back'] for r in allrows)} "
              f"median_secs={np.median([r['secs'] for r in allrows]):.2f} "
              f"max_prog={max(r['prog'] for r in allrows):.2f}")


if __name__ == "__main__":
    main()
