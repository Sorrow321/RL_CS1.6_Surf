"""The state at first arrival at progress levels, per episode or spine (pbr7b4, 2026-10-27; ANALYSIS ONLY): position,
|vxy|, heading (deg, -90 = south), vz and energy height h = z + |v|^2/1600. Where two lines part at a gate (pbr7b4: the
beam turns ~18 deg into the 74.4% climb ramp and converts its speed into vz; the energetic greedy turns ~13 deg and
keeps it horizontal).

    python autoresearch/analysis/pb_approach.py L1,L2,... traj.jsonl|spine.npy[:lo:hi] ...
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402

F = score.field()
levels = [float(x) for x in sys.argv[1].split(",")]


def report(tag, p, v):
    d = F.sample(p.astype(np.float64)).astype(np.float64)
    prog = np.maximum.accumulate(np.where(d < F.reach_max - 16, 100 * (57149 - d) / 57149, -1))
    hh = p[:, 2] + (v * v).sum(1) / 1600
    out = [f"{tag} max {prog.max():.2f}%"]
    for L in levels:
        i = np.flatnonzero(prog >= L)
        if not len(i):
            out.append(f"{L}: -")
            continue
        i = i[0]
        hd = np.degrees(np.arctan2(v[i, 1], v[i, 0]))
        out.append(f"{L}: ({p[i,0]:.0f},{p[i,1]:.0f},{p[i,2]:.0f}) |vxy| {np.hypot(v[i,0], v[i,1]):.0f} "
                   f"hdg {hd:.0f} vz {v[i,2]:+.0f} h {hh[i]:.0f}")
    print(" | ".join(out))


for arg in sys.argv[2:]:
    if arg.split(":")[0].endswith(".npy"):
        parts = arg.split(":")
        S = np.load(parts[0])
        lo = int(parts[1]) if len(parts) > 1 else 0
        hi = int(parts[2]) if len(parts) > 2 else len(S)
        report(Path(parts[0]).stem + f"[{lo}:{hi}]", S["origin"][lo:hi], S["velocity"][lo:hi])
        continue
    for h, r, tr in score.episodes(arg):
        report(f"{Path(arg).parent.name} ep{h.get('episode', '?')}", r[:, 1:4], r[:, 4:7])
