#!/usr/bin/env python3
"""sid_arch_regions.py - ANALYSIS ONLY: what a --respawn-rare-arch checkpoint's archive holds around
sidistic's CP2 slot exit (measurement boxes only, never training).

    python3 sid_arch_regions.py <ckpt.pt>     (run on the box from /root/RL_Surf)

Per region: archived cells (256 u), their summed entry count N, the median N, and the deepest
field progress of the archived states there.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "python")
from surfgym.goalfield import GoalField  # noqa: E402

z = np.load("maps/surf_src_sidistic.goal_32.npz", allow_pickle=False)
GF = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
               float(z["cell"]), float(z["reach_max"]))
D0 = 97768.0
REGIONS = {
    "slot (x 2600-3700, z 11380-11560)": (2600, 3700, -1200, 1200, 11380, 11560),
    "shaft upper (x 2210-2520, z 11150-11560)": (2210, 2520, -1200, 1200, 11150, 11560),
    "shaft lower (x 2210-2520, z 10700-11150)": (2210, 2520, -1200, 1200, 10700, 11150),
    "centre gap area (x 2400-2700, |y|<300, z 10750-11100)": (2400, 2700, -300, 300, 10750, 11100),
    "cavity (x 2580-2990, z 10650-11260)": (2580, 2990, -1200, 1200, 10650, 11260),
    "below shaft (x 1700-2990, z 9500-10650)": (1700, 2990, -1200, 1200, 9500, 10650),
    "room (x 1200-1950, z 10480-11180)": (1200, 1950, -1200, 1200, 10480, 11180),
    "under room / to CP2 (x 700-1700, z 9500-10450)": (700, 1700, -1200, 1200, 9500, 10450),
}


def main():
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    rr = next(iter(ck["respawn_rare"].values()))
    counts = np.asarray(rr["counts"])
    dims = np.asarray(rr["dims"])
    cell = float(rr["cell"])
    st = rr.get("arch_store")
    if st is None:
        print("no archive in this checkpoint")
        return
    ring = int(rr["ring"])
    fill = np.asarray(rr["arch_fill"])
    bcell = np.asarray(rr["arch_cell"])
    org = np.asarray(st["origin"], np.float64).reshape(len(bcell), ring, 3)
    print(f"step {ck.get('global_step')}: archive {len(bcell):,} cells, "
          f"{int((counts > 0).sum()):,} cells counted, {int(counts.sum()):,} entries")
    first = org[:, 0, :]                         # each block's first slot is always filled
    d = GF.sample(first).astype(np.float64)
    prog = 100.0 * (D0 - d) / D0
    n = counts[bcell]
    for name, (x0, x1, y0, y1, z0, z1) in REGIONS.items():
        m = ((first[:, 0] >= x0) & (first[:, 0] < x1) & (first[:, 1] >= y0) & (first[:, 1] < y1)
             & (first[:, 2] >= z0) & (first[:, 2] < z1))
        if m.any():
            print(f"  {name:55s} cells {int(m.sum()):4d}  N sum {int(n[m].sum()):9,d}  "
                  f"N med {int(np.median(n[m])):6,d}  best prog {prog[m].max():6.2f}%")
        else:
            print(f"  {name:55s} cells    0")
    top = np.argsort(-prog)[:8]
    for i in top:
        print(f"  deepest: prog {prog[i]:6.2f}%  pos {np.round(first[i]).astype(int).tolist()}  "
              f"N {int(n[i]):,}  fill {int(fill[i])}")


if __name__ == "__main__":
    main()
