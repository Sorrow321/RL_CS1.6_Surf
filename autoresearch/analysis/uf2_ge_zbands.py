#!/usr/bin/env python3
"""uf2_ge_zbands.py - a GE archive's best energy height per height band (z), with where the best
state is and how it moves: does the search carry energy DOWN into a pit at all. Reporting only."""
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
st = z["state"]
org = st["origin"].astype(np.float64)
vel = st["velocity"].astype(np.float64)
E = org[:, 2] + (vel ** 2).sum(1) / 1600.0
depth = z["depth"]
for lo in range(-1000, 1300, 200):
    m = (org[:, 2] >= lo) & (org[:, 2] < lo + 200)
    if not m.any():
        continue
    i = np.flatnonzero(m)[np.argmax(E[m])]
    print(f"z [{lo:5d},{lo+200:5d}) n {int(m.sum()):7,d} E max {E[i]:6.0f} p99 {np.percentile(E[m], 99):6.0f} "
          f"at ({org[i,0]:6.0f},{org[i,1]:6.0f},{org[i,2]:5.0f}) v ({vel[i,0]:6.0f},{vel[i,1]:6.0f},{vel[i,2]:6.0f}) t {depth[i]/100:5.2f}s")
