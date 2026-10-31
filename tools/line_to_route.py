#!/usr/bin/env python3
"""line_to_route.py LINE.npy OUT.npz [SPACING] (zlr1b2, 2026-10-31) - A per-tick STATE_DTYPE line -> an arc route .npz (route = the polyline resampled at a uniform
arc spacing, spacing) for train_fast --race-arc. Generic: geometry only."""
import sys
import numpy as np
line = np.load(sys.argv[1])
sp = float(sys.argv[3]) if len(sys.argv) > 3 else 128.0
p = line["origin"].astype(np.float64)
seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
keep = np.concatenate([[True], seg > 1e-6])
p = p[keep]
s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
n = int(np.floor(s[-1] / sp)) + 1
t = np.arange(n) * sp
route = np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)
np.savez(sys.argv[2], route=route.astype(np.float32), spacing=np.float32(sp))
print(f"{len(line)} rows, arc {s[-1]:,.0f} u -> {n} vertices @ {sp:g} u -> {sys.argv[2]}")
