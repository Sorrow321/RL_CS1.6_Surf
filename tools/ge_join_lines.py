#!/usr/bin/env python3
"""ge_join_lines.py BASE.npy EXT.npy OUT.npy (zlr1b2, 2026-10-31) - Join a GE line rooted on another line: base[:k] + ext, k = the base row ext starts on."""
import sys
import numpy as np
base = np.load(sys.argv[1])
ext = np.load(sys.argv[2])
o, v = ext["origin"][0].astype(np.float64), ext["velocity"][0].astype(np.float64)
do = np.linalg.norm(base["origin"].astype(np.float64) - o, axis=1)
dv = np.linalg.norm(base["velocity"].astype(np.float64) - v, axis=1)
k = int(np.argmin(do + dv))
print(f"ext starts on base row {k}: |dpos| {do[k]:.4f} |dvel| {dv[k]:.4f}")
assert do[k] < 0.5 and dv[k] < 0.5, "the extension does not start on the base line"
full = np.concatenate([base[:k], ext])
full["tick"] = 0
full["stuck_ticks"] = 0
np.save(sys.argv[3], full)
print(f"{len(base[:k])} + {len(ext)} = {len(full)} rows -> {sys.argv[3]}")
