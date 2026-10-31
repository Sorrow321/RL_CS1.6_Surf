# MEASUREMENT ONLY (a bench, never a training file): the high-energy pit rows of a spawn file.
import sys
import numpy as np
s = np.load(sys.argv[1], allow_pickle=False)
o = s["origin"].astype(np.float64); v = s["velocity"].astype(np.float64)
E = o[:, 2] + (v ** 2).sum(1) / 1600.0
k = (o[:, 2] < 100) & (E > 1000)
np.save(sys.argv[2], s[k])
print(f"{k.sum()} of {len(s)} rows -> {sys.argv[2]}")
