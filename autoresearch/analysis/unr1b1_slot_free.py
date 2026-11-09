# analysis only (unr1b1): free intervals of a DUCKED player's origin z (zero-length duck-hull
# traces every 4 u from z 9,600 to 11,300) along x at a few y - MEASUREMENT only.
import sys
import numpy as np
sys.path.insert(0, "/home/steam/ar_uni/wt/b1/python")
from surfgym.core import SurfCore, default_config
cfg = default_config(num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0)
core = SurfCore("/home/steam/ar_uni/wt/b1/maps/surf_src_sidistic.bsp", cfg)
ys = [float(v) for v in sys.argv[1].split(",")]
x0, x1, dx = (int(v) for v in sys.argv[2].split(","))
zs = np.arange(9600.0, 11300.0, 4.0)
for y in ys:
    print(f"-- y {y:.0f}: duck-hull origin free z intervals")
    for x in range(x0, x1, dx):
        free = [not core.trace((x, y, z), (x, y, z), 1).startsolid for z in zs]
        iv = []; s = None
        for z, f in zip(zs, free):
            if f and s is None: s = z
            if not f and s is not None: iv.append((s, z - 4)); s = None
        if s is not None: iv.append((s, zs[-1]))
        print(f"  x {x:6d}: " + "  ".join(f"[{a:.0f},{b:.0f}]" for a, b in iv))
