"""Free standing-hull origin-z intervals over the chamber: for each (x, y), the z ranges where the
standing hull is not in solid. Kill trigger [451..762]x[-3393..-2605]x[3207..3360]: a standing hull
touches it when origin z < 3396 (duck: < 3378) and x in 435..778, y in -3409..-2589."""
import sys
import numpy as np
sys.path.insert(0, 'python')
from surfgym.core import SurfCore, default_config
cfg = default_config(num_envs=1, spawn_mode=2, max_episode_ticks=6000, water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
core = SurfCore('maps/surf_princessburglar_b26.bsp', cfg, dll_path='build/libsurfcore.so')
hull = int(sys.argv[1]) if len(sys.argv) > 1 else 0
zs = np.arange(3180, 3800, 4.0)
def free_intervals(x, y):
    fr = [not core.trace((x, y, z), (x, y, z), hull).startsolid for z in zs]
    out, cur = [], None
    for z, f in zip(zs, fr):
        if f and cur is None: cur = [z, z]
        elif f: cur[1] = z
        elif cur is not None: out.append(tuple(cur)); cur = None
    if cur is not None: out.append(tuple(cur))
    return out
for y in list(range(-2600, -2860, -40)) + list(range(-2900, -3140, -60)) + list(range(-3180, -3420, -40)):
    print(f"y {y}:")
    for x in (300, 400, 440, 500, 600, 700, 760, 800, 850, 900, 950, 1000, 1050):
        iv = free_intervals(x, y)
        print(f"   x {x:5d}: " + " ".join(f"[{a:.0f}-{b:.0f}]" for a, b in iv))
