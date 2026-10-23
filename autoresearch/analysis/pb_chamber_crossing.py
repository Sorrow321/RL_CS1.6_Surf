"""Ballistic crossing of the north pocket's kill floor through its east window (x 790, y -2660,
z 3480-3540), heading west at v with vertical vz, no keys. 'Y' = alive at x < 420."""
import sys
import numpy as np
sys.path.insert(0, 'python')
from surfgym.core import SurfCore, default_config
zs = np.arange(3480, 3541, 10.0); vs = np.arange(300, 1301, 100.0); vzs = (0.0, 150.0, 300.0)
ys = (-2645.0, -2660.0, -2675.0)
combos = [(y, z, v, vz) for y in ys for z in zs for v in vs for vz in vzs]
n = len(combos)
cfg = default_config(num_envs=n, spawn_mode=2, max_episode_ticks=6000, water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
core = SurfCore('maps/surf_princessburglar_b26.bsp', cfg, dll_path='build/libsurfcore.so')
core.set_teleport_fail(True)
core.reset(0)
tmpl = np.load('autoresearch/analysis/pb_goexplore_chain73.npy')[-1].copy()  # any STATE_DTYPE row
solid0 = np.zeros(n, bool)
for i, (y, z, v, vz) in enumerate(combos):
    s = tmpl.copy()
    s['origin'] = (790.0, y, z); s['velocity'] = (-v, 0.0, vz); s['basevelocity'] = 0
    s['yaw'] = 180.0; s['pitch'] = 0.0; s['onground'] = -1; s['ducked'] = 0; s['induck'] = 0
    s['duck_time'] = 0.0; s['stuck_ticks'] = 0; s['tick'] = 0; s['oldbuttons'] = 0
    core.set_state(i, s)
    solid0[i] = core.trace((790.0, y, z), (790.0, y, z), 0).startsolid
act = np.zeros((n, 6), np.int32); act[:, 0] = 7; act[:, 1] = 3; act[:, 2] = 1; act[:, 3] = 1
alive = np.ones(n, bool); ok = np.zeros(n, bool); xmin = np.full(n, 1e9)
for t in range(250):
    _, _, done, trunc, _ = core.step(act)
    alive &= ~((done | trunc) != 0)
    o = core.states_view['origin']
    xmin = np.where(alive, np.minimum(xmin, o[:, 0]), xmin)
    ok |= alive & (o[:, 0] < 420)
for y in ys:
    for vz in vzs:
        print(f"\ny {y:.0f}, vz {vz:+.0f}: rows z0, cols westward v (u/s); Y = alive past x 420; S = start in solid; else min x alive")
        print("  z0  " + "".join(f"{int(v):>6}" for v in vs))
        for z in zs:
            row = []
            for v in vs:
                i = combos.index((y, z, v, vz))
                row.append("     S" if solid0[i] else ("     Y" if ok[i] else ("  dead" if xmin[i] > 1e8 else f"{xmin[i]:6.0f}")))
            print(f"{int(z):5d} " + "".join(row))
