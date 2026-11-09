# analysis only (unr1b1): hands-off ducked flights from synthetic states over / in the shelf's slit
# (x -500..-650, |y| <= 80, z 10,580-10,700) - which velocities pass the ledge? MEASUREMENT only.
import sys, itertools
import numpy as np
sys.path.insert(0, "/home/steam/ar_uni/wt/b1/python")
from surfgym.core import SurfCore, default_config
MAP = "/home/steam/ar_uni/wt/b1/maps/surf_src_sidistic.bsp"
XS = [-500.0, -550.0, -600.0, -650.0]; YS = [-80.0, -40.0, 0.0, 40.0, 80.0]; ZS = [10580.0, 10620.0, 10660.0, 10700.0]
VS = [400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0]; VYS = [-300.0, -150.0, 0.0, 150.0, 300.0]; VZS = [-600.0, -400.0, -200.0, 0.0, 200.0]
combos = list(itertools.product(XS, YS, ZS, VS, VYS, VZS)); n = len(combos)
cfg = default_config(num_envs=n, spawn_mode=2, max_episode_ticks=6000, water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
cfg.phys.sv_maxvelocity = 4000.0
core = SurfCore(MAP, cfg); core.set_teleport_fail(True); core.reset(0)
base = core.get_states().copy()
free0 = np.array([not core.trace((c[0], c[1], c[2]), (c[0], c[1], c[2]), 1).startsolid for c in combos])
for i, (x, y, z, v, vy, vz) in enumerate(combos):
    s = base[i:i+1].copy()
    s["origin"] = (x, y, z); s["velocity"] = (-v, vy, vz); s["basevelocity"] = (0, 0, 0); s["yaw"] = 180.0
    s["pitch"] = 0.0; s["onground"] = -1; s["ducked"] = 1; s["induck"] = 0; s["duck_time"] = 0.0
    s["oldbuttons"] = 4; s["tick"] = 0; s["stuck_ticks"] = 0
    core.set_state(i, s)
acts = np.zeros((n, 6), np.int32); acts[:, 0] = 7; acts[:, 1] = 3; acts[:, 2] = 1; acts[:, 3] = 1; acts[:, 5] = 1
alive = np.ones(n, bool); ok = np.zeros(n, bool)
for t in range(200):
    _, _, done, trunc, _ = core.step(acts)
    st = core.get_states()
    alive &= ~(done | trunc).astype(bool)
    ok |= alive & (st["origin"][:, 0] < -975.0) & (st["origin"][:, 2] > 10000.0)
C = np.array(combos)
print(f"{n} flights, {int(free0.sum())} start in free space; passed {int(ok.sum())}")
for z in ZS:
    print(f"  start z {z:.0f}: " + "  ".join(f"V{v:.0f}:{int(ok[(C[:,2]==z)&(C[:,3]==v)&free0].sum())}/{int(((C[:,2]==z)&(C[:,3]==v)&free0).sum())}" for v in VS))
for x in XS:
    print(f"  start x {x:.0f}: passed {int(ok[(C[:,0]==x)&free0].sum())} of {int(((C[:,0]==x)&free0).sum())}")
for y in YS:
    print(f"  start y {y:.0f}: passed {int(ok[(C[:,1]==y)&free0].sum())}")
for vz in VZS:
    print(f"  vz {vz:.0f}: passed {int(ok[(C[:,5]==vz)&free0].sum())}")
p = np.flatnonzero(ok)
if len(p):
    print("min V passing:", C[p, 3].min(), "; examples:")
    for i in p[np.argsort(C[p, 3])][:8]:
        print("   ", C[i])
