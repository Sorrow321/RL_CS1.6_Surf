# analysis only (unr1b1): where does the uniform explorer's archive (x4L, one 3 h archive from
# v2's own greedy line, 4 s roots) reach near the post-CP2 slot? The slot (map geometry, read off
# the goal field - MEASUREMENT only): x -912..-944 is free only for z 10,340-10,532 (y about
# -736..672). For each x band toward the slot: cells, the highest z, and the cells at slot height.
import sys
import numpy as np

p = sys.argv[1]
a = np.load(p, allow_pickle=False)
st = a["state"]
dist = a["dist"]
depth = a["depth"]
o = st["origin"].astype(np.float64)
v = st["velocity"].astype(np.float64)
x, y, z = o[:, 0], o[:, 1], o[:, 2]
n = len(st)
print(f"{p}: {n:,} cells; best dist {dist.min():,.0f} at origin {o[np.argmin(dist)].round(0)} "
      f"vel {v[np.argmin(dist)].round(0)} depth {depth[np.argmin(dist)]}")
print(f"cells x range {x.min():,.0f}..{x.max():,.0f}  z range {z.min():,.0f}..{z.max():,.0f}")
print("x band        cells   max z   z>=10340   y of max-z cell   |v| at max z   vz at max z")
for x0 in range(-500, -1100, -32):
    m = (x <= x0) & (x > x0 - 32)
    if not m.any():
        print(f"{x0:6d}..{x0-32:6d}  {0:7d}")
        continue
    i = np.flatnonzero(m)[np.argmax(z[m])]
    hi = int((m & (z >= 10340)).sum())
    print(f"{x0:6d}..{x0-32:6d}  {int(m.sum()):7d}  {z[i]:7.0f}  {hi:9d}   {y[i]:8.0f}        "
          f"{np.linalg.norm(v[i]):7.0f}       {v[i, 2]:7.0f}")
# the highest cells by how far -x they got: the envelope the explorer can fly toward the slot
m = x <= -700
print(f"\ncells at x <= -700: {int(m.sum()):,}; of them z >= 10,340: {int((m & (z >= 10340)).sum()):,}")
for zc in (10600, 10532, 10340, 10200, 10000):
    mm = z >= zc
    if mm.any():
        j = np.flatnonzero(mm)[np.argmin(x[mm])]
        print(f"  most -x cell with z >= {zc}: x {x[j]:,.0f} y {y[j]:,.0f} z {z[j]:,.0f}  "
              f"v ({v[j,0]:.0f}, {v[j,1]:.0f}, {v[j,2]:.0f})  dist {dist[j]:,.0f}")
