"""ANALYSIS ONLY: surf-aligned states in the region under sidistic's CP2 slot (the helix), built from
geometry alone (SDF + hull trace + goal field), no demo. Each state sits 30-90 u off a solid
surface with a velocity of 1,500-2,200 u/s tangent to it (a random horizontal direction projected
onto the surface plane) - what a generic 'surf-aligned novel row' would spawn."""
import sys
import numpy as np
sys.path.insert(0, "python")
from surfgym.core import STATE_DTYPE, SurfCore
from surfgym.goalfield import GoalField

rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 64
s = np.load("maps/surf_src_sidistic.sdf_32.npz"); sdf = s["sdf"].astype(np.float32); smin = s["mins"].astype(np.float64); C = float(s["cell"])
g = np.load("maps/surf_src_sidistic.goal_32.npz")
GF = GoalField(g["grid"].astype(np.float32) * float(g["quant"]), g["mins"], float(g["cell"]), float(g["reach_max"]))
core = SurfCore("maps/surf_src_sidistic.bsp")
box = np.array([[1950, 2800], [-700, 900], [9800, 10950]], float)   # measurement box (the helix)

def idx(p):
    return ((p - smin) // C).astype(int)

def sd(p):
    i = idx(p); return float(sdf[i[2], i[1], i[0]])

rows = []
tries = 0
while len(rows) < n_want and tries < 200000:
    tries += 1
    p = box[:, 0] + rng.random(3) * (box[:, 1] - box[:, 0])
    d0 = sd(p)
    if not (30.0 <= d0 <= 90.0):
        continue
    # the helix side: below / east of the diagonal ramp z = 9,968 + 0.909 (x - 1,760) (measurement only)
    if len(sys.argv) > 4 and p[2] > 9968.0 + 0.909 * (p[0] - 1760.0) - 40.0:
        continue
    if float(GF.sample(p[None])[0]) >= float(g["reach_max"]) - 16:
        continue
    t = core.trace(p, p, 0)
    if t.startsolid or t.allsolid:
        continue
    gr = np.array([sd(p + e * C) - sd(p - e * C) for e in np.eye(3)])
    nrm = np.linalg.norm(gr)
    if nrm < 1e-3:
        continue
    n = gr / nrm                                   # away from the surface
    a = rng.random() * 2 * np.pi
    dvec = np.array([np.cos(a), np.sin(a), 0.0])
    tan = dvec - dvec.dot(n) * n
    if np.linalg.norm(tan) < 0.3:
        continue
    tan /= np.linalg.norm(tan)
    v = tan * rng.uniform(1500.0, 2200.0)
    st = np.zeros(1, STATE_DTYPE)
    st["origin"][0] = p; st["velocity"][0] = v
    st["yaw"][0] = np.degrees(np.arctan2(v[1], v[0])); st["pitch"][0] = 0.0
    st["onground"][0] = -1; st["ducked"][0] = 0
    rows.append(st[0])
arr = np.array(rows, dtype=STATE_DTYPE)
np.save(sys.argv[3] if len(sys.argv) > 3 else "surf_states.npy", arr)
print(f"{len(arr)} states from {tries} tries; z range {arr['origin'][:,2].min():.0f}-{arr['origin'][:,2].max():.0f}; "
      f"|v| median {np.median(np.linalg.norm(arr['velocity'], axis=1)):.0f}; vz>0 share {np.mean(arr['velocity'][:,2] > 0):.0%}")
