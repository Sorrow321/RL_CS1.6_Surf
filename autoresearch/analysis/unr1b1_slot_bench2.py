"""analysis only (unr1b1, 2026-11-09): from v2's OWN greedy positions on the approach to the
post-CP2 slot (kicker apex -> end wall), which velocities carry a hands-off, ducked flight past
the ledge? MEASUREMENT only (CLAUDE.md 0b): nothing here enters a recipe.

For each position the policy flew through (eval rows), the velocity is transformed: horizontal
speed x K, heading rotated by TH degrees, vz + DVZ. The flight runs hands-off (no keys, duck
held) for 3 s in the simulator; passed = origin x < X_PAST alive with z > 10,000 (past the
ledge, in the tunnel). Output: the passing share per K, the slowest passing launches, and the
policy's own velocity (K 1, TH 0, DVZ 0).
"""
import json
import math
import sys
import numpy as np

sys.path.insert(0, "/home/steam/ar_uni/wt/b1/python")
from surfgym.core import SurfCore, default_config  # noqa: E402

MAP = "/home/steam/ar_uni/wt/b1/maps/surf_src_sidistic.bsp"
R = "/home/steam/ar_uni/runs/unr1b1/unr1b1_v2/"
X_PAST = -975.0
TICKS = 300
K = [0.6, 0.8, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5]
TH = [-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0]
DVZ = [-400.0, -200.0, 0.0, 200.0, 400.0]


def episodes(path):
    eps, cur = [], None
    for line in open(path):
        d = json.loads(line)
        if isinstance(d, dict):
            cur = []
            eps.append(cur)
        else:
            cur.append(d)
    return [e for e in eps if e]


# positions: every 2nd row from the kicker apex (x < -150 above 10,600) to the end-wall hit
starts = []
for f, picks in (("traj_6302990336.jsonl", (0, 3, 5)), ("traj_12720275456.jsonl", (0, 2, 4))):
    eps = episodes(R + f)
    for k in picks:
        if k >= len(eps):
            continue
        ep = eps[k]
        try:
            i2 = next(i for i, r in enumerate(ep) if r[1] < -700 and r[3] > 10000)
        except StopIteration:
            continue
        i1 = next(i for i in range(max(0, i2 - 200), i2) if ep[i][1] < -150 and ep[i][3] > 10600)
        for r in ep[i1:i2 + 1:2]:
            starts.append((f, k, r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
print(f"{len(starts)} start positions from the policy's own approach rows")

combos = [(si, kk, th, dvz) for si in range(len(starts)) for kk in K for th in TH for dvz in DVZ]
N = len(combos)
print(f"{N} flights")
B = 20000
passed = np.zeros(N, bool)
minx = np.zeros(N)
for b0 in range(0, N, B):
    sub = combos[b0:b0 + B]
    n = len(sub)
    cfg = default_config(num_envs=n, spawn_mode=2, max_episode_ticks=6000, water_fail=1,
                         yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    cfg.phys.sv_maxvelocity = 4000.0
    core = SurfCore(MAP, cfg)
    core.set_teleport_fail(True)
    core.reset(0)
    base = core.get_states().copy()
    for i, (si, kk, th, dvz) in enumerate(sub):
        _, _, _, x, y, z, vx, vy, vz = starts[si]
        a = math.radians(th)
        hx, hy = kk * vx, kk * vy
        s = base[i:i + 1].copy()
        s["origin"] = (x, y, z)
        s["velocity"] = (hx * math.cos(a) - hy * math.sin(a), hx * math.sin(a) + hy * math.cos(a), vz + dvz)
        s["basevelocity"] = (0.0, 0.0, 0.0)
        s["yaw"] = math.degrees(math.atan2(s["velocity"][0][1], s["velocity"][0][0]))
        s["pitch"] = 0.0
        s["onground"] = -1
        s["ducked"] = 1
        s["induck"] = 0
        s["duck_time"] = 0.0
        s["oldbuttons"] = 4
        s["tick"] = 0
        s["stuck_ticks"] = 0
        core.set_state(i, s)
    acts = np.zeros((n, 6), np.int32)
    acts[:, 0] = 7
    acts[:, 1] = 3
    acts[:, 2] = 1
    acts[:, 3] = 1
    acts[:, 5] = 1
    st = core.get_states()
    alive = np.ones(n, bool)
    mx = st["origin"][:, 0].astype(np.float64).copy()
    ok = np.zeros(n, bool)
    for t in range(TICKS):
        _, _, done, trunc, _ = core.step(acts)
        st = core.get_states()
        alive &= ~(done | trunc).astype(bool)
        x = st["origin"][:, 0].astype(np.float64)
        z = st["origin"][:, 2].astype(np.float64)
        mx = np.where(alive, np.minimum(mx, x), mx)
        ok |= alive & (x < X_PAST) & (z > 10000.0)
    passed[b0:b0 + n] = ok
    minx[b0:b0 + n] = mx
    del core
print(f"passed {int(passed.sum())} of {N}")
C = np.array([(c[1], c[2], c[3]) for c in combos])
for kk in K:
    m = C[:, 0] == kk
    print(f"  K {kk:4.2f}: passed {int(passed[m].sum()):6d} of {int(m.sum())}  "
          f"(TH 0, DVZ 0: {int(passed[m & (C[:, 1] == 0) & (C[:, 2] == 0)].sum())})")
own = (C[:, 0] == 1.0) & (C[:, 1] == 0) & (C[:, 2] == 0)
print(f"the policy's own velocities (K 1, TH 0, DVZ 0): passed {int(passed[own].sum())} of {int(own.sum())}; "
      f"deepest min x {minx[own].min():.0f}")
idx = np.flatnonzero(passed)
if len(idx):
    sp = []
    for i in idx:
        si, kk, th, dvz = combos[i]
        f, k, tick, x, y, z, vx, vy, vz = starts[si]
        vh = kk * math.hypot(vx, vy)
        sp.append((vh, i))
    sp.sort()
    print("slowest passing launches (|v_h| after K; start row; TH; DVZ):")
    for vh, i in sp[:15]:
        si, kk, th, dvz = combos[i]
        f, k, tick, x, y, z, vx, vy, vz = starts[si]
        print(f"  |v_h| {vh:5.0f} vz {vz + dvz:6.0f}  K {kk:4.2f} TH {th:4.0f} DVZ {dvz:5.0f}  from {f[5:-6]} ep {k} t {tick} "
              f"at ({x:.0f}, {y:.0f}, {z:.0f}), own v ({vx:.0f}, {vy:.0f}, {vz:.0f})")
    xs = np.array([starts[combos[i][0]][3] for i in idx])
    print(f"passing start x range {xs.min():.0f}..{xs.max():.0f}")
