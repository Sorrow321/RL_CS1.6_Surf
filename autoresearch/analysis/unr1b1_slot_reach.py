"""analysis only (unr1b1, 2026-11-09): can the Go-Explore explore step (uniform random actions,
action-repeat 0.95 per 4-tick decision, view held at an offset from the velocity heading - the
exact draw of tools/explore_phase1.py with v2's search flags) reach the post-CP2 slot's passing
envelope from v2's OWN greedy states on the approach? MEASUREMENT only (CLAUDE.md 0b): nothing
here enters a recipe.

Starts: v2's greedy eval rows OFF ticks before the end-wall approach (first row with x < -700
above z 10,000). Per start, BURSTS explorer bursts of 400 ticks (100 decisions x 4).
Per burst: passed (alive with x < -975 above z 10,000: past the ledge, in the tunnel); the slit
band (first crossing of x = -560 with |y| < 48 and 10,560 <= z <= 10,640) and the -vx there; the
z and -vx at the first crossing of x = -560 anywhere. Passing envelope measured on the bench
(unr1b1_slot_bench3.py): in the slit at shelf height with -vx >= ~1,000 u/s, vz -200..-600.
"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, "/home/steam/ar_uni/wt/b1/python")
from surfgym.core import SurfCore, default_config, YAW_BINS, ACTION_NVEC  # noqa: E402

MAP = "/home/steam/ar_uni/wt/b1/maps/surf_src_sidistic.bsp"
TRAJ = sys.argv[1] if len(sys.argv) > 1 else "/home/steam/ar_uni/runs/unr1b1/unr1b1_v2/traj_13550747648.jsonl"
BURSTS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
EPS = (0, 4, 8)
OFF = (15, 25, 35, 50, 75, 100, 125, 150, 200, 250)
TICKS, EVERY, REPEAT = 400, 4, 0.95
VF = np.array([-90.0, -60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0, 90.0])
XS = -560.0


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


eps = episodes(TRAJ)
starts = []
for k in EPS:
    ep = eps[k]
    iw = next(i for i, r in enumerate(ep) if r[1] < -700 and r[3] > 10000)
    for off in OFF:
        r = ep[iw - off]
        starts.append((k, off, r))
print(f"{len(starts)} starts x {BURSTS} bursts of {TICKS} ticks (repeat {REPEAT}, view offsets {VF.tolist()})")
for k, off, r in starts:
    print(f"  ep {k} -{off:3d} ticks: ({r[1]:7.0f}, {r[2]:6.0f}, {r[3]:7.0f}) v ({r[4]:6.0f}, {r[5]:6.0f}, {r[6]:6.0f}) "
          f"buttons {r[8]}")
sys.stdout.flush()

rng = np.random.default_rng(20261109)
vf_bins = YAW_BINS.astype(np.float64)
N = BURSTS
res = []
t0 = time.time()
for si, (k, off, r) in enumerate(starts):
    cfg = default_config(num_envs=N, spawn_mode=2, max_episode_ticks=6000, water_fail=1,
                         yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    cfg.phys.sv_maxvelocity = 4000.0
    core = SurfCore(MAP, cfg)
    core.set_teleport_fail(True)
    core.reset(0)
    base = core.get_states().copy()
    # ducked when the row holds duck, or when the standing hull does not fit at the recorded origin
    # (a player still ducked in the air after releasing the key)
    duck = (int(r[8]) & 4 != 0) or bool(core.trace((r[1], r[2], r[3]), (r[1], r[2], r[3]), 0).startsolid)
    for i in range(N):
        s = base[i:i + 1].copy()
        s["origin"] = (r[1], r[2], r[3])
        s["velocity"] = (r[4], r[5], r[6])
        s["basevelocity"] = (0.0, 0.0, 0.0)
        s["yaw"] = r[7]
        s["pitch"] = 0.0
        s["onground"] = -1
        s["ducked"] = 1 if duck else 0
        s["induck"] = 0
        s["duck_time"] = 0.0
        s["oldbuttons"] = int(r[8])
        s["tick"] = 0
        s["stuck_ticks"] = 0
        core.set_state(i, s)
    act = np.zeros((N, 6), np.int32)
    vf_off = np.zeros(N)

    def fresh(idx):
        m = idx.size
        act[idx, 0] = rng.integers(0, ACTION_NVEC[0], m)
        act[idx, 1] = 3
        act[idx, 2] = rng.integers(0, ACTION_NVEC[2], m)
        act[idx, 3] = rng.integers(0, ACTION_NVEC[3], m)
        act[idx, 4] = rng.integers(0, ACTION_NVEC[4], m)
        act[idx, 5] = rng.integers(0, ACTION_NVEC[5], m)
        vf_off[idx] = VF[rng.integers(0, VF.size, m)]

    fresh(np.arange(N))
    alive = np.ones(N, bool)
    passed = np.zeros(N, bool)
    st = core.get_states()
    px = st["origin"][:, 0].astype(np.float64).copy()
    cross = np.zeros(N, bool)
    cz = np.full(N, np.nan)
    cvx = np.full(N, np.nan)
    cy = np.full(N, np.nan)
    minx = px.copy()
    for t in range(TICKS):
        if t % EVERY == 0 and t > 0:
            fresh(np.flatnonzero(alive & (rng.random(N) >= REPEAT)))
        vel = st["velocity"].astype(np.float64)
        fast = np.hypot(vel[:, 0], vel[:, 1]) >= 100.0
        want = np.degrees(np.arctan2(vel[:, 1], vel[:, 0])) + vf_off
        dlt = (want - st["yaw"].astype(np.float64) + 180.0) % 360.0 - 180.0
        dlt = np.clip(dlt, vf_bins[0], vf_bins[-1])
        b = np.abs(vf_bins[None, :] - dlt[:, None]).argmin(axis=1)
        act[fast, 0] = b[fast]
        _, _, done, trunc, _ = core.step(act)
        st = core.get_states()
        alive &= ~(done | trunc).astype(bool)
        x = st["origin"][:, 0].astype(np.float64)
        y = st["origin"][:, 1].astype(np.float64)
        z = st["origin"][:, 2].astype(np.float64)
        new = alive & ~cross & (px >= XS) & (x < XS)
        cz[new] = z[new]
        cy[new] = y[new]
        cvx[new] = -st["velocity"][new, 0]
        cross |= new
        px = x
        minx = np.where(alive, np.minimum(minx, x), minx)
        passed |= alive & (x < -975.0) & (z > 10000.0)
    slit = cross & (np.abs(cy) < 48.0) & (cz >= 10560.0) & (cz <= 10640.0)
    fastslit = slit & (cvx >= 1000.0)
    low = cross & (cz <= 10700.0)
    mid = cross & (np.abs(cy) < 48.0)
    print(f"    crossings at z <= 10,700: {int(low.sum())} (max -vx {np.nanmax(cvx[low]) if low.any() else float('nan'):.0f}); "
          f"at |y| < 48: {int(mid.sum())} (lowest z {np.nanmin(cz[mid]) if mid.any() else float('nan'):.0f}); "
          f"died {int((~alive).sum())} of {N}")
    res.append((k, off, int(passed.sum()), int(cross.sum()), int(slit.sum()), int(fastslit.sum()),
                float(np.nanmax(cvx[slit])) if slit.any() else float("nan"),
                float(np.nanmin(cz[cross])) if cross.any() else float("nan"),
                float(np.nanmax(cvx[cross])) if cross.any() else float("nan"),
                float(minx.min())))
    q = res[-1]
    print(f"ep {k} -{off:3d} (ducked {int(duck)}): passed {q[2]:5d} | crossed x {XS:.0f} {q[3]:5d} | in slit band {q[4]:5d} "
          f"(-vx >= 1000: {q[5]:4d}; max -vx there {q[6]:6.0f}) | lowest z at the crossing {q[7]:7.0f}, "
          f"max -vx at the crossing {q[8]:6.0f} | deepest x {q[9]:6.0f}   [{time.time() - t0:5.0f}s]")
    sys.stdout.flush()
    del core
tot = np.array([q[2:6] for q in res]).sum(axis=0)
print(f"TOTAL {len(starts) * N} bursts: passed {tot[0]}, crossed x {XS:.0f} {tot[1]}, in the slit band {tot[2]}, "
      f"in the slit band with -vx >= 1000 {tot[3]}")
