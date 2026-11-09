"""analysis only (unr1b1, 2026-11-09): is the post-CP2 slot of surf_src_sidistic reachable AT ALL
from v2's own greedy states, and from how far upstream? A cross-entropy search over the explore
step's own action set (one draw per 4-tick decision: view offset from the velocity heading x
forward x side x jump x duck, exactly what explore_phase1 --view-follow can express) from v2's
greedy eval states T seconds before the end wall. MEASUREMENT only (CLAUDE.md 0b): the boxes
below are map coordinates read off the benches, never a recipe input.

Score of a sample: the deepest -x reached while alive AND above the ledge top (z >= 10,340), in
16 u bins, +10,000 if it passes (alive, x < -975, z > 10,000: past the ledge, as the other
benches); ties broken by how near its x -560 crossing is to the slit's measured envelope.
Per start: best score per iteration, passing samples, and how the best sample crosses x -560
(y, z, -vx: the slit's envelope is |y| < ~40, z <= ~10,620, -vx >= ~1,000).

usage: slot_cem.py [traj.jsonl] [ep] [T seconds, comma list] [N] [iters] [envelope|depth] [uniform|policy|line] [w_pol]
"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, "/home/steam/ar_uni/wt/b1/python")
from surfgym.core import SurfCore, default_config, YAW_BINS  # noqa: E402

MAP = "/home/steam/ar_uni/wt/b1/maps/surf_src_sidistic.bsp"
TRAJ = sys.argv[1] if len(sys.argv) > 1 else "/home/steam/ar_uni/runs/unr1b1/unr1b1_v2/traj_13550747648.jsonl"
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TS = [float(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "0.5,1,1.5,2,3,4,6").split(",")]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
MODE = sys.argv[6] if len(sys.argv) > 6 else "envelope"     # or "depth"
INIT = sys.argv[7] if len(sys.argv) > 7 else "uniform"      # or "policy": start at the line's own keys
W_POL = float(sys.argv[8]) if len(sys.argv) > 8 else 0.7   # INIT policy/line: weight of the policy's choice
DELTA = np.array([-30.0, -15.0, -7.0, -3.0, 0.0, 3.0, 7.0, 15.0, 30.0])
# INIT line: the view is the LINE'S OWN recorded yaw (per tick, open loop) + a held delta per
# decision from DELTA (component 0), instead of an offset from the velocity heading
EVERY = 4
TAIL = 2.0                       # seconds of flight after the start's own time to the wall
VF = np.array([-90.0, -60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0, 90.0])
NK = (len(VF), 3, 3, 2, 2)       # view offset, forward, side, jump, duck
ELITE = 0.04
SMOOTH = 0.6
FLOOR = 0.01
LEDGE_Z = 10340.0
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


ep = episodes(TRAJ)[EP]
iw = next(i for i, r in enumerate(ep) if r[1] < -700 and r[3] > 10000)
rng = np.random.default_rng(1109)
vf_bins = YAW_BINS.astype(np.float64)
print(f"{TRAJ.split('/')[-1]} ep {EP}: end-wall row {iw} (tick {ep[iw][0]}); score {MODE}; init {INIT}; N {N}, iters {ITERS}, "
      f"elite {ELITE:g}, smooth {SMOOTH:g}, floor {FLOOR:g}")
sys.stdout.flush()

cfg = default_config(num_envs=N, spawn_mode=2, max_episode_ticks=6000, water_fail=1,
                     yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
cfg.phys.sv_maxvelocity = 4000.0
core = SurfCore(MAP, cfg)
core.set_teleport_fail(True)

for T in TS:
    off = int(round(T * 100))
    i0 = iw - off
    i0 -= int(ep[i0][0]) % EVERY          # the policy's decision phase: its keys change on tick % 4 == 0
    r = ep[i0]
    duck0 = (int(r[8]) & 4 != 0) or bool(core.trace((r[1], r[2], r[3]), (r[1], r[2], r[3]), 0).startsolid)
    H = int(round((T + TAIL) * 100 / EVERY))
    P = [np.full((H, k), 1.0 / k) for k in NK]
    if INIT in ("policy", "line"):
        # the eval rows carry the policy's own keys per tick ([.., buttons, onground, progress,
        # reward, pitch, fwd, side]) and its view; each decision starts at W_POL on the policy's
        # choice (the view as the nearest held offset from the velocity heading), the rest uniform
        for d in range(H):
            j = i0 + d * EVERY
            if j >= len(ep):
                break
            q = ep[j]
            hd = np.degrees(np.arctan2(q[5], q[4]))
            vo = (q[7] - hd + 180.0) % 360.0 - 180.0
            v0 = 4 if INIT == "line" else int(np.abs(VF - vo).argmin())
            pick = (v0, int(q[13]), int(q[14]), int((int(q[8]) & 2) != 0),
                    int((int(q[8]) & 4) != 0))
            for c in range(5):
                P[c][d] = (1.0 - W_POL) / NK[c]
                P[c][d, pick[c]] += W_POL
    print(f"\n== T {T:g} s before the wall: start ({r[1]:.0f}, {r[2]:.0f}, {r[3]:.0f}) v ({r[4]:.0f}, {r[5]:.0f}, "
          f"{r[6]:.0f}) ducked {int(duck0)}; {H} decisions ({H * EVERY} ticks)")
    best_all = -1e9
    line_yaw = np.array([q[7] for q in ep[i0:]], np.float64)
    best_rec = None
    t0 = time.time()
    for it in range(ITERS):
        core.reset(0)
        base = core.get_states().copy()
        for i in range(N):
            s = base[i:i + 1].copy()
            s["origin"] = (r[1], r[2], r[3])
            s["velocity"] = (r[4], r[5], r[6])
            s["basevelocity"] = (0.0, 0.0, 0.0)
            s["yaw"] = r[7]
            s["pitch"] = 0.0
            s["onground"] = -1
            s["ducked"] = 1 if duck0 else 0
            s["induck"] = 0
            s["duck_time"] = 0.0
            s["oldbuttons"] = int(r[8])
            s["tick"] = 0
            s["stuck_ticks"] = 0
            core.set_state(i, s)
        # sample every decision's action for every sample from the per-decision categoricals
        A = np.zeros((N, H, 5), np.int64)
        for c in range(5):
            cdf = np.cumsum(P[c], axis=1)
            u = rng.random((N, H))
            A[:, :, c] = np.minimum((u[:, :, None] > cdf[None, :, :]).sum(axis=2), NK[c] - 1)
        act = np.zeros((N, 6), np.int32)
        act[:, 1] = 3
        st = core.get_states()
        alive = np.ones(N, bool)
        passed = np.zeros(N, bool)
        deep = np.full(N, 1e9)                  # min x while alive and above the ledge top
        minx = np.full(N, 1e9)                  # min x while alive
        px = st["origin"][:, 0].astype(np.float64).copy()
        cross = np.zeros(N, bool)
        cy = np.full(N, np.nan)
        cz = np.full(N, np.nan)
        cvx = np.full(N, np.nan)
        for t in range(H * EVERY):
            d = t // EVERY
            if t % EVERY == 0:
                act[:, 2] = A[:, d, 1]
                act[:, 3] = A[:, d, 2]
                act[:, 4] = A[:, d, 3]
                act[:, 5] = A[:, d, 4]
                vf_off = VF[A[:, d, 0]]
            if INIT == "line":
                want = line_yaw[min(t, len(line_yaw) - 1)] + DELTA[A[:, d, 0]]
            else:
                vel = st["velocity"].astype(np.float64)
                want = np.degrees(np.arctan2(vel[:, 1], vel[:, 0])) + vf_off
            dlt = (want - st["yaw"].astype(np.float64) + 180.0) % 360.0 - 180.0
            dlt = np.clip(dlt, vf_bins[0], vf_bins[-1])
            act[:, 0] = np.abs(vf_bins[None, :] - dlt[:, None]).argmin(axis=1)
            _, _, done, trunc, _ = core.step(act)
            st = core.get_states()
            alive &= ~(done | trunc).astype(bool)
            x = st["origin"][:, 0].astype(np.float64)
            y = st["origin"][:, 1].astype(np.float64)
            z = st["origin"][:, 2].astype(np.float64)
            up = alive & (z >= LEDGE_Z)
            deep = np.where(up, np.minimum(deep, x), deep)
            minx = np.where(alive, np.minimum(minx, x), minx)
            passed |= alive & (x < -975.0) & (z > 10000.0)
            new = alive & ~cross & (px >= XS) & (x < XS)
            cy[new] = y[new]
            cz[new] = z[new]
            cvx[new] = -st["velocity"][new, 0]
            cross |= new
            px = x
        if MODE == "depth":
            # primary: depth above the ledge top in 16 u bins (+10,000 if passed); tie-break inside
            # a bin (< 16): how near the x -560 crossing is to the slit's measured envelope
            dist = np.abs(cy) + np.maximum(0.0, cz - 10600.0) + 0.5 * np.maximum(0.0, 1000.0 - cvx)
            sec = np.where(np.isfinite(dist), 1.0 / (1.0 + np.nan_to_num(dist, nan=1e9) / 100.0), 0.0)
            score = 10000.0 * passed + 16.0 * np.floor(-deep / 16.0) + 15.0 * sec
        else:
            # envelope: distance of the x -560 crossing from the box the hands-off bench passed from
            # (|y| < 30, z 10,580-10,620, -vx >= 1,000); no crossing: 5,000 + how far short of -560
            env = (np.maximum(0.0, np.abs(cy) - 30.0) + np.maximum(0.0, cz - 10620.0)
                   + np.maximum(0.0, 10580.0 - cz) + 0.5 * np.maximum(0.0, 1000.0 - cvx))
            short = 5000.0 + np.maximum(0.0, minx - XS)
            dist = np.where(cross, env, short)
            score = 10000.0 * passed - dist
        k = max(1, int(ELITE * N))
        el = np.argsort(-score)[:k]
        for c in range(5):
            freq = np.zeros((H, NK[c]))
            for j in range(NK[c]):
                freq[:, j] = (A[el, :, c] == j).mean(axis=0)
            P[c] = SMOOTH * freq + (1.0 - SMOOTH) * P[c]
            P[c] = np.maximum(P[c], FLOOR)
            P[c] /= P[c].sum(axis=1, keepdims=True)
        b = int(el[0])
        if score[b] > best_all:
            best_all = float(score[b])
            best_rec = (it, float(cy[b]), float(cz[b]), float(cvx[b]), bool(passed[b]))
        slit = cross & (np.abs(cy) < 48.0) & (cz <= 10640.0)
        print(f"  it {it:2d}: best dist {dist[b]:7.1f} (in the envelope {int((cross & (dist <= 0.0)).sum()):4d}) | "
              f"deepest x above the ledge top {deep[b]:7.0f} | "
              f"passed {int(passed.sum()):4d} | elite median {np.median(deep[el]):7.0f} | "
              f"in the slit at <= 10,640 {int(slit.sum()):4d} (max -vx {np.nanmax(cvx[slit]) if slit.any() else float('nan'):5.0f}) | "
              f"best's x -560 crossing y {cy[b]:6.0f} z {cz[b]:6.0f} -vx {cvx[b]:5.0f}   [{time.time() - t0:4.0f}s]")
        sys.stdout.flush()
    print(f"T {T:g}: best {best_all:.0f} (it {best_rec[0]}; crossing y {best_rec[1]:.0f} z {best_rec[2]:.0f} "
          f"-vx {best_rec[3]:.0f}; passed {best_rec[4]})")
