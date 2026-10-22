"""ANALYSIS ONLY (never training): can ANY simple held macro sequence climb out of sidistic's lower lane
(past the second thin wall) into the upper area / the funnel / the finish, from the POLICY'S OWN states there?

States: the checkpoint's reservoir states in the lane region (x 2,470-3,300, |y| 200-720, z 850-1,200),
plus the policy's greedy-eval arrival states right after the underpass (rows of its own traj files, rebuilt
on a reservoir template). Macros: 3 segments of 0.3-1.5 s, each holding a yaw offset U(-120,120) deg in the
velocity frame (core view_mode 1, the trainer's yaw_adaptive / 10 deg-per-tick ceiling), pitch 0, random
forward/side keys, jump 20%, duck 20%. 6 s per rollout. Physics = the eval header's phys block (maxvel 4000).
Reports: finishes, max z, how many get above z 1,260 / over kill *61 (|y| < 330, z > 1,230) / east of x 2,700
above z 1,260."""
import sys, json, numpy as np, torch, glob
sys.path.insert(0, 'python'); sys.path.insert(0, 'autoresearch')  # run from the repo root
from surfgym.core import SurfCore, default_config, STATE_DTYPE
import score as S

CK = sys.argv[1]  # a checkpoint whose reservoir holds lane states; NROLL = int(sys.argv[2]) if len(sys.argv) > 2 else 4096; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(SEED)
f = S.field()
ck = torch.load(CK, map_location='cpu', weights_only=False)
st = ck['respawn']['states']
o = st['origin']
m = (o[:, 0] > 2470) & (o[:, 0] < 3300) & (np.abs(o[:, 1]) > 200) & (np.abs(o[:, 1]) < 720) & (o[:, 2] > 850) & (o[:, 2] < 1200)
src = st[m]
print('lane reservoir states', len(src), 'speeds', np.percentile(np.hypot(src['velocity'][:, 0], src['velocity'][:, 1]), [10, 50, 90]).round())
# the policy's own greedy arrivals right after the underpass (x 2,470-2,560 moving east, z < 1,150)
arr = []
tmpl = st[0:1].copy()
for fn in sorted(glob.glob('/home/steam/ar/runs/r5b3/r5b3_*/traj_*.jsonl'))[-40:]:
    for h, r, t in S.episodes(fn):
        k = np.flatnonzero((r[:, 1] > 2470) & (r[:, 1] < 2560) & (r[:, 3] < 1150) & (np.abs(r[:, 2]) < 720) & (r[:, 4] > 300))
        if len(k):
            for kk in k[:40:8]:
                s = tmpl.copy()
                s['origin'][0] = r[kk, 1:4]; s['velocity'][0] = r[kk, 4:7]; s['basevelocity'][0] = 0
                s['yaw'][0] = r[kk, 7]; s['pitch'][0] = 0; s['onground'][0] = -1; s['ducked'][0] = 0; s['induck'][0] = 0
                s['tick'][0] = 0; s['stuck_ticks'][0] = 0
                arr.append(s[0])
arr = np.array(arr, dtype=STATE_DTYPE) if arr else np.zeros(0, STATE_DTYPE)
print('greedy arrival states', len(arr), 'speeds', np.percentile(np.hypot(arr['velocity'][:, 0], arr['velocity'][:, 1]), [10, 50, 90]).round() if len(arr) else '-')
pool = np.concatenate([src, arr]) if len(arr) else src
N = NROLL
cfg = default_config(num_envs=N, sv_maxvelocity=4000.0, yaw_adaptive=1, view_mode=1, lidar_w=0, lidar_h=0,
                     max_episode_ticks=100000, pitch_rate_max_deg=1.33)
core = SurfCore('maps/surf_src_sidistic.bsp', cfg)
core.set_teleport_fail(True)
z = json.load(open('maps/surf_src_sidistic.zones.json'))['end']
core.set_goal_box(z['mins'], z['maxs'])
core.reset(SEED)
pick = rng.integers(0, len(pool), N)
start = pool[pick].copy()
for i in range(N):
    core.set_state(i, start[i:i + 1])
TICKS = 600
seg_len = rng.integers(30, 151, (N, 3))
seg_yaw = rng.uniform(-120, 120, (N, 3)).astype(np.float32)
seg_fwd = rng.integers(0, 3, (N, 3)); seg_side = rng.integers(0, 3, (N, 3))
seg_jump = (rng.random((N, 3)) < 0.2).astype(np.int32); seg_duck = (rng.random((N, 3)) < 0.2).astype(np.int32)
bounds = np.cumsum(seg_len, 1)
alive = np.ones(N, bool); fin = np.zeros(N, bool); zmax = start['origin'][:, 2].copy()
over61 = np.zeros(N, bool); upper = np.zeros(N, bool); hi = np.zeros(N, bool); dmin = f.sample(start['origin'].astype(np.float64)).astype(np.float64)
act = np.zeros((N, 6), np.int32); view = np.zeros((N, 2), np.float32)
for t in range(TICKS):
    sidx = np.minimum((t >= bounds).sum(1), 2)
    ii = np.arange(N)
    view[:, 0] = seg_yaw[ii, sidx]; view[:, 1] = 0.0
    act[:, 2] = seg_fwd[ii, sidx]; act[:, 3] = seg_side[ii, sidx]; act[:, 4] = seg_jump[ii, sidx]; act[:, 5] = seg_duck[ii, sidx]
    pre = core.get_states()['origin'].astype(np.float64)
    _, _, done, trunc, _ = core.step(act, view)
    g = core.goal_hits.astype(bool).copy()
    ended = (done.astype(bool) | trunc.astype(bool)) & alive
    fin |= alive & g
    p = pre
    zmax = np.where(alive, np.maximum(zmax, p[:, 2]), zmax)
    hi |= alive & (p[:, 2] > 1260)
    over61 |= alive & (np.abs(p[:, 1]) < 330) & (p[:, 2] > 1230) & (p[:, 0] > 2480)
    upper |= alive & (p[:, 0] > 2700) & (p[:, 2] > 1260)
    d = f.sample(p).astype(np.float64)
    dmin = np.where(alive, np.minimum(dmin, d), dmin)
    alive &= ~ended
    if not alive.any():
        break
D0 = 97768.375
print('rollouts', N, 'ticks', t + 1, 'alive at end', int(alive.sum()))
print('finishes', int(fin.sum()), '| z>1260', int(hi.sum()), '| over *61 (|y|<330, z>1230)', int(over61.sum()), '| east of x 2700 above z 1260', int(upper.sum()))
print('best progress %.2f%%' % (100 * (D0 - dmin.min()) / D0), 'zmax p50/p99/max', np.percentile(zmax, [50, 99, 100]).round())
for i in np.argsort(dmin)[:5]:
    print('  best', '%.2f%%' % (100 * (D0 - dmin[i]) / D0), 'start', np.round(start['origin'][i]).astype(int).tolist(), 'v', np.round(start['velocity'][i]).astype(int).tolist(),
          'segs', seg_len[i].tolist(), np.round(seg_yaw[i]).tolist(), seg_fwd[i].tolist(), seg_side[i].tolist(), seg_jump[i].tolist(), seg_duck[i].tolist(), 'zmax', int(zmax[i]))
