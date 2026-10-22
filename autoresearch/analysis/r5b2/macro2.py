"""ANALYSIS ONLY, two stages: (1) random held macros from the policy's own room states; keep the
states at the tick each rollout enters the shaft through a hole; (2) random macros from those
post-launch states: does any reach the funnel and the finish depth?"""
import sys, time, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python'))
from surfgym import SurfCore, default_config
CK = sys.argv[1]; SEED = int(sys.argv[2]); N1 = int(sys.argv[3]); REPS = int(sys.argv[4]); N2 = int(sys.argv[5])
rng = np.random.default_rng(SEED)
ck = torch.load(CK, map_location='cpu', weights_only=False)
G = ck['respawn_greedy']['rows']
o = G['origin'].astype(np.float64)
S = G[(o[:, 0] > 2250) & (o[:, 0] < 2750) & (o[:, 2] < 1200) & (o[:, 2] > 700) & (np.abs(o[:, 1]) > 250)]
def make(n):
    c = SurfCore(str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'maps' / 'surf_src_sidistic.bsp'),
                 default_config(num_envs=n, spawn_mode=2, max_episode_ticks=10**6, sv_maxvelocity=4000.0,
                                lidar_w=0, lidar_h=0, view_mode=1, yaw_adaptive=1))
    c.set_teleport_fail(True); return c
def rollout(core, pool, secs, nseg, stop_on_shaft):
    n = len(pool); core.set_spawn_pool(pool); core.reset(seed=int(rng.integers(1 << 30)))
    for i in range(n): core.set_state(i, pool[i])
    dur = rng.uniform(0.15, 1.0, (n, nseg)); bounds = np.cumsum(dur, 1) * 100
    yaw = rng.uniform(-150, 150, (n, nseg)).astype(np.float32)
    fwd = rng.integers(0, 3, (n, nseg)); side = rng.integers(0, 3, (n, nseg))
    jump = (rng.random((n, nseg)) < 0.25).astype(np.int32); duck = (rng.random((n, nseg)) < 0.3).astype(np.int32)
    acts = np.zeros((n, 6), np.int32); view = np.zeros((n, 2), np.float32)
    alive = np.ones(n, bool); got = {}; funnel = np.zeros(n, bool); fin = np.zeros(n, bool); idx = np.arange(n)
    for t in range(int(secs * 100)):
        seg = np.minimum((t >= bounds).sum(1), nseg - 1)
        acts[:, 2] = fwd[idx, seg]; acts[:, 3] = side[idx, seg]; acts[:, 4] = jump[idx, seg]; acts[:, 5] = duck[idx, seg]
        view[:, 0] = yaw[idx, seg]
        _, _, done, trunc, _ = core.step(acts, view)
        st = core.states_view
        p = np.asarray(st['origin'], np.float64)
        shaft = (p[:, 0] > 2464) & (p[:, 2] > 1232)
        if stop_on_shaft:
            for i in np.flatnonzero(alive & shaft):
                if i not in got: got[i] = st[i].copy()
        funnel |= alive & (p[:, 0] > 2872) & (p[:, 0] < 3048) & (np.abs(p[:, 1]) < 88) & (p[:, 2] < 1180)
        fin |= alive & (p[:, 2] < 260) & (p[:, 0] > 2872) & (p[:, 0] < 3048) & (np.abs(p[:, 1]) < 88)
        alive &= ~(np.asarray(done, bool) | np.asarray(trunc, bool))
    return got, funnel, fin, alive
t0 = time.time()
c1 = make(N1)
entries = []
for r in range(REPS):
    pool = S[rng.integers(0, len(S), N1)].copy(); pool['tick'] = 0; pool['stuck_ticks'] = 0
    got, _, _, _ = rollout(c1, pool, 3.0, 4, True)
    entries += list(got.values())
print(f"stage 1: {REPS * N1} rollouts -> {len(entries)} shaft entries ({time.time()-t0:.0f}s)")
if not entries: sys.exit()
E = np.array(entries); E['tick'] = 0; E['stuck_ticks'] = 0
c2 = make(N2)
pool = E[rng.integers(0, len(E), N2)].copy()
got, funnel, fin, alive = rollout(c2, pool, 4.0, 5, False)
print(f"stage 2: {N2} rollouts from {len(E)} post-launch states: funnel {funnel.sum()}, finish depth {fin.sum()}, alive {alive.sum()} ({time.time()-t0:.0f}s)")
v = np.asarray(E['velocity'], np.float64); p = np.asarray(E['origin'], np.float64)
print('post-launch states: x', np.percentile(p[:, 0], [5, 50, 95]).round(), 'z', np.percentile(p[:, 2], [5, 50, 95]).round(), 'vz', np.percentile(v[:, 2], [5, 50, 95]).round(), '|vh|', np.percentile(np.hypot(v[:, 0], v[:, 1]), [5, 50, 95]).round())
