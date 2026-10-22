"""ANALYSIS ONLY: random held macros from the greedy line's own launch-attempt states (att.npy);
count shaft entries, funnel entries and finish-depth arrivals; dump the winning plans."""
import sys, time, json, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python'))
from surfgym import SurfCore, default_config
S = np.load(sys.argv[1]); N = int(sys.argv[2]); REPS = int(sys.argv[3]); SEED = int(sys.argv[4])
SCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
rng = np.random.default_rng(SEED)
core = SurfCore(str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'maps' / 'surf_src_sidistic.bsp'),
                default_config(num_envs=N, spawn_mode=2, max_episode_ticks=10**6, sv_maxvelocity=4000.0,
                               lidar_w=0, lidar_h=0, view_mode=1, yaw_adaptive=1))
core.set_teleport_fail(True)
tot = {'shaft': 0, 'shaft_fast': 0, 'funnel': 0, 'finish': 0, 'n': 0}
wins = []
fast_states = []
t0 = time.time()
for rep in range(REPS):
    src = rng.integers(0, len(S), N)
    pool = S[src].copy(); pool['tick'] = 0; pool['stuck_ticks'] = 0
    pool['velocity'] = pool['velocity'] * np.float32(SCALE)
    core.set_spawn_pool(pool); core.reset(seed=int(rng.integers(1 << 30)))
    for i in range(N): core.set_state(i, pool[i])
    nseg = 5
    dur = rng.uniform(0.1, 0.8, (N, nseg)); bounds = np.cumsum(dur, 1) * 100
    yaw = rng.uniform(-150, 150, (N, nseg)).astype(np.float32)
    fwd = rng.integers(0, 3, (N, nseg)); side = rng.integers(0, 3, (N, nseg))
    jump = (rng.random((N, nseg)) < 0.15).astype(np.int32); duck = (rng.random((N, nseg)) < 0.2).astype(np.int32)
    acts = np.zeros((N, 6), np.int32); view = np.zeros((N, 2), np.float32); idx = np.arange(N)
    alive = np.ones(N, bool); shaft = np.zeros(N, bool); fast = np.zeros(N, bool); funnel = np.zeros(N, bool); fin = np.zeros(N, bool)
    for t in range(400):
        seg = np.minimum((t >= bounds).sum(1), nseg - 1)
        acts[:, 2] = fwd[idx, seg]; acts[:, 3] = side[idx, seg]; acts[:, 4] = jump[idx, seg]; acts[:, 5] = duck[idx, seg]
        view[:, 0] = yaw[idx, seg]
        _, _, done, trunc, _ = core.step(acts, view)
        st = core.states_view; p = np.asarray(st['origin'], np.float64); v = np.asarray(st['velocity'], np.float64)
        insh = alive & (p[:, 0] > 2464) & (p[:, 0] < 3424) & (np.abs(p[:, 1]) < 704) & (p[:, 2] > 1232) & (p[:, 2] < 3000)
        newfast = insh & ~shaft & (v[:, 2] > 600)
        for i in np.flatnonzero(newfast)[:64]:
            fast_states.append(st[i].copy())
            if len(wins) < 40:
                s0 = pool[i]
                wins.append({'t': t, 'src_o': [round(float(x)) for x in s0['origin']], 'src_v': [round(float(x)) for x in s0['velocity']],
                             'yaw': yaw[i].round().tolist(), 'fwd': fwd[i].tolist(), 'side': side[i].tolist(), 'jump': jump[i].tolist(),
                             'duck': duck[i].tolist(), 'dur': dur[i].round(2).tolist(),
                             'entry_o': [round(float(x)) for x in p[i]], 'entry_v': [round(float(x)) for x in v[i]]})
        fast |= newfast
        shaft |= insh
        funnel |= alive & shaft & (p[:, 0] > 2872) & (p[:, 0] < 3048) & (np.abs(p[:, 1]) < 88) & (p[:, 2] < 1250)
        fin |= alive & shaft & (p[:, 2] < 260) & (p[:, 0] > 2872) & (p[:, 0] < 3048)
        alive &= ~(np.asarray(done, bool) | np.asarray(trunc, bool))
    tot['n'] += N; tot['shaft'] += int(shaft.sum()); tot['shaft_fast'] += int(fast.sum()); tot['funnel'] += int(funnel.sum()); tot['finish'] += int(fin.sum())
    for i in np.flatnonzero(fin | funnel)[:20]:
        wins.append({'src': int(src[i]), 'o': [round(float(x)) for x in pool[i]['origin']], 'v': [round(float(x)) for x in pool[i]['velocity']],
                     'yaw': yaw[i].round().tolist(), 'fwd': fwd[i].tolist(), 'side': side[i].tolist(), 'jump': jump[i].tolist(),
                     'duck': duck[i].tolist(), 'dur': dur[i].round(2).tolist(), 'finish': bool(fin[i])})
print(f"scale {SCALE}: {tot} in {time.time()-t0:.0f}s")
if fast_states and len(sys.argv) > 6:
    F = np.array(fast_states); F['tick'] = 0; F['stuck_ticks'] = 0
    np.save(sys.argv[6], F); print('saved', len(F), 'fast post-launch states')
for w in wins[:8]:
    print(' ', w)
