# MEASUREMENT: survival of archived pit states restored two ways - core.set_state (the explorer's
# restore) vs the spawn pool (set_spawn_pool + reset/autoreset: the trainer's and record_ckpt's).
import sys
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.core import SurfCore, default_config
S = np.load(sys.argv[1], allow_pickle=False)[:12]
cfg = default_config(num_envs=1, spawn_mode=2, max_episode_ticks=6000, water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
cfg.phys.sv_maxvelocity = 3500.0
core = SurfCore("/root/RL_Surf/maps/surf_unitfarmer2.bsp", cfg)
core.set_teleport_fail(True)
a = np.array([[7, 3, 1, 1, 0, 0]], np.int32)   # neutral keys, no turn
def run(path_set_state, s):
    s = s.copy().reshape(1); s["tick"] = 0; s["stuck_ticks"] = 0
    if path_set_state:
        core.reset(0); core.set_state(0, s)
    else:
        core.set_spawn_pool(np.repeat(s, 4)); core.reset(0)
    st0 = core.get_states()[0].copy()
    for t in range(400):
        _, _, done, trunc, _ = core.step(a)
        if done[0] or trunc[0]:
            return t, st0
    return 400, st0
for i, s in enumerate(S):
    t1, a1 = run(True, s); t2, a2 = run(False, s)
    print(f"row {i:2d} z {float(s['origin'][2]):6.0f} |v| {float(np.linalg.norm(s['velocity'])):5.0f}  set_state: dies at tick {t1:3d}  pool: dies at tick {t2:3d}  pool origin diff {float(np.linalg.norm(a2['origin'].astype(float) - s['origin'].astype(float))):6.1f} vel diff {float(np.linalg.norm(a2['velocity'].astype(float) - s['velocity'].astype(float))):6.1f} duck {int(s['ducked']) if 'ducked' in s.dtype.names else -1}->{int(a2['ducked']) if 'ducked' in a2.dtype.names else -1}")
