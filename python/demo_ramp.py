"""Gate F demo: drop a player onto a real surf_ski_2 ramp with zero input and
record the slide as a viewer-ready trajectory. The surf mechanic, end to end."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from surfgym import SurfCore, default_config, record_rollout


class NeutralPolicy:
    """Bin 7 = 0 yaw delta, bin 3 = 0 pitch; fwd/side neutral; no jump/duck."""
    def act(self, obs):
        n = obs.shape[0]
        a = np.zeros((n, 6), dtype=np.int32)
        a[:, 0] = 7   # yaw: no turn
        a[:, 1] = 3   # pitch: level
        a[:, 2] = 1   # forward neutral
        a[:, 3] = 1   # side neutral
        return a


def main():
    root = Path(__file__).parent.parent
    cfg = default_config(num_envs=4, spawn_mode=0, max_episode_ticks=2000)
    core = SurfCore(str(root / "maps" / "surf_ski_2.bsp"), cfg)

    # Known surfable ramp (from test_physics find_ramp): point + normal
    ramp = np.array([-3106.0, -2359.0, -615.0], dtype=np.float64)
    n = np.array([0.189, 0.731, 0.656], dtype=np.float64)
    # downslope horizontal direction ~ (n_x, n_y) normalized; dz/dh = -sin/cos
    dh = n[:2] / np.linalg.norm(n[:2])
    slope = -np.sqrt(1.0 - n[2] ** 2) / n[2]

    # waypoints straight down the ramp line, ~1600u of travel
    pts = []
    for t in (-200.0, 200.0, 600.0, 1000.0, 1400.0):
        pts.append([ramp[0] + dh[0] * t, ramp[1] + dh[1] * t, ramp[2] + slope * t + 30.0])
    core.set_waypoints(np.asarray(pts, dtype=np.float32))

    core.reset(seed=7)
    # place env 0 just above the ramp, slight downslope drift
    st = core.get_states()[0].copy()
    st["origin"] = (ramp + np.array([0, 0, 40.0])).astype(np.float32)
    st["velocity"] = np.array([dh[0] * 50, dh[1] * 50, 0], dtype=np.float32)
    st["yaw"] = np.degrees(np.arctan2(dh[1], dh[0])) % 360.0
    st["onground"] = -1
    # zero the tracking fields — stale spawn-projected progress poisons rewards
    st["progress"] = 0.0
    st["best_progress"] = 0.0
    st["seg_hint"] = 0
    st["tick"] = 0
    st["stuck_ticks"] = 0
    core.set_state(0, st)

    out = root / "runs" / "ramp_slide.traj.jsonl"
    out.parent.mkdir(exist_ok=True)
    summaries = record_rollout(core, NeutralPolicy(), str(out), episodes=2, max_ticks=1200,
                               seed=None)   # keep the placed state — do not reset

    states = core.get_states()
    print(f"wrote {out}")
    for i, s in enumerate(summaries):
        print(f"episode {i}: end={s.get('end')} ticks={s.get('ticks')} "
              f"best_progress={s.get('best_progress', 0):.1f}")
    # quick speed audit from the file
    import json
    speeds = []
    with open(out) as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, list):
                speeds.append((row[4] ** 2 + row[5] ** 2) ** 0.5)
    print(f"max horizontal speed in recording: {max(speeds):.1f} u/s over {len(speeds)} ticks")
    assert max(speeds) > 400.0, "ramp slide did not accelerate — physics or waypoint problem"
    print("DEMO OK: the agent surfs.")


if __name__ == "__main__":
    main()
