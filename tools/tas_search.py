"""tas_search.py - can a better line be FOUND on a segment the agent already runs?

The value-ceiling analysis said the record is a line-geometry problem:
perfect strafing on the champion's line tops out at 73.66 s, still 5.7 s
short of 1:08, so the remaining time has to come from losing less energy
into ramp normals - i.e. a different line. This asks the question
directly and cheaply: restore the exact physics state at some tick of a
recorded run, then search action sequences over the next W ticks and see
whether anything beats what the policy actually did.

Why this is affordable here and not for the TrackMania bruteforcers who
invented the technique: the env is vectorized, so a candidate population
of 2048 mutations is evaluated in ONE batched rollout of W ticks. At
~13M eyeless env-steps/s that is a full generation every few
milliseconds, against their one-candidate-at-a-time loop.

Method: (1+lambda) hill climb. Restore the state, replay the recorded
actions to establish the baseline, then repeatedly perturb the incumbent
in a window and keep any candidate that ends the window with more
geodesic progress. Deliberately NOT a policy: no network is involved, so
what it finds is a statement about the map and the physics, not about
the agent.

  python tools/tas_search.py runs/<run>/traj_X.jsonl --ep 9 \
      --t0 3000 --window 300 --iters 40
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from surfgym.core import SurfCore, default_config          # noqa: E402
from surfgym.goalfield import build_goal_field             # noqa: E402
from surfgym.zones import load_zones                       # noqa: E402

# must match src/env.c
YAW_BINS = np.array([-10., -7., -4., -2., -1., -.5, -.25, 0.,
                     .25, .5, 1., 2., 4., 7., 10.])
IN_JUMP, IN_DUCK = 2, 4


def load_episode(path, ep):
    eps, rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r, dict) and "map" in r:
                rows = []
            elif isinstance(r, list):
                rows.append(r)
            elif isinstance(r, dict) and "end" in r:
                if rows:
                    eps.append(np.asarray(rows, dtype=np.float64))
                rows = []
    return eps[ep - 1]


def infer_actions(a):
    """Recover action bins from a recording.

    fwd/side/buttons are stored verbatim; the yaw BIN is not, so it is
    recovered from the realized per-tick yaw delta (wrapped, then matched
    to the nearest bin). Pitch is not reconstructed - it only aims the
    depth camera and does not enter movement (env.c passes pitch 0 to the
    physics), so it cannot affect what this search measures.
    """
    dy = (np.diff(a[:, 7]) + 180.0) % 360.0 - 180.0
    yb = np.abs(dy[:, None] - YAW_BINS[None, :]).argmin(axis=1)
    n = len(dy)
    act = np.zeros((n, 6), np.int32)
    act[:, 0] = yb
    act[:, 1] = 3                                   # pitch bin 3 == 0 deg
    act[:, 2] = a[:-1, 13].astype(np.int32)
    act[:, 3] = a[:-1, 14].astype(np.int32)
    btn = a[:-1, 8].astype(np.int32)
    act[:, 4] = (btn & IN_JUMP) > 0
    act[:, 5] = (btn & IN_DUCK) > 0
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--ep", type=int, default=9)
    ap.add_argument("--t0", type=int, default=3000, help="tick to branch at")
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--p-mutate", type=float, default=0.06)
    ap.add_argument("--map", default="maps/surf_src_cannonball.bsp")
    a = ap.parse_args()

    ep = load_episode(a.traj, a.ep)
    base_all = infer_actions(ep)
    t1 = min(a.t0 + a.window, len(base_all))
    W = t1 - a.t0
    if W < 10:
        raise SystemExit("window too short - is --t0 past the episode end?")

    core = SurfCore(a.map, default_config(num_envs=a.envs, lidar_w=0, lidar_h=0,
                                          sv_maxvelocity=4000.0,
                                          max_episode_ticks=1_000_000))
    core.reset(0)
    zones = load_zones(a.map)
    gf = build_goal_field(core, zones["end"], cell=32, cache_dir="maps",
                          device="cpu")

    # the recorded state at t0, broadcast to every env
    st = core.get_states()
    proto = st[0].copy()
    proto["origin"] = ep[a.t0, 1:4]
    proto["velocity"] = ep[a.t0, 4:7]
    proto["yaw"] = ep[a.t0, 7]
    proto["onground"] = int(ep[a.t0, 9])

    def evaluate(cands):
        """cands (envs, W, 6) -> geodesic distance-to-finish at window end."""
        for i in range(a.envs):
            core.set_state(i, proto)
        for t in range(W):
            core.step(np.ascontiguousarray(cands[:, t, :]))
        return gf.sample(core.states_view["origin"].copy())

    base = base_all[a.t0:t1]
    d_start = float(gf.sample(ep[a.t0:a.t0 + 1, 1:4])[0])
    d_recorded = float(gf.sample(ep[t1:t1 + 1, 1:4])[0])
    print(f"segment ticks {a.t0}..{t1} ({W/100:.2f}s), {a.envs} candidates/gen")
    print(f"  recorded: {d_start:,.0f}u -> {d_recorded:,.0f}u "
          f"({d_start - d_recorded:,.0f}u of progress)")

    # sanity: replaying the inferred actions must reproduce the recording
    rep = evaluate(np.repeat(base[None], a.envs, axis=0))[0]
    err = abs(rep - d_recorded)
    print(f"  replay of inferred actions: {rep:,.0f}u  (error {err:,.0f}u)")
    if err > 2000:
        print("  !! replay does not track the recording - the inferred action"
              " bins are not faithful, so any 'improvement' below is suspect")

    best, best_d = base.copy(), rep
    rng = np.random.default_rng(0)
    for it in range(a.iters):
        cands = np.repeat(best[None], a.envs, axis=0)
        m = rng.random((a.envs, W)) < a.p_mutate
        m[0] = False                                  # keep the incumbent
        ny = int(m.sum())
        if ny:
            idx = np.flatnonzero(m.ravel())
            flat = cands.reshape(-1, 6)
            flat[idx, 0] = rng.integers(0, 15, ny)    # yaw bin
            side = rng.integers(0, 3, ny)
            flat[idx, 3] = side
        d = evaluate(cands)
        k = int(np.argmin(d))
        if d[k] < best_d - 1.0:
            best_d, best = float(d[k]), cands[k].copy()
            print(f"  gen {it:3d}: {best_d:,.0f}u "
                  f"({d_recorded - best_d:+,.0f}u vs recorded)")
    gain = d_recorded - best_d
    print(f"\nbest found: {best_d:,.0f}u vs recorded {d_recorded:,.0f}u "
          f"-> {gain:+,.0f}u")
    print("  (positive = search beat the policy on this segment; at ~2,900 u/s"
          f" that is worth about {gain/2900:+.2f}s if it holds up)")


if __name__ == "__main__":
    main()
