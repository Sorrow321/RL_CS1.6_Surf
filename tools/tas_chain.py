"""tas_chain.py - optimize a whole recorded run, window by window.

tas_search.py answers "is this segment improvable?" one segment at a
time, each from the ORIGINAL state. That cannot produce a faster run,
because segment gains do not compose: a better exit from window k
changes the state window k+1 starts from.

This chains them. Walk the run in windows; for each, search mutations of
the recorded actions starting from the best state reached so far, keep
the winner, advance, repeat. That is how a TAS is actually built, and it
is the honest test of the value-ceiling claim that the champion's line
is worth ~72-74 s: if chaining cannot get near that, the estimate was
optimistic.

Scoring is geodesic distance-to-finish at the window end, so each window
is greedy about progress rather than about final time. Greedy is not
optimal - a window that ends slightly further back but much faster can
win later - so treat the result as a lower bound on what search can do.

  python tools/tas_chain.py runs/<run>/traj_X.jsonl --ep 9 \
      --window 150 --iters 25 --envs 1024
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
from tas_search import load_episode, infer_actions         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--ep", type=int, default=9)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--p-mutate", type=float, default=0.05)
    ap.add_argument("--map", default="maps/surf_src_cannonball.bsp")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = whole run")
    a = ap.parse_args()

    ep = load_episode(a.traj, a.ep)
    base_all = infer_actions(ep)
    # the run clock starts at the opening cliff drop, same rule as play.py
    z0 = ep[0, 3]
    drop = np.flatnonzero(ep[:, 3] < z0 - 100.0)
    t_start = int(drop[0]) if len(drop) else 0

    core = SurfCore(a.map, default_config(num_envs=a.envs, lidar_w=0, lidar_h=0,
                                          sv_maxvelocity=4000.0,
                                          max_episode_ticks=1_000_000))
    core.reset(0)
    zones = load_zones(a.map)
    gf = build_goal_field(core, zones["end"], cell=32, cache_dir="maps",
                          device="cpu")

    st = core.get_states()
    cur = st[0].copy()
    cur["origin"] = ep[t_start, 1:4]
    cur["velocity"] = ep[t_start, 4:7]
    cur["yaw"] = ep[t_start, 7]
    cur["onground"] = int(ep[t_start, 9])

    n_windows = (len(base_all) - t_start) // a.window
    if a.max_windows:
        n_windows = min(n_windows, a.max_windows)
    rng = np.random.default_rng(0)
    total_ticks = 0
    chained = []
    print(f"chaining {n_windows} windows of {a.window} ticks "
          f"({a.envs} candidates x {a.iters} gens each)")
    print(f"  recorded run: {len(ep) - t_start} ticks from the clock start "
          f"= {(len(ep) - t_start) / 100:.2f}s\n")

    for w in range(n_windows):
        lo = t_start + w * a.window
        hi = lo + a.window
        base = base_all[lo:hi]
        best, best_d = base.copy(), None
        for it in range(a.iters + 1):
            cands = np.repeat(best[None], a.envs, axis=0)
            if it > 0:
                m = rng.random((a.envs, a.window)) < a.p_mutate
                m[0] = False
                idx = np.flatnonzero(m.ravel())
                flat = cands.reshape(-1, 6)
                flat[idx, 0] = rng.integers(0, 15, idx.size)
                flat[idx, 3] = rng.integers(0, 3, idx.size)
            for i in range(a.envs):
                core.set_state(i, cur)
            for t in range(a.window):
                core.step(np.ascontiguousarray(cands[:, t, :]))
            d = gf.sample(core.states_view["origin"].copy())
            k = int(np.argmin(d))
            if best_d is None or d[k] < best_d:
                best_d, best = float(d[k]), cands[k].copy()
        # replay the winner once to advance the chain state
        for i in range(1):
            core.set_state(i, cur)
        for t in range(a.window):
            core.step(np.ascontiguousarray(
                np.repeat(best[t][None], a.envs, axis=0)))
        cur = core.states_view[0].copy()
        chained.append(best)
        total_ticks += a.window
        rec_d = float(gf.sample(ep[hi:hi + 1, 1:4])[0]) if hi < len(ep) else 0.0
        print(f"  window {w:3d} [{lo:5d}-{hi:5d}]  chained {best_d:9,.0f}u   "
              f"recorded {rec_d:9,.0f}u   {rec_d - best_d:+8,.0f}u")
        if best_d < 200.0:
            print(f"\n  *** reached the finish zone after {total_ticks} ticks "
                  f"= {total_ticks / 100:.2f}s (recorded "
                  f"{(len(ep) - t_start) / 100:.2f}s) ***")
            break

    out = Path(a.traj).with_suffix(".chained.npy")
    np.save(out, np.concatenate(chained, axis=0))
    print(f"\nchained action sequence saved to {out}")


if __name__ == "__main__":
    main()
