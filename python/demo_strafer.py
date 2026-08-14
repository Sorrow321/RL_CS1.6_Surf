"""ScriptedStrafer demo — the physics smoke test (docs/03).

Creates a 256-env SurfCore, sets waypoints (from --waypoints JSON if given,
else a placeholder straight 2-point line across the map bounds), runs the
sinusoidal strafe-sync bot for 2000 ticks, records env 0 to
runs/strafer.traj.jsonl (drag into viewer/index.html), and prints the max
horizontal speed reached plus progress stats.  If the strafer gains speed on
a ramp, the port breathes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find surfgym
from surfgym import ScriptedStrafer, SurfCore, default_config, record_rollout

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_waypoints(path: str) -> np.ndarray:
    """maps/<map>.waypoints.json: {"map": "...", "points": [[x,y,z], ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = np.asarray(data["points"], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        raise ValueError(f"{path}: points must be (M>=2, 3), got {pts.shape}")
    return pts


def placeholder_waypoints(core: SurfCore) -> np.ndarray:
    """Straight 2-point line across the map bounds (longest horizontal axis,
    other axes at mid) — a placeholder until a real spline is authored in the
    viewer's waypoint editor."""
    mins, maxs = core.map_bounds()
    mid = (mins + maxs) * 0.5
    axis = int(np.argmax((maxs - mins)[:2]))  # longest of x/y
    p0, p1 = mid.copy(), mid.copy()
    p0[axis] = mins[axis] + 64.0
    p1[axis] = maxs[axis] - 64.0
    return np.stack([p0, p1]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bsp", default=str(REPO_ROOT / "maps" / "surf_ski_2.bsp"))
    ap.add_argument("--waypoints", default=None,
                    help="waypoints JSON; default: placeholder straight line")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--period", type=int, default=100,
                    help="strafer oscillation period in ticks")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "strafer.traj.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dll", default=None, help="explicit surfcore.dll path")
    args = ap.parse_args()

    try:
        core = SurfCore(args.bsp, default_config(num_envs=args.envs),
                        dll_path=args.dll)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"demo aborted: {exc}", file=sys.stderr)
        return 2

    try:
        if args.waypoints:
            pts = load_waypoints(args.waypoints)
            src = args.waypoints
        else:
            pts = placeholder_waypoints(core)
            src = "placeholder straight line (author a real spline in the viewer)"
        core.set_waypoints(pts)
        seg = np.diff(pts, axis=0)
        track_len = float(np.sqrt((seg * seg).sum(axis=1)).sum())
        print(f"map: {args.bsp}")
        print(f"waypoints: {len(pts)} points, {track_len:.0f} u  [{src}]")
        print(f"envs: {core.num_envs}, obs_dim: {core.obs_dim}, "
              f"ticks: {args.ticks}")

        stats = {"max_hspeed": 0.0, "best_progress": 0.0, "final_states": None}

        def on_tick(t, states, rewards, done, trunc):
            v = states["velocity"]
            hs = float(np.hypot(v[:, 0], v[:, 1]).max())
            if hs > stats["max_hspeed"]:
                stats["max_hspeed"] = hs
            bp = float(states["best_progress"].max())
            if bp > stats["best_progress"]:
                stats["best_progress"] = bp
            stats["final_states"] = states  # get_states() copies; safe to keep

        policy = ScriptedStrafer(core, period_ticks=args.period)
        summaries = record_rollout(
            core, policy, args.out,
            episodes=None, max_ticks=args.ticks,
            seed=args.seed, on_tick=on_tick,
        )

        print()
        print(f"max horizontal speed (all envs): {stats['max_hspeed']:.1f} u/s")
        print(f"best progress (all envs):        {stats['best_progress']:.1f} u "
              f"({100.0 * stats['best_progress'] / max(track_len, 1e-9):.1f}% "
              "of track)")
        fs = stats["final_states"]
        if fs is not None:
            print(f"mean best_progress (final tick): "
                  f"{float(fs['best_progress'].mean()):.1f} u")
        print(f"env-0 episodes recorded: {len(summaries)}")
        for k, s in enumerate(summaries):
            print(f"  ep {k}: end={s['end']} ticks={s['ticks']} "
                  f"best_progress={s['best_progress']:.1f} "
                  f"return={s['ep_return']:.2f}")
        print(f"trajectory: {args.out}  (drag into viewer/index.html)")
        return 0
    finally:
        core.close()


if __name__ == "__main__":
    sys.exit(main())
