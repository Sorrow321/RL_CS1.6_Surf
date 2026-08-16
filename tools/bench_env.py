#!/usr/bin/env python3
"""bench_env.py — how fast is one batched core.step, and at how many threads?

`surf_step` is `#pragma omp parallel for` over the envs (src/env.c:503), so it
forks a thread team on every call — 384 times per training iteration. The team
defaults to the core count, which is right at 16 cores and absurd at 192: 2048
envs over 192 threads is ~10 envs each, so the fork/join and the barrier cost
more than the physics.

The phase timer puts `env` at 6.8% of a baseline iteration on a 16-core box,
and the first runs on a 192-core box came back 2.3x worse, which is what
prompted this. OMP_NUM_THREADS is an environment knob — no rebuild needed —
but OpenMP reads it when the .so loads, so it must be set before surfgym is
imported. This script re-execs itself per setting to get that right.

    python3 tools/bench_env.py                      # sweep 1,2,4,...,nproc
    OMP_NUM_THREADS=8 python3 tools/bench_env.py --once
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def measure(envs: int, k: int, iters: int, warmup: int) -> tuple[float, float]:
    sys.path.insert(0, str(ROOT / "python"))
    import numpy as np
    from surfgym import SurfCore, default_config
    core = SurfCore(str(ROOT / "maps" / "surf_src_cannonball.bsp"),
                    default_config(num_envs=envs, spawn_mode=2,
                                   max_episode_ticks=12000, water_fail=1,
                                   sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0,
                                   pitch_rate_max_deg=1.33))
    core.reset(0)
    act = np.zeros((envs, 6), np.int32)
    act[:, 0] = 7
    act[:, 1] = 3
    act[:, 2] = act[:, 3] = 1
    for _ in range(warmup):
        core.step(act)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        core.step(act)
        ts.append((time.perf_counter() - t0) * 1e3)
    ms = statistics.median(ts)
    return ms, envs / (ms / 1e3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--once", action="store_true",
                    help="measure with the inherited OMP_NUM_THREADS only")
    ap.add_argument("--threads", default=None,
                    help="comma list to sweep (default: 1,2,4,... up to nproc)")
    args = ap.parse_args()

    if args.once:
        ms, rate = measure(args.envs, 3, args.iters, args.warmup)
        print(f"{os.environ.get('OMP_NUM_THREADS', 'default'):>8}  "
              f"{ms:7.3f} ms/step  {rate / 1e6:6.2f}M env-ticks/s")
        return

    ncpu = os.cpu_count() or 8
    if args.threads:
        sweep = [int(v) for v in args.threads.split(",")]
    else:
        sweep, t = [], 1
        while t <= ncpu:
            sweep.append(t)
            t *= 2
        if ncpu not in sweep:
            sweep.append(ncpu)
    print(f"{ncpu} cores, {args.envs} envs, median of {args.iters} steps\n")
    print(f"{'threads':>8}  {'ms/step':>10}  {'env-ticks/s':>14}")
    print("-" * 36)
    best = None
    for t in sweep:
        env = dict(os.environ, OMP_NUM_THREADS=str(t))
        out = subprocess.run([sys.executable, __file__, "--once",
                              "--envs", str(args.envs), "--iters", str(args.iters),
                              "--warmup", str(args.warmup)],
                             env=env, capture_output=True, text=True)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if not line:
            print(f"{t:>8}  FAILED: {out.stderr.strip().splitlines()[-1:]}")
            continue
        ms = float(line.split()[1])
        print(f"{t:>8}  {ms:>10.3f}  {args.envs / (ms / 1e3) / 1e6:>11.2f}M")
        if best is None or ms < best[1]:
            best = (t, ms)
    if best:
        print(f"\nbest: OMP_NUM_THREADS={best[0]} at {best[1]:.3f} ms/step")
        print(f"per training iteration (384 steps): {384 * best[1]:.0f} ms")


if __name__ == "__main__":
    main()
