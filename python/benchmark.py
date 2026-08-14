"""Throughput benchmark for surfcore.dll via the surfgym binding.

Usage (from C:\\RL_Surf\\python):

    python benchmark.py --bsp ../maps/surf_ski_2.bsp --envs 1,64,1024 --ticks 1000

Random int32 action batches are precomputed outside the timed loop; timing
uses time.perf_counter with a warmup (default 50 steps).  surfcore steps envs
with OpenMP internally — control the core count with OMP_NUM_THREADS.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find surfgym
from surfgym import ACTION_DIM, ACTION_NVEC, SurfCore, default_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def bench_one(
    bsp: str,
    num_envs: int,
    ticks: int,
    warmup: int,
    seed: int,
    dll_path: str | None,
) -> float:
    """Returns wall seconds for `ticks` timed surf_step calls at `num_envs`."""
    core = SurfCore(bsp, default_config(num_envs=num_envs), dll_path=dll_path)
    try:
        rng = np.random.default_rng(seed)
        nvec = np.asarray(ACTION_NVEC, dtype=np.int32)
        # Precompute every action batch; slices along axis 0 are C-contiguous.
        acts = rng.integers(0, nvec, size=(ticks, num_envs, ACTION_DIM),
                            dtype=np.int32)
        wacts = rng.integers(0, nvec, size=(warmup, num_envs, ACTION_DIM),
                             dtype=np.int32)
        core.reset(seed)
        for t in range(warmup):
            core.step(wacts[t])
        t0 = time.perf_counter()
        for t in range(ticks):
            core.step(acts[t])
        return time.perf_counter() - t0
    finally:
        core.close()


def fmt_rate(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bsp", default=str(REPO_ROOT / "maps" / "surf_ski_2.bsp"),
                    help="path to the .bsp map")
    ap.add_argument("--envs", default="1,64,1024",
                    help="comma-separated batch sizes to benchmark")
    ap.add_argument("--ticks", type=int, default=1000,
                    help="timed steps per batch size")
    ap.add_argument("--warmup", type=int, default=50,
                    help="untimed warmup steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dll", default=None, help="explicit surfcore.dll path")
    ap.add_argument("--threads-note", nargs="?", const="", default=None,
                    help="annotate the run (thread setup etc.); printed with "
                         "the OMP_NUM_THREADS status line")
    args = ap.parse_args()

    env_counts = [int(x) for x in str(args.envs).split(",") if x.strip()]
    omp = os.environ.get("OMP_NUM_THREADS")
    print(f"map: {args.bsp}")
    print(f"threads: OMP_NUM_THREADS={omp if omp else '<unset>'} "
          "(surfcore parallelizes over envs with OpenMP)")
    if args.threads_note:
        print(f"note: {args.threads_note}")
    print(f"ticks per run: {args.ticks} (+{args.warmup} warmup), "
          f"numpy {np.__version__}")
    print()

    header = f"{'num_envs':>9} {'wall s':>9} {'calls/s':>9} {'env steps/s':>12}"
    print(header)
    print("-" * len(header))
    for n in env_counts:
        try:
            wall = bench_one(args.bsp, n, args.ticks, args.warmup, args.seed,
                             args.dll)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\nbenchmark aborted at num_envs={n}: {exc}",
                  file=sys.stderr)
            return 2
        calls_s = args.ticks / wall
        steps_s = n * args.ticks / wall
        print(f"{n:>9} {wall:>9.3f} {fmt_rate(calls_s):>9} "
              f"{fmt_rate(steps_s):>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
