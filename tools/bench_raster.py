#!/usr/bin/env python3
"""bench_raster.py - pinhole RASTERIZER vs the shipped SDF MARCH, same camera.

Answers one question with numbers instead of intuition: at the resolution and
env count this project actually trains at, is rasterizing the map's triangles
cheaper than sphere-tracing the precomputed SDF?

The two renderers are put on identical footing - same pinhole convention,
same fov, same range/near, same depth encoding - so the images can be
differenced pixel for pixel and the timings compared directly.

    python tools/bench_raster.py --map maps/surf_petrus_lite.bsp
    python tools/bench_raster.py --map maps/surf_src_cannonball.bsp --envs 2048

Reports, per renderer: milliseconds per frame over `--iters` timed runs after
a warmup, and what that costs per 100 Hz decision. For the rasterizer it also
reports the CLIPPED fraction, because a bounded-bbox rasterizer that drops
big triangles is fast for the wrong reason and the number is meaningless
without it.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

from surfgym import SurfCore, default_config         # noqa: E402
from surfgym.vision import GpuLidar, pick_cell       # noqa: E402
from surfgym.raster import Rasterizer, tris_from_mesh  # noqa: E402


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3          # ms/frame


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="maps/surf_petrus_lite.bsp")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--range", type=float, default=11500.0)
    ap.add_argument("--near", type=float, default=2000.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--maxbox", type=int, default=8)
    ap.add_argument("--compare-envs", type=int, default=64,
                    help="how many cameras to diff for the accuracy check")
    a = ap.parse_args()

    mp = Path(a.map)
    if not mp.is_absolute():
        mp = ROOT / mp
    stem = mp.stem
    mesh = ROOT / "viewer" / "assets" / f"{stem}.mesh.json"
    if not mesh.exists():
        raise SystemExit(f"no mesh export at {mesh}\n"
                         f"run: python tools/export_map.py {mp}")

    core = SurfCore(str(mp), default_config(num_envs=a.envs, spawn_mode=2,
                                            lidar_w=0, lidar_h=0))
    core.reset(0)
    cell = pick_cell(core)

    tris = tris_from_mesh(mesh)
    print(f"map {stem}: {len(tris):,} triangles, {a.envs:,} cameras, "
          f"{a.w}x{a.h} = {a.w * a.h:,} px/camera, sdf cell {cell:g}")
    print(f"  -> {len(tris) * a.envs:,} triangle-camera pairs vs "
          f"{a.w * a.h * a.envs:,} rays per frame")

    march = GpuLidar(core, a.w, a.h, range_units=a.range, near_range=a.near,
                     cell=cell, device="cuda", pinhole=True)
    rast = Rasterizer(tris, a.w, a.h, range_units=a.range, near_range=a.near,
                      device="cuda", maxbox=a.maxbox)

    sv = core.states_view
    org = torch.as_tensor(np.ascontiguousarray(sv["origin"]),
                          dtype=torch.float32, device="cuda")
    rng = np.random.default_rng(0)
    yaw = torch.as_tensor(rng.uniform(-180, 180, a.envs).astype(np.float32),
                          device="cuda")
    pit = torch.as_tensor(rng.uniform(-20, 10, a.envs).astype(np.float32),
                          device="cuda")
    duck = torch.zeros(a.envs, dtype=torch.int32, device="cuda")

    print("\n--- timing ---")
    ms_m = timeit(lambda: march.render(org, yaw, pit, duck), a.iters)
    ms_r = timeit(lambda: rast.render(org, yaw, pit, duck), a.iters)
    print(f"  SDF march (pinhole)  {ms_m:8.3f} ms/frame")
    print(f"  rasterizer           {ms_r:8.3f} ms/frame   "
          f"(clipped {100 * rast.last_clipped_frac:.3f}% of tri-camera pairs, "
          f"maxbox {a.maxbox})")
    faster = "FASTER" if ms_r < ms_m else "SLOWER"
    print(f"  -> rasterizer is {ms_r / ms_m:.2f}x the march = {faster}")

    print("\n--- accuracy (first "
          f"{min(a.compare_envs, a.envs)} cameras, encoded depth) ---")
    k = min(a.compare_envs, a.envs)
    dm = march.render(org[:k], yaw[:k], pit[:k], duck[:k]).float()
    dr = rast.render(org[:k], yaw[:k], pit[:k], duck[:k]).float()
    d = (dm - dr).abs()
    print(f"  mean |march - raster| {d.mean():.4f}   median "
          f"{d.median():.4f}   p95 {torch.quantile(d.flatten().float(), 0.95):.4f}")
    print(f"  pixels within 0.02    {100 * (d < 0.02).float().mean():.1f}%")
    print(f"  both at max range     {100 * ((dm > 1.24) & (dr > 1.24)).float().mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
