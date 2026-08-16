#!/usr/bin/env python3
"""bench_lidar.py — how many march steps do the depth rays actually take?

S7 (hierarchical SDF march) is premised on the audit's hypothesis that in the
maxvel-4000 regime the agent is high over open air, sky rays never hit, and
every one of them burns all 64 sphere-trace steps. That premise is worth
checking before spending two days on a two-level kernel: the fine march is
already adaptive (steps are 0.9 * SDF), so in genuinely open space it should
leave in a handful of long strides.

The production kernel has no early exit — `for _ in range(MAX_STEPS)` always
runs, with dead lanes merely masked — so the loop trip count is paid by every
ray in the block regardless. That makes two numbers decisive:

  * the distribution of steps-to-hit per ray (how much of the loop is real
    work vs masked no-ops), and
  * render time as a function of MAX_STEPS (flat => memory-bound on the live
    lanes; linear => the trip count itself is the cost).

Poses come from the checkpoint's own respawn reservoir, so they are real
mid-run states in the regime being optimised, not spawn points.

    python3 tools/bench_lidar.py --ckpt runs_ckpt.pt
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import triton                                            # noqa: E402
import triton.language as tl                             # noqa: E402

from surfgym import SurfCore, default_config             # noqa: E402
from surfgym.vision import GpuLidar, pick_cell           # noqa: E402


@triton.jit
def _count_kernel(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, steps_ptr, hit_ptr,
                  sdf_ptr, yoff_ptr, poff_ptr,
                  total, HW, W,
                  nx, ny, nz, stride_z, stride_y,
                  mnx, mny, mnz, inv_cell, cell, rng,
                  MAX_STEPS: tl.constexpr, BLOCK: tl.constexpr):
    """surfgym.vision._march_kernel, instrumented: per-ray step count + hit."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    n = offs // HW
    pix = offs % HW
    r = pix // W
    c = pix % W
    ex = tl.load(eye_ptr + n * 3 + 0, mask=m, other=0.0)
    ey = tl.load(eye_ptr + n * 3 + 1, mask=m, other=0.0)
    ez = tl.load(eye_ptr + n * 3 + 2, mask=m, other=0.0)
    dk = tl.load(duck_ptr + n, mask=m, other=0)
    ez += tl.where(dk != 0, 12.0, 17.0)
    yw = tl.load(yaw_ptr + n, mask=m, other=0.0) + tl.load(yoff_ptr + c, mask=m, other=0.0)
    pt = tl.load(pitch_ptr + n, mask=m, other=0.0) + tl.load(poff_ptr + r, mask=m, other=0.0)
    cp = tl.cos(pt)
    dx = cp * tl.cos(yw)
    dy = cp * tl.sin(yw)
    dz = tl.sin(pt)
    t = tl.zeros([BLOCK], tl.float32)
    nst = tl.zeros([BLOCK], tl.int32)
    alive = m
    hit_eps = 0.6 * cell
    min_step = 0.3 * cell
    for _ in range(MAX_STEPS):
        px = ex + dx * t
        py = ey + dy * t
        pz = ez + dz * t
        ix = tl.minimum(tl.maximum(((px - mnx) * inv_cell).to(tl.int64), 0), nx - 1)
        iy = tl.minimum(tl.maximum(((py - mny) * inv_cell).to(tl.int64), 0), ny - 1)
        iz = tl.minimum(tl.maximum(((pz - mnz) * inv_cell).to(tl.int64), 0), nz - 1)
        d = tl.load(sdf_ptr + iz * stride_z + iy * stride_y + ix,
                    mask=alive, other=0.0).to(tl.float32)
        alive = alive & (d > hit_eps) & (t < rng)
        nst += tl.where(alive, 1, 0)
        t += tl.where(alive, tl.maximum(d * 0.9, min_step), 0.0)
    tl.store(steps_ptr + offs, nst, mask=m)
    tl.store(hit_ptr + offs, tl.where(t < rng, 1, 0), mask=m)


def poses_from_ckpt(path: Path, n: int, device):
    """Real mid-run states out of the checkpoint's respawn reservoir."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = (ck.get("respawn") or {}).get("states")
    if st is None or not len(st):
        return None, (ck.get("config") or {})
    st = np.asarray(st)
    idx = np.random.default_rng(0).integers(0, len(st), n)
    s = st[idx]
    return (torch.as_tensor(np.ascontiguousarray(s["origin"]), dtype=torch.float32,
                            device=device),
            torch.as_tensor(s["yaw"].copy(), dtype=torch.float32, device=device),
            torch.as_tensor(s["pitch"].copy(), dtype=torch.float32, device=device),
            torch.as_tensor(s["ducked"].copy().astype(np.int32), device=device),
            ck.get("config") or {})


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs_ckpt.pt"))
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps-sweep", default="8,16,24,32,48,64")
    args = ap.parse_args()
    dev = torch.device("cuda")

    got = poses_from_ckpt(Path(args.ckpt), args.envs, dev)
    if got[0] is None:
        raise SystemExit("checkpoint carries no respawn reservoir to sample poses from")
    origin, yaw, pitch, duck, cfg = got
    mp = ROOT / "maps" / f"{cfg.get('map', 'surf_src_cannonball')}.bsp"
    print(f"map {mp.stem}  envs {args.envs}  "
          f"lidar {cfg.get('lidar_w')}x{cfg.get('lidar_h')} "
          f"range {cfg.get('lidar_range')} near {cfg.get('lidar_near')} "
          f"cell {cfg.get('lidar_cell')}")

    core = SurfCore(str(mp), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    lid = GpuLidar(core, int(cfg.get("lidar_w") or 128), int(cfg.get("lidar_h") or 64),
                   range_units=float(cfg.get("lidar_range") or 2000.0),
                   near_range=cfg.get("lidar_near"),
                   cell=float(cfg.get("lidar_cell") or pick_cell(core)),
                   device=dev)
    print(f"sdf grid {lid.nz}x{lid.ny}x{lid.nx} = "
          f"{lid.nz * lid.ny * lid.nx / 1e6:.0f}M voxels, "
          f"{lid.sdf_flat.numel() * 2 / 1e9:.2f} GB f16")

    base = timed(lambda: lid.render(origin, yaw, pitch, duck))
    rays = args.envs * lid.H * lid.W
    print(f"\nrender: {base:.2f} ms for {rays / 1e6:.1f}M rays "
          f"= {rays / base * 1e-6:.0f} Grays/s")

    print("\n-- render ms vs MAX_STEPS --")
    keep = lid.max_steps
    for s in [int(v) for v in args.steps_sweep.split(",")]:
        lid.max_steps = s
        t = timed(lambda: lid.render(origin, yaw, pitch, duck))
        print(f"  max_steps {s:3d}   {t:6.2f} ms   {base / t:5.2f}x vs {keep}")
    lid.max_steps = keep

    # -- per-ray step census -------------------------------------------------
    total = args.envs * lid.H * lid.W
    steps = torch.empty(total, dtype=torch.int32, device=dev)
    hit = torch.empty(total, dtype=torch.int32, device=dev)
    d2r = float(np.pi / 180.0)
    BLOCK = 256
    _count_kernel[(triton.cdiv(total, BLOCK),)](
        origin.contiguous(), (yaw * d2r).contiguous(), (pitch * d2r).contiguous(),
        duck.to(torch.int32).contiguous(), steps, hit,
        lid.sdf_flat, lid.yoff, lid.poff,
        total, lid.H * lid.W, lid.W,
        lid.nx, lid.ny, lid.nz, lid.stride_z, lid.stride_y,
        lid.mins_f[0], lid.mins_f[1], lid.mins_f[2],
        1.0 / lid.cell, lid.cell, lid.range,
        MAX_STEPS=keep, BLOCK=BLOCK)
    s = steps.float()
    q = torch.quantile(s, torch.tensor([0.5, 0.9, 0.99], device=dev)).tolist()
    print(f"\n-- per-ray march steps (MAX_STEPS={keep}) --")
    print(f"  mean {s.mean():.2f}  median {q[0]:.0f}  p90 {q[1]:.0f}  "
          f"p99 {q[2]:.0f}  max {int(s.max())}")
    print(f"  rays that exhaust the loop: "
          f"{100.0 * (steps >= keep).float().mean():.2f}%")
    print(f"  rays that never hit (reach range): "
          f"{100.0 * (1 - hit.float().mean()):.2f}%")
    h = torch.histc(s, bins=8, min=0, max=keep).tolist()
    w = keep / 8
    for i, c in enumerate(h):
        print(f"    steps {int(i * w):3d}-{int((i + 1) * w):3d}: "
              f"{100.0 * c / total:5.1f}%")
    print(f"\n  loop trips actually doing work: "
          f"{100.0 * s.mean() / keep:.1f}% "
          f"(the rest are masked no-op iterations)")


if __name__ == "__main__":
    main()
