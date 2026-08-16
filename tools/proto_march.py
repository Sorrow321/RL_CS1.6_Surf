#!/usr/bin/env python3
"""proto_march.py — candidate SDF-march kernels, A/B'd and pixel-verified.

The census (tools/bench_lidar.py) refuted S7's premise and replaced it with
two cheaper targets:

  1. **No early exit.** `for _ in range(MAX_STEPS)` is a constexpr loop with
     no break, so every block pays all 64 trips even though the mean ray
     finishes in 9.9 and only 0.12% exhaust the loop. A block-wide "is any
     lane still alive?" test lets a block cost its own max instead — 1.9x
     fewer trips at BLOCK=256, 2.6x at BLOCK=64.
  2. **An unroll cliff.** Render time is 1.65 ms at MAX_STEPS=32 but 4.90 ms
     at 48 — 1.5x the trips for 3x the time. A constexpr trip count gets
     fully unrolled; past ~32 the register pressure costs more than the
     work. A runtime-bounded `while` stops the unrolling.

Both are exactly equivalent to the current kernel, not approximations: once
a lane's `alive` goes false it stays false and its `t` stops advancing, so
stopping the loop early cannot change any output value. That matters — the
depth encoding is warm-start ABI. This script asserts bit-exactness against
the production kernel on real poses before reporting any speed.

    python3 tools/proto_march.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym import SurfCore, default_config             # noqa: E402
from surfgym.vision import GpuLidar, pick_cell           # noqa: E402

from bench_lidar import poses_from_ckpt, timed           # noqa: E402


@triton.jit
def _march_ee(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, out_ptr,
              sdf_ptr, yoff_ptr, poff_ptr,
              total, HW, W,
              nx, ny, nz, stride_z, stride_y,
              mnx, mny, mnz, inv_cell, cell, rng, near,
              max_steps, BLOCK: tl.constexpr):
    """Production march + a block-wide early exit, and max_steps passed as a
    RUNTIME argument so the loop is not unrolled."""
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
    alive = m
    hit_eps = 0.6 * cell
    min_step = 0.3 * cell
    k = 0
    while k < max_steps and tl.max(alive.to(tl.int32)) > 0:
        px = ex + dx * t
        py = ey + dy * t
        pz = ez + dz * t
        ix = tl.minimum(tl.maximum(((px - mnx) * inv_cell).to(tl.int64), 0), nx - 1)
        iy = tl.minimum(tl.maximum(((py - mny) * inv_cell).to(tl.int64), 0), ny - 1)
        iz = tl.minimum(tl.maximum(((pz - mnz) * inv_cell).to(tl.int64), 0), nz - 1)
        d = tl.load(sdf_ptr + iz * stride_z + iy * stride_y + ix,
                    mask=alive, other=0.0).to(tl.float32)
        alive = alive & (d > hit_eps) & (t < rng)
        t += tl.where(alive, tl.maximum(d * 0.9, min_step), 0.0)
        k += 1
    t = tl.minimum(t, rng)
    enc = tl.minimum(t, near) / near \
        + 0.25 * (1.0 - tl.exp(-tl.maximum(t - near, 0.0) / 2500.0))
    tl.store(out_ptr + offs, enc, mask=m)


@triton.jit
def _march_noexit_runtime(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, out_ptr,
                          sdf_ptr, yoff_ptr, poff_ptr,
                          total, HW, W,
                          nx, ny, nz, stride_z, stride_y,
                          mnx, mny, mnz, inv_cell, cell, rng, near,
                          max_steps, BLOCK: tl.constexpr):
    """Runtime trip count but NO early exit — isolates the unroll effect."""
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
    alive = m
    hit_eps = 0.6 * cell
    min_step = 0.3 * cell
    k = 0
    while k < max_steps:
        px = ex + dx * t
        py = ey + dy * t
        pz = ez + dz * t
        ix = tl.minimum(tl.maximum(((px - mnx) * inv_cell).to(tl.int64), 0), nx - 1)
        iy = tl.minimum(tl.maximum(((py - mny) * inv_cell).to(tl.int64), 0), ny - 1)
        iz = tl.minimum(tl.maximum(((pz - mnz) * inv_cell).to(tl.int64), 0), nz - 1)
        d = tl.load(sdf_ptr + iz * stride_z + iy * stride_y + ix,
                    mask=alive, other=0.0).to(tl.float32)
        alive = alive & (d > hit_eps) & (t < rng)
        t += tl.where(alive, tl.maximum(d * 0.9, min_step), 0.0)
        k += 1
    t = tl.minimum(t, rng)
    enc = tl.minimum(t, near) / near \
        + 0.25 * (1.0 - tl.exp(-tl.maximum(t - near, 0.0) / 2500.0))
    tl.store(out_ptr + offs, enc, mask=m)


def run(kernel, lid, origin, yaw, pitch, duck, block, warps):
    N = origin.shape[0]
    d2r = float(np.pi / 180.0)
    out = torch.empty(N, lid.H, lid.W, device=lid.device)
    total = N * lid.H * lid.W
    kernel[(triton.cdiv(total, block),)](
        origin.contiguous(), (yaw * d2r).contiguous(), (pitch * d2r).contiguous(),
        duck.to(torch.int32).contiguous(),
        out, lid.sdf_flat, lid.yoff, lid.poff,
        total, lid.H * lid.W, lid.W,
        lid.nx, lid.ny, lid.nz, lid.stride_z, lid.stride_y,
        lid.mins_f[0], lid.mins_f[1], lid.mins_f[2],
        1.0 / lid.cell, lid.cell, lid.range, lid.near,
        lid.max_steps, BLOCK=block, num_warps=warps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs_ckpt.pt"))
    ap.add_argument("--envs", type=int, default=2048)
    args = ap.parse_args()
    dev = torch.device("cuda")

    origin, yaw, pitch, duck, cfg = poses_from_ckpt(Path(args.ckpt), args.envs, dev)
    mp = ROOT / "maps" / f"{cfg.get('map', 'surf_src_cannonball')}.bsp"
    core = SurfCore(str(mp), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    lid = GpuLidar(core, int(cfg.get("lidar_w") or 128), int(cfg.get("lidar_h") or 64),
                   range_units=float(cfg.get("lidar_range") or 2000.0),
                   near_range=cfg.get("lidar_near"),
                   cell=float(cfg.get("lidar_cell") or pick_cell(core)),
                   device=dev)

    ref = lid.render(origin, yaw, pitch, duck)
    base = timed(lambda: lid.render(origin, yaw, pitch, duck))
    print(f"production kernel: {base:.2f} ms  (BLOCK=256, constexpr loop)\n")

    print(f"{'kernel':<22}{'BLOCK':>6}{'warps':>6}{'ms':>9}{'x':>7}   max|diff|")
    print("-" * 62)
    for name, kern in (("early-exit", _march_ee),
                       ("runtime-loop only", _march_noexit_runtime)):
        for block in (64, 128, 256, 512):
            for warps in (2, 4, 8):
                try:
                    out = run(kern, lid, origin, yaw, pitch, duck, block, warps)
                    diff = (out - ref).abs().max().item()
                    t = timed(lambda k=kern, b=block, w=warps:
                              run(k, lid, origin, yaw, pitch, duck, b, w))
                    flag = "" if diff == 0.0 else "  <-- NOT BIT-EXACT"
                    print(f"{name:<22}{block:>6}{warps:>6}{t:>9.2f}"
                          f"{base / t:>7.2f}   {diff:.3e}{flag}")
                except Exception as exc:
                    print(f"{name:<22}{block:>6}{warps:>6}   FAILED  {exc!r}")


if __name__ == "__main__":
    main()
