#!/usr/bin/env python3
"""raster_compare.py - the SDF march and the triangle rasterizer, same camera.

Renders a trajectory twice from identical poses and stacks the two images:

    top    = SDF march (pinhole)   - what the policy sees today
    bottom = triangle rasterizer   - the candidate replacement

Both use the same pinhole convention, fov, range and depth encoding, so any
difference is the renderer. Measured statistically the rasterizer is never
NEARER than the march and often farther, which means missing surfaces; this
is the tool for seeing WHICH surfaces.

    python tools/raster_compare.py runs/xMM/traj_0604766208_cannonball.jsonl
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import cv2                                            # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

from surfgym import SurfCore, default_config          # noqa: E402
from surfgym.vision import GpuLidar, pick_cell        # noqa: E402
from surfgym.raster import Rasterizer, tris_from_mesh  # noqa: E402


def episodes(path):
    rows, hdr = [], None
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if i == 0:
                hdr = json.loads(line)
                continue
            rows.append(json.loads(line))
    return hdr, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj")
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from surfgym.route import episodes_from_traj
    with open(a.traj, encoding="utf-8") as fh:
        hdr = json.loads(fh.readline())
    eps = episodes_from_traj(a.traj)
    rows = np.concatenate(eps, axis=0)
    stem = (hdr or {}).get("map") or "surf_src_cannonball"
    bsp = ROOT / "maps" / f"{stem}.bsp"
    mesh = ROOT / "viewer" / "assets" / f"{stem}.mesh.json"
    print(f"{stem}: {len(rows):,} ticks")

    core = SurfCore(str(bsp), default_config(num_envs=1, spawn_mode=2,
                                             lidar_w=0, lidar_h=0))
    core.reset(0)
    cell = pick_cell(core)
    march = GpuLidar(core, a.w, a.h, range_units=11500.0, near_range=2000.0,
                     cell=cell, device="cuda", pinhole=True)
    tris = tris_from_mesh(mesh, brushes=True)
    rast = Rasterizer(tris, a.w, a.h, range_units=11500.0, near_range=2000.0,
                      device="cuda", exact=True)
    print(f"march cell {cell:g}  |  raster {len(tris):,} triangles")

    # rows are flat lists: [tick, x,y,z, vx,vy,vz, yaw, buttons, ..., pitch, ...]
    arr = np.asarray(rows, dtype=np.float64)
    n = min(len(arr), a.max_frames)
    arr = arr[:n]
    pos = arr[:, 1:4].astype(np.float32)
    yaw = arr[:, 7].astype(np.float32) if arr.shape[1] > 7 else np.zeros(n, np.float32)
    pit = arr[:, 12].astype(np.float32) if arr.shape[1] > 12 else np.zeros(n, np.float32)

    HFOV, VFOV = 120.0, 90.0
    d2r = np.pi / 180.0
    aspect = ((np.tan(VFOV / 2 * d2r) / a.h) / (np.tan(HFOV / 2 * d2r) / a.w))
    W = a.w * a.scale
    H = int(round(a.h * a.scale * aspect))
    out = Path(a.out or Path(a.traj).with_suffix(".rastercmp.mp4"))
    ff = shutil.which("ffmpeg")
    enc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H * 2}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
         "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    B = 256
    for s0 in range(0, n, B):
        sl = slice(s0, min(s0 + B, n))
        o = torch.tensor(pos[sl], dtype=torch.float32, device="cuda")
        y = torch.tensor(yaw[sl], dtype=torch.float32, device="cuda")
        p = torch.tensor(pit[sl], dtype=torch.float32, device="cuda")
        d = torch.zeros(len(o), dtype=torch.int32, device="cuda")
        dm = march.render(o, y, p, d).cpu().numpy()
        dr = rast.render(o, y, p, d).cpu().numpy()
        for i in range(len(o)):
            top = cv2.applyColorMap(
                (np.clip(1.0 - dm[i] / 1.25, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO)
            bot = cv2.applyColorMap(
                (np.clip(1.0 - dr[i] / 1.25, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO)
            top = cv2.resize(top, (W, H), interpolation=cv2.INTER_NEAREST)
            bot = cv2.resize(bot, (W, H), interpolation=cv2.INTER_NEAREST)
            cv2.putText(top, "SDF march (current)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(bot, f"rasterizer ({len(tris):,} tris)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.line(bot, (0, 0), (W, 0), (60, 60, 60), 1)
            enc.stdin.write(np.ascontiguousarray(
                np.vstack((top, bot))).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {n} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
