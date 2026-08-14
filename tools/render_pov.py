"""Render what the agent SEES along a recorded trajectory — a first-person
depth-vision video (the exact 128x64 lidar image the policy's CNN receives,
re-rendered from the trajectory's poses via the map SDF).

    python tools\render_pov.py runs\eyes\traj_0100000000.jsonl
    python tools\render_pov.py runs\eyes\traj_X.jsonl --scale 8 --fps 50

Output: <traj>.pov.mp4 next to the input (near = bright/warm, far = dark;
overlay: tick, h-speed, view pitch). Trajectory rows carry [t, x,y,z,
vx,vy,vz, yaw, buttons, onground, progress, reward, pitch]; recordings from
before the pitch column render with pitch 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "python"))

import cv2
import numpy as np
import torch

from surfgym import SurfCore, default_config
from surfgym.vision import GpuLidar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_ski_2.bsp"))
    ap.add_argument("--w", type=int, default=128, help="lidar width (match training)")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--scale", type=int, default=6, help="upscale factor")
    ap.add_argument("--fps", type=int, default=100, help="100 = real-time (10ms ticks)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, episodes = [], []
    for line in open(args.traj, encoding="utf-8"):
        o = json.loads(line)
        if isinstance(o, dict) and "map" in o:
            rows = []
        elif isinstance(o, list):
            rows.append(o)
        elif isinstance(o, dict) and "end" in o and rows:
            episodes.append(np.asarray(rows, dtype=np.float64))
            rows = []
    if not episodes:
        raise SystemExit("no episodes in trajectory file")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = SurfCore(args.map, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    lidar = GpuLidar(core, args.w, args.h, device=device)

    out_path = Path(args.out) if args.out else Path(args.traj).with_suffix(".pov.mp4")
    W, H = args.w * args.scale, args.h * args.scale

    # system ffmpeg (libx264 ultrafast) is ~5x faster than cv2's mp4v writer
    # and makes browser-playable files; fall back to cv2 if it's missing
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff:
        enc = subprocess.Popen(
            [ff, "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(args.fps),
             "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "23", "-pix_fmt", "yuv420p", str(out_path)],
            stdin=subprocess.PIPE)
        write = enc.stdin.write

        def close():
            enc.stdin.close()
            enc.wait()
    else:
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))
        write = lambda buf: vw.write(np.frombuffer(buf, np.uint8).reshape(H, W, 3))

        def close():
            vw.release()

    B = 512                                      # render B poses per GPU batch
    total = 0
    for ei, a in enumerate(episodes):
        n = len(a)
        pitch = a[:, 12] if a.shape[1] > 12 else np.zeros(n)
        duck = (a[:, 8].astype(np.int64) & 4) != 0     # buttons IN_DUCK bit
        for s0 in range(0, n, B):
            sl = slice(s0, min(s0 + B, n))
            k = sl.stop - sl.start
            o = torch.tensor(a[sl, 1:4], dtype=torch.float32, device=device)
            yw = torch.tensor(a[sl, 7], dtype=torch.float32, device=device)
            pt = torch.tensor(pitch[sl], dtype=torch.float32, device=device)
            dk = torch.tensor(duck[sl].astype(np.int32), device=device)
            d = lidar.render(o, yw, pt, dk).cpu().numpy()      # (k, h, w)
            for i in range(k):
                img = (np.clip(1.0 - d[i], 0, 1) * 255).astype(np.uint8)
                frame = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
                r = a[sl.start + i]
                hs = float(np.hypot(r[4], r[5]))
                txt = (f"ep {ei+1}  tick {int(r[0]):4d}  {hs:5.0f} u/s  "
                       f"pitch {pitch[sl.start + i]:+.0f}")
                cv2.putText(frame, txt, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
                write(np.ascontiguousarray(frame).tobytes())
                total += 1
    close()
    print(f"wrote {total} frames ({total / args.fps:.1f}s at {args.fps}fps) -> {out_path}")


if __name__ == "__main__":
    main()
