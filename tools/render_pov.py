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


def _draw_key(frame, x, y, w, h, label, on):
    """One keycap: filled bright when pressed, dim outline otherwise."""
    if on:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 200, 255), -1)
        fg = (20, 20, 20)
    else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)
        fg = (150, 150, 150)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, label, (x + (w - tw) // 2, y + (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg, 1, cv2.LINE_AA)


def _draw_keys(frame, W, H, fwd, side, jump, duck):
    """WASD + SPACE + DUCK cluster, bottom-right. fwd/side None = unrecorded
    (old trajectories) — those keys stay dim."""
    k, gap = 30, 4
    x0 = W - 3 * k - 2 * gap - 12
    y0 = H - 2 * k - gap - 34
    _draw_key(frame, x0 + k + gap, y0, k, k, "W", fwd == 2)
    _draw_key(frame, x0, y0 + k + gap, k, k, "A", side == 0)
    _draw_key(frame, x0 + k + gap, y0 + k + gap, k, k, "S", fwd == 0)
    _draw_key(frame, x0 + 2 * (k + gap), y0 + k + gap, k, k, "D", side == 2)
    y1 = y0 + 2 * (k + gap)
    _draw_key(frame, x0 - 40, y1, 2 * k + gap + 40, 22, "SPACE", jump)
    _draw_key(frame, x0 + 2 * (k + gap), y1, k + 14, 22, "DUCK", duck)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--map", default=None,
                    help="defaults to the run.json map next to the traj "
                         "(surf_ski_2 when neither is available)")
    ap.add_argument("--w", type=int, default=128, help="lidar width (match training)")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--scale", type=int, default=6, help="upscale factor")
    ap.add_argument("--fps", type=int, default=100, help="100 = real-time (10ms ticks)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--surf-mask", action="store_true",
                    help="render the SECOND channel the --surf-mask policy "
                         "sees (the hit surface's |n_z|, i.e. how surfable "
                         "that pixel is) stacked BELOW the depth image, so "
                         "the two panels are pixel-aligned and you can read "
                         "off whether a ramp was in view at all")
    ap.add_argument("--goal-ball", type=int, default=0,
                    help="--goal-obs ball runs: render the goal-ball view "
                         "channels the policy receives (1 or 4 views, from "
                         "each episode's recorded goal) in a panel under the "
                         "depth image. 0 = off; filled from run.json when "
                         "the run trained with the ball")
    ap.add_argument("--goal-radius", type=float, default=192.0)
    ap.add_argument("--normals", action="store_true",
                    help="--normals runs: render the three ego-frame normal "
                         "channels the policy receives (x forward, y left, "
                         "z up; [-1, 1] -> [0, 255] as R, G, B, a miss is "
                         "mid-grey) in a panel under the depth image. "
                         "Filled from run.json when the run trained with it")
    ap.add_argument("--hfov", type=float, default=None,
                    help="lidar horizontal fov, degrees (run.json's "
                         "lidar_hfov, else 120)")
    ap.add_argument("--vfov", type=float, default=None,
                    help="lidar vertical fov, degrees (run.json's "
                         "lidar_vfov, else 90)")
    ap.add_argument("--horizon", action="store_true",
                    help="draw the world-horizon line (off by default)")
    ap.add_argument("--ep", type=int, default=None,
                    help="render only this episode (1-based)")
    args = ap.parse_args()

    rows, episodes, headers, hdr = [], [], [], {}
    for line in open(args.traj, encoding="utf-8"):
        o = json.loads(line)
        if isinstance(o, dict) and "map" in o:
            rows, hdr = [], o
        elif isinstance(o, list):
            rows.append(o)
        elif isinstance(o, dict) and "end" in o and rows:
            episodes.append(np.asarray(rows, dtype=np.float64))
            headers.append(hdr)
            rows = []
    if not episodes:
        raise SystemExit("no episodes in trajectory file")
    if args.ep is not None:
        if not 1 <= args.ep <= len(episodes):
            raise SystemExit(f"--ep {args.ep} out of range (file has "
                             f"{len(episodes)} episodes)")
        headers = [headers[args.ep - 1]]
        episodes = [episodes[args.ep - 1]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # match the run's actual sensor (map/dims/range/encoding) via run.json
    # when the traj sits inside a run directory
    rng_u, near, cell, pinhole = 2000.0, None, None, False
    explicit_map = args.map is not None      # an explicit --map beats both
    rj = Path(args.traj).parent / "run.json"
    if rj.exists():
        c = json.loads(rj.read_text(encoding="utf-8")).get("config", {})
        args.w = int(c.get("lidar_w", args.w))
        args.h = int(c.get("lidar_h", args.h))
        rng_u = float(c.get("lidar_range", rng_u))
        near = c.get("lidar_near")
        cell = c.get("lidar_cell")
        pinhole = bool(c.get("pinhole", 0))
        if c.get("normals"):
            args.normals = True
        if args.hfov is None and c.get("lidar_hfov"):
            args.hfov = float(c["lidar_hfov"])
        if args.vfov is None and c.get("lidar_vfov"):
            args.vfov = float(c["lidar_vfov"])
        if not args.goal_ball and c.get("goal_obs") in ("ball", "both"):
            args.goal_ball = int(c.get("goal_views") or 4)
            args.goal_radius = float(c.get("goal_radius") or args.goal_radius)
        if args.map is None and c.get("map"):
            args.map = str(ROOT / "maps" / f"{c['map']}.bsp")
    # A --maps run trains on several maps, so run.json names only ONE of them
    # and every trajectory would be marched through that map's SDF. Each
    # recording states its own map in its header line, so the FILE wins.
    # Getting this wrong is silent and looks like a broken sensor: petrus
    # coordinates inside cannonball's geometry return depth 0 on every ray,
    # which renders as a flat red screen.
    if not explicit_map:
        try:
            with open(args.traj, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line[0] != "{":
                        break
                    hm = json.loads(line).get("map")
                    if hm:
                        args.map = str(ROOT / "maps" / f"{hm}.bsp")
                    break
        except (OSError, ValueError):
            pass
    if args.map is None:
        args.map = str(ROOT / "maps" / "surf_ski_2.bsp")
    core = SurfCore(args.map, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    if cell is None:
        from surfgym.vision import pick_cell
        cell = pick_cell(core)
    # the camera's fov follows the run (--lidar-hfov/--lidar-vfov); the
    # shipped default is 120 x 90, which every checkpoint before the flags
    # was trained on
    HFOV = float(args.hfov) if args.hfov else 120.0
    VFOV = float(args.vfov) if args.vfov else 90.0
    lidar = GpuLidar(core, args.w, args.h, hfov_deg=HFOV, vfov_deg=VFOV,
                     range_units=rng_u, near_range=near,
                     cell=float(cell), device=device, pinhole=pinhole,
                     surf_mask=bool(args.surf_mask),
                     normals=bool(args.normals))

    out_path = Path(args.out) if args.out else Path(args.traj).with_suffix(".pov.mp4")
    # the lidar is EQUIANGULAR (fisheye-like) with anisotropic pixels:
    # 120/128 = 0.94 deg/px horizontal vs 90/64 = 1.41 deg/px vertical.
    # display with square angular pixels so proportions read correctly
    aspect_fix = (VFOV / args.h) / (HFOV / args.w)
    if pinhole:
        # a rectilinear frame's pixels are uniform on the TANGENT PLANE, not
        # in angle, so squareness is a ratio of tangents — undo the
        # equiangular correction rather than layering it on
        d2r = np.pi / 180.0
        aspect_fix = ((np.tan(VFOV / 2 * d2r) / args.h)
                      / (np.tan(HFOV / 2 * d2r) / args.w))
    W, H = args.w * args.scale, int(round(args.h * args.scale * aspect_fix))
    # --surf-mask stacks a second, pixel-aligned panel underneath
    # --goal-ball: a second panel of the same size holds the view(s) -
    # four views as a 2x2 grid at half scale (every lidar pixel still
    # >= 3 px), one view full size
    # --normals: a third kind of panel, the ego normal as RGB, stacked
    # directly under the depth (then the ball panel, when both are on)
    ball_panel = args.goal_ball > 0
    if ball_panel and args.surf_mask:
        raise SystemExit("--goal-ball and --surf-mask are exclusive")
    if args.normals and args.surf_mask:
        raise SystemExit("--normals and --surf-mask are exclusive (|n_z| is "
                         "the normal's third channel)")
    n_panels = 1 + int(bool(args.normals)) + int(args.surf_mask or ball_panel)
    FRAME_H = H * n_panels

    # system ffmpeg (libx264 ultrafast) is ~5x faster than cv2's mp4v writer
    # and makes browser-playable files; fall back to cv2 if it's missing
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff:
        enc = subprocess.Popen(
            [ff, "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{W}x{FRAME_H}", "-r", str(args.fps),
             "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out_path)],
            stdin=subprocess.PIPE)
        write = enc.stdin.write

        def close():
            enc.stdin.close()
            enc.wait()
    else:
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, FRAME_H))
        write = lambda buf: vw.write(np.frombuffer(buf, np.uint8).reshape(FRAME_H, W, 3))

        def close():
            vw.release()

    B = 512                                      # render B poses per GPU batch
    ball = None
    if ball_panel:
        from surfgym.goalball import GoalBallLidar
        ball = GoalBallLidar(lidar, B, radius=args.goal_radius,
                             views=args.goal_ball)
        print(ball.describe())
    VIEW_NAMES = {1: ("ball",), 4: ("ball front", "ball back",
                                    "ball left", "ball right")}
    total = 0
    for ei, a in enumerate(episodes):
        n = len(a)
        if ball is not None:
            g = headers[ei].get("goal") if ei < len(headers) else None
            if isinstance(g, dict):
                gc, gr = g.get("center"), g.get("radius")
            else:
                gc, gr = g, None
            if gc is None:
                print(f"episode {ei + 1}: no goal in its header - ball panel "
                      f"blank")
                gc = [float("nan")] * 3
            ball.set_goals(np.arange(B), np.repeat(
                np.asarray(gc, np.float32)[None, :3], B, 0),
                radius=np.full(B, float(gr or args.goal_radius), np.float32))
        pitch = a[:, 12] if a.shape[1] > 12 else np.zeros(n)
        duck = (a[:, 8].astype(np.int64) & 4) != 0     # buttons IN_DUCK bit
        for s0 in range(0, n, B):
            sl = slice(s0, min(s0 + B, n))
            k = sl.stop - sl.start
            o = torch.tensor(a[sl, 1:4], dtype=torch.float32, device=device)
            yw = torch.tensor(a[sl, 7], dtype=torch.float32, device=device)
            pt = torch.tensor(pitch[sl], dtype=torch.float32, device=device)
            dk = torch.tensor(duck[sl].astype(np.int32), device=device)
            if ball is not None:
                # (k, h, w, 1 + views): channel 0 is the map depth
                d = ball.render(o, yw, pt, dk, idx=np.arange(k)).cpu().numpy()
            else:
                d = lidar.render(o, yw, pt, dk).cpu().numpy()      # (k, h, w)
            enc_max = 1.25 if (near and near < rng_u) else 1.0
            for i in range(k):
                dep = d[i][..., 0] if d[i].ndim == 3 else d[i]
                img = (np.clip(1.0 - dep / enc_max, 0, 1) * 255).astype(np.uint8)
                frame = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
                r = a[sl.start + i]
                p = float(pitch[sl.start + i])
                btn = int(r[8])
                fwd = int(r[13]) if a.shape[1] > 13 else None
                sde = int(r[14]) if a.shape[1] > 14 else None
                _draw_keys(frame, W, H, fwd, sde,
                           bool(btn & 2), bool(btn & 4))
                if args.horizon:
                    # world-horizon line: the row whose absolute ray pitch is 0
                    # (rows span view_pitch +VFOV/2 .. -VFOV/2 top to bottom)
                    hy = (0.5 + p / VFOV) * H
                    if 0 <= hy < H:
                        cv2.line(frame, (0, int(hy)), (W, int(hy)),
                                 (200, 200, 200), 1, cv2.LINE_AA)
                cv2.drawMarker(frame, (W // 2, H // 2), (255, 255, 255),
                               cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
                hs = float(np.hypot(r[4], r[5]))
                txt = (f"ep {ei+1}  tick {int(r[0]):4d}  {hs:5.0f} u/s  "
                       f"pitch {p:+.0f}")
                cv2.putText(frame, txt, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
                if args.normals or ball is not None or args.surf_mask:
                    cv2.putText(frame, "depth", (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)
                if args.normals:
                    # channels 1..3: the hit surface's unit normal in the
                    # player's ego frame (x forward, y left, z up), each in
                    # [-1, 1]; (0, 0, 0) = no surface (a miss, or solid the
                    # mesh bake has no face for). Shown as R, G, B on
                    # [0, 255]: a floor is (128, 128, 255), the wall ahead
                    # (0, 128, 128), a wall on the right (128, 255, 128), a
                    # miss mid-grey. cv2 frames are BGR, hence the flip.
                    nrm = np.clip(d[i][..., 1:4], -1.0, 1.0)
                    rgb = ((nrm + 1.0) * 127.5).astype(np.uint8)
                    npan = cv2.resize(np.ascontiguousarray(rgb[..., ::-1]),
                                      (W, H), interpolation=cv2.INTER_NEAREST)
                    known = float((np.abs(nrm).sum(-1) > 0).mean()) * 100.0
                    cv2.putText(npan, f"normals  R=fwd G=left B=up   "
                                f"{known:4.1f}% surfaced", (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(npan, (0, 0), (W, 0), (60, 60, 60), 1)
                    frame = np.vstack((frame, npan))
                if ball is not None:
                    # the ball in each view rides AFTER the lidar's own
                    # channels (depth, or depth + 3 normals), the same
                    # distance encoding as the depth (0 = no ball on that
                    # ray, drawn black; near = warm, far = cool)
                    base = 4 if args.normals else 1
                    names = VIEW_NAMES[args.goal_ball]
                    tiles = []
                    for j, nm in enumerate(names):
                        v = d[i][..., base + j]
                        # brightness floor: a far ball encodes near enc_max
                        # (dark in the depth palette) and would vanish
                        # against the no-ball black for a human; the
                        # policy sees the raw value, where 0 vs 1.25 is
                        # unmistakable
                        vimg = np.where(v > 0.0,
                                        96 + np.clip(1.0 - v / enc_max, 0, 1) * 159,
                                        0.0).astype(np.uint8)
                        tile = cv2.applyColorMap(vimg, cv2.COLORMAP_TURBO)
                        tile[v <= 0.0] = (16, 16, 16)
                        tw_, th_ = (W, H) if len(names) == 1 else (W // 2, H // 2)
                        tile = cv2.resize(tile, (tw_, th_),
                                          interpolation=cv2.INTER_NEAREST)
                        seen = float((v > 0.0).mean()) * 100.0
                        cv2.putText(tile, f"{nm}  {seen:4.1f}% px", (6, 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.rectangle(tile, (0, 0), (tw_ - 1, th_ - 1),
                                      (60, 60, 60), 1)
                        tiles.append(tile)
                    if len(tiles) == 1:
                        panel = tiles[0]
                    else:
                        panel = np.vstack((np.hstack(tiles[0:2]),
                                           np.hstack(tiles[2:4])))
                    if panel.shape[0] != H or panel.shape[1] != W:
                        panel = cv2.resize(panel, (W, H),
                                           interpolation=cv2.INTER_NEAREST)
                    frame = np.vstack((frame, panel))
                if args.surf_mask:
                    # channel 1 is |n_z| in [0,1]: 1 = flat floor/ceiling,
                    # ~0.6-0.9 = a rideable ramp, 0 = vertical wall or no hit.
                    # A different colormap on purpose - this panel is NOT
                    # distance and should never be read as depth.
                    nz = np.clip(d[i][..., 1], 0.0, 1.0)
                    mimg = (nz * 255).astype(np.uint8)
                    mfr = cv2.applyColorMap(mimg, cv2.COLORMAP_VIRIDIS)
                    mfr = cv2.resize(mfr, (W, H),
                                     interpolation=cv2.INTER_NEAREST)
                    surfable = float((nz > 0.05).mean()) * 100.0
                    cv2.putText(mfr, f"surfable |n_z|   {surfable:4.1f}% of view",
                                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(mfr, (0, 0), (W, 0), (60, 60, 60), 1)
                    frame = np.vstack((frame, mfr))
                write(np.ascontiguousarray(frame).tobytes())
                total += 1
    close()
    print(f"wrote {total} frames ({total / args.fps:.1f}s at {args.fps}fps) -> {out_path}")


if __name__ == "__main__":
    main()
