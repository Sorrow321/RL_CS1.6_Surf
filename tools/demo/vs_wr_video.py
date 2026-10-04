"""skate_laby: the RL agent (top) vs the human world record (bottom), one synchronized video.

Both runs are rendered by the same camera from their recorded poses (origin, view yaw, duck),
re-marched through the map's SDF with surface normals (GpuLidar), shaded with one world light.
The camera pitch is FIXED for both panels (pitch has no effect on movement; the agent's pitch
head is free and bobs), so the two views are framed the same way.
Time zero is the start teleport for both; the human's run is the WR demo (ANALYSIS / display
only - CLAUDE.md section 0).

usage: python tools/demo/vs_wr_video.py --agent <traj.jsonl> [--agent-ep 1] [--demo skate_laby.dem] --out x.mp4
       [--frames 5,30,60  -> PNG stills instead of the video]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("C:/RL_Surf")
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "demo"))
from surfgym import SurfCore, default_config          # noqa: E402
from surfgym.goalfield import GoalField, GROUND_CELL  # noqa: E402
from surfgym.vision import GpuLidar                   # noqa: E402
import parse_hldemo as P                              # noqa: E402

FPS = 60
LW, LH = 640, 360            # rendered resolution (upscaled x2)
SCALE = 2
HFOV = 105.0
VFOV = HFOV * LH / LW        # square angular pixels
PITCH = 4.0                  # fixed camera pitch (deg, down) for both panels
RANGE = 12000.0
BAR_H = 120
BOX_LO = np.array([2123.0, -406.0, -230.0])
BOX_HI = np.array([2200.0, -298.0, -122.0])


def load_agent(path, ep):
    eps, rows, hdr = [], [], None
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        if isinstance(o, dict) and "map" in o:
            rows, hdr = [], o
        elif isinstance(o, list):
            rows.append(o)
        elif isinstance(o, dict) and "end" in o:
            eps.append((hdr, np.asarray(rows, np.float64), o))
    hdr, a, end = eps[ep - 1]
    if end.get("end") != "done":
        raise SystemExit(f"agent episode {ep} did not finish ({end})")
    tick = float(hdr["tick_ms"]) / 1000.0
    n = len(a)
    btn = a[:, 8].astype(np.int64)
    return {"name": "AGENT", "t": np.arange(n) * tick, "pos": a[:, 1:4], "vel": a[:, 4:7],
            "yaw": a[:, 7], "duck": (btn & 4) != 0, "jump": (btn & 2) != 0,
            "fwd": a[:, 13].astype(int), "side": a[:, 14].astype(int), "finish": n * tick}


def load_human(dem):
    data = open(dem, "rb").read()
    h = P.read_header(data)
    entries = P.read_directory(data, h["directory_offset"])
    e = next((x for x in entries if x["description"].lower().startswith("playback")), entries[-1])
    out = P.parse_frames(data, e["offset"], e["offset"] + e["length"])
    org = out["net_simorg"].astype(np.float64)
    d = np.hypot(org[:, 0] + 1056.0, org[:, 1] + 1792.0)
    arr = [i for i in np.flatnonzero(d < 40) if i > 0 and d[i - 1] > 200]
    k0 = int(arr[-1])                                      # the start teleport (timer start)
    inside = np.all((org >= BOX_LO - 16) & (org <= BOX_HI + 16), axis=1)
    kf = k0 + int(np.argmax(inside[k0:]))                  # first frame in the finish box
    sl = slice(k0, kf + 1)
    t = out["net_t"].astype(np.float64)[sl]
    t = t - t[0]
    btn = out["net_uc_buttons"].astype(np.int64)[sl]       # parse_hldemo: buttons << 8
    fsu = out["net_uc_fsu"].astype(np.float64)[sl]
    va = out["net_cl_viewangles"].astype(np.float64)[sl]
    return {"name": "HUMAN", "t": t, "pos": org[sl], "vel": out["net_simvel"].astype(np.float64)[sl],
            "yaw": va[:, 1], "duck": (btn & 1024) != 0, "jump": (btn & 512) != 0,
            "fwd": np.where(fsu[:, 0] > 0, 2, np.where(fsu[:, 0] < 0, 0, 1)),
            "side": np.where(fsu[:, 1] > 0, 2, np.where(fsu[:, 1] < 0, 0, 1)),
            "finish": float(t[-1])}


def add_progress(tr, F):
    d = F.sample(tr["pos"])
    tr["prog"] = (d[0] - np.minimum.accumulate(d)) / d[0]
    tr["yaw_u"] = np.degrees(np.unwrap(np.radians(tr["yaw"])))


def sample(tr, tau):
    """pose at time tau (clamped to the run), linear in position / unwrapped yaw / speed."""
    t = tr["t"]
    tau = np.clip(tau, 0.0, t[-1])
    i = np.clip(np.searchsorted(t, tau, side="right") - 1, 0, len(t) - 2)
    w = np.clip((tau - t[i]) / np.maximum(t[i + 1] - t[i], 1e-9), 0.0, 1.0)
    pos = tr["pos"][i] * (1 - w)[:, None] + tr["pos"][i + 1] * w[:, None]
    yaw = tr["yaw_u"][i] * (1 - w) + tr["yaw_u"][i + 1] * w
    spd = np.hypot(tr["vel"][i, 0], tr["vel"][i, 1]) * (1 - w) + np.hypot(tr["vel"][i + 1, 0], tr["vel"][i + 1, 1]) * w
    return {"pos": pos, "yaw": yaw, "spd": spd, "duck": tr["duck"][i], "jump": tr["jump"][i],
            "fwd": tr["fwd"][i], "side": tr["side"][i], "prog": tr["prog"][i]}


def t_at_prog(tr, p):
    k = np.searchsorted(tr["prog"], p)
    return tr["t"][np.minimum(k, len(tr["t"]) - 1)]


class Shader:
    def __init__(self, lidar, device):
        self.lidar, self.dev = lidar, device
        L = torch.tensor([0.35, 0.25, 0.90], device=device)
        self.L = L / L.norm()
        # BGR albedos: floors light warm grey, walls neutral grey; sky dark blue-grey
        self.floor = torch.tensor([0.74, 0.79, 0.84], device=device)
        self.wall = torch.tensor([0.80, 0.78, 0.76], device=device)
        self.sky = torch.tensor([0.30, 0.24, 0.20], device=device)

    @torch.no_grad()
    def render(self, pos, yaw, duck):
        o = torch.tensor(pos, dtype=torch.float32, device=self.dev)
        yw = torch.tensor(yaw, dtype=torch.float32, device=self.dev)
        pt = torch.full_like(yw, PITCH)
        dk = torch.tensor(np.asarray(duck, np.int32), device=self.dev)
        out = self.lidar.render(o, yw, pt, dk)                 # (B, H, W, 4)
        dep, n = out[..., 0], out[..., 1:4]
        yr = torch.deg2rad(yw)[:, None, None]
        c, s = torch.cos(yr), torch.sin(yr)
        wx = n[..., 0] * c - n[..., 1] * s
        wy = n[..., 0] * s + n[..., 1] * c
        wz = n[..., 2]
        lam = (wx * self.L[0] + wy * self.L[1] + wz * self.L[2]).clamp(0, 1)
        known = n.abs().sum(-1) > 0
        base = torch.where(known, 0.28 + 0.72 * lam, torch.full_like(lam, 0.5))
        alb = torch.where((wz > 0.7)[..., None], self.floor, self.wall)
        col = base[..., None] * alb
        dist = dep * RANGE
        fog = (1.0 - torch.exp(-dist / 6500.0))[..., None]
        col = col * (1 - fog) + self.sky * fog
        miss = dep >= 0.999
        col = torch.where(miss[..., None], self.sky, col)
        # depth-discontinuity outlines: darken where the log depth jumps
        ld = torch.log(dist.clamp_min(1.0))
        gx = torch.zeros_like(ld)
        gy = torch.zeros_like(ld)
        gx[:, :, 1:] = (ld[:, :, 1:] - ld[:, :, :-1]).abs()
        gy[:, 1:, :] = (ld[:, 1:, :] - ld[:, :-1, :]).abs()
        edge = ((gx + gy) > 0.35).float()[..., None]
        col = col * (1 - 0.45 * edge)
        return (col.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def text(img, s, org, scale, thick=1, col=(255, 255, 255), anchor="left"):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if anchor == "right":
        x -= tw
    elif anchor == "center":
        x -= tw // 2
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def key(img, x, y, w, h, label, on):
    if on:
        cv2.rectangle(img, (x, y), (x + w, y + h), (235, 235, 235), -1)
        fg = (20, 20, 20)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 2)
        fg = (170, 170, 170)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, label, (x + (w - tw) // 2, y + (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2,
                cv2.LINE_AA)


def overlay(img, tr, s, j, tau, title, sub):
    Wd, Hd = img.shape[1], img.shape[0]
    text(img, title, (20, 46), 1.15, 2)
    text(img, sub, (20, 80), 0.7, 1, (220, 220, 220))
    fin = tau >= tr["finish"]
    shown = tr["finish"] if fin else tau
    text(img, f"{shown:6.2f} s", (Wd - 20, 56), 1.6, 3, anchor="right")
    text(img, f"{s['spd'][j]:5.0f} u/s", (20, Hd - 24), 1.0, 2)
    k, g = 44, 6
    x0, y0 = Wd - 3 * k - 2 * g - 24, Hd - 2 * k - g - 24
    key(img, x0 + k + g, y0, k, k, "W", s["fwd"][j] == 2)
    key(img, x0, y0 + k + g, k, k, "A", s["side"][j] == 0)
    key(img, x0 + k + g, y0 + k + g, k, k, "S", s["fwd"][j] == 0)
    key(img, x0 + 2 * (k + g), y0 + k + g, k, k, "D", s["side"][j] == 2)
    key(img, x0 - 110, y0 + k + g, 96, k, "DUCK", bool(s["duck"][j]))
    if fin:
        ov = img.copy()
        cv2.rectangle(ov, (0, Hd // 2 - 60), (Wd, Hd // 2 + 40), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        text(img, f"FINISHED  {tr['finish']:.2f} s", (Wd // 2, Hd // 2 + 8), 1.8, 3, anchor="center")


def bar(Wd, a, h, A, Hm, j, tau):
    img = np.full((BAR_H, Wd, 3), 24, np.uint8)
    x0, x1, yl = 70, Wd - 70, 78
    cv2.line(img, (x0, yl), (x1, yl), (150, 150, 150), 2, cv2.LINE_AA)
    for q in range(11):
        x = int(x0 + (x1 - x0) * q / 10)
        cv2.line(img, (x, yl - 6), (x, yl + 6), (150, 150, 150), 1, cv2.LINE_AA)
    text(img, "route", (8, yl + 6), 0.55, 1, (200, 200, 200))
    pa, ph = float(a["prog"][j]), float(h["prog"][j])
    xa, xh = int(x0 + (x1 - x0) * pa), int(x0 + (x1 - x0) * ph)
    # agent: filled triangle ABOVE the line pointing down; human: hollow triangle BELOW pointing up
    tri_a = np.array([[xa - 11, yl - 30], [xa + 11, yl - 30], [xa, yl - 8]], np.int32)
    cv2.fillPoly(img, [tri_a], (255, 255, 255), cv2.LINE_AA)
    text(img, "A", (xa - 30, yl - 14), 0.6, 2)
    tri_h = np.array([[xh - 11, yl + 30], [xh + 11, yl + 30], [xh, yl + 8]], np.int32)
    cv2.polylines(img, [tri_h], True, (255, 255, 255), 2, cv2.LINE_AA)
    text(img, "H", (xh - 30, yl + 30), 0.6, 2)
    # the gap: time between the two at the trailing runner's route position
    if tau >= A["finish"] and tau < Hm["finish"]:
        msg = f"AGENT FINISHED - HUMAN {Hm['finish'] - A['finish']:.2f} s BEHIND AT THE LINE"
    elif tau >= Hm["finish"]:
        msg = f"AGENT {A['finish']:.2f} s  vs  HUMAN WR {Hm['finish']:.2f} s   (-{Hm['finish'] - A['finish']:.2f} s)"
    elif pa >= ph:
        gap = tau - float(t_at_prog(A, ph))
        msg = f"AGENT AHEAD {gap:4.2f} s"
    else:
        gap = tau - float(t_at_prog(Hm, pa))
        msg = f"HUMAN AHEAD {gap:4.2f} s"
    text(img, msg, (Wd // 2, 30), 0.8, 2, anchor="center")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--agent-ep", type=int, default=1)
    ap.add_argument("--agent-title", default="AGENT - RL policy")
    ap.add_argument("--agent-sub", default="")
    ap.add_argument("--demo", default=str(ROOT / "skate_laby.dem"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None, help="comma list of seconds -> PNG stills only")
    ap.add_argument("--tail", type=float, default=2.5, help="seconds held after the last finish")
    ap.add_argument("--cell", type=float, default=16.0, help="SDF / normal grid cell (u)")
    args = ap.parse_args()

    z = np.load(ROOT / "maps" / f"skate_laby.goalw_{GROUND_CELL:g}.npz", allow_pickle=False)
    F = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]),
                  float(z["reach_max"]))
    A = load_agent(args.agent, args.agent_ep)
    Hm = load_human(args.demo)
    for tr in (A, Hm):
        add_progress(tr, F)
    print(f"agent finish {A['finish']:.3f} s ({len(A['t'])} rows); human finish {Hm['finish']:.3f} s "
          f"({len(Hm['t'])} frames)")

    dev = torch.device("cuda")
    core = SurfCore(str(ROOT / "maps" / "skate_laby.bsp"), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    lidar = GpuLidar(core, LW, LH, hfov_deg=HFOV, vfov_deg=VFOV, range_units=RANGE, cell=args.cell,
                     max_steps=256, device=dev, normals=True)
    sh = Shader(lidar, dev)
    Wd, Hd = LW * SCALE, LH * SCALE
    titles = {"AGENT": (args.agent_title, args.agent_sub or f"finish {A['finish']:.2f} s"),
              "HUMAN": ("HUMAN - world record", f"1:14.90 on the server = {Hm['finish']:.2f} s on this clock")}

    if args.frames:
        taus = np.array([float(x) for x in args.frames.split(",")])
    else:
        taus = np.arange(0.0, max(A["finish"], Hm["finish"]) + args.tail, 1.0 / FPS)
    frame_h = 2 * Hd + BAR_H
    enc = None
    if not args.frames:
        enc = subprocess.Popen([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo",
                                "-pix_fmt", "bgr24", "-s", f"{Wd}x{frame_h}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    B = 96
    for b0 in range(0, len(taus), B):
        tb = taus[b0:b0 + B]
        sa, shm = sample(A, tb), sample(Hm, tb)
        ia = sh.render(sa["pos"], sa["yaw"], sa["duck"])
        ih = sh.render(shm["pos"], shm["yaw"], shm["duck"])
        for j, tau in enumerate(tb):
            top = cv2.resize(ia[j], (Wd, Hd), interpolation=cv2.INTER_LINEAR)
            bot = cv2.resize(ih[j], (Wd, Hd), interpolation=cv2.INTER_LINEAR)
            overlay(top, A, sa, j, tau, *titles["AGENT"])
            overlay(bot, Hm, shm, j, tau, *titles["HUMAN"])
            frame = np.vstack([top, bar(Wd, sa, shm, A, Hm, j, tau), bot])
            if args.frames:
                p = Path(args.out).with_suffix("") .as_posix() + f"_{tau:05.1f}s.png"
                cv2.imwrite(p, frame)
                print("wrote", p)
            else:
                enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if b0 // B % 10 == 0:
            print(f"{tb[-1]:6.1f} s / {taus[-1]:.1f} s", flush=True)
    if enc is not None:
        enc.stdin.close()
        enc.wait()
        print("wrote", args.out)


if __name__ == "__main__":
    main()
