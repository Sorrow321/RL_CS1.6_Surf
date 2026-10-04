"""Per-frame data for the viewer-rendered skate_laby agent-vs-WR video (viewer/vs_render.html).

Resamples both runs to FPS by time - the agent's trajectory (record_ckpt / beam_tas jsonl) and the
human WR demo (ANALYSIS / display only, CLAUDE.md section 0) - with the same loaders as
tools/demo/vs_wr_video.py, and writes one JSON the render page plays frame by frame:
poses (GoldSrc origin, view yaw, duck), speed, keys, route progress, the gap line, a --pre
second countdown before the start, and a top-down map image (the ground field's walkable
floor) with its world transform.

usage: python tools/demo/vs_wr_frames.py --agent <traj.jsonl> [--agent-ep 1] --out frames.json
       [--agent-title ...] [--agent-sub ...] [--human-title ...] [--human-sub ...] [--pre 3]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vs_wr_video as V  # noqa: E402

FLOOR_BGR = (178, 196, 210)     # sand
WALL_BGR = (58, 50, 44)         # dark blue-grey


def minimap(z, png: Path, long_side: int = 600):
    """The walkable floor seen from above (ground field columns with a reachable node) as a PNG,
    north up; -> its world extents so the page can place positions on it."""
    d = z["grid"].astype(np.float32) * float(z["quant"])
    walk = (d <= float(z["reach_max"]) + 1.0).any(axis=0)          # (ny, nx), row = +y
    mins, cell = z["mins"], float(z["cell"])
    ys, xs = np.nonzero(walk)
    y0i, y1i = max(0, ys.min() - 3), min(walk.shape[0], ys.max() + 4)
    x0i, x1i = max(0, xs.min() - 3), min(walk.shape[1], xs.max() + 4)
    m = walk[y0i:y1i, x0i:x1i].astype(np.float32)[::-1, :]          # north up
    h, w = m.shape
    sc = long_side / max(h, w)
    W, H = int(round(w * sc)), int(round(h * sc))
    up = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    a = np.clip((up - 0.35) / 0.3, 0.0, 1.0)[..., None]
    img = (np.array(WALL_BGR, np.float32) * (1 - a) + np.array(FLOOR_BGR, np.float32) * a)
    cv2.imwrite(str(png), img.astype(np.uint8))
    return {"x0": float(mins[0] + x0i * cell), "x1": float(mins[0] + x1i * cell),
            "y0": float(mins[1] + y0i * cell), "y1": float(mins[1] + y1i * cell), "w": W, "h": H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--agent-ep", type=int, default=1)
    ap.add_argument("--agent-title", default="AGENT")
    ap.add_argument("--agent-sub", default="")
    ap.add_argument("--human-title", default="HUMAN - world record")
    ap.add_argument("--human-sub", default="")
    ap.add_argument("--demo", default=str(ROOT / "skate_laby.dem"))
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--pre", type=float, default=3.0, help="countdown seconds before the start")
    ap.add_argument("--tail", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(ROOT / "maps" / f"skate_laby.goalw_{V.GROUND_CELL:g}.npz", allow_pickle=False)
    F = V.GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]),
                    float(z["reach_max"]))
    A = V.load_agent(args.agent, args.agent_ep)
    H = V.load_human(args.demo)
    for tr in (A, H):
        V.add_progress(tr, F)
    taus = np.arange(-args.pre, max(A["finish"], H["finish"]) + args.tail, 1.0 / args.fps)
    sa, sh = V.sample(A, taus), V.sample(H, taus)
    # the countdown holds both at the start teleport, facing the way they leave it (the demo's
    # first frames still carry the pre-teleport view; it snaps to the teleport's yaw 0.04 s in)
    for tr, s in ((A, sa), (H, sh)):
        y05 = float(V.sample(tr, np.array([0.05]))["yaw"][0])
        pre = taus < 0
        s["yaw"] = np.where(pre, y05, s["yaw"])
        s["spd"] = np.where(pre, 0.0, s["spd"])
        s["fwd"] = np.where(pre, 1, s["fwd"])
        s["side"] = np.where(pre, 1, s["side"])
        s["duck"] = np.where(pre, False, s["duck"])

    def pack(s):
        return [[round(float(s["pos"][j, 0]), 1), round(float(s["pos"][j, 1]), 1),
                 round(float(s["pos"][j, 2]), 1), round(float(s["yaw"][j]), 2), int(bool(s["duck"][j])),
                 int(round(float(s["spd"][j]))), int(s["fwd"][j]), int(s["side"][j]),
                 round(float(s["prog"][j]), 5)] for j in range(len(taus))]

    gaps = []
    for j, tau in enumerate(taus):
        pa, ph = float(sa["prog"][j]), float(sh["prog"][j])
        if tau < 0:
            msg = "GET READY"
        elif tau >= A["finish"] and tau < H["finish"]:
            msg = f"AGENT FINISHED - HUMAN {H['finish'] - A['finish']:.2f} s BEHIND AT THE LINE"
        elif tau >= H["finish"] and tau >= A["finish"]:
            msg = (f"AGENT {A['finish']:.2f} s   vs   HUMAN {H['finish']:.2f} s   "
                   f"({A['finish'] - H['finish']:+.2f} s)")
        elif tau >= H["finish"]:
            msg = f"HUMAN FINISHED - AGENT {A['finish'] - H['finish']:.2f} s BEHIND"
        elif pa >= ph:
            msg = f"AGENT AHEAD {tau - float(V.t_at_prog(A, ph)):.2f} s"
        else:
            msg = f"HUMAN AHEAD {tau - float(V.t_at_prog(H, pa)):.2f} s"
        gaps.append(msg)
    out_path = Path(args.out)
    png = out_path.with_name(out_path.stem + "_minimap.png")
    mm = minimap(z, png)
    try:
        mm["url"] = "/" + png.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        mm["url"] = png.name
    out = {"fps": args.fps, "n": len(taus), "pre": args.pre, "minimap": mm,
           "start": [-1056.0, -1792.0], "finish": [2161.5, -352.0],
           "agent": {"title": args.agent_title,
                     "sub": args.agent_sub or f"{A['finish']:.2f} s",
                     "finish": round(A["finish"], 3), "f": pack(sa)},
           "human": {"title": args.human_title,
                     "sub": args.human_sub or f"{H['finish']:.2f} s",
                     "finish": round(H["finish"], 3), "f": pack(sh)},
           "gap": gaps}
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out_path}: {len(taus)} frames at {args.fps} fps ({args.pre:g} s countdown); "
          f"agent {A['finish']:.3f} s, human {H['finish']:.3f} s; minimap {png} {mm['w']}x{mm['h']}")


if __name__ == "__main__":
    main()
