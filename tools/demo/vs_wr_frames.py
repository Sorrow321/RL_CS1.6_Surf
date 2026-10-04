"""Per-frame data for the viewer-rendered skate_laby agent-vs-WR video (viewer/vs_render.html).

Resamples both runs to FPS by time - the agent's trajectory (record_ckpt / beam_tas jsonl) and the
human WR demo (ANALYSIS / display only, CLAUDE.md section 0) - with the same loaders as
tools/demo/vs_wr_video.py, and writes one JSON the render page plays frame by frame:
poses (GoldSrc origin, view yaw, duck), speed, keys, route progress and the gap line.

usage: python tools/demo/vs_wr_frames.py --agent <traj.jsonl> [--agent-ep 1] --out frames.json
       [--agent-title ...] [--agent-sub ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vs_wr_video as V  # noqa: E402


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
    taus = np.arange(0.0, max(A["finish"], H["finish"]) + args.tail, 1.0 / args.fps)
    sa, sh = V.sample(A, taus), V.sample(H, taus)

    def pack(s):
        return [[round(float(s["pos"][j, 0]), 1), round(float(s["pos"][j, 1]), 1),
                 round(float(s["pos"][j, 2]), 1), round(float(s["yaw"][j]), 2), int(bool(s["duck"][j])),
                 int(round(float(s["spd"][j]))), int(s["fwd"][j]), int(s["side"][j]),
                 round(float(s["prog"][j]), 5)] for j in range(len(taus))]

    gaps = []
    for j, tau in enumerate(taus):
        pa, ph = float(sa["prog"][j]), float(sh["prog"][j])
        if tau >= A["finish"] and tau < H["finish"]:
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
    out = {"fps": args.fps, "n": len(taus),
           "agent": {"title": args.agent_title,
                     "sub": args.agent_sub or f"{A['finish']:.2f} s",
                     "finish": round(A["finish"], 3), "f": pack(sa)},
           "human": {"title": args.human_title,
                     "sub": args.human_sub or f"{H['finish']:.2f} s",
                     "finish": round(H["finish"], 3), "f": pack(sh)},
           "gap": gaps}
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.out}: {len(taus)} frames at {args.fps} fps; agent {A['finish']:.3f} s, "
          f"human {H['finish']:.3f} s")


if __name__ == "__main__":
    main()
