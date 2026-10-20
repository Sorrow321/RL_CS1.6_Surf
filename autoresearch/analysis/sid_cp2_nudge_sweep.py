#!/usr/bin/env python3
"""nudge_sweep.py - WON'T vs CAN'T at sidistic's CP2 pit (ANALYSIS ONLY, never training).

For each (tick, hold, nudge) the policy's own greedy episode is recorded with
record_ckpt.py --nudge-*: for `hold` decisions from physics tick `tick` the yaw is
either HELD at an absolute world heading (abs) or offset from the policy's own
command (off); then the policy flies free. Each recording is scored on the goal
field: best progress (% of d0), end, duration, end point. A run that survives
past ~27.6% / 28 s got over the pit, which the unnudged line never does.

    python3 nudge_sweep.py <ckpt> <out_dir> [ticks] [holds]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/root/RL_Surf")
sys.path.insert(0, str(ROOT / "python"))
from surfgym.goalfield import GoalField  # noqa: E402

z = np.load(ROOT / "maps/surf_src_sidistic.goal_32.npz", allow_pickle=False)
GF = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
               float(z["cell"]), float(z["reach_max"]))


def episodes(path):
    eps, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                eps.append((hdr, np.asarray(cur, np.float64), d))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps


def main():
    ck, out = sys.argv[1], Path(sys.argv[2])
    ticks = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "2250,2325,2400").split(",")]
    holds = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "12,25").split(",")]
    out.mkdir(parents=True, exist_ok=True)
    nudges = [("none", 0.0)] + [("abs", a) for a in (0, 45, 90, 135, -45, -90)] \
        + [("off", o) for o in (-60, -30, 30, 60)]
    res = out / "sweep.jsonl"
    t0 = time.time()
    done_none = False
    for T in ticks:
        for h in holds:
            for kind, v in nudges:
                if kind == "none":
                    if done_none:
                        continue
                    done_none = True
                tag = f"t{T}_h{h}_{kind}{v:+g}" if kind != "none" else "base"
                jl = out / f"{tag}.jsonl"
                cmd = [sys.executable, str(ROOT / "tools/record_ckpt.py"), ck,
                       "--map", str(ROOT / "maps/surf_src_sidistic.bsp"),
                       "--episodes", "1", "--seed", "0", "--out", str(jl),
                       "--ep-ticks", "3600"]
                if kind != "none":
                    cmd += ["--nudge-tick", str(T), "--nudge-hold", str(h)]
                    if kind == "abs":
                        cmd += ["--nudge-yaw-abs", f"{v:g}"]
                    else:
                        cmd += ["--nudge-yaw", f"{v:g}"]
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
                if r.returncode != 0 or not jl.exists():
                    print(tag, "FAILED", r.stderr[-300:], flush=True)
                    continue
                for hdr, a, tr in episodes(jl):
                    d = GF.sample(a[:, 1:4]).astype(np.float64)
                    ok = d < GF.reach_max - 0.5 * GF.cell
                    dmin = float(d[ok].min()) if ok.any() else float(d[0])
                    row = dict(tag=tag, tick=T, hold=h, kind=kind, val=v,
                               end=tr.get("end"), secs=len(a) / 100.0,
                               prog=round(100 * (d[0] - dmin) / d[0], 2),
                               last=[round(float(x)) for x in a[-1, 1:4]])
                    print(json.dumps(row), f"({time.time() - t0:.0f}s)", flush=True)
                    with open(res, "a") as fh:
                        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
