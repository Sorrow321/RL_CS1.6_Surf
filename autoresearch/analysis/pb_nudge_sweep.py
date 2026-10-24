#!/usr/bin/env python3
"""pb_nudge_sweep.py - perturb the policy's OWN greedy line from the true start (pbr2b4, round 2).

For each (tick, hold, yaw offset) one greedy episode is recorded with record_ckpt --nudge-*: for
`hold` decisions from physics tick `tick` the policy's yaw command is offset by `yaw` degrees,
then it flies free. Scored on the grader field: best progress, the max energy height
h = z + |v|^2/(2g) over the last part, the end. Runs in parallel on the box's CPU. Ticks come from
the policy's own energy profile (where its h drops), never from map coordinates; a run that goes
deeper than the unnudged line is a self-generated line (the policy + one yaw deviation), dumped
with --dump-states so it can become a curriculum spine.

  python3 autoresearch/analysis/pb_nudge_sweep.py CKPT OUT --ticks 650,700 --holds 5,15 \\
      --yaws -40,-20,-10,10,20,40 [--jobs 12 --ep-ticks 3000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "autoresearch" / "analysis"))
import pb_line_tools as T  # noqa: E402

OMP = 4


def run_one(ck, out, tick, hold, yaw, ep_ticks):
    tag = "base" if hold == 0 else f"t{tick}_h{hold}_y{yaw:+g}"
    jl, dz = out / f"{tag}.jsonl", out / f"{tag}.npz"
    cmd = [sys.executable, str(ROOT / "tools" / "record_ckpt.py"), ck, "--episodes", "1",
           "--ep-ticks", str(ep_ticks), "--out", str(jl), "--dump-states", str(dz)]
    if hold:
        cmd += ["--nudge-tick", str(tick), "--nudge-hold", str(hold), "--nudge-yaw", f"{yaw:g}"]
    env = dict(__import__("os").environ, OMP_NUM_THREADS=str(OMP), MKL_NUM_THREADS=str(OMP))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    return tag, r.returncode, dz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--ticks", required=True)
    ap.add_argument("--holds", default="5,15")
    ap.add_argument("--yaws", default="-40,-20,-10,10,20,40")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--ep-ticks", type=int, default=3000)
    a = ap.parse_args(argv)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(0, 0, 0.0)]
    for t in (int(x) for x in a.ticks.split(",")):
        for h in (int(x) for x in a.holds.split(",")):
            for y in (float(x) for x in a.yaws.split(",")):
                jobs.append((t, h, y))
    f = T.field()
    res = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, a.ckpt, out, t, h, y, a.ep_ticks) for t, h, y in jobs]
        for (t, h, y), fu in zip(jobs, futs):
            tag, rc, dz = fu.result()
            if rc != 0 or not dz.exists():
                print(tag, "FAILED", flush=True)
                continue
            e = T.eps_of(dz)[0]
            p = T.prog(f, e)
            o = e["origin"].astype(np.float64)
            v = e["velocity"].astype(np.float64)
            hh = o[:, 2] + (v * v).sum(1) / 1600.0
            i = int(np.nanargmax(p))
            row = {"tag": tag, "tick": t, "hold": h, "yaw": y, "best": float(p[i]), "at": i,
                   "end": o[-1].round(0).tolist(), "len": int(len(e)),
                   "h_max_after": float(hh[min(len(e) - 1, t + 100):].max()) if len(e) > t + 100 else None}
            res.append(row)
            print(json.dumps(row), flush=True)
    res.sort(key=lambda r: -r["best"])
    (out / "sweep.json").write_text(json.dumps(res, indent=1))
    print("TOP:")
    for r in res[:10]:
        print(f"  {r['tag']:>22}: best {r['best']:.3f}% at tick {r['at']}, h_max_after {r['h_max_after']}")


if __name__ == "__main__":
    main()
