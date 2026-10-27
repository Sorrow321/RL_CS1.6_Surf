"""Keep per-minute copies of a run's ckpt_latest.pt as ckpt_<step>.pt (pbr8b4, ops only).

    nohup python autoresearch/analysis/pb_snapcopy.py runs/<run> [keep_minutes] &

The trainer writes ckpt_latest.pt about once a minute but its own ckpt_<step>.pt snapshots
come every --ckpt-every (2e8) steps, so a best greedy eval can fall ~200M steps after the last
snapshot and the hand-on checkpoint is far from the eval's weights (PROTOCOL: the newest
snapshot at or before the winning eval). This copies ckpt_latest to ckpt_<global_step>.pt
whenever it changes and prunes copies older than keep_minutes (default 45), except the
trainer's own snapshots (any step it already wrote before this started or on its cadence is
left alone: only files this script created are pruned) and any step listed in
<run>/snap_keep.txt (one step per line). Training is untouched: it only reads the file."""
import os
import shutil
import sys
import time
from pathlib import Path

import torch

run = Path(sys.argv[1])
keep_min = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
made = {}                      # step -> wall time this script created it
last_mtime = 0.0
while True:
    src = run / "ckpt_latest.pt"
    try:
        mt = src.stat().st_mtime
        if mt != last_mtime and time.time() - mt > 3.0:      # not mid-write
            tmp = run / ".snapcopy.tmp"
            shutil.copyfile(src, tmp)
            ck = torch.load(tmp, map_location="cpu", weights_only=False)
            step = int(ck.get("global_step", 0))
            del ck
            dst = run / f"ckpt_{step}.pt"
            if step and not dst.exists():
                os.replace(tmp, dst)
                made[step] = time.time()
                print(f"{time.strftime('%H:%M:%S')} copied ckpt_{step}.pt", flush=True)
            else:
                tmp.unlink(missing_ok=True)
            last_mtime = mt
    except Exception as e:                                   # never die on a race
        print(f"{time.strftime('%H:%M:%S')} skip: {type(e).__name__}: {e}", flush=True)
    keep = set()
    try:
        keep = {int(x) for x in (run / "snap_keep.txt").read_text().split()}
    except Exception:
        pass
    for step, t in list(made.items()):
        if step not in keep and time.time() - t > keep_min * 60:
            (run / f"ckpt_{step}.pt").unlink(missing_ok=True)
            del made[step]
    time.sleep(20)
