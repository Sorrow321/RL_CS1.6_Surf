"""Keep per-minute snapshots of a run's ckpt_latest.pt, so that any greedy eval has a
checkpoint within ~one minute of training before it (the trainer's own --ckpt-every
cadence can be 2e8 steps, ~12 min, and the round's hand-on is the newest snapshot at or
before the best eval).

Every --every seconds: when ckpt_latest.pt's mtime has moved, copy it (via a .part file)
to ckpt_<global_step>.pt. Prune: a copy older than --keep-min minutes is deleted unless it
is the newest copy at or before some eval (runs/<run>/traj_<step>.jsonl), so each eval
keeps exactly its hand-on candidate. The trainer's own ckpt_<step>.pt files are never
touched (only files this process wrote are pruned). Exits when --pid is gone.

    nohup python3 tools/snapcopy.py runs/<run> --pid <trainer pid> > runs/<run>_snapcopy.txt 2>&1 &
"""
import argparse
import os
import re
import shutil
import time
from pathlib import Path

import torch


def step_of(path: Path) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return int(ck["global_step"])


def eval_steps(run: Path) -> list:
    out = []
    for p in run.glob("traj_*.jsonl"):
        m = re.fullmatch(r"traj_(\d+)\.jsonl", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def prune(run: Path, mine: dict, keep_s: float) -> None:
    now = time.time()
    evals = eval_steps(run)
    keep = set()
    steps = sorted(mine)
    for e in evals:
        before = [s for s in steps if s <= e]
        if before:
            keep.add(before[-1])
    for s in steps:
        t = mine[s]
        if s in keep or now - t < keep_s:
            continue
        p = run / f"ckpt_{s}.pt"
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        del mine[s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--keep-min", type=float, default=45.0)
    a = ap.parse_args()
    run = Path(a.run)
    src = run / "ckpt_latest.pt"
    mine = {}
    last_mtime = None
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print("trainer gone, exiting", flush=True)
            return
        try:
            mt = src.stat().st_mtime
            if mt != last_mtime and time.time() - mt > 5:
                tmp = run / "snapcopy.part"
                shutil.copyfile(src, tmp)
                s = step_of(tmp)
                dst = run / f"ckpt_{s}.pt"
                if dst.exists():
                    tmp.unlink()
                else:
                    os.replace(tmp, dst)
                    mine[s] = time.time()
                    print(f"{time.strftime('%H:%M:%S')} ckpt_{s}.pt", flush=True)
                last_mtime = mt
            prune(run, mine, a.keep_min * 60.0)
        except Exception as e:  # a half-written ckpt_latest: try again next round
            print(f"{time.strftime('%H:%M:%S')} skip: {e!r}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
