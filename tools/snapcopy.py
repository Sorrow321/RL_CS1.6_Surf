"""Keep per-minute snapshots of a run's ckpt_latest.pt as ckpt_<step>.pt (ops; pbr8b3, 2026-10-27).

    python3 tools/snapcopy.py runs/<run> [--every 60] [--keep-secs 1800]

The trainer rewrites ckpt_latest.pt about once a minute but keeps ckpt_<step>.pt only every
--ckpt-every steps (2e8 in this lineage), so the weights behind a best greedy eval are usually
100M+ steps from any snapshot, and the hand-on rule (PROTOCOL.md) then picks a checkpoint far
from the eval. This copies ckpt_latest.pt to ckpt_<global_step>.pt whenever it changes. A copy
that does not load is a half-written file (torch.save is not atomic) and is dropped.

Pruning keeps the disk bounded: a snapshot of OURS (listed in snapcopy.json; the trainer's own
ckpt_<step>.pt files are never touched) younger than --keep-secs is kept; an older one is kept only
while it is the newest snapshot at or before some eval (traj_<step>.jsonl). The loop ends when the
trainer's pid (runs/<run>.pid, else --pid) is gone.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _step_of(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return int(ck["global_step"])


def keep_set(snaps, evals, now, keep_secs):
    """snaps: {step: mtime}; evals: eval steps. The snapshot steps to keep."""
    keep = {s for s, t in snaps.items() if now - t < keep_secs}
    steps = sorted(snaps)
    for e in evals:
        before = [s for s in steps if s <= e]
        if before:
            keep.add(before[-1])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--keep-secs", type=float, default=1800.0)
    ap.add_argument("--pid", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.run_dir)
    pid = a.pid
    pid_file = out.parent / f"{out.name}.pid"
    if pid is None and pid_file.exists():
        pid = int(pid_file.read_text().split()[0])
    reg = out / "snapcopy.json"
    snaps = {int(k): v for k, v in json.loads(reg.read_text()).items()} if reg.exists() else {}
    last = None
    while True:
        src = out / "ckpt_latest.pt"
        try:
            sig = (src.stat().st_mtime_ns, src.stat().st_size)
        except OSError:
            sig = None
        if sig is not None and sig != last:
            tmp = out / "ckpt_snapcopy.tmp"
            try:
                shutil.copyfile(src, tmp)
                step = _step_of(tmp)
            except Exception as ex:  # a half-written ckpt_latest: try again next time
                print(f"skip ({type(ex).__name__}: {ex})", flush=True)
                tmp.unlink(missing_ok=True)
            else:
                dst = out / f"ckpt_{step}.pt"
                if dst.exists():
                    tmp.unlink()
                else:
                    tmp.rename(dst)
                    snaps[step] = time.time()
                    print(f"{time.strftime('%H:%M:%S')} ckpt_{step}.pt", flush=True)
                last = sig
        evals = [int(p.stem.split("_")[1]) for p in out.glob("traj_*.jsonl")
                 if p.stem.split("_")[1].isdigit()]
        keep = keep_set(snaps, evals, time.time(), a.keep_secs)
        for s in [s for s in snaps if s not in keep]:
            (out / f"ckpt_{s}.pt").unlink(missing_ok=True)
            del snaps[s]
        reg.write_text(json.dumps({str(k): v for k, v in sorted(snaps.items())}))
        if pid is not None and not _alive(pid):
            print("trainer gone; done", flush=True)
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
