#!/usr/bin/env python3
"""pull_run_mss.py - box.py pull's harvest over an MSS-clamped ssh (unr1b1, 2026-11-08).

A path-MTU black hole between a vast box and the droplet drops every segment over ~1,400 bytes
(1,000-byte replies arrive, 1,400-byte ones never do), so box.py pull's tar stream and scp
both stall. tools/mss_proxy.py is an ssh ProxyCommand that sets TCP_MAXSEG on its own socket
(no root, nothing system-wide), so the box never sends a segment the path drops. This tool runs
box.py pull's exact tar stream through it into $AR_HOME/runs/<agent>/<run>/ (run.json,
progress.csv, the launch log, every greedy traj_<step>.jsonl) and with --ckpt fetches
ckpt_latest.pt (or --snapshot STEP: ckpt_<STEP>.pt) into a .part, kept only when its md5 matches
the box's.

    python tools/pull_run_mss.py --agent unr1b1 --run unr1b1_gel [--ckpt] [--snapshot STEP]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "tools" / "mss_proxy.py"


def opts():
    return ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o",
            "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o",
            f"ProxyCommand={sys.executable} {PROXY} %h %p"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--snapshot", action="append", default=[])
    a = ap.parse_args()
    home = Path(os.environ.get("AR_HOME", "/home/steam/ar_uni"))
    rec = json.loads((home / "boxes" / f"{a.agent}.json").read_text())
    tgt = ["-p", str(rec["port"]), f"root@{rec['host']}"]
    dst = home / "runs" / a.agent / a.run
    dst.mkdir(parents=True, exist_ok=True)
    r_ = shlex.quote(a.run)
    remote = (f"cd /root/RL_Surf/runs && tar czf - {shlex.quote(a.run + '_launch.txt')} "
              f"$(cd {r_} && ls run.json progress.csv 2>/dev/null | sed 's#^#{a.run}/#') "
              f"$(cd {r_} && ls traj_*.jsonl 2>/dev/null | grep -E '^traj_[0-9]+[.]jsonl$' | "
              f"sed 's#^#{a.run}/#')")
    tgz = dst / "_pull_mss.tgz"
    with open(tgz, "wb") as fh:
        r = subprocess.run(["ssh", *opts(), *tgt, remote], stdout=fh, stdin=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=1800)
    if r.returncode != 0:
        print(f"!! tar stream exit {r.returncode}: {r.stderr.decode(errors='replace')[-300:]}")
    n = 0
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if m.isfile() and (name in ("run.json", "progress.csv", a.run + "_launch.txt")
                               or (name.startswith("traj_") and name.endswith(".jsonl"))):
                with tf.extractfile(m) as fsrc:
                    data = fsrc.read()
                (dst / name).write_bytes(data)
                n += 1
    tgz.unlink(missing_ok=True)
    cks = (["ckpt_latest.pt"] if a.ckpt else []) + [f"ckpt_{s}.pt" for s in a.snapshot]
    for ck in cks:
        w = subprocess.run(["ssh", *opts(), *tgt,
                            f"md5sum /root/RL_Surf/runs/{r_}/{ck} | cut -d' ' -f1"],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
        want = w.stdout.strip()
        if not want:
            print(f"!! {ck}: no md5 on the box")
            continue
        part = dst / (ck + ".part")
        part.unlink(missing_ok=True)
        subprocess.run(["scp", *opts(), "-q", "-P", str(rec["port"]),
                        f"root@{rec['host']}:/root/RL_Surf/runs/{a.run}/{ck}", str(part)],
                       stdin=subprocess.DEVNULL, timeout=1800)
        got = hashlib.md5(part.read_bytes()).hexdigest() if part.exists() else ""
        if got != want:
            part.unlink(missing_ok=True)
            print(f"!! {ck}: md5 {got} != {want} on the box; not kept")
        else:
            part.replace(dst / ck)
            print(f"pulled {ck} (md5 {want})")
    print(f"pulled {n} files -> {dst} ({len(list(dst.glob('traj_*.jsonl')))} eval trajectories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
