#!/usr/bin/env python3
"""refine_arch.py - return-then-explore over an ARCHIVE of a policy's own fastest search lines.

An attempt picks a base line from the archive, replays its first J ticks open-loop into every
env (beam_tas --prefix-line, which carries the wrapper's per-decision feeds and held keys through
the replay) and lets the population search the rest with a checkpoint's own proposals. The winner
is a spliced start-to-finish line whose open-loop replay beam_tas asserts bit-exact. A result
enters the archive when the archive is not full or it beats the archive's slowest line, and does
not repeat a line already there (same action table, or the same finish tick: a re-search late in a
line that lands on the same tick would fill the archive with one line).

Why an archive and not only the best line (princessburglar, pbx 2026-10-28/29): beam_tas
populations collapse onto one lineage, so a wave re-draws the whole line; re-searching the best
line's own tail never got past a later speed loss its 80-90% states approach too low, while
slower lines on other approaches did - all four record lines of the night came from a NON-best
base (60.283 -> 60.069 s in 170 attempts). Nothing here knows the map: junctions are fixed
fractions of each base line, bases / fractions / checkpoints / seeds follow a fixed schedule.

Several workers can share one archive (--worker W, flock on OUT/archive.lock): worker W numbers
its attempts from 5000 W (attempt dirs OUT/a<i>) and offsets its schedule. A restart resumes
from OUT/history.jsonl. Stops at --hours or at OUT/STOP.

    python tools/refine_arch.py --map maps/X.bsp --seeds "runs/A/round_*/wave_*/summary.json" \\
        --ckpts runs/R1/ckpt_final.pt,runs/R2/ckpt_final.pt --hours 2 --out runs/X_arch [--worker 1]
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# the planner settings of the loop that produced the seed lines (exit_local --plan-args)
BEAM = ["--envs", "4096", "--resample-every", "25", "--score", "d", "--elite-frac", "0.0625",
        "--greedy-envs", "128", "--max-ticks", "9800", "--keep-finishers", "16",
        "--route-file", "none", "--allow-nonfinisher"]
FRACS = "0.8,0.2,0.9,0.4,0.75,0.35,0.85,0.3"


def line_key(npz) -> str:
    """md5 of a line's action table, view rows and spawn state: the same line twice."""
    z = np.load(npz, allow_pickle=False)
    h = hashlib.md5(np.ascontiguousarray(z["acts"]).tobytes())
    if "view" in z.files:
        h.update(np.ascontiguousarray(z["view"]).tobytes())
    h.update(np.ascontiguousarray(z["spawn_state"]).tobytes())
    return h.hexdigest()


def offer(arch: list, ticks: int, npz: str, key: str, src: str, m: int) -> tuple[list, bool]:
    """-> (archive sorted by finish tick, entered?) after offering one finished line."""
    arch = sorted(arch, key=lambda a: a["ticks"])
    if key in {a["key"] for a in arch} or int(ticks) in {a["ticks"] for a in arch}:
        return arch, False
    if len(arch) >= m and int(ticks) >= arch[-1]["ticks"]:
        return arch, False
    arch = sorted(arch + [{"ticks": int(ticks), "npz": str(npz), "key": key, "src": str(src)}],
                  key=lambda a: a["ticks"])[:m]
    return arch, True


def seed_archive(globs, m: int, root: Path = ROOT) -> list:
    """The m fastest distinct bit-exact lines among the summary.json files the globs name."""
    cands = []
    for g in globs:
        for s in glob.glob(str(root / g)):
            try:
                j = json.loads(Path(s).read_text())
            except (OSError, ValueError):
                continue
            npz = Path(s).parent / "beam_best.npz"
            if j.get("replay_bit_exact") and j.get("best_ticks") and npz.exists():
                cands.append((int(j["best_ticks"]), str(npz)))
    arch = []
    for t, npz in sorted(cands):
        arch, _ = offer(arch, t, npz, line_key(npz), npz, m)
    return arch


def schedule(i: int, worker: int, n_arch: int, m: int, fracs, ckpts):
    """Attempt i -> (archive rank of its base, junction fraction, checkpoint). One cycle visits
    every base once; worker W is offset so parallel workers do not repeat each other."""
    c = (i - 5000 * worker) // m
    return ((i + 3 * worker) % n_arch, fracs[(c + 5 * worker) % len(fracs)],
            ckpts[(c // len(fracs) + worker) % len(ckpts)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--seeds", required=True,
                    help="comma-separated globs (relative to the repo) of summary.json files")
    ap.add_argument("--ckpts", required=True, help="comma-separated proposal checkpoints")
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive-size", type=int, default=6)
    ap.add_argument("--fracs", default=FRACS, help="junction fractions of a base line's ticks")
    ap.add_argument("--worker", type=int, default=int(os.environ.get("ARCH_WORKER", "0")))
    ap.add_argument("--seed0", type=int, default=41000, help="attempt i draws seed seed0 + i")
    ap.add_argument("--beam-extra", default="",
                    help="beam_tas flags appended after the fixed ones, one string; a repeated flag "
                         "overrides (pbx 2026-10-30: '--elite-frac 0.25' for a policy whose "
                         "1/16-elite populations die together, as the loop runs it)")
    a = ap.parse_args(argv)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    m, w = int(a.archive_size), int(a.worker)
    fracs = [float(x) for x in a.fracs.split(",")]
    ckpts = a.ckpts.split(",")
    stop = time.time() + a.hours * 3600
    ap_ = out / "archive.json"
    lock = out / "archive.lock"
    with open(lock, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not ap_.exists():
            arch = seed_archive(a.seeds.split(","), m)
            if not arch:
                raise SystemExit("no bit-exact seed lines")
            ap_.write_text(json.dumps(arch, indent=1))
            shutil.copyfile(arch[0]["npz"], out / "best.npz")
            shutil.copyfile(Path(arch[0]["npz"]).with_suffix(".jsonl"), out / "best.jsonl")
            print("seed archive:", [(x["ticks"], x["src"]) for x in arch], flush=True)
    hist = out / "history.jsonl"
    mine = [json.loads(ln)["i"] for ln in open(hist)] if hist.exists() else []
    mine = [x for x in mine if 5000 * w <= x < 5000 * (w + 1)]
    for i in itertools.count((max(mine) + 1) if mine else 5000 * w):
        if time.time() > stop or (out / "STOP").exists():
            break
        arch = sorted(json.loads(ap_.read_text()), key=lambda x: x["ticks"])
        bi, frac, ck = schedule(i, w, len(arch), m, fracs, ckpts)
        base = arch[bi]
        z = np.load(base["npz"], allow_pickle=False)
        k = int(z["act_every"])
        jt = int(frac * int(base["ticks"])) // k * k
        d = out / f"a{i:03d}"
        cmd = [sys.executable, "-u", str(ROOT / "tools" / "beam_tas.py"), ck, "--map", a.map,
               *BEAM, *shlex.split(a.beam_extra), "--prefix-line", f"{base['npz']}:{jt}",
               "--seed", str(a.seed0 + i),
               "--torch-seed", str(a.seed0 + i), "--out-dir", str(d)]
        t0 = time.time()
        with open(out / f"a{i:03d}.log", "w") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
        try:
            s = json.loads((d / "summary.json").read_text())
        except (OSError, ValueError):
            s = {}
        bt, ok = s.get("best_ticks"), bool(s.get("replay_bit_exact"))
        rec = {"i": i, "base_ticks": base["ticks"], "base": base["src"], "frac": frac,
               "junction": jt, "ckpt": ck, "seed": a.seed0 + i, "rc": rc, "best_ticks": bt,
               "best_s": s.get("best_s"), "bit_exact": ok, "wall_s": round(time.time() - t0, 1),
               "entered": False, "new_best": False}
        with open(lock, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)     # the other workers' entries since this attempt began
            arch = json.loads(ap_.read_text())
            if bt is not None and ok and (d / "beam_best.npz").exists():
                arch, rec["entered"] = offer(arch, int(bt), str(d / "beam_best.npz"),
                                             line_key(d / "beam_best.npz"), str(d), m)
                if rec["entered"]:
                    ap_.with_suffix(".tmp").write_text(json.dumps(arch, indent=1))
                    os.replace(ap_.with_suffix(".tmp"), ap_)
                    if arch[0]["src"] == str(d):
                        for src, dst in (("beam_best.npz", "best.npz"),
                                         ("beam_best.jsonl", "best.jsonl")):
                            shutil.copyfile(d / src, out / (dst + ".tmp"))
                            os.replace(out / (dst + ".tmp"), out / dst)
                        rec["new_best"] = True
            rec["archive"] = [x["ticks"] for x in sorted(arch, key=lambda x: x["ticks"])]
            print(json.dumps(rec), flush=True)
            with open(hist, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
