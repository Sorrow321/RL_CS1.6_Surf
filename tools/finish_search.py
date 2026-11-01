#!/usr/bin/env python3
"""finish_search.py - search for a FINISH from the policy's OWN late states (zlr3b4, 2026-11-01).

A policy can reach the last structure of a map and swing within a few hundred units of the
finish without ever entering it: its greedy line is one sample of a manoeuvre whose finishing
variant it has never experienced, so the success bonus never reaches the gradient. This driver
asks the policy's own search for that variant:

1. the GREEDY line: tools/beam_tas.py with every env greedy and --frontier-last --save-frontier
   (the keys-hold-safe way to save the greedy's own action table, beam_frontier.npz), unless
   --greedy-line names one;
2. for every start second S in --at (seconds of that greedy line) and every --seeds seed: a
   population search from the greedy's own state at S (beam_tas --prefix-line LINE:ticks(S),
   physics-only prefix replay), the policy's own proposals plus held view offsets
   (--view-hold), ranked by the goal field d (the finish is d = 0), --objective finish, for
   --window seconds; --save-frontier writes the finishing line (or the deepest one) with its
   per-tick states from the TRUE start, replay-checked;
3. one summary row per search in OUT/summary.jsonl; --stop-on-finish ends the sweep at the first
   crossing.

Nothing here reads a place on the map: the start states are the policy's own greedy line, the
proposals are the policy's own, the ranking is the goal field every run already shapes on.

    python tools/finish_search.py runs/x/ckpt.pt --out runs/fs --at 120,130,140,150 \
        --window 30 --seeds 0,1 --envs 4096 --greedy-envs 512 --view-hold 1.0 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, log: Path, timeout: float) -> int:
    cmd = [str(c) for c in cmd]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    print(f"  exit {rc} after {time.time() - t0:.0f}s (log {log})", flush=True)
    return rc


def line_info(npz: Path) -> dict:
    if not npz.exists():
        return {"line": None}
    z = np.load(npz, allow_pickle=False)
    out = {"line": str(npz), "decisions": int(len(z["acts"])), "act_every": int(z["act_every"]),
           "states": int(len(z["states"])), "frontier_d": float(z["frontier_d"]),
           "frontier_tick": int(z["frontier_tick"]), "replay_ok": int(z["replay_ok"]),
           "finished": int(z["finished"]) if "finished" in z.files else 0}
    if len(z["states"]):
        st = z["states"]
        o = np.asarray(st["origin"], np.float64)
        v = np.asarray(st["velocity"], np.float64)
        e = o[:, 2] + (v ** 2).sum(1) / 1600.0
        out.update(z_max=float(o[:, 2].max()), e_max=float(e.max()),
                   end=[round(float(x)) for x in o[-1]])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy-line", default=None,
                    help="a beam_frontier.npz of the greedy's own line; default: record one")
    ap.add_argument("--greedy-ticks", type=int, default=26000,
                    help="--max-ticks of the greedy recording (it ends where the greedy dies)")
    ap.add_argument("--at", default="120,130,140,150",
                    help="start seconds on the greedy line: a comma list of seconds and/or "
                         "FROM:TO:STEP ranges; a negative second counts back from the line's end "
                         "(-40:-4:2 = every 2 s over the greedy's last 40 s)")
    ap.add_argument("--window", type=float, default=30.0, help="search seconds after the start")
    ap.add_argument("--seeds", default="0", help="torch seeds (comma list)")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--greedy-envs", type=int, default=512)
    ap.add_argument("--resample-every", type=int, default=25)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--score", default="d", choices=["d", "dh"])
    ap.add_argument("--score-energy", type=float, default=1.0)
    ap.add_argument("--view-hold", type=float, nargs=2, default=None, metavar=("SIGMA", "PERIOD"))
    ap.add_argument("--view-temp", type=float, default=None)
    ap.add_argument("--keys-temp", type=float, default=None)
    ap.add_argument("--tick-ms", type=float, default=None,
                    help="seconds -> ticks; default: the greedy line's own (the checkpoint's)")
    ap.add_argument("--branch", type=int, default=0,
                    help="fork this many envs at the search's first resample boundary into held "
                         "view offsets (beam_tas --branch-at --branch-view-only: the policy's own "
                         "keys); 0 = off")
    ap.add_argument("--branch-jitter", type=float, default=20.0,
                    help="--branch: held yaw offset in degrees, uniform in [-J, J], on the "
                         "policy's own target")
    ap.add_argument("--branch-burst", type=float, default=1.5, help="--branch: seconds of burst")
    ap.add_argument("--branch-hold", type=float, default=0.4,
                    help="--branch: seconds one offset is held")
    ap.add_argument("--branch-protect", type=int, default=4,
                    help="--branch: generations the forked lineages are exempt from culling")
    ap.add_argument("--timeout", type=float, default=2400.0, help="wall cap per search (s)")
    ap.add_argument("--stop-on-finish", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    beam = ROOT / "tools" / "beam_tas.py"
    base = [py, "-u", beam, a.ckpt, "--skip-gate", "--allow-nonfinisher", "--save-frontier",
            "--route-file", "none", "--log-every", 5]

    line = Path(a.greedy_line) if a.greedy_line else out / "greedy" / "beam_frontier.npz"
    if not line.exists():
        gd = out / "greedy"
        gd.mkdir(parents=True, exist_ok=True)
        print(f"recording the greedy line -> {gd}", flush=True)
        run(base + ["--envs", 1, "--greedy-envs", 1, "--frontier-last",
                    "--max-ticks", a.greedy_ticks, "--out-dir", gd], out / "greedy.log",
            a.timeout)
    gi = line_info(line)
    print("greedy line:", json.dumps(gi), flush=True)
    if gi["line"] is None:
        raise SystemExit("no greedy line")
    with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "greedy", **gi, "t": time.time()}) + "\n")

    tick_ms = float(a.tick_ms) if a.tick_ms else None
    if tick_ms is None:
        hdr = (line.parent / "beam_frontier.jsonl")
        try:
            tick_ms = float(json.loads(open(hdr, encoding="utf-8").readline())["tick_ms"])
        except (OSError, ValueError, KeyError):
            tick_ms = 7.0
    k = int(gi["act_every"])
    n_line = int(gi["decisions"]) * k
    line_s = n_line * tick_ms / 1000.0
    starts = []
    for x in str(a.at).split(","):
        x = x.strip()
        if not x:
            continue
        if ":" in x:
            lo, hi, st = (float(u) for u in x.split(":"))
            lo, hi = (lo + line_s if lo < 0 else lo), (hi + line_s if hi < 0 else hi)
            starts += list(np.arange(lo, hi + 1e-9, st))
        else:
            v = float(x)
            starts.append(v + line_s if v < 0 else v)
    print(f"greedy line {line_s:.1f}s; starts {[round(float(v), 1) for v in starts]}", flush=True)
    for s in starts:
        t0 = int(round(s * 1000.0 / tick_ms))
        t0 -= t0 % k
        if t0 <= 0 or t0 >= n_line:
            print(f"start {s:g}s = tick {t0} is outside the greedy line ({n_line} ticks)")
            continue
        for seed in [int(x) for x in str(a.seeds).split(",") if x.strip()]:
            tag = f"s{int(round(s))}_k{seed}" + (f"_b{int(a.branch)}" if a.branch > 0 else "")
            sd = out / tag
            sd.mkdir(parents=True, exist_ok=True)
            cmd = base + ["--prefix-line", f"{line}:{t0}", "--envs", a.envs,
                          "--greedy-envs", a.greedy_envs, "--resample-every", a.resample_every,
                          "--elite-frac", a.elite_frac, "--score", a.score,
                          "--score-energy", a.score_energy, "--torch-seed", seed,
                          "--max-ticks", t0 + int(round(a.window * 1000.0 / tick_ms)),
                          "--out-dir", sd]
            if a.view_hold:
                cmd += ["--view-hold", a.view_hold[0], int(a.view_hold[1])]
            if a.view_temp is not None:
                cmd += ["--view-temp", a.view_temp]
            if a.keys_temp is not None:
                cmd += ["--keys-temp", a.keys_temp]
            if a.branch > 0:
                tps = 1000.0 / tick_ms
                cmd += ["--branch-at", f"t{t0 + a.resample_every * k}:{int(a.branch)}",
                        "--branch-view-only", "--branch-jitter", a.branch_jitter,
                        "--branch-burst", int(round(a.branch_burst * tps)),
                        "--branch-hold", int(round(a.branch_hold * tps)),
                        "--branch-protect", int(a.branch_protect)]
            print(f"search {tag}: start tick {t0} ({s:g}s), window {a.window:g}s", flush=True)
            t1 = time.time()
            rc = run(cmd, out / f"{tag}.log", a.timeout)
            info = line_info(sd / "beam_frontier.npz")
            try:
                summ = json.loads((sd / "summary.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summ = {}
            row = {"kind": "search", "tag": tag, "start_s": s, "start_tick": t0, "seed": seed,
                   "rc": rc, "wall_s": round(time.time() - t1), "crossed": summ.get("crossed"),
                   "best_s": summ.get("best_s"), **info}
            print(json.dumps(row), flush=True)
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if a.stop_on_finish and info.get("finished"):
                print(f"FINISH found by {tag}: {info['line']}", flush=True)
                return


if __name__ == "__main__":
    main()
