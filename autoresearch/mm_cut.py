"""mm_cut.py - the cut rule of a --maps pilot (mm/s1, 2026-11-16; P1's numbers by default).

Reads a run directory - on the box (runs/<run>) or a pulled copy - and decides, from the grader
(autoresearch/score_maps.py: every map's greedy evals on its own config) and progress.csv:

  STOP (exit 11), early, a broken or hopeless run:
    * throughput: past --fps-after-s (30 min) of training, the run's steps/s (the trainer's
      time/fps: fleet steps over wall time since the launch) is below --min-fps (150k);
    * no map has finished by --no-finish-by fleet steps (3.0B).
  CUT (exit 10), the run is done:
    * the fleet step reached --cut-steps (6.0B), or
    * EVERY map has finished at least once and has --after-finish-evals (5) of its own evals
      after its first finishing one.
  CONTINUE (exit 0) otherwise. Exit 2: nothing to read yet.

A finish is the grader's (a done episode inside the finish trigger), never the trainer's count.
The per-map rows also give each map's first finish at the fleet step and at its OWN step (D2),
the number to set against the single-map v14 runs.

Every rule is a flag (0 turns a step or count rule off). --mm7 (the seven-map run: no step cap,
stop rules only for a broken run) sets --cut-steps 0 --no-finish-by 0 --after-finish-evals 0
--min-fps 120000 --stale-min 10 --refs autoresearch/map_configs/v14_alone.json, and adds:
  * STOP for a dead trainer: --pid-file gone (on the box: runs/<run>.pid), or progress.csv not
    written for --stale-min minutes (CLAUDE.md: stationary > 10 min = failed);
  * STOP at the credit floor: --credit X below --credit-floor Y (config budget_reserve_usd);
  * the read against v14 alone (--refs): each map's first finish in its own steps, and pb's
    milestones (room exit, past the x -1,005 wall, the shaft, the finish) by the same rule as
    the v14-alone references were computed with;
  * --log runs/<run>_launch.txt: per map, the searches started and how long its requests waited
    in the queue (fleet steps, own steps, minutes).

    python autoresearch/mm_cut.py runs/mmp1_1 [--root DIR] [--json]
    python autoresearch/mm_cut.py runs/mm7_1 --mm7 --pid-file runs/mm7_1.pid \
        --log runs/mm7_1_launch.txt [--credit 41.5 --credit-floor 10]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_maps  # noqa: E402

CONTINUE, CUT, STOP, NOTHING = 0, 10, 11, 2


def progress(run_dir: Path) -> dict:
    """(fleet step, fps, hours) of the last progress.csv row; fps is the trainer's time/fps
    (fleet steps since the launch over wall time), so hours = (step - step0) / fps."""
    p = Path(run_dir) / "progress.csv"
    if not p.exists():
        return {}
    last = None
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("time/total_timesteps") and row.get("time/fps"):
                last = row
    if last is None:
        return {}
    step, fps = int(float(last["time/total_timesteps"])), float(last["time/fps"])
    return {"step": step, "fps": fps}


def queue_waits(log_path, n_maps):
    """{map tag: {"searches", "waited" (fleet steps, summed), "max_wait"}} from the launch log:
    a request's wait runs from its first "request queued" line to its map's next search start"""
    out, pend = {}, {}
    pat_q = re.compile(r"\[\s*([\d,]+)\] auto-spine\[([^\]]+)\]: (?:quorum|patience) request queued")
    pat_s = re.compile(r"\[\s*([\d,]+)\] auto-spine\[([^\]]+)\] search \d+: running beside")
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = pat_q.search(line)
        if m:
            pend.setdefault(m.group(2), int(m.group(1).replace(",", "")))
            continue
        m = pat_s.search(line)
        if m:
            tag, step = m.group(2), int(m.group(1).replace(",", ""))
            w = step - pend.pop(tag, step)
            r = out.setdefault(tag, {"searches": 0, "waited": 0, "max_wait": 0})
            r["searches"] += 1
            r["waited"] += w
            r["max_wait"] = max(r["max_wait"], w)
    for tag, since in pend.items():
        out.setdefault(tag, {"searches": 0, "waited": 0, "max_wait": 0})["pending_since"] = since
    return out


def against_refs(rows, refs, n_maps):
    """lines: each map's first finish in its own steps against v14 alone, and the milestones
    (best grader progress above a threshold, or the first finish) of the maps that list them"""
    lines = []
    for r in rows:
        ref = refs.get(r["map"]) or {}
        ev = r.get("evals", [])
        own = (lambda st: None if st is None else st // n_maps)
        ff = next((e[0] for e in ev if e[1] > 0), None)
        alone = ref.get("first_finish")
        lines.append(f"{r['map']}: first finish "
                     + (f"{own(ff):,} own" if ff is not None else "not yet")
                     + (f" | v14 alone {alone:,}" if alone else "")
                     + (f" | x{own(ff) / alone:.2f}" if ff is not None and alone else ""))
        for name, thr, alone_m in ref.get("milestones", []):
            hit = next((e[0] for e in ev if (e[1] > 0 if thr is None else e[3] > thr)), None)
            rule = "a finish" if thr is None else f"best > {thr:g}%"
            lines.append(f"   {name} ({rule}): "
                         + (f"{own(hit):,} own" if hit is not None else "not yet")
                         + (f" | v14 alone {alone_m:,}" if alone_m else "")
                         + (f" | x{own(hit) / alone_m:.2f}" if hit is not None and alone_m else ""))
        if ref.get("caveat"):
            lines.append("   (" + ref["caveat"] + ")")
    return lines


def decide(rows, prog, a, step0=0) -> tuple:
    """(code, reasons, notes) from score_maps rows and the last progress row"""
    step = max([prog.get("step", 0)] + [e[0] for r in rows for e in r.get("evals", [])])
    fps = prog.get("fps")
    # the wall time is the progress row's own: its step over its fps (the trainer's average)
    secs = (prog.get("step", 0) - step0) / fps if fps else 0.0
    notes = [f"fleet step {step:,}" + (f", {fps:,.0f} steps/s over {secs / 3600:.2f} h"
                                       if fps else "")]
    first = {}
    after = {}
    for r in rows:
        ev = r.get("evals", [])
        k = next((i for i, e in enumerate(ev) if e[1] > 0), None)
        first[r["map"]] = None if k is None else ev[k][0]
        after[r["map"]] = 0 if k is None else len(ev) - 1 - k
        notes.append(f"{r['map']}: {len(ev)} valid evals, "
                     + ("no finish" if k is None else
                        f"first finish at {ev[k][0]:,} ({r.get('first_finish_own_step') or 0:,} "
                        f"own), {after[r['map']]} evals after it"))
    if getattr(a, "dead", None):
        return STOP, [a.dead], notes
    if getattr(a, "credit", None) is not None and a.credit < a.credit_floor:
        return STOP, [f"credit ${a.credit:.2f} < the floor ${a.credit_floor:.2f}"], notes
    if fps and secs >= a.fps_after_s and fps < a.min_fps:
        return STOP, [f"{fps:,.0f} steps/s < {a.min_fps:,.0f} after {secs / 60:.0f} min"], notes
    if a.no_finish_by > 0 and step >= a.no_finish_by and \
            not any(v is not None for v in first.values()):
        return STOP, [f"no map finished by {a.no_finish_by:,.0f} fleet steps"], notes
    if a.cut_steps > 0 and step >= a.cut_steps:
        return CUT, [f"fleet step {step:,} >= {a.cut_steps:,.0f}"], notes
    if a.after_finish_evals > 0 and rows and all(
            first[r["map"]] is not None and after[r["map"]] >= a.after_finish_evals
            for r in rows):
        return CUT, [f"every map finished, each with >= {a.after_finish_evals} evals after its "
                     "first finish"], notes
    return CONTINUE, [], notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--root", default=None, help="the maps' directory root for the grader")
    ap.add_argument("--cut-steps", type=float, default=6.0e9)
    ap.add_argument("--after-finish-evals", type=int, default=5)
    ap.add_argument("--min-fps", type=float, default=150e3)
    ap.add_argument("--fps-after-s", type=float, default=1800.0)
    ap.add_argument("--no-finish-by", type=float, default=3.0e9)
    ap.add_argument("--step0", type=float, default=0.0,
                    help="the fleet step the launch started at (0 from scratch; a resume's)")
    ap.add_argument("--mm7", action="store_true",
                    help="the seven-map run's rules: no step cap, stop only for a broken run")
    ap.add_argument("--stale-min", type=float, default=0.0,
                    help="STOP when progress.csv is older than this many minutes (0 = off)")
    ap.add_argument("--pid-file", default=None, help="STOP when this pid file's process is gone")
    ap.add_argument("--credit", type=float, default=None, help="the account's credit now ($)")
    ap.add_argument("--credit-floor", type=float, default=10.0)
    ap.add_argument("--refs", default=None, help="v14-alone references (JSON) for the read")
    ap.add_argument("--log", default=None, help="the launch log: per-map queue waits")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.mm7:
        given = set(argv if argv is not None else sys.argv[1:])
        for flag, val in (("--cut-steps", 0.0), ("--no-finish-by", 0.0),
                          ("--after-finish-evals", 0), ("--min-fps", 120e3),
                          ("--stale-min", 10.0),
                          ("--refs", str(HERE / "map_configs" / "v14_alone.json"))):
            if flag not in given:
                setattr(a, flag[2:].replace("-", "_"), val)
    run = Path(a.run_dir)
    if not (run / "run.json").exists():
        print(f"mm_cut: no run.json in {run} yet")
        return NOTHING
    a.dead = None
    if a.pid_file:
        try:
            os.kill(int(Path(a.pid_file).read_text().strip()), 0)
        except (OSError, ValueError):
            a.dead = f"the trainer is gone ({a.pid_file})"
    pc = run / "progress.csv"
    if a.dead is None and a.stale_min > 0 and pc.exists() and \
            time.time() - pc.stat().st_mtime > 60.0 * a.stale_min:
        a.dead = (f"progress.csv not written for {(time.time() - pc.stat().st_mtime) / 60:.0f} "
                  f"min (> {a.stale_min:g})")
    rows = score_maps.grade(run, score_maps.configs_of_run(run), a.root)
    prog = progress(run)
    code, why, notes = decide(rows, prog, a, int(a.step0))
    n_maps = max(1, len(rows))
    if a.refs:
        notes += ["v14 alone (own steps):"] + against_refs(
            rows, json.loads(Path(a.refs).read_text(encoding="utf-8")), n_maps)
    if a.log and Path(a.log).exists():
        fps = prog.get("fps") or 0.0
        for tag, q in sorted(queue_waits(a.log, n_maps).items()):
            notes.append(f"queue {tag}: {q['searches']} searches, waited {q['waited']:,} fleet "
                         f"({q['waited'] // n_maps:,} own"
                         + (f", {q['waited'] / fps / 60:.0f} min" if fps else "")
                         + f"), longest {q['max_wait']:,}"
                         + (f", a request pending since {q['pending_since']:,}"
                            if "pending_since" in q else ""))
    verdict = {CONTINUE: "CONTINUE", CUT: "CUT", STOP: "STOP"}[code]
    if a.json:
        print(json.dumps({"verdict": verdict, "reasons": why, "notes": notes,
                          "progress": prog, "maps": rows}, indent=1, default=str))
    else:
        print(f"{verdict}" + (": " + "; ".join(why) if why else ""))
        for n in notes:
            print("   " + n)
    return code


if __name__ == "__main__":
    sys.exit(main())
