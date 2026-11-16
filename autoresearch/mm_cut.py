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

    python autoresearch/mm_cut.py runs/mmp1_1 [--root DIR] [--json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
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
    if fps and secs >= a.fps_after_s and fps < a.min_fps:
        return STOP, [f"{fps:,.0f} steps/s < {a.min_fps:,.0f} after {secs / 60:.0f} min"], notes
    if step >= a.no_finish_by and not any(v is not None for v in first.values()):
        return STOP, [f"no map finished by {a.no_finish_by:,.0f} fleet steps"], notes
    if step >= a.cut_steps:
        return CUT, [f"fleet step {step:,} >= {a.cut_steps:,.0f}"], notes
    if rows and all(first[r["map"]] is not None and after[r["map"]] >= a.after_finish_evals
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
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    if not (run / "run.json").exists():
        print(f"mm_cut: no run.json in {run} yet")
        return NOTHING
    rows = score_maps.grade(run, score_maps.configs_of_run(run), a.root)
    prog = progress(run)
    code, why, notes = decide(rows, prog, a, int(a.step0))
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
