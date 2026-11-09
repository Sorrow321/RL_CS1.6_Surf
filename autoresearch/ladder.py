"""ladder.py - time-to-wall of a run against autoresearch/ladder.json (the 2x stop rule).

The one-search program (2026-11-09) judges a recipe by WHEN its greedy evals from the true start
first get past each wall of the map, against run 5 of the unified-recipe program (unr1b3_mcts5,
the hybrid that finished the map). A wall counts as cleared in an eval when the eval's best
episode (score.py's best_progress_pct over the first config.score_episodes episodes) exceeds the
wall's progress, or the eval entered the wall's checkpoint curtain (score.py's max_cp), whichever
ladder.json names.

    python autoresearch/ladder.py $AR_HOME/runs/$AR_AGENT/<run>/      # one run, trainer steps
    python autoresearch/ladder.py --total <run dir>                   # total simulator ticks
    python autoresearch/ladder.py --json <run dir> [<run dir> ...]

TWO units (ladder.json). Without a switch, a wall's time is the trainer's step counter (the
simulator ticks of the PPO rollouts) against run5_step / deadline_step. With --total it is that
step PLUS the ticks of every search the run had started by then, read from the run directory's
search_ticks.json (autoresearch/search_ticks.py writes it), against run5_total / deadline_total.
--total-is-step is for a trainer whose step counter already counts every simulator tick it
spends, search included (a run with no separate rollouts).

Verdict per wall: CLEARED at <step> (ON TIME when step <= deadline, else LATE), or PENDING while
the run's newest eval is below the deadline, or FAILED once the newest eval is past the deadline
with the wall not cleared. A run with a FAILED wall is stopped (PROTOCOL.md). Steps are the
trainer's global step (the traj_<step>.jsonl name), so a crash-resume of the same run keeps
counting; a warm start from another recipe's checkpoint does not reset it and is not a valid run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score  # noqa: E402  (the grader: same field, same validation, same episode count)
import search_ticks  # noqa: E402

LADDER = json.loads((HERE / "ladder.json").read_text(encoding="utf-8"))


def evals(run: Path):
    """[(step, path)] of the run's greedy evals, oldest first (.jsonl or .jsonl.gz)."""
    out = []
    for p in run.iterdir():
        m = re.fullmatch(r"traj_(\d+)\.jsonl(\.gz)?", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def score_any(path: Path) -> dict:
    if path.suffix != ".gz":
        return score.score_file(path)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / path.name[:-3]
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return score.score_file(tmp)


def cleared(s: dict, wall: dict) -> bool:
    if s.get("score") is None:
        return False
    if "progress_pct" in wall and s["best_progress_pct"] > float(wall["progress_pct"]):
        return True
    if "cp" in wall and int(s.get("max_cp") or 0) >= int(wall["cp"]):
        return True
    if wall.get("finish") and int(s.get("finishes") or 0) > 0:
        return True
    return False


def search_rows(run: Path):
    """The run's searches [{start_step, ticks}], or None when search_ticks.json is missing."""
    p = run / "search_ticks.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["searches"]


def ladder(run: Path, total: bool = False, total_is_step: bool = False) -> dict:
    """The run's wall table. ``total``: in total simulator ticks (trainer step + the ticks of
    every search started at or before it); ``total_is_step``: the step already is that total."""
    ev = evals(run)
    scored = [(st, score_any(p)) for st, p in ev]
    newest = ev[-1][0] if ev else 0
    unit, ref_key, dl_key = "steps", "run5_step", "deadline_step"
    x = lambda st: st                                       # noqa: E731
    if total or total_is_step:
        unit, ref_key, dl_key = "total ticks", "run5_total", "deadline_total"
        if not total_is_step:
            rows_s = search_rows(run)
            if rows_s is None:
                raise SystemExit(f"{run}: no search_ticks.json - write it with "
                                 "autoresearch/search_ticks.py (see its docstring), or pass "
                                 "--total-is-step if the trainer's step counts every tick")
            x = lambda st: st + search_ticks.cumulative(rows_s, st)   # noqa: E731
    newest_x = x(newest)
    rows = []
    for w in LADDER["walls"]:
        hit = next((st for st, s in scored if cleared(s, w)), None)
        dl = int(w[dl_key])
        if hit is not None:
            verdict = "ON TIME" if x(hit) <= dl else "LATE"
        else:
            verdict = "FAILED" if newest_x > dl else "PENDING"
        rows.append({"wall": w["name"], "cleared_at": None if hit is None else x(hit),
                     "cleared_step": hit, "deadline": dl, "run5": int(w[ref_key]),
                     "verdict": verdict})
    best = max((s for _, s in scored if s.get("score") is not None),
               key=lambda s: s["score"], default=None)
    return {"run": str(run), "unit": unit, "evals": len(ev), "newest_step": newest,
            "newest": newest_x, "walls": rows,
            "alive": not any(r["verdict"] in ("FAILED", "LATE") for r in rows),
            "best": None if best is None else {k: best.get(k) for k in
                                                ("file", "score", "best_progress_pct",
                                                 "finishes", "max_cp")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories holding traj_<step>.jsonl evals")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--total", action="store_true",
                    help="total simulator ticks: trainer step + search ticks (search_ticks.json)")
    ap.add_argument("--total-is-step", action="store_true",
                    help="the trainer's step counter already counts every simulator tick")
    a = ap.parse_args(argv)
    out = [ladder(Path(r), total=a.total, total_is_step=a.total_is_step) for r in a.runs]
    if a.json:
        print(json.dumps(out, indent=1))
        return
    for r in out:
        print(f"{r['run']}: {r['evals']} evals, newest {r['newest']:,} {r['unit']}, "
              f"{'alive' if r['alive'] else 'STOP: a wall was missed'}")
        for w in r["walls"]:
            at = f"{w['cleared_at']:,}" if w["cleared_at"] is not None else "-"
            print(f"  {w['wall']:<24} cleared {at:>16}   deadline {w['deadline']:>16,}   "
                  f"(run 5 {w['run5']:,})  {w['verdict']}")
        if r["best"]:
            b = r["best"]
            print(f"  best eval: score {b['score']}  progress {b['best_progress_pct']}%  "
                  f"CP{b['max_cp']}  finishes {b['finishes']}")


if __name__ == "__main__":
    main()
