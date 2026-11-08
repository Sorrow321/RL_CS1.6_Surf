"""ladder.py - time-to-wall of a run against autoresearch/ladder.json (the 2x stop rule).

The unified-recipe program (2026-11-08) judges a recipe by WHEN its greedy evals from the true
start first get past each wall of the map, against the October lineage that solved it one wall
at a time. A wall counts as cleared in an eval when the eval's best episode (score.py's
best_progress_pct over the first config.score_episodes episodes) exceeds the wall's progress, or
the eval entered the wall's checkpoint curtain (score.py's max_cp), whichever ladder.json names.

    python autoresearch/ladder.py $AR_HOME/runs/$AR_AGENT/<run>/      # one run
    python autoresearch/ladder.py --json <run dir> [<run dir> ...]

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


def ladder(run: Path) -> dict:
    ev = evals(run)
    scored = [(st, score_any(p)) for st, p in ev]
    newest = ev[-1][0] if ev else 0
    rows = []
    for w in LADDER["walls"]:
        hit = next((st for st, s in scored if cleared(s, w)), None)
        dl = int(w["deadline_step"])
        if hit is not None:
            verdict = "ON TIME" if hit <= dl else "LATE"
        else:
            verdict = "FAILED" if newest > dl else "PENDING"
        rows.append({"wall": w["name"], "cleared_at": hit, "deadline": dl,
                     "october": int(w["october_step"]), "verdict": verdict})
    best = max((s for _, s in scored if s.get("score") is not None),
               key=lambda s: s["score"], default=None)
    return {"run": str(run), "evals": len(ev), "newest_step": newest, "walls": rows,
            "alive": not any(r["verdict"] in ("FAILED", "LATE") for r in rows),
            "best": None if best is None else {k: best.get(k) for k in
                                                ("file", "score", "best_progress_pct",
                                                 "finishes", "max_cp")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories holding traj_<step>.jsonl evals")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = [ladder(Path(r)) for r in a.runs]
    if a.json:
        print(json.dumps(out, indent=1))
        return
    for r in out:
        print(f"{r['run']}: {r['evals']} evals, newest step {r['newest_step']:,}, "
              f"{'alive' if r['alive'] else 'STOP: a wall was missed'}")
        for w in r["walls"]:
            at = f"{w['cleared_at']:,}" if w["cleared_at"] is not None else "-"
            print(f"  {w['wall']:<24} cleared {at:>16}   deadline {w['deadline']:>16,}   "
                  f"(October {w['october']:,})  {w['verdict']}")
        if r["best"]:
            b = r["best"]
            print(f"  best eval: score {b['score']}  progress {b['best_progress_pct']}%  "
                  f"CP{b['max_cp']}  finishes {b['finishes']}")


if __name__ == "__main__":
    main()
