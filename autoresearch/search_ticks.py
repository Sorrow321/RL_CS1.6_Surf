"""search_ticks.py - the simulator ticks a run's searches spent, by the trainer step they started at.

The one-search program (2026-11-09) compares recipes in TWO units: the trainer's step counter
(simulator ticks spent in PPO rollouts) and the TOTAL simulator ticks, rollouts plus every search
the run ran beside them. A recipe that moves work from the rollouts into the search is only
comparable in the second unit. This file counts the search half from the run's own files:

* each search's ticks = search_wall_s x env_steps_per_s of `<run>/auto/k<NN>/beam/summary.json`
  (beam_tas's summary; for the MCTS this equals waves x envs x macro_decisions x act_every);
* each search's start step = the first `[<step>] auto-spine search <N>:` line of the launch log.

    # on the box (stdlib only), one line "SEARCH_TICKS {...}" on stdout:
    python3 autoresearch/search_ticks.py runs/<run>
    # from the droplet, into the pulled run directory, where ladder.py --total reads it:
    python autoresearch/box.py ssh --agent $AR_AGENT -- \
        "cd /root/RL_Surf && python3 autoresearch/search_ticks.py runs/<run>" \
        | python autoresearch/search_ticks.py --save $AR_HOME/runs/$AR_AGENT/<run>/search_ticks.json

A search with no summary yet (still running, or it died) is counted at the median of the run's
finished searches and flagged "estimated". A run whose searches do not go through --auto-spine
writes the same file itself: {"searches": [{"k": 1, "start_step": 0, "ticks": 0}, ...]}.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

START = re.compile(r"^\[\s*([\d,]+)\] auto-spine search (\d+):")


def starts(launch_log: Path) -> dict:
    """{search number: the trainer step of its first launch-log line}."""
    out = {}
    with open(launch_log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = START.match(line)
            if m and int(m.group(2)) not in out:
                out[int(m.group(2))] = int(m.group(1).replace(",", ""))
    return out


def one(summary: Path):
    """Ticks of one finished search, or None when its summary does not say."""
    try:
        d = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if d.get("search_wall_s") and d.get("env_steps_per_s"):
        return int(round(float(d["search_wall_s"]) * float(d["env_steps_per_s"])))
    return None


def collect(run: Path, launch_log: Path | None = None) -> dict:
    run = run.resolve()
    if launch_log is None:
        cands = [run.parent / f"{run.name}_launch.txt", run / f"{run.name}_launch.txt"]
        launch_log = next((p for p in cands if p.exists()), None)
        if launch_log is None:
            raise SystemExit(f"no launch log beside or inside {run} (give --launch-log)")
    st = starts(launch_log)
    known = {}
    for p in sorted((run / "auto").glob("k*/beam/summary.json")):
        m = re.fullmatch(r"k(\d+)", p.parent.parent.name)
        t = one(p)
        if m and t is not None:
            known[int(m.group(1))] = t
    med = int(statistics.median(known.values())) if known else 0
    rows = []
    for k in sorted(st):
        rows.append({"k": k, "start_step": st[k], "ticks": known.get(k, med),
                     "estimated": k not in known})
    return {"run": run.name, "searches": rows, "total": sum(r["ticks"] for r in rows)}


def cumulative(rows, step: int) -> int:
    """Ticks of every search that had STARTED at or before `step`."""
    return sum(int(r["ticks"]) for r in rows if int(r["start_step"]) <= step)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="the run directory (holding auto/k*/beam/summary.json)")
    ap.add_argument("--launch-log", default=None)
    ap.add_argument("--save", default=None, metavar="FILE",
                    help="read a SEARCH_TICKS line from stdin and write its JSON to FILE")
    a = ap.parse_args(argv)
    if a.save:
        line = next((ln for ln in sys.stdin if ln.startswith("SEARCH_TICKS ")), None)
        if line is None:
            raise SystemExit("no SEARCH_TICKS line on stdin")
        d = json.loads(line[len("SEARCH_TICKS "):])
        Path(a.save).write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f"{a.save}: {len(d['searches'])} searches, {d['total']:,} ticks")
        return
    if not a.run:
        ap.error("give the run directory, or --save FILE")
    d = collect(Path(a.run), Path(a.launch_log) if a.launch_log else None)
    print("SEARCH_TICKS " + json.dumps(d))


if __name__ == "__main__":
    main()
