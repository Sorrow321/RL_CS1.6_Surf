#!/usr/bin/env python3
"""wr_metric.py - the world-record metric of a run, from its greedy evals.

At every eval step, over the run's maps:

  finished   maps with at least one finish the engine would fire (score.py's count: the
             finish trigger's clip hull when it is not a plain box, surfgym.goalhull)
  beat       maps with a comparable record whose best finish on the RECORD clock (from the
             last touch of the start trigger, surfgym.wrclock) is below the record
  beat ever  the same over every eval so far (the best time a map has ever posted)
  time/WR    the median of best record-clock time / record over the comparable maps that
             finished at that eval - it moves long before any record falls

"Comparable" = the map has a record in the snapshot AND both its start and its finish are brush
triggers (a button finish is a padded box the simulator arrives in, CLAUDE.md 4b). Only the
first --episodes episodes of each eval count (score.py's score_episodes, 9 = what the trainer
records). Reads each map's bsp + zones.json (no field) and the trajectories.

    python3 autoresearch/wr_metric.py runs/m20_ctl --maps-root maps20/ --records records.json
    python3 autoresearch/wr_metric.py RUN --variant ema0 --csv wr.csv     # the EMA weights' evals

--records: a snapshot from `tools/leaderboard.py snapshot` ({"maps": {stem: {"wr_s": ...}}});
without it the committed docs/leaderboards/<stem>.json are read.

The caveats travel with the number: our physics is the C core (not the game; a line nobody
has replayed in the engine is not a record), the record servers ran maxvelocity 3500 at ~131 fps,
and a record with few entries (sidistic: one) is a weak target.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from surfgym.goalhull import for_zone          # noqa: E402
from surfgym.mapfleet import map_tag            # noqa: E402
from surfgym.wrclock import (StartZone, grade, load_records, read_episodes,  # noqa: E402
                             summarise, _is_button)


def run_maps(run_dir: Path) -> list:
    cfg = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    cfg = cfg.get("config", cfg)
    maps = cfg.get("maps")
    return [str(m) for m in maps] if maps else [str(cfg["map"])]


def eval_files(run_dir: Path, tag: str | None, variant: str = "") -> dict:
    """{step: path} of the run's greedy evals of one map (traj_<step>[_<tag>][_<variant>].jsonl,
    .gz too, the plain file winning a tie); never the record gate's _rec_ / _stoch / _mirror."""
    pat = re.compile(r"traj_(\d+)" + (f"_{re.escape(tag)}" if tag else "")
                     + (f"_{re.escape(variant)}" if variant else "") + r"\.jsonl(\.gz)?")
    out = {}
    for p in sorted(run_dir.glob("traj_*.jsonl*"), key=lambda q: q.name.endswith(".gz")):
        m = pat.fullmatch(p.name)
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = p
    return out


def first_episodes(eps, n: int):
    """score.py's rule: header episode index < n, else the first n in file order."""
    if n <= 0:
        return eps
    if all(isinstance(h.get("episode"), int) for h, _, _ in eps):
        return [e for e in eps if e[0]["episode"] < n]
    return eps[:n]


def find_bsp(stem: str, roots) -> Path | None:
    for r in roots:
        for p in (Path(r) / f"{stem}.bsp", Path(r) / "maps" / f"{stem}.bsp"):
            if p.exists():
                return p
    return None


def records_for(stems, path) -> dict:
    if path:
        return load_records(path)
    from leaderboard import load_board
    out = {}
    for s in stems:
        b = load_board(s)
        if b and b.get("wr_s") is not None:
            out[s] = float(b["wr_s"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run directory (run.json + traj_*.jsonl)")
    ap.add_argument("--maps-root", action="append", default=[],
                    help="where <stem>.bsp + <stem>.zones.json live (repeatable; maps/ of the "
                         "repo is always tried last)")
    ap.add_argument("--records", default=None, help="a records snapshot JSON")
    ap.add_argument("--variant", default="", help="eval file suffix, e.g. ema0")
    ap.add_argument("--episodes", type=int, default=9)
    ap.add_argument("--csv", default=None, help="write the per-step table here")
    ap.add_argument("--last", type=int, default=0, help="print only the last N steps")
    a = ap.parse_args(argv)

    run_dir = Path(a.run)
    stems = run_maps(run_dir)
    multi = len(stems) > 1
    roots = a.maps_root + [str(REPO / "maps")]
    wr = records_for(stems, a.records)
    per_map, comparable, why = {}, {}, {}
    for s in stems:
        bsp = find_bsp(s, roots)
        if bsp is None:
            raise SystemExit(f"{s}.bsp not found under {roots}")
        zones = json.loads(bsp.with_suffix(".zones.json").read_text(encoding="utf-8"))
        sz = StartZone.for_map(bsp)
        goal = for_zone(bsp, zones["end"])
        trig = (not _is_button(zones["end"], zones)) and sz is not None and sz.trigger
        comparable[s] = bool(trig and s in wr)
        why[s] = ("" if comparable[s] else "no record" if s not in wr
                  else "button / no start trigger")
        res = {}
        for step, f in eval_files(run_dir, map_tag(s) if multi else None, a.variant).items():
            eps = first_episodes(read_episodes(f), a.episodes)
            if not eps or eps[0][0].get("map") != s:
                continue
            res[step] = grade(eps, sz, goal)
        per_map[s] = res

    steps = sorted(set().union(*[set(r) for r in per_map.values()])) if per_map else []
    best_ever = {s: None for s in stems}
    rows = []
    for st in steps:
        rec_now, fin_now = {}, 0
        for s in stems:
            g = per_map[s].get(st)
            t = g["best_record_s"] if g else None
            rec_now[s] = t
            if g and g["verified"]:
                fin_now += 1
            if t is not None and (best_ever[s] is None or t < best_ever[s]):
                best_ever[s] = t
        now = summarise(rec_now, wr, comparable)
        ever = summarise(dict(best_ever), wr, comparable)
        rows.append({"step": st, "n_maps": len(stems), "finished": fin_now,
                     "comparable": now["n_comparable"], "beat": now["n_beat"],
                     "beat_ever": ever["n_beat"], "ratio_median": now["ratio_median"],
                     **{f"rec_s.{map_tag(s)}": rec_now[s] for s in stems}})

    show = rows[-a.last:] if a.last else rows
    print(f"run {run_dir}  ({len(stems)} maps, {len(steps)} evals"
          + (f", variant {a.variant}" if a.variant else "") + ")")
    print(f"{'step':>15s}  {'finished':>9s}  {'beat WR':>8s}  {'ever':>5s}  {'time/WR':>8s}")
    for r in show:
        print(f"{r['step']:>15,d}  {r['finished']:>4d}/{r['n_maps']:<4d}  "
              f"{r['beat']:>3d}/{r['comparable']:<4d}  {r['beat_ever']:>5d}  "
              + (f"{r['ratio_median']:8.3f}" if r["ratio_median"] == r["ratio_median"] else
                 f"{'-':>8s}"))
    print()
    print(f"{'map':28s} {'record':>8s}  {'our best (record clock)':>24s}  {'x WR':>6s}  note")
    for s in stems:
        t = best_ever[s]
        at = None
        if t is not None:
            at = min(st for st, g in per_map[s].items() if g["best_record_s"] == t)
        print(f"{s:28s} {('%.2f' % wr[s]) if s in wr else '-':>8s}  "
              + (f"{t:8.2f} s at {at:>13,d}" if t is not None else f"{'-':>24s}")
              + (f"  {t / wr[s]:6.3f}" if (t is not None and s in wr) else f"  {'-':>6s}")
              + ("  BEATS THE RECORD" if (comparable[s] and t is not None and t < wr[s]) else "")
              + (f"  ({why[s]})" if why[s] else ""))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["step"])
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if (v is None or (isinstance(v, float) and v != v)) else v)
                            for k, v in r.items()})
    return rows


if __name__ == "__main__":
    main()
