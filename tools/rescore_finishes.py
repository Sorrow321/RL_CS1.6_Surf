#!/usr/bin/env python3
"""rescore_finishes.py - repair the box-finish columns of an existing run.

Until 2026-09-11 (commit 3a1349e) ``race_coverage`` swept the recorded path
against the RAW finish box, and the recorder's last row is the tick BEFORE
the crossing, so ``race/eval_finishes.<tag>``, ``race/maps_finished`` and
``race/maps_finished_trigger`` read 0 on 9/9 finishers (petrus's trigger is a
2 u curtain). The trainer is fixed; this recomputes those columns for a run
already on disk from its own eval recordings, with the fixed test (the
env's hull-inflated box plus the unrecorded finishing tick), and rewrites
progress.csv in place after saving a backup. Nothing else in the file moves.

    python tools/rescore_finishes.py --run runs/jtANCHU            # dry run
    python tools/rescore_finishes.py --run runs/jtANCHU --apply

Only the true-start eval recordings are read (``traj_<step>.jsonl`` and
``traj_<step>_<tag>.jsonl``); the dashboard's record-button files
(``_reservoir``, ``_mixed``, ``_stoch``) are not evals and are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goalfield import GoalField                    # noqa: E402
from surfgym.mapfleet import map_tag                       # noqa: E402
from train_fast import race_coverage                       # noqa: E402


def load_field(maps_dir: Path, stem: str, cell) -> GoalField:
    p = maps_dir / f"{stem}.goal_{float(cell):g}.npz"
    if not p.is_file():
        raise SystemExit(f"no baked field {p}")
    z = np.load(p, allow_pickle=False)
    grid = z["grid"].astype(np.float32) * float(z["quant"])
    return GoalField(grid, z["mins"], float(z["cell"]), float(z["reach_max"]))


def finish_kind(zones: dict, goal_box: dict) -> str:
    """train_fast.py's own rule (the trigger / button split of CLAUDE.md 4b)."""
    return ("button"
            if (goal_box.get("true_aabb") is not None
                or goal_box.get("from") == "func_button"
                or zones.get("source") == "gateway")
            else "trigger")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--maps-dir", default=str(ROOT / "maps"))
    ap.add_argument("--apply", action="store_true",
                    help="rewrite progress.csv (a timestamped backup is kept)")
    a = ap.parse_args()
    run = Path(a.run)
    maps_dir = Path(a.maps_dir)
    cfg = json.loads((run / "run.json").read_text(encoding="utf-8"))
    cfg = cfg.get("config", cfg)
    stems = list(cfg.get("maps") or [cfg["map"]])
    multi = len(stems) > 1
    cells = cfg.get("goal_cells") or {}
    tags = {map_tag(s): s for s in stems}

    fields, boxes, kinds = {}, {}, {}
    for s in stems:
        t = map_tag(s)
        # a single-map run.json carries goal_cell (often null = the 32 u
        # default); a joint one a per-tag goal_cells dict
        cell = (cells.get(t) or cells.get(s) or cfg.get("goal_cell")
                or cfg.get("lidar_cell") or 32)
        if isinstance(cell, str):
            cell = float(cell.split(",")[0])
        fields[t] = load_field(maps_dir, s, cell)
        zones = json.loads((maps_dir / f"{s}.zones.json").read_text(
            encoding="utf-8"))
        boxes[t] = zones["end"]
        kinds[t] = finish_kind(zones, zones["end"])

    # the eval recordings: step -> tag -> (n_box, n_ep)
    pat = re.compile(r"^traj_(\d{10})(?:_([A-Za-z0-9_]+))?\.jsonl$")
    per_step: dict = {}
    for p in sorted(run.glob("traj_*.jsonl")):
        m = pat.match(p.name)
        if not m:
            continue
        step, tag = int(m.group(1)), m.group(2)
        if multi:
            if tag not in tags:
                continue                      # a record-button file
        else:
            if tag is not None:
                continue
            tag = map_tag(stems[0])
        _, n_ep, n_box = race_coverage(p, fields[tag], boxes[tag])
        per_step.setdefault(step, {})[tag] = (n_box, n_ep)

    if not per_step:
        raise SystemExit("no eval recordings found")
    steps = sorted(per_step)
    print(f"{run.name}: {len(steps)} eval steps, maps {list(tags)}")
    for st in steps:
        print(f"  {st:>13,d}  " + "  ".join(
            f"{t} box {nb}/{ne}" for t, (nb, ne) in per_step[st].items()))

    rows = list(csv.DictReader(open(run / "progress.csv", encoding="utf-8")))
    hdr = list(rows[0].keys())
    changed = 0

    def agg(res: dict, only_trigger: bool):
        vals = [float(nb > 0) for t, (nb, ne) in res.items()
                if ne > 0 and (not only_trigger or kinds[t] == "trigger")]
        return (sum(vals) / len(vals)) if vals else None

    for r in rows:
        step = int(float(r["time/total_timesteps"]))
        # the eval whose values this row carries: the latest at or before it
        cur = max((s for s in steps if s <= step), default=None)
        if cur is None:
            continue
        res = per_step[cur]
        upd = {}
        if multi:
            for t, (nb, ne) in res.items():
                upd[f"race/eval_finishes.{t}"] = str(int(nb))
        mf = agg(res, False)
        if mf is not None:
            upd["race/maps_finished"] = f"{mf:.4f}"
        mft = agg(res, True)
        if mft is not None:
            upd["race/maps_finished_trigger"] = f"{mft:.4f}"
        for k, v in upd.items():
            if k in r and r[k] != v:
                try:
                    same = abs(float(r[k]) - float(v)) < 1e-9
                except ValueError:
                    same = False
                if not same:
                    r[k] = v
                    changed += 1
    print(f"cells to change: {changed}")
    if not a.apply:
        print("dry run - pass --apply to rewrite progress.csv")
        return
    if changed == 0:
        print("nothing to rewrite")
        return
    bak = run / f"progress.csv.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    (run / "progress.csv").replace(bak)
    with open(run / "progress.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print(f"rewritten; backup {bak.name}")


if __name__ == "__main__":
    main()
