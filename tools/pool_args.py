#!/usr/bin/env python3
"""pool_args.py - emit the --maps and --goal-cell flags for a pool run.

The pool ships each map's goal field at the cell that map was GATED at
(mostly 48, some 32). `--goal-cell` must be given those exact cells or the
trainer asks for a cell nobody baked, misses the cache and rebakes the
field at startup - minutes to hours per map, on rented time, with nothing
in the log that distinguishes it from a cold start.

The cells are not guessed: they are read off the .goal_<cell>.npz files
actually present next to each .bsp, so the flags can never disagree with
what is on disk. A map with no field, or with more than one, is skipped
loudly rather than silently rebaked.

    python3 tools/pool_args.py                       # every map with a field
    python3 tools/pool_args.py --limit 40            # first N, for a smaller run
    python3 tools/pool_args.py --only-trigger        # skip button finishes
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL_RE = re.compile(r"\.goal_(\d+(?:\.\d+)?)\.npz$")


def pool(maps_dir: Path, only_trigger=False):
    kinds = {}
    man = maps_dir.parent / "manifest.json"
    if man.exists():
        try:
            for r in json.loads(man.read_text(encoding="utf-8"))["maps"]:
                kinds[r["map"]] = str(r.get("finish_kind") or "?")
        except Exception:
            pass
    rows, skipped = [], []
    for bsp in sorted(maps_dir.glob("*.bsp")):
        cells = sorted({float(m.group(1)) for m in
                        (CELL_RE.search(p.name) for p in
                         maps_dir.glob(f"{bsp.stem}.goal_*.npz")) if m})
        if len(cells) != 1:
            skipped.append((bsp.stem, f"{len(cells)} baked fields"))
            continue
        if not (maps_dir / f"{bsp.stem}.zones.json").exists():
            skipped.append((bsp.stem, "no zones.json"))
            continue
        kind = kinds.get(bsp.stem, "?")
        if only_trigger and "trigger" not in kind.lower():
            skipped.append((bsp.stem, f"finish_kind={kind}"))
            continue
        rows.append((bsp.stem, cells[0], kind))
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default=str(ROOT / "maps"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-trigger", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="human summary instead of the flags")
    a = ap.parse_args()

    rows, skipped = pool(Path(a.maps_dir), a.only_trigger)
    if a.limit:
        rows = rows[:a.limit]
    if a.report:
        from collections import Counter
        print(f"{len(rows)} maps usable, {len(skipped)} skipped")
        for k, n in Counter(k for _, _, k in rows).most_common():
            print(f"   finish_kind {k:22s} {n}")
        for k, n in Counter(c for _, c, _ in rows).most_common():
            print(f"   goal cell   {k:<22g} {n}")
        for m, why in skipped[:8]:
            print(f"   skip {m:32s} {why}")
        return 0
    # emit the ACTUAL directory, not a hardcoded "maps/" - the pool is often
    # unpacked somewhere else (a local checkout already has its own maps/)
    d = Path(a.maps_dir)
    try:
        d = d.relative_to(ROOT)
    except ValueError:
        pass
    maps = ",".join(f"{d.as_posix()}/{m}.bsp" for m, _, _ in rows)
    cells = ",".join(f"{c:g}" for _, c, _ in rows)
    print(f"--maps {maps} --goal-cell {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
