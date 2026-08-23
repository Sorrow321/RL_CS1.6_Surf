#!/usr/bin/env python3
"""bake_gather.py - verify gathered goal fields and build the gate manifest.

`scp` has silently truncated a file and exited 0 on this project before
(a 153 MB checkpoint arrived as 3.9 MB), so nothing gathered is trusted on
its exit code. Every row written by bake_pool.py carries the npz's md5 as
computed ON THE BOX; this re-computes it locally and refuses to count a map
whose bytes do not match.

    python tools/bake_gather.py --dir runs/research/goalfields \\
        --pool pool.json [--dropped dropped.json] [--md docs/table.md]

Writes ``<dir>/manifest.json`` (every map, every number) and prints the
summary: how many maps have a usable field, how many are safe at the coarser
cell, and every failure by name.
"""
import argparse
import hashlib
import json
from pathlib import Path


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--dropped")
    ap.add_argument("--md", help="also write the gate table as markdown")
    args = ap.parse_args()

    d = Path(args.dir)
    pool = {r["map"]: r for r in json.loads(Path(args.pool).read_text())}
    rows = {}
    for rp in sorted((d / "rows").glob("*.json")):
        r = json.loads(rp.read_text())
        rows[r["map"]] = r

    total_bytes = 0
    for name, r in rows.items():
        if "cells" not in r:
            continue
        r["files_ok"] = True
        for tag, c in r["cells"].items():
            f = d / f"{name}.goal_{c['cell']:g}.npz"
            if not f.exists():
                c["local"] = "MISSING"
                r["files_ok"] = False
                continue
            got = md5(f)
            sz = f.stat().st_size
            if got != c["npz_md5"] or sz != c["npz_bytes"]:
                c["local"] = (f"CORRUPT (got {sz} B / {got[:8]}, "
                              f"box said {c['npz_bytes']} B / "
                              f"{c['npz_md5'][:8]})")
                r["files_ok"] = False
            else:
                c["local"] = "ok"
                total_bytes += sz

    # A map whose spawns cannot reach the finish IN THE BAKED FIELD is not a
    # coarsening question at all - it is the silent null the multimap plan
    # warns about: it would train forever at 0% and drag the aggregate down
    # while looking like "the agent does not generalise". Split it out, or
    # it hides inside keep_ref (where d0 == the sentinel == reach_max, so
    # both ratios look like ordinary reach_max drift).
    baked = [n for n, r in rows.items() if "cells" in r and r.get("files_ok")]
    unreachable = [n for n in baked
                   if rows[n]["cells"]["ref"]["n_reachable"] == 0]
    usable = [n for n in baked if n not in set(unreachable)]
    coarse = [n for n in usable if rows[n]["verdict"] == "coarse_ok"]
    keepref = [n for n in usable if rows[n]["verdict"] == "keep_ref"]
    broken = [n for n, r in rows.items()
              if "cells" in r and not r.get("files_ok")]
    failed = {n: r.get("reason", "?") for n, r in rows.items()
              if "cells" not in r}
    missing = [n for n in pool if n not in rows]

    dropped = []
    if args.dropped:
        dropped = json.loads(Path(args.dropped).read_text())

    man = {
        "pool_size": len(pool),
        "baked": len(baked),
        "usable": len(usable),
        "coarse_ok": len(coarse),
        "keep_ref": len(keepref),
        "no_reachable_spawn": unreachable,
        "corrupt": broken,
        "failed": failed,
        "never_attempted": missing,
        "dropped": [{"map": r["map"], "reason": r.get("dropped", ""),
                     "vox_at_32": r.get("vox_ref")} for r in dropped],
        "total_bytes": total_bytes,
        "maps": rows,
    }
    (d / "manifest.json").write_text(json.dumps(man, indent=1))

    # what the multi-map run actually consumes: one cell per map, already
    # gated. coarse_ok maps take the coarser cell (3.3x less RAM per rank);
    # keep_ref maps must stay fine or the shaping drives the agent into a
    # tunnel that does not exist.
    chosen = {}
    for n in usable:
        r = rows[n]
        c = r["cells"]["cand" if r["verdict"] == "coarse_ok" else "ref"]
        chosen[n] = {"goal_cell": c["cell"],
                     "npz": f"{n}.goal_{c['cell']:g}.npz",
                     "d0": round(c["d0"], 1),
                     "reach_max": round(c["reach_max"], 1),
                     "ref_is_32": r["ref_is_32"],
                     "verdict": r["verdict"]}
    (d / "chosen_cells.json").write_text(json.dumps(chosen, indent=1))
    gb = sum(rows[n]["cells"]["cand" if rows[n]["verdict"] == "coarse_ok"
                              else "ref"]["grid_dims"] and
             __import__("math").prod(
                 rows[n]["cells"]["cand" if rows[n]["verdict"] == "coarse_ok"
                                  else "ref"]["grid_dims"])
             for n in usable) * 2 / 1e9
    print(f"chosen-cell field RAM (uint16, all maps) {gb:.2f} GB")

    print(f"pool            {len(pool)}")
    print(f"baked ok        {len(baked)}")
    print(f"USABLE FIELD    {len(usable)}   <- the real input to a multimap run")
    print(f"  coarse_ok(48) {len(coarse)}")
    print(f"  keep_ref (32) {len(keepref)}")
    print(f"no spawn reaches the finish (silent nulls) {len(unreachable)} "
          f"{unreachable}")
    print(f"corrupt         {len(broken)} {broken}")
    print(f"failed          {len(failed)} {list(failed)}")
    print(f"never attempted {len(missing)} {missing}")
    print(f"dropped         {len(dropped)} {[r['map'] for r in dropped]}")
    print(f"bytes           {total_bytes/1e6:.1f} MB")

    if args.md:
        L = ["| map | ref | d0 ref | d0 48 | ratio | reach ref | reach 48 |"
             " reach ratio | verdict | bake s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for n in sorted(rows):
            r = rows[n]
            if "cells" not in r:
                L.append(f"| {n} | - | - | - | - | - | - | - | "
                         f"{r.get('verdict','failed')} | - |")
                continue
            a, b = r["cells"]["ref"], r["cells"]["cand"]
            L.append(
                f"| {n} | {a['cell']:g} | {a['d0']:,.0f} | {b['d0']:,.0f} | "
                f"{r['d0_ratio']:.3f} | {a['reach_max']:,.0f} | "
                f"{b['reach_max']:,.0f} | {r['reach_ratio']:.3f} | "
                f"{r['verdict']} | {r['bake_s']:.0f} |")
        Path(args.md).write_text("\n".join(L) + "\n")
        print(f"-> {args.md}")


if __name__ == "__main__":
    main()
