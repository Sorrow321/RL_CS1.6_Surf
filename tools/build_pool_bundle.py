#!/usr/bin/env python3
"""build_pool_bundle.py - assemble everything a training box needs, in the
layout it already expects, as ONE archive.

The pool's assets are deliberately scattered: maps live in the user's
read-only corpus, goal fields came off a rented bake, zone files came from
three different detectors. A box needs all of it flat in `maps/`, because
`zones.load_zones` and the field cache both resolve
``<bsp>.parent/<stem>.<ext>`` - so a flat layout needs no path flags and no
code change.

    maps/<name>.bsp            the map
    maps/<name>.goal_<cell>.npz the gated geodesic field
    maps/<name>.sdf_32.npz     the lidar SDF, prebaked
    maps/<name>.zones.json     the finish box, pinned

The SDF is included on purpose. It rebuilds at ~107 s per billion voxels,
which is ~13 minutes for this pool - **per box, every time**, before
training can start. 181 MB of archive removes that from every future rent,
and removes a failure mode (an interrupted bake) at the worst moment.

Zones are WRITTEN OUT rather than re-detected, so the box pins the exact
finish box measured here - including the inflated on-touch button boxes -
instead of re-deriving one that a later change to zones.py might move.

    python tools/build_pool_bundle.py --out runs/research/pool_bundle
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

CORPUS = ROOT / "maps_full_dataset"
FIELDS = ROOT / "runs" / "research" / "goalfields"
GWZONE = ROOT / "runs" / "research" / "gateway_zones"

# Maps removed from the pool, with cause. The selection JSONs under runs/
# are gitignored measurement artifacts, so the exclusion has to live in the
# builder itself or a v3 bundle would quietly re-include the map.
EXCLUDED = {
    "surf_bucetation": "2026-08-25: map_pct=100 in 18/18 evals from a ~4,600u "
        "free-fall that clips the goal-field zero halo 89u OUTSIDE the padded "
        "finish box at ~2,400 u/s and dies at kill_z; 0 finishes ever. GoldSrc "
        "fall damage would kill every such attempt; the sim has none and the "
        "user ruled out adding it, so removal is the fix (round 26).",
}


def load_pool():
    chosen = json.loads((FIELDS / "chosen_cells.json").read_text(encoding="utf-8"))
    tr = json.loads((ROOT / "runs" / "research" / "maps_trainable.json")
                    .read_text(encoding="utf-8"))
    rows = tr.get("maps") if isinstance(tr, dict) else tr
    if isinstance(rows, dict):
        rows = [dict(v, map=k) for k, v in rows.items()]
    kind = {r["map"]: str(r.get("finish_kind") or "?") for r in rows}
    passing = set(kind)
    out = []
    for m, meta in chosen.items():
        if m not in passing or m in EXCLUDED:
            continue
        out.append({"map": m, "cell": meta["goal_cell"], "npz": meta["npz"],
                    "finish_kind": kind.get(m, "?"),
                    "d0": meta.get("d0"), "verdict": meta.get("verdict")})
    return sorted(out, key=lambda r: r["map"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/research/pool_bundle")
    ap.add_argument("--only-trigger", action="store_true",
                    help="just the strong-evidence maps")
    ap.add_argument("--lidar-cell", type=float, default=32.0)
    ap.add_argument("--skip-sdf", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    maps_dir = out / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool()
    if a.only_trigger:
        pool = [r for r in pool if "trigger" in r["finish_kind"].lower()]
    print(f"{len(pool)} maps into {out}")

    from surfgym import SurfCore, default_config
    from surfgym.vision import build_sdf
    from surfgym.zones import detect_zones

    manifest, t0, failed = [], time.time(), []
    for i, r in enumerate(pool, 1):
        m = r["map"]
        bsp_src = CORPUS / f"{m}.bsp"
        if not bsp_src.exists():
            failed.append((m, "no bsp")); continue
        fld_src = FIELDS / r["npz"]
        if not fld_src.exists():
            failed.append((m, "no field")); continue

        bsp_dst = maps_dir / f"{m}.bsp"
        if not bsp_dst.exists():
            shutil.copy2(bsp_src, bsp_dst)
        shutil.copy2(fld_src, maps_dir / r["npz"])

        # zones: prefer the gateway file (type 2), else detect from the BSP
        # (types 1 and 3). Written out either way so the box never re-derives.
        gw = GWZONE / f"{m}.zones.json"
        if gw.exists():
            zdoc = json.loads(gw.read_text(encoding="utf-8"))
        else:
            z = detect_zones(str(bsp_dst))
            if not z.get("end"):
                failed.append((m, "no end zone")); continue
            zdoc = {"map": m, "source": "manual", "start": z.get("start"),
                    "end": z["end"]}
        zdoc.setdefault("source", "manual")     # never regenerable on the box
        (maps_dir / f"{m}.zones.json").write_text(
            json.dumps(zdoc, indent=1), encoding="utf-8")

        if not a.skip_sdf:
            sdf = maps_dir / f"{m}.sdf_{int(a.lidar_cell)}.npz"
            if not sdf.exists():
                try:
                    core = SurfCore(str(bsp_dst), default_config(
                        num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))
                    build_sdf(core, cell=a.lidar_cell)
                    del core
                except Exception as ex:
                    failed.append((m, f"sdf: {type(ex).__name__}")); continue
        manifest.append(r)
        if i % 10 == 0 or i == len(pool):
            mb = sum(f.stat().st_size for f in maps_dir.iterdir()) / 1e6
            print(f"  [{i:3d}/{len(pool)}] {mb:7.0f} MB  "
                  f"{time.time() - t0:5.0f}s elapsed")

    (out / "manifest.json").write_text(json.dumps(
        {"maps": manifest, "failed": failed, "lidar_cell": a.lidar_cell,
         "note": "flat layout: unpack over the repo root so files land in maps/"},
        indent=1), encoding="utf-8")
    mb = sum(f.stat().st_size for f in maps_dir.iterdir()) / 1e6
    print(f"\n{len(manifest)} maps, {mb:.0f} MB in {out}")
    if failed:
        print(f"{len(failed)} failed:")
        for m, why in failed[:10]:
            print(f"   {m:34s} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
