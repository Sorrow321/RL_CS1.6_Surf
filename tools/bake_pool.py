#!/usr/bin/env python3
"""bake_pool.py - bake the geodesic goal fields for a shard of the map pool
and run the per-map CELL-SIZE GATE.

Two cells per map:

  reference  = max(32, pick_cell(core))   - pick_cell auto-coarsens the maps
               whose cell-32 grid blows the 700M-voxel budget, so a handful
               of monsters get a 64u REFERENCE and their gate result is
               weaker evidence than a true cell-32 comparison. The manifest
               records ``ref_is_32`` so that is visible, never inferred.
  candidate  = 1.5 x reference            - 32 -> 48, 64 -> 96.

Gate (measured on surf_src_cannonball, docs/multimap-ddp-plan.md): coarsening
is NOT free. At cell 64 the grid stops resolving thin geometry, the geodesic
wavefront TUNNELS through floors, d0 halves (198,353 -> 95,122) and
monotonicity along the champion line falls 95.6% -> 70.6%. Only cannonball
has a champion route, so monotonicity is unavailable pool-wide; the tell that
IS available everywhere is d0 (the spawn's geodesic distance) and reach_max
collapsing between cells. Cannonball's cell-64 failure reads as 0.48 on the
d0 ratio, so the signal is enormous.

    accept the coarser cell iff  d0_cand/d0_ref >= 0.95
                            and  |reach_cand/reach_ref - 1| <= 0.05

The RATIO is recorded per map either way, so a different cut can be applied
later without re-baking.

Usage on a rented box::

    python3 tools/bake_pool.py --pool pool.json --shard shard.txt \\
        --maps-dir /root/pool/maps --out /root/pool/out [--timeout 3600]

Each map is baked in its OWN subprocess (``--one``) so a CUDA OOM, a hang or
a segfault costs that map and not the shard; ``--timeout`` skips a
pathological map and records why. Rows land in ``<out>/rows/<map>.json`` as
they finish, so a killed box still yields everything it had done.

``maps_full_dataset/`` is the user's data and READ-ONLY: the BSPs are copied
to ``--maps-dir`` and every derived file goes to ``--out``.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

D0_MIN_RATIO = 0.95
REACH_TOL = 0.05


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pool(path):
    return {r["map"]: r for r in json.loads(Path(path).read_text())}


def pin_mtimes(pool, maps_dir):
    """The cache signature embeds the bsp's size AND st_mtime_ns
    (vision._map_sig), so a field baked against a COPIED bsp is rejected
    when it comes home and silently re-bakes. Restore the workstation's
    mtime on the box before baking anything."""
    n = 0
    for name, r in pool.items():
        bsp = Path(maps_dir) / f"{name}.bsp"
        if not bsp.exists():
            continue
        if bsp.stat().st_size != r["size"]:
            raise SystemExit(f"{bsp} is {bsp.stat().st_size} B, pool says "
                             f"{r['size']} B - truncated transfer")
        m = int(r["mtime_ns"])
        os.utime(bsp, ns=(m, m))
        n += 1
    return n


def bake_one(name, pool, maps_dir, out):
    """Bake both cells for one map and return the manifest row."""
    import numpy as np
    from surfgym import SurfCore, default_config
    from surfgym.goalfield import build_goal_field
    from surfgym.rewards import map_spawn_pool

    meta = pool[name]
    bsp = Path(maps_dir) / f"{name}.bsp"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    row = {"map": name, "bsp_size": meta["size"],
           "ref_cell": meta["ref_cell"], "cand_cell": meta["cand_cell"],
           "ref_is_32": meta["ref_cell"] == 32.0,
           "pick_cell": meta["pick_cell"],
           "vox_ref": meta["vox_ref"], "vox_cand": meta["vox_cand"],
           "host": os.uname().nodename if hasattr(os, "uname") else "",
           "cells": {}}

    core = SurfCore(str(bsp), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    zone = meta["end"]
    if not zone:
        raise RuntimeError("map has no end zone")
    spawns = map_spawn_pool(core)["origin"]
    row["n_spawns"] = int(len(spawns))

    reach_ok = None
    for tag, cell in (("ref", float(meta["ref_cell"])),
                      ("cand", float(meta["cand_cell"]))):
        t0 = time.time()
        gf = build_goal_field(core, zone, cell=cell, cache_dir=out)
        dt = time.time() - t0
        d = np.asarray(gf.sample(spawns), np.float64)
        ok = np.asarray(gf.reachable(spawns), bool)
        npz = out / f"{bsp.stem}.goal_{cell:g}.npz"
        row["cells"][tag] = {
            "cell": cell,
            "bake_s": round(dt, 1),
            "reach_max": float(gf.reach_max),
            # what train_fast.py actually computes (scale = 100/d0): the mean
            # over EVERY spawn, sentinels included
            "d0_all": float(d.mean()),
            "n_reachable": int(ok.sum()),
            "npz_bytes": npz.stat().st_size if npz.exists() else 0,
            "npz_md5": md5(npz) if npz.exists() else "",
            "grid_dims": [int(v) for v in gf.grid.shape],
        }
        # the honest comparison basis: spawns reachable at BOTH cells. A
        # sentinel is reach_max + 2*cell, so mixing one in swamps a mean.
        reach_ok = ok if reach_ok is None else (reach_ok & ok)
        row["cells"][tag]["_d"] = d
        del gf

    both = reach_ok
    for tag in ("ref", "cand"):
        d = row["cells"][tag].pop("_d")
        row["cells"][tag]["d0"] = (float(d[both].mean()) if both.any()
                                   else float(d.mean()))
    row["n_reach_both"] = int(both.sum())

    a, b = row["cells"]["ref"], row["cells"]["cand"]
    row["d0_ratio"] = round(b["d0"] / a["d0"], 4) if a["d0"] > 0 else 0.0
    row["reach_ratio"] = (round(b["reach_max"] / a["reach_max"], 4)
                          if a["reach_max"] > 0 else 0.0)
    row["spawns_lost"] = a["n_reachable"] - b["n_reachable"]
    fails = []
    if row["d0_ratio"] < D0_MIN_RATIO:
        fails.append("d0")
    if abs(row["reach_ratio"] - 1.0) > REACH_TOL:
        fails.append("reach_max")
    if a["n_reachable"] == 0:
        fails.append("no_reachable_spawn_at_ref")
    row["verdict"] = "coarse_ok" if not fails else "keep_ref"
    row["gate_fail"] = fails
    row["bake_s"] = round(a["bake_s"] + b["bake_s"], 1)

    # occupancy/SDF caches are scratch - only the goal fields go home
    for f in out.glob(f"{bsp.stem}.*"):
        if ".goal_" not in f.name:
            f.unlink()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool json (per-map meta)")
    ap.add_argument("--shard", help="file with one map name per line")
    ap.add_argument("--maps-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="per-map seconds before it is skipped")
    ap.add_argument("--one", help="internal: bake exactly this map")
    ap.add_argument("--pin-mtimes", action="store_true",
                    help="restore the workstation mtime on every bsp and exit")
    args = ap.parse_args()

    pool = load_pool(args.pool)
    out = Path(args.out)
    rows_dir = out / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    if args.pin_mtimes:
        print(f"pinned {pin_mtimes(pool, args.maps_dir)} bsp mtimes")
        return

    if args.one:
        row = bake_one(args.one, pool, args.maps_dir, out)
        (rows_dir / f"{args.one}.json").write_text(json.dumps(row, indent=1))
        print(json.dumps({k: v for k, v in row.items() if k != "cells"}))
        return

    names = [ln.strip() for ln in Path(args.shard).read_text().splitlines()
             if ln.strip()]
    # heaviest FIRST: a slow straggler must show up at the start of the
    # shard, not at the end when nothing can be done about it
    names.sort(key=lambda n: -(pool[n]["vox_ref"] + pool[n]["vox_cand"]))
    print(f"shard: {len(names)} maps, "
          f"{sum(pool[n]['vox_ref'] + pool[n]['vox_cand'] for n in names)/1e9:.2f} Gvox",
          flush=True)
    pin_mtimes(pool, args.maps_dir)

    t_all = time.time()
    done = fail = 0
    for i, name in enumerate(names):
        rp = rows_dir / f"{name}.json"
        if rp.exists():
            done += 1
            continue
        w = (pool[name]["vox_ref"] + pool[name]["vox_cand"]) / 1e6
        print(f"[{i+1}/{len(names)}] {name} ({w:.0f} Mvox) ...", flush=True)
        t0 = time.time()
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--pool", args.pool, "--maps-dir", args.maps_dir,
               "--out", args.out, "--one", name]
        try:
            p = subprocess.run(cmd, timeout=args.timeout,
                               capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError((p.stderr or p.stdout or "")[-800:])
        except subprocess.TimeoutExpired:
            fail += 1
            rp.write_text(json.dumps(
                {"map": name, "verdict": "skipped",
                 "reason": f"timeout after {args.timeout:.0f}s",
                 "vox_ref": pool[name]["vox_ref"]}, indent=1))
            print(f"    SKIP timeout {args.timeout:.0f}s", flush=True)
            for f in out.glob(f"{name}.*"):
                f.unlink()
            continue
        except Exception as e:
            fail += 1
            rp.write_text(json.dumps(
                {"map": name, "verdict": "failed", "reason": str(e)[-800:],
                 "vox_ref": pool[name]["vox_ref"]}, indent=1))
            print(f"    FAIL {str(e)[-300:]}", flush=True)
            for f in out.glob(f"{name}.*"):
                if ".goal_" not in f.name:
                    f.unlink()
            continue
        done += 1
        r = json.loads(rp.read_text())
        print(f"    {r['verdict']:9s} d0 {r['cells']['ref']['d0']:.0f} -> "
              f"{r['cells']['cand']['d0']:.0f}  ratio {r['d0_ratio']:.3f}  "
              f"reach {r['reach_ratio']:.3f}  {time.time()-t0:.0f}s",
              flush=True)

    el = time.time() - t_all
    print(f"DONE {done} ok, {fail} failed, {el/60:.1f} min", flush=True)
    (out / "shard_done.json").write_text(json.dumps(
        {"done": done, "failed": fail, "elapsed_s": round(el, 1),
         "maps": names}, indent=1))
    free = shutil.disk_usage(out).free / 1e9
    print(f"disk free {free:.1f} GB; "
          f"{len(list(out.glob('*.goal_*.npz')))} goal npz", flush=True)


if __name__ == "__main__":
    main()
