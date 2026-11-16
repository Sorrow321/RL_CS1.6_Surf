#!/usr/bin/env python3
"""mm_assets.py - the map files a v14 multi-map run needs, checked and packed (CPU only).

A v14 run reads, per map, at cell 32: the bsp and zones.json, goal_32 (the standard field: the
eval metric and the spawn yaw), goalc_32 (--race-no-corner-cut: the shaping field and the
potential channel), sdf_32 (the depth march), occ_32 and slabocc_32 (what the SDF and both
fields are built from) and faceid_32 (--obs-normal). A file that is missing, or whose stored
signature differs from what the builder computes now, is RE-BAKED at startup on the rented
box (goal fields and the face-id grid on its GPU; cannonball's field bake holds ~21 GiB).

    python3 tools/mm_assets.py check --dir STAGE [--maps stem1,stem2,...]
    python3 tools/mm_assets.py pack  --dir STAGE --out maps7.tar.gz [--maps ...]

`check` computes every signature exactly as the builders do (vision.map_occupancy /
slab_occupancy / build_sdf, goalfield.build_goal_field, texmap.TextureMap.for_lidar) with the
built core, and reports OK / MISS / MISSING per file. A map with more solid brush entities than
the pre-6bd7f931 core held (128) carries the entity tag "_e1" in every grid signature since mm/s1
(vision.entity_tag), so its grids built by the old core MISS and are re-baked at startup
(surf_princessburglar_b26: four func_walls missing, see the mm/s0 notes); such a map is also
re-sampled (base occupancy only, seconds) and flagged STALE when its occ_32 differs from the
fixed core's. Run tools/restamp_maps.py --maps STAGE first: tar drops sub-second mtimes, and a
restamp is what turns those MISSes back into OKs.

`pack` tars STAGE/<stem>.* for the maps under maps/ (the layout box.py deploys), sorted, and
prints the md5. It refuses unless `check` passes (--force packs anyway, for a staging run).

--startup-bake faceid (check and pack): a missing or stale faceid_32 is accepted - the trainer
bakes it on the box's GPU at its first start (texmap.TextureMap.for_lidar: CUDA + triton, the
"face-id grid BAKED" line) - and is left out of the pack. Every other file must be OK.

THE PACKS (mm/s1, 2026-11-16; sources under ~/ar_m2/maps, S = a fresh staging directory):

    for t in sidistic_maps.tar.gz blue200cf_maps.tar.gz cannonballc_maps.tar.gz \
             petrus_litec_maps.tar.gz celestialc_maps.tar.gz uf2_maps.tar.gz \
             pb_maps.tar.gz pb_e1_grids.tar.gz; do          # (P1: blue200cf, petrus_litec,
        tar xzf ~/ar_m2/maps/$t -C $S --strip-components=1 --wildcards 'maps/*'   # celestialc)
    done
    cp -p ~/ar_m2/maps/celestial_faceid/surf_src_celestial.faceid_32.npz $S/
    rm -f $S/surf_princessburglar_b26.faceid_32.npz      # the old core's: stale since the tag
    python3 tools/restamp_maps.py --maps $S
    python3 tools/mm_assets.py check --dir $S --startup-bake faceid
    python3 tools/mm_assets.py pack --dir $S --startup-bake faceid --out ~/ar_m2/maps/<name>

* pb_e1_grids.tar.gz: princessburglar's occ / slabocc / sdf / goal / goalc rebuilt on the
  droplet's CPU by the fixed core with the entity tag "_e1" (ed329d70); it is unpacked AFTER
  pb_maps.tar.gz, whose grids it replaces.
* faceid: celestial's comes from celestial_faceid/ (not in its tarball); petrus never had one
  and pb's is stale, so both are baked on the box's GPU at the trainer's first start
  (--startup-bake faceid), or once ahead with the trainer's --warm-caches and pulled back
  into the sources.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

CELL = 32.0
KINDS = ("goal", "goalc", "sdf", "occ", "slabocc", "faceid")
# src/sim.h before 6bd7f931 (2026-10-29) and src/bsp.c parse_entities' solid rule: one copy, in
# surfgym.vision (the entity tag reads them)
from surfgym.vision import OLD_MAX_SOLIDS, solid_entity_count  # noqa: E402,F401


def expected_sigs(core, bsp) -> dict:
    """The signature each builder computes for this map now (cell 32, no kill mask, no gravity
    rule, no --vision-clip - the v14 caches)."""
    from surfgym import vision
    from surfgym.goalfield import _GOAL_BUILDER_VERSION
    from surfgym.texmap import TEXMAP_SEMANTICS
    from surfgym.zones import load_zones
    msig = vision._map_sig(Path(bsp))
    ptag = vision.pane_tag(core) + vision.entity_tag(core)   # (each builder appends both)
    sdf_sig = f"{msig}_{vision._SDF_SEMANTICS}{ptag}"
    zone = load_zones(str(bsp))["end"]
    box = np.round(np.asarray(zone["mins"] + zone["maxs"], np.float64), 1)
    tail = f"{msig}_" + "_".join(f"{v:g}" for v in box) + ptag
    if zone.get("boxes"):
        tail += "_b" + "_".join(f"{v:g}" for b in zone["boxes"]
                                for v in np.round(np.asarray(b["mins"] + b["maxs"],
                                                             np.float64), 1))
    return {"goal": f"g{_GOAL_BUILDER_VERSION}_{tail}",
            "goalc": f"g{_GOAL_BUILDER_VERSION}_c1_{tail}",
            "sdf": sdf_sig, "slabocc": sdf_sig,
            "occ": f"{msig}_{vision._OCC_SEMANTICS}{vision.entity_tag(core)}",
            "faceid": f"{sdf_sig}_{TEXMAP_SEMANTICS}"}


def stored_sig(path):
    try:
        z = np.load(path, allow_pickle=False)
        return str(z["sig"]) if "sig" in z.files else None
    except Exception:                                   # noqa: BLE001
        return None


def check_map(d: Path, stem: str, startup_bake=()) -> dict:
    """OK / MISS / MISSING per file; ``startup_bake`` names the kinds (only "faceid") that may
    be missing or stale because the trainer bakes them on the box's GPU at its first start
    (their stale files are listed in rep["skip"]: pack leaves them out)."""
    from surfgym.core import SurfCore, default_config
    bsp = d / f"{stem}.bsp"
    rep = {"map": stem, "files": {}, "ok": True, "notes": [], "skip": []}
    for need in (bsp, d / f"{stem}.zones.json"):
        if not need.exists():
            rep["files"][need.name] = "MISSING"
            rep["ok"] = False
    if not rep["ok"]:
        return rep
    core = SurfCore(str(bsp), default_config(num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))
    want = expected_sigs(core, bsp)
    for k in KINDS:
        p = d / f"{stem}.{k}_{CELL:g}.npz"
        late = k in startup_bake          # declared: baked on the box's GPU at the first start
        if not p.exists():
            rep["files"][p.name] = ("MISSING - baked on the GPU at the first start (declared)"
                                    if late else "MISSING (re-baked at startup)")
            rep["ok"] = rep["ok"] and late
            continue
        got = stored_sig(p)
        if got == want[k]:
            rep["files"][p.name] = "OK"
        else:
            same_but_mtime = (got is not None and got.split("_")[0:2] == want[k].split("_")[0:2])
            rep["files"][p.name] = ("MISS (signature differs" + (": run restamp_maps.py first"
                                    if same_but_mtime else "") + f"; stored {got!r}, now "
                                    f"{want[k]!r}) - re-baked at startup"
                                    + (" on the GPU (declared; not packed)" if late else ""))
            rep["ok"] = rep["ok"] and late
            if late:
                rep["skip"].append(p.name)
    n_sol = solid_entity_count(bsp)
    rep["solids"] = n_sol
    occ_p = d / f"{stem}.occ_{CELL:g}.npz"
    if n_sol > OLD_MAX_SOLIDS and occ_p.exists():
        from surfgym import vision
        mins, nx, ny, nz = vision.grid_dims(core, CELL)
        z = np.load(occ_p, allow_pickle=False)
        now = core.occupancy_grid(mins, CELL, nx, ny, nz)
        diff = int((now != z["occ"]).sum()) if now.shape == z["occ"].shape else -1
        rep["occ_vs_core"] = diff
        if diff != 0:
            rep["ok"] = False
            rep["notes"].append(
                f"STALE: {n_sol} solid brush entities > the old core's {OLD_MAX_SOLIDS}, and "
                f"occ_32 differs from this core's sampling in {diff:,} voxels: every grid of "
                "this map (occ, slabocc, sdf, goal, goalc, faceid) was built by the pre-6bd7f931 "
                "core; their signatures lack the entity tag, so they MISS and re-bake at startup "
                "(goal fields and faceid on the GPU) - re-bake them before packing instead")
    return rep


def cmd_check(a) -> int:
    d = Path(a.dir)
    stems = ([s.strip() for s in a.maps.split(",") if s.strip()] if a.maps
             else sorted(p.stem for p in d.glob("*.bsp")))
    ok = True
    skip = []
    for s in stems:
        r = check_map(d, s, tuple(getattr(a, "startup_bake", None) or ()))
        ok &= r["ok"]
        skip += r["skip"]
        print(f"{'OK  ' if r['ok'] else 'FAIL'} {s}  (solid brush entities {r.get('solids', '?')})")
        for f, v in r["files"].items():
            print(f"      {f:44s} {v}")
        for n in r["notes"]:
            print(f"      !! {n}")
    print("== all maps ready" if ok else "== NOT ready: see above")
    a._skip = skip          # (pack: the declared-stale files it leaves out)
    return 0 if ok else 1


def cmd_pack(a) -> int:
    d = Path(a.dir)
    stems = ([s.strip() for s in a.maps.split(",") if s.strip()] if a.maps
             else sorted(p.stem for p in d.glob("*.bsp")))
    if not a.force and cmd_check(a) != 0:
        print("!! pack refused: check failed (--force packs anyway)")
        return 1
    skip = set(getattr(a, "_skip", None) or ())
    files = []
    for s in stems:
        files += sorted(p for p in d.glob(f"{s}.*") if p.is_file() and p.name not in skip)
    out = Path(a.out)
    with tarfile.open(out, "w:gz") as tf:
        for p in files:
            tf.add(p, arcname=f"maps/{p.name}")
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    print(f"{out} {out.stat().st_size:,} bytes md5 {md5}: {len(files)} files for {len(stems)} maps "
          "(tar keeps whole-second mtimes: run tools/restamp_maps.py after unpacking)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "pack"):
        p = sub.add_parser(name)
        p.add_argument("--dir", required=True, help="the staging directory (maps' files flat)")
        p.add_argument("--maps", default=None, help="stems, comma-separated (default: every bsp)")
        p.add_argument("--startup-bake", action="append", choices=("faceid",), default=None,
                       help="this kind may be missing or stale: the trainer bakes it on the "
                            "box's GPU at its first start (a stale file is not packed)")
        if name == "pack":
            p.add_argument("--out", required=True)
            p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return cmd_check(a) if a.cmd == "check" else cmd_pack(a)


if __name__ == "__main__":
    sys.exit(main())
