#!/usr/bin/env python3
"""stage_maps.py - assemble the SUBSET of map assets one run needs, and only
that subset.

The pool's assets are scattered by design and none of the three locations is
the one a box reads from:

    maps_full_dataset/<name>.bsp                 read-only corpus
    runs/research/goalfields/<name>.goal_<c>.npz the gated geodesic field
    runs/research/gateway_zones/<name>.zones.json  type-2 finish buttons
    (type-1/3 zones are detected from the BSP and written out here)

``load_zones`` and the field/SDF caches all resolve ``<bsp>.parent/<stem>.*``,
so a FLAT ``maps/`` layout needs no path flags and no code change - that is
the layout this writes.

WHY A SUBSET TOOL EXISTS. ``deploy_box.sh``'s default glob pulls ~152 MB
where 57 MB sufficed, and the workstation uplink has been the pole often
enough to cost a night (a 153 MB checkpoint once arrived as 3.9 MB with exit
code 0). A 5-map smoke run needs ~25 MB. Shipping the whole pool to prove
plumbing is 6x the transfer for none of the evidence.

Three assets per map, and each is here for a reason:
  * the BSP;
  * the goal field AT ITS GATED CELL - per map, not global: 21 of the 110
    usable maps TUNNEL at cell 48 (the wavefront flows through thin floors,
    d0 collapses - surf_texture to a ratio of 0.134) and must keep cell 32,
    while the other 89 are 3.3x cheaper at 48. The tool emits the matching
    ``--goal-cell`` list so the trainer is told, per map, which one it is;
  * the lidar SDF, prebaked. It rebuilds at ~107 s per billion voxels - per
    box, every time, before training can start - and an interrupted bake is
    a failure mode at the worst possible moment.
  * the zone file, WRITTEN OUT rather than re-detected, so the box pins the
    exact finish box measured here instead of re-deriving one that a later
    change to zones.py might move.

    python tools/stage_maps.py --n 5 --only-trigger --out runs/research/stage5
    python tools/stage_maps.py --maps surf_prechasm,surf_latebra --out ...

NOTHING THIS WRITES MAY BE COMMITTED. Maps, baked fields and everything
derived from the Surf Gateway service are a third party's data;
``.gitignore`` was hardened for it. The default --out is under runs/, which
is ignored - keep it that way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# The corpus, the fields and the gateway zones are all gitignored, so they
# exist only in the MAIN checkout - a worktree has none of them. --src-root
# points at that checkout; CLAUDE.md's rule about absolute paths into
# C:/RL_Surf is the same rule (a worktree copy has different mtimes, and the
# cache signature embeds mtime_ns, so a relative path silently re-bakes).
SRC = ROOT
CORPUS = FIELDS = GWZONE = BUNDLE = LOCAL = None


def set_src(root: Path) -> None:
    global SRC, CORPUS, FIELDS, GWZONE, BUNDLE, LOCAL
    SRC = Path(root).resolve()
    CORPUS = SRC / "maps_full_dataset"
    FIELDS = SRC / "runs" / "research" / "goalfields"
    GWZONE = SRC / "runs" / "research" / "gateway_zones"
    BUNDLE = SRC / "runs" / "research" / "pool_bundle" / "maps"
    LOCAL = SRC / "maps"


set_src(ROOT)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pool() -> list:
    """Maps that have BOTH a gated field and a verify_maps.py pass.

    Gated on the BAKE, not on the survey: 48 maps have a spawn and an end
    zone and no path between them, and only the baked field shows it. They
    would train forever at 0% and read as "the agent does not generalise".
    """
    chosen = json.loads((FIELDS / "chosen_cells.json").read_text("utf-8"))
    tr = json.loads((SRC / "runs" / "research" / "maps_trainable.json")
                    .read_text("utf-8"))
    rows = tr.get("maps") if isinstance(tr, dict) else tr
    if isinstance(rows, dict):
        rows = [dict(v, map=k) for k, v in rows.items()]
    by = {r["map"]: r for r in rows if str(r.get("verdict")) == "pass"}
    out = []
    for m, meta in chosen.items():
        r = by.get(m)
        if r is None:
            continue
        bsp = CORPUS / f"{m}.bsp"
        out.append({
            "map": m,
            "goal_cell": float(meta["goal_cell"]),
            "npz": meta["npz"],
            "d0": meta.get("d0"),
            "cell_verdict": meta.get("verdict"),
            "finish_kind": str(r.get("finish_kind") or "?"),
            "finish_face_area": r.get("finish_face_area"),
            "bsp_bytes": bsp.stat().st_size if bsp.exists() else 0,
        })
    return sorted(out, key=lambda r: r["map"])


def find(name: str, *dirs) -> Path | None:
    for d in dirs:
        p = d / name
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default=str(ROOT),
                    help="checkout that holds maps_full_dataset/ and "
                         "runs/research/ (they are gitignored, so a worktree "
                         "has neither)")
    ap.add_argument("--out", default="runs/research/stage",
                    help="staging dir; a maps/ subdir is created inside")
    ap.add_argument("--maps", default=None,
                    help="explicit comma-separated map stems; overrides --n")
    ap.add_argument("--n", type=int, default=5,
                    help="how many maps to pick when --maps is not given")
    ap.add_argument("--only-trigger", action="store_true",
                    help="only real trigger_multiple finishes. A +use button "
                         "box is ~8x smaller in face area and the simulator "
                         "cannot press a button at all, so a null on one is "
                         "much weaker evidence (CLAUDE.md 4b)")
    ap.add_argument("--lidar-cell", type=float, default=32.0)
    ap.add_argument("--skip-sdf", action="store_true",
                    help="do not bake a missing SDF (the box will, at ~107 s "
                         "per billion voxels, before it can train)")
    ap.add_argument("--tar", action="store_true",
                    help="also write <out>.tar.gz, ready to scp")
    ap.add_argument("--mesh", action="store_true",
                    help="also export viewer/assets/<map>.mesh.json so the "
                         "dashboard can render each map's recordings")
    a = ap.parse_args()

    set_src(Path(a.src_root))
    pool = load_pool()
    bykey = {r["map"]: r for r in pool}
    if a.maps:
        want = [m.strip() for m in a.maps.split(",") if m.strip()]
        missing = [m for m in want if m not in bykey]
        if missing:
            print(f"!! not in the gated+verified pool: {missing}")
            return 1
        sel = [bykey[m] for m in want]
    else:
        cand = pool
        if a.only_trigger:
            cand = [r for r in cand if r["finish_kind"] == "trigger"]
        # smallest first: the bake and the transfer are what a smoke run is
        # paying for, and neither is the point of the test
        cand = sorted(cand, key=lambda r: r["bsp_bytes"])
        # drop _b2/_b3/_b1 re-releases of a map already picked - they are the
        # same track and would make the map count a lie
        sel, seen = [], set()
        for r in cand:
            base = r["map"]
            for suf in ("_b1", "_b2", "_b3"):
                if base.endswith(suf):
                    base = base[:-len(suf)]
            if base in seen:
                continue
            seen.add(base)
            sel.append(r)
            if len(sel) >= a.n:
                break

    out = Path(a.out)
    maps_dir = out / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    print(f"staging {len(sel)} maps into {maps_dir}")

    from surfgym import SurfCore, default_config          # noqa: E402
    from surfgym.vision import build_sdf                  # noqa: E402
    from surfgym.zones import detect_zones                # noqa: E402

    manifest, t0, failed = [], time.time(), []
    for r in sel:
        m = r["map"]
        files = {}

        bsp_src = find(f"{m}.bsp", BUNDLE, CORPUS, LOCAL)
        if bsp_src is None:
            failed.append((m, "no bsp"))
            continue
        bsp_dst = maps_dir / f"{m}.bsp"
        if not bsp_dst.exists():
            shutil.copy2(bsp_src, bsp_dst)
        files[bsp_dst.name] = md5(bsp_dst)

        fld_src = find(r["npz"], BUNDLE, FIELDS, LOCAL)
        if fld_src is None:
            failed.append((m, f"no field {r['npz']}"))
            continue
        shutil.copy2(fld_src, maps_dir / r["npz"])
        files[r["npz"]] = md5(maps_dir / r["npz"])

        # zones: the gateway file wins (type 2), else detect off the BSP
        # (types 1 and 3). Stamped source: manual, so the box never
        # regenerates it - load_zones only ever rebuilds "auto" files.
        zs = find(f"{m}.zones.json", BUNDLE, GWZONE)
        if zs is not None:
            zdoc = json.loads(zs.read_text("utf-8"))
        else:
            z = detect_zones(str(bsp_dst))
            if not z.get("end"):
                failed.append((m, "no end zone"))
                continue
            zdoc = {"map": m, "start": z.get("start"), "end": z["end"]}
        zdoc["source"] = "manual"
        zp = maps_dir / f"{m}.zones.json"
        zp.write_text(json.dumps(zdoc, indent=1), encoding="utf-8")
        files[zp.name] = md5(zp)

        for ext in (f"sdf_{int(a.lidar_cell)}", f"occ_{int(a.lidar_cell)}",
                    f"slabocc_{int(a.lidar_cell)}"):
            src = find(f"{m}.{ext}.npz", BUNDLE, LOCAL)
            if src is not None:
                shutil.copy2(src, maps_dir / src.name)
                files[src.name] = md5(maps_dir / src.name)
        sdf = maps_dir / f"{m}.sdf_{int(a.lidar_cell)}.npz"
        if not sdf.exists() and not a.skip_sdf:
            print(f"  baking SDF for {m} (cell {a.lidar_cell:g}) ...")
            try:
                core = SurfCore(str(bsp_dst), default_config(
                    num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))
                build_sdf(core, cell=a.lidar_cell)
                del core
                if sdf.exists():
                    files[sdf.name] = md5(sdf)
            except Exception as ex:                    # pragma: no cover
                failed.append((m, f"sdf: {type(ex).__name__}: {ex}"))
                continue

        if a.mesh:
            mesh = ROOT / "viewer" / "assets" / f"{m}.mesh.json"
            if not mesh.exists():
                import subprocess
                subprocess.run([sys.executable, str(ROOT / "tools" /
                                                    "export_map.py"),
                                str(bsp_dst), str(mesh)], check=False)

        r = dict(r, files=files)
        manifest.append(r)
        print(f"  {m:<28} {r['finish_kind']:<8} cell {r['goal_cell']:>4g} "
              f"({r['cell_verdict']})  d0 {r['d0']:>10,.0f}  "
              f"{sum(f.stat().st_size for f in maps_dir.glob(m + '.*')) / 1e6:6.1f} MB")

    names = [r["map"] for r in manifest]
    cells = ",".join(f"{r['goal_cell']:g}" for r in manifest)
    mb = sum(f.stat().st_size for f in maps_dir.iterdir()) / 1e6
    doc = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_maps": len(manifest),
        "total_mb": round(mb, 1),
        "lidar_cell": a.lidar_cell,
        "layout": "flat: unpack over the repo root so files land in maps/",
        "maps_arg": ",".join(f"maps/{m}.bsp" for m in names),
        "goal_cell_arg": cells,
        "finish_kinds": {r["map"]: r["finish_kind"] for r in manifest},
        "d0": {r["map"]: r["d0"] for r in manifest},
        "maps": manifest,
        "failed": failed,
    }
    (out / "manifest.json").write_text(json.dumps(doc, indent=1),
                                       encoding="utf-8")

    tar = None
    if a.tar:
        tar = out.with_suffix(".tar.gz")
        with tarfile.open(tar, "w:gz") as tf:
            for p in sorted(maps_dir.iterdir()):
                tf.add(p, arcname=f"maps/{p.name}")
            tf.add(out / "manifest.json", arcname="maps/manifest.json")
        print(f"\ntar: {tar}  {tar.stat().st_size / 1e6:.1f} MB "
              f"(md5 {md5(tar)})")

    print(f"\n{len(manifest)} maps, {mb:.1f} MB, {time.time() - t0:.0f}s")
    if failed:
        print(f"{len(failed)} FAILED:")
        for m, why in failed:
            print(f"   {m:<30} {why}")
    print("\n-- trainer flags --")
    print(f"  --maps {doc['maps_arg']}")
    print(f"  --goal-cell {cells}")
    print(f"  --lidar-cell {a.lidar_cell:g}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
