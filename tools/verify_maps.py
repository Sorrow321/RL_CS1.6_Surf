#!/usr/bin/env python3
"""verify_maps.py - does a BSP-classified "ready" map actually train?

`survey_maps.py` classifies a map from its entities alone: start/end zones
detected, every trigger_teleport a death catch. That is a NECESSARY
condition, not proof. Three things it cannot see, in increasing order of
how expensive they are to discover the hard way:

1. **It loads.** `surf_create` rejects malformed/oversized BSPs, and a map
   with zero `info_player_start` resets every env to the origin.
2. **The spawn is sane.** A spawn inside geometry, on a lethal
   `trigger_hurt`, or under water fails the episode on tick 1 for every
   env, forever.
3. **The finish is REACHABLE FROM THE SPAWN.** This is the killer.
   `surf_src_sidistic` passes every entity check and is untrainable: its
   start sits in a different free-space component from its finish, so the
   geodesic bake covers 0.2% of free space, `d0` comes back as the
   unreachable sentinel and `scale = 100/d0` shapes on a field the agent
   can never enter (round 19, xSID - ledger). Training such a map produces
   a null run that looks like a hard map.

METHOD for (3), and its error bars
----------------------------------
`build_goal_field` is authoritative but costs a GPU bake of minutes per
map. Its graph, however, is a plain 26-connected lattice over the FREE
voxels of `vision.slab_occupancy` at `vision.pick_cell`'s cell - so
"is the spawn in the goal's component?" is a connected-component label,
not a shortest path. `scipy.ndimage.label` answers it on the CPU in
seconds, on the identical grid at the identical cell, and reproduces the
xSID bake's numbers exactly (4 components at cell 32, finish component
0.21 M voxels, 0/2 spawns reachable).

  * cell = `pick_cell(core)` - the project's own 700M-voxel budget, so the
    grid is the one a bake would use. 16u for a normal 8k-unit map, 32u
    for a source port, 64u for the 48k-unit monsters.
  * occupancy = `slab_occupancy` semantics (13-point per-axis lattice at
    cell/4 + exact AABB raster of thin solid brush entities), rebuilt
    in-process so nothing is written next to the .bsp.
  * seed = the end zone AABB grown by `max(0.75*cell, 20)` - the same
    inflation `goalfield._zone_seed_box` uses so a 1u timer curtain still
    covers a voxel layer.
  * test = every `info_player_start` origin, grown the same way. Under
    `spawn_mode` 0/2-with-empty-pool the core copies that origin verbatim
    (src/env.c reset_env), so these ARE the start states.

**False negatives.** Slab occupancy dilates geometry by up to cell/2 per
axis, so a passage narrower than ~1 cell can read as sealed even though
the 32u player hull fits. That is real: at cell 64 this test wrongly calls
`surf_petrus_lite` unreachable, and petrus is a finished, trained map.
Two guards:
  * the cell is the finest that fits the standard budget, so small maps
    (where narrow passages live) are tested at 16u, dilation +-8u;
  * every FAILURE is re-tested at cell/2 (`--retry-fine`) and, at the same
    cell, against the *permissive* centre-sampled occupancy, which cannot
    dilate at all. A map that is unreachable under both is disconnected for
    reasons no resolution will fix; a map reachable under the permissive
    grid only is reported `ambiguous` - the link exists but is thinner than
    the wall model, exactly the sidistic situation, where the seal measured
    64u of solid and a 64u wall really is a wall.
  * for failures, `gap_units` reports how far the goal component has to be
    dilated (through solid) before it touches the spawn's - the thickness
    of the seal. >= 64u is a wall; 1 cell is a modelling artifact.

**False positives.** A slab lattice at cell/4 can still thread a brush
thinner than ~cell/4 (4u at cell 16, 16u at cell 64), which would merge two
components that are really sealed. Thin *entity* brushes are rasterized
exactly, so this is limited to thin worldspawn geometry on the 64u maps.

Usage
-----
    python tools/verify_maps.py --survey runs/research/map_survey.json \
        --maps maps_full_dataset --json runs/research/maps_verified.json

    # sanity check against three maps with known answers
    python tools/verify_maps.py --bsp maps/surf_petrus_lite.bsp \
        maps/surf_src_cannonball.bsp maps/surf_src_sidistic.bsp
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym import SurfCore, default_config                    # noqa: E402
from surfgym.vision import SOLID_ENT_CLASSES, grid_dims, pick_cell  # noqa: E402
from surfgym.zones import detect_zones, parse_bsp               # noqa: E402

STRUCT = np.ones((3, 3, 3), bool)      # 26-connected: _bfs_geodesic's graph
NEUTRAL = np.array([[7, 3, 1, 1, 0, 0]], np.int32)  # yaw 0, pitch 0, no move
SMOKE_TICKS = 20                       # 0.2 s - long enough to die, too short to fall


# --------------------------------------------------------------------------
# occupancy (in-process; never writes next to the .bsp)
# --------------------------------------------------------------------------
def base_occupancy(core, cell):
    mins, nx, ny, nz = grid_dims(core, cell)
    return core.occupancy_grid(mins, cell, nx, ny, nz), mins, (nx, ny, nz)


def slab_occupancy_inline(core, cell, occ, mins, dims):
    """vision.slab_occupancy, minus the npz cache write. ``occ`` is consumed."""
    nx, ny, nz = dims
    step = cell / 4.0
    for axis in range(3):
        for k in (-2, -1, 1, 2):
            off = np.zeros(3)
            off[axis] = k * step
            occ |= core.occupancy_grid(mins + off, cell, nx, ny, nz)
    ents, bboxes = parse_bsp(core.bsp_path)
    thin = 0
    for ent in ents:
        model = ent.get("model", "")
        if (ent.get("classname") == "func_conveyor"
                and int(float(ent.get("spawnflags", 0))) & 2):
            continue                                  # SF_CONVEYOR_NOTSOLID
        if ent.get("classname") not in SOLID_ENT_CLASSES or not model.startswith("*"):
            continue
        try:
            mi = int(model[1:])
        except ValueError:
            continue
        if mi >= len(bboxes):
            continue
        bmn, bmx = np.asarray(bboxes[mi][0]), np.asarray(bboxes[mi][1])
        if float((bmx - bmn).min()) > 20.0:
            continue                                  # thick: lattice-sample it
        lo = np.maximum(np.floor((bmn - mins) / cell - .001), 0).astype(int)
        hi = np.minimum(np.ceil((bmx - mins) / cell + .001),
                        [nx, ny, nz]).astype(int)
        occ[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = 1
        thin += 1
    return occ, thin


# --------------------------------------------------------------------------
# component bookkeeping
# --------------------------------------------------------------------------
def _box_slice(bmn, bmx, mins, cell, dims):
    nx, ny, nz = dims
    lo = np.maximum(np.floor((np.asarray(bmn, float) - mins) / cell), 0).astype(int)
    hi = np.minimum(np.ceil((np.asarray(bmx, float) - mins) / cell) + 1,
                    [nx, ny, nz]).astype(int)
    if np.any(hi <= lo):
        return None
    return (slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0]))


def labels_in_box(lab, bmn, bmx, mins, cell, dims):
    sl = _box_slice(bmn, bmx, mins, cell, dims)
    if sl is None:
        return set()
    u = np.unique(lab[sl])
    return {int(v) for v in u if v != 0}


def component_test(core, cell, spawns, endzone, slab=True):
    """-> dict: which spawns share a free-space component with the end zone."""
    occ, mins, dims = base_occupancy(core, cell)
    thin = 0
    if slab:
        occ, thin = slab_occupancy_inline(core, cell, occ, mins, dims)
    free = occ == 0
    del occ
    n_free = int(free.sum())
    lab, ncomp = ndimage_label(free)
    del free
    grow = max(0.75 * cell, 20.0)
    end_labs = labels_in_box(lab, np.asarray(endzone["mins"]) - grow,
                             np.asarray(endzone["maxs"]) + grow, mins, cell, dims)
    spawn_labs = [labels_in_box(lab, p - grow, p + grow, mins, cell, dims)
                  for p in spawns]
    reach = [bool(s & end_labs) for s in spawn_labs]
    goal_vox = int(sum((lab == L).sum() for L in end_labs)) if end_labs else 0
    return {
        "cell": cell, "dims": list(dims), "n_vox": int(np.prod(dims)),
        "n_free": n_free, "n_components": int(ncomp),
        "thin_ents": thin,
        "goal_labels": sorted(end_labs), "goal_free_vox": goal_vox,
        "goal_share_of_free": round(goal_vox / max(n_free, 1), 5),
        "spawns_reachable": int(sum(reach)), "spawns_total": len(reach),
        "reach": reach,
        "_lab": lab, "_mins": mins, "_dims": dims,
        "_end_labs": end_labs, "_spawn_labs": spawn_labs,
    }


def ndimage_label(free):
    from scipy import ndimage
    return ndimage.label(free, structure=STRUCT)


def seal_thickness(res, max_cells=None):
    """Cells of solid between the goal component and the nearest spawn
    component (dilating the goal THROUGH solid until they touch)."""
    from scipy import ndimage
    if max_cells is None:                 # measure out to ~320 units either way
        max_cells = int(np.clip(round(320.0 / res["cell"]), 5, 24))
    lab, end_labs = res["_lab"], res["_end_labs"]
    spawn_labs = set().union(*res["_spawn_labs"]) if res["_spawn_labs"] else set()
    spawn_labs -= end_labs
    if not end_labs or not spawn_labs:
        return None
    goal = np.isin(lab, list(end_labs))
    want = np.isin(lab, list(spawn_labs))
    # crop to the goal bbox + max_cells so the dilation is cheap
    idx = np.array(np.nonzero(goal))
    lo = np.maximum(idx.min(1) - max_cells - 1, 0)
    hi = np.minimum(idx.max(1) + max_cells + 2, np.array(goal.shape))
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    g, w = goal[sl].copy(), want[sl]
    for k in range(1, max_cells + 1):
        g = ndimage.binary_dilation(g, structure=STRUCT)
        if (g & w).any():
            return k
    return None


# --------------------------------------------------------------------------
# per-map verification
# --------------------------------------------------------------------------
def verify(bsp, retry_fine=True, max_vox=2.6e9, verbose=True):
    t0 = time.time()
    r = {"map": Path(bsp).stem, "size_mb": round(Path(bsp).stat().st_size / 1e6, 1)}
    fails = []

    # --- check 1: it loads ------------------------------------------------
    try:
        core = SurfCore(bsp, default_config(num_envs=8, spawn_mode=2,
                                            lidar_w=0, lidar_h=0))
        core.reset(0)
        r["loads"] = True
    except Exception as ex:
        r.update(loads=False, load_error=f"{type(ex).__name__}: {ex}",
                 checks_failed=["loads"], verdict="fail", secs=round(time.time() - t0, 1))
        return r

    try:
        mn, mx = (np.asarray(v, float) for v in core.map_bounds())
        r["extent"] = [int(v) for v in (mx - mn)]
        r["bounds_min"] = [int(v) for v in mn]
        spawn_ents = core.spawns()
        spawns = np.array([s[0] for s in spawn_ents], float) if spawn_ents else \
            np.zeros((0, 3))
        r["n_spawns"] = len(spawns)

        zones = detect_zones(bsp)
        r["has_start"] = zones.get("start") is not None
        r["has_end"] = zones.get("end") is not None
        end = zones.get("end")
        if end is None or not len(spawns):
            fails.append("zones" if end is None else "spawn_sane")
            r.update(checks_failed=fails, verdict="fail",
                     secs=round(time.time() - t0, 1))
            return r

        # --- check 2: spawn sanity ---------------------------------------
        kill_z = float(mn[2]) - 256.0          # cfg.kill_z auto (src/env.c:423)
        in_solid, below_kill = [], []
        for i, p in enumerate(spawns):
            tr = core.trace(p, p + np.array([0.0, 0.0, -1.0]), 0)   # standing hull
            if tr.startsolid or tr.allsolid:
                in_solid.append(i)
            if p[2] < kill_z:
                below_kill.append(i)
        # 20 neutral ticks over 8 envs: catches trigger_hurt / water / stuck
        core.reset(0)
        acts = np.repeat(NEUTRAL, core.num_envs, axis=0)
        died = np.zeros(core.num_envs, bool)
        for _ in range(SMOKE_TICKS):
            _, _, done, _, _ = core.step(acts)
            died |= done.astype(bool)
        r.update(spawns_in_solid=len(in_solid), spawns_below_kill=len(below_kill),
                 kill_z=round(kill_z, 1),
                 smoke_died=int(died.sum()), smoke_envs=int(core.num_envs))
        # DISQUALIFYING only when it takes the whole map out: one embedded
        # spawn of 32 wastes 1/32 of episodes, it does not make the map
        # untrainable. Partial damage is recorded as a warning instead.
        warn = []
        if in_solid:
            warn.append(f"{len(in_solid)}/{len(spawns)} spawns inside geometry")
        if below_kill:
            warn.append(f"{len(below_kill)}/{len(spawns)} spawns below kill_z")
        if died.any() and not died.all():
            warn.append(f"{int(died.sum())}/{core.num_envs} envs died in "
                        f"{SMOKE_TICKS} neutral ticks")
        dead_map = (len(in_solid) == len(spawns) or len(below_kill) == len(spawns)
                    or bool(died.all()))
        if dead_map:
            fails.append("spawn_sane")

        # --- check 3: reachability ---------------------------------------
        cell = pick_cell(core)
        res = component_test(core, cell, spawns, end, slab=True)
        for k in ("cell", "dims", "n_vox", "n_free", "n_components", "thin_ents",
                  "goal_free_vox", "goal_share_of_free",
                  "spawns_reachable", "spawns_total"):
            r[k] = res[k]
        r["reach_mode"] = "slab"
        reachable = res["spawns_reachable"] > 0

        if not reachable:
            # permissive bracket at the same cell: no dilation at all
            perm = component_test(core, cell, spawns, end, slab=False)
            r["permissive_reachable"] = perm["spawns_reachable"] > 0
            r["permissive_components"] = perm["n_components"]
            r["seal_cells"] = seal_thickness(res)
            r["gap_units"] = None if r["seal_cells"] is None \
                else int(r["seal_cells"] * cell)
            for k in ("_lab", "_mins", "_dims", "_end_labs", "_spawn_labs"):
                perm.pop(k, None)
            del perm
            # finer retry: halve the cell if it fits
            fine = cell / 2.0
            fine_vox = float(np.prod([math.ceil(e / fine) for e in
                                      (mx - mn) + 8.0 * fine]))
            if retry_fine and fine >= 16.0 and fine_vox <= max_vox:
                fres = component_test(core, fine, spawns, end, slab=True)
                r["fine_cell"] = fine
                r["fine_n_components"] = fres["n_components"]
                r["fine_goal_free_vox"] = fres["goal_free_vox"]
                r["fine_spawns_reachable"] = fres["spawns_reachable"]
                r["fine_seal_cells"] = seal_thickness(fres)
                r["fine_gap_units"] = None if r["fine_seal_cells"] is None \
                    else int(r["fine_seal_cells"] * fine)
                if fres["spawns_reachable"] > 0:
                    reachable = True
                    r["reach_mode"] = "slab@fine"
                    r["spawns_reachable"] = fres["spawns_reachable"]
                for k in ("_lab", "_mins", "_dims", "_end_labs", "_spawn_labs"):
                    fres.pop(k, None)
                del fres
            else:
                r["fine_cell"] = None
                r["fine_skipped_vox"] = fine_vox
        for k in ("_lab", "_mins", "_dims", "_end_labs", "_spawn_labs"):
            res.pop(k, None)
        del res
        r["reachable"] = bool(reachable)
        if not reachable:
            fails.append("reachable")
        elif r["spawns_reachable"] < r["spawns_total"]:
            warn.append(f"{r['spawns_total'] - r['spawns_reachable']}/"
                        f"{r['spawns_total']} spawns in a component the finish "
                        f"is not in")
        r["warnings"] = warn

        # --- check 4: d0 + extent ----------------------------------------
        emn, emx = np.asarray(end["mins"], float), np.asarray(end["maxs"], float)
        q = np.clip(spawns, emn, emx)
        d = np.linalg.norm(spawns - q, axis=1)
        r["d0_euclid_min"] = int(d.min())
        r["d0_euclid_mean"] = int(d.mean())
        r["d0_euclid_max"] = int(d.max())
        r["end_center"] = [int(v) for v in (emn + emx) / 2.0]
        r["free_volume_e9"] = round(r["n_free"] * r["cell"] ** 3 / 1e9, 2)
    finally:
        core.close()

    r["checks_failed"] = fails
    if not fails:
        r["verdict"] = "pass"
    elif fails == ["reachable"] and r.get("permissive_reachable"):
        r["verdict"] = "ambiguous"
    else:
        r["verdict"] = "fail"
    r["secs"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  {r['map']:34s} {r['verdict']:9s} cell {r.get('cell', 0):4.0f} "
              f"free {r.get('n_free', 0)/1e6:7.2f}M comp {r.get('n_components', 0):5d} "
              f"reach {r.get('spawns_reachable', 0)}/{r.get('spawns_total', 0):<3d} "
              f"d0 {r.get('d0_euclid_mean', 0):7d} {r['secs']:6.1f}s"
              + ("  FAILED: " + ",".join(fails) if fails else "")
              + ("  WARN: " + "; ".join(r.get("warnings", []))
                 if r.get("warnings") else ""), flush=True)
    return r


def diagnose(bsp, cell=None):
    """Why a map failed: what the free-space components ARE, and whether the
    goal's component is entered only through a teleport (a finish room you
    are meant to be TELEPORTED into is unreachable under teleport_fail by
    design, not by a voxel artifact)."""
    from scipy import ndimage
    core = SurfCore(bsp, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    zones = detect_zones(bsp)
    end, start = zones.get("end"), zones.get("start")
    spawns = np.array([s[0] for s in core.spawns()], float)
    cell = cell or pick_cell(core)
    occ, mins, dims = base_occupancy(core, cell)
    occ, _ = slab_occupancy_inline(core, cell, occ, mins, dims)
    free = occ == 0
    del occ
    lab, ncomp = ndimage.label(free, structure=STRUCT)
    del free
    grow = max(0.75 * cell, 20.0)
    end_labs = labels_in_box(lab, np.asarray(end["mins"]) - grow,
                             np.asarray(end["maxs"]) + grow, mins, cell, dims)
    spawn_labs = [labels_in_box(lab, p - grow, p + grow, mins, cell, dims)
                  for p in spawns]
    sizes = np.bincount(lab.ravel())
    print(f"=== {Path(bsp).stem}  cell {cell:g}  dims {dims}  "
          f"{ncomp} free components")
    print(f"  end zone box {np.round(end['mins'],0)} .. {np.round(end['maxs'],0)}"
          f"  -> labels {sorted(end_labs) or 'NONE (seed box has no free voxel)'}")
    sp_all = sorted(set().union(*spawn_labs)) if spawn_labs else []
    print(f"  {len(spawns)} spawns -> labels {sp_all}")
    objs = ndimage.find_objects(lab)
    order = np.argsort(-sizes[1:]) + 1
    for L in list(order[:8]) + [x for x in sorted(end_labs) + sp_all
                                if x not in order[:8]]:
        sl = objs[L - 1]
        lo = mins + np.array([sl[2].start, sl[1].start, sl[0].start]) * cell
        hi = mins + np.array([sl[2].stop, sl[1].stop, sl[0].stop]) * cell
        tag = []
        if L in end_labs:
            tag.append("FINISH")
        if L in sp_all:
            tag.append("SPAWN")
        print(f"    comp {L:5d} {sizes[L]/1e6:8.3f} Mvox "
              f"({100*sizes[L]/max(sizes[1:].sum(),1):5.1f}% of free) "
              f"x {lo[0]:8.0f}..{hi[0]:8.0f} y {lo[1]:8.0f}..{hi[1]:8.0f} "
              f"z {lo[2]:8.0f}..{hi[2]:8.0f} {' '.join(tag)}")
    # do any teleports bridge into the finish component?
    ents, boxes = parse_bsp(bsp)
    dests = {e["targetname"]: e.get("origin") for e in ents
             if e.get("targetname") and not (e.get("model", "").startswith("*"))}
    bridges = []
    for e in ents:
        if e.get("classname") != "trigger_teleport":
            continue
        m = e.get("model", "")
        if not m.startswith("*"):
            continue
        mi = int(m[1:])
        if (mi >= len(boxes) or e.get("target") not in dests
                or not dests[e["target"]]):
            continue
        d = np.array([float(v) for v in dests[e["target"]].split()[:3]])
        dl = labels_in_box(lab, d - grow, d + grow, mins, cell, dims)
        bmn, bmx = np.asarray(boxes[mi][0]), np.asarray(boxes[mi][1])
        sl_ = labels_in_box(lab, bmn, bmx, mins, cell, dims)
        bridges.append((e["target"], sorted(sl_)[:4], sorted(dl)))
    into_end = [b for b in bridges if set(b[2]) & end_labs]
    from_spawn = [b for b in into_end if set(b[1]) & set(sp_all)]
    print(f"  teleports landing INSIDE the finish component: {len(into_end)}"
          f" (of which sourced in the spawn component: {len(from_spawn)})")
    for b in into_end[:6]:
        print(f"    src comps {b[1]} -> {b[0]} -> dest comps {b[2]}")
    core.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", default=None, help="map_survey.json")
    ap.add_argument("--maps", default="maps_full_dataset", help="folder of .bsp")
    ap.add_argument("--only-class", default="ready")
    ap.add_argument("--bsp", nargs="*", default=None, help="explicit .bsp paths")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-retry-fine", action="store_true")
    ap.add_argument("--diagnose", action="store_true",
                    help="explain --bsp maps' components instead of verifying")
    a = ap.parse_args()

    if a.bsp:
        paths = [Path(p) for p in a.bsp]
    else:
        s = json.loads(Path(a.survey).read_text(encoding="utf-8"))
        names = sorted(m["map"] for m in s["maps"] if m["class"] == a.only_class)
        paths = [Path(a.maps) / f"{n}.bsp" for n in names]

    if a.diagnose:
        for p in paths:
            diagnose(p)
        return 0

    print(f"verifying {len(paths)} maps\n")
    rows = []
    for p in paths:
        try:
            rows.append(verify(p, retry_fine=not a.no_retry_fine))
        except Exception as ex:
            import traceback
            traceback.print_exc()
            rows.append({"map": p.stem, "verdict": "error",
                         "error": f"{type(ex).__name__}: {ex}",
                         "checks_failed": ["error"]})
            print(f"  {p.stem:34s} ERROR {ex}", flush=True)

    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n" + json.dumps(counts, indent=1))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"counts": counts, "maps": rows}, indent=1), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
