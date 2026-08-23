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
  * test = the trainer's OWN rule at every `info_player_start` origin
    (`reset_env` copies that origin verbatim under `spawn_mode` 0 / 2 with
    an empty pool, so these ARE the start states). `GoalField.sample` is
    trilinear over the eight voxel corners around the point, renormalized
    over the honest ones, and `train_fast` takes
    `race_d0 = mean(goal_field.sample(raw origin))` with `scale = 100/d0`.
    So `spawns_reachable` = at least one of those eight corners is free AND
    in the finish's component. `spawns_reachable_hull` is a second, looser
    probe over the UPPER half of the player's standing hull (origin+0..+36
    in z, never dipping through the floor) plus the zone grow; it answers
    "is the start AREA connected" rather than "what does the trainer sample
    at this point". Hull-yes / corner-no is the `spawn_misplaced` verdict.

VERDICTS
--------
`pass` all four checks; `spawn_misplaced` the map is connected but the
spawn entity is not in free space (the trainer would shape on
100/sentinel); `ambiguous` sealed in the slab grid, connected in the
undilated one, seal thinner than 2 cells; `fail` everything else, with
`checks_failed` naming the check.

**False negatives.** Slab occupancy dilates geometry by up to cell/2 per
axis, so a passage narrower than ~1 cell can read as sealed even though
the 32u player hull fits. That is real: at cell 64 this test wrongly calls
`surf_petrus_lite` unreachable, and petrus is a finished, trained map.
Three guards:
  * the cell is the finest that fits the standard budget, so small maps
    (where narrow passages live) are tested at 16u, dilation +-8u;
  * every FAILURE is re-tested at the same cell against the *permissive*
    centre-sampled occupancy, which cannot dilate at all - unreachable
    under both is not a dilation artifact - and at cell/2 (floored at 16u)
    when the permissive grid says the two sides DO connect;
  * `gap_units` = the solid the finish component must be dilated through
    before it touches the spawn's. Each side grows by at most cell/2, so a
    seal of >= 2 cells is thicker than the dilation could have made it and
    the verdict is promoted from `ambiguous` to `fail`. Calibrated on
    sidistic: 224u at cell 32, where the ledger's manual read of that map
    found 64u of worldspawn, and the player hull is 32u wide.

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

# A directory of `<map>.zones.json` files to use INSTEAD of `detect_zones`.
# Type-2 maps (the Surf Gateway service) carry their timer buttons outside
# the BSP entirely, so the only way to put them through the same four checks
# is to hand the zone in. `None` = read the BSP, i.e. every existing call
# behaves exactly as before.
ZONES_DIR = None


def zones_for(bsp):
    """`detect_zones`, or the sidecar in `--zones-dir` when one exists."""
    if ZONES_DIR:
        zp = Path(ZONES_DIR) / f"{Path(bsp).stem}.zones.json"
        if zp.exists():
            doc = json.loads(zp.read_text(encoding="utf-8"))
            return {"start": doc.get("start"), "end": doc.get("end"),
                    "_source": doc.get("source", "file")}
    return detect_zones(bsp)


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


# TWO spawn probes, because they answer two different questions.
#
# `GoalField.sample` is trilinear over the EIGHT voxel corners around the
# point, renormalized over the honest (non-sentinel) ones, and
# `GoalField.reachable` is `sample < reach_max`. So what the trainer
# actually sees at a spawn - `race_d0 = mean(goal_field.sample(raw origin))`
# in train_fast.py, and `scale = 100/d0` - is finite iff at least ONE of
# those eight corners is free AND in the finish's component. That is
# `spawns_reachable`: an exact reproduction of the trainer's own test, no
# grow factor invented for it.
#
# The player's standing hull is 32x32x72 with the origin at its centre, and
# a spawn resting on the floor has its lower half inside the floor's dilated
# voxels. `spawns_reachable_hull` probes the UPPER half of that hull
# (origin+0..+36 in z, never dipping through the floor) plus the zone grow,
# and answers the different question "is the start AREA connected to the
# finish at all". Hull-reachable but not corner-reachable means the map is
# fine and the spawn point is badly placed - nudging it or settling it fixes
# the run. Neither means the map is disconnected the way sidistic is.
def _corner_labels(lab, pos, mins, cell, dims):
    nx, ny, nz = dims
    g = (np.asarray(pos, float) - mins) / cell - 0.5
    i0 = np.floor(g).astype(np.int64)
    out = set()
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ix = int(np.clip(i0[0] + dx, 0, nx - 1))
                iy = int(np.clip(i0[1] + dy, 0, ny - 1))
                iz = int(np.clip(i0[2] + dz, 0, nz - 1))
                v = int(lab[iz, iy, ix])
                if v:
                    out.add(v)
    return out


def _hull_box(p, grow):
    lo = np.asarray(p, float) + np.array([-16.0, -16.0, 0.0]) - grow
    hi = np.asarray(p, float) + np.array([16.0, 16.0, 36.0]) + grow
    return lo, hi


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
    spawn_labs = [_corner_labels(lab, p, mins, cell, dims) for p in spawns]
    hull_labs = [labels_in_box(lab, *_hull_box(p, grow), mins, cell, dims)
                 for p in spawns]
    reach = [bool(s & end_labs) for s in spawn_labs]
    reach_hull = [bool(s & end_labs) for s in hull_labs]
    goal_vox = int(sum((lab == L).sum() for L in end_labs)) if end_labs else 0
    return {
        "cell": cell, "dims": list(dims), "n_vox": int(np.prod(dims)),
        "n_free": n_free, "n_components": int(ncomp),
        "thin_ents": thin,
        "goal_labels": sorted(end_labs), "goal_free_vox": goal_vox,
        "goal_share_of_free": round(goal_vox / max(n_free, 1), 5),
        "spawns_reachable": int(sum(reach)), "spawns_total": len(reach),
        "spawns_reachable_hull": int(sum(reach_hull)),
        "reach": reach,
        "_lab": lab, "_mins": mins, "_dims": dims,
        "_end_labs": end_labs, "_spawn_labs": hull_labs,
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


def _zone_entities(bsp):
    """The two trigger_multiple entities detect_zones would pick, so a caller
    can see whether either carried an origin-brush offset."""
    ents, _ = parse_bsp(bsp)
    buttons = {e.get("targetname", ""): e for e in ents
               if e.get("classname") == "func_button" and e.get("targetname")}

    def role(name):
        low = name.lower()
        tgt = buttons[name].get("target", "").lower()
        if any(k in low or k in tgt for k in ("stop", "off", "end", "finish")):
            return "end"
        if "start" in low or "start" in tgt:
            return "start"
        return None

    out = {}
    for e in ents:
        if e.get("classname") != "trigger_multiple":
            continue
        t = e.get("target", "")
        rl = role(t) if t in buttons else None
        if rl is None and any(k in t.lower() for k in
                              ("mapend", "map_end", "finishzone", "endzone")):
            rl = "end"
        if rl and rl not in out:
            out[rl] = e
    return out


def teleport_bridges(bsp, res):
    """Is the finish's component entered by a TELEPORT? A staged map whose
    finish sits past a stage link is unreachable under teleport_fail by
    design, not by a voxel artifact - and `survey_maps.py`'s end-ward test
    misses the link whenever the source brush happens to sit near the finish
    in straight-line terms. Distinguishing the two matters: one is a map to
    stage-split, the other is a map to drop."""
    lab, mins, dims = res["_lab"], res["_mins"], res["_dims"]
    cell, end_labs = res["cell"], res["_end_labs"]
    sp_labs = set().union(*res["_spawn_labs"]) if res["_spawn_labs"] else set()
    grow = max(0.75 * cell, 20.0)
    ents, boxes = parse_bsp(bsp)
    dests = {e["targetname"]: e.get("origin") for e in ents
             if e.get("targetname") and not e.get("model", "").startswith("*")}
    into_end, into_end_from_spawn = 0, 0
    for e in ents:
        if e.get("classname") != "trigger_teleport":
            continue
        m = e.get("model", "")
        if not m.startswith("*"):
            continue
        try:
            mi = int(m[1:])
        except ValueError:
            continue
        o = dests.get(e.get("target"))
        if mi >= len(boxes) or not o:
            continue
        d = np.array([float(v) for v in o.split()[:3]])
        if not (labels_in_box(lab, d - grow, d + grow, mins, cell, dims)
                & end_labs):
            continue
        into_end += 1
        src = labels_in_box(lab, np.asarray(boxes[mi][0]),
                            np.asarray(boxes[mi][1]), mins, cell, dims)
        if src & sp_labs:
            into_end_from_spawn += 1
    return {"tp_into_finish_component": into_end,
            "tp_into_finish_from_spawn_component": into_end_from_spawn}


# --------------------------------------------------------------------------
# per-map verification
# --------------------------------------------------------------------------
def verify(bsp, retry_fine=True, max_vox=2.6e9, verbose=True, cell=None):
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

        zones = zones_for(bsp)
        r["zone_source"] = zones.get("_source") or (
            "func_button" if (zones.get("end") or {}).get("from") == "func_button"
            else "trigger_multiple" if zones.get("end") else None)
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
        # zone provenance: a start line thousands of units from every spawn,
        # or a zone centre inside solid, means detect_zones picked the wrong
        # brush - the reachability verdict below is then about a phantom box
        emn, emx = np.asarray(end["mins"], float), np.asarray(end["maxs"], float)
        r["end_contents"] = int(core.point_contents((emn + emx) / 2.0))
        start = zones.get("start")
        if start:
            smn = np.asarray(start["mins"], float)
            smx = np.asarray(start["maxs"], float)
            r["start_contents"] = int(core.point_contents((smn + smx) / 2.0))
            q = np.clip(spawns, smn, smx)
            r["d_spawn_startzone"] = int(np.linalg.norm(spawns - q, axis=1).min())
            r["d_startzone_endzone"] = int(np.linalg.norm(
                (smn + smx) / 2.0 - (emn + emx) / 2.0))
        r["zone_origin_offset"] = [
            k for k, ent in _zone_entities(bsp).items()
            if ent.get("origin")
            and ent["origin"].split()[:3] != ["0", "0", "0"]]
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
        # `startsolid` on the standing hull is NOT disqualifying on its own:
        # a spawn placed exactly on the floor reads solid at the hull's
        # bottom plane while the sim settles it fine (surf_src_quickie: 8/8
        # startsolid, 200 neutral ticks, 0 deaths, stuck_ticks 0, onground).
        # The simulator's own verdict - 20 neutral ticks - is the test.
        dead_map = (len(below_kill) == len(spawns) or bool(died.all()))
        if dead_map:
            fails.append("spawn_sane")

        # --- check 3: reachability ---------------------------------------
        cell = float(cell) if cell else pick_cell(core)
        res = component_test(core, cell, spawns, end, slab=True)
        for k in ("cell", "dims", "n_vox", "n_free", "n_components", "thin_ents",
                  "goal_free_vox", "goal_share_of_free",
                  "spawns_reachable", "spawns_reachable_hull", "spawns_total"):
            r[k] = res[k]
        r["reach_mode"] = "slab"
        reachable = res["spawns_reachable"] > 0

        if not reachable:
            r.update(teleport_bridges(bsp, res))
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
            # Only worth paying for when the permissive (undilated) grid
            # says the two sides DO connect: if they are apart even with no
            # dilation at all, no finer cell can join them, and the retry is
            # minutes of CPU for a foregone answer.
            if (retry_fine and r["permissive_reachable"]
                    and fine >= 16.0 and fine_vox <= max_vox):
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
                r["fine_skipped_because"] = (
                    "permissive grid also disconnected" if not
                    r["permissive_reachable"] else
                    "cell floor 16u" if fine < 16.0 else "voxel budget")
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
        # slab says sealed, the undilated grid says connected. That is the
        # dilation's own error bar UNLESS the seal is thicker than the
        # dilation could have made it: each side grows by at most cell/2, so
        # a gap of >= 2 cells of solid is real wall, not model. sidistic is
        # the calibration point - 224u at cell 32, and the ledger's manual
        # read of that map found 64u of worldspawn between the two sides.
        gap = r.get("gap_units")
        r["verdict"] = ("fail" if gap is not None and gap >= 2 * r["cell"]
                        else "ambiguous")
    else:
        r["verdict"] = "fail"
    # the start AREA reaches the finish but the spawn POINT does not sit in
    # free space: the trainer would sample a sentinel d0 at it and shape on
    # 100/sentinel. The map is fine; the spawn entity is not.
    if (r.get("verdict") in ("fail", "ambiguous")
            and r.get("checks_failed") == ["reachable"]
            and r.get("spawns_reachable", 0) == 0
            and r.get("spawns_reachable_hull", 0) > 0):
        r["verdict"] = "spawn_misplaced"
    r["secs"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  {r['map']:34s} {r['verdict']:9s} cell {r.get('cell', 0):4.0f} "
              f"free {r.get('n_free', 0)/1e6:7.2f}M comp {r.get('n_components', 0):5d} "
              f"reach {r.get('spawns_reachable', 0)}"
              f"+{r.get('spawns_reachable_hull', 0)}h"
              f"/{r.get('spawns_total', 0):<3d} "
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
    zones = zones_for(bsp)
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
    spawn_labs = [labels_in_box(lab, *_hull_box(p, grow), mins, cell, dims)
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

    # Door gating. src/bsp.c makes func_door (skin >= -1) a permanently SOLID
    # brush entity and there is no entity-I/O system anywhere in src/ - no
    # button, no target, nothing opens. surf_occupancy_grid clips solid brush
    # entities through a zero-length hull-2 trace, so a closed start-room door
    # is a wall in the grid AND in the physics. If a door's AABB straddles the
    # spawn component and another one, that door IS the seal, and the map is
    # untrainable until the sim can open it.
    door_cls = {"func_door", "func_door_rotating", "func_wall_toggle",
                "func_breakable", "func_train", "func_rotating"}
    for e in ents:
        if e.get("classname") not in door_cls:
            continue
        m = e.get("model", "")
        if not m.startswith("*"):
            continue
        try:
            mi = int(m[1:])
        except ValueError:
            continue
        if mi >= len(boxes):
            continue
        o = np.array([float(v) for v in
                      (e.get("origin") or "0 0 0").split()[:3]])
        bmn = np.asarray(boxes[mi][0]) + o
        bmx = np.asarray(boxes[mi][1]) + o
        touch = labels_in_box(lab, bmn - grow, bmx + grow, mins, cell, dims)
        if len(touch) > 1:
            print(f"    {e['classname']} {m} skin={e.get('skin','0')} "
                  f"{np.round(bmn,0)}..{np.round(bmx,0)} straddles comps "
                  f"{sorted(touch)}"
                  + ("   <== SEALS SPAWN FROM FINISH"
                     if (touch & set(sp_all)) and (touch & end_labs) else ""))
    core.close()


def stage1(bsp, verbose=True):
    """Option (b) feasibility, measured: can stage 1 of a staged map be cut
    out of the BSP with no code change?

    The free-space component holding the spawn IS stage 1 - a stage link is
    a teleport whose SOURCE brush is reachable inside that component and
    whose DESTINATION is not. That is a topological test, so it does not care
    that the brush geometry and the targetnames are the same for links and
    for catch nets (measured: single-feature AUC 0.63, tools/stage_links.py).

    If the real end zone already lies in the spawn's component the map is
    single-stage and was mis-binned by the end-ward distance rule; nothing
    needs splitting."""
    r = {"map": Path(bsp).stem}
    core = SurfCore(bsp, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    try:
        zones = zones_for(bsp)
        end = zones.get("end")
        spawns = np.array([s[0] for s in core.spawns()], float)
        if end is None or not len(spawns):
            r["error"] = "no end zone or no spawn"
            return r
        cell = pick_cell(core)
        occ, mins, dims = base_occupancy(core, cell)
        occ, _ = slab_occupancy_inline(core, cell, occ, mins, dims)
        free = occ == 0
        del occ
        lab, ncomp = ndimage_label(free)
        del free
        grow = max(0.75 * cell, 20.0)
        end_labs = labels_in_box(lab, np.asarray(end["mins"]) - grow,
                                 np.asarray(end["maxs"]) + grow, mins, cell, dims)
        sp_labs = set()
        for p in spawns:
            sp_labs |= labels_in_box(lab, *_hull_box(p, grow),
                                     mins, cell, dims)
        r.update(cell=cell, n_components=int(ncomp),
                 finish_in_spawn_component=bool(end_labs & sp_labs))

        ents, boxes = parse_bsp(bsp)
        dests = {e["targetname"]: e.get("origin") for e in ents
                 if e.get("targetname") and not e.get("model", "").startswith("*")}
        exits = {}          # destination name -> [source brush AABBs]
        internal = 0
        for e in ents:
            if e.get("classname") != "trigger_teleport":
                continue
            m = e.get("model", "")
            if not m.startswith("*"):
                continue
            try:
                mi = int(m[1:])
            except ValueError:
                continue
            o = dests.get(e.get("target"))
            if mi >= len(boxes) or not o:
                continue
            bmn, bmx = np.asarray(boxes[mi][0]), np.asarray(boxes[mi][1])
            src = labels_in_box(lab, bmn, bmx, mins, cell, dims)
            if not (src & sp_labs):
                continue                       # not reachable in stage 1
            d = np.array([float(v) for v in o.split()[:3]])
            dl = labels_in_box(lab, d - grow, d + grow, mins, cell, dims)
            if dl & sp_labs:
                internal += 1                  # a catch net inside stage 1
            else:
                exits.setdefault(e["target"], []).append(
                    [list(np.round(bmn, 1)), list(np.round(bmx, 1))])
        r["stage1_catch_nets"] = internal
        r["stage1_exit_dests"] = len(exits)
        r["stage1_exit_brushes"] = sum(len(v) for v in exits.values())
        r["exits"] = {k: v for k, v in exits.items()}
        if verbose:
            print(f"  {r['map']:34s} cell {cell:4.0f} comps {ncomp:5d} "
                  f"finish-in-stage1 {str(r['finish_in_spawn_component']):5s} "
                  f"stage1 exits {r['stage1_exit_dests']} dest / "
                  f"{r['stage1_exit_brushes']} brush   catch nets "
                  f"{internal}", flush=True)
    finally:
        core.close()
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", default=None, help="map_survey.json")
    ap.add_argument("--maps", default="maps_full_dataset", help="folder of .bsp")
    ap.add_argument("--only-class", default="ready")
    ap.add_argument("--bsp", nargs="*", default=None, help="explicit .bsp paths")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-retry-fine", action="store_true")
    ap.add_argument("--zones-dir", default=None,
                    help="directory of <map>.zones.json to use instead of "
                         "detect_zones - the only way to check a type-2 map, "
                         "whose buttons are not in the BSP at all")
    ap.add_argument("--cell", type=float, default=None,
                    help="override pick_cell for check 3 (the six 30-48k-unit "
                         "maps land on 64u, where the shaping field is known "
                         "to tunnel through thin floors - re-run those at 32)")
    ap.add_argument("--stage1", action="store_true",
                    help="option (b) feasibility: locate stage 1's exits")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--diagnose", action="store_true",
                    help="explain --bsp maps' components instead of verifying")
    a = ap.parse_args()
    if a.zones_dir:
        global ZONES_DIR
        ZONES_DIR = a.zones_dir

    if a.bsp:
        paths = [Path(p) for p in a.bsp]
    else:
        s = json.loads(Path(a.survey).read_text(encoding="utf-8"))
        names = sorted(m["map"] for m in s["maps"] if m["class"] == a.only_class)
        paths = [Path(a.maps) / f"{n}.bsp" for n in names]

    if a.limit:
        paths = paths[:a.limit]
    if a.diagnose:
        for p in paths:
            diagnose(p)
        return 0
    if a.stage1:
        print(f"stage-1 analysis of {len(paths)} maps")
        rows = []
        for p in paths:
            try:
                rows.append(stage1(p))
            except Exception as ex:
                rows.append({"map": p.stem, "error": f"{type(ex).__name__}: {ex}"})
                print(f"  {p.stem:34s} ERROR {ex}", flush=True)
        ok = [x for x in rows if "error" not in x]
        mis = sum(x["finish_in_spawn_component"] for x in ok)
        one = sum(1 for x in ok if not x["finish_in_spawn_component"]
                  and x["stage1_exit_dests"] == 1)
        none_ = sum(1 for x in ok if not x["finish_in_spawn_component"]
                    and x["stage1_exit_dests"] == 0)
        print(f"{len(ok)} analysed: finish already in the spawn's component "
              f"(single-stage, mis-binned) {mis}; staged with exactly ONE "
              f"stage-1 exit destination {one}; staged with NO reachable exit "
              f"{none_}")
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.json).write_text(json.dumps({"maps": rows}, indent=1),
                                    encoding="utf-8")
            print(f"wrote {a.json}")
        return 0

    print(f"verifying {len(paths)} maps\n")
    rows = []
    for p in paths:
        try:
            rows.append(verify(p, retry_fine=not a.no_retry_fine,
                               cell=a.cell))
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
