#!/usr/bin/env python3
"""free_wins.py - what a map that FAILED reachability would need to train.

`verify_maps.py` answers "can the trainer, as configured today, get from the
spawn entity to the finish?". Most of the corpus answers no, and the two
biggest reasons are not defects in the map:

**(a) The sealed start room.** A staged map spawns you in a closed lobby
whose only exit is a `trigger_teleport`. Under race rules every teleport with
a live destination is a kill volume (`zones._kill_entities`), so the spawn's
free-space component is a box with no way out and the finish is in another
component. Seeding the start pool at the teleport's DESTINATION recovers the
map: the geometry was never broken.

**(b) The stage link.** A multi-stage map's finish sits past one or more
links. `src/env.c:601` is `int trig = complete ? 0 : apply_triggers(...)`, so
the goal test runs FIRST and short-circuits the trigger: **a goal box placed
on the link's teleport brush COMPLETES the episode instead of failing it.**
Stage 1 is therefore trainable as its own map with no code change - only a
`<map>.zones.json` whose `end` is the link brush.

Both questions are the same graph. Label the free voxels once (the method and
the error bars are `verify_maps.py`'s - 26-connected `scipy.ndimage.label`
over `slab_occupancy` at `vision.pick_cell`'s cell), then add one directed
edge per teleport from every component its SOURCE brush touches to the
component holding its DESTINATION point, and ask:

  * `hops` - fewest teleports from a spawn component to the finish's. 0 means
    `verify_maps` already passes it; 1 means one seed point fixes it.
  * `seed_points` - the destinations of the first-hop teleports that lead to
    the finish. That is the free win (a), and it is per-map data, not code.
  * `stage1_exits` - links leaving stage 1, i.e. candidate goal boxes for the
    free win (b). Reported with whether the brush has a standable point,
    because a goal box the player cannot occupy is not a finish line.

The measured yield over the 241 in-BSP maps that failed `verify_maps.py`
(round 22): 73 at `hops == 1`, i.e. one seed recovers the WHOLE map; 80 more
at 2-16 hops, where a seed exists but lands past further links and therefore
trains only the last stage; 130 with exactly one stage-1 exit destination, 94
of those with a standable exit brush.

    # one map, or a file of stems
    python tools/free_wins.py --bsp maps_full_dataset/surf_1ramp.bsp \
        --json runs/research/free_wins.json
    python tools/free_wins.py --names fails.txt --maps maps_full_dataset \
        --json runs/research/free_wins.json

    # a type-2 map, whose zone is not in the BSP; and the coarse-cell retry
    python tools/free_wins.py --zones-dir runs/research/gateway_zones ...
    python tools/free_wins.py --cell 32 ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym import SurfCore, default_config                    # noqa: E402
from surfgym.vision import pick_cell                            # noqa: E402
from surfgym.zones import hull_probe, parse_bsp                  # noqa: E402
import verify_maps                                              # noqa: E402
from verify_maps import (_hull_box, base_occupancy, labels_in_box,  # noqa: E402
                         ndimage_label, slab_occupancy_inline, zones_for)


def _vec3(s):
    if not s:
        return None
    try:
        v = [float(x) for x in s.split()[:3]]
    except ValueError:
        return None
    return v if len(v) == 3 else None


def analyse(bsp, verbose=True, cell=None):
    t0 = time.time()
    r = {"map": Path(bsp).stem}
    core = SurfCore(bsp, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    try:
        zones = zones_for(bsp)
        end = zones.get("end")
        spawns = np.array([s[0] for s in core.spawns()], float)
        if end is None or not len(spawns):
            r["error"] = "no end zone or no spawn"
            return r
        cell = float(cell) if cell else pick_cell(core)
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
            sp_labs |= labels_in_box(lab, *_hull_box(p, grow), mins, cell, dims)
        r.update(cell=cell, n_components=int(ncomp),
                 n_spawn_components=len(sp_labs), n_end_components=len(end_labs),
                 finish_in_spawn_component=bool(end_labs & sp_labs))

        # ---- the teleport graph over components
        ents, boxes = parse_bsp(bsp)
        dests = {e["targetname"]: _vec3(e.get("origin")) for e in ents
                 if e.get("targetname") and not e.get("model", "").startswith("*")}
        edges = []              # (src_label_set, dst_label_set, dest_point, name)
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
            d = np.asarray(o, float)
            dst = labels_in_box(lab, d - grow, d + grow, mins, cell, dims)
            edges.append((src, dst, d, e.get("target"),
                          [list(np.round(bmn, 1)), list(np.round(bmx, 1))]))
        r["n_teleports"] = len(edges)

        # ---- BFS over components, teleports as directed edges
        depth = {L: 0 for L in sp_labs}
        frontier, hop = set(sp_labs), 0
        hops = 0 if (end_labs & sp_labs) else None
        first_hop_seeds = []
        while frontier and hops is None and hop < 16:
            hop += 1
            nxt = set()
            for src, dst, d, name, brush in edges:
                if not (src & set(depth)) or not dst:
                    continue
                new = dst - set(depth)
                if not new:
                    continue
                if not (src & frontier):
                    continue
                for L in new:
                    depth[L] = hop
                nxt |= new
            if not nxt:
                break
            if nxt & end_labs:
                hops = hop
            frontier = nxt
        r["hops_to_finish"] = hops

        # ---- (a) which first-hop destinations actually reach the finish?
        # a destination component reaches the finish iff the finish's label is
        # in the set the BFS could paint starting FROM that component alone
        reach_cache = {}

        def reaches_end(start_labs):
            key = frozenset(start_labs)
            if key in reach_cache:
                return reach_cache[key]
            seen, fr = set(start_labs), set(start_labs)
            ok = bool(seen & end_labs)
            k = 0
            while fr and not ok and k < 16:
                k += 1
                nx = set()
                for src, dst, *_ in edges:
                    if src & fr:
                        nx |= dst - seen
                seen |= nx
                fr = nx
                ok = bool(seen & end_labs)
            reach_cache[key] = ok
            return ok

        if hops is not None and hops > 0:
            for src, dst, d, name, brush in edges:
                if not (src & sp_labs) or not dst or (dst & sp_labs):
                    continue                   # not an exit from the start room
                if reaches_end(dst):
                    first_hop_seeds.append({
                        "dest": [round(float(v), 1) for v in d],
                        "target": name, "brush": brush})
        # de-duplicate identical destination points
        seen_pts, uniq = set(), []
        for s in first_hop_seeds:
            k = tuple(s["dest"])
            if k not in seen_pts:
                seen_pts.add(k)
                uniq.append(s)
        r["seed_points"] = uniq
        r["free_win_seed"] = bool(uniq)

        # ---- (b) stage-1 exits: candidate goal boxes on the link brush
        solid = hull_probe(bsp)
        exits = {}
        for src, dst, d, name, brush in edges:
            if not (src & sp_labs) or (dst & sp_labs):
                continue                       # internal catch net, not a link
            bmn, bmx = np.asarray(brush[0], float), np.asarray(brush[1], float)
            axes = [np.linspace(bmn[i], bmx[i], 5) for i in range(3)]
            pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            free_frac = float((~solid(0, pts)).mean())
            exits.setdefault(name, []).append({
                "brush": brush, "free_frac": round(free_frac, 3),
                "dest_reaches_finish": reaches_end(dst)})
        r["stage1_exit_dests"] = len(exits)
        r["stage1_exit_brushes"] = sum(len(v) for v in exits.values())
        r["stage1_exits_standable"] = sum(
            1 for v in exits.values() for b in v if b["free_frac"] > 0)
        r["exits"] = exits
        r["free_win_stage1"] = bool(exits) and not r["finish_in_spawn_component"]
    finally:
        core.close()
    r["secs"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  {r['map']:34s} comps {r.get('n_components', 0):5d} "
              f"finish-in-stage1 {str(r.get('finish_in_spawn_component')):5s} "
              f"hops {str(r.get('hops_to_finish')):4s} "
              f"seeds {len(r.get('seed_points', [])):3d} "
              f"stage1 exits {r.get('stage1_exit_dests', 0):2d}d/"
              f"{r.get('stage1_exit_brushes', 0):3d}b "
              f"({r.get('stage1_exits_standable', 0)} standable) "
              f"{r['secs']:6.1f}s", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", default="maps_full_dataset")
    ap.add_argument("--bsp", nargs="*", default=None)
    ap.add_argument("--names", default=None,
                    help="file of map stems, one per line")
    ap.add_argument("--json", default=None)
    ap.add_argument("--zones-dir", default=None,
                    help="directory of <map>.zones.json to use instead of "
                         "detect_zones (type-2 maps: the Surf Gateway service)")
    ap.add_argument("--cell", type=float, default=None,
                    help="override pick_cell. The 21 maps over ~30k units "
                         "land on 64 u, where the slab dilation is +-32 u "
                         "against a 32 u player hull - re-run those at 32.")
    a = ap.parse_args()
    if a.zones_dir:
        verify_maps.ZONES_DIR = a.zones_dir

    if a.bsp:
        paths = [Path(p) for p in a.bsp]
    else:
        names = [x.strip() for x in Path(a.names).read_text().splitlines()
                 if x.strip()]
        paths = [Path(a.maps) / f"{n}.bsp" for n in names]

    print(f"free-win analysis of {len(paths)} maps")
    rows = []
    for p in paths:
        try:
            rows.append(analyse(p, cell=a.cell))
        except Exception as ex:
            rows.append({"map": p.stem, "error": f"{type(ex).__name__}: {ex}"})
            print(f"  {p.stem:34s} ERROR {ex}", flush=True)
    ok = [x for x in rows if "error" not in x]
    print(f"\n{len(ok)} analysed: seed-point free win {sum(x['free_win_seed'] for x in ok)}; "
          f"stage-1 goal-box free win {sum(x['free_win_stage1'] for x in ok)}; "
          f"neither {sum(1 for x in ok if not x['free_win_seed'] and not x['free_win_stage1'])}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"maps": rows}, indent=1),
                                encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
