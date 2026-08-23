#!/usr/bin/env python3
"""stage_links.py - anatomy of the maps `survey_maps.py` classes
`zones_but_links`: timed zones present, but at least one trigger_teleport
carries the player TOWARD the finish.

Under race rules `zones._kill_entities` makes EVERY live trigger_teleport a
kill volume (`teleport_fail`), which is right for a fall-catch net and
makes a staged map unfinishable by construction. This tool measures what
the BSP actually offers for telling the two apart, and what a per-stage
split would have to work with:

  * per teleport: source brush AABB (footprint, thickness, height in the
    map), destination origin + targetname, and whether it moves the player
    materially closer to the end zone;
  * per map: how many DISTINCT end-ward destinations there are (one = a
    single stage advance; many = a chain), and how many source brushes feed
    each of them (that set is the candidate end zone for a per-stage split);
  * separability: do end-ward sources and catch nets differ in geometry
    (nets are wide thin horizontal slabs low in the map) or in naming?

    python tools/stage_links.py --survey runs/research/map_survey.json \
        --maps maps_full_dataset --json runs/research/stage_links.json
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.zones import detect_zones, parse_bsp, _model_index   # noqa: E402

FWD_MARGIN = 512.0        # survey_maps.py's own end-ward threshold


def _origin(ent):
    o = ent.get("origin")
    if not o:
        return None
    try:
        v = [float(x) for x in o.split()[:3]]
        return v if len(v) == 3 else None
    except ValueError:
        return None


def analyse(path):
    ents, boxes = parse_bsp(path)
    zones = detect_zones(path)
    start, end = zones.get("start"), zones.get("end")
    if not end:
        return None
    endc = [(end["mins"][i] + end["maxs"][i]) / 2.0 for i in range(3)]
    startc = None
    if start:
        startc = [(start["mins"][i] + start["maxs"][i]) / 2.0 for i in range(3)]

    world = boxes[0] if boxes else ([0, 0, 0], [1, 1, 1])
    wz0, wz1 = world[0][2], world[1][2]
    span_z = max(wz1 - wz0, 1.0)

    spawns = [_origin(e) for e in ents
              if e.get("classname") in ("info_player_start",
                                        "info_player_deathmatch")]
    spawns = [s for s in spawns if s]

    dests = {e["targetname"]: _origin(e) for e in ents
             if e.get("targetname") and _model_index(e) is None}

    tps = []
    for e in ents:
        if e.get("classname") != "trigger_teleport":
            continue
        mi = _model_index(e)
        if mi is None or mi >= len(boxes):
            continue
        tgt = e.get("target")
        d = dests.get(tgt)
        mins, maxs = boxes[mi]
        dx, dy, dz = (maxs[i] - mins[i] for i in range(3))
        src = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
        rec = {
            "target": tgt, "targetname": e.get("targetname", ""),
            "live": tgt in dests and d is not None,
            "dest": d, "src": [round(v, 1) for v in src],
            "dims": [round(dx, 1), round(dy, 1), round(dz, 1)],
            "footprint": round(dx * dy, 1),
            "min_dim": round(min(dx, dy, dz), 1),
            "vol_e6": round(dx * dy * dz / 1e6, 3),
            "z_frac": round((src[2] - wz0) / span_z, 3),
            "horizontal_slab": bool(dz <= 64.0 and min(dx, dy) >= 256.0),
        }
        if rec["live"]:
            rec["d_src_end"] = round(math.dist(src, endc), 1)
            rec["d_dest_end"] = round(math.dist(d, endc), 1)
            rec["gain_end"] = round(rec["d_src_end"] - rec["d_dest_end"], 1)
            rec["fwd"] = rec["gain_end"] > FWD_MARGIN
            if spawns:
                rec["d_dest_spawn"] = round(
                    min(math.dist(d, s) for s in spawns), 1)
            if startc:
                rec["d_dest_start"] = round(math.dist(d, startc), 1)
        else:
            rec["fwd"] = False
        tps.append(rec)

    live = [t for t in tps if t["live"]]
    fwd = [t for t in live if t["fwd"]]
    back = [t for t in live if not t["fwd"]]
    fwd_dests = Counter(t["target"] for t in fwd)
    all_dests = Counter(t["target"] for t in live)

    def med(vals):
        v = sorted(vals)
        return round(v[len(v) // 2], 1) if v else None

    return {
        "map": Path(path).stem,
        "size_mb": round(Path(path).stat().st_size / 1e6, 1),
        "world_z": [round(wz0, 1), round(wz1, 1)],
        "n_tp": len(tps), "n_live": len(live), "n_dead": len(tps) - len(live),
        "n_fwd": len(fwd), "n_back": len(back),
        "n_dests": len(all_dests), "n_fwd_dests": len(fwd_dests),
        "fwd_dests": dict(fwd_dests),
        "fwd_dest_names": sorted(fwd_dests),
        "back_dest_names": sorted({t["target"] for t in back}),
        "single_link": len(fwd_dests) == 1,
        "single_link_sources": (list(fwd_dests.values())[0]
                                if len(fwd_dests) == 1 else None),
        "fwd_med_footprint": med([t["footprint"] for t in fwd]),
        "back_med_footprint": med([t["footprint"] for t in back]),
        "fwd_med_mindim": med([t["min_dim"] for t in fwd]),
        "back_med_mindim": med([t["min_dim"] for t in back]),
        "fwd_med_zfrac": med([t["z_frac"] for t in fwd]),
        "back_med_zfrac": med([t["z_frac"] for t in back]),
        "fwd_slab_frac": round(sum(t["horizontal_slab"] for t in fwd)
                               / max(len(fwd), 1), 3),
        "back_slab_frac": round(sum(t["horizontal_slab"] for t in back)
                                / max(len(back), 1), 3),
        "fwd_med_gain": med([t["gain_end"] for t in fwd]),
        "d_start_end": round(math.dist(startc, endc), 1) if startc else None,
        "teleports": tps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", default="runs/research/map_survey.json")
    ap.add_argument("--maps", default="maps_full_dataset")
    ap.add_argument("--only-class", default="zones_but_links")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    s = json.loads(Path(a.survey).read_text(encoding="utf-8"))
    names = sorted(m["map"] for m in s["maps"] if m["class"] == a.only_class)
    rows, errs = [], []
    for n in names:
        try:
            r = analyse(Path(a.maps) / f"{n}.bsp")
            if r:
                rows.append(r)
        except Exception as ex:
            errs.append({"map": n, "error": f"{type(ex).__name__}: {ex}"})

    n1 = sum(r["single_link"] for r in rows)
    print(f"{len(rows)} maps parsed, {len(errs)} errors")
    print(f"  exactly ONE end-ward destination : {n1} ({100*n1/max(len(rows),1):.0f}%)")
    for k in (2, 3, 5, 10):
        c = sum(1 for r in rows if r["n_fwd_dests"] >= k)
        print(f"  >= {k} end-ward destinations      : {c}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"maps": rows, "errors": errs},
                                           indent=1), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
