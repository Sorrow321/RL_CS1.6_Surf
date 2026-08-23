#!/usr/bin/env python3
"""survey_maps.py - which maps in a folder are usable for training TODAY.

Two things gate a surf map, and both are readable from the BSP without a
GPU, a bake, or loading the sim:

1. **Timed zones.** `zones.detect_zones` finds the start/end volumes from
   the standard wiring: a `trigger_multiple` brush that targets a
   `func_button` whose name marks it as the timer control. NOTE what that
   means - the zone is the TRIGGER BRUSH, not the button, so a map whose
   timer you start by *touching* a volume is already handled. A map that
   needs an actual +use press, or whose timer lives in a server-side AMXX
   plugin rather than in the BSP, simply has no such entities and comes out
   as "no zones". Those two failure modes are indistinguishable from the
   BSP alone and are reported together.

2. **Teleports.** `zones._kill_entities` treats EVERY `trigger_teleport`
   with a live destination as a kill volume, because under race rules any
   teleport touch fails the episode. That is correct for a fall-catch net
   and WRONG for a teleport that links two stages of a multi-level map -
   it would make the map unfinishable by construction.

   The discriminator used here: a **death** teleport sends you back to the
   spawn/start, a **link** teleport sends you somewhere else. So classify
   by the distance from each teleport's destination to the nearest spawn
   (`info_player_start`, falling back to the start zone centre). Below
   `--near` units it is a death catch; above, it links stages.

    python tools/survey_maps.py maps_full_dataset --json runs/research/map_survey.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.zones import detect_zones, parse_bsp, _model_index  # noqa: E402


def _origin(ent):
    o = ent.get("origin")
    if not o:
        return None
    try:
        v = [float(x) for x in o.split()[:3]]
        return v if len(v) == 3 else None
    except ValueError:
        return None


def survey_one(path, near):
    ents, boxes = parse_bsp(path)
    zones = detect_zones(path)
    start, end = zones.get("start"), zones.get("end")

    spawns = [_origin(e) for e in ents
              if e.get("classname") in ("info_player_start",
                                        "info_player_deathmatch")]
    spawns = [s for s in spawns if s]
    if not spawns and start:
        spawns = [[(start["mins"][i] + start["maxs"][i]) / 2.0
                   for i in range(3)]]

    # point entities that teleports can aim at
    dests = {e["targetname"]: _origin(e) for e in ents
             if e.get("targetname") and _model_index(e) is None}

    endc = None
    if end:
        endc = [(end["mins"][i] + end["maxs"][i]) / 2.0 for i in range(3)]

    # DISCRIMINATOR. Distance-to-spawn does NOT work: measured over 620 maps
    # the nearest-link distances are a smooth continuum with no gap, because
    # a fall-catch on a big map returns you to a stage checkpoint rather than
    # to info_player_start. What does separate them is how many DISTINCT
    # destinations the map's teleports share, plus whether a teleport carries
    # you TOWARD the finish:
    #   * one shared destination  -> a single fall-catch net; every touch is
    #     a death and the map is usable as-is (surf_src_twist: 184 teleports,
    #     1 destination, 0% end-ward - my distance rule called all 184 stage
    #     links, which was exactly backwards)
    #   * several destinations, none end-ward -> per-stage catch nets: still
    #     all deaths
    #   * end-ward teleports -> genuine stage links, and THOSE are the maps
    #     that cannot be trained under teleport_fail without handling
    death, link, destless, forward = 0, 0, 0, 0
    link_dists = []
    dest_used = {}
    for e in ents:
        if e.get("classname") != "trigger_teleport":
            continue
        mi = _model_index(e)
        if mi is None:
            continue
        tgt = e.get("target")
        if tgt not in dests:
            destless += 1                 # inert pad: GoldSrc no-ops these
            continue
        d = dests[tgt]
        if d is None:
            death += 1
            continue
        dest_used[tgt] = dest_used.get(tgt, 0) + 1
        # does this teleport move the player materially closer to the finish?
        is_fwd = False
        if endc is not None and mi < len(boxes):
            mins, maxs = boxes[mi]
            src = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
            is_fwd = math.dist(d, endc) < math.dist(src, endc) - 512.0
        if is_fwd:
            forward += 1
            link += 1
        else:
            death += 1
        if spawns:
            link_dists.append(min(math.dist(d, s) for s in spawns))

    return {
        "map": Path(path).stem,
        "has_start": start is not None,
        "has_end": end is not None,
        "spawns": len(spawns),
        "tp_death": death,
        "tp_link": link,
        "tp_forward": forward,
        "tp_destless": destless,
        "tp_distinct_dests": len(dest_used),
        "link_dist_min": round(min(link_dists), 1) if link_dists else None,
        "size_mb": round(Path(path).stat().st_size / 1e6, 1),
    }


def classify(r):
    if not (r["has_start"] and r["has_end"]):
        return "no_zones"
    if not r["spawns"]:
        return "no_spawn"
    if r["tp_forward"] > 0:
        return "zones_but_links"
    return "ready"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--near", type=float, default=512.0,
                    help="a teleport landing within this of a spawn is a "
                         "death catch, not a stage link")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    maps = sorted(Path(a.folder).glob("*.bsp"))
    rows, errs = [], []
    for m in maps:
        try:
            rows.append(survey_one(m, a.near))
        except Exception as ex:
            errs.append({"map": m.stem, "error": f"{type(ex).__name__}: {ex}"})

    for r in rows:
        r["class"] = classify(r)
    order = ["ready", "zones_but_links", "no_spawn", "no_zones"]
    counts = {k: sum(1 for r in rows if r["class"] == k) for k in order}

    print(f"{len(maps)} maps, {len(rows)} parsed, {len(errs)} failed\n")
    for k in order:
        print(f"  {k:16s} {counts[k]:4d}  ({100 * counts[k] / max(len(rows), 1):4.1f}%)")
    ready = [r for r in rows if r["class"] == "ready"]
    links = [r for r in rows if r["class"] == "zones_but_links"]
    print(f"\nREADY ({len(ready)}):")
    for r in sorted(ready, key=lambda x: x["map"]):
        print(f"  {r['map']:38s} {r['size_mb']:6.1f} MB  "
              f"tp: {r['tp_death']} deaths / {r['tp_distinct_dests']} dest, "
              f"{r['tp_destless']} inert")
    print(f"\nZONES BUT STAGE LINKS ({len(links)}) - nearest link landing:")
    for r in sorted(links, key=lambda x: -(x["tp_link"]))[:25]:
        print(f"  {r['map']:38s} end-ward {r['tp_forward']:3d}  deaths {r['tp_death']:3d}  "
              f"dests {r['tp_distinct_dests']:3d}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"near": a.near, "counts": counts, "maps": rows, "errors": errs},
            indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
