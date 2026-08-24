#!/usr/bin/env python3
"""zone_audit.py - are the three zone sources consistent, and is pad 64 right?

Three mechanisms mark start/finish on a surf map (CLAUDE.md 4b):

  type 1  `trigger_multiple` in the BSP     - a real volume you fly through
  type 2  the Surf Gateway service          - AMXX-spawned buttons, not in the BSP
  type 3  `func_button` in the BSP          - `target = counter_start`/`counter_off`

Types 2 and 3 are BUTTONS. The simulator cannot press one (`pm.c` handles
movement buttons, not `+use`), so completion is a pure AABB test on a box
padded by `zones.BUTTON_PAD` = 64 u, the engine's `+use` reach. This tool
answers the two questions that raises.

**Do the sources agree where they overlap?** Types 2 and 3 both claim to
locate the same physical timer button, so on a map covered by both they must
land in the same place. A disagreement means one source is wrong about that
map, and the map cannot be trusted until it is resolved. Reported per map as
centre distance, split into horizontal and vertical, plus whether the two
boxes overlap once padded.

**Is 64 the right pad for the in-BSP buttons?** The gateway work swept the
pad over ITS population (point entities carrying a small local AABB) and
found 9 of 171 finishes with no standable point unpadded and 0 from pad 32
up. The in-BSP buttons are a different population - brush entities, whose
AABB is the real world-space brush - so the sweep has to be redone on them.
"Standable" is hull-1 of the WORLD model: a point that comes back
CONTENTS_SOLID is a place the player's origin cannot be, so a box with no
free sample is a finish the agent can never occupy.

Two padding hazards the sweep also has to rule out, both created BY the pad
rather than found by it: a padded finish that swallows a spawn (the episode
completes on tick 1) and a padded finish that overlaps the padded start.

    python tools/zone_audit.py --maps maps_full_dataset \
        --gateway runs/research/gateway_buttons.json \
        --json runs/research/zone_audit.json --jobs 6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.zones import (BUTTON_PAD, detect_zones, hull_probe,  # noqa: E402
                           parse_bsp)

PADS = [0.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 320.0]
GRID = 9                       # samples per axis inside a padded box


def _vec3(s):
    if not s:
        return None
    try:
        v = [float(x) for x in s.split()[:3]]
    except ValueError:
        return None
    return v if len(v) == 3 else None


def _box(z):
    return np.asarray(z["mins"], float), np.asarray(z["maxs"], float)


def _true_box(z):
    """The honest, unpadded box: `true_aabb` when the source padded one."""
    t = z.get("true_aabb")
    return _box(t) if t else _box(z)


def _centre(mn, mx):
    return (mn + mx) / 2.0


def _face_area(mn, mx):
    """Area of the largest face - the target the agent has to hit."""
    d = np.sort(mx - mn)
    return float(d[1] * d[2])


def _free_stats(solid, mn, mx, pad, n=GRID):
    """Fraction of a lattice inside the box+pad that a player origin could
    occupy (hull-1 of the world model says CONTENTS_SOLID or not)."""
    axes = [np.linspace(mn[i] - pad, mx[i] + pad, n) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    s = solid(0, pts)
    return float((~s).mean())


def _overlap(a, b):
    (amn, amx), (bmn, bmx) = a, b
    return bool(np.all(amn <= bmx) and np.all(bmn <= amx))


def audit_one(name, maps_dir, gateway):
    bsp = Path(maps_dir) / f"{name}.bsp"
    r = {"map": name}
    try:
        ents, boxes = parse_bsp(bsp)
        z = detect_zones(bsp)
        solid = hull_probe(bsp)
    except Exception as ex:
        r["error"] = f"{type(ex).__name__}: {ex}"
        return r

    spawns = [v for v in (_vec3(e.get("origin")) for e in ents
                          if e.get("classname") in ("info_player_start",
                                                    "info_player_deathmatch"))
              if v]
    r["n_spawns"] = len(spawns)
    # how many entities could have been picked for each role. detect_zones
    # takes the FIRST match, so >1 is an arbitrary choice worth knowing about.
    cand = {"start": 0, "end": 0}
    for e in ents:
        if e.get("classname") != "func_button":
            continue
        t = (e.get("target") or "").lower()
        if t == "counter_start":
            cand["start"] += 1
        elif t in ("counter_off", "counter_stop"):
            cand["end"] += 1
    r["button_candidates"] = cand
    gb = (gateway.get(name, {}) or {}).get("buttons") or {}
    r["gateway_status"] = (gateway.get(name, {}) or {}).get("status")
    r["gateway_suspect"] = (gateway.get(name, {}) or {}).get("suspect")

    padded = {}
    for role, gkey in (("start", "start"), ("end", "stop")):
        d = {}
        zb = z.get(role)
        if zb is not None:
            kind = "button" if zb.get("from") == "func_button" else "trigger"
            tmn, tmx = _true_box(zb)
            pmn, pmx = _box(zb)
            padded[role] = (pmn, pmx)
            d["bsp_kind"] = kind
            d["bsp_true_mins"] = [round(float(v), 1) for v in tmn]
            d["bsp_true_maxs"] = [round(float(v), 1) for v in tmx]
            d["bsp_centre"] = [round(float(v), 1) for v in _centre(tmn, tmx)]
            d["bsp_dims"] = [round(float(v), 1) for v in (tmx - tmn)]
            d["bsp_face_area"] = round(_face_area(tmn, tmx), 1)
            d["bsp_volume"] = round(float(np.prod(np.maximum(tmx - tmn, 1e-6))), 1)
            # the pad sweep, on the source's own honest box
            d["free_by_pad"] = {str(int(p)): round(_free_stats(solid, tmn, tmx, p), 4)
                                for p in PADS}
            d["centre_solid"] = bool(solid(0, np.array([_centre(tmn, tmx)]))[0])
            if spawns:
                sp = np.asarray(spawns, float)
                q = np.clip(sp, pmn, pmx)
                d["spawn_in_padded_box"] = int(
                    np.sum(np.linalg.norm(sp - q, axis=1) < 1e-6))
                d["d_spawn_true_box"] = round(float(np.linalg.norm(
                    sp - np.clip(sp, tmn, tmx), axis=1).min()), 1)
        g = gb.get(gkey)
        if g:
            gmn, gmx = (np.asarray(g["true_aabb"]["mins"], float),
                        np.asarray(g["true_aabb"]["maxs"], float))
            d["gw_centre"] = [round(float(v), 1) for v in _centre(gmn, gmx)]
            d["gw_dims"] = [round(float(v), 1) for v in (gmx - gmn)]
            d["gw_face_area"] = round(_face_area(gmn, gmx), 1)
            d["gw_free_by_pad"] = {str(int(p)): round(_free_stats(solid, gmn, gmx, p), 4)
                                   for p in PADS}
            if spawns:
                sp = np.asarray(spawns, float)
                d["gw_d_spawn_box"] = round(float(np.linalg.norm(
                    sp - np.clip(sp, gmn, gmx), axis=1).min()), 1)
            if zb is not None:
                c1, c2 = _centre(*_true_box(zb)), _centre(gmn, gmx)
                d["agree_d"] = round(float(np.linalg.norm(c1 - c2)), 1)
                d["agree_dxy"] = round(float(np.linalg.norm(c1[:2] - c2[:2])), 1)
                d["agree_dz"] = round(float(abs(c1[2] - c2[2])), 1)
                bmn, bmx = _true_box(zb)
                d["agree_dbox"] = round(float(np.linalg.norm(
                    c2 - np.clip(c2, bmn, bmx))), 1)
                d["agree_padded_overlap"] = _overlap(
                    (_true_box(zb)[0] - BUTTON_PAD, _true_box(zb)[1] + BUTTON_PAD),
                    (gmn - BUTTON_PAD, gmx + BUTTON_PAD))
        if d:
            r[role] = d

    if "start" in padded and "end" in padded:
        r["padded_start_end_overlap"] = _overlap(padded["start"], padded["end"])
    return r


def _worker(args):
    return audit_one(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", default="maps_full_dataset")
    ap.add_argument("--gateway", default="runs/research/gateway_buttons.json")
    ap.add_argument("--json", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    gw = json.loads(Path(a.gateway).read_text(encoding="utf-8"))["maps"]
    names = sorted(p.stem for p in Path(a.maps).glob("*.bsp"))
    if a.limit:
        names = names[:a.limit]
    print(f"auditing {len(names)} maps, jobs {a.jobs}")

    tasks = [(n, a.maps, gw) for n in names]
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            rows = []
            for i, r in enumerate(pool.imap(_worker, tasks, chunksize=4), 1):
                rows.append(r)
                if i % 50 == 0:
                    print(f"  {i}/{len(names)}", flush=True)
    else:
        rows = [_worker(t) for t in tasks]

    out = {"pads": PADS, "grid": GRID, "button_pad": BUTTON_PAD, "maps": rows}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
