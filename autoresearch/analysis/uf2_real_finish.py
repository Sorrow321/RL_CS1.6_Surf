#!/usr/bin/env python3
"""uf2_real_finish.py - MEASUREMENT ONLY: does an eval episode that the core labelled "done" really
touch the finish trigger the way GoldSrc fires it?

The core's goal test (src/env.c) is the finish trigger's brush AABB, inflated by the player hull.
The engine (SV_TouchLinks) uses the AABB only as a precheck and fires a brush trigger when the
player's origin is inside the model's CLIP HULL (hull 1 standing, hull 3 ducked) - the same test
src/trace.c trigger_contains() runs for teleports and hurts. On surf_unitfarmer2 the finish (*23,
counter_stop_button) is three thin curtains along the box's west / diagonal / north sides, so the
box's empty north-west corner is a finish to the core and nothing to the game (found 2026-10-31).

This re-reads the BSP's own clip hull (models / clipnodes / planes lumps) in python, the
trigger_contains rule (origin - (clip_mins - player_mins) inside the hull; both offsets are 0 for
hulls 1 and 3), and checks every row of each episode against hull 1 AND hull 3 (a row counts if
either contains it - the episode rows do not say whether the player was ducked).

    python autoresearch/analysis/uf2_real_finish.py runs/<agent>/<run>/traj_*.jsonl
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "autoresearch"))
import score as S  # noqa: E402

LUMP_PLANES, LUMP_CLIPNODES, LUMP_MODELS = 1, 9, 14
CONTENTS_SOLID = -2


def load_hulls(bsp_path: Path):
    data = bsp_path.read_bytes()
    version = struct.unpack_from("<i", data, 0)[0]
    lumps = [struct.unpack_from("<ii", data, 4 + 8 * i) for i in range(15)]
    off, ln = lumps[LUMP_PLANES]
    planes = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("d", "<f4"), ("t", "<i4")]),
                           count=ln // 20, offset=off)
    off, ln = lumps[LUMP_CLIPNODES]
    clip = np.frombuffer(data, dtype=np.dtype([("p", "<i4"), ("c", "<i2", 2)]), count=ln // 8,
                         offset=off)
    off, ln = lumps[LUMP_MODELS]
    models = np.frombuffer(data, dtype=np.dtype([("mins", "<f4", 3), ("maxs", "<f4", 3),
                                                 ("org", "<f4", 3), ("head", "<i4", 4),
                                                 ("vis", "<i4"), ("ff", "<i4"), ("nf", "<i4")]),
                           count=ln // 64, offset=off)
    return version, planes, clip, models


def hull_contents(planes, clip, num, p):
    while num >= 0:
        node = clip[num]
        pl = planes[node["p"]]
        t = int(pl["t"])
        d = (p[t] if t < 3 else float(np.dot(pl["n"], p))) - float(pl["d"])
        num = int(node["c"][0] if d >= 0 else node["c"][1])
    return num


def finish_model(bsp_path: Path) -> int:
    from surfgym.zones import parse_bsp
    ents, _ = parse_bsp(str(bsp_path))
    for e in ents:
        if e.get("classname") == "trigger_multiple" and "stop" in e.get("target", "") \
                and e.get("model", "").startswith("*"):
            return int(e["model"][1:])
    raise SystemExit("no trigger_multiple targeting a stop button")


def main():
    bsp = REPO / S.CFG["map"]
    if not bsp.exists():
        bsp = Path("/home/steam/RL_Surf_uf2") / S.CFG["map"]
    version, planes, clip, models = load_hulls(bsp)
    mi = finish_model(bsp)
    m = models[mi]
    print(f"{bsp.name}: v{version}, finish model *{mi} box {m['mins'].tolist()}..{m['maxs'].tolist()}")
    for path in sys.argv[1:]:
        eps = S.first_episodes(S.episodes(Path(path)), S.SCORE_EPISODES)
        core_done = real = 0
        times = []
        for h, r, tr in eps:
            if tr.get("end") != "done":
                continue
            core_done += 1
            p = r[:, 1:4]
            hit = -1
            for i in range(max(0, len(p) - 400), len(p)):
                if any(hull_contents(planes, clip, int(m["head"][hi]), p[i]) == CONTENTS_SOLID
                       for hi in (1, 3)):
                    hit = i
                    break
            if hit >= 0:
                real += 1
                times.append(hit * float(h.get("tick_ms", 10.0)) / 1000.0)
        print(f"{Path(path).name}: core finishes {core_done}/{len(eps)}, REAL (clip hull) {real}"
              + (f", real times {sorted(round(t, 2) for t in times)}" if times else ""))


if __name__ == "__main__":
    main()
