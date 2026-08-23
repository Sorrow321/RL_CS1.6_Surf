"""zones.py — race start/finish zones for linear surf maps.

Timed surf maps carry their own labeling: a thin ``trigger_multiple`` brush
at the start line fires the timer's start button, another at the finish fires
the stop button (surf_src_cannonball wires ``counter_start_button`` /
``counter_stop_button`` exactly this way). This module extracts those brush
AABBs straight from the .bsp — no manual labeling needed when the pattern is
present — and persists them to an *editable* ``maps/<map>.zones.json``:

    {"map": "...", "source": "auto",
     "start": {"mins": [x,y,z], "maxs": [x,y,z]},
     "end":   {"mins": [x,y,z], "maxs": [x,y,z]}}

Hand-labeling a map without a timer = writing that file yourself (set
``"source": "manual"``; auto-extraction never overwrites a manual file).
Zone brushes are often 1u thin — fine: the env's goal test sweeps the
per-tick segment, so thin curtains register at any speed.

Pure stdlib on purpose — usable from tools without loading the DLL.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

_LUMP_ENTITIES = 0
_LUMP_MODELS = 14


def parse_bsp(bsp_path):
    """Return (entities, model_bboxes) from a v30 BSP.

    entities: list of key->value dicts; model_bboxes: list of (mins, maxs)
    tuples, index == the ``*N`` brush-model reference in entity values."""
    data = Path(bsp_path).read_bytes()
    version, = struct.unpack_from("<i", data, 0)
    if version != 30:
        raise ValueError(f"{bsp_path}: BSP version {version}, expected 30")
    e_off, e_len = struct.unpack_from("<ii", data, 4 + 8 * _LUMP_ENTITIES)
    m_off, m_len = struct.unpack_from("<ii", data, 4 + 8 * _LUMP_MODELS)
    ents_text = data[e_off:e_off + e_len].rstrip(b"\x00").decode("latin-1")
    entities = [dict(re.findall(r'"([^"]+)"\s+"([^"]*)"', block))
                for block in re.findall(r"\{([^}]*)\}", ents_text)]
    bboxes = []
    for i in range(m_len // 64):
        v = struct.unpack_from("<9f", data, m_off + 64 * i)
        bboxes.append((list(v[0:3]), list(v[3:6])))
    return entities, bboxes


def _model_index(ent):
    m = ent.get("model", "")
    return int(m[1:]) if m.startswith("*") else None


def _kill_entities(ents, models):
    """Pure selection rule, split out for unit testing.

    Under race rules ANY teleport touch fails the episode (teleport_fail),
    so every trigger_teleport WITH a live destination is a kill volume —
    destless pads are inert (GoldSrc TeleportTouch no-ops; mirrored in
    src/env.c). trigger_hurt kills only from dmg >= 90 — that is the
    simulator's own rule (src/env.c trigger test); below 90 the sim models
    no damage at all, so such a brush must NOT become a wall."""
    dests = {e.get("targetname") for e in ents
             if e.get("targetname") and _model_index(e) is None}
    out = []
    for e in ents:
        mi = _model_index(e)
        if mi is None or mi >= len(models):
            continue
        cls = e.get("classname")
        try:
            dmg = float(e.get("dmg", 0.0))
        except (TypeError, ValueError):
            dmg = 0.0                     # kv_float parity: garbage -> 0
        deadly = ((cls == "trigger_teleport" and e.get("target") in dests)
                  or (cls == "trigger_hurt" and dmg >= 90.0))
        if deadly:
            mins, maxs = models[mi]
            org = [0.0, 0.0, 0.0]
            if e.get("origin"):
                org = [float(v) for v in e["origin"].split()[:3]]
            out.append({"mins": list(mins), "maxs": list(maxs),
                        "class": cls, "model": e.get("model"),
                        "origin": org})
    return out


def kill_zones(bsp_path):
    """Kill volumes for the kill-aware goal graph (docs/research-plan.md
    arm S2): the plain geodesic paints a descending gradient THROUGH fail
    nets, pulling agents into the very volume that kills them.

    Entries carry the model AABB (broadphase only!) and the ``*N`` model
    reference — the sim kills by clipnode-hull containment, not by AABB
    (triggers are often thin curtains inside huge boxes), so callers must
    classify voxels with :func:`hull_probe`, never by the box alone."""
    ents, models = parse_bsp(bsp_path)
    return _kill_entities(ents, models)


_LUMP_PLANES = 1
_LUMP_CLIPNODES = 9


def hull_probe(bsp_path):
    """Return ``contains(model_index, points) -> bool array``: point-in-
    trigger via the model's HULL-1 clipnode tree — the same 32u-standing-
    player-inflated hull src/trace.c walks for the sim's trigger test, so
    the mask matches what actually kills."""
    import numpy as np
    data = Path(bsp_path).read_bytes()
    p_off, p_len = struct.unpack_from("<ii", data, 4 + 8 * _LUMP_PLANES)
    c_off, c_len = struct.unpack_from("<ii", data, 4 + 8 * _LUMP_CLIPNODES)
    m_off, m_len = struct.unpack_from("<ii", data, 4 + 8 * _LUMP_MODELS)
    n_planes = p_len // 20
    pl = np.frombuffer(data, "<f4", count=n_planes * 5,
                       offset=p_off).reshape(n_planes, 5)  # nx ny nz dist type
    normals, dists = pl[:, :3].copy(), pl[:, 3].copy()
    n_clip = c_len // 8
    raw = np.frombuffer(data, "<i4", count=n_clip * 2,
                        offset=c_off).reshape(n_clip, 2)
    planenum = raw[:, 0].copy()
    children = np.frombuffer(data, "<i2", count=n_clip * 4,
                             offset=c_off).reshape(n_clip, 4)[:, 2:4].astype(
                                 np.int32)                 # child0, child1
    headnodes = {}
    for i in range(m_len // 64):
        hn = struct.unpack_from("<4i", data, m_off + 64 * i + 36)
        headnodes[i] = hn[1]                               # hull 1

    def contains(model_index, points):
        pts = np.atleast_2d(np.asarray(points, np.float64))
        node = np.full(len(pts), headnodes[int(model_index)], np.int32)
        active = node >= 0
        while active.any():
            n = node[active]
            p = planenum[n]
            d = (pts[active] * normals[p]).sum(1) - dists[p]
            node[active] = np.where(d >= 0, children[n, 0], children[n, 1])
            active = node >= 0
        return node == -2                                  # CONTENTS_SOLID

    return contains


def detect_zones(bsp_path):
    """Auto-detect race start/end zones. Returns {"start": box|None,
    "end": box|None} where box = {"mins": [...], "maxs": [...]}.

    Heuristic (the standard timed-map wiring): a ``func_button`` whose name
    marks it as the timer's start/stop control, touched via a
    ``trigger_multiple`` brush — that brush's AABB is the zone. Falls back to
    name keywords on the trigger target itself."""
    entities, bboxes = parse_bsp(bsp_path)

    def box_of(ent):
        idx = _model_index(ent)
        if idx is None or idx >= len(bboxes):
            return None
        mins, maxs = bboxes[idx]
        # A brush entity built around an ORIGIN BRUSH stores its model
        # vertices relative to that origin and carries the world offset in
        # the "origin" key — world_point = model_point + origin, which is
        # the convention goalfield's kill-volume masking already assumes
        # (`hull_probe.contains(mi, pts - origin)`). Without this the zone
        # AABB lands near the world origin instead of on the map: measured
        # over the 620-map dataset it silently misplaces the finish on 5 of
        # the 47 shortlisted maps (surf_bomzis, surf_sg_china, surf_sg_dash,
        # surf_sg_oldtemple, surf_src_whoknows2_b1) and shifts sidistic's
        # finish 512u. No map in maps/ except sidistic has such a trigger,
        # so every trained checkpoint's zone boxes are unchanged.
        org = [0.0, 0.0, 0.0]
        if ent.get("origin"):
            try:
                v = [float(x) for x in ent["origin"].split()[:3]]
                if len(v) == 3:
                    org = v
            except ValueError:
                pass
        return {"mins": [mins[i] + org[i] for i in range(3)],
                "maxs": [maxs[i] + org[i] for i in range(3)]}

    buttons = {e.get("targetname", ""): e for e in entities
               if e.get("classname") == "func_button" and e.get("targetname")}

    def button_role(name):
        low = name.lower()
        tgt = buttons[name].get("target", "").lower()
        if any(k in low or k in tgt for k in ("stop", "off", "end", "finish")):
            return "end"
        if "start" in low or "start" in tgt:
            return "start"
        return None

    zones = {"start": None, "end": None}
    for ent in entities:
        if ent.get("classname") != "trigger_multiple":
            continue
        target = ent.get("target", "")
        role = button_role(target) if target in buttons else None
        if role is None:
            low = target.lower()
            if any(k in low for k in ("mapend", "map_end", "finishzone", "endzone")):
                role = "end"
        if role and zones[role] is None:
            zones[role] = box_of(ent)
    return zones


def zones_path(bsp_path) -> Path:
    p = Path(bsp_path)
    return p.parent / f"{p.stem}.zones.json"


def _bsp_sig(bsp_path) -> str:
    st = Path(bsp_path).stat()
    return f"{st.st_size}_{st.st_mtime_ns}"


def load_zones(bsp_path, create: bool = True):
    """Load ``maps/<map>.zones.json``, auto-extracting it from the BSP when
    missing or stale. Hand-labeled files (``"source": "manual"``) are always
    trusted; auto files carry the BSP's size+mtime signature and re-extract
    when the map changes (a recompiled finish must never race against a
    stale zone box). Failed detections are returned but never persisted, so
    a map that later gains timer triggers is retried automatically."""
    zp = zones_path(bsp_path)
    sig = _bsp_sig(bsp_path)
    if zp.exists():
        doc = json.loads(zp.read_text(encoding="utf-8"))
        if doc.get("source") == "manual" or doc.get("bsp_sig") == sig:
            return doc
    zones = detect_zones(bsp_path)
    doc = {"map": Path(bsp_path).stem, "source": "auto", "bsp_sig": sig,
           **zones}
    if create and (zones["start"] or zones["end"]):
        zp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def main():
    import argparse
    ap = argparse.ArgumentParser(description="extract race zones from a BSP")
    ap.add_argument("bsp")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if a zones.json exists "
                         "(refuses to clobber source: manual)")
    args = ap.parse_args()
    zp = zones_path(args.bsp)
    if zp.exists() and args.force:
        cur = json.loads(zp.read_text(encoding="utf-8"))
        if cur.get("source") == "manual":
            raise SystemExit(f"{zp} is hand-labeled (source: manual) — not "
                             "overwriting; delete it yourself if you mean it")
        zp.unlink()
    doc = load_zones(args.bsp)
    print(json.dumps(doc, indent=2))
    print(f"-> {zp}")


if __name__ == "__main__":
    main()
