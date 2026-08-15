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
        return {"mins": mins, "maxs": maxs}

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
