#!/usr/bin/env python3
"""fetch_gateway_buttons.py - timer button positions from the Surf Gateway
buttons service, for maps whose BSP carries no timer entities at all.

447 of the 620 maps in ``maps_full_dataset`` come out of
``tools/survey_maps.py`` as ``no_zones``: no ``trigger_multiple`` wired to a
named ``func_button``, so nothing in the BSP says where the race starts or
ends. On those maps the timer is not in the map - it is spawned at runtime by
a server plugin ("Surf Gateway Buttons" 1.1.3), which asks a central service
for the two button entities and creates them itself.

That service is a plain form POST::

    POST http://buttons.surfcs.net/
    Content-Type: application/x-www-form-urlencoded
    plugin=1.1.3&amxx=<ver>&port=<port>&map=<mapname>

and answers in plain text with zero or more ``[ ... ]`` blocks of
``key = value`` lines, optionally plus a ``{ message = ... }`` block::

    [
    classname = func_button
    targetname = surfgateway_start
    target = counter_start
    model = models/SurfGateway/starttimer.mdl
    solid = 0
    origin = -3132.403320 2881.999755 -150.580566
    angles = 0.000000 90.000000 0.000000
    mins = -10.000000 -3.000000 -10.000000
    maxs = 10.000000 3.000000 10.000000
    ]

The world AABB is ``origin + mins`` .. ``origin + maxs``. Those boxes are
TINY (about 20 x 6 x 20 units) and ``solid = 0``: they are +use targets, not
trigger volumes. This tool records the true unpadded AABB; turning it into a
trainable zone (and choosing the padding) is ``--zones``, see below.

This is somebody else's free service. The fetcher is therefore:

* **rate limited** - one request per second by default, strictly sequential,
  never parallel;
* **resumable** - every response is cached in ``--cache`` and a map already
  in the cache is never requested again (``--retry-errors`` re-asks only the
  ones that failed);
* **gentle on failure** - exactly one retry with backoff, then the failure is
  recorded and the pass moves on;
* **raw-preserving** - the verbatim response text is stored per map, so the
  parser can be changed and re-run offline with ``--parse-only``.

Usage::

    # single pass over every .bsp in the dataset (620 maps ~= 11 min)
    python tools/fetch_gateway_buttons.py maps_full_dataset \
        --cache runs/research/gateway_buttons.json

    # re-parse the cache in place after a parser change - no network
    python tools/fetch_gateway_buttons.py --parse-only \
        --cache runs/research/gateway_buttons.json

    # emit zone files for every map that has BOTH buttons
    python tools/fetch_gateway_buttons.py --parse-only \
        --cache runs/research/gateway_buttons.json \
        --zones runs/research/gateway_zones --pad 64
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SERVICE = "http://buttons.surfcs.net/"
PLUGIN_VERSION = "1.1.3"          # the plugin these responses are cut for
AMXX_VERSION = "1.8.2"            # the AMXX build the plugin reports
SERVER_PORT = "27015"             # default GoldSrc port

# The service is free and unpaid-for. One request per second, one pass.
DEFAULT_RATE = 1.0
RETRY_BACKOFF = 5.0               # exactly one retry, then give up on the map

_KV = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_BLOCK = re.compile(r"\[(.*?)\]", re.S)
_MESSAGE = re.compile(r"\{(.*?)\}", re.S)


# ---------------------------------------------------------------- parsing

def _vec3(s):
    """'1.0 2.0 3.0' -> [1.0, 2.0, 3.0]; anything else -> None."""
    try:
        v = [float(x) for x in str(s).split()]
    except ValueError:
        return None
    return v[:3] if len(v) >= 3 else None


def parse_block(text):
    """One ``[ ... ]`` body -> dict of key -> raw string value."""
    kv = {}
    for line in text.splitlines():
        m = _KV.match(line)
        if m:
            kv[m.group(1).lower()] = m.group(2)
    return kv


def button_role(kv):
    """'start' / 'stop' / None from targetname, falling back to target and
    the model name. The plugin names them surfgateway_start /
    surfgateway_stop, but the service is a free-text database and a couple of
    entries are hand-entered, so do not hard-code the exact string."""
    hay = " ".join((kv.get("targetname", ""), kv.get("target", ""),
                    kv.get("model", ""))).lower()
    # order matters: 'stoptimer' contains neither 'start' nor a lone 'top'
    if "stop" in hay or "_off" in hay or "counter_off" in hay:
        return "stop"
    if "start" in hay:
        return "start"
    return None


def parse_response(text):
    """Full response -> (buttons, messages, blocks, suspect).

    ``buttons`` maps role -> button record with the WORLD AABB computed as
    ``origin + mins`` .. ``origin + maxs``. A block with no usable origin or
    extent is kept in ``blocks`` but cannot become a button; a block
    REJECTED as bogus is listed in ``suspect`` with the reason."""
    blocks = [parse_block(b) for b in _BLOCK.findall(text or "")]
    messages = []
    for m in _MESSAGE.findall(text or ""):
        kv = parse_block(m)
        messages.append(kv.get("message", m.strip()) if kv else m.strip())

    buttons, suspect = {}, []
    for kv in blocks:
        role = button_role(kv)
        if role is None or role in buttons:
            continue                       # first block of a role wins
        org = _vec3(kv.get("origin"))
        mins = _vec3(kv.get("mins"))
        maxs = _vec3(kv.get("maxs"))
        if org is None:
            continue
        if org == [0.0, 0.0, 0.0]:
            # the service's "never filled in" sentinel, not a position.
            # Three of 343 blocks: surf_green_pot's stop (solid=2, and the
            # map's geometry is nowhere near the world origin),
            # surf_sg_speedway's start (that map HAS a real BSP start
            # curtain, 3,142 u away) and surf_meow_brokengame's start
            # (5,130 u from every spawn). Kept in `raw`, excluded from
            # `buttons` - a zone built on it would be a phantom finish, the
            # same defect the origin-brush bug caused in detect_zones.
            suspect.append({"role": role, "why": "origin 0 0 0",
                            "targetname": kv.get("targetname")})
            continue
        if mins is None or maxs is None:
            # a point button with no extent still pins the position; give it
            # a zero box and let the consumer's padding do the work
            mins = mins or [0.0, 0.0, 0.0]
            maxs = maxs or [0.0, 0.0, 0.0]
        lo = [min(mins[i], maxs[i]) for i in range(3)]
        hi = [max(mins[i], maxs[i]) for i in range(3)]
        buttons[role] = {
            "classname": kv.get("classname"),
            "targetname": kv.get("targetname"),
            "target": kv.get("target"),
            "model": kv.get("model"),
            "solid": kv.get("solid"),
            "origin": org,
            "angles": _vec3(kv.get("angles")),
            "local_mins": lo,
            "local_maxs": hi,
            # the ground truth, unpadded: what a human record is timed against
            "true_aabb": {"mins": [org[i] + lo[i] for i in range(3)],
                          "maxs": [org[i] + hi[i] for i in range(3)]},
        }
    return buttons, messages, blocks, suspect


def reparse_entry(entry):
    """Recompute the parsed fields of one cache entry from its raw text."""
    if entry.get("raw") is None:
        return entry
    buttons, messages, blocks, suspect = parse_response(entry["raw"])
    entry["buttons"] = buttons
    entry["messages"] = messages
    entry["n_blocks"] = len(blocks)
    entry["suspect"] = suspect
    if entry.get("http") == 200:
        entry["status"] = "ok" if buttons else "no_buttons"
    return entry


# ------------------------------------------------------------------ cache

def load_cache(path):
    p = Path(path)
    if p.exists():
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc.setdefault("maps", {})
        return doc
    return {"service": SERVICE,
            "params": {"plugin": PLUGIN_VERSION, "amxx": AMXX_VERSION,
                       "port": SERVER_PORT},
            "rate_limit_s": DEFAULT_RATE,
            "maps": {}}


def save_cache(path, doc):
    """Atomic-ish write: a Ctrl-C mid-save must not shred 620 responses."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------- fetching

def fetch_one(session, mapname, args):
    """One map -> cache entry. At most two HTTP attempts, ever."""
    body = {"plugin": args.plugin, "amxx": args.amxx, "port": args.port,
            "map": mapname}
    last = None
    for attempt in (0, 1):
        if attempt:
            time.sleep(RETRY_BACKOFF)
        try:
            r = session.post(args.url, data=body, timeout=args.timeout)
        except Exception as ex:                       # network / DNS / TLS
            last = f"{type(ex).__name__}: {ex}"
            continue
        if r.status_code >= 500:                      # transient server side
            last = f"HTTP {r.status_code}"
            continue
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "http": r.status_code, "raw": r.text}
        if r.status_code != 200:
            entry["status"] = "error"
            entry["error"] = f"HTTP {r.status_code}"
            return entry
        return reparse_entry(entry)
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "http": None, "raw": None, "status": "error", "error": last}


def map_names(args):
    if args.maps_list:
        return [l.strip() for l in Path(args.maps_list).read_text(
            encoding="utf-8").splitlines() if l.strip()]
    if args.folder:
        return sorted(p.stem for p in Path(args.folder).glob("*.bsp"))
    return []


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=None,
                    help="folder of .bsp files; each stem is one map name")
    ap.add_argument("--maps-list", default=None,
                    help="file with one map name per line (instead of folder)")
    ap.add_argument("--cache", default=str(ROOT / "runs" / "research" /
                                           "gateway_buttons.json"))
    ap.add_argument("--url", default=SERVICE)
    ap.add_argument("--plugin", default=PLUGIN_VERSION)
    ap.add_argument("--amxx", default=AMXX_VERSION)
    ap.add_argument("--port", default=SERVER_PORT)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE,
                    help="minimum seconds between request STARTS (>= 1.0; "
                         "this is somebody else's free service)")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N new requests (0 = no limit)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-ask maps whose cached status is 'error'")
    ap.add_argument("--parse-only", action="store_true",
                    help="re-parse cached raw text; makes no requests at all")
    ap.add_argument("--zones", default=None,
                    help="write <dir>/<map>.zones.json for every map with "
                         "BOTH buttons")
    ap.add_argument("--pad", type=float, default=64.0,
                    help="units the button AABB is inflated by in the zone "
                         "file (recorded in the file as 'pad'). 64 is "
                         "GoldSrc's PLAYER_SEARCH_RADIUS - the radius inside "
                         "which the engine itself lets a player +use an "
                         "entity, i.e. the honest 'could have pressed it' box")
    ap.add_argument("--report", action="store_true",
                    help="coverage / BSP-agreement / spawn-distance report")
    ap.add_argument("--survey", default=str(ROOT / "runs" / "research" /
                                            "map_survey.json"))
    ap.add_argument("--maps-dir", default=str(ROOT / "maps_full_dataset"))
    ap.add_argument("--report-json", default=str(ROOT / "runs" / "research" /
                                                 "gateway_report.json"))
    ap.add_argument("--tol", type=float, default=128.0,
                    help="a gateway button further than this from the BSP "
                         "zone AABB is reported as a disagreement")
    ap.add_argument("--show", type=int, default=15,
                    help="how many outliers to list per section")
    ap.add_argument("--min-race", type=float, default=700.0,
                    help="start-to-stop button distance below which the two "
                         "timer controls are in the same room and the map is "
                         "not a race")
    args = ap.parse_args()

    if args.rate < 1.0 and not args.parse_only:
        ap.error("--rate below 1.0 s would hammer a free third-party service")

    doc = load_cache(args.cache)
    cache = doc["maps"]

    # offline modes: --parse-only, or --report/--zones with no map source
    offline = args.parse_only or not (args.folder or args.maps_list)

    if args.parse_only:
        n = 0
        for name, entry in cache.items():
            before = json.dumps(entry.get("buttons"), sort_keys=True)
            reparse_entry(entry)
            if json.dumps(entry.get("buttons"), sort_keys=True) != before:
                n += 1
        save_cache(args.cache, doc)
        print(f"re-parsed {len(cache)} cached responses, {n} changed")
    elif offline:
        if not (args.report or args.zones):
            ap.error("give a folder of .bsp files or --maps-list")
    else:
        names = map_names(args)
        todo = [m for m in names
                if m not in cache
                or (args.retry_errors and cache[m].get("status") == "error")]
        print(f"{len(names)} maps, {len(cache)} cached, {len(todo)} to fetch "
              f"at {args.rate:.1f} s/request "
              f"(~{len(todo) * args.rate / 60:.0f} min)")
        if args.limit:
            todo = todo[:args.limit]

        import requests
        session = requests.Session()
        session.headers.update(
            {"Content-Type": "application/x-www-form-urlencoded"})
        doc["params"] = {"plugin": args.plugin, "amxx": args.amxx,
                         "port": args.port}
        doc["rate_limit_s"] = args.rate

        next_at = 0.0
        try:
            for i, name in enumerate(todo, 1):
                wait = next_at - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                next_at = time.monotonic() + args.rate
                entry = fetch_one(session, name, args)
                cache[name] = entry
                st = entry.get("status")
                roles = ",".join(sorted(entry.get("buttons", {}))) or "-"
                print(f"[{i:4d}/{len(todo)}] {name:44s} {st:10s} {roles}",
                      flush=True)
                if i % 25 == 0:
                    save_cache(args.cache, doc)
        except KeyboardInterrupt:
            print("\ninterrupted - saving what we have")
        finally:
            save_cache(args.cache, doc)

    counts = {}
    for e in cache.values():
        counts[e.get("status", "?")] = counts.get(e.get("status", "?"), 0) + 1
    both = sum(1 for e in cache.values()
               if {"start", "stop"} <= set(e.get("buttons", {})))
    print(f"\ncache: {len(cache)} maps  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())) +
          f"  both_buttons={both}")
    print(f"-> {args.cache}")

    if args.zones:
        n = write_zones(cache, args.zones, args.pad)
        print(f"wrote {n} zone files -> {args.zones}")
    if args.report:
        report(cache, args)
    return 0


# ----------------------------------------------------------------- report

def _center(box):
    return [(box["mins"][i] + box["maxs"][i]) / 2.0 for i in range(3)]


def _dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _dist_to_box(p, box, axes=(0, 1, 2)):
    """0 if p is inside the box on ``axes``; else distance to its surface.

    Split by axis on purpose. A BSP timer zone is usually a CURTAIN - the
    measured median thinnest axis over the 173 zoned maps is 8 u for the
    finish, 4 u for the start - and the gateway button is a plate bolted to
    a wall at chest height inside the same room. Comparing them in 3-D
    reports a "disagreement" that is really the offset between the floor
    plane the curtain lies in and the height of the button above it, so
    horizontal and vertical are reported separately."""
    d = 0.0
    for i in axes:
        e = max(box["mins"][i] - p[i], 0.0, p[i] - box["maxs"][i])
        d += e * e
    return d ** 0.5


def report(cache, args):
    """Coverage, BSP-vs-gateway agreement, and start-button-to-spawn
    distance. Reads BSPs only for the maps that need them."""
    sys.path.insert(0, str(ROOT / "python"))
    import numpy as np
    from surfgym.zones import parse_bsp, detect_zones, hull_probe

    surv = json.loads(Path(args.survey).read_text(encoding="utf-8"))
    rows = {r["map"]: r for r in surv["maps"]}
    maps_dir = Path(args.maps_dir)

    # -------- coverage by survey class
    order = ["ready", "zones_but_links", "no_spawn", "no_zones"]
    cov = {k: {"n": 0, "ok": 0, "no_buttons": 0, "error": 0,
               "start": 0, "stop": 0, "both": 0} for k in order}
    cov["_uncased"] = dict(cov["ready"])
    for name, r in rows.items():
        c = cov.get(r["class"], cov["_uncased"])
        c["n"] += 1
        e = cache.get(name)
        if e is None:
            continue
        c[e.get("status", "error")] = c.get(e.get("status", "error"), 0) + 1
        b = e.get("buttons") or {}
        if "start" in b:
            c["start"] += 1
        if "stop" in b:
            c["stop"] += 1
        if {"start", "stop"} <= set(b):
            c["both"] += 1

    print("\n=== coverage by survey class ===")
    print(f"{'class':17s} {'maps':>5s} {'ok':>5s} {'none':>5s} {'err':>4s} "
          f"{'start':>6s} {'stop':>6s} {'both':>6s}")
    for k in order:
        c = cov[k]
        print(f"{k:17s} {c['n']:5d} {c['ok']:5d} {c['no_buttons']:5d} "
              f"{c['error']:4d} {c['start']:6d} {c['stop']:6d} {c['both']:6d}")

    # -------- per-map geometry: agreement + spawn distance
    need = [m for m in sorted(rows)
            if (cache.get(m, {}).get("buttons")
                or (rows[m]["has_start"] and rows[m]["has_end"]))]
    details, agree_rows, spawn_rows = {}, [], []
    for m in need:
        bsp = maps_dir / f"{m}.bsp"
        if not bsp.exists():
            continue
        try:
            ents, boxes = parse_bsp(bsp)
            bz = detect_zones(bsp)
        except Exception as ex:
            details[m] = {"error": f"{type(ex).__name__}: {ex}"}
            continue
        world = {"mins": boxes[0][0], "maxs": boxes[0][1]} if boxes else None
        spawns = []
        for e in ents:
            if e.get("classname") in ("info_player_start",
                                      "info_player_deathmatch"):
                v = _vec3(e.get("origin"))
                if v:
                    spawns.append((e["classname"], v))
        gb = (cache.get(m, {}).get("buttons") or {})
        d = {"class": rows[m]["class"], "n_spawns": len(spawns)}
        # hull-1 of the WORLD model is the standing player's clip hull: a
        # point that comes back CONTENTS_SOLID is a place the player's origin
        # cannot be. So "is there anywhere inside the padded zone a player
        # could stand?" is the free-GPU trainability precheck, and it is also
        # the only independent check available on the 447 no_zones maps,
        # where there is no BSP zone to compare the button against.
        try:
            solid = hull_probe(bsp)
        except Exception:
            solid = None

        for role, zkey in (("start", "start"), ("stop", "end")):
            if role not in gb:
                continue
            c = _center(gb[role]["true_aabb"])
            d[f"{role}_center"] = [round(x, 1) for x in c]
            if world:
                d[f"{role}_in_world"] = all(
                    world["mins"][i] - 64 <= c[i] <= world["maxs"][i] + 64
                    for i in range(3))
            if solid is not None:
                t = gb[role]["true_aabb"]
                grid = np.stack(np.meshgrid(*[
                    np.linspace(t["mins"][i] - args.pad,
                                t["maxs"][i] + args.pad, 5)
                    for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
                s = solid(0, grid)
                d[f"{role}_center_solid"] = bool(solid(0, np.array([c]))[0])
                d[f"{role}_free_frac"] = round(float((~s).mean()), 3)
            zb = bz.get(zkey)
            if zb:
                d[f"{role}_bsp_dcenter"] = round(_dist(c, _center(zb)), 1)
                d[f"{role}_bsp_dbox"] = round(_dist_to_box(c, zb), 1)
                d[f"{role}_bsp_dxy"] = round(_dist_to_box(c, zb, (0, 1)), 1)
                d[f"{role}_bsp_dz"] = round(_dist_to_box(c, zb, (2,)), 1)
                agree_rows.append((m, role, d[f"{role}_bsp_dxy"],
                                   d[f"{role}_bsp_dz"],
                                   rows[m]["class"]))
        if "start" in gb and spawns:
            c = _center(gb["start"]["true_aabb"])
            ips = [v for cl, v in spawns if cl == "info_player_start"]
            pool = ips or [v for _, v in spawns]
            dmin = min(_dist(c, s) for s in pool)
            d["start_to_spawn"] = round(dmin, 1)
            d["spawn_kind"] = "info_player_start" if ips else "deathmatch"
            spawn_rows.append((m, dmin, len(pool), rows[m]["class"]))
        if "start" in gb and "stop" in gb:
            d["start_to_stop"] = round(
                _dist(_center(gb["start"]["true_aabb"]),
                      _center(gb["stop"]["true_aabb"])), 1)
        details[m] = d

    # -------- is the button somewhere a player could actually be?
    print(f"\n=== button placement sanity (world model hull 1, pad "
          f"{args.pad:.0f}) ===")
    for role in ("start", "stop"):
        got = [(m, d) for m, d in details.items()
               if f"{role}_free_frac" in d]
        if not got:
            continue
        oow = [m for m, d in got if d.get(f"{role}_in_world") is False]
        solidc = [m for m, d in got if d.get(f"{role}_center_solid")]
        nofree = [m for m, d in got if d[f"{role}_free_frac"] == 0.0]
        fr = sorted(d[f"{role}_free_frac"] for _, d in got)

        def q(p):
            return fr[int(p * (len(fr) - 1))]
        print(f"  {role:5s} n={len(got):3d}  outside world AABB {len(oow):3d}"
              f"  centre in solid {len(solidc):3d}  padded box entirely "
              f"solid {len(nofree):3d}")
        print(f"        free fraction of the padded box: p10 {q(.1):.2f}  "
              f"med {q(.5):.2f}  p90 {q(.9):.2f}")
        for m in (oow + solidc)[:args.show]:
            print(f"      {m:38s} {details[m].get(role + '_center')}")

    # -------- agreement
    print(f"\n=== BSP zone vs gateway button ({len(agree_rows)} "
          f"role-comparisons where both sources exist) ===")
    for role in ("start", "stop"):
        rs = [a for a in agree_rows if a[1] == role]
        if not rs:
            continue
        inside = sum(1 for a in rs if a[2] == 0.0)
        near = sum(1 for a in rs if 0.0 < a[2] <= args.tol)
        far = [a for a in rs if a[2] > args.tol]
        print(f"  {role:5s} n={len(rs):3d}  HORIZONTALLY inside the BSP zone "
              f"{inside:3d}  within {args.tol:.0f}u {near:3d}  "
              f"DISAGREE {len(far):3d}")
        for a in sorted(rs, key=lambda x: -x[2]):
            flag = "DISAGREE" if a[2] > args.tol else "ok"
            print(f"      {a[0]:38s} dxy {a[2]:8.1f}  dz {a[3]:8.1f}  "
                  f"{flag} ({a[4]})")

    # -------- start button vs spawn
    print(f"\n=== start button -> nearest spawn ({len(spawn_rows)} maps) ===")
    v = sorted(s[1] for s in spawn_rows)
    if v:
        def q(p):
            return v[int(p * (len(v) - 1))]
        print(f"  min {v[0]:8.1f}  p25 {q(.25):8.1f}  med {q(.5):8.1f}  "
              f"p75 {q(.75):8.1f}  p90 {q(.9):8.1f}  max {v[-1]:8.1f}")
        for cut in (64, 128, 256, 512, 1024, 4096):
            n = sum(1 for x in v if x <= cut)
            print(f"  <= {cut:5d} u: {n:4d} / {len(v)}  "
                  f"({100.0 * n / len(v):5.1f}%)")
    print(f"  farthest {args.show}:")
    for m, dm, ns, cl in sorted(spawn_rows, key=lambda x: -x[1])[:args.show]:
        print(f"      {m:38s} {dm:10.1f} u  ({ns} spawns, {cl})")

    # -------- is there a race between the two buttons at all?
    # Euclidean, so a floor on the route length, never an estimate of it. A
    # map whose start and stop buttons are a few hundred units apart has both
    # timer controls in the same room: it passes every geometric check and is
    # still not a race, so it must not be counted as a map this unlocked.
    st2 = sorted(((d["start_to_stop"], m) for m, d in details.items()
                  if "start_to_stop" in d))
    if st2:
        v = [x[0] for x in st2]

        def q(p):
            return v[int(p * (len(v) - 1))]
        print(f"\n=== start button -> stop button ({len(v)} maps, euclidean "
              f"= a FLOOR on race length) ===")
        print(f"  min {v[0]:8.0f}  p10 {q(.1):8.0f}  med {q(.5):8.0f}  "
              f"p90 {q(.9):8.0f}  max {v[-1]:8.0f}")
        short = [x for x in st2 if x[0] < args.min_race]
        print(f"  under {args.min_race:.0f} u - both timer buttons in one "
              f"room, NOT a race: {len(short)}")
        for dm, m in short:
            print(f"      {m:38s} {dm:10.0f} u")

    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"coverage": cov, "tol": args.tol,
                               "maps": details}, indent=1), encoding="utf-8")
    print(f"\n-> {out}")


# ------------------------------------------------------------------ zones

def write_zones(cache, out_dir, pad):
    """Emit ``<out>/<map>.zones.json`` in the schema ``surfgym.zones``
    reads, for every map that has BOTH buttons.

    ``source: gateway`` is deliberately NOT ``auto``: ``load_zones`` re-runs
    BSP detection whenever an auto file's map signature misses, and these
    maps have nothing in the BSP to detect, so an auto file would be thrown
    away on first load. Anything that is not ``auto`` is kept as-is.

    The emitted box is the true button AABB inflated by ``pad`` on every
    axis; both the padding and the unpadded ``true_aabb`` are recorded in the
    file so a timing comparison against a human record can undo it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, e in sorted(cache.items()):
        b = e.get("buttons") or {}
        if not {"start", "stop"} <= set(b):
            continue
        doc = {"map": name, "source": "gateway",
               "service": SERVICE, "pad": pad,
               "note": "func_button (+use target, solid=0) from the Surf "
                       "Gateway buttons service; 'start'/'end' are the true "
                       "button AABB inflated by 'pad' units per axis, "
                       "'true_aabb' is unpadded ground truth"}
        for role, key in (("start", "start"), ("stop", "end")):
            t = b[role]["true_aabb"]
            doc[key] = {
                "mins": [t["mins"][i] - pad for i in range(3)],
                "maxs": [t["maxs"][i] + pad for i in range(3)],
                "true_aabb": t,
                "origin": b[role]["origin"],
                "targetname": b[role].get("targetname"),
            }
        (out / f"{name}.zones.json").write_text(
            json.dumps(doc, indent=1), encoding="utf-8")
        n += 1
    return n


if __name__ == "__main__":
    raise SystemExit(main())
