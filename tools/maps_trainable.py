#!/usr/bin/env python3
"""maps_trainable.py - the final trainable set, and what each map's finish IS.

Folds `verify_maps.py`'s per-map verdicts, `survey_maps.py`'s class and
`zone_audit.py`'s zone provenance into one file for the multi-map run to
consume, and prints the funnel and the failure table.

The provenance is not decoration. A **type-1** finish is a `trigger_multiple`
volume with a median largest face of ~799,000 u^2 - the agent flies through
it. A **type-3** finish is a `func_button` brush AABB, median ~2,000 u^2
unpadded and ~31,000 u^2 once padded by the 64 u `+use` reach: still **26x
smaller** than a real trigger zone, and the simulator arrives in it rather
than pressing it. So a null on a button-finish map is much weaker evidence
than a null on a trigger-finish map, and the two must never be pooled into
one aggregate without saying which is which. Each row carries
`finish_kind` and `evidence` for exactly that reason.

    python tools/maps_trainable.py --shards runs/research/verify_corpus \
        --survey runs/research/map_survey.json \
        --audit runs/research/zone_audit.json \
        --json runs/research/maps_trainable.json
"""
import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TRAINABLE = ("pass",)


def load_shards(*dirs):
    rows = {}
    for d in dirs:
        if not d:
            continue
        for f in glob.glob(str(Path(d) / "*.json")):
            for m in json.loads(Path(f).read_text(encoding="utf-8"))["maps"]:
                rows[m["map"]] = m
    return rows


def family(name):
    """`surf_x_b3` and `surf_x` are the same map re-released; group them."""
    return re.sub(r"(_b\d+|_v\d+|_final|_fix(ed)?|_ez|_s|_sc|_nosc|_h)$", "",
                  name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="*", default=["runs/research/verify_corpus"])
    ap.add_argument("--survey", default="runs/research/map_survey.json")
    ap.add_argument("--audit", default="runs/research/zone_audit.json")
    ap.add_argument("--maps", default="maps_full_dataset")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    ver = load_shards(*a.shards)
    survey = {m["map"]: m for m in
              json.loads(Path(a.survey).read_text(encoding="utf-8"))["maps"]}
    audit = {m["map"]: m for m in
             json.loads(Path(a.audit).read_text(encoding="utf-8"))["maps"]}

    out, fails = [], []
    for name in sorted(ver):
        r = ver[name]
        az = audit.get(name, {})
        end = az.get("end", {})
        kind = end.get("bsp_kind")
        if kind is None and "gw_centre" in end:
            kind = "gateway"
        row = {
            "map": name,
            "bsp": str(Path(a.maps) / f"{name}.bsp"),
            "verdict": r.get("verdict"),
            "survey_class": survey.get(name, {}).get("class"),
            "finish_kind": kind,
            "evidence": ("strong" if kind == "trigger" else
                         "weak" if kind in ("button", "gateway") else None),
            "finish_face_area": end.get("bsp_face_area", end.get("gw_face_area")),
            "finish_true_mins": end.get("bsp_true_mins"),
            "finish_true_maxs": end.get("bsp_true_maxs"),
            "cell": r.get("cell"),
            "extent": r.get("extent"),
            "n_spawns": r.get("n_spawns"),
            "spawns_reachable": r.get("spawns_reachable"),
            "d0_euclid_mean": r.get("d0_euclid_mean"),
            "n_free": r.get("n_free"),
            "n_components": r.get("n_components"),
            "free_volume_e9": r.get("free_volume_e9"),
            "warnings": r.get("warnings", []),
            "family": family(name),
        }
        if r.get("verdict") in TRAINABLE:
            out.append(row)
        else:
            row["checks_failed"] = r.get("checks_failed")
            row["permissive_reachable"] = r.get("permissive_reachable")
            row["gap_units"] = r.get("gap_units")
            row["tp_into_finish_from_spawn_component"] = r.get(
                "tp_into_finish_from_spawn_component")
            fails.append(row)

    print(f"=== funnel ===\n  verified {len(ver)}  ->  trainable {len(out)}  "
          f"(not trainable {len(fails)})")
    print(f"  distinct families among the trainable: "
          f"{len({r['family'] for r in out})}")

    print("\n=== trainable by finish kind ===")
    for k, v in Counter(r["finish_kind"] for r in out).most_common():
        print(f"  {str(k):10s} {v:4d}   evidence "
              f"{'strong' if k == 'trigger' else 'weak'}")

    print("\n=== trainable by survey class ===")
    for k, v in Counter(r["survey_class"] for r in out).most_common():
        print(f"  {str(k):18s} {v:4d}")

    print("\n=== failures, by check ===")
    grp = defaultdict(list)
    for r in fails:
        grp[(r["verdict"], "+".join(r.get("checks_failed") or ["?"]))].append(r["map"])
    for k, v in sorted(grp.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k[0]:16s} {k[1]:24s} {len(v):4d}")

    doc = {"trainable": len(out), "not_trainable": len(fails),
           "counts_by_finish_kind": dict(Counter(r["finish_kind"] for r in out)),
           "maps": out, "rejected": fails}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
