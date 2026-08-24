#!/usr/bin/env python3
"""restamp_maps.py - make the shipped cache files match their BSPs again.

Every cache in this project (goal field, SDF, occupancy, slab occupancy)
keys itself on ``_map_sig`` = ``v2_<size>_<mtime_ns>`` of the .bsp. That is
a fine cache key on one machine and a TRAP the moment the pool is shipped:

    baked on the bake box   v2_3330396_1761347437279944800
    after tar + download    v2_3330396_1761347437000000000
                                                ^^^^^^^^^ gone

**tar does not preserve sub-second mtimes.** So the signature misses, and
546 MB of prebaked fields is silently ignored - on 102 of the 108 pool maps
measured. The trainer then rebakes at startup, minutes to hours per map, on
a rented box. Nothing warns you: a cache miss looks exactly like a cold
start. (The 6 that survived just happened to have whole-second mtimes.)

The mtime the cache WANTS is written inside the cache itself, so nothing
has to be rebaked - the BSP is restamped to the value its fields were baked
against. Size is checked first, so this can only ever re-date a file that
is byte-for-byte the one that was baked.

    python3 tools/restamp_maps.py            # fix maps/
    python3 tools/restamp_maps.py --check    # report only, exit 1 if stale

Run it after any transfer of the pool. fetch_pool.sh does it for you.
"""
import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIG_RE = re.compile(r"v\d+_(\d+)_(\d+)")


def stored_sig(npz):
    """(size, mtime_ns) the cache was baked against, or None."""
    try:
        import numpy as np
        z = np.load(npz, allow_pickle=False)
        if "sig" not in z:
            return None
        m = SIG_RE.search(str(z["sig"]))
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default=str(ROOT / "maps"))
    ap.add_argument("--check", action="store_true", help="report only")
    a = ap.parse_args()
    maps = Path(a.maps)

    tally = Counter()
    fixed, stale, conflict = [], [], []
    for bsp in sorted(maps.glob("*.bsp")):
        caches = sorted(maps.glob(f"{bsp.stem}.goal*_*.npz")) + \
            sorted(maps.glob(f"{bsp.stem}.sdf_*.npz")) + \
            sorted(maps.glob(f"{bsp.stem}.occ_*.npz")) + \
            sorted(maps.glob(f"{bsp.stem}.slabocc_*.npz"))
        if not caches:
            tally["no cache"] += 1
            continue
        st = bsp.stat()
        want = {s for s in (stored_sig(c) for c in caches) if s}
        if not want:
            tally["cache has no sig"] += 1
            continue
        # only ever restamp to a signature whose SIZE is this exact file
        want = {(sz, mt) for sz, mt in want if sz == st.st_size}
        if not want:
            tally["size differs - genuinely a different bsp"] += 1
            continue
        if len({mt for _, mt in want}) > 1:
            conflict.append(bsp.stem)
            tally["caches disagree - left alone"] += 1
            continue
        mt = next(iter(want))[1]
        if mt == st.st_mtime_ns:
            tally["already matching"] += 1
            continue
        stale.append(bsp.stem)
        if not a.check:
            os.utime(bsp, ns=(st.st_atime_ns, mt))
            fixed.append(bsp.stem)
        tally["restamped" if not a.check else "STALE"] += 1

    for k, v in tally.most_common():
        print(f"  {v:4d}  {k}")
    if conflict:
        print(f"\n!! {len(conflict)} maps have caches baked against different "
              f"mtimes; delete the odd one out and rebake: "
              f"{', '.join(conflict[:5])}")
    if a.check and stale:
        print(f"\n!! {len(stale)} maps would rebake at startup. "
              f"Run: python3 tools/restamp_maps.py")
        return 1
    if fixed:
        print(f"\n{len(fixed)} maps restamped - their prebaked fields, SDFs "
              f"and occupancy grids are live again (no rebake)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
