#!/usr/bin/env python3
"""energy_splice.py - the loop's ENERGY stage applied to an archive that is already there (pbr7b2).

tools/auto_spine.py --energy-gain G runs Go-Explore before the policy line's largest energy losses
and hands back the line to the DEEPEST archive cell keeping >= G u more energy height
(z + |v|^2 / 2g) than the line at the same progress (auto_spine.energy_cell; energy_pick's filters:
on the route, not plunging, not slower than the line), spliced onto the line at its root, its views
along the velocity and its held keys translated with them. This tool does the same with an
exact-chains archive from a search run by hand (explore_phase1 --exact-chains rooted at the line's
own states), so the spine is the object the loop itself would have produced, without waiting for
the loop's quorum to fire its search. Nothing comes from a demo: the archive is a reward-free search
rooted at the policy's own line.

    python tools/energy_splice.py --line SPINE.npy --keys SPINE.keys.npy --archive A/archive.npz \\
        --field F.npz --map M.bsp --gain 150 --lo 72 --hi 74.5 --out OUT.npy

writes OUT.npy, OUT.keys.npy (when --keys is given) and OUT.json (the pick, the root row, the
boost of the root, the end's energy height: train_fast --demo-success-energy M asks h >= that - M).
Exit 2 when no cell keeps --gain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--line", required=True, help="the reference line (STATE_DTYPE .npy spine)")
    ap.add_argument("--keys", default=None, help="the line's held keys ((n, 3) .npy)")
    ap.add_argument("--archive", required=True, help="explore_phase1 --exact-chains archive.npz")
    ap.add_argument("--field", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--gain", type=float, required=True, help="u of energy height over the line")
    ap.add_argument("--lo", type=float, required=True, help="progress window, %% of the start's d")
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import auto_spine
    from spine_canon_view import canon_view
    from surfgym.core import STATE_DTYPE
    line = np.load(a.line)
    lkeys = np.asarray(np.load(a.keys), np.int32) if a.keys else None
    field = auto_spine.load_field(a.field)
    d = auto_spine.field_d(field, line)
    fin = np.isfinite(d)
    d0 = float(d[0]) if fin[0] else float(d[fin].max())
    z = np.load(a.archive, allow_pickle=False)
    c, n = auto_spine.energy_cell(z["state"], z["parent"], z["depth"], line, field, d0, a.lo,
                                  a.hi, a.gain)
    out = Path(a.out)
    info = {"line": a.line, "archive": a.archive, "gain_min": a.gain, "window": [a.lo, a.hi],
            "candidates": int(n), "pick": c}
    if c is None:
        print(f"no cell keeps {a.gain:g} u more than the line ({n} candidates)")
        out.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        return 2
    chain_path = out.with_name(out.stem + "_chain.npy")
    rc = auto_spine.run([sys.executable, "-u", ROOT / "tools" / "ge_replay.py", "--map", a.map,
                         "--archive", a.archive, "--field", a.field, "--out", chain_path,
                         "--cell", int(c["cell"])], out.with_name(out.stem + "_replay.log"), 1200)
    if rc != 0 or not chain_path.exists():
        raise SystemExit(f"ge_replay failed (exit {rc})")
    gl = np.load(chain_path)
    if gl.dtype != STATE_DTYPE or len(gl) < 2:
        raise SystemExit("ge_replay wrote no chain")
    # the chain starts at one of the roots, an exact copy of a line state (its velocity possibly
    # boosted): find it on the line, as auto_spine.go_explore does
    ro = np.asarray(gl[0]["origin"], np.float64)
    dist = np.linalg.norm(np.asarray(line["origin"], np.float64) - ro, axis=1)
    j = int(np.argmin(dist))
    if dist[j] > 1.0:
        raise SystemExit(f"the chain's root is {dist[j]:.1f} u from the line: not one of its states")
    vr = float(np.linalg.norm(np.asarray(gl[0]["velocity"], np.float64))
               / max(1e-6, float(np.linalg.norm(np.asarray(line[j]["velocity"], np.float64)))))
    chain, _m = canon_view(gl[1:])
    sp = auto_spine.clean(np.concatenate([np.asarray(line[:j + 1], STATE_DTYPE),
                                          np.asarray(chain, STATE_DTYPE)]))
    np.save(out, sp)
    eh = float(auto_spine.energy_height(gl[-1:])[0])
    info.update(root_row=j, root_err=float(dist[j]), root_boost=round(vr, 3),
                chain_len=int(len(gl)), n=int(len(sp)), energy_h=round(eh, 1),
                end_d=float(auto_spine.field_d(field, sp[-1:])[0]))
    if lkeys is not None:
        ck = auto_spine.chain_keys(a.archive, field, len(gl), cell=int(c["cell"]))
        if ck is None:
            raise SystemExit("the chain's link actions do not cover it: no keys")
        from spine_canon_keys import translate
        v = np.asarray(gl["velocity"][1:], np.float64)
        hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        fast = np.hypot(v[:, 0], v[:, 1]) >= 100.0
        ck = ck.copy()
        ck[fast] = translate(ck[fast], np.asarray(gl["yaw"][1:], np.float64)[fast], hd[fast])
        keys = np.concatenate([lkeys[:j + 1], ck.astype(np.int32)])
        assert len(keys) == len(sp)
        kp = out.with_name(out.stem + ".keys.npy")
        np.save(kp, keys)
        info["keys"] = str(kp)
    import hashlib
    info["md5"] = hashlib.md5(out.read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"energy spine: line rows 0-{j} (root boost x{vr:.3f}) + a {len(gl)}-state chain to cell "
          f"{c['cell']} at {c['progress']:.2f}% with h {eh:,.0f} ({c['gain']:+,.0f} u over the line), "
          f"{len(sp)} states, end d {info['end_d']:,.0f} -> {out}")
    print("ENERGY_SPLICE " + json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
