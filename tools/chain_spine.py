#!/usr/bin/env python3
"""chain_spine.py - a keyed curriculum spine from a policy line + one or more exact Go-Explore
chains spliced end to end (pbr5b1, 2026-10-25).

    python tools/chain_spine.py --line BEAM.npz --out SPINE.npy \\
        --chain ARCH1.npz:CELL1 [--chain ARCH2.npz:CELL2 ...] [--map M.bsp] [--field F.npz]

The line is a beam_tas --save-frontier .npz (states + the policy's own per-decision actions) or a
spine .npy with a .keys.npy beside it. Each --chain names an explore_phase1 --exact-chains archive
and a cell of it (tools/ge_energy_pick.py or the deepest); the chain is replayed exactly with
tools/ge_replay.py, matched to the current line at its root (an exact copy of a line state) and
spliced there; its views are put along the velocity and its held keys translated so the wish
direction survives (tools/spine_canon_keys.translate). The result is cut at its best depth on the
field and written with SPINE.keys.npy for train_fast --demo-keys. Chains are spliced in order, so
a second chain is rooted on the first chain's states (an energy pick, then a depth search from
it). Nothing comes from a demo or a record; no constant names a map (CLAUDE.md 0, 0b)."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None):
    from auto_spine import keys_from_acts, load_field, field_d, clean
    from ge_replay import chain_to
    from spine_canon_view import canon_view
    from spine_canon_keys import translate
    from surfgym.core import STATE_DTYPE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--line", required=True)
    ap.add_argument("--chain", action="append", required=True, metavar="ARCHIVE:CELL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_princessburglar_b26.bsp"))
    ap.add_argument("--field", default=str(ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"))
    a = ap.parse_args(argv)
    if a.line.endswith(".npz"):
        z = np.load(a.line, allow_pickle=False)
        line = np.asarray(z["states"], STATE_DTYPE)
        keys = keys_from_acts(z["acts"], int(z["act_every"]), len(line))
    else:
        line = np.asarray(np.load(a.line), STATE_DTYPE)
        keys = np.asarray(np.load(a.line[:-4] + ".keys.npy"), np.int32)
    field = load_field(a.field)
    info = {"line": a.line, "chains": []}
    for spec in a.chain:
        arch, cell = spec.rsplit(":", 1)
        cell = int(cell)
        tmp = Path(a.out).with_suffix(f".chain{len(info['chains'])}.npy")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "ge_replay.py"), "--map", a.map,
                            "--archive", arch, "--field", a.field, "--out", str(tmp),
                            "--cell", str(cell)], capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists():
            raise SystemExit(f"ge_replay failed on {spec}: {r.stdout[-400:]} {r.stderr[-400:]}")
        gl = np.asarray(np.load(tmp), STATE_DTYPE)
        zz = np.load(arch, allow_pickle=False)
        ch = chain_to(zz["parent"], cell)
        seq = (np.concatenate([zz["link_acts"][zz["link_off"][c]:zz["link_off"][c + 1]]
                               for c in ch[1:]]) if len(ch) > 1 else np.zeros((0, 6), np.int64))
        if len(seq) != len(gl) - 1:
            raise SystemExit(f"{spec}: {len(seq)} link actions for {len(gl)} states")
        ro = np.asarray(gl[0]["origin"], np.float64)
        dist = np.linalg.norm(np.asarray(line["origin"], np.float64) - ro, axis=1)
        j = int(np.argmin(dist))
        if dist[j] > 1.0:
            raise SystemExit(f"{spec}: the chain's root is {dist[j]:.1f} u from the line")
        boost = float(np.linalg.norm(gl[0]["velocity"])
                      / max(1e-6, float(np.linalg.norm(line[j]["velocity"]))))
        chain, _m = canon_view(gl[1:])
        ck = np.asarray(seq[:, [2, 3, 5]], np.int32).copy()
        v = np.asarray(gl["velocity"][1:], np.float64)
        hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        fast = np.hypot(v[:, 0], v[:, 1]) >= 100.0
        ck[fast] = translate(ck[fast], np.asarray(gl["yaw"][1:], np.float64)[fast], hd[fast])
        line = np.concatenate([np.asarray(line[:j + 1], STATE_DTYPE), np.asarray(chain, STATE_DTYPE)])
        keys = np.concatenate([np.asarray(keys[:j + 1], np.int32), ck])
        info["chains"].append({"spec": spec, "root_index": j, "root_boost": round(boost, 3),
                               "chain_len": int(len(gl))})
    d = field_d(field, line)
    e = int(np.argmin(d))
    sp = clean(line[:e + 1])
    np.save(a.out, sp)
    kp = a.out[:-4] + ".keys.npy"
    np.save(kp, np.asarray(keys[:e + 1], np.int32))
    D0 = 57149.125
    info.update(n=int(len(sp)), end_d=round(float(d[e]), 1), end_pct=round(100 * (D0 - d[e]) / D0, 2),
                md5=hashlib.md5(open(a.out, "rb").read()).hexdigest(),
                keys_md5=hashlib.md5(open(kp, "rb").read()).hexdigest())
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
