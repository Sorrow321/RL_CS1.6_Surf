#!/usr/bin/env python3
"""zl_ge_spine.py - one curriculum spine (+ its held keys) from a Go-Explore archive (zlr1b3).

The chain to the archive's deepest cell on config.field (or --cell) is replayed exactly
(tools/ge_replay.py), its held keys are read off the chain's own link actions
(tools/auto_spine.chain_keys: state t holds the keys of the action that produced it, state 0
neutral), and its view is put along the velocity on every row moving >= 100 u/s
(tools/spine_canon_view.py: a --view-absolute velocity policy looks there, the explorer's
--view-follow offsets do not). The line is the search's own, from the TRUE spawn: no demo, no
record, no policy (CLAUDE.md section 0: an own-search spawn file, declared with
box.py launch --self-states).

    python autoresearch/analysis/zl_ge_spine.py ARCHIVE.npz OUT.npy [--cell N]
-> OUT.npy (STATE_DTYPE rows, clocks zeroed) and OUT.keys.npy ((rows, 3) int32 fwd/side/duck)
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))
CFG = json.loads((REPO / "autoresearch" / "config.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--cell", type=int, default=None)
    a = ap.parse_args()
    from surfgym.goalfield import GoalField
    from auto_spine import chain_keys
    out = Path(a.out)
    raw = out.with_name(out.stem + ".raw.npy")
    cmd = [sys.executable, str(REPO / "tools" / "ge_replay.py"), "--map", str(REPO / CFG["map"]),
           "--archive", a.archive, "--field", str(REPO / CFG["field"]), "--out", str(raw)]
    if a.cell is not None:
        cmd += ["--cell", str(a.cell)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit("ge_replay did not replay the whole chain")
    zf = np.load(REPO / CFG["field"], allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    line = np.load(raw, allow_pickle=False)
    k = chain_keys(a.archive, f, len(line), cell=a.cell)
    if k is None:
        raise SystemExit("chain keys do not match the replayed line's length")
    keys = np.concatenate([np.array([[1, 1, 0]], np.int32), k]).astype(np.int32)
    r = subprocess.run([sys.executable, str(REPO / "tools" / "spine_canon_view.py"), str(raw),
                        str(out)], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit("spine_canon_view failed")
    sp = np.load(out, allow_pickle=False)
    assert len(sp) == len(keys), (len(sp), len(keys))
    kp = out.with_name(out.stem + ".keys.npy")
    np.save(kp, keys)
    d = f.sample(sp["origin"].astype(np.float64)).astype(np.float64)
    o, v = sp["origin"].astype(np.float64), sp["velocity"].astype(np.float64)
    E = o[:, 2] + (v * v).sum(1) / 1600.0
    d0 = float(CFG["start_d"])
    print(f"spine {out} ({len(sp)} rows, md5 {hashlib.md5(out.read_bytes()).hexdigest()}), keys "
          f"{kp.name} (md5 {hashlib.md5(kp.read_bytes()).hexdigest()}); start d {d[0]:,.0f}, end d "
          f"{d[-1]:,.0f} = {100 * (d0 - d[-1]) / d0:.2f}%, best {100 * (d0 - d.min()) / d0:.2f}%; "
          f"E start {E[0]:,.0f} max {E.max():,.0f} end {E[-1]:,.0f}; end origin "
          f"{o[-1].round(0).tolist()}")


if __name__ == "__main__":
    main()
