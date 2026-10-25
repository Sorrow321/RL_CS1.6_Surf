#!/usr/bin/env python3
"""ge_energy_picks.py - the explorer's most energetic states AGAINST the field (pbr3b4, 2026-10-25).

    python tools/ge_energy_picks.py ARCHIVE.npz OUT_PREFIX [--n 4] [--field F] [--min-sep 256]

From a tools/explore_phase1.py --exact-chains archive rooted at the true spawn: the cells whose
shaping-field distance is ABOVE the spawn's (behind the start - where a field-follower never
goes) ranked by energy height h = z + |v|^2 / 2g, the top --n at least --min-sep u apart. Each
pick's state is written to OUT_PREFIX_<k>.npy (a --start-state for beam_tas, where the policy's
own proposals take over from the explorer) and its exact chain from the spawn (ge_replay) to
OUT_PREFIX_<k>_chain.npy. Nothing names a place on a map: on any map this is "the energy a
start that runs against the field could collect". The user's insight that motivated it
(2026-10-25): a human surfs a field-opposed first ramp for momentum before the ramp the policy
starts on."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("out_prefix")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--field", default=str(ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"))
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_princessburglar_b26.bsp"))
    ap.add_argument("--min-sep", type=float, default=256.0)
    ap.add_argument("--gravity", type=float, default=800.0)
    a = ap.parse_args(argv)
    from surfgym.goalfield import GoalField
    z = np.load(a.archive, allow_pickle=False)
    st, parent = z["state"], z["parent"]
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"], float(zf["cell"]),
                  float(zf["reach_max"]))
    o = st["origin"].astype(np.float64)
    v = st["velocity"].astype(np.float64)
    d = f.sample(o).astype(np.float64)
    ok = d < f.reach_max - 0.5 * f.cell
    roots = np.flatnonzero(parent < 0)
    d0 = float(np.min(d[roots]))
    h = o[:, 2] + (v * v).sum(axis=1) / (2.0 * a.gravity)
    # VIABLE cells only: ones the explorer later restored from and found new cells beyond (some
    # cell's parent) - a cell reached a tick before a death has the most kinetic energy of all
    # (pbr3b4's first picks: states plunging into a fatal pit and a push-down trigger)
    viable = np.zeros(len(st), bool)
    viable[parent[parent >= 0]] = True
    cand = np.flatnonzero(ok & (d > d0) & viable)
    cand = cand[np.argsort(-h[cand])]
    picks = []
    for k in cand:
        if all(np.linalg.norm(o[k] - o[j]) >= a.min_sep for j in picks):
            picks.append(int(k))
        if len(picks) >= a.n:
            break
    h0 = float(h[roots].max())
    print(f"{len(cand):,} cells behind the start (d > {d0:,.0f}); spawn h {h0:,.0f}")
    for i, k in enumerate(picks):
        row = np.array(st[k:k + 1], copy=True)
        row["tick"] = 0
        row["stuck_ticks"] = 0
        np.save(f"{a.out_prefix}_{i}.npy", row)
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "ge_replay.py"), "--map", a.map,
                            "--archive", a.archive, "--field", a.field, "--cell", str(k),
                            "--out", f"{a.out_prefix}_{i}_chain.npy"], capture_output=True,
                           text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"pick {i}: cell {k} h {h[k]:,.0f} (+{h[k] - h0:,.0f}) d {d[k]:,.0f} at "
              f"{np.round(o[k])} |v| {np.linalg.norm(v[k]):,.0f}; chain: {tail}")


if __name__ == "__main__":
    main()
