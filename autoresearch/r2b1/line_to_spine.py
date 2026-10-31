#!/usr/bin/env python3
"""line_to_spine.py (uf2r2b1) - a beam_tas --save-frontier line -> a train_fast --demo-file spine.

The line is the policy's OWN search line from the true start (beam_tas on its weights; nothing from
the record). Writes <out>.npy (STATE_DTYPE, one state per tick, tick/stuck_ticks cleared - the
auto_spine.clean rule) and <out>.keys.npy (the held keys each state was reached with:
auto_spine.keys_from_acts), and prints the line's time, end d and energy profile.

    python autoresearch/r2b1/line_to_spine.py runs/beam_r2/s1/beam_frontier.npz runs/beam/r2s1 \
        [--field maps/surf_unitfarmer2.goalc_32.npz]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out", help="output stem: <out>.npy and <out>.keys.npy")
    ap.add_argument("--field", default=str(ROOT / "maps" / "surf_unitfarmer2.goalc_32.npz"))
    ap.add_argument("--tick-ms", type=float, default=7.666667)
    a = ap.parse_args()
    from auto_spine import clean, keys_from_acts, load_field, field_d
    z = np.load(a.npz, allow_pickle=False)
    st = z["states"]
    fin = int(z["finished"]) if "finished" in z.files else 0
    ok = int(z["replay_ok"])
    keys = keys_from_acts(z["acts"], int(z["act_every"]), len(st))
    f = load_field(a.field)
    d = field_d(f, st)
    o = np.asarray(st["origin"], np.float64)
    v = np.asarray(st["velocity"], np.float64)
    E = o[:, 2] + (v * v).sum(1) / 1600.0
    tk = a.tick_ms / 1000.0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out) + ".npy", clean(st))
    np.save(str(out) + ".keys.npy", keys)
    info = {"src": str(a.npz), "states": int(len(st)), "finished": fin, "replay_ok": ok,
            "secs": round(len(st) * tk, 3), "end_d": round(float(d[-1]), 1),
            "best_d": round(float(np.min(d)), 1)}
    print(json.dumps(info))
    d0 = float(d[0])
    for L in (2, 5, 10, 20, 40, 60, 70, 80, 90, 95, 99):
        pr = np.maximum.accumulate(100.0 * (d0 - d) / d0)
        i = int(np.argmax(pr >= L)) if (pr >= L).any() else -1
        if i >= 0:
            print(f"  {L:3d}% at {i * tk:6.2f}s  E {E[i]:6.0f} |v| {np.linalg.norm(v[i]):5.0f}")


if __name__ == "__main__":
    main()
