"""Read a Go-Explore archive on surf_za_loop (zlr1b2, 2026-10-31). ANALYSIS ONLY.

For an explore_phase1 archive.npz: how many states, the best progress on config.field, the
energy height h = z + |v|^2 / 2g distribution, the top states by h and by progress (position,
speed, depth in ticks), and a coarse x/y/z occupancy. Optionally writes a spawn-state file of
one state per (--pick-cell u) position cell - the most energetic one - with clocks zeroed
(--out), the generic subset the trainer reads with --spawn-states.

    python autoresearch/analysis/zl_ge_report.py runs/ge/zl1/archive.npz [--out x.npy]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
G = 800.0


def field():
    from surfgym.goalfield import GoalField
    cfg = json.loads((REPO / "autoresearch" / "config.json").read_text())
    z = np.load(REPO / cfg["field"], allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"])), float(cfg["start_d"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pick-cell", type=float, default=256.0)
    ap.add_argument("--forward", action="store_true",
                    help="keep only picks FORWARD of the start: deeper on the field (d < d0) or "
                         "with more energy height than the spawn at rest (h > h0 + 32 u). The "
                         "trainer's --ge-minutes keeps d < d0 only, which on a loop map drops the "
                         "half-pipe the run has to pump in")
    a = ap.parse_args()
    z = np.load(a.archive, allow_pickle=True)
    st = z["state"]
    pos = np.asarray(st["origin"], np.float64)
    vel = np.asarray(st["velocity"], np.float64)
    h = pos[:, 2] + (vel * vel).sum(1) / (2 * G)
    f, d0 = field()
    d = np.asarray(f.sample(pos), np.float64)
    ok = d < float(f.reach_max) - 16.0
    prog = np.where(ok, 100.0 * (d0 - d) / d0, -999.0)
    spd = np.linalg.norm(vel, axis=1)
    tick = np.asarray(st["tick"]) if "tick" in st.dtype.names else np.zeros(len(st))
    print(f"{a.archive}: {len(st):,} states; keys {list(z.keys())}")
    print(f"progress: max {prog.max():.2f}%  p99 {np.percentile(prog[ok], 99):.2f}%  "
          f"p90 {np.percentile(prog[ok], 90):.2f}%")
    print(f"h: min {h.min():.0f} p50 {np.median(h):.0f} p90 {np.percentile(h, 90):.0f} "
          f"p99 {np.percentile(h, 99):.0f} max {h.max():.0f}")
    print("h histogram (u):")
    edges = np.arange(np.floor(h.min() / 250) * 250, h.max() + 250, 250)
    cnt, _ = np.histogram(h, edges)
    for lo, c in zip(edges[:-1], cnt):
        if c:
            print(f"  {lo:7.0f}..{lo + 250:7.0f}  {c:9,d}")
    for name, key in (("h", h), ("progress", prog)):
        idx = np.argsort(-key)[:12]
        print(f"top by {name}:")
        for i in idx:
            print(f"  x {pos[i, 0]:7.0f} y {pos[i, 1]:7.0f} z {pos[i, 2]:7.0f}  spd {spd[i]:6.0f}"
                  f"  h {h[i]:7.0f}  prog {prog[i]:6.2f}%  tick {int(tick[i])}")
    # coarse occupancy by z band
    print("z bands (count, max h, max prog):")
    for lo in range(-4100, 3400, 500):
        m = (pos[:, 2] >= lo) & (pos[:, 2] < lo + 500)
        if m.any():
            print(f"  z {lo:6d}..{lo + 500:6d}  {int(m.sum()):9,d}  h {h[m].max():7.0f}"
                  f"  prog {prog[m].max():6.2f}%")
    if a.out:
        from surfgym.core import STATE_DTYPE as SD
        c = np.floor(pos / a.pick_cell).astype(np.int64)
        key = (c[:, 0] * 1_000_003 + c[:, 1]) * 1_000_003 + c[:, 2]
        order = np.lexsort((-h, key))
        first = np.ones(len(order), bool)
        first[1:] = key[order][1:] != key[order][:-1]
        pick = order[first]
        if a.forward:
            zp = json.loads((REPO / Path(json.loads((REPO / "autoresearch" / "config.json").read_text())["map"]).with_suffix(".zones.json")).read_text())
            h0 = float(zp["spawns"][0]["origin"][2])          # the spawn is at rest
            fw = (ok[pick] & (d[pick] < d0)) | (h[pick] > h0 + 32.0)
            print(f"--forward: {int(fw.sum()):,} of {len(pick):,} picks are deeper than the start "
                  f"(d < {d0:,.0f}) or above its energy (h > {h0 + 32.0:,.0f})")
            pick = pick[fw]
        rows = np.zeros(len(pick), SD)
        for nm in SD.names:
            rows[nm] = st[nm][pick]
        rows["tick"] = 0
        rows["stuck_ticks"] = 0
        np.save(a.out, rows)
        print(f"wrote {len(rows):,} states (the most energetic per {a.pick_cell:g} u cell) -> "
              f"{a.out}; h of those: p50 {np.median(h[pick]):.0f} max {h[pick].max():.0f}")


if __name__ == "__main__":
    main()
