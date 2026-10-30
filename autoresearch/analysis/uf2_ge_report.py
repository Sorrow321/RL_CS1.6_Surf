#!/usr/bin/env python3
"""uf2_ge_report.py - read a Go-Explore archive (tools/explore_phase1.py) on the grader's field.

Prints, for the archive's cells: progress on config.field (goalc_32, d0 = config.start_d), energy
height E = z + |v|^2/1600, depth (ticks from the spawn along the chain), and the deepest cells
clustered by position. Reporting only: nothing here feeds a search, a reward or a spawn rule.

    python autoresearch/analysis/uf2_ge_report.py runs/ge/uf2a/archive.npz [--top 25]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
CFG = json.loads((REPO / "autoresearch" / "config.json").read_text())


def field():
    from surfgym.goalfield import GoalField
    z = np.load(REPO / CFG["field"], allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tick-ms", type=float, default=10.0)
    ap.add_argument("--wr", default=None, help="ANALYSIS ONLY: the record jsonl for the same-motion reading")
    a = ap.parse_args()
    z = np.load(a.archive, allow_pickle=False)
    st = z["state"]
    n = len(st)
    org = st["origin"].astype(np.float64)
    vel = st["velocity"].astype(np.float64)
    f = field()
    d = f.sample(org).astype(np.float64)
    bad = d >= f.reach_max - 0.5 * f.cell
    D0 = float(CFG["start_d"])
    prog = np.where(bad, np.nan, 100.0 * (D0 - d) / D0)
    E = org[:, 2] + (vel ** 2).sum(1) / 1600.0
    sp = np.linalg.norm(vel, axis=1)
    depth = z["depth"].astype(np.int64) if "depth" in z.files else np.zeros(n, np.int64)
    print(f"{a.archive}: {n:,} cells; off-field {int(bad.sum()):,}; "
          f"progress max {np.nanmax(prog):.2f}%; E max {E.max():.0f}; z min {org[:,2].min():.0f}")
    # progress bands
    print("band      cells   E max   E p90   speed max  depth(s) min")
    for lo in [-12, -8, -4, -2, 0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]:
        hi = {-12: -8, -8: -4, -4: -2, -2: 0, 0: 2, 2: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
              10: 12, 12: 15, 15: 20, 20: 30, 30: 50, 50: 101}[lo]
        m = (prog >= lo) & (prog < hi)
        if m.any():
            print(f"[{lo:4d},{hi:4d}) {int(m.sum()):8,d} {E[m].max():7.0f} {np.percentile(E[m],90):7.0f}"
                  f" {sp[m].max():9.0f} {depth[m].min()*a.tick_ms/1000:8.2f}")
    # deepest cells, clustered on a 256 u grid
    o = np.argsort(np.nan_to_num(-prog, nan=1e9))
    seen = set()
    print(f"deepest cells (one per 256 u block), top {a.top}:")
    k = 0
    for i in o:
        if np.isnan(prog[i]):
            continue
        b = tuple((org[i] // 256).astype(int))
        if b in seen:
            continue
        seen.add(b)
        print(f"  {prog[i]:6.2f}%  ({org[i,0]:7.0f},{org[i,1]:7.0f},{org[i,2]:6.0f}) |v| {sp[i]:5.0f} "
              f"v ({vel[i,0]:6.0f},{vel[i,1]:6.0f},{vel[i,2]:6.0f}) E {E[i]:6.0f} t {depth[i]*a.tick_ms/1000:5.2f}s "
              f"ground {int(st['onground'][i])} idx {i}")
        k += 1
        if k >= a.top:
            break
    return a




def same_motion(archive, wr, times=(1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                                    8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 15.0), radius=200.0):
    """ANALYSIS ONLY (CLAUDE.md section 0): per record time (from its start curtain), the best
    archive energy height within RADIUS u of the record's position among states moving the
    same way (velocity cosine > 0.8) - the October 2026-10-16 'same-motion energy' reading."""
    z = np.load(archive, allow_pickle=False)
    st = z["state"]
    org = st["origin"].astype(np.float64)
    vel = st["velocity"].astype(np.float64)
    E = org[:, 2] + (vel ** 2).sum(1) / 1600.0
    vn = vel / np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-6)
    rows = [json.loads(l) for l in open(wr) if l.strip() and l.strip()[0] == "["]
    r = np.asarray(rows, np.float64)
    ic = int(np.argmax(r[:, 2] <= -2400))
    print("same-motion energy (record t from the curtain: record E / best archive E / n):")
    for t in times:
        i = ic + int(round(t * 100))
        if i >= len(r):
            break
        p, v = r[i, 1:4], r[i, 4:7]
        Er = p[2] + (v ** 2).sum() / 1600.0
        m = np.linalg.norm(org - p, axis=1) < radius
        m &= (vn @ (v / max(np.linalg.norm(v), 1e-6))) > 0.8
        best = f"{E[m].max():6.0f}" if m.any() else "  none"
        print(f"  t+{t:4.1f} ({p[0]:6.0f},{p[1]:6.0f},{p[2]:5.0f}) record {Er:6.0f}  archive {best}  n {int(m.sum())}")


if __name__ == "__main__":
    a = main()
    if a.wr:
        same_motion(a.archive, a.wr)
