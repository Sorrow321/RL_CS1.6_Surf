"""pbf_ge_report.py - what a Go-Explore archive from the true start reached (pbfr1b1, 2026-10-29).

    python autoresearch/analysis/pbf_ge_report.py runs/<ge>/archive.npz

MEASUREMENT ONLY (CLAUDE.md 0 / 0b): the boxes below are map coordinates used to REPORT where the
search got (the ridge room east of the spawn, the arch, the north shelf, the tunnel); nothing here
feeds a reward, a spawn rule or a selection. Prints: cells, the deepest cell on the grader field
(progress % past the start), the energy height h = z + |v|^2/1600 maxima by region, and how many
cells sit on the shelf / in the tunnel."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

F, D0 = score.field(), score.START_D
for fn in _args:
    z = np.load(fn)
    st = z["state"]
    o = st["origin"].astype(np.float64)
    v = st["velocity"].astype(np.float64)
    h = o[:, 2] + (v ** 2).sum(1) / 1600.0
    d = F.sample(o).astype(np.float64)
    reach = d < F.reach_max - 0.5 * F.cell
    prog = np.where(reach, 100.0 * (D0 - d) / D0, -99.0)
    depth = z["depth"]
    print(f"{fn}: {len(st):,} cells; best progress {prog.max():.3f}% "
          f"(cell {int(prog.argmax())}, origin {o[prog.argmax()].round(0).tolist()}, "
          f"depth {int(depth[prog.argmax()])} ticks)")
    regions = {
        "ridge room (x>1190)": o[:, 0] > 1190,
        "arch (1120<x<1200)": (o[:, 0] > 1120) & (o[:, 0] < 1200),
        "spawn room east (860<x<1120)": (o[:, 0] > 860) & (o[:, 0] <= 1120) & (o[:, 1] > -3200),
        "spawn room west (660<x<=860)": (o[:, 0] > 660) & (o[:, 0] <= 860) & (o[:, 1] > -3200),
        "shelf (660<x<900, y>-2900, z>3718)": (o[:, 0] > 660) & (o[:, 0] < 900) & (o[:, 1] > -2900)
                                              & (o[:, 2] > 3718),
        "tunnel (x<=660, z>3700)": (o[:, 0] <= 660) & (o[:, 2] > 3700) & (o[:, 1] > -3000),
        "past start (prog>0.5%)": prog > 0.5,
    }
    for name, m in regions.items():
        if m.any():
            i = np.flatnonzero(m)[np.argmax(h[m])]
            print(f"  {name:38s} {int(m.sum()):7,} cells; h max {h[i]:6.0f} at "
                  f"{o[i].round(0).tolist()} |v| {np.linalg.norm(v[i]):5.0f} depth {int(depth[i])}")
        else:
            print(f"  {name:38s}       0 cells")
    # energy histogram in the ridge room
    m = o[:, 0] > 1190
    if m.any():
        hb = np.arange(3500, 4200, 50)
        cnt, _ = np.histogram(h[m], hb)
        print("  ridge-room h histogram: " + " ".join(f"{int(a)}:{int(c)}" for a, c in zip(hb[:-1], cnt) if c))
