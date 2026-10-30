"""uf2_eval_report.py - what each greedy eval episode on surf_unitfarmer2 did (MEASUREMENT ONLY).

Per episode of runs/<run>/traj_<step>.jsonl: how it ended, its seconds, its best progress on the
grader field (goalc_32), its lowest z, its top speed, and the PIT columns: whether it went below
z -400 inside the start pit's footprint (the AABB of trigger_teleport *7, the pit's own fall
catch: x -2,880..-1,472, y -3,936..192), its energy height h = z + |v|^2/1600 at the pit's lowest
point, and whether it came back up past z +300 afterwards ("out"). The footprint and thresholds
are a bench instrument (CLAUDE.md 0b: map-specific instruments are for measurement); nothing
here enters a reward, a spawn rule or a schedule.

    python autoresearch/analysis/uf2_eval_report.py runs/uf2r1b3_x/traj_000750000000.jsonl [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autoresearch"))
import score  # noqa: E402

PIT_LO = np.array([-2880.0, -3936.0])
PIT_HI = np.array([-1472.0, 192.0])


def report(path: Path) -> None:
    f = score.field()
    eps = score.first_episodes(score.episodes(path), score.SCORE_EPISODES)
    print(f"{path.name}: {len(eps)} episodes")
    for hdr, r, tr in eps:
        p, v = r[:, 1:4], r[:, 4:7]
        d = f.sample(p).astype(np.float64)
        ok = d < f.reach_max - 0.5 * f.cell
        d0 = float(d[0])
        prog = 100.0 * (d0 - (d[ok].min() if ok.any() else d0)) / d0
        sp = np.linalg.norm(v, axis=1)
        h = p[:, 2] + sp ** 2 / 1600.0
        secs = len(r) * float(hdr.get("tick_ms", 10.0)) / 1000.0
        inpit = (np.all((p[:, :2] >= PIT_LO) & (p[:, :2] <= PIT_HI), axis=1)
                 & (p[:, 2] < -400.0))
        pit = ""
        if inpit.any():
            k = int(np.flatnonzero(inpit)[np.argmin(p[inpit, 2])])
            out = bool(np.any(p[k:, 2] > 300.0))
            pit = (f" | PIT at {k * float(hdr.get('tick_ms', 10.0)) / 1000:.1f}s z "
                   f"{p[k, 2]:.0f} |v| {sp[k]:.0f} h {h[k]:.0f}" + (" OUT" if out else ""))
        print(f"  ep {hdr.get('episode')}: {tr.get('end'):5s} {secs:5.1f}s best "
              f"{prog:5.2f}% zmin {p[:, 2].min():6.0f} vmax {sp.max():5.0f} end "
              f"({p[-1, 0]:6.0f},{p[-1, 1]:6.0f},{p[-1, 2]:5.0f}){pit}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        report(Path(a))
