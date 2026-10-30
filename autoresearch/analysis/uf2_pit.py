#!/usr/bin/env python3
"""uf2_pit.py - MEASUREMENT ONLY (CLAUDE.md 0b: map instruments read results, never train): per
greedy eval episode, did it go into unitfarmer2's start pit, with how much energy, and did it climb
back out. Pit = below z -400 inside x -2,900..-1,450, y -3,950..200 (the `*7` catch's footprint).
E = z + |v|^2/1600. "climb" = the highest z after the pit bottom; "out" = later above z 300 with
y > -2,000 (where the record leaves the pit northwards, t+7.5..t+8.5).

    python autoresearch/analysis/uf2_pit.py runs/<agent>/<run>/traj_<step>.jsonl [...]
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "autoresearch"))
import score as S  # noqa: E402


def pit_stats(r, tick):
    p, v = r[:, 1:4], r[:, 4:7]
    E = p[:, 2] + (v ** 2).sum(1) / 1600.0
    inpit = (p[:, 2] < -400) & (p[:, 0] > -2900) & (p[:, 0] < -1450) & (p[:, 1] > -3950) & (p[:, 1] < 200)
    if not inpit.any():
        return None
    i0 = int(np.argmax(inpit))
    idx = np.flatnonzero(inpit)
    ib = int(idx[np.argmin(p[idx, 2])])                 # the bottom, among in-pit ticks only
    w = slice(ib, min(len(p), ib + int(6.0 / tick)))    # the 6 s after it
    ic = ib + int(np.argmax(p[w, 2]))
    out = bool(np.any((p[w, 2] > 300) & (p[w, 1] > -2000)))
    sp = np.linalg.norm(v, axis=1)
    return dict(t_in=i0 * tick, E_in=float(E[i0]), E_bot=float(E[ib]), v_pit=float(sp[idx].max()),
                zbot=float(p[ib, 2]), climb=float(p[ic, 2]), E_climb=float(E[ic]), t_climb=ic * tick,
                out=out, E_min_pit=float(E[idx].min()))


def main():
    for path in sys.argv[1:]:
        eps = S.first_episodes(S.episodes(Path(path)), S.SCORE_EPISODES)
        n_in = n_out = 0
        lines = []
        for h, r, tr in eps:
            tick = float(h.get("tick_ms", 10.0)) / 1000.0
            s = pit_stats(r, tick)
            if s is None:
                lines.append(f"  ep{h.get('episode')}: no pit ({tr.get('end')}, {len(r)*tick:.1f}s)")
                continue
            n_in += 1
            n_out += s["out"]
            lines.append(f"  ep{h.get('episode')}: in @{s['t_in']:.2f}s E {s['E_in']:.0f} | bottom z {s['zbot']:.0f} "
                         f"E {s['E_bot']:.0f} | vmax {s['v_pit']:.0f} | climb to z {s['climb']:.0f} @{s['t_climb']:.2f}s "
                         f"E {s['E_climb']:.0f} | out {s['out']} | {tr.get('end')} {len(r)*tick:.1f}s")
        print(f"{Path(path).name}: pit {n_in}/{len(eps)}, out {n_out}")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
