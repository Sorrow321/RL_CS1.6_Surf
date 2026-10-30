# one line per greedy episode: end, time, best progress (grader rule), where it ended, whether it
# reached the pit column's north end (y > -1400), which side region it entered, its energy there
import sys, numpy as np
sys.path.insert(0, "python"); sys.path.insert(0, "autoresearch")
import score
f = score.field()
for path in sys.argv[1:]:
    s = score.score_file(path)
    print(f"== {path.split('/')[-1]}: score {s.get('score')} best {s.get('best_progress_pct')}% median {s.get('median_progress_pct')}% fin {s.get('finishes')}/{s.get('episodes')}")
    for hdr, r, tr in score.first_episodes(score.episodes(path), score.SCORE_EPISODES):
        tick = float(hdr.get("tick_ms", 10.0)) / 1000
        p = r[:, 1:4]; v = r[:, 4:7]
        d = f.sample(p).astype(np.float64); d0 = d[0]
        E = p[:, 2] + (v ** 2).sum(1) / 1600
        ok = d < f.reach_max - 16
        best = 100 * (d0 - d[ok].min()) / d0 if ok.any() else 0
        north = np.flatnonzero(p[:, 1] > -1400)
        side = "-"
        if north.size:
            i = north[0]
            w = np.any((p[i:, 0] < -2832) & (p[i:, 1] > -1500)); e = np.any((p[i:, 0] > -1560) & (p[i:, 1] > -1500))
            side = ("W" if w else "") + ("E" if e else "") or "col"
            nE = f"E@north {E[i]:5.0f} t {i*tick:4.1f}s"
        else:
            nE = "never north"
        room = np.any(p[:, 1] > -400)
        print(f"  ep {hdr.get('episode')}: {tr.get('end'):5s} {len(r)*tick:6.1f}s best {best:5.2f}% end ({p[-1,0]:6.0f},{p[-1,1]:6.0f},{p[-1,2]:5.0f}) zmin {p[:,2].min():5.0f} Emax {E.max():5.0f} | {nE} side {side} room {'Y' if room else 'n'}")
