"""Per run: greedy eval episodes reaching the final room - entry speed, launch attempts (rising near
the holes), the attempt's max vz and max |y|, shaft entries, finishes."""
import sys, glob, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2])); sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python'))
import score
for run in sys.argv[1:]:
    files = sorted(glob.glob(f'{run}/traj_[0-9]*.jsonl'))
    n_ep = n_room = n_att = n_shaft = n_fin = 0; entry = []; vz = []; ymax = []
    for f in files:
        try:
            eps = score.episodes(f)
        except Exception:
            continue
        for h, r, tr in eps:
            n_ep += 1
            p = r[:, 1:4]; v = r[:, 4:7]
            if tr.get('end') == 'done':
                n_fin += 1
            room = np.flatnonzero((p[:, 0] > 2300) & (p[:, 2] < 1180))
            if not len(room):
                continue
            n_room += 1
            entry.append(np.linalg.norm(v[room[0]]))
            att = np.flatnonzero((p[:, 0] > 2440) & (p[:, 0] < 2720) & (v[:, 2] > 300) & (p[:, 2] > 980) & (p[:, 2] < 1180))
            if len(att):
                n_att += 1; vz.append(v[att, 2].max()); ymax.append(np.abs(p[att, 1]).max())
            q = p[room[0]:]
            if ((q[:, 0] > 2464) & (q[:, 0] < 3424) & (np.abs(q[:, 1]) < 704) & (q[:, 2] > 1232) & (q[:, 2] < 3000)).any():
                n_shaft += 1
    pc = lambda a: np.percentile(a, [10, 50, 90]).round(0) if len(a) else '-'
    print(f"{run}: {len(files)} evals {n_ep} eps; room {n_room}; entry |v| p10/50/90 {pc(entry)}; attempts {n_att} "
          f"(max vz {pc(vz)}, max |y| {pc(ymax)}); shaft {n_shaft}; finishes {n_fin}")
