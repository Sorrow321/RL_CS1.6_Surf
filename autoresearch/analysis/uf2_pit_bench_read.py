# MEASUREMENT: from pit starts, did an episode climb out (z back above 400 after its lowest point,
# and a field progress above 2%)? Progress is measured from the MAP START's d0 (30,647.5).
import sys, json
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
def eps(path):
    out, hdr, cur = [], None, []
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                out.append((hdr, np.asarray(cur, np.float64), d)); hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return out
for path in sys.argv[1:]:
    E = eps(path); n_out = 0; rows = []
    for h, r, tr in E:
        p = r[:, 1:4]; v = r[:, 4:7]
        d = f.sample(p).astype(np.float64); prog = 100 * (30647.5 - d) / 30647.5
        imin = int(np.argmin(p[:, 2]))
        zafter = p[imin:, 2].max()
        out = bool(zafter > 400 and prog.max() > 2.0)
        n_out += out
        tick = float(h.get("tick_ms", 10)) / 1000
        rows.append(f"    start z {p[0,2]:6.0f} prog {prog[0]:6.2f}% |v| {np.linalg.norm(v[0]):5.0f} -> best {prog.max():6.2f}% minz {p[:,2].min():6.0f} z-after-min {zafter:6.0f} t {len(r)*tick:5.1f}s {tr.get('end')}{'  OUT' if out else ''}")
    print(f"{path.split('/')[-1]}: {n_out}/{len(E)} climbed out")
    print("\n".join(rows[:12]))
