"""Box-side: the deepest reservoir states of a run's ckpt_latest.pt (where training episodes
survived deepest). Usage: python3 res_probe.py runs/<run>/ckpt_latest.pt"""
import sys, time, numpy as np, torch
sys.path.insert(0, 'python')
from surfgym.goalfield import GoalField
for _ in range(10):
    try:
        ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); break
    except Exception:
        time.sleep(3)
z = np.load('maps/surf_src_sidistic.goal_32.npz', allow_pickle=False)
f = GoalField(z['grid'].astype(np.float32) * float(z['quant']), z['mins'], float(z['cell']), float(z['reach_max']))
st = ck['respawn']['states']
p = np.asarray(st['origin'], np.float64); v = np.asarray(st['velocity'], np.float64)
d = f.sample(p).astype(np.float64)
ok = d < f.reach_max - 0.5 * f.cell
print('step', ck['global_step'], 'reservoir rows', len(st), 'on-field', int(ok.sum()))
o = np.argsort(np.where(ok, d, 1e9))
for i in o[:25]:
    hs = np.hypot(v[i, 0], v[i, 1]); hd = np.degrees(np.arctan2(v[i, 1], v[i, 0]))
    print(f"  d {d[i]:7.0f} ({100*d[i]/97768:5.2f}%) pos ({p[i,0]:6.0f},{p[i,1]:6.0f},{p[i,2]:6.0f}) hs {hs:5.0f} vz {v[i,2]:6.0f} head {hd:6.1f} gnd {int(st['onground'][i])} tick {int(st['tick'][i])}")
deep = ok & (d < 72800)
print('rows deeper than 72,800u:', int(deep.sum()))
if deep.any():
    print('  x range', p[deep, 0].min(), p[deep, 0].max(), ' y', p[deep, 1].min(), p[deep, 1].max(), ' z', p[deep, 2].min(), p[deep, 2].max())
