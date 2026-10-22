import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[2])); sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python'))
import score
F=score.field()
for f in sys.argv[1:]:
    eps=score.episodes(f)
    print(f.split('/')[-1])
    for i,(h,r,tr) in enumerate(eps):
        p=r[:,1:4]; d=F.sample(p).astype(float)
        ok=d<F.reach_max-0.5*F.cell
        j=int(np.argmin(np.where(ok,d,1e18)))
        sp=np.linalg.norm(r[j,4:6])
        print(f"  ep{i} {len(r)/100:6.1f}s {tr.get('end'):5s} best {100*(97768-d[j])/97768:6.2f}% at {j/100:6.2f}s pos {np.round(p[j]).astype(int)} hspd {sp:5.0f} vz {r[j,6]:6.0f} | end {np.round(p[-1]).astype(int)}")
