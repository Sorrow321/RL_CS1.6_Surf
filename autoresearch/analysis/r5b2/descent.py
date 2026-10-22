import sys, json, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[2])); sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python'))
import score
F=score.field()
G=F.grid; mins=np.asarray(F.mins); c=F.cell
nz,ny,nx=G.shape
def idx(p):
    q=((np.asarray(p,float)-mins)/c).astype(int); return (q[2],q[1],q[0])
def pos(i): return mins + (np.array([i[2],i[1],i[0]])+0.5)*c
start=tuple(float(v) for v in sys.argv[1:4]) if len(sys.argv)>3 else (123,-607,2018)
i=idx(start)
# nearest honest cell
if G[i]>F.reach_max:
    best=None
    for a in range(-2,3):
        for b in range(-2,3):
            for cc in range(-2,3):
                j=(i[0]+a,i[1]+b,i[2]+cc)
                if G[j]<F.reach_max and (best is None or G[j]<G[best]): best=j
    i=best
path=[i]
nb=[(a,b,cc) for a in (-1,0,1) for b in (-1,0,1) for cc in (-1,0,1) if (a,b,cc)!=(0,0,0)]
for step in range(5000):
    best=None; bd=G[i]
    for a,b,cc in nb:
        j=(i[0]+a,i[1]+b,i[2]+cc)
        if min(j)<0 or j[0]>=nz or j[1]>=ny or j[2]>=nx: continue
        if G[j]<bd: bd=G[j]; best=j
    if best is None: break
    i=best; path.append(i)
every=int(sys.argv[4]) if len(sys.argv)>4 else 6
for k in list(range(0,len(path),every))+[len(path)-1]:
    p=pos(path[k]); print(k, np.round(p).astype(int), round(float(G[path[k]])), f"{100*(97768-G[path[k]])/97768:.2f}%")
