#!/usr/bin/env python3
# Usage: ZL_CKPT=<ckpt.pt> ZL_REC_DIR=<dir with inc_greedy.jsonl + inc_greedy_states.npy (record_ckpt
#   --episodes 1 --out <dir>/inc_greedy.jsonl --dump-states <dir>/inc_greedy_states.npz, then save
#   ep_0000 as inc_greedy_states.npy)> python autoresearch/analysis/zl_perturb_search.py [T0_TICK] [STRIDE]
# Exact replay = each recorded row's NEXT yaw relative to the sim's velocity + its keys (0.0-0.2 u
# over 5 s); the search adds one held offset c (deg) for D ticks at every root.
# zlr2b2: generic perturbation search over the policy's OWN greedy line (physics only, exact replay).
# Roots every STRIDE ticks along the line; at each root one held view offset c for D ticks, then the
# policy's own recorded controls again (open loop); keep the deepest field progress. No region, no
# record, no map constant: the roots are evenly spaced in time, the offsets a fixed grid.
import sys, json, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python')); sys.path.insert(0, str(ROOT/'tools')); sys.path.insert(0, str(ROOT/'autoresearch'))
import beam_tas as B
from surfgym.tick import TickClock
import score as S
import os
SP = Path(os.environ.get('ZL_REC_DIR', '.'))          # dir with the recording (inc_greedy.jsonl + inc_greedy_states.npy)
cfg = (torch.load(os.environ['ZL_CKPT'], map_location='cpu', weights_only=False).get('config') or {})
pst = np.load(SP/'inc_greedy_states.npy')
R=np.array([r for r in (json.loads(l) for l in open(SP/'inc_greedy.jsonl')) if isinstance(r,list)],float)
yaw=R[:,7]; side=R[:,14].astype(int); fwd=R[:,13].astype(int); btn=R[:,8].astype(int)
f=S.field(); d0=6361.8
T0=int(sys.argv[1]) if len(sys.argv)>1 else 14286; STRIDE=int(sys.argv[2]) if len(sys.argv)>2 else 35
CS=[-2.0,-1.5,-1.0,-0.5,0.5,1.0,1.5,2.0]; DS=[70,140,280]; POST=420
nlast=len(pst)-1-max(DS)-POST
roots=list(range(T0, nlast, STRIDE))
combos=[(r,c,dd) for r in roots for c in CS for dd in DS]
print(f"{len(roots)} roots x {len(CS)*len(DS)} = {len(combos)} envs", flush=True)
CH=4096
best_all=[]
t0=time.time()
for s0 in range(0,len(combos),CH):
    cb=combos[s0:s0+CH]; N=len(cb)
    core=B.build_sim(cfg, Path(os.environ.get('ZL_MAP', str(ROOT/'maps/surf_za_loop.bsp'))), N, 200000, tick=TickClock(7.0))
    core.set_teleport_fail(True); core.reset(0)
    rr=np.array([r for r,c,dd in cb]); cc=np.array([c for r,c,dd in cb],np.float32); DD=np.array([dd for r,c,dd in cb])
    for i,(r,c,dd) in enumerate(cb):
        row=pst[r:r+1].copy(); core.set_state(i,row)
    alive=np.ones(N,bool); best=np.full(N,-1e9); bk=np.zeros(N,int)
    acts=np.zeros((N,6),np.int32); acts[:,0]=7; acts[:,1]=3; view=np.zeros((N,2),np.float32)
    H=max(DS)+POST
    for k in range(H):
        ii=rr+k
        sv=np.asarray(core.states_view['velocity'],np.float64); hv=np.degrees(np.arctan2(sv[:,1],sv[:,0]))
        base=((yaw[ii+1]-hv+180)%360)-180
        view[:,0]=(base+np.where(k<DD,cc,0.0)).astype(np.float32)
        acts[:,2]=fwd[ii]; acts[:,3]=side[ii]; acts[:,4]=(btn[ii]&2)>0; acts[:,5]=(btn[ii]&4)>0
        _o,_r,done,trunc,_=core.step(acts,view=view)
        p=np.asarray(core.states_view['origin'],np.float64)
        alive&=~(np.asarray(done,bool)|np.asarray(trunc,bool))
        pr=100*(d0-f.sample(p).astype(np.float64))/d0
        u=alive&(pr>best); best[u]=pr[u]; bk[u]=k
    for i in np.argsort(-best)[:20]:
        best_all.append((float(best[i]),int(rr[i]),float(cc[i]),int(DD[i]),int(bk[i])))
    print(f"chunk {s0//CH}: best {best.max():.2f}% [{time.time()-t0:.0f}s]", flush=True)
best_all.sort(reverse=True)
# the policy's own best on the line (unperturbed), for reference
pp=np.asarray(pst['origin'],np.float64); prog=100*(d0-f.sample(pp).astype(np.float64))/d0
print(f"the line's own best: {prog.max():.2f}% at tick {int(np.argmax(prog))}")
for b in best_all[:15]:
    print(f"  {b[0]:.2f}%  root tick {b[1]} (t {b[1]*0.007:.1f}s)  c {b[2]:+.1f}  hold {b[3]*0.007:.2f}s  best at +{b[4]*0.007:.2f}s")
json.dump(best_all[:50], open(SP/'psearch_best.json','w'))
