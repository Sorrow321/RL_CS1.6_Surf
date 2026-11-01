#!/usr/bin/env python3
# Usage: ZL_CKPT=... ZL_REC_DIR=... python autoresearch/analysis/zl_perturb_spine.py ROOT_TICK C_DEG HOLD_TICKS OUT.npy
# zlr2b2: rebuild one perturbation-search line exactly and write it as a spine (STATE_DTYPE .npy):
# the policy's own greedy states [0, root) + the perturbed continuation up to its best d.
import sys, json
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
root=int(sys.argv[1]); c=float(sys.argv[2]); dd=int(sys.argv[3]); out=sys.argv[4]
core=B.build_sim(cfg, Path(os.environ.get('ZL_MAP', str(ROOT/'maps/surf_za_loop.bsp'))), 1, 200000, tick=TickClock(7.0))
core.set_teleport_fail(True); core.reset(0)
core.set_state(0, pst[root:root+1].copy())
seg=[core.get_states()[0].copy()]; prog=[]
for k in range(700):
    i=root+k
    sv=np.asarray(core.get_states()[0]['velocity'],np.float64); hv=np.degrees(np.arctan2(sv[1],sv[0]))
    off=((yaw[i+1]-hv+180)%360)-180 + (c if k<dd else 0.0)
    a=np.array([[7,3,fwd[i],side[i],1 if btn[i]&2 else 0,1 if btn[i]&4 else 0]],np.int32)
    _o,_r,done,trunc,_=core.step(a, view=np.array([[off,0.0]],np.float32))
    if bool(done[0]) or bool(trunc[0]): break
    s=core.get_states()[0].copy(); seg.append(s)
seg=np.array(seg)
pp=np.asarray(seg['origin'],np.float64); pr=100*(d0-f.sample(pp).astype(np.float64))/d0
kb=int(np.argmax(pr))
line=np.concatenate([pst[:root], seg[:kb+1]])
o=np.asarray(line['origin'],np.float64); v=np.asarray(line['velocity'],np.float64)
E=o[:,2]+(v**2).sum(1)/1600
lp=100*(d0-f.sample(o).astype(np.float64))/d0
# consistency: every consecutive pair is one physics tick apart in position (no teleport/jump)
step=np.linalg.norm(np.diff(o,axis=0),axis=1)
print(f"line {len(line)} states ({len(line)*0.007:.1f}s): end progress {lp[-1]:.2f}% (max {lp.max():.2f}%), end z {o[-1,2]:.0f}, E end {E[-1]:.0f}; max step {step.max():.1f}u; perturbed segment {kb+1} ticks")
np.save(out, line)
print('saved', out)
