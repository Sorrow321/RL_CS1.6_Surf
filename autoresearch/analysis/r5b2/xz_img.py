import sys, numpy as np
SP=str(__import__('pathlib').Path(__file__).resolve().parent)
sys.path.insert(0,SP); sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[2])); sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python'))
import score
from png import write_png, colormap, text
from surfgym.zones import kill_zones, hull_probe
BSP=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'maps' / 'surf_src_sidistic.bsp')
F=score.field(); G=F.grid; mins=np.asarray(F.mins); c=F.cell
x0,x1,z0,z1=[float(v) for v in sys.argv[2:6]]
ylo,yhi=float(sys.argv[6]),float(sys.argv[7])
traj=sys.argv[8]
eplist=[int(v) for v in sys.argv[9].split(',')] if len(sys.argv)>9 else [8,6]
tmin=float(sys.argv[10]) if len(sys.argv)>10 else 80.0
res=float(__import__("os").environ.get("RES","16"))
xs=np.arange(x0,x1,res); zs=np.arange(z0,z1,res)
X,Z=np.meshgrid(xs,zs)
# progress: best over y samples in [ylo,yhi]
best=np.full(X.shape,np.nan)
kill=np.zeros(X.shape,bool)
ks=kill_zones(BSP); hp=hull_probe(BSP)
for y in np.arange(ylo,yhi+1,32.0):
    P=np.stack([X.ravel(),np.full(X.size,y),Z.ravel()],1)
    d=F.sample(P).reshape(X.shape)
    pr=np.where(d<F.reach_max-0.5*c,100*(97768-d)/97768,np.nan)
    best=np.fmax(best,pr)
    for k in ks:
        lo=np.asarray(k['mins'])+k['origin']; hi=np.asarray(k['maxs'])+k['origin']
        if hi[0]+16<x0 or lo[0]-16>x1 or hi[2]+36<z0 or lo[2]-36>z1 or hi[1]+16<y or lo[1]-16>y: continue
        m=(P[:,0]>=lo[0]-16)&(P[:,0]<=hi[0]+16)&(P[:,2]>=lo[2]-36)&(P[:,2]<=hi[2]+36)
        if m.any():
            idx=np.where(m)[0]
            inside=hp(int(k['model'][1:]),P[idx])
            kk=kill.ravel(); kk[idx[inside]]=True; kill=kk.reshape(X.shape)
img=colormap(best,float(sys.argv[11]) if len(sys.argv)>11 else 80,100)
img[kill]=[230,30,30]
img=img[::-1]  # z up
H,W,_=img.shape
def to_px(x,z): return int((x-x0)/res), H-1-int((z-z0)/res)
eps=score.episodes(traj)
cols=[(255,255,255),(255,0,255),(0,255,255),(255,160,0),(160,255,0),(0,128,255),(255,255,0),(128,0,255),(255,128,128)]
for n,e in enumerate(eplist):
    r=eps[e][1]; col=cols[n%len(cols)]
    for t in range(int(tmin*100),len(r)):
        y=r[t,2]
        px,pz=to_px(r[t,1],r[t,3])
        if 0<=px<W and 0<=pz<H: img[pz,px]=col
    for t in range(int(tmin*100),len(r),100):
        px,pz=to_px(r[t,1],r[t,3])
        if 0<=px<W and 0<=pz<H: text(img,px+2,pz-12,f"{t/100:.0f}",col,1)
# axis ticks every 500 u
for x in np.arange(np.ceil(x0/500)*500,x1,500):
    px,_=to_px(x,z0); img[H-6:H,px]=255; text(img,px+1,H-16,f"{int(x)}",(255,255,255),1)
for z in np.arange(np.ceil(z0/500)*500,z1,500):
    _,pz=to_px(x0,z); img[pz,0:6]=255; text(img,8,pz-3,f"{int(z)}",(255,255,255),1)
# finish box
for x in np.arange(2872,3048,res):
    px,pz=to_px(x,180)
    if 0<=px<W and 0<=pz<H: img[max(0,pz-2):pz+2,px]=[255,255,255]
write_png(sys.argv[1],img)
print('wrote',sys.argv[1],img.shape)
