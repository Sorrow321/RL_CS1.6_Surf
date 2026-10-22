import sys, os, numpy as np
SP=str(__import__('pathlib').Path(__file__).resolve().parent)
sys.path.insert(0,SP); sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[2])); sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python'))
import score
from png import write_png, colormap, text
from surfgym.zones import kill_zones, hull_probe
BSP=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'maps' / 'surf_src_sidistic.bsp')
F=score.field(); c=F.cell
out=sys.argv[1]; y0,y1,z0,z1=[float(v) for v in sys.argv[2:6]]
xl=[float(v) for v in sys.argv[6].split(',')]
res=float(os.environ.get('RES','8'))
ks=kill_zones(BSP); hp=hull_probe(BSP)
ys=np.arange(y0,y1,res); zs=np.arange(z0,z1,res)
Y,Z=np.meshgrid(ys,zs)
panels=[]
for x in xl:
    P=np.stack([np.full(Y.size,x),Y.ravel(),Z.ravel()],1)
    d=F.sample(P).reshape(Y.shape)
    pr=np.where(d<F.reach_max-0.5*c,100*(97768-d)/97768,np.nan)
    kill=np.zeros(Y.size,bool)
    for k in ks:
        lo=np.asarray(k['mins'])+k['origin']; hi=np.asarray(k['maxs'])+k['origin']
        if lo[0]-16>x or hi[0]+16<x: continue
        m=(P[:,1]>=lo[1]-16)&(P[:,1]<=hi[1]+16)&(P[:,2]>=lo[2]-36)&(P[:,2]<=hi[2]+36)
        if m.any():
            idx=np.where(m)[0]; inside=hp(int(k['model'][1:]),P[idx]); kill[idx[inside]]=True
    img=colormap(pr,float(sys.argv[7]) if len(sys.argv)>7 else 85,100)
    img[kill.reshape(Y.shape)]=[230,30,30]
    img=img[::-1].copy(); H,W,_=img.shape
    text(img,4,4,f"x {int(x)}",(255,255,255),2)
    for y in np.arange(np.ceil(y0/250)*250,y1,250):
        px=int((y-y0)/res); img[H-6:H,px]=255; text(img,px+1,H-16,f"{int(y)}",(255,255,255),1)
    for z in np.arange(np.ceil(z0/100)*100,z1,100):
        pz=H-1-int((z-z0)/res); img[pz,0:6]=255; text(img,8,pz-3,f"{int(z)}",(255,255,255),1)
    panels.append(img); panels.append(np.full((H,4,3),255,np.uint8))
write_png(out,np.concatenate(panels[:-1],1)); print('ok')
