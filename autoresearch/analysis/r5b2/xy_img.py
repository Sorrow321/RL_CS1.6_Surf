import sys, numpy as np
SP=str(__import__('pathlib').Path(__file__).resolve().parent)
sys.path.insert(0,SP); sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[2])); sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python'))
import score
from png import write_png, colormap, text
from surfgym.zones import kill_zones, hull_probe
BSP=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'maps' / 'surf_src_sidistic.bsp')
F=score.field(); c=F.cell
out=sys.argv[1]; x0,x1,y0,y1=[float(v) for v in sys.argv[2:6]]
zl=[float(v) for v in sys.argv[6].split(',')]
res=float(__import__("os").environ.get("RES","16"))
ks=kill_zones(BSP); hp=hull_probe(BSP)
xs=np.arange(x0,x1,res); ys=np.arange(y0,y1,res)
X,Y=np.meshgrid(xs,ys)
panels=[]
for z in zl:
    P=np.stack([X.ravel(),Y.ravel(),np.full(X.size,z)],1)
    d=F.sample(P).reshape(X.shape)
    pr=np.where(d<F.reach_max-0.5*c,100*(97768-d)/97768,np.nan)
    kill=np.zeros(X.size,bool)
    for k in ks:
        lo=np.asarray(k['mins'])+k['origin']; hi=np.asarray(k['maxs'])+k['origin']
        if lo[2]-36>z or hi[2]+36<z: continue
        m=(P[:,0]>=lo[0]-16)&(P[:,0]<=hi[0]+16)&(P[:,1]>=lo[1]-16)&(P[:,1]<=hi[1]+16)
        if m.any():
            idx=np.where(m)[0]; inside=hp(int(k['model'][1:]),P[idx]); kill[idx[inside]]=True
    img=colormap(pr,float(sys.argv[7]) if len(sys.argv)>7 else 85,100)
    img[kill.reshape(X.shape)]=[230,30,30]
    img=img[::-1].copy()
    H,W,_=img.shape
    text(img,4,4,f"z {int(z)}",(255,255,255),2)
    for x in np.arange(np.ceil(x0/500)*500,x1,500):
        px=int((x-x0)/res); img[H-6:H,px]=255; text(img,px+1,H-16,f"{int(x)}",(255,255,255),1)
    for y in np.arange(np.ceil(y0/500)*500,y1,500):
        py=H-1-int((y-y0)/res); img[py,0:6]=255; text(img,8,py-3,f"{int(y)}",(255,255,255),1)
    panels.append(img); panels.append(np.full((H,4,3),255,np.uint8))
write_png(out,np.concatenate(panels[:-1],1))
print('ok')
