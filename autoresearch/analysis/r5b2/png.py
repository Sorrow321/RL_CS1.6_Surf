import zlib, struct, numpy as np
def write_png(path, rgb):
    rgb=np.ascontiguousarray(rgb.astype(np.uint8)); h,w,_=rgb.shape
    raw=b''.join(b'\x00'+rgb[y].tobytes() for y in range(h))
    def chunk(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b'')
    open(path,'wb').write(png)
def colormap(v, lo, hi):
    """grey-blue-green-yellow ramp; nan -> black"""
    t=np.clip((v-lo)/(hi-lo),0,1)
    r=np.interp(t,[0,0.5,1],[40,30,250]); g=np.interp(t,[0,0.5,1],[40,160,230]); b=np.interp(t,[0,0.5,1],[140,90,30])
    out=np.stack([r,g,b],-1)
    out[np.isnan(v)]=0
    return out
# 5x7 digit font
FONT={'0':["111","101","101","101","111"],'1':["010","110","010","010","111"],'2':["111","001","111","100","111"],
'3':["111","001","111","001","111"],'4':["101","101","111","001","001"],'5':["111","100","111","001","111"],
'6':["111","100","111","101","111"],'7':["111","001","001","001","001"],'8':["111","101","111","101","111"],
'9':["111","101","111","001","111"],'.':["000","000","000","000","010"],'-':["000","000","111","000","000"],' ':["000"]*5}
def text(img, x, y, s, col, scale=2):
    h,w,_=img.shape
    for ch in s:
        g=FONT.get(ch,FONT[' '])
        for r,row in enumerate(g):
            for cc,bit in enumerate(row):
                if bit=='1':
                    for a in range(scale):
                        for b in range(scale):
                            yy=y+r*scale+a; xx=x+cc*scale+b
                            if 0<=yy<h and 0<=xx<w: img[yy,xx]=col
        x+=4*scale
