import re,base64,io,sys
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.colors import hsv_to_rgb
s=open('/home/claude/teawase/index.html').read()
A=s[s.find('var ARTH='):]
KMON=set(int(x) for x in re.findall(r'(\d+):1',re.search(r'var KMON=\{[^}]*\}',s).group(0)))
ids=[int(x) for x in re.findall(r'[{,](\d+):\[\d+,\d+,"data:image/webp',A[:A.find('};')+2] if False else A)]
ids=sorted(set(ids))
def art(i):
    m=re.search(r"[{,]%d:\[\d+,\d+,\"(data:image/[^\"]+)\""%i,A)
    im=Image.open(io.BytesIO(base64.b64decode(m.group(1).split(',')[1]))); im.seek(0); return im.convert('RGBA')
def ss(x,a,b): return np.clip((x-a)/(b-a),0,1)
def make(im,i):
    p=np.asarray(im).astype(float)/255; r,g,b,al=p[...,0],p[...,1],p[...,2],p[...,3]
    mx=p[...,:3].max(-1); mn=p[...,:3].min(-1); dl=mx-mn; d=np.where(dl==0,1,dl)
    h=np.where(mx==r,((g-b)/d)%6,np.where(mx==g,(b-r)/d+2,(r-g)/d+4))/6; sat=dl/np.where(mx==0,1,mx)
    kw=ss(dl,0.10,0.22)*ss(mx,0.22,0.32)*(al>0.03)
    band=ss(h,0.015,0.035)*(1-ss(h,0.165,0.185))
    cool=(kw*(1-band)).sum(); tot=kw.sum()+1e-6
    if i in KMON: w=kw; mode='all'
    elif cool/tot>0.15: w=kw*(1-band); mode='cool'
    else:
        vlo=0.36 if i in (234,241) else 0.50
        prot=ss(h,0.012,0.03)*(1-ss(h,0.115,0.135))*ss(mx,vlo,vlo+0.10)*(1-ss(sat,0.74,0.86)); w=kw*(1-prot); mode='pale'
    sh=(70+(i*47)%220)/360
    rgb=hsv_to_rgb(np.stack([(h+sh)%1,np.minimum(1,sat*1.12),mx],-1)); out=p.copy()
    out[...,:3]=p[...,:3]*(1-w[...,None])+rgb*w[...,None]
    return Image.fromarray((out*255).astype('uint8')),mode,cool/tot
if __name__=="__main__":
    ids=[i for i in ids if i not in KMON and make(art(i),i)[1]=='pale']; print(ids)
    per=24; T=150
    for pg in range((len(ids)+per-1)//per):
        sub=ids[pg*per:(pg+1)*per]; c=Image.new('RGB',(8*T*2//2*1+0,3*T*1),(40,40,50)); c=Image.new('RGB',(8*T,3*(T//2+T//2)),(40,40,50)); c=Image.new('RGB',(12*T,2*T*2//2+T),(40,40,50))
        c=Image.new('RGB',(12*T,4*T//2*1+2*T),(40,40,50)); c=Image.new('RGB',(12*T,4*T),(40,40,50)); d=ImageDraw.Draw(c)
        for k,i in enumerate(sub):
            im=art(i); k2,mode,fr=make(im,i); col=k%6; row=k//6
            for j,x in enumerate((im,k2)):
                x=x.copy(); x.thumbnail((T,T)); c.paste(x,(col*2*T+j*T,row*T),x)
            d.text((col*2*T+2,row*T+2),"%d %s %.2f"%(i,mode,fr),fill=(255,255,0))
        c.save('/tmp/kk/sheet%d.png'%pg)

