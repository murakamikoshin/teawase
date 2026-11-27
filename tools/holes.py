"""キャラの中に とじこめられた白(腕と武器のあいだ等)を 背景として抜く。
 python3 holes.py show 出力.png id,id,...   → 候補に番号をふった画像
 python3 holes.py punch id 番号,番号,...     → その番号の白を透明にする(元の絵に 透明として書きこむ)"""
import sys,os,importlib.util
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
def src(i):
    if i>=8300: return '/home/claude/anim/83_f%d.png'%(i-8300)
    c=['/home/claude/hires/%d.png'%i] if i!=18 else []
    if i in (1,2,3,4): c.append('/home/claude/flat2/%d.png'%i)
    c+=['/home/claude/newsrc/%d.png'%i,'/home/claude/lowres/%d.png'%i]
    for f in c:
        if os.path.exists(f): return f
def cands(f):
    im=Image.open(f).convert('RGBA'); a=np.asarray(im).astype(int)
    w=((a[:,:,:3].min(axis=2)>150)&((a[:,:,:3].max(axis=2)-a[:,:,:3].min(axis=2))<30))|(a[:,:,3]<128)
    lab,n=ndimage.label(w); bg=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
    ink=~np.isin(lab,list(bg)); ys,xs=np.nonzero(ink); area=(ys.max()-ys.min())*(xs.max()-xs.min())
    sz=ndimage.sum(w,lab,range(1,n+1)); out=[]
    for k in np.argsort(sz)[::-1]:
        if (k+1) in bg or sz[k]<area*0.0012: continue
        if (a[:,:,3][lab==k+1]<128).mean()>0.5: continue      # もう抜いてある
        out.append(k+1)
    return im,lab,out
if sys.argv[1]=="show":
    ids=[int(x) for x in sys.argv[3].split(',')]; C=5; W=400; R=(len(ids)+C-1)//C
    sheet=Image.new('RGB',(C*W,R*W),(90,200,120)); 
    for q,i in enumerate(ids):
        im,lab,cs=cands(src(i)); a=np.asarray(im.convert('RGB')).copy()
        bgl=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}; a[np.isin(lab,list(bgl))]=(90,200,120)
        for c in cs: a[lab==c]=(255,120,255)
        t=Image.fromarray(a); sc=min((W-10)/t.width,(W-10)/t.height); t=t.resize((int(t.width*sc),int(t.height*sc)))
        ox=(q%C)*W+5; oy=(q//C)*W+5; sheet.paste(t,(ox,oy)); d=ImageDraw.Draw(sheet); d.text((ox+2,oy+2),"#%d"%i,fill=(0,0,0))
        for n,c in enumerate(cs,1):
            ys,xs=np.nonzero(lab==c); k=len(ys)//2; d.text((ox+xs[k]*sc-3,oy+ys[k]*sc-5),str(n),fill=(0,0,160))
    sheet.save(sys.argv[2])
else:
    i=int(sys.argv[2]); f=src(i); im,lab,cs=cands(f); a=np.asarray(im).copy()
    for n in [int(x) for x in sys.argv[3].split(',')]:
        m=ndimage.binary_dilation(lab==cs[n-1],iterations=2); a[m,3]=0
    out='/home/claude/newsrc/%d.png'%i
    if i>=8300: out=f
    if f!=out and os.path.exists(f) and '/newsrc/' not in f:
        os.makedirs('/home/claude/oldart',exist_ok=True); os.replace(f,'/home/claude/oldart/pre_hole_'+os.path.basename(os.path.dirname(f))+'_%d.png'%i)
    Image.fromarray(a).save(out); print('punched',i,sys.argv[3])
