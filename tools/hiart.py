"""元の絵(イラスト)を 背景ぬきで 高さ300にそろえ、ゲームの ARTH に入れる。使い方: python3 hiart.py"""
import re,base64,io,os
from collections import deque
from PIL import Image, ImageFilter
import numpy as np
F='/home/claude/teawase/index.html'
s=open(F).read()
ids=[int(x) for x in re.findall(r'(\d+):\[\d+,\d+,"data:image/png',s[s.index('var ARTL={'):s.index('};',s.index('var ARTL={'))])]
def src(i):
    c=['/home/claude/hires/%d.png'%i] if i!=18 else []
    if i in (1,2,3,4): c.append('/home/claude/flat2/%d.png'%i)
    c+=['/home/claude/newsrc/%d.png'%i,'/home/claude/lowres/%d.png'%i]
    for f in c:
        if os.path.exists(f): return f
def cut(f,H=300,raw=False):
    im=Image.open(f).convert('RGBA'); srcA=im.getchannel('A'); bg=Image.new('RGBA',im.size,(255,255,255,255)); bg.alpha_composite(im); im=bg.convert('RGB')
    m=max(2,im.width//100); im=im.crop((m,m,im.width-m,im.height-m)); srcA=srcA.crop((m,m,srcA.width-m,srcA.height-m))
    if max(im.size)>900: im.thumbnail((900,900),Image.LANCZOS)
    srcA=srcA.resize(im.size,Image.LANCZOS)
    a=np.asarray(im).astype(int); h,w=a.shape[:2]
    white=(a.min(axis=2)>150)&((a.max(axis=2)-a.min(axis=2))<30)
    bgm=np.zeros((h,w),bool); q=deque()
    for x in range(w): q.extend(((x,0),(x,h-1)))
    for y in range(h): q.extend(((0,y),(w-1,y)))
    while q:
        x,y=q.popleft()
        if not(0<=x<w and 0<=y<h) or bgm[y,x] or not white[y,x]: continue
        bgm[y,x]=True; q.extend(((x+1,y),(x-1,y),(x,y+1),(x,y-1)))
    bgm|=(np.asarray(srcA)<128)      # holes.py で抜いた所(腕と武器のあいだ等)
    alpha=Image.fromarray(((~bgm)*255).astype('uint8')).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.7))
    out=im.convert('RGBA'); out.putalpha(alpha)
    if raw: return out
    out=out.crop(out.getbbox())
    W=max(1,round(out.width*H/out.height)); out=out.resize((W,H),Image.LANCZOS)
    b=io.BytesIO(); out.save(b,'WEBP',quality=86,method=6); return W,H,b.getvalue()
def cutAnim(g,H=300):
    """GIF の各コマを 同じ枠で切り抜いて、動くWebPにする(高レアのアニメ用)"""
    im=Image.open(g); fr=[]; dur=im.info.get('duration',200) or 200
    for k in range(im.n_frames):
        im.seek(k); t=g[:-4]+'_f%d.png'%k
        if not os.path.exists(t): t='/tmp/_fr%d.png'%k; im.convert('RGB').save(t)
        fr.append(cut(t,H,True))
    box=None
    for f in fr:
        b=f.getbbox(); box=b if box is None else (min(box[0],b[0]),min(box[1],b[1]),max(box[2],b[2]),max(box[3],b[3]))
    fr=[f.crop(box) for f in fr]; W=max(1,round(fr[0].width*H/fr[0].height)); fr=[f.resize((W,H),Image.LANCZOS) for f in fr]
    b=io.BytesIO(); fr[0].save(b,'WEBP',save_all=True,append_images=fr[1:],duration=dur,loop=0,quality=82,method=6); return W,H,b.getvalue()
ents=[]; tot=0; miss=[]
for i in ids:
    f=src(i)
    if not f: miss.append(i); continue
    g='/home/claude/anim/%d.gif'%i
    W,H,data=cutAnim(g) if os.path.exists(g) else cut(f,480); tot+=len(data)
    ents.append('%d:[%d,%d,"data:image/webp;base64,%s"]'%(i,W,H,base64.b64encode(data).decode()))
line='var ARTH={'+','.join(ents)+'};'
if 'var ARTH={' in s:
    a=s.index('var ARTH={'); e=s.index('};',a)+2; s=s[:a]+line+s[e:]
else:
    a=s.index('var ARTL={'); s=s[:a]+line+'\n'+s[a:]
# タイトルで大きく出す子は さらに高画質で(ARTT)
tents=[]
for i in (10,19,31,68,72,15,11,14,20):
    f=src(i)
    if f:
        W,H,data=cut(f,820); tot+=len(data); tents.append('%d:"data:image/webp;base64,%s"'%(i,base64.b64encode(data).decode()))
tline='var ARTT={'+','.join(tents)+'};'
if 'var ARTT={' in s:
    a=s.index('var ARTT={'); e=s.index('};',a)+2; s=s[:a]+tline+s[e:]
else:
    a=s.index('var ARTL={'); s=s[:a]+tline+'\n'+s[a:]
open(F,'w').write(s); print('hi-res art',len(ents),'missing',miss,'bytes',tot)
