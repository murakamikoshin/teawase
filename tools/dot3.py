"""ベタ塗りの元絵(高解像度) → ドット絵。マスごとに「いちばん多い色」を取るので、ぼけない。
黒い線は、となりの塗りを濃くした色に置きかえる(黒い枠は使わない。広い黒い面だけ黒のまま)。
使い方: python3 dot3.py 元絵.png 出力.png [高さ=72] [色数=16]"""
import sys
from collections import Counter, deque
from PIL import Image, ImageFilter, ImageEnhance
def lum(c): return 0.299*c[0]+0.587*c[1]+0.114*c[2]
def run(f, TH=72, K=16):
    im=Image.open(f).convert('RGBA'); im=Image.alpha_composite(Image.new('RGBA',im.size,(255,255,255,255)),im).convert('RGB')
    m=max(2,im.width//100); im=im.crop((m,m,im.width-m,im.height-m))      # ふちの線やにじみを落とす
    im.thumbnail((640,640),Image.LANCZOS); w,h=im.size; p=im.load()
    def white(c): return min(c)>236 and max(c)-min(c)<18
    bg=bytearray(w*h); q=deque()
    for x in range(w): q.extend(((x,0),(x,h-1)))
    for y in range(h): q.extend(((0,y),(w-1,y)))
    while q:
        x,y=q.popleft()
        if not(0<=x<w and 0<=y<h) or bg[y*w+x] or not white(p[x,y]): continue
        bg[y*w+x]=1; q.extend(((x+1,y),(x-1,y),(x,y+1),(x,y-1)))
    xs=[i%w for i in range(w*h) if not bg[i]]; ys=[i//w for i in range(w*h) if not bg[i]]
    x0,x1,y0,y1=min(xs),max(xs)+1,min(ys),max(ys)+1
    im2=ImageEnhance.Color(im).enhance(1.15).filter(ImageFilter.MedianFilter(5 if max(im.size)>=500 else 3))
    qz=im2.crop((x0,y0,x1,y1)).quantize(K,method=Image.MEDIANCUT,dither=Image.NONE)
    pal=qz.getpalette()[:K*3]; pal=[tuple(pal[i*3:i*3+3]) for i in range(K)]
    qp=qz.load(); cw,ch=x1-x0,y1-y0; s=ch/TH; W=max(1,round(cw/s))
    g=[[None]*W for _ in range(TH)]
    dark={i for i,c in enumerate(pal) if lum(c)<58}
    for Y in range(TH):
        for X in range(W):
            ax,bx=int(X*s),max(int(X*s)+1,int((X+1)*s)); ay,by=int(Y*s),max(int(Y*s)+1,int((Y+1)*s))
            c=Counter(); n=0; tot=0
            for yy in range(ay,min(by,ch)):
                for xx in range(ax,min(bx,cw)):
                    tot+=1
                    if not bg[(y0+yy)*w+x0+xx]: n+=1; c[qp[xx,yy]]+=1
            if tot and n*2>=tot:
                # 線(暗い色)より塗りを優先: 暗い色が過半数でなければ、塗りの中でいちばん多い色
                d=sum(v for k,v in c.items() if k in dark)
                if d*10<n*6:
                    c2=Counter({k:v for k,v in c.items() if k not in dark})
                    if c2: c=c2
                g[Y][X]=c.most_common(1)[0][0]
    def at(y,x): return g[y][x] if 0<=y<TH and 0<=x<W else None
    N4=((1,0),(-1,0),(0,1),(0,-1)); N8=N4+((1,1),(-1,-1),(1,-1),(-1,1))
    for _ in range(2):
        g2=[r[:] for r in g]
        for y in range(TH):
            for x in range(W):
                v=g[y][x]
                if v is None: continue
                nb=[at(y+dy,x+dx) for dx,dy in N4]
                if sum(1 for t in nb if t==v)==0:
                    if v in dark and any(at(y+dy,x+dx) in dark for dx,dy in N8): continue
                    c=Counter(t for t in nb if t is not None)
                    if c: g2[y][x]=c.most_common(1)[0][0]
        g=g2
    # ふちに残った白いにじみ(背景の白の残り)を消す
    lightc={i for i,c in enumerate(pal) if min(c)>222}
    for _ in range(2):
        kill=[]
        for y in range(TH):
            for x in range(W):
                v=g[y][x]
                if v is None or v not in lightc: continue
                if not any(at(y+dy,x+dx) is None for dx,dy in N4): continue
                if sum(1 for dx,dy in N8 if at(y+dy,x+dx) in lightc)<=2: kill.append((x,y))
        for x,y in kill: g[y][x]=None
    for y in range(TH):
        for x in range(W):
            if g[y][x] is not None and sum(1 for dx,dy in N4 if at(y+dy,x+dx) is not None)==0: g[y][x]=None
    out=Image.new('RGBA',(W,TH),(0,0,0,0)); op=out.load()
    for y in range(TH):
        for x in range(W):
            v=g[y][x]
            if v is None: continue
            col=pal[v]
            if v in dark:
                c=Counter(t for t in (at(y+dy,x+dx) for dx,dy in N8) if t is not None and t not in dark)
                nd=sum(1 for dx,dy in N8 if at(y+dy,x+dx) in dark)
                if c and nd<=5:
                    b=pal[c.most_common(1)[0][0]]; col=tuple(int(k*0.45) for k in b)
            op[x,y]=col+(255,)
    return out.crop(out.getbbox())
if __name__=='__main__':
    r=run(sys.argv[1], int(sys.argv[3]) if len(sys.argv)>3 else 72, int(sys.argv[4]) if len(sys.argv)>4 else 16)
    r.save(sys.argv[2]); print(sys.argv[2], r.size)
