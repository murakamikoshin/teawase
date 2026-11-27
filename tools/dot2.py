# イラスト → ドット絵。平均でぼかさず「いちばん多い色」で縮めるので、モザイクにならずベタ面になる
import sys,os,glob
from collections import Counter
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
src=sys.argv[1]; work=sys.argv[2]; TH=int(sys.argv[3]); K=int(sys.argv[4])
ids=[15,10,11,12,18,13,14,19,20,31,32,33,1,2,3,4,5,16,6,7,8,9,17,101,102,103,104,105,116,106,107,108,109,117]
files=sorted(glob.glob(os.path.join(src,'mcp-Canva-blob-*.png')))
def lum(c): return 0.299*c[0]+0.587*c[1]+0.114*c[2]
def run(f):
    im=Image.open(f).convert('RGB'); w,h=im.size
    cv=Image.new('RGB',(w+2,h+2),(255,255,255)); cv.paste(im,(1,1))
    ImageDraw.floodfill(cv,(0,0),(255,0,255),thresh=40)
    cp=cv.load()
    fg=[[cp[x+1,y+1]!=(255,0,255) for x in range(w)] for y in range(h)]
    xs=[x for y in range(h) for x in range(w) if fg[y][x]]; ys=[y for y in range(h) for x in range(w) if fg[y][x]]
    x0,x1,y0,y1=min(xs),max(xs)+1,min(ys),max(ys)+1
    im=ImageEnhance.Color(im).enhance(1.2).filter(ImageFilter.MedianFilter(3))
    q=im.crop((x0,y0,x1,y1)).quantize(K,method=Image.MEDIANCUT,dither=Image.NONE)
    pal=q.getpalette()[:K*3]; pal=[tuple(pal[i*3:i*3+3]) for i in range(K)]
    qp=q.load(); cw,ch=x1-x0,y1-y0
    s=ch/TH; W=max(1,round(cw/s))
    g=[[None]*W for _ in range(TH)]
    for Y in range(TH):
        for X in range(W):
            ax,bx=int(X*s),max(int(X*s)+1,int((X+1)*s)); ay,by=int(Y*s),max(int(Y*s)+1,int((Y+1)*s))
            c=Counter(); n=0; tot=0
            for yy in range(ay,min(by,ch)):
                for xx in range(ax,min(bx,cw)):
                    tot+=1
                    if fg[y0+yy][x0+xx]: n+=1; c[qp[xx,yy]]+=1
            if tot and n*2>=tot: g[Y][X]=c.most_common(1)[0][0]
    def at(y,x): return g[y][x] if 0<=y<TH and 0<=x<W else None
    N4=((1,0),(-1,0),(0,1),(0,-1)); N8=N4+((1,1),(-1,-1),(1,-1),(-1,1))
    dark={i for i,c in enumerate(pal) if lum(c)<62}
    # 散った1ドットを まわりの色にそろえる(暗い点は目かもしれないので、となりに暗い点があれば残す)
    for _ in range(2):
        g2=[r[:] for r in g]
        for y in range(TH):
            for x in range(W):
                v=g[y][x]
                if v is None: continue
                nb=[at(y+dy,x+dx) for dx,dy in N4]; same=sum(1 for t in nb if t==v)
                if same==0:
                    if v in dark and any(at(y+dy,x+dx) in dark for dx,dy in N8): continue
                    c=Counter(t for t in nb if t is not None)
                    if c: g2[y][x]=c.most_common(1)[0][0]
        g=g2
    # 1ドットだけ飛び出た角・ちぎれた点を消す
    for y in range(TH):
        for x in range(W):
            if g[y][x] is not None and sum(1 for dx,dy in N4 if at(y+dy,x+dx) is not None)==0: g[y][x]=None
    # 色: 黒い線は「となりの色を濃くした色」に。広い黒い面はそのまま
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
                    b=pal[c.most_common(1)[0][0]]; col=tuple(int(k*0.42) for k in b)
            op[x,y]=col+(255,)
    return out.crop(out.getbbox())
sizes={}
for i,f in zip(ids,files):
    r=run(f); os.makedirs(os.path.join(work,'m%d'%i),exist_ok=True); r.save(os.path.join(work,'m%d'%i,'dot2.png')); sizes[i]=r.size
print(sizes)
for n,part in (('a',ids[:17]),('b',ids[17:])):
    sheet=Image.new('RGB',(6*240,3*240),(20,16,54))
    for k,i in enumerate(part):
        t=Image.open(os.path.join(work,'m%d'%i,'dot2.png')); t=t.resize((t.width*3,t.height*3),Image.NEAREST)
        sheet.paste(t,((k%6)*240+(240-t.width)//2,(k//6)*240+(240-t.height)//2),t)
    sheet.save(os.path.join(work,'dot2_%s.png'%n))
