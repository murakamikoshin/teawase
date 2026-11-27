import sys,os,glob,base64,json,io
from PIL import Image, ImageDraw
src=sys.argv[1]; out=sys.argv[2]
ids=[15,10,11,12,18,13,14,19,20,31,32,33,1,2,3,4,5,16,6,7,8,9,17,101,102,103,104,105,116,106,107,108,109,117]
files=sorted(glob.glob(os.path.join(src,'mcp-Canva-blob-*.png')))
print(len(files),'files'); assert len(files)==len(ids)
N=48
art={}
for i,f in zip(ids,files):
    im=Image.open(f).convert('RGBA')
    if i==15: print('src size',im.size)
    w,h=im.size
    # 外周から白をぬりつぶして透明に
    rgb=im.convert('RGB')
    mask=Image.new('L',(w+2,h+2),0)
    canvas=Image.new('RGB',(w+2,h+2),(255,255,255)); canvas.paste(rgb,(1,1))
    ImageDraw.floodfill(canvas,(0,0),(255,0,255),thresh=42)
    px=canvas.load(); a=Image.new('L',(w,h),255); ap=a.load()
    for y in range(h):
        for x in range(w):
            if px[x+1,y+1]==(255,0,255): ap[x,y]=0
    im.putalpha(a)
    bb=a.getbbox(); im=im.crop(bb)
    s=max(im.size); sq=Image.new('RGBA',(s,s),(0,0,0,0)); sq.paste(im,((s-im.size[0])//2,s-im.size[1]))
    # premultiplied縮小
    small=sq.resize((N,N),Image.BOX)
    al=small.getchannel('A').point(lambda v:255 if v>=120 else 0)
    q=small.convert('RGB').quantize(28,method=Image.MEDIANCUT,dither=Image.NONE).convert('RGB')
    res=Image.new('RGBA',(N+2,N+2),(0,0,0,0)); res.paste(q,(1,1),al)
    rp=res.load()
    edge=[]
    for y in range(N+2):
        for x in range(N+2):
            if rp[x,y][3]==0:
                for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                    xx,yy=x+dx,y+dy
                    if 0<=xx<N+2 and 0<=yy<N+2 and rp[xx,yy][3]==255: edge.append((x,y)); break
    for x,y in edge: rp[x,y]=(27,31,51,255)
    res.save(os.path.join(out,'%d.png'%i),optimize=True)
    b=io.BytesIO(); res.save(b,'PNG',optimize=True)
    art[i]='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
json.dump(art,open(os.path.join(out,'art.json'),'w'))
print('total bytes',sum(len(v) for v in art.values()))
# contact sheet
sheet=Image.new('RGB',(9*104,4*104),(246,239,224))
for k,i in enumerate(ids):
    t=Image.open(os.path.join(out,'%d.png'%i)).resize((100,100),Image.NEAREST)
    sheet.paste(t,((k%9)*104+2,(k//9)*104+2),t)
sheet.save(os.path.join(out,'sheet.png'))
