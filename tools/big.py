import sys,os,glob,base64,json,io
from PIL import Image, ImageDraw, ImageFilter
src=sys.argv[1]; out=sys.argv[2]
ids=[15,10,11,12,18,13,14,19,20,31,32,33,1,2,3,4,5,16,6,7,8,9,17,101,102,103,104,105,116,106,107,108,109,117]
files=sorted(glob.glob(os.path.join(src,'mcp-Canva-blob-*.png')))
assert len(files)==len(ids)
art={}
for i,f in zip(ids,files):
    im=Image.open(f).convert('RGBA'); w,h=im.size
    canvas=Image.new('RGB',(w+2,h+2),(255,255,255)); canvas.paste(im.convert('RGB'),(1,1))
    ImageDraw.floodfill(canvas,(0,0),(255,0,255),thresh=42)
    px=canvas.load(); a=Image.new('L',(w,h),255); ap=a.load()
    for y in range(h):
        for x in range(w):
            if px[x+1,y+1]==(255,0,255): ap[x,y]=0
    a=a.filter(ImageFilter.GaussianBlur(0.6))
    im.putalpha(a)
    bb=a.point(lambda v:255 if v>40 else 0).getbbox(); im=im.crop(bb)
    s=max(im.size); sq=Image.new('RGBA',(s,s),(0,0,0,0)); sq.paste(im,((s-im.size[0])//2,s-im.size[1]))
    b=io.BytesIO(); sq.save(b,'WEBP',quality=84,method=6)
    art[i]='data:image/webp;base64,'+base64.b64encode(b.getvalue()).decode()
json.dump(art,open(os.path.join(out,'artl.json'),'w'))
print('total',sum(len(v) for v in art.values()))
