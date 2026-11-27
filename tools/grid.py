import sys
from PIL import Image, ImageDraw
f,out,x0,y0,x1,y1,z=sys.argv[1],sys.argv[2],*map(int,sys.argv[3:8])
im=Image.open(f).convert('RGB').crop((x0,y0,x1,y1)); im=im.resize((im.width*z,im.height*z),Image.NEAREST)
d=ImageDraw.Draw(im)
for x in range(x0,x1+1):
    if x%5==0:
        d.line([((x-x0)*z,0),((x-x0)*z,im.height)],fill=(255,0,255) if x%10==0 else (0,200,255),width=1)
        if x%10==0: d.text(((x-x0)*z+2,2),str(x),fill=(255,0,255))
for y in range(y0,y1+1):
    if y%5==0:
        d.line([(0,(y-y0)*z),(im.width,(y-y0)*z)],fill=(255,0,255) if y%10==0 else (0,200,255),width=1)
        if y%10==0: d.text((2,(y-y0)*z+2),str(y),fill=(255,0,255))
im.save(out)
