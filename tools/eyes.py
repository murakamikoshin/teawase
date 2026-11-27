from collections import Counter
from PIL import Image
def lum(c): return 0.299*c[0]+0.587*c[1]+0.114*c[2]
def dist(a,b): return sum((a[i]-b[i])**2 for i in range(3))**0.5
def sat(c): return max(c[:3])-min(c[:3])
def eye_pixels(P,box,w,h):
    x0,y0,x1,y1=box
    ring=Counter()
    for y in range(max(0,y0-3),min(h,y1+4)):
        for x in range(max(0,x0-3),min(w,x1+4)):
            if (x<x0 or x>x1 or y<y0 or y>y1) and P[x,y][3]: ring[P[x,y][:3]]+=1
    base=ring.most_common(1)[0][0]
    px=[(x,y) for y in range(y0,y1+1) for x in range(x0,x1+1) if P[x,y][3] and dist(P[x,y][:3],base)>55]
    return base,px
def no_highlight(im,boxes):
    im=im.copy(); P=im.load(); w,h=im.size
    for box in boxes:
        base,px=eye_pixels(P,box,w,h); S=set(px)
        for _ in range(3):
            ch=[]
            for (x,y) in px:
                c=P[x,y][:3]
                if min(c)>215:
                    nb=[P[x+dx,y+dy][:3] for dx in(-1,0,1) for dy in(-1,0,1) if (dx or dy) and (x+dx,y+dy) in S and min(P[x+dx,y+dy][:3])<=215]
                    if len(nb)>=4: ch.append((x,y,Counter(nb).most_common(1)[0][0]))
            for x,y,c in ch: P[x,y]=c+(255,)
            if not ch: break
    return im
def simple(im,boxes,hero):
    im=im.copy(); P=im.load(); w,h=im.size
    for k,box in enumerate(boxes):
        base,px=eye_pixels(P,box,w,h)
        if not px: continue
        cols=[P[x,y][:3] for x,y in px]
        darkest=min(cols,key=lum)
        iris=max(cols,key=lambda c:(sat(c) if lum(c)<200 else -1))
        xs=[x for x,y in px]; ys=[y for x,y in px]
        cx=round(sum(xs)/len(xs)); top=min(ys); bot=max(ys)
        for x,y in px: P[x,y]=base+(255,)
        if hero:
            ey=round((top+bot)/2)-3
            lid=tuple(int(v*0.55) for v in darkest) if lum(darkest)>60 else darkest
            ird=tuple(int(v*0.6) for v in iris)
            for i in range(-1,2):
                P[cx+i,ey]=lid+(255,)
                P[cx+i,ey+1]=ird+(255,); P[cx+i,ey+2]=ird+(255,)
                for j in range(3,6): P[cx+i,ey+j]=iris+(255,)
            P[cx+(-2 if k==0 else 2),ey]=lid+(255,)
        else:
            ew=max(3,round((max(xs)-min(xs)+1)*0.42)); eh=max(4,round((bot-top+1)*0.6)); ey=round((top+bot)/2-eh/2)
            for j in range(eh):
                for i in range(ew):
                    if (j in(0,eh-1)) and (i in(0,ew-1)) and ew>3: continue
                    P[cx-ew//2+i,ey+j]=darkest+(255,)
    return im
