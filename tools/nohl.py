"""目の中の白い点(キラキラ)を消す。白い小さなかたまりのうち、まわりが瞳の色(暗い色か濃い色)で囲まれているものだけ、まわりの色で埋める"""
import sys
from collections import Counter
from PIL import Image
def lum(c): return 0.299*c[0]+0.587*c[1]+0.114*c[2]
def run(src,out):
    im=Image.open(src).convert('RGBA'); P=im.load(); w,h=im.size
    seen=set(); n=0
    for y0 in range(int(h*0.66)):
        for x0 in range(w):
            if (x0,y0) in seen or not P[x0,y0][3] or min(P[x0,y0][:3])<=212: continue
            comp=[]; st=[(x0,y0)]; seen.add((x0,y0))
            while st:
                x,y=st.pop(); comp.append((x,y))
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    u,v=x+dx,y+dy
                    if 0<=u<w and 0<=v<h and (u,v) not in seen and P[u,v][3] and min(P[u,v][:3])>212:
                        seen.add((u,v)); st.append((u,v))
            if len(comp)>34: continue
            S=set(comp); ring=[]
            for x,y in comp:
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        u,v=x+dx,y+dy
                        if (u,v) in S: continue
                        if not (0<=u<w and 0<=v<h) or not P[u,v][3]: ring.append(None)
                        else: ring.append(P[u,v][:3])
            if not ring or any(r is None for r in ring): continue
            eyeish=sum(1 for c in ring if lum(c)<112 or (max(c)-min(c)>110 and lum(c)<170))
            if eyeish/len(ring)<0.72: continue
            fill=Counter(c for c in ring if lum(c)<170).most_common(1)[0][0]
            for x,y in comp: P[x,y]=fill+(255,)
            n+=1
    im.save(out); return n
if __name__=='__main__': print(sys.argv[1], run(sys.argv[1],sys.argv[2]))
