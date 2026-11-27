# ボスのドット化(全工程)の結果に、消えた目・口などの暗い細部だけを 1段目(pixel4)の結果から戻す
import sys,os
from PIL import Image
work=sys.argv[1]
ids=[15,10,11,12,18,13,14,19,20,31,32,33,1,2,3,4,5,16,6,7,8,9,17,101,102,103,104,105,116,106,107,108,109,117]
def lum(c): return 0.299*c[0]+0.587*c[1]+0.114*c[2]
sizes={}
for i in ids:
    st=Image.open(os.path.join(work,'m%d'%i,'source','m%d_still.png'%i)).convert('RGBA')
    p4=Image.open(os.path.join(work,'m%d'%i,'t_p.png')).convert('RGBA')
    if st.size!=p4.size:
        print('size mismatch',i,st.size,p4.size); p4=p4.resize(st.size,Image.NEAREST)
    w,h=st.size; S=st.load(); P=p4.load()
    dark=set()
    for y in range(h):
        for x in range(w):
            s=S[x,y]; p=P[x,y]
            if not s[3] or not p[3]: continue
            edge=any(not(0<=x+dx<w and 0<=y+dy<h) or S[x+dx,y+dy][3]==0 for dx,dy in((1,0),(-1,0),(0,1),(0,-1)))
            if edge: continue
            if lum(p)<105 and lum(p)<lum(s)-60: dark.add((x,y))
    # 1ドットだけの点は戻さない(ノイズ)。となりにもう1つ以上あるものだけ
    keep={(x,y) for (x,y) in dark if any((x+dx,y+dy) in dark for dx in(-1,0,1) for dy in(-1,0,1) if dx or dy)}
    # 目の中の白い点(ハイライト)は埋める
    fill=set()
    for y in range(1,h-1):
        for x in range(1,w-1):
            if (x,y) in keep or not S[x,y][3]: continue
            n=sum(((x+dx,y+dy) in keep) for dx,dy in((1,0),(-1,0),(0,1),(0,-1)))
            if n>=3: fill.add((x,y))
    for (x,y) in keep: S[x,y]=P[x,y]
    for (x,y) in fill:
        for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
            if (x+dx,y+dy) in keep: S[x,y]=P[x+dx,y+dy]; break
    bb=st.getbbox(); st=st.crop(bb)
    st.save(os.path.join(work,'m%d'%i,'final.png')); sizes[i]=st.size
print(sizes)
for n,part in (('a',ids[:17]),('b',ids[17:])):
    sheet=Image.new('RGB',(6*240,3*240),(20,16,54))
    for k,i in enumerate(part):
        t=Image.open(os.path.join(work,'m%d'%i,'final.png')); t=t.resize((t.width*3,t.height*3),Image.NEAREST)
        sheet.paste(t,((k%6)*240+(240-t.width)//2,(k//6)*240+(240-t.height)//2),t)
    sheet.save(os.path.join(work,'final_%s.png'%n))
