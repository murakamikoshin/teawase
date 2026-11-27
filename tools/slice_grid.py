"""透過 PNG の 表を 等分の マス目で 切る(絵が 混みあって いて かたまり検出が 失敗する 時用)。
使い方: python3 slice_grid.py 表.png 列 行 出力フォルダ id1,id2,..."""
import sys
from PIL import Image
import numpy as np
from scipy import ndimage
f,cols,rows,out,ids=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4],sys.argv[5].split(',')
im=Image.open(f).convert('RGBA'); W,H=im.size; res=[]
for k,i in enumerate(ids):
    if not i: continue
    r,c=divmod(k,cols)
    cell=im.crop((c*W//cols,r*H//rows,(c+1)*W//cols,(r+1)*H//rows))
    a=np.asarray(cell)[:,:,3]>40
    lab,n=ndimage.label(ndimage.binary_dilation(a,iterations=3))
    if n==0: print('EMPTY',i); continue
    sizes=ndimage.sum(a,lab,range(1,n+1)); keep=np.zeros_like(a)
    big=sizes.max()
    for j,s in enumerate(sizes,1):
        sl=ndimage.find_objects((lab==j).astype(int))[0]; edge=sl[0].start<=1 or sl[1].start<=1 or sl[0].stop>=a.shape[0]-1 or sl[1].stop>=a.shape[1]-1
        if s>big*(0.15 if edge else 0.02): keep|=(lab==j)
    ys,xs=np.where(keep&a); b=(xs.min(),ys.min(),xs.max()+1,ys.max()+1)
    m=Image.fromarray((keep*255).astype('uint8')); cell.putalpha(Image.fromarray(np.minimum(np.asarray(cell)[:,:,3],np.asarray(m))))
    cell=cell.crop(b); p=8; s=max(cell.size)+2*p
    bg=Image.new('RGBA',(s,s),(255,255,255,255)); bg.alpha_composite(cell,((s-cell.width)//2,(s-cell.height)//2))
    bg.convert('RGB').save('%s/%s.png'%(out,i)); res.append((i,cell.size))
print(res)
